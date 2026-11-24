"""Deterministic experiment discovery; experimental metadata is never guessed."""
from __future__ import annotations

import csv
import inspect
import os
import stat
from pathlib import Path
from typing import Any, Mapping


OBJECTIVES = {
    "full_field": ("full_field", "full field", "2d dic", "全场应变", "二维全场应变"),
    "virtual_gauge": ("virtual_gauge", "virtual gauge", "虚拟标距", "标距应变"),
    "stress_strain": ("stress_strain", "stress strain", "应力应变", "应力应变曲线"),
}
GENERATED_DIRECTORIES = {
    ".git", ".venv", "__pycache__", "outputs", "output", "results", "runs",
    "plans", "checkpoints", "plots", "qc", "optional", "report",
    "_previous_runs", "_failed_runs", "dist", "build",
}


def _is_link_or_junction(path: Path) -> bool:
    if hasattr(Path, "is_junction"):
        try:
            if path.is_junction():
                return True
        except OSError:
            return False
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse_tag = getattr(info, "st_reparse_tag", 0)
    return reparse_tag in (
        getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C),
        getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003),
    )


def source_files(folder: Path):
    """Walk once, pruning outputs and directory links before descending."""
    for current, directories, names in os.walk(folder, followlinks=False):
        current_path = Path(current)
        if ({"project.json", "CURRENT.json"}.issubset(names) or "run_state.json" in names
                or any(name.startswith("frame_") and name.endswith("_parameters.txt") for name in names)):
            directories[:] = []
            continue
        directories[:] = sorted(
            name for name in directories
            if name.casefold() not in GENERATED_DIRECTORIES
            and not _is_link_or_junction(current_path / name)
        )
        for name in sorted(names):
            path = current_path / name
            if not _is_link_or_junction(path):
                yield path


def discover_experiment(folder: str | os.PathLike[str], *, objective: str = "full_field", options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from . import api
    from ._config import _ensure_known_fields, _positive_number
    from ._inventory import IMAGE_SUFFIXES, parse_frame_map, parse_native_frame
    from ._io import AgentError, _read_json

    if not isinstance(folder, (str, os.PathLike)) or not str(folder).strip():
        raise AgentError("CONFIG_TYPE_ERROR", "folder must be a non-empty path string", path="$.folder")
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise AgentError("EXPERIMENT_DIRECTORY_MISSING", f"experiment directory does not exist: {root}", path=str(root))
    if options is not None and not isinstance(options, Mapping):
        raise AgentError("CONFIG_TYPE_ERROR", "options must be an object", path="$.options")
    normalized = next((key for key, aliases in OBJECTIVES.items() if isinstance(objective, str) and objective.strip().casefold() in aliases), None)
    required: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    if normalized is None:
        required.append({"field": "objective", "reason": "This objective has no supported deterministic interpretation.", "choices": list(OBJECTIVES)})
    metadata_path = root / "experiment.json"
    metadata = _read_json(metadata_path) if metadata_path.is_file() else None
    if metadata is not None:
        _ensure_known_fields(metadata, {"schema", "options"}, path="$.experiment")
        if metadata.get("schema") != "straintrace.experiment/v1" or not isinstance(metadata.get("options"), dict):
            raise AgentError("EXPERIMENT_SCHEMA", "experiment.json requires schema straintrace.experiment/v1 and an options object", path=str(metadata_path))
    selected = {**(metadata["options"] if metadata else {}), **dict(options or {})}
    allowed = set(inspect.signature(api.init_project).parameters) - {"project"}
    _ensure_known_fields(selected, allowed, path="$.options")
    for key in ("image_dir", "frame_map_csv", "loads_csv", "mask_file"):
        if selected.get(key) is not None:
            if not isinstance(selected[key], (str, os.PathLike)) or not str(selected[key]).strip():
                raise AgentError("CONFIG_TYPE_ERROR", f"{key} must be a path", path=f"$.options.{key}")
            path = Path(selected[key]).expanduser()
            selected[key] = str((path if path.is_absolute() else root / path).resolve())
    mask_path = selected.get("mask_file")
    groups: dict[Path, list[Path]] = {}
    maps: list[Path] = []
    loads: list[dict[str, Any]] = []
    for path in source_files(root):
        if path.suffix.lower() in IMAGE_SUFFIXES and str(path.resolve()) != mask_path:
            groups.setdefault(path.parent, []).append(path)
        elif path.suffix.lower() == ".csv":
            try:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    columns = next(csv.reader(stream), [])
            except (OSError, UnicodeError, csv.Error) as exc:
                warnings.append({"code": "TABLE_HEADER_UNREADABLE", "message": f"{path}: {exc}"})
                continue
            names = {name.strip().lower() for name in columns}
            if names.intersection({"filename", "path", "file"}) and names.intersection({"native_frame", "frame"}):
                maps.append(path)
            else:
                loads.append({"path": str(path), "columns": columns})
    candidates = []
    for directory, files in sorted(groups.items(), key=lambda pair: str(pair[0]).casefold()):
        frames: list[int] = []
        ambiguous = 0
        for path in files:
            try:
                frames.append(parse_native_frame(path))
            except AgentError:
                ambiguous += 1
        values = sorted(set(frames))
        candidates.append({"image_dir": str(directory), "frame_count": len(files),
                           "native_frame_range": [values[0], values[-1]] if values else None,
                           "ambiguous_filename_count": ambiguous, "duplicate_native_frame_count": len(frames) - len(values),
                           "missing_native_frame_ranges": [[a + 1, b - 1] for a, b in zip(values, values[1:]) if b > a + 1]})
    if not selected.get("image_dir"):
        if len(candidates) == 1:
            selected["image_dir"] = candidates[0]["image_dir"]
        else:
            required.append({"field": "image_dir", "reason": "Select one independent image sequence." if candidates else "No source image sequence was found.", "choices": [item["image_dir"] for item in candidates]})
    chosen = next((item for item in candidates if item["image_dir"] == selected.get("image_dir")), None)
    if selected.get("image_dir") and chosen is None:
        required.append({"field": "image_dir", "reason": "The selected directory is not a discovered source sequence; select one candidate directory."})
    if chosen and chosen["frame_count"] < 2:
        required.append({"field": "image_dir", "reason": "At least a reference and a deformation image are required."})
    if chosen and any(Path(chosen["image_dir"]) in Path(item["image_dir"]).parents for item in candidates if item is not chosen):
        required.append({"field": "image_dir", "reason": "The selected directory contains another image sequence; choose an independent leaf sequence."})
    # A sole explicit map is safe to propose; full coverage is validated below.
    if not selected.get("frame_map_csv") and len(maps) == 1:
        selected["frame_map_csv"] = str(maps[0])
    explicit_times = False
    if chosen and selected.get("frame_map_csv"):
        try:
            mapping = parse_frame_map(Path(selected["frame_map_csv"]), Path(chosen["image_dir"]), groups[Path(chosen["image_dir"])])
            ordered = sorted(mapping.values(), key=lambda item: item["native_frame"])
            times = [item["time_s"] for item in ordered]
            natives = [item["native_frame"] for item in ordered]
            if len(set(natives)) != len(natives):
                raise AgentError("FRAME_ID_DUPLICATE", "frame map contains duplicate native frame numbers")
            if any(value is not None for value in times):
                if any(value is None for value in times) or any(b <= a for a, b in zip(times, times[1:])):
                    raise AgentError("FRAME_TIME_ORDER", "explicit frame times must be complete and strictly increasing")
                explicit_times = True
        except AgentError as exc:
            required.append({"field": "frame_map_csv", "reason": exc.message, "error_code": exc.code})
    elif chosen and (chosen["ambiguous_filename_count"] or chosen["duplicate_native_frame_count"]):
        required.append({"field": "frame_map_csv", "reason": "Native image identity requires an explicit unique filename/native_frame mapping.", "choices": [str(path) for path in maps]})
    if selected.get("time_basis") is not None and selected.get("time_basis") not in {"physical_time", "frame_index"}:
        required.append({"field": "time_basis", "reason": "time_basis must be physical_time or frame_index.", "choices": ["physical_time", "frame_index"]})
    if not explicit_times and selected.get("frame_interval_s") is None:
        if normalized in {"full_field", "virtual_gauge"} and not selected.get("loads_csv") and selected.get("time_basis", "frame_index") == "frame_index":
            selected["time_basis"] = "frame_index"
            warnings.append({"code": "FRAME_INDEX_AXIS", "message": "Acquisition time is unavailable; results use native image frame numbers and time_s remains null. Time-dependent quantities are unavailable."})
        else:
            required.append({"field": "frame_interval_s", "reason": "Provide the recorded acquisition interval in seconds or a complete frame-map time_s column; frame index is not seconds."})
    else:
        if selected.get("time_basis") == "frame_index":
            required.append({"field": "time_basis", "reason": "frame_index conflicts with recorded frame times or frame_interval_s; use physical_time.", "choices": ["physical_time"]})
        else:
            selected.setdefault("time_basis", "physical_time")
    if selected.get("time_basis") == "frame_index" and selected.get("loads_csv"):
        required.append({"field": "time_basis", "reason": "frame_index cannot be used with loads_csv; provide physical_time.", "choices": ["physical_time"]})
    if selected.get("frame_interval_s") is not None:
        _positive_number(selected["frame_interval_s"], "frame_interval_s")
    if normalized in {"virtual_gauge", "stress_strain"} and not selected.get("gauges"):
        required.append({"field": "gauges", "reason": "Specify measurement endpoints and loading axis; automatic geometry candidates do not establish a physical gauge."})
    if normalized == "stress_strain":
        for field in ("loads_csv", "load_time_column", "load_force_column", "load_time_offset_s", "max_gap_s", "area_mm2"):
            if selected.get(field) is None or (isinstance(selected.get(field), str) and not selected[field].strip()):
                required.append({"field": field, "reason": "Required for synchronized force / supplied cross-sectional area and DIC gauge strain."})
        if selected.get("clock_confirmed") is not True:
            required.append({"field": "clock_confirmed", "reason": "Confirm the recorded relationship between camera and load clocks."})
    elif selected.get("loads_csv"):
        for field in ("load_time_column", "load_time_offset_s", "max_gap_s"):
            if selected.get(field) is None or (isinstance(selected.get(field), str) and not selected[field].strip()):
                required.append({"field": field, "reason": "Explicit time-channel metadata is required when assigning a load table."})
        if not selected.get("load_force_column") and not selected.get("displacement_column"):
            required.append({"field": "load_force_column", "reason": "Configure load_force_column or displacement_column when loads_csv is provided."})
    if selected.get("loads_csv"):
        time_col = selected.get("load_time_column")
        if time_col and selected.get("load_time_unit") is None and time_col not in {"time_s", "timestamp_s", "time_seconds"}:
            required.append({"field": "load_time_unit", "reason": "Provide load_time_unit; only canonical *_s columns imply seconds.", "choices": ["s", "ms", "min", "h"]})
        force_col = selected.get("load_force_column")
        if force_col and selected.get("force_unit") is None and force_col not in {"force_N", "load_N", "force_kN", "load_kN"}:
            required.append({"field": "force_unit", "reason": "Provide force_unit; only canonical *_N or *_kN columns imply units.", "choices": ["N", "kN"]})
        if not force_col and (selected.get("area_mm2") is not None or selected.get("force_unit") is not None):
            required.append({"field": "load_force_column", "reason": "area_mm2 and force_unit require load_force_column."})
        disp_col = selected.get("displacement_column")
        if disp_col:
            if selected.get("displacement_unit") is None and disp_col not in {
                "displacement_mm", "machine_displacement_mm", "crosshead_mm",
                "displacement_um", "machine_displacement_um", "crosshead_um",
            }:
                required.append({"field": "displacement_unit", "reason": "Provide displacement_unit for non-canonical displacement columns.", "choices": ["mm", "um"]})
            if selected.get("displacement_source") is None:
                required.append({"field": "displacement_source", "reason": "Provide displacement_source when displacement_column is configured.", "choices": ["crosshead", "machine_extensometer", "other_machine_channel"]})
        elif any(selected.get(k) is not None for k in ("displacement_unit", "displacement_source", "machine_gauge_length_mm")):
            required.append({"field": "displacement_column", "reason": "Machine displacement metadata requires displacement_column."})
    elif any(selected.get(k) is not None for k in (
        "load_time_column", "load_force_column", "load_time_unit", "force_unit",
        "load_time_offset_s", "max_gap_s", "area_mm2", "displacement_column",
        "displacement_unit", "displacement_source", "machine_gauge_length_mm",
    )):
        if not any(item.get("field") == "loads_csv" for item in required):
            required.append({"field": "loads_csv", "reason": "Load/displacement channel metadata requires loads_csv."})
    if loads and not selected.get("loads_csv"):
        warnings.append({"code": "UNASSIGNED_TABLES", "message": "Tables were found but no load channels, units or clock relationship were assigned."})
    return {"schema": "straintrace.experiment-discovery/v1", "folder": str(root), "objective": normalized or objective,
            "status": "needs_input" if required else "ready", "method": "fixed_reference_2d_dic",
            "candidate_sequences": candidates, "candidate_load_tables": loads,
            "candidate_frame_maps": [str(path) for path in maps], "init_options": selected,
            "required_inputs": required, "warnings": warnings,
            "metadata_file": str(metadata_path) if metadata else None,
            "integrity_scope": "directory structure, native frame identity and supplied timing; image decoding and SHA-256 occur at initialization"}
