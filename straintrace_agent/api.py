"""Streaming, provenance-bound 2D DIC automation API.

The package calls the established ezdic_core solver one frame at a time. It
keeps source images read-only and stores numeric/string NPZ checkpoints without
pickle so a stopped run can resume without retaining the full image sequence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import socket
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .mechanics import (
    _auto_gauge_candidates, _gauge_group_summary, _interpolate_bounded,
    _interpolate_bounded_detail, _load_mechanics, _mechanics_at,
    _measure_gauges, _resolve_gauge_geometry,
)


PROJECT_SCHEMA = "straintrace.agent-project/v1"
RUN_SCHEMA = "straintrace.agent-run/v1"
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
_FRAME_TOKEN = re.compile(r"\d+")
QUALITY_GATE_SCOPE = "existing ezdic_core correlation and strain validity thresholds; not experimental validation or scientific acceptance"
_REPLACE_RETRY_DELAYS_S = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
_CORE_SOLVER_PARAMETER_KEYS = ("subset_size_px", "step_px", "solver", "strain_window_poi", "search_radius_px")


def _component_file_slug(component: str) -> str:
    """Return a stable filename token that stays unique on case-insensitive filesystems.

    Reuse the core's established component suffixes for its nine field
    components (notably distinguishing Green-Lagrange ``Exx`` from
    infinitesimal ``exx``). Other supported field-helper components receive a
    readable, case-fold-safe slug with a short hash of the original spelling.
    """
    import ezdic_core as core

    if not isinstance(component, str) or not component:
        raise AgentError("COMPONENT_INVALID", "component must be a non-empty string", path="$.component")
    suffixes = getattr(core, "DIC_COMPONENT_FILE_SUFFIXES", {})
    if component in suffixes:
        return str(suffixes[component])
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", component).strip("_").lower() or "component"
    digest = hashlib.sha256(component.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


class AgentError(RuntimeError):
    """Stable, machine-readable workflow error."""

    def __init__(self, code: str, message: str, *, path: str = "$", details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"error_code": self.code, "message": self.message, "path": self.path}
        if self.details is not None:
            result["details"] = _json_safe(self.details)
        return result


class _RunCancelled(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise AgentError("JSON_TYPE_ERROR", f"unsupported JSON value: {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AgentError("JSON_VALUE_ERROR", str(exc)) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AgentError("INPUT_READ_ERROR", f"cannot read {path}: {exc}", path=str(path)) from exc
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate_plan_checksum(plan: Mapping[str, Any]) -> str:
    plan_body = dict(plan)
    saved_sha = plan_body.pop("plan_sha256", None)
    if saved_sha != _hash_json(plan_body):
        raise AgentError("PLAN_CHECKSUM_MISMATCH", "saved analysis plan checksum is invalid")
    return str(saved_sha)


def _solver_parameters_match(recorded: Any, expected: Any) -> bool:
    if not isinstance(recorded, Mapping) or not isinstance(expected, Mapping):
        return False
    for key in _CORE_SOLVER_PARAMETER_KEYS:
        if key not in recorded or key not in expected:
            return False
        if type(recorded[key]) is not type(expected[key]) or recorded[key] != expected[key]:
            return False
    return True


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Replace a closed temporary file, retrying only bounded PermissionErrors."""
    for attempt in range(len(_REPLACE_RETRY_DELAYS_S) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(_REPLACE_RETRY_DELAYS_S):
                raise
            time.sleep(_REPLACE_RETRY_DELAYS_S[attempt])


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical(value) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite value {token}")))
    except FileNotFoundError as exc:
        raise AgentError("PROJECT_FILE_MISSING", f"required project file is missing: {path.name}", path=str(path)) from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AgentError("PROJECT_JSON_INVALID", f"cannot read {path.name}: {exc}", path=str(path)) from exc
    if not isinstance(raw, dict):
        raise AgentError("PROJECT_JSON_INVALID", f"{path.name} must contain a JSON object", path=str(path))
    return raw


def _positive_number(value: Any, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AgentError("CONFIG_TYPE_ERROR", f"{field} must be a finite number", path=f"$.{field}")
    number = float(value)
    if number < 0 if allow_zero else number <= 0:
        op = ">= 0" if allow_zero else "> 0"
        raise AgentError("CONFIG_VALUE_ERROR", f"{field} must be {op}", path=f"$.{field}")
    return number


def _validate_rect(value: Sequence[Any], field: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise AgentError("CONFIG_VALUE_ERROR", f"{field} must be [x, y, width, height]", path=f"$.{field}")
    if any(type(item) is not int for item in value) or any(item < 0 for item in value) or value[2] <= 0 or value[3] <= 0:
        raise AgentError("CONFIG_VALUE_ERROR", f"{field} must contain nonnegative integer origin and positive dimensions", path=f"$.{field}")
    return [int(item) for item in value]


def _validate_gauge(value: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AgentError("CONFIG_TYPE_ERROR", "gauge must be an object", path=f"$.gauges[{index}]")
    allowed = {"name", "axis", "a_point_px", "b_point_px", "a_roi_px", "b_roi_px", "max_distance_px", "geometry_status", "min_zncc"}
    _ensure_known_fields(value, allowed, path=f"$.gauges[{index}]")
    name = value.get("name")
    axis = value.get("axis")
    if not isinstance(name, str) or not name.strip() or axis not in {"x", "y", "distance"}:
        raise AgentError("GAUGE_CONFIG", "gauge requires a name and axis x, y, or distance", path=f"$.gauges[{index}]")
    result: dict[str, Any] = {"name": name.strip(), "axis": axis, "geometry_status": str(value.get("geometry_status", "user_defined"))}
    for side in ("a", "b"):
        point_key, roi_key = f"{side}_point_px", f"{side}_roi_px"
        if (point_key in value) == (roi_key in value):
            raise AgentError("GAUGE_ENDPOINT", f"provide exactly one of {point_key} or {roi_key}", path=f"$.gauges[{index}]")
        if point_key in value:
            point = value[point_key]
            if not isinstance(point, (list, tuple)) or len(point) != 2 or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in point):
                raise AgentError("GAUGE_ENDPOINT", f"{point_key} must be a finite [x, y] point", path=f"$.gauges[{index}].{point_key}")
            result[point_key] = [float(item) for item in point]
        else:
            result[roi_key] = _validate_rect(value[roi_key], f"gauges[{index}].{roi_key}")
    result["max_distance_px"] = _positive_number(value.get("max_distance_px", 0.0), f"gauges[{index}].max_distance_px", allow_zero=True)
    if "min_zncc" in value:
        threshold = value["min_zncc"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or not -1 <= threshold <= 1:
            raise AgentError("GAUGE_CONFIG", "min_zncc must be finite in [-1, 1]", path=f"$.gauges[{index}].min_zncc")
        result["min_zncc"] = float(threshold)
    return result


def _ensure_known_fields(value: Mapping[str, Any], allowed: set[str], *, path: str = "$") -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AgentError("CONFIG_UNKNOWN_FIELD", f"unknown field: {unknown[0]}", path=f"{path}.{unknown[0]}")


def _project_root(project: str | os.PathLike[str]) -> Path:
    return Path(project).expanduser().resolve(strict=False)


def _read_config(project: Path) -> dict[str, Any]:
    config = _read_json(project / "project.json")
    return _validate_project_config(config, project)


def _validate_project_config(config: Mapping[str, Any], project: Path) -> dict[str, Any]:
    """Validate a candidate configuration before replacing persisted state."""
    config = dict(config)
    allowed = {
        "schema", "project_id", "image_dir", "frame_interval_s", "frame_map_csv",
        "reference_native_frame", "loads", "calibration", "regions", "paths", "gauges",
        "mask_file", "mask_sha256", "roi_override_px", "exclusions_px", "step_hint_px", "plot_every",
        "time_basis",
    }
    _ensure_known_fields(config, allowed)
    missing_root_fields = sorted(allowed - set(config) - {"time_basis"})
    if missing_root_fields:
        raise AgentError("CONFIG_REQUIRED_FIELD", f"missing project configuration field: {missing_root_fields[0]}", path=f"$.{missing_root_fields[0]}")
    if config.get("schema") != PROJECT_SCHEMA:
        raise AgentError("PROJECT_SCHEMA_UNSUPPORTED", f"expected {PROJECT_SCHEMA}", path="$.schema")
    if not isinstance(config.get("project_id"), str) or not config["project_id"].strip():
        raise AgentError("CONFIG_TYPE_ERROR", "project_id must be a non-empty string", path="$.project_id")
    if not isinstance(config.get("image_dir"), str) or not config["image_dir"].strip():
        raise AgentError("CONFIG_TYPE_ERROR", "image_dir must be a non-empty path string", path="$.image_dir")
    for key in ("frame_map_csv", "mask_file"):
        if config.get(key) is not None and (not isinstance(config[key], str) or not config[key].strip()):
            raise AgentError("CONFIG_TYPE_ERROR", f"{key} must be a path string or null", path=f"$.{key}")
    for key in ("regions", "paths", "gauges", "exclusions_px"):
        value = config.get(key, [])
        if not isinstance(value, list):
            raise AgentError("CONFIG_TYPE_ERROR", f"{key} must be a JSON array", path=f"$.{key}")
    for key in ("regions", "paths"):
        if any(not isinstance(value, Mapping) for value in config.get(key, [])):
            raise AgentError("CONFIG_TYPE_ERROR", f"each {key} entry must be an object", path=f"$.{key}")

    loads = config.get("loads")
    if loads is not None:
        if not isinstance(loads, Mapping):
            raise AgentError("CONFIG_TYPE_ERROR", "loads must be an object or null", path="$.loads")
        _ensure_known_fields(loads, {
            "path", "time_column", "force_column", "time_unit", "time_unit_basis", "force_unit", "force_unit_basis",
            "displacement_column", "displacement_unit", "displacement_unit_basis", "displacement_source",
            "machine_gauge_length_mm", "clock_confirmed", "time_offset_s", "max_gap_s", "area_mm2",
        }, path="$.loads")
        required_load_fields = {
            "path", "time_column", "force_column", "time_unit", "time_unit_basis", "force_unit", "force_unit_basis",
            "displacement_column", "displacement_unit", "displacement_unit_basis", "displacement_source",
            "machine_gauge_length_mm", "clock_confirmed", "time_offset_s", "max_gap_s", "area_mm2",
        }
        missing_load_fields = sorted(required_load_fields - set(loads))
        if missing_load_fields:
            raise AgentError("CONFIG_REQUIRED_FIELD", f"missing loads configuration field: {missing_load_fields[0]}", path=f"$.loads.{missing_load_fields[0]}")
        if not isinstance(loads.get("path"), str) or not loads["path"].strip():
            raise AgentError("CONFIG_TYPE_ERROR", "loads.path must be a non-empty path string", path="$.loads.path")
        for key in ("time_column", "force_column", "displacement_column", "displacement_source"):
            if loads.get(key) is not None and (not isinstance(loads[key], str) or not loads[key].strip()):
                raise AgentError("CONFIG_TYPE_ERROR", f"loads.{key} must be a non-empty string or null", path=f"$.loads.{key}")
        if not isinstance(loads.get("clock_confirmed"), bool):
            raise AgentError("CONFIG_TYPE_ERROR", "loads.clock_confirmed must be a JSON boolean", path="$.loads.clock_confirmed")
        for basis_key in ("time_unit_basis", "force_unit_basis", "displacement_unit_basis"):
            if loads.get(basis_key, "explicit") not in {"explicit", "canonical_column_name"}:
                raise AgentError("CONFIG_VALUE_ERROR", f"loads.{basis_key} has an unsupported value", path=f"$.loads.{basis_key}")
        canonical_units = (
            ("time_unit_basis", "time_unit", "time_column", {"time_s": "s", "timestamp_s": "s", "time_seconds": "s"}),
            ("force_unit_basis", "force_unit", "force_column", {"force_N": "N", "load_N": "N", "force_kN": "kN", "load_kN": "kN"}),
            ("displacement_unit_basis", "displacement_unit", "displacement_column", {
                "displacement_mm": "mm", "machine_displacement_mm": "mm", "crosshead_mm": "mm",
                "displacement_um": "um", "machine_displacement_um": "um", "crosshead_um": "um",
            }),
        )
        for basis_key, unit_key, column_key, recognized in canonical_units:
            if loads.get(basis_key, "explicit") == "canonical_column_name":
                expected_unit = recognized.get(loads.get(column_key))
                if expected_unit is None or loads.get(unit_key) != expected_unit:
                    raise AgentError("CONFIG_UNIT_BASIS_MISMATCH", f"loads.{unit_key} disagrees with its canonical column name", path=f"$.loads.{unit_key}")
        if loads.get("force_column") is None and (loads.get("force_unit") is not None or loads.get("force_unit_basis") not in (None, "explicit")):
            raise AgentError("CONFIG_VALUE_ERROR", "force unit metadata requires force_column", path="$.loads.force_column")

    calibration = config.get("calibration")
    if calibration is not None and not isinstance(calibration, Mapping):
        raise AgentError("CONFIG_TYPE_ERROR", "calibration must be an object or null", path="$.calibration")
    for key in ("gauges", "exclusions_px"):
        if key not in config:
            config[key] = []

    validated = _config_from_arguments(
        image_dir=config["image_dir"], frame_interval_s=config.get("frame_interval_s"),
        frame_map_csv=config.get("frame_map_csv"), reference_native_frame=config.get("reference_native_frame"),
        loads_csv=loads.get("path") if loads else None,
        load_time_column=loads.get("time_column") if loads else None,
        load_force_column=loads.get("force_column") if loads else None,
        load_time_unit=(None if loads and loads.get("time_unit_basis") == "canonical_column_name" else loads.get("time_unit")) if loads else None,
        force_unit=(None if loads and loads.get("force_unit_basis") == "canonical_column_name" else loads.get("force_unit")) if loads else None,
        displacement_column=loads.get("displacement_column") if loads else None,
        displacement_unit=(None if loads and loads.get("displacement_unit_basis") == "canonical_column_name" else loads.get("displacement_unit")) if loads else None,
        displacement_source=loads.get("displacement_source") if loads else None,
        machine_gauge_length_mm=loads.get("machine_gauge_length_mm") if loads else None,
        clock_confirmed=loads.get("clock_confirmed", False) if loads else False,
        load_time_offset_s=loads.get("time_offset_s") if loads else None,
        max_gap_s=loads.get("max_gap_s") if loads else None,
        area_mm2=loads.get("area_mm2") if loads else None,
        calibration=calibration,
        regions=config.get("regions", []), paths=config.get("paths", []), gauges=config.get("gauges", []),
        mask_file=config.get("mask_file"), roi_override_px=config.get("roi_override_px"),
        exclusions_px=config.get("exclusions_px", []), step_hint_px=config.get("step_hint_px"),
        plot_every=config.get("plot_every", 0),
        time_basis=config.get("time_basis", "physical_time"),
    )
    validated["project_id"] = config["project_id"]
    if loads:
        # These values document why units were assigned; ensure they agree
        # with the validated channel names and the normalized unit choices.
        for key in ("time_unit_basis", "force_unit_basis", "displacement_unit_basis"):
            saved = loads.get(key, "explicit")
            if validated["loads"].get(key) != saved:
                raise AgentError("CONFIG_UNIT_BASIS_MISMATCH", f"loads.{key} disagrees with configured units", path=f"$.loads.{key}")
    if config.get("mask_file"):
        if not Path(config["mask_file"]).is_file():
            raise AgentError("MASK_FILE_MISSING", f"mask image is missing: {config['mask_file']}", path="$.mask_file")
        if config.get("mask_sha256") != validated.get("mask_sha256"):
            raise AgentError("CONFIG_MASK_HASH_MISMATCH", "mask file hash differs from the project configuration", path="$.mask_sha256")
    elif config.get("mask_sha256") is not None:
        raise AgentError("CONFIG_MASK_HASH_MISMATCH", "mask_sha256 requires mask_file", path="$.mask_sha256")
    try:
        project.relative_to(Path(validated["image_dir"]).resolve(strict=False))
    except ValueError:
        pass
    else:
        raise AgentError("PROJECT_INSIDE_IMAGE_DIRECTORY", "project output must be outside the raw image directory", path="$.image_dir")
    return validated


def _explicit_frame_map(path: Path, image_root: Path, image_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        raise AgentError("FRAME_MAP_READ_ERROR", f"cannot read frame map: {exc}", path=str(path)) from exc
    names = {str(column).strip().lower(): str(column) for column in table.columns}
    file_col = next((names[key] for key in ("path", "filename", "file") if key in names), None)
    frame_col = names.get("native_frame") or names.get("frame")
    time_col = names.get("time_s") or names.get("timestamp_s")
    if file_col is None or frame_col is None:
        raise AgentError("FRAME_MAP_COLUMNS", "frame map requires path/filename and native_frame columns", path=str(path))
    known = {item.resolve(strict=False) for item in image_paths}
    by_basename: dict[str, list[Path]] = {}
    for image_path in known:
        by_basename.setdefault(image_path.name.casefold(), []).append(image_path)
    by_key: dict[str, dict[str, Any]] = {}
    for row_index, row in table.iterrows():
        raw_path = str(row[file_col]).strip()
        if not raw_path:
            raise AgentError("FRAME_MAP_VALUE", f"empty path at row {row_index + 2}", path=str(path))
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = image_root / candidate
        candidate = candidate.resolve(strict=False)
        if candidate not in known:
            # A basename is accepted only if unique in the image directory.
            matches = by_basename.get(Path(raw_path).name.casefold(), [])
            if len(matches) != 1:
                raise AgentError("FRAME_MAP_PATH_UNKNOWN", f"frame map row does not identify one source image: {raw_path}", path=str(path))
            candidate = matches[0]
        try:
            native_value = float(row[frame_col])
            if not math.isfinite(native_value) or native_value != int(native_value) or native_value < 0:
                raise ValueError
            native = int(native_value)
            time_value = None
            if time_col is not None and pd.notna(row[time_col]):
                time_value = float(row[time_col])
                if not math.isfinite(time_value):
                    raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise AgentError("FRAME_MAP_VALUE", f"invalid native_frame or time at row {row_index + 2}", path=str(path)) from exc
        key = str(candidate).casefold()
        if key in by_key:
            raise AgentError("FRAME_MAP_DUPLICATE_IMAGE", f"image appears more than once in frame map: {candidate}", path=str(path))
        by_key[key] = {"native_frame": native, "time_s": time_value}
    if set(by_key) != {str(item.resolve(strict=False)).casefold() for item in image_paths}:
        raise AgentError("FRAME_MAP_INCOMPLETE", "frame map must cover every source image exactly once", path=str(path))
    return by_key


def _native_frame_from_name(path: Path) -> int:
    tokens = _FRAME_TOKEN.findall(path.stem)
    if len(tokens) != 1:
        raise AgentError(
            "FRAME_ID_AMBIGUOUS",
            f"expected one numeric frame token in {path.name}; provide an explicit frame map",
            path=str(path),
            details={"numeric_tokens": tokens},
        )
    return int(tokens[0])


def _scan_inventory(config: Mapping[str, Any], *, read_dimensions: bool = True) -> list[dict[str, Any]]:
    from .discovery import source_files
    image_root = Path(str(config["image_dir"])).expanduser().resolve(strict=False)
    if not image_root.is_dir():
        raise AgentError("IMAGE_DIRECTORY_MISSING", f"image directory does not exist: {image_root}", path="$.image_dir")
    mask_path = Path(str(config["mask_file"])).resolve(strict=False) if config.get("mask_file") else None
    paths = sorted((p.resolve(strict=False) for p in source_files(image_root) if p.suffix.lower() in IMAGE_SUFFIXES and (mask_path is None or p.resolve(strict=False) != mask_path)), key=lambda p: str(p).casefold())
    if not paths:
        raise AgentError("NO_IMAGES", f"no supported DIC images found in {image_root}", path="$.image_dir")
    explicit: dict[str, dict[str, Any]] | None = None
    if config.get("frame_map_csv"):
        map_path = Path(str(config["frame_map_csv"])).expanduser().resolve(strict=False)
        explicit = _explicit_frame_map(map_path, image_root, paths)
    records: list[dict[str, Any]] = []
    native_values: set[int] = set()
    times_explicit = explicit is not None and any(value.get("time_s") is not None for value in explicit.values())
    for path in paths:
        if explicit is None:
            native = _native_frame_from_name(path)
            time_value = None
        else:
            item = explicit[str(path).casefold()]
            native = item["native_frame"]
            time_value = item.get("time_s")
        if native in native_values:
            raise AgentError("FRAME_ID_DUPLICATE", f"duplicate native frame number: {native}", path=str(path))
        native_values.add(native)
        row: dict[str, Any] = {
            "native_frame": native,
            "path": str(path),
            "relative_path": path.relative_to(image_root).as_posix(),
            "sha256": _sha256(path),
            "time_s": time_value,
            "status": "pending",
        }
        if read_dimensions:
            try:
                import ezdic_core as core
                image = core.read_gray_image(path)
                row["shape_yx"] = [int(image.shape[0]), int(image.shape[1])]
                row["dtype"] = str(image.dtype)
            except Exception as exc:
                row["status"] = "unreadable"
                row["read_error"] = f"{type(exc).__name__}: {exc}"
        records.append(row)
    records.sort(key=lambda item: int(item["native_frame"]))
    if len(native_values) != len(records):
        raise AgentError("FRAME_ID_DUPLICATE", "native frame identity is not unique")
    native_order = [int(item["native_frame"]) for item in records]
    if any(right <= left for left, right in zip(native_order, native_order[1:])):
        raise AgentError("FRAME_ID_ORDER", "native frame numbers must be strictly increasing")
    if times_explicit:
        if config.get("time_basis") == "frame_index":
            raise AgentError("CONFIG_TIME_BASIS_CONFLICT", "frame_index cannot discard explicitly recorded frame times", path="$.time_basis")
        times = [item.get("time_s") for item in records]
        if any(value is None for value in times) or any(float(b) <= float(a) for a, b in zip(times, times[1:])):
            raise AgentError("FRAME_TIME_ORDER", "explicit frame times must be complete and strictly increasing")
    elif config.get("time_basis") != "frame_index":
        interval = config.get("frame_interval_s")
        if interval is None:
            raise AgentError("FRAME_TIME_REQUIRED", "provide a positive frame_interval_s or explicit frame-map time_s values", path="$.frame_interval_s")
        interval = _positive_number(interval, "frame_interval_s")
        first = native_order[0]
        for row in records:
            row["time_s"] = round((int(row["native_frame"]) - first) * interval, 12)
    reference_native = config.get("reference_native_frame")
    reference_row = next((row for row in records if row.get("native_frame") == reference_native and row.get("shape_yx")), None) if reference_native is not None else None
    reference_shape = tuple(reference_row["shape_yx"]) if reference_row else next((tuple(row["shape_yx"]) for row in records if row.get("shape_yx")), None)
    if reference_shape is not None:
        for row in records:
            if row.get("shape_yx") and tuple(row["shape_yx"]) != reference_shape:
                row["status"] = "dimension_mismatch"
    if config.get("reference_native_frame") is not None and int(config["reference_native_frame"]) not in native_values:
        raise AgentError("REFERENCE_FRAME_UNKNOWN", "reference_native_frame is not present in the image inventory", path="$.reference_native_frame")
    return records


def _config_from_arguments(
    *, image_dir: str | os.PathLike[str], frame_interval_s: float | None,
    frame_map_csv: str | os.PathLike[str] | None, reference_native_frame: int | None,
    loads_csv: str | os.PathLike[str] | None, load_time_column: str | None,
    load_force_column: str | None, load_time_unit: str | None, force_unit: str | None,
    displacement_column: str | None, displacement_unit: str | None, displacement_source: str | None,
    machine_gauge_length_mm: float | None,
    clock_confirmed: bool, load_time_offset_s: float | None, max_gap_s: float | None,
    area_mm2: float | None, calibration: Mapping[str, Any] | None,
    regions: Sequence[Mapping[str, Any]] | None, paths: Sequence[Mapping[str, Any]] | None,
    gauges: Sequence[Mapping[str, Any]] | None, mask_file: str | os.PathLike[str] | None,
    roi_override_px: Sequence[int] | None, exclusions_px: Sequence[Sequence[int]] | None,
    step_hint_px: int | None, plot_every: int,
    time_basis: str = "physical_time",
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema": PROJECT_SCHEMA,
        "project_id": uuid.uuid4().hex,
        "image_dir": str(Path(image_dir).expanduser().resolve(strict=False)),
        "frame_interval_s": frame_interval_s,
        "time_basis": time_basis,
        "frame_map_csv": str(Path(frame_map_csv).expanduser().resolve(strict=False)) if frame_map_csv else None,
        "reference_native_frame": reference_native_frame,
        "loads": None,
        "calibration": None,
        "regions": list(regions or []),
        "paths": list(paths or []),
        "gauges": list(gauges or []),
        "mask_file": str(Path(mask_file).expanduser().resolve(strict=False)) if mask_file else None,
        "mask_sha256": _sha256(Path(mask_file).expanduser().resolve(strict=False)) if mask_file else None,
        "roi_override_px": list(roi_override_px) if roi_override_px is not None else None,
        "exclusions_px": [list(item) for item in (exclusions_px or [])],
        "step_hint_px": step_hint_px,
        "plot_every": int(plot_every),
    }
    if reference_native_frame is not None and (type(reference_native_frame) is not int or reference_native_frame < 0):
        raise AgentError("CONFIG_VALUE_ERROR", "reference_native_frame must be a nonnegative integer", path="$.reference_native_frame")
    if not isinstance(time_basis, str) or time_basis not in {"physical_time", "frame_index"}:
        raise AgentError("CONFIG_TIME_BASIS", "time_basis must be physical_time or frame_index", path="$.time_basis")
    if time_basis == "frame_index" and (frame_interval_s is not None or loads_csv is not None):
        raise AgentError("CONFIG_TIME_BASIS_CONFLICT", "frame_index requires null frame_interval_s and no load table; time-dependent measurements require physical_time", path="$.time_basis")
    if frame_interval_s is not None:
        config["frame_interval_s"] = _positive_number(frame_interval_s, "frame_interval_s")
    if loads_csv is not None:
        required = {
            "load_time_column": load_time_column,
            "load_time_offset_s": load_time_offset_s,
            "max_gap_s": max_gap_s,
        }
        if any(value is None for value in required.values()):
            missing = next(key for key, value in required.items() if value is None)
            raise AgentError("LOAD_CONFIG_REQUIRED", f"{missing} is required when loads_csv is provided", path=f"$.{missing}")
        time_unit_basis = "explicit"
        if load_time_unit is None:
            if load_time_column in {"time_s", "timestamp_s", "time_seconds"}:
                load_time_unit = "s"
                time_unit_basis = "canonical_column_name"
            else:
                raise AgentError("LOAD_TIME_UNIT", "provide load_time_unit; only canonical *_s columns imply seconds", path="$.load_time_unit")
        if load_time_unit not in {"s", "ms", "min", "h"}:
            raise AgentError("LOAD_TIME_UNIT", "provide load_time_unit; only canonical *_s columns imply seconds", path="$.load_time_unit")
        force_unit_basis = "explicit"
        if load_force_column and force_unit is None:
            if load_force_column in {"force_N", "load_N"}:
                force_unit = "N"
                force_unit_basis = "canonical_column_name"
            elif load_force_column in {"force_kN", "load_kN"}:
                force_unit = "kN"
                force_unit_basis = "canonical_column_name"
        if load_force_column and force_unit not in {"N", "kN"}:
            raise AgentError("FORCE_UNIT", "provide force_unit; only canonical *_N or *_kN columns imply units", path="$.force_unit")
        if not load_force_column and not displacement_column:
            raise AgentError("LOAD_CHANNEL_REQUIRED", "configure force_column or displacement_column", path="$.loads")
        if area_mm2 is not None and not load_force_column:
            raise AgentError("AREA_WITHOUT_FORCE", "area_mm2 requires a configured force_column", path="$.area_mm2")
        if type(clock_confirmed) is not bool:
            raise AgentError("CONFIG_TYPE_ERROR", "clock_confirmed must be a JSON boolean", path="$.clock_confirmed")
        if isinstance(load_time_offset_s, bool) or not isinstance(load_time_offset_s, (int, float)) or not math.isfinite(float(load_time_offset_s)):
            raise AgentError("CONFIG_TYPE_ERROR", "load_time_offset_s must be a finite signed number", path="$.load_time_offset_s")
        displacement_unit_basis = "explicit"
        if displacement_column is not None and displacement_unit is None:
            if displacement_column in {"displacement_mm", "machine_displacement_mm", "crosshead_mm"}:
                displacement_unit = "mm"
                displacement_unit_basis = "canonical_column_name"
            elif displacement_column in {"displacement_um", "machine_displacement_um", "crosshead_um"}:
                displacement_unit = "um"
                displacement_unit_basis = "canonical_column_name"
        config["loads"] = {
            "path": str(Path(loads_csv).expanduser().resolve(strict=False)),
            "time_column": str(load_time_column),
            "force_column": str(load_force_column) if load_force_column else None,
            "time_unit": load_time_unit,
            "time_unit_basis": time_unit_basis,
            "force_unit": force_unit,
            "force_unit_basis": force_unit_basis,
            "displacement_column": str(displacement_column) if displacement_column else None,
            "displacement_unit": displacement_unit,
            "displacement_unit_basis": displacement_unit_basis,
            "displacement_source": displacement_source,
            "machine_gauge_length_mm": _positive_number(machine_gauge_length_mm, "machine_gauge_length_mm") if machine_gauge_length_mm is not None else None,
            "clock_confirmed": bool(clock_confirmed),
            "time_offset_s": float(load_time_offset_s),
            "max_gap_s": _positive_number(max_gap_s, "max_gap_s"),
            "area_mm2": _positive_number(area_mm2, "area_mm2") if area_mm2 is not None else None,
        }
        if displacement_column is not None:
            if displacement_unit not in {"mm", "um"}:
                raise AgentError("DISPLACEMENT_UNIT", "displacement_unit must be mm or um", path="$.displacement_unit")
            if displacement_source not in {"crosshead", "machine_extensometer", "other_machine_channel"}:
                raise AgentError("DISPLACEMENT_SOURCE", "provide displacement_source: crosshead, machine_extensometer, or other_machine_channel", path="$.displacement_source")
        elif any(value is not None for value in (displacement_unit, displacement_source, machine_gauge_length_mm)):
            raise AgentError("DISPLACEMENT_CONFIG_WITHOUT_COLUMN", "machine displacement metadata requires displacement_column", path="$.displacement_column")
    elif any(value is not None for value in (load_time_column, load_force_column, load_time_unit, force_unit, load_time_offset_s, max_gap_s, area_mm2, displacement_column, displacement_unit, displacement_source, machine_gauge_length_mm)):
        raise AgentError("LOAD_CONFIG_WITHOUT_DATA", "load metadata requires loads_csv", path="$.loads_csv")
    if calibration is not None:
        _ensure_known_fields(calibration, {"mm_per_px", "origin_px", "origin_mm", "axis_sign"}, path="$.calibration")
        if set(calibration) != {"mm_per_px", "origin_px", "origin_mm", "axis_sign"}:
            raise AgentError("CALIBRATION_INCOMPLETE", "calibration requires mm_per_px, origin_px, origin_mm, and axis_sign", path="$.calibration")
        _positive_number(calibration["mm_per_px"], "calibration.mm_per_px")
        for name, size in (("origin_px", 2), ("origin_mm", 2), ("axis_sign", 2)):
            pair = calibration[name]
            if not isinstance(pair, (list, tuple)) or len(pair) != size:
                raise AgentError("CALIBRATION_VALUE", f"calibration.{name} must contain two finite numbers", path=f"$.calibration.{name}")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in pair):
                raise AgentError("CALIBRATION_VALUE", f"calibration.{name} must contain two finite numbers", path=f"$.calibration.{name}")
        if any(float(value) not in (-1.0, 1.0) for value in calibration["axis_sign"]):
            raise AgentError("CALIBRATION_VALUE", "axis_sign values must be -1 or 1", path="$.calibration.axis_sign")
        config["calibration"] = _json_safe(calibration)
    if config["mask_file"] is not None and not Path(config["mask_file"]).is_file():
        raise AgentError("MASK_FILE_MISSING", f"mask image is missing: {config['mask_file']}", path="$.mask_file")
    if roi_override_px is not None:
        config["roi_override_px"] = _validate_rect(roi_override_px, "roi_override_px")
    for index, rect in enumerate(config["exclusions_px"]):
        config["exclusions_px"][index] = _validate_rect(rect, f"exclusions_px[{index}]")
    config["gauges"] = [_validate_gauge(gauge, index) for index, gauge in enumerate(config["gauges"])]
    if len({gauge["name"].casefold() for gauge in config["gauges"]}) != len(config["gauges"]):
        raise AgentError("GAUGE_CONFIG", "gauge names must be unique", path="$.gauges")
    from .regions import normalize_regions
    try:
        config["regions"] = normalize_regions(config["regions"])
    except ValueError as exc:
        raise AgentError("CONFIG_REGION_INVALID", str(exc), path="$.regions") from exc
    if step_hint_px is not None and (type(step_hint_px) is not int or step_hint_px < 1):
        raise AgentError("CONFIG_VALUE_ERROR", "step_hint_px must be a positive integer", path="$.step_hint_px")
    if type(plot_every) is not int or plot_every < 0:
        raise AgentError("CONFIG_VALUE_ERROR", "plot_every must be a nonnegative integer", path="$.plot_every")
    return config


def init_project(
    project: str | os.PathLike[str], image_dir: str | os.PathLike[str], *,
    frame_interval_s: float | None = None, frame_map_csv: str | os.PathLike[str] | None = None,
    reference_native_frame: int | None = None, loads_csv: str | os.PathLike[str] | None = None,
    load_time_column: str | None = None, load_force_column: str | None = None,
    load_time_unit: str | None = None, force_unit: str | None = None,
    displacement_column: str | None = None, displacement_unit: str | None = None,
    displacement_source: str | None = None, machine_gauge_length_mm: float | None = None,
    clock_confirmed: bool = False, load_time_offset_s: float | None = None,
    max_gap_s: float | None = None, area_mm2: float | None = None,
    calibration: Mapping[str, Any] | None = None,
    regions: Sequence[Mapping[str, Any]] | None = None,
    paths: Sequence[Mapping[str, Any]] | None = None,
    gauges: Sequence[Mapping[str, Any]] | None = None,
    mask_file: str | os.PathLike[str] | None = None,
    roi_override_px: Sequence[int] | None = None,
    exclusions_px: Sequence[Sequence[int]] | None = None,
    step_hint_px: int | None = None,
    plot_every: int = 0,
    time_basis: str = "physical_time",
) -> dict[str, Any]:
    """Create a new sample project and hash every source image once."""
    root = _project_root(project)
    if root.exists() and any(root.iterdir()):
        raise AgentError("PROJECT_ALREADY_EXISTS", f"refusing to overwrite a nonempty project: {root}", path=str(root))
    image_root = Path(image_dir).expanduser().resolve(strict=False)
    try:
        root.relative_to(image_root)
    except ValueError:
        pass
    else:
        raise AgentError("PROJECT_INSIDE_IMAGE_DIRECTORY", "project output must be outside the raw image directory", path=str(root))
    config = _config_from_arguments(
        image_dir=image_root, frame_interval_s=frame_interval_s, frame_map_csv=frame_map_csv,
        reference_native_frame=reference_native_frame, loads_csv=loads_csv,
        load_time_column=load_time_column, load_force_column=load_force_column,
        load_time_unit=load_time_unit, force_unit=force_unit,
        displacement_column=displacement_column, displacement_unit=displacement_unit,
        displacement_source=displacement_source, machine_gauge_length_mm=machine_gauge_length_mm,
        clock_confirmed=clock_confirmed,
        load_time_offset_s=load_time_offset_s, max_gap_s=max_gap_s, area_mm2=area_mm2,
        calibration=calibration, regions=regions, paths=paths, gauges=gauges,
        mask_file=mask_file, roi_override_px=roi_override_px, exclusions_px=exclusions_px,
        step_hint_px=step_hint_px,
        plot_every=plot_every,
        time_basis=time_basis,
    )
    records = _scan_inventory(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / "plans").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    _atomic_json(root / "project.json", config)
    input_fingerprint = _inventory_fingerprint(config, records)
    _record_inventory(root, records, input_fingerprint)
    _atomic_json(root / "CURRENT.json", {"schema": PROJECT_SCHEMA, "project_id": config["project_id"], "current_plan": None, "current_run": None})
    return {"project": str(root), "project_id": config["project_id"], "frame_count": len(records), "readable_count": sum(row["status"] == "pending" for row in records), "inventory_sha256": _hash_json(records)}


def inspect_project(project: str | os.PathLike[str]) -> dict[str, Any]:
    """Read-only summary of recorded inputs and analysis availability."""
    root = _project_root(project)
    config = _read_config(root)
    inventory = _read_json(root / "inventory.json")
    pointer = _read_json(root / "CURRENT.json")
    frames = inventory.get("frames", [])
    return {
        "project": str(root), "project_id": config.get("project_id"),
        "frame_count": len(frames), "readable_count": sum(row.get("status") == "pending" for row in frames),
        "unreadable_frames": [row.get("native_frame") for row in frames if row.get("status") != "pending"],
        "native_frames": [row.get("native_frame") for row in frames],
        "time_range_s": [frames[0].get("time_s"), frames[-1].get("time_s")] if frames else None,
        "image_shape_yx": next((row.get("shape_yx") for row in frames if row.get("shape_yx")), None),
        "frame_interval_s": config.get("frame_interval_s"),
        "time_basis": config.get("time_basis", "physical_time"),
        "load_data_configured": config.get("loads") is not None,
        "load_clock_confirmed": bool((config.get("loads") or {}).get("clock_confirmed", False)),
        "stress_available_in_principle": bool((config.get("loads") or {}).get("clock_confirmed") and (config.get("loads") or {}).get("area_mm2")),
        "physical_calibration_available": config.get("calibration") is not None,
        "analysis_rectangle_status": "candidate_not_physical_specimen_confirmation",
        "current_plan": pointer.get("current_plan"), "current_run": pointer.get("current_run"),
    }


def _code_fingerprint() -> dict[str, str]:
    import ezdic_core

    frozen_root = getattr(sys, "_MEIPASS", None)
    runtime_root = Path(frozen_root).resolve() if frozen_root else Path(__file__).resolve().parent.parent
    try:
        core_sources = ezdic_core.resolve_source_paths(
            include_cli=True, include_gui=False, include_benchmark=False, include_schema=False,
        )
    except Exception as exc:
        raise AgentError("CODE_PROVENANCE_UNAVAILABLE", f"cannot resolve core/CLI source files: {exc}") from exc
    package_root = runtime_root / "sources" / "straintrace_agent" if frozen_root else Path(__file__).resolve().parent
    if not package_root.is_dir():
        raise AgentError("CODE_PROVENANCE_UNAVAILABLE", f"agent source directory is missing: {package_root}")
    package_files = sorted(path for path in package_root.rglob("*.py") if path.is_file())
    required = {"__init__.py", "api.py", "cli.py", "mechanics.py", "fields.py"}
    missing = sorted(required - {path.name for path in package_files})
    if missing:
        raise AgentError("CODE_PROVENANCE_UNAVAILABLE", f"agent source file(s) are missing: {', '.join(missing)}")
    result = {}
    for path in core_sources:
        result[path.name] = _sha256(Path(path))
    for path in package_files:
        relative = path.relative_to(package_root).as_posix()
        result[f"straintrace_agent/{relative}"] = _sha256(path)
    return result


def _current_pointer(root: Path) -> dict[str, Any]:
    return _read_json(root / "CURRENT.json")


def _inventory_fresh(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    current = _scan_inventory(config, read_dimensions=False)
    return current, _inventory_fingerprint(config, current)


def _inventory_fingerprint(config: Mapping[str, Any], current: Sequence[Mapping[str, Any]]) -> str:
    """Bind an already hashed image inventory to its auxiliary source files."""
    auxiliary: dict[str, str] = {}
    for key in ("frame_map_csv", "mask_file"):
        if config.get(key):
            path = Path(str(config[key]))
            if not path.is_file():
                raise AgentError("INPUT_FILE_MISSING", f"configured input is missing: {path}", path=f"$.{key}")
            auxiliary[key] = _sha256(path)
    loads = config.get("loads")
    if loads:
        load_path = Path(str(loads["path"]))
        if not load_path.is_file():
            raise AgentError("LOAD_FILE_MISSING", f"load data file is missing: {load_path}", path="$.loads.path")
        current_load_sha = _sha256(load_path)
    else:
        current_load_sha = None
    fingerprint = _hash_json({
        "frames": [{key: item.get(key) for key in ("native_frame", "path", "relative_path", "sha256", "time_s")} for item in current],
        "loads_sha256": current_load_sha, "auxiliary_sha256": auxiliary,
    })
    return fingerprint


def _record_inventory(root: Path, records: Sequence[Mapping[str, Any]], fingerprint: str) -> None:
    inventory_path = root / "inventory.json"
    if inventory_path.exists():
        previous = _read_json(inventory_path)
        previous_fp = previous.get("input_fingerprint") or previous.get("inventory_sha256")
        if previous_fp != fingerprint:
            archive = root / "inventories" / f"inventory_{str(previous_fp)[:16]}.json"
            if not archive.exists():
                _atomic_json(archive, previous)
    _atomic_json(inventory_path, {"schema": "straintrace.agent-inventory/v1", "frames": _json_safe(list(records)), "input_fingerprint": fingerprint, "inventory_sha256": _hash_json(records), "updated_unix_s": time.time()})


def _load_image(path: str | os.PathLike[str]) -> np.ndarray:
    import ezdic_core as core
    try:
        image = np.asarray(core.read_gray_image(path))
    except Exception as exc:
        raise AgentError("IMAGE_DECODE_FAILED", f"cannot decode image {path}: {exc}", path=str(path)) from exc
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise AgentError("IMAGE_INVALID", f"decoded image is empty or non-finite: {path}", path=str(path))
    return image


def _load_specimen_mask(config: Mapping[str, Any], image_shape: Sequence[int]) -> np.ndarray | None:
    mask_path = config.get("mask_file")
    if not mask_path:
        if not config.get("exclusions_px"):
            return None
        mask = np.ones(tuple(int(v) for v in image_shape[:2]), dtype=bool)
    else:
        import ezdic_core as core
        if _sha256(Path(str(mask_path))) != config.get("mask_sha256"):
            raise AgentError("MASK_CHANGED", "mask file changed since project initialization", path="$.mask_file")
        try:
            mask_image = core.read_gray_image(mask_path)
        except Exception as exc:
            raise AgentError("MASK_UNREADABLE", f"cannot read specimen mask: {exc}", path=str(mask_path)) from exc
        if tuple(mask_image.shape[:2]) != tuple(image_shape[:2]):
            raise AgentError("MASK_DIMENSION_MISMATCH", "mask dimensions must match source images", path=str(mask_path))
        mask = np.asarray(mask_image) > 0
    for rect in config.get("exclusions_px") or []:
        x, y, width, height = (int(value) for value in rect)
        if x + width > mask.shape[1] or y + height > mask.shape[0]:
            raise AgentError("EXCLUSION_OUTSIDE_IMAGE", f"exclusion rectangle outside image: {rect}")
        mask[y:y + height, x:x + width] = False
    return mask


def _texture_roi(image: np.ndarray, subset_size: int) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    import cv2
    gray = np.asarray(image, dtype=np.float32)
    if min(gray.shape) < subset_size + 2:
        raise AgentError("IMAGE_TOO_SMALL", f"image is too small for subset_size={subset_size}")
    scale = min(1.0, 1024.0 / max(gray.shape))
    if scale < 1.0:
        analysis = cv2.resize(gray, (max(1, int(round(gray.shape[1] * scale))), max(1, int(round(gray.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    else:
        analysis = gray
    window = max(9, min(31, (subset_size // 2) * 2 + 1))
    mean = cv2.boxFilter(analysis, ddepth=-1, ksize=(window, window), normalize=True)
    mean_square = cv2.boxFilter(analysis * analysis, ddepth=-1, ksize=(window, window), normalize=True)
    local_std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    q50, q80, q95 = (float(value) for value in np.quantile(local_std, [0.50, 0.80, 0.95]))
    if not math.isfinite(q95) or q95 <= max(1e-6, q50 * 1.01):
        raise AgentError("TEXTURE_INSUFFICIENT", "no usable spatial texture variation was detected")
    threshold = max(q50 + 0.25 * (q80 - q50), q95 * 0.18)
    active = local_std >= threshold
    fraction = float(active.mean())
    if fraction < 0.03:
        raise AgentError("TEXTURE_INSUFFICIENT", f"only {fraction:.3%} of the image passes the texture screen")
    # When texture fills the view, preserve the full field. Otherwise the tight
    # box is a computational region proposal, not a segmented specimen mask.
    if fraction >= 0.72:
        x0, y0, x1, y1 = 0, 0, analysis.shape[1], analysis.shape[0]
        method = "full_view_texture_coverage"
    else:
        yy, xx = np.nonzero(active)
        pad = max(0, int(round((subset_size // 2) * scale)))
        x0 = max(0, int(xx.min()) - pad)
        y0 = max(0, int(yy.min()) - pad)
        x1 = min(analysis.shape[1], int(xx.max()) + pad + 1)
        y1 = min(analysis.shape[0], int(yy.max()) + pad + 1)
        method = "local_texture_quantile_bounding_box"
    if scale < 1.0:
        left = int(math.floor(x0 / scale))
        top = int(math.floor(y0 / scale))
        right = min(gray.shape[1], int(math.ceil(x1 / scale)))
        bottom = min(gray.shape[0], int(math.ceil(y1 / scale)))
    else:
        left, top, right, bottom = x0, y0, x1, y1
    roi = (left, top, right - left, bottom - top)
    return roi, {"method": method, "window_px": window, "downsample_scale": scale, "threshold_local_std": threshold, "texture_fraction": fraction, "local_std_quantiles": {"q50": q50, "q80": q80, "q95": q95}}


def _safe_roi_for_image(roi: Sequence[int], image_shape: Sequence[int], *, search_radius_px: int) -> tuple[int, int, int, int]:
    """Keep translation search plus spline support inside image bounds."""
    x, y, width, height = (int(value) for value in roi)
    margin = int(search_radius_px) + 3  # interpolation support; no numerical thresholds changed.
    x0, y0 = max(x, margin), max(y, margin)
    x1, y1 = min(x + width, int(image_shape[1]) - margin), min(y + height, int(image_shape[0]) - margin)
    return x0, y0, x1 - x0, y1 - y0


def _reference_record(config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    requested = config.get("reference_native_frame")
    if requested is None:
        requested = records[0]["native_frame"]
    found = next((row for row in records if int(row["native_frame"]) == int(requested)), None)
    if found is None:
        raise AgentError("REFERENCE_FRAME_UNKNOWN", f"native reference frame {requested} is absent")
    return found


def _automatic_reference(config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], np.ndarray, dict[str, Any]]:
    if config.get("reference_native_frame") is not None:
        record = _reference_record(config, records)
        image = _load_image(record["path"])
        roi, texture = _texture_roi(image, 21)
        return record, image, {"selection": "user_selected", **texture}
    failures = []
    first = records[0]
    # The first readable frame with spatial texture is a defensible fixed
    # reference. Skips retain their native IDs/times and appear in inventory.
    for record in records[: min(len(records), 12)]:
        try:
            image = _load_image(record["path"])
            _, texture = _texture_roi(image, 21)
            return record, image, {"selection": "first_readable_textured_frame", "initial_inventory_frame": int(first["native_frame"]), **texture}
        except AgentError as exc:
            failures.append({"native_frame": int(record["native_frame"]), "code": exc.code, "message": exc.message})
    raise AgentError("REFERENCE_FRAME_UNAVAILABLE", "no readable textured reference found among the first 12 frames", details=failures)


def _pilot_frame(records: Sequence[Mapping[str, Any]], reference: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    for row in records:
        if int(row["native_frame"]) == int(reference["native_frame"]):
            continue
        try:
            image = _load_image(row["path"])
            if tuple(image.shape) != tuple(reference.get("shape_yx") or image.shape):
                continue
            return row, None
        except AgentError as exc:
            continue
    return None, "no readable non-reference frame is available for bounded parameter trials"


@contextmanager
def _project_lock(root: Path, *, purpose: str):
    lock = root / ".agent.lock"
    payload = {"pid": os.getpid(), "host": socket.gethostname(), "process_start_token": _process_start_token(os.getpid()), "purpose": purpose, "created_unix_s": time.time(), "owner_token": uuid.uuid4().hex}
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        existing = None
        try:
            existing = _read_json(lock)
        except AgentError:
            pass
        raise AgentError("PROJECT_LOCKED", "another writer owns this project; stale-lock recovery is an explicit operation", path=str(lock), details=existing) from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        try:
            current = _read_json(lock)
            if current.get("owner_token") == payload["owner_token"]:
                lock.unlink()
        except (AgentError, FileNotFoundError):
            pass


def _process_start_token(pid: int) -> str | None:
    if os.name != "nt":
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
            return stat[21]
        except (OSError, IndexError):
            return None
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4))
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
    try:
        if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        stamp = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        return f"{stamp:016x}"
    finally:
        kernel32.CloseHandle(handle)


def _pid_state(pid: int, expected_start_token: str | None) -> bool | None:
    """Return alive/dead/unknown without opening a process for termination."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            return False if error == 87 else None  # ERROR_INVALID_PARAMETER means no such process.
        code = wintypes.DWORD()
        try:
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            if code.value != 259:  # STILL_ACTIVE
                return False
            actual_start = _process_start_token(pid)
            if actual_start is None or expected_start_token is None:
                return None
            if actual_start != expected_start_token:
                return False  # PID has been reused by a different process.
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return None
    return True


def recover_lock(project: str | os.PathLike[str], *, expected_pid: int) -> dict[str, Any]:
    root = _project_root(project)
    lock = root / ".agent.lock"
    if type(expected_pid) is not int or expected_pid <= 0:
        raise AgentError("LOCK_PID_INVALID", "expected_pid must be a positive integer", path="$.expected_pid")
    record = _read_json(lock)
    if record.get("host") != socket.gethostname():
        raise AgentError("LOCK_REMOTE_OWNER", "lock host differs from this machine; remote ownership cannot be verified", path=str(lock))
    if record.get("pid") != expected_pid:
        raise AgentError("LOCK_PID_MISMATCH", "recorded lock PID does not match expected_pid", path="$.expected_pid", details={"recorded_pid": record.get("pid")})
    state = _pid_state(expected_pid, record.get("process_start_token"))
    if state is True:
        raise AgentError("LOCK_OWNER_ALIVE", f"PID {expected_pid} is still alive; lock cannot be recovered", path=str(lock))
    if state is None:
        raise AgentError("LOCK_OWNER_UNKNOWN", f"cannot safely determine whether PID {expected_pid} is alive", path=str(lock))
    # Re-read immediately before unlink to avoid deleting a replacement lock.
    if _read_json(lock).get("owner_token") != record.get("owner_token"):
        raise AgentError("LOCK_CHANGED", "lock owner changed during recovery", path=str(lock))
    lock.unlink()
    return {"project": str(root), "recovered_pid": expected_pid, "recovered": True}


def _write_progress(run_root: Path, event: Mapping[str, Any], callback: Callable[[dict[str, Any]], None] | None = None) -> None:
    record = {"time_unix_s": time.time(), **_json_safe(dict(event))}
    path = run_root / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
    _atomic_json(run_root / "progress.json", record)
    if callback is not None:
        callback(record)


def _append_frame_event(run_root: Path, event: Mapping[str, Any], callback: Callable[[dict[str, Any]], None] | None = None) -> None:
    _write_progress(run_root, event, callback)


def _plan_path(root: Path, pointer: Mapping[str, Any]) -> Path:
    relative = pointer.get("current_plan")
    if not relative:
        raise AgentError("PLAN_REQUIRED", "run plan_project before running DIC")
    path = root / str(relative)
    if not path.is_file():
        raise AgentError("PLAN_MISSING", f"current analysis plan is missing: {path}", path=str(path))
    return path


def plan_project(project: str | os.PathLike[str], *, progress: Callable[[dict[str, Any]], None] | None = None, force: bool = False) -> dict[str, Any]:
    """Propose a texture ROI and choose among a small, fixed solver trial set."""
    root = _project_root(project)
    config = _read_config(root)
    with _project_lock(root, purpose="plan"):
        records, input_sha = _inventory_fresh(root, config)
        reference_record, reference, reference_selection = _automatic_reference(config, records)
        try:
            detected_roi, texture = _texture_roi(reference, 21)
        except AgentError as exc:
            if not config.get("roi_override_px"):
                raise
            detected_roi, texture = tuple(config["roi_override_px"]), {"method": "user_roi_override_texture_not_assessed", "assessment_error": exc.as_dict()}
        if config.get("roi_override_px"):
            roi = tuple(config["roi_override_px"])
            texture = {**texture, "selected_by": "explicit_user_roi_override"}
        else:
            roi = detected_roi
        x, y, width, height = roi
        specimen_mask = _load_specimen_mask(config, reference.shape)
        if specimen_mask is not None and not config.get("roi_override_px"):
            supported_rows = np.flatnonzero(np.any(specimen_mask, axis=1))
            supported_cols = np.flatnonzero(np.any(specimen_mask, axis=0))
            if supported_cols.size and supported_rows.size:
                mx0, my0 = int(supported_cols[0]), int(supported_rows[0])
                mx1, my1 = int(supported_cols[-1]) + 1, int(supported_rows[-1]) + 1
                x0, y0 = max(x, mx0), max(y, my0)
                x1, y1 = min(x + width, mx1), min(y + height, my1)
                if x1 > x0 and y1 > y0:
                    roi = (x0, y0, x1 - x0, y1 - y0)
                    x, y, width, height = roi
                    texture = {**texture, "selected_by": "intersection_of_texture_and_user_mask_bounds"}
        if x + width > reference.shape[1] or y + height > reference.shape[0]:
            raise AgentError("ROI_OUTSIDE_IMAGE", f"analysis rectangle {roi} exceeds reference image shape {reference.shape}")
        if width < 35 or height < 35:
            raise AgentError("ROI_TOO_SMALL", f"texture analysis candidate {roi} cannot support a 3x3 POI field")
        pilot_record, pilot_problem = _pilot_frame(records, reference_record)
        candidates: list[dict[str, Any]] = []
        import ezdic_core as core
        if pilot_record is not None:
            try:
                pilot = _load_image(pilot_record["path"])
                # Bounded central window avoids an unbounded planning pass on
                # very large images while keeping the solver's real semantics.
                pilot_w = min(width, 160)
                pilot_h = min(height, 160)
                px = x + max(0, (width - pilot_w) // 2)
                py = y + max(0, (height - pilot_h) // 2)
                pilot_roi = (px, py, pilot_w, pilot_h)
                pilot_roi = _safe_roi_for_image(pilot_roi, reference.shape, search_radius_px=15)
                pilot_step = int(config.get("step_hint_px") or 8)
                subset_values = [size for size in (21, 31) if size + 2 * pilot_step <= min(pilot_w, pilot_h)]
                if not subset_values:
                    subset_values = [15] if min(pilot_w, pilot_h) >= 15 + 2 * pilot_step else []
                for subset in subset_values[:2]:
                    try:
                        field = core.run_2d_dic(
                            reference, pilot, pilot_roi, subset_size=subset, step=pilot_step,
                            solver=core.DIC_SOLVER_ICGN, strain_window=5,
                            search_radius=max(8, subset // 2),
                            specimen_mask=specimen_mask,
                        )
                        quality = field.get("quality_summary") or core.field_quality_summary(field)
                        candidates.append({
                            "subset_size_px": int(subset), "step_px": pilot_step, "strain_window_poi": 5,
                            "solver": core.DIC_SOLVER_ICGN,
                            "scientific_ok": bool(quality.get("scientific_ok", False)),
                            "correlation_valid_fraction": float(quality.get("correlation_valid_fraction", 0.0)),
                            "strain_valid_fraction": float(quality.get("strain_valid_fraction", 0.0)),
                            "quality_reasons": list(quality.get("scientific_reasons") or []),
                            "thresholds": quality.get("thresholds"),
                        })
                    except Exception as exc:
                        candidates.append({"subset_size_px": int(subset), "step_px": pilot_step, "strain_window_poi": 5,
                                           "solver": core.DIC_SOLVER_ICGN, "scientific_ok": False,
                                           "correlation_valid_fraction": 0.0, "strain_valid_fraction": 0.0,
                                           "quality_reasons": ["pilot_error"], "error": f"{type(exc).__name__}: {exc}"})
                pilot_problem = None if candidates else "no candidate subset fits a 3x3 pilot grid"
            except AgentError as exc:
                pilot_problem = exc.message
        if candidates:
            chosen = max(candidates, key=lambda item: (bool(item["scientific_ok"]), min(item["correlation_valid_fraction"], item["strain_valid_fraction"]), -item["subset_size_px"]))
            plan_status = "ready" if chosen["scientific_ok"] else "review_required"
        else:
            fallback_step = int(config.get("step_hint_px") or 8)
            fallback_subset = 21
            if min(width, height) < fallback_subset + 2 * fallback_step:
                fallback_subset = 15
                fallback_step = max(1, min(fallback_step, (min(width, height) - fallback_subset) // 2))
            chosen = {"subset_size_px": fallback_subset, "step_px": fallback_step, "strain_window_poi": 5, "solver": core.DIC_SOLVER_ICGN}
            plan_status = "review_required"
        chosen["search_radius_px"] = max(8, int(chosen["subset_size_px"]) // 2)
        roi_before_support_crop = roi
        roi = _safe_roi_for_image(roi, reference.shape, search_radius_px=int(chosen["search_radius_px"]))
        x, y, width, height = roi
        if width < int(chosen["subset_size_px"]) + 2 * int(chosen["step_px"]) or height < int(chosen["subset_size_px"]) + 2 * int(chosen["step_px"]):
            plan_status = "review_required"
        gauge_reference_geometry = _resolve_gauge_geometry(
            config.get("gauges") or [], roi, reference.shape,
            int(chosen["subset_size_px"]), int(chosen["step_px"]), specimen_mask,
            calibration=config.get("calibration"),
        )
        if any(item.get("reference_status") != "resolved" for item in gauge_reference_geometry):
            plan_status = "review_required"
        chosen_subset = int(chosen["subset_size_px"])
        chosen_step = int(chosen["step_px"])
        plan = {
            "schema": "straintrace.agent-plan/v1", "plan_id": uuid.uuid4().hex,
            "created_unix_s": time.time(), "project_id": config["project_id"], "status": plan_status,
            "input_fingerprint": input_sha, "configuration_sha256": _hash_json(config), "code_sha256": _code_fingerprint(),
            "reference_native_frame": int(reference_record["native_frame"]),
            "reference_path": reference_record["path"],
            "reference_selection": reference_selection,
            "analysis_rectangle_candidate_px": list(roi), "texture_roi_bbox_px": list(roi_before_support_crop), "analysis_rectangle_status": "candidate_not_physical_specimen_confirmation",
            "specimen_geometry_confirmed": False, "roi_detection": texture,
            "mask_file": config.get("mask_file"), "mask_sha256": config.get("mask_sha256"),
            "exclusions_px": config.get("exclusions_px", []),
            "pilot_native_frame": int(pilot_record["native_frame"]) if pilot_record else None,
            "pilot_rectangle_px": list(pilot_roi) if pilot_record and "pilot_roi" in locals() else None,
            "pilot_issue": pilot_problem, "candidate_trials": candidates,
            "automatic_gauge_candidates": _auto_gauge_candidates(roi, chosen_subset, chosen_step),
            "gauge_reference_geometry": gauge_reference_geometry,
            "selected_parameters": chosen,
            "parameter_decision": "bounded central-pilot comparison; existing core thresholds unchanged; no threshold relaxation",
            "strain_support_span_px": int((int(chosen["strain_window_poi"]) - 1) * chosen_step),
            "subset_support_span_px": int(chosen_subset + (int(chosen["strain_window_poi"]) - 1) * chosen_step),
            "correlation_search_radius_px": int(chosen["search_radius_px"]),
            "environment": __import__("ezdic_core").collect_environment(),
        }
        from .regions import recommend_regions
        plan["region_recommendations"] = recommend_regions(reference, chosen_subset, chosen_step, specimen_mask=specimen_mask)
        plan_sha = _hash_json(plan)
        plan["plan_sha256"] = plan_sha
        plans_root = root / "plans"
        plans_root.mkdir(exist_ok=True)
        prior = sorted(plans_root.glob("plan_v*.json"))
        version = len(prior) + 1
        plan_file = plans_root / f"plan_v{version:03d}.json"
        _atomic_json(plan_file, plan)
        pointer = _current_pointer(root)
        _record_inventory(root, records, input_sha)
        pointer["current_plan"] = plan_file.relative_to(root).as_posix()
        pointer["current_run"] = None
        _atomic_json(root / "CURRENT.json", pointer)
        if progress:
            progress({"event": "plan_complete", "status": plan_status, "candidate_count": len(candidates), "plan": str(plan_file)})
        return plan


def _serialize_field(path: Path, field: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in field.items():
        if isinstance(value, np.ndarray):
            if key == "specimen_mask":
                continue
            array = np.asarray(value)
            if array.dtype.kind == "O":
                array = array.astype(str)
            if array.dtype.kind not in "biufSU":
                continue
            arrays[str(key)] = array
    if not arrays:
        raise AgentError("CHECKPOINT_EMPTY", "DIC solver returned no persistable arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    metadata_path = path.with_suffix(".json")
    record = {"schema": "straintrace.agent-checkpoint/v1", **_json_safe(dict(metadata)), "checkpoint": path.name, "checkpoint_sha256": _sha256(path), "array_names": sorted(arrays)}
    _atomic_json(metadata_path, record)
    return {"path": path, "metadata_path": metadata_path, "sha256": record["checkpoint_sha256"], "metadata": record}


def _load_field(checkpoint: Path, metadata_path: Path | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if metadata_path is None:
        metadata_path = checkpoint.with_suffix(".json")
    metadata = _read_json(metadata_path)
    actual_sha = _sha256(checkpoint)
    if actual_sha != metadata.get("checkpoint_sha256"):
        raise AgentError("CHECKPOINT_CHECKSUM_MISMATCH", f"checkpoint checksum mismatch: {checkpoint}", path=str(checkpoint))
    try:
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise AgentError("CHECKPOINT_INVALID", f"cannot load safe NPZ checkpoint: {exc}", path=str(checkpoint)) from exc
    return arrays, metadata


















def _run_fingerprint(config: Mapping[str, Any], plan: Mapping[str, Any], input_sha: str, code: Mapping[str, str]) -> str:
    return _hash_json({"config": config, "plan_sha256": plan.get("plan_sha256"), "input_fingerprint": input_sha, "code_sha256": code})


def _frame_key(native_frame: int) -> str:
    return f"frame_native_{native_frame:08d}"


def _checkpoint_valid(
    run_root: Path, record: Mapping[str, Any], frame_position: int,
    reference_native_frame: int, run_fingerprint: str, input_sha: str,
    code_sha: Mapping[str, str], expected_parameters: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    native_frame = int(record["native_frame"])
    checkpoint = run_root / "checkpoints" / f"{_frame_key(native_frame)}.npz"
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.is_file():
        return None

    def matches_expected_identity(metadata: Mapping[str, Any]) -> bool:
        if metadata_path.name != f"{_frame_key(native_frame)}.json":
            return False
        if (metadata.get("native_frame") != native_frame
                or metadata.get("frame_position") != frame_position
                or metadata.get("source_path") != record.get("path")
                or metadata.get("source_sha256") != record.get("sha256")
                or metadata.get("reference_native_frame") != reference_native_frame
                or metadata.get("run_fingerprint") != run_fingerprint
                or metadata.get("input_fingerprint") != input_sha
                or metadata.get("code_sha256") != code_sha
                or not _solver_parameters_match(metadata.get("parameters"), expected_parameters)):
            return False
        expected_time = record.get("time_s")
        if expected_time is None:
            if metadata.get("time_s") is not None:
                return False
        else:
            try:
                actual_time = float(metadata.get("time_s"))
                if not math.isfinite(actual_time) or not math.isclose(actual_time, float(expected_time), rel_tol=0.0, abs_tol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
        summary = metadata.get("summary_row")
        if not isinstance(summary, Mapping):
            return False
        if (summary.get("native_frame") != native_frame
                or summary.get("frame_position_1based") != frame_position
                or summary.get("reference_native_frame") != reference_native_frame
                or summary.get("status") != metadata.get("status")):
            return False
        summary_time = summary.get("time_s")
        if expected_time is None:
            if summary_time is not None:
                return False
        else:
            try:
                if not math.isclose(float(summary_time), float(expected_time), rel_tol=0.0, abs_tol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
        if metadata.get("status") == "failed":
            return metadata.get("checkpoint") is None and summary.get("checkpoint") is None
        return (metadata.get("status") in {"complete", "quality_failed"}
                and metadata.get("checkpoint") == checkpoint.name
                and summary.get("checkpoint") == f"checkpoints/{checkpoint.name}")

    try:
        metadata = _read_json(metadata_path)
        if not matches_expected_identity(metadata):
            return None
        if metadata.get("status") == "failed":
            # Deterministic input failures remain reusable. Runtime failures
            # are retried on resume so transient filesystem/solver errors can
            # recover without discarding other valid frame checkpoints.
            if metadata.get("retryable") is True:
                return None
            return metadata_path, metadata
        if not checkpoint.is_file():
            return None
        arrays, metadata = _load_field(checkpoint, metadata_path)
        if not matches_expected_identity(metadata) or not arrays:
            return None
        return checkpoint, metadata
    except AgentError:
        return None


def _frame_failure(run_root: Path, record: Mapping[str, Any], run_fingerprint: str, input_sha: str, code_sha: Mapping[str, str], error: AgentError, *, frame_position: int | None = None, reference_native_frame: int | None = None, retryable: bool = False, parameters: Mapping[str, Any] | None = None, mechanics_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    native = int(record["native_frame"])
    path = run_root / "checkpoints" / f"{_frame_key(native)}.json"
    row = {"frame_position_1based": frame_position, "native_frame": native, "time_s": record["time_s"], "reference_native_frame": reference_native_frame, "status": "failed", "quality_status": "unavailable", "scientific_ok": False, "failure_code": error.code, "failure_reason": error.message, "checkpoint": None, "force_N": None, "stress_MPa": None, "mechanics_status": "not_evaluated", **dict(mechanics_row or {})}
    payload = {
        "schema": "straintrace.agent-checkpoint/v1", "native_frame": native,
        "frame_position": frame_position, "time_s": record.get("time_s"),
        "reference_native_frame": reference_native_frame,
        "source_path": record["path"], "source_sha256": record["sha256"],
        "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha,
        "code_sha256": code_sha, "status": "failed", "checkpoint": None,
        "parameters": dict(parameters or {}),
        "retryable": bool(retryable), "failure_class": "runtime" if retryable else "deterministic",
        "error": error.as_dict(), "summary_row": row,
    }
    _atomic_json(path, payload)
    return row


def _load_run_pointer(root: Path) -> tuple[dict[str, Any], Path]:
    pointer = _current_pointer(root)
    relative = pointer.get("current_run")
    if not relative:
        raise AgentError("RUN_REQUIRED", "project does not have a current run")
    run_root = root / str(relative)
    if not run_root.is_dir():
        raise AgentError("RUN_MISSING", f"current run directory is missing: {run_root}", path=str(run_root))
    return pointer, run_root


def _require_current_run_binding(
    root: Path, config: Mapping[str, Any], run_root: Path,
    plan: Mapping[str, Any], run_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    """Reject derived-output writes when their run no longer binds current inputs."""
    plan_copy = dict(plan)
    saved_plan_sha = _validate_plan_checksum(plan_copy)
    records, input_sha = _inventory_fresh(root, config)
    if (plan.get("input_fingerprint") != input_sha
            or run_state.get("input_fingerprint") != input_sha):
        raise AgentError("RUN_STALE_INPUT", "raw images, load data, frame map, or mask changed since this run")
    config_sha = _hash_json(config)
    if plan.get("configuration_sha256") != config_sha:
        raise AgentError("RUN_STALE_CONFIG", "project configuration changed since this run")
    if run_root.joinpath("manifest.json").is_file():
        manifest = _read_json(run_root / "manifest.json")
    else:
        raise AgentError("RUN_MANIFEST_MISSING", "the current run has no manifest", path=str(run_root / "manifest.json"))
    if manifest.get("configuration_sha256") != config_sha:
        raise AgentError("RUN_STALE_CONFIG", "run manifest does not match the current project configuration")
    if manifest.get("plan_sha256") != saved_plan_sha:
        raise AgentError("RUN_STALE_PLAN", "run manifest does not match the saved analysis plan")
    if manifest.get("input_fingerprint") != input_sha:
        raise AgentError("RUN_STALE_INPUT", "run manifest does not match current source inputs")
    current_code = _code_fingerprint()
    if plan.get("code_sha256") != current_code or run_state.get("code_sha256") != current_code or manifest.get("code_sha256") != current_code:
        raise AgentError("RUN_STALE_CODE", "analysis code changed since this run")
    import ezdic_core as core
    current_environment = core.collect_environment()
    if (plan.get("environment") != current_environment
            or manifest.get("environment") != current_environment):
        raise AgentError("RUN_STALE_ENVIRONMENT", "runtime packages or platform changed since this run")
    expected_fingerprint = _run_fingerprint(config, plan, input_sha, current_code)
    if (run_state.get("run_fingerprint") != expected_fingerprint
            or manifest.get("run_fingerprint") != expected_fingerprint
            or run_root.name != str(run_state.get("run_id"))):
        raise AgentError("RUN_STALE_BINDING", "run identity does not match current inputs, configuration, and code")
    return records, input_sha, current_code


def run(
    project: str | os.PathLike[str], *, progress: Callable[[dict[str, Any]], None] | None = None,
    make_plots: bool = True, field_csv: bool = False,
) -> dict[str, Any]:
    """Stream fixed-reference frame calculations and rebuild deliverables."""
    root = _project_root(project)
    config = _read_config(root)
    with _project_lock(root, purpose="run"):
        records, input_sha = _inventory_fresh(root, config)
        pointer = _current_pointer(root)
        plan_path = _plan_path(root, pointer)
        plan = _read_json(plan_path)
        _validate_plan_checksum(plan)
        code_sha = _code_fingerprint()
        if plan.get("input_fingerprint") != input_sha:
            raise AgentError("PLAN_STALE_INPUT", "source image, load, frame-map, or mask identity changed; create a fresh plan")
        if plan.get("configuration_sha256") != _hash_json(config):
            raise AgentError("PLAN_STALE_CONFIG", "project parameters changed after planning; create a fresh plan")
        if plan.get("code_sha256") != code_sha:
            raise AgentError("PLAN_STALE_CODE", "analysis code changed after planning; create a fresh plan")
        import ezdic_core as core
        current_environment = core.collect_environment()
        if plan.get("environment") != current_environment:
            raise AgentError("PLAN_STALE_ENVIRONMENT", "runtime packages or platform changed after planning; create a fresh plan")
        if plan.get("status") != "ready":
            raise AgentError(
                "PLAN_REVIEW_REQUIRED",
                "saved plan is not computationally ready; inspect its ROI, pilot, and parameter candidates before running",
                details={
                    "plan_status": plan.get("status"),
                    "pilot_issue": plan.get("pilot_issue"),
                    "analysis_rectangle_candidate_px": plan.get("analysis_rectangle_candidate_px"),
                    "candidate_trials": plan.get("candidate_trials", []),
                },
            )
        _record_inventory(root, records, input_sha)
        run_fingerprint = _run_fingerprint(config, plan, input_sha, code_sha)
        run_id = f"run-{run_fingerprint[:20]}"
        run_root = root / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "checkpoints").mkdir(exist_ok=True)
        (run_root / "plots").mkdir(exist_ok=True)
        pointer["current_run"] = run_root.relative_to(root).as_posix()
        _atomic_json(root / "CURRENT.json", pointer)
        cancel_file = root / "CANCEL.request"
        try:
            cancel_file.unlink()
        except FileNotFoundError:
            pass
        _atomic_json(run_root / "run_state.json", {"schema": RUN_SCHEMA, "status": "running", "run_id": run_id, "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha, "code_sha256": code_sha, "started_unix_s": time.time(), "completed_frames": 0})
        reference_record = next(row for row in records if int(row["native_frame"]) == int(plan["reference_native_frame"]))
        reference_sha_before = _sha256(Path(reference_record["path"]))
        reference = None
        try:
            reference = _load_image(reference_record["path"])
            if reference_sha_before != reference_record["sha256"]:
                raise AgentError("INPUT_CHANGED_DURING_RUN", "reference image changed after inventory")
        except AgentError as exc:
            raise AgentError("REFERENCE_FRAME_UNREADABLE", exc.message, path=str(reference_record["path"])) from exc
        roi = tuple(int(value) for value in plan["analysis_rectangle_candidate_px"])
        parameters = dict(plan["selected_parameters"])
        specimen_mask = _load_specimen_mask(config, reference.shape)
        mechanics, mechanics_status = _load_mechanics(config)
        displacement_reference = None
        displacement_reference_alignment = None
        if mechanics is not None and mechanics_status.get("status") == "aligned" and "displacement_time_s" in mechanics:
            displacement_reference_alignment = _interpolate_bounded_detail(
                float(reference_record["time_s"]), mechanics["displacement_time_s"], mechanics["displacement_mm"],
                float(config["loads"]["max_gap_s"]), mechanics.get("displacement_source_rows"),
            )
            displacement_reference = displacement_reference_alignment.get("value")
        if mechanics_status.get("status") == "unavailable" and mechanics_status.get("reason") not in {"clock_not_confirmed"}:
            # Keep image analysis independent of optional/invalid mechanics.
            pass
        _write_progress(run_root, {"event": "run_started", "run_id": run_id, "frame_count": len(records), "reference_native_frame": int(reference_record["native_frame"]), "run_fingerprint": run_fingerprint}, progress)
        rows: list[dict[str, Any]] = []
        completed = 0
        for position, record in enumerate(records, 1):
            native = int(record["native_frame"])
            existing = _checkpoint_valid(
                run_root, record, position, int(reference_record["native_frame"]),
                run_fingerprint, input_sha, code_sha, parameters,
            )
            if existing is not None:
                meta = existing[1]
                row = dict(meta.get("summary_row") or {})
                if row:
                    row["frame_position_1based"] = position
                    rows.append(row)
                    completed += 1
                    _write_progress(run_root, {"event": "frame_reused", "native_frame": native, "position": position, "total": len(records)}, progress)
                    continue
            if cancel_file.exists():
                _atomic_json(run_root / "run_state.json", {"schema": RUN_SCHEMA, "status": "cancelled", "run_id": run_id, "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha, "code_sha256": code_sha, "completed_frames": completed, "updated_unix_s": time.time()})
                _write_progress(run_root, {"event": "run_cancelled", "completed_frames": completed, "remaining_frames": len(records) - position + 1}, progress)
                _build_outputs(root, run_root, config, plan, records, rows, mechanics_status, run_fingerprint, input_sha, code_sha, field_csv=False, plots=False)
                return {"project": str(root), "run_id": run_id, "status": "cancelled", "scientific_ok": False, "numerical_quality_passed": False, "scientific_acceptance": "not_assessed", "completed_frames": completed, "total_frames": len(records), "run_dir": str(run_root)}
            _write_progress(run_root, {"event": "frame_started", "native_frame": native, "position": position, "total": len(records)}, progress)
            try:
                if record.get("status") != "pending":
                    raise AgentError("FRAME_UNREADABLE", record.get("read_error") or record.get("status") or "frame is not readable", path=record["path"])
                if _sha256(Path(record["path"])) != record["sha256"]:
                    raise AgentError("INPUT_CHANGED_DURING_RUN", "source image changed after inventory", path=record["path"])
                deformed = _load_image(record["path"])
                if tuple(deformed.shape) != tuple(reference.shape):
                    raise AgentError("FRAME_DIMENSION_MISMATCH", f"shape {deformed.shape} differs from reference {reference.shape}", path=record["path"])
                import ezdic_core as core
                last_progress = 0
                def dic_progress(done: int, total: int) -> None:
                    nonlocal last_progress
                    if cancel_file.exists():
                        raise _RunCancelled("cancel requested during DIC frame")
                    interval = max(1, int(total) // 20)
                    if done == total or done - last_progress >= interval:
                        last_progress = done
                        _write_progress(run_root, {"event": "dic_progress", "native_frame": native, "position": position, "total_frames": len(records), "points_done": int(done), "points_total": int(total), "fraction": float(done / total) if total else 1.0}, progress)
                field = core.run_2d_dic(
                    reference, deformed, roi,
                    subset_size=int(parameters["subset_size_px"]), step=int(parameters["step_px"]),
                    solver=str(parameters["solver"]), strain_window=int(parameters["strain_window_poi"]),
                    search_radius=int(parameters["search_radius_px"]),
                    specimen_mask=specimen_mask,
                    progress_callback=dic_progress,
                )
                quality = field.get("quality_summary") or core.field_quality_summary(field)
                field["provenance"] = {
                    "project_id": config["project_id"], "run_id": run_id,
                    "reference_frame_1based": records.index(reference_record) + 1,
                    "reference_native_frame": int(reference_record["native_frame"]),
                    "reference_filename": Path(reference_record["path"]).name,
                    "frame_global_1based": position, "native_frame": native,
                    "frame_filename": Path(record["path"]).name,
                    "time_s": record.get("time_s"), "roi_candidate_px": list(roi),
                    "analysis_rectangle_status": "candidate_not_physical_specimen_confirmation",
                    "input_sha256": record["sha256"],
                }
                valid = np.asarray(field.get("valid", []), dtype=bool)
                strain_valid = np.asarray(field.get("strain_valid", []), dtype=bool)
                exx = np.asarray(field.get("exx", []), dtype=float)
                finite_exx = exx[np.isfinite(exx) & strain_valid] if exx.size == strain_valid.size else np.asarray([], dtype=float)
                mechanics_row = _mechanics_at(record.get("time_s"), mechanics, config, displacement_reference_mm=displacement_reference, displacement_reference_alignment=displacement_reference_alignment)
                quality_ok = bool(quality.get("scientific_ok", False)) and bool(core.fullfield_field_has_finite_strain(field))
                status_text = "complete" if quality_ok else "quality_failed"
                gauge_rows = _measure_gauges(field, plan.get("gauge_reference_geometry") or [], quality_ok=quality_ok)
                gauge_summary = _gauge_group_summary(gauge_rows)
                summary_row = {
                    "frame_position_1based": position, "native_frame": native, "time_s": record.get("time_s"),
                    "reference_native_frame": int(reference_record["native_frame"]),
                    "status": status_text, "quality_status": "pass" if quality_ok else "fail",
                    "scientific_ok": quality_ok,
                    "eligible_point_count": int(quality.get("eligible_point_count", 0)),
                    "correlation_valid_fraction": float(quality.get("correlation_valid_fraction", 0.0)),
                    "strain_valid_fraction": float(quality.get("strain_valid_fraction", 0.0)),
                    "max_abs_local_exx": float(np.max(np.abs(finite_exx))) if quality_ok and finite_exx.size else None,
                    **gauge_summary,
                    "engineering_strain": gauge_summary["dic_virtual_gauge_engineering_strain"],
                    **mechanics_row,
                    "failure_code": None, "failure_reason": None,
                    "checkpoint": f"checkpoints/{_frame_key(native)}.npz",
                }
                checkpoint = run_root / "checkpoints" / f"{_frame_key(native)}.npz"
                meta = {
                    "native_frame": native, "frame_position": position, "time_s": record.get("time_s"),
                    "source_path": record["path"], "source_sha256": record["sha256"],
                    "reference_native_frame": int(reference_record["native_frame"]), "run_fingerprint": run_fingerprint,
                    "input_fingerprint": input_sha, "code_sha256": code_sha, "status": status_text,
                    "quality": quality, "parameters": parameters, "roi_candidate_px": list(roi),
                    "gauge_measurements": gauge_rows,
                    "summary_row": summary_row,
                }
                _serialize_field(checkpoint, field, meta)
                rows.append(summary_row)
                completed += 1
                _write_progress(run_root, {"event": "frame_complete", "native_frame": native, "position": position, "status": status_text, "quality": quality}, progress)
            except _RunCancelled:
                _atomic_json(run_root / "run_state.json", {"schema": RUN_SCHEMA, "status": "cancelled", "run_id": run_id, "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha, "code_sha256": code_sha, "completed_frames": len(rows), "updated_unix_s": time.time()})
                _write_progress(run_root, {"event": "run_cancelled", "completed_frames": len(rows), "remaining_frames": len(records) - position + 1}, progress)
                _build_outputs(root, run_root, config, plan, records, rows, mechanics_status, run_fingerprint, input_sha, code_sha, field_csv=False, plots=False)
                return {"project": str(root), "run_id": run_id, "status": "cancelled", "scientific_ok": False, "numerical_quality_passed": False, "scientific_acceptance": "not_assessed", "completed_frames": len(rows), "total_frames": len(records), "run_dir": str(run_root)}
            except AgentError as exc:
                mechanics_row = _mechanics_at(record.get("time_s"), mechanics, config, displacement_reference_mm=displacement_reference, displacement_reference_alignment=displacement_reference_alignment)
                row = _frame_failure(run_root, record, run_fingerprint, input_sha, code_sha, exc, frame_position=position, reference_native_frame=int(reference_record["native_frame"]), parameters=parameters, mechanics_row=mechanics_row)
                rows.append(row)
                _write_progress(run_root, {"event": "frame_failed", "native_frame": native, "position": position, "error": exc.as_dict()}, progress)
            except Exception as exc:
                error = AgentError("DIC_FRAME_ERROR", f"{type(exc).__name__}: {exc}", path=record["path"])
                mechanics_row = _mechanics_at(record.get("time_s"), mechanics, config, displacement_reference_mm=displacement_reference, displacement_reference_alignment=displacement_reference_alignment)
                row = _frame_failure(run_root, record, run_fingerprint, input_sha, code_sha, error, frame_position=position, reference_native_frame=int(reference_record["native_frame"]), retryable=True, parameters=parameters, mechanics_row=mechanics_row)
                rows.append(row)
                _write_progress(run_root, {"event": "frame_failed", "native_frame": native, "position": position, "error": error.as_dict()}, progress)
        if _sha256(Path(reference_record["path"])) != reference_sha_before:
            raise AgentError("INPUT_CHANGED_DURING_RUN", "reference image changed during analysis")
        failed = sum(row.get("status") not in {"complete"} for row in rows)
        passed = sum(row.get("status") == "complete" for row in rows)
        run_status = "complete" if failed == 0 and passed > 0 else "partial" if passed > 0 else "failed"
        run_state = {"schema": RUN_SCHEMA, "status": run_status, "run_id": run_id, "run_fingerprint": run_fingerprint, "input_fingerprint": input_sha, "code_sha256": code_sha, "completed_frames": len(rows), "passed_frames": passed, "failed_frames": failed, "updated_unix_s": time.time()}
        _atomic_json(run_root / "run_state.json", run_state)
        output = _build_outputs(root, run_root, config, plan, records, rows, mechanics_status, run_fingerprint, input_sha, code_sha, field_csv=field_csv, plots=make_plots)
        _write_progress(run_root, {"event": "run_finished", "status": run_status, "passed_frames": passed, "failed_frames": failed, "scientific_ok": output["quality"]["scientific_ok"], "numerical_quality_passed": output["quality"]["numerical_quality_passed"]}, progress)
        return {"project": str(root), "run_id": run_id, "status": run_status,
                "scientific_ok": output["quality"]["scientific_ok"],
                "numerical_quality_passed": output["quality"]["numerical_quality_passed"],
                "scientific_acceptance": "not_assessed", "experimental_uncertainty_calibrated": False,
                "passed_frames": passed, "failed_frames": failed, "total_frames": len(records), "run_dir": str(run_root), "artifacts": output["artifacts"]}


def _csv_atomic(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                safe = {key: (json.dumps(_json_safe(row[key]), ensure_ascii=False, allow_nan=False) if isinstance(row.get(key), (list, dict, tuple))
                              else "" if row.get(key) is None or isinstance(row.get(key), float) and not math.isfinite(row[key]) else row.get(key)) for key in columns}
                writer.writerow(safe)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _iter_checkpoint_metadata(run_root: Path) -> Iterable[dict[str, Any]]:
    for metadata_path in sorted((run_root / "checkpoints").glob("frame_native_*.json")):
        yield _read_json(metadata_path)


def _iter_checkpoint_rows(run_root: Path, *, validate_fingerprint: str | None = None) -> Iterable[tuple[dict[str, np.ndarray], dict[str, Any]]]:
    for meta in _iter_checkpoint_metadata(run_root):
        if meta.get("status") == "failed":
            yield {}, meta
            continue
        if validate_fingerprint and meta.get("run_fingerprint") != validate_fingerprint:
            raise AgentError("CHECKPOINT_STALE", f"checkpoint belongs to another run: {meta.get('checkpoint')}", path=str(meta.get("checkpoint")))
        checkpoint = run_root / "checkpoints" / str(meta.get("checkpoint"))
        metadata_path = checkpoint.with_suffix(".json")
        arrays, meta = _load_field(checkpoint, metadata_path)
        yield arrays, meta


def _field_from_arrays(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> dict[str, Any]:
    field: dict[str, Any] = {key: value for key, value in arrays.items()}
    field["quality_summary"] = dict(metadata.get("quality") or {})
    field["roi"] = tuple(metadata.get("roi_candidate_px") or ())
    field["subset_size"] = (metadata.get("parameters") or {}).get("subset_size_px")
    field["step"] = (metadata.get("parameters") or {}).get("step_px")
    field["strain_window"] = (metadata.get("parameters") or {}).get("strain_window_poi")
    field["provenance"] = {"native_frame": metadata.get("native_frame"), "reference_native_frame": metadata.get("reference_native_frame"), "time_s": metadata.get("time_s")}
    return field


def _build_outputs(
    root: Path, run_root: Path, config: Mapping[str, Any], plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
    mechanics_status: Mapping[str, Any], run_fingerprint: str, input_sha: str,
    code_sha: Mapping[str, str], *, field_csv: bool, plots: bool,
) -> dict[str, Any]:
    run_state = _read_json(run_root / "run_state.json")
    expected_native_frames = [int(record["native_frame"]) for record in records]
    try:
        row_native_frames = [int(row["native_frame"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        row_native_frames = []
    native_frames_match_expected = row_native_frames == expected_native_frames
    run_status = run_state.get("status", "unknown")
    expected_frame_count = len(records)
    completed_frame_count = len(rows)
    summary_columns = [
        "frame_position_1based", "native_frame", "time_s", "reference_native_frame", "status", "quality_status", "scientific_ok",
        "eligible_point_count", "correlation_valid_fraction", "strain_valid_fraction", "max_abs_local_exx",
        "dic_virtual_gauge_engineering_strain", "dic_virtual_gauge_engineering_strain_sd", "dic_virtual_gauge_true_strain",
        "dic_virtual_gauge_valid_count", "dic_virtual_gauge_total_count",
        "dic_virtual_gauge_engineering_strain_x", "dic_virtual_gauge_engineering_strain_sd_x", "dic_virtual_gauge_valid_count_x",
        "dic_virtual_gauge_engineering_strain_y", "dic_virtual_gauge_engineering_strain_sd_y", "dic_virtual_gauge_valid_count_y", "engineering_strain",
        "force_N", "stress_MPa", "mechanics_status", "force_alignment_kind",
        "force_bracket_left_time_s", "force_bracket_right_time_s", "force_bracket_left_source_row", "force_bracket_right_source_row",
        "machine_displacement_mm", "machine_engineering_strain", "machine_true_strain", "machine_displacement_source",
        "displacement_alignment_kind", "displacement_bracket_left_time_s", "displacement_bracket_right_time_s",
        "displacement_bracket_left_source_row", "displacement_bracket_right_source_row",
        "displacement_reference_alignment_kind", "displacement_reference_bracket_left_time_s", "displacement_reference_bracket_right_time_s",
        "displacement_reference_bracket_left_source_row", "displacement_reference_bracket_right_source_row",
        "failure_code", "failure_reason", "checkpoint",
    ]
    _csv_atomic(run_root / "summary.csv", rows, summary_columns)
    numerical_quality_passed = (
        run_status == "complete"
        and expected_frame_count > 0
        and completed_frame_count == expected_frame_count
        and native_frames_match_expected
        and all(row.get("status") == "complete" and row.get("scientific_ok") is True for row in rows)
    )
    quality = {
        "schema": "straintrace.agent-quality/v1", "scientific_ok": numerical_quality_passed,
        "numerical_quality_passed": numerical_quality_passed,
        "run_status": run_status,
        "expected_frame_count": expected_frame_count,
        "completed_frame_count": completed_frame_count,
        "native_frames_match_expected": native_frames_match_expected,
        "scientific_ok_scope": QUALITY_GATE_SCOPE,
        "scientific_acceptance": "not_assessed", "experimental_uncertainty_calibrated": False,
        "quality_gate_scope": QUALITY_GATE_SCOPE,
        "frame_count": len(rows), "passed_frame_count": sum(row.get("scientific_ok") is True for row in rows),
        "failed_frame_count": sum(row.get("scientific_ok") is not True for row in rows),
        "threshold_policy": "existing ezdic_core field_quality_summary thresholds; never relaxed by Agent",
        "aggregate_engineering_strain": None,
        "mechanics_alignment": _json_safe(dict(mechanics_status)),
        "frames": [],
    }
    for metadata in _iter_checkpoint_metadata(run_root):
        quality["frames"].append({"native_frame": metadata.get("native_frame"), "status": metadata.get("status"), "quality": metadata.get("quality"), "failure": metadata.get("error")})
    _atomic_json(run_root / "quality.json", quality)
    gauge_csv = run_root / "gauges.csv"
    gauge_tmp = gauge_csv.with_name(f".{gauge_csv.name}.{uuid.uuid4().hex}.tmp")
    try:
        with gauge_tmp.open("w", encoding="utf-8-sig", newline="") as stream:
            gauge_writer = csv.DictWriter(stream, fieldnames=_GAUGE_COLUMNS, extrasaction="ignore")
            gauge_writer.writeheader()
            for metadata in _iter_checkpoint_metadata(run_root):
                summary = metadata.get("summary_row") or {}
                for gauge_row in metadata.get("gauge_measurements") or []:
                    gauge_writer.writerow({"frame_position_1based": summary.get("frame_position_1based"), "native_frame": metadata.get("native_frame"), "time_s": metadata.get("time_s"), **gauge_row})
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(gauge_tmp, gauge_csv)
    finally:
        try:
            gauge_tmp.unlink()
        except FileNotFoundError:
            pass
    if field_csv:
        import ezdic_core as core
        fields_dir = run_root / "fields_csv"
        fields_dir.mkdir(exist_ok=True)
        for arrays, metadata in _iter_checkpoint_rows(run_root, validate_fingerprint=run_fingerprint):
            if not arrays:
                continue
            field = _field_from_arrays(arrays, metadata)
            path = fields_dir / f"{_frame_key(int(metadata['native_frame']))}.csv"
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                core.dic_field_to_dataframe(field).to_csv(temporary, index=False)
                _replace_with_retry(temporary, path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    plot_result = None
    history_summary = None
    if plots:
        plot_result = _plot_run_locked(root, run_root, config, plan, plot_every=int(config.get("plot_every", 0)), run_fingerprint=run_fingerprint)
        history_summary = plot_result.get("history_plots")
    elif (run_root / "plots" / "history_summary.json").is_file():
        history_summary = _read_json(run_root / "plots" / "history_summary.json")
    report_lines = [
        "# StrainTrace Agent analysis report", "",
        f"- Project ID: `{config['project_id']}`", f"- Run: `{run_root.name}`",
        f"- Reference native frame: `{plan['reference_native_frame']}`",
        f"- Analysis rectangle candidate (px): `{plan['analysis_rectangle_candidate_px']}`",
        "- The rectangle is a texture-based analysis candidate; physical specimen geometry is not confirmed.",
        f"- Run execution status: `{run_status}`",
        f"- Frame coverage: `{completed_frame_count}/{expected_frame_count}` expected native frames; identity matches: `{native_frames_match_expected}`.",
        f"- Core-quality frames among completed rows: `{quality['passed_frame_count']}/{completed_frame_count}`",
        f"- Core numerical quality gates: `{'PASS' if quality['numerical_quality_passed'] else 'NOT PASSED'}`",
        "- Scientific acceptance: not assessed; a numerical quality-gate pass is not experimental validation.",
        f"- Load alignment: `{mechanics_status.get('status')}`",
        "- DIC virtual-gauge engineering strain is emitted only for configured valid endpoints; machine displacement strain is reported separately with its channel source.",
        "- Stress is force_N / user-confirmed area_mm2. Missing clock alignment or area leaves dependent outputs unavailable.",
        "- Machine-channel `alignment_kind` distinguishes exact samples from bounded linear interpolation; bracket times/source rows are in `summary.csv`, and the source CSV hash is in `manifest.json`.",
        "- Local strain fit span is `(strain_window_poi - 1) * step_px`; full support span including subset is `subset_size_px + (strain_window_poi - 1) * step_px`.",
        "- This is in-plane 2D DIC. Out-of-plane motion correction and experimental uncertainty calibration are not provided.",
        "- Invalid POIs and unsupported regions remain NaN; no interpolation is performed by this workflow.", "",
        "## Selected solver parameters", "",
        "```json", json.dumps(plan["selected_parameters"], indent=2, ensure_ascii=False, allow_nan=False), "```", "",
        "See `summary.csv`, `quality.json`, per-frame checkpoint sidecars, and `manifest.json` for machine-readable evidence.",
    ]
    _atomic_bytes(run_root / "report.md", ("\n".join(report_lines) + "\n").encode("utf-8"))
    _update_report_visualizations(run_root, history_summary, plot_count=plot_result.get("plot_count") if plot_result else None)
    artifact_paths = [run_root / name for name in ("summary.csv", "quality.json", "report.md")]
    artifact_paths.append(gauge_csv)
    artifact_paths.extend(path for path in sorted((run_root / "plots").rglob("*")) if path.is_file())
    artifact_paths.extend(sorted((run_root / "extractions").rglob("*")) if (run_root / "extractions").exists() else [])
    if field_csv:
        artifact_paths.extend(sorted((run_root / "fields_csv").glob("*.csv")))
    manifest = {
        "schema": "straintrace.agent-manifest/v1", "project_id": config["project_id"],
        "run_id": run_root.name, "run_fingerprint": run_fingerprint,
        "run_status": run_status,
        "expected_frame_count": expected_frame_count,
        "completed_frame_count": completed_frame_count,
        "native_frames_match_expected": native_frames_match_expected,
        "input_fingerprint": input_sha, "configuration_sha256": _hash_json(config),
        "plan_sha256": plan.get("plan_sha256"), "code_sha256": code_sha,
        "source_images": [{"native_frame": row["native_frame"], "path": row["path"], "sha256": row["sha256"], "time_s": row["time_s"]} for row in records],
        "load_file_sha256": _sha256(Path(config["loads"]["path"])) if config.get("loads") else None,
        "mask_sha256": config.get("mask_sha256"),
        "time_basis": config.get("time_basis", "physical_time"),
        "environment": __import__("ezdic_core").collect_environment(),
        "artifacts": {path.relative_to(run_root).as_posix(): _sha256(path) for path in artifact_paths if path.is_file()},
        "checkpoint_receipts": {
            metadata_path.name: {"sha256": _sha256(metadata_path), "checkpoint_sha256": _read_json(metadata_path).get("checkpoint_sha256"), "native_frame": _read_json(metadata_path).get("native_frame"), "status": _read_json(metadata_path).get("status")}
            for metadata_path in sorted((run_root / "checkpoints").glob("frame_native_*.json"))
        },
        "checkpoint_count": sum(1 for path in (run_root / "checkpoints").glob("*.npz")),
        "checkpoint_failures": sum(1 for path in (run_root / "checkpoints").glob("frame_native_*.json") if _read_json(path).get("status") == "failed"),
        "scientific_ok": quality["scientific_ok"], "numerical_quality_passed": quality["numerical_quality_passed"],
        "quality_gate_scope": QUALITY_GATE_SCOPE, "scientific_acceptance": "not_assessed",
        "experimental_uncertainty_calibrated": False,
        "units": {"x_px": "px", "u_px": "px", "strain": "dimensionless", "force_N": "N", "stress_MPa": "MPa"},
    }
    _atomic_json(run_root / "manifest.json", manifest)
    _write_region_outputs(run_root, config, plan, run_fingerprint)
    _refresh_manifest_artifacts(run_root)
    portable_report = _build_report_transaction(run_root, config, plan, include_checkpoints=False)
    return {"quality": quality, "artifacts": {"summary": str(run_root / "summary.csv"), "gauges": str(gauge_csv), "regions": str(run_root / "regions.csv"), "geometry": str(run_root / "geometry.json"), "quality": str(run_root / "quality.json"), "manifest": str(run_root / "manifest.json"), "report": str(run_root / "report.md"), "report_html": portable_report["report_html"], "plots": str(run_root / "plots")}}


def _refresh_manifest_artifacts(run_root: Path) -> None:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path)
    files = [run_root / name for name in ("summary.csv", "quality.json", "gauges.csv", "regions.csv", "geometry.json", "report.md")]
    for subdir in ("plots", "fields_csv", "extractions", "report"):
        directory = run_root / subdir
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    manifest["artifacts"] = {path.relative_to(run_root).as_posix(): _sha256(path) for path in sorted(set(files)) if path.is_file()}
    _atomic_json(manifest_path, manifest)


def _build_report_transaction(
    run_root: Path, config: Mapping[str, Any], plan: Mapping[str, Any], *, include_checkpoints: bool,
) -> dict[str, Any]:
    """Commit the report and its outer hashes together, retaining rollback data."""
    from . import reporting

    run_root = run_root.resolve()
    target = reporting._safe_file(run_root, "report")
    if target.exists() and _read_json(target / "manifest.json").get("schema") != reporting.PACKAGE_SCHEMA:
        raise AgentError("REPORT_OUTPUT_UNOWNED", "refusing to replace an unrecognized report directory", path=str(target))
    candidate = run_root / f".report-candidate-{uuid.uuid4().hex}"
    backup = run_root / f".report-backup-{uuid.uuid4().hex}"
    manifest_path = run_root / "manifest.json"
    previous_manifest = manifest_path.read_bytes()
    committed = False
    try:
        result = reporting.build_report(
            run_root, config=config, plan=plan, regions_csv=run_root / "regions.csv",
            output_dir=candidate, include_checkpoints=include_checkpoints,
        )
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(candidate, target)
            try:
                _refresh_manifest_artifacts(run_root)
            except BaseException:
                os.replace(target, candidate)
                if backup.exists():
                    os.replace(backup, target)
                _atomic_bytes(manifest_path, previous_manifest)
                raise
        except BaseException:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        committed = True

        def relocate(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(str(candidate)):
                return str(target) + value[len(str(candidate)):]
            if isinstance(value, dict):
                return {key: relocate(item) for key, item in value.items()}
            if isinstance(value, list):
                return [relocate(item) for item in value]
            return value

        return relocate(result)
    finally:
        # These unique staging paths are owned by this operation. Validate the
        # complete path before recursive cleanup; retain a failed rollback.
        for path in (candidate, backup) if committed else (candidate,):
            safe_path = reporting._safe_file(run_root, path.name)
            if safe_path.exists():
                shutil.rmtree(safe_path)


def _plot_selection(run_root: Path, frame_metas: Sequence[Mapping[str, Any]], plot_every: int) -> list[Mapping[str, Any]]:
    successes = [meta for meta in frame_metas if meta.get("status") in {"complete", "quality_failed"} and meta.get("checkpoint")]
    if not successes:
        return []
    if plot_every > 0:
        selected = successes[::plot_every]
        if successes[-1] not in selected:
            selected.append(successes[-1])
        if len(selected) > 24:
            indices = np.linspace(0, len(selected) - 1, 24).round().astype(int)
            selected = [selected[index] for index in sorted(set(indices.tolist()))]
        return selected
    if len(successes) <= 3:
        return successes
    # Bounded load/strain stage selection for long experiments.
    by_native = {int(meta["native_frame"]): meta for meta in successes}
    selected = [successes[0], successes[len(successes) // 2], successes[-1]]
    strain_peak = max(successes, key=lambda meta: (meta.get("summary_row") or {}).get("max_abs_local_exx") or -1.0)
    load_peak = max(successes, key=lambda meta: abs((meta.get("summary_row") or {}).get("stress_MPa") or 0.0))
    selected.extend((strain_peak, load_peak))
    return list({int(meta["native_frame"]): by_native[int(meta["native_frame"])] for meta in selected}.values())


def _plot_run_locked(root: Path, run_root: Path, config: Mapping[str, Any], plan: Mapping[str, Any], *, plot_every: int, run_fingerprint: str) -> dict[str, Any]:
    from . import fields as field_tools
    selected = _plot_selection(run_root, list(_iter_checkpoint_metadata(run_root)), plot_every)
    plots_root = run_root / "plots"
    plots_root.mkdir(exist_ok=True)
    generated = []
    issues = []
    components = ("u", "v", "zncc", "Exx", "Eyy", "Exy", "exx", "eyy", "exy")
    for component in components:
        def fields_for_limits():
            for metadata in _iter_checkpoint_metadata(run_root):
                if metadata.get("status") == "failed" or not metadata.get("checkpoint"):
                    continue
                arrays, loaded = _load_field(run_root / "checkpoints" / str(metadata["checkpoint"]))
                yield _field_from_arrays(arrays, loaded)
        try:
            color_limits = field_tools.common_color_limits(fields_for_limits(), component, symmetric=component not in {"u", "v", "zncc"})
        except ValueError as exc:
            issues.append({"component": component, "reason": str(exc)})
            continue
        for meta in selected:
            if meta.get("status") == "failed" or not meta.get("checkpoint"):
                continue
            checkpoint = run_root / "checkpoints" / str(meta["checkpoint"])
            arrays, metadata = _load_field(checkpoint)
            field = _field_from_arrays(arrays, metadata)
            name = _frame_key(int(meta["native_frame"]))
            slug = _component_file_slug(component)
            path = plots_root / f"{name}_{slug}.png"
            temp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
            try:
                fig, _ax, _artist = field_tools.plot_field_map(
                    field, component, color_limits=color_limits,
                    title=f"native frame {meta['native_frame']} · {meta.get('status')}",
                )
                fig.savefig(temp_path, dpi=220, bbox_inches="tight", format="png")
                import matplotlib.pyplot as plt
                plt.close(fig)
                _replace_with_retry(temp_path, path)
                generated.append(path.relative_to(run_root).as_posix())
            except ValueError as exc:
                issues.append({"native_frame": meta.get("native_frame"), "component": component, "reason": str(exc)})
            finally:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
    _atomic_json(plots_root / "plot_issues.json", issues)
    history_plots = _plot_standard_histories(run_root)
    _atomic_json(plots_root / "history_summary.json", history_plots)
    # Position-time data is loaded and released one checkpoint at a time by
    # the field module integration when configured; keep the API bounded.
    return {"plot_count": len(generated), "paths": generated, "selected_native_frames": [meta.get("native_frame") for meta in selected], "plot_issues": issues, "plot_selection_cap": 24, "history_plots": history_plots}


def _plot_standard_histories(run_root: Path) -> dict[str, Any]:
    """Emit sparse instrument and virtual-gauge curves without connecting gaps."""
    table_path = run_root / "summary.csv"
    try:
        table = pd.read_csv(table_path)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"summary_unreadable: {exc}", "plots": []}
    plots_dir = run_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    def values(column: str) -> np.ndarray:
        if column not in table:
            return np.full(len(table), np.nan, dtype=float)
        return pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)

    frame_axis = not np.isfinite(values("time_s")).any()
    history_column = "native_frame" if frame_axis else "time_s"
    history_label = "Native frame" if frame_axis else "Time (s)"

    specifications: list[dict[str, Any]] = []
    for plot_name, x_column, y_column, x_label, y_label, label, source, alignment_column, unavailable_reason in (
        ("time_force.png", "time_s", "force_N", "Time (s)", "Force (N)", "Force", "force_N", "force_alignment_kind", "requires a configured force channel, confirmed clock, and in-support source samples"),
        ("time_machine_displacement.png", "time_s", "machine_displacement_mm", "Time (s)", "Machine displacement from reference (mm)", "Machine displacement", "machine_displacement_mm", "displacement_alignment_kind", "requires a configured displacement channel, confirmed clock, and source support at the frame and reference times"),
        ("time_machine_engineering_strain.png", "time_s", "machine_engineering_strain", "Time (s)", "Machine engineering strain", "Machine engineering strain", "machine_engineering_strain", "displacement_alignment_kind", "requires machine displacement and a user-supplied machine gauge length"),
        ("time_stress.png", "time_s", "stress_MPa", "Time (s)", "Stress (MPa)", "Stress", "stress_MPa", "force_alignment_kind", "requires confirmed force, user-supplied area_mm2, and in-support source samples"),
    ):
        alignment_values = table[alignment_column].astype(str).tolist() if alignment_column in table else []
        source_values = table["machine_displacement_source"].dropna().astype(str).unique().tolist() if y_column.startswith("machine_") and "machine_displacement_source" in table else []
        series_label = f"{label} ({', '.join(source_values)})" if source_values and y_column.startswith("machine_") else label
        specifications.append({
            "plot": plot_name, "x_label": x_label, "y_label": y_label,
            "series": [{"label": series_label, "x": values(x_column), "y": values(y_column), "source": source,
                        "alignment_column": alignment_column, "alignment_values": alignment_values}],
            "unavailable_reason": unavailable_reason,
        })

    gauge_series = []
    for axis in ("x", "y"):
        column = f"dic_virtual_gauge_engineering_strain_{axis}"
        gauge_series.append({
            "label": f"DIC virtual-gauge engineering strain ({axis})", "x": values(history_column),
            "y": values(column), "source": column, "alignment_column": "quality_status",
            "alignment_values": table["quality_status"].astype(str).tolist() if "quality_status" in table else [],
        })
    specifications.append({
        "plot": ("frame" if frame_axis else "time") + "_dic_virtual_gauge_engineering_strain.png", "x_label": history_label,
        "y_label": "DIC virtual-gauge engineering strain", "series": gauge_series,
        "unavailable_reason": "requires configured fixed-reference DIC virtual-gauge endpoints; no full-field mean is substituted",
    })
    stress_gauge_series = []
    for axis in ("x", "y"):
        column = f"dic_virtual_gauge_engineering_strain_{axis}"
        stress_gauge_series.append({
            "label": f"DIC virtual-gauge strain ({axis})", "x": values(column),
            "y": values("stress_MPa"), "source": column, "alignment_column": "force_alignment_kind",
            "alignment_values": table["force_alignment_kind"].astype(str).tolist() if "force_alignment_kind" in table else [],
        })
    specifications.append({
        "plot": "stress_vs_dic_virtual_gauge_engineering_strain.png",
        "x_label": "DIC virtual-gauge engineering strain", "y_label": "Stress (MPa)", "series": stress_gauge_series,
        "unavailable_reason": "requires both force/area stress and valid fixed-reference DIC virtual-gauge engineering strain",
    })

    plot_summaries = []
    import matplotlib.pyplot as plt
    for specification in specifications:
        valid_series = []
        summary_series = []
        for series in specification["series"]:
            x = np.asarray(series["x"], dtype=float)
            y = np.asarray(series["y"], dtype=float)
            count = min(x.size, y.size)
            valid = np.isfinite(x[:count]) & np.isfinite(y[:count])
            alignment_values = series.get("alignment_values") or []
            alignment_counts: dict[str, int] = {}
            if alignment_values:
                for item in set(alignment_values):
                    alignment_counts[item] = alignment_values.count(item)
            item_summary = {
                "label": series["label"], "source_column": series["source"],
                "finite_sample_count": int(valid.sum()), "alignment_counts": alignment_counts,
            }
            summary_series.append(item_summary)
            if valid.any():
                valid_series.append((series["label"], x[:count][valid], y[:count][valid]))
        path = plots_dir / specification["plot"]
        if not valid_series:
            plot_summaries.append({"plot": specification["plot"], "status": "unavailable", "reason": specification.get("unavailable_reason", "no finite aligned samples"), "series": summary_series})
            continue
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for label, x, y in valid_series:
            ax.scatter(x, y, s=22, alpha=0.85, label=label, edgecolors="none")
        ax.set_xlabel(specification["x_label"])
        ax.set_ylabel(specification["y_label"])
        ax.grid(True, alpha=0.25)
        if len(valid_series) > 1:
            ax.legend(frameon=False)
        temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
        try:
            fig.savefig(temp, dpi=220, bbox_inches="tight", format="png")
            _replace_with_retry(temp, path)
        finally:
            plt.close(fig)
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        plot_summaries.append({"plot": specification["plot"], "status": "generated", "path": path.relative_to(run_root).as_posix(), "point_count": sum(len(x) for _, x, _ in valid_series), "point_rendering": "scatter_no_interframe_lines", "series": summary_series})
    return {"status": "complete", "plots": plot_summaries}


def _update_report_visualizations(run_root: Path, history_summary: Mapping[str, Any] | None, *, plot_count: int | None = None) -> None:
    report_path = run_root / "report.md"
    if not report_path.is_file():
        return
    lines = ["## Standard visualizations", ""]
    if plot_count is not None:
        lines.append(f"- Field maps: {plot_count} PNG file(s) generated for selected native frames.")
    if not history_summary or history_summary.get("status") != "complete":
        reason = (history_summary or {}).get("reason") or "plot generation was not requested"
        lines.append(f"- Time-history, displacement/gauge and stress–DIC curves: unavailable ({reason}).")
    else:
        for item in history_summary.get("plots", []):
            if item.get("status") == "generated":
                lines.append(f"- `{item['plot']}`: generated with {item.get('point_count', 0)} scatter point(s); gaps are not connected.")
            else:
                lines.append(f"- `{item['plot']}`: unavailable ({item.get('reason', 'no finite aligned samples')}).")
    block = "\n".join(lines) + "\n\n"
    text = report_path.read_text(encoding="utf-8")
    marker = "## Standard visualizations"
    start = text.find(marker)
    if start >= 0:
        end = text.find("## Selected solver parameters", start)
        if end >= 0:
            text = text[:start] + text[end:]
    insertion = text.find("## Selected solver parameters")
    if insertion >= 0:
        text = text[:insertion] + block + text[insertion:]
    else:
        text = text.rstrip() + "\n\n" + block
    _atomic_bytes(report_path, text.encode("utf-8"))


def plot(project: str | os.PathLike[str], *, plot_every: int | None = None) -> dict[str, Any]:
    root = _project_root(project)
    with _project_lock(root, purpose="plot"):
        _, run_root = _load_run_pointer(root)
        config = _read_config(root)
        pointer = _current_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        run_state = _read_json(run_root / "run_state.json")
        _require_current_run_binding(root, config, run_root, plan, run_state)
        count = int(config.get("plot_every", 0) if plot_every is None else plot_every)
        if count < 0:
            raise AgentError("CONFIG_VALUE_ERROR", "plot_every must be >= 0", path="$.plot_every")
        result = _plot_run_locked(root, run_root, config, plan, plot_every=count, run_fingerprint=str(run_state["run_fingerprint"]))
        _update_report_visualizations(run_root, result.get("history_plots"), plot_count=result.get("plot_count"))
        _refresh_manifest_artifacts(run_root)
        return {"run_id": run_root.name, "status": "complete", **result}


def export(project: str | os.PathLike[str], *, field_csv: bool = False) -> dict[str, Any]:
    root = _project_root(project)
    config = _read_config(root)
    with _project_lock(root, purpose="export"):
        pointer, run_root = _load_run_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        run_state = _read_json(run_root / "run_state.json")
        records, input_sha, current_code = _require_current_run_binding(root, config, run_root, plan, run_state)
        rows = []
        mechanics, mechanics_status = _load_mechanics(config)
        displacement_reference_alignment = None
        displacement_reference = None
        if mechanics is not None and mechanics_status.get("status") == "aligned" and "displacement_time_s" in mechanics:
            reference_record = next((record for record in records if int(record["native_frame"]) == int(plan["reference_native_frame"])), None)
            if reference_record is not None:
                displacement_reference_alignment = _interpolate_bounded_detail(
                    float(reference_record["time_s"]), mechanics["displacement_time_s"], mechanics["displacement_mm"],
                    float(config["loads"]["max_gap_s"]), mechanics.get("displacement_source_rows"),
                )
                displacement_reference = displacement_reference_alignment.get("value")
        by_native = {int(meta["native_frame"]): meta for meta in _iter_checkpoint_metadata(run_root)}
        for index, record in enumerate(records, 1):
            native = int(record["native_frame"])
            meta = by_native.get(native)
            if meta is None:
                rows.append({"frame": index, "native_frame": native, "time_s": record.get("time_s"), "status": "pending", "quality_status": "unavailable", "scientific_ok": False, "engineering_strain": None, **_mechanics_at(record.get("time_s"), mechanics, config, displacement_reference_mm=displacement_reference, displacement_reference_alignment=displacement_reference_alignment)})
                continue
            row = dict(meta.get("summary_row") or {})
            if row.get("status") == "failed":
                rows.append(row)
                continue
            row.update(_mechanics_at(record.get("time_s"), mechanics, config, displacement_reference_mm=displacement_reference, displacement_reference_alignment=displacement_reference_alignment))
            row["engineering_strain"] = row.get("dic_virtual_gauge_engineering_strain")
            rows.append(row)
        output = _build_outputs(root, run_root, config, plan, records, rows, mechanics_status, str(run_state["run_fingerprint"]), input_sha, current_code, field_csv=field_csv, plots=False)
        output["portable_report"] = _build_report_transaction(run_root, config, plan, include_checkpoints=True)
        return {"run_id": run_root.name, "status": run_state.get("status"),
                "scientific_ok": output["quality"]["scientific_ok"],
                "numerical_quality_passed": output["quality"]["numerical_quality_passed"],
                "scientific_acceptance": "not_assessed", "experimental_uncertainty_calibrated": False,
                "artifacts": output["artifacts"]}


def extract(
    project: str | os.PathLike[str], *, component: str = "exx",
    roi: Sequence[float] | None = None, path_points: Sequence[Sequence[float]] | None = None,
    max_distance: float | None = None, coordinate_system: str = "image",
) -> dict[str, Any]:
    """Extract configured spatial queries using the optional field helpers."""
    try:
        from . import fields
    except ImportError as exc:
        raise AgentError("FIELD_API_UNAVAILABLE", "spatial field extraction support is not installed") from exc
    root = _project_root(project)
    config = _read_config(root)
    with _project_lock(root, purpose="extract"):
        pointer, run_root = _load_run_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        state = _read_json(run_root / "run_state.json")
        records, _, _ = _require_current_run_binding(root, config, run_root, plan, state)
        output: dict[str, Any] = {"run_id": run_root.name, "component": component, "coordinate_system": coordinate_system}
        checkpoint_root = run_root / "checkpoints"
        metadata_by_native = {int(meta["native_frame"]): meta for meta in _iter_checkpoint_metadata(run_root)}
        dest = run_root / "extractions"
        dest.mkdir(exist_ok=True)

        def load_field(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
            meta = metadata_by_native.get(int(record["native_frame"]))
            if not meta or meta.get("status") == "failed" or not meta.get("checkpoint"):
                return None, meta
            arrays, loaded = _load_field(checkpoint_root / str(meta["checkpoint"]))
            field = _field_from_arrays(arrays, loaded)
            if config.get("calibration"):
                field.update(fields.calibrate_field(field, config["calibration"]))
            return field, loaded

        if roi is not None:
            output["rois"] = []
            def roi_rows():
                for record in records:
                    field, meta = load_field(record)
                    native = int(record["native_frame"])
                    if field is None:
                        continue
                    summary = fields.summarize_roi(field, component, roi, coordinate_system=coordinate_system)
                    output["rois"].append({"native_frame": native, "time_s": meta.get("time_s"), **_json_safe(summary)})
                    for row in fields.iter_roi_rows(field, component, roi, coordinate_system=coordinate_system):
                        yield {"native_frame": native, "time_s": meta.get("time_s"), **row}
            component_slug = _component_file_slug(component)
            roi_csv = fields.write_rows_csv(roi_rows(), dest / f"roi_{component_slug}.csv", fieldnames=["native_frame", "time_s", "poi_index", "x", "y", "coordinate_system", "component", "value", "value_unit", "status", "interpolation"])
            _atomic_json(dest / f"roi_{component_slug}_summary.json", output["rois"])
            output["roi_csv"] = str(roi_csv)
            output["roi_summary"] = str(dest / f"roi_{component_slug}_summary.json")
        if path_points is not None:
            if max_distance is None:
                raise AgentError("PATH_TOLERANCE_REQUIRED", "max_distance is required for path extraction", path="$.max_distance")
            def sample_rows():
                for record in records:
                    field, meta = load_field(record)
                    native = int(record["native_frame"])
                    if field is None:
                        for index, point in enumerate(path_points):
                            yield {"native_frame": native, "time_s": record.get("time_s"), "run_status": "unavailable", "path_index": index, "target_x": point[0], "target_y": point[1], "poi_index": None, "poi_x": None, "poi_y": None, "distance": None, "coordinate_system": coordinate_system, "interpolation": "none", component: None, "status": "frame_failed"}
                        continue
                    for row in fields.sample_path(field, path_points, [component], max_distance, coordinate_system=coordinate_system):
                        yield {"native_frame": native, "time_s": meta.get("time_s"), "run_status": meta.get("status"), **row}
            component_slug = _component_file_slug(component)
            path_csv = fields.write_rows_csv(sample_rows(), dest / f"path_{component_slug}.csv")
            output["paths_csv"] = str(path_csv)
            frame_axis = config.get("time_basis") == "frame_index"
            times = [record["native_frame"] if frame_axis else record.get("time_s") for record in records]
            def plot_fields():
                for record in records:
                    field, _meta = load_field(record)
                    yield field
            finite_steps = np.diff([float(value) for value in times]) if len(times) > 1 else np.asarray([])
            max_time_gap = float(np.median(finite_steps) * 1.5) if finite_steps.size and not frame_axis else None
            figure_path = dest / f"position_{'frame' if frame_axis else 'time'}_{component_slug}.png"
            fig, _ax, _artist, plot_summary = fields.plot_position_time(plot_fields(), times, path_points, component, max_distance,
                         max_time_gap_s=max_time_gap, axis_unit="frame" if frame_axis else "s", axis_label="Native frame" if frame_axis else "Time (s)")
            fig.savefig(figure_path, dpi=220, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            output["position_time_plot"] = str(figure_path)
            output["position_time_summary"] = plot_summary
        _refresh_manifest_artifacts(run_root)
        return output


def _current_plan_if_fresh(root: Path, config: Mapping[str, Any]) -> dict[str, Any] | None:
    pointer = _current_pointer(root)
    if not pointer.get("current_plan"):
        return None
    plan = _read_json(_plan_path(root, pointer))
    _validate_plan_checksum(plan)
    _, input_sha = _inventory_fresh(root, config)
    import ezdic_core as core
    return plan if (
        plan.get("configuration_sha256") == _hash_json(config)
        and plan.get("input_fingerprint") == input_sha
        and plan.get("code_sha256") == _code_fingerprint()
        and plan.get("environment") == core.collect_environment()
    ) else None


def analyze(project: str | os.PathLike[str], *, progress: Callable[[dict[str, Any]], None] | None = None, field_csv: bool = False, make_plots: bool = True) -> dict[str, Any]:
    root = _project_root(project)
    config = _read_config(root)
    plan = _current_plan_if_fresh(root, config)
    if plan is None:
        plan = plan_project(root, progress=progress)
    elif progress:
        progress({"event": "plan_reused", "plan_id": plan.get("plan_id"), "status": plan.get("status")})
    result = run(root, progress=progress, field_csv=field_csv, make_plots=make_plots)
    return {"plan": plan, "run": result}


def cancel(project: str | os.PathLike[str]) -> dict[str, Any]:
    root = _project_root(project)
    if not root.is_dir():
        raise AgentError("PROJECT_NOT_FOUND", f"project does not exist: {root}", path=str(root))
    _atomic_json(root / "CANCEL.request", {"pid": os.getpid(), "requested_unix_s": time.time()})
    return {"project": str(root), "cancel_requested": True}


def status(project: str | os.PathLike[str]) -> dict[str, Any]:
    root = _project_root(project)
    config = _read_config(root)
    pointer = _current_pointer(root)
    lock_path = root / ".agent.lock"
    lock = _read_json(lock_path) if lock_path.exists() else None
    run_state = None
    progress_state = None
    if pointer.get("current_run"):
        run_root = root / str(pointer["current_run"])
        state_path = run_root / "run_state.json"
        if state_path.exists():
            run_state = _read_json(state_path)
        progress_path = run_root / "progress.json"
        if progress_path.exists():
            progress_state = _read_json(progress_path)
    return {"project": str(root), "project_id": config.get("project_id"), "locked": lock is not None, "lock_owner": lock, "cancel_requested": (root / "CANCEL.request").exists(), "current_plan": pointer.get("current_plan"), "current_run": pointer.get("current_run"), "run": run_state, "progress": progress_state}


def verify(project: str | os.PathLike[str]) -> dict[str, Any]:
    """Read-only integrity check for source inventory, current plan/run, and outputs."""
    root = _project_root(project)
    config = _read_config(root)
    pointer = _current_pointer(root)
    records, input_sha = _inventory_fresh(root, config)
    issues: list[dict[str, Any]] = []
    stored_inventory = _read_json(root / "inventory.json")
    if stored_inventory.get("input_fingerprint") != input_sha:
        issues.append({"code": "PROJECT_INVENTORY_STALE"})
    plan = None
    if pointer.get("current_plan"):
        plan = _read_json(root / str(pointer["current_plan"]))
        expected = plan.pop("plan_sha256", None)
        if expected != _hash_json(plan):
            issues.append({"code": "PLAN_CHECKSUM_MISMATCH"})
        plan["plan_sha256"] = expected
    else:
        issues.append({"code": "PLAN_MISSING"})
    if not pointer.get("current_run"):
        return {"project": str(root), "valid": False, "issues": [{"code": "RUN_MISSING"}], "input_fingerprint": input_sha}
    run_root = root / str(pointer["current_run"])
    state = _read_json(run_root / "run_state.json")
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        issues.append({"code": "MANIFEST_MISSING"})
        manifest = {}
    else:
        manifest = _read_json(manifest_path)
    current_code = _code_fingerprint()
    import ezdic_core as core
    current_environment = core.collect_environment()
    if state.get("run_fingerprint") != _run_fingerprint(config, plan or {}, input_sha, current_code):
        issues.append({"code": "RUN_FINGERPRINT_STALE"})
    if manifest.get("run_status") != state.get("status"):
        issues.append({"code": "RUN_STATUS_MISMATCH"})
    if manifest.get("expected_frame_count") != len(records):
        issues.append({"code": "EXPECTED_FRAME_COUNT_MISMATCH", "expected": len(records), "actual": manifest.get("expected_frame_count")})
    if manifest.get("input_fingerprint") != input_sha:
        issues.append({"code": "INPUT_FINGERPRINT_MISMATCH"})
    if manifest.get("code_sha256") != current_code:
        issues.append({"code": "CODE_FINGERPRINT_MISMATCH"})
    if manifest.get("environment") != current_environment or (plan or {}).get("environment") != current_environment:
        issues.append({"code": "ENVIRONMENT_MISMATCH"})
    for relative, expected in (manifest.get("artifacts") or {}).items():
        path = run_root / relative
        if not path.is_file():
            issues.append({"code": "ARTIFACT_MISSING", "path": relative})
        elif _sha256(path) != expected:
            issues.append({"code": "ARTIFACT_CHECKSUM_MISMATCH", "path": relative})
    expected_frames = {int(row["native_frame"]): row for row in records}
    manifest_receipts = manifest.get("checkpoint_receipts") or {}
    seen: dict[int, Path] = {}
    checkpoint_count = 0
    receipt_paths = sorted((run_root / "checkpoints").glob("frame_native_*.json"))
    for metadata_path in receipt_paths:
        metadata = _read_json(metadata_path)
        try:
            native = int(metadata["native_frame"])
        except (KeyError, TypeError, ValueError):
            issues.append({"code": "RECEIPT_NATIVE_FRAME_INVALID", "path": metadata_path.name})
            continue
        if native not in expected_frames:
            issues.append({"code": "UNEXPECTED_FRAME_RECEIPT", "native_frame": native})
        if native in seen:
            issues.append({"code": "DUPLICATE_FRAME_RECEIPT", "native_frame": native})
        seen[native] = metadata_path
        expected = expected_frames.get(native)
        if expected and (metadata.get("source_sha256") != expected.get("sha256") or metadata.get("source_path") != expected.get("path")):
            issues.append({"code": "RECEIPT_SOURCE_MISMATCH", "native_frame": native})
        if metadata.get("run_fingerprint") != state.get("run_fingerprint") or metadata.get("input_fingerprint") != input_sha or metadata.get("code_sha256") != current_code:
            issues.append({"code": "RECEIPT_STALE", "native_frame": native})
        if not _solver_parameters_match(metadata.get("parameters"), (plan or {}).get("selected_parameters")):
            issues.append({"code": "CHECKPOINT_PARAMETER_MISMATCH", "native_frame": native})
        receipt_record = manifest_receipts.get(metadata_path.name)
        if not receipt_record or receipt_record.get("sha256") != _sha256(metadata_path):
            issues.append({"code": "RECEIPT_CHECKSUM_MISMATCH", "native_frame": native})
        if metadata.get("status") == "failed":
            if not isinstance(metadata.get("error"), Mapping):
                issues.append({"code": "FAILED_RECEIPT_REASON_MISSING", "native_frame": native})
            if metadata.get("checkpoint") is not None:
                issues.append({"code": "FAILED_RECEIPT_HAS_NPZ", "native_frame": native})
            continue
        checkpoint = metadata_path.with_suffix(".npz")
        try:
            arrays, loaded = _load_field(checkpoint, metadata_path)
            if not arrays:
                issues.append({"code": "CHECKPOINT_EMPTY", "path": checkpoint.name})
            if loaded.get("checkpoint_sha256") != (receipt_record or {}).get("checkpoint_sha256"):
                issues.append({"code": "CHECKPOINT_MANIFEST_CHECKSUM_MISMATCH", "native_frame": native})
            if loaded.get("input_fingerprint") != input_sha or loaded.get("code_sha256") != current_code:
                issues.append({"code": "CHECKPOINT_STALE", "path": checkpoint.name})
            checkpoint_count += 1
        except (OSError, AgentError) as exc:
            issues.append({"code": exc.code if isinstance(exc, AgentError) else "CHECKPOINT_MISSING", "path": checkpoint.name})
    for native in sorted(set(expected_frames) - set(seen)):
        issues.append({"code": "FRAME_RECEIPT_MISSING", "native_frame": native})
    if len(receipt_paths) != len(records):
        issues.append({"code": "FRAME_RECEIPT_COUNT_MISMATCH", "expected": len(records), "actual": len(receipt_paths)})
    if checkpoint_count != manifest.get("checkpoint_count"):
        issues.append({"code": "CHECKPOINT_COUNT_MISMATCH", "expected": manifest.get("checkpoint_count"), "actual": checkpoint_count})
    summary_native_matches_expected = False
    summary_quality_passed = False
    try:
        summary_table = pd.read_csv(run_root / "summary.csv")
        if len(summary_table) != len(records):
            issues.append({"code": "SUMMARY_FRAME_COUNT_MISMATCH", "expected": len(records), "actual": len(summary_table)})
        if "native_frame" in summary_table:
            summary_native_frames = [int(value) for value in summary_table["native_frame"].tolist()]
            summary_native_matches_expected = summary_native_frames == [int(row["native_frame"]) for row in records]
        if not summary_native_matches_expected:
            issues.append({"code": "SUMMARY_FRAME_IDENTITY_MISMATCH"})
        if "status" in summary_table and "scientific_ok" in summary_table:
            summary_quality_passed = bool(
                len(summary_table)
                and summary_table["status"].eq("complete").all()
                and summary_table["scientific_ok"].eq(True).all()
            )
        if manifest.get("completed_frame_count") != len(summary_table):
            issues.append({"code": "COMPLETED_FRAME_COUNT_MISMATCH", "expected": len(summary_table), "actual": manifest.get("completed_frame_count")})
    except Exception as exc:
        issues.append({"code": "SUMMARY_INVALID", "message": str(exc)})
    numerical_quality_passed = (
        not issues
        and state.get("status") == "complete"
        and manifest.get("run_status") == "complete"
        and manifest.get("expected_frame_count") == len(records)
        and manifest.get("completed_frame_count") == len(records)
        and manifest.get("native_frames_match_expected") is True
        and summary_native_matches_expected
        and summary_quality_passed
        and manifest.get("numerical_quality_passed", manifest.get("scientific_ok") is True) is True
    )
    return {"project": str(root), "valid": not issues, "scientific_ok": numerical_quality_passed,
            "numerical_quality_passed": numerical_quality_passed,
            "quality_gate_scope": manifest.get("quality_gate_scope", QUALITY_GATE_SCOPE),
            "scientific_acceptance": manifest.get("scientific_acceptance", "not_assessed"),
            "experimental_uncertainty_calibrated": manifest.get("experimental_uncertainty_calibrated", False),
            "issues": issues, "input_fingerprint": input_sha, "checkpoint_count": checkpoint_count, "receipt_count": len(receipt_paths), "expected_frame_count": len(records), "run_status": state.get("status")}


def _batch_one(project_path: str, field_csv: bool, make_plots: bool) -> dict[str, Any]:
    try:
        result = analyze(project_path, field_csv=field_csv, make_plots=make_plots)
        run_result = result.get("run", {})
        scientific_pass = run_result.get("scientific_ok") is True and run_result.get("status") == "complete"
        numerical_status = "pass" if scientific_pass else "attention_required"
        return {"project": project_path, "execution_status": "complete",
                "scientific_status": numerical_status, "numerical_quality_status": numerical_status,
                "scientific_acceptance": "not_assessed", "result": result}
    except AgentError as exc:
        return {"project": project_path, "execution_status": "failed", "scientific_status": "not_evaluated", "numerical_quality_status": "not_evaluated", "scientific_acceptance": "not_assessed", "error": exc.as_dict()}
    except Exception as exc:
        return {"project": project_path, "execution_status": "failed", "scientific_status": "not_evaluated", "numerical_quality_status": "not_evaluated", "scientific_acceptance": "not_assessed", "error": {"error_code": "BATCH_RUNTIME_ERROR", "message": f"{type(exc).__name__}: {exc}", "path": "$"}}


def batch(projects: Sequence[str | os.PathLike[str]], *, max_workers: int = 2, field_csv: bool = False, make_plots: bool = True) -> dict[str, Any]:
    """Run independent sample projects with bounded cross-project concurrency."""
    paths = [str(_project_root(item)) for item in projects]
    if not paths:
        raise AgentError("BATCH_EMPTY", "batch requires at least one project", path="$.projects")
    if len(set(path.casefold() for path in paths)) != len(paths):
        raise AgentError("BATCH_DUPLICATE_PROJECT", "each project may appear only once in a batch", path="$.projects")
    if type(max_workers) is not int or max_workers < 1 or max_workers > 8:
        raise AgentError("BATCH_WORKERS", "max_workers must be in [1, 8]", path="$.max_workers")
    results = []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(paths)), mp_context=multiprocessing.get_context("spawn")) as executor:
        future_map = {executor.submit(_batch_one, path, field_csv, make_plots): path for path in paths}
        for future in as_completed(future_map):
            project_path = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "project": project_path, "execution_status": "failed",
                    "scientific_status": "not_evaluated", "numerical_quality_status": "not_evaluated",
                    "scientific_acceptance": "not_assessed",
                    "error": {"error_code": "BATCH_WORKER_FAILED", "message": f"{type(exc).__name__}: {exc}", "path": project_path},
                })
    results.sort(key=lambda item: paths.index(item["project"]))
    completed = sum(item["execution_status"] == "complete" for item in results)
    passed = sum(item["scientific_status"] == "pass" for item in results)
    attention = sum(item["scientific_status"] == "attention_required" for item in results)
    failed = len(results) - completed
    return {
        "status": "complete" if failed == 0 else "partial",
        "scientific_status": "pass" if completed == len(results) and passed == len(results) else "attention_required",
        "numerical_quality_status": "pass" if completed == len(results) and passed == len(results) else "attention_required",
        "scientific_acceptance": "not_assessed",
        "experimental_uncertainty_calibrated": False,
        "project_count": len(results), "execution_completed": completed, "execution_failed": failed,
        "scientific_passed": passed, "attention_required": attention, "results": results,
    }


_GAUGE_COLUMNS = ["frame_position_1based", "native_frame", "time_s", "gauge_name", "axis", "geometry_status", "status",
                  "initial_length_px", "initial_length_mm", "current_length_px", "extension_px", "current_length_mm", "extension_mm",
                  "a_expected_poi_count", "b_expected_poi_count", "a_valid_poi_count", "b_valid_poi_count", "endpoint_policy",
                  "engineering_strain", "true_strain", "calibration_status", "measurement_kind", "loading_axis", "min_zncc", "reasons"]
_REGION_COLUMNS = ["frame_position_1based", "native_frame", "time_s", "frame_status", "region_name", "component", "coordinate_system",
                   "coordinate_unit", "bounds", "unit", "n_poi_total", "n_poi_valid", "valid_fraction", "n_displacement_valid",
                   "n_strain_valid", "displacement_valid_fraction", "strain_valid_fraction", "min_valid_pois", "min_valid_fraction", "min_zncc",
                   "mean", "median", "sd_sample", "min", "max", "status", "reasons", "source", "interpolation", "calibration_status",
                   "quality_ok", "sd_interpretation", "field_support_bounds", "coverage_interpretation"]


def _region_rows(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from .regions import summarize_regions
    frame = {"frame_position_1based": metadata.get("frame_position"), "native_frame": metadata.get("native_frame"),
             "time_s": metadata.get("time_s"), "frame_status": metadata.get("status")}
    if arrays:
        field = _field_from_arrays(arrays, metadata)
        field["calibration"] = config.get("calibration")
        rows = summarize_regions(field, config.get("regions") or [], quality_ok=metadata.get("status") == "complete")
    else:
        rows = [{"region_name": region["name"], "bounds": region["bounds"], "component": component,
                 "coordinate_system": region["coordinate_system"], "coordinate_unit": "px" if region["coordinate_system"] == "image" else "mm",
                 "unit": "dimensionless", "status": "unavailable", "reasons": ["frame_failed"], "source": "original_poi",
                 "quality_ok": False, "interpolation": "none", "n_poi_total": 0, "n_poi_valid": 0}
                for region in config.get("regions") or [] for component in ("Exx", "Eyy", "Exy")]
    return [{**frame, **row} for row in rows]


def _write_region_outputs(run_root: Path, config: Mapping[str, Any], plan: Mapping[str, Any], run_fingerprint: str) -> None:
    rows = (row for arrays, metadata in _iter_checkpoint_rows(run_root, validate_fingerprint=run_fingerprint)
            for row in _region_rows(arrays, metadata, config)) if config.get("regions") else iter(())
    _csv_atomic(run_root / "regions.csv", rows, _REGION_COLUMNS)
    _atomic_json(run_root / "geometry.json", {"schema": "straintrace.geometry/v1", "regions": config.get("regions", []),
                 "gauges": config.get("gauges", []), "gauge_reference_geometry": plan.get("gauge_reference_geometry", []),
                 "analysis_rectangle_candidate_px": plan.get("analysis_rectangle_candidate_px"),
                 "reference_native_frame": plan.get("reference_native_frame"), "reference_path": plan.get("reference_path"),
                 "mask_file": config.get("mask_file"), "calibration": config.get("calibration"),
                 "specimen_geometry_confirmed": False, "loading_axis": None})


def reprocess(project: str | os.PathLike[str], *, regions: Sequence[Mapping[str, Any]] | None = None,
              gauges: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Extract new regions/gauges from saved original POIs without rerunning DIC."""
    root = _project_root(project)
    with _project_lock(root, purpose="reprocess"):
        checked = verify(root)
        if not checked.get("valid"):
            raise AgentError("REPROCESS_SOURCE_INVALID", "current run verification failed", details=checked.get("issues"))
        config = _read_config(root)
        candidate = {**config, "regions": config["regions"] if regions is None else list(regions),
                     "gauges": config["gauges"] if gauges is None else list(gauges)}
        derived_config = _validate_project_config(candidate, root)
        pointer, run_root = _load_run_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        source_manifest = _read_json(run_root / "manifest.json")
        reference = _load_image(plan["reference_path"])
        parameters = plan["selected_parameters"]
        geometry = _resolve_gauge_geometry(derived_config["gauges"], plan["analysis_rectangle_candidate_px"], reference.shape,
                     parameters["subset_size_px"], parameters["step_px"], _load_specimen_mask(config, reference.shape),
                     calibration=config.get("calibration"))
        derived_plan = {**plan, "gauge_reference_geometry": geometry, "source_plan_sha256": plan["plan_sha256"],
                        "configuration_sha256": _hash_json(derived_config), "analysis_kind": "saved_field_postprocessing"}
        derived_plan.pop("plan_sha256")
        derived_plan["plan_sha256"] = _hash_json(derived_plan)
        receipt = {"schema": "straintrace.reprocess/v1", "source_run": run_root.name,
                   "run_fingerprint": source_manifest["run_fingerprint"], "source_plan_sha256": plan["plan_sha256"],
                   "regions": derived_config["regions"], "gauges": derived_config["gauges"],
                   "checkpoint_receipts": source_manifest["checkpoint_receipts"], "solver_recomputed": False}
        destination = run_root / "extractions" / f"reprocess-{_hash_json(receipt)[:20]}"
        if destination.exists():
            saved = _read_json(destination / "reprocess.json")
            if saved != receipt:
                raise AgentError("REPROCESS_IDENTITY_CONFLICT", "existing extraction has a different identity")
            from .reporting import verify_report
            report_check = verify_report(destination / "report")
            if not report_check.get("valid"):
                raise AgentError("REPROCESS_ARTIFACT_INVALID", "saved extraction report failed verification", details=report_check)
            saved_result = _read_json(destination / "result.json")
            measurement_status = saved_result["status"]
            return {"project": str(root), "run_id": run_root.name, "status": "reused" if measurement_status == "complete" else measurement_status,
                    "output_dir": str(destination), "solver_recomputed": False,
                    "postprocessing_quality": saved_result.get("postprocessing_quality"),
                    "report": {"report_html": str(destination / "report/index.html"), "verification": report_check},
                    "artifacts": {name: str(destination / name) for name in ("summary.csv", "gauges.csv", "regions.csv", "geometry.json", "reprocess.json")}}
        # A failed extraction remains outside the published artifact tree and
        # can never be mistaken for a completed immutable result.
        extraction = run_root / "_postprocess_staging" / uuid.uuid4().hex
        extraction.mkdir(parents=True)
        with (run_root / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            summary_columns = list(reader.fieldnames or [])
            summaries = {int(row["native_frame"]): row for row in reader}
        gauge_rows, region_rows = [], []
        for arrays, metadata in _iter_checkpoint_rows(run_root, validate_fingerprint=source_manifest["run_fingerprint"]):
            frame = {"frame_position_1based": metadata.get("frame_position"), "native_frame": metadata["native_frame"], "time_s": metadata.get("time_s")}
            field = _field_from_arrays(arrays, metadata) if arrays else {}
            measured = _measure_gauges(field, geometry, quality_ok=metadata.get("status") == "complete")
            gauge_rows.extend({**frame, **row} for row in measured)
            region_rows.extend(_region_rows(arrays, metadata, derived_config))
            summary = summaries[int(metadata["native_frame"])]
            summary.update(_gauge_group_summary(measured))
            summary["engineering_strain"] = summary["dic_virtual_gauge_engineering_strain"]
        _csv_atomic(extraction / "summary.csv", summaries.values(), summary_columns)
        _csv_atomic(extraction / "gauges.csv", gauge_rows, _GAUGE_COLUMNS)
        _csv_atomic(extraction / "regions.csv", region_rows, _REGION_COLUMNS)
        _atomic_json(extraction / "reprocess.json", receipt)
        _atomic_json(extraction / "geometry.json", {"regions": derived_config["regions"], "gauges": derived_config["gauges"], "gauge_reference_geometry": geometry})
        complete = source_manifest.get("numerical_quality_passed") is True and all(row.get("status") == "valid" for row in (*gauge_rows, *region_rows))
        def measurement_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
            valid = sum(row.get("status") == "valid" for row in rows)
            return {"total": len(rows), "valid": valid, "unavailable": len(rows) - valid}
        postprocessing_quality = {
            "schema": "straintrace.postprocessing-quality/v1",
            "status": "complete" if complete else "attention_required",
            "region_measurements": measurement_counts(region_rows),
            "gauge_measurements": measurement_counts(gauge_rows),
            "source_dic_numerical_quality_passed": source_manifest.get("numerical_quality_passed") is True,
            "source_run_id": run_root.name,
        }
        # Keep the source DIC gate distinct from this extraction's support gate.
        # The report carries quality.json, so this status survives relocation.
        derived_quality = {**_read_json(run_root / "quality.json"), "postprocessing_quality": postprocessing_quality}
        _atomic_json(extraction / "quality.json", derived_quality)
        _atomic_json(extraction / "result.json", {"status": postprocessing_quality["status"], "solver_recomputed": False,
                                                  "postprocessing_quality": postprocessing_quality})
        manifest = {**source_manifest, "artifacts": {path.name: _sha256(path) for path in extraction.iterdir() if path.is_file()},
                    "postprocessing": receipt, "configuration_sha256": _hash_json(derived_config),
                    "postprocessing_quality": postprocessing_quality,
                    "source_configuration_sha256": source_manifest["configuration_sha256"], "plan_sha256": derived_plan["plan_sha256"]}
        _atomic_json(extraction / "manifest.json", manifest)
        from .reporting import build_report
        portable_report = build_report(extraction, config=derived_config, plan=derived_plan, regions_csv=extraction / "regions.csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(extraction, destination)
        def relocate(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(str(extraction)):
                return str(destination) + value[len(str(extraction)):]
            if isinstance(value, dict):
                return {key: relocate(item) for key, item in value.items()}
            if isinstance(value, list):
                return [relocate(item) for item in value]
            return value
        portable_report = relocate(portable_report)
        _refresh_manifest_artifacts(run_root)
        return {"project": str(root), "run_id": run_root.name, "status": "complete" if complete else "attention_required",
                "output_dir": str(destination), "solver_recomputed": False, "report": portable_report,
                "postprocessing_quality": postprocessing_quality,
                "artifacts": {name: str(destination / name) for name in ("summary.csv", "gauges.csv", "regions.csv", "geometry.json", "reprocess.json")}}


def read_project_config(project: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the validated, editable configuration without mutating the project."""
    return _read_config(_project_root(project))


def update_project(project: str | os.PathLike[str], changes: Mapping[str, Any]) -> dict[str, Any]:
    """Validate top-level replacements, archive the old state and invalidate plans."""
    root = _project_root(project)
    if not isinstance(changes, Mapping):
        raise AgentError("CONFIG_TYPE_ERROR", "changes must be an object", path="$.changes")
    immutable = {"schema", "project_id", "image_dir", "mask_sha256"}
    if immutable.intersection(changes):
        raise AgentError("CONFIG_IMMUTABLE_FIELD", f"cannot update {sorted(immutable.intersection(changes))[0]}", path="$.changes")
    with _project_lock(root, purpose="update"):
        previous = _read_config(root)
        candidate = {**previous, **dict(changes)}
        if "mask_file" in changes:
            candidate["mask_sha256"] = _sha256(Path(str(candidate["mask_file"]))) if candidate.get("mask_file") else None
        validated = _validate_project_config(candidate, root)
        if _hash_json(validated) == _hash_json(previous):
            return {"project": str(root), "status": "unchanged", "configuration_sha256": _hash_json(previous)}
        records = _scan_inventory(validated)
        fingerprint = _inventory_fingerprint(validated, records)
        pointer = _current_pointer(root)
        revision = root / "configuration_history" / f"{time.time_ns()}-{_hash_json(previous)[:12]}.json"
        _atomic_json(revision, {"schema": "straintrace.configuration-revision/v1", "config": previous,
                                "pointer": pointer, "replacement_sha256": _hash_json(validated)})
        # The old run is retained. A crash between these writes cannot make it
        # current: all execution/export boundaries compare configuration hashes.
        _atomic_json(root / "project.json", validated)
        _record_inventory(root, records, fingerprint)
        pointer.update(current_plan=None, current_run=None)
        _atomic_json(root / "CURRENT.json", pointer)
        return {"project": str(root), "status": "updated", "configuration_sha256": _hash_json(validated),
                "previous_configuration": str(revision), "next_action": "analyze", "previous_runs_preserved": True}


def discover_experiment(folder: str | os.PathLike[str], *, objective: str = "full_field", options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from .discovery import discover_experiment as discover
    return discover(folder, objective=objective, options=options)


def _config_init_options(config: Mapping[str, Any]) -> dict[str, Any]:
    options = {key: value for key, value in config.items() if key not in {"schema", "project_id", "loads", "mask_sha256"}}
    if config.get("loads"):
        names = {"path": "loads_csv", "time_column": "load_time_column", "force_column": "load_force_column",
                 "time_unit": "load_time_unit", "time_offset_s": "load_time_offset_s"}
        for key, value in config["loads"].items():
            if not key.endswith("_basis"):
                options[names.get(key, key)] = value
        for basis, unit in (("time_unit_basis", "load_time_unit"), ("force_unit_basis", "force_unit"), ("displacement_unit_basis", "displacement_unit")):
            if config["loads"].get(basis) == "canonical_column_name":
                options[unit] = None
    return options


def auto_analyze(folder: str | os.PathLike[str], project: str | os.PathLike[str], *, objective: str = "full_field",
                 options: Mapping[str, Any] | None = None, progress: Callable[[dict[str, Any]], None] | None = None,
                 field_csv: bool = False, make_plots: bool = True) -> dict[str, Any]:
    """Discover one sequence and execute the established reproducible workflow."""
    root = _project_root(project)
    existing = _read_config(root) if (root / "project.json").is_file() else None
    if options is not None and not isinstance(options, Mapping):
        raise AgentError("CONFIG_TYPE_ERROR", "options must be an object", path="$.options")
    effective_options = {**(_config_init_options(existing) if existing else {}), **dict(options or {})}
    discovery = discover_experiment(folder, objective=objective, options=effective_options)
    result: dict[str, Any] = {"project": str(root), "discovery": discovery, "status": discovery["status"],
                             "scientific_acceptance": "not_assessed"}
    if discovery["status"] != "ready":
        return result
    if existing:
        import inspect
        binding = inspect.signature(init_project).bind(root, **discovery["init_options"])
        binding.apply_defaults()
        args = dict(binding.arguments)
        args.pop("project")
        candidate = _config_from_arguments(**args)
        candidate["project_id"] = existing["project_id"]
        if _hash_json(candidate) != _hash_json(existing):
            result.update(status="needs_input", required_inputs=[{"field": "project", "reason": "Supplied options change the existing configuration. Apply update_project explicitly or select a new project."}])
            return result
    else:
        result["initialization"] = init_project(root, **discovery["init_options"])
    _atomic_json(root / "discovery.json", discovery)
    config = _read_config(root)
    try:
        plan = _current_plan_if_fresh(root, config) or plan_project(root, progress=progress)
    except AgentError as exc:
        if exc.code not in {"TEXTURE_INSUFFICIENT", "ROI_TOO_SMALL", "IMAGE_TOO_SMALL", "ROI_OUTSIDE_IMAGE", "REFERENCE_FRAME_UNREADABLE"}:
            raise
        result.update(status="review_required", error=exc.as_dict())
        return result
    result["plan"] = plan
    if plan.get("status") != "ready":
        result["status"] = "review_required"
        return result
    result["run"] = run(root, progress=progress, field_csv=field_csv, make_plots=make_plots)
    if result["run"].get("status") == "cancelled":
        result["status"] = "cancelled"
        run_root = Path(result["run"]["run_dir"])
        result["report"] = {"report_html": str(run_root / "report" / "index.html"), "partial": True}
        return result
    expected_run = str(result["run"]["run_id"])
    result["report"] = report(root, include_checkpoints=True, expected_run_id=expected_run)
    # A writer can change CURRENT between the independently locked steps.
    # Bind verification, goal checks and the final receipt to this same run.
    with _project_lock(root, purpose="auto_finalize"):
        _require_expected_run(root, expected_run)
        result["verification"] = verify(root)
        complete = result["run"].get("status") == "complete" and result["run"].get("numerical_quality_passed") is True and result["verification"].get("valid") is True
        result["status"] = "complete" if complete else "attention_required"
        if config.get("regions"):
            region_table = pd.read_csv(Path(result["run"]["run_dir"]) / "regions.csv")
            if region_table.empty or not region_table["status"].eq("valid").all():
                result.update(status="attention_required", region_issue="One or more requested regions lack valid measurement support.")
        if discovery["objective"] in {"virtual_gauge", "stress_strain"}:
            run_root = Path(result["run"]["run_dir"])
            table = pd.read_csv(run_root / "summary.csv")
            gauge_table = pd.read_csv(run_root / "gauges.csv")
            expected = {(int(frame), gauge["name"]) for frame in table["native_frame"] for gauge in config["gauges"]}
            actual = list(zip(gauge_table["native_frame"].astype(int), gauge_table["gauge_name"]))
            available = (len(actual) == len(expected) and set(actual) == expected and gauge_table["status"].eq("valid").all()
                         and np.isfinite(pd.to_numeric(gauge_table["engineering_strain"], errors="coerce")).all())
            if discovery["objective"] == "stress_strain":
                available = available and "stress_MPa" in table and np.isfinite(pd.to_numeric(table["stress_MPa"], errors="coerce")).all()
            if not available:
                result.update(status="attention_required", objective_issue="One or more frames lack the requested supported gauge or stress measurement.")
        _atomic_json(root / "automation.json", {"schema": "straintrace.automation/v1", "objective": discovery["objective"],
                                               "status": result["status"], "run_id": expected_run,
                                               "verification_valid": result["verification"].get("valid"), "scientific_acceptance": "not_assessed"})
    return result


def _require_expected_run(root: Path, expected_run_id: str) -> None:
    _, run_root = _load_run_pointer(root)
    if run_root.name != expected_run_id:
        raise AgentError("AUTOMATION_STALE_RUN", "the current project run changed during automatic analysis; retry with the intended configuration",
                         details={"expected_run_id": expected_run_id, "current_run_id": run_root.name})


def report(project: str | os.PathLike[str], *, include_checkpoints: bool = True, expected_run_id: str | None = None) -> dict[str, Any]:
    """Rebuild the portable report package from a verified current run."""
    root = _project_root(project)
    with _project_lock(root, purpose="report"):
        if expected_run_id is not None:
            _require_expected_run(root, expected_run_id)
        checked = verify(root)
        # A killed report commit can leave only its regenerated output hashes
        # stale. Rebuild those outputs after every underlying source still
        # passes verification; never waive a source or checkpoint failure.
        issues = checked.get("issues") or []
        report_only = bool(issues) and all(
            issue.get("code") in {"ARTIFACT_MISSING", "ARTIFACT_CHECKSUM_MISMATCH"}
            and isinstance(issue.get("path"), str) and issue["path"].startswith("report/")
            for issue in issues
        )
        if not checked.get("valid") and not report_only:
            raise AgentError("REPORT_SOURCE_INVALID", "current run verification failed", details=checked.get("issues"))
        config = _read_config(root)
        pointer, run_root = _load_run_pointer(root)
        plan = _read_json(_plan_path(root, pointer))
        result = _build_report_transaction(run_root, config, plan, include_checkpoints=include_checkpoints)
        return {"run_id": run_root.name, **result}
