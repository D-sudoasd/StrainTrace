"""Project configuration normalization, validation, and plan-parameter checks."""
from __future__ import annotations

import math
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

from ._io import (
    AgentError,
    _api_attr,
    _hash_json,
    _json_safe,
    _project_root,
    _read_json,
    _sha256,
)


PROJECT_SCHEMA = "straintrace.agent-project/v1"
RUN_SCHEMA = "straintrace.agent-run/v1"
QUALITY_GATE_SCOPE = "existing ezdic_core correlation and strain validity thresholds; not experimental validation or scientific acceptance"
_CORE_SOLVER_PARAMETER_KEYS = ("subset_size_px", "step_px", "solver", "strain_window_poi", "search_radius_px")


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
    # Old v1 plans and sidecars predate the explicit policy and both omitted
    # it. Preserve their read-only comparison while refusing a mixed pair.
    key = "reject_nonconverged"
    if key not in recorded and key not in expected:
        return True
    if key not in recorded or key not in expected:
        return False
    if type(recorded[key]) is not bool or type(expected[key]) is not bool:
        return False
    if recorded[key] != expected[key]:
        return False
    return True


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


def _ensure_known_fields(value: Mapping[str, Any], allowed: set[str], *, path: str = "$") -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AgentError("CONFIG_UNKNOWN_FIELD", f"unknown field: {unknown[0]}", path=f"{path}.{unknown[0]}")


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


def _read_config(project: Path) -> dict[str, Any]:
    config = _read_json(project / "project.json")
    return _validate_project_config(config, project)


def read_project_config(project: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the validated, editable configuration without mutating the project."""
    return _read_config(_project_root(project))


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
    sha256_fn = _api_attr("_sha256", _sha256)
    if not isinstance(image_dir, (str, os.PathLike)) or not str(image_dir).strip():
        raise AgentError("CONFIG_TYPE_ERROR", "image_dir must be a non-empty path string", path="$.image_dir")
    for field_name, raw_path in (("frame_map_csv", frame_map_csv), ("loads_csv", loads_csv), ("mask_file", mask_file)):
        if raw_path is not None and (not isinstance(raw_path, (str, os.PathLike)) or not str(raw_path).strip()):
            raise AgentError("CONFIG_TYPE_ERROR", f"{field_name} must be a non-empty path string or null", path=f"$.{field_name}")
    for field_name, seq_val in (("regions", regions), ("paths", paths), ("gauges", gauges), ("exclusions_px", exclusions_px)):
        if seq_val is not None and not isinstance(seq_val, (list, tuple)):
            raise AgentError("CONFIG_TYPE_ERROR", f"{field_name} must be a JSON array", path=f"$.{field_name}")
    if paths is not None and any(not isinstance(item, Mapping) for item in paths):
        raise AgentError("CONFIG_TYPE_ERROR", "each paths entry must be an object", path="$.paths")
    resolved_mask = Path(mask_file).expanduser().resolve(strict=False) if mask_file is not None else None
    if resolved_mask is not None and not resolved_mask.is_file():
        raise AgentError("MASK_FILE_MISSING", f"mask image is missing: {resolved_mask}", path="$.mask_file")
    if isinstance(plot_every, bool) or not isinstance(plot_every, int) or plot_every < 0:
        raise AgentError("CONFIG_VALUE_ERROR", "plot_every must be a nonnegative integer", path="$.plot_every")
    config: dict[str, Any] = {
        "schema": PROJECT_SCHEMA,
        "project_id": uuid.uuid4().hex,
        "image_dir": str(Path(image_dir).expanduser().resolve(strict=False)),
        "frame_interval_s": frame_interval_s,
        "time_basis": time_basis,
        "frame_map_csv": str(Path(frame_map_csv).expanduser().resolve(strict=False)) if frame_map_csv is not None else None,
        "reference_native_frame": reference_native_frame,
        "loads": None,
        "calibration": None,
        "regions": list(regions or []),
        "paths": [dict(item) for item in (paths or [])],
        "gauges": list(gauges or []),
        "mask_file": str(resolved_mask) if resolved_mask is not None else None,
        "mask_sha256": sha256_fn(resolved_mask) if resolved_mask is not None else None,
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
        if not isinstance(load_time_column, str) or not load_time_column.strip():
            raise AgentError("CONFIG_TYPE_ERROR", "load_time_column must be a non-empty string", path="$.load_time_column")
        for col_name, col_value in (("load_force_column", load_force_column), ("displacement_column", displacement_column)):
            if col_value is not None and (not isinstance(col_value, str) or not col_value.strip()):
                raise AgentError("CONFIG_TYPE_ERROR", f"{col_name} must be a non-empty string or null", path=f"$.{col_name}")
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
        if not load_force_column and force_unit is not None:
            raise AgentError("CONFIG_VALUE_ERROR", "force_unit requires load_force_column", path="$.force_unit")
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
        if not isinstance(calibration, Mapping):
            raise AgentError("CONFIG_TYPE_ERROR", "calibration must be an object or null", path="$.calibration")
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
    return config


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
        required_load_fields = {
            "path", "time_column", "force_column", "time_unit", "time_unit_basis", "force_unit", "force_unit_basis",
            "displacement_column", "displacement_unit", "displacement_unit_basis", "displacement_source",
            "machine_gauge_length_mm", "clock_confirmed", "time_offset_s", "max_gap_s", "area_mm2",
        }
        _ensure_known_fields(loads, required_load_fields, path="$.loads")
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


def _config_init_options(config: Mapping[str, Any]) -> dict[str, Any]:
    options = {key: value for key, value in config.items() if key not in {"schema", "project_id", "loads", "mask_sha256"}}
    if config.get("loads"):
        names = {
            "path": "loads_csv", "time_column": "load_time_column", "force_column": "load_force_column",
            "time_unit": "load_time_unit", "time_offset_s": "load_time_offset_s",
        }
        for key, value in config["loads"].items():
            if not key.endswith("_basis"):
                options[names.get(key, key)] = value
        for basis, unit in (("time_unit_basis", "load_time_unit"), ("force_unit_basis", "force_unit"), ("displacement_unit_basis", "displacement_unit")):
            if config["loads"].get(basis) == "canonical_column_name":
                options[unit] = None
    return options
