"""Strict, GUI-free command line interface for ezDIC.

The CLI owns the JSON boundary and process contract. Numerical work and
transactional publication remain in :mod:`ezdic_core`; this module imports the
core lazily only for ``run`` and ``verify-manifest`` so ``--help`` is safe in a
headless process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "run_config_v1.json"
if not SCHEMA_PATH.is_file():
    from importlib.resources import files

    SCHEMA_PATH = Path(str(files("straintrace_data").joinpath("run_config_v1.json")))
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 2
EXIT_RUNTIME_ERROR = 3
EXIT_GATE_ERROR = 4

IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"})


class ConfigError(ValueError):
    """Machine-readable configuration/input validation error."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.path = path

    def as_dict(self) -> dict[str, Any]:
        return {"error_code": self.code, "message": self.message, "path": self.path}


def _fail(code: str, message: str, path: str = "$") -> None:
    raise ConfigError(code, message, path)


def _path(parent: str, key: str | int) -> str:
    return f"{parent}[{key}]" if isinstance(key, int) else f"{parent}.{key}"


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("CONFIG_TYPE_ERROR", "expected a JSON object", path)
    return value


def _array(value: Any, path: str, *, min_items: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        _fail("CONFIG_TYPE_ERROR", "expected a JSON array", path)
    if min_items is not None and len(value) < min_items:
        _fail("CONFIG_VALUE_ERROR", f"must contain at least {min_items} item(s)", path)
    return value


def _reject_unknown(obj: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        _fail("CONFIG_UNKNOWN_FIELD", f"unknown field: {unknown[0]}", _path(path, unknown[0]))


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:  # noqa: E721 - bool/float/string are not JSON integers here.
        _fail("CONFIG_TYPE_ERROR", "expected a JSON integer (not a string or float)", path)
    if minimum is not None and value < minimum:
        _fail("CONFIG_VALUE_ERROR", f"must be >= {minimum}", path)
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("CONFIG_TYPE_ERROR", "expected a JSON number", path)
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        _fail("CONFIG_NONFINITE_NUMBER", "number must be finite", path)
    numeric = float(value)
    if minimum is not None and (numeric <= minimum if exclusive_minimum else numeric < minimum):
        operator = ">" if exclusive_minimum else ">="
        _fail("CONFIG_VALUE_ERROR", f"must be {operator} {minimum:g}", path)
    if maximum is not None and numeric > maximum:
        _fail("CONFIG_VALUE_ERROR", f"must be <= {maximum:g}", path)
    return value


def _nullable_number(value: Any, path: str, *, minimum: float | None = None, exclusive_minimum: bool = False) -> int | float | None:
    if value is None:
        return None
    return _number(value, path, minimum=minimum, exclusive_minimum=exclusive_minimum)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("CONFIG_TYPE_ERROR", "expected a non-empty JSON string", path)
    if "\x00" in value:
        _fail("CONFIG_VALUE_ERROR", "NUL is not allowed in strings or paths", path)
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:  # noqa: E721 - reject 0/1 as well.
        _fail("CONFIG_TYPE_ERROR", "expected a JSON boolean", path)
    return value


def _strict_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def parse_config_text(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text, parse_constant=_strict_json_constant)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ConfigError("CONFIG_JSON_ERROR", f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        _fail("CONFIG_SCHEMA_ERROR", "configuration root must be a JSON object")
    return value


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError("CONFIG_INPUT_ERROR", f"cannot read config: {exc}") from exc
    return parse_config_text(text)


def load_schema() -> dict[str, Any]:
    schema_candidates = [SCHEMA_PATH]
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        root = Path(bundle_root).resolve()
        schema_candidates.extend((root / "schemas" / "run_config_v1.json", root / "sources" / "schemas" / "run_config_v1.json"))
    schema_path = next((path for path in schema_candidates if path.is_file()), schema_candidates[0])
    try:
        value = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError("SCHEMA_FILE_ERROR", f"cannot read schema: {exc}") from exc
    if not isinstance(value, dict):
        _fail("SCHEMA_FILE_ERROR", "schema root must be an object")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(config: Mapping[str, Any]) -> str:
    # Runtime-only bindings (notably ``_code_paths``) must never alter the
    # user-facing configuration identity recorded in a manifest.
    canonical_snapshot = config.get("_canonical_config")
    if isinstance(canonical_snapshot, Mapping):
        payload = dict(canonical_snapshot)
    else:
        payload = {key: value for key, value in config.items() if key not in {"_code_paths", "_canonical_config"}}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _walk_finite(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        _number(value, path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_finite(item, _path(path, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail("CONFIG_TYPE_ERROR", "object keys must be strings", path)
            _walk_finite(item, _path(path, key))
        return
    _fail("CONFIG_TYPE_ERROR", "unsupported JSON value", path)


def _canonical_path(value: str, base_dir: Path | None) -> str:
    path = Path(value).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return os.fspath(path.resolve(strict=False))


def _rect(value: Any, path: str) -> list[int]:
    values = _array(value, path)
    if len(values) != 4:
        _fail("CONFIG_VALUE_ERROR", "ROI must be [x, y, width, height]", path)
    result = [_integer(item, _path(path, index), minimum=0) for index, item in enumerate(values)]
    if result[2] <= 0 or result[3] <= 0:
        _fail("CONFIG_VALUE_ERROR", "ROI width and height must be > 0", path)
    return result


def _roi_group(value: Any, index: int) -> dict[str, Any]:
    path = f"$.roi_groups[{index}]"
    group = _object(value, path)
    allowed = {"name", "roi1", "roi2", "strain_mode", "role"}
    _reject_unknown(group, allowed, path)
    if "roi1" not in group or "roi2" not in group:
        _fail("CONFIG_SCHEMA_ERROR", "ROI group requires roi1 and roi2", path)
    name = _string(group.get("name", f"ROI{index + 1}"), _path(path, "name"))
    strain_mode = _string(group.get("strain_mode", "auto"), _path(path, "strain_mode"))
    if strain_mode not in {"auto", "x", "y", "distance"}:
        _fail("CONFIG_VALUE_ERROR", "strain_mode must be auto, x, y, or distance", _path(path, "strain_mode"))
    role = _string(group.get("role", "none"), _path(path, "role"))
    if role not in {"none", "axial", "transverse"}:
        _fail("CONFIG_VALUE_ERROR", "role must be none, axial, or transverse", _path(path, "role"))
    return {
        "name": name,
        "roi1": _rect(group["roi1"], _path(path, "roi1")),
        "roi2": _rect(group["roi2"], _path(path, "roi2")),
        "strain_mode": strain_mode,
        "role": role,
    }


_DEFAULT_TRACKING = {
    "search_radius_px": 30,
    "hard_corr": 0.55,
    "soft_corr": 0.35,
    "enable_adaptive": True,
    "use_prev_frame_template": False,
    "template_policy": "fixed_reference",
    "template_alpha": 0.70,
    "max_frame_jump_px": None,
    "pixel_size_mm": None,
}
_DEFAULT_TEXTURE = {
    "min_texture_std": 8.0,
    "min_texture_contrast": 25.0,
    "max_saturated_frac": 0.20,
    "min_structure_ratio": 0.02,
}
_DEFAULT_QUALITY = {
    "zncc_min": 0.75,
    "peak_margin_min": 0.02,
    "best_to_second_peak_ratio_min": 1.02,
    "min_correlation_valid_fraction": 0.95,
    "min_strain_valid_fraction": 0.80,
    "min_valid_frames": 1,
    "min_valid_frame_ratio": 0.0,
    "max_condition_number": 1e12,
    "max_residual_rms": None,
    "reject_nonconverged": False,
    "enable_fb_check": True,
    "fb_tolerance_px": 12.0,
}
_DEFAULT_SOLVER = {
    "name": "IC-GN",
    "subset_size_px": 21,
    "step_px": 5,
    "strain_window_px": 5,
    "smooth_sigma_poi": 0.0,
    "search_radius_px": 20,
    "max_iterations": 25,
    "tolerance": 1e-3,
    "strain_degree": 2,
    "robust_strain": True,
    "outlier_threshold_px": 1.0,
    "initialization": "auto",
}
_DEFAULT_PYRAMID = {"levels": 1, "scale": 0.5}
_DEFAULT_NORMALIZATION = {
    "policy": "reference_percentile",
    "lower_percentile": 1.0,
    "upper_percentile": 99.0,
    "bounds": None,
    "clip": True,
}
_DEFAULT_EXPORT = {
    "write_manifest": True,
    "write_qc": True,
    "write_full_csv": True,
    "write_origin_txt": False,
    "write_origin_opju": False,
    "write_engineering_png": False,
    "write_publication_figures": False,
    "write_corr_plot": False,
    "write_overlays": False,
    "write_parameters": True,
    "origin_opju_required": False,
}
_DEFAULT_TRANSACTION = {
    "enabled": True,
    "archive_previous": True,
    "retain_failed_staging": True,
}


def _section(raw: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if raw is None:
        return {}
    value = _object(raw, f"$.{name}")
    _reject_unknown(value, allowed, f"$.{name}")
    return value


def _normalize_tracking(raw: Any) -> dict[str, Any]:
    source = _section(raw, "tracking", set(_DEFAULT_TRACKING))
    result = dict(_DEFAULT_TRACKING)
    result.update(source)
    result["search_radius_px"] = _integer(result["search_radius_px"], "$.tracking.search_radius_px", minimum=1)
    for key in ("hard_corr", "soft_corr"):
        result[key] = _number(result[key], f"$.tracking.{key}", minimum=0, maximum=1)
    result["enable_adaptive"] = _boolean(result["enable_adaptive"], "$.tracking.enable_adaptive")
    result["use_prev_frame_template"] = _boolean(result["use_prev_frame_template"], "$.tracking.use_prev_frame_template")
    result["template_policy"] = _string(result["template_policy"], "$.tracking.template_policy")
    if result["template_policy"] not in {"fixed_reference", "follow_deformed_experimental"}:
        _fail("CONFIG_VALUE_ERROR", "template_policy must be fixed_reference or follow_deformed_experimental", "$.tracking.template_policy")
    if result["template_policy"] == "fixed_reference" and result["use_prev_frame_template"]:
        _fail("CONFIG_VALUE_ERROR", "fixed_reference template_policy requires use_prev_frame_template=false", "$.tracking.use_prev_frame_template")
    if result["template_policy"] == "follow_deformed_experimental" and not result["use_prev_frame_template"]:
        _fail("CONFIG_VALUE_ERROR", "follow_deformed_experimental requires use_prev_frame_template=true", "$.tracking.use_prev_frame_template")
    result["template_alpha"] = _number(result["template_alpha"], "$.tracking.template_alpha", minimum=0, maximum=1)
    result["max_frame_jump_px"] = _nullable_number(result["max_frame_jump_px"], "$.tracking.max_frame_jump_px", minimum=0, exclusive_minimum=True)
    result["pixel_size_mm"] = _nullable_number(result["pixel_size_mm"], "$.tracking.pixel_size_mm", minimum=0, exclusive_minimum=True)
    return result


def _normalize_texture(raw: Any) -> dict[str, Any]:
    source = _section(raw, "texture", set(_DEFAULT_TEXTURE))
    result = dict(_DEFAULT_TEXTURE)
    result.update(source)
    result["min_texture_std"] = _number(result["min_texture_std"], "$.texture.min_texture_std", minimum=0)
    result["min_texture_contrast"] = _number(result["min_texture_contrast"], "$.texture.min_texture_contrast", minimum=0)
    result["max_saturated_frac"] = _number(result["max_saturated_frac"], "$.texture.max_saturated_frac", minimum=0, maximum=1)
    result["min_structure_ratio"] = _number(result["min_structure_ratio"], "$.texture.min_structure_ratio", minimum=0, maximum=1)
    return result


def _normalize_quality(raw: Any, mode: str) -> dict[str, Any]:
    if mode == "extensometer":
        allowed = {
            "peak_margin_min",
            "best_to_second_peak_ratio_min",
            "min_strain_valid_fraction",
            "min_valid_frames",
            "min_valid_frame_ratio",
            "enable_fb_check",
            "fb_tolerance_px",
        }
        defaults = {key: _DEFAULT_QUALITY[key] for key in allowed}
    else:
        allowed = {
            "zncc_min",
            "peak_margin_min",
            "best_to_second_peak_ratio_min",
            "min_correlation_valid_fraction",
            "min_strain_valid_fraction",
            "min_valid_frames",
            "min_valid_frame_ratio",
            "max_condition_number",
            "max_residual_rms",
            "reject_nonconverged",
        }
        defaults = {key: _DEFAULT_QUALITY[key] for key in allowed}
    source = _section(raw, "quality", allowed)
    result = defaults
    result.update(source)
    if mode == "fullfield":
        result["zncc_min"] = _number(result["zncc_min"], "$.quality.zncc_min", minimum=0, maximum=1)
    result["peak_margin_min"] = _number(result["peak_margin_min"], "$.quality.peak_margin_min", minimum=0)
    result["best_to_second_peak_ratio_min"] = _number(result["best_to_second_peak_ratio_min"], "$.quality.best_to_second_peak_ratio_min", minimum=1)
    if mode == "fullfield":
        result["min_correlation_valid_fraction"] = _number(result["min_correlation_valid_fraction"], "$.quality.min_correlation_valid_fraction", minimum=0, maximum=1)
    result["min_strain_valid_fraction"] = _number(result["min_strain_valid_fraction"], "$.quality.min_strain_valid_fraction", minimum=0, maximum=1)
    result["min_valid_frames"] = _integer(result["min_valid_frames"], "$.quality.min_valid_frames", minimum=1)
    result["min_valid_frame_ratio"] = _number(result["min_valid_frame_ratio"], "$.quality.min_valid_frame_ratio", minimum=0, maximum=1)
    if mode == "fullfield":
        result["max_condition_number"] = _number(result["max_condition_number"], "$.quality.max_condition_number", minimum=0, exclusive_minimum=True)
        try:
            result["max_residual_rms"] = _nullable_number(result["max_residual_rms"], "$.quality.max_residual_rms", minimum=0, exclusive_minimum=True)
        except ConfigError as exc:
            raise ConfigError(
                exc.code,
                "max_residual_rms is normalized grayscale residual RMS (dimensionless): " + exc.message,
                "$.quality.max_residual_rms",
            ) from exc
        result["reject_nonconverged"] = _boolean(result["reject_nonconverged"], "$.quality.reject_nonconverged")
    else:
        result["enable_fb_check"] = _boolean(result["enable_fb_check"], "$.quality.enable_fb_check")
        result["fb_tolerance_px"] = _number(result["fb_tolerance_px"], "$.quality.fb_tolerance_px", minimum=0)
    return result


def _normalize_solver(raw: Any) -> dict[str, Any]:
    source = _section(raw, "solver", set(_DEFAULT_SOLVER))
    result = dict(_DEFAULT_SOLVER)
    result.update(source)
    result["name"] = _string(result["name"], "$.solver.name")
    if result["name"] not in {"IC-GN", "IC-LM"}:
        _fail("CONFIG_VALUE_ERROR", "solver name must be IC-GN or IC-LM", "$.solver.name")
    result["subset_size_px"] = _integer(result["subset_size_px"], "$.solver.subset_size_px", minimum=9)
    if result["subset_size_px"] % 2 == 0:
        _fail("CONFIG_VALUE_ERROR", "subset_size_px must be odd", "$.solver.subset_size_px")
    result["step_px"] = _integer(result["step_px"], "$.solver.step_px", minimum=1)
    result["strain_window_px"] = _integer(result["strain_window_px"], "$.solver.strain_window_px", minimum=3)
    if result["strain_window_px"] % 2 == 0:
        _fail("CONFIG_VALUE_ERROR", "strain_window_px must be odd", "$.solver.strain_window_px")
    result["smooth_sigma_poi"] = _number(result["smooth_sigma_poi"], "$.solver.smooth_sigma_poi", minimum=0)
    result["search_radius_px"] = _integer(result["search_radius_px"], "$.solver.search_radius_px", minimum=1)
    result["max_iterations"] = _integer(result["max_iterations"], "$.solver.max_iterations", minimum=1)
    result["tolerance"] = _number(result["tolerance"], "$.solver.tolerance", minimum=0, exclusive_minimum=True)
    result["strain_degree"] = _integer(result["strain_degree"], "$.solver.strain_degree", minimum=1)
    if result["strain_degree"] not in (1, 2):
        _fail("CONFIG_VALUE_ERROR", "strain degree must be 1 or 2", "$.solver.strain_degree")
    result["robust_strain"] = _boolean(result["robust_strain"], "$.solver.robust_strain")
    result["outlier_threshold_px"] = _number(result["outlier_threshold_px"], "$.solver.outlier_threshold_px", minimum=0)
    if result["initialization"] not in ("auto","local"):
        _fail("CONFIG_VALUE_ERROR", "initialization must be auto or local", "$.solver.initialization")
    return result


def _normalize_mask(raw, base_dir):
    result = {"mode": "none", "path": None, "texture_threshold": 3.0, "exclusions": []}
    result.update(_section(raw, "mask", set(result)))
    if result["mode"] not in ("none", "auto", "file"):
        _fail("CONFIG_VALUE_ERROR", "mask mode must be none, auto or file", "$.mask.mode")
    result["texture_threshold"] = _number(result["texture_threshold"], "$.mask.texture_threshold", minimum=0, exclusive_minimum=True)
    result["exclusions"] = [_rect(rect, f"$.mask.exclusions[{index}]") for index, rect in enumerate(_array(result["exclusions"], "$.mask.exclusions"))]
    if result["mode"] == "file":
        result["path"] = _canonical_path(_string(result["path"], "$.mask.path"), base_dir)
    elif result["path"] is not None:
        _fail("CONFIG_VALUE_ERROR", "mask path is only used with file mode", "$.mask.path")
    return result


def _normalize_display(raw):
    result = {"style": "contour", "color_mode": "range", "vmin": None, "vmax": None,
              "coordinate_frame": "reference", "background": "none", "percent": False, "alpha": .65,
              "cmap": "RdBu_r"}
    result.update(_section(raw, "display", set(result)))
    for key, allowed in (("style", ("contour", "points")), ("color_mode", ("range", "symmetric", "manual")),
                         ("coordinate_frame", ("reference", "deformed")), ("background", ("none", "reference", "deformed")),
                         ("cmap", ("RdBu_r", "viridis", "cividis", "turbo"))):
        if result[key] not in allowed:
            _fail("CONFIG_VALUE_ERROR", f"{key} must be one of {allowed}", f"$.display.{key}")
    result["percent"] = _boolean(result["percent"], "$.display.percent")
    result["alpha"] = _number(result["alpha"], "$.display.alpha", minimum=0, maximum=1)
    if result["color_mode"] == "manual":
        result["vmin"] = _number(result["vmin"], "$.display.vmin")
        result["vmax"] = _number(result["vmax"], "$.display.vmax")
        if result["vmin"] >= result["vmax"]:
            _fail("CONFIG_VALUE_ERROR", "display limits must be increasing", "$.display.vmax")
    else:
        if result["vmin"] is not None or result["vmax"] is not None:
            _fail("CONFIG_VALUE_ERROR", "display limits require manual color mode", "$.display.color_mode")
    return result


def _normalize_pyramid(raw: Any) -> dict[str, Any]:
    source = _section(raw, "pyramid", set(_DEFAULT_PYRAMID))
    result = dict(_DEFAULT_PYRAMID)
    result.update(source)
    result["levels"] = _integer(result["levels"], "$.pyramid.levels", minimum=1)
    if result["levels"] > 8:
        _fail("CONFIG_VALUE_ERROR", "pyramid levels must be <= 8", "$.pyramid.levels")
    result["scale"] = _number(result["scale"], "$.pyramid.scale", minimum=0, maximum=1, exclusive_minimum=True)
    return result


def _normalize_normalization(raw: Any) -> dict[str, Any]:
    source = _section(raw, "normalization", set(_DEFAULT_NORMALIZATION))
    result = dict(_DEFAULT_NORMALIZATION)
    result.update(source)
    result["policy"] = _string(result["policy"], "$.normalization.policy")
    if result["policy"] not in {"reference_percentile", "fixed_bounds"}:
        _fail("CONFIG_VALUE_ERROR", "normalization policy must be reference_percentile or fixed_bounds", "$.normalization.policy")
    result["lower_percentile"] = _number(result["lower_percentile"], "$.normalization.lower_percentile", minimum=0, maximum=100)
    result["upper_percentile"] = _number(result["upper_percentile"], "$.normalization.upper_percentile", minimum=0, maximum=100)
    if result["lower_percentile"] >= result["upper_percentile"]:
        _fail("CONFIG_VALUE_ERROR", "lower_percentile must be < upper_percentile", "$.normalization")
    result["clip"] = _boolean(result["clip"], "$.normalization.clip")
    if result["clip"] is not True:
        _fail("CONFIG_VALUE_ERROR", "normalization clip must be true", "$.normalization.clip")
    bounds = result["bounds"]
    if result["policy"] == "fixed_bounds":
        bound_obj = _object(bounds, "$.normalization.bounds")
        _reject_unknown(bound_obj, {"lo", "hi"}, "$.normalization.bounds")
        if "lo" not in bound_obj or "hi" not in bound_obj:
            _fail("CONFIG_SCHEMA_ERROR", "fixed_bounds requires bounds.lo and bounds.hi", "$.normalization.bounds")
        lo = _number(bound_obj["lo"], "$.normalization.bounds.lo")
        hi = _number(bound_obj["hi"], "$.normalization.bounds.hi")
        if lo >= hi:
            _fail("CONFIG_VALUE_ERROR", "normalization bounds require lo < hi", "$.normalization.bounds")
        result["bounds"] = {"lo": lo, "hi": hi}
    elif bounds is not None:
        _fail("CONFIG_VALUE_ERROR", "bounds are allowed only with fixed_bounds policy", "$.normalization.bounds")
    return result


def _normalize_export(raw: Any) -> dict[str, Any]:
    source = _section(raw, "export", set(_DEFAULT_EXPORT))
    result = dict(_DEFAULT_EXPORT)
    result.update(source)
    for key in result:
        result[key] = _boolean(result[key], _path("$.export", key))
    if result["write_manifest"] is not True:
        _fail("CONFIG_VALUE_ERROR", "export.write_manifest must be true for a verifiable run", "$.export.write_manifest")
    return result


def _normalize_transaction(raw: Any) -> dict[str, Any]:
    source = _section(raw, "transaction", set(_DEFAULT_TRANSACTION))
    result = dict(_DEFAULT_TRANSACTION)
    result.update(source)
    for key in result:
        result[key] = _boolean(result[key], _path("$.transaction", key))
    invalid = [key for key, value in result.items() if value is not True]
    if invalid:
        _fail(
            "CONFIG_VALUE_ERROR",
            "transaction.enabled/archive_previous/retain_failed_staging must all be true",
            _path("$.transaction", invalid[0]),
        )
    return result


def _folder_image_count(path: Path) -> int | None:
    if not path.exists():
        return None
    if not path.is_dir():
        _fail("CONFIG_INPUT_ERROR", "image_folder is not a directory", "$.image_folder")
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def normalize_config(config: Mapping[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    root = _object(dict(config), "$")
    allowed = {
        "schema_version",
        "analysis_mode",
        "image_paths",
        "image_folder",
        "start_frame_1based",
        "end_frame_1based",
        "reference_frame_1based",
        "output_dir",
        "roi_groups",
        "field_roi",
        "field_roi_reference_frame_1based",
        "tracking",
        "texture",
        "quality",
        "solver",
        "pyramid",
        "mask",
        "display",
        "normalization",
        "export",
        "transaction",
        "metadata",
    }
    _reject_unknown(root, allowed, "$")
    required = ("schema_version", "analysis_mode", "start_frame_1based", "end_frame_1based", "reference_frame_1based", "output_dir")
    for key in required:
        if key not in root:
            _fail("CONFIG_SCHEMA_ERROR", f"missing required field: {key}", _path("$", key))
    version = _integer(root["schema_version"], "$.schema_version", minimum=1)
    if version != SCHEMA_VERSION:
        _fail("CONFIG_SCHEMA_ERROR", f"unsupported schema_version {version}; expected {SCHEMA_VERSION}", "$.schema_version")
    mode = _string(root["analysis_mode"], "$.analysis_mode")
    if mode not in {"extensometer", "fullfield"}:
        _fail("CONFIG_VALUE_ERROR", "analysis_mode must be extensometer or fullfield", "$.analysis_mode")

    has_paths = "image_paths" in root
    has_folder = "image_folder" in root
    if has_paths == has_folder:
        _fail("CONFIG_SCHEMA_ERROR", "provide exactly one of image_paths or image_folder", "$.image_paths")
    normalized: dict[str, Any] = {"schema_version": version, "analysis_mode": mode}
    frame_count: int | None
    if has_paths:
        paths = _array(root["image_paths"], "$.image_paths", min_items=2)
        normalized_paths: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(paths):
            path = _canonical_path(_string(item, _path("$.image_paths", index)), base_dir)
            identity = os.path.normcase(path)
            if identity in seen:
                _fail("CONFIG_VALUE_ERROR", "image_paths must not contain duplicates", "$.image_paths")
            seen.add(identity)
            normalized_paths.append(path)
        normalized["image_paths"] = normalized_paths
        frame_count = len(normalized_paths)
    else:
        folder = _canonical_path(_string(root["image_folder"], "$.image_folder"), base_dir)
        normalized["image_folder"] = folder
        frame_count = _folder_image_count(Path(folder))
        if frame_count == 0:
            _fail("CONFIG_INPUT_ERROR", "image_folder contains no supported image files", "$.image_folder")

    start = _integer(root["start_frame_1based"], "$.start_frame_1based", minimum=1)
    end = _integer(root["end_frame_1based"], "$.end_frame_1based", minimum=1)
    reference = _integer(root["reference_frame_1based"], "$.reference_frame_1based", minimum=1)
    if end <= start:
        _fail("CONFIG_INPUT_ERROR", "at least two selected frames are required", "$.end_frame_1based")
    if reference < start or reference > end:
        _fail("CONFIG_VALUE_ERROR", "reference_frame_1based must be within the selected frame range", "$.reference_frame_1based")
    if frame_count is not None and end > frame_count:
        _fail("CONFIG_INPUT_ERROR", f"end_frame_1based {end} exceeds {frame_count} input frame(s)", "$.end_frame_1based")
    if mode == "extensometer" and reference != start:
        _fail("CONFIG_VALUE_ERROR", "extensometer requires reference_frame_1based == start_frame_1based", "$.reference_frame_1based")
    normalized.update({"start_frame_1based": start, "end_frame_1based": end, "reference_frame_1based": reference})
    normalized["output_dir"] = _canonical_path(_string(root["output_dir"], "$.output_dir"), base_dir)

    if mode == "extensometer":
        for key in ("field_roi", "field_roi_reference_frame_1based", "solver", "pyramid", "mask", "display"):
            if key in root:
                _fail("CONFIG_MODE_FIELD", f"{key} is not valid in extensometer mode", _path("$", key))
        groups = _array(root.get("roi_groups"), "$.roi_groups", min_items=1)
        normalized_groups = [_roi_group(item, index) for index, item in enumerate(groups)]
        names = [item["name"] for item in normalized_groups]
        if len(set(names)) != len(names):
            _fail("CONFIG_VALUE_ERROR", "ROI group names must be unique", "$.roi_groups")
        normalized["roi_groups"] = normalized_groups
        normalized["tracking"] = _normalize_tracking(root.get("tracking"))
    else:
        for key in ("roi_groups", "tracking"):
            if key in root:
                _fail("CONFIG_MODE_FIELD", f"{key} is not valid in fullfield mode", _path("$", key))
        if "field_roi" not in root:
            _fail("CONFIG_SCHEMA_ERROR", "fullfield mode requires field_roi", "$.field_roi")
        normalized["field_roi"] = _rect(root["field_roi"], "$.field_roi")
        field_reference = _integer(root.get("field_roi_reference_frame_1based", reference), "$.field_roi_reference_frame_1based", minimum=1)
        if field_reference != reference:
            _fail("CONFIG_VALUE_ERROR", "field ROI reference must equal reference_frame_1based", "$.field_roi_reference_frame_1based")
        normalized["field_roi_reference_frame_1based"] = field_reference
        normalized["solver"] = _normalize_solver(root.get("solver"))
        normalized["pyramid"] = _normalize_pyramid(root.get("pyramid"))
        normalized["mask"] = _normalize_mask(root.get("mask"), base_dir)
        normalized["display"] = _normalize_display(root.get("display"))

    normalized["texture"] = _normalize_texture(root.get("texture"))
    normalized["quality"] = _normalize_quality(root.get("quality"), mode)
    normalized["normalization"] = _normalize_normalization(root.get("normalization"))
    normalized["export"] = _normalize_export(root.get("export"))
    if mode == "fullfield" and normalized["export"]["write_parameters"] is not True:
        _fail("CONFIG_VALUE_ERROR", "fullfield requires export.write_parameters=true", "$.export.write_parameters")
    normalized["transaction"] = _normalize_transaction(root.get("transaction"))
    metadata = _object(root.get("metadata", {}), "$.metadata")
    _walk_finite(metadata, "$.metadata")
    normalized["metadata"] = metadata
    return normalized


validate_config = normalize_config


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass
    return str(value)


def _import_core() -> Any:
    try:
        import ezdic_core as core  # noqa: PLC0415 - lazy import is the Tk-free contract.
    except Exception as exc:
        raise RuntimeError(f"cannot import ezdic_core: {exc}") from exc
    return core


def _resolve_runtime_paths(normalized: Mapping[str, Any], core: Any) -> list[str]:
    if "image_paths" in normalized:
        paths = [str(value) for value in normalized["image_paths"]]
    else:
        try:
            paths = [os.fspath(value) for value in core.collect_images(normalized["image_folder"])]
        except Exception as exc:
            raise ConfigError("CONFIG_INPUT_ERROR", f"cannot collect image_folder: {exc}", "$.image_folder") from exc
    if len(paths) < 2:
        raise ConfigError("CONFIG_INPUT_ERROR", "at least two input images are required", "$.image_paths")
    for index, path in enumerate(paths):
        if not Path(path).is_file():
            raise ConfigError("CONFIG_INPUT_ERROR", f"input image does not exist: {path}", _path("$.image_paths", index))
    if int(normalized["end_frame_1based"]) > len(paths):
        raise ConfigError("CONFIG_INPUT_ERROR", "selected frame range exceeds collected inputs", "$.end_frame_1based")
    return paths


def _resolve_code_paths(core_path: Path | None = None) -> list[Path]:
    """Resolve the complete real source set for source and PyInstaller layouts.

    The release spec stores executable source copies below ``sources/`` while
    keeping the schema at the bundle root.  A frozen module's ``__file__`` can
    point into an archive, so resolution must be based on ``_MEIPASS`` and must
    include every executable implementation used by a run.
    """

    module_path = Path(core_path).resolve() if core_path is not None else None
    roots: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        roots.append(Path(bundle_root).resolve())
    roots.append(Path(__file__).resolve().parent)
    if module_path is not None and module_path.exists():
        roots.append(module_path.parent)
    seen_roots: set[str] = set()
    for root in roots:
        root_key = os.path.normcase(os.fspath(root))
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        candidates_sets = [
            # Actual ezDIC.spec layout: source copies in ``sources/`` and the
            # schema in the root ``schemas/`` directory.
            [
                root / "sources" / "ezdic_core.py",
                root / "sources" / "ezdic_cli.py",
                root / "sources" / "ezdic_benchmark.py",
                root / "schemas" / "run_config_v1.json",
            ],
            # Compatibility layout used by older source/frozen fixtures where
            # all data were copied below ``sources/``.
            [
                root / "sources" / "ezdic_core.py",
                root / "sources" / "ezdic_cli.py",
                root / "sources" / "ezdic_benchmark.py",
                root / "sources" / "schemas" / "run_config_v1.json",
            ],
            [
                root / "ezdic_core.py",
                root / "ezdic_cli.py",
                root / "ezdic_benchmark.py",
                root / "schemas" / "run_config_v1.json",
            ],
        ]
        for candidates in candidates_sets:
            if all(path.is_file() for path in candidates):
                return candidates
        # A frozen runtime must never fall back to hashing only the importer or
        # schema when one executable source copy is absent.  That would create
        # a superficially valid but incomplete code fingerprint.
        if bundle_root:
            raise ConfigError(
                "CODE_FILE_SET_INCOMPLETE",
                "frozen bundle must contain ezdic_core.py, ezdic_cli.py, ezdic_benchmark.py, and run_config_v1.json",
                "$.code_paths",
            )
    fallback = [
        module_path if module_path is not None and module_path.is_file() else Path(__file__).resolve(),
        Path(__file__).resolve(),
        SCHEMA_PATH.resolve(),
    ]
    return list(dict.fromkeys(path for path in fallback if path.is_file()))


class _CoreSettings(dict[str, Any]):
    """Runtime aliases with a canonical snapshot for core manifest sealing."""

    def __init__(self, runtime: Mapping[str, Any], canonical: Mapping[str, Any]) -> None:
        super().__init__(runtime)
        self._canonical_snapshot = copy.deepcopy(dict(canonical))

    def items(self):  # type: ignore[override]
        # ezdic_core._canonical_settings snapshots through items(). Keep
        # private runtime aliases and _code_paths out of the config hash.
        return self._canonical_snapshot.items()


def build_core_settings(normalized: Mapping[str, Any], core: Any) -> dict[str, Any]:
    """Map canonical config to the current public core settings boundary."""

    settings = copy.deepcopy(dict(normalized))
    settings["image_paths"] = _resolve_runtime_paths(normalized, core)
    core_path = Path(getattr(core, "__file__", __file__)).resolve()
    settings["_code_paths"] = _resolve_code_paths(core_path)
    settings["_canonical_config"] = copy.deepcopy(dict(normalized))
    quality = dict(settings["quality"])
    if normalized["analysis_mode"] == "fullfield" and quality["max_residual_rms"] is None:
        # ``null`` is the canonical disabled value; infinity is only a private
        # compatibility value required by the current core call signature.
        quality["max_residual_rms"] = float("inf")
    settings["quality"] = quality
    settings["peak_margin_min"] = quality["peak_margin_min"]
    settings["peak_ratio_min"] = quality["best_to_second_peak_ratio_min"]
    settings["min_strain_valid_fraction"] = quality["min_strain_valid_fraction"]
    if normalized["analysis_mode"] == "fullfield":
        settings["max_residual_rms"] = quality["max_residual_rms"]
        settings["max_condition_number"] = quality["max_condition_number"]
        settings["min_correlation_valid_fraction"] = quality["min_correlation_valid_fraction"]
        settings["reject_nonconverged"] = quality["reject_nonconverged"]

    if normalized["analysis_mode"] == "extensometer":
        tracking = normalized["tracking"]
        settings.update(
            {
                "search_radius_base": tracking["search_radius_px"],
                "hard_corr": tracking["hard_corr"],
                "soft_corr": tracking["soft_corr"],
                "enable_adaptive": tracking["enable_adaptive"],
                "use_prev_frame_template": tracking["use_prev_frame_template"],
                "template_alpha": tracking["template_alpha"],
                "max_frame_jump": tracking["max_frame_jump_px"],
                "enable_fb_check": quality["enable_fb_check"],
                "fb_tolerance": quality["fb_tolerance_px"],
                "pixel_size_mm": tracking["pixel_size_mm"],
                "min_valid_frames": quality["min_valid_frames"],
                "min_strain_valid_ratio": quality["min_strain_valid_fraction"],
            }
        )
    else:
        solver = normalized["solver"]
        pyramid = normalized["pyramid"]
        settings.update(
            {
                "field_roi": tuple(normalized["field_roi"]),
                "subset_size": solver["subset_size_px"],
                "step": solver["step_px"],
                "strain_window": solver["strain_window_px"],
                "smooth_sigma": solver["smooth_sigma_poi"],
                "search_radius": solver["search_radius_px"],
                "solver_name": solver["name"],
                "max_iter": solver["max_iterations"],
                "conv_tol": solver["tolerance"],
                "pyramid_levels": pyramid["levels"],
                "pyramid_scale": pyramid["scale"],
                "strain_degree": solver["strain_degree"],
                "robust_strain": solver["robust_strain"],
                "outlier_threshold_px": solver["outlier_threshold_px"],
                "initialization": solver["initialization"],
                "mask": normalized["mask"],
                "display": normalized["display"],
            }
        )
    return _CoreSettings(settings, normalized)


def _core_error_code(exc: BaseException) -> str:
    return str(getattr(exc, "code", "RUNTIME_ERROR"))


_CONFIG_CORE_CODES = {
    "NO_INPUT_IMAGES",
    "INPUT_FILE_ERROR",
    "INVALID_IMAGE",
    "NONFINITE_IMAGE",
    "IMAGE_DIMENSION_MISMATCH",
    "INVALID_FRAME_RANGE",
    "INSUFFICIENT_FRAMES",
    "INVALID_REFERENCE_FRAME",
    "UNSUPPORTED_REFERENCE_ORDER",
    "DUPLICATE_INPUT_IMAGES",
    "NO_ROI_GROUPS",
    "INVALID_ROI_GROUP",
    "ROI_OUT_OF_BOUNDS",
    "NO_FIELD_ROI",
    "INVALID_FIELD_ROI",
    "UNUSABLE_POI_GRID",
    "INVALID_DIC_SETTINGS",
    "INVALID_NORMALIZATION_POLICY",
    "INVALID_NORMALIZATION_BOUNDS",
    "UNSUPPORTED_NORMALIZATION_CLIP",
}
_GATE_CORE_CODES = {
    "LOW_TEXTURE",
    "SATURATED_TEXTURE",
    "AMBIGUOUS_TEXTURE",
    "SCIENTIFIC_GATE_FAILED",
    "MANIFEST_INVALID",
    "MANIFEST_VERIFY_FAILED",
}


def _exit_for_core_error(exc: BaseException) -> int:
    code = _core_error_code(exc)
    if code in _CONFIG_CORE_CODES:
        return EXIT_CONFIG_ERROR
    if code in _GATE_CORE_CODES:
        return EXIT_GATE_ERROR
    return EXIT_RUNTIME_ERROR


def _write_error(code: str, message: str, *, path: str | None = None, exit_code: int | None = None) -> None:
    payload: dict[str, Any] = {"error_code": str(code), "message": str(message)}
    if path is not None:
        payload["path"] = path
    if exit_code is not None:
        payload["exit_code"] = int(exit_code)
    sys.stderr.write(canonical_json(payload) + "\n")


def _emit_progress(event: Mapping[str, Any]) -> None:
    sys.stdout.write(canonical_json(_json_safe(dict(event))) + "\n")
    sys.stdout.flush()


def _result_summary(result: Mapping[str, Any], normalized: Mapping[str, Any], manifest: Mapping[str, Any], verification: Mapping[str, Any]) -> dict[str, Any]:
    candidate = result.get("json_summary") if isinstance(result.get("json_summary"), Mapping) else {}
    manifest_path = result.get("manifest_path") or candidate.get("manifest_path")
    status = result.get("status") or candidate.get("status") or manifest.get("status")
    scientific = result.get("scientific_ok")
    if scientific is None:
        scientific = candidate.get("scientific_ok", manifest.get("scientific_ok"))
    integrity = result.get("integrity_ok")
    if integrity is None:
        integrity = candidate.get("integrity_ok", verification.get("ok", False))
    integrity = bool(integrity is True and verification.get("ok") is True)
    gate = candidate.get("gate") or result.get("gate") or manifest.get("scientific_gate")
    outputs = result.get("outputs")
    if outputs is None:
        outputs = manifest.get("outputs", [])
    return {
        "status": str(status) if status is not None else "unknown",
        "scientific_ok": bool(scientific is True),
        "integrity_ok": bool(integrity is True),
        "manifest_verified": bool(verification.get("ok") is True),
        "mode": normalized["analysis_mode"],
        "manifest_path": os.fspath(manifest_path) if manifest_path is not None else None,
        "config_hash": config_hash(normalized),
        "gate": _json_safe(gate),
        "output_count": len(outputs) if isinstance(outputs, (list, tuple)) else 0,
        "warnings": _json_safe(result.get("warnings", [])),
        "errors": _json_safe(result.get("errors", [])),
    }


def _run_command(config_path: Path, progress_json: bool) -> int:
    normalized: dict[str, Any] | None = None
    mode = None
    try:
        raw = load_config(config_path)
        normalized = normalize_config(raw, base_dir=config_path.resolve().parent)
        mode = normalized["analysis_mode"]
        output_path = Path(normalized["output_dir"])
        if output_path.exists() and not output_path.is_dir():
            raise ConfigError("CONFIG_INPUT_ERROR", "output_dir exists but is not a directory", "$.output_dir")
        core = _import_core()
        settings = build_core_settings(normalized, core)
        if progress_json:
            _emit_progress(
                {
                    "event": "run_started",
                    "mode": mode,
                    "fraction": 0.0,
                    "start_frame_1based": normalized["start_frame_1based"],
                    "end_frame_1based": normalized["end_frame_1based"],
                    "reference_frame_1based": normalized["reference_frame_1based"],
                    "status": "running",
                }
            )

        def progress(payload: Any = None, *args: Any, **kwargs: Any) -> None:
            if not progress_json:
                return
            details = dict(payload) if isinstance(payload, Mapping) else {}
            if isinstance(payload, (int, float)):
                details["fraction"] = float(payload)
            details.update(kwargs)
            fraction = details.get("fraction", args[0] if args and isinstance(args[0], (int, float)) else 0.0)
            try:
                fraction = max(0.0, min(1.0, float(fraction)))
            except (TypeError, ValueError):
                fraction = 0.0
            _emit_progress(
                {
                    "event": "progress",
                    "mode": mode,
                    "fraction": fraction,
                    "frame_global_1based": details.get("frame_global_1based"),
                    "status": details.get("status", "running"),
                }
            )

        runner = core.run_extensometer_sequence if mode == "extensometer" else core.run_fullfield_sequence
        result = runner(settings, progress_callback=progress)
        if not isinstance(result, Mapping):
            raise RuntimeError("core runner returned a non-object result")
        manifest_path_value = result.get("manifest_path")
        if not manifest_path_value:
            raise RuntimeError("core runner returned no manifest_path")
        manifest_path = Path(manifest_path_value)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot read generated manifest: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise RuntimeError("generated manifest is not a JSON object")
        verification = _json_safe(core.verify_run_manifest(manifest_path, verify_code=True))
        summary = _result_summary(result, normalized, manifest, verification)
        if not summary["manifest_verified"]:
            exit_code = EXIT_GATE_ERROR
        elif summary["status"] not in {"completed", "completed_with_warnings"} or not summary["scientific_ok"] or not summary["integrity_ok"]:
            exit_code = EXIT_GATE_ERROR
        else:
            exit_code = EXIT_SUCCESS
        summary["exit_code"] = exit_code
        if progress_json:
            _emit_progress(
                {
                    "event": "run_finished",
                    "mode": mode,
                    "fraction": 1.0,
                    "frame_global_1based": None,
                    "status": summary["status"],
                    "scientific_ok": summary["scientific_ok"],
                    "integrity_ok": summary["integrity_ok"],
                    "manifest_verified": summary["manifest_verified"],
                    "manifest_path": summary["manifest_path"],
                    "exit_code": exit_code,
                }
            )
        else:
            sys.stdout.write(canonical_json(summary) + "\n")
        return exit_code
    except ConfigError as exc:
        if progress_json and mode is not None:
            _emit_progress({"event": "run_finished", "mode": mode, "fraction": 0.0, "frame_global_1based": None, "status": "config_error", "scientific_ok": False, "integrity_ok": False, "manifest_verified": False, "manifest_path": None, "exit_code": EXIT_CONFIG_ERROR})
        _write_error(exc.code, exc.message, path=exc.path, exit_code=EXIT_CONFIG_ERROR)
        return EXIT_CONFIG_ERROR
    except Exception as exc:
        exit_code = _exit_for_core_error(exc)
        if progress_json and mode is not None:
            _emit_progress({"event": "run_finished", "mode": mode, "fraction": 0.0, "frame_global_1based": None, "status": "failed", "scientific_ok": False, "integrity_ok": False, "manifest_verified": False, "manifest_path": None, "exit_code": exit_code})
        _write_error(_core_error_code(exc), str(exc), path="$.run", exit_code=exit_code)
        return exit_code


def _verify_command(manifest_path: Path) -> int:
    try:
        core = _import_core()
    except Exception as exc:
        _write_error("CORE_IMPORT_ERROR", str(exc), path="$.verify-manifest", exit_code=EXIT_RUNTIME_ERROR)
        return EXIT_RUNTIME_ERROR
    try:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError("MANIFEST_READ_ERROR", str(exc), "$.manifest") from exc
        if not isinstance(payload, dict):
            raise ConfigError("MANIFEST_SCHEMA_ERROR", "manifest root must be an object", "$.manifest")
        required = ("manifest_version", "manifest_hash", "config_hash", "inputs", "outputs", "code_fingerprint", "code_files", "status", "scientific_ok", "scientific_gate")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ConfigError("MANIFEST_REQUIRED_FIELD_MISSING", f"missing manifest fields: {', '.join(missing)}", "$.manifest")
        verification = _json_safe(core.verify_run_manifest(manifest_path, verify_code=True))
        scientific_ok = payload.get("scientific_ok") is True
        gate = payload.get("scientific_gate")
        gate_ok = isinstance(gate, Mapping) and gate.get("scientific_ok") is True
        status_ok = payload.get("status") in {"completed", "completed_with_warnings"}
        ok = bool(verification.get("ok") is True and scientific_ok and gate_ok and status_ok)
        summary = {
            "manifest": os.fspath(manifest_path.resolve()),
            "ok": ok,
            "integrity_ok": bool(verification.get("ok") is True),
            "scientific_ok": scientific_ok and gate_ok,
            "status": payload.get("status"),
            "errors": verification.get("errors", []) if not verification.get("ok") else ([] if status_ok and scientific_ok and gate_ok else [{"code": "SCIENTIFIC_GATE_FAILED"}]),
        }
        sys.stdout.write(canonical_json(summary) + "\n")
        if not ok:
            _write_error("MANIFEST_INVALID", "manifest integrity or scientific gate verification failed", path="$.manifest", exit_code=EXIT_GATE_ERROR)
            return EXIT_GATE_ERROR
        return EXIT_SUCCESS
    except ConfigError as exc:
        _write_error(exc.code, exc.message, path=exc.path, exit_code=EXIT_GATE_ERROR)
        return EXIT_GATE_ERROR
    except Exception as exc:
        _write_error("MANIFEST_VERIFY_ERROR", str(exc), path="$.manifest", exit_code=EXIT_GATE_ERROR)
        return EXIT_GATE_ERROR


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        _write_error("CLI_USAGE_ERROR", message, exit_code=EXIT_CONFIG_ERROR)
        raise SystemExit(EXIT_CONFIG_ERROR)


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="path to run_config_v1 JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="ezdic", description="Strict headless ezDIC CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    validate = subparsers.add_parser("validate-config", help="validate and print canonical configuration")
    _config_argument(validate)
    run = subparsers.add_parser("run", help="run an analysis without GUI/Tk")
    _config_argument(run)
    run.add_argument("--progress-json", action="store_true", help="emit line-delimited JSON progress events")
    verify = subparsers.add_parser("verify-manifest", help="recompute manifest/input/output/code integrity")
    verify.add_argument("--manifest", type=Path, required=True, help="path to run_manifest.json")
    benchmark = subparsers.add_parser("benchmark", help="run the locked synthetic benchmark")
    benchmark.add_argument("--cases", type=Path, help="exact locked benchmark document")
    benchmark.add_argument("--output", type=Path, help="benchmark output directory")
    return parser


def _benchmark_command(cases_path: Path | None, output_dir: Path | None) -> int:
    if output_dir is None:
        _write_error("BENCHMARK_OUTPUT_REQUIRED", "benchmark requires --output", path="$.benchmark.output", exit_code=EXIT_CONFIG_ERROR)
        return EXIT_CONFIG_ERROR
    try:
        from benchmarks.run_benchmark import BenchmarkError, run_benchmark  # noqa: PLC0415

        report = run_benchmark(cases_path=cases_path, output_dir=output_dir)
    except BenchmarkError as exc:
        _write_error(exc.code, str(exc), path="$.benchmark", exit_code=exc.exit_code)
        return exc.exit_code
    except Exception as exc:
        _write_error("BENCHMARK_RUNTIME_ERROR", str(exc), path="$.benchmark", exit_code=EXIT_RUNTIME_ERROR)
        return EXIT_RUNTIME_ERROR
    sys.stdout.write(canonical_json(report) + "\n")
    return int(report.get("exit_code", EXIT_RUNTIME_ERROR))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        try:
            raw = load_config(args.config)
            normalized = normalize_config(raw, base_dir=args.config.resolve().parent)
        except ConfigError as exc:
            _write_error(exc.code, exc.message, path=exc.path, exit_code=EXIT_CONFIG_ERROR)
            return EXIT_CONFIG_ERROR
        sys.stdout.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return EXIT_SUCCESS
    if args.command == "run":
        return _run_command(args.config, args.progress_json)
    if args.command == "verify-manifest":
        return _verify_command(args.manifest)
    if args.command == "benchmark":
        return _benchmark_command(args.cases, args.output)
    _write_error("CLI_USAGE_ERROR", f"unsupported command: {args.command}", exit_code=EXIT_CONFIG_ERROR)
    return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
