"""Canonical, provenance-first synthetic quality benchmark for ezDIC."""

from __future__ import annotations

import csv
import hashlib
import importlib
import math
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .synthetic_cases import (
    CASE_DOCUMENT_VERSION,
    CORRUPTION_PANEL_VERSION,
    ERROR_TOLERANCE_PX,
    IMAGE_SHAPE,
    LOCKED_COORDINATES,
    QUALITY_SCORE_COMPONENTS,
    QUALITY_SCORE_NORMALIZATION,
    QUALITY_SCORE_VERSION,
    QUALITY_SCORE_VALIDITY_REQUIRED,
    ROI,
    STEP,
    SOLVER_SETTINGS,
    SUBSET_SIZE,
    TEXTURE_PREFLIGHT,
    apply_image_corruption,
    canonical_json,
    default_cases_path,
    load_locked_cases,
    locked_case_hash,
    make_case,
)


REPORT_VERSION = "ezdic-benchmark-report-v6"
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 3
EXIT_GATE_ERROR = 4


class BenchmarkError(RuntimeError):
    code = "BENCHMARK_RUNTIME_ERROR"
    exit_code = EXIT_RUNTIME_ERROR


class BenchmarkConfigError(BenchmarkError):
    code = "BENCHMARK_CONFIG_ERROR"
    exit_code = 2


class BenchmarkIOError(BenchmarkError):
    code = "BENCHMARK_IO_ERROR"
    exit_code = EXIT_RUNTIME_ERROR


class BenchmarkGateError(BenchmarkError):
    code = "BENCHMARK_GATE_FAILED"
    exit_code = EXIT_GATE_ERROR


BenchmarkResultError = BenchmarkGateError


_DIAGNOSTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "zncc": ("zncc",),
    "second_peak_margin": ("second_peak_margin", "peak_margin"),
    "residual_rms": ("residual_rms",),
    "hessian_condition_number": ("hessian_condition_number", "condition_number"),
    "iterations": ("iterations",),
    "converged": ("converged",),
    "stop_reason": ("stop_reason",),
}

_CSV_FIELDS = (
    "case_id",
    "run_id",
    "panel_variant_id",
    "point_index",
    "x",
    "y",
    "oracle_u",
    "oracle_v",
    "u_raw",
    "v_raw",
    "error_u_px",
    "error_v_px",
    "error_px",
    "oracle_Exx",
    "oracle_Eyy",
    "oracle_Exy",
    "Exx",
    "Eyy",
    "Exy",
    "strain_error_Exx",
    "strain_error_Eyy",
    "strain_error_Exy",
    "valid",
    "strain_valid",
    "accepted",
    "quality_accept",
    "quality_label_good",
    "false_accept",
    "false_reject",
    "quality_false_accept",
    "quality_false_reject",
    "quality_score",
    "quality_score_version",
    "corruption_applied",
    "diagnostic_source",
    "ratio_direction",
    "zncc",
    "second_peak_margin",
    "second_peak_ratio_best_over_second",
    "second_to_best_peak_ratio",
    "best_peak",
    "second_peak",
    "peak_margin",
    "peak_ratio",
    "residual_rms",
    "hessian_condition_number",
    "iterations",
    "converged",
    "stop_reason",
    "invalid_reason",
    "preflight_code",
)


def _machine_ambiguity_code(value: Any) -> bool:
    """Recognize only an exact machine-readable texture rejection code."""

    return isinstance(value, str) and value.strip().upper() == "AMBIGUOUS_TEXTURE"


def _ambiguity_result_passes(result: Mapping[str, Any], *, expected_points: int = 81) -> bool:
    """Validate a typed negative result without treating free text as evidence."""

    import numpy as np

    code = result.get("code") or result.get("error_code") or result.get("failure_code") or result.get("status")
    if not _machine_ambiguity_code(code) or "valid" not in result:
        return False
    valid = np.asarray(result["valid"])
    return valid.dtype.kind == "b" and valid.size == int(expected_points) and not bool(valid.any())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BenchmarkIOError(f"cannot hash benchmark artifact: {path}: {exc}") from exc
    return digest.hexdigest()


def _runtime_root() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parents[1]


def _module_hash(module: Any | None, module_name: str, *, package_name: str | None = None) -> str:
    """Resolve source files using the same archive fallback as frozen entrypoint."""

    candidates: list[Path] = []
    module_path = getattr(module, "__file__", None)
    if module_path:
        try:
            candidates.append(Path(module_path))
        except (TypeError, ValueError):
            pass
    root = _runtime_root()
    candidates.append(root / "sources" / f"{module_name}.py")
    if package_name:
        candidates.extend((root / "sources" / package_name / f"{module_name}.py", root / package_name / f"{module_name}.py"))
    for candidate in candidates:
        try:
            if candidate.is_file():
                return _sha256_path(candidate)
        except (OSError, TypeError, ValueError):
            continue
    return "unavailable"


def _code_provenance(core: Any | None, cases_path: Path | None = None) -> dict[str, Any]:
    runner = sys.modules.get(__name__)
    runner_hash = _module_hash(runner, "run_benchmark", package_name="benchmarks")
    synthetic = importlib.import_module("benchmarks.synthetic_cases")
    synthetic_hash = _module_hash(synthetic, "synthetic_cases", package_name="benchmarks")
    cases_path = default_cases_path() if cases_path is None else Path(cases_path)
    try:
        facade = importlib.import_module("ezdic_benchmark")
    except Exception:
        facade = None
    facade_hash = _module_hash(facade, "ezdic_benchmark")
    core_hash = _module_hash(core, "ezdic_core")
    try:
        cli = importlib.import_module("ezdic_cli")
    except Exception:
        cli = None
    cli_hash = _module_hash(cli, "ezdic_cli")
    try:
        cases_hash = _sha256_path(cases_path) if cases_path.is_file() else "unavailable"
    except BenchmarkIOError:
        cases_hash = "unavailable"
    return {
        "module": "benchmarks.run_benchmark",
        "source_sha256": runner_hash,
        "benchmark_runner_source_sha256": runner_hash,
        "benchmark_source_sha256": facade_hash,
        "synthetic_cases_source_sha256": synthetic_hash,
        "cases_json_sha256": cases_hash,
        "core_source_sha256": core_hash,
        "cli_source_sha256": cli_hash,
        "benchmark_facade": {"module": "ezdic_benchmark", "source_sha256": facade_hash},
        "core": {"module": "ezdic_core", "source_sha256": core_hash},
        "cli": {"module": "ezdic_cli", "source_sha256": cli_hash},
    }


def _as_array(field: Mapping[str, Any], names: Sequence[str], n: int, *, dtype: Any = float) -> Any:
    import numpy as np

    selected = next((name for name in names if name in field), None)
    if selected is None:
        raise BenchmarkGateError(f"missing benchmark field: {names[0]}")
    raw = np.asarray(field[selected])
    if dtype is bool and raw.dtype.kind != "b":
        raise BenchmarkGateError(f"benchmark field {selected} must be boolean")
    try:
        values = np.asarray(raw, dtype=dtype).reshape(-1)
    except Exception as exc:
        raise BenchmarkGateError(f"benchmark field {selected} cannot be converted") from exc
    if values.size != n:
        raise BenchmarkGateError(f"benchmark field {selected} has {values.size} values; expected {n}")
    return values


def _canonical_peak_ratio(
    values: Any,
    *,
    direction: str,
) -> tuple[Any, str]:
    """Return best/second ratio (always >= 1 for a valid tie/order)."""

    import numpy as np

    array = np.asarray(values, dtype=float)
    if direction == "best_over_second":
        return array, "best_over_second"
    if direction == "second_over_best":
        with np.errstate(divide="ignore", invalid="ignore"):
            inverted = np.divide(1.0, array, out=np.full(array.shape, np.nan, dtype=float), where=np.abs(array) > 1e-12)
        inverted[array == 0.0] = np.inf
        return inverted, "best_over_second_inverted"
    raise BenchmarkGateError(f"unknown peak-ratio direction: {direction}")


def _ratio_from_field(field: Mapping[str, Any], n: int) -> tuple[Any, Any, str]:
    import numpy as np

    if "best_to_second_peak_ratio" in field:
        raw = _as_array(field, ("best_to_second_peak_ratio",), n)
        ratio, direction = _canonical_peak_ratio(raw, direction="best_over_second")
    elif "peak_ratio" in field:
        raw = _as_array(field, ("peak_ratio",), n)
        ratio, direction = _canonical_peak_ratio(raw, direction="best_over_second")
    elif "second_to_best_peak_ratio" in field:
        raw = _as_array(field, ("second_to_best_peak_ratio",), n)
        ratio, direction = _canonical_peak_ratio(raw, direction="second_over_best")
    elif "second_peak_ratio" in field:
        raw = _as_array(field, ("second_peak_ratio",), n)
        ratio, direction = _canonical_peak_ratio(raw, direction="second_over_best")
    else:
        raise BenchmarkGateError("missing benchmark field: peak ratio")
    finite = np.isfinite(ratio)
    if np.any(ratio[finite] < 1.0 - 1e-9):
        raise BenchmarkGateError("best/second peak ratio is below one")
    return ratio, raw, direction


def _required_field(field: Any, case_id: str) -> tuple[dict[str, Any], int]:
    import numpy as np

    if not isinstance(field, Mapping):
        raise BenchmarkGateError(f"solver result for {case_id} is not an object")
    n = len(LOCKED_COORDINATES)
    diagnostics: dict[str, Any] = {
        "x": _as_array(field, ("x",), n),
        "y": _as_array(field, ("y",), n),
        "u_raw": _as_array(field, ("u_raw",), n),
        "v_raw": _as_array(field, ("v_raw",), n),
        "valid": _as_array(field, ("valid",), n, dtype=bool),
        "strain_valid": _as_array(field, ("strain_valid",), n, dtype=bool),
    }
    for logical_name, aliases in _DIAGNOSTIC_ALIASES.items():
        diagnostics[logical_name] = _as_array(field, aliases, n, dtype=object if logical_name == "stop_reason" else bool if logical_name == "converged" else float)
    ratio, raw_ratio, direction = _ratio_from_field(field, n)
    diagnostics["second_peak_ratio_best_over_second"] = ratio
    diagnostics["second_peak_ratio_raw"] = raw_ratio
    diagnostics["ratio_direction"] = direction
    diagnostics["invalid_reason"] = _as_array(field, ("invalid_reason",), n, dtype=object) if "invalid_reason" in field else np.full(n, "", dtype=object)
    for name in ("best_peak", "second_peak", "peak_margin", "peak_ratio"):
        diagnostics[name] = _as_array(field, (name,), n) if name in field else np.full(n, np.nan, dtype=float)
    expected = np.asarray(LOCKED_COORDINATES, dtype=float)
    if not np.array_equal(diagnostics["x"], expected[:, 0]) or not np.array_equal(diagnostics["y"], expected[:, 1]):
        raise BenchmarkGateError(f"solver coordinates do not exactly match locked POIs for {case_id}")
    raw_finite = np.isfinite(diagnostics["u_raw"]) & np.isfinite(diagnostics["v_raw"])
    accepted_finite = diagnostics["valid"] & raw_finite
    for name in ("zncc", "second_peak_margin", "second_peak_ratio_best_over_second", "residual_rms", "hessian_condition_number", "iterations"):
        values = np.asarray(diagnostics[name], dtype=float)
        if not np.isfinite(values[accepted_finite]).all():
            raise BenchmarkGateError(f"benchmark diagnostic {name} is missing on an accepted point")
        if name == "second_peak_margin" and np.any(values[accepted_finite] < 0.0):
            raise BenchmarkGateError("second-peak margin is negative")
        if name == "hessian_condition_number" and np.any(values[accepted_finite] <= 0.0):
            raise BenchmarkGateError("Hessian condition number is not positive")
        if name == "iterations" and np.any(values[accepted_finite] < 0.0):
            raise BenchmarkGateError("iteration count is negative")
        if name == "iterations" and np.any(values[accepted_finite] != np.floor(values[accepted_finite])):
            raise BenchmarkGateError("iteration count is not integral")
        # Floating-point normalized correlations can exceed one by a few ulps;
        # tolerate only that numerical roundoff, while rejecting real range
        # corruption.
        if name == "zncc" and np.any((values[accepted_finite] < -1.0 - 1e-9) | (values[accepted_finite] > 1.0 + 1e-9)):
            raise BenchmarkGateError("ZNCC is outside [-1, 1]")
        if name == "residual_rms" and np.any(values[accepted_finite] < 0.0):
            raise BenchmarkGateError("residual RMS is negative")
    return diagnostics, n


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _quality_score_at(diagnostics: Mapping[str, Any], index: int) -> float | None:
    import numpy as np

    if QUALITY_SCORE_VALIDITY_REQUIRED and not bool(np.asarray(diagnostics["valid"]).reshape(-1)[index]):
        return 0.0
    def numeric(name: str) -> float | None:
        try:
            value = float(np.asarray(diagnostics[name]).reshape(-1)[index])
        except (TypeError, ValueError, IndexError):
            return None
        return value if math.isfinite(value) else None

    values = {name: numeric(name) for name in ("zncc", "second_peak_margin", "second_peak_ratio_best_over_second", "residual_rms", "hessian_condition_number", "iterations")}
    if any(value is None for value in values.values()):
        return None
    converged = bool(np.asarray(diagnostics["converged"]).reshape(-1)[index])
    normalization = QUALITY_SCORE_NORMALIZATION
    components = {
        "zncc": _clamp01((values["zncc"] - normalization["zncc_floor"]) / normalization["zncc_span"]),
        "second_peak_margin": _clamp01(values["second_peak_margin"] / normalization["second_peak_margin_span"]),
        "second_peak_ratio_best_over_second": _clamp01((values["second_peak_ratio_best_over_second"] - normalization["second_peak_ratio_floor"]) / normalization["second_peak_ratio_span"]),
        "residual_rms": math.exp(-max(0.0, values["residual_rms"]) / normalization["residual_rms_scale"]),
        "hessian_condition_number": _clamp01((normalization["hessian_condition_ceiling"] - values["hessian_condition_number"]) / normalization["hessian_condition_ceiling"]),
        "iterations": _clamp01(1.0 - max(0.0, values["iterations"] - normalization["iterations_free_ceiling"]) / normalization["iterations_span"]),
        "converged": 1.0 if converged else 0.0,
    }
    score = sum(float(QUALITY_SCORE_COMPONENTS[name]) * components[name] for name in QUALITY_SCORE_COMPONENTS)
    return _clamp01(float(score))


def _roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    import numpy as np

    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=bool)
    finite = np.isfinite(score_array)
    score_array, label_array = score_array[finite], label_array[finite]
    positive = int(np.count_nonzero(label_array))
    negative = int(label_array.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(sorted_scores.size, dtype=float)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    return float((np.sum(original_ranks[label_array]) - positive * (positive + 1) / 2.0) / (positive * negative))


def _preflight(core: Any, image: Any) -> dict[str, Any]:
    require_texture = getattr(core, "require_texture", None)
    if not callable(require_texture):
        return {"ok": False, "code": "TEXTURE_PREFLIGHT_UNAVAILABLE", "metrics": None, "details": {}, "thresholds": dict(TEXTURE_PREFLIGHT)}
    kwargs = {
        "min_std": TEXTURE_PREFLIGHT["min_std"],
        "min_contrast": TEXTURE_PREFLIGHT["min_contrast"],
        "max_saturated_frac": TEXTURE_PREFLIGHT["max_saturated_frac"],
        "min_structure_ratio": TEXTURE_PREFLIGHT["min_structure_ratio"],
        "max_directional_coherence": TEXTURE_PREFLIGHT["max_directional_coherence"],
        "min_periodicity_score": TEXTURE_PREFLIGHT["min_periodicity_score"],
    }
    threshold_record = {"version": TEXTURE_PREFLIGHT["version"], **kwargs}
    try:
        metrics = require_texture(image, ROI, **kwargs)
    except Exception as exc:
        code = getattr(exc, "code", None)
        details = getattr(exc, "details", {})
        if not isinstance(code, str):
            code = "TEXTURE_PREFLIGHT_ERROR"
        if not isinstance(details, Mapping):
            details = {}
        return {"ok": False, "code": code, "metrics": details.get("metrics"), "details": dict(details), "thresholds": threshold_record}
    if not isinstance(metrics, Mapping):
        return {"ok": False, "code": "TEXTURE_PREFLIGHT_INVALID", "metrics": None, "details": {}, "thresholds": threshold_record}
    return {"ok": True, "code": None, "metrics": dict(metrics), "details": {}, "thresholds": threshold_record}


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _case_hash(case: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(case).encode("utf-8")).hexdigest()


def _oracle_strain(case: Mapping[str, Any]) -> tuple[float, float, float]:
    import numpy as np

    if case["kind"] != "affine":
        return 0.0, 0.0, 0.0
    F = np.asarray(case["F"], dtype=float).reshape(2, 2)
    E = 0.5 * (F.T @ F - np.eye(2))
    return float(E[0, 0]), float(E[1, 1]), float(E[0, 1])


def _fit_affine(raw_u: Any, raw_v: Any, valid: Any, coordinates: Any) -> dict[str, Any]:
    import numpy as np

    finite = np.asarray(valid, dtype=bool) & np.isfinite(raw_u) & np.isfinite(raw_v)
    design = np.column_stack((coordinates[finite, 0], coordinates[finite, 1], np.ones(int(np.count_nonzero(finite)))))
    if design.shape[0] < 3 or np.linalg.matrix_rank(design) < 3:
        raise BenchmarkGateError("affine oracle fit is rank deficient")
    u_coeff, *_ = np.linalg.lstsq(design, np.asarray(raw_u)[finite], rcond=None)
    v_coeff, *_ = np.linalg.lstsq(design, np.asarray(raw_v)[finite], rcond=None)
    F = np.asarray([[1.0 + u_coeff[0], u_coeff[1]], [v_coeff[0], 1.0 + v_coeff[1]]], dtype=float)
    E = 0.5 * (F.T @ F - np.eye(2))
    return {"F": F, "E": E, "fit_count": int(np.count_nonzero(finite)), "fit_rank": int(np.linalg.matrix_rank(design))}


def _observation(
    case: Mapping[str, Any],
    fixture: Mapping[str, Any],
    field: Any,
    *,
    run_id: str,
    panel: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any],
    quality_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    diagnostics, n = _required_field(field, str(case["case_id"]))
    oracle_u = np.asarray(fixture["oracle_u"], dtype=float)
    oracle_v = np.asarray(fixture["oracle_v"], dtype=float)
    raw_u = np.asarray(diagnostics["u_raw"], dtype=float)
    raw_v = np.asarray(diagnostics["v_raw"], dtype=float)
    valid = np.asarray(diagnostics["valid"], dtype=bool)
    strain_valid = np.asarray(diagnostics["strain_valid"], dtype=bool)
    finite_raw = np.isfinite(raw_u) & np.isfinite(raw_v)
    valid_finite = valid & finite_raw
    error_u = raw_u - oracle_u
    error_v = raw_v - oracle_v
    error_px = np.hypot(error_u, error_v)
    strict_errors = error_px[valid_finite]
    rmse = float(np.sqrt(np.mean(strict_errors**2))) if strict_errors.size else None
    p95 = float(np.percentile(strict_errors, 95)) if strict_errors.size else None
    max_error = float(np.max(strict_errors)) if strict_errors.size else None
    oracle_exx, oracle_eyy, oracle_exy = _oracle_strain(case)
    strain_arrays: dict[str, Any] = {name: None for name in ("Exx", "Eyy", "Exy")}
    strain_errors: dict[str, Any] = {name: None for name in ("Exx", "Eyy", "Exy")}
    affine_fit: dict[str, Any] | None = None
    strain_abs_max = None
    strain_consistency_abs_max = None
    metadata_ok = True
    if case["kind"] == "affine":
        for name in strain_arrays:
            strain_arrays[name] = _as_array(field, (name,), n)
        affine_fit = _fit_affine(raw_u, raw_v, valid, np.asarray(LOCKED_COORDINATES, dtype=float))
        strain_stack = np.column_stack(
            (
                np.asarray(strain_arrays["Exx"], dtype=float) - oracle_exx,
                np.asarray(strain_arrays["Eyy"], dtype=float) - oracle_eyy,
                np.asarray(strain_arrays["Exy"], dtype=float) - oracle_exy,
            )
        )
        strain_ok = strain_valid & np.isfinite(strain_stack).all(axis=1)
        if np.any(strain_ok):
            strain_abs_max = float(np.max(np.abs(strain_stack[strain_ok])))
            estimated_E = np.asarray((affine_fit["E"][0, 0], affine_fit["E"][1, 1], affine_fit["E"][0, 1]), dtype=float)
            supplied = np.column_stack((strain_arrays["Exx"], strain_arrays["Eyy"], strain_arrays["Exy"]))
            strain_consistency_abs_max = float(np.max(np.abs(supplied[strain_ok] - estimated_E)))
        for index, name in enumerate(("Exx", "Eyy", "Exy")):
            strain_errors[name] = strain_stack[:, index]
        strain_type = str(field.get("strain_type", "")).lower().replace("_", "-")
        strain_convention = str(field.get("strain_convention", "")).lower()
        metadata_ok = (
            "green" in strain_type
            and "lagrange" in strain_type
            and "tensor" in strain_convention
            and "exy" in strain_convention
            and "engineering" not in strain_convention
            and "gamma" not in strain_convention
        )
    quality_scores: list[float | None] = []
    quality_min = float(quality_contract["illustrative_quality_threshold"]["quality_accept_score_min"])
    target_indices = set(int(value) for value in (panel or {}).get("target_point_indices", []))
    rows: list[dict[str, Any]] = []
    for index, coordinate in enumerate(np.asarray(LOCKED_COORDINATES, dtype=float)):
        quality = _quality_score_at(diagnostics, index)
        quality_scores.append(quality)
        quality_accept = None if quality is None else bool(quality >= quality_min)
        accepted = bool(valid[index])
        # A rejected/nonfinite measurement is a bad outcome for ranking even
        # when no finite displacement error can be formed.  This keeps AUC
        # population accounting honest while preserving raw diagnostics.
        label_good = bool(accepted and np.isfinite(error_px[index]) and error_px[index] <= ERROR_TOLERANCE_PX)
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "run_id": run_id,
                "panel_variant_id": None if panel is None else str(panel["variant_id"]),
                "point_index": index,
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "oracle_u": float(oracle_u[index]),
                "oracle_v": float(oracle_v[index]),
                "u_raw": _finite_or_none(raw_u[index]),
                "v_raw": _finite_or_none(raw_v[index]),
                "error_u_px": _finite_or_none(error_u[index]),
                "error_v_px": _finite_or_none(error_v[index]),
                "error_px": _finite_or_none(error_px[index]),
                "oracle_Exx": oracle_exx,
                "oracle_Eyy": oracle_eyy,
                "oracle_Exy": oracle_exy,
                "Exx": None if strain_arrays["Exx"] is None else _finite_or_none(strain_arrays["Exx"][index]),
                "Eyy": None if strain_arrays["Eyy"] is None else _finite_or_none(strain_arrays["Eyy"][index]),
                "Exy": None if strain_arrays["Exy"] is None else _finite_or_none(strain_arrays["Exy"][index]),
                "strain_error_Exx": None if strain_errors["Exx"] is None else _finite_or_none(strain_errors["Exx"][index]),
                "strain_error_Eyy": None if strain_errors["Eyy"] is None else _finite_or_none(strain_errors["Eyy"][index]),
                "strain_error_Exy": None if strain_errors["Exy"] is None else _finite_or_none(strain_errors["Exy"][index]),
                "valid": accepted,
                "strain_valid": bool(strain_valid[index]),
                "accepted": accepted,
                "quality_accept": quality_accept,
                "quality_label_good": label_good,
                "false_accept": bool(accepted and label_good is False),
                "false_reject": bool((not accepted) and label_good is True),
                "quality_false_accept": bool(quality_accept is True and label_good is False),
                "quality_false_reject": bool(quality_accept is False and label_good is True),
                "quality_score": quality,
                "quality_score_version": QUALITY_SCORE_VERSION,
                "corruption_applied": index in target_indices,
                "diagnostic_source": "solver",
                "ratio_direction": diagnostics["ratio_direction"],
                "zncc": _finite_or_none(diagnostics["zncc"][index]),
                "second_peak_margin": _finite_or_none(diagnostics["second_peak_margin"][index]),
                "second_peak_ratio_best_over_second": _finite_or_none(diagnostics["second_peak_ratio_best_over_second"][index]),
                "second_to_best_peak_ratio": _finite_or_none(1.0 / diagnostics["second_peak_ratio_best_over_second"][index]) if _finite_or_none(diagnostics["second_peak_ratio_best_over_second"][index]) not in (None, 0.0) else None,
                "best_peak": _finite_or_none(diagnostics["best_peak"][index]),
                "second_peak": _finite_or_none(diagnostics["second_peak"][index]),
                "peak_margin": _finite_or_none(diagnostics["peak_margin"][index]),
                "peak_ratio": _finite_or_none(diagnostics["peak_ratio"][index]),
                "residual_rms": _finite_or_none(diagnostics["residual_rms"][index]),
                "hessian_condition_number": _finite_or_none(diagnostics["hessian_condition_number"][index]),
                "iterations": _finite_or_none(diagnostics["iterations"][index]),
                "converged": bool(diagnostics["converged"][index]),
                "stop_reason": str(diagnostics["stop_reason"][index]),
                "invalid_reason": str(diagnostics["invalid_reason"][index]),
                "preflight_code": None,
            }
        )
    gates = {"diagnostics_complete": True}
    if panel is None:
        gates.update(
            {
                "valid_fraction": float(np.count_nonzero(valid) / n) >= float(thresholds["valid_fraction_min"]),
                "rmse_px": rmse is not None and rmse <= float(thresholds["rmse_px_max"]),
                "p95_error_px": p95 is not None and p95 <= float(thresholds["p95_error_px_max"]),
                "max_error_px": max_error is not None and max_error <= float(thresholds["max_error_px_max"]),
            }
        )
        if case["kind"] == "translation" and int(case["pyramid_levels"]) > 1:
            try:
                used = int(field.get("pyramid_levels_used", 0))
            except (TypeError, ValueError):
                used = 0
            gates["pyramid_levels"] = used >= int(case["pyramid_levels"])
        if case["kind"] == "affine":
            gates.update(
                {
                    "affine_fit": affine_fit is not None,
                    "affine_metadata": metadata_ok,
                    "strain_component_abs_error": strain_abs_max is not None and strain_abs_max <= float(thresholds["strain_component_abs_error_max"]),
                    "strain_consistency_abs_error": strain_consistency_abs_max is not None and strain_consistency_abs_max <= float(thresholds["strain_consistency_abs_error_max"]),
                    "strain_valid_fraction": float(np.count_nonzero(strain_valid) / n) >= float(thresholds["strain_valid_fraction_min"]),
                }
            )
    observed = {
        "valid_fraction": float(np.count_nonzero(valid) / n),
        "strain_valid_fraction": float(np.count_nonzero(strain_valid) / n),
        "raw_finite_count": int(np.count_nonzero(finite_raw)),
        "rejected_count": int(np.count_nonzero(~valid)),
        "rmse_px": rmse,
        "p95_error_px": p95,
        "max_error_px": max_error,
        "false_accept_count": int(sum(bool(row["false_accept"]) for row in rows)),
        "false_reject_count": int(sum(bool(row["false_reject"]) for row in rows)),
        "quality_false_accept_count": int(sum(bool(row["quality_false_accept"]) for row in rows)),
        "quality_false_reject_count": int(sum(bool(row["quality_false_reject"]) for row in rows)),
    }
    if case["kind"] == "affine":
        observed.update(
            {
                "estimated_F": None if affine_fit is None else affine_fit["F"].tolist(),
                "oracle_F": case["F"],
                "estimated_E": None if affine_fit is None else [float(affine_fit["E"][0, 0]), float(affine_fit["E"][1, 1]), float(affine_fit["E"][0, 1])],
                "oracle_E": [oracle_exx, oracle_eyy, oracle_exy],
                "fit_count": None if affine_fit is None else affine_fit["fit_count"],
                "fit_rank": None if affine_fit is None else affine_fit["fit_rank"],
                "strain_component_abs_error_max": strain_abs_max,
                "strain_consistency_abs_error_max": strain_consistency_abs_max,
            }
        )
    run_status = "PASS" if panel is None and all(gates.values()) else "PANEL_OBSERVATION" if panel is not None and all(gates.values()) else "FAIL"
    return {
        "run_id": run_id,
        "panel": None if panel is None else dict(panel),
        "status": run_status,
        "strict_gate_evaluated": panel is None,
        "observed": observed,
        "gates": gates,
        "diagnostic_keys": sorted(_DIAGNOSTIC_ALIASES),
        "point_count": len(rows),
    }, rows


def _near_record(
    case: Mapping[str, Any],
    fixture: Mapping[str, Any],
    preflight: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    expected = str(thresholds["expected_failure_code"])
    code = preflight.get("code")
    passed = code == expected and int(thresholds["max_successful_export_artifacts"]) == 0
    oracle_u = np.asarray(fixture["oracle_u"], dtype=float)
    oracle_v = np.asarray(fixture["oracle_v"], dtype=float)
    rows = []
    for index, coordinate in enumerate(np.asarray(LOCKED_COORDINATES, dtype=float)):
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "run_id": "preflight",
                "panel_variant_id": None,
                "point_index": index,
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "oracle_u": float(oracle_u[index]),
                "oracle_v": float(oracle_v[index]),
                "u_raw": None,
                "v_raw": None,
                "error_u_px": None,
                "error_v_px": None,
                "error_px": None,
                "oracle_Exx": 0.0,
                "oracle_Eyy": 0.0,
                "oracle_Exy": 0.0,
                "Exx": None,
                "Eyy": None,
                "Exy": None,
                "strain_error_Exx": None,
                "strain_error_Eyy": None,
                "strain_error_Exy": None,
                "valid": False,
                "strain_valid": False,
                "accepted": False,
                "quality_accept": None,
                "quality_label_good": None,
                "false_accept": False,
                "false_reject": False,
                "quality_false_accept": False,
                "quality_false_reject": False,
                "quality_score": None,
                "quality_score_version": QUALITY_SCORE_VERSION,
                "corruption_applied": False,
                "diagnostic_source": "preflight_rejected",
                "ratio_direction": None,
                "zncc": None,
                "second_peak_margin": None,
                "second_peak_ratio_best_over_second": None,
                "second_to_best_peak_ratio": None,
                "best_peak": None,
                "second_peak": None,
                "peak_margin": None,
                "peak_ratio": None,
                "residual_rms": None,
                "hessian_condition_number": None,
                "iterations": None,
                "converged": None,
                "stop_reason": "preflight_ambiguity",
                "invalid_reason": str(code or "TEXTURE_PREFLIGHT_ERROR"),
                "preflight_code": str(code or "TEXTURE_PREFLIGHT_ERROR"),
            }
        )
    return {
        "run_id": "preflight",
        "status": "REJECTED" if passed else "FAIL",
        "benchmark_pass": passed,
        "scientific_ok": False,
        "failure_code": str(code or "TEXTURE_PREFLIGHT_ERROR"),
        "outcome": str(code or "TEXTURE_PREFLIGHT_ERROR"),
        "solver_calls": 0,
        "successful_export_artifacts": 0,
        "texture_preflight": dict(preflight),
        "gates": {"ambiguous_texture_rejected": code == expected, "zero_successful_export_artifacts": True},
        "point_count": len(rows),
    }, rows


def _failure_case(case: Mapping[str, Any], message: str, *, case_hash: str) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        "kind": str(case["kind"]),
        "seed": int(case["seed"]),
        "case_hash": case_hash,
        "config_hash": case_hash,
        "status": "FAIL",
        "benchmark_pass": False,
        "scientific_ok": False,
        "failure_code": "BENCHMARK_GATE_FAILED",
        "failure_message": message,
        "gates": {"case_execution": False},
        "points": [],
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return format(value, ".17g") if math.isfinite(value) else ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS), extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_value(row.get(field)) for field in _CSV_FIELDS})
    except OSError as exc:
        raise BenchmarkIOError(f"cannot write benchmark CSV: {path}: {exc}") from exc


def run_benchmark(
    cases_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    core_module: Any | None = None,
) -> dict[str, Any]:
    """Run all locked cases, preserving both clean and corrupted observations."""

    # Historical callers passed only an output directory.  Keep that harmless
    # positional form while retaining the explicit ``cases_path, output_dir``
    # package API used by the release command.
    if output_dir is None and cases_path is not None:
        candidate = Path(cases_path)
        if candidate.is_dir() or candidate.suffix.casefold() != ".json":
            output_dir, cases_path = candidate, None
    try:
        selected_cases_path = Path(cases_path) if cases_path is not None else default_cases_path()
        document = load_locked_cases(selected_cases_path)
    except ValueError as exc:
        raise BenchmarkConfigError(str(exc)) from exc
    if output_dir is None:
        output = Path(tempfile.mkdtemp(prefix="ezdic-benchmark-"))
    else:
        output = Path(output_dir)
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BenchmarkIOError(f"cannot create benchmark output directory: {output}: {exc}") from exc
    if not output.is_dir():
        raise BenchmarkIOError(f"benchmark output path is not a directory: {output}")
    try:
        core = core_module or importlib.import_module("ezdic_core")
    except Exception as exc:
        raise BenchmarkError(f"cannot import ezdic_core: {exc}") from exc
    expected_ids = ["small_translation", "large_translation", "small_affine_strain", "near_1d_periodic"]
    if [str(case["case_id"]) for case in document["cases"]] != expected_ids:
        raise BenchmarkConfigError("locked benchmark case set is incomplete or reordered")
    thresholds = document["thresholds"]
    quality_contract = document["quality_contract"]
    panel_doc = quality_contract["corruption_panel"]
    records: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    expected_row_count = len(document["cases"]) * len(LOCKED_COORDINATES)
    expected_row_count += sum(
        len(LOCKED_COORDINATES)
        for case in document["cases"]
        for variant in panel_doc["variants"]
        if str(case["case_id"]) in variant["case_ids"]
    )
    for case in document["cases"]:
        case_id = str(case["case_id"])
        case_hash = _case_hash(case)
        try:
            fixture = make_case(case, core_module=core)
            preflight = _preflight(core, fixture["reference"])
            case_thresholds = thresholds[case_id]
            if case["kind"] == "near_1d_periodic":
                record, rows = _near_record(case, fixture, preflight, case_thresholds)
                record.update({"case_id": case_id, "kind": case["kind"], "seed": int(case["seed"]), "case_hash": case_hash, "config_hash": case_hash, "thresholds": dict(case_thresholds), "input_sha256": fixture["input_sha256"]})
                records.append(record)
                all_rows.extend(rows)
                continue
            if not preflight.get("ok"):
                record = _failure_case(case, f"reference texture preflight failed: {preflight.get('code')}", case_hash=case_hash)
                record.update({"texture_preflight": dict(preflight), "thresholds": dict(case_thresholds), "input_sha256": fixture["input_sha256"]})
                records.append(record)
                continue
            run_records: list[dict[str, Any]] = []
            baseline_field = core.run_2d_dic(
                fixture["reference"],
                fixture["deformed"],
                ROI,
                subset_size=SUBSET_SIZE,
                step=STEP,
                search_radius=int(case["search_radius"]),
                pyramid_levels=int(case["pyramid_levels"]),
                zncc_min=float(SOLVER_SETTINGS["zncc_min"]),
                strain_window=int(SOLVER_SETTINGS["strain_window"]),
                smooth_sigma=float(SOLVER_SETTINGS["smooth_sigma"]),
            )
            baseline_observation, baseline_rows = _observation(
                case,
                fixture,
                baseline_field,
                run_id="clean",
                panel=None,
                thresholds=case_thresholds,
                quality_contract=quality_contract,
            )
            run_records.append(baseline_observation)
            all_rows.extend(baseline_rows)
            for variant in panel_doc["variants"]:
                if case_id not in variant["case_ids"]:
                    continue
                corrupted_fixture, panel = apply_image_corruption(fixture, case, variant)
                field = core.run_2d_dic(
                    corrupted_fixture["reference"],
                    corrupted_fixture["deformed"],
                    ROI,
                    subset_size=SUBSET_SIZE,
                    step=STEP,
                    search_radius=int(case["search_radius"]),
                    pyramid_levels=int(case["pyramid_levels"]),
                    zncc_min=float(SOLVER_SETTINGS["zncc_min"]),
                    strain_window=int(SOLVER_SETTINGS["strain_window"]),
                    smooth_sigma=float(SOLVER_SETTINGS["smooth_sigma"]),
                )
                panel_observation, panel_rows = _observation(
                    case,
                    corrupted_fixture,
                    field,
                    run_id=str(panel["variant_id"]),
                    panel=panel,
                    thresholds=case_thresholds,
                    quality_contract=quality_contract,
                )
                run_records.append(panel_observation)
                all_rows.extend(panel_rows)
            baseline_pass = bool(baseline_observation.get("status") == "PASS" and all(baseline_observation["gates"].values()))
            panel_rows_for_case = [row for row in all_rows if row["case_id"] == case_id and row["corruption_applied"]]
            expected_panel_variants = [variant for variant in panel_doc["variants"] if case_id in variant["case_ids"]]
            panel_gate = len(panel_rows_for_case) == sum(len(variant["target_point_indices"]) for variant in expected_panel_variants)
            record = {
                "case_id": case_id,
                "kind": case["kind"],
                "seed": int(case["seed"]),
                "case_hash": case_hash,
                "config_hash": case_hash,
                "input_sha256": fixture["input_sha256"],
                "thresholds": dict(case_thresholds),
                "status": "PASS" if baseline_pass and panel_gate else "FAIL",
                "benchmark_pass": baseline_pass and panel_gate,
                "scientific_ok": baseline_pass,
                "failure_code": None if baseline_pass and panel_gate else "SCIENTIFIC_GATE_FAILED",
                "texture_preflight": dict(preflight),
                "metrics": dict(baseline_observation["observed"]),
                "gates": {**dict(baseline_observation["gates"]), "corruption_rows": panel_gate},
                "runs": run_records,
                "corruption_panel_version": CORRUPTION_PANEL_VERSION,
            }
            records.append(record)
        except BenchmarkError as exc:
            records.append(_failure_case(case, str(exc), case_hash=case_hash))
        except Exception as exc:
            records.append(_failure_case(case, f"benchmark execution failed: {exc}", case_hash=case_hash))

    # Keep every numeric solver outcome in the ranking population.  A rejected
    # or nonfinite measurement is explicitly a bad outcome even when no finite
    # error can be calculated; only the preflight-only near-1D case is excluded.
    quality_rows = [
        row
        for row in all_rows
        if row.get("case_id") != "near_1d_periodic" and row.get("quality_score") is not None
    ]
    scores = [float(row["quality_score"]) for row in quality_rows]
    labels = [bool(row["quality_label_good"]) for row in quality_rows]
    auc = _roc_auc(scores, labels)
    good_count = int(sum(labels))
    bad_count = int(len(labels) - good_count)
    finite_label_rows = [row for row in quality_rows if row.get("error_px") is not None]
    finite_good_count = int(sum(bool(row["quality_label_good"]) for row in finite_label_rows))
    finite_bad_count = int(len(finite_label_rows) - finite_good_count)
    false_accept = int(sum(bool(row["quality_false_accept"]) for row in finite_label_rows))
    false_reject = int(sum(bool(row["quality_false_reject"]) for row in finite_label_rows))
    ranking_false_accept = int(sum(bool(row["quality_false_accept"]) for row in quality_rows))
    ranking_false_reject = int(sum(bool(row["quality_false_reject"]) for row in quality_rows))
    bad_min_gate = bad_count >= int(quality_contract["minimum_bad_label_count"])
    corruption_row_gate = sum(bool(row["corruption_applied"]) for row in all_rows) == int(quality_contract["minimum_corruption_row_count"])
    quality_ranking_pass = (
        auc is not None
        and auc >= float(quality_contract["roc_auc_min"])
        and good_count > 0
        and bad_count > 0
        and bad_min_gate
        and corruption_row_gate
    )
    finite_false_accept_rate = None if finite_bad_count == 0 else false_accept / finite_bad_count
    finite_false_reject_rate = None if finite_good_count == 0 else false_reject / finite_good_count
    ranking_false_accept_rate = None if bad_count == 0 else ranking_false_accept / bad_count
    ranking_false_reject_rate = None if good_count == 0 else ranking_false_reject / good_count
    quality_error = {
        "version": QUALITY_SCORE_VERSION,
        "threshold_status": "NOT_CALIBRATED",
        "quality_threshold_evaluated": False,
        "quality_threshold_pass": None,
        "error_tolerance_px": float(quality_contract["error_tolerance_px"]),
        "illustrative_quality_accept_score_min": float(quality_contract["illustrative_quality_threshold"]["quality_accept_score_min"]),
        "point_count": len(finite_label_rows),
        "finite_error_label_count": len(finite_label_rows),
        "ranking_point_count": len(quality_rows),
        "ranking_label_basis": "numeric solver rows; accepted finite error <= tolerance is good, rejected or nonfinite is bad; near-1D preflight excluded",
        "all_row_count": len(all_rows),
        "rejected_point_count": int(sum(not bool(row["accepted"]) for row in all_rows)),
        "good_label_count": finite_good_count,
        "bad_label_count": finite_bad_count,
        "ranking_good_label_count": good_count,
        "ranking_bad_label_count": bad_count,
        "ranking_rejected_bad_count": int(sum(not bool(row["accepted"]) for row in quality_rows)),
        "minimum_bad_label_count": int(quality_contract["minimum_bad_label_count"]),
        "corruption_row_count": int(sum(bool(row["corruption_applied"]) for row in all_rows)),
        "minimum_corruption_row_count": int(quality_contract["minimum_corruption_row_count"]),
        "roc_auc": auc,
        "roc_auc_min": float(quality_contract["roc_auc_min"]),
        "roc_auc_gate": bool(quality_ranking_pass),
        "false_accept_count": false_accept,
        "false_reject_count": false_reject,
        "false_accept_rate": finite_false_accept_rate,
        "false_reject_rate": finite_false_reject_rate,
        "ranking_false_accept_count": ranking_false_accept,
        "ranking_false_reject_count": ranking_false_reject,
        "ranking_false_accept_rate": ranking_false_accept_rate,
        "ranking_false_reject_rate": ranking_false_reject_rate,
        "class_conditional_rates": {
            "finite_error_labels": {
                "false_accept_count": false_accept,
                "bad_label_count": finite_bad_count,
                "false_accept_rate": finite_false_accept_rate,
                "false_reject_count": false_reject,
                "good_label_count": finite_good_count,
                "false_reject_rate": finite_false_reject_rate,
            },
            "ranking_outcomes": {
                "false_accept_count": ranking_false_accept,
                "bad_label_count": bad_count,
                "false_accept_rate": ranking_false_accept_rate,
                "false_reject_count": ranking_false_reject,
                "good_label_count": good_count,
                "false_reject_rate": ranking_false_reject_rate,
            },
        },
    }
    numeric_baseline_pass = len(records) == 4 and all(
        bool(record.get("benchmark_pass"))
        for record in records
        if record.get("case_id") != "near_1d_periodic"
    )
    near_1d_preflight_pass = any(
        record.get("case_id") == "near_1d_periodic" and bool(record.get("benchmark_pass"))
        for record in records
    )
    provenance = _code_provenance(core, selected_cases_path)
    provenance_keys = (
        "synthetic_cases_source_sha256",
        "cases_json_sha256",
        "benchmark_runner_source_sha256",
        "benchmark_source_sha256",
        "core_source_sha256",
        "cli_source_sha256",
    )
    provenance_gate = all(isinstance(provenance.get(key), str) and len(str(provenance[key])) == 64 for key in provenance_keys)
    expected_rows_gate = len(all_rows) == expected_row_count
    csv_path = output / "benchmark_report.csv"
    _write_csv(csv_path, all_rows)
    try:
        csv_exists_gate = csv_path.is_file() and csv_path.stat().st_size > 0
    except OSError:
        csv_exists_gate = False
    csv_hash = _sha256_path(csv_path) if csv_exists_gate else "unavailable"
    overall_pass = bool(
        numeric_baseline_pass
        and near_1d_preflight_pass
        and quality_ranking_pass
        and provenance_gate
        and expected_rows_gate
        and csv_exists_gate
    )
    gate_summary = {
        "numeric_baseline_pass": numeric_baseline_pass,
        "near_1d_preflight_pass": near_1d_preflight_pass,
        "quality_ranking_pass": quality_ranking_pass,
        "quality_threshold_evaluated": False,
        "quality_threshold_pass": None,
        "code_provenance": provenance_gate,
        "csv_rows": expected_rows_gate,
        "csv_exists": csv_exists_gate,
    }
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "cases_version": CASE_DOCUMENT_VERSION,
        "migration": {
            "previous_report_version": "ezdic-benchmark-report-v5",
            "previous_cases_version": "ezdic-benchmark-cases-v3",
            "reason": "independent continuous image oracle; stricter numerical accuracy gates; per-run CSV integrity rather than a platform-dependent CSV byte baseline",
        },
        "locked_cases_hash": locked_case_hash(),
        "app": {"name": "ezDIC", "benchmark": "quality_to_known_error", "version": "v0.2.0-dev"},
        "code": provenance,
        "environment": {"python": platform.python_version(), "platform": platform.system(), "image_shape": list(IMAGE_SHAPE)},
        "contract": document["contract"],
        "texture_preflight_contract": TEXTURE_PREFLIGHT,
        "quality_contract": quality_contract,
        "quality_error": quality_error,
        "quality_score": {"version": QUALITY_SCORE_VERSION, "components": QUALITY_SCORE_COMPONENTS, "normalization": QUALITY_SCORE_NORMALIZATION, "validity_required": QUALITY_SCORE_VALIDITY_REQUIRED, "ratio_direction": "best_over_second", "threshold_status": "NOT_CALIBRATED", "quality_threshold_evaluated": False, "quality_threshold_pass": None, "illustrative_accept_score_min": float(quality_contract["illustrative_quality_threshold"]["quality_accept_score_min"]), "error_tolerance_px": float(quality_contract["error_tolerance_px"]), "roc_auc": auc, "roc_auc_min": float(quality_contract["roc_auc_min"])},
        "quality_auc": auc,
        "thresholds": document["thresholds"],
        "cases": [{key: value for key, value in record.items() if key != "points"} for record in records],
        "artifacts": {"benchmark_report_csv": "benchmark_report.csv", "benchmark_report_csv_sha256": csv_hash},
        "gate_summary": gate_summary,
        "gates": dict(gate_summary),
        "overall_pass": overall_pass,
        "exit_code": EXIT_SUCCESS if overall_pass else EXIT_GATE_ERROR,
    }
    if not provenance_gate:
        report["provenance_failure"] = "CODE_PROVENANCE_UNAVAILABLE"
    try:
        (output / "benchmark_report.json").write_text(canonical_json(_json_safe(report)) + "\n", encoding="utf-8")
    except OSError as exc:
        raise BenchmarkIOError(f"cannot write benchmark report: {output / 'benchmark_report.json'}: {exc}") from exc
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the locked ezDIC quality-to-known-error benchmark")
    parser.add_argument("--cases", type=Path, default=None, help="locked cases JSON")
    parser.add_argument("--output", type=Path, default=None, help="output directory for benchmark JSON/CSV")
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.cases, args.output)
    except BenchmarkError as exc:
        sys.stderr.write(canonical_json({"error_code": exc.code, "message": str(exc), "exit_code": exc.exit_code}) + "\n")
        return int(exc.exit_code)
    sys.stdout.write(canonical_json(report) + "\n")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkConfigError",
    "BenchmarkError",
    "BenchmarkGateError",
    "BenchmarkIOError",
    "BenchmarkResultError",
    "REPORT_VERSION",
    "_canonical_peak_ratio",
    "_code_provenance",
    "_ambiguity_result_passes",
    "_machine_ambiguity_code",
    "_roc_auc",
    "main",
    "run_benchmark",
]
