"""Versioned synthetic benchmark definitions and deterministic image fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


CASE_DOCUMENT_VERSION = "ezdic-benchmark-cases-v4"
QUALITY_SCORE_VERSION = "quality_score_v1"
CORRUPTION_PANEL_VERSION = "image_corruption_panel_v1"
ERROR_TOLERANCE_PX = 0.25
QUALITY_SCORE_VALIDITY_REQUIRED = True

IMAGE_SHAPE = (192, 192)
ROI = (40, 60, 88, 88)
SUBSET_SIZE = 21
STEP = 8
STRAIN_WINDOW = 5
SOLVER_SETTINGS = {
    "subset_size": SUBSET_SIZE,
    "step": STEP,
    "zncc_min": 0.75,
    "strain_window": STRAIN_WINDOW,
    "smooth_sigma": 0.0,
    "initialization": "local",
}
TEXTURE_PREFLIGHT = {
    "version": "texture_preflight_v2",
    "min_std": 8.0,
    "min_contrast": 25.0,
    "max_saturated_frac": 0.20,
    "min_structure_ratio": 0.02,
    "max_directional_coherence": 0.85,
    "min_periodicity_score": 0.90,
}

LOCKED_X = tuple(range(ROI[0] + SUBSET_SIZE // 2, ROI[0] + ROI[2] - SUBSET_SIZE // 2, STEP))
LOCKED_Y = tuple(range(ROI[1] + SUBSET_SIZE // 2, ROI[1] + ROI[3] - SUBSET_SIZE // 2, STEP))
LOCKED_COORDINATES = tuple((float(x), float(y)) for y in LOCKED_Y for x in LOCKED_X)

# These constants are duplicated in the immutable document below on purpose:
# a modified JSON file cannot silently redefine the actual gate or score.
QUALITY_SCORE_COMPONENTS = {
    "zncc": 0.25,
    "second_peak_margin": 0.15,
    "second_peak_ratio_best_over_second": 0.15,
    "residual_rms": 0.15,
    "hessian_condition_number": 0.15,
    "iterations": 0.05,
    "converged": 0.10,
}
QUALITY_SCORE_NORMALIZATION = {
    "zncc_floor": 0.50,
    "zncc_span": 0.50,
    "second_peak_margin_span": 0.25,
    "second_peak_ratio_floor": 1.0,
    "second_peak_ratio_span": 0.75,
    "residual_rms_scale": 0.10,
    "hessian_condition_ceiling": 1000.0,
    "iterations_free_ceiling": 8.0,
    "iterations_span": 40.0,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _locked_document() -> dict[str, Any]:
    return {
        "version": CASE_DOCUMENT_VERSION,
        "migration": {
            "previous_version": "ezdic-benchmark-cases-v3",
            "reason": "independent continuous texture and inverse deformation; remove OpenCV sampling bias; tighten numeric error limits",
        },
        "contract": {
            "image_shape": list(IMAGE_SHAPE),
            "roi": list(ROI),
            "subset_size": SUBSET_SIZE,
            "step": STEP,
            "strain_window": STRAIN_WINDOW,
            "coordinate_order": "row-major y then x; exactly 81 locked POIs",
            "reference_frame": "fixed clean reference; each deformed image is compared to it",
            "solver": SOLVER_SETTINGS,
            "image_generator": "analytic_random_fourier_v1",
        },
        "quality_contract": {
            "version": QUALITY_SCORE_VERSION,
            "error_tolerance_px": ERROR_TOLERANCE_PX,
            "quality_validity_required": QUALITY_SCORE_VALIDITY_REQUIRED,
            "illustrative_quality_threshold": {
                "status": "NOT_CALIBRATED",
                "quality_accept_score_min": 0.50,
            },
            "roc_auc_min": 0.90,
            "minimum_bad_label_count": 2,
            "minimum_corruption_row_count": 4,
            "score_components": QUALITY_SCORE_COMPONENTS,
            "score_normalization": QUALITY_SCORE_NORMALIZATION,
            "texture_preflight": TEXTURE_PREFLIGHT,
            "corruption_panel": {
                "version": CORRUPTION_PANEL_VERSION,
                "apply_to": "deformed_image_only",
                "target_coordinate": "oracle_deformed_poi_center",
                "variants": [
                    {
                        "variant_id": "gaussian_noise_target_0",
                        "case_ids": ["small_translation", "large_translation"],
                        "target_point_indices": [0],
                        "radius_px": 3,
                        "sigma_gray": 160.0,
                        "seed": 1701,
                    },
                    {
                        "variant_id": "gaussian_noise_target_80",
                        "case_ids": ["small_translation", "large_translation"],
                        "target_point_indices": [80],
                        "radius_px": 4,
                        "sigma_gray": 160.0,
                        "seed": 1701,
                    },
                ],
            },
        },
        "thresholds": {
            "small_translation": {
                "valid_fraction_min": 0.95,
                "rmse_px_max": 0.001,
                "p95_error_px_max": 0.002,
                "max_error_px_max": 0.005,
            },
            "large_translation": {
                "valid_fraction_min": 0.95,
                "rmse_px_max": 0.001,
                "p95_error_px_max": 0.002,
                "max_error_px_max": 0.005,
            },
            "small_affine_strain": {
                "valid_fraction_min": 0.95,
                "rmse_px_max": 0.001,
                "p95_error_px_max": 0.002,
                "max_error_px_max": 0.005,
                "strain_component_abs_error_max": 1e-4,
                "strain_valid_fraction_min": 0.80,
                "strain_consistency_abs_error_max": 1e-4,
            },
            "near_1d_periodic": {
                "expected_failure_code": "AMBIGUOUS_TEXTURE",
                "max_successful_export_artifacts": 0,
            },
        },
        "cases": [
            {
                "case_id": "small_translation",
                "kind": "translation",
                "seed": 17,
                "translation": [2.3, -1.2],
                "pyramid_levels": 1,
                "search_radius": 8,
            },
            {
                "case_id": "large_translation",
                "kind": "translation",
                "seed": 17,
                "translation": [28.0, -18.0],
                "pyramid_levels": 3,
                "search_radius": 8,
            },
            {
                "case_id": "small_affine_strain",
                "kind": "affine",
                "seed": 23,
                "F": [[1.012, 0.007], [-0.004, 0.989]],
                "center": [95.5, 95.5],
                "pyramid_levels": 1,
                "search_radius": 8,
            },
            {
                "case_id": "near_1d_periodic",
                "kind": "near_1d_periodic",
                "seed": 29,
                "period_px": 16,
                "translation": [2.0, -1.0],
                "pyramid_levels": 1,
                "search_radius": 8,
            },
        ],
    }


LOCKED_CASE_DOCUMENT = _locked_document()


def locked_case_document() -> dict[str, Any]:
    return copy.deepcopy(LOCKED_CASE_DOCUMENT)


def locked_case_hash() -> str:
    return hashlib.sha256(canonical_json(LOCKED_CASE_DOCUMENT).encode("utf-8")).hexdigest()


def default_cases_path() -> Path:
    """Resolve cases for source and PyInstaller onedir execution."""

    candidates: list[Path] = [Path(__file__).resolve().parent / "cases_v1.json"]
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        root = Path(bundle_root).resolve()
        candidates.extend(
            (
                root / "benchmarks" / "cases_v1.json",
                root / "sources" / "benchmarks" / "cases_v1.json",
                root / "sources" / "cases_v1.json",
            )
        )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except (OSError, TypeError, ValueError):
            continue
    return candidates[0]


def load_locked_cases(path: str | Path | None = None) -> dict[str, Any]:
    selected = Path(path) if path is not None else default_cases_path()
    try:
        raw = json.loads(
            selected.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read locked benchmark cases: {selected}: {exc}") from exc
    if canonical_json(raw) != canonical_json(LOCKED_CASE_DOCUMENT):
        raise ValueError("cases_v1.json does not exactly match the migrated locked benchmark contract")
    return copy.deepcopy(LOCKED_CASE_DOCUMENT)


def _core(core_module: Any | None = None) -> Any:
    if core_module is not None:
        return core_module
    try:
        import ezdic_core as core  # noqa: PLC0415 - benchmark remains GUI/Tk-free.
    except Exception as exc:  # pragma: no cover - installation dependent.
        raise RuntimeError(f"cannot import ezdic_core: {exc}") from exc
    return core


def _oracle_for_case(case: Mapping[str, Any], coordinates: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    if case["kind"] == "translation" or case["kind"] == "near_1d_periodic":
        tx, ty = (float(value) for value in case["translation"])
        return np.full(len(coordinates), tx), np.full(len(coordinates), ty), np.zeros(3, dtype=float)
    F = np.asarray(case["F"], dtype=float).reshape(2, 2)
    center = np.asarray(case["center"], dtype=float).reshape(2)
    displacement = (coordinates - center) @ (F - np.eye(2)).T
    E = 0.5 * (F.T @ F - np.eye(2))
    return displacement[:, 0], displacement[:, 1], np.asarray((E[0, 0], E[1, 1], E[0, 1]), dtype=float)


def analytic_texture(x, y, seed=381):
    """Continuous texture evaluated independently of the DIC sampler."""
    import numpy as np
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2*np.pi, 100)
    frequencies = rng.uniform(.025, .145, 100)
    phases = rng.uniform(0, 2*np.pi, 100)
    amplitudes = rng.uniform(.4, 1., 100)
    result = np.full(np.broadcast_shapes(np.shape(x),np.shape(y)),128.)
    scale = 34/np.sqrt(np.sum(amplitudes**2)/2)
    for angle,frequency,phase,amplitude in zip(angles,frequencies,phases,amplitudes):
        result += scale*amplitude*np.cos(2*np.pi*frequency*(np.cos(angle)*x+np.sin(angle)*y)+phase)
    return np.clip(result,0,255)


def make_case(case: Mapping[str, Any], *, core_module: Any | None = None) -> dict[str, Any]:
    """Return clean deterministic images and analytic POI oracles."""

    import numpy as np

    core = _core(core_module)
    coordinates = np.asarray(LOCKED_COORDINATES, dtype=np.float64)
    case_id = str(case["case_id"])
    oracle_u, oracle_v, oracle_strain = _oracle_for_case(case, coordinates)
    if case["kind"] == "near_1d_periodic":
        _, xx = np.indices(IMAGE_SHAPE, dtype=np.float64)
        period = float(case["period_px"])
        reference = (127.5 + 100.0 * np.sin(2.0 * np.pi * xx / period)).astype(np.float32)
        tx, ty = (float(value) for value in case["translation"])
        deformed = np.asarray(core.warp_image_translation(reference, tx, ty), dtype=np.float32)
    else:
        yy, xx = np.indices(IMAGE_SHAPE, dtype=float)
        reference = analytic_texture(xx, yy, seed=int(case["seed"]))
        if case["kind"] == "translation":
            tx, ty = (float(value) for value in case["translation"])
            deformed = analytic_texture(xx-tx, yy-ty, seed=int(case["seed"]))
        elif case["kind"] == "affine":
            F = np.asarray(case["F"], dtype=float).reshape(2, 2)
            center = np.asarray(case["center"], dtype=float)
            inverse = np.linalg.inv(F)
            dx, dy = xx-center[0], yy-center[1]
            rx, ry = inverse[0,0]*dx+inverse[0,1]*dy+center[0], inverse[1,0]*dx+inverse[1,1]*dy+center[1]
            deformed = analytic_texture(rx, ry, seed=int(case["seed"]))
        else:
            raise ValueError(f"unsupported locked case kind: {case['kind']}")
    return {
        "case_id": case_id,
        "reference": reference,
        "deformed": deformed,
        "coordinates": coordinates,
        "oracle_u": np.asarray(oracle_u, dtype=np.float64),
        "oracle_v": np.asarray(oracle_v, dtype=np.float64),
        "oracle_strain": np.asarray(oracle_strain, dtype=np.float64),
        "input_sha256": hashlib.sha256(reference.tobytes() + deformed.tobytes()).hexdigest(),
    }


def apply_image_corruption(
    fixture: Mapping[str, Any],
    case: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one locked image-level corruption without touching solver output."""

    import numpy as np

    reference = np.asarray(fixture["reference"]).copy()
    deformed = np.asarray(fixture["deformed"]).copy()
    coordinates = np.asarray(fixture["coordinates"], dtype=float)
    if str(case["case_id"]) not in variant.get("case_ids", []):
        raise ValueError(f"corruption variant {variant.get('variant_id')} is not locked for {case['case_id']}")
    radius = int(variant["radius_px"])
    sigma = float(variant["sigma_gray"])
    rng = np.random.default_rng(int(variant["seed"]))
    target_indices = [int(value) for value in variant["target_point_indices"]]
    for index in target_indices:
        if index < 0 or index >= len(coordinates):
            raise ValueError(f"corruption target point is outside the locked grid: {index}")
        center = np.asarray(
            (
                coordinates[index, 0] + fixture["oracle_u"][index],
                coordinates[index, 1] + fixture["oracle_v"][index],
            ),
            dtype=float,
        )
        cx, cy = (int(round(value)) for value in center)
        y0, y1 = max(0, cy - radius), min(deformed.shape[0], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(deformed.shape[1], cx + radius + 1)
        deformed[y0:y1, x0:x1] += rng.normal(0.0, sigma, size=(y1 - y0, x1 - x0)).astype(np.float32)
    corrupted = dict(fixture)
    corrupted["reference"] = reference
    corrupted["deformed"] = deformed
    corrupted["input_sha256"] = hashlib.sha256(reference.tobytes() + deformed.tobytes()).hexdigest()
    panel = {
        "panel_version": CORRUPTION_PANEL_VERSION,
        "variant_id": str(variant["variant_id"]),
        "target_point_indices": target_indices,
        "radius_px": radius,
        "sigma_gray": sigma,
        "seed": int(variant["seed"]),
        "applied_to": "deformed_image_only",
    }
    return corrupted, panel


__all__ = [
    "CASE_DOCUMENT_VERSION",
    "CORRUPTION_PANEL_VERSION",
    "ERROR_TOLERANCE_PX",
    "IMAGE_SHAPE",
    "LOCKED_CASE_DOCUMENT",
    "LOCKED_COORDINATES",
    "QUALITY_SCORE_COMPONENTS",
    "QUALITY_SCORE_NORMALIZATION",
    "QUALITY_SCORE_VERSION",
    "QUALITY_SCORE_VALIDITY_REQUIRED",
    "ROI",
    "STEP",
    "STRAIN_WINDOW",
    "SOLVER_SETTINGS",
    "SUBSET_SIZE",
    "TEXTURE_PREFLIGHT",
    "apply_image_corruption",
    "canonical_json",
    "default_cases_path",
    "load_locked_cases",
    "locked_case_document",
    "locked_case_hash",
    "make_case",
]
