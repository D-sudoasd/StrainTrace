"""Known-displacement tests of global initialization and localized tracking.

Run with ``python -m benchmarks.necking_search_validation --output report.json``.
The continuous input texture and inverse image map are generated independently
of the DIC interpolator. The local case is an in-plane displacement band; it
does not reproduce all optical or mechanical features of a tensile neck.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import scipy

# Support both direct execution from any working directory and ``python -m``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ezdic_core as core
from benchmarks.synthetic_cases import analytic_texture


IMAGE_SIZE = 450
ROI = (30, 30, 360, 360)
CENTER = np.array([225.0, 225.0])
TRANSLATION = np.array([15.0, 5.0])
AFFINE = np.array([[0.98, 0.01], [0.005, 1.06]])
BAND_AMPLITUDE_PX = 14.0
BAND_WIDTH_PX = 27.0
SETTINGS = {
    "subset_size": 21,
    "step": 30,
    "search_radius": 18,
    "zncc_min": 0.75,
    "strain_window": 3,
    "outlier_threshold_px": 1.0,
    "max_iter": 40,
    "initialization": "auto",
    "reject_nonconverged": True,
}


def make_case(case_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Generate independently evaluated reference and deformed images."""
    y, x = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    reference = analytic_texture(x, y, seed=593).astype(np.float32)
    if case_name == "localized_shear_band":
        # Material map: x' = x + tx + A exp[-(y-y0)^2/(2 sigma^2)],
        # y' = y + ty. Because y is unchanged apart from translation, its
        # inverse is exact, and no iterative inverse interpolation is needed.
        ref_y = y - TRANSLATION[1]
        shift = BAND_AMPLITUDE_PX * np.exp(-0.5 * ((ref_y - CENTER[1]) / BAND_WIDTH_PX) ** 2)
        ref_x = x - TRANSLATION[0] - shift
    elif case_name == "uniform_affine":
        inverse = np.linalg.inv(AFFINE)
        dx = x - CENTER[0] - TRANSLATION[0]
        dy = y - CENTER[1] - TRANSLATION[1]
        ref_x = CENTER[0] + inverse[0, 0] * dx + inverse[0, 1] * dy
        ref_y = CENTER[1] + inverse[1, 0] * dx + inverse[1, 1] * dy
    else:
        raise ValueError(f"unknown validation case {case_name!r}")
    deformed = analytic_texture(ref_x, ref_y, seed=593).astype(np.float32)
    return reference, deformed


def displacement_truth(case_name: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if case_name == "localized_shear_band":
        return (
            TRANSLATION[0] + BAND_AMPLITUDE_PX * np.exp(-0.5 * ((y - CENTER[1]) / BAND_WIDTH_PX) ** 2),
            np.full_like(y, TRANSLATION[1], dtype=float),
        )
    if case_name == "uniform_affine":
        displacement = np.column_stack((x - CENTER[0], y - CENTER[1])) @ (AFFINE - np.eye(2)).T
        displacement += TRANSLATION
        return displacement[:, 0], displacement[:, 1]
    raise ValueError(f"unknown validation case {case_name!r}")


def exact_global_seed(case_name: str) -> dict:
    """A deterministic global initializer for tests; it is not an oracle field."""
    linear = np.eye(2) if case_name == "localized_shear_band" else AFFINE
    if case_name not in {"localized_shear_band", "uniform_affine"}:
        raise ValueError(f"unknown validation case {case_name!r}")
    offset = CENTER + TRANSLATION - linear @ CENTER
    return {"matrix": np.column_stack((linear, offset)), "median_reprojection_error_px": 0.1}


def evaluate_case(case_name: str, *, experimental_full_search=False) -> tuple[dict, dict]:
    reference, deformed = make_case(case_name)
    start = time.perf_counter()
    options = {"experimental_full_search": True} if experimental_full_search else {}
    field = core.run_2d_dic(reference, deformed, ROI, **SETTINGS, **options)
    elapsed = time.perf_counter() - start
    expected_u, expected_v = displacement_truth(case_name, field["x"], field["y"])
    error = np.hypot(field["u_raw"] - expected_u, field["v_raw"] - expected_v)
    valid = np.asarray(field["valid"], dtype=bool) & np.isfinite(error)
    band = np.abs(np.asarray(field["y"]) - CENTER[1]) < 35.0
    valid_errors = error[valid]
    reasons, counts = np.unique(np.asarray(field["invalid_reason"], dtype=str), return_counts=True)
    input_hash = hashlib.sha256(reference.tobytes() + deformed.tobytes()).hexdigest()
    metrics = {
        "case": case_name,
        "input_sha256": input_hash,
        "field_point_count": int(valid.size),
        "correlation_valid_count": int(valid.sum()),
        "local_band_point_count": int(band.sum()),
        "local_band_valid_count": int((valid & band).sum()),
        "valid_displacement_mean_error_px": float(valid_errors.mean()) if valid_errors.size else None,
        "valid_displacement_rmse_px": float(np.sqrt(np.mean(valid_errors**2))) if valid_errors.size else None,
        "valid_displacement_max_error_px": float(valid_errors.max()) if valid_errors.size else None,
        "valid_error_over_1_px_count": int((valid_errors > 1.0).sum()),
        "accepted_error_over_0_5_px_count": int((valid_errors > 0.5).sum()),
        "known_converged_count": int((valid & np.asarray(field["convergence_known"], dtype=bool) & np.asarray(field["converged"], dtype=bool)).sum()),
        "invalid_reasons": {str(reason): int(count) for reason, count in zip(reasons, counts) if reason},
        "effective_search_radius_px": int(field["effective_search_radius_px"]),
        "initial_transform": field["initial_transform"],
        "elapsed_seconds": float(elapsed),
        "experimental_full_search": bool(field.get("experimental_full_search", False)),
        "scientific_ok": bool(field["quality_summary"]["scientific_ok"]),
        "scientific_reasons": field["quality_summary"]["scientific_reasons"],
    }
    return field, metrics


def report(*, experimental_full_search=False) -> dict:
    cases = [evaluate_case(name, experimental_full_search=experimental_full_search)[1]
             for name in ("localized_shear_band", "uniform_affine")]
    return {
        "report_version": "necking-search-validation-v3",
        "purpose": "Known-displacement assessment of global initialization and local search radius",
        "limitations": [
            "Localized case is in-plane shear, not a complete optical or mechanical tensile-neck model.",
            "Uses directly sampled floating-point synthetic images, not the experimental image-import path.",
            "Uses the normal 1 px spatial outlier threshold and explicit strict convergence.",
            "Error statistics use accepted points only; inspect valid counts to detect selection bias.",
            "Timing is a single elapsed run on this machine and is not a calibrated performance benchmark.",
            "No VIC-2D data or independent experimental ground truth is evaluated.",
            "Full-radius search is explicitly experimental and cannot pass the scientific gate, even at full coverage.",
        ],
        "provenance": {
            "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
        },
        "settings": {**SETTINGS, "roi": list(ROI), "experimental_full_search": bool(experimental_full_search)},
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this path; print to stdout when omitted")
    parser.add_argument("--core-file", type=Path, help="Load a frozen core for matched baseline evaluation")
    parser.add_argument("--experimental-full-search", action="store_true",
                        help="Explicitly evaluate the unvalidated full-radius/secondary-peak research mode")
    args = parser.parse_args()
    global core
    if args.core_file:
        spec = importlib.util.spec_from_file_location("necking_search_frozen_core", args.core_file.resolve())
        if spec is None or spec.loader is None:
            parser.error("cannot load the requested core file")
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
    output = json.dumps(report(experimental_full_search=args.experimental_full_search), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # An explicit destination is required; refuse to overwrite historical results.
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
