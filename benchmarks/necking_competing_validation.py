"""Measure competing-peak decisions against a manufactured displacement field.

The in-memory control disables the competing-peak probe on the same solver and
images. Its accepted candidate is compared with analytic displacement truth;
the illustrative 0.5 px bound is not an experimental uncertainty estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ezdic_core as core
from benchmarks import necking_validation as neck


def run_case(noise_sigma: float) -> dict:
    params = neck.CASES["narrow_localization"]
    reference, target = neck.image_pair(params)
    if noise_sigma:
        rng = np.random.default_rng(20261028)
        reference = reference + rng.normal(0, noise_sigma, neck.SHAPE)
        target = target + rng.normal(0, noise_sigma, neck.SHAPE)
    settings = dict(neck.PARAMETERS, subset_size=15, step=4, strain_window=3)
    original_peak, original_refine = core._competing_correlation_peak, core._refine_competing_peak
    try:
        with_probe = core.run_2d_dic(reference, target, neck.ROI, **settings)
        core._competing_correlation_peak = lambda *_args, **_kwargs: None
        core._refine_competing_peak = lambda *_args, **_kwargs: {"status": "PROBE_DISABLED_FOR_CONTROL", "result": None}
        without_probe = core.run_2d_dic(reference, target, neck.ROI, **settings)
    finally:
        core._competing_correlation_peak, core._refine_competing_peak = original_peak, original_refine
    if not (np.array_equal(with_probe["x"], without_probe["x"]) and np.array_equal(with_probe["y"], without_probe["y"])):
        raise RuntimeError("POI grids differ")
    x, y = with_probe["x"], with_probe["y"]
    truth_u, truth_v, _ = neck.truth(x, y, params)
    error_on = np.hypot(with_probe["u_raw"] - truth_u, with_probe["v_raw"] - truth_v)
    error_off = np.hypot(without_probe["u_raw"] - truth_u, without_probe["v_raw"] - truth_v)
    valid_on, valid_off = with_probe["valid"], without_probe["valid"]
    lost = valid_off & ~valid_on
    finite_lost = lost & np.isfinite(error_off)
    lost_rows = []
    for i in np.flatnonzero(finite_lost):
        def finite_or_none(value):
            return float(value) if np.isfinite(value) else None
        lost_rows.append({
            "x_px": float(x[i]), "y_px": float(y[i]),
            "control_displacement_error_px": float(error_off[i]),
            "invalid_reason": str(with_probe["invalid_reason"][i]),
            "competing_status": str(with_probe["competing_peak_status"][i]),
            "competing_raw_zncc": finite_or_none(with_probe["competing_peak_zncc"][i]),
            "competing_refined_zncc": finite_or_none(with_probe["competing_peak_refined_zncc"][i]),
        })
    return {
        "noise_sigma_gray": noise_sigma,
        "input_sha256": hashlib.sha256(reference.tobytes() + target.tobytes()).hexdigest(),
        "settings": dict(settings, roi=list(neck.ROI)),
        "point_count": int(valid_on.size),
        "control_valid_count": int(valid_off.sum()),
        "probe_valid_count": int(valid_on.sum()),
        "control_accepted_error_over_0_5_px": int((valid_off & (error_off > 0.5)).sum()),
        "probe_accepted_error_over_0_5_px": int((valid_on & (error_on > 0.5)).sum()),
        "control_valid_probe_invalid_count": int(lost.sum()),
        "control_valid_probe_invalid_control_error_at_most_0_5_px_count": int((finite_lost & (error_off <= .5)).sum()),
        "control_valid_probe_invalid_control_error_over_0_5_px_count": int((finite_lost & (error_off > .5)).sum()),
        "probe_valid_control_invalid_count": int((valid_on & ~valid_off).sum()),
        "control_valid_probe_invalid_points": lost_rows,
    }


def run_copied_patch(noise_sigma: float) -> dict:
    """Known (5, 0) px translation plus an exact displaced texture copy."""
    rng = np.random.default_rng(3)
    reference = gaussian_filter(rng.normal(size=(130, 130)), 1.2).astype("float32")
    target = np.roll(reference, 5, axis=1).copy()
    target[50:71, 55:76] += rng.normal(size=(21, 21)).astype("float32") * noise_sigma
    target[67:88, 72:93] = reference[50:71, 50:71]
    transform = {"matrix": np.array([[1., 0., 5.], [0., 1., 0.]]), "median_reprojection_error_px": 0.1}
    settings = {
        "subset_size": 21, "step": 30, "search_radius": 18, "strain_window": 5,
        "max_iter": 40, "zncc_min": 0.75, "reject_nonconverged": True,
    }
    field = core.run_2d_dic(reference, target, (20, 20, 81, 81), **settings,
                            _global_initial_transform=transform)
    index = 4
    def number(value):
        return float(value) if np.isfinite(value) else None
    u = number(field["u_raw"][index])
    v = number(field["v_raw"][index])
    return {
        "target_patch_noise_sigma": noise_sigma,
        "input_sha256": hashlib.sha256(reference.tobytes() + target.tobytes()).hexdigest(),
        "settings": dict(settings, roi=[20, 20, 81, 81]),
        "true_displacement_uv_px": [5.0, 0.0],
        "copied_displacement_uv_px": [22.0, 17.0],
        "point_count": int(len(field["valid"])),
        "valid_count": int(field["valid"].sum()),
        "strain_valid_count": int(field["strain_valid"].sum()),
        "numeric_quality_gate_scientific_ok": bool(field["quality_summary"]["scientific_ok"]),
        "center_valid": bool(field["valid"][index]),
        "center_displacement_uv_px": [u, v],
        "center_error_px": None if u is None or v is None else float(np.hypot(u-5.0, v)),
        "center_invalid_reason": str(field["invalid_reason"][index]),
        "center_peak_is_ambiguous": bool(field["peak_is_ambiguous"][index]),
        "competing_raw_zncc": number(field["competing_peak_zncc"][index]),
        "competing_refined_zncc": number(field["competing_peak_refined_zncc"][index]),
        "competing_refined_zncc_gap": number(field["competing_peak_zncc_gap"][index]),
        "competing_peak_status": str(field["competing_peak_status"][index]),
        "competing_peak_max_zncc_gap": field["competing_peak_max_zncc_gap"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema": "competing-peak-validation-v1",
        "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_script_sha256": hashlib.sha256(Path(neck.__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__},
        "limits": [
            "Synthetic 2-D deformation and texture; no out-of-plane motion, fracture, or experimentally calibrated uncertainty.",
            "The control disables the competing-peak probe only; it does not supply a ground-truth candidate for all rejections.",
            "The 0.5 px displacement threshold is an illustrative assessment criterion, not a calibrated tolerance.",
            "Accepted candidates may still be inaccurate; counts apply only to these images and thresholds.",
            "The copied-patch sweep is an adversarial synthetic construction; it demonstrates a false accept that ordinary quality gates can miss.",
        ],
        "cases": {"clean": run_case(0.0), "noise_sigma_1_5": run_case(1.5)},
        "copied_patch_noise_sweep": {
            f"sigma_{sigma:.2f}": run_copied_patch(sigma) for sigma in (0.06, 0.08, 0.10)
        },
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
