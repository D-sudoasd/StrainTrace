"""Manufactured full-field DIC benchmark for localization, not a camera calibration.

The image is generated from an independent continuous texture through the
inverse of a smooth, invertible 2-D deformation.  The reference-coordinate
displacement and Green--Lagrange strains have closed-form true values.  The
only numerical inversion is monotone Newton iteration in the axial direction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ezdic_core as core
from benchmarks.synthetic_cases import analytic_texture
import benchmarks.synthetic_cases as synthetic_cases

SHAPE = (160, 160)
ROI = (20, 20, 120, 120)
SEED = 381
PARAMETERS = dict(subset_size=25, step=8, search_radius=10, strain_window=5,
                  smooth_sigma=0.0, max_iter=50, conv_tol=1e-4, zncc_min=0.8,
                  reject_nonconverged=True, initialization="local")
CASES = {
    "uniform": dict(axial=0.02, peak=0.0, width=18.0, lateral=-0.006, neck=0.0),
    "broad_localization": dict(axial=0.02, peak=0.12, width=18.0, lateral=-0.006, neck=-0.05),
    "narrow_localization": dict(axial=0.04, peak=0.24, width=8.0, lateral=-0.012, neck=-0.10),
}
OCCLUSION = (52, 52, 108, 108)  # target coordinates, x0/y0 inclusive; x1/y1 exclusive


def truth(x, y, params):
    """Return u, v, and Eyy in reference-image coordinates, in px and 1."""
    cx, cy = SHAPE[1] / 2, SHAPE[0] / 2
    axial, peak, width = params["axial"], params["peak"], params["width"]
    lateral, neck = params["lateral"], params["neck"]
    dy = y - cy
    g = np.exp(-0.5 * (dy / width) ** 2)
    u = (lateral + neck * g) * (x - cx)
    v = axial * dy + peak * width * np.sqrt(np.pi / 2) * erf(dy / (np.sqrt(2) * width))
    f12 = -neck * g * dy / width**2 * (x - cx)
    f22 = 1 + axial + peak * g
    eyy = 0.5 * (f12**2 + f22**2 - 1)
    return u, v, eyy


def image_pair(params):
    """Make target pixels by solving for reference positions, not raster warping."""
    y, x = np.indices(SHAPE, dtype=float)
    cy, cx = SHAPE[0] / 2, SHAPE[1] / 2
    # y' = y + v(y) is strictly increasing in all manufactured cases.
    ry = (y - cy) / (1 + params["axial"])
    for _ in range(30):
        g = np.exp(-0.5 * (ry / params["width"]) ** 2)
        residual = ((1 + params["axial"]) * ry +
                    params["peak"] * params["width"] * np.sqrt(np.pi / 2) *
                    erf(ry / (np.sqrt(2) * params["width"])) - (y - cy))
        ry -= residual / (1 + params["axial"] + params["peak"] * g)
    if np.max(np.abs(residual)) > 1e-10:
        raise RuntimeError("inverse deformation failed to converge")
    g = np.exp(-0.5 * (ry / params["width"]) ** 2)
    lateral_stretch = 1 + params["lateral"] + params["neck"] * g
    if np.min(lateral_stretch) <= 0:
        raise ValueError("deformation is not invertible")
    rx = cx + (x - cx) / lateral_stretch
    return analytic_texture(x, y, seed=SEED), analytic_texture(rx, cy + ry, seed=SEED)


def _stats(error):
    error = np.asarray(error, dtype=float)
    error = error[np.isfinite(error)]
    if not error.size:
        return {"count": 0, "bias": None, "rmse": None, "p95_abs": None, "max_abs": None}
    return {"count": int(error.size), "bias": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "p95_abs": float(np.percentile(np.abs(error), 95)),
            "max_abs": float(np.max(np.abs(error)))}


def _distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {"count": int(values.size), "min": float(np.min(values)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def _footprint_categories(X, Y, params, solver_params):
    """Classify oracle-deformed subset footprints with a 3-pixel buffer.

    The spline sampler has support beyond the center coordinates, so a
    fully covered subset must be at least three pixels inside the occlusion.
    """
    x0, y0, x1, y1 = OCCLUSION
    h = solver_params["subset_size"] // 2
    oy, ox = np.mgrid[-h:h + 1, -h:h + 1]
    covered = np.zeros(X.shape, dtype=bool)
    clear = np.zeros(X.shape, dtype=bool)
    for index in np.ndindex(X.shape):
        rx, ry = X[index] + ox, Y[index] + oy
        u, v, _ = truth(rx, ry, params)
        px, py = rx + u, ry + v
        covered[index] = np.all((px >= x0 + 3) & (px < x1 - 3) &
                                (py >= y0 + 3) & (py < y1 - 3))
        clear[index] = (np.all(px < x0 - 3) or np.all(px >= x1 + 3) or
                        np.all(py < y0 - 3) or np.all(py >= y1 + 3))
    return covered, clear


def run_case(params, solver_core=core, corruption=None, solver_params=None, noise_sigma=0.0):
    settings = dict(PARAMETERS)
    if solver_params:
        settings.update(solver_params)
    reference, target = image_pair(params)
    if corruption == "occlusion":
        x0, y0, x1, y1 = OCCLUSION
        target[y0:y1, x0:x1] = 128.0
    elif corruption == "unrelated_texture":
        y, x = np.indices(SHAPE, dtype=float)
        target = analytic_texture(x, y, seed=SEED + 1)
    elif corruption is not None:
        raise ValueError("unknown corruption")
    if noise_sigma:
        rng = np.random.default_rng(20261028)
        reference = reference + rng.normal(0, noise_sigma, SHAPE)
        target = target + rng.normal(0, noise_sigma, SHAPE)
    started = time.perf_counter()
    field = solver_core.run_2d_dic(reference, target, ROI, **settings)
    elapsed = time.perf_counter() - started
    X, Y = np.asarray(field["X"]), np.asarray(field["Y"])
    u, v, eyy = truth(X, Y, params)
    valid = np.asarray(field["valid"], dtype=bool).reshape(X.shape)
    strain_valid = np.asarray(field["strain_valid"], dtype=bool).reshape(X.shape)
    measured_eyy = np.asarray(field["Eyy_grid"], dtype=float)
    oracle_fit = solver_core.compute_strain_fields(
        X, Y, np.where(valid, u, np.nan), np.where(valid, v, np.nan),
        window=settings["strain_window"], smooth_sigma=settings["smooth_sigma"])
    oracle_fit_eyy = np.asarray(oracle_fit["Eyy"], dtype=float)
    disp_error = np.hypot(field["U"] - u, field["V"] - v)
    disp_error[~valid] = np.nan
    strain_error = np.where(strain_valid, measured_eyy - eyy, np.nan)
    center_column = int(np.argmin(np.abs(X[0] - SHAPE[1] / 2)))
    line = [{"y_px": float(Y[i, center_column]), "true_eyy": float(eyy[i, center_column]),
             "estimated_eyy": (float(measured_eyy[i, center_column])
                               if strain_valid[i, center_column] else None)}
            for i in range(X.shape[0])]
    counts = {}
    for reason in np.asarray(field["invalid_reason"], dtype=object)[~valid.ravel()]:
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    peak_index = np.unravel_index(np.nanargmax(np.where(strain_valid, measured_eyy, np.nan)), X.shape) if strain_valid.any() else None
    true_peak_index = np.unravel_index(np.argmax(eyy), X.shape)
    line_error = np.asarray([p["estimated_eyy"] - p["true_eyy"]
                             for p in line if p["estimated_eyy"] is not None])
    line_estimates = [p for p in line if p["estimated_eyy"] is not None]
    line_peak = max(line_estimates, key=lambda p: p["estimated_eyy"]) if line_estimates else None
    line_true_peak = max(line, key=lambda p: p["true_eyy"])
    quality = field["quality_summary"]
    corrupted_acceptance = None
    if corruption == "occlusion":
        covered, clear = _footprint_categories(X, Y, params, settings)
        categories = {"fully_covered": covered, "clear": clear, "partial_or_buffer": ~(covered | clear)}
        corrupted_acceptance = {key: {"count": int(mask.sum()), "accepted": int((mask & valid).sum())}
                                for key, mask in categories.items()}
    elif corruption == "unrelated_texture":
        corrupted_acceptance = {"no_physical_correspondence": {"count": int(valid.size),
                                                                "accepted": int(valid.sum())}}
    # Maximum measured only among valid values and true maximum over the entire
    # eligible grid are reported separately so rejected peaks cannot look good.
    result = {"input_sha256": hashlib.sha256(reference.tobytes() + target.tobytes()).hexdigest(),
            "solver_parameters": settings, "independent_gaussian_image_noise_sigma_gray": noise_sigma,
            "point_count": int(valid.size), "correlation_valid_count": int(valid.sum()),
            "strain_valid_count": int(strain_valid.sum()),
            "invalid_reason_counts": counts,
            "displacement_vector_error_px": _stats(disp_error),
            "accepted_vector_error_over_0_5_px_count": int(np.count_nonzero(disp_error > 0.5)),
            "eyy_error": _stats(strain_error),
            "centerline_eyy_error": _stats(line_error),
            "centerline_true_peak": {"y_px": line_true_peak["y_px"], "eyy": line_true_peak["true_eyy"]},
            "centerline_estimated_peak": ({"y_px": line_peak["y_px"], "eyy": line_peak["estimated_eyy"]}
                                          if line_peak else None),
            "quality": {"scientific_ok": quality["scientific_ok"],
                        "scientific_reasons": quality["scientific_reasons"],
                        "thresholds": quality["thresholds"],
                        "zncc_accepted": _distribution(np.asarray(field["zncc"])[valid.ravel()]),
                        "residual_rms_accepted": _distribution(np.asarray(field["residual_rms"])[valid.ravel()])},
            "true_peak_eyy_on_grid": float(np.max(eyy)),
            "true_peak_location_xy_px": [float(X[true_peak_index]), float(Y[true_peak_index])],
            "estimated_peak_eyy_on_valid_grid": (float(np.max(measured_eyy[strain_valid]))
                                                  if strain_valid.any() else None),
            "estimated_peak_location_xy_px": ([float(X[peak_index]), float(Y[peak_index])]
                                               if peak_index is not None else None),
            "exact_displacement_windowed_eyy_error": _stats(np.where(strain_valid, oracle_fit_eyy - eyy, np.nan)),
            "exact_displacement_windowed_peak_eyy": (float(np.nanmax(np.where(strain_valid, oracle_fit_eyy, np.nan)))
                                                      if strain_valid.any() else None),
            "corruption": corruption, "corruption_acceptance": corrupted_acceptance,
            "centerline": line, "elapsed_s": elapsed}
    if corruption == "unrelated_texture":
        # No displacement or strain truth exists for unrelated random images.
        for key in ("displacement_vector_error_px", "accepted_vector_error_over_0_5_px_count",
                    "eyy_error", "centerline_eyy_error", "centerline_true_peak", "centerline_estimated_peak", "true_peak_eyy_on_grid",
                    "true_peak_location_xy_px", "estimated_peak_eyy_on_valid_grid",
                    "estimated_peak_location_xy_px", "exact_displacement_windowed_eyy_error",
                    "exact_displacement_windowed_peak_eyy", "centerline"):
            result[key] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-file", type=Path, help="Archived core for a matched comparison")
    args = parser.parse_args()
    solver_core = core
    if args.core_file:
        spec = importlib.util.spec_from_file_location("archived_necking_core", args.core_file)
        solver_core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solver_core)
    result = {"schema": "necking-validation-v1", "core_sha256": hashlib.sha256(
                  Path(solver_core.__file__).read_bytes()).hexdigest(),
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "texture_source_sha256": hashlib.sha256(Path(synthetic_cases.__file__).read_bytes()).hexdigest(),
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "shape": SHAPE, "roi": ROI, "texture_seed": SEED, "solver_parameters": PARAMETERS,
              "strain_definition": "Green-Lagrange Eyy in reference coordinates, dimensionless",
              "limitations": ["Ideal 2-D static speckle and prescribed smooth mapping; no out-of-plane motion, optical blur, camera noise, fracture, or experimental uncertainty calibration.",
                              "Timing is environment-dependent; this is not a VIC-2D comparison."],
              "cases": {name: {"deformation": params, **run_case(params, solver_core)}
                        for name, params in CASES.items()}}
    result["cases"]["occluded_broad"] = {"deformation": CASES["broad_localization"],
                                          "occlusion_xyxy_target_px": OCCLUSION,
                                          **run_case(CASES["broad_localization"], solver_core, "occlusion")}
    result["cases"]["unrelated_target"] = {"deformation": CASES["uniform"],
                                            **run_case(CASES["uniform"], solver_core, "unrelated_texture")}
    fine = {"subset_size": 15, "step": 4, "strain_window": 3}
    for name, settings, noise in (
        ("narrow_fine", fine, 0.0),
        ("narrow_default_noisy", {}, 1.5),
        ("narrow_fine_noisy", fine, 1.5),
    ):
        result["cases"][name] = {"deformation": CASES["narrow_localization"],
                                  **run_case(CASES["narrow_localization"], solver_core,
                                             solver_params=settings, noise_sigma=noise)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
