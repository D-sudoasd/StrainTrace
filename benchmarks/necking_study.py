"""Separate correlation, finite strain-gauge and texture-loss errors in 2-D DIC.

Reuses the independent continuous-texture necking generator reviewed in the
28 October branch. All runs use strict convergence, the same quality gates,
zero added displacement smoothing, and preserve invalid points. The central
POI coordinates are identical when changing subset size; a smaller ROI
padding merely compensates for the changed subset half-width.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ezdic_core as core
from benchmarks import necking_validation as bench
from benchmarks import synthetic_cases

SCHEMA = "straintrace-necking-study-v1"
ERROR_DIAGNOSTIC_PX = 0.5  # Descriptive, never passed to a solver quality gate.
POINT_COLUMNS = (
    "case", "x_px", "y_px", "eligible", "correlation_valid", "strain_valid",
    "u_px", "v_px", "true_u_px", "true_v_px", "Eyy", "true_Eyy",
    "exact_displacement_full_Eyy", "exact_displacement_masked_Eyy", "zncc",
    "invalid_reason", "strain_invalid_reason", "spatial_outlier",
    "u_pre_outlier_px", "v_pre_outlier_px", "u_candidate_px", "v_candidate_px",
    "zncc_candidate", "candidate_observed",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_core(path):
    spec = importlib.util.spec_from_file_location("necking_study_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locked_roi(subset_size):
    """The unchanged 12 x 12 POI domain is x,y=32,40,...,120 px."""
    if subset_size < 3 or subset_size % 2 != 1:
        raise ValueError("use an odd subset size")
    half = subset_size // 2
    return 32-half, 32-half, 89+2*half, 89+2*half


def cases():
    # Each one-factor comparison has the identical POI grid and image hashes.
    narrow = bench.CASES["narrow_localization"]
    broad = bench.CASES["broad_localization"]
    rows = [(name, params, {}, {}) for name, params in bench.CASES.items()]
    rows.extend([
        ("narrow_window3", narrow, {"strain_window": 3}, {}),
        ("narrow_window7", narrow, {"strain_window": 7}, {}),
        ("narrow_subset15", narrow, {"subset_size": 15}, {}),
        ("narrow_subset15_window3", narrow, {"subset_size": 15, "strain_window": 3}, {}),
        ("narrow_noise1_5", narrow, {}, {"noise_sigma": 1.5}),
        ("narrow_subset15_noise1_5", narrow, {"subset_size": 15}, {"noise_sigma": 1.5}),
    ])
    for mix in (0.25, 0.5, 0.75, 1.0):
        rows.append((f"broad_texture_mix_{mix:g}", broad, {}, {"texture_mix": mix}))
    rows.append(("broad_blur2", broad, {}, {"blur_sigma": 2.0}))
    return rows


def _array(field, key, shape, default=np.nan):
    return np.asarray(field.get(key, np.full(shape, default))).reshape(shape)


def _count_reasons(values, mask):
    names, counts = np.unique(np.asarray(values)[mask], return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)}


def _number(value):
    return float(value) if np.isfinite(value) else None


def _peak(X, Y, values, mask):
    accepted = mask & np.isfinite(values)
    if not accepted.any():
        return None
    maximum = float(values[accepted].max())
    indices = np.argwhere(accepted & np.isclose(values, maximum, rtol=1e-12, atol=1e-12))
    return {"value": maximum,
            "locations_xy_px": [[float(X[i, j]), float(Y[i, j])] for i, j in indices]}


def summarize_field(field, params, settings, solver_core):
    X, Y = field["X"], field["Y"]
    shape = X.shape
    eligible = _array(field, "eligible", shape, True).astype(bool)
    valid = _array(field, "valid", shape, False).astype(bool) & eligible
    sv = _array(field, "strain_valid", shape, False).astype(bool) & valid
    true_u, true_v, true_Eyy = bench.truth(X, Y, params)
    U, V = _array(field, "U", shape), _array(field, "V", shape)
    Eyy = _array(field, "Eyy_grid", shape)
    subset_F = np.asarray(field["P"])
    subset_supported = solver_core._positive_deformation_jacobian(
        1 + subset_F[..., 1], subset_F[..., 2], subset_F[..., 4], 1 + subset_F[..., 5])
    fit_support = valid & subset_supported
    fitting = dict(window=settings["strain_window"], smooth_sigma=0,
                   degree=settings.get("strain_degree", 2),
                   robust=settings.get("robust_strain", True),
                   max_condition_number=field.get("max_condition_number", 1e12))
    full_oracle = solver_core.compute_strain_fields(X, Y, true_u, true_v, **fitting)
    masked_oracle = solver_core.compute_strain_fields(
        X, Y, np.where(fit_support, true_u, np.nan),
        np.where(fit_support, true_v, np.nan), **fitting)
    oracle_full_E = np.asarray(full_oracle["Eyy"])
    oracle_masked_E = np.asarray(masked_oracle["Eyy"])
    delta = np.hypot(U-true_u, V-true_v)
    raw_u, raw_v = _array(field, "u_raw", shape), _array(field, "v_raw", shape)
    raw_error = np.hypot(raw_u-true_u, raw_v-true_v)
    outlier = _array(field, "spatial_outlier", shape, False).astype(bool)
    # This exposes false rejection of a non-polynomial but exact displacement.
    exact_outliers = solver_core._displacement_outliers(
        X, Y, true_u, true_v, settings.get("outlier_threshold_px", 1.0))
    candidate_present = all(key in field for key in ("u_candidate", "v_candidate"))
    candidate_u = _array(field, "u_candidate", shape)
    candidate_v = _array(field, "v_candidate", shape)
    candidate_error = np.hypot(candidate_u-true_u, candidate_v-true_v)
    candidate_known = np.isfinite(candidate_error) & eligible
    rejected_known = ~valid & candidate_known
    neck = eligible & (np.abs(Y-bench.SHAPE[0]/2) <= params["width"])
    regions = {}
    for name, domain in (("entire_poi_domain", eligible), ("neck_band", neck)):
        accepted = domain & valid
        strain_accepted = domain & sv
        denominator = int(domain.sum())
        max_truth = float(true_Eyy[domain].max()) if denominator else np.nan
        true_peak = domain & np.isclose(true_Eyy, max_truth, rtol=1e-12, atol=1e-12)
        common = strain_accepted & np.isfinite(oracle_masked_E)
        regions[name] = {
            "eligible_count": denominator,
            "correlation_valid_count": int(accepted.sum()),
            "strain_valid_count": int(strain_accepted.sum()),
            "correlation_coverage": float(accepted.sum()/denominator) if denominator else None,
            "strain_coverage": float(strain_accepted.sum()/denominator) if denominator else None,
            "displacement_vector_error_px": bench._stats(delta[accepted]),
            "accepted_error_above_0_5_px_count": int(np.count_nonzero(accepted & (delta > ERROR_DIAGNOSTIC_PX))),
            "Eyy_error": bench._stats((Eyy-true_Eyy)[strain_accepted]),
            "true_peak_full_domain": _peak(X, Y, true_Eyy, domain),
            "true_peak_on_measured_support": _peak(X, Y, true_Eyy, strain_accepted),
            "measured_peak": _peak(X, Y, Eyy, strain_accepted),
            "true_peak_point_count": int(true_peak.sum()),
            "true_peak_points_with_measured_strain": int((true_peak & sv).sum()),
            "exact_displacement_full_grid_peak": _peak(X, Y, oracle_full_E, domain),
            "exact_displacement_full_grid_error": bench._stats((oracle_full_E-true_Eyy)[domain]),
            "exact_displacement_same_support_error": bench._stats((oracle_masked_E-true_Eyy)[common]),
            "measured_minus_exact_displacement_same_support": bench._stats((Eyy-oracle_masked_E)[common]),
            "strain_invalid_reasons": _count_reasons(_array(field, "strain_invalid_reason", shape, ""), domain & ~sv),
            "correlation_invalid_reasons": _count_reasons(_array(field, "invalid_reason", shape, ""), domain & ~valid),
        }
    result = {
        "locked_poi_domain_xy_px": [32, 32, 120, 120],
        "grid_sha256": hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
        "strain_gauge_center_span_px": (settings["strain_window"]-1)*settings["step"],
        "nominal_image_support_width_px": settings["subset_size"]+(settings["strain_window"]-1)*settings["step"],
        "neck_band_definition": "abs(y - 80 px) <= Gaussian width parameter",
        "regions": regions,
        "spatial_outlier_diagnostic": {
            "rejected_count": int(outlier.sum()),
            "rejected_pre_outlier_error_le_0_5_px": int((outlier & (raw_error <= ERROR_DIAGNOSTIC_PX)).sum()),
            "rejected_pre_outlier_error_gt_0_5_px": int((outlier & (raw_error > ERROR_DIAGNOSTIC_PX)).sum()),
            "exact_displacement_rejected_count": int(exact_outliers.sum()),
        },
        "refinement_candidate_diagnostic": {
            "available": candidate_present,
            "finite_candidate_count": int(candidate_known.sum()) if candidate_present else None,
            "rejected_candidate_error_le_0_5_px": int((rejected_known & (candidate_error <= ERROR_DIAGNOSTIC_PX)).sum()) if candidate_present else None,
            "rejected_candidate_error_gt_0_5_px": int((rejected_known & (candidate_error > ERROR_DIAGNOSTIC_PX)).sum()) if candidate_present else None,
            "rejected_with_unknown_candidate_count": int((eligible & ~valid & ~candidate_known).sum()),
        },
        "decomposition_note": "At identical valid points, measured-minus-truth equals measured-minus-masked-oracle plus masked-oracle-minus-truth. These are conditional diagnostics, not independent uncertainty sources.",
    }
    points = []
    for index in np.ndindex(shape):
        i, j = index
        points.append(dict(zip(POINT_COLUMNS[1:], (
            X[index], Y[index], bool(eligible[index]), bool(valid[index]), bool(sv[index]),
            U[index], V[index], true_u[index], true_v[index], Eyy[index], true_Eyy[index],
            oracle_full_E[index], oracle_masked_E[index], _array(field, "zncc", shape)[index],
            _array(field, "invalid_reason", shape, "")[index],
            _array(field, "strain_invalid_reason", shape, "")[index], bool(outlier[index]),
            raw_u[index], raw_v[index], candidate_u[index], candidate_v[index],
            _array(field, "zncc_candidate", shape)[index], bool(candidate_known[index]),
        ))))
    return result, points


def deformation_gradient(x, y, params):
    dy = y-bench.SHAPE[0]/2
    dx = x-bench.SHAPE[1]/2
    g = np.exp(-0.5*(dy/params["width"])**2)
    return np.array([[1+params["lateral"]+params["neck"]*g,
                      -params["neck"]*g*dy/params["width"]**2*dx],
                     [0., 1+params["axial"]+params["peak"]*g]])


def reference_strategy_diagnostic(solver_core):
    """Compare local affine model fits with an ideal, error-free intermediate.

    True reference-to-middle positions and gradients are supplied. Therefore
    this is an upper-bound diagnostic of the final step, not an incremental
    implementation, a measurement of accumulated errors, or a quality gate.
    """
    rows = []
    for name in ("uniform", "broad_localization", "narrow_localization"):
        final_params = bench.CASES[name]
        middle_params = {key: value if key == "width" else value*0.5
                         for key, value in final_params.items()}
        reference, final_image = bench.image_pair(final_params)
        _, middle_image = bench.image_pair(middle_params)
        prepared_fixed = solver_core._prepare_dic_images(reference, final_image)
        prepared_middle = solver_core._prepare_dic_images(middle_image, final_image)
        for x, y in ((80., 72.), (80., 80.), (96., 72.), (112., 80.)):
            uf, vf, _ = bench.truth(x, y, final_params)
            um, vm, _ = bench.truth(x, y, middle_params)
            final_displacement = np.array([uf, vf])
            middle_displacement = np.array([um, vm])
            F_final = deformation_gradient(x, y, final_params)
            F_middle = deformation_gradient(x, y, middle_params)
            F_relative = F_final @ np.linalg.inv(F_middle)
            fixed_seed = np.array([uf, F_final[0, 0]-1, F_final[0, 1],
                                   vf, F_final[1, 0], F_final[1, 1]-1])
            relative_seed = np.array([uf-um, F_relative[0, 0]-1, F_relative[0, 1],
                                      vf-vm, F_relative[1, 0], F_relative[1, 1]-1])
            fixed = solver_core.refine_subset_icgn(
                reference, final_image, x, y, 25, p0=fixed_seed,
                max_iter=50, tol=1e-4, _prepared=prepared_fixed)
            middle = solver_core.refine_subset_icgn(
                middle_image, final_image, x+um, y+vm, 25, p0=relative_seed,
                max_iter=50, tol=1e-4, _prepared=prepared_middle)
            record = {"case": name, "original_material_point_xy_px": [x, y],
                      "known_middle_point_xy_px": [float(x+um), float(y+vm)],
                      "true_final_displacement_uv_px": final_displacement.tolist(),
                      "input_sha256": hashlib.sha256(reference.tobytes()+middle_image.tobytes()+final_image.tobytes()).hexdigest()}
            for key, fit, offset in (("fixed", fixed, np.zeros(2)),
                                     ("oracle_middle", middle, middle_displacement)):
                if fit is None:
                    record[key] = {"candidate_exists": False}
                    continue
                total = np.array([fit["u"], fit["v"]])+offset
                record[key] = {
                    "candidate_exists": True, "total_displacement_uv_px": total.tolist(),
                    "total_displacement_vector_error_px": float(np.linalg.norm(total-final_displacement)),
                    "converged": bool(fit["converged"]), "zncc": float(fit["zncc"]),
                    "residual_rms": float(fit["residual_rms"]), "stop_reason": str(fit["stop_reason"]),
                }
            rows.append(record)
    return {
        "subset_size_px": 25, "solver": "IC-GN", "known_middle_deformation_fraction": 0.5,
        "initialization": "exact material point, displacement and deformation gradients in both routes",
        "composition": "u_total(X) = u_middle_exact(X) + u_middle_to_final(X + u_middle_exact(X))",
        "interpretation": "Only the final local solve is measured; earlier correspondence and displacement are exact. Equal pixel subset widths in different reference images do not represent identical material footprints. No accumulated tracking error or full-field strain composition is measured.",
        "points": rows,
    }


def _csv_value(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def matched_case_comparisons(records, points_by_case):
    pairs = (("narrow_localization", "narrow_window3"),
             ("narrow_localization", "narrow_window7"),
             ("narrow_localization", "narrow_subset15"),
             ("narrow_window3", "narrow_subset15_window3"),
             ("narrow_noise1_5", "narrow_subset15_noise1_5"))
    comparisons = []
    for before, after in pairs:
        if before not in records or after not in records:
            continue
        a, b = records[before], records[after]
        if a["input_sha256"] != b["input_sha256"] or a["grid_sha256"] != b["grid_sha256"]:
            raise ValueError("Sensitivity comparison requires identical images and POI coordinates")
        pa, pb = points_by_case[before], points_by_case[after]
        va = np.array([p["correlation_valid"] for p in pa], bool)
        vb = np.array([p["correlation_valid"] for p in pb], bool)
        sa = np.array([p["strain_valid"] for p in pa], bool)
        sb = np.array([p["strain_valid"] for p in pb], bool)
        common_u = va & vb
        common_s = sa & sb
        record = {
            "before": before, "after": after, "point_count": len(pa),
            "common_correlation_valid_count": int(common_u.sum()),
            "gained_correlation_valid_count": int((~va & vb).sum()),
            "lost_correlation_valid_count": int((va & ~vb).sum()),
            "common_strain_valid_count": int(common_s.sum()),
            "gained_strain_valid_count": int((~sa & sb).sum()),
            "lost_strain_valid_count": int((sa & ~sb).sum()),
        }
        for label, rows in (("before", pa), ("after", pb)):
            eu = np.array([p["u_px"]-p["true_u_px"] for p in rows])
            ev = np.array([p["v_px"]-p["true_v_px"] for p in rows])
            ee = np.array([p["Eyy"]-p["true_Eyy"] for p in rows])
            record[label+"_common_displacement_error_px"] = bench._stats(np.hypot(eu, ev)[common_u])
            record[label+"_common_Eyy_error"] = bench._stats(ee[common_s])
        comparisons.append(record)
    return comparisons


def run(output, core_file=None, selected_cases=None, *, experimental_full_search=False):
    output = Path(output)
    points_path = output.with_name(output.stem+".points.csv")
    if output.exists() or points_path.exists():
        raise FileExistsError("Use new output files; existing research evidence is preserved")
    solver_core = core if core_file is None else load_core(core_file)
    core_path = Path(solver_core.__file__)
    source_hash = sha256(core_path)
    selected = set(selected_cases) if selected_cases else None
    available = {name for name, *_ in cases()}
    if selected is not None and not selected <= available:
        raise ValueError(f"unknown cases: {sorted(selected-available)}")
    report = {
        "schema": SCHEMA, "core_sha256": source_hash,
        "benchmark_sha256": sha256(__file__), "generator_sha256": sha256(bench.__file__),
        "texture_source_sha256": sha256(synthetic_cases.__file__),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "opencv": cv2.__version__},
        "deformation_reference": "fixed initial image; x right, y down; lengths in px",
        "strain_definition": "Green-Lagrange Eyy, dimensionless; tensor strain in reference coordinates",
        "diagnostic_error_cutoff_px": ERROR_DIAGNOSTIC_PX,
        "search_policy": {
            "experimental_full_search": bool(experimental_full_search),
            "interpretation": (
                "Explicit experimental broad search and competing-candidate checks; numerical approval must remain false."
                if experimental_full_search else "Default solver policy; no experimental parameter is supplied to the core."
            ),
        },
        "cases": {},
        "limitations": [
            "Manufactured static in-plane images; no experimental or VIC-2D ground truth.",
            "Local texture mixing and Gaussian blur are controlled stress tests, not a calibrated camera degradation model.",
            "Accepted-point error is always accompanied by whole-domain and neck-band coverage; a numerical quality pass does not certify resolved peaks.",
            "No temporal reference update is implemented or inferred from this fixed-reference study.",
        ],
    }
    all_points = []
    points_by_case = {}
    cv2.setRNGSeed(0)
    for name, deformation, overrides, degradation in cases():
        if selected is not None and name not in selected:
            continue
        settings = {**bench.PARAMETERS, **overrides}
        # Do not pass a new keyword with False to archived cores: an omitted
        # flag preserves their native default and makes the comparison exact.
        if experimental_full_search:
            settings["experimental_full_search"] = True
        record, field = bench.run_case(
            deformation, solver_core, solver_params=settings,
            roi=locked_roi(settings["subset_size"]), return_field=True, **degradation)
        if experimental_full_search and record["quality"]["scientific_ok"]:
            raise RuntimeError("Experimental search must not report numerical approval")
        extra, points = summarize_field(field, deformation, settings, solver_core)
        if degradation.get("texture_mix") == 1.0:
            covered, clear = bench._footprint_categories(field["X"], field["Y"], deformation, settings)
            extra["fully_destroyed_texture"] = {
                "fully_covered_subsets": int(covered.sum()),
                "accepted_fully_covered_subsets": int((covered.ravel() & field["valid"]).sum()),
                "interpretation": "No original material texture remains inside these buffered subset footprints; retained kinematic truth is the prescribed underlying motion, not an observable correspondence.",
            }
        report["cases"][name] = {"deformation": deformation, **record, **extra}
        points_by_case[name] = points
        all_points.extend({"case": name, **point} for point in points)
        region = extra["regions"]["neck_band"]
        print(json.dumps({"case": name, "accepted": record["correlation_valid_count"],
                          "neck_strain_coverage": region["strain_coverage"],
                          "Eyy_rmse": record["eyy_error"]["rmse"]}), flush=True)
    report["ideal_intermediate_reference_diagnostic"] = reference_strategy_diagnostic(solver_core)
    report["matched_sensitivity_comparisons"] = matched_case_comparisons(report["cases"], points_by_case)
    if sha256(core_path) != source_hash:
        raise RuntimeError("Core changed during the run; repeat with a frozen core")
    output.parent.mkdir(parents=True, exist_ok=True)
    with points_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=POINT_COLUMNS)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in all_points)
    report["pointwise_csv"] = {"filename": points_path.name, "sha256": sha256(points_path),
                               "row_count": len(all_points), "missing_numeric_values": "empty cells"}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--core-file", type=Path)
    parser.add_argument("--case", action="append", dest="cases", choices=sorted(name for name, *_ in cases()))
    parser.add_argument("--experimental-full-search", action="store_true",
                        help="Explicitly test the unvalidated broad-search policy; never a scientific pass")
    args = parser.parse_args()
    run(args.output, core_file=args.core_file, selected_cases=args.cases,
        experimental_full_search=args.experimental_full_search)


if __name__ == "__main__":
    main()
