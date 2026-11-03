"""Paired known-truth search and copied-speckle stress comparison.

Example: python -m benchmarks.necking_candidate_validation --baseline-core
<frozen-main-ezdic_core.py> --output <new-report.json>
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
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ezdic_core as current
from benchmarks.necking_search_validation import (
    ROI, SETTINGS, displacement_truth, make_case,
)


def load_core(path):
    spec = importlib.util.spec_from_file_location("candidate_baseline_core", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load baseline core")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def copied_patch(noise):
    rng = np.random.default_rng(3)
    reference = gaussian_filter(rng.normal(size=(130, 130)), 1.2).astype("float32")
    target = np.roll(reference, 5, axis=1).copy()
    target[50:71, 55:76] += rng.normal(size=(21, 21)).astype("float32")*noise
    target[67:88, 72:93] = reference[50:71, 50:71]
    return reference, target


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "rmse_px": None, "max_px": None, "over_0_5_px_count": 0}
    return {"count": int(values.size), "rmse_px": float(np.sqrt(np.mean(values**2))),
            "max_px": float(values.max()), "over_0_5_px_count": int((values>.5).sum())}


def field_metrics(field, truth_u, truth_v, center=None):
    valid = np.asarray(field["valid"], dtype=bool)
    error = np.hypot(field["u_raw"]-truth_u, field["v_raw"]-truth_v)
    candidate_u = field.get("u_candidate", field["u_raw"])
    candidate_v = field.get("v_candidate", field["v_raw"])
    candidate_error = np.hypot(candidate_u-truth_u, candidate_v-truth_v)
    reasons, counts = np.unique(np.asarray(field["invalid_reason"], dtype=str), return_counts=True)
    metrics = {"point_count": int(valid.size), "accepted_count": int(valid.sum()),
               "accepted_error": stats(error[valid]),
               "finite_rejected_candidate_error": stats(candidate_error[~valid]),
               "rejected_candidate_unknown_count": int((~valid & ~np.isfinite(candidate_error)).sum()),
               "scientific_ok": bool(field["quality_summary"]["scientific_ok"]),
               "invalid_reasons": {str(k): int(v) for k,v in zip(reasons,counts) if k}}
    if center is not None:
        def number(values):
            value = values[center]
            return float(value) if np.isfinite(value) else None
        metrics["center"] = {"valid": bool(valid[center]),
            "candidate_u": number(candidate_u), "candidate_v": number(candidate_v),
            "candidate_error_px": number(candidate_error),
            "candidate_zncc": number(field.get("zncc_candidate", field["zncc"])),
            "competing_status": str(field.get("competing_peak_status", ["UNAVAILABLE"]*valid.size)[center]),
            "competing_u": number(field.get("competing_peak_refined_u", np.full(valid.size,np.nan))),
            "competing_v": number(field.get("competing_peak_refined_v", np.full(valid.size,np.nan))),
            "competing_zncc": number(field.get("competing_peak_refined_zncc", np.full(valid.size,np.nan)))}
    return metrics, error, valid


def compare(baseline, *, experimental_full_search=False):
    rows = []
    cases = []
    for name in ("uniform_affine", "localized_shear_band"):
        reference, target = make_case(name)
        cases.append((name, reference, target, ROI, SETTINGS, None))
    for noise in (.06, .10, .20, .35):
        reference, target = copied_patch(noise)
        for label, seed in (("correct_seed",(5,0)), ("wrong_seed",(22,17)), ("no_seed",None)):
            transform = None if seed is None else {
                "matrix": np.array([[1.,0.,seed[0]],[0.,1.,seed[1]]]),
                "median_reprojection_error_px": .1}
            settings = dict(subset_size=21, step=30, search_radius=30, strain_window=5,
                            max_iter=40, zncc_min=.75, reject_nonconverged=True,
                            outlier_threshold_px=1., initialization="local", _global_initial_transform=transform)
            cases.append((f"copied_noise_{noise:.2f}_{label}", reference, target, (20,20,81,81), settings, 4))
    for name, reference, target, roi, settings, center in cases:
        before = baseline.run_2d_dic(reference, target, roi, **settings)
        default = current.run_2d_dic(reference, target, roi, **settings)
        after = (current.run_2d_dic(reference, target, roi, **settings, experimental_full_search=True)
                 if experimental_full_search else default)
        np.testing.assert_array_equal(before["x"], after["x"])
        np.testing.assert_array_equal(before["y"], after["y"])
        truth_u, truth_v = ((np.full_like(after["x"],5.), np.zeros_like(after["y"])) if center is not None
                            else displacement_truth(name, after["x"], after["y"]))
        bm, be, bv = field_metrics(before, truth_u, truth_v, center)
        dm, de, dv = field_metrics(default, truth_u, truth_v, center)
        am, ae, av = field_metrics(after, truth_u, truth_v, center)
        common = bv & av
        rows.append({"case": name, "input_sha256": hashlib.sha256(reference.tobytes()+target.tobytes()).hexdigest(),
            "roi": list(roi), "settings": current.canonicalize_json(settings),
            "after_experimental_full_search": bool(experimental_full_search),
            "before": bm, "after": am,
            "current_default": dm, "experimental": am if experimental_full_search else None,
            "default_changed_validity_count": int((bv != dv).sum()),
            "default_common_before_error": stats(be[bv & dv]),
            "default_common_after_error": stats(de[bv & dv]),
            "common_accepted_count": int(common.sum()),
            "common_before_error": stats(be[common]), "common_after_error": stats(ae[common]),
            "newly_accepted_error": stats(ae[~bv & av]),
            "newly_rejected_previous_inliers_count": int((bv & ~av & (be<=.5)).sum()),
            "newly_rejected_previous_error_over_0_5_px_count": int((bv & ~av & (be>.5)).sum())})
    timings = []
    for name in ("uniform_affine", "localized_shear_band"):
        reference, target = make_case(name)
        samples = {"before": [], "current_default": [], "after": []}
        for iteration in range(4):
            forward = (("before",baseline),("current_default",current),("after",current))
            order = forward if iteration%2 == 0 else tuple(reversed(forward))
            for label, core in order:
                start = time.perf_counter()
                options = {"experimental_full_search": experimental_full_search} if label == "after" else {}
                core.run_2d_dic(reference,target,ROI,**SETTINGS,**options)
                samples[label].append(time.perf_counter()-start)
        timings.append({"case":name,"elapsed_seconds":samples,
                        "median_default_over_before":float(np.median(samples["current_default"])/np.median(samples["before"])),
                        "median_after_over_before":float(np.median(samples["after"])/np.median(samples["before"]))})
    return {"report_version":"necking-candidate-paired-v3",
        "after_experimental_full_search": bool(experimental_full_search), "environment":{
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"opencv":cv2.__version__},
        "source_sha256":{label:hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
                         for label,core in (("before",baseline),("after",current))},
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations":["The 0.5 px cutoff classifies this benchmark's errors; it is not an experimental uncertainty or universal accuracy requirement.",
                       "Copied patches deliberately violate texture conservation while retaining known nominal displacement.",
                       "A strongest-secondary-peak check can miss alternatives outside its domain, below its preselection threshold, or beyond the one retained peak.",
                       "Loss of all evidence at the true match can leave a high-correlation wrong match accepted; inspect reported misses.",
                       "Full-radius search is explicitly experimental and cannot pass scientific_ok, even with full coverage; default workflows retain the established search policy.",
                       "Timings are four alternating samples on a shared machine, not a controlled production performance claim.",
                       "No VIC-2D paired experimental data were used."],
        "cases":rows,"timing":timings}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-core",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--experimental-full-search", action="store_true",
                        help="Explicitly evaluate the unvalidated full-radius/secondary-peak research mode")
    args=parser.parse_args()
    result=compare(load_core(args.baseline_core.resolve()), experimental_full_search=args.experimental_full_search)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
