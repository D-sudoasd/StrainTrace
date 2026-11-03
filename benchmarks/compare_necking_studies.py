"""Compare two frozen-core necking studies on identical images and POIs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_points(report_path, report):
    artifact = report["pointwise_csv"]
    name = artifact["filename"]
    if Path(name).name != name:
        raise ValueError("pointwise artifact must be adjacent to its report")
    path = report_path.parent/name
    if digest(path) != artifact["sha256"]:
        raise ValueError("pointwise CSV content differs from its report")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != artifact["row_count"]:
        raise ValueError("pointwise row count differs from its report")
    result = {}
    for row in rows:
        points = result.setdefault(row["case"], {})
        position = (float(row["x_px"]), float(row["y_px"]))
        if position in points:
            raise ValueError("duplicate POI in pointwise CSV")
        points[position] = row
    return result


def number(value):
    return float(value) if value else float("nan")


def displacement_error(row):
    return math.hypot(number(row["u_px"])-number(row["true_u_px"]),
                      number(row["v_px"])-number(row["true_v_px"]))


def stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {"count": int(finite.size),
            "rmse": float(np.sqrt(np.mean(finite**2))) if finite.size else None,
            "max_abs": float(np.max(np.abs(finite))) if finite.size else None}


def _parameter_difference(before, after, *, allow_experimental_policy_difference=False):
    """Permit only an explicitly requested change in the experimental flag."""
    if not allow_experimental_policy_difference:
        if before != after:
            raise ValueError("solver_parameters differ; experimental policy changes require explicit opt-in")
        return {}
    a, b = dict(before), dict(after)
    old = a.pop("experimental_full_search", False)
    new = b.pop("experimental_full_search", False)
    if type(old) is not bool or type(new) is not bool:
        raise ValueError("experimental_full_search must be a Boolean")
    if a != b:
        raise ValueError("solver_parameters differ beyond experimental_full_search")
    return {"experimental_full_search": {"before": old, "after": new}} if old != new else {}


def compare(before_path, after_path, output, *, allow_experimental_policy_difference=False):
    before_path, after_path, output = map(Path, (before_path, after_path, output))
    if output.exists():
        raise FileExistsError("Use a new comparison output")
    a = json.loads(before_path.read_text(encoding="utf-8"))
    b = json.loads(after_path.read_text(encoding="utf-8"))
    if a.get("schema") != "straintrace-necking-study-v1":
        raise ValueError("Unsupported study schema")
    for key in ("schema", "benchmark_sha256", "generator_sha256", "texture_source_sha256",
                "environment", "deformation_reference", "strain_definition", "diagnostic_error_cutoff_px"):
        if a[key] != b[key]:
            raise ValueError(f"Study comparison differs in {key}")
    if set(a["cases"]) != set(b["cases"]):
        raise ValueError("Different case sets")
    parameter_differences = {}
    for name, old in a["cases"].items():
        new = b["cases"][name]
        for key in ("input_sha256", "deformation", "roi_xywh_px", "grid_sha256",
                    "local_unrelated_texture_mix", "target_blur_sigma_px",
                    "independent_gaussian_image_noise_sigma_gray"):
            if old[key] != new[key]:
                raise ValueError(f"Case {name} differs in {key}")
        change = _parameter_difference(old["solver_parameters"], new["solver_parameters"],
                                       allow_experimental_policy_difference=allow_experimental_policy_difference)
        if change:
            parameter_differences[name] = change
        for report, case in ((a, old), (b, new)):
            if case["solver_parameters"].get("experimental_full_search", False) != report.get("search_policy", {}).get("experimental_full_search", False):
                raise ValueError("Case solver parameters disagree with the declared search policy")
    ap, bp = load_points(before_path, a), load_points(after_path, b)
    result = {
        "schema": "straintrace-necking-study-comparison-v1",
        "before_report_sha256": digest(before_path), "after_report_sha256": digest(after_path),
        "before_core_sha256": a["core_sha256"], "after_core_sha256": b["core_sha256"],
        "environment": a["environment"], "cases": {},
        "comparison_kind": "experimental_policy_comparison" if parameter_differences else "matched_solver_comparison",
        "permitted_parameter_differences": parameter_differences,
        "experimental_policy_difference_explicitly_allowed": bool(allow_experimental_policy_difference),
        "search_policy_before_after": [a.get("search_policy"), b.get("search_policy")],
        "interpretation": "Errors use analytic truth; accepted-only comparisons are accompanied by gained/lost points and local coverage. The 0.5 px error cutoff is descriptive, not a relaxed quality gate.",
    }
    for name, old in a["cases"].items():
        new = b["cases"][name]
        pa, pb = ap[name], bp[name]
        if pa.keys() != pb.keys():
            raise ValueError("POI coordinates differ")
        positions = list(pa)
        for position in positions:
            for key in ("eligible", "true_u_px", "true_v_px", "true_Eyy"):
                if pa[position][key] != pb[position][key]:
                    raise ValueError(f"POI truth or eligibility differs for {name}")
        common = [p for p in positions if pa[p]["correlation_valid"] == pb[p]["correlation_valid"] == "1"]
        gained = [p for p in positions if pa[p]["correlation_valid"] == "0" and pb[p]["correlation_valid"] == "1"]
        lost = [p for p in positions if pa[p]["correlation_valid"] == "1" and pb[p]["correlation_valid"] == "0"]
        common_strain = [p for p in positions if pa[p]["strain_valid"] == pb[p]["strain_valid"] == "1"]
        lost_errors = [displacement_error(pa[p]) for p in lost]
        gained_errors = [displacement_error(pb[p]) for p in gained]
        cutoff = a["diagnostic_error_cutoff_px"]
        record = {
            "correlation_valid_before_after": [old["correlation_valid_count"], new["correlation_valid_count"]],
            "strain_valid_before_after": [old["strain_valid_count"], new["strain_valid_count"]],
            "common_valid_count": len(common), "common_strain_valid_count": len(common_strain),
            "gained_valid_count": len(gained), "lost_valid_count": len(lost),
            "newly_rejected_old_error_le_0_5_px": sum(error <= cutoff for error in lost_errors),
            "newly_rejected_old_error_gt_0_5_px": sum(error > cutoff for error in lost_errors),
            "newly_accepted_error_le_0_5_px": sum(error <= cutoff for error in gained_errors),
            "newly_accepted_error_gt_0_5_px": sum(error > cutoff for error in gained_errors),
            "gained_valid_xy_px": gained, "lost_valid_xy_px": lost,
            "common_displacement_error_before_px": stats([displacement_error(pa[p]) for p in common]),
            "common_displacement_error_after_px": stats([displacement_error(pb[p]) for p in common]),
            "common_displacement_change_px": stats([math.hypot(number(pb[p]["u_px"])-number(pa[p]["u_px"]),
                                                                number(pb[p]["v_px"])-number(pa[p]["v_px"])) for p in common]),
            "common_Eyy_change": stats([number(pb[p]["Eyy"])-number(pa[p]["Eyy"]) for p in common_strain]),
            "neck_strain_coverage_before_after": [old["regions"]["neck_band"]["strain_coverage"], new["regions"]["neck_band"]["strain_coverage"]],
            "centerline_peak_before_after": [old["centerline_estimated_peak"], new["centerline_estimated_peak"]],
            "accepted_error_gt_0_5_px_before_after": [old["accepted_vector_error_over_0_5_px_count"], new["accepted_vector_error_over_0_5_px_count"]],
            "invalid_reasons_before_after": [old["invalid_reason_counts"], new["invalid_reason_counts"]],
            "scientific_ok_before_after": [old["quality"]["scientific_ok"], new["quality"]["scientific_ok"]],
            "scientific_reasons_before_after": [old["quality"]["scientific_reasons"], new["quality"]["scientific_reasons"]],
        }
        result["cases"][name] = record
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-experimental-policy-difference", action="store_true",
                        help="Explicit policy study: permit only experimental_full_search to differ")
    args = parser.parse_args()
    compare(args.before, args.after, args.output,
            allow_experimental_policy_difference=args.allow_experimental_policy_difference)


if __name__ == "__main__":
    main()
