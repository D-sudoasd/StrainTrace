"""Compare declared, aligned DIC exports without resampling or filling holes.

Run with ``python benchmarks/compare_dic_exports.py --manifest comparison.json
--output new-directory``. See docs/VIC2D_COMPARISON.md for the input contract.
The result measures agreement between two exports, never accuracy/equivalence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree

SCHEMA = "straintrace-paired-dic-comparison-v1"
FLOAT_FIELDS = ("x", "y", "u", "v", "strain_xx", "strain_yy", "strain_xy")
BOOL_FIELDS = ("eligible", "correlation_valid", "strain_valid")
OPTIONAL_FIELDS = ("quality", "invalid_reason", "strain_invalid_reason")
IDENTITY_FIELDS = (
    "reference_image_sha256", "reference_frame", "coordinate_frame",
    "coordinate_unit", "displacement_unit", "displacement_reference",
    "strain_measure", "strain_unit", "shear_convention", "roi_xywh_px",
)
PARAMETER_FIELDS = (
    "subset_size_px", "step_px", "strain_window_points", "shape_function",
    "reference_strategy", "strain_filter", "displacement_smoothing", "quality_policy",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _number(value, context, *, missing=False):
    if missing and (value is None or str(value).strip().lower() in {"", "nan", "na", "null"}):
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a number, got {value!r}") from exc
    if not math.isfinite(result):
        if missing and math.isnan(result):
            return result
        raise ValueError(f"{context}: nonfinite number")
    return result


def _boolean(value, context):
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{context}: validity must be explicit true/false or 1/0, got {value!r}")


def _read_csv(path: Path, columns, *, grid=False):
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    headers = reader.fieldnames or []
    if len(set(headers)) != len(headers):
        raise ValueError(f"{path}: duplicate CSV column names")
    required = ("x", "y", "eligible") if grid else FLOAT_FIELDS + BOOL_FIELDS
    if not isinstance(columns, dict) or any(name not in columns for name in required):
        raise ValueError(f"{path}: explicit column map must include {required}")
    unknown = set(columns) - set(FLOAT_FIELDS + BOOL_FIELDS + OPTIONAL_FIELDS)
    if unknown:
        raise ValueError(f"{path}: unknown canonical column mappings: {sorted(unknown)}")
    if any(not isinstance(value, str) or value not in headers for value in columns.values()):
        raise ValueError(f"{path}: mapped CSV column is missing; column names are case-sensitive")
    if len(set(columns.values())) != len(columns):
        raise ValueError(f"{path}: each canonical field must map to a different CSV column")
    rows = []
    for index, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}:{index}: malformed CSV row")
        record = {}
        for key, column in columns.items():
            context = f"{path}:{index}:{column}"
            if key in BOOL_FIELDS:
                record[key] = _boolean(row[column], context)
            elif key in FLOAT_FIELDS or key == "quality":
                record[key] = _number(row[column], context, missing=key not in {"x", "y"})
            else:
                record[key] = row[column]
        if not grid:
            if record["correlation_valid"] and not record["eligible"]:
                raise ValueError(f"{path}:{index}: correlation-valid point is declared ineligible")
            if record["strain_valid"] and not record["correlation_valid"]:
                raise ValueError(f"{path}:{index}: strain-valid point lacks valid correlation")
            # StrainTrace deliberately keeps the correlation-acceptance flag
            # when a folded subset makes its published displacement unavailable.
            # Retain that distinction; raw candidates are never substituted.
            record["displacement_available"] = record["correlation_valid"] and all(
                math.isfinite(record[key]) for key in ("u", "v"))
            for names, valid in ((('strain_xx', 'strain_yy', 'strain_xy'), 'strain_valid'),):
                if record[valid] and not all(math.isfinite(record[key]) for key in names):
                    raise ValueError(f"{path}:{index}: {valid} is true but its measurements are nonfinite")
            if record["strain_valid"] and not record["displacement_available"]:
                raise ValueError(f"{path}:{index}: strain-valid point lacks finite published displacement")
        rows.append(record)
    return rows, {"path": str(path), "sha256": _sha256(data), "rows": len(rows)}


def _inside(row, roi):
    x, y, width, height = roi
    return x <= row["x"] <= x + width and y <= row["y"] <= y + height


def _coordinates(rows):
    return np.asarray([(row["x"], row["y"]) for row in rows], dtype=float).reshape(-1, 2)


def _unique(rows, tolerance, context):
    if len(rows) > 1 and cKDTree(_coordinates(rows)).query_pairs(tolerance):
        raise ValueError(f"{context}: duplicate or ambiguous coordinates within matching tolerance")


def _align(domain, rows, tolerance):
    """Return unique exact-coordinate matches; tolerance only absorbs rounding."""
    if not rows:
        return [None] * len(domain)
    tree = cKDTree(_coordinates(rows))
    candidates = tree.query_ball_point(_coordinates(domain), tolerance)
    matched = set()
    result = []
    for items in candidates:
        if len(items) > 1 or (items and items[0] in matched):
            raise ValueError("Coordinates do not define a unique one-to-one match")
        if items:
            matched.add(items[0])
        result.append(rows[items[0]] if items else None)
    return result


def _distribution(values):
    data = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not data.size:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": int(data.size), "min": float(data.min()),
            "median": float(np.median(data)), "max": float(data.max())}


def _extremum(rows, component, *, kind="maximum"):
    """Return a signed extremum; compression and negative shear need minima."""
    valid = [row for row in rows if row is not None and row["eligible"] and row["strain_valid"]]
    if not valid:
        return None
    operation = max if kind == "maximum" else min
    value = operation(row[component] for row in valid)
    peaks = [row for row in valid if row[component] == value]
    # Ties are exposed; the first reported location is not a unique peak.
    first = min(peaks, key=lambda row: (row["y"], row["x"]))
    return {"kind": f"signed_{kind}", "value": value, "x": first["x"], "y": first["y"], "tied_point_count": len(peaks)}


def _compare_extrema(left, right, *, kind):
    result = {}
    for key in FLOAT_FIELDS[4:]:
        apeak = _extremum(left, key, kind=kind)
        bpeak = _extremum(right, key, kind=kind)
        result[key] = {"kind": f"signed_{kind}", "straintrace": apeak, "comparison": bpeak,
            "amplitude_difference": apeak["value"] - bpeak["value"] if apeak and bpeak else None,
            "location_distance_px": math.hypot(apeak["x"] - bpeak["x"], apeak["y"] - bpeak["y"]) if apeak and bpeak and apeak["tied_point_count"] == bpeak["tied_point_count"] == 1 else None}
    return result


def _side_summary(rows, source, *, denominator=None):
    present = [row for row in rows if row is not None]
    eligible = [row for row in present if row["eligible"]]
    correlated = [row for row in eligible if row["correlation_valid"]]
    displacement = [row for row in eligible if row["displacement_available"]]
    strain = [row for row in eligible if row["strain_valid"]]
    complete = source["rows_representation"] == "complete_grid"
    n = denominator if denominator is not None else (len(eligible) if complete else None)
    quality = source.get("quality")
    return {
        "present_n": len(present), "eligible_n": len(eligible),
        "excluded_n": len(present) - len(eligible),
        "missing_n": len(rows) - len(present),
        "correlation_valid_n": len(correlated), "strain_valid_n": len(strain),
        "displacement_available_n": len(displacement),
        "accepted_correlation_without_displacement_n": len(correlated) - len(displacement),
        "coverage_denominator_n": n,
        "correlation_valid_fraction": len(correlated) / n if n else None,
        "displacement_available_fraction": len(displacement) / n if n else None,
        "strain_valid_fraction": len(strain) / n if n else None,
        "coverage_basis": "fixed_comparison_grid" if denominator is not None else
                          ("complete_native_eligible_grid" if complete else "unavailable_for_partial_rows"),
        "correlation_failure_reasons": dict(Counter(row.get("invalid_reason") or "unspecified"
                                                    for row in eligible if not row["correlation_valid"])),
        "strain_failure_reasons": dict(Counter(row.get("strain_invalid_reason") or "unspecified"
                                               for row in eligible if not row["strain_valid"])),
        "quality": None if quality is None else {
            "definition": quality,
            "all_eligible": _distribution([row.get("quality", float("nan")) for row in eligible]),
            "valid_correlation": _distribution([row.get("quality", float("nan")) for row in correlated]),
        },
        # Keep the established peak fields as signed maxima and expose minima
        # separately; never silently switch to magnitude-based extrema.
        "strain_peaks": {key: _extremum(present, key) for key in FLOAT_FIELDS[4:]},
        "strain_minima": {key: _extremum(present, key, kind="minimum") for key in FLOAT_FIELDS[4:]},
    }


def _difference(values):
    data = np.asarray(values, dtype=float)
    if not data.size:
        return {"n": 0, "bias": None, "rmse": None, "p95_abs": None, "max_abs": None}
    return {"n": int(data.size), "bias": float(data.mean()),
            "rmse": float(np.sqrt(np.mean(data ** 2))),
            "p95_abs": float(np.percentile(np.abs(data), 95)), "max_abs": float(np.max(np.abs(data)))}


def _status(row):
    if row is None:
        return "missing"
    if not row["eligible"]:
        return "excluded"
    if not row["correlation_valid"]:
        return "correlation_invalid"
    if not row["displacement_available"]:
        return "displacement_unavailable"
    return "valid" if row["strain_valid"] else "strain_invalid"


def _validate(config):
    if config.get("schema") != SCHEMA:
        raise ValueError(f"Expected schema {SCHEMA}")
    sources = config.get("sources", {})
    if set(sources) != {"straintrace", "comparison"}:
        raise ValueError("sources must contain straintrace and comparison")
    for label, source in sources.items():
        metadata = source.get("metadata", {})
        if any(key not in metadata for key in IDENTITY_FIELDS):
            raise ValueError(f"{label}: all identity/coordinate/strain/ROI metadata must be explicit")
        if not _digest(metadata["reference_image_sha256"]) or not isinstance(metadata["reference_frame"], str) or not metadata["reference_frame"]:
            raise ValueError(f"{label}: reference frame and lowercase SHA-256 are required")
        if metadata["coordinate_frame"] != "reference_image_x_right_y_down" or metadata["coordinate_unit"] != "px" or metadata["displacement_unit"] != "px":
            raise ValueError("Export both sides in reference image pixel coordinates, x right and y down")
        if metadata["displacement_reference"] != "total_from_reference":
            raise ValueError("Incremental displacement must be correctly composed to the declared reference before comparison")
        if metadata["strain_measure"] not in {"green_lagrange", "infinitesimal"} or metadata["strain_unit"] != "1" or metadata["shear_convention"] != "tensor":
            raise ValueError("Use matching dimensionless Green-Lagrange or infinitesimal strain and tensor shear")
        if source.get("rows_representation") not in {"complete_grid", "partial_rows"}:
            raise ValueError(f"{label}: declare complete_grid or partial_rows; omitted failures affect coverage")
        parameters = source.get("parameters", {})
        if any(key not in parameters or parameters[key] is None for key in PARAMETER_FIELDS):
            raise ValueError(f"{label}: record subset, step, strain window, shape function, reference/filter and quality parameters")
        if any(isinstance(value, str) and (not value.strip() or value.startswith("REPLACE_WITH")) for value in parameters.values()):
            raise ValueError(f"{label}: replace template placeholders with the actual processing parameters")
        for key in ("subset_size_px", "step_px", "strain_window_points"):
            if _number(parameters[key], f"{label}:{key}") <= 0:
                raise ValueError(f"{label}:{key} must be positive")
        if "quality" in source.get("columns", {}) and not all(source.get("quality", {}).get(key) for key in ("name", "unit", "meaning")):
            raise ValueError(f"{label}: quality column requires its own name, unit and meaning")
        if source.get("quality") and "quality" not in source.get("columns", {}):
            raise ValueError(f"{label}: quality definition requires a mapped quality column")
    left, right = (sources[name]["metadata"] for name in ("straintrace", "comparison"))
    for key in IDENTITY_FIELDS:
        if left[key] != right[key]:
            raise ValueError(f"Incomparable metadata: {key} differs")
    roi = left["roi_xywh_px"]
    if not isinstance(roi, list) or len(roi) != 4 or any(type(value) not in (int, float) or not math.isfinite(value) for value in roi) or min(roi[2:]) <= 0:
        raise ValueError("roi_xywh_px must contain finite x, y and positive width, height")
    tolerance = _number(config.get("coordinate_tolerance_px"), "coordinate_tolerance_px")
    if not 0 <= tolerance <= 1e-3:
        raise ValueError("coordinate_tolerance_px must be in [0, 0.001]; it is only for coordinate rounding")
    frames = config.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must explicitly enumerate all intended stages, including missing exports")
    identifiers = set()
    for frame in frames:
        identity = frame.get("frame_id")
        if not isinstance(identity, str) or not identity or identity in identifiers:
            raise ValueError("Each frame_id must be a nonempty unique string")
        identifiers.add(identity)
        if not isinstance(frame.get("stage"), str) or not frame["stage"] or not _digest(frame.get("image_sha256")):
            raise ValueError(f"{identity}: stage and target image SHA-256 are required")
        for label in sources:
            if label not in frame:
                raise ValueError(f"{identity}: declare {label}: null for a missing export")
            item = frame[label]
            if item is not None and (not isinstance(item, dict) or not isinstance(item.get("csv"), str) or item.get("image_sha256") != frame["image_sha256"]):
                raise ValueError(f"{identity}: {label} export must bind the same target image SHA-256")
    return sources, roi, tolerance


def compare(manifest: Path, output: Path):
    """Write a fresh, atomic comparison directory; all source files stay read-only."""
    manifest = manifest.resolve(strict=True)
    manifest_bytes = manifest.read_bytes()
    config = json.loads(manifest_bytes)
    sources, roi, tolerance = _validate(config)
    base = manifest.parent
    grid_config = config.get("comparison_grid", {})
    if not isinstance(grid_config.get("csv"), str):
        raise ValueError("comparison_grid.csv must identify the fixed reference domain")
    grid, grid_identity = _read_csv((base / grid_config["csv"]).resolve(strict=True), grid_config.get("columns"), grid=True)
    domain = [row for row in grid if row["eligible"] and _inside(row, roi)]
    if not domain:
        raise ValueError("The fixed eligible comparison grid is empty inside the ROI")
    _unique(domain, tolerance, "comparison_grid")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dic-compare-", dir=output.parent))
    report = {
        "schema": SCHEMA, "kind": "agreement_not_ground_truth", "equivalence_claim": False,
        "identity_verification": "Image identities and processing settings are supplied by the manifest; export and grid bytes are SHA-256 recorded.",
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest_bytes)},
        "script_sha256": _sha256(Path(__file__).read_bytes()), "comparison_grid": grid_identity,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "fixed_domain_n": len(domain), "coordinate_tolerance_px": tolerance,
        "roi_membership": {
            "native_roi": "Reported coordinates inside the closed ROI; no boundary expansion.",
            "fixed_domain": "Fixed grid coordinates inside the closed ROI; export coordinates match only within coordinate_tolerance_px, including rounding across an ROI edge.",
        },
        "difference_direction": "straintrace_minus_comparison", "sources": sources,
        "parameter_differences": {key: {label: source["parameters"][key] for label, source in sources.items()}
                                  for key in PARAMETER_FIELDS if sources["straintrace"]["parameters"][key] != sources["comparison"]["parameters"][key]},
        "frames": [],
    }
    try:
        for ordinal, frame in enumerate(config["frames"], 1):
            native, aligned, identities = {}, {}, {}
            for label, source in sources.items():
                item = frame[label]
                if item is None:
                    rows, identity = [], None
                else:
                    rows, identity = _read_csv((base / item["csv"]).resolve(strict=True), source["columns"])
                _unique(rows, tolerance, f"{frame['frame_id']}:{label}")
                native[label] = [row for row in rows if _inside(row, roi)]
                # Membership of the fixed domain was established from the
                # reference grid. Cropping export rows first loses valid edge
                # matches when coordinate rounding falls just outside the ROI.
                # Keep the native ROI strict so unrelated outside peaks cannot
                # enter its extrema through a blanket tolerance expansion.
                aligned[label] = _align(domain, rows, tolerance)
                identities[label] = identity
            pairs = list(zip(aligned["straintrace"], aligned["comparison"]))
            common_correlation = [(a, b) for a, b in pairs if a is not None and b is not None and a["eligible"] and b["eligible"] and a["correlation_valid"] and b["correlation_valid"]]
            common_displacement = [(a, b) for a, b in common_correlation if a["displacement_available"] and b["displacement_available"]]
            common_strain = [(a, b) for a, b in common_displacement if a["strain_valid"] and b["strain_valid"]]
            differences = {key: _difference([a[key] - b[key] for a, b in (common_displacement if key in {"u", "v"} else common_strain)]) for key in FLOAT_FIELDS[2:]}
            vectors = [math.hypot(a["u"] - b["u"], a["v"] - b["v"]) for a, b in common_displacement]
            common_left = [a for a, _ in common_strain]
            common_right = [b for _, b in common_strain]
            common_peaks = _compare_extrema(common_left, common_right, kind="maximum")
            common_minima = _compare_extrema(common_left, common_right, kind="minimum")
            point_file = f"frame_{ordinal:04d}_points.csv"
            with (staging / point_file).open("w", newline="", encoding="utf-8") as stream:
                fieldnames = ["x", "y", "straintrace_status", "comparison_status"]
                for label in sources:
                    fieldnames += [f"{label}_{key}" for key in FLOAT_FIELDS[2:] + ("quality", "invalid_reason", "strain_invalid_reason")]
                fieldnames += [f"difference_{key}" for key in FLOAT_FIELDS[2:]]
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for point, (a, b) in zip(domain, pairs):
                    record = {"x": point["x"], "y": point["y"], "straintrace_status": _status(a), "comparison_status": _status(b)}
                    for label, row in (("straintrace", a), ("comparison", b)):
                        for key in FLOAT_FIELDS[2:]:
                            gate = "displacement_available" if key in {"u", "v"} else "strain_valid"
                            record[f"{label}_{key}"] = row[key] if row is not None and row[gate] else None
                        for key in ("quality", "invalid_reason", "strain_invalid_reason"):
                            value = row.get(key) if row else None
                            record[f"{label}_{key}"] = value if not isinstance(value, float) or math.isfinite(value) else None
                    for key in FLOAT_FIELDS[2:]:
                        gate = "displacement_available" if key in {"u", "v"} else "strain_valid"
                        record[f"difference_{key}"] = a[key] - b[key] if a is not None and b is not None and a[gate] and b[gate] else None
                    writer.writerow(record)
            report["frames"].append({
                "frame_id": frame["frame_id"], "stage": frame["stage"], "image_sha256": frame["image_sha256"],
                "exports": identities, "status": "no_common_displacement" if not common_displacement else "no_common_strain" if not common_strain else "compared",
                "native_roi": {label: _side_summary(native[label], source) for label, source in sources.items()},
                "native_roi_extrema_comparison": {
                    "support": "separate_native_valid_sets",
                    "signed_maximum": _compare_extrema(native["straintrace"], native["comparison"], kind="maximum"),
                    "signed_minimum": _compare_extrema(native["straintrace"], native["comparison"], kind="minimum"),
                },
                "fixed_domain": {label: _side_summary(aligned[label], source, denominator=len(domain)) for label, source in sources.items()},
                "common_correlation_n": len(common_correlation), "common_strain_n": len(common_strain),
                "common_displacement_n": len(common_displacement),
                "common_correlation_fraction": len(common_correlation) / len(domain),
                "common_displacement_fraction": len(common_displacement) / len(domain),
                "common_strain_fraction": len(common_strain) / len(domain),
                "component_differences": differences,
                "displacement_vector_rmse_px": math.sqrt(sum(v*v for v in vectors)/len(vectors)) if vectors else None,
                "common_domain_peaks": common_peaks,
                "common_domain_minima": common_minima,
                "point_table": {"path": point_file, "sha256": _sha256((staging / point_file).read_bytes())},
            })
        count_strain = sum(frame["common_strain_n"] > 0 for frame in report["frames"])
        count_displacement = sum(frame["common_displacement_n"] > 0 for frame in report["frames"])
        report["comparison_status"] = "compared_all_frames" if count_strain == len(report["frames"]) else "partial_comparison" if count_displacement else "no_common_measurements"
        report["frames_with_common_strain"] = count_strain
        (staging / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing output is never overwritten")
    args = parser.parse_args()
    report = compare(args.manifest, args.output)
    print(json.dumps({"report": str(args.output / "report.json"), "frames": len(report["frames"]),
                      "comparison_status": report["comparison_status"],
                      "frames_with_common_strain": report["frames_with_common_strain"],
                      "equivalence_claim": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
