"""Named spatial summaries and conservative image-space ROI recommendations.

Regions select original reference POIs; they never rerun DIC, interpolate a
hole, or change a displacement/strain value. Image texture is evidence for
correlation suitability, not evidence that a pixel belongs to the specimen.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from . import fields


def _number(value, name: str, *, lower: float, upper: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < lower or (upper is not None and result > upper):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def normalize_regions(regions: Sequence[Mapping] | None) -> list[dict]:
    """Validate independent named rectangles in inclusive coordinate bounds.

    Bounds are ``[xmin, xmax, ymin, ymax]``. They deliberately differ from
    solver/gauge ``*_roi_px`` rectangles, which use ``[x, y, width, height]``.
    """
    if regions is None:
        return []
    if not isinstance(regions, (list, tuple)):
        raise ValueError("regions must be a list of objects")
    allowed = {"name", "bounds", "coordinate_system", "min_valid_fraction", "min_valid_pois", "min_zncc"}
    result, names = [], set()
    for index, region in enumerate(regions):
        if not isinstance(region, Mapping):
            raise ValueError(f"regions[{index}] must be an object")
        unknown = set(region) - allowed
        if unknown:
            raise ValueError(f"unknown region field: {sorted(unknown)[0]}")
        name = region.get("name")
        if not isinstance(name, str) or not name.strip() or name.strip().casefold() in names:
            raise ValueError("region names must be nonempty and unique")
        names.add(name.strip().casefold())
        bounds = region.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            raise ValueError("region bounds must be [xmin, xmax, ymin, ymax]")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) for value in bounds):
            raise ValueError("region bounds must contain numbers, not booleans or strings")
        xmin, xmax, ymin, ymax = fields._validate_roi(bounds)
        if xmin == xmax or ymin == ymax:
            raise ValueError("region bounds must have positive width and height")
        system = region.get("coordinate_system", "image")
        if system not in {"image", "calibrated"}:
            raise ValueError("region coordinate_system must be image or calibrated")
        count = region.get("min_valid_pois", 3)
        if type(count) is not int or count < 1:
            raise ValueError("min_valid_pois must be a positive integer")
        item = {
            "name": name.strip(), "bounds": [xmin, xmax, ymin, ymax], "coordinate_system": system,
            "min_valid_fraction": _number(region.get("min_valid_fraction", 0.5), "min_valid_fraction", lower=0, upper=1),
            "min_valid_pois": count,
        }
        if region.get("min_zncc") is not None:
            item["min_zncc"] = _number(region["min_zncc"], "min_zncc", lower=-1, upper=1)
        result.append(item)
    return result


def _support_masks(field: Mapping) -> tuple[np.ndarray, np.ndarray]:
    shape = fields._shape(field)
    valid = fields._array(field, "valid", required=False, dtype=bool)
    valid = np.zeros(shape, dtype=bool) if valid is None else valid.copy()
    displacement = valid.copy()
    for key in ("u", "v"):
        array = fields._array(field, key + "_raw" if key + "_raw" in field else key, required=False)
        displacement &= False if array is None else np.isfinite(array)
    strain = fields._array(field, "strain_valid", required=False, dtype=bool)
    strain = np.zeros(shape, dtype=bool) if strain is None else valid & strain
    tensor_keys = ("Exx", "Eyy", "Exy") if all(key in field for key in ("Exx", "Eyy", "Exy")) else ("exx", "eyy", "exy")
    for key in tensor_keys:
        array = fields._array(field, key, required=False)
        strain &= False if array is None else np.isfinite(array)
    if fields._has_jacobian_support(field) and len(shape) == 2 and min(shape) >= 2:
        strain &= ~fields.nonpositive_jacobian_poi_mask(field)
    return displacement, strain


def _field_bounds(field: Mapping, system: str, x: np.ndarray, y: np.ndarray) -> list[float] | None:
    roi = field.get("roi")
    if roi is not None and len(roi) == 4 and np.isfinite(roi).all() and roi[2] > 0 and roi[3] > 0:
        left, top, width, height = map(float, roi)
        xb, yb = np.array([left, left + width]), np.array([top, top + height])
        if system == "calibrated":
            calibration = field["calibration"]
            scale = calibration["mm_per_px"]
            xb = calibration["origin_mm"][0] + calibration["axis_sign"][0] * (xb - calibration["origin_px"][0]) * scale
            yb = calibration["origin_mm"][1] + calibration["axis_sign"][1] * (yb - calibration["origin_px"][1]) * scale
        return [float(xb.min()), float(xb.max()), float(yb.min()), float(yb.max())]
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return None
    # Without the original solver rectangle, only the recorded POI extent is
    # known. Do not assume that the computed field extends outside that extent.
    return [float(x[finite].min()), float(x[finite].max()), float(y[finite].min()), float(y[finite].max())]


def summarize_regions(
    field: Mapping,
    regions: Sequence[Mapping],
    components: Sequence[str] = ("Exx", "Eyy", "Exy"),
    *,
    quality_ok: bool = True,
) -> list[dict]:
    """Return one row per named region/component from the same computed field.

    Coverage is a fraction of original POI centers, not a continuous area
    fraction. Displacement and strain support are reported separately. A
    failed gate leaves statistics null while preserving support evidence.
    ``min_zncc`` screens existing POI centers; it does not recompute the
    solver's neighboring strain-fit support with a different threshold.
    Partial out-of-field rectangles are rejected, rather than silently
    reporting the smaller overlapping region as the requested measurement.
    """
    definitions = normalize_regions(regions)
    requested = [components] if isinstance(components, str) else list(components)
    if not requested or any(not isinstance(key, str) or not key for key in requested) or len(set(requested)) != len(requested):
        raise ValueError("components must be unique nonempty field component names")
    if not definitions:
        return []
    if not any(key in field for key in ("X", "x")):
        field = {**field, "x": np.array([], dtype=float), "y": np.array([], dtype=float)}
    calibrated = None
    if field.get("calibration") is not None:
        calibrated = {**field, **fields.calibrate_field(field, field["calibration"])}
    working = calibrated or field
    displacement, strain = _support_masks(working)
    values_by_component = {}
    for component in requested:
        try:
            values = fields.component_values(working, component)
        except ValueError:
            if component not in fields._STRAIN_COMPONENTS and component not in {"u", "v", "u_raw", "v_raw", "u_mm", "v_mm", "zncc"}:
                raise
            values = None
        if values is not None:
            mask = np.isfinite(values)
            if component in {"u", "v", "u_raw", "v_raw", "u_mm", "v_mm"}:
                mask &= displacement
            elif component in fields._STRAIN_COMPONENTS:
                mask &= strain
            elif component == "zncc":
                valid = fields._array(working, "valid", required=False, dtype=bool)
                mask &= False if valid is None else valid
            values_by_component[component] = (values, mask)
        else:
            values_by_component[component] = (None, None)
    coordinates = {}
    result = []
    for region in definitions:
        system = region["coordinate_system"]
        common_reasons = []
        inside = np.zeros(fields._shape(working), dtype=bool)
        support_bounds = None
        if system == "calibrated" and calibrated is None:
            common_reasons.append("calibration_unavailable")
        else:
            if system not in coordinates:
                coordinates[system] = fields._coordinates(working, system)
            x, y = coordinates[system]
            xmin, xmax, ymin, ymax = region["bounds"]
            inside = np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            support_bounds = _field_bounds(working, system, x, y)
            if support_bounds is not None:
                sxmin, sxmax, symin, symax = support_bounds
                tolerance = 1e-12 * max(1.0, *(abs(value) for value in support_bounds))
                if xmin < sxmin - tolerance or xmax > sxmax + tolerance or ymin < symin - tolerance or ymax > symax + tolerance:
                    common_reasons.append("outside_field_support")
        n_total = int(inside.sum())
        if n_total == 0:
            common_reasons.append("no_poi_support")
        quality_mask = np.ones(inside.shape, dtype=bool)
        if "min_zncc" in region:
            correlation = fields._array(working, "zncc", required=False)
            if correlation is None:
                quality_mask[:] = False
                common_reasons.append("correlation_quality_unavailable")
            else:
                quality_mask &= np.isfinite(correlation) & (correlation >= region["min_zncc"])
        n_displacement = int((inside & displacement & quality_mask).sum())
        n_strain = int((inside & strain & quality_mask).sum())
        if not quality_ok:
            common_reasons.append("frame_quality_rejected")
        for component, (values, valid) in values_by_component.items():
            reasons = list(common_reasons)
            selected = inside & quality_mask & valid if valid is not None else np.zeros(inside.shape, dtype=bool)
            n_valid = int(selected.sum())
            fraction = n_valid / n_total if n_total else None
            if values is None:
                reasons.append("component_unavailable")
            if n_valid < region["min_valid_pois"]:
                reasons.append("insufficient_valid_pois")
            if fraction is not None and fraction < region["min_valid_fraction"]:
                reasons.append("insufficient_valid_fraction")
            samples = values[selected] if not reasons else []
            result.append({
                "region_name": region["name"], "component": component, "bounds": list(region["bounds"]),
                "coordinate_system": system, "coordinate_unit": "px" if system == "image" else "mm",
                "unit": fields.component_unit(component), "n_poi_total": n_total, "n_poi_valid": n_valid,
                "valid_fraction": fraction, "n_displacement_valid": n_displacement, "n_strain_valid": n_strain,
                "displacement_valid_fraction": n_displacement / n_total if n_total else None,
                "strain_valid_fraction": n_strain / n_total if n_total else None,
                "min_valid_pois": region["min_valid_pois"], "min_valid_fraction": region["min_valid_fraction"],
                "min_zncc": region.get("min_zncc"), "status": "valid" if not reasons else "unavailable",
                "reasons": reasons, "quality_ok": bool(quality_ok), "field_support_bounds": support_bounds,
                "calibration_status": "available" if calibrated is not None else "unavailable",
                **fields._summarize_samples(samples, component),
                "source": "original_poi", "interpolation": "none",
                "coverage_interpretation": "original_poi_count_fraction_not_continuous_area_fraction",
            })
    return result


def recommend_regions(
    image: np.ndarray, subset_size: int, step: int, *,
    specimen_mask: np.ndarray | None = None, max_regions: int = 3,
) -> dict:
    """Rank a bounded set of sizes/positions using existing texture checks.

    Output candidates require review. Neither texture nor a supplied mask
    establishes physical specimen identity or the experimental loading axis.
    """
    import cv2
    from scipy import ndimage
    import ezdic_core as core

    raw = np.asarray(image)
    if raw.ndim != 2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError("image must be a finite nonempty grayscale array")
    if type(subset_size) is not int or subset_size < 9 or subset_size % 2 != 1:
        raise ValueError("subset_size must be an odd integer >= 9")
    if type(step) is not int or step < 1 or type(max_regions) is not int or max_regions < 1:
        raise ValueError("step and max_regions must be positive integers")
    mask = None
    support = None
    if specimen_mask is not None:
        mask = np.asarray(specimen_mask)
        if mask.shape != raw.shape or mask.dtype != np.bool_:
            raise ValueError("specimen_mask must be a boolean array matching the image")
        support = ndimage.minimum_filter(mask.astype(np.uint8), size=subset_size, mode="constant", cval=0).astype(bool)
    normalization = core.compute_reference_normalization(raw)
    image8 = core.normalize_with_bounds(raw, normalization["reference_bounds"])
    height, width = raw.shape
    candidates, rejected, seen = [], [], set()
    for fraction in (0.35, 0.55, 0.80):
        rw = min(width, max(subset_size + 2 * step, int(round(width * fraction))))
        rh = min(height, max(subset_size + 2 * step, int(round(height * fraction))))
        for fy in (0.25, 0.50, 0.75):
            for fx in (0.25, 0.50, 0.75):
                left = max(0, min(width - rw, int(round(width * fx - rw / 2))))
                top = max(0, min(height - rh, int(round(height * fy - rh / 2))))
                rect = (left, top, rw, rh)
                if rect in seen:
                    continue
                seen.add(rect)
                X, Y = core.build_poi_grid(rect, subset_size, step, raw.shape)
                if not core.poi_grid_is_usable(X, Y):
                    rejected.append({"roi_px": list(rect), "reason": "insufficient_grid_support"})
                    continue
                eligible = np.ones(X.shape, dtype=bool) if support is None else support[Y.astype(int), X.astype(int)]
                eligible_fraction = float(eligible.mean())
                if int(eligible.sum()) < 9 or eligible_fraction < 0.5:
                    rejected.append({"roi_px": list(rect), "reason": "insufficient_mask_support"})
                    continue
                patch = image8[top:top + rh, left:left + rw]
                scale = min(1.0, 256.0 / max(patch.shape))
                if scale < 1:
                    patch = cv2.resize(patch, (max(2, int(round(rw * scale))), max(2, int(round(rh * scale)))), interpolation=cv2.INTER_AREA)
                metrics = core.roi_texture_metrics(patch, (0, 0, patch.shape[1], patch.shape[0]))
                failure = core.texture_failure_code(metrics, 8.0, 25.0, 0.20)
                if failure:
                    rejected.append({"roi_px": list(rect), "reason": failure})
                    continue
                score = float(metrics["gradient_energy"] * metrics["structure_tensor_ratio"] * eligible_fraction)
                candidates.append({
                    "bounds": [left, left + rw, top, top + rh], "roi_px": list(rect), "coordinate_system": "image",
                    "size_fraction": fraction, "eligible_poi_count": int(eligible.sum()), "total_poi_count": int(X.size),
                    "eligible_fraction": eligible_fraction, "texture_metrics": metrics, "texture_analysis_scale": scale,
                    "score": score, "geometry_status": "candidate_unconfirmed_physical_specimen",
                    "specimen_geometry_confirmed": False, "loading_axis": None,
                })
    # Keep different spatial resolutions when available, then add other high
    # quality positions. No result is silently promoted to an approved ROI.
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected = []
    for fraction in (0.35, 0.55, 0.80):
        match = next((item for item in ranked if item["size_fraction"] == fraction), None)
        if match is not None and len(selected) < max_regions:
            selected.append(match)
    selected.extend(item for item in ranked if item not in selected and len(selected) < max_regions)
    return {
        "schema": "straintrace.region-recommendations/v1", "method": "multiscale_grid_existing_texture_checks",
        "status": "candidates_require_review" if selected else "no_supported_candidate",
        "specimen_geometry_confirmed": False, "loading_axis": None,
        "mask_source": "supplied_boolean_mask" if mask is not None else "not_provided",
        "candidates": [{"name": f"candidate_{index}", **item} for index, item in enumerate(selected, 1)],
        "rejected": rejected, "evaluated_rectangle_count": len(seen),
        "interpretation": "texture_suitability_candidates_not_specimen_segmentation_or_experimental_acceptance",
    }
