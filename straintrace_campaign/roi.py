"""Conservative ROI proposal and review helpers for campaign analyses.

The module proposes image-space geometry only. A proposal is not permission to
run: approval binds every rectangle and endpoint to the exact reference bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from scipy import ndimage

import ezdic_core as core
from .storage import proposal_hash as _shared_proposal_hash

_PROPOSAL_VERSION = "campaign_roi_v1"
_GEOMETRY_KEYS = ("reference_frame", "reference_path", "reference_sha256", "image_shape", "free_region", "field_roi", "exclusions", "groups")
_DEFAULT_GAUGE_COUNT = 5
_GAUGE_FRACTIONS = (0.18, 0.34, 0.50, 0.66, 0.82)
_TEXTURE_THRESHOLDS = {
    "min_texture_std": 8.0,
    "min_texture_contrast": 25.0,
    "max_saturated_frac": 0.20,
    "min_structure_ratio": core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
    "max_directional_coherence": core.DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE,
    "min_periodicity_score": core.DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE,
}


def _strict_json(value: Any) -> Any:
    """Convert NumPy values and reject non-finite proposal data."""
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(_strict_json(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_records(acquisition: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = acquisition.get("image_frames")
    if not isinstance(frames, list) or not frames:
        raise core.CoreError("NO_INPUT_IMAGES", {"message": "acquisition.image_frames is empty"})
    records = []
    seen = set()
    for item in frames:
        if not isinstance(item, Mapping):
            raise core.CoreError("INVALID_IMAGE_FRAME", {"message": "image frame records must be objects"})
        try:
            frame = int(item["frame"])
            native = int(item.get("native_frame", frame))
            path = Path(item["path"]).expanduser().resolve()
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise core.CoreError("INVALID_IMAGE_FRAME", {"record": dict(item), "message": str(exc)}) from exc
        if frame in seen:
            raise core.CoreError("DUPLICATE_IMAGE_FRAME", {"frame": frame})
        seen.add(frame)
        records.append({**dict(item), "frame": frame, "native_frame": native, "path": path})
    # Frame identity, never directory enumeration position, defines sequence order.
    return sorted(records, key=lambda record: (record["frame"], record["native_frame"], str(record["path"]).casefold()))


def _read_reference(acquisition: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, str]:
    frames = _frame_records(acquisition)
    existing_roi = acquisition.get("roi") if isinstance(acquisition.get("roi"), Mapping) else {}
    requested = existing_roi.get("reference_frame")
    if requested is not None:
        candidates = [item for item in frames if item["frame"] == int(requested)]
        if not candidates:
            raise core.CoreError("REFERENCE_FRAME_NOT_FOUND", {"reference_frame": requested})
    else:
        candidates = frames
    errors = []
    for item in candidates:
        try:
            raw = core.read_gray_image(item["path"])
            return item, raw, _sha256_file(item["path"])
        except Exception as exc:  # a corrupt frame may be skipped only for choosing the reference
            errors.append({"frame": item["frame"], "path": str(item["path"]), "error": str(exc)})
            if requested is not None:
                break
    raise core.CoreError("REFERENCE_IMAGE_UNAVAILABLE", {"errors": errors})


def _normalization8(raw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    record = core.compute_reference_normalization(raw)
    bounds = record["reference_bounds"]
    return core.normalize_with_bounds(raw, bounds), {"lo": float(bounds["lo"]), "hi": float(bounds["hi"])}


def _border_values(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    thickness = max(2, min(height, width) // 40)
    pieces = (image[:thickness, :].ravel(), image[-thickness:, :].ravel(), image[:, :thickness].ravel(), image[:, -thickness:].ravel())
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _foreground_component(image8: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Find an isolated, elongated specimen silhouette without a fixed axis."""
    height, width = image8.shape
    border = _border_values(image8)
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    # A robust border-relative threshold avoids assuming whether the specimen is
    # brighter or darker than the background.
    threshold = max(12.0, 3.0 * 1.4826 * mad)
    candidate = (np.abs(image8.astype(np.float32) - background) >= threshold).astype(np.uint8)
    radius = max(2, min(9, int(round(min(height, width) * 0.012))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = ndimage.binary_fill_holes(candidate.astype(bool)).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, 8)
    viable = []
    for label in range(1, count):
        x, y, box_w, box_h, area = map(int, stats[label])
        if area < max(250, int(height * width * 0.015)) or area > height * width * 0.90:
            continue
        if x <= 0 or y <= 0 or x + box_w >= width or y + box_h >= height:
            continue
        ys, xs = np.nonzero(labels == label)
        points = np.column_stack((xs, ys)).astype(np.float64)
        if len(points) < 100:
            continue
        centered = points - points.mean(axis=0)
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        normal = np.array([-axis[1], axis[0]])
        along = centered @ axis
        across = centered @ normal
        # Quantiles suppress isolated mask pixels while retaining the specimen body.
        s_lo, s_hi = np.percentile(along, [1, 99])
        t_lo, t_hi = np.percentile(across, [2, 98])
        length = float(s_hi - s_lo)
        breadth = float(t_hi - t_lo)
        aspect = length / max(breadth, 1.0)
        if length < 60 or breadth < 18 or aspect < 1.45:
            continue
        # Prefer a dominant long component and reject ambiguous image silhouettes.
        score = float(area) * min(aspect, 8.0)
        viable.append((score, labels == label, points.mean(axis=0), axis, normal, s_lo, s_hi, t_lo, t_hi, area, aspect))
    if not viable:
        return None, {
            "status": "UNRESOLVED_MANUAL_REVIEW_REQUIRED",
            "method": "border_relative_connected_component_pca_v1",
            "background_median": background,
            "background_mad": mad,
            "foreground_threshold": threshold,
            "candidate_components": max(0, count - 1),
            "reason": "no isolated elongated specimen silhouette passed geometry checks",
        }
    viable.sort(key=lambda entry: entry[0], reverse=True)
    best = viable[0]
    second_score = viable[1][0] if len(viable) > 1 else 0.0
    confidence = float(np.clip((best[0] - second_score) / max(best[0], 1.0), 0.0, 1.0))
    return best[1], {
        "status": "CANDIDATE_REQUIRES_HUMAN_REVIEW",
        "method": "border_relative_connected_component_pca_v1",
        "background_median": background,
        "background_mad": mad,
        "foreground_threshold": threshold,
        "candidate_components": len(viable),
        "selected_component_area_px": int(best[9]),
        "selected_component_aspect_ratio": float(best[10]),
        "orientation_xy_unit": [float(best[3][0]), float(best[3][1])],
        "orientation_confidence": confidence,
        "segmentation_mask": best[1],
        "geometry": {
            "center": best[2], "axis": best[3], "normal": best[4],
            "s_min": float(best[5]), "s_max": float(best[6]),
            "t_min": float(best[7]), "t_max": float(best[8]),
        },
    }


def _aabb_from_oriented(center, axis, normal, s_lo, s_hi, t_lo, t_hi, shape) -> list[int]:
    corners = []
    for s in (s_lo, s_hi):
        for t in (t_lo, t_hi):
            point = center + axis * s + normal * t
            corners.append(point)
    points = np.asarray(corners)
    h, w = shape
    x0 = int(np.clip(math.floor(points[:, 0].min()), 0, w - 1))
    y0 = int(np.clip(math.floor(points[:, 1].min()), 0, h - 1))
    x1 = int(np.clip(math.ceil(points[:, 0].max()) + 1, x0 + 1, w))
    y1 = int(np.clip(math.ceil(points[:, 1].max()) + 1, y0 + 1, h))
    return [x0, y0, x1 - x0, y1 - y0]


def _inside(rect: list[int] | tuple[int, ...], outer: list[int] | tuple[int, ...]) -> bool:
    x, y, w, h = rect
    ox, oy, ow, oh = outer
    return x >= ox and y >= oy and x + w <= ox + ow and y + h <= oy + oh


def _overlap(first, second) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _texture_metrics(image8, rect):
    metrics = core.roi_texture_metrics(image8, rect)
    code = core.texture_failure_code(
        metrics, _TEXTURE_THRESHOLDS["min_texture_std"], _TEXTURE_THRESHOLDS["min_texture_contrast"],
        _TEXTURE_THRESHOLDS["max_saturated_frac"], _TEXTURE_THRESHOLDS["min_structure_ratio"],
        _TEXTURE_THRESHOLDS["max_directional_coherence"], _TEXTURE_THRESHOLDS["min_periodicity_score"],
    )
    return metrics, code


def _mask_occupancy(mask: np.ndarray | None, rect: list[int]) -> float:
    if mask is None:
        return 1.0
    x, y, w, h = rect
    patch = mask[y:y + h, x:x + w]
    if patch.shape != (h, w) or patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def _make_groups(image8, free_region, exclusions, geometry, mask, *, count=_DEFAULT_GAUGE_COUNT):
    x, y, width, height = free_region
    if width < 60 or height < 40:
        return [], [{"gauge": f"G{i + 1:02d}", "status": "NOT_PROPOSED", "reason": "free region is too small for safe texture windows"} for i in range(count)]
    if geometry is None:
        # Explicit manual boxes need no camera-axis convention: their longer
        # dimension defines the gauge direction when silhouette PCA is absent.
        if width >= height:
            center = np.array([x + width / 2, y + height / 2], dtype=float)
            axis = np.array([1.0, 0.0]); normal = np.array([0.0, 1.0])
            s_min, s_max = -0.40 * width, 0.40 * width
            t_min, t_max = -0.25 * height, 0.25 * height
        else:
            center = np.array([x + width / 2, y + height / 2], dtype=float)
            axis = np.array([0.0, 1.0]); normal = np.array([-1.0, 0.0])
            s_min, s_max = -0.40 * height, 0.40 * height
            t_min, t_max = -0.25 * width, 0.25 * width
    else:
        center = np.asarray(geometry["center"], dtype=float)
        axis = np.asarray(geometry["axis"], dtype=float)
        normal = np.asarray(geometry["normal"], dtype=float)
        s_min, s_max = geometry["s_min"], geometry["s_max"]
        t_min, t_max = geometry["t_min"], geometry["t_max"]
    long_span = float(s_max - s_min)
    cross_span = float(t_max - t_min)
    patch_side = int(np.clip(round(min(cross_span * 0.30, long_span * 0.12)), 15, 81))
    if patch_side % 2 == 0:
        patch_side -= 1
    if cross_span < patch_side * 2.2 or long_span < patch_side * 3.0:
        patch_side = int(min(cross_span / 2.4, long_span / 3.2))
        patch_side = patch_side if patch_side % 2 else patch_side - 1
    if patch_side < 15:
        return [], [{"gauge": f"G{i + 1:02d}", "status": "NOT_PROPOSED", "reason": "specimen cannot safely contain the minimum texture window"} for i in range(count)]
    failures = []
    groups = []
    # The auto free-region box already excludes both specimen ends by 16%;
    # keep each tracking window fully inside that inner region as well.
    end_margin = 0.16 * long_span + patch_side / 2.0
    interior_s = (s_min + end_margin, s_max - end_margin)
    t_lo = t_min + 0.16 * cross_span
    t_hi = t_max - 0.16 * cross_span
    exclusions = [list(map(int, item)) for item in exclusions]
    for index in range(count):
        fraction = _GAUGE_FRACTIONS[index] if index < len(_GAUGE_FRACTIONS) else (index + 1) / (count + 1)
        center_t_target = t_lo + fraction * (t_hi - t_lo)
        rects = []
        rect_metrics = []
        reason = None
        for station in interior_s:
            center_s = station
            candidates = []
            max_offset = max(0, min(patch_side // 3, int(cross_span / 20)))
            offsets = [0]
            for delta in range(3, max_offset + 1, max(1, patch_side // 8)):
                offsets.extend((-delta, delta))
            for offset in offsets:
                point = center + axis * center_s + normal * (center_t_target + offset)
                rx = int(round(point[0] - patch_side / 2))
                ry = int(round(point[1] - patch_side / 2))
                rect = [rx, ry, patch_side, patch_side]
                if not _inside(rect, free_region) or any(_overlap(rect, excluded) for excluded in exclusions):
                    continue
                occupancy = _mask_occupancy(mask, rect)
                if occupancy < 0.90:
                    continue
                metrics, code = _texture_metrics(image8, rect)
                if code is not None:
                    continue
                score = float(metrics.get("structure_tensor_ratio", 0.0)) * float(metrics.get("std_gray", 0.0))
                candidates.append((score, rect, metrics, occupancy))
            if not candidates:
                reason = "NO_UNAMBIGUOUS_TEXTURE_OR_MASKED_WINDOW"
                break
            candidates.sort(key=lambda item: item[0], reverse=True)
            _score, rect, metrics, occupancy = candidates[0]
            rects.append(rect); rect_metrics.append({"texture": metrics, "mask_occupancy": occupancy})
        if reason or len(rects) != 2:
            failures.append({"gauge": f"G{index + 1:02d}", "status": "NOT_PROPOSED", "reason": reason or "NO_SAFE_ENDPOINT_WINDOWS"})
            continue
        dx = (rects[1][0] + rects[1][2] / 2) - (rects[0][0] + rects[0][2] / 2)
        dy = (rects[1][1] + rects[1][3] / 2) - (rects[0][1] + rects[0][3] / 2)
        if math.hypot(dx, dy) < patch_side * 2.5:
            failures.append({"gauge": f"G{index + 1:02d}", "status": "NOT_PROPOSED", "reason": "GAUGE_LENGTH_TOO_SHORT"})
            continue
        groups.append({
            "name": f"G{index + 1:02d}", "roi1": rects[0], "roi2": rects[1],
            "actual_mode": "distance", "selected_mode": "distance", "role": "axial",
            "proposal_status": "CANDIDATE_REQUIRES_HUMAN_REVIEW",
            "texture_diagnostics": rect_metrics,
        })
    return groups, failures


def _preview(image8, path, proposal, label):
    canvas = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR)
    free = proposal.get("free_region")
    if free:
        x, y, w, h = map(int, free)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 220, 220), 2)
    for rect in proposal.get("exclusions", []):
        x, y, w, h = map(int, rect)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for group in proposal.get("groups", []):
        for key, color in (("roi1", (0, 255, 0)), ("roi2", (255, 120, 0))):
            x, y, w, h = map(int, group[key])
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        c1 = (int(round(group["roi1"][0] + group["roi1"][2] / 2)), int(round(group["roi1"][1] + group["roi1"][3] / 2)))
        c2 = (int(round(group["roi2"][0] + group["roi2"][2] / 2)), int(round(group["roi2"][1] + group["roi2"][3] / 2)))
        cv2.line(canvas, c1, c2, (255, 0, 255), 1)
        cv2.putText(canvas, group["name"], (c1[0], max(12, c1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    core.write_image_checked(path, canvas)


def _geometry_basis(proposal: Mapping[str, Any]) -> dict[str, Any]:
    return {key: proposal.get(key) for key in _GEOMETRY_KEYS}


def proposal_hash(proposal: Mapping[str, Any]) -> str:
    """Return the storage layer's canonical hash of the whole reviewed proposal."""
    return _shared_proposal_hash(proposal)


def propose(acquisition: Mapping[str, Any], project_dir: str | Path, free_region=None) -> dict[str, Any]:
    """Propose a conservative specimen region and up to five textured gauges.

    ``free_region`` is an optional human supplied [x, y, width, height] override.
    Automatic output always requires visual review and explicit approval.
    """
    if not isinstance(acquisition, Mapping):
        raise core.CoreError("INVALID_ACQUISITION", {"message": "acquisition must be a mapping"})
    frame_records = _frame_records(acquisition)
    ref, raw, reference_digest = _read_reference(acquisition)
    image8, normalization = _normalization8(raw)
    height, width = image8.shape
    segmentation_mask, segmentation = _foreground_component(image8)
    geometry = segmentation.get("geometry")
    geometry_summary = None
    if geometry is not None:
        geometry_summary = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in geometry.items()}
    exclusions = []
    old_roi = acquisition.get("roi") if isinstance(acquisition.get("roi"), Mapping) else {}
    for item in old_roi.get("exclusions", []) or []:
        rect = _validate_rect(item, image8.shape, "exclusion")
        exclusions.append(rect)
    manual = free_region is not None
    if manual:
        region = _validate_rect(free_region, image8.shape, "free_region")
        free_status = "MANUAL_OVERRIDE_REQUIRES_HUMAN_REVIEW"
        if geometry is None or _mask_occupancy(segmentation_mask, region) < 0.70:
            # Trust the human's rectangular free-segment selection; use its longer
            # image-space axis only when no silhouette securely overlaps the box.
            geometry_for_gauges = None
            mask_for_gauges = None
        else:
            geometry_for_gauges = geometry
            mask_for_gauges = segmentation_mask
    elif geometry is not None:
        # Exclude the silhouette ends (where shoulders/grips commonly intrude) and
        # narrow borders; geometry remains only a candidate pending human review.
        s_span = geometry["s_max"] - geometry["s_min"]
        t_span = geometry["t_max"] - geometry["t_min"]
        region = _aabb_from_oriented(
            geometry["center"], geometry["axis"], geometry["normal"],
            geometry["s_min"] + 0.16 * s_span, geometry["s_max"] - 0.16 * s_span,
            geometry["t_min"] + 0.12 * t_span, geometry["t_max"] - 0.12 * t_span,
            image8.shape,
        )
        free_status = "AUTO_CANDIDATE_REQUIRES_HUMAN_REVIEW"
        geometry_for_gauges = geometry
        mask_for_gauges = segmentation_mask
    else:
        region = [0, 0, width, height]
        free_status = "UNRESOLVED_MANUAL_REVIEW_REQUIRED"
        geometry_for_gauges = None
        mask_for_gauges = None
    if free_status == "UNRESOLVED_MANUAL_REVIEW_REQUIRED":
        groups, rejected = [], [{"gauge": f"G{i + 1:02d}", "status": "NOT_PROPOSED", "reason": "SPECIMEN_FREE_REGION_UNRESOLVED"} for i in range(_DEFAULT_GAUGE_COUNT)]
    else:
        groups, rejected = _make_groups(image8, region, exclusions, geometry_for_gauges, mask_for_gauges)
    extra_frames = _frame_records(acquisition)
    previews = []
    preview_root = Path(project_dir).expanduser().resolve() / "proposals" / str(acquisition.get("acquisition_id", "unknown"))
    # Proposal hash determines the immutable review folder after geometry is known.
    proposal = {
        "version": _PROPOSAL_VERSION,
        "free_region": region,
        "field_roi": region,
        "free_region_status": free_status,
        "exclusions": exclusions,
        "groups": groups,
        "reference_frame": int(ref["frame"]),
        "reference_path": str(ref["path"]),
        "reference_sha256": reference_digest,
        "image_shape": [int(height), int(width)],
        "diagnostics": {
            "status": free_status,
            "manual_free_region_override": manual,
            "mask_method": segmentation.get("method"),
            "mask_status": segmentation.get("status"),
            "mask_diagnostics": {key: value for key, value in segmentation.items() if key not in {"segmentation_mask", "geometry"}},
            "orientation": geometry_summary,
            "normalization": normalization,
            "texture_thresholds": dict(_TEXTURE_THRESHOLDS),
            "proposed_gauge_count": len(groups),
            "requested_gauge_count": _DEFAULT_GAUGE_COUNT,
            "rejected_gauges": rejected,
            "needs_manual_review": True,
            "interpretation": "Candidate rectangles are image-space suggestions. Confirm the free specimen segment and exclude grips, shadows, edges, or occlusions before approval.",
        },
        "preview_paths": [],
    }
    preview_key = _canonical_hash(_geometry_basis(proposal))
    preview_root = preview_root / preview_key
    preview_root.mkdir(parents=True, exist_ok=True)
    preview_indices = (0, len(extra_frames) // 2, len(extra_frames) - 1)
    for index, label in zip(preview_indices, ("first", "middle", "last")):
        record = extra_frames[index]
        try:
            preview_raw = core.read_gray_image(record["path"])
            if tuple(preview_raw.shape) != tuple(image8.shape):
                raise core.CoreError("IMAGE_DIMENSION_MISMATCH", {"frame": record["frame"], "shape": list(preview_raw.shape), "reference_shape": [height, width]})
            preview8 = core.normalize_with_bounds(preview_raw, normalization)
            target = preview_root / f"{label}_{record['frame']}.png"
            _preview(preview8, target, proposal, f"{label}: frame {record['frame']} | review required")
            previews.append(str(target.relative_to(Path(project_dir).expanduser().resolve())).replace("\\", "/"))
        except Exception as exc:
            proposal["diagnostics"].setdefault("preview_errors", []).append({"frame": record["frame"], "error": str(exc)})
    proposal["preview_paths"] = previews
    proposal = _strict_json(proposal)
    proposal["proposal_hash"] = proposal_hash(proposal)
    return _strict_json(proposal)


def _validate_rect(rect, image_shape, name) -> list[int]:
    try:
        if len(rect) != 4:
            raise ValueError("rectangle must contain [x, y, width, height]")
        values = [int(value) for value in rect]
    except (TypeError, ValueError) as exc:
        raise core.CoreError("INVALID_ROI", {"name": name, "roi": rect, "message": str(exc)}) from exc
    if any(float(original) != converted for original, converted in zip(rect, values)):
        raise core.CoreError("INVALID_ROI", {"name": name, "roi": rect, "message": "ROI coordinates must be integers"})
    if values[2] <= 0 or values[3] <= 0 or not core.rect_is_inside_image(values, image_shape):
        raise core.CoreError("ROI_OUT_OF_BOUNDS", {"name": name, "roi": values, "image_shape": list(image_shape)})
    return values


def validate_approval(acquisition: Mapping[str, Any]) -> None:
    """Reject missing, stale, or geometry-edited ROI approvals."""
    proposal = acquisition.get("roi")
    approval = acquisition.get("approval")
    if not isinstance(proposal, Mapping):
        raise core.CoreError("ROI_NOT_PROPOSED", {"acquisition_id": acquisition.get("acquisition_id")})
    if not isinstance(approval, Mapping):
        raise core.CoreError("ROI_NOT_APPROVED", {"acquisition_id": acquisition.get("acquisition_id")})
    expected = proposal_hash(proposal)
    if proposal.get("proposal_hash") != expected or approval.get("proposal_hash") != expected:
        raise core.CoreError("ROI_APPROVAL_STALE", {"expected_proposal_hash": expected, "proposal_hash": proposal.get("proposal_hash"), "approved_hash": approval.get("proposal_hash")})
    if approval.get("reference_sha256") != proposal.get("reference_sha256"):
        raise core.CoreError("ROI_APPROVAL_STALE", {"reason": "approval reference hash differs from current proposal"})
    if approval.get("specimen_confirmed") is not True:
        raise core.CoreError("SPECIMEN_NOT_CONFIRMED", {"message": "specimen confirmation is required before analysis"})
    if proposal.get("free_region_status") == "UNRESOLVED_MANUAL_REVIEW_REQUIRED":
        raise core.CoreError("ROI_REVIEW_REQUIRED", {"message": "automatic segmentation could not identify a safe free region; supply a human-selected free_region and review it"})
    if not str(approval.get("reviewer") or "").strip() or not str(approval.get("approved_at") or "").strip():
        raise core.CoreError("ROI_APPROVAL_INCOMPLETE", {"message": "reviewer and approval time are required"})
    if proposal.get("diagnostics", {}).get("needs_manual_review") and not approval.get("specimen_confirmed"):
        raise core.CoreError("ROI_REVIEW_REQUIRED", {"message": "automatic ROI candidates require human confirmation"})
    reference_path = Path(str(proposal.get("reference_path", ""))).expanduser()
    try:
        actual_sha = _sha256_file(reference_path)
        raw = core.read_gray_image(reference_path)
    except Exception as exc:
        raise core.CoreError("ROI_REFERENCE_UNAVAILABLE", {"path": str(reference_path), "message": str(exc)}) from exc
    if actual_sha != proposal.get("reference_sha256"):
        raise core.CoreError("ROI_REFERENCE_CHANGED", {"expected_sha256": proposal.get("reference_sha256"), "actual_sha256": actual_sha})
    if list(raw.shape) != list(proposal.get("image_shape", [])):
        raise core.CoreError("ROI_REFERENCE_CHANGED", {"expected_shape": proposal.get("image_shape"), "actual_shape": list(raw.shape)})
    region = _validate_rect(proposal.get("free_region"), raw.shape, "free_region")
    exclusions = [_validate_rect(item, raw.shape, "exclusion") for item in proposal.get("exclusions", [])]
    if str(acquisition.get("analysis_mode", "extensometer")) == "fullfield":
        field_roi = _validate_rect(proposal.get("field_roi", region), raw.shape, "field_roi")
        if not _inside(field_roi, region):
            raise core.CoreError("ROI_OUTSIDE_FREE_REGION", {"field_roi": field_roi, "free_region": region})
        return None
    if not isinstance(proposal.get("groups"), list):
        raise core.CoreError("ROI_GROUPS_UNAVAILABLE", {"message": "proposal groups must be an array"})
    if not proposal["groups"]:
        raise core.CoreError("ROI_GROUPS_UNAVAILABLE", {"message": "at least one approved extensometer group is required"})
    names = set()
    for group in proposal["groups"]:
        if not isinstance(group, Mapping) or not str(group.get("name") or "").strip():
            raise core.CoreError("INVALID_ROI_GROUP", {"group": group})
        if group["name"] in names:
            raise core.CoreError("DUPLICATE_ROI_GROUP", {"name": group["name"]})
        names.add(group["name"])
        for key in ("roi1", "roi2"):
            rect = _validate_rect(group.get(key), raw.shape, f"{group['name']}.{key}")
            if not _inside(rect, region):
                raise core.CoreError("ROI_OUTSIDE_FREE_REGION", {"group": group["name"], "roi": key, "rect": rect, "free_region": region})
            if any(_overlap(rect, excluded) for excluded in exclusions):
                raise core.CoreError("ROI_OVERLAPS_EXCLUSION", {"group": group["name"], "roi": key})


def validate_proposal(acquisition: Mapping[str, Any], proposal: Mapping[str, Any], project_dir=None) -> None:
    """Validate proposal hash, explicit reference identity, and all geometry."""
    expected = proposal_hash(proposal)
    if proposal.get("proposal_hash") != expected:
        raise core.CoreError("ROI_PROPOSAL_HASH_MISMATCH", {"expected": expected, "actual": proposal.get("proposal_hash")})
    frames = _frame_records(acquisition)
    match = [item for item in frames if item["frame"] == proposal.get("reference_frame")]
    if len(match) != 1 or str(match[0]["path"]) != str(Path(str(proposal.get("reference_path"))).resolve()):
        raise core.CoreError("ROI_REFERENCE_FRAME_MISMATCH", {"reference_frame": proposal.get("reference_frame"), "reference_path": proposal.get("reference_path")})
    if _sha256_file(match[0]["path"]) != proposal.get("reference_sha256"):
        raise core.CoreError("ROI_REFERENCE_CHANGED", {"reference_path": str(match[0]["path"])})
    raw = core.read_gray_image(match[0]["path"])
    region = _validate_rect(proposal.get("free_region"), raw.shape, "free_region")
    exclusions = [_validate_rect(item, raw.shape, "exclusion") for item in proposal.get("exclusions", [])]
    if str(acquisition.get("analysis_mode", "extensometer")) == "fullfield":
        field_roi = _validate_rect(proposal.get("field_roi", region), raw.shape, "field_roi")
        if not _inside(field_roi, region):
            raise core.CoreError("ROI_OUTSIDE_FREE_REGION", {"field_roi": field_roi, "free_region": region})
        return
    if not isinstance(proposal.get("groups"), list):
        raise core.CoreError("ROI_GROUPS_UNAVAILABLE", {"message": "proposal groups must be an array"})
    if not proposal["groups"]:
        return None
    names = set()
    for group in proposal["groups"]:
        if not isinstance(group, Mapping) or not str(group.get("name") or "").strip() or group["name"] in names:
            raise core.CoreError("INVALID_ROI_GROUP", {"group": group})
        names.add(group["name"])
        for key in ("roi1", "roi2"):
            rect = _validate_rect(group.get(key), raw.shape, f"{group['name']}.{key}")
            if not _inside(rect, region):
                raise core.CoreError("ROI_OUTSIDE_FREE_REGION", {"group": group["name"], "roi": key, "rect": rect, "free_region": region})
            if any(_overlap(rect, excluded) for excluded in exclusions):
                raise core.CoreError("ROI_OVERLAPS_EXCLUSION", {"group": group["name"], "roi": key})


def approve(acquisition: Mapping[str, Any], reviewer: str, specimen_confirmed: bool) -> dict[str, Any]:
    proposal = acquisition.get("roi")
    if not isinstance(proposal, Mapping):
        raise core.CoreError("ROI_NOT_PROPOSED", {"acquisition_id": acquisition.get("acquisition_id")})
    if not str(reviewer or "").strip():
        raise core.CoreError("INVALID_REVIEWER", {"message": "reviewer must be a non-empty name or identifier"})
    if specimen_confirmed is not True:
        raise core.CoreError("SPECIMEN_NOT_CONFIRMED", {"message": "approval requires explicit specimen confirmation"})
    record = {
        "proposal_hash": proposal_hash(proposal),
        "reference_sha256": proposal.get("reference_sha256"),
        "reviewer": str(reviewer).strip(),
        "approved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "specimen_confirmed": True,
        "review_status": "APPROVED",
    }
    candidate = {**dict(acquisition), "approval": record}
    validate_approval(candidate)
    return record
