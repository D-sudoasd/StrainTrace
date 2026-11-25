"""Read-only checks of saved trial images from verified TIFF bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from .config import SENSOR_WINDOW_FIELDS, CaptureError, QualityRegion
from .image_inputs import frame_part_selector, recorded_image
from .image_quality_watch import quality_region_from_image, quality_region_layout_mismatch, quality_region_settings_mismatch
from .preview import QUALITY_ALERT_FRACTION, image_quality
from .storage import TERMINAL_STATES, ExperimentReader, image_review_indices, json_text


_IMAGE_ERRORS = (CaptureError, OSError, ValueError, cv2.error)


def _indices(rows, *, all_frames):
    largest = 0 if all_frames or len(rows) <= 2 else max(row["file_bytes"] for row in rows)
    return image_review_indices(len(rows), largest, all_frames=all_frames)


def _layout(row):
    return {
        "shape": list(row["shape"]), "dtype": row["dtype"], "pixel_format": row["pixel_format"],
        "color_model": row["color_model"], "significant_bits": row["significant_bits"], "part": frame_part_selector(row),
    }


def _read_image(reader, row):
    image = recorded_image(reader.image_path(row), row, verify=True)
    if list(image.pixels.shape) != list(row["shape"]) or image.pixels.dtype.name != row["dtype"]:
        raise CaptureError("FRAME_GEOMETRY_CHANGED", row["frame_id"])
    return image


def _position(row):
    return {
        "sequence": row["sequence"], "ordinal": row["ordinal"], "frame_id": row["frame_id"], "sha256": row["sha256"],
        "received_utc_ns": (row.get("timestamps") or {}).get("received_utc", {}).get("value"),
        "layout": _layout(row),
    }


def _measured_row(item, image, roi, *, measure_texture=True):
    result = image_quality(image, roi, measure_texture=measure_texture)
    dark, saturated = result["dark_fraction"], result["bright_fraction"]
    texture = result["texture"]
    metrics = {"dark_fraction": dark, "near_saturation_fraction": saturated,
               "brightness_fraction": [value / result["code_max"] for value in result["percentiles_p5_p50_p95"]],
               "focus_response": result["focus_laplacian_variance"], "texture_measured": texture is not None,
               "texture_error": result["texture_error"],
               "sample_stride_px": result["histogram_stride_px"],
               "contrast_8bit": texture["contrast_p95_p5"] if texture is not None else None}
    flags = []
    if dark >= QUALITY_ALERT_FRACTION:
        flags.append("TRIAL_IMAGE_NEAR_BLACK")
    if saturated >= QUALITY_ALERT_FRACTION:
        flags.append("TRIAL_IMAGE_NEAR_SATURATION")
    if texture and texture["failure_code"]:
        flags.append(texture["failure_code"])
    return {**item, "metrics": metrics, "flags": flags}


def _failed_row(item, error):
    return {**item, "metrics": None, "flags": [error.code if isinstance(error, CaptureError) else "FRAME_UNREADABLE"]}


def _result(rows, available, digest, roi):
    return {
        "available": available, "selected": len(rows), "checked": sum(item["metrics"] is not None for item in rows),
        "flagged": sum(bool(item["flags"]) for item in rows), "mode": "all" if len(rows) == available else "sampled",
        "series_sha256": digest, "roi": list(roi) if roi is not None else None, "rows": rows,
    }


def trial_sensor_values(started, camera):
    features = started.get("camera_features", {}).get(camera, {})
    return {name: features[name].get("value") for name in SENSOR_WINDOW_FIELDS if name in features}


def inspect_trial_images(reader, rows, *, roi=None, region_layout=None, all_frames=False, cancelled=None, progress=None,
                         quality_region=None, sensor_values=None):
    """Check all small trials, or uniformly select frame sequences from larger trials."""
    ordered = sorted(rows, key=lambda row: row["sequence"])
    selected = [ordered[index] for index in _indices(ordered, all_frames=all_frames)]
    series = hashlib.sha256()
    region = None if roi is not None else quality_region
    fallback = "no_images" if region is not None and not ordered else None
    for row in ordered:
        if cancelled and cancelled():
            raise CaptureError("TRIAL_IMAGE_CHECK_CANCELLED", "trial image check cancelled")
        series.update(json_text(_position(row)).encode("utf-8") + b"\n")
        if region is not None and fallback is None:
            fallback = quality_region_layout_mismatch(region, row["shape"], row["pixel_format"], row["significant_bits"],
                                                      row.get("backend_metadata") or {}, sensor_values or {})
    checked, regional = [], []
    for index, row in enumerate(selected):
        if cancelled and cancelled():
            raise CaptureError("TRIAL_IMAGE_CHECK_CANCELLED", "trial image check cancelled")
        item = _position(row)
        try:
            if roi is not None and item["layout"] != region_layout:
                raise CaptureError("TRIAL_IMAGE_REGION_CHANGED", row["frame_id"])
            image = _read_image(reader, row)
            whole = _measured_row(item, image, roi)
        except _IMAGE_ERRORS as exc:
            whole = _failed_row(item, exc)
            if region is not None and fallback is None:
                regional.append(whole)
        else:
            if region is not None and fallback is None:
                try:
                    regional.append(_measured_row(item, image, region.bounds, measure_texture=False))
                except _IMAGE_ERRORS as exc:
                    regional.append(_failed_row(item, exc))
        checked.append(whole)
        if progress:
            progress(index + 1, len(selected))
    digest = series.hexdigest()
    result = _result(checked, len(ordered), digest, roi)
    if region is not None:
        check = _result(regional, len(ordered), digest, region.bounds) if fallback is None else None
        if check is not None and check["checked"] != check["selected"]:
            fallback = "incomplete"
        result["region_check"] = {"region": region.to_dict(), "fallback": fallback, "check": check}
    return result


def _check_experiment(reader, expected_id):
    record = reader.experiment
    if record["experiment_id"] != expected_id:
        raise CaptureError("FRAME_REVIEW_CHANGED", "the trial directory contains another experiment")
    if record["state"] not in TERMINAL_STATES:
        raise CaptureError("TRIAL_ACTIVE", "wait until the trial finishes writing")
    return record


def rescan_trial_images(root: Path, experiment_id, camera_id, *, roi=None, region_layout=None, all_frames=False,
                        cancelled=None, progress=None, quality_region=None):
    with ExperimentReader(root) as reader:
        _check_experiment(reader, experiment_id)
        region, values = None, None
        if quality_region is not None:
            region = QualityRegion.from_dict(quality_region)
            started = next(iter(reader.events(kind="phase_started")), {})
            values = trial_sensor_values(started, camera_id)
        return inspect_trial_images(reader, reader.frames(camera_id=camera_id), roi=roi, region_layout=region_layout,
                                    all_frames=all_frames, cancelled=cancelled, progress=progress,
                                    quality_region=region, sensor_values=values)


def _verified_quality_frame(reader, camera_id, expected):
    row = reader.frame_at_sequence(camera_id, expected["sequence"])
    position = _position(row) if row is not None else {}
    if any(position.get(key) != expected[key] for key in ("ordinal", "frame_id", "sha256", "layout")):
        raise CaptureError("FRAME_REVIEW_CHANGED", expected["frame_id"])
    try:
        return row, _read_image(reader, row)
    except (ValueError, cv2.error) as exc:
        raise CaptureError("FRAME_UNREADABLE", expected["frame_id"]) from exc


def open_quality_frame(root: Path, experiment_id, camera_id, expected):
    with ExperimentReader(root) as reader:
        _check_experiment(reader, experiment_id)
        return _verified_quality_frame(reader, camera_id, expected)[1]


open_trial_image = open_quality_frame


def read_trial_quality_region(root: Path, experiment_id, camera_id, expected, bounds, settings):
    """Bind the inspection rectangle to the verified source frame and its phase readbacks."""
    from .trial import trial_metadata

    with ExperimentReader(root) as reader:
        config = _check_experiment(reader, experiment_id)["config"]
        camera = next((item for item in config["cameras"] if item["id"] == camera_id), None)
        if trial_metadata(config) is None or camera is None:
            raise CaptureError("TRIAL_REGION_SOURCE", camera_id)
        row, image = _verified_quality_frame(reader, camera_id, expected)
        started = next((event for event in reader.events(kind="phase_started") if event.get("phase") == row["phase"]), {})
        try:
            region = quality_region_from_image(image, bounds, trial_sensor_values(started, camera_id))
        except CaptureError as exc:
            raise CaptureError("TRIAL_REGION_BOUNDS", str(exc)) from exc
        if reason := quality_region_settings_mismatch(region, settings):
            raise CaptureError(f"TRIAL_REGION_{reason.upper()}", camera_id)
        original = camera["settings"]
        unread = [name for name in SENSOR_WINDOW_FIELDS if name not in region.sensor_window
                  and settings.get(name) != original.get(name)]
        if unread:
            raise CaptureError("TRIAL_REGION_SENSOR_WINDOW", ", ".join(unread))
        return region
