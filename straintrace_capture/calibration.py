"""Pinhole and stereo calibration from committed checkerboard captures."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CaptureError, positive_integer, positive_number
from .image_inputs import bayer_luminance, decode_samples
from .storage import ExperimentReader, sha256_file


CALIBRATION_SCHEMA = "straintrace.capture-calibration/v1"


@dataclass
class _Observation:
    frame: dict
    shape: tuple[int, int]
    corners: np.ndarray


def _observation(path: Path, frame: dict, pattern, stereo):
    if sha256_file(path) != frame["sha256"]:
        raise CaptureError("CALIBRATION_INPUT_CHANGED", f"checksum mismatch for {frame['frame_id']}")
    color = frame["color_model"]
    data = (bayer_luminance(path, frame["pixel_format"]) if color == "bayer"
            else decode_samples(path, color_model=color))
    if data.ndim == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY if data.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    gray = cv2.convertScaleAbs(data, alpha=255 / ((1 << frame["significant_bits"]) - 1))
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    if stereo:
        # Canonicalize the two 180-degree indexings by the top-left square's
        # color. Mixed inner-corner parity makes opposite squares different.
        points = corners.reshape(pattern[1], pattern[0], 2)
        origin = points[0, 0]
        dx = points[0, 1] - origin
        dy = points[1, 0] - origin
        side = [origin - .3 * dx - .3 * dy, origin + .3 * dx - .3 * dy]
        p0, p1 = (tuple(map(float, point)) for point in side)
        if any(not (1 <= x < gray.shape[1] - 1 and 1 <= y < gray.shape[0] - 1) for x, y in (p0, p1)):
            return None
        s0, s1 = (float(cv2.getRectSubPix(gray, (3, 3), point).mean()) for point in (p0, p1))
        if abs(s0 - s1) < 18:
            return None
        if s0 > s1:
            corners = corners[::-1].copy()
    return _Observation(frame, (gray.shape[1], gray.shape[0]), corners)


def _fit_mono(observations, points_3d, minimum_views, geometry):
    if len(observations) < minimum_views:
        raise CaptureError("CALIBRATION_VIEWS_INSUFFICIENT", f"{len(observations)} usable board views; {minimum_views} required")
    sizes = {item.shape for item in observations}
    if len(sizes) != 1:
        raise CaptureError("CALIBRATION_GEOMETRY_CHANGED", "calibration images have mixed widths or heights")
    size = sizes.pop()
    distinct = []
    for item in observations:
        if all(np.sqrt(np.mean((item.corners - prior) ** 2)) >= 3 for prior in distinct):
            distinct.append(item.corners)
    if len(distinct) < minimum_views:
        raise CaptureError("CALIBRATION_VIEWS_REPEATED", f"only {len(distinct)} board poses differ by at least 3 px RMS")
    images = [item.corners for item in observations]
    result = cv2.calibrateCameraExtended([points_3d] * len(images), images, size, None, None)
    rms, matrix, coefficients, _, _, _, _, per_view = result
    if not math.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(coefficients).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise CaptureError("CALIBRATION_FIT_INVALID", "pinhole fit is degenerate or non-finite")
    all_points = np.concatenate(images, axis=0).reshape(-1, 2)
    span = np.ptp(all_points, axis=0) / np.asarray(size)
    return {
        "image_size_px": list(size), "sensor_geometry": geometry,
        "camera_matrix": matrix.tolist(), "distortion_coefficients": coefficients.ravel().tolist(),
        "reprojection_rms_px": float(rms), "corner_span_fraction_xy": span.tolist(),
        "distinct_poses": len(distinct),
        "views": [{"frame_id": item.frame["frame_id"], "sha256": item.frame["sha256"], "rms_px": float(error)}
                  for item, error in zip(observations, per_view.ravel())],
    }


def _geometry(reader, frames, camera_id):
    keys = ("width", "height", "offset_x", "offset_y", "binning_x", "binning_y", "decimation_x", "decimation_y")
    settings = {}
    for event in reader.events():
        if event["kind"] == "phase_started":
            feature = event["camera_features"][camera_id]
            settings[event["phase"]] = {key: feature.get(key, {}).get("value") for key in keys if key in feature}
    contexts = [settings.get(index, {}) for index in sorted({frame["phase"] for frame in frames})]
    if any(item != contexts[0] for item in contexts):
        raise CaptureError("CALIBRATION_GEOMETRY_CHANGED", "sensor ROI or sampling changed among board views")
    return contexts[0] if contexts else {}


def calibrate(root, camera_ids, *, inner_columns: int, inner_rows: int, square_mm: float, min_views: int = 6) -> dict:
    """Fit one or two cameras using the calibration role and explicit paired events."""
    cols = positive_integer(inner_columns, "inner columns", minimum=3)
    rows = positive_integer(inner_rows, "inner rows", minimum=3)
    square = positive_number(square_mm, "checkerboard spacing (mm)")
    minimum = positive_integer(min_views, "minimum board views", minimum=5)
    identities = tuple(camera_ids)
    if len(identities) not in (1, 2) or len(set(identities)) != len(identities):
        raise CaptureError("CALIBRATION_CAMERAS_INVALID", "select one camera for pinhole or two distinct cameras for stereo")
    stereo = len(identities) == 2
    if stereo and cols % 2 == rows % 2:
        raise CaptureError("STEREO_BOARD_PARITY", "use a board with one odd and one even inner-corner count for stereo")
    object_points = np.zeros((cols * rows, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square
    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        device_ids = {camera["device"]["id"] for camera in reader.cameras()}
        if not set(identities) <= device_ids:
            raise CaptureError("CALIBRATION_CAMERAS_INVALID", "selected cameras are not registered in this experiment")
        frames = {
            camera_id: {frame["frame_id"]: frame for frame in reader.frames(camera_id=camera_id)
                        if frame["role"] == "calibration"}
            for camera_id in identities
        }
        observations = {
            camera: {key: _observation(reader.image_path(row), row, (cols, rows), stereo)
                     for key, row in entries.items()} for camera, entries in frames.items()
        }
        rejected = [key for entries in observations.values() for key, item in entries.items() if item is None]
        paired = []
        if stereo:
            for group in reader.pairing()["groups"]:
                associated = group["frames_by_camera"]
                if group["status"] != "associated" or any(key not in associated for key in identities):
                    continue
                match = tuple(observations[camera].get(associated[camera][0]) for camera in identities)
                if all(item is not None for item in match):
                    paired.append((group["event_id"], *match))
            selected = [[pair[index + 1] for pair in paired] for index in range(2)]
        else:
            selected = [[item for item in observations[identities[0]].values() if item is not None]]
        models = {}
        for camera, selection in zip(identities, selected):
            geometry = _geometry(reader, [item.frame for item in selection], camera)
            models[camera] = _fit_mono(selection, object_points, minimum, geometry)

    relationship = None
    if stereo:
        left, right = (models[camera] for camera in identities)
        if left["image_size_px"] != right["image_size_px"]:
            raise CaptureError("STEREO_IMAGE_SIZE_MISMATCH", "the current stereo calibration requires equal image size")
        result = cv2.stereoCalibrate(
            [object_points] * len(paired), [item.corners for item in selected[0]], [item.corners for item in selected[1]],
            np.asarray(left["camera_matrix"]), np.asarray(left["distortion_coefficients"]),
            np.asarray(right["camera_matrix"]), np.asarray(right["distortion_coefficients"]),
            tuple(left["image_size_px"]), flags=cv2.CALIB_FIX_INTRINSIC,
        )
        rms, _, _, _, _, rotation, translation, essential, fundamental = result
        if not all(np.isfinite(item).all() for item in (rms, rotation, translation, essential, fundamental)):
            raise CaptureError("STEREO_FIT_INVALID", "stereo fit is non-finite")
        relationship = {
            "left_camera": identities[0], "right_camera": identities[1], "reprojection_rms_px": float(rms),
            "rotation_left_to_right": rotation.tolist(), "translation_left_to_right_mm": translation.ravel().tolist(),
            "essential": essential.tolist(), "fundamental": fundamental.tolist(),
            "paired_trigger_events": [pair[0] for pair in paired], "target_condition": "stationary_per_pair",
        }
    return {
        "schema": CALIBRATION_SCHEMA, "capture_experiment_id": experiment["experiment_id"],
        "target": {"type": "checkerboard", "inner_columns": cols, "inner_rows": rows, "square_mm": square},
        "fit": "opencv_pinhole_calibration", "opencv_version": cv2.__version__,
        "cameras": models, "stereo": relationship, "corner_detection_rejections": rejected,
    }


def parse_profile(content):
    data = json.loads(content)
    if not isinstance(data, dict) or data.get("schema") != CALIBRATION_SCHEMA or not data.get("cameras"):
        raise CaptureError("CALIBRATION_SCHEMA_INVALID", "expected a versioned Capture calibration profile")
    for camera in data["cameras"].values():
        matrix, distortion = np.asarray(camera["camera_matrix"]), np.asarray(camera["distortion_coefficients"])
        size = camera["image_size_px"]
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0
                or distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all()
                or len(size) != 2 or any(type(item) is not int or item <= 0 for item in size)):
            raise CaptureError("CALIBRATION_PROFILE_INVALID", "profile has invalid camera geometry or distortion coefficients")
    return data


def read_profile(path):
    return parse_profile(Path(path).read_bytes())


def undistort(pixels, model):
    """Correct a derived preview only, using a complete matching image geometry."""
    if tuple(pixels.shape[:2][::-1]) != tuple(model["image_size_px"]):
        raise CaptureError("CALIBRATION_GEOMETRY_MISMATCH", "calibration size does not match the displayed image")
    return cv2.undistort(pixels, np.asarray(model["camera_matrix"]), np.asarray(model["distortion_coefficients"]))
