"""Pinhole and stereo calibration from committed checkerboard captures."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .cameras._gentl_multipart import IMAGE_PART, parse_image_part
from .config import (
    REPEAT_OF, REPEAT_PARENT, SENSOR_WINDOW_FIELDS, CaptureConfig, CaptureError, inside_repeat_source,
    positive_integer, positive_number,
)
from .output import resolve_output_path
from .continuation import CONTINUATION_REFERENCE
from .image_inputs import bayer_luminance, decode_samples, select_frame_image, validate_analysis_part
from .storage import ExperimentReader, sha256_file


CALIBRATION_SCHEMA = "straintrace.capture-calibration/v1"
CHECKERBOARD_CAPTURE_SCHEMA = "straintrace.capture.checkerboard/v1"
CHECKERBOARD_PHASE = {"mode": "manual", "role": "calibration"}
DEFAULT_MIN_VIEWS = 6
SENSOR_GEOMETRY = ("width", "height", *SENSOR_WINDOW_FIELDS)


def phase_sensor_geometry(reader, camera_id):
    """Use the camera readback recorded when each capture phase started."""
    return {
        str(event["phase"]): {name: value.get("value") for name, value in features.items() if name in SENSOR_GEOMETRY}
        for event in reader.events(kind="phase_started")
        if (features := event.get("camera_features", {}).get(camera_id)) is not None
    }


def calibration_geometry(shape, features, metadata):
    """Describe the delivered pixels and the acquisition geometry reported for them."""
    height, width = shape[:2]
    sensor = {name: features[name] for name in SENSOR_GEOMETRY if features.get(name) is not None}
    sensor.update(width=width, height=height)
    geometry = {"image_size_px": [width, height], "sensor_geometry": sensor}
    if IMAGE_PART in metadata:
        part = metadata[IMAGE_PART]
        selector = part.get("selector") if isinstance(part, dict) else None
        try:
            key = parse_image_part(selector) if selector else None
        except CaptureError:
            key = None
        if not isinstance(key, tuple):
            raise CaptureError("CALIBRATION_PART_UNVERIFIED", "the frame does not identify a specific GenTL part")
        geometry[IMAGE_PART] = selector
        sensor.update(offset_x=key[1], offset_y=key[2])
    return geometry


def require_matching_calibration(model, geometry):
    if tuple(model["image_size_px"]) != tuple(geometry["image_size_px"]):
        raise CaptureError("CALIBRATION_GEOMETRY_MISMATCH", "the recorded image size differs from the calibration")
    expected, observed = model.get(IMAGE_PART), geometry.get(IMAGE_PART)
    if observed is not None and expected is None:
        raise CaptureError("CALIBRATION_PART_UNVERIFIED", f"no GenTL part was recorded by the profile (image {observed})")
    if expected != observed:
        raise CaptureError("CALIBRATION_PART_MISMATCH", f"calibrated part {expected}, observed part {observed}")
    for field, expected in model.get("sensor_geometry", {}).items():
        actual = geometry["sensor_geometry"].get(field)
        if field not in ("width", "height") and expected is not None and actual != expected:
            raise CaptureError("CALIBRATION_GEOMETRY_MISMATCH", f"calibration {field}={expected}, observed {actual}")


def matching_calibration(models, geometry):
    """Select the latest compatible profile for the current camera."""
    first_problem = None
    for model in reversed(models):
        try:
            require_matching_calibration(model, geometry)
        except CaptureError as exc:
            if first_problem is None:
                first_problem = exc
        else:
            return model
    if first_problem is not None:
        raise first_problem
    raise CaptureError("CALIBRATION_UNAVAILABLE", "请先为此相机加载或拟合镜头标定。")


@dataclass(frozen=True)
class CheckerboardTarget:
    camera_ids: tuple[str, ...]
    inner_columns: int
    inner_rows: int
    square_mm: float

    def __post_init__(self):
        positive_integer(self.inner_columns, "列内角点数", minimum=3)
        positive_integer(self.inner_rows, "行内角点数", minimum=3)
        positive_number(self.square_mm, "实测方格边长（mm）")
        if (len(self.camera_ids) not in (1, 2) or not all(isinstance(key, str) and key for key in self.camera_ids)
                or len(set(self.camera_ids)) != len(self.camera_ids)):
            raise CaptureError("CALIBRATION_CAMERAS_INVALID", "select one camera for pinhole or two distinct cameras for stereo")
        if self.stereo and self.inner_columns % 2 == self.inner_rows % 2:
            raise CaptureError("STEREO_BOARD_PARITY", "use a board with one odd and one even inner-corner count for stereo")

    @property
    def stereo(self):
        return len(self.camera_ids) == 2

    @property
    def pattern(self):
        return self.inner_columns, self.inner_rows

    def parameters(self):
        return {"inner_columns": self.inner_columns, "inner_rows": self.inner_rows, "square_mm": float(self.square_mm)}


def prepare_checkerboard_capture(config: CaptureConfig, output, target: CheckerboardTarget):
    """Create a manual calibration plan beside the proposed experiment."""
    formal = resolve_output_path(output)
    if inside_repeat_source(config, formal):
        raise CaptureError("OUTPUT_INSIDE_REPEAT_SOURCE", "请选择来源实验目录之外的新输出位置。")
    data = config.to_dict()
    cameras = {row["id"]: row for row in data["cameras"]}
    if not set(target.camera_ids) <= cameras.keys():
        raise CaptureError("CALIBRATION_CAMERAS_INVALID", "selected cameras are not in the current plan")
    data["name"] = f"{config.name} · 棋盘格标定"
    data["cameras"] = [cameras[key] for key in target.camera_ids]
    if "quality_regions" in data:
        data["quality_regions"] = {key: region for key, region in data["quality_regions"].items() if key in target.camera_ids}
    data["phases"] = [dict(CHECKERBOARD_PHASE)]
    data["calibration_files"] = []
    data["metadata"].pop("capture_trial", None)
    data["metadata"].pop("continuation_of", None)
    data["metadata"].pop("continuation_parent", None)
    data["metadata"].pop(CONTINUATION_REFERENCE, None)
    data["metadata"].pop(REPEAT_OF, None)
    data["metadata"].pop(REPEAT_PARENT, None)
    data["metadata"]["capture_checkerboard"] = {
        "schema": CHECKERBOARD_CAPTURE_SCHEMA, "camera_ids": list(target.camera_ids), **target.parameters(),
    }
    parent = formal.parent
    return CaptureConfig.from_dict(data), parent / f"calibration-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"


def checkerboard_target(config: dict) -> CheckerboardTarget | None:
    """Read the acquisition target only when it matches the recorded cameras and phase."""
    value = config.get("metadata", {}).get("capture_checkerboard")
    phases = config.get("phases", [])
    if (not isinstance(value, dict) or value.get("schema") != CHECKERBOARD_CAPTURE_SCHEMA
            or len(phases) != 1 or phases[0]["role"] != "calibration" or phases[0]["mode"] != "manual"
            or value.get("camera_ids") != [row["id"] for row in config["cameras"]]):
        return None
    try:
        return CheckerboardTarget(tuple(value["camera_ids"]), value["inner_columns"], value["inner_rows"], value["square_mm"])
    except (KeyError, CaptureError):
        return None


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


def _distinct_observations(observations):
    distinct = []
    for item in observations:
        if all(np.sqrt(np.mean((item.corners - prior.corners) ** 2)) >= 3 for prior in distinct):
            distinct.append(item)
    return distinct


def _corner_span(observations):
    points = np.concatenate([item.corners for item in observations], axis=0).reshape(-1, 2)
    return (np.ptp(points, axis=0) / np.asarray(observations[0].shape)).tolist()


def _paired_observations(reader, identities, observations):
    for group in reader.pairing()["groups"]:
        associated = group["frames_by_camera"]
        if group["status"] != "associated" or any(key not in associated for key in identities):
            continue
        match = tuple(observations[camera].get(associated[camera][0]) for camera in identities)
        if all(item is not None for item in match):
            yield group["event_id"], *match


class CheckerboardScan:
    """Inspect new committed calibration frames once as they are saved."""

    def __init__(self, root, target: CheckerboardTarget):
        self.root = root
        self.target = target
        self._observations = {key: {} for key in target.camera_ids}
        self._read_errors = set()
        self._ordinal = 0
        self._latest = None

    def scan(self):
        target = self.target
        with ExperimentReader(self.root) as reader:
            for frame in reader.frames(after=self._ordinal):
                camera = frame["camera_id"]
                if frame["role"] == "calibration" and camera in self._observations:
                    frame_id = frame["frame_id"]
                    try:
                        item = _observation(reader.image_path(frame), frame, target.pattern, target.stereo)
                    except (OSError, ValueError, CaptureError, cv2.error):
                        self._read_errors.add(frame_id)
                        item = None
                    self._observations[camera][frame_id] = item
                    self._latest = frame
                self._ordinal = frame["ordinal"]
            pairs = list(_paired_observations(reader, target.camera_ids, self._observations)) if target.stereo else []

        cameras = {}
        distinct_ids = set()
        sizes = set()
        for index, (camera, entries) in enumerate(self._observations.items()):
            valid = [item for item in entries.values() if item is not None]
            selected = [pair[index + 1] for pair in pairs] if target.stereo else valid
            sizes.update(item.shape for item in selected)
            distinct = _distinct_observations(selected)
            distinct_ids.update(item.frame["frame_id"] for item in distinct)
            last_id = next(reversed(entries), None)
            last = entries.get(last_id)
            overlay = None
            if last is not None:
                stamps = last.frame["timestamps"]
                points = tuple(map(tuple, last.corners.reshape(-1, 2).tolist()))
                overlay = (stamps["received_utc"]["value"], stamps["received_monotonic"]["value"], points)
            cameras[camera] = {
                "saved": len(entries), "detected": len(valid), "distinct_poses": len(distinct),
                "corner_span_fraction_xy": _corner_span(selected) if selected else None,
                "latest": ("read_error" if last_id in self._read_errors else "detected" if last is not None
                           else "not_detected" if last_id else "waiting"),
                "overlay": overlay,
            }
        latest = "waiting"
        if self._latest:
            frame_id = self._latest["frame_id"]
            if target.stereo:
                trigger = self._latest["trigger_event_id"]
                pair = next((pair for pair in reversed(pairs) if pair[0] == trigger), None)
                latest_frames = [item.frame["frame_id"] for item in pair[1:]] if pair else []
                latest = "unpaired"
            elif self._observations[target.camera_ids[0]][frame_id] is not None:
                latest_frames = [frame_id]
            else:
                latest = cameras[target.camera_ids[0]]["latest"]
            if latest_frames:
                latest = "new_pose" if all(key in distinct_ids for key in latest_frames) else "similar_pose"
        distinct_poses = min(row["distinct_poses"] for row in cameras.values())
        return {"cameras": cameras, "usable_pairs": len(pairs), "distinct_poses": distinct_poses,
                "mixed_image_sizes": len(sizes) > 1, "read_errors": len(self._read_errors), "latest": latest}


def _fit_mono(observations, points_3d, minimum_views, geometry):
    if len(observations) < minimum_views:
        raise CaptureError("CALIBRATION_VIEWS_INSUFFICIENT", f"{len(observations)} usable board views; {minimum_views} required")
    sizes = {item.shape for item in observations}
    if len(sizes) != 1:
        raise CaptureError("CALIBRATION_GEOMETRY_CHANGED", "calibration images have mixed widths or heights")
    size = sizes.pop()
    distinct_count = len(_distinct_observations(observations))
    if distinct_count < minimum_views:
        raise CaptureError("CALIBRATION_VIEWS_REPEATED", f"only {distinct_count} board poses differ by at least 3 px RMS")
    images = [item.corners for item in observations]
    result = cv2.calibrateCameraExtended([points_3d] * len(images), images, size, None, None)
    rms, matrix, coefficients, _, _, _, _, per_view = result
    if not math.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(coefficients).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise CaptureError("CALIBRATION_FIT_INVALID", "pinhole fit is degenerate or non-finite")
    return {
        **geometry,
        "camera_matrix": matrix.tolist(), "distortion_coefficients": coefficients.ravel().tolist(),
        "reprojection_rms_px": float(rms), "corner_span_fraction_xy": _corner_span(observations),
        "distinct_poses": distinct_count,
        "views": [{"frame_id": item.frame["frame_id"], "path": item.frame["path"],
                   "sha256": item.frame["sha256"], "rms_px": float(error)}
                  for item, error in zip(observations, per_view.ravel())],
    }


def _geometry(observations, phase_geometry, registered):
    geometry = None
    for item in observations:
        frame = item.frame
        values = phase_geometry.get(str(frame["phase"]), registered)
        context = calibration_geometry(item.shape[::-1], values, frame.get("backend_metadata") or {})
        if geometry is not None and context != geometry:
            raise CaptureError("CALIBRATION_GEOMETRY_CHANGED", "image part, dimensions or sensor sampling changed among board views")
        geometry = context
    return geometry


def calibrate(root, camera_ids, *, inner_columns: int, inner_rows: int, square_mm: float,
              min_views: int = DEFAULT_MIN_VIEWS, image_parts=None) -> dict:
    """Fit one or two cameras using the calibration role and explicit paired events."""
    target = CheckerboardTarget(tuple(camera_ids), inner_columns, inner_rows, square_mm)
    cols, rows = target.pattern
    square = float(target.square_mm)
    minimum = positive_integer(min_views, "minimum board views", minimum=5)
    identities, stereo = target.camera_ids, target.stereo
    parts = dict(image_parts or {})
    if not parts.keys() <= set(identities):
        raise CaptureError("CALIBRATION_CAMERAS_INVALID", "image parts must identify the selected calibration cameras")
    for part in parts.values():
        validate_analysis_part(part)
    object_points = np.zeros((cols * rows, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square
    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        cameras = {camera["device"]["id"]: camera["features"] for camera in reader.cameras()}
        if not set(identities) <= cameras.keys():
            raise CaptureError("CALIBRATION_CAMERAS_INVALID", "selected cameras are not registered in this experiment")
        frames = {
            camera_id: {frame["frame_id"]: select_frame_image(frame, parts.get(camera_id)) for frame in reader.frames(camera_id=camera_id)
                        if frame["role"] == "calibration"}
            for camera_id in identities
        }
        observations = {
            camera: {key: _observation(reader.image_path(row), row, (cols, rows), stereo)
                     for key, row in entries.items()} for camera, entries in frames.items()
        }
        rejected = [key for entries in observations.values() for key, item in entries.items() if item is None]
        paired = list(_paired_observations(reader, identities, observations)) if stereo else []
        if stereo:
            selected = [[pair[index + 1] for pair in paired] for index in range(2)]
        else:
            selected = [[item for item in observations[identities[0]].values() if item is not None]]
        models = {}
        for camera, selection in zip(identities, selected):
            registered = {name: value["value"] for name, value in cameras[camera].items() if name in SENSOR_GEOMETRY}
            geometry = _geometry(selection, phase_sensor_geometry(reader, camera), registered)
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
        if IMAGE_PART in camera:
            try:
                selector = parse_image_part(camera[IMAGE_PART])
            except CaptureError as exc:
                raise CaptureError("CALIBRATION_PROFILE_INVALID", "invalid GenTL image part in calibration profile") from exc
            if not isinstance(selector, tuple):
                raise CaptureError("CALIBRATION_PROFILE_INVALID", "the profile must identify one specific GenTL image part")
    return data


def profile_bytes(path, *, expected_hash=None):
    content = Path(path).read_bytes()
    if expected_hash is not None and sha256(content).hexdigest() != expected_hash:
        raise CaptureError("CALIBRATION_HASH_MISMATCH", f"calibration checksum mismatch for {path}")
    return content


def read_profile(path, *, expected_hash=None):
    return parse_profile(profile_bytes(path, expected_hash=expected_hash))


def archived_profile_path(root, attachment):
    folder = Path(root).resolve() / "calibration"
    path = (Path(root) / attachment["path"]).resolve()
    if not path.is_relative_to(folder) or not path.is_file():
        raise CaptureError("CALIBRATION_FILE_MISSING", attachment["path"])
    return path


def undistort(pixels, model):
    """Correct a derived preview only, using a complete matching image geometry."""
    if tuple(pixels.shape[:2][::-1]) != tuple(model["image_size_px"]):
        raise CaptureError("CALIBRATION_GEOMETRY_MISMATCH", "calibration size does not match the displayed image")
    return cv2.undistort(pixels, np.asarray(model["camera_matrix"]), np.asarray(model["distortion_coefficients"]))
