"""OpenCV camera input. Camera indices identify host inputs, not physical serials."""

from __future__ import annotations

import hashlib
import math
import re
import sys
import threading
from dataclasses import replace

import cv2

from ..config import CaptureError
from .base import Camera, Device, Feature, Image, receipt_times


_API = {"win32": cv2.CAP_DSHOW, "linux": cv2.CAP_V4L2, "darwin": cv2.CAP_AVFOUNDATION}.get(sys.platform, cv2.CAP_ANY)
_PROPERTIES = {
    "width": (cv2.CAP_PROP_FRAME_WIDTH, "px"),
    "height": (cv2.CAP_PROP_FRAME_HEIGHT, "px"),
    "fps": (cv2.CAP_PROP_FPS, "Hz"),
    "driver_exposure": (cv2.CAP_PROP_EXPOSURE, "driver_units"),
    "driver_gain": (cv2.CAP_PROP_GAIN, "driver_units"),
}


def _device(index, api=None, info=None, *, stable=False) -> Device:
    path = info.path if info else None
    if stable and path:
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]
        identity, scope = f"uvc:os-{digest}", "host_device_path"
    else:
        identity, scope = f"uvc:{index}", "host_index"
    name = info.name if info else f"Video input {index}"
    return Device(identity, "uvc", "OpenCV", name, None, "uvc_interface", scope, api)


def _enumerate():
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except (ImportError, OSError) as exc:
        raise CaptureError("UVC_DISCOVERY_UNAVAILABLE",
                           "install straintrace[capture-uvc] for non-opening discovery; a known uvc:<index> still works") from exc
    return [(item, _device(item.index, cv2.videoio_registry.getBackendName(item.backend), item, stable=True))
            for item in enumerate_cameras(_API)]


def discover(cti_files=()) -> list[Device]:
    return [device for _, device in _enumerate()]


def camera(identity: str, cti_files=()) -> UvcCamera:
    suffix = identity.removeprefix("uvc:")
    if re.fullmatch(r"os-[a-f0-9]{24}", suffix):
        return UvcCamera(identity=identity)
    if not suffix.isascii() or not suffix.isdecimal() or len(suffix) > 4 or str(int(suffix)) != suffix:
        raise CaptureError("INVALID_CAMERA_ID", f"invalid video input ID: {identity}")
    return UvcCamera(int(suffix))


class UvcCamera(Camera):
    trigger_method = "host_next_receipt"

    def __init__(self, index: int | None = None, *, identity: str | None = None):
        self._index = index
        self._identity = identity
        self._handle = None
        self._mode = None
        self._condition = threading.Condition()
        self._worker = None
        self._generation = 0
        self._sample_after = None
        self._latest = None
        self._failure = None
        self._stop = threading.Event()
        self._abandoned = False
        self._request = threading.Event()

    def open(self) -> None:
        if self._identity:
            matches = [(entry, device) for entry, device in _enumerate() if device.id == self._identity]
            if len(matches) != 1:
                raise CaptureError("CAMERA_ID_UNRESOLVED", f"cannot uniquely resolve {self._identity}")
            entry, self.device = matches[0]
            self._index = entry.index
        self._handle = cv2.VideoCapture(self._index, _API)
        if not self._handle.isOpened():
            raise CaptureError("CAMERA_UNAVAILABLE", f"cannot open video input {self._index}; check OS camera permissions and competing applications")
        if not self._identity:
            self.device = _device(self._index, self._handle.getBackendName())

    def close(self) -> None:
        if self._handle is not None and not self._abandoned:
            self.stop()
            self._handle.release()
            self._handle = None

    def features(self) -> dict[str, Feature]:
        features = {"pixel_format": Feature("pixel_format", "BGR8", choices=("BGR8",), writable=False)}
        if self._mode is not None:
            return {**self._cached_features, "pixel_format": features["pixel_format"]}
        for name, (prop, unit) in _PROPERTIES.items():
            value = self._handle.get(prop)
            if not math.isfinite(value):
                continue
            if name in ("width", "height") and value > 0:
                value = int(value)
            available = value != 0 or name == "fps"
            if available:
                features[name] = Feature(name, value if value != 0 else None, readable=value != 0,
                                         writable=self._mode is None, unit=unit, native_name=f"CAP_PROP_{prop}")
        return features

    def set_feature(self, name, value) -> None:
        feature = self.features().get(name)
        if feature is None:
            raise CaptureError("FEATURE_UNAVAILABLE", f"video input has no observable feature {name}")
        feature.validate(value)
        if type(value) not in (int, float):
            raise CaptureError("FEATURE_TYPE", f"{name} requires a number")
        prop, _ = _PROPERTIES[name]
        if not self._handle.set(prop, value):
            raise CaptureError("FEATURE_WRITE_FAILED", f"video driver rejected {name}={value}")
        observed = self._handle.get(prop)
        if not observed and name == "fps":
            # Some DirectShow drivers set FPS but never report it. Keep it unknown;
            # observed cadence is measured by Capture, independently of this request.
            return
        if not math.isclose(observed, value, abs_tol=1e-4, rel_tol=0.01):
            raise CaptureError("FEATURE_READBACK_MISMATCH", f"video driver returned {observed} after setting {name}={value}")

    def start(self, trigger="free", source=None) -> None:
        if trigger not in ("free", "software"):
            raise CaptureError("TRIGGER_UNSUPPORTED", "OpenCV video input has no hardware FrameStart")
        self._failure = self._latest = self._sample_after = None
        self._cached_features = {name: replace(feature, writable=False) for name, feature in self.features().items()}
        self._mode = trigger
        self._stop.clear()
        self._request.clear()
        self._worker_done = False
        self._worker = threading.Thread(target=self._pump, args=(trigger,), name=f"uvc-read-{self._index}", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        if self._abandoned:
            return
        if self._worker is not None:
            self._stop.set()
            self._request.set()
            self._worker.join(timeout=2)
            with self._condition:
                # Hand the handle back to the blocked reader. release() on a
                # handle while the driver owns it can itself block indefinitely.
                if self._worker.is_alive() and not self._worker_done:
                    self._abandoned = True
                    raise CaptureError("UVC_DRIVER_TIMEOUT", f"video driver input {self._index} is still blocked")
            self._worker = None
        self._mode = None

    def _get_image(self) -> Image:
        ok, pixels = self._handle.read()
        utc, monotonic = receipt_times()
        if not ok or pixels is None:
            raise CaptureError("UVC_FRAME_UNAVAILABLE", f"video input {self._index} stopped delivering images")
        return Image(pixels, "BGR8", 8, "bgr", utc, monotonic,
                     metadata={"pixel_source": "driver_decoded", "exposure_time_observed": False})

    def _pump(self, mode) -> None:
        try:
            while not self._stop.is_set():
                if mode == "free":
                    self._request.wait()
                    self._request.clear()
                    if self._stop.is_set():
                        break
                image = self._get_image()
                with self._condition:
                    self._latest = image
                    self._generation += 1
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._failure = exc
                self._condition.notify_all()
        finally:
            with self._condition:
                if self._abandoned:
                    self._handle.release()
                    self._handle = None
                self._worker_done = True

    def trigger(self) -> None:
        with self._condition:
            if self._mode != "software" or self._sample_after is not None:
                raise CaptureError("CAMERA_NOT_READY", f"{self.device.id} is not ready to sample a frame")
            self._sample_after = self._generation

    def read(self, timeout_s: float) -> Image | None:
        with self._condition:
            if self._mode == "free" and self._sample_after is None:
                self._sample_after = self._generation
                self._request.set()
            if self._mode not in ("free", "software") or self._sample_after is None:
                raise CaptureError("CAMERA_NOT_READY", f"{self.device.id} is not armed")
            self._condition.wait_for(
                lambda: self._failure is not None or self._generation > self._sample_after,
                timeout=timeout_s,
            )
            if self._failure is not None:
                raise self._failure
            if self._generation <= self._sample_after:
                return None
            self._sample_after = None
            return self._latest
