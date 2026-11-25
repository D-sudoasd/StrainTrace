"""Deterministic speckle camera for development and setup rehearsals."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from ..config import CaptureError
from .base import Camera, Device, Feature, Image, receipt_times


FORMATS = {"Mono8": 8, "Mono10": 10, "Mono12": 12, "Mono14": 14, "Mono16": 16}
FEATURES = {
    "width": Feature("width", 640, 32, 4096, 2, unit="px"),
    "height": Feature("height", 480, 32, 3072, 2, unit="px"),
    "offset_x": Feature("offset_x", 0, 0, 4096, 2, unit="px"),
    "offset_y": Feature("offset_y", 0, 0, 3072, 2, unit="px"),
    "pixel_format": Feature("pixel_format", "Mono8", choices=tuple(FORMATS)),
    "fps": Feature("fps", 25.0, 0.01, 1000, unit="Hz"),
    "exposure_us": Feature("exposure_us", 2000.0, 1, 1000000, unit="us"),
    "gain_db": Feature("gain_db", 0.0, 0, 24, unit="dB"),
    "motion_px": Feature("motion_px", 0.16, -4, 4, unit="px/frame"),
    "strain_per_frame": Feature("strain_per_frame", 0.0005, -0.01, 0.01, unit="1/frame"),
    "disconnect_after": Feature("disconnect_after", 0, 0, 10000000, 1, unit="frames"),
}


def simulated_device(index: int) -> Device:
    return Device(f"sim:{index}", "sim", "StrainTrace", "Speckle simulator", f"SIM-{index:04}", "simulated")


def discover(cti_files=()) -> list[Device]:
    return [simulated_device(0), simulated_device(1)]


def camera(identity: str, cti_files=()) -> SimulatedCamera:
    index = identity.removeprefix("sim:")
    if not index.isascii() or not index.isdecimal() or len(index) > 6 or str(int(index)) != index:
        raise CaptureError("INVALID_CAMERA_ID", f"invalid speckle simulator ID: {identity}")
    return SimulatedCamera(int(index))


class SimulatedCamera(Camera):
    def __init__(self, index: int):
        self.device = simulated_device(index)
        self._index = index
        self._values = {name: feature.value for name, feature in FEATURES.items()}
        self._mode: str | None = None
        self._reference: np.ndarray | None = None
        self._counter = 0
        self._pending = False
        self._next_frame = 0.0

    def open(self) -> None:
        self._counter = 0

    def close(self) -> None:
        self.stop()
        self._reference = None

    def features(self) -> dict[str, Feature]:
        result = {
            name: replace(feature, value=self._values[name], writable=self._mode is None)
            for name, feature in FEATURES.items()
        }
        for size, offset in (("width", "offset_x"), ("height", "offset_y")):
            extent = FEATURES[size].maximum
            result[size] = replace(result[size], maximum=extent - self._values[offset])
            result[offset] = replace(result[offset], maximum=extent - self._values[size])
        return result

    def set_feature(self, name: str, value: Any) -> None:
        feature = self.features().get(name)
        if feature is None:
            raise CaptureError("FEATURE_UNAVAILABLE", f"simulator has no feature {name}")
        feature.validate(value)
        self._values[name] = value
        self._reference = None

    def _make_reference(self) -> np.ndarray:
        values = self._values
        width, height = values["width"], values["height"]
        x, y = values["offset_x"], values["offset_y"]
        rng = np.random.default_rng(3201 + self._index)
        pixels = rng.random((height + y, width + x), dtype=np.float32)
        pixels = cv2.GaussianBlur(pixels, (0, 0), 1.15)[y:, x:]
        scale = min(4.0, values["exposure_us"] / 2000 * 10 ** (values["gain_db"] / 20))
        bits = FORMATS[values["pixel_format"]]
        reference = (np.clip((pixels - 0.5) * 2.6 + 0.5, 0, 1) * scale).clip(0, 1)
        return np.ascontiguousarray(np.rint(reference * ((1 << bits) - 1)).astype(np.uint8 if bits == 8 else np.uint16))

    def start(self, trigger: str = "free", source: str | None = None) -> None:
        if trigger not in ("free", "software"):
            raise CaptureError("TRIGGER_UNSUPPORTED", "the speckle simulator has no hardware trigger input")
        if self._reference is None:
            self._reference = self._make_reference()
        self._pending = False
        self._next_frame = time.perf_counter()
        self._mode = trigger

    def stop(self) -> None:
        self._mode = None
        self._pending = False

    def trigger(self) -> None:
        if self._mode != "software" or self._pending:
            raise CaptureError("CAMERA_NOT_READY", f"{self.device.id} is not armed for a new software trigger")
        self._pending = True

    def read(self, timeout_s: float) -> Image | None:
        if self._mode is None:
            raise CaptureError("CAMERA_NOT_READY", f"{self.device.id} is not acquiring")
        if self._mode == "software" and not self._pending:
            return None
        if self._mode == "free":
            delay = max(0.0, self._next_frame - time.perf_counter())
            if delay > timeout_s:
                time.sleep(timeout_s)
                return None
            time.sleep(delay)
            self._next_frame = max(self._next_frame + 1 / self._values["fps"], time.perf_counter())
        if self._values["disconnect_after"] and self._counter >= self._values["disconnect_after"]:
            raise CaptureError("CAMERA_DISCONNECTED", f"{self.device.id} simulated disconnect")
        self._pending = False
        height, width = self._reference.shape
        motion = self._counter * self._values["motion_px"]
        strain = self._counter * self._values["strain_per_frame"]
        pixels = self._reference
        if motion or strain:
            transform = np.float32([[1, 0, motion], [0, 1 + strain, -height / 2 * strain]])
            pixels = cv2.warpAffine(pixels, transform, (width, height), borderMode=cv2.BORDER_REFLECT_101)
        self._counter += 1
        pixel_format = self._values["pixel_format"]
        return Image(pixels, pixel_format, FORMATS[pixel_format], "mono", *receipt_times(), self._counter, self._counter)
