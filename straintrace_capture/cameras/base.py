"""Camera contracts independent of tkinter, analysis and vendor SDKs."""

from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import CaptureError


HOST_NEXT_RECEIPT = "host_next_receipt"


@dataclass(frozen=True)
class Device:
    id: str
    backend: str
    vendor: str
    model: str
    serial: str | None
    support: str
    identity_scope: str = "serial_number"
    transport: str | None = None


@dataclass(frozen=True)
class TriggerSupport:
    software: bool | None = None
    hardware: bool | None = None
    basis: str = "unknown"
    missing_nodes: tuple[str, ...] = ()
    error: str | None = None
    physical_inputs: tuple[str, ...] | None = None
    software_method: str | None = None


def is_physical_trigger_input(value: str | None) -> bool:
    return isinstance(value, str) and re.fullmatch(r"(?:Line|CC|CxpTrigger|CXPTrigger)[0-9]+", value) is not None


@dataclass(frozen=True)
class TriggerWaitSample:
    line: str | None = None
    asserted: bool | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Feature:
    name: str
    value: Any
    minimum: int | float | None = None
    maximum: int | float | None = None
    increment: int | float | None = None
    choices: tuple[str, ...] = ()
    readable: bool = True
    writable: bool = True
    unit: str | None = None
    native_name: str | None = None

    def validate(self, value: Any) -> None:
        if not self.writable:
            raise CaptureError("FEATURE_READ_ONLY", f"{self.name} is read-only in the current state")
        if self.choices and value not in self.choices:
            raise CaptureError("FEATURE_CHOICE", f"{self.name} must be one of {', '.join(self.choices)}")
        if type(self.value) in (int, float) or self.minimum is not None or self.maximum is not None:
            if type(value) not in (int, float) or abs(value) > 1e15 or not math.isfinite(value):
                raise CaptureError("FEATURE_TYPE", f"{self.name} requires a finite number")
            if type(self.value) is int and type(value) is not int:
                raise CaptureError("FEATURE_TYPE", f"{self.name} requires an integer")
            if self.minimum is not None and value < self.minimum or self.maximum is not None and value > self.maximum:
                raise CaptureError("FEATURE_RANGE", f"{self.name}={value} outside [{self.minimum}, {self.maximum}]")
            if self.increment and self.minimum is not None:
                steps = (value - self.minimum) / self.increment
                if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-7):
                    raise CaptureError("FEATURE_INCREMENT", f"{self.name} must advance in steps of {self.increment}")
        if type(self.value) is bool and type(value) is not bool:
            raise CaptureError("FEATURE_TYPE", f"{self.name} requires a boolean")


@dataclass(frozen=True)
class ClockReading:
    value: int
    unit: str
    source: str
    clock_id: str
    reference: str
    sync_status: str = "unverified"
    uncertainty_ns: int | None = None
    tick_frequency_hz: int | None = None
    status: str = "recorded"


@dataclass(frozen=True)
class Image:
    pixels: np.ndarray
    pixel_format: str
    significant_bits: int
    color_model: str
    received_utc_ns: int
    received_monotonic_ns: int
    native_frame_id: int | str | None = None
    native_counter: int | None = None
    device_time: ClockReading | None = None
    exposure_start: ClockReading | None = None
    exposure_end: ClockReading | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    auxiliary_images: tuple[Image, ...] = ()

    def __post_init__(self) -> None:
        pixels = self.pixels
        allowed = ("mono", "bayer", "rgb", "bgr", "rgba", "bgra")
        color = self.color_model
        if color not in allowed or pixels.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise CaptureError("UNSUPPORTED_IMAGE", f"unsupported array {pixels.dtype} / {color}")
        if color in ("mono", "bayer"):
            valid_shape = pixels.ndim == 2
        else:
            valid_shape = pixels.ndim == 3 and pixels.shape[2] == len(color)
        if not valid_shape or not pixels.size or not 1 <= self.significant_bits <= pixels.dtype.itemsize * 8:
            raise CaptureError("UNSUPPORTED_IMAGE", f"invalid array shape or bit depth: {pixels.shape}")
        receipt = self.received_utc_ns, self.received_monotonic_ns, self.native_frame_id
        for part in self.auxiliary_images:
            if part.auxiliary_images or receipt != (part.received_utc_ns, part.received_monotonic_ns, part.native_frame_id):
                raise CaptureError("IMAGE_PART_RECEIPT_MISMATCH", "image parts must belong to one received frame")
        # Borrowed SDK buffers must already have been copied by the backend.
        pixels.setflags(write=False)

    @property
    def raw_bytes(self) -> int:
        size = self.pixels.nbytes
        return size + sum(part.pixels.nbytes for part in self.auxiliary_images) if self.auxiliary_images else size


def receipt_times() -> tuple[int, int]:
    return time.time_ns(), time.perf_counter_ns()


class Camera(ABC):
    device: Device
    trigger_method = "software_frame_start"

    def trigger_wait_monitor(self):
        """Return a context for sampling an already configured hardware wait output."""
        return nullcontext(lambda: TriggerWaitSample(reason="driver_unavailable"))

    @abstractmethod
    def open(self) -> None:
        """Connect to the selected device."""

    @abstractmethod
    def close(self) -> None:
        """Release device and SDK resources."""

    @abstractmethod
    def features(self) -> dict[str, Feature]:
        """Query current values, access, and constraints."""

    @abstractmethod
    def set_feature(self, name: str, value: Any) -> None:
        """Write a feature or report a device-specific rejection."""

    @abstractmethod
    def start(self, trigger: str = "free", source: str | None = None) -> None:
        """Start free-running or arm software/physical FrameStart."""

    @abstractmethod
    def stop(self) -> None:
        """Stop or disarm acquisition."""

    @abstractmethod
    def read(self, timeout_s: float) -> Image | None:
        """Transfer an owned image; None indicates a timeout."""

    def trigger(self) -> None:
        raise CaptureError("TRIGGER_UNSUPPORTED", f"{self.device.id} does not support software trigger")

    def trigger_support(self) -> TriggerSupport:
        """Report evidence from an open camera without arming or editing it."""
        return TriggerSupport()

    def set_features(self, settings) -> None:
        for name, value in settings.items():
            self.set_feature(name, value)

    def __enter__(self) -> Camera:
        try:
            self.open()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.close()
