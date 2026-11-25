"""Native Basler acquisition through the optional pypylon SDK."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np

from ..config import CaptureError
from ._genapi import CHUNK_FIELDS, GenApi, exposure_marks, image_layout, optional
from .base import Camera, ClockReading, Device, Image, receipt_times


_PACKED_MOSAIC = re.compile(r"(Mono|Bayer..)(10|12|14)(p|Packed)")


def _sdk():
    try:
        from pypylon import genicam, pylon
    except (ImportError, OSError) as exc:
        raise CaptureError("PYLON_UNAVAILABLE", "install straintrace[capture-pylon] and the required Basler runtime") from exc
    return pylon, genicam


def _device(info) -> Device:
    serial = info.GetSerialNumber()
    transport = info.GetDeviceClass()
    emulated = transport == "BaslerCamEmu"
    name = info.GetFullName()
    return Device(f"pylon:{serial or name}", "pylon", info.GetVendorName(), info.GetModelName(), serial or None,
                  "vendor_emulator" if emulated else "native_interface", "serial_number" if serial else "sdk_device_name",
                  transport)


def discover(cti_files=()) -> list[Device]:
    pylon, _ = _sdk()
    return [_device(info) for info in pylon.TlFactory.GetInstance().EnumerateDevices()]


def camera(identity: str, cti_files=()) -> PylonCamera:
    return PylonCamera(identity)


class PylonCamera(Camera):
    def __init__(self, identity: str):
        self._pylon, self._api = _sdk()
        self._identity = identity
        self._instant = None
        self._nodes = None
        self._format = None
        self._tick_frequency = None
        self._clock_id = None
        self._ptp_status = None
        self._timestamp_selector = None
        self._chunk_enabled = False

    def open(self) -> None:
        factory = self._pylon.TlFactory.GetInstance()
        matches = [info for info in factory.EnumerateDevices() if _device(info).id == self._identity]
        if len(matches) != 1:
            raise CaptureError("CAMERA_ID_UNRESOLVED", f"expected one Basler camera for {self._identity}, found {len(matches)}")
        self.device = _device(matches[0])
        self._instant = self._pylon.InstantCamera(factory.CreateDevice(matches[0]))
        self._instant.Open()
        self._nodes = GenApi(self._instant.GetNodeMap(), pylon_api=self._api)

    def close(self) -> None:
        if self._instant is not None:
            try:
                self.stop()
            finally:
                try:
                    self._instant.Close()
                finally:
                    self._instant = None
                    self._nodes = None

    def features(self):
        features = self._nodes.public_features()
        if "pixel_format" in features:
            node = features["pixel_format"]
            choices = tuple(value for value in node.choices if not _PACKED_MOSAIC.fullmatch(value))
            features["pixel_format"] = replace(node, choices=choices, writable=node.writable and bool(choices))
        return features

    def set_feature(self, name, value) -> None:
        if name == "pixel_format":
            self._validate_pixel_format(value)
        self._nodes.write(name, value)

    def set_features(self, settings) -> None:
        if "pixel_format" in settings:
            self._validate_pixel_format(settings["pixel_format"])
        self._nodes.configure(settings)

    @staticmethod
    def _validate_pixel_format(value):
        image_layout(value)
        if _PACKED_MOSAIC.fullmatch(value):
            raise CaptureError("PIXEL_FORMAT_UNSUPPORTED", "select the unpacked Basler pixel format for this bit depth")

    def start(self, trigger="free", source=None) -> None:
        self._nodes.trigger_setup(trigger, source)
        self._format = self._nodes.get("PixelFormat")
        self._validate_pixel_format(self._format)
        tick = self._nodes.feature("timestamp_frequency")
        ptp = self._nodes.feature("ptp_status")
        self._tick_frequency = tick.value if tick and tick.value and tick.value > 0 else None
        self._ptp_status = str(ptp.value) if ptp and ptp.value is not None else "unverified"
        self._clock_id = f"{self.device.id}:pylon_timestamp"
        selected = self._nodes.feature("chunk_timestamp_selector")
        self._timestamp_selector = selected.value if selected else None
        self._chunk_enabled = bool(self._nodes.get("ChunkModeActive"))
        self._instant.StartGrabbing(self._pylon.GrabStrategy_OneByOne)

    def stop(self) -> None:
        if self._instant is not None and self._instant.IsGrabbing():
            self._instant.StopGrabbing()

    def trigger(self) -> None:
        if self._instant.CanWaitForFrameTriggerReady():
            self._instant.WaitForFrameTriggerReady(500, self._pylon.TimeoutHandling_ThrowException)
        self._instant.ExecuteSoftwareTrigger()

    def read(self, timeout_s: float) -> Image | None:
        if self._instant.IsCameraDeviceRemoved():
            raise CaptureError("CAMERA_DISCONNECTED", f"{self.device.id} was removed")
        result = self._instant.RetrieveResult(max(1, round(timeout_s * 1000)), self._pylon.TimeoutHandling_Return)
        utc, monotonic = receipt_times()
        with result:
            if not result.IsValid():
                return None
            if not result.GrabSucceeded():
                raise CaptureError("PYLON_GRAB_FAILED", f"pylon {result.ErrorCode}: {result.ErrorDescription}")
            bits, color = image_layout(self._format)
            pixels = np.array(result.GetArray(), copy=True, order="C")
            block = result.GetBlockID()
            # SDK sentinel IDs and transports without a block counter are common.
            native = int(block) if 0 < block < 2**64 - 1 else None
            timestamp = result.GetTimeStamp()
            device_time = ClockReading(
                int(timestamp), "ticks", "pylon_grab_timestamp", self._clock_id, "device_clock_unspecified_epoch",
                self._ptp_status, tick_frequency_hz=self._tick_frequency,
            ) if timestamp else None
            metadata = {
                "pylon_image_number": int(result.GetImageNumber()), "pylon_skipped_images": int(result.GetNumberOfSkippedImages()),
            }
            if self._chunk_enabled:
                for feature in CHUNK_FIELDS:
                    value = optional(lambda: getattr(result, feature).Value)
                    if value is not None:
                        metadata[feature] = value
            start, end = exposure_marks(metadata, self._timestamp_selector, clock_id=self._clock_id,
                                        frequency=self._tick_frequency, sync_status=self._ptp_status)
            return Image(pixels, self._format, bits, color, utc, monotonic, native, native,
                         device_time, start, end, metadata)
