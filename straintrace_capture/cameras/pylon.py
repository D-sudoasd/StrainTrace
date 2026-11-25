"""Native Basler acquisition through the optional pypylon SDK."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import CaptureError
from ._genapi import CHUNK_FIELDS, GenApi, exposure_marks, image_layout, is_packed_mono_or_bayer, optional
from .base import Camera, ClockReading, Device, Image, receipt_times


def _sdk():
    try:
        from pypylon import genicam, pylon
    except (ImportError, OSError) as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "pypylon":
            raise CaptureError("PYLON_UNAVAILABLE", "install straintrace[capture-pylon]") from exc
        raise CaptureError("PYLON_SDK_LOAD_FAILED", f"could not load the Basler SDK: {exc}") from exc
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
        self._packed_type = None
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
                    if self._instant.IsCameraDeviceRemoved():
                        self._instant.DestroyDevice()
                    else:
                        self._instant.Close()
                finally:
                    self._instant = None
                    self._nodes = None

    def features(self):
        features = self._nodes.public_features()
        if "pixel_format" in features:
            node = features["pixel_format"]
            choices = tuple(value for value in node.choices if not is_packed_mono_or_bayer(value)
                            or optional(lambda: self._validate_pixel_format(value)) is not None)
            features["pixel_format"] = replace(node, choices=choices, writable=node.writable and bool(choices))
        return features

    def trigger_support(self):
        return self._nodes.trigger_support()

    def trigger_wait_monitor(self):
        return self._nodes.trigger_wait_monitor()

    def set_feature(self, name, value) -> None:
        if name == "pixel_format":
            self._validate_pixel_format(value)
        self._nodes.write(name, value)

    def set_features(self, settings) -> None:
        if "pixel_format" in settings:
            self._validate_pixel_format(settings["pixel_format"])
        self._nodes.configure(settings)

    def _validate_pixel_format(self, value):
        bits, color = image_layout(value)
        if not is_packed_mono_or_bayer(value):
            return None
        symbol = value
        if value.endswith(("Packed", "packed")):
            # The SDK spells the legacy mono and Bayer enum suffixes differently.
            symbol = value[:-6] + ("packed" if color == "mono" else "Packed")
        pixel_type = getattr(self._pylon, f"PixelType_{symbol}", None)
        if bits not in (10, 12) or pixel_type is None:
            raise CaptureError("PIXEL_FORMAT_UNSUPPORTED", f"Basler packed format {value} is not supported by this adapter")
        return pixel_type

    def start(self, trigger="free", source=None) -> None:
        self._nodes.trigger_setup(trigger, source)
        self._format = self._nodes.get("PixelFormat")
        self._packed_type = self._validate_pixel_format(self._format)
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
        if self._instant is not None and not self._instant.IsCameraDeviceRemoved() and self._instant.IsGrabbing():
            self._instant.StopGrabbing()

    def trigger(self) -> None:
        if self._instant.CanWaitForFrameTriggerReady():
            self._instant.WaitForFrameTriggerReady(500, self._pylon.TimeoutHandling_ThrowException)
        self._instant.ExecuteSoftwareTrigger()

    def _check_connection(self):
        if self._instant.IsCameraDeviceRemoved():
            raise CaptureError("CAMERA_DISCONNECTED", f"{self.device.id} was removed")

    def read(self, timeout_s: float) -> Image | None:
        self._check_connection()
        try:
            result = self._instant.RetrieveResult(max(1, round(timeout_s * 1000)), self._pylon.TimeoutHandling_Return)
        except self._api.GenericException:
            self._check_connection()
            raise
        utc, monotonic = receipt_times()
        with result:
            if not result.IsValid():
                return None
            if not result.GrabSucceeded():
                self._check_connection()
                raise CaptureError("PYLON_GRAB_FAILED", f"pylon {result.ErrorCode}: {result.ErrorDescription}")
            bits, color = image_layout(self._format)
            if self._packed_type is not None:
                actual_type = result.GetPixelType()
                if actual_type != self._packed_type:
                    raise CaptureError("PYLON_PIXEL_FORMAT_MISMATCH",
                                       f"{self.device.id}: expected {self._format}, received pixel type 0x{actual_type:08x}")
            try:
                # GetArray unpacks the SDK buffer while preserving the sensor samples and Bayer pattern.
                pixels = np.array(result.GetArray(), copy=True, order="C")
            except (ValueError, self._api.GenericException) as exc:
                raise CaptureError("PYLON_BAD_PAYLOAD", f"{self.device.id}: could not read {self._format} samples: {exc}") from exc
            if self._packed_type is not None and (pixels.dtype != np.uint16
                                                  or pixels.shape != (result.GetHeight(), result.GetWidth())):
                raise CaptureError("PYLON_BAD_PAYLOAD", f"{self.device.id}: invalid decoded {self._format} array")
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
