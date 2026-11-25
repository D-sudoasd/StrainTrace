"""GenTL acquisition through Harvesters and user-provided CTI producers."""

from __future__ import annotations

import os
import struct
from pathlib import Path
from threading import Lock
from urllib.parse import quote

import numpy as np

from ..config import CaptureError
from ._genapi import CHUNK_FIELDS, GenApi, exposure_marks, image_layout, optional
from .base import Camera, ClockReading, Device, Image, receipt_times


def _sdk():
    try:
        from genicam import genapi
        from harvesters.core import Harvester
    except (ImportError, OSError) as exc:
        raise CaptureError("GENTL_UNAVAILABLE", "install straintrace[capture-gentl] and a compatible CTI producer") from exc
    return Harvester, genapi


def _ctis(configured) -> list[Path]:
    if configured:
        return [Path(path) for path in configured]
    variable = f"GENICAM_GENTL{struct.calcsize('P') * 8}_PATH"
    paths = []
    for entry in os.environ.get(variable, "").split(os.pathsep):
        if entry and Path(entry).is_dir():
            paths.extend(sorted(Path(entry).glob("*.cti")))
    if not paths:
        raise CaptureError("GENTL_NO_CTI", f"supply trusted vendor .cti files or set {variable}")
    return list(dict.fromkeys(path.resolve() for path in paths))


def _load(harvester, cti_files) -> None:
    for path in _ctis(cti_files):
        if path.suffix.lower() != ".cti" or not path.is_file():
            raise CaptureError("GENTL_CTI_INVALID", f"cannot find a CTI producer at {path}")
        harvester.add_file(str(path), check_existence=True, check_validity=True)
    harvester.update()


def _device(info) -> Device:
    data = info.property_dict
    vendor, model = data.get("vendor") or "", data.get("model") or ""
    serial = data.get("serial_number") or None
    identity = f"gentl:{quote(vendor, safe='')}:{quote(serial or data['id_'], safe='')}"
    label = "vendor_emulator" if model == "Viky" and vendor == "Machine Vision Games Ltd." else "gentl_interface"
    return Device(identity, "gentl", vendor, model, serial, label, "serial_number" if serial else "producer_device_id",
                  data.get("tl_type"))


def discover(cti_files=()) -> list[Device]:
    harvester, _ = _sdk()
    with harvester() as client:
        _load(client, cti_files)
        devices = [_device(info) for info in client.device_info_list]
        if len({device.id for device in devices}) != len(devices):
            raise CaptureError("DUPLICATE_CAMERA_ID", "multiple producers expose the same camera ID; select one CTI per camera")
        return devices


def camera(identity: str, cti_files=()) -> GenTLCamera:
    return GenTLGroup(cti_files).camera(identity)


class GenTLGroup:
    """A producer shares one client across its simultaneously open devices."""

    def __init__(self, cti_files):
        self._Harvester, self._api = _sdk()
        self._cti_files = cti_files
        self._client = None
        self._acquirers = {}
        self._mutex = Lock()

    def camera(self, identity):
        return GenTLCamera(identity, self)

    def open(self, identity):
        with self._mutex:
            if identity in self._acquirers:
                raise CaptureError("CAMERA_ALREADY_OPEN", f"GenTL camera {identity} already connected")
            try:
                if self._client is None:
                    self._client = self._Harvester()
                    _load(self._client, self._cti_files)
                matches = [(index, _device(info)) for index, info in enumerate(self._client.device_info_list)
                           if _device(info).id == identity]
                if len(matches) != 1:
                    raise CaptureError("CAMERA_ID_UNRESOLVED",
                                       f"expected one GenTL camera for {identity}, found {len(matches)}")
                index, device = matches[0]
                acquirer = self._client.create(index)
                self._acquirers[identity] = acquirer
                return device, acquirer, self._api
            except Exception:
                self._close_empty_client()
                raise

    def close(self, identity):
        with self._mutex:
            try:
                acquirer = self._acquirers.pop(identity, None)
                if acquirer is not None:
                    acquirer.destroy()
            finally:
                self._close_empty_client()

    def _close_empty_client(self):
        if not self._acquirers and self._client is not None:
            try:
                self._client.reset()
            finally:
                self._client = None


class GenTLCamera(Camera):
    def __init__(self, identity: str, group: GenTLGroup):
        self._identity = identity
        self._group = group
        self._acquirer = self._nodes = None
        self._ptp_status = None
        self._timestamp_selector = None
        self._camera_tick_frequency = None
        self._active = False

    def open(self) -> None:
        self.device, self._acquirer, api = self._group.open(self._identity)
        self._nodes = GenApi(self._acquirer.remote_device.node_map, genicam_api=api)

    def close(self) -> None:
        if self._acquirer is not None:
            try:
                self.stop()
            finally:
                try:
                    self._group.close(self._identity)
                finally:
                    self._acquirer = self._nodes = None

    def features(self):
        return self._nodes.public_features()

    def set_feature(self, name, value) -> None:
        self._nodes.write(name, value)

    def set_features(self, settings) -> None:
        self._nodes.configure(settings)

    def start(self, trigger="free", source=None) -> None:
        self._nodes.trigger_setup(trigger, source)
        fmt = self._nodes.get("PixelFormat")
        if fmt is not None:
            image_layout(fmt)
        ptp = self._nodes.feature("ptp_status")
        self._ptp_status = str(ptp.value) if ptp and ptp.value is not None else "unverified"
        selected = self._nodes.feature("chunk_timestamp_selector")
        self._timestamp_selector = selected.value if selected else None
        tick = self._nodes.feature("timestamp_frequency")
        self._camera_tick_frequency = tick.value if tick and tick.value and tick.value > 0 else None
        self._acquirer.start()
        self._active = True

    def stop(self) -> None:
        if self._acquirer is not None and self._active:
            self._acquirer.stop()
            self._active = False

    def trigger(self) -> None:
        try:
            self._acquirer.remote_device.node_map.get_node("TriggerSoftware").execute()
        except Exception as exc:
            raise CaptureError("SOFTWARE_TRIGGER_FAILED", f"{self.device.id}: {exc}") from exc

    def read(self, timeout_s: float) -> Image | None:
        buffer = self._acquirer.try_fetch(timeout=timeout_s)
        utc, monotonic = receipt_times()
        if buffer is None:
            return None
        with buffer:
            if optional(lambda: buffer.module.is_incomplete, False):
                raise CaptureError("GENTL_INCOMPLETE_BUFFER", f"{self.device.id} returned an incomplete image buffer")
            payload = buffer.payload
            components = getattr(payload, "components", ())
            if len(components) != 1:
                raise CaptureError("GENTL_MULTIPART_UNSUPPORTED", f"expected one image, received {len(components)} components")
            component = components[0]
            fmt = component.data_format
            bits, color = image_layout(fmt)
            height, width = component.height, component.width
            shape = (height, width) if color in ("mono", "bayer") else (height, width, len(color))
            try:
                pixels = np.array(component.data.reshape(shape), copy=True, order="C")
            except ValueError as exc:
                raise CaptureError("GENTL_BAD_PAYLOAD", f"payload does not match {fmt} {width} × {height}") from exc
            frame_id = optional(lambda: buffer.module.frame_id)
            native_id = int(frame_id) if frame_id is not None and 0 <= frame_id < 2**64 - 1 else None
            ticks = optional(lambda: buffer.timestamp)
            frequency = optional(lambda: buffer.timestamp_frequency)
            device_time = ClockReading(
                int(ticks), "ticks", "gentl_buffer_timestamp", f"{self.device.id}:gentl_timestamp",
                "producer_clock_unspecified_epoch", self._ptp_status,
                tick_frequency_hz=int(frequency) if frequency and frequency > 0 else None,
            ) if ticks else None
            metadata = {}
            for feature in CHUNK_FIELDS:
                value = self._nodes.get(feature)
                if value is not None:
                    metadata[feature] = value
            start, end = exposure_marks(metadata, self._timestamp_selector,
                                        clock_id=f"{self.device.id}:camera_chunk_timestamp",
                                        frequency=self._camera_tick_frequency, sync_status=self._ptp_status)
            return Image(pixels, fmt, bits, color, utc, monotonic, native_id, native_id,
                         device_time, start, end, metadata)
