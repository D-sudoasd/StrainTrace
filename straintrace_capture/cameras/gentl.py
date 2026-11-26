"""GenTL acquisition through Harvesters and user-provided CTI producers."""

from __future__ import annotations

import os
import struct
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Lock
from urllib.parse import quote

import numpy as np

from ..config import CaptureError
from ._genapi import CHUNK_FIELDS, GenApi, exposure_marks, image_layout, optional
from ._gentl_multipart import AUTO_PART, IMAGE_PART, SAVE_ALL_2D, parse_image_part, part_pixels, select_image_part
from ._gentl_packed import packed_format, packed_samples
from .base import Camera, ClockReading, Device, Feature, Image, receipt_times


_SINGLE_IMAGE = object()


def _sdk():
    try:
        from genicam import genapi
        from harvesters.core import Harvester
    except (ImportError, OSError) as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name in ("genicam", "harvesters"):
            raise CaptureError("GENTL_UNAVAILABLE", "install straintrace[capture-gentl]") from exc
        raise CaptureError("GENTL_SDK_LOAD_FAILED", f"could not load the GenTL SDK: {exc}") from exc
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


@contextmanager
def _without_loader_dialogs():
    if os.name != "nt":
        yield
        return
    from ctypes import WinDLL, WinError, get_last_error

    kernel = WinDLL("kernel32", use_last_error=True)
    old_mode = kernel.GetThreadErrorMode()
    # SEM_FAILCRITICALERRORS returns DLL loader errors to the caller.
    if not kernel.SetThreadErrorMode(old_mode | 0x0001, None):
        raise WinError(get_last_error())
    try:
        yield
    finally:
        kernel.SetThreadErrorMode(old_mode, None)


def _load(harvester, cti_files) -> None:
    files = _ctis(cti_files)
    with _without_loader_dialogs():
        for path in files:
            if path.suffix.lower() != ".cti" or not path.is_file():
                raise CaptureError("GENTL_CTI_INVALID", f"cannot find a CTI producer at {path}")
            try:
                harvester.add_file(str(path), check_existence=True, check_validity=True)
            except OSError as exc:
                raise CaptureError("GENTL_CTI_LOAD_FAILED", f"{path}: {exc}") from exc
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
        self._interfaces = ()
        self._acquirers = {}
        self._mutex = Lock()

    def camera(self, identity, *, refresh=False):
        return GenTLCamera(identity, self, refresh=refresh)

    def open(self, identity, *, refresh=False):
        with self._mutex:
            if identity in self._acquirers:
                raise CaptureError("CAMERA_ALREADY_OPEN", f"GenTL camera {identity} already connected")
            try:
                if self._client is None:
                    self._client = self._Harvester()
                    _load(self._client, self._cti_files)
                    self._interfaces = tuple({id(info.parent): info.parent
                                              for info in self._client.device_info_list}.values())
                elif refresh:
                    self._refresh_devices()
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

    def _refresh_devices(self):
        from harvesters.core import DeviceInfo

        # Harvester.update() destroys its acquirers. Update the lists on the
        # existing interfaces instead, without closing any camera handles.
        devices = []
        for interface in self._interfaces:
            interface.update_device_info_list(self._client.timeout_for_update)
            devices.extend(DeviceInfo(module=info, parent=interface) for info in interface.device_info_list)
        self._client.device_info_list[:] = devices

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
                self._interfaces = ()


class GenTLCamera(Camera):
    def __init__(self, identity: str, group: GenTLGroup, *, refresh=False):
        self._identity = identity
        self._group = group
        self._refresh = refresh
        self._acquirer = self._nodes = None
        self._ptp_status = None
        self._timestamp_selector = None
        self._camera_tick_frequency = None
        self._packed = None
        self._part_choice = AUTO_PART
        self._save_all_2d = False
        self._part_selector = None
        self._part_binding = None
        self._all_part_bindings = None
        self.image_part_options = ()
        self._active = False

    def open(self) -> None:
        self.device, self._acquirer, api = self._group.open(self._identity, refresh=self._refresh)
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
                    self._part_binding = None
                    self._all_part_bindings = None
                    self.image_part_options = ()

    def features(self):
        return {**self._nodes.public_features(), IMAGE_PART: Feature(IMAGE_PART, self._part_choice),
                SAVE_ALL_2D: Feature(SAVE_ALL_2D, self._save_all_2d)}

    def trigger_support(self):
        return self._nodes.trigger_support()

    def trigger_wait_monitor(self):
        return self._nodes.trigger_wait_monitor()

    def set_feature(self, name, value) -> None:
        self.set_features({name: value})

    def set_features(self, settings) -> None:
        choice = settings.get(IMAGE_PART, self._part_choice)
        selector = parse_image_part(choice)
        save_all = settings.get(SAVE_ALL_2D, self._save_all_2d)
        Feature(SAVE_ALL_2D, self._save_all_2d).validate(save_all)
        self._nodes.configure({name: value for name, value in settings.items() if name not in (IMAGE_PART, SAVE_ALL_2D)})
        self._part_choice, self._part_selector = choice, selector
        self._save_all_2d = save_all
        if settings:
            self._part_binding = None
            self._all_part_bindings = None

    def start(self, trigger="free", source=None) -> None:
        self._nodes.trigger_setup(trigger, source)
        fmt = self._nodes.get("PixelFormat")
        # The remote node can refer to a different component than the delivered 2D image.
        self._packed = packed_format(fmt)
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

    def _read_component(self, buffer, utc, monotonic):
        components = getattr(buffer.payload, "components", ())
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
        return self._image(pixels, fmt, bits, color, utc, monotonic, buffer.module,
                           optional(lambda: buffer.timestamp), optional(lambda: buffer.timestamp_frequency))

    def read(self, timeout_s: float) -> Image | None:
        from genicam.gentl import GenericException
        from genicam.gentl import PAYLOADTYPE_INFO_IDS as Payload
        from genicam.gentl import PIXELFORMAT_NAMESPACE_IDS as Namespace
        from harvesters.core import Buffer
        from harvesters.util.pfnc import dict_by_ints

        raw = self._acquirer.try_fetch(timeout=timeout_s, is_raw=True)
        utc, monotonic = receipt_times()
        if raw is None:
            return None
        try:
            if optional(lambda: raw.is_complete()) is False or optional(lambda: raw.has_received_larger_payload(), False):
                raise CaptureError("GENTL_INCOMPLETE_BUFFER", f"{self.device.id} returned an incomplete image buffer")
            payload = optional(lambda: raw.payload_type)
            if payload == Payload.PAYLOAD_TYPE_MULTI_PART:
                return self._read_multipart(raw, utc, monotonic)
            if self._part_selector is not None or self._part_binding not in (None, _SINGLE_IMAGE):
                raise CaptureError("GENTL_PART_MISSING", "当前缓冲区没有采集计划选定的图像部件。")
            code = optional(lambda: raw.pixel_format)
            planned = self._packed
            layout = planned if code is None or (planned is not None and planned.code == code) else packed_format(dict_by_ints.get(code))
            if layout is None or payload not in (Payload.PAYLOAD_TYPE_IMAGE, Payload.PAYLOAD_TYPE_CHUNK_DATA):
                try:
                    wrapped = Buffer(module=raw, node_map=self._acquirer.remote_device.node_map, acquire=self._acquirer)
                except GenericException:
                    return None
                except (ValueError, KeyError) as exc:
                    raise CaptureError("GENTL_BAD_PAYLOAD", f"{self.device.id}: could not interpret the image buffer: {exc}") from exc
                image = self._read_component(wrapped, utc, monotonic)
                self._part_binding = _SINGLE_IMAGE
                return image
            namespace = optional(lambda: raw.pixel_format_name_space)
            if namespace not in (None, Namespace.PIXELFORMAT_NAMESPACE_UNKNOWN, Namespace.PIXELFORMAT_NAMESPACE_GEV,
                                 Namespace.PIXELFORMAT_NAMESPACE_PFNC_32BIT):
                raise CaptureError("GENTL_BAD_PAYLOAD", f"unsupported packed pixel format namespace {namespace}")
            width = optional(lambda: raw.width) or self._nodes.get("Width")
            height = optional(lambda: raw.delivered_image_height) or optional(lambda: raw.height) or self._nodes.get("Height")
            legacy_chunks = payload == Payload.PAYLOAD_TYPE_CHUNK_DATA
            has_chunks = True if legacy_chunks else optional(lambda: raw.is_containing_chunk_data())
            if has_chunks is None and self._nodes.get("ChunkModeActive") is False:
                has_chunks = False
            offset = optional(lambda: raw.image_offset, None if legacy_chunks else 0)
            pixels = packed_samples(layout, raw.raw_buffer, width=width, height=height, offset=offset,
                                    filled=self._filled_bytes(raw),
                                    padding_x=optional(lambda: raw.padding_x, 0), padding_y=optional(lambda: raw.padding_y, 0),
                                    includes_chunks=has_chunks)
            ticks, frequency = self._raw_clock(raw)
            bits, color = image_layout(layout.name)
            image = self._image(pixels, layout.name, bits, color, utc, monotonic, raw, ticks, frequency)
            self._part_binding = _SINGLE_IMAGE
            return image
        finally:
            raw.parent.queue_buffer(raw)

    def _read_multipart(self, raw, utc, monotonic):
        if self._part_binding is _SINGLE_IMAGE:
            raise CaptureError("GENTL_PART_MISSING", "采集中途收到多部件缓冲区，原图像来源已改变。")
        selected, options, candidates = select_image_part(raw, self._part_selector, self._part_binding)
        additional = ()
        bindings = None
        if self._save_all_2d:
            if len({part.key for part in candidates}) != len(candidates):
                raise CaptureError("GENTL_PART_SET_AMBIGUOUS", "有多个二维部件使用相同编号，无法追踪各自的原图。")
            bindings = frozenset(part.binding for part in candidates)
            if self._all_part_bindings is not None and self._all_part_bindings != bindings:
                raise CaptureError("GENTL_PART_SET_CHANGED", "可归档的二维部件编号或尺寸在采集中发生变化。")
            additional = sorted((part for part in candidates if part is not selected), key=lambda part: part.selector)
        filled = self._filled_bytes(raw)
        data = raw.raw_buffer
        endian = optional(lambda: raw.pixel_endianness)
        pixels, byte_order = part_pixels(data, selected, filled, endian)
        ticks, frequency = self._raw_clock(raw)
        image = self._image(pixels, selected.name, selected.bits, selected.color, utc, monotonic,
                            raw, ticks, frequency, part=selected.metadata(byte_order))
        if additional:
            auxiliary = []
            for part in additional:
                samples, order = part_pixels(data, part, filled, endian)
                auxiliary.append(replace(image, pixels=samples, pixel_format=part.name, significant_bits=part.bits,
                                         color_model=part.color, metadata={IMAGE_PART: part.metadata(order)}))
            image = replace(image, auxiliary_images=tuple(auxiliary))
        self._part_binding = selected.binding
        self._all_part_bindings = bindings
        self.image_part_options = options
        return image

    def _filled_bytes(self, raw):
        filled = optional(lambda: raw.size_filled)
        intended = optional(lambda: raw.data_size)
        if intended and filled is not None and intended != filled:
            raise CaptureError("GENTL_INCOMPLETE_BUFFER", f"{self.device.id} reported {filled} of {intended} payload bytes")
        return intended if filled is None else filled

    def _raw_clock(self, raw):
        nanos = optional(lambda: raw.timestamp_ns)
        if nanos is not None:
            return nanos, 1_000_000_000
        return (optional(lambda: raw.timestamp),
                optional(lambda: raw.parent.parent.timestamp_frequency) or self._camera_tick_frequency)

    def _image(self, pixels, fmt, bits, color, utc, monotonic, module, ticks, frequency, *, part=None):
        frame_id = optional(lambda: module.frame_id)
        native_id = int(frame_id) if frame_id is not None and 0 <= frame_id < 2**64 - 1 else None
        device_time = ClockReading(
            int(ticks), "ticks", "gentl_buffer_timestamp", f"{self.device.id}:gentl_timestamp",
            "producer_clock_unspecified_epoch", self._ptp_status,
            tick_frequency_hz=int(frequency) if frequency and frequency > 0 else None,
        ) if ticks else None
        metadata = {IMAGE_PART: part} if part is not None else {}
        for feature in CHUNK_FIELDS:
            value = self._nodes.get(feature)
            if value is not None:
                metadata[feature] = value
        start, end = exposure_marks(metadata, self._timestamp_selector,
                                    clock_id=f"{self.device.id}:camera_chunk_timestamp",
                                    frequency=self._camera_tick_frequency, sync_status=self._ptp_status)
        return Image(pixels, fmt, bits, color, utc, monotonic, native_id, native_id,
                     device_time, start, end, metadata)
