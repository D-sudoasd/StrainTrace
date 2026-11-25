"""Map selected GenApi nodes, retaining native names and access restrictions."""

from __future__ import annotations

import re
from dataclasses import replace

from ..config import CaptureError
from .base import ClockReading, Feature


NODES = {
    "offset_x": ("OffsetX",), "offset_y": ("OffsetY",),
    "width": ("Width",), "height": ("Height",), "pixel_format": ("PixelFormat",),
    "sensor_width": ("SensorWidth",), "sensor_height": ("SensorHeight",),
    "binning_x": ("BinningHorizontal",), "binning_y": ("BinningVertical",),
    "decimation_x": ("DecimationHorizontal",), "decimation_y": ("DecimationVertical",),
    "exposure_us": ("ExposureTime", "ExposureTimeAbs"), "exposure_raw": ("ExposureTimeRaw",),
    "exposure_auto": ("ExposureAuto",), "gain_db": ("Gain", "GainAbs"), "gain_raw": ("GainRaw",),
    "gain_auto": ("GainAuto",), "fps": ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
    "fps_enable": ("AcquisitionFrameRateEnable",),
    "trigger_mode": ("TriggerMode",), "trigger_source": ("TriggerSource",),
    "trigger_selector": ("TriggerSelector",), "trigger_activation": ("TriggerActivation",),
    "acquisition_mode": ("AcquisitionMode",), "device_temperature": ("DeviceTemperature",),
    "ptp_enable": ("PtpEnable", "GevIEEE1588"),
    "ptp_status": ("PtpStatus", "GevIEEE1588Status", "BslPtpStatus"),
    "ptp_reported_offset": ("PtpOffsetFromMaster", "GevIEEE1588OffsetFromMaster"),
    "timestamp_frequency": ("GevTimestampTickFrequency", "TimestampTickFrequency"),
    "chunk_mode": ("ChunkModeActive",), "chunk_selector": ("ChunkSelector",),
    "chunk_timestamp_selector": ("BslChunkTimestampSelector", "ChunkTimestampSelector"),
    "chunk_enable": ("ChunkEnable",), "counter_selector": ("CounterSelector",),
    "counter_event_source": ("CounterEventSource",), "counter_event_activation": ("CounterEventActivation",),
    "test_pattern": ("TestPattern",),
}
CHUNK_FIELDS = (
    "ChunkCounterValue", "ChunkFramecounter", "ChunkFrameID", "ChunkExposureTime", "ChunkTimestamp",
    "ChunkExposureStartTimestamp", "ChunkExposureEndTimestamp", "BslChunkTimestampValue",
)
UNITS = {
    "width": "px", "height": "px", "offset_x": "px", "offset_y": "px",
    "sensor_width": "px", "sensor_height": "px", "exposure_us": "us",
    "gain_db": "dB", "fps": "Hz", "timestamp_frequency": "Hz",
    "exposure_raw": "device_units", "gain_raw": "device_units",
}
PRIORITY = {
    key: i for i, key in enumerate((
        "acquisition_mode", "exposure_auto", "gain_auto", "fps_enable", "trigger_selector",
        "chunk_mode", "chunk_selector", "counter_selector", "counter_event_source", "counter_event_activation",
        "pixel_format", "binning_x", "binning_y", "decimation_x", "decimation_y",
        "width", "height", "offset_x", "offset_y", "chunk_enable",
    ))
}
_MONO_OR_BAYER = re.compile(r"(Mono|Bayer(?:RG|BG|GR|GB))(8|10|12|14|16)(?:p|Packed)?$")
_COLOR = re.compile(r"(RGBa|BGRa|RGBA|BGRA|RGB|BGR)(8|10|12|14|16)(?:Packed)?$")


def image_layout(pixel_format: str) -> tuple[int, str]:
    if not isinstance(pixel_format, str):
        raise CaptureError("PIXEL_FORMAT_UNSUPPORTED", f"could not determine the camera pixel format: {pixel_format!r}")
    match = _MONO_OR_BAYER.fullmatch(pixel_format)
    if match:
        return int(match[2]), "mono" if match[1] == "Mono" else "bayer"
    match = _COLOR.fullmatch(pixel_format)
    if match:
        bits, model = int(match[2]), match[1].lower()
        if len(model) == 3 or bits == 8:
            return bits, model
    raise CaptureError("PIXEL_FORMAT_UNSUPPORTED", f"lossless capture does not support {pixel_format!r}")


def optional(getter, default=None):
    """Read optional SDK metadata. Nodes can disappear with selector and device state."""
    try:
        return getter()
    except Exception:
        return default


def exposure_marks(metadata, selector, *, clock_id, frequency, sync_status):
    def stamp(position, name, value):
        return ClockReading(value, "ticks", f"{name}:{position}", clock_id, "device_clock_unspecified_epoch",
                            sync_status, tick_frequency_hz=frequency) if type(value) is int and value >= 0 else None

    names = {"ExposureStart": "ChunkExposureStartTimestamp", "ExposureEnd": "ChunkExposureEndTimestamp"}
    readings = {position: stamp(position, name, metadata.get(name)) for position, name in names.items()}
    if selector in names and readings[selector] is None:
        for name in ("BslChunkTimestampValue", "ChunkTimestamp"):
            if value := stamp(selector, name, metadata.get(name)):
                readings[selector] = value
                break
    return readings["ExposureStart"], readings["ExposureEnd"]


class GenApi:
    def __init__(self, node_map, *, pylon_api=None, genicam_api=None):
        self._map = node_map
        self._pylon = pylon_api
        self._genicam = genicam_api

    def _node(self, name):
        return optional(lambda: self._map.GetNode(name) if self._pylon else self._map.get_node(name))

    def _readable(self, node):
        return optional(lambda: self._pylon.IsReadable(node) if self._pylon else self._genicam.is_readable(node), False)

    def _writable(self, node):
        return optional(lambda: self._pylon.IsWritable(node) if self._pylon else self._genicam.is_writable(node), False)

    def _get(self, node, name):
        return optional(lambda: getattr(node, name.capitalize() if self._pylon else name))

    def get(self, native_name):
        node = self._node(native_name)
        return self._get(node, "value") if self._readable(node) else None

    def feature(self, code: str) -> Feature | None:
        for name in NODES[code]:
            node = self._node(name)
            readable, writable = self._readable(node), self._writable(node)
            if not (readable or writable):
                continue
            unit = self._get(node, "unit") or UNITS.get(code)
            if code == "gain_db" and unit and unit.lower() != "db":
                continue
            if code == "exposure_us" and unit and unit.lower() not in ("us", "µs", "μs"):
                continue
            choices = tuple(self._get(node, "symbolics") or ())
            if code == "pixel_format":
                choices = tuple(entry for entry in choices if optional(lambda: image_layout(entry)) is not None)
                if not choices:
                    writable = False
            return Feature(
                code, self._get(node, "value") if readable else None, self._get(node, "min"),
                self._get(node, "max"), self._get(node, "inc"), choices,
                bool(readable), bool(writable), unit, name,
            )
        return None

    def features(self) -> dict[str, Feature]:
        return {key: feature for key in NODES if (feature := self.feature(key)) is not None}

    def write(self, code: str, value) -> None:
        feature = self.feature(code) if code in NODES else None
        if feature is None:
            raise CaptureError("FEATURE_UNAVAILABLE", f"camera does not expose {code}")
        if code in ("acquisition_mode", "trigger_mode", "trigger_source", "trigger_selector"):
            raise CaptureError("PHASE_OWNS_TRIGGER", f"configure {code} via the capture phase")
        self._write(feature, value)

    def _write(self, feature: Feature, value) -> None:
        feature.validate(value)
        node = self._node(feature.native_name)
        try:
            setattr(node, "Value" if self._pylon else "value", value)
        except Exception as exc:
            raise CaptureError("FEATURE_WRITE_FAILED", f"{feature.native_name}: {exc}") from exc

    def configure(self, settings) -> None:
        # The maximum width/height can depend on the current offset.
        for dimension, offset in (("width", "offset_x"), ("height", "offset_y")):
            if dimension in settings and offset in settings:
                feature = self.feature(offset)
                if feature and feature.minimum is not None and feature.value != feature.minimum:
                    self._write(feature, feature.minimum)
        for code in sorted(settings, key=lambda name: PRIORITY.get(name, len(PRIORITY))):
            self.write(code, settings[code])

    def trigger_setup(self, trigger: str, source: str | None) -> None:
        acquisition = self.feature("acquisition_mode")
        if acquisition and acquisition.value != "Continuous" and "Continuous" in acquisition.choices:
            self._write(acquisition, "Continuous")
        mode = self.feature("trigger_mode")
        if mode is None:
            if trigger == "free":
                return
            raise CaptureError("TRIGGER_UNSUPPORTED", "camera does not expose TriggerMode")
        selector = self.feature("trigger_selector")
        # A previous vendor profile may leave burst or exposure triggering enabled.
        # Free-run/FrameStart must have exactly one effective trigger strategy.
        if selector and selector.writable:
            selectors = selector.choices
        else:
            selectors = ()
        for entry in selectors:
            self._write(self.feature("trigger_selector"), entry)
            candidate = self.feature("trigger_mode")
            if candidate and candidate.value != "Off":
                self._write(candidate, "Off")
        if selector and selector.value != "FrameStart":
            if "FrameStart" not in selector.choices and trigger != "free":
                raise CaptureError("TRIGGER_UNSUPPORTED", "camera does not expose FrameStart")
            if "FrameStart" in selector.choices:
                self._write(self.feature("trigger_selector"), "FrameStart")
        elif selectors and "FrameStart" in selectors:
            self._write(self.feature("trigger_selector"), "FrameStart")
        mode = self.feature("trigger_mode")
        if trigger == "free":
            if mode.value != "Off":
                self._write(mode, "Off")
            return
        trigger_source = self.feature("trigger_source")
        if trigger_source is None:
            raise CaptureError("TRIGGER_UNSUPPORTED", "camera does not expose TriggerSource")
        if trigger == "hardware" and not (source and re.fullmatch(r"Line[0-9]+|CC[0-9]+", source)):
            raise CaptureError("TRIGGER_SOURCE_UNSUPPORTED", "hardware triggering requires a physical Line<n> or CC<n> input")
        self._write(trigger_source, "Software" if trigger == "software" else source)
        self._write(self.feature("trigger_mode"), "On")

    def public_features(self) -> dict[str, Feature]:
        controlled = {"acquisition_mode", "trigger_selector", "trigger_mode", "trigger_source"}
        return {
            name: replace(value, writable=False) if name in controlled else value
            for name, value in self.features().items()
        }
