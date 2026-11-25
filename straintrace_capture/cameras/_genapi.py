"""Map selected GenApi nodes, retaining native names and access restrictions."""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import replace

from ..config import CaptureError, HARDWARE_COUNTER_FEATURES
from .base import ClockReading, Feature, TriggerSupport, TriggerWaitSample, is_physical_trigger_input


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
    *HARDWARE_COUNTER_FEATURES, "ChunkExposureTime", "ChunkTimestamp",
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
ROI_AXES = (("width", "offset_x"), ("height", "offset_y"))
PHASE_FEATURES = frozenset(("acquisition_mode", "trigger_mode", "trigger_source", "trigger_selector"))
_MONO_OR_BAYER = re.compile(r"(Mono|Bayer(?:RG|BG|GR|GB))(8|10|12|14|16)(p|[Pp]acked)?$")
_COLOR = re.compile(r"(RGBa|BGRa|RGBA|BGRA|RGB|BGR)(8|10|12|14|16)(?:Packed)?$")


def setting_order(settings):
    return sorted(settings, key=lambda name: PRIORITY.get(name, len(PRIORITY)))


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


def is_packed_mono_or_bayer(pixel_format: str) -> bool:
    match = _MONO_OR_BAYER.fullmatch(pixel_format)
    return match is not None and match[3] is not None


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

    def _wait_output_sample(self, selector, line):
        if self._get(selector, "value") != line:
            return TriggerWaitSample(line, reason="line_selection_changed")
        status = self._node("LineStatus")
        state = None
        if self._readable(status):
            getter = "GetValue" if self._pylon else "get_value"
            # Ignore the GenApi register cache for an observation used to start a test.
            state = optional(lambda: getattr(status, getter)(False, True))
        return (TriggerWaitSample(line, state) if type(state) is bool
                else TriggerWaitSample(line, reason="line_status_unreadable"))

    @contextmanager
    def trigger_wait_monitor(self):
        """Select for reading only. Preserve the hardware wiring and output configuration."""
        selector = self._node("LineSelector")
        original = self._get(selector, "value") if self._readable(selector) else None
        if not isinstance(original, str) or self._node("LineStatus") is None:
            yield lambda: TriggerWaitSample(reason="line_catalog_unavailable")
            return
        attribute = "Value" if self._pylon else "value"
        writable = self._writable(selector)
        choices = self._get(selector, "symbolics") if writable else ()
        changed = False
        rejected = None
        try:
            for line in dict.fromkeys((original, *(choices or ()))):
                if line != original:
                    changed = True
                    optional(lambda: setattr(selector, attribute, line))
                    if self._get(selector, "value") != line:
                        continue
                if self.get("LineMode") != "Output" or self.get("LineSource") != "FrameTriggerWait":
                    continue
                if self.get("LineInverter") is not False or self.get("LineLogic") not in (None, "Positive"):
                    rejected = line
                    continue
                yield lambda: self._wait_output_sample(selector, line)
                return
            reason = "output_polarity_unconfirmed" if rejected else "wait_output_not_configured"
            yield lambda: TriggerWaitSample(rejected, reason=reason)
        finally:
            if changed and self._get(selector, "value") != original:
                optional(lambda: setattr(selector, attribute, original))
                if self._get(selector, "value") != original:
                    raise CaptureError("WAIT_OUTPUT_SELECTOR_RESTORE_FAILED", f"could not restore LineSelector to {original}")
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
        if code in PHASE_FEATURES:
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
        for dimension, offset in ROI_AXES:
            if dimension in settings and offset in settings:
                feature = self.feature(offset)
                if feature and feature.minimum is not None and feature.value != feature.minimum:
                    self._write(feature, feature.minimum)
        for code in setting_order(settings):
            self.write(code, settings[code])

    def trigger_support(self) -> TriggerSupport:
        # Use the node catalog. An inaccessible feature can still be present on
        # another TriggerSelector setting. Some pypylon lookups also return a
        # placeholder for names that are absent from the catalog.
        try:
            if self._pylon:
                names = {(node.GetNode() if hasattr(node, "GetNode") else node).GetName() for node in self._map.GetNodes()}
            else:
                names = {node.node.name for node in self._map.nodes}
        except Exception as exc:
            return TriggerSupport(error=str(exc))
        if not names:
            return TriggerSupport(error="the remote node catalog is empty")
        missing = tuple(key for key in ("TriggerMode", "TriggerSource") if key not in names)
        if missing:
            return TriggerSupport(software=False, hardware=False, basis="device_nodes", missing_nodes=missing)
        inputs = None
        # A catalog can depend on the selector and mode. Check it only in the
        # configuration used to set the source, without changing device state.
        if (self.get("TriggerSelector") == "FrameStart" and self.get("TriggerMode") == "Off"
            and self.get("AcquisitionMode") in (None, "Continuous")):
            source = self.feature("trigger_source")
            if source and source.writable and source.choices:
                inputs = tuple(name for name in source.choices if is_physical_trigger_input(name))
        if not self._pylon and "TriggerSoftware" not in names:
            return TriggerSupport(software=False, basis="device_nodes", missing_nodes=("TriggerSoftware",), physical_inputs=inputs)
        return TriggerSupport(basis="device_nodes", physical_inputs=inputs)

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
        if trigger == "hardware" and not is_physical_trigger_input(source):
            raise CaptureError("TRIGGER_SOURCE_UNSUPPORTED", "hardware triggering requires a Line, CC or CoaXPress trigger input")
        self._write(trigger_source, "Software" if trigger == "software" else source)
        self._write(self.feature("trigger_mode"), "On")

    def public_features(self) -> dict[str, Feature]:
        controlled = {"acquisition_mode", "trigger_selector", "trigger_mode", "trigger_source"}
        return {
            name: replace(value, writable=False) if name in controlled else value
            for name, value in self.features().items()
        }
