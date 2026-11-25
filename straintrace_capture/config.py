"""Versioned, SDK-independent Capture configuration."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from sys import float_info
from types import MappingProxyType
from typing import Any, Literal


HARDWARE_COUNTER_FEATURES = ("ChunkCounterValue", "ChunkFramecounter", "ChunkFrameID")
EVENT_CONCURRENT_LIMIT = 16
REPEAT_OF = "repeat_of"
REPEAT_PARENT = "repeat_parent"
SENSOR_WINDOW_FIELDS = ("offset_x", "offset_y", "binning_x", "binning_y", "decimation_x", "decimation_y")


def repeat_source_directory(metadata: Mapping) -> Path | None:
    hint = metadata.get(REPEAT_PARENT)
    directory = hint.get("directory") if isinstance(hint, Mapping) else None
    if metadata.get(REPEAT_OF) and isinstance(directory, str) and "\0" not in directory:
        path = Path(directory)
        if path.is_absolute():
            return path
    return None


def inside_repeat_source(config: CaptureConfig, output: Path) -> bool:
    """Compare a resolved output with the recorded path without reading the source locator."""
    source = repeat_source_directory(config.metadata)
    return source is not None and output.is_relative_to(source)


class CaptureError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


def invalid(message: str) -> CaptureError:
    return CaptureError("INVALID_CONFIG", message)


def fields(value: Any, allowed: set[str], where: str) -> Mapping:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise invalid(f"{where} must be an object with string keys")
    extra = value.keys() - allowed
    if extra:
        raise invalid(f"{where} has unknown fields: {', '.join(sorted(extra))}")
    return value


def positive_integer(value: Any, where: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise invalid(f"{where} must be an integer >= {minimum}")
    return value


def automatic_queue_capacity(max_buffer_bytes: int) -> int:
    """Bound write-queue entries even when the source produces very small images."""
    return min(8192, max(32, max_buffer_bytes // (64 * 1024)))


def positive_number(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not (0 < value <= 1e12) or not math.isfinite(value):
        raise invalid(f"{where} must be a finite positive number")
    return float(value)


def nonempty_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise invalid(f"{where} must be nonempty text")
    return value.strip()


def settings_object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise invalid("camera settings must be an object")
    for key, item in value.items():
        nonempty_text(key, "camera setting name")
        if type(item) not in (int, float, str, bool) or (type(item) is float and not math.isfinite(item)):
            raise invalid(f"camera setting {key!r} must be a finite number, string or boolean")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class CameraConfig:
    id: str
    settings: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> CameraConfig:
        value = fields(value, {"id", "settings"}, "camera")
        identity = nonempty_text(value.get("id"), "camera id")
        backend, _, address = identity.partition(":")
        if not backend or not address:
            raise invalid(f"camera ID {identity!r} must have a backend prefix")
        return cls(identity, settings_object(value.get("settings", {})))


@dataclass(frozen=True)
class QualityRegion:
    bounds: tuple[int, int, int, int]
    image_size_px: tuple[int, int]
    pixel_format: str
    significant_bits: int
    image_part: str | None
    sensor_window: Mapping[str, int]

    @classmethod
    def from_dict(cls, value: Any) -> QualityRegion:
        value = fields(value, set(cls.__dataclass_fields__), "capture quality region")
        bounds, size = value.get("bounds"), value.get("image_size_px")
        if not isinstance(bounds, list) or len(bounds) != 4 or not isinstance(size, list) or len(size) != 2:
            raise invalid("quality region requires [x, y, width, height] and [image width, image height]")
        x, y = (positive_integer(item, "quality region origin", minimum=0) for item in bounds[:2])
        width, height = (positive_integer(item, "quality region size") for item in bounds[2:])
        image_width, image_height = (positive_integer(item, "quality region image size") for item in size)
        if x + width > image_width or y + height > image_height:
            raise invalid("quality region must lie within its original image")
        bits = positive_integer(value.get("significant_bits"), "quality region bit depth")
        if bits > 16:
            raise invalid("quality region bit depth must be at most 16")
        part = value.get("image_part")
        sensor = fields(value.get("sensor_window", {}), set(SENSOR_WINDOW_FIELDS), "quality region sensor window")
        return cls((x, y, width, height), (image_width, image_height),
                   nonempty_text(value.get("pixel_format"), "quality region pixel format"), bits,
                   nonempty_text(part, "quality region image part") if part is not None else None,
                   MappingProxyType({name: positive_integer(item, name, minimum=int(not name.startswith("offset_")))
                                     for name, item in sensor.items()}))

    def to_dict(self) -> dict[str, Any]:
        return {"bounds": list(self.bounds), "image_size_px": list(self.image_size_px),
                "pixel_format": self.pixel_format, "significant_bits": self.significant_bits,
                "image_part": self.image_part, "sensor_window": dict(self.sensor_window)}


def hardware_mapping_error(camera_ids, source, counters) -> str:
    if counters is not None and counters.keys() != camera_ids:
        return "hardware pairing anchors must match the configured cameras exactly"
    if isinstance(source, Mapping) and source.keys() != camera_ids:
        return "physical trigger line mapping must match the configured cameras exactly"
    return ""


@dataclass(frozen=True)
class HardwarePairing:
    source_id: str
    counter_feature: str
    first_counter: Mapping[str, int]
    first_event_number: int = 1

    @classmethod
    def from_dict(cls, value: Any) -> HardwarePairing:
        data = fields(value, set(cls.__dataclass_fields__), "hardware pairing")
        source = nonempty_text(data.get("source_id"), "physical shared trigger source ID")
        feature = data.get("counter_feature")
        if feature not in HARDWARE_COUNTER_FEATURES:
            raise invalid("hardware pairing requires an explicit supported chunk counter feature")
        counters = data.get("first_counter")
        if not isinstance(counters, Mapping) or not counters:
            raise invalid("hardware pairing requires the first counter for every camera on a known shared pulse")
        return cls(
            source, feature, MappingProxyType({
                nonempty_text(key, "hardware camera ID"): positive_integer(counter, f"first counter for {key}", minimum=0)
                for key, counter in counters.items()
            }),
            positive_integer(data.get("first_event_number", 1), "first shared event number", minimum=0),
        )

    def to_dict(self):
        return {**vars(self), "first_counter": dict(self.first_counter)}


@dataclass(frozen=True)
class CsvTrigger:
    path: str
    value_column: str
    threshold: float
    direction: Literal["rising", "falling"]
    quantity: str
    unit: str
    time_column: str | None = None
    header_sha256: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> CsvTrigger:
        data = fields(value, set(cls.__dataclass_fields__), "CSV trigger")
        threshold = data.get("threshold")
        if type(threshold) not in (int, float) or not -float_info.max <= threshold <= float_info.max:
            raise invalid("CSV trigger threshold must be a finite number")
        direction = data.get("direction")
        if direction not in ("rising", "falling"):
            raise invalid("CSV trigger direction must be rising or falling")
        column = nonempty_text(data.get("value_column"), "CSV trigger value column")
        time_column = nonempty_text(data["time_column"], "CSV trigger time column") if "time_column" in data else None
        if time_column == column:
            raise invalid("CSV trigger value and time columns must be different")
        fingerprint = data.get("header_sha256")
        if "header_sha256" in data and (not isinstance(fingerprint, str) or not re.fullmatch("[0-9a-f]{64}", fingerprint)):
            raise invalid("CSV trigger header_sha256 must contain 64 hexadecimal digits")
        return cls(nonempty_text(data.get("path"), "CSV trigger path"), column, float(threshold), direction,
                   nonempty_text(data.get("quantity"), "CSV trigger quantity"),
                   nonempty_text(data.get("unit"), "CSV trigger unit"), time_column, fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in vars(self).items() if value is not None}


@dataclass(frozen=True)
class Phase:
    mode: str
    role: str = "test"
    frame_count: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    frame_timeout_s: float = 3.0
    hardware_source: str | Mapping[str, str] | None = None
    hardware_pairing: HardwarePairing | None = None
    pre_frames: int = 0
    post_frames: int | None = None
    software_trigger_fallback: bool = False
    monitor_trigger_wait: bool = False
    start_confirmation: str | None = None
    csv_trigger: CsvTrigger | None = None
    event_count: int = 1
    overlap_events: bool = False

    @property
    def camera_trigger(self) -> str:
        return {"continuous": "free", "hardware": "hardware"}.get(self.mode, "software")

    @property
    def repeats_events(self) -> bool:
        return self.mode == "event" and (self.event_count != 1 or self.overlap_events)

    @property
    def minimum_event_rounds(self) -> int:
        events = self.event_count or 1
        if self.overlap_events:
            groups = (events + EVENT_CONCURRENT_LIMIT - 1) // EVENT_CONCURRENT_LIMIT
            return self.pre_frames + self.post_frames * groups
        return (self.pre_frames + self.post_frames) * events

    def hardware_input(self, camera_id: str) -> str | None:
        source = self.hardware_source
        return source[camera_id] if isinstance(source, Mapping) else source

    @classmethod
    def from_dict(cls, value: Any) -> Phase:
        value = fields(value, set(cls.__dataclass_fields__), "phase")
        mode = value.get("mode")
        if mode not in ("single", "timed", "continuous", "manual", "hardware", "event"):
            raise invalid(f"unsupported phase mode: {mode!r}")
        role = value.get("role", "test")
        if role not in ("reference", "test", "calibration"):
            raise invalid(f"unsupported frame role: {role!r}")
        count = positive_integer(value["frame_count"], "frame_count") if "frame_count" in value else None
        duration = positive_number(value["duration_s"], "duration_s") if "duration_s" in value else None
        fps = positive_number(value["fps"], "fps") if "fps" in value else None
        timeout = positive_number(value.get("frame_timeout_s", 3.0), "frame_timeout_s")
        source = value.get("hardware_source")
        if isinstance(source, Mapping):
            source = MappingProxyType({
                nonempty_text(key, "hardware camera ID"): nonempty_text(line, "physical trigger line")
                for key, line in source.items()
            })
        elif "hardware_source" in value:
            source = nonempty_text(source, "hardware_source")
        pairing = HardwarePairing.from_dict(value["hardware_pairing"]) if "hardware_pairing" in value else None
        pre = positive_integer(value.get("pre_frames", 0), "pre_frames", minimum=0)
        post = positive_integer(value["post_frames"], "post_frames") if "post_frames" in value else None
        event_count = positive_integer(value.get("event_count", 1), "event_count", minimum=0)
        overlap = value.get("overlap_events", False)
        if type(overlap) is not bool or ("overlap_events" in value and mode != "event"):
            raise invalid("overlap_events must be boolean and applies only to event phases")
        fallback = value.get("software_trigger_fallback", False)
        if type(fallback) is not bool or ("software_trigger_fallback" in value and mode in ("continuous", "hardware")):
            raise invalid("software_trigger_fallback must be boolean and applies only to software request phases")
        monitor = value.get("monitor_trigger_wait", False)
        if type(monitor) is not bool or ("monitor_trigger_wait" in value and mode != "hardware"):
            raise invalid("monitor_trigger_wait must be boolean and applies only to hardware phases")
        confirmation = None
        if "start_confirmation" in value:
            confirmation = nonempty_text(value["start_confirmation"], "start_confirmation").strip()
        csv_trigger = CsvTrigger.from_dict(value["csv_trigger"]) if "csv_trigger" in value else None
        if mode == "single":
            if count not in (None, 1) or duration is not None:
                raise invalid("single takes one frame per camera and does not accept duration_s")
            count = 1
        if mode in ("timed", "event") and fps is None:
            raise invalid(f"{mode} requires fps")
        if fps is not None and mode not in ("timed", "event"):
            raise invalid("phase fps applies to timed/event modes; set camera fps for free-running acquisition")
        if mode == "hardware" and not source:
            raise invalid("hardware requires hardware_source")
        if (source is not None or pairing is not None) and mode != "hardware":
            raise invalid("hardware_source and hardware_pairing apply only to hardware phases")
        if mode == "event":
            if post is None or count is not None:
                raise invalid("event requires post_frames and uses pre_frames instead of frame_count")
        elif "pre_frames" in value or post is not None or csv_trigger is not None or "event_count" in value:
            raise invalid("pre_frames, post_frames, csv_trigger and event_count apply only to event phases")
        if mode == "timed" and count is None and duration is None:
            raise invalid("timed requires frame_count or duration_s")
        return cls(mode, role, count, duration, fps, timeout, source, pairing, pre, post,
                   fallback, monitor, confirmation, csv_trigger, event_count, overlap)

    def to_dict(self) -> dict[str, Any]:
        settings = {key: value for key, value in vars(self).items()
                    if value is not None and (key not in ("pre_frames", "software_trigger_fallback", "monitor_trigger_wait") or value)}
        if self.event_count == 1:
            settings.pop("event_count")
        if not self.overlap_events:
            settings.pop("overlap_events")
        if self.hardware_pairing:
            settings["hardware_pairing"] = self.hardware_pairing.to_dict()
        if self.csv_trigger:
            settings["csv_trigger"] = self.csv_trigger.to_dict()
        if isinstance(self.hardware_source, Mapping):
            settings["hardware_source"] = dict(self.hardware_source)
        return settings


@dataclass(frozen=True)
class CaptureConfig:
    name: str
    cameras: tuple[CameraConfig, ...]
    phases: tuple[Phase, ...]
    queue_capacity: int | Literal["auto"] = "auto"
    max_buffer_bytes: int = 256 * 1024**2
    min_free_bytes: int = 256 * 1024**2
    durability: str = "buffered"
    segment_frames: int = 0
    cti_files: tuple[str, ...] = ()
    calibration_files: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    quality_regions: Mapping[str, QualityRegion] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def effective_queue_capacity(self) -> int:
        if self.queue_capacity == "auto":
            return automatic_queue_capacity(self.max_buffer_bytes)
        return self.queue_capacity

    @classmethod
    def from_dict(cls, value: Any) -> CaptureConfig:
        value = fields(value, {"config_version", *cls.__dataclass_fields__}, "capture configuration")
        if type(value.get("config_version")) is not int or value["config_version"] != 1:
            raise invalid("config_version must be 1")
        name = nonempty_text(value.get("name"), "experiment name")
        cameras = value.get("cameras")
        phases = value.get("phases")
        if not isinstance(cameras, list) or not cameras:
            raise invalid("cameras must contain at least one camera")
        if not isinstance(phases, list) or not phases:
            raise invalid("phases must contain at least one phase")
        devices = tuple(CameraConfig.from_dict(camera) for camera in cameras)
        if len({device.id for device in devices}) != len(devices):
            raise invalid("camera IDs must be unique within an experiment")
        file_lists = {}
        for key in ("cti_files", "calibration_files"):
            paths = value.get(key, [])
            if not isinstance(paths, list):
                raise invalid(f"{key} must be an array of paths")
            file_lists[key] = tuple(nonempty_text(path, key) for path in paths)
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise invalid("metadata must be an object")
        try:
            metadata = json.loads(json.dumps(dict(metadata), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise invalid(f"metadata must contain finite JSON values: {exc}") from exc
        parsed_phases = tuple(Phase.from_dict(phase) for phase in phases)
        durability = value.get("durability", "buffered")
        if durability not in ("buffered", "write_through"):
            raise invalid("durability must be buffered or write_through")
        camera_ids = {device.id for device in devices}
        quality_regions = fields(value.get("quality_regions", {}), camera_ids, "capture quality region cameras")
        for phase in parsed_phases:
            counters = phase.hardware_pairing.first_counter if phase.hardware_pairing else None
            if problem := hardware_mapping_error(camera_ids, phase.hardware_source, counters):
                raise invalid(problem)
        queue_capacity = value.get("queue_capacity", "auto")
        return cls(
            name=name,
            cameras=devices,
            phases=parsed_phases,
            queue_capacity="auto" if queue_capacity == "auto" else positive_integer(queue_capacity, "queue_capacity"),
            max_buffer_bytes=positive_integer(value.get("max_buffer_bytes", 256 * 1024**2), "max_buffer_bytes"),
            min_free_bytes=positive_integer(value.get("min_free_bytes", 256 * 1024**2), "min_free_bytes", minimum=0),
            durability=durability,
            segment_frames=positive_integer(value.get("segment_frames", 0), "segment_frames", minimum=0),
            metadata=MappingProxyType(metadata),
            quality_regions=MappingProxyType({key: QualityRegion.from_dict(region) for key, region in quality_regions.items()}),
            **file_lists,
        )

    @classmethod
    def load(cls, path: str | Path) -> CaptureConfig:
        path = Path(path).resolve()
        try:
            config = cls.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError) as exc:
            raise invalid(f"cannot read {path}: {exc}") from exc
        data = config.to_dict()
        for key in ("cti_files", "calibration_files"):
            data[key] = [str((path.parent / item).resolve()) for item in data[key]]
        for phase in data["phases"]:
            if trigger := phase.get("csv_trigger"):
                trigger["path"] = str((path.parent / Path(trigger["path"]).expanduser()).resolve())
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_version": 1, "name": self.name,
            "cameras": [{"id": camera.id, "settings": dict(camera.settings)} for camera in self.cameras],
            "phases": [phase.to_dict() for phase in self.phases],
            "queue_capacity": self.queue_capacity, "max_buffer_bytes": self.max_buffer_bytes,
            "min_free_bytes": self.min_free_bytes, "cti_files": list(self.cti_files),
            "durability": self.durability, "segment_frames": self.segment_frames,
            "calibration_files": list(self.calibration_files),
            "metadata": dict(self.metadata),
            **({"quality_regions": {key: region.to_dict() for key, region in self.quality_regions.items()}}
               if self.quality_regions else {}),
        }
