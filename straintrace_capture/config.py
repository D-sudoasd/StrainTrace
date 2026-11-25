"""Versioned, SDK-independent Capture configuration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


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
        if feature not in ("ChunkCounterValue", "ChunkFramecounter", "ChunkFrameID"):
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
        elif "pre_frames" in value or post is not None:
            raise invalid("pre_frames and post_frames apply only to event phases")
        if mode == "timed" and count is None and duration is None:
            raise invalid("timed requires frame_count or duration_s")
        return cls(mode, role, count, duration, fps, timeout, source, pairing, pre, post)

    def to_dict(self) -> dict[str, Any]:
        settings = {key: value for key, value in vars(self).items()
                    if value is not None and (key != "pre_frames" or value)}
        if self.hardware_pairing:
            settings["hardware_pairing"] = self.hardware_pairing.to_dict()
        if isinstance(self.hardware_source, Mapping):
            settings["hardware_source"] = dict(self.hardware_source)
        return settings


@dataclass(frozen=True)
class CaptureConfig:
    name: str
    cameras: tuple[CameraConfig, ...]
    phases: tuple[Phase, ...]
    queue_capacity: int = 32
    max_buffer_bytes: int = 256 * 1024**2
    min_free_bytes: int = 256 * 1024**2
    durability: str = "buffered"
    segment_frames: int = 0
    cti_files: tuple[str, ...] = ()
    calibration_files: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

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
        devices = []
        for camera in cameras:
            camera = fields(camera, {"id", "settings"}, "camera")
            identity = nonempty_text(camera.get("id"), "camera id")
            backend, _, address = identity.partition(":")
            if not backend or not address:
                raise invalid(f"camera ID {identity!r} must have a backend prefix")
            devices.append(CameraConfig(identity, settings_object(camera.get("settings", {}))))
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
        for phase in parsed_phases:
            if phase.hardware_pairing and set(phase.hardware_pairing.first_counter) != camera_ids:
                raise invalid("hardware pairing anchors must match the configured cameras exactly")
            if isinstance(phase.hardware_source, Mapping) and set(phase.hardware_source) != camera_ids:
                raise invalid("physical trigger line mapping must match the configured cameras exactly")
        return cls(
            name=name,
            cameras=tuple(devices),
            phases=parsed_phases,
            queue_capacity=positive_integer(value.get("queue_capacity", 32), "queue_capacity"),
            max_buffer_bytes=positive_integer(value.get("max_buffer_bytes", 256 * 1024**2), "max_buffer_bytes"),
            min_free_bytes=positive_integer(value.get("min_free_bytes", 256 * 1024**2), "min_free_bytes", minimum=0),
            durability=durability,
            segment_frames=positive_integer(value.get("segment_frames", 0), "segment_frames", minimum=0),
            metadata=MappingProxyType(metadata),
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
        }
