"""Same-run pulse associations bounded by two independently identified exposures."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .config import HARDWARE_COUNTER_FEATURES, CaptureError, fields, nonempty_text, positive_integer
from .storage import sha256_file


PULSE_BINDING_SCHEMA = "straintrace.capture-pulse-anchors/v1"


@dataclass(frozen=True)
class PulseBinding:
    experiment_id: str
    phase: int
    source_id: str
    counter_feature: str
    first_event_number: int
    last_event_number: int
    camera_frames: Mapping[str, tuple[int, int]]
    evidence: str

    @classmethod
    def from_dict(cls, data):
        row = fields(data, {*cls.__dataclass_fields__, "schema"}, "post-capture pulse anchors")
        if row.get("schema") != PULSE_BINDING_SCHEMA:
            raise CaptureError("PULSE_BINDING_INVALID", "锚点文件需要可识别的采后补配版本。")
        if not isinstance(cameras := row.get("camera_frames"), Mapping) or not cameras:
            raise CaptureError("PULSE_BINDING_INVALID", "请为每台相机填写起止原图序号。")
        selections = {}
        for name, values in cameras.items():
            camera = nonempty_text(name, "camera ID")
            if (not isinstance(values, (tuple, list)) or len(values) != 2 or
                any(type(value) is not int for value in values) or not 1 <= values[0] < values[1] < 2**63):
                raise CaptureError("PULSE_BINDING_INVALID", f"{camera} 的起止原图需要两个递增的正整数序号。")
            if camera in selections:
                raise CaptureError("PULSE_BINDING_INVALID", f"相机编号 {camera} 重复。")
            selections[camera] = tuple(values)
        first = positive_integer(row.get("first_event_number"), "first pulse", minimum=0)
        last = positive_integer(row.get("last_event_number"), "last pulse", minimum=0)
        feature = row.get("counter_feature")
        if first >= last or feature not in HARDWARE_COUNTER_FEATURES:
            raise CaptureError("PULSE_BINDING_INVALID", "请选择帧附带计数器，并填写两个递增的 DAQ 脉冲编号。")
        return cls(nonempty_text(row.get("experiment_id"), "experiment ID"), positive_integer(row.get("phase"), "phase"),
                   nonempty_text(row.get("source_id"), "DAQ source ID"), feature, first, last,
                   MappingProxyType(selections), nonempty_text(row.get("evidence"), "anchor evidence"))

    def to_dict(self):
        return {**vars(self), "schema": PULSE_BINDING_SCHEMA,
                "camera_frames": {camera: list(pair) for camera, pair in self.camera_frames.items()}}

    @classmethod
    def load(cls, path):
        row = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(row, dict) and isinstance(association := row.get("association"), dict) and "binding" in association:
            row = association["binding"]
        return cls.from_dict(row)

    def check_context(self, experiment, phase):
        if self.experiment_id != experiment["experiment_id"] or self.phase != phase:
            raise CaptureError("PULSE_BINDING_SCOPE", "补配锚点须对应当前实验及所选阶段。")
        cameras = {row["id"] for row in experiment["config"]["cameras"]}
        if self.camera_frames.keys() != cameras:
            raise CaptureError("PULSE_BINDING_SCOPE", "锚点须包括该实验登记的每台相机。")

    def audit_range(self, first, last):
        first = self.first_event_number if first is None else first
        last = self.last_event_number if last is None else last
        if not self.first_event_number <= first <= last <= self.last_event_number:
            raise CaptureError("PULSE_BINDING_SCOPE", "核对范围须位于两个确认脉冲之间，含起止编号。")
        return first, last


def _usable(frame):
    return (frame["ledger_consistent"] and frame["event_id"] is None and
            type(frame["counter"]) is int and frame["counter"] >= 0)


def _checked_anchor(reader, frame, camera, sequence):
    if frame is None or not _usable(frame):
        raise CaptureError("PULSE_BINDING_FRAME", f"{camera} 的序号 {sequence} 缺少本阶段可用的计数记录。")
    digest = frame.get("sha256")
    try:
        verified = sha256_file(reader.image_path(frame)) == digest
    except OSError:
        verified = False
    if not verified:
        raise CaptureError("PULSE_BINDING_FRAME", f"{camera} 的锚点原图 {sequence} 无法通过文件校验。")
    return frame


def counter_associations(reader, binding, events, frames):
    if any(event["kind"] == "hardware_trigger" for event in events):
        raise CaptureError("PULSE_BINDING_SCOPE", "本阶段已有硬件触发台账记录。请先核实原记录。")
    groups, by_camera = defaultdict(list), defaultdict(list)
    unassociated = []
    for frame in frames:
        if frame["phase"] == binding.phase and frame["camera_id"] in binding.camera_frames:
            by_camera[frame["camera_id"]].append(frame)
        else:
            unassociated.append({**frame, "reason": "outside_config"})
    pairing = {"counter_feature": binding.counter_feature, "first_event_number": binding.first_event_number,
               "source_id": binding.source_id, "first_counter": {}}
    evidence = {"basis": "post_capture_counter_anchors", "binding": binding.to_dict(), "anchors": {}, "outside_frames": {}}
    interval = binding.last_event_number - binding.first_event_number
    for camera, (first_seq, last_seq) in binding.camera_frames.items():
        rows = sorted(by_camera[camera], key=lambda row: row["sequence"])
        ends = {row["sequence"]: row for row in rows if row["sequence"] in (first_seq, last_seq)}
        first, last = (_checked_anchor(reader, ends.get(seq), camera, seq) for seq in (first_seq, last_seq))
        origin = first["counter"]
        if last["counter"] - origin != interval:
            raise CaptureError("PULSE_BINDING_COUNTER", f"{camera} 的两端计数差为 {last['counter'] - origin}，DAQ 脉冲编号差为 {interval}。")
        pairing["first_counter"][camera] = origin
        evidence["anchors"][camera] = {"first": first, "last": last}
        outside = evidence["outside_frames"][camera] = {"before": 0, "after": 0}
        previous = origin
        for frame in rows:
            sequence = frame["sequence"]
            if sequence < first_seq:
                outside["before"] += 1
            elif sequence > last_seq:
                outside["after"] += 1
            elif not _usable(frame):
                unassociated.append({**frame, "reason": "unusable_counter"})
            else:
                counter = frame["counter"]
                if counter < previous:
                    raise CaptureError("PULSE_BINDING_COUNTER", f"{camera} 在原图序号 {sequence} 的计数回退。请选择同一连续计数区间。")
                groups[binding.first_event_number + counter - origin].append(frame)
                previous = counter
    return pairing, groups, evidence, unassociated


def daq_unlisted_ranges(pulses, first, last):
    cursor = first
    gaps = []
    for number in sorted(pulses):
        if number > cursor:
            gaps.append({"first_event": cursor, "last_event": number - 1})
        cursor = number + 1
    if cursor <= last:
        gaps.append({"first_event": cursor, "last_event": last})
    return gaps
