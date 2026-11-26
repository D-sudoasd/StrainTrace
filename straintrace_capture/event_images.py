"""Resolve a finished event and an explicit baseline for its saved images."""

from collections import Counter

from .config import CaptureError
from .storage import TERMINAL_STATES


def validate_event_id(value):
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(character) < 32 for character in value)):
        raise CaptureError("EVENT_IMAGES_SCOPE", "事件标识无效，请从采集时间线重新选择。")


def event_scope(reader, experiment, context):
    validate_event_id(context)
    if experiment["state"] not in TERMINAL_STATES:
        raise CaptureError("EVENT_IMAGES_ACTIVE", "请先停止采集并等待原图写入，再分析或导出这次事件。")
    signal = reader.received_signal(context)
    phases = experiment["config"]["phases"]
    phase = signal.get("phase") if signal else None
    if (type(phase) is not int or not 1 <= phase <= len(phases)
            or phases[phase - 1]["mode"] != "event" or phases[phase - 1]["role"] != "test"):
        raise CaptureError("EVENT_IMAGES_SCOPE", "这条触发没有可用于分析的实验图像。")
    return {"signal_id": context, "phase": phase, "event_number": signal.get("event_number", 1),
            "signal_utc_ns": signal["received_utc_ns"]}


def event_sequence(reader, camera, event, *, after=0, through=None):
    while page := reader.test_sequence_page(camera, after=after, through=through, phase=event["phase"],
                                             event_context_id=event["signal_id"]):
        yield from page
        after = page[-1]["sequence"]


def event_first_frame(reader, camera, event):
    rows = reader.test_sequence_page(camera, after=0, phase=event["phase"], event_context_id=event["signal_id"], limit=1)
    if not rows:
        raise CaptureError("EVENT_IMAGES_EMPTY", "这台相机在本次事件中没有登记实验图像。")
    return rows[0]


def select_event_reference(reader, camera, event, sequence):
    first = event_first_frame(reader, camera, event)
    if sequence is not None:
        candidates = reader.reference_frames(camera, sequence=sequence)
        if candidates and candidates[0]["role"] == "reference" and sequence < first["sequence"]:
            return candidates[0], "explicit_sequence"
        scoped = reader.test_sequence_page(camera, after=sequence - 1, through=sequence, phase=event["phase"],
                                           event_context_id=event["signal_id"], limit=1)
        if scoped:
            return scoped[0], "explicit_sequence"
        raise CaptureError("EVENT_REFERENCE_OUTSIDE_SCOPE", "请选本次事件中的图像，或拍摄于它之前的实验参考图。")
    earlier = reader.reference_frames(camera, before=first["sequence"])
    if len(earlier) == 1:
        return earlier[0], "capture_reference_role"
    if len(earlier) > 1:
        raise CaptureError("EVENT_REFERENCE_REQUIRED", "本次事件之前有多张实验参考图。请选择计算所用的基准图。")
    if first.get("event_position") == "pre":
        return first, "first_event_pre_frame"
    raise CaptureError("EVENT_REFERENCE_REQUIRED", "本次没有可直接使用的事件前图。请明确选择一张参考图。")


def event_image_choices(reader, experiment, camera, context):
    event = event_scope(reader, experiment, context)
    frames = list(event_sequence(reader, camera, event))
    if not frames:
        raise CaptureError("EVENT_IMAGES_EMPTY", "这台相机在本次事件中没有登记实验图像。")
    references = reader.reference_frames(camera, before=frames[0]["sequence"])
    counts = Counter(frame.get("event_position") for frame in frames)
    options = references + frames[:-1]
    if not options:
        raise CaptureError("TOO_FEW_FRAMES", "需要一张参考图，以及至少一张晚于它的同事件图像。")
    return event, {key: counts[key] for key in ("pre", "boundary", "post")}, options
