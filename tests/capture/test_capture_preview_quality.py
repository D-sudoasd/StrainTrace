from collections import deque
from threading import Event

import numpy as np

from straintrace_capture import CaptureConfig, inspect_plan
from straintrace_capture.cameras.base import Image
from straintrace_capture.presentation import format_readiness
from straintrace_capture.preview import _image_layout
from straintrace_capture.preview_quality import PreviewQualityTarget, inspect_preview_quality, with_preview_quality


def mono(code, receipt, *, shape=(40, 64)):
    return Image(np.full(shape, code, dtype=np.uint16), "Mono12", 12, "mono", receipt, receipt)


class PreviewSamples:
    def __init__(self, streams):
        self.streams = {key: deque(frames) for key, frames in streams.items()}

    def bound_image(self, camera_id):
        stream = self.streams[camera_id]
        return stream.popleft() if len(stream) > 1 else stream[0]


def test_preview_quality_uses_distinct_new_frames_for_each_camera_and_reports_its_warning(tmp_path):
    left, right = object(), object()
    before = mono(0, 10)
    clipped = mono(4095, 13)
    right_pixels = np.random.default_rng(34).integers(800, 2100, size=(40, 64), dtype=np.uint16)
    right_pixels[:, 36:] = 4095
    right_image = Image(right_pixels, "Mono12", 12, "mono", 1, 1)
    roi = (4, 4, 30, 26)
    targets = (
        PreviewQualityTarget("sim:0", None, left, _image_layout(before)),
        PreviewQualityTarget("sim:1", roi, right, _image_layout(right_image)),
    )
    preview = PreviewSamples({
        "sim:0": [(left, frame) for frame in (before, mono(1300, 11), mono(1300, 11), mono(1400, 12), clipped,
                                             mono(0, 14), mono(1300, 15))],
        "sim:1": [(right, Image(right_pixels, "Mono12", 12, "mono", i, i)) for i in range(11, 16)],
    })
    result = inspect_preview_quality(preview, targets, after=10, cancelled=Event(), timeout=1.)
    assert [row.state for row in result] == ["ready", "ready"]
    assert [frame.received_ns for frame in result[0].readings] == [11, 12, 13, 14, 15]
    assert result[0].summary()["dark_frames"] == result[0].summary()["saturated_frames"] == 1
    assert result[1].summary()["maximum_saturated_fraction"] == 0.
    assert result[1].summary()["texture_frames"] == 5 and not result[1].summary()["texture_alerts"]
    assert result[1].target.matches(targets[1])
    assert not result[1].target.matches(PreviewQualityTarget("sim:1", None, right, targets[1].layout))

    config = CaptureConfig.from_dict({
        "config_version": 1, "name": "Preview quality",
        "cameras": [{"id": identity} for identity in ("sim:0", "sim:1")],
        "phases": [{"mode": "single", "role": "reference"}],
    })
    plan = inspect_plan(config, tmp_path / "unused")
    report = with_preview_quality(plan, result)
    warning, = (issue for issue in report["issues"] if issue["code"] == "PREVIEW_IMAGE_QUALITY")
    assert warning["camera"] == "sim:0" and report["ready"] == plan["ready"] is True
    assert warning["severity"] == "warning" and "preview_quality" not in plan
    assert "1 张近饱和比例达到 20%" in format_readiness(report)
    assert not (tmp_path / "unused").exists()


def test_preview_quality_does_not_credit_a_changed_connection_or_wrong_image_layout():
    binding = object()
    original = mono(900, 2)
    layout = _image_layout(original)
    targets = (PreviewQualityTarget("sim:left", None, binding, layout),
               PreviewQualityTarget("sim:right", None, binding, layout),
               PreviewQualityTarget("sim:missing", None, unavailable="检查接线"))
    preview = PreviewSamples({"sim:left": [(binding, original), (object(), mono(900, 3))],
                              "sim:right": [(binding, original), (binding, mono(900, 3, shape=(50, 64)))]})
    result = inspect_preview_quality(preview, targets, after=1, cancelled=Event(), timeout=.75)
    assert [row.state for row in result] == ["changed", "changed", "unavailable"]
    assert result[-1].detail == "检查接线"
    report = {"cameras": {target.camera_id: {} for target in targets}, "issues": [], "ready": True}
    assert with_preview_quality(report, result) is report


def test_preview_quality_does_not_turn_a_slow_stream_into_evidence():
    binding = object()
    current = mono(4095, 9)
    targets = (PreviewQualityTarget("sim:live", None, binding, _image_layout(current)),
               PreviewQualityTarget("sim:slow", None, binding, _image_layout(current)))
    preview = PreviewSamples({"sim:live": [(binding, mono(1400, i)) for i in range(10, 15)],
                              "sim:slow": [(binding, current)]})
    result = inspect_preview_quality(preview, targets, after=8, cancelled=Event(), timeout=.32)
    assert [row.state for row in result] == ["ready", "timeout"]
    assert result[1].summary()["frames"] == result[1].summary()["saturated_frames"] == 1
    report = with_preview_quality({"cameras": {"sim:live": {}, "sim:slow": {}}, "issues": []}, result)
    assert tuple(report["preview_quality"]) == ("sim:live",) and not report["issues"]
    cancelled = Event()
    cancelled.set()
    assert inspect_preview_quality(preview, targets, after=8, cancelled=cancelled) == ()
