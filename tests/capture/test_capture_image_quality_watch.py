import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from jsonschema import Draft202012Validator

from straintrace_capture import (
    CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan, prepare_trial, trial_report,
)
from straintrace_capture import image_quality_watch as quality
from straintrace_capture.cameras.base import Feature, Image
from straintrace_capture.cameras.settings_probe import SettingsCheck
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.presentation import format_trial
from straintrace_capture.preview import image_quality
from straintrace_capture.storage import ExperimentWriter


def texture():
    rng = np.random.default_rng(42)
    return np.clip(rng.normal(1800, 480, (64, 96)), 50, 3650).astype(np.uint16)


def raw(pixels):
    return Image(pixels, "Mono12", 12, "mono", time.time_ns(), time.perf_counter_ns())


def test_quality_watch_debounces_raw_indicators_and_restarts_on_new_phase():
    original = raw(texture())
    blurred = raw(cv2.GaussianBlur(original.pixels, (11, 11), 2.4))
    flash = raw(np.full(original.pixels.shape, 4095, np.uint16))
    watch = quality.ImageQualityWatch()
    index = 0

    def update(image, phase=1):
        nonlocal index
        index += 1
        return watch.update("sim:0", phase, index, image, quality.measure_primary_image(image),
                            compare_sharpness=True)

    for candidate in (original, original, flash, original):
        assert update(candidate) is None
    assert watch.snapshot()["cameras"]["sim:0"]["reference_samples"] == 3
    assert update(blurred) is None
    assert update(blurred) is None
    alert = update(blurred)
    assert alert["started"] == ["sharpness_down"]
    assert alert["metrics"]["sharpness_ratio"] < quality.SHARPNESS_ALERT
    assert update(original) is None
    assert update(original) is None
    assert update(original)["cleared"] == ["sharpness_down"]

    for _ in range(2):
        assert update(flash) is None
    assert update(flash)["started"] == ["near_saturation"]
    assert update(flash, phase=2) is None
    report = watch.snapshot()["cameras"]["sim:0"]
    assert report["phase"] == 2 and report["reference_samples"] == 0 and not report["alerts"]
    report["metrics"]["near_saturation_fraction"] = -1
    assert watch.snapshot()["cameras"]["sim:0"]["metrics"]["near_saturation_fraction"] == 1
    unlit = raw(np.zeros(original.pixels.shape, np.uint16))
    assert update(unlit, phase=2) is None
    assert update(unlit, phase=2) is None
    assert update(unlit, phase=2)["started"] == ["near_black"]


def test_exposure_sampling_caps_wide_images_and_checks_color_channels(monkeypatch):
    pixels = np.full((3, 2048, 3), 1800, np.uint16)
    pixels[..., 2] = 4095
    image = Image(pixels, "BGR12", 12, "bgr", 1, 2)
    report = image_quality(image, pixel_limit=89)
    assert sum(report["histogram"]) <= 89
    assert report["bright_fraction"] == 1 and report["dark_fraction"] == 0

    def unexpected_texture_scan(*_args):
        pytest.fail("the capture watch must keep focused region measurements bounded")

    monkeypatch.setattr("ezdic_core.roi_texture_metrics", unexpected_texture_scan)
    sample = quality.measure_primary_image(image, roi=(3, 0, 2000, 3))
    assert sample["near_saturation_fraction"] == 1 and sample["near_black_fraction"] == 0


class ChangingCamera(SimulatedCamera):
    def __init__(self, index):
        super().__init__(index)
        self.sharp = texture()
        self.soft = cv2.GaussianBlur(self.sharp, (11, 11), 2.4)
        self.clipped = np.full(self.sharp.shape, 4095, np.uint16)

    def read(self, timeout_s):
        frame = super().read(timeout_s)
        if frame is None:
            return None
        return replace(frame, pixels=self.expected(frame.native_counter))

    def expected(self, sequence):
        if 13 <= sequence <= 33:
            return self.soft if self.device.id == "sim:1" else self.clipped
        return self.sharp


def simulated_plan(count=48):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "采集画质验证",
        "cameras": [{"id": f"sim:{i}", "settings": {"width": 96, "height": 64, "pixel_format": "Mono12"}}
                    for i in range(2)],
        "phases": [{"mode": "timed", "fps": 18, "frame_count": count}],
    })


def registry():
    return CameraRegistry(custom={"sim": (list, lambda identity: ChangingCamera(int(identity[-1])))})


def test_quality_regions_round_trip_and_preflight_detects_a_new_sensor_window(tmp_path):
    frame = raw(texture())
    region = quality.quality_region_from_preview(frame, (6, 8, 28, 30), {"offset_x": 4})
    data = simulated_plan().to_dict()
    data["quality_regions"] = {"sim:0": region.to_dict()}
    config = CaptureConfig.from_dict(data)
    assert config.to_dict() == data
    schema = Path(__file__).resolve().parents[2] / "schemas" / "capture_config_v1.json"
    Draft202012Validator(json.loads(schema.read_text(encoding="utf8"))).validate(config.to_dict())

    def check(offset):
        return inspect_plan(config, tmp_path / "formal", preview_images={"sim:0": frame},
                            camera_settings={"sim:0": SettingsCheck("preview_readback", readback={
                                "offset_x": Feature("offset_x", offset, 0, 100, 2),
                            })})

    assert check(4)["quality_regions"]["sim:0"]["verified_preview"]
    displaced = check(8)
    assert displaced["ready"]
    change, = [issue for issue in displaced["issues"] if issue["code"] == "QUALITY_REGION_CHANGED"]
    assert change["camera"] == "sim:0" and change["fallback"] == "sensor_window"
    assert quality.quality_region_mismatch(region, raw(texture()[:, :90]), {"offset_x": 4}) == "image_layout"
    data["cameras"][0]["settings"]["offset_x"] = 8
    planned = inspect_plan(CaptureConfig.from_dict(data), tmp_path / "formal")
    assert planned["quality_regions"]["sim:0"]["fallback"] == "sensor_window"

    with pytest.raises(CaptureError, match="quality region cameras"):
        CaptureConfig.from_dict({**data, "quality_regions": {"sim:9": region.to_dict()}})
    data["quality_regions"]["sim:0"]["bounds"][2] = 96
    with pytest.raises(CaptureError, match="within its original image"):
        CaptureConfig.from_dict(data)


def test_trial_uses_selection_and_records_whole_frame_fallback_without_cropping(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "SAMPLE_INTERVAL_S", .040)
    bounds = (17, 11, 40, 28)
    x, y, width, height = bounds
    pixels = np.zeros((64, 96), np.uint16)
    pixels[y:y + height, x:x + width] = texture()[:height, :width]

    class DarkBackgroundCamera(ChangingCamera):
        def expected(self, _sequence):
            return pixels

    data = simulated_plan(count=26).to_dict()
    data["quality_regions"] = {
        f"sim:{index}": quality.quality_region_from_preview(raw(pixels), bounds, {"offset_x": offset}).to_dict()
        for index, offset in enumerate((0, 4))
    }
    plan = CaptureConfig.from_dict(data)
    formal = tmp_path / "formal"
    trial, destination = prepare_trial(plan, formal, seconds=4)
    source = CameraRegistry(custom={"sim": (list, lambda identity: DarkBackgroundCamera(int(identity[-1])))})
    run = CaptureSession(trial, destination, registry=source, watch_quality=True)
    run.start()
    assert run.wait(10)["state"] == "completed"

    report = trial_report(destination, inspect_images=False)
    assert report["observed"] and not formal.exists()
    scoped, fallback = (report["image_quality_watch"]["cameras"][f"sim:{index}"] for index in range(2))
    assert scoped["region"] == {"bounds": list(bounds), "fallback": None}
    assert not scoped["alerts"] and scoped["metrics"]["near_black_fraction"] == 0
    assert fallback["region"]["fallback"] == "sensor_window" and fallback["alerts"] == ["near_black"]
    assert fallback["metrics"]["near_black_fraction"] == pytest.approx(1 - width * height / pixels.size)
    assert "传感器画幅偏移或采样参数不匹配" in format_trial(report)
    with ExperimentReader(destination) as reader:
        checked = reader.verify()
        assert checked["complete"] and checked["committed_images"] == 52
        assert checked["file_hashes_checked"]
        assert all(np.array_equal(tifffile.imread(reader.image_path(frame)), pixels) for frame in reader.frames())
        notices = [row for row in reader.timeline()["events"] if row["kind"] == quality.NOTICE_EVENT]
        scope_rows = [row for row in notices if row["details"].get("region_changed")]
        assert {row["camera_id"] for row in scope_rows} == {"sim:0", "sim:1"} and len(scope_rows) == 2
        for row in scope_rows:
            frame = reader.frame_at_sequence(row["camera_id"], row["details"]["sequence"])
            assert frame["timestamps"]["received_utc"]["value"] == row["utc_ns"]


def test_live_watch_records_exact_sample_frames_and_preserves_originals(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "SAMPLE_INTERVAL_S", .040)
    run = CaptureSession(simulated_plan(), tmp_path / "watched", registry=registry(), watch_quality=True)
    run.start()
    status = run.wait(10)
    assert status["state"] == "completed" and status["scheduled_misses"] == 0
    assert status["image_quality_watch"]["measurements"] > 12
    assert not status["image_quality_watch"]["ledger_errors"]
    armed = {"sim:0": "near_saturation", "sim:1": "sharpness_down"}
    with ExperimentReader(run.output) as reader:
        assert reader.verify()["complete"]
        assert reader.experiment["summary"]["image_quality_watch"] == status["image_quality_watch"]
        notices = reader.events(kind=quality.NOTICE_EVENT)
        assert len(notices) == 4
        timeline = reader.timeline()
        assert sum(row["kind"] == quality.NOTICE_EVENT for row in timeline["events"]) == 4
        assert sum(bucket["incidents"] for bucket in timeline["overview"]["buckets"]) == 0
        for item in notices:
            cam = item["camera_id"]
            code = armed[cam]
            assert code in (item["started"] or item["cleared"])
            sequence = item["sequence"]
            assert (13 <= sequence <= 33) if item["started"] else sequence > 33
            frame = reader.frame_at_sequence(cam, sequence)
            assert frame["phase"] == item["phase"] == 1
            assert frame["timestamps"]["received_utc"]["value"] == item["received_utc_ns"]
        for i in range(2):
            source = ChangingCamera(i)
            for frame in reader.frames(camera_id=f"sim:{i}"):
                assert np.array_equal(tifffile.imread(reader.image_path(frame)), source.expected(frame["sequence"]))


def test_write_pressure_defers_background_samples(tmp_path):
    run = CaptureSession(simulated_plan(), tmp_path / "deferred", watch_quality=True)
    # Supply the same state the worker reads while waiting on a slow writer.
    run._running_phase = object()
    run._queue_bytes = (run.config.max_buffer_bytes + 1) // 2
    candidate = run._next_quality_sample(deque(("sim:0",)), {})
    report = run._quality_watch.snapshot()
    assert candidate is None and report["deferred"]
    assert report["deferred_intervals"] == 1 and not report["measurements"]


def test_optional_monitor_failures_leave_capture_completion_and_hashes_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "SAMPLE_INTERVAL_S", .025)
    measure = quality.measure_primary_image
    store_event = ExperimentWriter.event
    attempts = 0

    def unreliable(image):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("test-only measurement failure")
        return measure(image)

    def event(writer, kind, record, **kwargs):
        if kind == quality.NOTICE_EVENT:
            raise OSError("test-only notice write failure")
        return store_event(writer, kind, record, **kwargs)

    monkeypatch.setattr(quality, "measure_primary_image", unreliable)
    monkeypatch.setattr(ExperimentWriter, "event", event)
    run = CaptureSession(simulated_plan(36), tmp_path / "keep-recording", registry=registry(), watch_quality=True)
    run.start()
    state = run.wait(10)
    assert state["state"] == "completed"
    assert state["image_quality_watch"]["measurement_errors"] == 1
    assert state["image_quality_watch"]["ledger_errors"] >= 2
    with ExperimentReader(run.output) as reader:
        assert reader.verify()["committed_frames"] == 72
