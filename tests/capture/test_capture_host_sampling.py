import queue
import time
from contextlib import closing
from dataclasses import replace
from threading import Event

import cv2
import numpy as np
import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader
from straintrace_capture.cameras.base import Device, Image, TriggerSupport, receipt_times
from straintrace_capture.cameras.host_sampler import HostFrameSampler
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.cameras.uvc import UvcCamera
from straintrace_capture.config import Phase
from straintrace_capture.storage import ExperimentWriter
from straintrace_capture.trial import prepare_trial, trial_report


def test_sampler_retains_first_post_request_receipt_and_excludes_stale_frames():
    class Stream:
        device = Device("test:stream", "test", "Test", "Live", None, "simulated")
        incoming = queue.Queue()

        def read(self, timeout_s):
            try:
                return self.incoming.get(timeout=timeout_s)
            except queue.Empty:
                return None

    def make_frame(value):
        return Image(np.full((4, 4), value, dtype=np.uint8), "Mono8", 8, "mono", *receipt_times(), value, value)

    source = Stream()
    observed = queue.Queue()
    errors = []
    sampler = HostFrameSampler(source, lambda image, chosen: observed.put((image.native_counter, chosen)), errors.append)
    stale = make_frame(1)
    with sampler:
        sampler.trigger()
        source.incoming.put(stale)
        source.incoming.put(make_frame(2))
        source.incoming.put(make_frame(3))
        assert [observed.get(timeout=2) for _ in range(3)] == [(1, False), (2, True), (3, False)]
        selected = sampler.read(1)
        assert selected.native_counter == 2 and int(selected.pixels[0, 0]) == 2
        assert stale.received_monotonic_ns < selected.metadata["host_selection_request_monotonic_ns"] < selected.received_monotonic_ns
        assert sampler.read(.02) is None
    assert not errors


def test_uvc_latches_first_eligible_driver_receipt_even_if_read_is_delayed(monkeypatch):
    camera = UvcCamera(0)
    monkeypatch.setattr(camera, "features", lambda: {})
    stale = Image(np.zeros((4, 5, 3), dtype=np.uint8), "BGR8", 8, "bgr", *receipt_times())
    stalled, stale_processed = Event(), Event()
    release_stale, release_new, newer_published = Event(), Event(), Event()
    calls = iter(range(1, 6))

    def driver_receipt():
        number = next(calls)
        if number == 1:
            stalled.set()
            assert release_stale.wait(3)
            return stale
        if number == 2:
            stale_processed.set()
            assert release_new.wait(3)
        if number == 4:
            newer_published.set()
            assert camera._stop.wait(3)
        return Image(np.full(stale.pixels.shape, number, dtype=np.uint8), "BGR8", 8, "bgr", *receipt_times())

    monkeypatch.setattr(camera, "_get_image", driver_receipt)
    camera.start("software")
    try:
        assert stalled.wait(3)
        camera.trigger()
        release_stale.set()
        assert stale_processed.wait(3)
        assert camera.read(0.01) is None
        release_new.set()
        assert newer_published.wait(3)
        selected = camera.read(0.5)
        assert selected is not None and int(selected.pixels[0, 0, 0]) == 2
        requested = selected.metadata["host_selection_request_monotonic_ns"]
        assert stale.received_monotonic_ns < requested < selected.received_monotonic_ns
        assert selected.metadata["frame_selection"] == "first_host_receipt_after_request"
    finally:
        release_stale.set()
        release_new.set()
        camera.stop()


def test_uvc_driver_selection_is_preserved_in_trial_and_trigger_group(tmp_path, monkeypatch):
    class Driver:
        def isOpened(self):
            return True

        def getBackendName(self):
            return "Fixture"

        def get(self, key):
            return {cv2.CAP_PROP_FRAME_WIDTH: 80, cv2.CAP_PROP_FRAME_HEIGHT: 60, cv2.CAP_PROP_FPS: 35.}.get(key, 0.)

        def read(self):
            time.sleep(.02)
            return True, np.zeros((60, 80, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _index, _api: Driver())
    original = CaptureConfig.from_dict({
        "config_version": 1, "name": "Driver and camera trigger", "min_free_bytes": 0,
        "phases": [{"mode": "timed", "fps": 5., "frame_count": 3}],
        "cameras": [{"id": "uvc:0"}, {"id": "sim:1", "settings": {"width": 80, "height": 60}}],
    })
    config, path = prepare_trial(original, tmp_path / "planned", phase_number=1, seconds=1.)
    session = CaptureSession(config, path)
    session.start()
    result = session.wait(8)
    assert result["state"] == "completed", result
    with ExperimentReader(path) as reader:
        assert reader.verify()["committed_images"] == 6
        assert reader.verify()["complete"]
        raw_software = reader.events(kind="software_trigger")
        assert len(raw_software) == 3 and all(event["synchronization"] == "host_receipt_only" for event in raw_software)
        groups = reader.pairing()["groups"]
        assert all(group["camera_methods"] == {"uvc:0": "host_next_receipt", "sim:1": "software_frame_start"}
                   for group in groups)
        for row in reader.frames(camera_id="uvc:0"):
            details = row["backend_metadata"]
            assert details["frame_selection"] == "first_host_receipt_after_request"
            assert details["exposure_time_observed"] is False
            assert details["software_trigger_dispatch_monotonic_ns"] <= details["host_selection_request_monotonic_ns"]
            assert details["host_selection_request_monotonic_ns"] < row["timestamps"]["received_monotonic"]["value"]
    summary = trial_report(path)["cameras"]
    assert summary["uvc:0"]["host_unselected_frames"] is None
    assert summary["uvc:0"]["host_selection_delay_ms"]["median"] > 0
    assert summary["sim:1"]["host_selection_delay_ms"] is None

    historical = tmp_path / "legacy"
    with closing(ExperimentWriter(historical, original)) as writer:
        writer.event("software_trigger", {
            "phase": 1, "expected_cameras": ["uvc:0", "sim:1"], "pairing_basis": "common_capture_request",
            "synchronization": "unverified", "camera_methods": {"uvc:0": "host_next_receipt"},
        }, event_id="legacy-uvc")
        writer.set_state("stopped")
    with ExperimentReader(historical) as reader:
        group, _ = reader.trigger_group("legacy-uvc")
        assert group["synchronization"] == "host_receipt_only"
        assert reader.events(kind="software_trigger")[0]["synchronization"] == "unverified"


class StreamOnlyCamera(SimulatedCamera):
    def __init__(self):
        super().__init__(0)
        self.device = replace(self.device, id="test:free-run", backend="test")

    def trigger_support(self):
        return TriggerSupport(False, False, "device_nodes", ("TriggerMode",))

    def start(self, trigger="free", source=None):
        assert trigger == "free"
        super().start(trigger, source)

    def trigger(self):
        pytest.fail("host selection must not trigger the free-running device")


def _mixed_config(phase):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "Mixed software and streamed receipt", "min_free_bytes": 0,
        "phases": [{**phase, "software_trigger_fallback": True}],
        "cameras": [{"id": key, "settings": {"width": 64, "height": 48, "fps": 40.}}
                    for key in ("test:free-run", "sim:1")],
    })


@pytest.mark.parametrize(("phase", "rounds"), [
    ({"mode": "timed", "fps": 6, "frame_count": 4}, 4),
    ({"mode": "manual", "frame_count": 2, "duration_s": 4}, 2),
    ({"mode": "event", "fps": 6, "pre_frames": 2, "post_frames": 2, "duration_s": 4}, 4),
])
def test_mixed_host_selection_preserves_the_request_schedule_and_source_counters(tmp_path, phase, rounds):
    registry = CameraRegistry(custom={"test": (lambda: [], lambda _: StreamOnlyCamera())})
    session = CaptureSession(_mixed_config(phase), tmp_path / "requests", registry=registry)
    session.start()
    try:
        limit = time.monotonic() + 5
        if phase["mode"] != "timed":
            while True:
                progress = session.status()["phase_progress"]
                if progress and progress["trigger_available"]:
                    if phase["mode"] == "manual" or progress["event_window"]["state"] == "waiting":
                        break
                if session.finished or time.monotonic() > limit:
                    pytest.fail(f"could not arm capture: {session.status()}")
                time.sleep(.02)
            assert progress["host_sampling_cameras"] == ["test:free-run"]
            if phase["mode"] == "manual":
                time.sleep(.12)
                session.request_trigger(expected_phase=1)
                time.sleep(.12)
                session.request_trigger(expected_phase=1)
            else:
                session.request_trigger(expected_phase=1, require_prebuffer=True)
        result = session.wait(8)
        assert result["state"] == "completed", result
    finally:
        if not session.finished:
            session.request_stop()
            session.wait(8)

    stats = result["cameras"]["test:free-run"]
    assert stats["host_unselected_frames"] > 0 and stats["native_counter_gaps"] == 0
    assert result["cameras"]["sim:1"]["host_unselected_frames"] == 0
    with ExperimentReader(session.output) as reader:
        verified = reader.verify()
        assert verified["complete"] and verified["committed_frames"] == 2 * rounds
        groups = reader.pairing()["groups"]
        assert len(groups) == rounds
        for group in groups:
            assert group["status"] == "associated" and group["synchronization"] == "host_receipt_only"
            assert group["camera_methods"] == {"test:free-run": "host_next_receipt", "sim:1": "software_frame_start"}
        frames = reader.frames(camera_id="test:free-run")
        assert any(right["native_counter"] > left["native_counter"] + 1 for left, right in zip(frames, frames[1:]))
        for frame in frames:
            detail = frame["backend_metadata"]
            assert detail["frame_selection"] == "first_host_receipt_after_request"
            assert detail["software_trigger_dispatch_monotonic_ns"] <= detail["host_selection_request_monotonic_ns"]
            assert detail["host_selection_request_monotonic_ns"] < frame["timestamps"]["received_monotonic"]["value"]


def test_idle_host_sample_dispatch_fails_promptly_on_device_loss(tmp_path):
    class FailedCamera(StreamOnlyCamera):
        def read(self, timeout_s):
            raise CaptureError("CAMERA_DISCONNECTED", "the free-running stream was lost")

    registry = CameraRegistry(custom={"test": (lambda: [], lambda _: FailedCamera())})
    session = CaptureSession(_mixed_config({"mode": "manual"}), tmp_path / "disconnected", registry=registry)
    session.start()
    status = session.wait(6)
    assert status["state"] == "failed" and status["error"]["code"] == "CAMERA_DISCONNECTED"


def test_phase_rejects_fallback_for_physical_or_unrequested_stream_modes():
    assert "software_trigger_fallback" not in Phase.from_dict({"mode": "single"}).to_dict()
    with pytest.raises(CaptureError, match="software_trigger_fallback"):
        Phase.from_dict({"mode": "hardware", "hardware_source": "Line1", "software_trigger_fallback": True})
    with pytest.raises(CaptureError, match="software_trigger_fallback"):
        Phase.from_dict({"mode": "timed", "fps": 2, "frame_count": 3, "software_trigger_fallback": "True"})
