"""Fault-injected acquisition checks using synthetic cameras and temporary data."""

import threading
import time
from dataclasses import replace

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.storage import ExperimentWriter


def _config(camera_ids, phase, **options):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "acquisition integrity", "min_free_bytes": 0,
        "cameras": [{"id": identity, "settings": {"width": 64, "height": 48}}
                    for identity in camera_ids],
        "phases": [phase], **options,
    })


def test_stop_interrupts_an_outstanding_software_round_and_keeps_received_frame(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class WaitingCamera(SimulatedCamera):
        def __init__(self):
            super().__init__(1)
            self.device = replace(self.device, id="test:waiting", backend="test")

        def read(self, timeout_s):
            entered.set()
            if release.wait(timeout_s):
                return super().read(0)
            return None

    registry = CameraRegistry(custom={"test": (lambda: [], lambda _: WaitingCamera())})
    capture = CaptureSession(_config(["sim:0", "test:waiting"],
                                    {"mode": "single", "frame_timeout_s": 20}),
                             tmp_path / "stop", registry=registry)
    capture.start()
    try:
        assert entered.wait(5)
        deadline = time.monotonic() + 5
        while not capture.status()["cameras"]["sim:0"]["received"] and time.monotonic() < deadline:
            time.sleep(.01)
        assert capture.status()["cameras"]["sim:0"]["received"] == 1
        capture.request_stop()
        result = capture.wait(3)
    finally:
        # Unblock the old behaviour too, so a regression never leaves a worker running.
        release.set()
        if not capture.finished:
            capture.wait(25)

    assert result["state"] == "stopped" and result["error"] is None
    assert result["cameras"]["sim:0"]["saved"] == 1
    assert result["cameras"]["test:waiting"]["received"] == 0
    with ExperimentReader(capture.output) as reader:
        assert len(reader.frames(camera_id="sim:0")) == 1
        assert reader.events(kind="trigger_incomplete")
        assert not reader.verify()["complete"]


def test_counter_diagnostic_write_failure_keeps_the_received_image(tmp_path, monkeypatch):
    class DuplicateCounter(SimulatedCamera):
        def __init__(self):
            super().__init__(0)
            self.device = replace(self.device, id="test:duplicate", backend="test")

        def read(self, timeout_s):
            image = super().read(timeout_s)
            return replace(image, native_counter=1) if image is not None else None

    event = ExperimentWriter.event

    def fail_diagnostic(writer, kind, details, **options):
        if kind == "native_duplicate":
            raise OSError("injected counter diagnostic write failure")
        return event(writer, kind, details, **options)

    monkeypatch.setattr(ExperimentWriter, "event", fail_diagnostic)
    registry = CameraRegistry(custom={"test": (lambda: [], lambda _: DuplicateCounter())})
    capture = CaptureSession(_config(["test:duplicate"], {"mode": "continuous", "frame_count": 2}),
                             tmp_path / "counter", registry=registry)
    capture.start()
    result = capture.wait(8)

    assert result["state"] == "failed"
    assert result["error"]["code"] == "DIAGNOSTIC_LEDGER_WRITE_FAILED"
    stats = result["cameras"]["test:duplicate"]
    assert (stats["received"], stats["saved"], stats["pending_write"], stats["app_unsaved"]) == (2, 2, 0, 0)
    with ExperimentReader(capture.output) as reader:
        assert len(reader.frames()) == 2
        assert not reader.verify()["complete"]


def test_hardware_identity_diagnostic_failure_keeps_the_received_image(tmp_path, monkeypatch):
    class HardwareCamera(SimulatedCamera):
        def __init__(self):
            super().__init__(0)
            self.device = replace(self.device, id="test:hardware", backend="test")

        def start(self, trigger="free", source=None):
            assert trigger == "hardware" and source == "Line1"
            super().start("free")

    event = ExperimentWriter.event

    def fail_diagnostic(writer, kind, details, **options):
        if kind == "hardware_identity_unavailable":
            raise OSError("injected hardware identity diagnostic failure")
        return event(writer, kind, details, **options)

    monkeypatch.setattr(ExperimentWriter, "event", fail_diagnostic)
    registry = CameraRegistry(custom={"test": (lambda: [], lambda _: HardwareCamera())})
    phase = {"mode": "hardware", "frame_count": 1, "hardware_source": "Line1", "hardware_pairing": {
        "source_id": "test-pulse", "counter_feature": "ChunkFrameID", "first_counter": {"test:hardware": 1},
    }}
    capture = CaptureSession(_config(["test:hardware"], phase), tmp_path / "hardware", registry=registry)
    capture.start()
    result = capture.wait(8)

    assert result["state"] == "failed" and result["error"]["code"] == "DIAGNOSTIC_LEDGER_WRITE_FAILED"
    stats = result["cameras"]["test:hardware"]
    assert (stats["received"], stats["saved"], stats["hardware_identity_missing"], stats["app_unsaved"]) == (1, 1, 1, 0)
    with ExperimentReader(capture.output) as reader:
        assert len(reader.frames()) == 1
        assert not reader.verify()["complete"]


def test_incomplete_trigger_diagnostic_failure_keeps_other_camera_frame(tmp_path, monkeypatch):
    good_read = threading.Event()

    class GoodCamera(SimulatedCamera):
        def __init__(self):
            super().__init__(0)
            self.device = replace(self.device, id="test:good", backend="test")

        def read(self, timeout_s):
            image = super().read(timeout_s)
            good_read.set()
            return image

    class FailedCamera(SimulatedCamera):
        def __init__(self):
            super().__init__(1)
            self.device = replace(self.device, id="test:failed", backend="test")

        def read(self, timeout_s):
            assert good_read.wait(5)
            raise CaptureError("CAMERA_DISCONNECTED", "injected disconnect")

    event = ExperimentWriter.event

    def fail_diagnostic(writer, kind, details, **options):
        if kind == "trigger_incomplete":
            raise OSError("injected trigger diagnostic write failure")
        return event(writer, kind, details, **options)

    monkeypatch.setattr(ExperimentWriter, "event", fail_diagnostic)
    registry = CameraRegistry(custom={"test": (lambda: [],
                                               lambda identity: GoodCamera() if identity == "test:good" else FailedCamera())})
    capture = CaptureSession(_config(["test:good", "test:failed"], {"mode": "single"}),
                             tmp_path / "partial", registry=registry)
    capture.start()
    result = capture.wait(8)

    assert result["state"] == "failed" and result["error"]["code"] == "CAMERA_DISCONNECTED"
    stats = result["cameras"]["test:good"]
    assert (stats["received"], stats["saved"], stats["pending_write"], stats["app_unsaved"]) == (1, 1, 0, 0)
    with ExperimentReader(capture.output) as reader:
        assert len(reader.frames(camera_id="test:good")) == 1
        assert not reader.verify()["complete"]


def test_unexpected_writer_exception_cannot_be_reported_as_completed(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def fail_checkpoint(_writer):
        entered.set()
        assert release.wait(8)
        raise RuntimeError("injected unexpected writer failure")

    monkeypatch.setattr(ExperimentWriter, "frame_journal_needs_checkpoint", fail_checkpoint)
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "writer failure", "min_free_bytes": 0, "durability": "buffered",
        "cameras": [{"id": "sim:0", "settings": {"width": 2048, "height": 1024, "fps": 50.}}],
        "phases": [{"mode": "continuous", "frame_count": 4}],
    })
    capture = CaptureSession(plan, tmp_path / "writer")
    capture.start()
    try:
        assert entered.wait(8)
        deadline = time.monotonic() + 8
        while capture.status()["cameras"]["sim:0"]["received"] < 4 and time.monotonic() < deadline:
            time.sleep(.01)
        assert capture.status()["cameras"]["sim:0"]["received"] == 4
    finally:
        release.set()
        result = capture.wait(10)

    assert result["state"] == "failed"
    assert result["error"]["code"] == "WRITER_THREAD_FAILED"
    with ExperimentReader(capture.output) as reader:
        committed = len(reader.frames())
        assert reader.experiment["state"] == "failed"
        assert not reader.verify()["complete"]
    stats = result["cameras"]["sim:0"]
    assert stats["saved"] == committed
    assert stats["app_unsaved"] == stats["received"] - committed
    assert stats["pending_write"] == 0


def test_writer_exception_releases_event_drain_and_pending_marker(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    capture = CaptureSession(_config(["sim:0"], {
        "mode": "event", "fps": 5, "pre_frames": 0, "post_frames": 2, "duration_s": 10,
    }), tmp_path / "event-crash")
    write_events = capture._write_event_records

    def fail_on_signal():
        if not capture._event_writes.empty():
            entered.set()
            assert release.wait(8)
            raise RuntimeError("injected event writer crash")
        write_events()

    monkeypatch.setattr(capture, "_write_event_records", fail_on_signal)
    capture.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            progress = capture.status()["phase_progress"]
            if progress and progress["trigger_available"]:
                break
            time.sleep(.01)
        else:
            raise AssertionError("event phase did not become ready")
        capture.request_trigger()
        assert entered.wait(5)
        _marker_id, receipt = capture.request_marker("test marker")
    finally:
        release.set()
        result = capture.wait(10)

    assert result["state"] == "failed" and result["error"]["code"] == "WRITER_THREAD_FAILED"
    assert receipt.done()
    assert isinstance(receipt.exception(), CaptureError)
    assert receipt.exception().code == "MARKER_WRITE_FAILED"


def test_writer_failure_is_signalled_while_an_earlier_commit_is_still_blocked(tmp_path, monkeypatch):
    from straintrace_capture import storage

    entered, release = threading.Event(), threading.Event()
    insert = storage._insert_frames

    def hold_commit(db, rows):
        entered.set()
        assert release.wait(8)
        insert(db, rows)

    def fail_after_commit_started(_writer):
        assert entered.wait(5)
        raise RuntimeError("injected writer crash during a blocked commit")

    monkeypatch.setattr(storage, "_insert_frames", hold_commit)
    monkeypatch.setattr(ExperimentWriter, "frame_journal_needs_checkpoint", fail_after_commit_started)
    plan = CaptureConfig.from_dict({
        "config_version": 1, "name": "blocked ledger", "min_free_bytes": 0, "durability": "buffered",
        "cameras": [{"id": "sim:0", "settings": {"width": 2048, "height": 1024}}],
        "phases": [{"mode": "single"}],
    })
    capture = CaptureSession(plan, tmp_path / "blocked-ledger")
    capture.start()
    try:
        assert entered.wait(5)
        deadline = time.monotonic() + 2
        while capture.status()["error"] is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert capture.status()["error"]["code"] == "WRITER_THREAD_FAILED"
    finally:
        release.set()
        result = capture.wait(10)
    assert result["state"] == "failed"
