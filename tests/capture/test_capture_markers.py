import json
import queue
import threading
import time

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, CaptureSession, ExperimentReader
from straintrace_capture import session as session_module
from straintrace_capture.cameras.base import Image
from straintrace_capture.cameras.simulated import simulated_device
from straintrace_capture.cli import main
from straintrace_capture.control import CaptureControl
from straintrace_capture.frame_search import FrameLocation
from straintrace_capture.image_inputs import verified_image_path
from straintrace_capture.markers import MARKER_EVENT
from straintrace_capture.storage import ExperimentWriter


_UTC = 1_795_598_100_123_456_789


def plan(phases, n=2):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "实验标记",
        "cameras": [{"id": f"sim:{i}", "settings": {"width": 64, "height": 48}} for i in range(n)],
        "phases": phases,
    })


def wait_for(condition):
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        if found := condition():
            return found
        time.sleep(.025)
    raise AssertionError("condition did not become true")


def test_marker_neighbors_use_host_monotonic_across_utc_changes_missing_sides_and_ties(tmp_path):
    cfg = plan([{"mode": "single", "role": "reference"}, {"mode": "timed", "fps": 4, "frame_count": 3}], n=3)
    root = tmp_path / "recording"
    writer = ExperimentWriter(root, cfg)
    try:
        for i in range(3):
            writer.register_camera(simulated_device(i), {})
        clock = f"{writer.experiment_id}:host_monotonic"
        for camera, sequence, phase, utc_offset, monotonic in (
                ("sim:0", 1, 1, 20, 99), ("sim:0", 2, 2, -50, 105),
                ("sim:0", 3, 2, 3, 150), ("sim:0", 4, 2, 6, 105),
                ("sim:1", 1, 1, 200, 81)):
            image = Image(np.full((8, 10), sequence, np.uint8), "Mono8", 8, "mono", _UTC + utc_offset, monotonic)
            writer.commit_frames([writer.prepare_frame(camera, sequence, phase, cfg.phases[phase - 1].role,
                                                       image, None, None, None, 0)])
        marker = {"phase": 2, "marked_monotonic_ns": 100, "marked_utc_ns": _UTC,
                  "monotonic_clock_id": clock, "label": "开始滑移", "source": "operator"}
        writer.record_markers([("slip", marker), ("exact", {**marker, "marked_monotonic_ns": 105}),
                               ("foreign-clock", {**marker, "monotonic_clock_id": "another:host_monotonic"})])
        writer.set_state("completed")
    finally:
        writer.close()

    with ExperimentReader(root) as reader:
        access = reader.marker_context("slip")
        first = access["cameras"]["sim:0"]
        assert access["time_basis"] == "host_receive_monotonic"
        assert first["before"]["sequence"] == 1 and first["before"]["offset_ns"] == -1
        assert first["after"]["sequence"] == 2 and first["after"]["offset_ns"] == 5
        assert first["before"]["phase"] == 1 and first["after"]["phase"] == 2
        assert first["before"]["received_utc_ns"] > access["marker"]["marked_utc_ns"] > first["after"]["received_utc_ns"]
        assert access["cameras"]["sim:1"]["before"]["offset_ns"] == -19
        assert access["cameras"]["sim:1"]["after"] is None
        assert access["cameras"]["sim:2"] == {"before": None, "after": None}
        exact = reader.marker_context("exact")["cameras"]["sim:0"]
        assert exact["before"]["offset_ns"] == 0 and exact["after"]["sequence"] == 3
        row = first["after"]
        location = FrameLocation(access["experiment_id"], row["camera_id"], row["sequence"],
                                 row["ordinal"], row["frame_id"], row["sha256"])
        assert verified_image_path(reader, location.read(reader)).is_file()
        with pytest.raises(CaptureError, match="comparable"):
            reader.marker_context("foreign-clock")
        with pytest.raises(CaptureError, match="different experiment"):
            reader.marker_context("slip", expected_experiment_id="wrong-run")
        with pytest.raises(CaptureError, match="no saved marker"):
            reader.marker_context("missing")

        page = reader.timeline(phase=2, camera_id="sim:1", markers_only=True, limit=2)
        assert page["markers_only"]
        assert all(item["kind"] == MARKER_EVENT and item["utc_ns"] == _UTC for item in page["events"])
        assert sum(bucket["markers"] for bucket in page["overview"]["buckets"]) == 3
        older = reader.timeline(phase=2, camera_id="sim:1", markers_only=True, limit=2,
                                before=page["older_before"], snapshot=page["snapshot_ordinal"])
        assert [item["marker_id"] for item in older["events"]] == ["slip"]


def test_live_marker_command_preserves_manual_wait_until_explicit_shot(tmp_path, capsys):
    capture = CaptureSession(plan([{"mode": "manual", "frame_count": 1}]), tmp_path / "live")
    capture.start()
    try:
        with CaptureControl(capture):
            wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
            args = ["mark", str(capture.output), "--label", "  观察到滑移  ", "--source", "load-controller"]
            assert main(args) == 0
            receipt = json.loads(capsys.readouterr().out)
            assert receipt["state"] == "marker_queued"

            def committed():
                with ExperimentReader(capture.output) as reader:
                    return reader.events(kind=MARKER_EVENT)

            saved, = wait_for(committed)
            assert (saved["event_id"], saved["label"], saved["source"]) == (
                receipt["marker_id"], "观察到滑移", "load-controller")
            status = capture.status()
            assert not status["phase_progress"]["queued_signals"]
            assert all(item["received"] == 0 for item in status["cameras"].values())
            capture.request_trigger()
            assert capture.wait(7)["state"] == "completed"
    finally:
        capture.request_stop()
        capture.wait(7)
    with pytest.raises(CaptureError) as stopped:
        capture.request_marker()
    assert stopped.value.code == "MARKER_NOT_RECORDING"
    assert main(["marker-context", str(capture.output), receipt["marker_id"]]) == 0
    context = json.loads(capsys.readouterr().out)
    for sides in context["cameras"].values():
        assert sides["before"] is None and sides["after"]["offset_ns"] > 0
    with ExperimentReader(capture.output) as reader:
        assert reader.verify()["complete"]


def test_stop_drains_accepted_markers_before_closing_the_ledger(tmp_path, monkeypatch):
    entered, resume = threading.Event(), threading.Event()
    original = ExperimentWriter.record_markers

    def pause(writer, batch):
        entered.set()
        assert resume.wait(7)
        return original(writer, batch)

    monkeypatch.setattr(ExperimentWriter, "record_markers", pause)
    capture = CaptureSession(plan([{"mode": "manual", "frame_count": 1}]), tmp_path / "stopping")
    capture.start()
    try:
        wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
        identity, future = capture.request_marker("夹具开始移动")
        assert entered.wait(5)
        capture.request_stop()
        assert not future.done() and not capture.finished
    finally:
        resume.set()
        capture.request_stop()
        assert capture.wait(7)["state"] == "stopped"
    assert future.result(0) == identity
    with ExperimentReader(capture.output) as reader:
        assert [event["event_id"] for event in reader.events(kind=MARKER_EVENT)] == [identity]


def test_marker_and_producer_end_wake_idle_writer(tmp_path, monkeypatch):
    capture = CaptureSession(plan([{"mode": "manual"}], n=1), tmp_path / "idle-writer")
    idle_waits = queue.Queue()
    idle_wait = capture._writer_ready.wait

    def observe_wait(timeout=None):
        if timeout is None:
            idle_waits.put(None)
        return idle_wait(timeout)

    monkeypatch.setattr(capture._writer_ready, "wait", observe_wait)
    capture.start()
    try:
        wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
        idle_waits.get(timeout=5)
        identity, receipt = capture.request_marker("加载开始")
        assert receipt.result(5) == identity
        idle_waits.get(timeout=5)
    finally:
        capture.request_stop()
        end = capture.wait(7)
    assert end["state"] == "stopped" and end["queue_size"] == 0
    with ExperimentReader(capture.output) as reader:
        assert not reader.frames()
        assert [event["event_id"] for event in reader.events(kind=MARKER_EVENT)] == [identity]


def test_marker_wakes_optional_batch_wait_after_published_frame(tmp_path, monkeypatch):
    capture = CaptureSession(plan([{"mode": "manual"}], n=1), tmp_path / "batch-writer")
    waiting_for_more = threading.Event()
    writer_wait = capture._writer_ready.wait

    def observe_wait(timeout=None):
        if timeout is not None:
            waiting_for_more.set()
        return writer_wait(timeout)

    monkeypatch.setattr(session_module, "_SMALL_BUFFERED_BATCH_WAIT_S", 6.)
    monkeypatch.setattr(capture._writer_ready, "wait", observe_wait)
    capture.start()
    try:
        wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
        capture.request_trigger()
        assert waiting_for_more.wait(5)
        identity, receipt = capture.request_marker("核对这一帧")
        assert receipt.result(3) == identity
        with ExperimentReader(capture.output) as reader:
            assert [row["sequence"] for row in reader.frames()] == [1]
            assert [event["event_id"] for event in reader.events(kind=MARKER_EVENT)] == [identity]
    finally:
        capture.request_stop()
        capture.wait(8)


def test_marker_write_failure_returns_receipt_error_while_capture_can_continue(tmp_path, monkeypatch):
    def rejected(_writer, _batch):
        raise OSError("test marker write failure")

    monkeypatch.setattr(ExperimentWriter, "record_markers", rejected)
    capture = CaptureSession(plan([{"mode": "manual", "frame_count": 1}], n=1), tmp_path / "marker-error")
    capture.start()
    try:
        wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
        with pytest.raises(CaptureError) as invalid:
            capture.request_marker("逾" * 161)
        assert invalid.value.code == "MARKER_LABEL_INVALID"
        _, pending = capture.request_marker("加载变化")
        with pytest.raises(CaptureError) as failed:
            pending.result(5)
        assert failed.value.code == "MARKER_WRITE_FAILED"
        capture.request_trigger()
        assert capture.wait(7)["state"] == "completed"
    finally:
        capture.request_stop()
        capture.wait(7)
    with ExperimentReader(capture.output) as reader:
        assert reader.verify()["committed_images"] == 1
        assert not reader.events(kind=MARKER_EVENT)
