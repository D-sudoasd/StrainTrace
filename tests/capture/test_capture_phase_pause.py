import json
import threading
import time

import pytest

from straintrace_capture import CaptureConfig, CaptureError, CaptureSession, ExperimentReader
from straintrace_capture.cli import main
from straintrace_capture.control import CaptureControl, send_command
from straintrace_capture.storage import ExperimentWriter


def plan(*phases):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "暂停采集",
        "cameras": [{"id": f"sim:{i}", "settings": {"width": 64, "height": 48}} for i in range(2)],
        "phases": list(phases),
    })


def wait_for(condition, timeout=6):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if result := condition():
            return result
        time.sleep(.01)
    raise AssertionError("Capture did not reach the expected state")


def pause_state(capture, state):
    gate = capture.status().get("phase_pause")
    return gate if gate and gate["state"] == state else None


def test_timed_pause_keeps_camera_rounds_and_freezes_the_plan_clock(tmp_path, monkeypatch):
    capture = CaptureSession(plan({"mode": "timed", "fps": 5, "duration_s": .76}), tmp_path / "timed")
    first_round, finish_round = threading.Event(), threading.Event()
    writing_resume, finish_resume = threading.Event(), threading.Event()
    take_round = capture._round
    event = ExperimentWriter.event

    def hold_first(*args, **kwargs):
        batch = take_round(*args, **kwargs)
        if not first_round.is_set():
            first_round.set()
            assert finish_round.wait(6)
        return batch

    def hold_resume(writer, kind, record, **kwargs):
        if kind == "phase_resume_requested":
            writing_resume.set()
            assert finish_resume.wait(6)
        return event(writer, kind, record, **kwargs)

    monkeypatch.setattr(capture, "_round", hold_first)
    monkeypatch.setattr(ExperimentWriter, "event", hold_resume)
    capture.start()
    try:
        assert first_round.wait(6)
        identity = capture.request_phase_pause(1, source="fixture-pause")
        assert pause_state(capture, "requested")["pause_id"] == identity
        with pytest.raises(CaptureError) as premature:
            capture.request_phase_resume(1, identity)
        assert premature.value.code == "PHASE_PAUSE_TRANSITION"
        finish_round.set()
        paused = wait_for(lambda: pause_state(capture, "paused"))
        before = capture.status()["phase_progress"]
        assert not before["accepting_images"]
        assert list(before["received_by_camera"].values()) == [1, 1]
        observation, receipt = capture.request_marker("夹具调整")
        assert receipt.result(6) == observation
        assert capture.request_phase_resume(1, identity, source="fixture-resume") == identity
        assert writing_resume.wait(6)
        time.sleep(.85)
        status = capture.status()
        assert status["phase_pause"]["state"] == "resuming"
        assert status["phase_progress"]["elapsed_s"] == before["elapsed_s"]
        assert status["phase_progress"]["received_by_camera"] == before["received_by_camera"]
        assert status["scheduled_misses"] == 0
        finish_resume.set()
        final = capture.wait(8)
    finally:
        finish_round.set()
        finish_resume.set()
        capture.request_stop()
        capture.wait(8)

    assert final["state"] == "completed" and final["scheduled_misses"] == 0
    assert "phase_pause" not in final
    with ExperimentReader(capture.output) as reader:
        assert reader.verify()["complete"]
        assert len(reader.frames()) == 8
        begun, = reader.events(kind="phase_paused")
        continued, = reader.events(kind="phase_resume_requested")
        finished, = reader.events(kind="phase_ended")
        assert begun["paused_monotonic_ns"] == paused["paused_monotonic_ns"]
        assert begun["pause_id"] == continued["pause_id"] == identity
        assert (begun["source"], continued["source"]) == ("fixture-pause", "fixture-resume")
        shift = finished["paused_duration_ns"]
        assert shift >= 850_000_000 and shift > continued["pause_to_request_ns"]
        times = [row["scheduled_monotonic_ns"] for row in reader.events(kind="software_trigger")]
        assert len(times) == 4
        assert abs(times[1] - times[0] - (200_000_000 + shift)) < 1_000
        assert all(abs(b - a - 200_000_000) < 1_000 for a, b in zip(times[1:], times[2:]))
        assert not reader.events(kind="schedule_miss")
        assert [row["status"] for row in reader.pairing()["groups"]] == ["associated"] * 4
        event_page = reader.timeline(phase=1, camera_id="sim:0")
        rows = {row["kind"]: row for row in event_page["events"]}
        assert rows["phase_paused"]["utc_ns"] == begun["paused_utc_ns"]
        assert rows["phase_resume_requested"]["details"]["pause_id"] == identity
        assert rows["phase_ended"]["details"]["paused_duration_ns"] == shift
        assert any(bucket.get("paused") for bucket in event_page["overview"]["buckets"])
        assert sum(bucket["incidents"] for bucket in event_page["overview"]["buckets"]) == 0


def test_cli_pause_and_resume_are_bound_to_the_active_manual_pause(tmp_path, capsys):
    capture = CaptureSession(plan({"mode": "manual", "frame_count": 1}), tmp_path / "manual")
    capture.start()

    def cli(command, *options, code=0):
        assert main([command, str(capture.output), *options]) == code
        return json.loads(capsys.readouterr().out)

    try:
        wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
        with CaptureControl(capture):
            started = cli("pause-phase", "--phase", "1")
            assert started["state"] == "pause_requested"
            identity = started["pause_id"]
            wait_for(lambda: pause_state(capture, "paused"))
            live = cli("status", "--live")["status"]
            assert live["phase_pause"]["pause_id"] == identity
            assert not live["phase_progress"]["trigger_available"]
            assert cli("trigger", "--phase", "1", code=3)["error"]["code"] == "PHASE_PAUSE_TRANSITION"
            stale = cli("resume-phase", "--phase", "1", "--pause-id", "previous", code=3)
            assert stale["error"]["code"] == "PHASE_PAUSE_CHANGED"
            assert cli("resume-phase", "--phase", "1", "--pause-id", identity)["state"] == "resume_requested"
            wait_for(lambda: capture.status()["phase_progress"]["trigger_available"])
            assert cli("trigger", "--phase", "1")["ok"]
            final = capture.wait(8)
    finally:
        capture.request_stop()
        capture.wait(8)
    assert final["state"] == "completed"
    with ExperimentReader(capture.output) as reader:
        assert reader.verify()["committed_frames"] == 2
        assert len(reader.events(kind="signal_received")) == 1


def test_confirmation_only_control_does_not_enable_external_pause(tmp_path):
    capture = CaptureSession(plan({"mode": "manual"}), tmp_path / "limited-control")
    capture.start()
    try:
        wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
        with CaptureControl(capture, allow_trigger=False):
            result = send_command(capture.output, "pause-phase", expected_phase=1)
            assert result["error"]["code"] == "CONTROL_PHASE_PAUSE_DISABLED"
            assert "phase_pause" not in capture.status()
    finally:
        capture.request_stop()
        capture.wait(8)


def test_external_stop_wakes_paused_phase_and_keeps_saved_frames(tmp_path):
    external_stop = threading.Event()
    capture = CaptureSession(plan({"mode": "manual"}, {"mode": "single"}), tmp_path / "stopped")
    capture.start(stop_event=external_stop)
    try:
        wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
        capture.request_trigger()
        wait_for(lambda: all(row["saved"] == 1 for row in capture.status()["cameras"].values()))
        identity = capture.request_phase_pause(1)
        wait_for(lambda: pause_state(capture, "paused"))
        external_stop.set()
        final = capture.wait(6)
    finally:
        external_stop.set()
        capture.wait(8)
    assert final["state"] == "stopped"
    assert "phase_pause" not in final
    with ExperimentReader(capture.output) as reader:
        assert reader.verify()["ok"] and len(reader.frames()) == 2
        stopped, = reader.events(kind="phase_pause_cancelled")
        assert stopped["pause_id"] == identity and stopped["reason"] == "stopped"
        assert stopped["pause_duration_ns"] > 0
        assert [row["phase"] for row in reader.events(kind="phase_started")] == [1]


def test_pause_requested_in_last_round_is_closed_before_next_phase(tmp_path, monkeypatch):
    capture = CaptureSession(
        plan({"mode": "timed", "fps": 2, "frame_count": 1}, {"mode": "single"}),
        tmp_path / "last-round",
    )
    last_round, finish = threading.Event(), threading.Event()
    original = capture._round

    def hold(*args, **kwargs):
        if args[3] == 1:
            last_round.set()
            assert finish.wait(6)
        return original(*args, **kwargs)

    monkeypatch.setattr(capture, "_round", hold)
    capture.start()
    try:
        assert last_round.wait(6)
        identity = capture.request_phase_pause(1)
        finish.set()
        final = capture.wait(8)
    finally:
        finish.set()
        capture.request_stop()
        capture.wait(8)
    assert final["state"] == "completed" and "phase_pause" not in final
    with ExperimentReader(capture.output) as reader:
        assert reader.verify()["committed_frames"] == 4
        cancelled, = reader.events(kind="phase_pause_cancelled")
        assert cancelled["pause_id"] == identity and cancelled["reason"] == "phase_ended"
        assert "pause_duration_ns" not in cancelled and not reader.events(kind="phase_paused")


def test_pause_cannot_delay_an_already_queued_manual_trigger(tmp_path, monkeypatch):
    capture = CaptureSession(plan({"mode": "manual", "frame_count": 2}), tmp_path / "queued-trigger")
    first_round, finish = threading.Event(), threading.Event()
    original = capture._round

    def hold(*args, **kwargs):
        if not first_round.is_set():
            first_round.set()
            assert finish.wait(6)
        return original(*args, **kwargs)

    monkeypatch.setattr(capture, "_round", hold)
    capture.start()
    try:
        wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
        capture.request_trigger()
        assert first_round.wait(6)
        capture.request_trigger()
        with pytest.raises(CaptureError) as queued:
            capture.request_phase_pause(1)
        assert queued.value.code == "PHASE_PAUSE_SIGNALS_QUEUED"
        finish.set()
        assert capture.wait(8)["state"] == "completed"
    finally:
        finish.set()
        capture.request_stop()
        capture.wait(8)
    with ExperimentReader(capture.output) as reader:
        assert reader.verify()["committed_frames"] == 4
        assert not reader.events(kind="phase_paused")


def test_failed_resume_write_ends_acquisition_before_a_new_round(tmp_path, monkeypatch):
    capture = CaptureSession(plan({"mode": "manual"}), tmp_path / "cannot-resume")
    event = ExperimentWriter.event

    def reject(writer, kind, record, **kwargs):
        if kind == "phase_resume_requested":
            raise OSError("injected ledger write failure")
        return event(writer, kind, record, **kwargs)

    monkeypatch.setattr(ExperimentWriter, "event", reject)
    capture.start()
    try:
        wait_for(lambda: (capture.status()["phase_progress"] or {}).get("trigger_available"))
        capture.request_trigger()
        wait_for(lambda: all(row["saved"] == 1 for row in capture.status()["cameras"].values()))
        identity = capture.request_phase_pause(1)
        wait_for(lambda: pause_state(capture, "paused"))
        capture.request_phase_resume(1, identity)
        final = capture.wait(8)
    finally:
        capture.request_stop()
        capture.wait(8)
    assert final["state"] == "failed"
    assert final["error"]["code"] == "PHASE_PAUSE_LEDGER_FAILED"
    assert "phase_pause" not in final
    with ExperimentReader(capture.output) as reader:
        assert reader.verify()["committed_frames"] == 2
        assert not reader.events(kind="phase_resume_requested")
