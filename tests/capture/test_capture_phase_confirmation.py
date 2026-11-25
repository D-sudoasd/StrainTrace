import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from straintrace_capture import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan, prepare_trial
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.cli import main
from straintrace_capture.config import Phase
from straintrace_capture.control import send_command
from straintrace_capture.presentation import format_integrity, format_readiness
from straintrace_capture.storage import ExperimentWriter


NOTE = "核对夹具和载荷准备后开始。"


def plan(*phases):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "阶段确认试验",
        "cameras": [{"id": "sim:0", "settings": {"width": 96, "height": 80}}],
        "phases": list(phases),
    })


def wait_for(check, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := check():
            return result
        time.sleep(.015)
    raise AssertionError("timed out waiting for phase confirmation test")


def pending(session, state="waiting"):
    gate = session.status().get("phase_confirmation")
    return gate if gate and gate["state"] == state else None


class LoggedCamera(SimulatedCamera):
    def __init__(self, experiment):
        super().__init__(0)
        self.experiment = experiment
        self.starts = []

    def start(self, trigger="free", source=None):
        with ExperimentReader(self.experiment) as reader:
            events = reader.events(kind="phase_confirmed")
        self.starts.append((time.perf_counter_ns(), events))
        super().start(trigger, source)


def make_session(tmp_path, *phases):
    output = tmp_path / "experiment"
    camera = LoggedCamera(output)
    registry = CameraRegistry(custom={"sim": (lambda: [camera.device], lambda _key: camera)})
    return CaptureSession(plan(*phases), output, registry=registry), camera


def test_confirmation_waits_for_commit_and_then_arms_with_a_fresh_timer(tmp_path, monkeypatch):
    session, camera = make_session(
        tmp_path, {"mode": "single", "role": "reference"},
        {"mode": "timed", "fps": 12, "frame_count": 2, "duration_s": .40, "start_confirmation": NOTE},
    )
    writing, release = threading.Event(), threading.Event()
    commit = ExperimentWriter.commit_frames

    def hold_reference(writer, items):
        if items[0].record["role"] == "reference":
            writing.set()
            assert release.wait(7)
        return commit(writer, items)

    monkeypatch.setattr(ExperimentWriter, "commit_frames", hold_reference)
    session.start()
    try:
        assert writing.wait(5)
        locked = wait_for(lambda: pending(session, "saving_previous"))
        assert session.status()["queue_size"] == 0
        assert session.status()["buffered_raw_bytes"] == 0
        assert session.status()["cameras"]["sim:0"]["saved"] == 0
        assert len(camera.starts) == 1 and camera._mode is None
        with pytest.raises(CaptureError) as early:
            session.request_phase_confirmation(2, confirmation_id=locked["confirmation_id"])
        assert early.value.code == "PHASE_CONFIRMATION_SAVING"
        with ExperimentReader(session.output) as reader:
            assert not reader.frames() and not reader.events(kind="phase_confirmation_waiting")
        release.set()

        gate = wait_for(lambda: pending(session))
        assert gate["confirmation_id"] == locked["confirmation_id"] and gate["instruction"] == NOTE
        with ExperimentReader(session.output) as reader:
            assert [frame["role"] for frame in reader.frames()] == ["reference"]
            assert reader.events(kind="phase_confirmation_waiting")[0]["confirmation_id"] == gate["confirmation_id"]
        time.sleep(.45)
        assert pending(session) and session.status()["phase_progress"] is None and len(camera.starts) == 1
        for number, identity in ((1, gate["confirmation_id"]), (2, "wrong")):
            with pytest.raises(CaptureError) as stale:
                session.request_phase_confirmation(number, confirmation_id=identity)
            assert stale.value.code == "PHASE_CONFIRMATION_CHANGED"
        with pytest.raises(CaptureError) as shot:
            session.request_trigger(expected_phase=2)
        assert shot.value.code in {"NOT_ARMED", "SIGNAL_PHASE_CHANGED"}

        assert session.request_phase_confirmation(2, confirmation_id=gate["confirmation_id"],
                                                  source="操作员甲") == gate["confirmation_id"]
        assert session.wait(7)["state"] == "completed"
        with ExperimentReader(session.output) as reader:
            report = reader.verify()
            assert report["complete"] and report["committed_frames"] == 3
            assert [frame["phase"] for frame in reader.frames()] == [1, 2, 2]
        waiting, confirmed, started = report["phase_confirmation_events"]
        assert [event["kind"] for event in (waiting, confirmed, started)] == [
            "phase_confirmation_waiting", "phase_confirmed", "phase_started",
        ]
        assert {event["confirmation_id"] for event in (waiting, confirmed, started)} == {gate["confirmation_id"]}
        assert confirmed["source"] == "操作员甲"
        arm_time, already_committed_events = camera.starts[1]
        assert waiting["waiting_since_monotonic_ns"] <= confirmed["requested_monotonic_ns"] <= arm_time
        assert arm_time <= started["camera_start_returned_monotonic_ns"]
        assert already_committed_events == [confirmed]
        assert "阶段开始确认记录（主机 UTC）" in format_integrity(report)
    finally:
        release.set()
        if not session.finished:
            session.request_stop()
            session.wait(8)


def test_stop_while_waiting_never_arms_or_takes_the_next_image(tmp_path):
    session, camera = make_session(
        tmp_path, {"mode": "single", "role": "reference"}, {"mode": "single", "start_confirmation": NOTE},
    )
    session.start()
    try:
        gate = wait_for(lambda: pending(session))
        session.request_stop()
        with pytest.raises(CaptureError) as rejected:
            session.request_phase_confirmation(2, confirmation_id=gate["confirmation_id"])
        assert rejected.value.code == "PHASE_CONFIRMATION_NOT_PENDING"
        assert session.wait(5)["state"] == "stopped"
        assert len(camera.starts) == 1
        with ExperimentReader(session.output) as reader:
            report = reader.verify()
            assert report["ok"] and report["committed_frames"] == 1
        waiting, cancelled = report["phase_confirmation_events"]
        assert cancelled["kind"] == "phase_confirmation_cancelled" and not cancelled["confirmation_requested"]
        assert cancelled["confirmation_id"] == waiting["confirmation_id"] == gate["confirmation_id"]
    finally:
        if not session.finished:
            session.request_stop()
            session.wait(6)


def test_duplicate_confirmation_cannot_retry_and_a_failed_ledger_write_prevents_arming(tmp_path, monkeypatch):
    session, camera = make_session(tmp_path, {"mode": "single", "start_confirmation": NOTE})
    recording, release = threading.Event(), threading.Event()
    event = ExperimentWriter.event

    def fail_confirmation(writer, kind, details, **kwargs):
        if kind == "phase_confirmed":
            recording.set()
            assert release.wait(7)
            raise OSError("cannot commit phase confirmation")
        return event(writer, kind, details, **kwargs)

    monkeypatch.setattr(ExperimentWriter, "event", fail_confirmation)
    session.start()
    try:
        gate = wait_for(lambda: pending(session))
        session.request_phase_confirmation(1, confirmation_id=gate["confirmation_id"])
        assert recording.wait(5) and pending(session, "arming")
        with pytest.raises(CaptureError) as duplicate:
            session.request_phase_confirmation(1, confirmation_id=gate["confirmation_id"])
        assert duplicate.value.code == "PHASE_CONFIRMATION_REQUESTED"
        release.set()
        status = session.wait(6)
        assert status["state"] == "failed" and status["error"]["code"] == "PHASE_CONFIRMATION_LEDGER_FAILED"
        assert not camera.starts
        with ExperimentReader(session.output) as reader:
            assert not reader.frames()
            assert not reader.events(kind="phase_confirmed") and not reader.events(kind="phase_started")
    finally:
        release.set()
        if not session.finished:
            session.request_stop()
            session.wait(8)


def test_stop_after_confirmation_is_recorded_can_still_cancel_arming(tmp_path, monkeypatch):
    session, camera = make_session(tmp_path, {"mode": "single", "start_confirmation": NOTE})
    recording, release = threading.Event(), threading.Event()
    event = ExperimentWriter.event

    def hold_after_confirmation(writer, kind, details, **kwargs):
        event(writer, kind, details, **kwargs)
        if kind == "phase_confirmed":
            recording.set()
            assert release.wait(7)

    monkeypatch.setattr(ExperimentWriter, "event", hold_after_confirmation)
    session.start()
    try:
        gate = wait_for(lambda: pending(session))
        session.request_phase_confirmation(1, confirmation_id=gate["confirmation_id"])
        assert recording.wait(5)
        session.request_stop()
        release.set()
        assert session.wait(6)["state"] == "stopped" and not camera.starts
        with ExperimentReader(session.output) as reader:
            assert not reader.frames() and not reader.events(kind="phase_started")
            cancelled, = reader.events(kind="phase_confirmation_cancelled")
            assert cancelled["confirmation_requested"] and cancelled["confirmation_id"] == gate["confirmation_id"]
    finally:
        release.set()
        if not session.finished:
            session.request_stop()
            session.wait(8)


def test_confirmation_is_round_tripped_in_plans_and_preflight_but_a_trial_runs_directly(tmp_path):
    phase = Phase.from_dict({"mode": "timed", "fps": 12, "frame_count": 4, "start_confirmation": f"  {NOTE}  "})
    assert phase.start_confirmation == NOTE
    updated = plan(phase.to_dict())
    path = tmp_path / "formal"
    inspection = inspect_plan(updated, path)
    assert inspection["ready"] and inspection["phases"][0]["start_confirmation"] == NOTE
    assert f"确认后开始 · {NOTE}" in format_readiness(inspection)
    trial, _ = prepare_trial(updated, path)
    assert trial.phases[0].start_confirmation is None
    assert updated.phases[0].start_confirmation == NOTE
    assert "start_confirmation" not in Phase.from_dict({"mode": "single"}).to_dict()


@pytest.mark.parametrize("value", [None, True, "", " \t "])
def test_empty_or_nontext_confirmation_fails_configuration(value):
    with pytest.raises(CaptureError, match="start_confirmation"):
        Phase.from_dict({"mode": "single", "start_confirmation": value})


def test_headless_plan_announces_confirmation_and_preserves_external_trigger_opt_in(tmp_path, capsys, monkeypatch):
    output, config = tmp_path / "output", tmp_path / "plan.json"
    config.write_text(json.dumps(plan(
        {"mode": "single", "role": "reference"}, {"mode": "single", "start_confirmation": NOTE},
    ).to_dict()), encoding="utf-8")
    receipt, messages = tmp_path / "receipt.json", tmp_path / "messages.jsonl"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with receipt.open("wb") as stdout, messages.open("wb") as stderr:
        process = subprocess.Popen([sys.executable, "-B", "-m", "straintrace_capture", "run", "--config", str(config),
                                    "--output", str(output)], stdout=stdout, stderr=stderr, creationflags=flags)
        try:
            def announced():
                assert process.poll() is None, messages.read_text(encoding="utf-8")
                lines = messages.read_text(encoding="utf-8").splitlines()
                return next((data for line in lines if (data := json.loads(line))["event"] == "phase_confirmation_required"),
                            None)

            prompt = wait_for(announced, timeout=10)
            assert prompt["phase"] == 2 and prompt["instruction"] == NOTE
            signal = send_command(output, "trigger", expected_phase=2)
            assert signal["error"]["code"] == "CONTROL_TRIGGER_DISABLED"
            assert main(["status", str(output), "--live"]) == 0
            status = json.loads(capsys.readouterr().out)["status"]
            assert status["phase_confirmation"]["confirmation_id"] == prompt["confirmation_id"]
            blocked = []
            read_text = Path.read_text

            def read_with_sharing_violation(path, *args, **kwargs):
                if path.suffix == ".response" and path.is_file() and not blocked:
                    blocked.append(path)
                    raise PermissionError(13, "response temporarily locked", str(path))
                return read_text(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(Path, "read_text", read_with_sharing_violation)
                assert main(["confirm-phase", str(output), "--phase", "2", "--confirmation-id", prompt["confirmation_id"],
                             "--source", "workstation"]) == 0
            assert len(blocked) == 1
            reply = json.loads(capsys.readouterr().out)
            assert reply == {"ok": True, "state": "confirmation_requested", "confirmation_id": prompt["confirmation_id"]}
            assert process.wait(10) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(5)
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "completed"
    with ExperimentReader(output) as reader:
        assert reader.verify()["complete"]
        confirmed, = reader.events(kind="phase_confirmed")
        assert confirmed["source"] == "workstation"
