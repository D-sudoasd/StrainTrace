import json
import subprocess
import sys
import threading
import time

import pytest

from straintrace_capture import CaptureConfig, CaptureError, CaptureSession, ExperimentReader, recover_experiment
from straintrace_capture.cli import main
from straintrace_capture.control import CaptureControl
from straintrace_capture.handoff import analysis_snapshot


def config(phases, *, cameras=("sim:0",), **options):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "Test coupon",
        "cameras": [{"id": camera, "settings": {
            "width": 128, "height": 120, **options.pop(camera, {}),
        }} for camera in cameras],
        "phases": phases, **options,
    })


def wait_until(check, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if result := check():
            return result
        time.sleep(0.025)
    raise AssertionError("condition did not become true")


def test_calibration_frames_excluded_and_two_cameras_paired_by_request(tmp_path):
    capture = CaptureSession(config([
        {"mode": "single", "role": "calibration"},
        {"mode": "single", "role": "reference"},
        {"mode": "timed", "fps": 8, "frame_count": 2},
    ], cameras=("sim:0", "sim:1")), tmp_path / "experiment")
    capture.start()
    assert capture.wait(10)["state"] == "completed"
    with ExperimentReader(capture.output) as reader:
        pairs = reader.pairing()
        assert len(pairs["groups"]) == 4
        assert all(pair["status"] == "associated" and pair["synchronization"] == "unverified" for pair in pairs["groups"])
        assert all(set(pair["frames_by_camera"]) == {"sim:0", "sim:1"} for pair in pairs["groups"])
    with pytest.raises(CaptureError, match="select one registered camera"):
        analysis_snapshot(capture.output)
    snapshot = analysis_snapshot(capture.output, "sim:1")
    assert [frame["sequence"] for frame in snapshot["frames"]] == [2, 3, 4]
    assert snapshot["reference_frame_id"].endswith("/sim:1/2")


def test_remote_signal_preserves_pre_and_post_buffers(tmp_path, capsys):
    capture = CaptureSession(config([
        {"mode": "event", "fps": 15, "pre_frames": 3, "post_frames": 2, "duration_s": 6},
    ]), tmp_path / "event")
    capture.start()
    try:
        with CaptureControl(capture):
            wait_until(lambda: capture.status()["cameras"]["sim:0"]["received"] >= 5)
            assert main(["status", str(capture.output), "--live"]) == 0
            progress = json.loads(capsys.readouterr().out)["status"]["phase_progress"]
            assert progress["trigger_available"] and progress["event_window"]["pre_buffered_rounds"] == 3
            assert not list((capture.output / "control").iterdir())
            assert main(["trigger", str(capture.output), "--source", "DAQ", "--payload", '{"label":"start"}',
                         "--phase", "1", "--require-prebuffer"]) == 0
            receipt = json.loads(capsys.readouterr().out)
            assert receipt["ok"]
            assert capture.wait(10)["state"] == "completed"
    finally:
        capture.request_stop()
        capture.wait(10)
    with ExperimentReader(capture.output) as reader:
        rows = reader.frames()
        assert [row["event_position"] for row in rows if row["event_position"] != "boundary"] == [
            "pre", "pre", "pre", "post", "post",
        ]
        assert rows[0]["sequence"] > 1
        assert {row["event_context_id"] for row in rows} == {receipt["signal_id"]}
        signal, = [event for event in reader.events() if event["kind"] == "signal_received"]
        assert signal["payload"] == {"label": "start"}
        window, = [event for event in reader.events() if event["kind"] == "event_window"]
        assert window["signal_id"] == receipt["signal_id"]
        assert window["pre_retained_rounds"] == 3 and window["post_received_rounds"] == 2
        assert window["boundary_rounds"] == sum(row["event_position"] == "boundary" for row in rows)
        assert reader.verify()["complete"]


def test_guarded_early_signal_can_be_retried_and_unguarded_signal_remains_available(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    capture = CaptureSession(config([
        {"mode": "event", "fps": 5, "pre_frames": 4, "post_frames": 2, "duration_s": 6},
    ]), tmp_path / "early")
    collect_prebuffer = capture._prebuffer

    def pause_prebuffer(*args):
        entered.set()
        assert release.wait(5)
        collect_prebuffer(*args)

    monkeypatch.setattr(capture, "_prebuffer", pause_prebuffer)
    capture.start()
    try:
        assert entered.wait(5)
        progress = capture.status()["phase_progress"]
        assert progress["trigger_available"] and progress["event_window"]["state"] == "buffering"
        with pytest.raises(CaptureError) as early:
            capture.request_trigger(expected_phase=1, require_prebuffer=True)
        assert early.value.code == "PREBUFFER_NOT_READY"
        signal_id = capture.request_trigger(expected_phase=1)
        queued = capture.status()["phase_progress"]
        assert not queued["trigger_available"] and queued["event_window"]["state"] == "trigger_queued"
        with pytest.raises(CaptureError) as duplicate:
            capture.request_trigger(expected_phase=1)
        assert duplicate.value.code == "EVENT_ALREADY_RECEIVED"
    finally:
        release.set()
        result = capture.wait(10)
    assert result["state"] == "completed" and result["short_pre_windows"] == 1
    with ExperimentReader(capture.output) as reader:
        window, = [event for event in reader.events() if event["kind"] == "event_window"]
        assert window["signal_id"] == signal_id
        assert window["pre_retained_rounds"] == 0 and window["post_received_rounds"] == 2
        assert [row["event_position"] for row in reader.frames()] == ["post", "post"]
        assert reader.verify()["ok"] and not reader.verify()["complete"]


def test_guard_checks_retained_receipts_instead_of_a_stale_full_indication(tmp_path, monkeypatch):
    from straintrace_capture import session as capture_module

    capture = CaptureSession(config([
        {"mode": "event", "fps": 10, "pre_frames": 3, "post_frames": 2, "duration_s": 6},
    ]), tmp_path / "stale")
    capture.start()

    def event_ready():
        progress = capture.status()["phase_progress"]
        return progress and progress["event_window"]["state"] == "waiting"

    try:
        wait_until(event_ready)
        stale_receipt = capture_module.receipt_times()
        earlier_count = capture.status()["cameras"]["sim:0"]["received"]
        wait_until(lambda: capture.status()["cameras"]["sim:0"]["received"] >= earlier_count + 5)
        real_receipt_times = capture_module.receipt_times
        caller = threading.current_thread()
        with monkeypatch.context() as patched:
            patched.setattr(capture_module, "receipt_times",
                            lambda: stale_receipt if threading.current_thread() is caller else real_receipt_times())
            with pytest.raises(CaptureError) as stale:
                capture.request_trigger(expected_phase=1, require_prebuffer=True)
        assert stale.value.code == "PREBUFFER_NOT_READY"
        capture.request_trigger(expected_phase=1, require_prebuffer=True)
        assert capture.wait(10)["short_pre_windows"] == 0
    finally:
        capture.request_stop()
        capture.wait(10)


def test_signal_arriving_before_a_rotation_keeps_its_qualified_window(tmp_path, monkeypatch):
    from straintrace_capture import session as capture_module

    capture = CaptureSession(config([
        {"mode": "event", "fps": 10, "pre_frames": 2, "post_frames": 2, "duration_s": 6},
    ]), tmp_path / "rotation")
    entered, release = threading.Event(), threading.Event()
    retained = []
    collect_prebuffer = capture._prebuffer

    def pause_before_eviction(executor, cameras, phase, index, due, ring, window):
        if len(ring) == 3 and not entered.is_set():
            retained.extend(item.image.received_monotonic_ns for batch in ring for item in batch)
            entered.set()
            assert release.wait(5)
        collect_prebuffer(executor, cameras, phase, index, due, ring, window)

    monkeypatch.setattr(capture, "_prebuffer", pause_before_eviction)
    capture.start()
    try:
        assert entered.wait(5)
        during_window = retained[1] + 1
        assert during_window < retained[2]
        real_receipt_times = capture_module.receipt_times
        caller = threading.current_thread()
        with monkeypatch.context() as patched:
            patched.setattr(capture_module, "receipt_times",
                            lambda: (time.time_ns(), during_window) if threading.current_thread() is caller else real_receipt_times())
            signal = capture.request_trigger(expected_phase=1, require_prebuffer=True)
    finally:
        release.set()
        result = capture.wait(10)
    assert result["state"] == "completed" and result["short_pre_windows"] == 0
    with ExperimentReader(capture.output) as reader:
        window, = [event for event in reader.events() if event["kind"] == "event_window"]
        assert window["signal_id"] == signal and window["pre_retained_rounds"] == 2
        assert [row["event_position"] for row in reader.frames()] == ["pre", "pre", "boundary", "post", "post"]
        assert reader.verify()["complete"]


def test_a_delayed_operator_command_cannot_fire_a_new_manual_phase(tmp_path):
    capture = CaptureSession(config([
        {"mode": "manual", "role": "reference", "frame_count": 1},
        {"mode": "manual", "frame_count": 1},
    ]), tmp_path / "phases")
    capture.start()
    try:
        wait_until(lambda: capture.status()["mode"] == "manual")
        capture.request_trigger(expected_phase=1)
        wait_until(lambda: capture.status()["phase"] == 2)
        with pytest.raises(CaptureError) as delayed:
            capture.request_trigger(expected_phase=1)
        assert delayed.value.code == "SIGNAL_PHASE_CHANGED"
        progress = capture.status()["phase_progress"]
        assert progress["received_by_camera"] == {"sim:0": 0}
        capture.request_trigger(expected_phase=2)
        assert capture.wait(10)["state"] == "completed"
    finally:
        capture.request_stop()
        capture.wait(10)
    with ExperimentReader(capture.output) as reader:
        signals = [event for event in reader.events() if event["kind"] == "signal_received"]
        assert [event["phase"] for event in signals] == [1, 2]


def test_disk_reserve_and_disconnect_protect_existing_images(tmp_path):
    stopped = CaptureSession(config([{"mode": "single"}], min_free_bytes=10**18), tmp_path / "reserve")
    stopped.start()
    result = stopped.wait(5)
    assert result["state"] == "failed"
    assert result["error"]["code"] == "DISK_RESERVE"
    assert result["cameras"]["sim:0"]["app_unsaved"] == 1
    with ExperimentReader(stopped.output) as reader:
        assert not reader.verify()["complete"]
        assert not reader.frames()
    failure = CaptureSession(config([{"mode": "continuous", "frame_count": 9}],
                                    **{"sim:0": {"disconnect_after": 3}}), tmp_path / "disconnect")
    failure.start()
    result = failure.wait(5)
    assert result["state"] == "failed"
    assert result["error"]["code"] == "CAMERA_DISCONNECTED"
    with ExperimentReader(failure.output) as reader:
        assert reader.verify()["ok"]
        assert reader.verify()["committed_frames"] == 3
        assert reader.pairing()["unassociated_frames"] == {"sim:0": 3}
    with pytest.raises(CaptureError, match="new experiment directory"):
        CaptureSession(failure.config, failure.output).start()


def test_write_failure_reports_unsaved_and_retains_prior_commits(tmp_path, monkeypatch):
    from straintrace_capture import storage

    stage = storage._stage_tiff

    def fail_third(path, image, **options):
        if path.stem == "000000000003":
            path.with_suffix(".tiff.part").write_bytes(b"torn write")
            raise OSError("injected media error after creating temporary TIFF")
        return stage(path, image, **options)

    monkeypatch.setattr(storage, "_stage_tiff", fail_third)
    capture = CaptureSession(config([{"mode": "continuous", "frame_count": 10}]), tmp_path / "fault")
    capture.start()
    final = capture.wait(6)
    assert final["state"] == "failed" and final["error"]["code"] == "WRITE_FAILED"
    assert final["cameras"]["sim:0"]["saved"] == 2
    assert final["cameras"]["sim:0"]["pending_write"] == 0
    assert final["cameras"]["sim:0"]["app_unsaved"] >= 1
    with ExperimentReader(capture.output) as reader:
        result = reader.verify()
        assert result["committed_frames"] == 2
        assert [issue["code"] for issue in result["issues"]] == ["UNCOMMITTED_FILE"]
        failed, = (row for row in reader.timeline()["events"] if row["kind"] == "error")
        assert (failed["phase"], failed["camera_id"], failed["details"]["sequence"]) == (1, "sim:0", 3)
        assert 0 < failed["details"]["frame_received_utc_ns"] <= failed["utc_ns"]


def test_byte_pressure_stops_capture_without_inventing_frames(tmp_path):
    capture = CaptureSession(config([{"mode": "single"}], max_buffer_bytes=100), tmp_path / "pressure")
    capture.start()
    status = capture.wait(6)
    assert status["state"] == "failed"
    assert status["error"]["code"] == "RAW_BUFFER_FULL"
    with ExperimentReader(capture.output) as reader:
        assert reader.frames() == []
        assert reader.verify()["reported_camera_losses"] == {"sim:0": {"app_unsaved": 1}}


def test_kill_then_recover_preserves_committed_frames_and_reports_orphans(tmp_path):
    procedure = config([{"mode": "continuous"}])
    settings = tmp_path / "plan.json"
    settings.write_text(json.dumps(procedure.to_dict()), encoding="utf-8")
    experiment = tmp_path / "interrupted"
    process = subprocess.Popen([sys.executable, "-m", "straintrace_capture", "run",
                                "--config", str(settings), "--output", str(experiment)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def committed_reader():
        if not (experiment / "capture.sqlite3").is_file():
            return None
        try:
            with ExperimentReader(experiment) as reader:
                return reader.frames() if reader.experiment["state"] == "running" else None
        except CaptureError:
            return None

    try:
        first_frame = wait_until(committed_reader)[0]
        with pytest.raises(CaptureError, match="writer lease"):
            recover_experiment(experiment)
    finally:
        process.kill()
        process.wait(timeout=6)
    artificial = experiment / first_frame["path"]
    artificial.with_name("uncommitted.tiff.part").write_bytes(b"incomplete write")
    def try_recovery():
        try:
            return recover_experiment(experiment)
        except CaptureError as exc:
            if exc.code != "WRITER_ACTIVE":
                raise
            return None

    evidence = wait_until(try_recovery)
    assert evidence["state"] == "interrupted"
    assert evidence["committed_frames"] >= 1
    assert any(issue["path"].endswith("uncommitted.tiff.part") for issue in evidence["issues"])
    with ExperimentReader(experiment) as reader:
        assert reader.frames()[0] == first_frame
