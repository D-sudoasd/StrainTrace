import json
import subprocess
import sys
import time

import pytest

from straintrace_capture import CaptureConfig, CaptureError, CaptureSession, ExperimentReader, recover_experiment
from straintrace_capture.control import CaptureControl, send_command
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


def test_remote_signal_preserves_pre_and_post_buffers(tmp_path):
    capture = CaptureSession(config([
        {"mode": "event", "fps": 15, "pre_frames": 3, "post_frames": 2, "duration_s": 6},
    ]), tmp_path / "event")
    capture.start()
    try:
        with CaptureControl(capture):
            wait_until(lambda: capture.status()["cameras"]["sim:0"]["received"] >= 5)
            receipt = send_command(capture.output, "trigger", source="DAQ", payload={"label": "start"})
            assert receipt["ok"]
            assert capture.wait(10)["state"] == "completed"
    finally:
        capture.request_stop()
        capture.wait(10)
    with ExperimentReader(capture.output) as reader:
        rows = reader.frames()
        assert [row["event_position"] for row in rows] == ["pre", "pre", "pre", "post", "post"]
        assert rows[0]["sequence"] > 1
        assert {row["event_context_id"] for row in rows} == {receipt["signal_id"]}
        signal, = [event for event in reader.events() if event["kind"] == "signal_received"]
        assert signal["payload"] == {"label": "start"}
        assert reader.verify()["complete"]


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

    publish = storage._publish_tiff

    def fail_third(path, image, **options):
        if path.stem == "000000000003":
            path.with_suffix(".tiff.part").write_bytes(b"torn write")
            raise OSError("injected media error after creating temporary TIFF")
        return publish(path, image, **options)

    monkeypatch.setattr(storage, "_publish_tiff", fail_third)
    capture = CaptureSession(config([{"mode": "continuous", "frame_count": 10}]), tmp_path / "fault")
    capture.start()
    final = capture.wait(6)
    assert final["state"] == "failed" and final["error"]["code"] == "WRITE_FAILED"
    assert final["cameras"]["sim:0"]["saved"] == 2
    assert final["cameras"]["sim:0"]["app_unsaved"] >= 1
    with ExperimentReader(capture.output) as reader:
        result = reader.verify()
        assert result["committed_frames"] == 2
        assert [issue["code"] for issue in result["issues"]] == ["UNCOMMITTED_FILE"]


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
