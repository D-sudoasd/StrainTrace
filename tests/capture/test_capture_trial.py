import json
import threading
import time
from copy import deepcopy
from dataclasses import replace
from math import floor
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest

from straintrace_capture import (
    CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan, prepare_trial, trial_report,
)
from straintrace_capture.cameras.base import Image
from straintrace_capture.cameras.simulated import SimulatedCamera
from straintrace_capture.cli import main
from straintrace_capture.gui import CaptureWorkbench
from straintrace_capture.presentation import format_integrity, format_readiness, format_trial
from straintrace_capture.rate_trial import (
    assess_rate_trial, check_rate_trial, read_rate_trial_source, suggest_trial_rate, with_trial_rate,
)
from straintrace_capture.storage import ExperimentWriter


def _plan(phases, *, camera_id="sim:0", **options):
    return CaptureConfig.from_dict({
        "config_version": 1,
        "name": "setup",
        "cameras": [{"id": camera_id, "settings": {"width": 64, "height": 48, "pixel_format": "Mono12"}}],
        "phases": phases,
        **options,
    })


def test_trial_uses_the_selected_phase_and_preserves_the_formal_settings(tmp_path):
    source = _plan([
        {"mode": "single", "role": "reference"},
        {"mode": "hardware", "hardware_source": "Line1", "frame_count": 80, "duration_s": 20,
         "hardware_pairing": {"source_id": "daq", "counter_feature": "ChunkFramecounter",
                              "first_counter": {"pylon:123": 4}}},
    ], camera_id="pylon:123", durability="write_through", max_buffer_bytes=123456, segment_frames=20,
        metadata={"continuation_of": "old-run"})
    formal = tmp_path / "results" / "planned"
    trial, path = prepare_trial(source, formal, seconds=30)
    assert path.parent == formal.parent and path != formal and not formal.parent.exists()
    assert len(trial.phases) == 1
    phase = trial.phases[0]
    assert phase.mode == "hardware" and phase.hardware_source == "Line1"
    assert phase.frame_count == 80 and phase.duration_s == 20
    assert phase.hardware_pairing is None and source.phases[1].hardware_pairing is not None
    assert trial.cameras == source.cameras and trial.durability == source.durability
    assert trial.max_buffer_bytes == source.max_buffer_bytes and trial.segment_frames == source.segment_frames
    assert source.metadata["continuation_of"] == "old-run" and "continuation_of" not in trial.metadata
    assert trial.metadata["capture_trial"]["requires_fresh_hardware_anchors"]
    with pytest.raises(CaptureError, match="阶段"):
        prepare_trial(source, formal, phase_number=3)


def test_event_trial_checks_that_both_windows_fit_the_selected_duration(tmp_path):
    source = _plan([{"mode": "event", "fps": 10, "pre_frames": 8, "post_frames": 4}])
    trial, output = prepare_trial(source, tmp_path / "formal", seconds=1)
    report = inspect_plan(trial, output)
    blocker, = [issue for issue in report["issues"] if issue["code"] == "EVENT_WINDOW_EXCEEDS_DURATION"]
    assert blocker == {"severity": "block", "code": "EVENT_WINDOW_EXCEEDS_DURATION", "phase": 1,
                       "planned_rounds": 10, "pre_frames": 8, "post_frames": 4}
    assert not report["ready"] and not output.exists()
    feasible, _ = prepare_trial(source, tmp_path / "formal", seconds=2)
    assert inspect_plan(feasible, output)["ready"]
    assert feasible.phases[0].pre_frames == 8 and feasible.phases[0].post_frames == 4


def test_cli_trial_reopens_and_verifies_its_own_evidence(tmp_path, capsys):
    formal = tmp_path / "formal"
    config = _plan([
        {"mode": "single", "role": "reference"},
        {"mode": "timed", "fps": 5, "frame_count": 2},
    ])
    config_path = tmp_path / "plan.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert main(["trial", "--config", str(config_path), "--output", str(formal), "--seconds", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    output = Path(report["output"])
    assert output.parent == formal.parent and not formal.exists()
    assert report["trial"]["phase_number"] == 2 and report["observed"]
    assert report["verification"]["file_hashes_checked"]
    camera = report["cameras"]["sim:0"]
    assert camera["device_support"] == "simulated"
    assert camera["committed_frames"] == camera["frames_with_native_counter"] == 2
    assert camera["host_receive_fps"] > 0
    assert camera["formats"] == [{"shape": [48, 64], "pixel_format": "Mono12", "significant_bits": 12}]
    assert camera["phase_feature_readback"]["width"]["value"] == 64
    assert report["raw_buffer_high_water_bytes"] >= 64 * 48 * 2
    assert report["write_timing"]["acquisition_windows"] == [] and not report["sustained_write_lag"]
    assert main(["trial-report", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == report
    check = ["preflight", "--config", str(config_path), "--output", str(formal), "--trial", str(output)]
    assert main(check) == 0
    comparison = json.loads(capsys.readouterr().out)
    assessment, = comparison["trial_assessments"]
    assert assessment["status"] == "matched" and assessment["phase"] == 2 and assessment["observed"]
    assert comparison["phases"][1]["trial_geometry_cameras"] == ["sim:0"]
    assert comparison["phases"][0]["trial_geometry_cameras"] == []
    assert "使用已校验的试采尺寸" in format_readiness(comparison)

    with ExperimentReader(output) as reader:
        path = reader.image_path(reader.frames(limit=1)[0])
    path.write_bytes(b"damaged")
    assert main(["trial-report", str(output)]) == 4
    damaged = json.loads(capsys.readouterr().out)
    assert not damaged["observed"] and damaged["verification"]["issues"]
    assert any(item["code"] == "TRIAL_FILE_ISSUES" for item in damaged["findings"])
    assert main(check) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["ready"] and comparison["trial_assessments"][0]["status"] == "matched"
    assert not comparison["trial_assessments"][0]["observed"]
    assert not comparison["phases"][1]["trial_geometry_cameras"]
    assert any(item["code"] == "TRIAL_NEEDS_REVIEW" for item in comparison["issues"])


def test_armed_trial_without_any_trigger_reports_no_camera_evidence(tmp_path):
    config = _plan([{"mode": "manual"}])
    trial, output = prepare_trial(config, tmp_path / "formal", seconds=.2)
    session = CaptureSession(trial, output)
    session.start()
    assert session.wait(10)["state"] == "completed"
    report = trial_report(output)
    assert report["cameras_without_frames"] == ["sim:0"]
    assert not report["observed"]
    assert report["cameras"]["sim:0"]["host_receive_fps"] is None
    assert report["findings"] == [{"code": "TRIAL_NO_FRAMES", "cameras": ["sim:0"], "mode": "manual"}]
    contents = format_trial(report, camera_labels={"sim:0": "模拟散斑 0"})
    assert "模拟散斑 0 未保存图像。开始后点“触发一次”" in contents
    assert "模拟散斑 0 · 软件模拟" in contents


class ReadbackCamera(SimulatedCamera):
    def __init__(self, identity):
        super().__init__(0)
        self.device = replace(self.device, id=identity, backend="fixture")
        self.set_features({"width": 64, "height": 48, "pixel_format": "Mono12"})


def test_trial_sizes_apply_only_to_matching_phases_and_yield_to_current_preview(tmp_path, monkeypatch):
    identity = "fixture:0"
    config = _plan([{"mode": "single", "role": "reference"}, {"mode": "single", "role": "test"}],
                   cameras=[{"id": identity, "settings": {}}])
    registry = CameraRegistry(custom={"fixture": (lambda: [], ReadbackCamera)})
    destination = tmp_path / "formal"
    reports = []
    for phase in (2, 1):
        trial, output = prepare_trial(config, destination, phase_number=phase)
        capture = CaptureSession(trial, output, registry=registry)
        capture.start()
        assert capture.wait(10)["state"] == "completed"
        reports.append(trial_report(output))
    assert inspect_plan(config, destination)["raw_payload_bytes"] is None
    one = inspect_plan(config, destination, trial_reports=reports[:1])
    assert one["raw_payload_bytes"] is None
    assert one["phases"][0]["raw_payload_bytes"] is None
    assert one["phases"][1]["raw_payload_bytes"] == 64 * 48 * 2
    both = inspect_plan(config, tmp_path / "renamed-run", trial_reports=reports)
    assert both["ready"] and both["raw_payload_bytes"] == 2 * 64 * 48 * 2
    assert all(row["latest_match"] and row["status"] == "matched" for row in both["trial_assessments"])
    assert not (tmp_path / "renamed-run").exists()

    preview = Image(np.zeros((24, 32), dtype=np.uint8), "Mono8", 8, "mono", 1, 1)
    refreshed = inspect_plan(config, destination, trial_reports=reports, preview_images={identity: preview})
    assert refreshed["raw_payload_bytes"] == 2 * 24 * 32
    assert all(row["status"] == "changed" and row["changes"] == ["preview_geometry"] for row in refreshed["trial_assessments"])
    assert not any(row["trial_geometry_cameras"] for row in refreshed["phases"])

    for changes, group in (
        ({"durability": "write_through"}, "storage"),
        ({"cameras": [{"id": identity, "settings": {"gain_db": 2}}]}, "camera_settings"),
        ({"phases": [{"mode": "single", "role": "reference"}, {"mode": "single", "role": "calibration"}]}, "phase"),
    ):
        modified = CaptureConfig.from_dict({**config.to_dict(), **changes})
        compared = inspect_plan(modified, destination, trial_reports=reports[:1])
        assert compared["trial_assessments"][0]["changes"] == [group]
        assert compared["trial_assessments"][0]["status"] == "changed"
        assert not compared["phases"][1]["trial_geometry_cameras"]

    with monkeypatch.context() as patched:
        patched.setattr("straintrace_capture.readiness.volume_identity", lambda _path: "different volume")
        compared = inspect_plan(config, destination, trial_reports=reports[:1])
        assert compared["trial_assessments"][0]["changes"] == ["volume"]
        assert not compared["phases"][1]["trial_geometry_cameras"]

    archive = deepcopy(reports[0])
    archive["trial"].pop("scope")
    compared = inspect_plan(config, destination, trial_reports=[archive])
    assert compared["trial_assessments"][0]["status"] == "unbound"
    assert not compared["phases"][1]["trial_geometry_cameras"]
    assert "缺少可核对的试采设置" in format_readiness(compared)


def test_trial_reports_measured_rate_and_buffer_pressure_without_claiming_a_camera_fault(tmp_path):
    class SlowerCamera(ReadbackCamera):
        def read(self, timeout_s):
            time.sleep(.16)
            return super().read(timeout_s)

    identity = "fixture:0"
    config = _plan([{"mode": "continuous", "frame_count": 10}], max_buffer_bytes=64 * 48 * 2,
                   cameras=[{"id": identity, "settings": {"fps": 100}}])
    trial, output = prepare_trial(config, tmp_path / "formal", seconds=3)
    registry = CameraRegistry(custom={"fixture": (lambda: [], SlowerCamera)})
    capture = CaptureSession(trial, output, registry=registry)
    capture.start()
    assert capture.wait(10)["state"] == "completed"
    report = trial_report(output)
    assert report["observed"] and not report["verification"]["reported_camera_losses"]
    assert [item["code"] for item in report["findings"]] == ["TRIAL_HIGH_BUFFER", "TRIAL_LOW_RECEIPT_RATE"]
    assert report["findings"][1]["target_fps"] == 100
    assert 0 < report["findings"][1]["observed_fps"] < 60
    assert "缓冲峰值达到上限的 100%" in format_trial(report)
    assert "核对曝光时间、相机输出上限和传输设置" in format_trial(report)


def test_trial_reports_sustained_writer_lag_even_after_all_tiffs_verify(tmp_path, monkeypatch, capsys):
    source = _plan([{"mode": "continuous"}], cameras=[
        {"id": f"sim:{index}", "settings": {"width": 64, "height": 48, "fps": 18}}
        for index in range(2)
    ])
    formal = tmp_path / "formal"
    trial, output = prepare_trial(source, formal, seconds=5.5)
    original = ExperimentWriter.prepare_frame

    def slow_writer(writer, *args, **kwargs):
        time.sleep(.055)
        return original(writer, *args, **kwargs)

    monkeypatch.setattr(ExperimentWriter, "prepare_frame", slow_writer)
    capture = CaptureSession(trial, output)
    capture.start()
    status = capture.wait(25)

    assert status["state"] == "completed"
    assert main(["trial-report", str(output)]) == 4
    report = json.loads(capsys.readouterr().out)
    assert report["observed"] and report["verification"]["file_hashes_checked"]
    for identity, row in report["cameras"].items():
        assert row["committed_frames"] == status["cameras"][identity]["received"]
    assert report["sustained_write_lag"]
    assert "TRIAL_SUSTAINED_WRITE_LAG" in {item["code"] for item in report["findings"]}
    assert report["write_timing"] == status["write_timing"]
    windows = report["write_timing"]["acquisition_windows"]
    assert len(windows) == 2 and all(item["pending_raw_bytes_growth"] > 0 for item in windows)
    assert report["write_timing"]["post_capture_committed_frames"] > 16
    assert "图像通过校验，写入积压需复查" in format_trial(report)
    assert status["write_pressure"]["pending_raw_bytes"] == 0 and status["write_pressure"]["estimate"] is None
    assert status["write_pressure"]["accepted_raw_bytes"] == status["write_pressure"]["committed_raw_bytes"]
    persisted = report["verification"]["write_alerts"]
    alert = next(item for item in persisted if item["kind"] == "sustained")
    assert alert["phase"] == 1 and len(alert["windows"]) == 2 and persisted == status["write_alerts"]
    assert "采集中写入提醒" in format_integrity(report["verification"])

    compared = inspect_plan(source, formal, trial_reports=[report])
    assert compared["ready"] and not formal.exists()
    assert {"severity": "warning", "code": "TRIAL_NEEDS_REVIEW", "phase": 1} in compared["issues"]
    contents = format_readiness(compared)
    assert contents.startswith("采前检查完成 · 试采结果需复查")
    assert "计划设置匹配 · TIFF 校验通过 · 写入积压需复查" in contents

    suggestion, = compared["rate_suggestions"]
    assert suggestion["phase"] == 1 and suggestion["source_experiment_id"] == report["experiment_id"]
    assert suggestion["estimated_raw_bytes_per_s"] <= .8 * min(w["committed_raw_bytes_per_s"] for w in windows)
    assert suggestion["request_rate"] is None
    round_bytes = sum(row["geometry"]["frame_bytes"] for row in report["cameras"].values())
    expected_rate = floor(suggestion["write_budget_raw_bytes_per_s"] / round_bytes * 100) / 100
    assert all(item["current"] == 18 and item["proposed"] == expected_rate
               for item in suggestion["camera_rates"].values())
    revised = with_trial_rate(source, suggestion)
    assert revised.phases == source.phases and not formal.exists()
    assert all(item.settings["fps"] == 18 for item in source.cameras)
    checked, proposed_output = prepare_trial(
        revised, formal, phase_number=1, seconds=suggestion["trial_seconds"],
        rate_source_experiment_id=suggestion["source_experiment_id"], rate_source_directory=output,
    )
    candidate = CaptureSession(checked, proposed_output)
    candidate.start()
    assert candidate.wait(25)["state"] == "completed"
    proposed = trial_report(proposed_output)
    assert check_rate_trial(suggestion, report, proposed) == "verified"
    assert proposed["trial"]["rate_source_experiment_id"] == report["experiment_id"]
    assert checked.name.endswith("建议频率复测")
    assert Path(proposed["trial"]["rate_source"]["directory"]) == output
    verified_source = read_rate_trial_source(proposed)
    assert verified_source["output"] == str(output)
    assert assess_rate_trial(source, report["volume"], proposed, verified_source) == ("verified", revised)
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "isolated-history"))
    assert CaptureWorkbench._read_rate_trial_check((source, str(formal)), proposed, {}) == ("verified", revised)
    assert assess_rate_trial(revised, report["volume"], proposed, None) == ("current", None)
    assert assess_rate_trial(source, "different-volume", proposed, verified_source) == ("volume", None)
    assert assess_rate_trial(source, report["volume"], proposed, None) == ("source_unavailable", None)
    changed_plan = source.to_dict()
    changed_plan["cameras"][0]["settings"]["fps"] = 17
    assert assess_rate_trial(CaptureConfig.from_dict(changed_plan), report["volume"], proposed, verified_source) == (
        "original_plan", None,
    )

    legacy = deepcopy(proposed)
    del legacy["trial"]["rate_source"]
    assert read_rate_trial_source(legacy, [proposed_output, output])["experiment_id"] == report["experiment_id"]

    relocated_source = copytree(output, tmp_path / "relocated" / output.name)
    relocated_candidate = copytree(proposed_output, relocated_source.parent / proposed_output.name)
    moved = trial_report(relocated_candidate, inspect_images=False)
    assert read_rate_trial_source(moved)["output"] == str(relocated_source)
    for directory, expected in ((relocated_source, "source_integrity"), (relocated_candidate, "file_integrity")):
        frame = next(directory.rglob("*.tiff"))
        data = frame.read_bytes()
        try:
            frame.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            fresh = trial_report(relocated_candidate, inspect_images=False)
            assert assess_rate_trial(source, report["volume"], fresh, read_rate_trial_source(fresh)) == (expected, None)
        finally:
            frame.write_bytes(data)

    assert not formal.exists()
    assert inspect_plan(revised, formal, trial_reports=[report, proposed])["rate_suggestions"] == []
    insufficient = deepcopy(proposed)
    insufficient["write_timing"]["acquisition_windows"].clear()
    assert check_rate_trial(suggestion, report, insufficient) == "write_measurement"
    damaged = deepcopy(proposed)
    damaged["observed"] = False
    assert check_rate_trial(suggestion, report, damaged) == "file_integrity"


def test_timed_rate_suggestion_only_changes_the_matched_phase():
    source = _plan([{"mode": "single", "role": "reference"},
                    {"mode": "timed", "fps": 18, "duration_s": 20, "frame_count": 150},
                    {"mode": "timed", "fps": 6, "frame_count": 80}], durability="write_through")
    report = {
        "sustained_write_lag": True, "observed": True, "experiment_id": "measured", "volume": "local-volume",
        "verification": {"file_hashes_checked": True},
        "write_timing": {"acquisition_windows": [{"committed_raw_bytes_per_s": 30000},
                                                  {"committed_raw_bytes_per_s": 29000}]},
        "cameras": {"sim:0": {"geometry": {"frame_bytes": 6144}, "committed_frames": 120,
                              "host_receive_fps": 18, "host_receive_span_s": 6.6}},
    }
    suggestion = suggest_trial_rate(source, 2, report)
    assert suggestion["request_rate"] == {"current": 18, "proposed": 3.77}
    revised = with_trial_rate(source, suggestion)
    expected = source.to_dict()
    expected["phases"][1]["fps"] = 3.77
    assert revised.to_dict() == expected
    report["cameras"]["sim:0"]["host_receive_fps"] = 12
    assert suggest_trial_rate(source, 2, report) is None


def test_slow_suggested_rate_allows_one_round_in_flight_at_stop():
    cameras = {key: {"geometry": {"frame_bytes": frame_bytes}, "formats": [key],
                     "committed_frames": 8, "host_receive_fps": .77, "host_receive_span_s": 9.1}
               for key, frame_bytes in (("pylon:test", 76800), ("gentl:test", 307200))}
    suggestion = {"source_volume": "test-volume", "request_rate": {"proposed": .77}}
    reference = {"cameras": deepcopy(cameras)}
    report = {
        "observed": True, "sustained_write_lag": False, "volume": "test-volume", "cameras": cameras, "findings": [],
        "verification": {"file_hashes_checked": True},
        "write_timing": {"acquisition_windows": [
            {"duration_s": 2.5, "accepted_raw_bytes_per_s": 307200, "pending_raw_bytes_growth": 0},
            {"duration_s": 2.5, "accepted_raw_bytes_per_s": 460800, "pending_raw_bytes_growth": 384000},
        ]},
    }
    assert check_rate_trial(suggestion, reference, report) == "verified"
    report["write_timing"]["acquisition_windows"][0]["pending_raw_bytes_growth"] = 76800
    assert check_rate_trial(suggestion, reference, report) == "write_lag"
    cameras["gentl:test"]["host_receive_span_s"] = 2.6
    assert check_rate_trial(suggestion, reference, report) == "cadence"


def test_stop_during_writer_opening_never_opens_a_camera(tmp_path, monkeypatch):
    from straintrace_capture import session as capture

    factory = capture.ExperimentWriter
    entered, cancel = threading.Event(), threading.Event()
    opened = []

    def delayed_writer(*args):
        entered.set()
        assert cancel.wait(5)
        return factory(*args)

    class Registry:
        def camera(self, identity):
            opened.append(identity)
            raise AssertionError("startup cancellation allowed a camera to open")

    monkeypatch.setattr(capture, "ExperimentWriter", delayed_writer)
    session = CaptureSession(_plan([{"mode": "single"}]), tmp_path / "cancelled", registry=Registry())
    startup = threading.Thread(target=session.start, kwargs={"stop_event": cancel})
    startup.start()
    try:
        assert entered.wait(5)
    finally:
        cancel.set()
        startup.join(5)
    assert not startup.is_alive()
    assert session.wait(5)["state"] == "stopped" and not opened
    with ExperimentReader(session.output) as reader:
        assert reader.verify()["committed_frames"] == 0
