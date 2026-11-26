import threading
import time

import pytest

from straintrace_capture import CaptureConfig, CaptureSession, ExperimentReader, inspect_plan
from straintrace_capture.presentation import format_integrity, format_readiness, format_trial
from straintrace_capture.storage import ExperimentWriter
from straintrace_capture.trial import prepare_trial, trial_report


def _plan(**options):
    return CaptureConfig.from_dict({
        "config_version": 1,
        "name": "Write queue",
        "durability": "write_through",
        "cameras": [
            {"id": f"sim:{index}", "settings": {"width": 64, "height": 48, "pixel_format": "Mono12", "fps": 42.0}}
            for index in range(2)
        ],
        "phases": [{"mode": "continuous", "frame_count": 24}],
        **options,
    })


def test_preflight_reports_the_first_stall_limit_and_preserves_fixed_plans(tmp_path):
    fixed = CaptureConfig.from_dict(_plan(queue_capacity=32).to_dict())
    default = CaptureConfig.from_dict(_plan().to_dict())
    assert fixed.queue_capacity == 32
    assert default.queue_capacity == "auto" and default.effective_queue_capacity == 4096
    for plan in (fixed, default):
        report = inspect_plan(plan, tmp_path / "unused")
        assert report["write_queue"]["capacity"] == plan.effective_queue_capacity
        stall = report["phases"][0]["write_stall_buffer"]
        assert stall["queue_seconds"] == pytest.approx(plan.effective_queue_capacity / 84)
        assert stall["byte_seconds"] == pytest.approx(plan.max_buffer_bytes / (84 * 64 * 48 * 2))
    assert inspect_plan(fixed, tmp_path / "fixed")["write_queue"]["suggest_auto"]
    assert "写入暂停" in format_readiness(inspect_plan(fixed, tmp_path / "fixed"))
    large = _plan(cameras=[
        {"id": "sim:0", "settings": {"width": 640, "height": 480, "pixel_format": "Mono12", "fps": 12.0}},
    ])
    assert inspect_plan(large, tmp_path / "large")["phases"][0]["write_stall_buffer"]["first_limit"] == "bytes"
    short = _plan(queue_capacity=32, phases=[{"mode": "continuous", "frame_count": 2}])
    assert not inspect_plan(short, tmp_path / "short")["write_queue"]["suggest_auto"]


def test_fixed_queue_reports_pressure_while_automatic_queue_saves_the_burst(tmp_path, monkeypatch):
    original = ExperimentWriter.prepare_frame
    started, release = threading.Event(), threading.Event()

    def hold_first(writer, *args, **kwargs):
        if not started.is_set():
            started.set()
            assert release.wait(8), "test writer was not released"
        return original(writer, *args, **kwargs)

    monkeypatch.setattr(ExperimentWriter, "prepare_frame", hold_first)
    for options in ({"queue_capacity": 32}, {}):
        started.clear()
        release.clear()
        planned = _plan(**options)
        trial, output = prepare_trial(planned, tmp_path / "formal", seconds=3)
        session = CaptureSession(trial, output)
        session.start()
        try:
            assert started.wait(4)
            limit = time.monotonic() + 4
            while time.monotonic() < limit:
                status = session.status()
                if status["error"] or status["queue_size"] >= 36:
                    break
                time.sleep(.008)
            else:
                pytest.fail(f"camera producers failed to reach the stalled queue: {session.status()}")
        finally:
            release.set()
            status = session.wait(8)

        report = trial_report(output)
        with ExperimentReader(output) as reader:
            integrity = reader.verify()
        assert integrity["ok"] and integrity["file_hashes_checked"]
        if options:
            assert status["error"]["code"] == "WRITE_QUEUE_FULL"
            assert status["queue_high_water"] == 32
            assert status["raw_buffer_high_water_bytes"] < 2**20
            assert not integrity["complete"]
            findings = {row["code"] for row in report["findings"]}
            assert {"TRIAL_HIGH_WRITE_QUEUE", "TRIAL_UNSAVED_FRAMES"} <= findings
            assert "TRIAL_CAMERA_LOSSES" not in findings
            assert "待写队列峰值 32 / 32 批" in format_trial(report)
            assert "待写队列达到批次上限" in format_integrity(integrity)
        else:
            assert status["state"] == "completed"
            assert status["queue_high_water"] >= 36
            assert integrity["complete"] and integrity["committed_frames"] == 48
            assert status["raw_buffer_limit_bytes"] == planned.max_buffer_bytes
