import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from straintrace_capture import CaptureSession, ExperimentReader
from straintrace_capture.config import CaptureError
from straintrace_capture.live import LiveAnalysis
from straintrace_capture.live_results import LIVE_RESULTS_DIR, LiveResultLog, export_live_csv
from straintrace_capture.storage import DATABASE, ExperimentWriter, sha256_file
from test_capture_lifecycle import config, wait_until


def test_live_waits_until_capture_publishes_the_initial_ledger(tmp_path, monkeypatch):
    preparing, release = Event(), Event()
    archive_calibrations = ExperimentWriter._save_calibrations

    def delayed_archive(writer, calibrations):
        preparing.set()
        assert release.wait(6)
        return archive_calibrations(writer, calibrations)

    monkeypatch.setattr(ExperimentWriter, "_save_calibrations", delayed_archive)
    capture = CaptureSession(config([{"mode": "single", "role": "reference"}]), tmp_path / "experiment")
    with LiveAnalysis(capture.output, field_roi=(24, 24, 80, 72), field_step=12, field_subset=19) as follower:
        with ThreadPoolExecutor(max_workers=1) as worker:
            started = worker.submit(capture.start)
            try:
                assert preparing.wait(6)
                assert follower.poll() is None
                assert follower.progress()["pending_committed_frames"] is None
                assert not (capture.output / DATABASE).exists()
            finally:
                release.set()
                identity = started.result(timeout=8)
                assert capture.wait(8)["state"] == "completed"
        with ExperimentReader(capture.output) as reader:
            assert reader.experiment["experiment_id"] == identity
            reference, = reader.frames()
        assert follower.poll()["frame_id"] == reference["frame_id"]


def test_capture_continues_while_live_analysis_stalls_exits_and_restarts(tmp_path):
    session = CaptureSession(config([
        {"mode": "single", "role": "reference"},
        {"mode": "continuous", "duration_s": 30},
    ], **{"sim:0": {"fps": 20.0, "motion_px": 0.015, "strain_per_frame": 0.00012, "pixel_format": "Mono12"}}),
        tmp_path / "capture")
    marker = tmp_path / "first-analysis-frame.json"
    ready = marker.with_suffix(".ready")
    live_config = {
        "gauges": [{"name": "G01", "roi1": [41, 20, 27, 27], "roi2": [41, 73, 27, 27], "strain_mode": "y"}],
    }
    child = """
import sys
import time
from pathlib import Path
from straintrace_capture.live import LiveAnalysis
from straintrace_capture.storage import json_text
with LiveAnalysis(sys.argv[1], **__import__('json').loads(sys.argv[3])) as analyst:
    marker = Path(sys.argv[2])
    marker.with_suffix('.ready').touch()
    while (row := analyst.poll()) is None:
        time.sleep(.05)
    pending = marker.with_suffix('.tmp')
    pending.write_text(json_text(row), encoding='utf-8')
    pending.rename(marker)
    time.sleep(60)
"""
    process = subprocess.Popen([sys.executable, "-c", child, str(session.output), str(marker), json.dumps(live_config)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started = False
    try:
        wait_until(ready.is_file, timeout=35)
        session.start()
        started = True
        wait_until(marker.is_file, timeout=20)
        first = json.loads(marker.read_text(encoding="utf-8"))
        assert first["provisional"] and first["gauges"][0]["strain_valid"]
        assert process.poll() is None
        before = session.status()["cameras"]["sim:0"]["saved"]
        wait_until(lambda: session.status()["cameras"]["sim:0"]["saved"] > before + 2)
        assert process.poll() is None
        process.kill()
        process.wait(timeout=8)
        after_exit = session.status()["cameras"]["sim:0"]["saved"]
        wait_until(lambda: session.status()["cameras"]["sim:0"]["saved"] > after_exit + 2)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=8)
        if started:
            session.request_stop()
            final = session.wait(12)
    assert final["state"] == "stopped"
    cfg = tmp_path / "follow.json"
    cfg.write_text(json.dumps(live_config), encoding="utf-8")
    restarted = subprocess.run([sys.executable, "-m", "straintrace_capture", "follow", str(session.output),
                                "--config", str(cfg)], text=True, capture_output=True, timeout=30)
    assert restarted.returncode == 0, restarted.stderr + restarted.stdout
    reference, latest = [json.loads(line) for line in restarted.stdout.splitlines()]
    assert reference["reference_frame_id"] == first["reference_frame_id"]
    with ExperimentReader(session.output) as reader:
        frames = reader.frames()
        assert latest["skipped_committed_frames"] == len(frames) - 2
        assert latest["frame_id"] == frames[-1]["frame_id"]
        integrity = reader.verify()
        assert integrity["ok"] and not integrity["reported_camera_losses"]


def test_live_progress_and_saved_results_across_calculation_modes(tmp_path):
    run = CaptureSession(config([{"mode": "single", "role": "reference"},
                                 {"mode": "manual", "frame_count": 8}]), tmp_path / "capture")
    gauges = [{"name": "G01", "roi1": [41, 20, 27, 27], "roi2": [41, 73, 27, 27], "strain_mode": "y"}]
    log = LiveResultLog(run.output)
    run.start()
    try:
        wait_until(lambda: run.status()["mode"] == "manual")
        with LiveAnalysis(run.output, gauges=gauges) as live:
            reference = wait_until(live.poll)
            log.append(reference, live, "latest")
            assert reference["analyzed_test_frames"] == 0
            assert live.progress()["pending_committed_frames"] == 0

            def save_test(count):
                run.request_trigger(expected_phase=2)
                wait_until(lambda: run.status()["cameras"]["sim:0"]["saved"] == count + 1)

            for index in range(1, 4):
                save_test(index)
            assert live.progress()["pending_committed_frames"] == 3
            newest = live.poll(latest=True)
            log.append(newest, live, "latest")
            assert (newest["sequence"], newest["analyzed_test_frames"], newest["skipped_committed_frames"]) == (4, 1, 2)
            assert live.progress()["pending_committed_frames"] == 0
            early = tmp_path / "early.csv"
            assert export_live_csv(log.path, early) == {"frames": 2, "test_frames": 1, "skipped": 2, "incomplete_tail": False}
            for index in range(4, 9):
                save_test(index)
            assert run.wait(8)["state"] == "completed"
            ledger_digest = sha256_file(run.output / DATABASE)
            assert live.progress()["pending_committed_frames"] == 5
            assert live.progress()["pending_committed_frames"] == 5
            ordered = []
            for pending in (4, 3, 2, 1, 0):
                row = live.poll(latest=False)
                log.append(row, live, "ordered")
                ordered.append(row["sequence"])
                assert live.progress()["pending_committed_frames"] == pending
            assert ordered == [5, 6, 7, 8, 9]
            assert live.poll(latest=False) is None
            assert live.progress() == {"experiment_state": "completed", "reference_ready": True, "analyzed_test_frames": 6,
                                       "skipped_committed_frames": 2, "pending_committed_frames": 0}
    finally:
        if not run.finished:
            run.request_stop()
            run.wait(8)
    with ExperimentReader(run.output) as reader:
        frames = reader.frames()
        assert reader.verify()["complete"] and len(frames) == 9
    assert log.error is None and log.test_frames == 6
    rows = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert [row["sequence"] for row in rows] == [1, 4, 5, 6, 7, 8, 9]
    assert [row["follow_mode"] for row in rows] == ["latest"] * 2 + ["ordered"] * 5
    assert rows[0]["settings"]["gauges"][0]["roi1"] == gauges[0]["roi1"]
    assert all(row["image_sha256"] == frames[row["sequence"] - 1]["sha256"] for row in rows)
    assert all(row["reference_sha256"] == frames[0]["sha256"] for row in rows)
    assert sha256_file(run.output / DATABASE) == ledger_digest
    assert len(early.read_text(encoding="utf-8-sig").splitlines()) == 3

    # A partial final write keeps every previous line available for export.
    output = tmp_path / "result.csv"
    with log.path.open("ab") as stream:
        stream.write(b'{"record_schema":')
    assert export_live_csv(log.path, output) == {"frames": 7, "test_frames": 6, "skipped": 2, "incomplete_tail": True}
    with output.open(encoding="utf-8-sig", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert [row["原图标识"] for row in exported] == [row["frame_id"] for row in rows]
    assert float(exported[1]["工程应变 (%)"]) == pytest.approx(rows[1]["gauges"][0]["engineering_strain"] * 100)
    digest = sha256_file(output)
    with log.path.open("ab") as stream:
        stream.write(b"invalid}\n")
    with pytest.raises(CaptureError, match="第 8 行"):
        export_live_csv(log.path, output)
    assert sha256_file(output) == digest


def test_result_directory_error_preserves_live_measurement_and_raw_data(tmp_path):
    run = CaptureSession(config([{"mode": "single", "role": "reference"}]), tmp_path / "capture")
    run.start()
    assert run.wait(8)["state"] == "completed"
    (run.output / LIVE_RESULTS_DIR).write_text("existing file", encoding="utf-8")
    ledger_digest = sha256_file(run.output / DATABASE)
    log = LiveResultLog(run.output)
    with LiveAnalysis(run.output, field_roi=(24, 24, 80, 72), field_step=12, field_subset=19) as live:
        reference = live.poll()
        log.append(reference, live, "latest")
        assert reference["field_quality"]["scientific_ok"]
        assert live.poll() is None
    assert log.path is None and isinstance(log.error, OSError)
    assert sha256_file(run.output / DATABASE) == ledger_digest
