import csv
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from straintrace_capture import CaptureSession, ExperimentReader
from straintrace_capture.config import CaptureError
from straintrace_capture.live import LiveAnalysis, read_linked_reference_image
from straintrace_capture.live_results import LIVE_RESULTS_DIR, LiveResultLog, export_live_csv
from straintrace_capture.storage import DATABASE, ExperimentWriter, sha256_file
from test_capture_lifecycle import config, wait_until


def test_linked_reference_uses_the_same_tiff_bytes_for_identity_and_pixels(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import ezdic_core

    path = tmp_path / "reference.tiff"
    source = np.full((12, 16), 1200, dtype=np.uint16)
    other = cv2.imencode(".tiff", np.zeros_like(source))[1].tobytes()
    path.write_bytes(cv2.imencode(".tiff", source)[1].tobytes())
    reference = {"absolute_path": str(path), "frame_id": "sim:0/1", "color_model": "mono",
                 "sha256": sha256_file(path)}
    decode = ezdic_core.read_gray_image

    def replace_before_decode(path_to_read, **kwargs):
        path.write_bytes(other)
        return decode(path_to_read, **kwargs)

    monkeypatch.setattr(ezdic_core, "read_gray_image", replace_before_decode)
    np.testing.assert_array_equal(read_linked_reference_image(reference, debayer=False), source)
    with pytest.raises(CaptureError) as changed:
        read_linked_reference_image(reference, debayer=False)
    assert changed.value.code == "CAPTURE_IMAGE_CHANGED"


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


def _stall_live_reader(root, config, events):
    with events, LiveAnalysis(root, **config) as analyst:
        events.send("ready")
        while (row := analyst.poll()) is None:
            time.sleep(.05)
        events.send(row)
        time.sleep(60)


def test_capture_continues_while_live_analysis_stalls_exits_and_restarts(tmp_path):
    session = CaptureSession(config([
        {"mode": "single", "role": "reference"},
        {"mode": "continuous", "duration_s": 30},
    ], **{"sim:0": {"fps": 20.0, "motion_px": 0.015, "strain_per_frame": 0.00012, "pixel_format": "Mono12"}}),
        tmp_path / "capture")
    live_config = {
        "gauges": [{"name": "G01", "roi1": [41, 20, 27, 27], "roi2": [41, 73, 27, 27], "strain_mode": "y"}],
    }
    ctx = multiprocessing.get_context("spawn")
    events, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_stall_live_reader, args=(session.output, live_config, sender))
    process.start()
    sender.close()
    started = False
    try:
        assert events.poll(35) and events.recv() == "ready"
        session.start()
        started = True
        assert events.poll(20)
        first = events.recv()
        assert first["provisional"] and first["gauges"][0]["strain_valid"]
        assert process.is_alive()
        before = session.status()["cameras"]["sim:0"]["saved"]
        wait_until(lambda: session.status()["cameras"]["sim:0"]["saved"] > before + 2)
        assert process.is_alive()
        process.kill()
        process.join(timeout=8)
        assert process.exitcode is not None
        after_exit = session.status()["cameras"]["sim:0"]["saved"]
        wait_until(lambda: session.status()["cameras"]["sim:0"]["saved"] > after_exit + 2)
    finally:
        events.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=8)
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


@pytest.mark.parametrize(("field", "changed"), [("offset_x", 12), ("binning_x", 2)])
def test_live_rejects_changed_same_run_sensor_geometry(tmp_path, field, changed):
    from straintrace_capture.cameras.base import Device, Image
    from straintrace_capture.live_history import read_history_index
    from straintrace_capture.live_replay import replay_live_results
    from test_capture_continuation_analysis import _images

    pixels, sample, _ = _images()
    root = tmp_path / "run"
    plan = config([{"mode": "single", "role": "reference"}, {"mode": "single"}, {"mode": "single"}])
    features = {name: {"value": value} for name, value in
                (("width", 176), ("height", 144), ("offset_x", 0), ("binning_x", 1))}
    writer = ExperimentWriter(root, plan)
    try:
        writer.register_camera(Device("sim:0", "sim", "Fixture", "Camera", "0", "simulated"), features)
        writer.set_state("running")
        writer.event("phase_started", {"phase": 1, "camera_features": {"sim:0": features}})
        image = Image(pixels, "Mono8", 8, "mono", 1_800_000_000_000_000_000, 1_000_000_000)
        writer.commit_frames([writer.prepare_frame("sim:0", 1, 1, "reference", image, None, None, None, 0)])
        gauges = [{"name": "G01", "roi1": (45, 25, 25, 25), "roi2": (45, 83, 25, 25), "strain_mode": "y"}]
        log = LiveResultLog(root)
        with LiveAnalysis(root, gauges=gauges, latest=False) as live:
            reference = live.poll()
            log.append(reference, live, "ordered")
            assert reference["sequence"] == 1
            writer.event("phase_started", {"phase": 2, "camera_features": {"sim:0": features}})
            image = Image(sample, "Mono8", 8, "mono", 1_800_000_000_100_000_000, 1_100_000_000)
            writer.commit_frames([writer.prepare_frame("sim:0", 2, 2, "test", image, None, None, None, 0)])
            result = live.poll()
            log.append(result, live, "ordered")
            assert result["sequence"] == 2 and log.error is None
            updated = {**features, field: {"value": changed}}
            writer.event("phase_started", {"phase": 3, "camera_features": {"sim:0": updated}})
            writer.commit_frames([writer.prepare_frame("sim:0", 3, 3, "test", image, None, None, None, 0)])
            with pytest.raises(CaptureError) as problem:
                live.poll()
            assert problem.value.code == "FRAME_GEOMETRY_CHANGED"
            assert live.analyzed_test_frames == 1
        index = read_history_index(root, log.path)
        with pytest.raises(CaptureError) as problem:
            replay_live_results(index, images_only=True)
        assert problem.value.code == "FRAME_GEOMETRY_CHANGED"
        assert set(log.path.parent.iterdir()) == {log.path}
    finally:
        writer.close()


@pytest.mark.parametrize("case", ["earlier", "different-clock", "different-reference", "different-unit", "missing-value"])
def test_same_run_live_checks_receipt_clocks_and_preserves_signed_time(tmp_path, case):
    from test_capture_mechanics_setup import capture_with_known_receipts

    root = capture_with_known_receipts(tmp_path / "run")
    gauges = [{"name": "G01", "roi1": (45, 25, 25, 25), "roi2": (45, 83, 25, 25), "strain_mode": "y"}]
    with LiveAnalysis(root, gauges=gauges, latest=False) as live:
        assert live.poll()["elapsed_receive_s"] == 0
        row, _ = live.reader.next_test_frame("sim:0", after=live._cursor, latest=False)
        stamps = dict(row["timestamps"])
        receipt = dict(stamps["received_monotonic"])
        if case == "earlier":
            receipt["value"] = live.reference["timestamps"]["received_monotonic"]["value"] - 1
        elif case == "different-clock":
            receipt["clock_id"] = "different-host-clock"
        elif case == "different-reference":
            receipt["reference"] = "another-origin"
        elif case == "different-unit":
            receipt["unit"] = "ticks"
        else:
            receipt["value"] = None
        stamps["received_monotonic"] = receipt
        result = live.analyze_committed_test({**row, "timestamps": stamps})
        assert result["elapsed_receive_s"] == (-1e-9 if case == "earlier" else None)
