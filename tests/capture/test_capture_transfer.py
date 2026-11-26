import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, ExperimentReader, copy_experiment, inspect_transfer, verify_transfer
from straintrace_capture.calibration import CALIBRATION_SCHEMA, archived_profile_path, read_profile
from straintrace_capture.cameras.base import Device, Image, receipt_times
from straintrace_capture.cli import main
from straintrace_capture.continuation import continuation_plan
from straintrace_capture.handoff import analysis_snapshot
from straintrace_capture.storage import DATABASE, FRAME_WRITE_JOURNAL, ExperimentWriter, WriterLease, frame_image_records, sha256_file
from straintrace_capture.transfer import TRANSFER_MANIFEST


@pytest.fixture
def experiment(tmp_path):
    camera = "gentl:fixture"
    calibration = tmp_path / "原始标定.json"
    calibration.write_text(json.dumps({"schema": CALIBRATION_SCHEMA, "capture_experiment_id": "checkerboard-run", "cameras": {
        camera: {"camera_matrix": [[45, 0, 12], [0, 45, 8], [0, 0, 1]], "distortion_coefficients": [0] * 5,
                 "image_size_px": [24, 16]},
    }}), encoding="utf-8")
    root = tmp_path / "原始实验"
    plan = CaptureConfig.from_dict({"config_version": 1, "name": "交接测试", "cameras": [{"id": camera}],
                                    "phases": [{"mode": "single", "role": "reference"}, {"mode": "single"}],
                                    "calibration_files": [str(calibration)], "segment_frames": 2})
    writer = ExperimentWriter(root, plan)
    try:
        writer.register_camera(Device(camera, "gentl", "Transfer", "Camera", "1", "fixture"), {})
        times = receipt_times()
        raw = Image(np.arange(384, dtype=np.uint16).reshape(16, 24), "Mono12", 12, "mono", *times)
        auxiliary = Image(np.full((8, 12), 24, dtype=np.uint8), "Mono8", 8, "mono", *times)
        for sequence, role in ((1, "reference"), (2, "test")):
            frame = writer.prepare_frame(camera, sequence, sequence, role, replace(raw, auxiliary_images=(auxiliary,)),
                                         None, None, None, 0)
            writer.commit_frames([frame])
        writer.event("operator_note", {"note": "标记随台账一同交接"}, event_id="operator-1")
        writer.prepare_frame(camera, 3, 2, "test", raw, None, None, None, 0)
        writer.set_state("interrupted")
    finally:
        writer.close()
    (root / "attachments" / "notes").mkdir(parents=True)
    (root / "attachments" / "notes" / "说明.txt").write_bytes(b"existing attachment\n")
    (root / "analysis-results").mkdir()
    (root / "analysis-results" / "working.csv").write_bytes(b"optional derived result\n")
    return root, camera, calibration


def file_hashes(root):
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*") if path.is_file()}


def test_copy_keeps_raw_parts_snapshot_calibration_and_interruption_evidence(experiment, tmp_path):
    source, camera, original_profile = experiment
    daq = tmp_path / "脉冲.csv"
    daq.write_bytes(b"event_number,time_ns\n1,20\n")
    target = tmp_path / "交接副本"
    before = file_hashes(source)
    with ExperimentReader(source, immutable=True) as reader:
        original = reader.experiment
        frames = reader.frames()
        events = reader.events()

    scope = inspect_transfer(source, target, attachments=[daq])
    assert (scope["frames"], scope["registered_images"], scope["calibration_files"], scope["recovery_files"]) == (2, 4, 1, 1)
    assert scope["attachment_files"] == 2 and scope["available_bytes"] > 0
    assert not target.exists() and not Path(scope["staging"]).exists()
    copied = copy_experiment(source, target, attachments=[daq], expected_id=original["experiment_id"])
    assert copied["files"] == 10 and copied["reused_files"] == 0 and copied["ok"]
    assert not Path(scope["staging"]).exists()
    assert file_hashes(source) == before
    manifest = json.loads((target / TRANSFER_MANIFEST).read_text(encoding="utf-8"))
    assert {item["kind"] for item in manifest["entries"]} == {"ledger", "image", "calibration", "recovery", "journal", "attachment"}
    for item in manifest["entries"]:
        if item["kind"] not in {"ledger", "attachment"}:
            assert item["sha256"] == before[item["path"]]
    assert (target / FRAME_WRITE_JOURNAL).read_bytes() == (source / FRAME_WRITE_JOURNAL).read_bytes()
    assert (target / "attachments" / daq.name).read_bytes() == daq.read_bytes()
    assert not (target / "analysis-results").exists() and not (target / ".capture-writer").exists()
    verified = verify_transfer(target)
    assert verified["ok"] and verified["state"] == "interrupted"

    with ExperimentReader(target) as reader:
        assert reader.experiment == original and reader.frames() == frames and reader.events() == events
        assert reader.frame_at_sequence(camera, 1)["auxiliary_images"] == frames[0]["auxiliary_images"]
        saved = reader.experiment["calibrations"][0]
        archived = archived_profile_path(target, saved)
        assert read_profile(archived, expected_hash=saved["sha256"])["capture_experiment_id"] == "checkerboard-run"
        assert any(item["code"] == "UNCOMMITTED_FILE" for item in reader.verify()["issues"])
    original_profile.unlink()
    plan = continuation_plan(target)
    assert list(plan.calibration_files) == [str(archived)]
    assert analysis_snapshot(target, camera)["camera_id"] == camera


def test_copy_reads_committed_events_that_are_only_in_wal(experiment, tmp_path):
    source, _, _ = experiment
    wal = source / (DATABASE + "-wal")
    target = tmp_path / "含WAL的副本"
    with closing(sqlite3.connect(source / DATABASE)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        body = {"kind": "operator_note", "event_id": "wal-event", "utc_ns": 2, "note": "仅在 WAL 中"}
        connection.execute("INSERT INTO events (event_id, kind, record) VALUES (?, ?, ?)",
                           (body["event_id"], body["kind"], json.dumps(body)))
        connection.commit()
        assert wal.stat().st_size > 0
        source_hash = sha256_file(source / DATABASE)
        wal_hash = sha256_file(wal)
        with ExperimentReader(source, immutable=True) as reader:
            assert not any(event["event_id"] == "wal-event" for event in reader.events())
        copy_experiment(source, target)
        assert sha256_file(source / DATABASE) == source_hash and sha256_file(wal) == wal_hash
    assert not list(target.glob(DATABASE + "-*"))
    with ExperimentReader(target) as reader:
        assert any(event["event_id"] == "wal-event" and event["note"] == body["note"] for event in reader.events())
    assert verify_transfer(target)["ok"]


def test_cancel_resume_rehashes_both_ends_and_repairs_incomplete_temporary_data(experiment, tmp_path):
    source, _, _ = experiment
    target = tmp_path / "断点续传"
    selected = tmp_path / "记录.csv"
    selected.write_bytes(b"1,20\n")
    stop = False

    def progress(update):
        nonlocal stop
        if update.files_done >= 3:
            stop = True

    with pytest.raises(CaptureError) as failed:
        copy_experiment(source, target, attachments=[selected], progress=progress, cancelled=lambda: stop)
    assert failed.value.code == "TRANSFER_CANCELLED" and not target.exists()
    scope = inspect_transfer(source, target, attachments=[selected])
    assert scope["can_resume"]
    stage = Path(scope["staging"])
    staged = sorted((stage / "experiment" / "images").rglob("*.tiff"))
    assert len(staged) == 2
    staged[0].write_bytes(b"x" * staged[0].stat().st_size)
    (stage / "scratch" / "file.part").write_bytes(b"crash before file complete")
    with pytest.raises(CaptureError) as wrong:
        copy_experiment(source, target)
    assert wrong.value.code == "TRANSFER_SCOPE"
    done = copy_experiment(source, target, attachments=[selected])
    assert done["reused_files"] == 1 and verify_transfer(target)["ok"] and not stage.exists()


def test_no_replace_when_destination_appears_during_publication(experiment, tmp_path):
    source, _, _ = experiment
    target = tmp_path / "有人先创建"

    def progress(update):
        if update.stage == "publishing":
            target.mkdir()

    with pytest.raises((CaptureError, FileExistsError, PermissionError)):
        copy_experiment(source, target, progress=progress)
    assert list(target.iterdir()) == []
    with pytest.raises(CaptureError) as existing:
        inspect_transfer(source, target)
    assert existing.value.code == "OUTPUT_EXISTS"


def test_copy_refuses_active_or_moved_source_and_changed_raw_bytes(experiment, tmp_path):
    source, _, _ = experiment
    target = tmp_path / "new"
    with WriterLease(source):
        with pytest.raises(CaptureError) as exc:
            copy_experiment(source, target)
        assert exc.value.code == "WRITER_ACTIVE"
    with pytest.raises(CaptureError) as exc:
        copy_experiment(source, target, expected_id="a-different-experiment")
    assert exc.value.code == "TRANSFER_SCOPE"
    with pytest.raises(CaptureError) as inside:
        inspect_transfer(source, source / "new")
    assert inside.value.code == "TRANSFER_PATH"
    with ExperimentReader(source, immutable=True) as reader:
        image = source / frame_image_records(reader.frames(limit=1)[0])[1]["path"]
    original = image.read_bytes()
    image.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(CaptureError) as changed:
        copy_experiment(source, target)
    assert changed.value.code == "TRANSFER_CHANGED" and image.name in str(changed.value) and not target.exists()


def test_cli_reports_tampering_invalid_scope_and_missing_manifest(experiment, tmp_path, capsys):
    source, _, _ = experiment
    target = tmp_path / "副本"
    assert main(["copy-experiment", str(source), "--output", str(target), "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["registered_images"] == 4
    assert main(["copy-experiment", str(source), "--output", str(target), "--progress"]) == 0
    progress = capsys.readouterr()
    assert json.loads(progress.out)["ok"] and "publishing" in progress.err
    assert main(["verify-copy", str(target)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    path = target / TRANSFER_MANIFEST
    body = json.loads(path.read_text(encoding="utf-8"))
    body["entries"][1]["path"] = "images/../../outside.tiff"
    path.write_text(json.dumps(body), encoding="utf-8")
    assert main(["verify-copy", str(target)]) == 4
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "TRANSFER_PATH"
    path.unlink()
    assert main(["verify-copy", str(target)]) == 4
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "TRANSFER_MISSING"


def test_verifier_detects_attachment_tampering_and_snapshot_sidecars(experiment, tmp_path):
    source, _, _ = experiment
    target = tmp_path / "copy"
    copy_experiment(source, target)
    sentinel = target / (DATABASE + "-wal")
    sentinel.write_bytes(b"outside the sealed SQLite snapshot")
    with pytest.raises(CaptureError) as extra:
        verify_transfer(target)
    assert extra.value.code == "TRANSFER_CHANGED"
    sentinel.unlink()
    asset = target / "attachments" / "notes" / "说明.txt"
    asset.write_bytes(b"X" * asset.stat().st_size)
    with pytest.raises(CaptureError) as tampered:
        verify_transfer(target)
    assert tampered.value.code == "TRANSFER_CHANGED" and asset.name in str(tampered.value)


def test_resume_does_not_publish_a_staged_hardlink_to_source(experiment, tmp_path):
    source, _, _ = experiment
    target = tmp_path / "copy"
    stop = False

    def progress(update):
        nonlocal stop
        stop = update.files_done >= 2

    with pytest.raises(CaptureError):
        copy_experiment(source, target, cancelled=lambda: stop, progress=progress)
    stage = Path(inspect_transfer(source, target)["staging"])
    staged, = (stage / "experiment" / "images").rglob("*.tiff")
    corresponding = source / staged.relative_to(stage / "experiment")
    staged.unlink()
    try:
        os.link(corresponding, staged)
    except OSError as exc:
        pytest.skip(f"hard links unavailable on this filesystem: {exc}")
    before = sha256_file(corresponding)
    with pytest.raises(CaptureError) as redirect:
        copy_experiment(source, target)
    assert redirect.value.code == "TRANSFER_PATH" and not target.exists() and sha256_file(corresponding) == before


def test_killed_copy_releases_destination_lease_and_can_resume(experiment, tmp_path):
    source, _, _ = experiment
    target, ready = tmp_path / "resume-after-exit", tmp_path / "copy-paused"
    program = """
import sys
import time
from pathlib import Path
from straintrace_capture import copy_experiment

def pause(update):
    if update.files_done >= 2:
        Path(sys.argv[3]).touch()
        time.sleep(30)

copy_experiment(sys.argv[1], sys.argv[2], progress=pause)
"""
    process = subprocess.Popen([sys.executable, "-B", "-c", program, str(source), str(target), str(ready)],
                               stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.025)
        assert process.poll() is None and ready.exists() and not target.exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stderr.close()
    assert inspect_transfer(source, target)["can_resume"]
    assert copy_experiment(source, target)["reused_files"] == 1
    assert verify_transfer(target)["ok"]


def test_manifest_checks_folder_case_and_source_that_has_not_finalized(experiment, tmp_path):
    source, _, _ = experiment
    target = tmp_path / "checked-copy"
    copy_experiment(source, target)
    path = target / TRANSFER_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    attached = next(item for item in manifest["entries"] if item["kind"] == "attachment")
    manifest["entries"].append({**attached, "path": "attachments/NOTES/extra.log"})
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CaptureError) as ambiguous:
        verify_transfer(target)
    assert ambiguous.value.code == "TRANSFER_PATH"

    pending = tmp_path / "unfinalized-capture"
    writer = ExperimentWriter(pending, CaptureConfig.from_dict({"config_version": 1, "name": "pending",
                                                                 "cameras": [{"id": "sim:0"}], "phases": [{"mode": "single"}]}))
    writer.close()
    with pytest.raises(CaptureError) as active:
        copy_experiment(pending, tmp_path / "new-target")
    assert active.value.code == "TRANSFER_ACTIVE"


def test_copy_rechecks_a_source_without_a_writer_marker_before_publication(experiment, tmp_path):
    source, _, _ = experiment
    first = tmp_path / "received"
    copy_experiment(source, first)
    assert not (first / ".capture-writer").exists()
    second = tmp_path / "duplicate"

    def recovery_starts(update):
        if update.stage == "publishing":
            with WriterLease(first):
                pass

    with pytest.raises(CaptureError) as changed:
        copy_experiment(first, second, progress=recovery_starts)
    assert changed.value.code == "TRANSFER_SCOPE" and not second.exists()
    assert copy_experiment(first, second)["reused_files"] > 0 and verify_transfer(second)["ok"]


def test_copy_reports_insufficient_space_before_writing_image_data(experiment, tmp_path, monkeypatch):
    source, _, _ = experiment
    target = tmp_path / "small-destination"
    monkeypatch.setattr("straintrace_capture.transfer.shutil.disk_usage", lambda _path: SimpleNamespace(free=10))
    scope = inspect_transfer(source, target)
    assert scope["available_bytes"] == 10 and scope["bytes"] > 10
    with pytest.raises(CaptureError) as exc:
        copy_experiment(source, target)
    assert exc.value.code == "TRANSFER_SPACE" and not target.exists()
    assert not (Path(scope["staging"]) / "experiment" / "images").exists()
