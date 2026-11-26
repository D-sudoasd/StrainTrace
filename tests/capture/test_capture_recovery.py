import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, CaptureSession, ExperimentReader, recover_experiment
from straintrace_capture import storage
from straintrace_capture.cameras.base import Device, Image, receipt_times
from straintrace_capture.cli import main as capture_cli
from straintrace_capture.continuation import continuation_plan
from straintrace_capture.handoff import analysis_snapshot
from straintrace_capture.recovery import inspect_recovery
from straintrace_capture.storage import DATABASE, FRAME_WRITE_JOURNAL, ExperimentWriter, TriggerEvent, frame_image_records


def capture_plan(*, durability="buffered"):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "Recovery fixture",
        "cameras": [{"id": "sim:0", "settings": {"width": 96, "height": 80, "pixel_format": "Mono12"}}],
        "phases": [{"mode": "single", "role": "reference"},
                   {"mode": "timed", "fps": 5, "frame_count": 4}],
        "segment_frames": 2,
        "durability": durability,
    })


@pytest.mark.parametrize(("cut", "durability"), [
    ("published", "buffered"),
    ("staged", "write_through"),
    ("temporary", "write_through"),
    ("published", "write_through"),
])
def test_force_exit_recovers_native_frame_identity_and_trigger(tmp_path, cut, durability):
    root = tmp_path / "experiment"
    program = """
import json
import os
import sys
import time
from pathlib import Path
from straintrace_capture import CaptureConfig, CaptureSession
from straintrace_capture import storage

root, pause, cut = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
config = CaptureConfig.from_dict(json.loads(sys.argv[4]))
publish = storage._publish_path
commit = storage.ExperimentWriter.commit_frames

def hold(path):
    pause.write_text(json.dumps({"path": str(path), "pid": os.getpid()}), encoding="utf-8")
    time.sleep(30)
    raise TimeoutError("test never stopped the writer")

def pause_for_kill(partial, target, *, write_through):
    if cut == "staged":
        return publish(partial, target, write_through=write_through)
    if target.name != "000000000003.tiff":
        return publish(partial, target, write_through=write_through)
    if cut == "published":
        publish(partial, target, write_through=write_through)
    hold(target if cut == "published" else partial)

def pause_commit(self, prepared):
    if cut == "staged":
        for item in prepared:
            if item.record["sequence"] == 3:
                hold((self.root / item.record["path"]).with_suffix(".tiff.part"))
    return commit(self, prepared)

storage._publish_path = pause_for_kill
storage.ExperimentWriter.commit_frames = pause_commit
run = CaptureSession(config, root)
run.start()
run.wait(40)
"""
    marker = tmp_path / "paused"
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", program, str(root), str(marker), cut,
         json.dumps(capture_plan(durability=durability).to_dict())],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    paused = None
    try:
        deadline = time.monotonic() + 12
        while paused is None:
            try:
                paused = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            if paused is not None:
                break
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode(errors="replace"))
            if time.monotonic() >= deadline:
                pytest.fail("writer did not reach the requested interruption point")
            time.sleep(.025)
        with pytest.raises(CaptureError) as error:
            recover_experiment(root)
        assert error.value.code == "WRITER_ACTIVE"
    finally:
        if process.poll() is None:
            # The venv launcher may be a different process from the writer.
            if paused is not None and paused["pid"] != process.pid:
                os.kill(paused["pid"], signal.SIGTERM)
            else:
                process.kill()
        process.wait(timeout=6)
        process.stderr.close()

    raw_path = Path(paused["path"])
    raw = raw_path.read_bytes()
    journal = (root / FRAME_WRITE_JOURNAL).read_bytes().splitlines()
    descriptors = [json.loads(line.split(b" ", 1)[1]) for line in journal]
    descriptor, = [item for item in descriptors if item["frame"]["sequence"] == 3]
    with ExperimentReader(root) as reader:
        committed = reader.frames()
        assert [row["sequence"] for row in committed] == [1, 2]
    result = recover_experiment(root)
    assert result["state"] == "interrupted" and result["ok"] and not result["complete"]
    assert not result["runtime_summary_available"]
    restored, = result["recovery"]["recovered"]
    assert restored["source_path"] == raw_path.relative_to(root).as_posix()
    assert (root / restored["path"]).read_bytes() == raw
    with ExperimentReader(root) as reader:
        previously_committed = reader.frames(limit=2)
        assert previously_committed == committed
        recovered, = reader.frames(after=committed[-1]["ordinal"])
        ordinal = recovered.pop("ordinal")
        assert ordinal > committed[-1]["ordinal"] and recovered == descriptor["frame"]
        associations = reader.pairing()["groups"]
        assert all(group["status"] == "associated" for group in associations)
        assert any(row["kind"] == "recovery" and row["recovered"] == [restored] for row in reader.events())
    snapshot = analysis_snapshot(root, "sim:0")
    assert [row["sequence"] for row in snapshot["frames"]] == [1, 2, 3]
    second = recover_experiment(root)
    assert second["recovery"]["recovered"] == [] and second["recovery"]["already_committed"] == 1
    assert second["committed_frames"] == 3


def _prepare_frames(writer, count):
    writer.register_camera(Device("sim:0", "sim", "StrainTrace", "Fixture", "SIM-0000", "simulated"), {})
    utc, monotonic = receipt_times()
    image = Image(np.zeros((80, 96), dtype=np.uint16), "Mono12", 12, "mono", utc, monotonic)
    return [writer.prepare_frame("sim:0", number, 2, "test", image, None, None, None, 0)
            for number in range(1, count + 1)]


def stage_frames(root, count, *, durability="buffered"):
    writer = ExperimentWriter(root, capture_plan(durability=durability))
    try:
        return [item.record for item in _prepare_frames(writer, count)]
    finally:
        writer.close()


def test_recovery_check_preserves_files_and_matches_an_explicit_restore(tmp_path, capsys):
    root = tmp_path / "experiment"
    first, bad, missing = stage_frames(root, 3, durability="write_through")
    def partial(row):
        return (root / row["path"]).with_suffix(".tiff.part")

    partial(bad).write_bytes(b"damaged fixture")
    partial(missing).unlink()
    evidence = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    assert capture_cli(["recover", str(root), "--check"]) == 0
    check = json.loads(capsys.readouterr().out)
    assert check["schema"] == "straintrace.capture-recovery-check/v1"
    assert check["state"] == "opening" and check["needs_finalize"]
    assert check["committed_frames"] == check["committed_images"] == 0
    assert check["cameras"] == {"sim:0": {"frames": 0, "images": 0}}
    scan = check["recovery"]
    assert [row["frame_id"] for row in scan["recoverable"]] == [first["frame_id"]]
    assert [row["code"] for row in scan["issues"]] == ["FRAME_PAYLOAD_MISMATCH", "FRAME_PAYLOAD_MISSING"]
    assert {path: path.read_bytes() for path in evidence} == evidence
    assert partial(first).is_file() and not (root / first["path"]).exists()

    restored = recover_experiment(root, expected_experiment_id=check["experiment_id"])
    assert restored["state"] == "interrupted" and restored["committed_frames"] == 1
    assert restored["recovery"]["recovered"] == scan["recoverable"]
    assert (root / first["path"]).read_bytes() == evidence[partial(first)]
    subsequent = inspect_recovery(root)
    assert not subsequent["needs_finalize"] and not subsequent["recovery"]["recoverable"]


@pytest.mark.parametrize("with_invalid", [False, True])
def test_recovery_restores_a_shared_tiff_and_keeps_its_other_event_link(tmp_path, with_invalid):
    cfg = CaptureConfig.from_dict({
        "config_version": 1, "name": "共享原图恢复", "cameras": [{"id": "sim:0"}],
        "phases": [{"mode": "event", "pre_frames": 1, "post_frames": 2, "fps": 4, "overlap_events": True}],
    })
    root = tmp_path / "shared"
    writer = ExperimentWriter(root, cfg)
    try:
        camera = "sim:0"
        writer.register_camera(Device(camera, "sim", "StrainTrace", "Fixture", "SIM-0000", "simulated"), {})
        utc, stamp = receipt_times()
        for event, shift in (("first", 0), ("second", 10)):
            writer.event("signal_received", {"phase": 1, "id": event, "overlap": True, "source": "operator",
                                             "received_utc_ns": utc + shift, "received_monotonic_ns": stamp + shift},
                         event_id=event)
        image = Image(np.zeros((80, 96), dtype=np.uint8), "Mono8", 8, "mono", utc + 5, stamp + 5)
        good = writer.prepare_frame(camera, 1, 1, "test", image, None, "first", "post", 0,
                                    event_memberships=(("first", "post"),)).record
        writer.record_capture_events((), links=[("second", camera, 1, "pre")])
        if with_invalid:
            writer.prepare_frame(camera, 2, 1, "test", image, None, "unknown", "post", 0,
                                 event_memberships=(("unknown", "post"),))
    finally:
        writer.close()

    check = inspect_recovery(root)["recovery"]
    assert [entry["frame_id"] for entry in check["recoverable"]] == [good["frame_id"]]
    assert [entry["code"] for entry in check["issues"]] == (["FRAME_JOURNAL_ENTRY_INVALID"] if with_invalid else [])
    result = recover_experiment(root)
    assert result["state"] == "interrupted" and result["committed_images"] == 1
    assert result["ok"] is not with_invalid
    with ExperimentReader(root) as reader:
        one, = reader.test_sequence_page(camera, after=0, event_context_id="first")
        two, = reader.test_sequence_page(camera, after=0, event_context_id="second")
        assert one["frame_id"] == two["frame_id"] == good["frame_id"]
        assert one["sha256"] == two["sha256"] == good["sha256"]
        assert (one["event_position"], two["event_position"]) == ("post", "pre")


def test_recovery_check_accounts_for_duplicates_and_conflicts_between_pending_entries(tmp_path):
    root = tmp_path / "experiment"
    frame, = stage_frames(root, 1)
    journal = root / FRAME_WRITE_JOURNAL
    original = journal.read_bytes()
    payload = json.loads(original.split(b" ", 1)[1])
    payload["frame"]["backend_metadata"] = {"ChangedReceipt": 1}
    encoded = json.dumps(payload).encode("utf-8")
    journal.write_bytes(original + original + hashlib.sha256(encoded).hexdigest().encode("ascii") + b" " + encoded + b"\n")

    check = inspect_recovery(root)["recovery"]
    assert len(check["recoverable"]) == 1 and check["recoverable"][0]["frame_id"] == frame["frame_id"]
    assert check["already_committed"] == 0 and check["duplicate_pending"] == 1
    assert [(row["code"], row["line"]) for row in check["issues"]] == [("FRAME_LEDGER_CONFLICT", 3)]
    restored = recover_experiment(root)["recovery"]
    assert restored["recovered"] == check["recoverable"]
    assert restored["already_committed"] == 1 and restored["issues"] == check["issues"]


def test_recovery_check_refuses_an_active_writer_and_uses_the_original_experiment_identity(tmp_path):
    root = tmp_path / "experiment"
    writer = ExperimentWriter(root, capture_plan())
    try:
        with pytest.raises(CaptureError) as locked:
            inspect_recovery(root)
        assert locked.value.code == "WRITER_ACTIVE"
    finally:
        writer.close()
    lease_path = root / ".capture-writer"
    lease_path.unlink()
    assert inspect_recovery(root)["experiment_id"] == writer.experiment_id and not lease_path.exists()

    before = (root / DATABASE).read_bytes()
    for action in (inspect_recovery, recover_experiment, continuation_plan):
        with pytest.raises(CaptureError) as changed:
            action(root, expected_experiment_id="another-experiment")
        assert changed.value.code == "RECOVERY_SCOPE_CHANGED"
    assert (root / DATABASE).read_bytes() == before


def test_write_through_syncs_journal_once_before_syncing_and_publishing_a_batch(tmp_path, monkeypatch):
    from straintrace_capture import storage

    root = tmp_path / "experiment"
    writer = ExperimentWriter(root, capture_plan(durability="write_through"))
    try:
        prepared = _prepare_frames(writer, 3)
        assert not list(root.rglob("*.tiff"))
        with ExperimentReader(root) as reader:
            assert reader.frames() == []
        partials = [root / item.record["path"] for item in prepared]
        inode_sequences = {path.with_suffix(".tiff.part").stat().st_ino: seq for seq, path in enumerate(partials, 1)}
        assert len(inode_sequences) == 3
        operations = []
        fsync, publish = os.fsync, storage._publish_path
        journal_fd = writer._frame_journal.fileno()

        def traced_fsync(fd):
            fsync(fd)
            if fd == journal_fd:
                operations.append("journal")
            elif (sequence := inode_sequences.get(os.fstat(fd).st_ino)) is not None:
                operations.append(f"sync {sequence}")

        def traced_publish(partial, target, *, write_through):
            assert write_through
            operations.append(f"publish {int(target.stem)}")
            publish(partial, target, write_through=write_through)

        with monkeypatch.context() as patched:
            patched.setattr(os, "fsync", traced_fsync)
            patched.setattr(storage, "_publish_path", traced_publish)
            writer.commit_frames(prepared)
        assert operations == ["journal", "sync 1", "publish 1", "sync 2", "publish 2", "sync 3", "publish 3"]
    finally:
        writer.close()
    assert (root / FRAME_WRITE_JOURNAL).stat().st_size == 0
    with ExperimentReader(root) as reader:
        assert reader.verify()["ok"] and len(reader.frames()) == 3


@pytest.mark.parametrize("failure", ("journal_sync", "second_publication"))
def test_write_through_batch_failure_recovers_frames_and_preserves_saved_counts(tmp_path, monkeypatch, failure):
    from straintrace_capture import storage

    config = capture_plan(durability="write_through").to_dict()
    config["phases"] = [{"mode": "single", "role": "reference"}]
    config["cameras"].append({**config["cameras"][0], "id": "sim:1"})
    run = CaptureSession(CaptureConfig.from_dict(config), tmp_path / "experiment")
    fsync, publish = os.fsync, storage._publish_path
    attempted = []

    def fail_sync(fd):
        if run._store and (stream := run._store._frame_journal) is not None and fd == stream.fileno():
            raise OSError("injected journal sync failure")
        fsync(fd)

    def fail_publication(partial, target, *, write_through):
        if target.suffix == ".tiff":
            attempted.append(target)
            if len(attempted) == 2:
                raise OSError("injected second publication failure")
        publish(partial, target, write_through=write_through)

    with monkeypatch.context() as patched:
        if failure == "journal_sync":
            patched.setattr(os, "fsync", fail_sync)
        else:
            patched.setattr(storage, "_publish_path", fail_publication)
        run.start()
        status = run.wait(8)
    saved = int(failure == "second_publication")
    assert status["state"] == "failed"
    assert sum(camera["saved"] for camera in status["cameras"].values()) == saved
    assert sum(camera["app_unsaved"] for camera in status["cameras"].values()) == 2 - saved
    assert status["error"]["code"] == ("WRITE_FAILED" if saved else "FRAME_JOURNAL_WRITE_FAILED")
    if saved:
        missed, = [camera_id for camera_id, values in status["cameras"].items() if values["app_unsaved"]]
        assert status["error"]["message"].startswith(f"{missed} sequence 1")
    assert len(list(run.output.rglob("*.tiff"))) == saved
    parts_before = {part.with_suffix(""): part.read_bytes() for part in run.output.rglob("*.tiff.part")}
    assert len(parts_before) == 2 - saved

    restoration = recover_experiment(run.output)
    assert restoration["committed_frames"] == 2
    assert len(restoration["recovery"]["recovered"]) == 2 - saved
    assert restoration["recovery"]["already_committed"] == saved and not restoration["recovery"]["issues"]
    assert all(path.read_bytes() == payload for path, payload in parts_before.items())
    with ExperimentReader(run.output) as reader:
        group, = reader.pairing()["groups"]
        assert group["status"] == "associated"
        assert reader.experiment["summary"] == status


def test_write_through_recovery_requires_journal_sync_before_publishing(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    frame, = stage_frames(root, 1, durability="write_through")
    target = root / frame["path"]
    partial = target.with_suffix(".tiff.part")
    evidence_inode = (root / FRAME_WRITE_JOURNAL).stat().st_ino
    fsync = os.fsync

    def fail_sync(fd):
        if os.fstat(fd).st_ino == evidence_inode:
            raise OSError("injected recovery sync failure")
        fsync(fd)

    with monkeypatch.context() as patched:
        patched.setattr(os, "fsync", fail_sync)
        failed = recover_experiment(root)
    assert failed["committed_frames"] == 0 and not failed["ok"]
    assert failed["recovery"]["issues"][0]["code"] == "FRAME_JOURNAL_UNREADABLE"
    assert partial.exists() and not target.exists()
    assert recover_experiment(root)["committed_frames"] == 1


def test_recovery_preserves_corrupt_payload_and_ignores_a_torn_descriptor(tmp_path):
    entries = stage_frames(tmp_path / "experiment", 2)
    root = tmp_path / "experiment"
    damaged = root / entries[1]["path"]
    byte_values = bytearray(damaged.read_bytes())
    byte_values[-1] ^= 1
    damaged.write_bytes(byte_values)
    with (root / FRAME_WRITE_JOURNAL).open("ab") as stream:
        stream.write(b"incomplete descriptor")

    result = recover_experiment(root)
    assert not result["ok"] and result["committed_frames"] == 1
    assert [row["code"] for row in result["recovery"]["issues"]] == [
        "FRAME_PAYLOAD_MISMATCH", "FRAME_JOURNAL_ENTRY_INVALID",
    ]
    assert damaged.read_bytes() == byte_values
    assert result["recovery"]["recovered"][0]["frame_id"] == entries[0]["frame_id"]


def test_recovery_can_retry_after_a_partial_file_has_been_published(tmp_path, monkeypatch):
    from straintrace_capture import recovery

    root = tmp_path / "experiment"
    frame, = stage_frames(root, 1)
    target = root / frame["path"]
    temporary = target.with_suffix(".tiff.part")
    content = target.read_bytes()
    target.rename(temporary)

    def fail_insert(*args):
        raise sqlite3.OperationalError("injected ledger failure")

    with monkeypatch.context() as patched:
        patched.setattr(recovery, "_insert_frames", fail_insert)
        with pytest.raises(sqlite3.OperationalError, match="injected ledger failure"):
            recover_experiment(root)
    assert target.read_bytes() == content
    with ExperimentReader(root) as reader:
        assert reader.frames() == []
    assert recover_experiment(root)["committed_frames"] == 1
    assert recover_experiment(root)["recovery"]["already_committed"] == 1


def test_recovery_requires_all_tiffs_from_the_same_receipt(tmp_path):
    root = tmp_path / "parts"
    writer = ExperimentWriter(root, capture_plan(durability="write_through"))
    try:
        camera = Device("sim:0", "sim", "StrainTrace", "Fixture", "SIM-0000", "simulated")
        writer.register_camera(camera, {})
        utc, monotonic = receipt_times()
        image = Image(np.zeros((80, 96), dtype=np.uint16), "Mono12", 12, "mono", utc, monotonic, 42, 42)
        part = replace(image, pixels=np.full((20, 30), 429, dtype=np.uint16))
        event = TriggerEvent("recovery-part-test", "software_trigger", {
            "phase": 1, "expected_cameras": [camera.id], "pairing_basis": "common_capture_request",
            "synchronization": "unverified",
        })
        frame = writer.prepare_frame(camera.id, 1, 1, "reference", replace(image, auxiliary_images=(part,)),
                                     event, None, None, 0).record
    finally:
        writer.close()
    primary, auxiliary = (root / asset["path"] for asset in frame_image_records(frame))
    partial = auxiliary.with_suffix(".tiff.part")
    payload = partial.read_bytes()
    partial.rename(tmp_path / "part-backup.tiff")
    for error in ("FRAME_PAYLOAD_MISSING", "FRAME_PAYLOAD_MISMATCH"):
        result = recover_experiment(root)
        assert result["committed_frames"] == result["committed_images"] == 0
        assert result["recovery"]["issues"][0]["code"] == error
        assert result["recovery"]["issues"][0]["image_path"] == frame["auxiliary_images"][0]["path"]
        assert not primary.exists() and primary.with_suffix(".tiff.part").exists()
        with ExperimentReader(root) as reader:
            assert not reader.events(kind="software_trigger")
        if error == "FRAME_PAYLOAD_MISSING":
            partial.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    partial.write_bytes(payload)
    result = recover_experiment(root)
    assert result["ok"] and result["committed_frames"] == 1 and result["committed_images"] == 2
    assert auxiliary.read_bytes() == payload
    with ExperimentReader(root) as reader:
        assert len(reader.events(kind="software_trigger")) == 1
        assert reader.frames()[0]["native_frame_id"] == 42
    assert recover_experiment(root)["recovery"]["already_committed"] == 1


def test_failed_ledger_commit_preserves_frame_for_recovery(tmp_path, monkeypatch):
    config = capture_plan().to_dict()
    config["phases"] = [{"mode": "single", "role": "reference"}]
    run = CaptureSession(CaptureConfig.from_dict(config), tmp_path / "experiment")

    def fail_commit(self, prepared):
        raise sqlite3.OperationalError("injected capture commit failure")

    with monkeypatch.context() as patched:
        patched.setattr(ExperimentWriter, "commit_frames", fail_commit)
        run.start()
        status = run.wait(8)
    assert status["state"] == "failed" and status["cameras"]["sim:0"]["app_unsaved"] == 1
    result = recover_experiment(run.output)
    assert result["state"] == "failed" and not result["complete"] and result["committed_frames"] == 1
    assert result["runtime_summary_available"]
    assert len(result["recovery"]["recovered"]) == 1
    with ExperimentReader(run.output) as reader:
        assert reader.experiment["summary"] == status


@pytest.mark.parametrize("fail_commit", [False, True], ids=["commit", "recover"])
def test_large_buffered_tiff_publication_can_continue_across_batches_during_a_commit(tmp_path, monkeypatch, fail_commit):
    from straintrace_capture.cameras.simulated import SimulatedCamera

    original_insert = storage._insert_frames
    original_publish = ExperimentWriter._publish_frame
    original_read = SimulatedCamera.read
    entered, release, published_last = threading.Event(), threading.Event(), threading.Event()
    frame_count = 5
    commit_attempts = []

    def hold_first_commit(db, rows):
        commit_attempts.append([row.record["sequence"] for row in rows])
        if rows[0].record["sequence"] == 1:
            entered.set()
            assert release.wait(30)
            if fail_commit:
                raise sqlite3.OperationalError("injected frame commit failure after TIFF publication")
        original_insert(db, rows)

    def mark_last_tiff(writer, row):
        original_publish(writer, row)
        if row.record["sequence"] == frame_count:
            published_last.set()

    def wait_until_first_commit(camera, timeout_s):
        # Keep the first batch to one frame, then let the stream proceed while
        # its ledger transaction is held. No timed capture ticks can be missed.
        if camera._counter and not entered.wait(15):
            raise TimeoutError("the first frame did not reach its ledger commit")
        return original_read(camera, timeout_s)

    monkeypatch.setattr(storage, "_insert_frames", hold_first_commit)
    monkeypatch.setattr(ExperimentWriter, "_publish_frame", mark_last_tiff)
    monkeypatch.setattr(SimulatedCamera, "read", wait_until_first_commit)
    data = capture_plan().to_dict()
    data["cameras"][0]["settings"] = {"width": 1600, "height": 720, "pixel_format": "Mono16", "fps": 5.}
    data["phases"] = [{"mode": "continuous", "role": "test", "frame_count": frame_count}]
    data["max_buffer_bytes"] = 12 * 2**20
    session = CaptureSession(CaptureConfig.from_dict(data), tmp_path / "concurrent-tiffs")
    session.start()
    try:
        # The property under test is that subsequent TIFF publication does not
        # wait for the held SQLite commit, regardless of camera or disk speed.
        assert entered.wait(15)
        assert published_last.wait(15), "the first SQLite transaction stalled a later TIFF"
        deadline = time.monotonic() + 5
        while session.status()["buffered_raw_bytes"] and time.monotonic() < deadline:
            time.sleep(.005)
        status = session.status()
        assert status["buffered_raw_bytes"] == 0
        assert status["cameras"]["sim:0"]["pending_write"] == frame_count
        with ExperimentReader(session.output) as reader:
            assert not reader.frames()
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((session.output / "images").rglob("*.tiff"))]
        assert len(hashes) == frame_count
    finally:
        release.set()
        result = session.wait(8)
    monkeypatch.setattr(storage, "_insert_frames", original_insert)

    if fail_commit:
        assert result["state"] == "failed" and result["cameras"]["sim:0"]["app_unsaved"] == frame_count
        assert commit_attempts == [[1]]
        with ExperimentReader(session.output) as reader:
            assert not reader.frames()
        recovered = recover_experiment(session.output)
        assert recovered["ok"] and len(recovered["recovery"]["recovered"]) == frame_count
    else:
        assert result["state"] == "completed"
        assert commit_attempts[0] == [1] and len(commit_attempts) > 1
        assert [sequence for batch in commit_attempts for sequence in batch] == list(range(1, frame_count + 1))
    with ExperimentReader(session.output) as reader:
        assert [row["sha256"] for row in reader.frames()] == hashes
        assert reader.verify()["ok"]


def test_an_earlier_commit_keeps_journal_entries_for_newly_published_tiffs(tmp_path, monkeypatch):
    insert = storage._insert_frames
    entered, release = threading.Event(), threading.Event()

    def hold_while_the_next_frame_is_published(db, rows):
        if rows[0].record["sequence"] == 1:
            entered.set()
            assert release.wait(4)
        insert(db, rows)

    monkeypatch.setattr(storage, "_insert_frames", hold_while_the_next_frame_is_published)
    root = tmp_path / "checkpoint"
    camera = "sim:0"
    writer = ExperimentWriter(root, capture_plan())
    writer.register_camera(Device(camera, "sim", "StrainTrace", "Fixture", "SIM-0000", "simulated"), {})
    pixels = np.arange(80 * 96, dtype=np.uint16).reshape(80, 96) % 4096
    image = Image(pixels, "Mono12", 12, "mono", *receipt_times())
    try:
        first = writer.prepare_frame(camera, 1, 2, "test", image, None, None, None, 0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            ack = pool.submit(writer.commit_frames, [first])
            try:
                assert entered.wait(4)
                second = writer.prepare_frame(camera, 2, 2, "test", image, None, None, None, 0)
            finally:
                release.set()
            ack.result(timeout=4)
        third = writer.prepare_frame(camera, 3, 2, "test", image, None, None, None, 0)
    finally:
        writer.close()

    wanted = [second.record["frame_id"], third.record["frame_id"]]
    pending = inspect_recovery(root)["recovery"]
    assert pending["already_committed"] == 1
    assert [row["frame_id"] for row in pending["recoverable"]] == wanted
    restored = recover_experiment(root)
    assert restored["ok"] and restored["committed_images"] == 3
    assert [row["frame_id"] for row in restored["recovery"]["recovered"]] == wanted


def test_an_older_ledger_keeps_unattributed_files_unregistered(tmp_path):
    root = tmp_path / "legacy"
    frame, = stage_frames(root, 1)
    with closing(sqlite3.connect(root / DATABASE)) as db, db:
        record = json.loads(db.execute("SELECT record FROM experiment").fetchone()[0])
        record.pop("frame_write_journal")
        db.execute("UPDATE experiment SET record=?", (json.dumps(record),))

    result = recover_experiment(root)
    assert result["state"] == "interrupted" and result["committed_frames"] == 0
    assert result["recovery"]["journal"] == "not_recorded"
    assert result["issues"] == [{"code": "UNCOMMITTED_FILE", "path": frame["path"]}]
