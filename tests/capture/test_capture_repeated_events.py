import hashlib
import json
import sqlite3
import threading
import time

import pytest

from straintrace_capture import storage
from straintrace_capture import (
    CaptureConfig, CaptureError, CaptureSession, ExperimentReader, SequenceSelection, export_sequence, inspect_plan,
    verify_sequence,
)
from straintrace_capture.cli import main
from straintrace_capture.control import CaptureControl
from straintrace_capture.handoff import analysis_snapshot
from straintrace_capture.storage import DATABASE
from straintrace_capture.trial import prepare_trial


def plan(**phase):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "重复事件",
        "cameras": [{"id": "sim:0"}, {"id": "sim:1"}],
        "phases": [{"mode": "event", "pre_frames": 3, "post_frames": 4, "fps": 7,
                    "event_count": 2, "duration_s": 8, **phase}],
        "min_free_bytes": 0,
    })


def until(check, *, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if result := check():
            return result
        time.sleep(.01)
    raise AssertionError("timed out waiting for an event phase")


def progress(session):
    return session.status()["phase_progress"] or {}


def window(session, number, state):
    active = progress(session).get("event_window", {})
    return active if active.get("event_number") == number and active.get("state") == state else None


def stop(session):
    session.request_stop()
    session.wait(6)


@pytest.mark.parametrize("overlap", [False, True], ids=["successive", "shared"])
def test_software_rounds_continue_while_the_signal_ledger_transaction_is_held(tmp_path, monkeypatch, overlap):
    entered, release = threading.Event(), threading.Event()
    insert = storage._insert_event

    def hold_first_signal(db, kind, details, **options):
        insert(db, kind, details, **options)
        if kind == "signal_received" and not entered.is_set():
            entered.set()
            assert release.wait(4)

    monkeypatch.setattr(storage, "_insert_event", hold_first_signal)
    session = CaptureSession(plan(overlap_events=overlap, event_count=2 if overlap else 1, fps=6,
                                  post_frames=9, duration_s=8), tmp_path / "delayed-signal")
    session.start()
    try:
        until(lambda: window(session, 1, "waiting"))
        first = session.request_trigger(expected_phase=1, expected_window=1)
        assert entered.wait(3)
        with ExperimentReader(session.output) as reader:
            assert reader.received_signal(first) is None
        before = progress(session)["received_by_camera"]
        until(lambda: all(progress(session).get("received_by_camera", {}).get(camera, 0) >= count + 2
                          for camera, count in before.items()), timeout=.85)
        if overlap:
            second = session.request_trigger(expected_phase=1, expected_window=2)
        release.set()
        result = session.wait(6)
    finally:
        release.set()
        stop(session)

    assert result["state"] == "completed" and result["scheduled_misses"] == 0, result
    assert result["buffered_raw_bytes"] == 0
    accepted = [first, second] if overlap else [first]
    with ExperimentReader(session.output) as reader:
        assert [item["event_id"] for item in reader.events(kind="signal_received")] == accepted
        assert [item["signal_id"] for item in reader.events(kind="event_window")] == accepted
        for context in accepted:
            frames = reader.test_sequence_page("sim:0", after=0, event_context_id=context)
            assert sum(frame["event_position"] == "post" for frame in frames) == 9
        assert reader.verify()["complete"]


def test_rearming_continues_while_the_previous_window_record_is_waiting_to_commit(tmp_path, monkeypatch):
    from straintrace_capture import session as capture_module

    entered, release = threading.Event(), threading.Event()
    steps = threading.Semaphore(0)
    insert = storage._insert_event

    class ControllerClock:
        # Advance the capture schedule only when the test releases a round.
        # Other threads and image receipt timestamps retain their real clock.
        value = time.perf_counter()

        def perf_counter(self):
            return self.value if threading.current_thread().name == "capture-controller" else time.perf_counter()

        def __getattr__(self, name):
            return getattr(time, name)

    clock = ControllerClock()

    class SteppedStop(threading.Event):
        def wait(self, timeout=None):
            if threading.current_thread().name != "capture-controller":
                return super().wait(timeout)
            while not self.is_set():
                if steps.acquire(timeout=.05):
                    clock.value += timeout or 0
                    return False
            return True

    def advance(rounds):
        for _ in range(rounds):
            steps.release()

    def hold_first_window(db, kind, details, **options):
        insert(db, kind, details, **options)
        if kind == "event_window" and details["event_number"] == 1:
            entered.set()
            assert release.wait(4)

    monkeypatch.setattr(storage, "_insert_event", hold_first_window)
    monkeypatch.setattr(capture_module, "time", clock)
    data = plan(fps=6, post_frames=3, duration_s=8).to_dict()
    for camera in data["cameras"]:
        camera["settings"].update(width=32, height=32)
    session = CaptureSession(CaptureConfig.from_dict(data), tmp_path / "delayed-window")
    session._stop = SteppedStop()
    session.start()
    try:
        advance(3)
        until(lambda: window(session, 1, "waiting"))
        first = session.request_trigger(expected_window=1)
        advance(3)
        assert entered.wait(3)
        advance(3)
        until(lambda: window(session, 2, "waiting"), timeout=1.3)
        second = session.request_trigger(expected_window=2)
        release.set()
        advance(3)
        result = session.wait(6)
    finally:
        release.set()
        stop(session)

    assert result["state"] == "completed" and result["scheduled_misses"] == 0, result
    with ExperimentReader(session.output) as reader:
        windows = reader.events(kind="event_window")
        assert [event["signal_id"] for event in windows] == [first, second]
        assert all((event["pre_retained_rounds"], event["post_received_rounds"]) == (3, 3) for event in windows)
        for identity in (first, second):
            for camera in ("sim:0", "sim:1"):
                frames = [row for row in reader.frames(camera_id=camera) if row["event_context_id"] == identity]
                assert sum(row["event_position"] == "pre" for row in frames) == 3
                assert sum(row["event_position"] == "post" for row in frames) == 3
        assert reader.verify()["complete"]


def test_failed_signal_transaction_never_admits_images_without_their_required_event(tmp_path, monkeypatch):
    insert = storage._insert_event

    def fail_after_insert(db, kind, details, **options):
        insert(db, kind, details, **options)
        if kind == "signal_received":
            raise sqlite3.OperationalError("test rollback after inserting the signal")

    monkeypatch.setattr(storage, "_insert_event", fail_after_insert)
    session = CaptureSession(plan(overlap_events=True, fps=5), tmp_path / "failed-signal")
    session.start()
    try:
        until(lambda: window(session, 1, "waiting"))
        identity = session.request_trigger(expected_window=1)
        result = session.wait(6)
    finally:
        stop(session)

    assert result["state"] == "failed" and result["error"]["code"] == "EVENT_LEDGER_WRITE_FAILED", result
    assert all(camera["app_unsaved"] >= 3 for camera in result["cameras"].values())
    assert result["buffered_raw_bytes"] == 0
    with ExperimentReader(session.output) as reader:
        assert reader.received_signal(identity) is None
        assert not reader.events(kind="event_window") and not reader.frames()
        assert reader.verify()["ok"]
    assert not list((session.output / "images").rglob("*.tiff*"))


def test_raw_queue_still_stops_admission_during_a_prolonged_event_ledger_write(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    insert = storage._insert_event

    def hold_signal(db, kind, details, **options):
        insert(db, kind, details, **options)
        if kind == "signal_received":
            entered.set()
            assert release.wait(4)

    data = plan(overlap_events=True, pre_frames=0, post_frames=8, fps=6).to_dict()
    data["queue_capacity"] = 2
    monkeypatch.setattr(storage, "_insert_event", hold_signal)
    session = CaptureSession(CaptureConfig.from_dict(data), tmp_path / "bounded-queue")
    session.start()
    try:
        until(lambda: window(session, 1, "waiting"))
        identity = session.request_trigger(expected_window=1)
        assert entered.wait(3)
        until(lambda: (session.status()["error"] or {}).get("code") == "WRITE_QUEUE_FULL", timeout=1.6)
        release.set()
        result = session.wait(6)
    finally:
        release.set()
        stop(session)

    assert result["state"] == "failed" and result["queue_high_water"] == result["queue_capacity"] == 2
    assert all(camera["saved"] == 2 and camera["app_unsaved"] == 1 for camera in result["cameras"].values())
    assert result["buffered_raw_bytes"] == 0
    with ExperimentReader(session.output) as reader:
        assert reader.received_signal(identity) is not None
        event, = reader.events(kind="event_window")
        assert (event["signal_id"], event["result"], event["post_received_rounds"]) == (identity, "short_post", 2)
        assert reader.verify()["committed_images"] == 4


def test_overlapping_rounds_preserve_shared_tiff_identity_and_each_event_position(tmp_path, monkeypatch):
    boundary_started, release = threading.Event(), threading.Event()
    fire = CaptureSession._fire

    def hold_one_camera(session, camera, phase, index, event):
        if camera.device.id == "sim:1" and session.status()["cameras"]["sim:1"]["received"] == 4:
            boundary_started.set()
            assert release.wait(4)
        return fire(session, camera, phase, index, event)

    monkeypatch.setattr(CaptureSession, "_fire", hold_one_camera)
    session = CaptureSession(plan(overlap_events=True, fps=4), tmp_path / "overlap")
    session.start()
    try:
        until(lambda: window(session, 1, "waiting"))
        first = session.request_trigger(expected_window=1)
        assert boundary_started.wait(4)
        until(lambda: progress(session).get("received_by_camera", {}).get("sim:0") == 5)
        state = until(lambda: window(session, 2, "waiting"))
        assert state["active_events"] == 1 and state["completed_events"] == 0
        with pytest.raises(CaptureError, match="no longer active"):
            session.request_trigger(expected_window=1)
        second = session.request_trigger(expected_window=2)
        state = progress(session)["event_window"]
        assert state["active_events"] == 2 and state["event_target_reached"]
        with pytest.raises(CaptureError) as unavailable:
            session.request_trigger()
        assert unavailable.value.code == "EVENT_TARGET_REACHED"
        release.set()
        result = session.wait(6)
    finally:
        release.set()
        stop(session)
    assert result["state"] == "completed", result
    assert result["event_phases"][0]["completed_events"] == 2
    assert result["buffered_raw_bytes"] == 0
    with ExperimentReader(session.output) as reader:
        events = {event["signal_id"]: event for event in reader.events(kind="event_window")}
        assert events[first]["boundary_rounds"] == 0 and events[second]["boundary_rounds"] == 1
        assert [item["details"]["signal_id"] for item in reader.timeline(event_windows_only=True)["events"]] == [second, first]
        for camera in ("sim:0", "sim:1"):
            a = reader.test_sequence_page(camera, after=0, event_context_id=first)
            b = reader.test_sequence_page(camera, after=0, event_context_id=second)
            assert [row["sequence"] for row in a] == list(range(1, 8))
            assert [row["sequence"] for row in b] == list(range(2, 10))
            assert [row["event_position"] for row in a] == ["pre"] * 3 + ["post"] * 4
            assert [row["event_position"] for row in b] == ["pre"] * 3 + ["boundary"] + ["post"] * 4
            for old, new in zip(a[1:], b):
                assert (old["frame_id"], old["path"], old["sha256"]) == (new["frame_id"], new["path"], new["sha256"])
            boundary = b[3]["timestamps"]["received_monotonic"]["value"]
            assert (boundary >= events[second]["signal_monotonic_ns"]) == (camera == "sim:1")
            page = reader.frame_page(camera_id=camera, event_context_id=second, limit=2)
            seen = page["frames"]
            while cursor := page["older_before"]:
                page = reader.frame_page(camera_id=camera, event_context_id=second, limit=2, before=cursor,
                                         snapshot=page["snapshot_ordinal"])
                seen += page["frames"]
            assert [(row["sequence"], row["event_position"]) for row in reversed(seen)] == [
                (row["sequence"], row["event_position"]) for row in b
            ]
            exact = reader.frame_page(camera_id=camera, event_context_id=second, first_sequence=4, last_sequence=4)
            assert exact["frames"][0]["event_position"] == "pre"
        verified = reader.verify()
        assert verified["complete"] and verified["committed_images"] == 18

    snapshot = analysis_snapshot(session.output, camera_id="sim:0", event_context_id=second, reference_sequence=4)
    assert [row["sequence"] for row in snapshot["frames"]] == list(range(4, 10))
    assert [row["event_position"] for row in snapshot["frames"]] == ["pre", "boundary", "post", "post", "post", "post"]
    target = tmp_path / "export"
    exported = export_sequence(session.output, target, SequenceSelection(
        camera_id="sim:0", event_context_id=second, reference_sequence=4,
    ))
    assert exported["event"]["signal_id"] == second and exported["test_frames"] == 5
    assert verify_sequence(target)["ok"]
    assert len(list((session.output / "images").rglob("*.tiff"))) == 18

    with sqlite3.connect(session.output / DATABASE) as db:
        db.execute("DELETE FROM event_frame_links WHERE event_id=? AND camera_id=? AND sequence=4", (second, "sim:0"))
    with ExperimentReader(session.output) as reader:
        assert any(issue["code"] == "EVENT_WINDOW_FRAME_COUNT_MISMATCH" and issue["event_id"] == second
                   for issue in reader.verify()["issues"])


def test_overlapping_csv_recrosses_and_links_two_windows_at_once(tmp_path):
    source = tmp_path / "load.csv"
    source.write_bytes(b"seconds,force\n0,4\n")
    cfg = plan(overlap_events=True, pre_frames=2, post_frames=3, fps=4, csv_trigger={
        "path": str(source), "value_column": "force", "time_column": "seconds",
        "threshold": 5, "unit": "kN", "quantity": "load", "direction": "rising",
    })
    session = CaptureSession(cfg, tmp_path / "csv")
    session.start()
    try:
        until(lambda: window(session, 1, "waiting")
              and (progress(session).get("csv_trigger") or {}).get("state") == "watching")
        with source.open("ab") as stream:
            stream.write(b"1,4\n2,6\n3,3\n4,7\n")
        result = session.wait(6)
    finally:
        stop(session)
    assert result["state"] == "completed", result
    final, = result["csv_triggers"]
    assert final["events_accepted"] == 2 and not final["busy_crossings"]
    with ExperimentReader(session.output) as reader:
        events = reader.events(kind="event_window")
        assert [event["csv_crossing"]["sample"]["source_time_text"] for event in events] == ["2", "4"]
        ids = [event["signal_id"] for event in events]
        shared = [set(row["frame_id"] for row in reader.test_sequence_page("sim:0", after=0, event_context_id=context))
                  for context in ids]
        assert shared[0] & shared[1]
        assert reader.verify()["complete"]


def test_stop_records_each_accepted_window_and_limits_simultaneous_requests(tmp_path):
    session = CaptureSession(plan(overlap_events=True, event_count=0, pre_frames=2, post_frames=8, fps=4),
                             tmp_path / "stopped")
    session.start()
    try:
        until(lambda: window(session, 1, "waiting"))
        accepted = [session.request_trigger(expected_window=number) for number in range(1, 17)]
        with pytest.raises(CaptureError) as busy:
            session.request_trigger()
        assert busy.value.code == "EVENT_ALREADY_RECEIVED"
    finally:
        stop(session)
    assert session.status()["event_phases"][0]["partial_events"] == 16
    assert session.status()["buffered_raw_bytes"] == 0
    with ExperimentReader(session.output) as reader:
        rows = reader.events(kind="event_window")
        assert [row["signal_id"] for row in rows] == accepted
        assert all(row["result"] == "short_post" and row["pre_retained_rounds"] == 2 for row in rows)
        first_sequences = [reader.test_sequence_page("sim:0", after=0, event_context_id=context, limit=1)[0]["sequence"]
                           for context in accepted]
        assert len(set(first_sequences)) == 1
        assert reader.verify()["ok"]


def test_overlap_readiness_uses_shared_rounds_and_preserves_the_trial_count(tmp_path):
    config = plan(overlap_events=True, duration_s=1.1)
    checks = inspect_plan(config, tmp_path / "formal")
    assert not [item for item in checks["issues"] if item["code"] == "EVENT_WINDOW_EXCEEDS_DURATION"]
    trial, _output = prepare_trial(config, tmp_path / "formal", seconds=1)
    assert trial.phases[0].overlap_events and trial.phases[0].event_count == 2
    too_short = inspect_plan(plan(overlap_events=True, duration_s=1.1, event_count=17), tmp_path / "short")
    assert any(item["code"] == "EVENT_WINDOW_EXCEEDS_DURATION" for item in too_short["issues"])


def test_rearms_with_fresh_frames_and_fences_the_external_window(tmp_path, capsys):
    session = CaptureSession(plan(), tmp_path / "run")
    session.start()
    try:
        with CaptureControl(session):
            until(lambda: window(session, 1, "buffering") or window(session, 1, "waiting"))
            until(lambda: window(session, 1, "waiting"))
            first = session.request_trigger(expected_phase=1, expected_window=1)
            with pytest.raises(CaptureError) as busy:
                session.request_trigger(expected_phase=1, expected_window=1)
            assert busy.value.code == "EVENT_ALREADY_RECEIVED"
            fresh = until(lambda: window(session, 2, "buffering"))
            assert fresh["completed_events"] == 1 and not fresh["signal_id"]
            assert not progress(session)["trigger_available"]
            with pytest.raises(CaptureError) as early:
                session.request_trigger(expected_phase=1, expected_window=2)
            assert early.value.code == "PREBUFFER_NOT_READY"
            until(lambda: window(session, 2, "waiting"))
            command = ["trigger", str(session.output), "--phase", "1", "--event-window"]
            assert main([*command, "1"]) == 3
            assert json.loads(capsys.readouterr().out)["error"]["code"] == "SIGNAL_WINDOW_CHANGED"
            assert main([*command, "2"]) == 0
            second = json.loads(capsys.readouterr().out)["signal_id"]
            result = session.wait(6)
    finally:
        stop(session)
    assert result["state"] == "completed", result
    assert result["event_phases"] == [{
        "phase": 1, "event_count": 2, "completed_events": 2, "partial_events": 0, "busy_signals": 1, "early_signals": 1,
    }]
    with ExperimentReader(session.output) as reader:
        events = reader.events(kind="event_window")
        assert [row["result"] for row in events] == ["complete", "complete"]
        assert [row["signal_id"] for row in events] == [first, second]
        assert [row["event_number"] for row in events] == [1, 2]
        page = reader.timeline(event_windows_only=True, limit=1)
        assert [row["details"]["signal_id"] for row in page["events"]] == [second]
        previous = reader.timeline(event_windows_only=True, limit=1, before=page["older_before"], snapshot=page["snapshot_ordinal"])
        assert [row["details"]["signal_id"] for row in previous["events"]] == [first]
        for camera in ("sim:0", "sim:1"):
            first_sequences = []
            for event in events:
                frames = reader.frame_page(camera_id=camera, event_context_id=event["signal_id"])["frames"]
                ordered = list(reversed(frames))
                assert [row["event_position"] for row in ordered] == (
                    ["pre"] * 3 + ["boundary"] * event["boundary_rounds"] + ["post"] * 4
                )
                for item in ordered:
                    frame = reader.frame_at_sequence(camera, item["sequence"])
                    stamp = frame["timestamps"]["received_monotonic"]["value"]
                    if item["event_position"] == "pre":
                        assert stamp < event["signal_monotonic_ns"]
                    if item["event_position"] == "post":
                        assert stamp >= event["signal_monotonic_ns"]
                sequences = [row["sequence"] for row in ordered]
                if first_sequences:
                    assert min(sequences) > max(first_sequences)
                first_sequences = sequences
        verified = reader.verify()
        assert verified["committed_images"] == 28 + 2 * sum(row["boundary_rounds"] for row in events)
        assert verified["ok"]

    with sqlite3.connect(session.output / DATABASE) as db:
        lookup = db.execute("""EXPLAIN QUERY PLAN SELECT ordinal FROM frames WHERE camera_id=?
            AND json_extract(record, '$.event_context_id')=? AND json_type(record, '$.event_context_id')='text'
            ORDER BY ordinal DESC""", ("sim:0", second)).fetchall()
        assert any("frame_event_window" in row[3] for row in lookup)
        target = first_sequences[-1]
        db.execute("UPDATE frames SET record=json_set(record, '$.event_context_id', ?) WHERE camera_id=? AND sequence=?",
                   (first, "sim:1", target))
    with ExperimentReader(session.output) as reader:
        codes = {issue["code"] for issue in reader.verify()["issues"]}
        assert "EVENT_WINDOW_FRAME_COUNT_MISMATCH" in codes


def test_csv_watches_recrossings_and_audits_signals_during_another_window(tmp_path):
    csv = tmp_path / "load.csv"
    csv.write_bytes(b"seconds,force\n0,9\n")
    cfg = plan(csv_trigger={"path": str(csv), "value_column": "force", "time_column": "seconds",
                            "threshold": 5, "unit": "kN", "quantity": "load", "direction": "rising"})
    session = CaptureSession(cfg, tmp_path / "run")

    def append(data):
        with csv.open("ab") as stream:
            stream.write(data)

    def watcher(key, value):
        return (progress(session).get("csv_trigger") or {}).get(key) == value

    session.start()
    try:
        until(lambda: watcher("state", "watching"))
        append(b"1,4\n2,5\n")
        until(lambda: watcher("early_crossings", 1))
        until(lambda: window(session, 1, "waiting"))
        append(b"3,3\n4,6\n")
        until(lambda: watcher("events_accepted", 1))
        append(b"5,2\n6,8\n")
        until(lambda: watcher("busy_crossings", 1))
        until(lambda: window(session, 2, "waiting"))
        assert watcher("events_accepted", 1)
        append(b"7,4\n8,5.5\n")
        result = session.wait(6)
    finally:
        stop(session)
    assert result["state"] == "completed", result
    final, = result["csv_triggers"]
    assert (final["events_accepted"], final["early_crossings"], final["busy_crossings"]) == (2, 1, 1)
    data = csv.read_bytes()
    with ExperimentReader(session.output) as reader:
        rows = reader.timeline(event_windows_only=True)["events"]
        assert [row["kind"] for row in rows] == ["event_window", "event_window"]
        assert [row["details"]["event_number"] for row in rows] == [2, 1]
        for row, samples in zip(rows, (("7", "8"), ("3", "4"))):
            evidence = row["details"]["csv_crossing"]
            assert row["time_basis"] == "host_signal" and row["utc_ns"] >= evidence["detected_utc_ns"]
            for field, time_text in zip(("previous_sample", "sample"), samples):
                item = evidence[field]
                assert item["source_time_text"] == time_text
                assert hashlib.sha256(data[item["offset"]:item["offset"] + item["length"]]).hexdigest() == item["sha256"]


def test_open_ended_monitoring_can_finish_without_an_event_and_short_windows_are_kept(tmp_path):
    quiet = CaptureSession(plan(event_count=0, pre_frames=0, post_frames=1, duration_s=.25), tmp_path / "quiet")
    quiet.start()
    summary = quiet.wait(5)
    assert summary["state"] == "completed" and summary["event_phases"][0]["completed_events"] == 0
    with ExperimentReader(quiet.output) as reader:
        assert not reader.timeline(event_windows_only=True)["events"]

    session = CaptureSession(plan(event_count=0, pre_frames=0, post_frames=4, fps=3, duration_s=5), tmp_path / "partial")
    session.start()
    try:
        until(lambda: window(session, 1, "waiting"))
        identity = session.request_trigger(expected_window=1)
        until(lambda: (window(session, 1, "recording_post") or {}).get("post_received_rounds") == 1)
    finally:
        stop(session)
    with ExperimentReader(session.output) as reader:
        row, = reader.timeline(event_windows_only=True)["events"]
        assert row["details"]["signal_id"] == identity and row["details"]["result"] == "short_post"
        assert reader.verify()["committed_images"] == 2
    assert session.status()["event_phases"][0]["partial_events"] == 1


def test_finite_deadline_reports_the_incomplete_target_and_retains_the_first_event(tmp_path):
    cfg = plan(pre_frames=0, post_frames=1, fps=10, duration_s=.8)
    session = CaptureSession(cfg, tmp_path / "deadline")
    session.start()
    try:
        until(lambda: window(session, 1, "waiting"))
        identity = session.request_trigger(expected_window=1)
        final = session.wait(5)
    finally:
        stop(session)
    assert final["state"] == "failed" and final["error"]["code"] == "EVENT_WINDOW_INCOMPLETE"
    assert final["event_phases"][0]["completed_events"] == 1
    with ExperimentReader(session.output) as reader:
        assert reader.frame_page(camera_id="sim:0", event_context_id=identity)["frames"]


def test_readiness_accounts_for_every_window_and_trial_keeps_the_formal_plan(tmp_path):
    config = plan(event_count=20, duration_s=30)
    formal = inspect_plan(config, tmp_path / "formal")
    assert formal["phases"][0]["frame_limit_per_camera"] == 160
    short = inspect_plan(plan(event_count=20, duration_s=5), tmp_path / "short")
    block, = [item for item in short["issues"] if item["code"] == "EVENT_WINDOW_EXCEEDS_DURATION"]
    assert block["event_count"] == 20 and block["planned_rounds"] == 35
    trial, output = prepare_trial(config, tmp_path / "formal", seconds=3)
    assert trial.phases[0].event_count == 0
    assert trial.metadata["capture_trial"]["scope"]["phase"]["event_count"] == 20
    assert inspect_plan(trial, output)["ready"]
    assert "event_count" not in plan(event_count=1).phases[0].to_dict()
