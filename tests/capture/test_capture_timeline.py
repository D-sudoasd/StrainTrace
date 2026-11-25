import json
import time

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError, CaptureSession, ExperimentReader
from straintrace_capture.capture_review import inspect_capture_review
from straintrace_capture.cameras.base import Image
from straintrace_capture.cameras.simulated import SimulatedCamera, simulated_device
from straintrace_capture.cli import main
from straintrace_capture.storage import ExperimentWriter


_START = 1_795_598_100_123_456_789


@pytest.fixture
def timeline_run(tmp_path):
    config = CaptureConfig.from_dict({
        "name": "时间线", "config_version": 1, "cameras": [{"id": "sim:0"}, {"id": "sim:1"}],
        "phases": [{"mode": "single", "role": "reference"}, {"mode": "timed", "duration_s": 2, "fps": 5}],
    })
    root = tmp_path / "时间线"
    writer = ExperimentWriter(root, config)
    try:
        for index in range(2):
            writer.register_camera(simulated_device(index), {})
        writer.set_state("running")
        writer.event("phase_started", {"phase": 1, "utc_ns": _START})
        writer.event("phase_ended", {"phase": 1, "utc_ns": _START + 4})
        writer.event("phase_confirmation_waiting", {"phase": 2, "utc_ns": _START + 6, "waiting_since_utc_ns": _START + 5})
        writer.event("phase_confirmed", {"phase": 2, "utc_ns": _START + 11, "requested_utc_ns": _START + 9})
        writer.event("phase_started", {"phase": 2, "utc_ns": _START + 14})
        for camera, sequence, phase, offset in (("sim:0", 1, 1, 3), ("sim:0", 2, 2, 16),
                                                 ("sim:0", 3, 2, 21), ("sim:1", 1, 2, 22)):
            image = Image(np.zeros((8, 10), np.uint8), "Mono8", 8, "mono", _START + offset, 10 + sequence)
            writer.commit_frames([writer.prepare_frame(camera, sequence, phase, config.phases[phase - 1].role,
                                                      image, None, None, None, 0)])
        writer.event("native_gap_unattributed", {"phase": 2, "camera_id": "sim:1", "previous": 1, "current": 3,
                                                 "utc_ns": _START + 24, "received_utc_ns": _START + 22})
        writer.event("trigger_incomplete", {"phase": 2, "utc_ns": _START + 30, "issued_utc_ns": _START + 18,
                                             "trigger_event_id": "trigger-5", "failed": [{"camera_id": "sim:1"}]})
        writer.event("schedule_miss", {"phase": 2, "first_tick": 2, "last_tick": 3, "first_due_utc_ns": _START + 17,
                                        "last_due_utc_ns": _START + 19, "utc_ns": _START + 41})
        writer.event("phase_ended", {"phase": 2, "utc_ns": _START + 42})
        writer.event("schedule_miss", {"phase": 2, "first_tick": 7, "last_tick": 7, "utc_ns": _START + 300})
        yield root, writer
    finally:
        writer.set_state("completed")
        writer.close()


def test_chart_uses_host_receipt_and_planned_times_with_inclusive_nanoseconds(timeline_run):
    root, _ = timeline_run
    with ExperimentReader(root) as reader:
        page = reader.timeline(phase=2)
        only_first = reader.timeline(camera_id="sim:0", phase=2)
        only_second = reader.timeline(camera_id="sim:1", phase=1)
        identity = reader.experiment["experiment_id"]
        with pytest.raises(CaptureError, match="different experiment"):
            reader.timeline(expected_experiment_id=identity + "-other")
    chart = page["overview"]
    assert (chart["first_utc_ns"], chart["last_utc_ns"], chart["bucket_ns"]) == (_START + 5, _START + 42, 1)
    assert chart["frame_count"] == chart["timed_frame_count"] == 3
    assert chart["buckets"][21 - 5]["camera_counts"] == {"sim:0": 1}
    assert chart["buckets"][22 - 5]["camera_counts"] == {"sim:1": 1}
    assert [mark["utc_ns"] for mark in chart["phase_marks"]] == [_START + k for k in (5, 9, 14, 42)]
    assert sum(bucket["incidents"] for bucket in chart["buckets"]) == 3
    assert chart["buckets"][17 - 5]["incidents"] == 1
    assert chart["buckets"][18 - 5]["incidents"] == 1
    assert chart["buckets"][22 - 5]["incidents"] == 1
    legacy, _ended, scheduled, trigger, native, *_phase = page["events"]
    assert (legacy["utc_ns"], legacy["time_basis"], legacy["locatable"]) == (_START + 300, "ledger", False)
    assert (scheduled["utc_ns"], scheduled["time_basis"], scheduled["details"]["last_due_utc_ns"]) == (
        _START + 17, "scheduled", _START + 19,
    )
    assert trigger["trigger_event_id"] == "trigger-5"
    assert trigger["details"]["failed_cameras"] == ["sim:1"]
    assert native["time_basis"] == "host_receipt"
    assert only_first["overview"]["frame_count"] == 2
    assert not any(row["kind"] == "native_gap_unattributed" for row in only_first["events"])
    assert only_second["overview"]["frame_count"] == 0
    assert len(only_second["overview"]["phase_marks"]) == 2


def test_event_pages_hold_snapshot_and_cli_reuses_scope(timeline_run, capsys):
    root, writer = timeline_run
    with ExperimentReader(root) as reader:
        first = reader.timeline(phase=2, limit=2)
        writer.event("error", {"phase": 2, "utc_ns": _START + 34, "code": "WRITE_FAILED", "message": "fixture"})
        before = first["older_before"]
        snapshot = first["snapshot_ordinal"]
        older = reader.timeline(phase=2, before=before, snapshot=snapshot, limit=2)
        again = reader.timeline(phase=2, after=older["newer_after"], snapshot=snapshot, limit=2)
        refreshed = reader.timeline(phase=2, limit=2)
    assert again["events"] == first["events"]
    assert "overview" not in older
    assert refreshed["events"][0]["kind"] == "error"
    assert refreshed["snapshot_ordinal"] > snapshot
    assert main(["timeline", str(root), "--phase", "2", "--before-ordinal", str(before),
                 "--snapshot-ordinal", str(snapshot), "--limit", "2"]) == 0
    assert json.loads(capsys.readouterr().out) == older


def test_incident_filter_preserves_frame_scope_and_snapshot_with_legacy_records(timeline_run, capsys):
    root, writer = timeline_run
    writer.event("event_window", {"phase": 2, "signal_id": "short-csv", "event_number": 1, "event_count": 1,
                                  "signal_utc_ns": _START + 36, "utc_ns": _START + 100, "result": "short_post",
                                  "csv_crossing": {"detected_utc_ns": _START + 35}})
    writer.event("event_window", {"phase": 2, "signal_id": "complete", "event_number": 2, "event_count": 2,
                                  "signal_utc_ns": _START + 38, "result": "complete"})
    writer.event("event_phase_summary", {"phase": 2, "event_count": 2, "completed_events": 1, "partial_events": 1,
                                          "early_signals": 2, "busy_signals": 1, "utc_ns": _START + 900})
    writer.event("csv_trigger", {"phase": 1, "state": "fault", "utc_ns": _START + 39})
    writer.event("error", {"phase": 2, "camera_id": "sim:1", "code": "WRITE_FAILED", "sequence": 3,
                           "frame_received_utc_ns": _START + 29, "utc_ns": _START + 200})
    with ExperimentReader(root) as reader:
        identity = reader.experiment["experiment_id"]
        first = reader.timeline(phase=2, incidents_only=True, limit=3)
        other_camera = reader.timeline(camera_id="sim:0", phase=2, incidents_only=True)
        csv_fault = reader.timeline(phase=1, incidents_only=True)["events"]
        assert [row["kind"] for row in csv_fault] == ["csv_trigger"]
        assert [row["details"]["signal_id"] for row in reader.timeline(event_windows_only=True)["events"]] == ["complete"]
        writer.event("error", {"phase": 2, "utc_ns": _START + 60})
        snapshot = first["snapshot_ordinal"]
        older = reader.timeline(phase=2, incidents_only=True, limit=3, before=first["older_before"], snapshot=snapshot)
        again = reader.timeline(phase=2, incidents_only=True, limit=3, after=older["newer_after"], snapshot=snapshot)
        assert again["events"] == first["events"]
        assert reader.timeline(phase=2, incidents_only=True)["overview"]["incident_count"] == 8
        with pytest.raises(CaptureError, match="only one"):
            reader.timeline(incidents_only=True, event_windows_only=True)
    review = inspect_capture_review(root, expected_id=identity)
    assert review.has_incidents and review.incident_records and not review.runtime_recorded
    chart = first["overview"]
    assert chart["incident_count"] == 7 and chart["frame_count"] == 3
    assert chart["last_utc_ns"] == _START + 42
    assert sum(bucket["incidents"] for bucket in chart["buckets"]) == 5
    write, summary, window = first["events"]
    assert write["camera_id"] == "sim:1" and write["utc_ns"] == _START + 29
    assert write["time_basis"] == "host_receipt" and write["ledger_utc_ns"] == _START + 200
    assert summary["kind"] == "event_phase_summary" and not summary["locatable"]
    assert summary["details"] == {"early_signals": 2, "busy_signals": 1}
    assert window["details"]["signal_id"] == "short-csv"
    assert not {"error", "native_gap_unattributed"} & {row["kind"] for row in other_camera["events"]}
    assert other_camera["overview"]["frame_count"] == 2
    assert main(["timeline", str(root), "--phase", "2", "--incidents-only", "--before-ordinal",
                 str(first["older_before"]), "--snapshot-ordinal", str(snapshot), "--limit", "3"]) == 0
    assert json.loads(capsys.readouterr().out) == older


def test_trigger_group_page_refuses_an_unavailable_anchor(timeline_run):
    root, writer = timeline_run
    details = {"phase": 2, "expected_cameras": ["sim:0"], "pairing_basis": "common_capture_request",
               "synchronization": "unverified"}
    writer.event("software_trigger", {**details, "retention": "save"}, event_id="older-available")
    writer.event("software_trigger", {**details, "retention": "rolling_prebuffer"}, event_id="discarded-buffer")
    with ExperimentReader(root) as reader:
        assert reader.trigger_group_page()[0]["event_id"] == "older-available"
        for identity in ("discarded-buffer", "missing-id"):
            with pytest.raises(CaptureError) as missing:
                reader.trigger_group_page(through_event=identity)
            assert missing.value.code == "TRIGGER_GROUP_UNAVAILABLE"


def test_live_timed_misses_record_projected_bounds_in_same_phase(tmp_path, monkeypatch):
    original = SimulatedCamera.trigger

    def delayed(self):
        time.sleep(.19)
        return original(self)

    monkeypatch.setattr(SimulatedCamera, "trigger", delayed)
    config = CaptureConfig.from_dict({
        "name": "Missed times", "config_version": 1, "cameras": [{"id": "sim:0"}],
        "phases": [{"mode": "timed", "fps": 20, "duration_s": .4}],
    })
    capture = CaptureSession(config, tmp_path / "slow")
    capture.start()
    assert capture.wait(10)["state"] == "completed"
    with ExperimentReader(capture.output) as reader:
        start, = reader.events(kind="phase_started")
        misses = reader.events(kind="schedule_miss")
        timeline = reader.timeline()
    assert misses
    for miss in misses:
        assert start["utc_ns"] <= miss["first_due_utc_ns"] <= miss["last_due_utc_ns"] <= miss["utc_ns"]
        assert miss["last_due_utc_ns"] - miss["first_due_utc_ns"] == (miss["last_tick"] - miss["first_tick"]) * 50_000_000
    assert sum(row["time_basis"] == "scheduled" for row in timeline["events"]) == len(misses)
