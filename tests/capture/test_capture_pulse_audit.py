import csv
import json
from hashlib import sha256

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, CaptureError
from straintrace_capture.cameras.base import Device, Image, receipt_times
from straintrace_capture.cli import main
from straintrace_capture.pulse_audit import audit_external_pulses, pulse_audit_phases
from straintrace_capture.storage import ExperimentWriter, TriggerEvent


CAMERAS = ("sim:left", "sim:right")
SOURCE = "DAQ-A"


def pulse_fixture(root, *, variations=True, active=False, start=101, cameras=CAMERAS):
    anchors = {camera: 2**63 + 31 + index * 59 for index, camera in enumerate(cameras)}
    pairing = {"source_id": SOURCE, "counter_feature": "ChunkCounterValue", "first_event_number": start,
               "first_counter": anchors}
    phase = {"mode": "hardware", "hardware_source": "Line1", "hardware_pairing": pairing}
    plan = CaptureConfig.from_dict({"config_version": 1, "name": "Independent pulse fixture",
                                   "cameras": [{"id": name} for name in cameras], "phases": [phase, phase]})
    writer = ExperimentWriter(root, plan)
    counts = dict.fromkeys(cameras, 0)

    def record(camera, number, stage=1, *, counter_offset=0, event_name=None):
        counts[camera] += 1
        stamp, mono = receipt_times()
        trigger = None
        metadata = {}
        if number is not None:
            metadata["ChunkCounterValue"] = anchors[camera] + number - start + counter_offset
            identity = event_name or f"{writer.experiment_id}:phase-{stage}:ttl-{number}"
            trigger = TriggerEvent(identity, "hardware_trigger", {
                "phase": stage, "expected_cameras": list(cameras), "external_source_id": SOURCE,
                "external_event_number": number, "pairing_basis": "explicit_shared_counter_anchor",
                "synchronization": "hardware_association_clock_unverified", "counter_feature": "ChunkCounterValue",
                "first_cam_counter": anchors,
            })
        image = Image(np.full((20, 28), counts[camera], dtype=np.uint8), "Mono8", 8, "mono", stamp, mono, metadata=metadata)
        return writer.prepare_frame(camera, counts[camera], stage, "test", image, trigger, None, None, 0)

    try:
        for name in cameras:
            writer.register_camera(Device(name, "sim", "Fixture", name, name, "simulated"), {})
        writer.event("phase_started", {"phase": 1})
        frames = [record(name, start) for name in cameras]
        if variations:
            left, right = CAMERAS
            frames += [record(left, start + 1), record(left, start + 2), record(left, start + 2),
                       record(right, start + 2, counter_offset=1), record(left, start + 4), record(right, start + 4),
                       record(right, None), record(right, start + 4, event_name="another-identity")]
        writer.commit_frames(frames)
        if not active:
            writer.event("phase_ended", {"phase": 1})
            writer.event("phase_started", {"phase": 2})
            writer.commit_frames([record(name, start + 6, stage=2) for name in cameras])
            writer.event("phase_ended", {"phase": 2})
            writer.set_state("completed")
    finally:
        writer.close()
    return root


def write_daq(path, text):
    path.write_text(text, encoding="utf-8-sig")
    return path


def test_independent_daq_finds_first_and_last_missing_pulses_and_frame_conflicts(tmp_path):
    root = pulse_fixture(tmp_path / "experiment")
    csv = write_daq(tmp_path / "pulses.csv", "event_number,source_id,phase\n100,DAQ-A,1\n101,DAQ-A,1\n"
                    "102,DAQ-A,1\n103,DAQ-A,1\n107,DAQ-A,1\n999,DAQ-B,1\n108,DAQ-A,2\n")
    report = audit_external_pulses(root, csv, phase=1)
    assert not report["complete"] and report["phase_execution"] == "ended"
    assert report["input"]["sha256"] == sha256(csv.read_bytes()).hexdigest()
    assert report["input"]["excluded_rows"] == {"other_source": 1, "other_phase": 1}
    rows = {row["event_number"]: row for row in report["pulses"]}
    assert {key: row["status"] for key, row in rows.items()} == {
        100: "missed", 101: "matched", 102: "partial", 103: "conflict", 105: "conflict", 107: "missed",
    }
    assert rows[100]["missing_cameras"] == rows[107]["missing_cameras"] == list(CAMERAS)
    assert rows[102]["missing_cameras"] == [CAMERAS[1]]
    assert rows[103]["duplicate_cameras"] == [CAMERAS[0]]
    assert rows[103]["missing_cameras"] == [CAMERAS[1]] and len(rows[103]["invalid_frame_ids"]) == 1
    assert rows[105]["conflicts"] == ["duplicate_capture_events", "duplicate_camera_frames"]
    assert rows[101]["frames"][0]["counter"] == 2**63 + 31
    assert len(report["unassociated_frames"]) == 1 and report["unassociated_frames"][0]["reason"] == "no_event"
    assert pulse_audit_phases(root).keys() == {1, 2}


def test_range_filter_and_duplicate_csv_ids_are_explicit(tmp_path):
    root = pulse_fixture(tmp_path / "experiment", variations=False)
    source = write_daq(tmp_path / "custom.csv", "Pulse\n100\n101\n107\n")
    audit = audit_external_pulses(root, source, phase=1, event_column="Pulse", first_event=101, last_event=106)
    assert audit["complete"] and audit["summary"]["matched"] == 1
    assert audit["input"]["excluded_rows"] == {"outside_range": 2}
    write_daq(source, "Pulse\n101\n101\n")
    audit = audit_external_pulses(root, source, phase=1, event_column="Pulse")
    row, = audit["pulses"]
    assert not audit["complete"] and row["status"] == "conflict" and row["daq_lines"] == [2, 3]
    assert row["conflicts"] == ["duplicate_daq_rows"]


def test_capture_only_is_reported_when_the_daq_does_not_list_the_pulse(tmp_path):
    root = pulse_fixture(tmp_path / "experiment", variations=False)
    source = write_daq(tmp_path / "pulse.csv", "event_number\n100\n")
    audit = audit_external_pulses(root, source, phase=1)
    assert [(row["event_number"], row["status"]) for row in audit["pulses"]] == [(100, "missed"), (101, "capture_only")]
    limited = audit_external_pulses(root, source, phase=1, last_event=100)
    assert len(limited["pulses"]) == 1 and limited["summary"]["capture_pulses_outside_range"] == 1


def test_long_daq_ids_stay_exact_when_adjacent_rows_differ_by_one(tmp_path):
    number = 2**63 + 7
    root = pulse_fixture(tmp_path / "experiment", variations=False, start=number)
    source = write_daq(tmp_path / "long-ids.csv", f"event_number\n{number - 1}\n{number}\n{number + 1}\n")
    report = audit_external_pulses(root, source, phase=1)
    assert [(row["event_number"], row["status"]) for row in report["pulses"]] == [
        (number - 1, "missed"), (number, "matched"), (number + 1, "missed"),
    ]


@pytest.mark.parametrize(("values", "code"), [
    ("event_number,event_number\n101,101\n", "PULSE_CSV_INVALID"),
    ("event_number\n101,wrong\n", "PULSE_CSV_INVALID"),
    ("event_number\n1.25\n", "PULSE_CSV_INVALID"),
    ("event_number\n1e2\n", "PULSE_CSV_INVALID"),
    ("event_number,source_id\n101,\n", "PULSE_CSV_INVALID"),
    ("event_number,source_id\n101,OTHER\n", "PULSE_CSV_EMPTY_SCOPE"),
    ("unit,time\nHz,1100\n", "PULSE_CSV_EVENT_COLUMN"),
])
def test_no_green_report_for_invalid_or_empty_daq_input(tmp_path, values, code):
    root = pulse_fixture(tmp_path / "experiment", variations=False)
    path = write_daq(tmp_path / "pulses.csv", values)
    with pytest.raises(CaptureError) as err:
        audit_external_pulses(root, path, phase=1)
    assert err.value.code == code


def test_cli_writes_json_and_spreadsheet_ready_csv_without_overwriting(tmp_path, capsys):
    root = pulse_fixture(tmp_path / "experiment", variations=False)
    source = write_daq(tmp_path / "pulses.csv", "event_number\n100\n101\n")
    args = ["audit-pulses", str(root), "--source", str(source), "--phase", "1", "--output"]
    json_path = tmp_path / "audit.json"
    assert main([*args, str(json_path)]) == 4
    printed = json.loads(capsys.readouterr().out)
    assert printed["missed"] == 1 and printed["matched"] == 1
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["phase"] == 1 and len(report["pulses"]) == 2
    csv_path = tmp_path / "audit.csv"
    assert main([*args, str(csv_path), "--first-event", "101"]) == 0
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["event_number"] == "101" and row["status"] == "matched"
    assert len(json.loads(row["frame_ids"])) == 2
    assert main([*args, str(source)]) != 0
    assert source.read_text(encoding="utf-8-sig") == "event_number\n100\n101\n"


def test_only_finished_runs_with_explicit_phase_anchors_can_be_audited(tmp_path):
    source = write_daq(tmp_path / "pulses.csv", "event_number\n101\n")
    root = pulse_fixture(tmp_path / "opening", active=True)
    with pytest.raises(CaptureError) as err:
        audit_external_pulses(root, source, phase=1)
    assert err.value.code == "PULSE_AUDIT_ACTIVE"
    root = pulse_fixture(tmp_path / "ended", variations=False)
    with pytest.raises(CaptureError) as err:
        audit_external_pulses(root, source, phase=3)
    assert err.value.code == "PULSE_AUDIT_PHASE"
