import csv
import json
from hashlib import sha256

import pytest

from straintrace_capture.cli import main
from straintrace_capture.config import CaptureError
from straintrace_capture.detector_link import link_audited_detector, write_detector_links
from straintrace_capture.pulse_audit import audit_external_pulses, write_pulse_audit
from straintrace_capture.pulse_binding import PulseBinding

from test_capture_pulse_audit import CAMERAS, SOURCE, pulse_fixture, write_daq
from test_capture_pulse_binding import FIRST, binding_fixture, daq_file


def _detector(path, rows, header=("event_number", "detector_frame", "phase", "source_id")):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_reconstructed_run_links_only_unambiguous_frames_and_preserves_raw_files(tmp_path):
    run, anchors = binding_fixture(tmp_path / "experiment")
    original = {path: sha256(path.read_bytes()).hexdigest() for path in run.rglob("*")
                if path.suffix == ".tiff" or path.name == "capture.sqlite3"}
    daq = daq_file(tmp_path / "daq.csv", [FIRST + value for value in (0, 1, 2, 3, 5)])
    audit = audit_external_pulses(run, daq, phase=1, binding=PulseBinding.from_dict(anchors))
    source = _detector(tmp_path / "frames.csv", [
        (FIRST + 4, "probe04.cbf", 1, SOURCE), (FIRST, "probe00.cbf", 1, SOURCE),
        (FIRST + 2, "probe02.cbf", 1, SOURCE), (FIRST + 3, "probe03.cbf", 1, SOURCE),
        (FIRST + 1, "scan.h5[1]", 1, SOURCE), (FIRST + 6, "outside.cbf", 1, SOURCE),
        (FIRST, "probe00.cbf", 1, "DAQ-B"), (FIRST, "probe00.cbf", 2, SOURCE),
    ], header=("Pulse", "Image", "phase", "source_id"))
    result = link_audited_detector(run, source, audit, event_column="Pulse", frame_column="Image")
    rows = result["links"]
    assert not result["complete"] and result["summary"]["camera_frame_links"] == 3
    assert [row["event_number"] for row in rows] == list(range(FIRST, FIRST + 6))
    assert [row["status"] for row in rows] == [
        "identity_match", "partial_capture", "conflict", "missing_cameras", "daq_unlisted", "missing_detector",
    ]
    assert rows[0]["daq_lines"] == [2] and rows[0]["detector_rows"] == [{"line": 3, "frame_id": "probe00.cbf"}]
    assert rows[2]["conflicts"] == ["duplicate_camera_frames"] and not rows[4]["daq_lines"]
    assert result["association"]["daq_unlisted_ranges"] == [{"first_event": FIRST + 4, "last_event": FIRST + 4}]
    assert result["daq_input"]["sha256"] == sha256(daq.read_bytes()).hexdigest()
    assert result["detector_input"]["sha256"] == sha256(source.read_bytes()).hexdigest()
    assert result["detector_input"]["excluded_rows"] == {"outside_range": 1, "other_source": 1, "other_phase": 1}

    csv_path = write_detector_links(result, tmp_path / "usable.csv")
    exported = _csv_rows(csv_path)
    assert [(row["event_number"], row["detector_frame"], row["camera_id"], row["camera_sequence"]) for row in exported] == [
        (str(FIRST), "probe00.cbf", CAMERAS[0], "2"), (str(FIRST), "probe00.cbf", CAMERAS[1], "2"),
        (str(FIRST + 1), "scan.h5[1]", CAMERAS[1], "3"),
    ]
    assert [row["status"] for row in exported] == ["identity_match", "identity_match", "partial_capture"]
    assert all((run / row["camera_path"]).is_file() for row in exported)
    with pytest.raises(FileExistsError):
        write_detector_links(result, csv_path)
    assert _csv_rows(csv_path) == exported
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in original.items())


def test_duplicate_detector_rows_and_reused_frame_outside_range_cannot_export(tmp_path):
    run = pulse_fixture(tmp_path / "experiment", variations=False)
    daq = write_daq(tmp_path / "daq.csv", "event_number\n101\n102\n103\n")
    audit = audit_external_pulses(run, daq, phase=1, first_event=101, last_event=103)
    detector = _detector(tmp_path / "frames.csv", [
        (101, "reused.cbf", 1, SOURCE), (107, "reused.cbf", 1, SOURCE),
        (102, "second.cbf", 1, SOURCE), (102, "duplicate.cbf", 1, SOURCE),
    ])
    result = link_audited_detector(run, detector, audit)
    assert [(row["status"], row["conflicts"]) for row in result["links"]] == [
        ("conflict", ["reused_detector_frames"]), ("conflict", ["duplicate_detector_pulses"]), ("missing_both", []),
    ]
    assert result["links"][0]["reused_detector_frames"] == ["reused.cbf"]
    assert result["links"][1]["detector_rows"] == [{"line": 4, "frame_id": "second.cbf"}, {"line": 5, "frame_id": "duplicate.cbf"}]
    assert not _csv_rows(write_detector_links(result, tmp_path / "empty.csv"))


def test_cli_rechecks_audit_scope_and_source_bytes_and_keeps_legacy_mode(tmp_path, capsys):
    run = pulse_fixture(tmp_path / "experiment", variations=False)
    daq = write_daq(tmp_path / "daq.csv", "event_number\n101\n")
    audit = audit_external_pulses(run, daq, phase=1)
    detector = _detector(tmp_path / "frames.csv", [(101, "scan.cbf", 1, SOURCE)])
    # The exported result rows are not used as evidence for the new links.
    audit["pulses"][0]["frames"] = []
    report = write_pulse_audit(audit, tmp_path / "pulse-audit.json")
    relocated = daq.rename(tmp_path / "relocated.csv")
    output = tmp_path / "links.json"
    args = ["link-detector", str(run), "--source", str(detector)]
    audited = [*args, "--pulse-report", str(report), "--daq-source", str(relocated)]
    assert main([*audited, "--output", str(output)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["complete"] and receipt["camera_frame_links"] == 2
    assert result["schema"] == "straintrace.capture-detector-link/v2" and result["daq_input"]["path"] == str(relocated.resolve())
    assert {frame["camera_id"] for frame in result["links"][0]["capture"]["frames"]} == set(CAMERAS)
    foreign = pulse_fixture(tmp_path / "foreign", variations=False)
    with pytest.raises(CaptureError, match="不一致") as exc:
        link_audited_detector(foreign, detector, audit, daq_source=relocated)
    assert exc.value.code == "DETECTOR_AUDIT_SCOPE"

    missing = tmp_path / "bad.json"
    relocated.write_bytes(b"different file")
    assert main([*audited, "--output", str(missing)]) == 3
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "PULSE_CSV_SOURCE_CHANGED"
    assert not missing.exists()

    legacy = tmp_path / "legacy.json"
    assert main([*args, "--trigger-source", SOURCE, "--output", str(legacy)]) == 0
    assert json.loads(legacy.read_text(encoding="utf-8"))["schema"] == "straintrace.capture-detector-link/v1"


def test_audited_export_publishes_only_complete_output_files(tmp_path):
    target = tmp_path / "links.json"
    with pytest.raises(TypeError):
        write_detector_links({"bad_input": object()}, target)
    assert list(tmp_path.iterdir()) == []
