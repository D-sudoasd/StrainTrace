import csv
import json
from hashlib import sha256

import pytest

from straintrace_capture import CaptureError
from straintrace_capture.cli import main
from straintrace_capture.pulse_audit import audit_external_pulses
from straintrace_capture.pulse_timing import ExposureLimits, add_exposure_audit

from test_capture_pulse_audit import CAMERAS, SOURCE, pulse_fixture, write_daq


EDGE_COLUMNS = ("event_number", "clock_id", "edge", "camera_id", "time_ns", "uncertainty_ns", "phase", "source_id", "channel")


def write_edges(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(EDGE_COLUMNS)
        writer.writerows(rows)
    return path


def edges(number, time_ns, *, offsets=(500, 650), bounds=(10, 20, 30), phase=1, clock="counter-A", source=SOURCE):
    trigger = (number, clock, "trigger", "", time_ns, bounds[0], phase, source, "DI0")
    exposure = [(number, clock, "exposure_start", camera, time_ns + offset, bound, phase, source, f"DI{index}")
                for index, (camera, offset, bound) in enumerate(zip(CAMERAS, offsets, bounds[1:]), 1) if offset is not None]
    return [trigger, *exposure]


@pytest.mark.parametrize("values", [(None, None, None), (None, 0, None), (None, 5, 2), (-1, None, None)])
def test_exposure_limits_require_a_complete_valid_interval_or_skew_limit(values):
    with pytest.raises(CaptureError) as exc:
        ExposureLimits(*values)
    assert exc.value.code == "PULSE_TIMING_LIMITS"


def test_one_daq_clock_reports_skew_and_delay_without_using_camera_receipt(tmp_path):
    number = 2**63 + 7
    root = pulse_fixture(tmp_path / "experiment", variations=False, start=number)
    pulses = write_daq(tmp_path / "pulses.csv", f"event_number\n{number}\n")
    trace = [*edges(number, 2**63), *edges(number + 9, 5000, phase=2, clock="other-phase"),
             *edges(number - 1, 0, clock="outside-range"), *edges(number, 0, clock="other-source", source="OTHER")]
    timing_source = write_edges(tmp_path / "timing.csv", trace)
    report = audit_external_pulses(root, pulses, phase=1, first_event=number, last_event=number)
    timed = add_exposure_audit(report, timing_source, limits=ExposureLimits(200, 470, 690))
    row, = timed["pulses"]
    result = row["exposure_timing"]
    assert report["complete"] and "exposure_timing" not in report["pulses"][0]
    assert timed["complete"] and timed["identity_complete"] and result["status"] == "within_limits"
    assert (result["skew_ns"], result["skew_bounds_ns"]) == (150, [100, 200])
    assert result["cameras"][CAMERAS[0]]["latency_bounds_ns"] == [470, 530]
    assert result["cameras"][CAMERAS[1]]["latency_bounds_ns"] == [610, 690]
    assert result["observations"][1]["channel"] == "DI1" and result["observations"][1]["line"] == 3
    assert timed["exposure_timing"]["input"]["sha256"] == sha256(timing_source.read_bytes()).hexdigest()
    assert timed["exposure_timing"]["input"]["excluded_rows"] == {"other_phase": 3, "outside_range": 3, "other_source": 3}
    borderline = add_exposure_audit(timed, timing_source, limits=ExposureLimits(199))
    assert borderline["identity_complete"] and not borderline["complete"]
    assert borderline["pulses"][0]["exposure_timing"]["status"] == "inconclusive"


def test_timing_findings_explain_missing_tiffs_and_keep_unmatched_measurements(tmp_path):
    root = pulse_fixture(tmp_path / "experiment")
    pulses = write_daq(tmp_path / "pulses.csv", "event_number\n100\n101\n102\n103\n104\n107\n")
    trace = [*edges(100, 1000), *edges(101, 2000, offsets=(300, 800)),
             *edges(102, 3000, offsets=(500, None)), *edges(103, 4000), edges(103, 4010)[1],
             *edges(105, 5000), *edges(107, 6000, bounds=(10, 20, None)), *edges(109, 7000)]
    source = write_edges(tmp_path / "timing.csv", trace)
    report = add_exposure_audit(audit_external_pulses(root, pulses, phase=1), source, limits=ExposureLimits(250))
    rows = {row["event_number"]: row for row in report["pulses"]}
    states = {number: row["exposure_timing"]["status"] for number, row in rows.items() if "exposure_timing" in row}
    timing = report["exposure_timing"]
    assert rows[100]["status"] == "missed" and states[100] == "within_limits"
    assert rows[101]["status"] == "matched" and states[101] == "outside_limits"
    assert rows[102]["exposure_timing"]["missing_cameras"] == [CAMERAS[1]]
    assert rows[103]["exposure_timing"]["duplicate_cameras"] == [CAMERAS[0]]
    assert states == {100: "within_limits", 101: "outside_limits", 102: "missing_edges", 103: "conflicting_edges",
                      104: "missing_edges", 107: "inconclusive"}
    assert not timing["within_limits"] and not report["complete"]
    assert [row["event_number"] for row in timing["unmatched"]] == [105, 109]


def test_tight_bounds_compare_distinct_cameras_and_allow_signed_latency(tmp_path):
    root = pulse_fixture(tmp_path / "experiment", variations=False)
    pulses = write_daq(tmp_path / "pulses.csv", "event_number\n101\n")
    trace = write_edges(tmp_path / "timing.csv", edges(101, 1000, offsets=(-1, 0), bounds=(2, 0, 10)))
    report = audit_external_pulses(root, pulses, phase=1)
    row, = add_exposure_audit(report, trace, limits=ExposureLimits(11, -12, 12))["pulses"]
    timing = row["exposure_timing"]
    assert timing["status"] == "within_limits" and timing["skew_bounds_ns"] == [0, 11]
    assert timing["cameras"][CAMERAS[0]]["latency_bounds_ns"] == [-3, 1]
    row, = add_exposure_audit(report, trace, limits=ExposureLimits(latency_min_ns=2, latency_max_ns=12))["pulses"]
    assert row["exposure_timing"]["status"] == "outside_limits"


@pytest.mark.parametrize(("scenario", "error"), [
    ("mixed clocks", "PULSE_TIMING_CLOCK"), ("timer reset", "PULSE_TIMING_CLOCK"),
    ("noninteger ns", "PULSE_TIMING_CSV_INVALID"), ("duplicate header", "PULSE_TIMING_CSV_INVALID"),
])
def test_invalid_or_unrelated_clock_measurements_are_rejected(tmp_path, scenario, error):
    root = pulse_fixture(tmp_path / "experiment", variations=False)
    pulses = write_daq(tmp_path / "pulses.csv", "event_number\n100\n101\n")
    report = audit_external_pulses(root, pulses, phase=1)
    trace = edges(100, 10000)
    supplied = edges(101, 11000)
    if scenario == "mixed clocks":
        supplied = edges(101, 11000, clock="counter-B")
    if scenario == "timer reset":
        supplied = edges(101, 500)
    if scenario == "noninteger ns":
        supplied = edges(101, 11000, bounds=("1e1", 20, 30))
    path = write_edges(tmp_path / "timing.csv", [*trace, *supplied])
    if scenario == "duplicate header":
        path.write_text("event_number,event_number\n100,101\n", encoding="utf-8")
    with pytest.raises(CaptureError) as exc:
        add_exposure_audit(report, path, limits=ExposureLimits(200))
    assert exc.value.code == error


def test_fewer_than_two_cameras_have_delay_evidence_only(tmp_path):
    root = pulse_fixture(tmp_path / "experiment", variations=False, cameras=(CAMERAS[0],))
    pulses = write_daq(tmp_path / "pulses.csv", "event_number\n101\n")
    trace = write_edges(tmp_path / "timing.csv", edges(101, 1000)[:2])
    report = audit_external_pulses(root, pulses, phase=1)
    with pytest.raises(CaptureError, match="至少需要两台相机"):
        add_exposure_audit(report, trace, limits=ExposureLimits(200))
    row, = add_exposure_audit(report, trace, limits=ExposureLimits(latency_min_ns=470, latency_max_ns=530))["pulses"]
    assert row["exposure_timing"]["status"] == "within_limits" and row["exposure_timing"]["skew_ns"] is None


def test_cli_integrates_timing_and_preserves_identity_with_json_and_csv(tmp_path, capsys):
    root = pulse_fixture(tmp_path / "experiment", variations=False)
    pulses = write_daq(tmp_path / "pulses.csv", "event_number\n101\n")
    trace = write_edges(tmp_path / "timing.csv", edges(101, 1000))
    args = ["audit-pulses", str(root), "--phase", "1", "--source", str(pulses), "--timing-source", str(trace)]
    report_path = tmp_path / "report.json"
    assert main([*args, "--max-skew-ns", "90", "--output", str(report_path)]) == 4
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["identity_complete"] and stdout["exposure_timing"]["outside_limits"] == 1
    full = json.loads(report_path.read_text(encoding="utf-8"))
    assert full["exposure_timing"]["limits_ns"]["max_skew_ns"] == 90
    table_path = tmp_path / "report.csv"
    assert main([*args, "--max-skew-ns", "200", "--output", str(table_path)]) == 0
    with table_path.open(encoding="utf-8-sig", newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["exposure_timing_status"] == "within_limits" and row["start_skew_max_ns"] == "200"
    assert json.loads(row["camera_timing"])[CAMERAS[1]]["latency_ns"] == 650
