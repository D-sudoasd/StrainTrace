import hashlib
import json
import time
from pathlib import Path

import pytest

from straintrace_capture import CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan
from straintrace_capture.cli import main
from straintrace_capture.csv_trigger import CSV_TRIGGER_EVENT, CsvTriggerWatch, inspect_csv_trigger


def append(path, data):
    with path.open("ab") as file:
        file.write(data)


def wait_for(get, timeout=6):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if result := get():
            return result
        time.sleep(.01)
    raise AssertionError("timed out waiting for the CSV watcher")


def rule(path, **kwargs):
    return {"path": str(path), "value_column": "载荷(kN)", "threshold": 5, "direction": "rising",
            "quantity": "载荷", "unit": "kN", "time_column": "时间(s)", **kwargs}


def plan(path, *, trigger=None, two=False, **stage):
    return CaptureConfig.from_dict({
        "config_version": 1, "name": "CSV 阈值试验",
        "cameras": [{"id": "sim:0"}, *([{"id": "sim:1"}] if two else [])],
        "phases": [{"mode": "event", "fps": 10, "pre_frames": 3, "post_frames": 2, "duration_s": 4,
                    "csv_trigger": trigger or rule(path), **stage}],
        "min_free_bytes": 0,
    })


def watcher(report, **values):
    status = report.status()
    return status if all(status.get(key) == value for key, value in values.items()) else None


def test_plan_checks_relative_file_and_changed_header_before_creating_output(tmp_path, capsys):
    path = tmp_path / "machine.csv"
    path.write_bytes("时间(s),载荷(kN)\n".encode())
    config = plan(path, trigger=rule(path.name, header_sha256=inspect_csv_trigger(path).sha256))
    config_path = tmp_path / "capture.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    loaded = CaptureConfig.load(config_path)
    assert loaded.phases[0].csv_trigger.path == str(path)
    assert CaptureConfig.from_dict(loaded.to_dict()) == loaded
    schema = json.loads((Path(__file__).resolve().parents[2] / "schemas" / "capture_config_v1.json").read_text())
    from jsonschema import Draft202012Validator
    Draft202012Validator(schema).validate(loaded.to_dict())
    missing = tmp_path / "missing"
    assert not any(item["code"].startswith("CSV_TRIGGER_") for item in inspect_plan(loaded, missing)["issues"])
    path.write_bytes("载荷(kN),时间(s)\n".encode())
    assert main(["run", "--config", str(config_path), "--output", str(missing)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert any(row["code"] == "CSV_TRIGGER_HEADER_CHANGED" and row["phase"] == 1 for row in result["issues"])
    assert not missing.exists()
    path.unlink()
    assert any(row["code"] == "CSV_TRIGGER_SOURCE_UNAVAILABLE" for row in inspect_plan(loaded, missing)["issues"])
    with pytest.raises(CaptureError, match="only to event"):
        CaptureConfig.from_dict({**loaded.to_dict(), "phases": [{"mode": "single", "csv_trigger": rule(path)}]})


@pytest.mark.parametrize("encoding,delimiter", [("utf-8-sig", ","), ("gb18030", ";"), ("utf-16", "\t")])
def test_new_complete_rows_and_exact_original_byte_evidence(tmp_path, encoding, delimiter):
    path = tmp_path / "machine.csv"
    line_encoding = {"utf-8-sig": "utf-8", "utf-16": "utf-16-le"}.get(encoding, encoding)
    original = (f"时间(s){delimiter}载荷(kN)\r\n0{delimiter}3\r\n1{delimiter}9\r\n2{delimiter}2").encode(encoding)
    path.write_bytes(original)
    previous = f"5{delimiter}7\r\n".encode(line_encoding)
    current = f"\"6,1\"{delimiter}4\r\n".encode(line_encoding)
    config = plan(path, trigger=rule(path, direction="falling"))
    calls = []

    def request(source, *, payload):
        calls.append((source, payload))
        return "accepted"

    with CsvTriggerWatch(config.phases[0].csv_trigger, request) as watch:
        wait_for(lambda: watcher(watch, state="watching"))
        prefix = f"0\r\n3{delimiter}8\r\n4{delimiter}bad\r\n4.5{delimiter}4\r\n".encode(line_encoding)
        append(path, prefix + previous + current[:-1])
        wait_for(lambda: watcher(watch, samples=3, invalid_rows=1))
        assert not calls
        append(path, current[-1:])
        report = wait_for(lambda: watcher(watch, state="triggered"))
    assert len(calls) == 1 and calls[0][0] == "instrument_csv_threshold"
    data = path.read_bytes()
    crossing = report["crossing"]
    assert crossing["previous"] == 7 and crossing["value"] == 4
    for key, raw, offset in (("previous_sample", previous, len(original + prefix)),
                             ("sample", current, len(original + prefix + previous))):
        row = crossing[key]
        assert row["offset"] == offset and row["length"] == len(raw)
        assert data[offset:offset + row["length"]] == raw
        assert row["sha256"] == hashlib.sha256(raw).hexdigest()
    assert crossing["sample"]["source_time_text"] == "6,1"


@pytest.mark.parametrize("rewrite", ["truncate", "replace"])
def test_change_of_source_stops_auto_trigger(tmp_path, rewrite):
    path = tmp_path / "machine.csv"
    header = "时间(s),载荷(kN)\n".encode()
    path.write_bytes(header + b"0,2\n")
    config = plan(path)
    with CsvTriggerWatch(config.phases[0].csv_trigger, lambda *_args, **_kw: pytest.fail("unexpected trigger")) as watch:
        wait_for(lambda: watcher(watch, state="watching"))
        append(path, b"1,4\n")
        wait_for(lambda: watcher(watch, samples=1))
        if rewrite == "truncate":
            path.write_bytes(header)
        else:
            replacement = path.with_suffix(".new")
            replacement.write_bytes(header + b"0,2\n1,4\n2,6\n")

            def replace():
                try:
                    replacement.replace(path)
                except PermissionError:
                    return False
                return True

            wait_for(replace)
        assert wait_for(lambda: watcher(watch, state="fault"))["error"]["code"] == "CSV_TRIGGER_SOURCE_CHANGED"


def test_two_camera_session_links_samples_with_the_exact_event_window(tmp_path):
    path = tmp_path / "machine.csv"
    original = "时间(s),载荷(kN)\n0,4\n1,9\n".encode()
    path.write_bytes(original)
    config = plan(path, two=True, pre_frames=3, fps=4)
    output = tmp_path / "experiment"
    session = CaptureSession(config, output)
    session.start()

    def progress():
        return session.status().get("phase_progress") or {}

    try:
        wait_for(lambda: (progress().get("csv_trigger") or {}).get("state") == "watching")
        append(path, b"3,4\n4,5\n")
        wait_for(lambda: (progress().get("csv_trigger") or {}).get("early_crossings") == 1)
        wait_for(lambda: (progress().get("event_window") or {}).get("state") == "waiting")
        append(path, b"5,7\n")
        wait_for(lambda: (progress().get("csv_trigger") or {}).get("samples") == 3)
        assert not progress()["event_window"]["signal_id"]
        append(path, b"6,4\n7,6\n")
        status = session.wait(6)
    finally:
        session.request_stop()
        session.wait(6)
    assert status["state"] == "completed", status
    outcome, = status["csv_triggers"]
    context = outcome["signal_id"]
    assert outcome["early_crossings"] == 1 and outcome["crossing"]["sample"]["source_time_text"] == "7"
    assert path.read_bytes() == original + b"3,4\n4,5\n5,7\n6,4\n7,6\n"
    with ExperimentReader(output) as reader:
        assert reader.experiment["summary"]["csv_triggers"] == [outcome]
        record, = reader.events(kind=CSV_TRIGGER_EVENT)
        assert record["signal_id"] == context
        signal, = reader.events(kind="signal_received")
        assert signal["event_id"] == context
        assert signal["payload"] == record["crossing"]
        assert record["detected_utc_ns"] <= signal["received_utc_ns"]
        timeline, = [row for row in reader.timeline(phase=1)["events"] if row["kind"] == CSV_TRIGGER_EVENT]
        assert timeline["time_basis"] == "host_csv" and timeline["utc_ns"] == record["detected_utc_ns"]
        for camera in ("sim:0", "sim:1"):
            page = reader.frame_page(camera_id=camera, event_context_id=context)
            assert [frame["event_position"] for frame in reversed(page["frames"])] == ["pre"] * 3 + ["post"] * 2
            assert not reader.frame_page(camera_id=camera, event_context_id="unrelated")["frames"]
        verified = reader.verify()
        assert verified["ok"], verified


def test_runtime_read_failure_leaves_manual_event_window_available(tmp_path):
    path = tmp_path / "machine.csv"
    header = "时间(s),载荷(kN)\n".encode()
    path.write_bytes(header + b"0,2\n")
    session = CaptureSession(plan(path, pre_frames=1), tmp_path / "experiment")
    session.start()

    def has_state():
        active = session.status()["phase_progress"]
        if active and active["event_window"]["state"] == "waiting":
            return active["csv_trigger"]["state"]

    try:
        wait_for(lambda: has_state() == "watching")
        path.write_bytes(header)
        wait_for(lambda: has_state() == "fault")
        context = session.request_trigger(require_prebuffer=True)
        summary = session.wait(5)
    finally:
        session.request_stop()
        session.wait(5)
    with ExperimentReader(session.output) as reader:
        assert summary["state"] == "completed"
        fault, = reader.events(kind=CSV_TRIGGER_EVENT)
        assert fault["state"] == "fault" and "signal_id" not in fault
        assert len(reader.frame_page(event_context_id=context)["frames"]) == 3
