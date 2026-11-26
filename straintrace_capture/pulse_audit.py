"""Reconcile a finished hardware-trigger phase against independent DAQ pulse IDs."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from .config import CaptureError, positive_integer
from .pulse_binding import PulseBinding, counter_associations, daq_unlisted_ranges
from .storage import ExperimentReader, TERMINAL_STATES


PULSE_AUDIT_SCHEMA = "straintrace.capture-pulse-audit/v1"


def pulse_integer(text: str) -> int | None:
    """Parse decimal identifiers without a floating-point conversion."""
    digits = text.strip()
    if not digits.isascii() or not digits.isdecimal():
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _phases(experiment, *, include_unpaired=False):
    if experiment["state"] not in TERMINAL_STATES:
        raise CaptureError("PULSE_AUDIT_ACTIVE", "采集停止并保存后才能核对外部脉冲。")
    return {number: phase.get("hardware_pairing") for number, phase in enumerate(experiment["config"]["phases"], 1)
            if phase["mode"] == "hardware" and (include_unpaired or phase.get("hardware_pairing"))}


def pulse_audit_phases(root):
    with ExperimentReader(root) as reader:
        return _phases(reader.experiment)


def pulse_audit_context(root):
    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        return experiment["experiment_id"], _phases(experiment, include_unpaired=True), tuple(
            row["id"] for row in experiment["config"]["cameras"]
        )


def _csv_reader(stream, *, code="PULSE_CSV_INVALID"):
    reader = csv.DictReader(stream, strict=True)
    fields = reader.fieldnames
    if not fields or any(not name.strip() for name in fields) or len(set(fields)) != len(fields):
        raise CaptureError(code, "CSV 首行需要非空且不可重复的列名。")
    return reader


def pulse_csv_columns(path, *, code="PULSE_CSV_INVALID"):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return tuple(_csv_reader(stream, code=code).fieldnames)
    except (UnicodeError, csv.Error) as exc:
        raise CaptureError(code, str(exc)) from exc


def _in_range(value, first, last):
    return (first is None or first <= value) and (last is None or value <= last)


def _scoped_rows(reader, phase, source_id, excluded, *, code="PULSE_CSV_INVALID"):
    for row in reader:
        line = reader.line_num
        if None in row or any(value is None for value in row.values()):
            raise CaptureError(code, f"第 {line} 行的列数与表头不一致。")
        if "source_id" in row:
            if not (row_source := row["source_id"].strip()):
                raise CaptureError(code, f"第 {line} 行的信号源为空。")
            if row_source != source_id:
                excluded["other_source"] += 1
                continue
        if "phase" in row:
            value = pulse_integer(row["phase"])
            if value is None or value < 1:
                raise CaptureError(code, f"第 {line} 行的阶段编号须为正整数。")
            if value != phase:
                excluded["other_phase"] += 1
                continue
        yield line, row


def _pulse_table(path, column, phase, source_id, first, last, *, anchors=(), expected_sha256=None):
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise CaptureError("PULSE_CSV_SOURCE_CHANGED", "DAQ 文件内容与原核对报告不同。请选用原文件，或重新核对。")
    pulses = defaultdict(list)
    excluded = Counter()
    anchor_lines = {number: [] for number in anchors}
    try:
        with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="") as stream:
            reader = _csv_reader(stream)
            fields = reader.fieldnames
            if column not in fields or column in ("source_id", "phase"):
                raise CaptureError("PULSE_CSV_EVENT_COLUMN", f"请选择脉冲编号列。当前指定为 {column!r}。")
            for line, row in _scoped_rows(reader, phase, source_id, excluded):
                number = pulse_integer(row[column])
                if number is None:
                    raise CaptureError("PULSE_CSV_INVALID", f"第 {line} 行的脉冲编号须为非负整数。")
                if number in anchor_lines:
                    anchor_lines[number].append(line)
                if not _in_range(number, first, last):
                    excluded["outside_range"] += 1
                    continue
                pulses[number].append(line)
    except (UnicodeError, csv.Error) as exc:
        raise CaptureError("PULSE_CSV_INVALID", str(exc)) from exc
    for number, lines in anchor_lines.items():
        if len(lines) != 1:
            raise CaptureError("PULSE_BINDING_DAQ", f"补配锚点 {number} 在所选 DAQ 信号源及阶段内须有一条记录，实际 {len(lines)} 条。")
    if not pulses:
        raise CaptureError("PULSE_CSV_EMPTY_SCOPE", "CSV 中没有符合所选阶段、信号源和编号范围的脉冲。")
    evidence = {"path": str(path), "sha256": digest, "event_column": column,
                "phase_column": "phase" if "phase" in fields else None,
                "source_column": "source_id" if "source_id" in fields else None,
                "included_rows": sum(map(len, pulses.values())), "excluded_rows": dict(excluded)}
    if anchor_lines:
        evidence["anchor_csv_lines"] = {str(number): lines[0] for number, lines in anchor_lines.items()}
    return pulses, evidence


def _events_by_number(events, pairing):
    numbers, by_id, invalid_events = defaultdict(list), {}, []
    for event in events:
        if event["kind"] != "hardware_trigger":
            continue
        identity, number = event.get("event_id"), event.get("external_event_number")
        cameras = event.get("expected_cameras")
        counters = event.get("first_cam_counter")
        if (not identity or not event["ledger_consistent"] or type(number) is not int or number < pairing["first_event_number"]
            or event.get("external_source_id") != pairing["source_id"]
            or event.get("pairing_basis") != "explicit_shared_counter_anchor"
            or event.get("counter_feature") != pairing["counter_feature"]
            or counters != pairing["first_counter"] or any(type(value) is not int for value in counters.values())
            or not isinstance(cameras, list) or sorted(cameras, key=str) != sorted(pairing["first_counter"])):
            invalid_events.append({"event_id": identity, "event_number": number, "source_id": event.get("external_source_id")})
            continue
        numbers[number].append(identity)
        by_id[identity] = number
    return numbers, by_id, invalid_events


def _pulse_row(number, csv_lines, identities, frames, pairing):
    expected = pairing["first_counter"]
    offset = number - pairing["first_event_number"]
    valid = Counter()
    all_cameras = Counter()
    invalid_frames = []
    for frame in frames:
        camera = frame["camera_id"]
        all_cameras[camera] += 1
        if camera in expected and frame["ledger_consistent"] and type(frame["counter"]) is int and frame["counter"] == expected[camera] + offset:
            valid[camera] += 1
        else:
            invalid_frames.append(frame["frame_id"])
    missing = sorted(expected.keys() - valid.keys())
    duplicates = sorted(camera for camera, count in all_cameras.items() if count > 1)
    unexpected = sorted(all_cameras.keys() - expected.keys())
    conflicts = [code for condition, code in (
        (len(csv_lines) > 1, "duplicate_daq_rows"), (len(identities) > 1, "duplicate_capture_events"),
        (duplicates, "duplicate_camera_frames"), (unexpected, "unexpected_cameras"), (invalid_frames, "invalid_camera_counters"),
    ) if condition]
    if conflicts:
        status = "conflict"
    elif not csv_lines:
        status = "capture_only"
    elif not frames:
        status = "missed"
    elif missing:
        status = "partial"
    else:
        status = "matched"
    return {"event_number": number, "status": status, "daq_lines": csv_lines, "capture_event_ids": identities,
            "frames": frames, "missing_cameras": missing, "duplicate_cameras": duplicates,
            "unexpected_cameras": unexpected, "invalid_frame_ids": invalid_frames, "conflicts": conflicts}


def audit_external_pulses(root, source, *, phase, event_column="event_number", first_event=None, last_event=None,
                          binding: PulseBinding | None = None, expected_source_sha256=None):
    """Compare exact DAQ pulse IDs with phase-local ledger events and recorded counters."""
    phase = positive_integer(phase, "phase")
    for key, value in (("first_event", first_event), ("last_event", last_event)):
        if value is not None:
            positive_integer(value, key, minimum=0)
    if first_event is not None and last_event is not None and first_event > last_event:
        raise CaptureError("PULSE_AUDIT_RANGE", "起始编号须小于或等于结束编号。")

    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        phases = _phases(experiment, include_unpaired=True)
        if phase not in phases:
            raise CaptureError("PULSE_AUDIT_PHASE", f"阶段 {phase} 不是硬件触发阶段。")
        pairing = phases[phase]
        if binding is not None and pairing:
            raise CaptureError("PULSE_BINDING_SCOPE", "阶段已有采集时登记的锚点，请用原台账核对。")
        if binding is None and not pairing:
            raise CaptureError("PULSE_AUDIT_PHASE", f"阶段 {phase} 没有带计数器锚点的硬件触发记录。")
        association = None
        if binding is not None:
            binding.check_context(experiment, phase)
            first_event, last_event = binding.audit_range(first_event, last_event)
            pulses, csv_evidence = _pulse_table(source, event_column, phase, binding.source_id, first_event, last_event,
                                               anchors=(binding.first_event_number, binding.last_event_number), expected_sha256=expected_source_sha256)
            experiment, events, frames = reader.hardware_phase_evidence(phase, binding.counter_feature, include_hash=True)
            pairing, groups, association, unassociated = counter_associations(reader, binding, events, frames)
            numbers, invalid_events = {}, []
        else:
            pulses, csv_evidence = _pulse_table(source, event_column, phase, pairing["source_id"], first_event, last_event,
                                               expected_sha256=expected_source_sha256)
            experiment, events, frames = reader.hardware_phase_evidence(phase, pairing["counter_feature"])
            numbers, by_id, invalid_events = _events_by_number(events, pairing)
            groups = defaultdict(list)
            unassociated = []
            for frame in frames:
                number = by_id.get(frame["event_id"])
                if frame["phase"] != phase or number is None:
                    unassociated.append({**frame, "reason": "wrong_phase" if frame["phase"] != phase else
                                         "invalid_event" if frame["event_id"] else "no_event"})
                else:
                    groups[number].append(frame)
    captured = numbers.keys() | groups.keys()
    selected = {key for key in captured if _in_range(key, first_event, last_event)}
    rows = [_pulse_row(number, pulses.get(number, []), numbers.get(number, []), groups.get(number, []), pairing)
            for number in sorted(pulses.keys() | selected)]
    execution = {event["kind"] for event in events}
    counts = Counter(row["status"] for row in rows)
    summary = {key: counts[key] for key in ("matched", "missed", "partial", "conflict", "capture_only")}
    summary.update(expected_pulses=len(pulses), unassociated_frames=len(unassociated), invalid_events=len(invalid_events),
                   capture_pulses_outside_range=len(captured) - len(selected), not_in_daq=sum(not row["daq_lines"] for row in rows))
    complete = summary["matched"] == len(rows) and not unassociated and not invalid_events
    report = {"schema": PULSE_AUDIT_SCHEMA, "complete": complete, "experiment_id": experiment["experiment_id"],
              "experiment_state": experiment["state"], "phase": phase, "source_id": pairing["source_id"],
              "phase_execution": ("ended" if "phase_ended" in execution else "interrupted" if execution or frames else "not_started"),
              "pairing": pairing, "first_event": first_event, "last_event": last_event, "input": csv_evidence,
              "summary": summary, "pulses": rows, "unassociated_frames": unassociated, "invalid_events": invalid_events}
    if association is not None:
        gaps = association["daq_unlisted_ranges"] = daq_unlisted_ranges(pulses, first_event, last_event)
        summary["outside_anchor_frames"] = sum(sum(counts.values()) for counts in association["outside_frames"].values())
        summary["daq_unlisted_pulses"] = sum(gap["last_event"] - gap["first_event"] + 1 for gap in gaps)
        report.update(association=association, complete=complete and not gaps)
    return report


def write_pulse_csv(report, path):
    """Write the pulse table. Array fields keep IDs and line numbers unambiguous."""
    from .output import resolve_output_path
    from .report_files import new_text_output

    path = resolve_output_path(path)
    columns = ("event_number", "status", "daq_lines", "capture_event_ids", "frame_ids", "missing_cameras",
               "duplicate_cameras", "unexpected_cameras", "invalid_frame_ids", "conflicts")
    timing_columns = (("exposure_timing_status", "start_skew_ns", "start_skew_min_ns", "start_skew_max_ns", "camera_timing")
                      if "exposure_timing" in report else ())
    binding_columns = ("association_basis", "first_event", "last_event", "daq_unlisted_pulses") if "association" in report else ()
    with new_text_output(path, encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((*columns, *timing_columns, *binding_columns))
        for row in report["pulses"]:
            frame_ids = [frame["frame_id"] for frame in row["frames"]]
            cells = [row["event_number"], row["status"], *(
                json.dumps(frame_ids if key == "frame_ids" else row[key], ensure_ascii=False) for key in columns[2:])]
            if timing := row.get("exposure_timing"):
                cells.extend((timing["status"], timing["skew_ns"], *(timing["skew_bounds_ns"] or (None, None)),
                              json.dumps(timing["cameras"], ensure_ascii=False)))
            elif timing_columns:
                cells.extend([None] * len(timing_columns))
            if binding_columns:
                cells.extend((report["association"]["basis"], report["first_event"], report["last_event"], report["summary"]["daq_unlisted_pulses"]))
            writer.writerow(cells)
    return path


def write_pulse_audit(report, path):
    from .handoff import write_snapshot

    path = Path(path)
    if path.suffix.lower() == ".csv":
        return write_pulse_csv(report, path)
    if path.suffix.lower() == ".json":
        return write_snapshot(report, path)
    raise CaptureError("PULSE_AUDIT_OUTPUT_FORMAT", "请以 .json 或 .csv 保存核对结果。")
