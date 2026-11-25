"""Link independently numbered detector frames to a rechecked DAQ and camera audit."""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import CaptureError
from .handoff import write_snapshot
from .pulse_audit import PULSE_AUDIT_SCHEMA, _csv_reader, _in_range, _scoped_rows, audit_external_pulses, pulse_integer
from .pulse_binding import PulseBinding
from .storage import _publish_path


_SCHEMA = "straintrace.capture-detector-link/v2"
_CSV_ERROR = "DETECTOR_CSV_INVALID"
_JOINED = {"identity_match", "partial_capture"}
_STATUSES = ("identity_match", "partial_capture", "missing_cameras", "missing_detector", "missing_both", "daq_unlisted", "conflict")


def _recheck_audit(root, audit, daq_source):
    if not isinstance(audit, dict) or audit.get("schema") != PULSE_AUDIT_SCHEMA:
        raise CaptureError("DETECTOR_AUDIT_INVALID", "请选择“核对 DAQ 脉冲”导出的 JSON 报告。")
    try:
        audit_input = audit["input"]
        scope = {key: audit[key] for key in ("phase", "first_event", "last_event")}
        identity = {key: audit[key] for key in ("experiment_id", "source_id", "pairing")}
        source_path = audit_input["path"] if daq_source is None else daq_source
        expected_sha, event_column = audit_input["sha256"], audit_input["event_column"]
        association = audit.get("association")
        binding = PulseBinding.from_dict(association["binding"]) if association is not None else None
        anchors = association["anchors"] if association is not None else None
    except (KeyError, TypeError) as exc:
        raise CaptureError("DETECTOR_AUDIT_INVALID", "核对报告缺少 DAQ 来源、阶段范围或锚点记录。") from exc
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise CaptureError("DETECTOR_AUDIT_INVALID", "核对报告缺少 DAQ 文件的 SHA-256 校验值。")

    current = audit_external_pulses(root, source_path, event_column=event_column, binding=binding,
                                   expected_source_sha256=expected_sha, **scope)
    if any(current[key] != value for key, value in (identity | scope).items()) or (
        binding is not None and current["association"]["anchors"] != anchors
    ):
        raise CaptureError("DETECTOR_AUDIT_SCOPE", "核对报告与当前实验、阶段、信号源或锚点原图不一致。")
    return current


def _detector_table(source, audit, event_column, frame_column):
    path = Path(source).expanduser().resolve()
    data = path.read_bytes()
    pulses, frame_counts, excluded = defaultdict(list), Counter(), Counter()
    first, last = audit["first_event"], audit["last_event"]
    try:
        with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="") as stream:
            rows = _csv_reader(stream, code=_CSV_ERROR)
            fields = rows.fieldnames
            selected = {event_column, frame_column}
            if len(selected) != 2 or selected & {"source_id", "phase"} or not selected.issubset(fields):
                raise CaptureError("DETECTOR_COLUMNS_MISSING", "请为脉冲编号和探测器帧标识分别选择两列。")
            for line, row in _scoped_rows(rows, audit["phase"], audit["source_id"], excluded, code=_CSV_ERROR):
                number, frame = pulse_integer(row[event_column]), row[frame_column].strip()
                if number is None or not frame:
                    raise CaptureError(_CSV_ERROR, f"第 {line} 行需要非负整数脉冲编号和非空的探测器帧标识。")
                frame_counts[frame] += 1
                if not _in_range(number, first, last):
                    excluded["outside_range"] += 1
                    continue
                pulses[number].append({"line": line, "frame_id": frame})
    except (UnicodeError, csv.Error) as exc:
        raise CaptureError(_CSV_ERROR, "请使用 UTF-8 CSV，并检查文本引号和列数。") from exc

    evidence = {"path": str(path), "sha256": sha256(data).hexdigest(), "event_column": event_column, "frame_column": frame_column,
                "source_column": "source_id" if "source_id" in fields else None, "phase_column": "phase" if "phase" in fields else None,
                "included_rows": sum(map(len, pulses.values())), "excluded_rows": dict(excluded)}
    return pulses, {name for name, count in frame_counts.items() if count > 1}, evidence


def _link_row(number, capture, detector, reused):
    daq_lines = capture["daq_lines"] if capture else []
    conflicts = list(capture["conflicts"]) if capture else []
    repeated_frames = sorted({row["frame_id"] for row in detector} & reused)
    if len(detector) > 1:
        conflicts.append("duplicate_detector_pulses")
    if repeated_frames:
        conflicts.append("reused_detector_frames")

    if conflicts:
        status = "conflict"
    elif not daq_lines:
        status = "daq_unlisted"
    elif not detector:
        status = "missing_both" if capture["status"] == "missed" else "missing_detector"
    else:
        status = {"matched": "identity_match", "partial": "partial_capture", "missed": "missing_cameras"}[capture["status"]]
    return {"event_number": number, "status": status, "daq_lines": daq_lines, "detector_rows": detector,
            "capture": capture, "conflicts": conflicts, "reused_detector_frames": repeated_frames}


def link_audited_detector(root, source, pulse_audit, *, event_column="event_number", frame_column="detector_frame", daq_source=None):
    """Rebuild associations from the run and source DAQ CSV, then join exact integer pulse IDs."""
    audit = _recheck_audit(root, pulse_audit, daq_source)
    detector, reused, evidence = _detector_table(source, audit, event_column, frame_column)
    camera = {row["event_number"]: row for row in audit["pulses"]}
    rows = [_link_row(pulse, camera.get(pulse), detector.get(pulse, []), reused) for pulse in sorted(camera.keys() | detector.keys())]
    counts = Counter(row["status"] for row in rows)
    summary = {status: counts[status] for status in _STATUSES}
    summary["camera_frame_links"] = sum(len(row["capture"]["frames"]) for row in rows if row["status"] in _JOINED)
    result = {"schema": _SCHEMA, "complete": audit["complete"] and counts["identity_match"] == len(rows),
              "experiment_id": audit["experiment_id"], "phase": audit["phase"], "source_id": audit["source_id"],
              "first_event": audit["first_event"], "last_event": audit["last_event"], "pairing": audit["pairing"],
              "daq_input": audit["input"], "detector_input": evidence, "capture_summary": audit["summary"], "summary": summary, "links": rows}
    if "association" in audit:
        result["association"] = audit["association"]
    return result


def write_detector_links(report, output):
    """Write the full JSON or a long-form CSV of unambiguous frame links."""
    output = Path(output).expanduser().resolve()
    suffix = output.suffix.lower()
    if suffix not in (".json", ".csv"):
        raise CaptureError("DETECTOR_OUTPUT_FORMAT", "请以 .json 或 .csv 保存探测器关联。")
    with TemporaryDirectory(prefix=".detector-link-", dir=output.parent) as directory:
        staged = Path(directory) / output.name
        if suffix == ".json":
            write_snapshot(report, staged)
        else:
            _write_frame_links(report, staged)
        _publish_path(staged, output, write_through=False)
    return output


def _write_frame_links(report, path):
    scope = report["experiment_id"], report["phase"], report["source_id"]
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("experiment_id", "phase", "source_id", "event_number", "status", "daq_csv_line", "detector_csv_line",
                         "detector_frame", "camera_id", "camera_frame_id", "camera_sequence", "camera_path"))
        for row in report["links"]:
            if row["status"] not in _JOINED:
                continue
            detector = row["detector_rows"][0]
            pulse = row["event_number"], row["status"], row["daq_lines"][0], detector["line"], detector["frame_id"]
            for frame in row["capture"]["frames"]:
                writer.writerow((*scope, *pulse, frame["camera_id"], frame["frame_id"], frame["sequence"], frame["path"]))
