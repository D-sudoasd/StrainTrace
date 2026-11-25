"""Use frame write evidence to recover sample identities after a writer exits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import closing
from functools import cache
from pathlib import Path

from .config import CaptureError
from .storage import (
    FRAME_WRITE_JOURNAL, FRAME_WRITE_SCHEMA, TERMINAL_STATES, ExperimentReader, PreparedFrame,
    TriggerEvent, WriterLease, _connection, _frame_target, _insert_event, _insert_frames, _part_target,
    _publish_staged_tiff, frame_image_records, json_text, sha256_file,
)


def _read_journal(reader, experiment, report, *, write_through):
    declaration = experiment.get("frame_write_journal")
    if declaration is None:
        report["journal"] = "not_recorded"
        return
    if declaration != {"schema": FRAME_WRITE_SCHEMA, "path": FRAME_WRITE_JOURNAL}:
        report["journal"] = "unavailable"
        report["issues"].append({"code": "FRAME_JOURNAL_UNSUPPORTED"})
        return
    path = reader.root / FRAME_WRITE_JOURNAL
    try:
        if path.resolve() != path:
            raise OSError("the frame journal is linked to another path")
        with path.open("r+b" if write_through else "rb") as stream:
            if write_through:
                # The previous process may have stopped before syncing its batch.
                os.fsync(stream.fileno())
            report["journal"] = "present"
            for number, line in enumerate(stream, 1):
                try:
                    digest, separator, payload = line[:-1].partition(b" ")
                    if not line.endswith(b"\n") or not separator or hashlib.sha256(payload).hexdigest().encode("ascii") != digest:
                        raise ValueError("descriptor checksum is missing or differs")
                    yield number, json.loads(payload)
                except (UnicodeError, ValueError) as exc:
                    report["issues"].append({"code": "FRAME_JOURNAL_ENTRY_INVALID", "line": number, "message": str(exc)})
    except FileNotFoundError:
        report["journal"] = "missing"
        report["issues"].append({"code": "FRAME_JOURNAL_MISSING", "path": FRAME_WRITE_JOURNAL})
    except OSError as exc:
        report["journal"] = "unavailable"
        report["issues"].append({"code": "FRAME_JOURNAL_UNREADABLE", "message": str(exc)})


def frame_write_records_need_review(reader: ExperimentReader, experiment: dict) -> bool:
    """Check whether failed or interrupted writes need inspection without opening image files."""
    if experiment["state"] not in ("interrupted", "failed"):
        return False
    evidence = {"issues": []}
    for _line, packet in _read_journal(reader, experiment, evidence, write_through=False):
        try:
            if packet["experiment_id"] != experiment["experiment_id"]:
                return True
            frame = packet["frame"]
            camera, sequence = frame["camera_id"], frame["sequence"]
            if not isinstance(camera, str) or not camera or type(sequence) is not int or not 1 <= sequence < 2**63:
                return True
            committed = reader.frame_at_sequence(camera, sequence)
        except (CaptureError, KeyError, TypeError, ValueError):
            return True
        if committed is None:
            return True
        committed.pop("ordinal")
        if committed != frame:
            return True
    return bool(evidence["issues"])


def _descriptor(reader, experiment, cameras, packet, received_signal):
    if not isinstance(packet, dict) or packet.get("experiment_id") != experiment["experiment_id"]:
        raise ValueError("descriptor belongs to a different experiment")
    frame = packet["frame"]
    if not isinstance(frame, dict):
        raise ValueError("invalid frame record")
    camera, sequence, phase = frame["camera_id"], frame["sequence"], frame["phase"]
    phases = experiment["config"]["phases"]
    if type(sequence) is not int or not 1 <= sequence < 2**63 or type(phase) is not int or not 1 <= phase <= len(phases):
        raise ValueError("invalid sequence or phase")
    camera_path = reader.root / cameras[camera]["path"]
    target = _frame_target(camera_path, sequence, experiment["config"].get("segment_frames", 0))
    if (frame["frame_id"] != f"{experiment['experiment_id']}/{camera}/{sequence}"
        or frame["role"] != phases[phase - 1]["role"]):
        raise ValueError("frame identity or role differs from acquisition")
    if "auxiliary_images" in frame and not isinstance(frame["auxiliary_images"], list):
        raise ValueError("invalid auxiliary image descriptors")
    for index, image in enumerate(frame_image_records(frame), 1):
        asset = target if index == 1 else _part_target(target, index)
        if image["path"] != asset.relative_to(reader.root).as_posix() or reader.image_path(image) != asset:
            raise ValueError("image part path differs from acquisition")
        size, digest = image["file_bytes"], image["sha256"]
        if type(size) is not int or size <= 0 or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("invalid TIFF size or checksum")
    for name in ("host_utc", "host_monotonic"):
        stamp_key = "received_utc" if name == "host_utc" else "received_monotonic"
        stamp = frame["timestamps"][stamp_key]
        if stamp["clock_id"] != experiment["clocks"][name]["clock_id"] or type(stamp["value"]) is not int:
            raise ValueError("frame receipt is not bound to the experiment clock")

    candidate = packet["trigger"]
    trigger = None
    if candidate is not None:
        trigger = TriggerEvent(**candidate)
        if (not isinstance(trigger.id, str) or not trigger.id
            or trigger.kind not in ("software_trigger", "hardware_trigger")
            or not isinstance(trigger.details, dict) or trigger.details.get("phase") != phase
            or not all(isinstance(trigger.details.get(key), str) for key in ("pairing_basis", "synchronization"))
            or camera not in trigger.details.get("expected_cameras", ())):
            raise ValueError("invalid trigger association")
    if frame["trigger_event_id"] != (trigger.id if trigger is not None else None):
        raise ValueError("frame and trigger identifiers differ")
    if phases[phase - 1].get("overlap_events"):
        positions = frame.get("event_memberships")
        if (not isinstance(positions, dict) or not positions
                or (frame.get("event_context_id"), frame.get("event_position")) not in positions.items()):
            raise ValueError("invalid shared event membership")
        for context, position in positions.items():
            signal = received_signal(context)
            if (position not in ("pre", "boundary", "post") or signal is None
                    or signal.get("phase") != phase or signal.get("overlap") is not True):
                raise ValueError("shared event membership does not refer to a signal from this phase")
    elif "event_memberships" in frame:
        raise ValueError("shared event membership is outside an overlapping event phase")
    return PreparedFrame(frame, trigger)


def _frame_keys(frame):
    return frame["path"], (frame["camera_id"], frame["sequence"])


def _previous_rows(db, frame, planned):
    records = [json.loads(row[0]) for row in db.execute(
        "SELECT record FROM frames WHERE path=? OR (camera_id=? AND sequence=?)",
        (frame["path"], frame["camera_id"], frame["sequence"]),
    )]
    for key in _frame_keys(frame):
        if key in planned and planned[key] not in records:
            records.append(planned[key])
    return records


def _trigger_conflicts(db, trigger, planned):
    if trigger is None:
        return False
    existing = db.execute("SELECT record FROM events WHERE event_id=?", (trigger.id,)).fetchone()
    event = json.loads(existing[0]) if existing is not None else planned.get(trigger.id)
    return event is not None and (event["kind"] != trigger.kind or
                                  any(event.get(key) != value for key, value in trigger.details.items()))


def _membership_conflicts(db, frame):
    positions = frame.get("event_memberships")
    if positions:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_frame_links'").fetchone():
            return True
        return any(db.execute("SELECT 1 FROM event_frame_links WHERE event_id=? AND camera_id=? AND sequence=?",
                              (context, frame["camera_id"], frame["sequence"])).fetchone() for context in positions)
    return False


def _verify_payload(source, frame):
    if source.resolve() != source:
        raise CaptureError("FRAME_PAYLOAD_PATH_CONFLICT", "image payload is linked to another path")
    try:
        size = source.stat().st_size
    except FileNotFoundError as exc:
        raise CaptureError("FRAME_PAYLOAD_MISSING", "no TIFF or partial file for this descriptor") from exc
    if size != frame["file_bytes"] or sha256_file(source) != frame["sha256"]:
        raise CaptureError("FRAME_PAYLOAD_MISMATCH", "payload size or checksum differs from the recorded TIFF")


def _scan_frames(db, reader, experiment, report, *, restore):
    cameras = {entry["device"]["id"]: entry for entry in reader.cameras()}
    write_through = experiment["config"].get("durability") == "write_through"
    planned_frames, planned_events = {}, {}
    received_signal = cache(reader.received_signal)
    for line, packet in _read_journal(reader, experiment, report, write_through=restore and write_through):
        try:
            prepared = _descriptor(reader, experiment, cameras, packet, received_signal)
        except (CaptureError, KeyError, TypeError, ValueError, OSError) as exc:
            report["issues"].append({"code": "FRAME_JOURNAL_ENTRY_INVALID", "line": line, "message": str(exc)})
            continue
        frame = prepared.record
        entry = {"path": frame["path"], "frame_id": frame["frame_id"], "line": line}
        previous = _previous_rows(db, frame, planned_frames)
        if previous == [frame]:
            key = "duplicate_pending" if frame["path"] in planned_frames else "already_committed"
            report[key] += 1
            continue
        if previous or _trigger_conflicts(db, prepared.trigger, planned_events) or _membership_conflicts(db, frame):
            report["issues"].append({"code": "FRAME_LEDGER_CONFLICT", **entry})
            continue
        sources = []
        try:
            for image in frame_image_records(frame):
                target = reader.root / image["path"]
                source = target
                if not target.exists():
                    source = target.with_suffix(".tiff.part")
                _verify_payload(source, image)
                sources.append((source, target))
            if restore:
                for source, target in sources:
                    if source != target:
                        _publish_staged_tiff(source, target, write_through=write_through)
        except (CaptureError, OSError) as exc:
            code = exc.code if isinstance(exc, CaptureError) else "FRAME_PAYLOAD_UNAVAILABLE"
            report["issues"].append({"code": code, **entry, "image_path": target.relative_to(reader.root).as_posix(),
                                     "source_path": source.relative_to(reader.root).as_posix(),
                                     "message": str(exc)})
            continue
        if restore:
            _insert_frames(db, (prepared,))
        else:
            for key in _frame_keys(frame):
                planned_frames[key] = frame
            if trigger := prepared.trigger:
                planned_events.setdefault(trigger.id, {"kind": trigger.kind, "event_id": trigger.id, **trigger.details})
        report["recovered" if restore else "recoverable"].append({
            **entry, "source_path": sources[0][0].relative_to(reader.root).as_posix(), "sha256": frame["sha256"],
        })


def _recovery_evidence(*, restore):
    evidence = {"journal": "not_applicable", "already_committed": 0, "issues": []}
    if restore:
        evidence["recovered"] = []
    else:
        evidence.update(recoverable=[], duplicate_pending=0)
    return evidence


def _can_scan(state):
    return state not in TERMINAL_STATES or state in ("interrupted", "failed")


def inspect_recovery(root: str | Path, *, expected_experiment_id: str | None = None) -> dict:
    """Inspect pending write evidence under the existing writer lease without updating the run."""
    root = Path(root).resolve()
    with WriterLease(root, create=False), ExperimentReader(root) as reader, closing(_connection(root, "ro")) as db:
        experiment, counts = reader.committed_state()
        if expected_experiment_id is not None and experiment["experiment_id"] != expected_experiment_id:
            raise CaptureError("RECOVERY_SCOPE_CHANGED", "实验目录中的标识已变化，请重新检查中断记录。")
        state = experiment["state"]
        evidence = _recovery_evidence(restore=False)
        if _can_scan(state):
            _scan_frames(db, reader, experiment, evidence, restore=False)
        return {
            "schema": "straintrace.capture-recovery-check/v1", "experiment_id": experiment["experiment_id"],
            "name": experiment["config"]["name"], "state": state, "needs_finalize": state not in TERMINAL_STATES,
            "capture_error": (experiment.get("summary") or {}).get("error"), "cameras": counts,
            "committed_frames": sum(count["frames"] for count in counts.values()),
            "committed_images": sum(count["images"] for count in counts.values()), "recovery": evidence,
        }


def recover(root: str | Path, *, expected_experiment_id: str | None = None) -> dict:
    """Finalize an interrupted ledger and admit only payloads with verified write evidence."""
    root = Path(root).resolve()
    with WriterLease(root), ExperimentReader(root) as reader:
        experiment = reader.experiment
        if expected_experiment_id is not None and experiment["experiment_id"] != expected_experiment_id:
            raise CaptureError("RECOVERY_SCOPE_CHANGED", "实验目录中的标识已变化，请重新检查中断记录。")
        prior = experiment["state"]
        interrupted = prior not in TERMINAL_STATES
        evidence = _recovery_evidence(restore=True)
        if _can_scan(prior):
            with closing(_connection(root, "rw")) as db, db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("PRAGMA foreign_keys=ON")
                _scan_frames(db, reader, experiment, evidence, restore=True)
                if interrupted:
                    experiment.update(state="interrupted", recovered_utc_ns=time.time_ns())
                    db.execute("UPDATE experiment SET record=? WHERE singleton=1", (json_text(experiment),))
                if interrupted or evidence["recovered"]:
                    _insert_event(db, "recovery", {"previous_state": prior, **evidence})
        result = reader.verify()
        result["recovery"] = evidence
        if evidence["issues"]:
            result["ok"] = False
        return result
