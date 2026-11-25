"""Host-clock timeline derived from committed frames and selected ledger events."""

from __future__ import annotations

import json

from .config import CaptureError
from .csv_trigger import CSV_TRIGGER_EVENT
from .image_quality_watch import NOTICE_EVENT
from .markers import MARKER_EVENT
from .storage import _SQLITE_INTEGER_MAX


_BUCKETS = 60
PHASE_EVENTS = ("phase_started", "phase_ended", "phase_confirmation_waiting",
                "phase_confirmed", "phase_confirmation_cancelled")
INCIDENTS = ("schedule_miss", "native_gap_unattributed", "native_duplicate", "native_reset_or_wrap",
             "trigger_incomplete", "hardware_identity_unavailable", "pre_window_short", "signal_unused", "error")
_PAUSE_EVENTS = ("phase_paused", "phase_resume_requested", "phase_pause_cancelled")
_EVENTS = PHASE_EVENTS + _PAUSE_EVENTS + INCIDENTS + (
    MARKER_EVENT, NOTICE_EVENT, CSV_TRIGGER_EVENT, "event_window", "event_phase_summary",
)
_RECEIPT_EVENTS = ("native_gap_unattributed", "native_duplicate", "native_reset_or_wrap")
_EVENT_TIMES = {
    "schedule_miss": ("first_due_utc_ns", "scheduled"),
    "phase_confirmation_waiting": ("waiting_since_utc_ns", "host"),
    "phase_confirmed": ("requested_utc_ns", "host"),
    "phase_paused": ("paused_utc_ns", "host"),
    "phase_resume_requested": ("requested_utc_ns", "host"),
    "phase_pause_cancelled": ("cancelled_utc_ns", "host"),
    "trigger_incomplete": ("issued_utc_ns", "trigger_request"),
    "error": ("frame_received_utc_ns", "host_receipt"),
    MARKER_EVENT: ("marked_utc_ns", "host"),
    NOTICE_EVENT: ("received_utc_ns", "host_receipt"),
    CSV_TRIGGER_EVENT: ("detected_utc_ns", "host_csv"),
    "event_window": ("signal_utc_ns", "host_signal"),
    "signal_unused": ("received_utc_ns", "host_signal"),
    **{kind: ("received_utc_ns", "host_receipt") for kind in (*_RECEIPT_EVENTS, "hardware_identity_unavailable")},
}
_DETAIL_FIELDS = {
    "schedule_miss": ("first_tick", "last_tick", "last_due_utc_ns"),
    **{kind: ("previous", "current") for kind in _RECEIPT_EVENTS},
    "hardware_identity_unavailable": ("counter_feature", "observed", "first_counter"),
    "pre_window_short": ("required_rounds", "observed_rounds", "signal_id"),
    "signal_unused": ("event_number", "source"),
    "event_phase_summary": ("early_signals", "busy_signals"),
    "phase_confirmed": ("source",),
    "phase_paused": ("pause_id", "source"),
    "phase_resume_requested": ("pause_id", "source", "pause_to_request_ns"),
    "phase_pause_cancelled": ("pause_id", "reason", "pause_duration_ns"),
    "phase_ended": ("paused_duration_ns",),
    MARKER_EVENT: ("label", "source"),
    NOTICE_EVENT: ("sequence", "pixel_format", "metrics", "compare_sharpness", "reference_samples",
                   "started", "cleared", "region", "region_changed"),
    CSV_TRIGGER_EVENT: ("state", "signal_id", "path", "value_column", "time_column", "unit", "quantity",
                        "direction", "threshold", "samples", "invalid_rows", "early_crossings", "last_value",
                        "crossing", "error", "events_accepted", "busy_crossings"),
    "event_window": ("signal_id", "event_number", "event_count", "source", "result", "pre_required_rounds",
                     "pre_retained_rounds", "boundary_rounds", "post_required_rounds", "post_received_rounds",
                     "csv_crossing", "overlap"),
    "error": ("code", "message", "sequence", "frame_received_utc_ns"),
}
_PHASE = "json_extract(record, '$.phase')"
_CAMERA = "json_extract(record, '$.camera_id')"
_CAPTURED_WINDOW = "(kind='event_window' OR (kind='csv_trigger' AND json_type(record,'$.signal_id')='text'))"
_EVENT_WINDOW_VISIBLE = """(kind!='event_window' OR (
    json_type(record,'$.event_number')='integer' AND json_type(record,'$.signal_id')='text'
    AND (json_type(record,'$.csv_crossing') IS NULL OR json_extract(record,'$.event_count')!=1
         OR json_extract(record,'$.overlap')=1)
))"""
_PARTIAL_WINDOW = """(kind='event_window' AND json_type(record,'$.event_number')='integer'
    AND json_type(record,'$.signal_id')='text' AND json_extract(record,'$.result') IN ('short_pre','short_post'))"""
_TIMED_INCIDENT = (
    f"(kind IN ({','.join(repr(kind) for kind in INCIDENTS)}) "
    "OR (kind='csv_trigger' AND json_extract(record,'$.state')='fault') "
    f"OR {_PARTIAL_WINDOW})"
)
_REJECTED_SIGNALS = """(kind='event_phase_summary' AND (
    (json_type(record,'$.early_signals')='integer' AND json_extract(record,'$.early_signals')>0)
    OR (json_type(record,'$.busy_signals')='integer' AND json_extract(record,'$.busy_signals')>0)
))"""
_INCIDENT = f"({_TIMED_INCIDENT} OR {_REJECTED_SIGNALS})"


def _utc_field(key):
    value = f"json_extract(record, '$.{key}')"
    return f"CASE WHEN json_type(record, '$.{key}')='integer' AND typeof({value})='integer' AND {value}>0 THEN {value} END"


_RECEIPT = _utc_field("timestamps.received_utc.value")
_EVENT_UTC = "COALESCE(CASE kind " + " ".join(
    f"WHEN '{kind}' THEN {_utc_field(key)}" for kind, (key, _basis) in _EVENT_TIMES.items()
) + f" END, {_utc_field('utc_ns')})"
_LOCATABLE = (f"kind!='event_phase_summary' AND "
              f"(kind!='schedule_miss' OR ({_utc_field('first_due_utc_ns')}) IS NOT NULL)")


def _valid_int(value, minimum=1, maximum=_SQLITE_INTEGER_MAX):
    return type(value) is int and minimum <= value <= maximum


def _event_row(ordinal, kind, record):
    key, basis = _EVENT_TIMES.get(kind, ("utc_ns", "ledger"))
    stamp = record.get(key)
    if not _valid_int(stamp):
        stamp, basis = record.get("utc_ns"), "ledger"
    if not _valid_int(stamp):
        stamp, basis = None, None
    detail = {field: record[field] for field in _DETAIL_FIELDS.get(kind, ()) if field in record}
    if kind == "trigger_incomplete":
        failed = [item for item in record.get("failed", ()) if item.get("camera_id")]
        detail["failed_cameras"] = [item["camera_id"] for item in failed]
        if messages := {item["camera_id"]: item["message"] for item in failed if item.get("message")}:
            detail["failure_messages"] = messages
    row = {"ordinal": ordinal, "kind": kind, "phase": record.get("phase"), "camera_id": record.get("camera_id"),
            "utc_ns": stamp, "time_basis": basis, "ledger_utc_ns": record.get("utc_ns"), "details": detail,
            "trigger_event_id": record.get("trigger_event_id") if kind == "trigger_incomplete" else None,
            "locatable": stamp is not None and kind != "event_phase_summary"
            and (kind != "schedule_miss" or basis == "scheduled")}
    if kind == MARKER_EVENT:
        row["marker_id"] = record.get("event_id")
    return row


def _event_page(db, scope, params, before, after, limit):
    seek = ""
    if before is not None:
        seek, cursor = " AND ordinal<?", (before,)
    elif after is not None:
        seek, cursor = " AND ordinal>?", (after,)
    else:
        cursor = ()
    order = "ASC" if after is not None else "DESC"
    rows = db.execute(
        f"SELECT ordinal,kind,record {scope}{seek} ORDER BY ordinal {order} LIMIT ?",
        (*params, *cursor, limit + 1),
    ).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    if after is not None:
        rows.reverse()
    older = newer = None
    if rows:
        first, last = rows[0][0], rows[-1][0]
        if after is None:
            older = last if more else None
            has_newer = before is not None and db.execute(
                f"SELECT 1 {scope} AND ordinal>? LIMIT 1", (*params, first),
            ).fetchone()
            newer = first if has_newer else None
        else:
            newer = first if more else None
            if db.execute(f"SELECT 1 {scope} AND ordinal<? LIMIT 1", (*params, last)).fetchone():
                older = last
    return {"events": [_event_row(ordinal, kind, json.loads(record)) for ordinal, kind, record in rows],
            "older_before": older, "newer_after": newer}


def _mark_pause_buckets(db, scope, params, start, width, buckets):
    bounds = db.execute(
        f"SELECT MAX(CASE WHEN kind='phase_paused' THEN {_EVENT_UTC} END), "
        f"COALESCE(MAX(CASE WHEN kind='phase_pause_cancelled' THEN {_EVENT_UTC} END), "
        f"MAX(CASE WHEN kind='phase_resume_requested' THEN {_EVENT_UTC} END)) "
        f"{scope} AND kind IN ({','.join('?' for _ in _PAUSE_EVENTS)}) "
        "AND json_type(record, '$.pause_id')='text' GROUP BY json_extract(record, '$.pause_id')",
        (*params, *_PAUSE_EVENTS),
    )
    edges = [0] * (len(buckets) + 1)
    for begin, end in bounds:
        if begin is not None:
            edges[(begin - start) // width] += 1
            last = max(begin, end - 1) if end is not None else begin
            edges[(last - start) // width + 1] -= 1
    active = 0
    for bucket, edge in zip(buckets, edges):
        active += edge
        if active:
            bucket["paused"] = True


def _overview(db, events, event_args, frame_filters, frame_args, *, include_incident_count=False):
    frames = "FROM frames" + (" WHERE " + " AND ".join(frame_filters) if frame_filters else "")
    frame_count, timed_count, frame_first, frame_last = db.execute(
        f"SELECT COUNT(*), COUNT({_RECEIPT}), MIN({_RECEIPT}), MAX({_RECEIPT}) {frames}", frame_args,
    ).fetchone()
    incident_count_sql = f", COUNT(CASE WHEN {_INCIDENT} THEN 1 END)" if include_incident_count else ""
    event_count, event_first, event_last, pause_count, *incident_count = db.execute(
        f"SELECT COUNT(*), MIN(CASE WHEN {_LOCATABLE} THEN {_EVENT_UTC} END), "
        f"MAX(CASE WHEN {_LOCATABLE} THEN {_EVENT_UTC} END), SUM(kind='phase_paused'){incident_count_sql} {events}",
        event_args,
    ).fetchone()
    totals = {"event_count": event_count, "frame_count": frame_count, "timed_frame_count": timed_count}
    if incident_count:
        totals["incident_count"] = incident_count[0]
    phase_marks = [{"phase": phase, "kind": kind, "utc_ns": stamp} for phase, kind, stamp in db.execute(
        f"SELECT {_PHASE}, kind, {_EVENT_UTC} {events} "
        f"AND kind IN ({','.join('?' for _ in PHASE_EVENTS)}) AND {_PHASE} IS NOT NULL ORDER BY ordinal",
        (*event_args, *PHASE_EVENTS),
    ) if stamp is not None]
    first = min((stamp for stamp in (frame_first, event_first) if stamp is not None), default=None)
    last = max(stamp for stamp in (frame_last, event_last) if stamp is not None) if first is not None else None
    if first is None:
        return {"first_utc_ns": None, "last_utc_ns": None, "bucket_ns": None, "buckets": [],
                "phase_marks": phase_marks, **totals}
    width = (last - first + _BUCKETS) // _BUCKETS
    buckets = [{"frames": 0, "incidents": 0, "markers": 0, "camera_counts": {}}
               for _ in range((last - first) // width + 1)]
    for camera_id, offset, count in db.execute(
        f"SELECT camera_id, (({_RECEIPT})-?)/? AS bucket, COUNT(*) {frames} GROUP BY camera_id,bucket",
        (first, width, *frame_args),
    ):
        if offset is not None:
            buckets[offset]["frames"] += count
            buckets[offset]["camera_counts"][camera_id] = count
    for marker, offset, count in db.execute(
        f"SELECT kind=? AS marker, (({_EVENT_UTC})-?)/? AS bucket, COUNT(*) {events} AND {_LOCATABLE} "
        f"AND (kind=? OR {_TIMED_INCIDENT}) GROUP BY marker,bucket",
        (MARKER_EVENT, first, width, *event_args, MARKER_EVENT),
    ):
        if offset is not None:
            buckets[offset]["markers" if marker else "incidents"] = count
    if pause_count:
        _mark_pause_buckets(db, events, event_args, first, width, buckets)
    return {"first_utc_ns": first, "last_utc_ns": last, "bucket_ns": width, "buckets": buckets,
            "phase_marks": phase_marks, **totals}


def has_timeline_incidents(db):
    return db.execute(f"SELECT 1 FROM events WHERE {_INCIDENT} LIMIT 1").fetchone() is not None


def read_timeline(db, experiment, *, camera_id, phase, before, after, snapshot, expected_experiment_id,
                  limit, markers_only=False, event_windows_only=False, incidents_only=False):
    identity = experiment["experiment_id"]
    if expected_experiment_id is not None and expected_experiment_id != identity:
        raise CaptureError("FRAME_REVIEW_CHANGED", "the selected directory now contains a different experiment")
    plan = experiment["config"]
    if camera_id is not None and not any(item["id"] == camera_id for item in plan["cameras"]):
        raise CaptureError("FRAME_REVIEW_CAMERA", "choose a camera recorded in the experiment plan")
    if phase is not None and not _valid_int(phase, maximum=len(plan["phases"])):
        raise CaptureError("INVALID_TIMELINE_QUERY", "choose a phase recorded in the experiment plan")
    if not _valid_int(limit, maximum=256):
        raise CaptureError("INVALID_TIMELINE_QUERY", "limit must be between 1 and 256")
    filters = (markers_only, event_windows_only, incidents_only)
    if any(type(flag) is not bool for flag in filters) or sum(filters) > 1:
        raise CaptureError("INVALID_TIMELINE_QUERY", "select only one valid event-list filter")
    if (before is not None and after is not None or
            any(value is not None and not _valid_int(value) for value in (before, after)) or
            snapshot is not None and not _valid_int(snapshot, minimum=0)):
        raise CaptureError("INVALID_TIMELINE_QUERY", "invalid event pagination cursor")

    head = db.execute("SELECT MAX(ordinal) FROM events").fetchone()[0] or 0
    if snapshot is not None and snapshot > head:
        raise CaptureError("FRAME_REVIEW_CHANGED", "the ledger has fewer events than the selected snapshot")
    boundary = head if snapshot is None else snapshot
    if any(value is not None and value > boundary for value in (before, after)):
        raise CaptureError("INVALID_TIMELINE_QUERY", "event cursor lies outside the selected snapshot")
    conditions = ["ordinal<=?", f"kind IN ({','.join('?' for _ in _EVENTS)})",
                  f"({_EVENT_WINDOW_VISIBLE} OR {_PARTIAL_WINDOW})",
                  f"(kind!='event_phase_summary' OR {_REJECTED_SIGNALS})"]
    params = [boundary, *_EVENTS]
    frame_filters, frame_args = [], []
    if camera_id is not None:
        conditions.append(f"({_CAMERA}=? OR {_CAMERA} IS NULL)")
        params.append(camera_id)
        frame_filters.append("camera_id=?")
        frame_args.append(camera_id)
    if phase is not None:
        conditions.append(f"{_PHASE}=?")
        params.append(phase)
        frame_filters.append(f"{_PHASE}=?")
        frame_args.append(phase)
    events = "FROM events WHERE " + " AND ".join(conditions)
    rows, row_params = events, list(params)
    if incidents_only:
        rows += " AND " + _INCIDENT
    if event_windows_only:
        rows += f" AND {_CAPTURED_WINDOW} AND {_EVENT_WINDOW_VISIBLE}"
    if markers_only:
        rows += " AND kind=?"
        row_params.append(MARKER_EVENT)
    page = _event_page(db, rows, row_params, before, after, limit)
    result = {"experiment_id": identity, "camera_id": camera_id, "phase": phase, "snapshot_ordinal": boundary,
              **page}
    if markers_only:
        result["markers_only"] = True
    if event_windows_only:
        result["event_windows_only"] = True
    if incidents_only:
        result["incidents_only"] = True
    if before is None and after is None and snapshot is None:
        result["overview"] = _overview(db, events, params, frame_filters, frame_args, include_incident_count=incidents_only)
    return result
