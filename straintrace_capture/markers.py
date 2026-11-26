"""Operator markers and their neighboring, committed camera frames."""

from __future__ import annotations

import json

from .config import CaptureError


MARKER_EVENT = "operator_marker"
MARKER_LABEL_LIMIT = 160
_RECEIPT_PATH = "$.timestamps.received_monotonic"


def marker_label(value):
    if not isinstance(value, str):
        raise CaptureError("MARKER_LABEL_INVALID", "marker label must be text")
    label = " ".join(value.split())
    if len(label) > MARKER_LABEL_LIMIT or any(ord(char) < 32 for char in label):
        raise CaptureError("MARKER_LABEL_INVALID", f"enter at most {MARKER_LABEL_LIMIT} characters of text")
    return label


def read_marker_context(db, experiment, marker_id, frame_at_sequence):
    """Find the nearest saved receipt on each side on the same Capture host clock."""
    entry = db.execute("SELECT record FROM events WHERE kind=? AND event_id=?", (MARKER_EVENT, marker_id)).fetchone()
    if entry is None:
        raise CaptureError("MARKER_NOT_FOUND", f"no saved marker {marker_id!r}")
    marker = json.loads(entry[0])
    clock = (experiment.get("clocks") or {}).get("host_monotonic", {}).get("clock_id")
    marked = marker.get("marked_monotonic_ns")
    utc = marker.get("marked_utc_ns")
    if (not clock or marker.get("monotonic_clock_id") != clock or marker.get("event_id") != marker_id
            or type(marked) is not int or not 0 <= marked < 1 << 63
            or type(utc) is not int or not 0 < utc < 1 << 63):
        raise CaptureError("MARKER_CLOCK_UNAVAILABLE", "the marker lacks a comparable Capture host clock")
    by_camera = {row["id"]: {"before": None, "after": None} for row in experiment["config"]["cameras"]}
    stamp = f"json_extract(record, '{_RECEIPT_PATH}.value')"
    matches = db.execute(
        f"""SELECT camera_id, sequence, ordinal, {stamp} FROM frames
            WHERE json_extract(record, '{_RECEIPT_PATH}.clock_id') = ?
              AND json_extract(record, '{_RECEIPT_PATH}.unit') = 'ns'
              AND json_extract(record, '{_RECEIPT_PATH}.source') = 'host_receive'
              AND json_type(record, '{_RECEIPT_PATH}.value') = 'integer'
              AND typeof({stamp}) = 'integer' AND {stamp} >= 0""", (clock,),
    )
    for camera, sequence, ordinal, received in matches:
        if camera not in by_camera:
            continue
        side = "before" if received <= marked else "after"
        candidate = (abs(received - marked), ordinal, sequence, received)
        if by_camera[camera][side] is None or candidate < by_camera[camera][side]:
            by_camera[camera][side] = candidate
    neighbors = {}
    for camera, sides in by_camera.items():
        neighbors[camera] = {"before": None, "after": None}
        for side, selected in sides.items():
            if selected is None:
                continue
            _, ordinal, sequence, received = selected
            row = frame_at_sequence(camera, sequence)
            if row is None or row["ordinal"] != ordinal:
                raise CaptureError("FRAME_REVIEW_CHANGED", f"the saved frame {camera}/{sequence} has changed")
            neighbors[camera][side] = {
                "camera_id": camera, "ordinal": ordinal, "sequence": sequence,
                "frame_id": row["frame_id"], "sha256": row["sha256"], "phase": row["phase"],
                "received_utc_ns": (row["timestamps"].get("received_utc") or {}).get("value"),
                "received_monotonic_ns": received, "offset_ns": received - marked,
            }
    return {"experiment_id": experiment["experiment_id"], "marker": marker,
            "time_basis": "host_receive_monotonic", "cameras": neighbors}
