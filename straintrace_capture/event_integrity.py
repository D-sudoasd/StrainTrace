"""Check the saved frame membership of repeated event windows."""

from collections import Counter


def event_window_issues(frames, signals, windows, phases, cameras, *, shared_phases, shared_links):
    receipts = {row["event_id"]: row for row in signals if row.get("phase") in phases}
    captured = [row for row in windows if row.get("phase") in phases]
    counts = Counter()
    issues = []
    for frame in frames:
        primary = (frame.get("event_context_id"), frame.get("event_position"))
        links = shared_links.get(frame["ordinal"], {})
        shared = frame["phase"] in shared_phases
        if shared:
            written = frame.get("event_memberships")
            if (not isinstance(written, dict) or not written or primary not in written.items()
                    or any(context not in links or links[context] != position for context, position in written.items())):
                issues.append({"code": "EVENT_FRAME_LINK_MISMATCH", "path": frame["path"], "phase": frame["phase"]})
        elif links:
            issues.append({"code": "EVENT_FRAME_LINK_MISMATCH", "path": frame["path"], "phase": frame["phase"]})
        memberships = (primary,) if not shared and frame["phase"] in phases else links.items()
        stamp = ((frame.get("timestamps") or {}).get("received_monotonic") or {}).get("value")
        for context, position in memberships:
            receipt = receipts.get(context) if isinstance(context, str) else None
            if receipt is None or receipt["phase"] != frame["phase"] or shared != (receipt.get("overlap") is True):
                code = "EVENT_CONTEXT_UNKNOWN"
            elif not _valid_position(position, stamp, receipt.get("received_monotonic_ns")):
                code = "EVENT_FRAME_POSITION_MISMATCH"
            else:
                counts[(context, frame["camera_id"], position)] += 1
                continue
            issues.append({"code": code, "path": frame["path"], "event_id": context, "phase": frame["phase"]})

    for row in captured:
        if row.get("result") != "complete":
            continue
        context = row["signal_id"]
        for camera in cameras:
            pre, post = counts[(context, camera, "pre")], counts[(context, camera, "post")]
            expected_pre, expected_post = row["pre_required_rounds"], row["post_required_rounds"]
            boundary = counts[(context, camera, "boundary")]
            bad_boundary = row["phase"] in shared_phases and boundary != row["boundary_rounds"]
            if (pre, post) != (expected_pre, expected_post) or bad_boundary:
                issues.append({"code": "EVENT_WINDOW_FRAME_COUNT_MISMATCH", "event_id": context, "camera_id": camera,
                               "phase": row["phase"], "event_number": row["event_number"], "pre": pre, "post": post,
                               "expected_pre": expected_pre, "expected_post": expected_post,
                               **({"boundary": boundary, "expected_boundary": row["boundary_rounds"]} if bad_boundary else {})})
    return issues


def _valid_position(position, stamp, signal):
    if type(stamp) is not int or type(signal) is not int:
        return False
    if position == "pre":
        return stamp < signal
    if position == "post":
        return stamp >= signal
    return position == "boundary"
