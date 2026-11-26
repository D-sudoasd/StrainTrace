"""Read and summarize integer frame-attached counters without inferring a pulse ID."""

from __future__ import annotations

from .config import HARDWARE_COUNTER_FEATURES


def frame_hardware_counters(metadata):
    if not isinstance(metadata, dict):
        return {}
    return {name: value for name in HARDWARE_COUNTER_FEATURES
            if type(value := metadata.get(name)) is int and value >= 0}


def summarize_hardware_counters(rows):
    summary, previous = {}, {}
    for row in rows:
        observed = frame_hardware_counters(row.get("backend_metadata"))
        for name, value in observed.items():
            sample = {"sequence": row["sequence"], "value": value}
            state = summary.setdefault(name, {"frames": 0, "first": sample, "last": None,
                                             "duplicates": 0, "decreases": 0, "skipped_values": 0})
            state["frames"] += 1
            state["last"] = sample
            if name in previous:
                advance = value - previous[name]
                state["duplicates"] += advance == 0
                state["decreases"] += advance < 0
                state["skipped_values"] += max(advance - 1, 0)
        previous = observed
    return summary
