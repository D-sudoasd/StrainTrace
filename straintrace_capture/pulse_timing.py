"""Audit independently measured trigger and exposure edges on one DAQ clock."""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .config import CaptureError
from .pulse_audit import _csv_reader, _in_range, _scoped_rows, pulse_integer


_EDGE_COLUMNS = ("event_number", "clock_id", "edge", "camera_id", "time_ns", "uncertainty_ns")
_INVALID = "PULSE_TIMING_CSV_INVALID"
_STATUSES = ("within_limits", "outside_limits", "inconclusive", "missing_edges", "conflicting_edges")


@dataclass(frozen=True)
class ExposureLimits:
    max_skew_ns: int | None = None
    latency_min_ns: int | None = None
    latency_max_ns: int | None = None

    def __post_init__(self):
        if self.max_skew_ns is not None and (type(self.max_skew_ns) is not int or self.max_skew_ns < 0):
            raise CaptureError("PULSE_TIMING_LIMITS", "相机间曝光起始偏差上限须为非负数。")
        low, high = self.latency_min_ns, self.latency_max_ns
        if (low is not None or high is not None) and (type(low) is not int or type(high) is not int or low > high):
            raise CaptureError("PULSE_TIMING_LIMITS", "请同时填写延迟下限和上限，且下限须小于或等于上限。")
        if self.max_skew_ns is None and low is None:
            raise CaptureError("PULSE_TIMING_LIMITS", "请填写相机间曝光起始偏差上限或触发至曝光的延迟范围。")


def _edge(row, line):
    clock_id, kind, camera = (row[key].strip() for key in ("clock_id", "edge", "camera_id"))
    if not clock_id or kind not in ("trigger", "exposure_start") or bool(camera) != (kind == "exposure_start"):
        raise CaptureError(_INVALID, f"第 {line} 行须填写 clock_id。trigger 行留空 camera_id，exposure_start 行须填写 camera_id。")
    time_ns = pulse_integer(row["time_ns"])
    bound = row["uncertainty_ns"].strip()
    uncertainty = pulse_integer(bound) if bound else None
    if time_ns is None or (bound and uncertainty is None):
        raise CaptureError(_INVALID, f"第 {line} 行的 time_ns 和已知的 uncertainty_ns 须为非负整数纳秒。")
    return {"line": line, "edge": kind, "camera_id": camera or None, "clock_id": clock_id,
            "time_ns": time_ns, "uncertainty_ns": uncertainty, "channel": row.get("channel", "").strip() or None}


def _read_edges(source, pulse_report):
    path = Path(source).expanduser().resolve()
    raw = path.read_bytes()
    excluded = Counter()
    groups = defaultdict(list)
    clocks = set()
    try:
        with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="") as stream:
            reader = _csv_reader(stream, code=_INVALID)
            missing = [key for key in _EDGE_COLUMNS if key not in reader.fieldnames]
            if missing:
                raise CaptureError(_INVALID, f"时序 CSV 缺少列 {', '.join(missing)}。")
            for line, row in _scoped_rows(reader, pulse_report["phase"], pulse_report["source_id"], excluded, code=_INVALID):
                number = pulse_integer(row["event_number"])
                if number is None:
                    raise CaptureError(_INVALID, f"第 {line} 行的 event_number 须为非负整数。")
                if not _in_range(number, pulse_report["first_event"], pulse_report["last_event"]):
                    excluded["outside_range"] += 1
                    continue
                edge = _edge(row, line)
                groups[number].append(edge)
                clocks.add(edge["clock_id"])
    except (UnicodeError, csv.Error) as exc:
        raise CaptureError(_INVALID, str(exc)) from exc
    if len(clocks) > 1:
        raise CaptureError("PULSE_TIMING_CLOCK", "时序 CSV 在当前范围内有多个 clock_id。请核对各路信号是否来自同一时基。")
    previous = None
    for number in sorted(groups):
        triggers = [edge for edge in groups[number] if edge["edge"] == "trigger"]
        if len(triggers) != 1:
            continue
        current = triggers[0]
        if previous is not None and current["time_ns"] <= previous["time_ns"]:
            raise CaptureError("PULSE_TIMING_CLOCK", f"触发时刻未随脉冲编号递增。请核对 CSV 第 {previous['line']} 和 {current['line']} 行的计时器及编号。")
        previous = current
    evidence = {"path": str(path), "sha256": sha256(raw).hexdigest(), "clock_id": next(iter(clocks), None),
                "included_rows": sum(map(len, groups.values())), "excluded_rows": dict(excluded)}
    return groups, evidence


def _interval_status(bounds, low, high):
    if high is None:
        return "not_checked"
    if bounds is None:
        return "unknown"
    first, last = bounds
    if first >= low and last <= high:
        return "within"
    return "outside" if last < low or first > high else "overlaps_limit"


def _latency(trigger, exposure, limits):
    value = bounds = None
    if trigger is not None:
        value = exposure["time_ns"] - trigger["time_ns"]
        if trigger["uncertainty_ns"] is not None and exposure["uncertainty_ns"] is not None:
            width = trigger["uncertainty_ns"] + exposure["uncertainty_ns"]
            bounds = [value - width, value + width]
    return {"exposure_start_ns": exposure["time_ns"], "uncertainty_ns": exposure["uncertainty_ns"],
            "latency_ns": value, "latency_bounds_ns": bounds, "line": exposure["line"],
            "latency_status": _interval_status(bounds, limits.latency_min_ns, limits.latency_max_ns)}


def _skew(exposures):
    times = [edge["time_ns"] for edge in exposures]
    nominal = max(times) - min(times)
    if any(edge["uncertainty_ns"] is None for edge in exposures):
        return nominal, None
    intervals = sorted((edge["time_ns"] - edge["uncertainty_ns"], edge["time_ns"] + edge["uncertainty_ns"])
                       for edge in exposures)
    # Upper bound compares distinct cameras, even when one interval contains all others.
    first_low, first_high = intervals[0]
    widest = max(max(high for _, high in intervals[1:]) - first_low, first_high - intervals[1][0])
    closest = max(0, intervals[-1][0] - min(high for _, high in intervals))
    return nominal, [closest, widest]


def _judge(statuses, conflicts, missing):
    if conflicts:
        return "conflicting_edges"
    if missing:
        return "missing_edges"
    if "outside" in statuses:
        return "outside_limits"
    if "unknown" in statuses or "overlaps_limit" in statuses:
        return "inconclusive"
    return "within_limits"


def _audit_edges(observations, cameras, limits):
    triggers, exposures = [], defaultdict(list)
    for edge in observations:
        if edge["edge"] == "trigger":
            triggers.append(edge)
        else:
            exposures[edge["camera_id"]].append(edge)
    trigger = triggers[0] if len(triggers) == 1 else None
    missing = sorted(cameras - exposures.keys())
    extras = sorted(exposures.keys() - cameras)
    duplicates = sorted(camera for camera, readings in exposures.items() if len(readings) > 1)
    chosen = {camera: exposures[camera][0] for camera in sorted(cameras & exposures.keys()) if len(exposures[camera]) == 1}
    metrics = {camera: _latency(trigger, edge, limits) for camera, edge in chosen.items()}
    skew, skew_bounds = _skew(list(chosen.values())) if len(chosen) == len(cameras) and len(chosen) > 1 else (None, None)
    skew_status = _interval_status(skew_bounds, 0, limits.max_skew_ns)
    checks = {entry["latency_status"] for entry in metrics.values()} | {skew_status}
    status = _judge(checks, len(triggers) > 1 or extras or duplicates, not triggers or missing)
    return {"status": status, "skew_ns": skew, "skew_bounds_ns": skew_bounds, "skew_status": skew_status,
            "trigger_count": len(triggers), "missing_cameras": missing, "unexpected_cameras": extras,
            "duplicate_cameras": duplicates, "cameras": metrics, "observations": observations}


def add_exposure_audit(pulse_report, source, *, limits: ExposureLimits):
    """Attach a timing audit without changing the pulse or frame association results."""
    cameras = set(pulse_report["pairing"]["first_counter"])
    if len(cameras) < 2 and limits.max_skew_ns is not None:
        raise CaptureError("PULSE_TIMING_LIMITS", "相机间偏差至少需要两台相机。单相机可检查触发至曝光的延迟。")
    groups, evidence = _read_edges(source, pulse_report)
    rows, counts = [], Counter()
    for pulse in pulse_report["pulses"]:
        if not pulse["daq_lines"]:
            rows.append(pulse)
            continue
        timing = _audit_edges(groups.pop(pulse["event_number"], []), cameras, limits)
        counts[timing["status"]] += 1
        rows.append({**pulse, "exposure_timing": timing})
    unmatched = [{"event_number": number, "observations": groups[number]} for number in sorted(groups)]
    summary = {key: counts[key] for key in _STATUSES}
    passed = counts["within_limits"] == pulse_report["summary"]["expected_pulses"] and not unmatched
    identity_complete = pulse_report.get("identity_complete", pulse_report["complete"])
    return {**pulse_report, "pulses": rows, "identity_complete": identity_complete, "complete": identity_complete and passed,
            "exposure_timing": {"within_limits": passed, "limits_ns": asdict(limits), "input": evidence,
                                "summary": {**summary, "unmatched_timing_pulses": len(unmatched)}, "unmatched": unmatched}}
