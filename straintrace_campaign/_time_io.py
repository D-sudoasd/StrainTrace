"""Shared stdlib-only CSV, timestamp, clock-domain, and load-table utilities.

Used by ``alignment.py``, ``adapters/desy.py``, and ``workflow.preview_alignment``
so CLI, GUI, and Python API apply identical parsing and clock-compatibility rules
without pulling any GUI, NumPy, or vision dependencies into headless imports.
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import CampaignError


ABSOLUTE_EPOCH_MIN = 1_000_000_000.0
EXACT_TIME_TOL_S = 2e-6
DERIVED_TIME_STATUS_WORDS = ("ESTIMAT", "RECONSTRUCT", "INFER", "PROTOCOL", "ADOPTED")
_MISSING_TEXT = {"", "nan", "none", "na", "n/a", "null", "<no-data>"}
_TRUE_TEXT = {"true", "1", "yes", "y"}
_FALSE_TEXT = {"false", "0", "no", "n"}
FORCE_UNIT_SCALES = {
    "n": 1.0,
    "newton": 1.0,
    "newtons": 1.0,
    "kn": 1000.0,
    "kilonewton": 1000.0,
    "kilonewtons": 1000.0,
}


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def strict_integer(value: Any) -> int | None:
    result = finite_number(value)
    if result is None:
        return None
    rounded = round(result)
    if abs(result - rounded) > 1e-8:
        return None
    return int(rounded)


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().casefold()
    if text in _TRUE_TEXT:
        return True
    if text in _FALSE_TEXT:
        return False
    return default


def sniff_csv_delimiter(sample: str, delimiters: str = ",;\t") -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=delimiters).delimiter
    except csv.Error:
        return ","


def read_csv_header(path: str | os.PathLike[str]) -> list[str]:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise CampaignError("LOAD_SOURCE_NOT_FOUND", f"load source does not exist: {file_path}")
    with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        delimiter = sniff_csv_delimiter(sample)
        return next(csv.reader(stream, delimiter=delimiter), [])


def force_unit_scale(unit_text: str | None) -> float | None:
    normalized = str(unit_text or "").strip().casefold().replace(" ", "")
    return FORCE_UNIT_SCALES.get(normalized)


def resolve_force_unit(
    row: Mapping[str, Any], force_column: str, force_unit: str | None = None
) -> tuple[str | None, float | None]:
    unit = force_unit
    if not unit:
        for unit_key in ("force_unit", "load_unit", "unit"):
            candidate_unit = row.get(unit_key)
            if candidate_unit is not None and str(candidate_unit).strip():
                unit = str(candidate_unit).strip()
                break
    if not unit and force_column == "force_N":
        unit = "N"
    return unit, force_unit_scale(unit)


def classify_time_value(value: Any) -> dict[str, Any]:
    """Classify a standalone timestamp without converting relative clocks to Unix time."""
    number = finite_number(value)
    if number is not None:
        if number >= ABSOLUTE_EPOCH_MIN:
            return {
                "timestamp_unix_s": number,
                "time_s": None,
                "timestamp_status": "RECORDED_ABSOLUTE_EPOCH",
                "timestamp_timezone": "UTC_EPOCH",
            }
        return {
            "timestamp_unix_s": None,
            "time_s": number,
            "timestamp_status": "RECORDED_RELATIVE_CLOCK",
            "timestamp_timezone": None,
        }
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() in _MISSING_TEXT:
        return {
            "timestamp_unix_s": None,
            "time_s": None,
            "timestamp_status": "MISSING_TIMESTAMP",
            "timestamp_timezone": None,
        }
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return {
            "timestamp_unix_s": None,
            "time_s": None,
            "timestamp_status": "UNPARSEABLE_TIMESTAMP",
            "timestamp_timezone": None,
            "timestamp_raw": text,
        }
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return {
            "timestamp_unix_s": None,
            "time_s": None,
            "timestamp_status": "NAIVE_DATETIME_TIMEZONE_UNKNOWN",
            "timestamp_timezone": None,
            "timestamp_raw": text,
        }
    return {
        "timestamp_unix_s": parsed.timestamp(),
        "time_s": None,
        "timestamp_status": "RECORDED_ABSOLUTE_TIMEZONE_AWARE",
        "timestamp_timezone": str(parsed.tzinfo),
    }


def parse_row_time(
    field: str, row: Mapping[str, Any], *, numeric_domain: str | None = None,
    numeric_scale: float = 1., default_clock_id: str | None = None,
) -> tuple[float | None, str, str | None, str | None]:
    """Return ``(value, domain, clock_id, reason)`` for the selected row time field."""
    if field not in row or row.get(field) is None or str(row.get(field)).strip() == "":
        if field == "timestamp_unix_s" and row.get("estimated_timestamp_unix_s") not in (None, ""):
            return None, "missing", None, "RECORDED_TIMESTAMP_MISSING_ESTIMATE_SEPARATE"
        return None, "missing", None, "TIME_VALUE_MISSING"
    status = str(row.get("timestamp_status") or row.get("time_status") or "").upper()
    is_estimated = field.startswith("estimated_")
    if not is_estimated and any(word in status for word in DERIVED_TIME_STATUS_WORDS):
        return None, "missing", None, "TIME_FIELD_MARKED_DERIVED"

    raw = row.get(field)
    number = finite_number(raw)
    clock_id = row.get("clock_id") or row.get("timestamp_clock_id") or row.get("time_clock_id")
    clock_id = str(clock_id).strip() if clock_id not in (None, "") else None
    if clock_id == "":
        clock_id = None
    if number is not None:
        number *= numeric_scale
        if not math.isfinite(number):
            return None, "missing", clock_id, "TIME_VALUE_OUTSIDE_RANGE"
        if numeric_domain == "relative":
            if is_estimated:
                return None, "missing", clock_id, "TIME_FIELD_MARKED_DERIVED"
            return number, "relative", clock_id or default_clock_id, None
        if numeric_domain == "absolute" or field.endswith("unix_s"):
            if number < ABSOLUTE_EPOCH_MIN:
                return None, "missing", clock_id, "UNIX_FIELD_VALUE_OUTSIDE_EPOCH_RANGE"
            return number, "unix_epoch_estimated" if is_estimated else "unix_epoch_recorded", clock_id, None
        if field == "time_s" or field.endswith("_time_s"):
            if not clock_id:
                return None, "relative", None, "RELATIVE_CLOCK_ID_REQUIRED"
            return number, "relative", clock_id, None
        if number >= ABSOLUTE_EPOCH_MIN:
            return number, "unix_epoch_estimated" if is_estimated else "unix_epoch_recorded", clock_id, None
        if not clock_id:
            return None, "relative", None, "RELATIVE_CLOCK_ID_REQUIRED"
        return number, "relative", clock_id, None

    if numeric_domain == "relative":
        return None, "missing", clock_id, "RELATIVE_TIME_MUST_BE_NUMERIC"
    text = str(raw).strip()
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None, "missing", clock_id, "TIME_VALUE_UNPARSEABLE"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, "missing", clock_id, "NAIVE_DATETIME_TIMEZONE_UNKNOWN"
    domain = "unix_epoch_estimated" if is_estimated else "unix_epoch_recorded"
    return parsed.timestamp(), domain, clock_id, None


def clock_pair_compatible(
    left: tuple[float | None, str, str | None, str | None],
    right: tuple[float | None, str, str | None, str | None],
    *,
    allow_explicit_estimated_pair: bool = False,
) -> tuple[bool, str]:
    if left[0] is None or right[0] is None:
        return False, left[3] or right[3] or "TIME_VALUE_MISSING"
    left_domain, right_domain = left[1], right[1]
    if left_domain == right_domain and left_domain in {"unix_epoch_recorded", "unix_epoch_estimated"}:
        return True, ""
    if {left_domain, right_domain} == {"unix_epoch_recorded", "unix_epoch_estimated"}:
        if allow_explicit_estimated_pair:
            return True, ""
        return False, "RECORDED_ESTIMATED_CLOCK_MISMATCH"
    if left_domain == right_domain == "relative":
        if left[2] and left[2] == right[2]:
            return True, ""
        return False, "RELATIVE_CLOCK_DOMAIN_MISMATCH"
    return False, "TIME_CLOCK_DOMAIN_MISMATCH"


def read_load_table(
    path: str | os.PathLike[str],
    time_column: str,
    force_column: str,
    force_unit: str | None = None,
    *,
    encoding: str = "utf-8-sig",
    time_parser=None,
) -> dict[str, Any]:
    """Read and validate a headered load CSV with strict monotonicity and unit checks."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise CampaignError("LOAD_SOURCE_NOT_FOUND", f"load source does not exist: {file_path}")
    with file_path.open("r", encoding=encoding, newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        delimiter = sniff_csv_delimiter(sample)
        reader = csv.DictReader(stream, delimiter=delimiter)
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers):
            raise CampaignError("LOAD_HEADER_INVALID", "load column names must be unique")
        if time_column not in headers:
            raise CampaignError("LOAD_TIME_COLUMN_MISSING", f"load source has no {time_column!r} column")
        if force_column not in headers:
            raise CampaignError("LOAD_FORCE_COLUMN_MISSING", f"load source has no {force_column!r} column")
        rows = [dict(raw) for raw in reader]

    parsed_times: list[float] = []
    time_domains: list[str] = []
    clock_ids: list[str | None] = []
    forces: list[float | None] = []
    valid: list[bool] = []
    force_reasons: list[str] = []
    parse_time = time_parser or parse_row_time
    for index, raw in enumerate(rows):
        time_value, domain, clock_id, time_error = parse_time(time_column, raw)
        if time_value is None:
            raise CampaignError("LOAD_TIME_AXIS_INVALID", f"load row {index + 2} has invalid time: {time_error}")
        parsed_times.append(time_value)
        time_domains.append(domain)
        clock_ids.append(clock_id)

        raw_force = finite_number(raw.get(force_column))
        _unit, scale = resolve_force_unit(raw, force_column, force_unit)
        if raw_force is None or scale is None:
            forces.append(None)
            valid.append(False)
            force_reasons.append("UNKNOWN_FORCE_UNIT" if raw_force is not None and scale is None else "INVALID_FORCE_VALUE")
        else:
            force = finite_number(raw_force * scale)
            forces.append(force)
            row_valid = raw.get("is_force_valid")
            valid.append(force is not None and (row_valid in (None, "") or parse_bool(row_valid)))
            force_reasons.append("INVALID_FORCE_VALUE" if force is None else
                                 "VALID_FORCE_N" if valid[-1] else "SOURCE_MARKED_FORCE_INVALID")

    if not rows:
        raise CampaignError("LOAD_SOURCE_EMPTY", f"load source has no data rows: {file_path}")
    if any(right <= left for left, right in zip(parsed_times, parsed_times[1:])):
        raise CampaignError(
            "LOAD_TIME_AXIS_NOT_STRICTLY_INCREASING",
            "load timestamps must be unique and increasing in source-row order",
        )
    return {
        "headers": headers,
        "rows": rows,
        "times": parsed_times,
        "forces": forces,
        "valid": valid,
        "time_domains": time_domains,
        "clock_ids": clock_ids,
        "force_reasons": force_reasons,
    }


def bracket_time(
    times: Sequence[float], query: float, max_gap_s: float, valid: Sequence[bool]
) -> dict[str, Any]:
    if not times:
        return {"status": "NO_TIME_SUPPORT", "indices": [], "weights": []}
    low = bisect.bisect_left(times, query)
    candidates = [index for index in (low - 1, low) if 0 <= index < len(times)]
    exact = next((index for index in candidates if abs(times[index] - query) <= EXACT_TIME_TOL_S), None)
    if exact is not None:
        if not valid[exact]:
            return {"status": "INVALID_EXACT_SAMPLE", "indices": [exact], "weights": [1.0]}
        return {"status": "EXACT_TIME_SAMPLE", "indices": [exact], "weights": [1.0]}
    if low == 0 or low == len(times):
        return {"status": "OUTSIDE_TIME_RANGE", "indices": [], "weights": []}
    left, right = low - 1, low
    delta = times[right] - times[left]
    if delta <= 0:
        return {"status": "NONMONOTONIC_TIME_AXIS", "indices": [left, right], "weights": []}
    if not valid[left] or not valid[right]:
        return {"status": "INVALID_BRACKET", "indices": [left, right], "weights": []}
    if delta > max_gap_s + EXACT_TIME_TOL_S:
        return {"status": "LONG_TIME_GAP", "indices": [left, right], "weights": []}
    alpha = (query - times[left]) / delta
    return {
        "status": "LINEAR_TIME_INTERPOLATION",
        "indices": [left, right],
        "weights": [1.0 - alpha, alpha],
        "alpha": alpha,
    }


def window_mean_support(
    times: Sequence[float],
    values: Sequence[float | None],
    valid: Sequence[bool],
    start: float | None,
    end: float | None,
    max_gap_s: float,
) -> dict[str, Any]:
    if start is None or end is None:
        return {"status": "EXPOSURE_WINDOW_MISSING", "value": None, "indices": [], "weights": []}
    if end <= start:
        return {"status": "INVALID_EXPOSURE_WINDOW", "value": None, "indices": [], "weights": []}
    if not times or start < times[0] - EXACT_TIME_TOL_S or end > times[-1] + EXACT_TIME_TOL_S:
        return {"status": "OUTSIDE_TIME_RANGE", "value": None, "indices": [], "weights": []}

    if len(times) == 1:
        if not valid[0] or values[0] is None:
            return {"status": "INVALID_LOAD_BRACKET", "value": None, "indices": [0], "weights": []}
        return {"status": "EXPOSURE_WINDOW_MEAN", "value": float(values[0]), "indices": [0], "weights": [1.0]}

    eff_start = max(float(start), float(times[0]))
    eff_end = min(float(end), float(times[-1]))
    if eff_end <= eff_start:
        boundary = 0 if eff_start <= times[0] else len(times) - 1
        if not valid[boundary] or values[boundary] is None:
            return {"status": "INVALID_LOAD_BRACKET", "value": None, "indices": [boundary], "weights": []}
        return {"status": "EXPOSURE_WINDOW_MEAN", "value": float(values[boundary]), "indices": [boundary], "weights": [1.0]}

    coefficients: dict[int, float] = {}
    idx = max(0, min(len(times) - 2, bisect.bisect_right(times, eff_start) - 1))
    while idx < len(times) - 1:
        left, right = idx, idx + 1
        seg_a = max(eff_start, float(times[left]))
        seg_b = min(eff_end, float(times[right]))
        if seg_b > seg_a:
            delta = float(times[right]) - float(times[left])
            if delta <= 0:
                return {"status": "NONMONOTONIC_TIME_AXIS", "value": None, "indices": [left, right], "weights": []}
            if delta > max_gap_s + EXACT_TIME_TOL_S:
                return {"status": "LONG_TIME_GAP", "value": None, "indices": [left, right], "weights": []}
            if not valid[left] or not valid[right] or values[left] is None or values[right] is None:
                return {"status": "INVALID_LOAD_BRACKET", "value": None, "indices": [left, right], "weights": []}
            r0 = max(0.0, min(1.0, (seg_a - float(times[left])) / delta))
            r1 = max(0.0, min(1.0, (seg_b - float(times[left])) / delta))
            segment_length = seg_b - seg_a
            coefficients[left] = coefficients.get(left, 0.0) + 0.5 * segment_length * ((1.0 - r0) + (1.0 - r1))
            coefficients[right] = coefficients.get(right, 0.0) + 0.5 * segment_length * (r0 + r1)
        if float(times[right]) >= eff_end:
            break
        idx += 1

    duration = eff_end - eff_start
    indices = sorted(coefficients)
    weights = [coefficients[index] / duration for index in indices]
    value = sum(float(values[index]) * weight for index, weight in zip(indices, weights) if values[index] is not None)
    return {"status": "EXPOSURE_WINDOW_MEAN", "value": value, "indices": indices, "weights": weights}
