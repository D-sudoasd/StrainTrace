"""Bounded, provenance-preserving load/strain alignment for campaign batches.

Recorded and estimated clocks stay separate. Native frame identities are used
only when a mapping is explicit; force and strain are never extrapolated.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import CampaignError
from .storage import (
    artifact_path,
    fingerprint_data,
    relative_artifact,
    sha256_file,
    stage_output_dir,
    write_jsonl_atomic,
)


_ABSOLUTE_EPOCH_MIN = 1_000_000_000.0
_EXACT_TIME_TOL_S = 2e-6
_TRUE = {"true", "1", "yes", "y"}
_FALSE = {"false", "0", "no", "n"}
_INVALID_STATUS_WORDS = ("FAIL", "REJECT", "INVALID", "QC_FAILED", "INSUFFICIENT")
_GAUGE_RE = re.compile(r"^(G\d{2})_engineering_strain$")


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    result = _number(value)
    if result is None:
        return None
    rounded = round(result)
    if abs(result - rounded) > 1e-8:
        return None
    return int(rounded)


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().casefold()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def _parse_time(value: Any, field: str, row: Mapping[str, Any]) -> tuple[float | None, str, str | None, str | None]:
    """Return (value, domain, clock_id, reason) for the selected time field."""
    if field not in row or row.get(field) is None or str(row.get(field)).strip() == "":
        if field == "timestamp_unix_s" and row.get("estimated_timestamp_unix_s") is not None:
            return None, "missing", None, "RECORDED_TIMESTAMP_MISSING_ESTIMATE_SEPARATE"
        return None, "missing", None, "TIME_VALUE_MISSING"
    status = str(row.get("timestamp_status") or row.get("time_status") or "").upper()
    if field == "timestamp_unix_s" and any(word in status for word in ("ESTIMAT", "RECONSTRUCT", "INFER", "PROTOCOL", "ADOPTED")):
        return None, "missing", None, "TIME_FIELD_MARKED_DERIVED"

    raw = row.get(field)
    number = _number(raw)
    clock_id = row.get("clock_id") or row.get("timestamp_clock_id") or row.get("time_clock_id")
    clock_id = str(clock_id) if clock_id not in (None, "") else None
    if number is not None:
        if field.endswith("unix_s"):
            if number < _ABSOLUTE_EPOCH_MIN:
                return None, "missing", clock_id, "UNIX_FIELD_VALUE_OUTSIDE_EPOCH_RANGE"
            return number, "unix_epoch_estimated" if field.startswith("estimated_") else "unix_epoch_recorded", clock_id, None
        if field == "time_s" or field.endswith("_time_s"):
            return number, "relative", clock_id, None if clock_id else "RELATIVE_CLOCK_ID_REQUIRED"
        if number >= _ABSOLUTE_EPOCH_MIN:
            return number, "unix_epoch_recorded", clock_id, None
        return number, "relative", clock_id, None if clock_id else "RELATIVE_CLOCK_ID_REQUIRED"

    text = str(raw).strip()
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None, "missing", clock_id, "TIME_VALUE_UNPARSEABLE"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, "missing", clock_id, "NAIVE_DATETIME_TIMEZONE_UNKNOWN"
    return parsed.timestamp(), "unix_epoch_recorded", clock_id, None


def _clock_pair_compatible(
    left: tuple[float | None, str, str | None, str | None],
    right: tuple[float | None, str, str | None, str | None],
) -> tuple[bool, str]:
    if left[0] is None or right[0] is None:
        return False, left[3] or right[3] or "TIME_VALUE_MISSING"
    left_domain, right_domain = left[1], right[1]
    if left_domain.startswith("unix_epoch") and right_domain.startswith("unix_epoch"):
        return True, ""
    if left_domain == right_domain == "relative":
        if left[2] and left[2] == right[2]:
            return True, ""
        return False, "RELATIVE_CLOCK_DOMAIN_MISMATCH"
    return False, "TIME_CLOCK_DOMAIN_MISMATCH"


def _read_load(
    path: Path, time_column: str, force_column: str, force_unit: str | None
) -> tuple[list[dict[str, Any]], list[float], list[float | None], list[bool], list[str], list[str | None], list[str]]:
    if not path.is_file():
        raise CampaignError("LOAD_SOURCE_NOT_FOUND", f"load source does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
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
    for i, raw in enumerate(rows):
        time_value, domain, clock_id, time_error = _parse_time(raw, time_column, raw)
        if time_value is None:
            raise CampaignError("LOAD_TIME_AXIS_INVALID", f"load row {i + 2} has invalid time: {time_error}")
        parsed_times.append(time_value)
        time_domains.append(domain)
        clock_ids.append(clock_id)

        raw_force = _number(raw.get(force_column))
        unit = force_unit
        if unit is None:
            unit = raw.get("force_unit") or raw.get("load_unit") or raw.get("unit")
        if unit is None and force_column == "force_N":
            unit = "N"
        unit_norm = str(unit or "").strip().casefold().replace(" ", "")
        scale = {
            "n": 1.0, "newton": 1.0, "newtons": 1.0,
            "kn": 1000.0, "kilonewton": 1000.0, "kilonewtons": 1000.0,
        }.get(unit_norm)
        if raw_force is None or scale is None:
            forces.append(None)
            valid.append(False)
            force_reasons.append("UNKNOWN_FORCE_UNIT" if raw_force is not None and scale is None else "INVALID_FORCE_VALUE")
        else:
            forces.append(raw_force * scale)
            row_valid = raw.get("is_force_valid")
            valid.append(True if row_valid in (None, "") else _bool(row_valid))
            force_reasons.append("VALID_FORCE_N" if valid[-1] else "SOURCE_MARKED_FORCE_INVALID")
    if not rows:
        raise CampaignError("LOAD_SOURCE_EMPTY", f"load source has no data rows: {path}")
    if any(b <= a for a, b in zip(parsed_times, parsed_times[1:])):
        raise CampaignError("LOAD_TIME_AXIS_NOT_STRICTLY_INCREASING", "load timestamps must be unique and increasing in source-row order")
    return rows, parsed_times, forces, valid, time_domains, clock_ids, force_reasons


def _axis_times(
    rows: Sequence[Mapping[str, Any]], field: str
) -> tuple[list[float | None], list[str], list[str | None], list[str | None]]:
    values: list[float | None] = []
    domains: list[str] = []
    clock_ids: list[str | None] = []
    errors: list[str | None] = []
    for row in rows:
        value, domain, clock_id, error = _parse_time(row, field, row)
        values.append(value)
        domains.append(domain)
        clock_ids.append(clock_id)
        errors.append(error)
    return values, domains, clock_ids, errors


def _source_rows(acquisition: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    result = acquisition.get("result")
    if not isinstance(result, Mapping):
        return []
    rows_path = result.get("rows_path")
    if not rows_path:
        return []
    try:
        from .storage import read_jsonl

        return read_jsonl(artifact_path(project_root, str(rows_path)))
    except CampaignError:
        raise
    except OSError as exc:
        raise CampaignError("DIC_SOURCE_ROWS_READ_ERROR", f"cannot read DIC rows: {exc}") from exc


def _hydrate_source_times(
    rows: list[dict[str, Any]], image_frames: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Join time metadata only by an explicit native DIC counter."""
    by_native: dict[int, list[Mapping[str, Any]]] = {}
    for image in image_frames:
        native = _integer(image.get("native_frame"))
        if native is not None:
            by_native.setdefault(native, []).append(image)
    enriched: list[dict[str, Any]] = []
    time_fields = (
        "timestamp_unix_s", "estimated_timestamp_unix_s", "time_s", "clock_id",
        "timestamp_source", "timestamp_status", "exposure_start_unix_s", "exposure_end_unix_s",
    )
    for source in rows:
        row = dict(source)
        native = _integer(row.get("native_dic_frame", row.get("native_frame", row.get("frame"))))
        candidates = by_native.get(native, []) if native is not None else []
        if len(candidates) == 1:
            image = candidates[0]
            for field in time_fields:
                if row.get(field) is None and image.get(field) is not None:
                    row[field] = image[field]
        elif len(candidates) > 1:
            row["_time_join_status"] = "DUPLICATE_NATIVE_DIC_COUNTER"
        enriched.append(row)
    return enriched


def _gauge_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    found: dict[str, float | None] = {}
    for key, value in row.items():
        match = _GAUGE_RE.fullmatch(str(key))
        if match:
            found[match.group(1)] = _number(value)
    return found


def _aggregate_strain(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    engineering = _number(row.get("engineering_strain"))
    if engineering is None:
        engineering = _number(row.get("engineering_strain_mean"))
    if engineering is None:
        engineering = _number(row.get("dic_engineering_strain"))
    true = _number(row.get("true_strain"))
    if true is None:
        true = _number(row.get("true_strain_mean"))
    return engineering, true


def _quality_eligible(row: Mapping[str, Any]) -> bool:
    count = _integer(row.get("valid_gauge_count"))
    if count is None or count < 3:
        return False
    for key in ("quality_status", "strain_status", "status"):
        status = str(row.get(key, "")).upper()
        if any(word in status for word in _INVALID_STATUS_WORDS):
            return False
    engineering, _ = _aggregate_strain(row)
    gauges = _gauge_values(row)
    return engineering is not None or sum(value is not None for value in gauges.values()) >= 3


def _bracket(times: Sequence[float], q: float, max_gap_s: float, valid: Sequence[bool]) -> dict[str, Any]:
    if not times:
        return {"status": "NO_TIME_SUPPORT", "indices": [], "weights": []}
    low, high = 0, len(times)
    while low < high:
        middle = (low + high) // 2
        if times[middle] < q:
            low = middle + 1
        else:
            high = middle
    candidates = [index for index in (low - 1, low) if 0 <= index < len(times)]
    exact = next((index for index in candidates if abs(times[index] - q) <= _EXACT_TIME_TOL_S), None)
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
    if delta > max_gap_s + _EXACT_TIME_TOL_S:
        return {"status": "LONG_TIME_GAP", "indices": [left, right], "weights": []}
    alpha = (q - times[left]) / delta
    return {
        "status": "LINEAR_TIME_INTERPOLATION",
        "indices": [left, right],
        "weights": [1 - alpha, alpha],
        "alpha": alpha,
    }


def _window_mean(
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
    if not times or start < times[0] - _EXACT_TIME_TOL_S or end > times[-1] + _EXACT_TIME_TOL_S:
        return {"status": "OUTSIDE_TIME_RANGE", "value": None, "indices": [], "weights": []}

    coefficients: dict[int, float] = {}
    cursor = start
    while cursor < end - _EXACT_TIME_TOL_S:
        insertion = 0
        while insertion < len(times) and times[insertion] < cursor - _EXACT_TIME_TOL_S:
            insertion += 1
        exact = insertion < len(times) and abs(times[insertion] - cursor) <= _EXACT_TIME_TOL_S
        if exact:
            left = insertion
            right = left + 1
        else:
            right = insertion
            left = right - 1
        if left < 0 or right >= len(times):
            return {"status": "OUTSIDE_TIME_RANGE", "value": None, "indices": [], "weights": []}
        delta = times[right] - times[left]
        if delta <= 0:
            return {"status": "NONMONOTONIC_TIME_AXIS", "value": None, "indices": [left, right], "weights": []}
        if delta > max_gap_s + _EXACT_TIME_TOL_S:
            return {"status": "LONG_TIME_GAP", "value": None, "indices": [left, right], "weights": []}
        if not valid[left] or not valid[right] or values[left] is None or values[right] is None:
            return {"status": "INVALID_LOAD_BRACKET", "value": None, "indices": [left, right], "weights": []}
        next_knot = min(end, times[right])
        if next_knot <= cursor + _EXACT_TIME_TOL_S:
            next_knot = end
        r0 = max(0.0, min(1.0, (cursor - times[left]) / delta))
        r1 = max(0.0, min(1.0, (next_knot - times[left]) / delta))
        segment_length = next_knot - cursor
        coefficients[left] = coefficients.get(left, 0.0) + 0.5 * segment_length * ((1 - r0) + (1 - r1))
        coefficients[right] = coefficients.get(right, 0.0) + 0.5 * segment_length * (r0 + r1)
        cursor = next_knot

    duration = end - start
    indices = sorted(coefficients)
    weights = [coefficients[index] / duration for index in indices]
    value = sum(values[index] * weight for index, weight in zip(indices, weights) if values[index] is not None)
    return {"status": "EXPOSURE_WINDOW_MEAN", "value": value, "indices": indices, "weights": weights}


def _confirmed_area(acquisition: Mapping[str, Any]) -> tuple[float | None, float | None, str | None, str]:
    dimensions = acquisition.get("dimensions") if isinstance(acquisition.get("dimensions"), Mapping) else {}
    status = str(dimensions.get("status", "")).strip().casefold()
    area_candidate = _number(dimensions.get("area_mm2"))
    if status not in {"confirmed", "verified_recorded_dimensions", "verified_recorded_area"}:
        width = _number(dimensions.get("width_mm"))
        thickness = _number(dimensions.get("thickness_mm"))
        if area_candidate is None and width is not None and thickness is not None and width > 0 and thickness > 0:
            area_candidate = width * thickness
        return None, area_candidate, dimensions.get("source"), "AREA_NOT_CONFIRMED"
    area = area_candidate
    source = dimensions.get("source")
    if area is None:
        width = _number(dimensions.get("width_mm"))
        thickness = _number(dimensions.get("thickness_mm"))
        if width is not None and thickness is not None and width > 0 and thickness > 0:
            area = width * thickness
            source = f"{source or 'confirmed dimensions'}; width_mm*thickness_mm"
    if area is None or area <= 0:
        return None, area_candidate, source, "CONFIRMED_AREA_VALUE_MISSING_OR_INVALID"
    return area, area_candidate, source, "CONFIRMED_AREA"


def _read_axis(
    acquisition: Mapping[str, Any], project_root: Path, source_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    measurement = acquisition.get("measurement_frames") or []
    if measurement:
        return [dict(row) for row in measurement], "explicit_measurement_frames"
    if source_rows:
        return [dict(row) for row in source_rows], "dic_result_rows"
    images = acquisition.get("image_frames") or []
    if images:
        axis = []
        for image in images:
            row = {
                key: image.get(key)
                for key in (
                    "frame", "native_frame", "timestamp_unix_s", "estimated_timestamp_unix_s", "time_s",
                    "clock_id", "timestamp_source", "timestamp_status", "exposure_start_unix_s", "exposure_end_unix_s",
                )
                if key in image
            }
            row["native_dic_frame"] = image.get("native_frame")
            axis.append(row)
        return axis, "image_frames"
    raise CampaignError("ALIGNMENT_AXIS_MISSING", "acquisition has no measurement axis, DIC result rows, or image frames")


def _dic_rows_for_axis(
    axis_rows: Sequence[Mapping[str, Any]],
    source_rows: list[dict[str, Any]],
    time_column: str,
    max_gap_s: float,
    *,
    preserve_imported: bool = False,
) -> list[dict[str, Any]]:
    if not source_rows:
        return [{"status": "NO_DIC_RESULT"} for _ in axis_rows]
    dic_rows = source_rows
    exact_by_native: dict[int, list[int]] = {}
    exact_by_cbf: dict[int, list[int]] = {}
    exact_by_frame: dict[int, list[int]] = {}
    for index, row in enumerate(dic_rows):
        native = _integer(row.get("native_dic_frame", row.get("native_frame", row.get("frame"))))
        if native is not None:
            exact_by_native.setdefault(native, []).append(index)
        native_cbf = _integer(row.get("native_cbf_frame"))
        if native_cbf is not None:
            exact_by_cbf.setdefault(native_cbf, []).append(index)
        source_frame = _integer(row.get("frame"))
        if source_frame is not None:
            exact_by_frame.setdefault(source_frame, []).append(index)

    direct_mapping: list[int | None] = []
    direct_status: list[str | None] = []
    for axis in axis_rows:
        native = _integer(axis.get("native_dic_frame", axis.get("dic_native_frame")))
        candidates: list[int] = []
        if native is not None:
            matches = exact_by_native.get(native, [])
            candidates = matches
        if not candidates:
            native_cbf = _integer(axis.get("native_cbf_frame"))
            if native_cbf is not None:
                candidates = exact_by_cbf.get(native_cbf, [])
        if not candidates:
            frame = _integer(axis.get("frame"))
            if frame is not None:
                candidates = exact_by_frame.get(frame, [])

        if len(candidates) == 1:
            source = dic_rows[candidates[0]]
            axis_cbf = _integer(axis.get("native_cbf_frame"))
            source_cbf = _integer(source.get("native_cbf_frame"))
            axis_global = _bool(axis.get("scan_id_is_global_fio"))
            source_global = _bool(source.get("scan_id_is_global_fio"))
            axis_scan = _integer(axis.get("scan_id")) if axis_global else None
            source_scan = _integer(source.get("scan_id")) if source_global else None
            if axis_cbf is not None and source_cbf is not None and axis_cbf != source_cbf:
                direct_mapping.append(None)
                direct_status.append("SOURCE_FRAME_IDENTITY_CONFLICT")
            elif axis_global and source_global and axis_scan is not None and source_scan is not None and axis_scan != source_scan:
                direct_mapping.append(None)
                direct_status.append("SOURCE_SCAN_IDENTITY_CONFLICT")
            else:
                direct_mapping.append(candidates[0])
                direct_status.append(None)
        elif len(candidates) > 1:
            direct_mapping.append(None)
            direct_status.append("DUPLICATE_SOURCE_NATIVE_FRAME")
        else:
            direct_mapping.append(None)
            direct_status.append(None)

    axis_times, axis_domains, axis_clocks, axis_errors = _axis_times(axis_rows, time_column)
    dic_times, dic_domains, dic_clocks, dic_errors = _axis_times(dic_rows, time_column)
    timed_indices = [index for index, time in enumerate(dic_times) if time is not None]
    timed_values = [float(dic_times[index]) for index in timed_indices]
    if any(right <= left for left, right in zip(timed_values, timed_values[1:])):
        raise CampaignError("DIC_TIME_AXIS_NOT_STRICTLY_INCREASING", "DIC result times must increase in source-frame order")
    timed_rows = [dic_rows[index] for index in timed_indices]
    timed_valid = [_quality_eligible(row) for row in timed_rows]
    output: list[dict[str, Any]] = []

    for axis_index, axis in enumerate(axis_rows):
        mapped = direct_mapping[axis_index]
        if mapped is not None:
            source = dic_rows[mapped]
            if preserve_imported:
                output.append({
                    "status": "IMPORTED_ACCEPTED_EXACT_IDENTITY",
                    "source_index": mapped,
                    "source_row": source,
                    "native_dic_frame": source.get(
                        "native_dic_frame", source.get("native_frame", source.get("frame"))
                    ),
                    "left_index": mapped,
                    "right_index": mapped,
                    "weights": [1.0],
                })
            elif not _quality_eligible(source):
                output.append({
                    "status": "DIC_FRAME_QC_FAILED",
                    "native_dic_frame": axis.get("native_dic_frame", axis.get("dic_native_frame")),
                    "left_index": mapped,
                    "right_index": mapped,
                    "weights": [1.0],
                })
            else:
                output.append({
                    "status": "DIC_EXACT_NATIVE_FRAME",
                    "source_index": mapped,
                    "source_row": source,
                    "native_dic_frame": source.get("native_dic_frame", source.get("native_frame", source.get("frame"))),
                    "left_index": mapped,
                    "right_index": mapped,
                    "weights": [1.0],
                })
            continue

        if direct_status[axis_index] is not None:
            output.append({"status": direct_status[axis_index], "reason": direct_status[axis_index]})
            continue

        if preserve_imported:
            output.append({
                "status": "IMPORTED_ACCEPTED_NO_EXACT_FRAME_IDENTITY",
                "reason": "IMPORTED_ACCEPTED_ROWS_REQUIRE_EXACT_FRAME_OR_NATIVE_COUNTER_MATCH",
            })
            continue

        if axis_times[axis_index] is None:
            reason = axis_errors[axis_index] or "NO_DIC_TIME_SUPPORT"
            output.append({"status": reason, "reason": reason})
            continue
        q = float(axis_times[axis_index])
        compatibility: list[bool] = []
        clock_reason = "NO_DIC_TIME_SUPPORT"
        for dic_index in timed_indices:
            ok, reason = _clock_pair_compatible(
                (q, axis_domains[axis_index], axis_clocks[axis_index], axis_errors[axis_index]),
                (dic_times[dic_index], dic_domains[dic_index], dic_clocks[dic_index], dic_errors[dic_index]),
            )
            compatibility.append(ok)
            if not ok:
                clock_reason = reason
        if not compatibility or not all(compatibility):
            output.append({"status": clock_reason, "reason": clock_reason})
            continue
        if not timed_values:
            output.append({"status": "NO_DIC_TIME_SUPPORT"})
            continue
        support = _bracket(timed_values, q, max_gap_s, timed_valid)
        local_indices = support.get("indices", [])
        if not local_indices or support["status"] not in {"EXACT_TIME_SAMPLE", "LINEAR_TIME_INTERPOLATION"}:
            output.append({
                "status": f"DIC_{support['status']}",
                "left_index": None,
                "right_index": None,
                "weights": support.get("weights", []),
            })
            continue
        source_indices = [timed_indices[index] for index in local_indices]
        if len(source_indices) == 1:
            source = dic_rows[source_indices[0]]
            output.append({
                "status": "DIC_EXACT_TIME_SAMPLE",
                "source_index": source_indices[0],
                "source_row": source,
                "native_dic_frame": source.get("native_dic_frame", source.get("native_frame", source.get("frame"))),
                "left_index": source_indices[0],
                "right_index": source_indices[0],
                "weights": [1.0],
            })
            continue
        left, right = (dic_rows[index] for index in source_indices)
        left_native = _integer(left.get("native_dic_frame", left.get("native_frame", left.get("frame"))))
        right_native = _integer(right.get("native_dic_frame", right.get("native_frame", right.get("frame"))))
        if left_native is None or right_native is None or right_native != left_native + 1:
            output.append({
                "status": "DIC_NATIVE_FRAME_GAP",
                "left_index": source_indices[0],
                "right_index": source_indices[1],
                "weights": [],
            })
            continue
        weights = support["weights"]
        left_gauges, right_gauges = _gauge_values(left), _gauge_values(right)
        gauge_names = sorted(set(left_gauges) & set(right_gauges))
        gauges: dict[str, float | None] = {}
        for name in gauge_names:
            lval, rval = left_gauges[name], right_gauges[name]
            gauges[name] = None if lval is None or rval is None else lval * weights[0] + rval * weights[1]
        accepted_gauges = [value for value in gauges.values() if value is not None]
        if len(accepted_gauges) >= 3:
            mean = sum(accepted_gauges) / len(accepted_gauges)
        else:
            e0, _ = _aggregate_strain(left)
            e1, _ = _aggregate_strain(right)
            mean = e0 * weights[0] + e1 * weights[1] if not gauge_names and e0 is not None and e1 is not None else None
        true = math.log1p(mean) if mean is not None and mean > -1 else None
        interpolated = {f"{name}_engineering_strain": value for name, value in gauges.items()}
        interpolated.update({
            "engineering_strain": mean,
            "true_strain": true,
            "valid_gauge_count": len(accepted_gauges) if gauge_names else min(
                _integer(left.get("valid_gauge_count")) or 0, _integer(right.get("valid_gauge_count")) or 0
            ),
            "status": "DIC_BOUNDED_INTERPOLATION",
            "source_indices": source_indices,
            "source_frames": [left.get("frame"), right.get("frame")],
            "weights": weights,
        })
        output.append(interpolated)
    return output


def align(
    acquisition: Mapping[str, Any],
    project_dir: str | os.PathLike[str],
    load_path: str | os.PathLike[str],
    mode: str = "instantaneous",
    max_gap_s: float = 2.5,
    time_column: str = "timestamp_unix_s",
    force_column: str = "force_N",
    *,
    measurement_time_field: str | None = None,
    load_time_column: str | None = None,
    force_unit: str | None = None,
    dic_max_gap_s: float | None = None,
    exposure_start_column: str = "exposure_start_unix_s",
    exposure_end_column: str = "exposure_end_unix_s",
) -> dict[str, Any]:
    """Align force and available DIC data onto an explicit physical frame axis.

    Instantaneous mode performs bounded exact/linear interpolation. Window
    mode computes an exact trapezoidal mean of the piecewise-linear load signal
    over each exposure window. Invalid load rows remain barriers. Relative
    clocks require matching explicit clock IDs. Selecting an estimated time
    column is an explicit choice recorded in stage provenance.
    """
    mode_norm = str(mode).strip().casefold()
    if mode_norm in {"exposure_window_mean", "window", "exposure_mean"}:
        mode_norm = "window_mean"
    if mode_norm not in {"instantaneous", "window_mean"}:
        raise CampaignError("ALIGNMENT_MODE_UNSUPPORTED", "mode must be instantaneous or window_mean")
    if measurement_time_field:
        if time_column != "timestamp_unix_s" and time_column != measurement_time_field:
            raise CampaignError(
                "ALIGNMENT_TIME_FIELD_CONFLICT",
                "time_column and measurement_time_field select different measurement clocks",
            )
        time_column = measurement_time_field
    load_time_column = load_time_column or time_column
    gap = _number(max_gap_s)
    dic_gap = _number(dic_max_gap_s if dic_max_gap_s is not None else max_gap_s)
    if gap is None or gap <= 0 or dic_gap is None or dic_gap <= 0:
        raise CampaignError("ALIGNMENT_MAX_GAP_INVALID", "max_gap_s and dic_max_gap_s must be positive finite seconds")
    if not isinstance(acquisition, Mapping):
        raise CampaignError("ALIGNMENT_ACQUISITION_INVALID", "acquisition must be a mapping")
    acquisition_id = str(acquisition.get("acquisition_id") or "")
    if not acquisition_id:
        raise CampaignError("ALIGNMENT_ACQUISITION_ID_MISSING", "acquisition_id is required")
    root = Path(project_dir).expanduser().resolve()
    load = Path(load_path).expanduser().resolve()
    source_rows = _source_rows(acquisition, root)
    source_rows = _hydrate_source_times(source_rows, acquisition.get("image_frames") or [])
    axis_rows, axis_kind = _read_axis(acquisition, root, source_rows)
    load_rows, load_times, load_forces, load_valid, load_domains, load_clock_ids, load_force_reasons = _read_load(
        load, load_time_column, force_column, force_unit
    )
    axis_times, axis_domains, axis_clock_ids, axis_errors = _axis_times(axis_rows, time_column)
    if axis_kind == "dic_result_rows":
        dic_alignment = [
            {
                "status": "DIC_SOURCE_ROW_PRESERVED",
                "source_row": row,
                "native_dic_frame": row.get("native_dic_frame", row.get("native_frame", row.get("frame"))),
            }
            for row in axis_rows
        ]
    else:
        dic_alignment = _dic_rows_for_axis(
            axis_rows,
            source_rows,
            time_column,
            dic_gap,
            preserve_imported=str((acquisition.get("result") or {}).get("status", "")).upper() == "IMPORTED_ACCEPTED",
        )
    area, candidate_area, area_source, area_status = _confirmed_area(acquisition)
    result_imported = str((acquisition.get("result") or {}).get("status", "")).upper() == "IMPORTED_ACCEPTED"
    aligned_rows: list[dict[str, Any]] = []
    summary = {
        "frame_count": len(axis_rows), "finite_force_count": 0, "finite_stress_count": 0,
        "finite_strain_count": 0, "interpolated_force_count": 0, "window_mean_force_count": 0,
        "interpolated_strain_count": 0, "missing_force_count": 0, "missing_strain_count": 0,
        "area_status": area_status,
    }

    for index, (axis, dic) in enumerate(zip(axis_rows, dic_alignment)):
        output = dict(axis)
        frame_no = _integer(axis.get("frame"))
        if frame_no is None:
            raise CampaignError("ALIGNMENT_FRAME_ID_MISSING", f"axis row {index} has no integer frame identity")
        output["frame"] = frame_no
        output["timestamp_unix_s"] = axis.get("timestamp_unix_s")
        output["estimated_timestamp_unix_s"] = axis.get("estimated_timestamp_unix_s")
        output["time_s"] = axis.get("time_s")
        output["timestamp_source"] = axis.get("timestamp_source")
        output["timestamp_status"] = axis.get("timestamp_status")
        output["time_column_used"] = time_column
        output["load_time_column_used"] = load_time_column
        output["time_domain_status"] = "RECORDED_OR_EXPLICIT_TIME" if axis_times[index] is not None else axis_errors[index]
        output["force_unit"] = "N"
        output["load_force_unit_status"] = (
            "N" if all(reason != "UNKNOWN_FORCE_UNIT" for reason in load_force_reasons)
            else "UNKNOWN_OR_UNSUPPORTED_UNIT"
        )
        output["area_mm2_candidate"] = candidate_area
        output["area_mm2_source"] = area_source
        output["area_status"] = area_status
        if area is not None:
            output["area_mm2"] = area
        elif _number(axis.get("area_mm2")) is None:
            output["area_mm2"] = None

        query = axis_times[index]
        clock_compat = False
        clock_status = axis_errors[index] or "NO_QUERY_TIME"
        if query is not None:
            sample = (query, axis_domains[index], axis_clock_ids[index], axis_errors[index])
            compatible_rows = []
            for load_index, load_time in enumerate(load_times):
                compatible, _ = _clock_pair_compatible(
                    sample,
                    (load_time, load_domains[load_index], load_clock_ids[load_index], None),
                )
                compatible_rows.append(compatible)
            if compatible_rows and all(compatible_rows):
                clock_compat, clock_status = True, "CLOCKS_COMPATIBLE"
            elif any(compatible_rows):
                clock_status = "MIXED_LOAD_CLOCK_DOMAINS"
            elif compatible_rows:
                _, clock_status = _clock_pair_compatible(
                    sample, (load_times[0], load_domains[0], load_clock_ids[0], None)
                )
        output["load_time_domain_status"] = clock_status
        force_support: dict[str, Any]
        if not clock_compat:
            force_support = {"status": clock_status, "indices": [], "weights": []}
            force_value = None
        elif mode_norm == "window_mean":
            start_info = _parse_time(
                {exposure_start_column: axis.get(exposure_start_column), "clock_id": axis_clock_ids[index]},
                exposure_start_column,
                axis,
            )
            end_info = _parse_time(
                {exposure_end_column: axis.get(exposure_end_column), "clock_id": axis_clock_ids[index]},
                exposure_end_column,
                axis,
            )
            start_ok, start_reason = _clock_pair_compatible(
                start_info,
                (query, axis_domains[index], axis_clock_ids[index], axis_errors[index]),
            )
            end_ok, end_reason = _clock_pair_compatible(
                end_info,
                (query, axis_domains[index], axis_clock_ids[index], axis_errors[index]),
            )
            if not start_ok or not end_ok:
                force_support = {
                    "status": start_reason if not start_ok else end_reason,
                    "indices": [],
                    "weights": [],
                    "value": None,
                }
            else:
                force_support = _window_mean(
                    load_times, load_forces, load_valid, start_info[0], end_info[0], gap
                )
            force_value = force_support.get("value")
        else:
            load_support = _bracket(load_times, float(query), gap, load_valid)
            force_support = load_support
            if load_support["status"] == "INVALID_EXACT_SAMPLE":
                source_index = load_support["indices"][0]
                reason = load_force_reasons[source_index]
                force_support["status"] = reason if reason == "UNKNOWN_FORCE_UNIT" else "INVALID_LOAD_SAMPLE"
                force_value = None
            elif load_support["status"] == "EXACT_TIME_SAMPLE":
                force_value = load_forces[load_support["indices"][0]]
                if force_value is None:
                    force_support["status"] = load_force_reasons[load_support["indices"][0]]
            elif load_support["status"] == "LINEAR_TIME_INTERPOLATION":
                left_i, right_i = load_support["indices"]
                w0, w1 = load_support["weights"]
                force_value = load_forces[left_i] * w0 + load_forces[right_i] * w1
            else:
                force_value = None
        support_indices = force_support.get("indices", [])
        support_weights = force_support.get("weights", [])
        output["force_left_index"] = support_indices[0] if support_indices else None
        output["force_right_index"] = support_indices[-1] if support_indices else None
        output["force_support_indices"] = support_indices
        output["force_support_weights"] = support_weights
        output["force_support_times"] = [load_times[i] for i in support_indices]
        output["force_support_sources"] = [
            {
                "load_row_index": i,
                "time_raw": load_rows[i].get(load_time_column),
                "time_domain": load_domains[i],
                "clock_id": load_clock_ids[i],
                "force_raw": load_rows[i].get(force_column),
                "force_unit_raw": load_rows[i].get("force_unit", load_rows[i].get("load_unit", load_rows[i].get("unit"))),
                "force_N": load_forces[i],
                "force_valid": load_valid[i],
                "force_status": load_force_reasons[i],
            }
            for i in support_indices
        ]
        output["force_interpolation_status"] = force_support["status"]
        output["force_interpolation_alpha"] = force_support.get("alpha")
        output["force_alignment_mode"] = mode_norm

        accepted_source = dic.get("source_row") if result_imported and isinstance(dic.get("source_row"), Mapping) else axis
        source_force = _number(accepted_source.get("force_N"))
        source_stress = _number(accepted_source.get("stress_MPa"))
        source_status = str(accepted_source.get("strain_status", "")).upper()
        imported = result_imported or "IMPORTED_ACCEPTED" in source_status
        if imported and source_force is not None:
            output["source_accepted_force_N"] = source_force
        if imported and source_stress is not None:
            output["source_accepted_stress_MPa"] = source_stress
        if imported:
            output["force_N"] = source_force
            output["force_N_candidate"] = force_value
            output["force_status"] = (
                "IMPORTED_ACCEPTED_PRESERVED" if source_force is not None else force_support["status"]
            )
            output["force_candidate_status"] = force_support["status"]
            if force_value is not None:
                summary["finite_force_count"] += 1
                if force_support["status"] == "LINEAR_TIME_INTERPOLATION":
                    summary["interpolated_force_count"] += 1
                if force_support["status"] == "EXPOSURE_WINDOW_MEAN":
                    summary["window_mean_force_count"] += 1
            else:
                summary["missing_force_count"] += 1
        elif force_value is not None:
            output["force_N"] = force_value
            output["force_status"] = force_support["status"]
            summary["finite_force_count"] += 1
            if force_support["status"] == "LINEAR_TIME_INTERPOLATION":
                summary["interpolated_force_count"] += 1
            if force_support["status"] == "EXPOSURE_WINDOW_MEAN":
                summary["window_mean_force_count"] += 1
        else:
            output["force_N"] = None
            output["force_status"] = force_support["status"]
            summary["missing_force_count"] += 1

        if imported:
            output["stress_MPa"] = source_stress
            output["stress_status"] = (
                "IMPORTED_ACCEPTED_PRESERVED" if source_stress is not None else "NO_IMPORTED_ACCEPTED_STRESS"
            )
            if area is not None and force_value is not None:
                calculated_stress = force_value / area
                output["stress_MPa_candidate"] = calculated_stress
                output["stress_candidate_status"] = "CALCULATED_FROM_ALIGNED_FORCE_AND_CONFIRMED_AREA"
                if source_stress is not None:
                    output["stress_source_conflict"] = not math.isclose(
                        source_stress, calculated_stress, rel_tol=1e-9, abs_tol=1e-9
                    )
            else:
                output["stress_MPa_candidate"] = None
                output["stress_candidate_status"] = (
                    "MISSING_CONFIRMED_AREA" if area is None else "MISSING_ALIGNED_FORCE"
                )
        elif area is not None and force_value is not None:
            calculated_stress = force_value / area
            output["stress_MPa"] = calculated_stress
            output["stress_status"] = "CALCULATED_FROM_FORCE_AND_CONFIRMED_AREA"
        else:
            output["stress_MPa"] = None
            output["stress_status"] = "MISSING_CONFIRMED_AREA" if area is None else "MISSING_ALIGNED_FORCE"
        if _number(output.get("stress_MPa")) is not None or _number(output.get("stress_MPa_candidate")) is not None:
            summary["finite_stress_count"] += 1

        output["dic_support_status"] = dic.get("status", "NO_DIC_RESULT")
        output["dic_left_index"] = dic.get("left_index")
        output["dic_right_index"] = dic.get("right_index")
        output["dic_support_weights"] = dic.get("weights", [])
        output["dic_source_frames"] = dic.get("source_frames")
        if isinstance(dic.get("source_row"), Mapping):
            source = dict(dic["source_row"])
            output["dic_source_frame"] = dic.get(
                "native_dic_frame", source.get("native_dic_frame", source.get("native_frame", source.get("frame")))
            )
            output["dic_source_status"] = source.get("strain_status", source.get("quality_status"))
            output["dic_source_values"] = source
            for gauge_name, value in _gauge_values(source).items():
                output[f"{gauge_name}_engineering_strain"] = value
            engineering, true = _aggregate_strain(source)
            output["engineering_strain"] = engineering
            output["true_strain"] = true
            output["valid_gauge_count"] = _integer(source.get("valid_gauge_count"))
            output["engineering_strain_sd"] = _number(
                source.get("engineering_strain_sd", source.get("engineering_strain_mean_sd"))
            )
            output["strain_status"] = source.get(
                "strain_status", source.get("quality_status", "SOURCE_VALUE_PRESERVED")
            )
        elif dic.get("status") == "DIC_BOUNDED_INTERPOLATION":
            for key, value in dic.items():
                if key.endswith("_engineering_strain"):
                    output[key] = value
            output["engineering_strain"] = dic.get("engineering_strain")
            output["true_strain"] = dic.get("true_strain")
            output["valid_gauge_count"] = dic.get("valid_gauge_count")
            accepted_gauges = [
                _number(value)
                for key, value in dic.items()
                if _GAUGE_RE.fullmatch(str(key)) and _number(value) is not None
            ]
            output["engineering_strain_sd"] = (
                statistics.stdev(accepted_gauges) if len(accepted_gauges) >= 3 else None
            )
            output["strain_status"] = "DIC_BOUNDED_INTERPOLATION"
            summary["interpolated_strain_count"] += int(dic.get("engineering_strain") is not None)
        else:
            output["engineering_strain"] = None
            output["true_strain"] = None
            output["valid_gauge_count"] = None
            output["engineering_strain_sd"] = None
            output["strain_status"] = dic.get("status", "NO_DIC_RESULT")

        if output.get("engineering_strain") is not None:
            summary["finite_strain_count"] += 1
        else:
            summary["missing_strain_count"] += 1
        aligned_rows.append(output)

    axis_digest = hashlib.sha256(
        json.dumps(
            [
                {
                    key: row.get(key)
                    for key in (
                        "frame", "native_cbf_frame", "native_dic_frame", "native_frame", "scan_id",
                        "scan_id_is_global_fio", time_column, load_time_column,
                        exposure_start_column, exposure_end_column,
                    )
                }
                for row in axis_rows
            ],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    load_digest = sha256_file(load)
    result_record = acquisition.get("result") if isinstance(acquisition.get("result"), Mapping) else {}
    fingerprint = fingerprint_data(
        "campaign_alignment_v1",
        {
            "acquisition_id": acquisition_id,
            "axis_kind": axis_kind,
            "axis_sha256": axis_digest,
            "dic_result_fingerprint": result_record.get("fingerprint"),
            "load_path": str(load),
            "load_sha256": load_digest,
            "mode": mode_norm,
            "max_gap_s": gap,
            "dic_max_gap_s": dic_gap,
            "time_column": time_column,
            "load_time_column": load_time_column,
            "force_column": force_column,
            "force_unit": force_unit,
            "exposure_start_column": exposure_start_column,
            "exposure_end_column": exposure_end_column,
        },
    )
    output_dir = stage_output_dir(root, acquisition_id, "alignment", fingerprint)
    rows_path = output_dir / "alignment.jsonl"
    write_jsonl_atomic(rows_path, aligned_rows)
    relative_rows_path = relative_artifact(root, rows_path)
    checksum = sha256_file(rows_path)
    return {
        "status": "COMPLETE",
        "fingerprint": fingerprint,
        "rows_path": relative_rows_path,
        "artifacts": [relative_rows_path],
        "checksums": {relative_rows_path: checksum},
        "provenance": {
            "load_source_path": str(load),
            "load_source_sha256": load_digest,
            "measurement_time_field": time_column,
            "load_time_column": load_time_column,
            "load_force_column": force_column,
            "load_force_unit_override": force_unit,
            "time_column_selection_is_explicit": time_column != "timestamp_unix_s",
            "axis_kind": axis_kind,
            "axis_sha256": axis_digest,
            "mode": mode_norm,
            "max_gap_s": gap,
            "dic_max_gap_s": dic_gap,
            "area_status": area_status,
            "area_mm2_used": area,
            "area_mm2_candidate": candidate_area,
            "area_source": area_source,
            "recorded_and_estimated_timestamps_kept_separate": True,
            "extrapolation": "PROHIBITED",
            "invalid_load_rows": "BARRIERS",
        },
        "summary": summary,
    }


__all__ = ["align"]
