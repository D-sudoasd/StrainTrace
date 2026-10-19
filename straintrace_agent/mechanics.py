"""Instrument alignment and fixed-reference virtual-gauge calculations."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _load_mechanics(config: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    loads = config.get("loads")
    if not loads:
        return None, {"status": "not_provided"}
    path = Path(str(loads["path"]))
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        return None, {"status": "unavailable", "reason": f"load_file_read_error: {exc}"}
    if loads.get("time_column") not in table:
        return None, {"status": "unavailable", "reason": "configured_time_column_missing", "force_status": "unavailable", "displacement_status": "unavailable"}
    raw_time = pd.to_numeric(table[loads["time_column"]], errors="coerce").to_numpy(dtype=float)
    unit_scale = {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}[loads["time_unit"]]
    aligned_time = raw_time * unit_scale + float(loads["time_offset_s"])
    values: dict[str, Any] = {}
    channel_status: dict[str, Any] = {}

    def channel(column: str | None, scale: float, output_time: str, output_value: str, channel_name: str) -> None:
        if not column:
            channel_status[channel_name] = "not_configured"
            return
        if column not in table:
            channel_status[channel_name] = "column_missing"
            return
        raw = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(aligned_time) & np.isfinite(raw)
        times = aligned_time[good]
        data = raw[good] * scale
        source_rows = np.flatnonzero(good).astype(np.int64) + 1
        order = np.argsort(times, kind="stable")
        times, data, source_rows = times[order], data[order], source_rows[order]
        if times.size < 2:
            channel_status[channel_name] = "fewer_than_two_finite_samples"
            return
        if np.any(np.diff(times) <= 0):
            channel_status[channel_name] = "duplicate_or_nonincreasing_times"
            return
        values[output_time] = times
        values[output_value] = data
        values[f"{channel_name}_source_rows"] = source_rows
        channel_status[channel_name] = "clock_unconfirmed" if not loads.get("clock_confirmed") else "available"

    force_scale = {"N": 1.0, "kN": 1000.0}.get(loads.get("force_unit"), 1.0)
    channel(loads.get("force_column"), force_scale, "force_time_s", "force_N", "force")
    displacement_scale = 1.0 if loads.get("displacement_unit") == "mm" else 1e-3
    channel(loads.get("displacement_column"), displacement_scale, "displacement_time_s", "displacement_mm", "displacement")
    overall_status = "aligned" if loads.get("clock_confirmed") else "unavailable"
    if not loads.get("clock_confirmed"):
        for name, state in list(channel_status.items()):
            if state == "available":
                channel_status[name] = "clock_unconfirmed"
    return values, {
        "status": overall_status, "reason": None if loads.get("clock_confirmed") else "clock_not_confirmed",
        "source_rows": int(len(table)), "time_offset_s": float(loads["time_offset_s"]),
        "max_gap_s": float(loads["max_gap_s"]), "area_mm2": loads.get("area_mm2"),
        "force_status": channel_status["force"], "displacement_status": channel_status["displacement"],
        "machine_displacement_source": loads.get("displacement_source"),
        "machine_gauge_length_mm": loads.get("machine_gauge_length_mm"),
    }

def _interpolate_bounded_detail(
    time_s: float, times: Sequence[float], values: Sequence[float], max_gap_s: float,
    source_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if not times.size or times.size != values.size:
        return {"value": None, "alignment_kind": "channel_unavailable", "left_time_s": None, "right_time_s": None, "left_source_row": None, "right_source_row": None}
    if source_rows is None:
        source_rows = np.arange(1, len(times) + 1, dtype=np.int64)
    source_rows = np.asarray(source_rows, dtype=np.int64)
    if source_rows.size != times.size or not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        return {"value": None, "alignment_kind": "channel_unavailable", "left_time_s": None, "right_time_s": None, "left_source_row": None, "right_source_row": None}
    index = int(np.searchsorted(times, time_s, side="left"))
    if index < len(times) and math.isclose(float(times[index]), float(time_s), rel_tol=0.0, abs_tol=1e-10):
        return {
            "value": float(values[index]), "alignment_kind": "exact_sample",
            "left_time_s": float(times[index]), "right_time_s": float(times[index]),
            "left_source_row": int(source_rows[index]), "right_source_row": int(source_rows[index]),
        }
    if index == 0 or index >= len(times):
        return {"value": None, "alignment_kind": "outside_time_support", "left_time_s": None, "right_time_s": None, "left_source_row": None, "right_source_row": None}
    left, right = index - 1, index
    gap = float(times[right] - times[left])
    support = {
        "left_time_s": float(times[left]), "right_time_s": float(times[right]),
        "left_source_row": int(source_rows[left]), "right_source_row": int(source_rows[right]),
    }
    if gap > max_gap_s:
        return {"value": None, "alignment_kind": "gap_exceeds_max_gap", **support}
    alpha = float((time_s - times[left]) / gap)
    return {"value": float(values[left] + alpha * (values[right] - values[left])), "alignment_kind": "bounded_linear_interpolation", **support}

def _interpolate_bounded(time_s: float, times: Sequence[float], values: Sequence[float], max_gap_s: float, source_rows: Sequence[int] | None = None) -> float | None:
    """Compatibility value-only wrapper; new exports retain alignment evidence."""
    return _interpolate_bounded_detail(time_s, times, values, max_gap_s, source_rows).get("value")

def _mechanics_at(
    time_s: float, mechanics: Mapping[str, Any] | None, config: Mapping[str, Any], *,
    displacement_reference_mm: float | None = None,
    displacement_reference_alignment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    loads = config.get("loads")
    if not loads:
        return {
            "force_N": None, "stress_MPa": None, "mechanics_status": "not_provided",
            "machine_displacement_mm": None, "machine_engineering_strain": None, "machine_true_strain": None,
            "machine_displacement_source": None, "force_alignment_kind": "not_provided",
            "force_bracket_left_time_s": None, "force_bracket_right_time_s": None,
            "force_bracket_left_source_row": None, "force_bracket_right_source_row": None,
            "displacement_alignment_kind": "not_provided", "displacement_bracket_left_time_s": None,
            "displacement_bracket_right_time_s": None, "displacement_bracket_left_source_row": None,
            "displacement_bracket_right_source_row": None, "displacement_reference_alignment_kind": "not_provided",
        }
    if not loads.get("clock_confirmed"):
        return {
            "force_N": None, "stress_MPa": None, "mechanics_status": "unavailable_clock_unconfirmed",
            "machine_displacement_mm": None, "machine_engineering_strain": None, "machine_true_strain": None,
            "machine_displacement_source": loads.get("displacement_source"),
            "force_alignment_kind": "clock_unconfirmed", "force_bracket_left_time_s": None,
            "force_bracket_right_time_s": None, "force_bracket_left_source_row": None, "force_bracket_right_source_row": None,
            "displacement_alignment_kind": "clock_unconfirmed", "displacement_bracket_left_time_s": None,
            "displacement_bracket_right_time_s": None, "displacement_bracket_left_source_row": None,
            "displacement_bracket_right_source_row": None, "displacement_reference_alignment_kind": "clock_unconfirmed",
        }
    max_gap = float(loads["max_gap_s"])
    if not loads.get("force_column"):
        force_detail = {"value": None, "alignment_kind": "not_configured", "left_time_s": None, "right_time_s": None, "left_source_row": None, "right_source_row": None}
    elif mechanics is None:
        force_detail = {"value": None, "alignment_kind": "channel_unavailable", "left_time_s": None, "right_time_s": None, "left_source_row": None, "right_source_row": None}
    else:
        force_detail = _interpolate_bounded_detail(
            time_s, mechanics.get("force_time_s", []), mechanics.get("force_N", []), max_gap,
            mechanics.get("force_source_rows"),
        )
    value = force_detail["value"]
    area = loads.get("area_mm2")
    stress = value / float(area) if value is not None and area is not None else None
    displacement_delta = None
    machine_engineering = None
    machine_true = None
    if not loads.get("displacement_column"):
        displacement_detail = {"value": None, "alignment_kind": "not_configured", "left_time_s": None, "right_time_s": None, "left_source_row": None, "right_source_row": None}
    elif mechanics is None:
        displacement_detail = {"value": None, "alignment_kind": "channel_unavailable", "left_time_s": None, "right_time_s": None, "left_source_row": None, "right_source_row": None}
    else:
        displacement_detail = _interpolate_bounded_detail(
            time_s, mechanics.get("displacement_time_s", []), mechanics.get("displacement_mm", []), max_gap,
            mechanics.get("displacement_source_rows"),
        )
    if displacement_detail["value"] is not None and displacement_reference_mm is not None:
        absolute_displacement = displacement_detail["value"]
        if absolute_displacement is not None:
            displacement_delta = float(absolute_displacement - displacement_reference_mm)
            gauge_length = loads.get("machine_gauge_length_mm")
            if gauge_length is not None:
                machine_engineering = displacement_delta / float(gauge_length)
                machine_true = math.log1p(machine_engineering) if machine_engineering > -1.0 else None
    reference_detail = dict(displacement_reference_alignment or {})
    reference_kind = reference_detail.get("alignment_kind", "reference_unavailable" if loads.get("displacement_column") else "not_configured")
    return {
        "force_N": value, "stress_MPa": stress,
        "mechanics_status": "aligned" if stress is not None else "force_aligned_area_unavailable" if value is not None else "force_channel_unavailable_or_outside_time_support",
        "machine_displacement_mm": displacement_delta,
        "machine_engineering_strain": machine_engineering,
        "machine_true_strain": machine_true,
        "machine_displacement_source": loads.get("displacement_source"),
        "force_alignment_kind": force_detail["alignment_kind"],
        "force_bracket_left_time_s": force_detail.get("left_time_s"),
        "force_bracket_right_time_s": force_detail.get("right_time_s"),
        "force_bracket_left_source_row": force_detail.get("left_source_row"),
        "force_bracket_right_source_row": force_detail.get("right_source_row"),
        "displacement_alignment_kind": displacement_detail["alignment_kind"],
        "displacement_bracket_left_time_s": displacement_detail.get("left_time_s"),
        "displacement_bracket_right_time_s": displacement_detail.get("right_time_s"),
        "displacement_bracket_left_source_row": displacement_detail.get("left_source_row"),
        "displacement_bracket_right_source_row": displacement_detail.get("right_source_row"),
        "displacement_reference_alignment_kind": reference_kind,
        "displacement_reference_bracket_left_time_s": reference_detail.get("left_time_s"),
        "displacement_reference_bracket_right_time_s": reference_detail.get("right_time_s"),
        "displacement_reference_bracket_left_source_row": reference_detail.get("left_source_row"),
        "displacement_reference_bracket_right_source_row": reference_detail.get("right_source_row"),
    }

def _resolve_gauge_geometry(
    gauges: Sequence[Mapping[str, Any]], roi: Sequence[int], image_shape: Sequence[int],
    subset_size: int, step: int, specimen_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Bind gauge endpoints to a fixed set of reference POIs once per plan."""
    import ezdic_core as core
    X, Y = core.build_poi_grid(roi, subset_size, step, image_shape)
    eligible = np.ones(X.shape, dtype=bool)
    if specimen_mask is not None and X.size:
        from scipy import ndimage
        supported = ndimage.minimum_filter(specimen_mask.astype(np.uint8), size=int(subset_size), mode="constant", cval=0).astype(bool)
        eligible = supported[np.rint(Y).astype(int), np.rint(X).astype(int)]
    out = []
    for gauge in gauges:
        endpoints = {}
        for side in ("a", "b"):
            roi_key, point_key = f"{side}_roi_px", f"{side}_point_px"
            if roi_key in gauge:
                x, y, width, height = (float(value) for value in gauge[roi_key])
                selected = eligible & (X >= x) & (X <= x + width) & (Y >= y) & (Y <= y + height)
                indices = np.flatnonzero(selected.ravel()).astype(int).tolist()
            else:
                target_x, target_y = (float(value) for value in gauge[point_key])
                support_indices = np.flatnonzero(eligible.ravel())
                if support_indices.size:
                    distances = np.hypot(X.ravel()[support_indices] - target_x, Y.ravel()[support_indices] - target_y)
                    nearest = int(np.argmin(distances))
                    indices = [int(support_indices[nearest])] if float(distances[nearest]) <= float(gauge.get("max_distance_px", 0.0)) else []
                else:
                    indices = []
            if indices:
                endpoints[side] = {
                    "poi_indices": indices,
                    "reference_xy": [float(np.mean(X.ravel()[indices])), float(np.mean(Y.ravel()[indices]))],
                }
            else:
                endpoints[side] = {"poi_indices": [], "reference_xy": None}
        valid = endpoints["a"]["reference_xy"] is not None and endpoints["b"]["reference_xy"] is not None
        initial_length = None
        direction = None
        if valid:
            dx = endpoints["b"]["reference_xy"][0] - endpoints["a"]["reference_xy"][0]
            dy = endpoints["b"]["reference_xy"][1] - endpoints["a"]["reference_xy"][1]
            if gauge["axis"] == "x":
                initial_length = abs(dx)
                direction = [1.0 if dx >= 0 else -1.0, 0.0]
            elif gauge["axis"] == "y":
                initial_length = abs(dy)
                direction = [0.0, 1.0 if dy >= 0 else -1.0]
            else:
                initial_length = math.hypot(dx, dy)
                direction = [dx / initial_length, dy / initial_length] if initial_length > 0 else None
            valid = bool(initial_length and initial_length > 0 and direction is not None)
        out.append({
            "name": gauge["name"], "axis": gauge["axis"],
            "geometry_status": gauge.get("geometry_status", "user_defined"),
            "a": endpoints["a"], "b": endpoints["b"],
            "initial_length_px": float(initial_length) if valid else None,
            "direction": direction if valid else None,
            "reference_status": "resolved" if valid else "reference_support_missing",
            "endpoint_policy": "fixed_reference_poi_set_all_pois_must_be_valid",
        })
    return out

def _measure_gauges(field: Mapping[str, Any], reference_gauges: Sequence[Mapping[str, Any]], *, quality_ok: bool) -> list[dict[str, Any]]:
    valid = np.asarray(field.get("valid", []), dtype=bool).ravel()
    u = np.asarray(field.get("u_raw", field.get("u", [])), dtype=float).ravel()
    v = np.asarray(field.get("v_raw", field.get("v", [])), dtype=float).ravel()
    result = []
    for gauge in reference_gauges:
        a_indices = np.asarray(gauge["a"]["poi_indices"], dtype=int)
        b_indices = np.asarray(gauge["b"]["poi_indices"], dtype=int)
        a_ok = bool(a_indices.size and a_indices.max() < valid.size and valid[a_indices].all() and np.isfinite(u[a_indices]).all() and np.isfinite(v[a_indices]).all())
        b_ok = bool(b_indices.size and b_indices.max() < valid.size and valid[b_indices].all() and np.isfinite(u[b_indices]).all() and np.isfinite(v[b_indices]).all())
        base = {
            "gauge_name": gauge["name"], "axis": gauge["axis"], "geometry_status": gauge.get("geometry_status"),
            "a_expected_poi_count": int(a_indices.size), "b_expected_poi_count": int(b_indices.size),
            "a_valid_poi_count": int(valid[a_indices].sum()) if a_indices.size and a_indices.max() < valid.size else 0,
            "b_valid_poi_count": int(valid[b_indices].sum()) if b_indices.size and b_indices.max() < valid.size else 0,
            "initial_length_px": gauge.get("initial_length_px"), "endpoint_policy": gauge.get("endpoint_policy"),
        }
        if gauge.get("reference_status") != "resolved" or not a_ok or not b_ok:
            result.append({**base, "status": "invalid_endpoint_support", "engineering_strain": None, "true_strain": None})
            continue
        a_u, a_v = float(np.mean(u[a_indices])), float(np.mean(v[a_indices]))
        b_u, b_v = float(np.mean(u[b_indices])), float(np.mean(v[b_indices]))
        du, dv = b_u - a_u, b_v - a_v
        length = float(gauge["initial_length_px"])
        if gauge["axis"] == "distance":
            dx = float(gauge["b"]["reference_xy"][0] - gauge["a"]["reference_xy"][0])
            dy = float(gauge["b"]["reference_xy"][1] - gauge["a"]["reference_xy"][1])
            engineering = math.hypot(dx + du, dy + dv) / length - 1.0
        else:
            engineering = float(np.dot(np.array([du, dv]), np.asarray(gauge["direction"], dtype=float)) / length)
        true = math.log1p(engineering) if engineering > -1.0 else None
        result.append({**base, "status": "valid" if quality_ok else "quality_rejected", "engineering_strain": engineering if quality_ok else None, "true_strain": true if quality_ok else None})
    return result

def _gauge_group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "valid" and row.get("engineering_strain") is not None]
    summary: dict[str, Any] = {"dic_virtual_gauge_engineering_strain": None, "dic_virtual_gauge_engineering_strain_sd": None, "dic_virtual_gauge_true_strain": None, "dic_virtual_gauge_valid_count": 0, "dic_virtual_gauge_total_count": len(rows)}
    for axis in ("x", "y"):
        group = [row for row in valid if row.get("axis") == axis]
        values = [float(row["engineering_strain"]) for row in group]
        truths = [float(row["true_strain"]) for row in group if row.get("true_strain") is not None]
        summary[f"dic_virtual_gauge_engineering_strain_{axis}"] = float(np.mean(values)) if values else None
        summary[f"dic_virtual_gauge_engineering_strain_sd_{axis}"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary[f"dic_virtual_gauge_valid_count_{axis}"] = len(values)
        summary[f"dic_virtual_gauge_true_strain_{axis}"] = float(np.mean(truths)) if truths else None
    compatible_axes = [axis for axis in ("x", "y") if summary[f"dic_virtual_gauge_valid_count_{axis}"]]
    if len(compatible_axes) == 1:
        axis = compatible_axes[0]
        summary["dic_virtual_gauge_engineering_strain"] = summary[f"dic_virtual_gauge_engineering_strain_{axis}"]
        summary["dic_virtual_gauge_engineering_strain_sd"] = summary[f"dic_virtual_gauge_engineering_strain_sd_{axis}"]
        summary["dic_virtual_gauge_true_strain"] = summary[f"dic_virtual_gauge_true_strain_{axis}"]
        summary["dic_virtual_gauge_valid_count"] = summary[f"dic_virtual_gauge_valid_count_{axis}"]
    return summary

def _auto_gauge_candidates(roi: Sequence[int], subset: int, step: int) -> list[dict[str, Any]]:
    x, y, width, height = (int(value) for value in roi)
    length = int(round(width * 0.6))
    if length < subset * 2 or height < subset * 2:
        return []
    radius = max(1, int(round((subset + 2 * step) / 2)))
    result = []
    for index, fraction in enumerate((0.3, 0.5, 0.7), 1):
        center_y = min(y + height - radius - 1, max(y + radius, y + int(round(height * fraction))))
        a_x = x + int(round(width * 0.2))
        b_x = a_x + length
        result.append({
            "name": f"auto_candidate_{index}", "axis": "x",
            "a_roi_px": [max(0, a_x - radius), max(0, center_y - radius), 2 * radius + 1, 2 * radius + 1],
            "b_roi_px": [max(0, b_x - radius), max(0, center_y - radius), 2 * radius + 1, 2 * radius + 1],
            "geometry_status": "automatic_candidate_unconfirmed_physical_specimen",
            "estimated_initial_length_px": length,
        })
    return result
