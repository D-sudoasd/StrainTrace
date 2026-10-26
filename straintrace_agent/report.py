"""Offline reports from saved results, with a portable standard-library verifier.

No DIC solver is called here. CSV cells are retained verbatim in the report data;
plots and HTML tables are both derived from those cells. Copying this module as
``verify_report.py`` provides the same verifier without importing StrainTrace.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping


REPORT_SCHEMA = "straintrace.agent-report/v1"
PACKAGE_SCHEMA = "straintrace.agent-report-package/v1"
_COLORS = ("#2563eb", "#c2410c", "#047857", "#9333ea", "#a21caf", "#0e7490")
_MAX_TABLE_ROWS = 100
_MAX_CURVE_POINTS = 600
_MAX_CURVE_SERIES = 12
_MAX_FIELD_MAPS = 12
_NOTE = (
    "数值质量检查不是实验精度或科学验收。空间标准差反映有效测点之间的离散，"
    "不是标准误或实验不确定度。缺失值不补零、不插值；失败帧和无效测点保留。"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"Nonfinite JSON token: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def _json_sha256(value: Mapping[str, Any]) -> str:
    """Match the API's canonical hash for its validated JSON snapshots."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_snapshot_binding(config: Mapping[str, Any], plan: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    config_sha = _json_sha256(config)
    if source.get("configuration_sha256") != config_sha:
        raise ValueError("Configuration checksum does not match source manifest")
    if plan.get("configuration_sha256") != config_sha:
        raise ValueError("Configuration checksum does not match plan")
    body = dict(plan)
    saved_sha = body.pop("plan_sha256", None)
    if saved_sha != _json_sha256(body):
        raise ValueError("Plan checksum does not match its contents")
    if source.get("plan_sha256") != saved_sha:
        raise ValueError("Plan checksum does not match source manifest")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "未提供"), quote=True)


def _read_table(path: Path, relative: str) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        if len(set(columns)) != len(columns):
            raise ValueError(f"Duplicate CSV column: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return {"path": relative, "columns": columns, "rows": rows}


def _safe_file(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe package path: {relative!r}")
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes package: {relative!r}")
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError(f"Symlink in package path: {relative!r}")
    return path


def _native(value: Any) -> str:
    number = _finite(value)
    return str(int(number)) if number is not None and number.is_integer() else str(value)


def _frame_axis(source: Mapping[str, Any], summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("rows", [])
    by_native = {_native(row.get("native_frame")): row for row in rows}
    frames = list(source.get("source_images") or [])
    known = {_native(frame.get("native_frame")) for frame in frames}
    if frames and any(_native(row.get("native_frame")) not in known for row in rows):
        raise ValueError("Summary contains a frame absent from the source inventory")
    frames.extend(row for row in rows if _native(row.get("native_frame")) not in known)
    result = []
    for frame in frames:
        row = by_native.get(_native(frame.get("native_frame")), {})
        if "time_s" in row and _finite(row["time_s"]) != _finite(frame.get("time_s")):
            raise ValueError(f"Summary/source inventory time mismatch at frame {frame.get('native_frame')}")
        result.append({"native_frame": frame.get("native_frame"), "time_s": _finite(frame.get("time_s")),
                       "status": row.get("status") or "missing", "quality_status": row.get("quality_status") or "unavailable",
                       "failure_code": row.get("failure_code", ""), "failure_reason": row.get("failure_reason", "")})
    if len({_native(item["native_frame"]) for item in result}) != len(result):
        raise ValueError("Duplicate native frame identity in report inputs")
    return result


def _curves(tables: Mapping[str, Any], frames: list[dict[str, Any]], time_basis: str = "physical_time") -> list[dict[str, Any]]:
    curves = []
    frame_times = {_native(frame["native_frame"]): frame["time_s"] for frame in frames}
    for name, table in tables.items():
        if "native_frame" not in table["columns"]:
            continue
        for row in table["rows"]:
            native = _native(row.get("native_frame"))
            if native not in frame_times:
                raise ValueError(f"{name} contains a frame absent from the source inventory: {native}")
            if "time_s" in row and _finite(row["time_s"]) != frame_times[native]:
                raise ValueError(f"{name}/source inventory time mismatch at frame {native}")
    specs = [("summary", "force_N", "载荷", "N", (), None),
             ("summary", "stress_MPa", "工程应力", "MPa", (), None),
             ("summary", "machine_engineering_strain", "试验机位移换算工程应变", "dimensionless", (), None),
             ("gauges", "engineering_strain", "逐标距工程应变", "dimensionless", ("gauge_name", "axis"), None),
             ("gauges", "true_strain", "逐标距对数应变", "dimensionless", ("gauge_name", "axis"), None)]
    regions = tables.get("regions", {}).get("rows", [])
    components = sorted({row.get("component", "") for row in regions})
    for component in components:
        units = {row.get("unit", "") for row in regions if row.get("component") == component}
        specs.append(("regions", "mean", f"ROI {component} 均值", "/".join(sorted(units)), ("region_name", "component"), component))
    for table_name, column, title, unit, groups, component_filter in specs:
        table = tables.get(table_name, {})
        rows = table.get("rows", [])
        if column not in table.get("columns", []):
            continue
        grouped: dict[tuple[str, ...], dict[str, tuple[int, dict[str, str]]]] = {}
        for index, row in enumerate(rows):
            if component_filter is not None and row.get("component") != component_filter:
                continue
            key = tuple(row.get(name, "") for name in groups)
            native = _native(row.get("native_frame"))
            group = grouped.setdefault(key, {})
            if native in group:
                raise ValueError(f"Duplicate {table_name} measurement identity: {key}, frame {native}")
            group[native] = (index, row)
        series = []
        for key, values in grouped.items():
            points = []
            for frame in frames:
                pair = values.get(_native(frame["native_frame"]))
                index, row = pair if pair else (None, {})
                # Source CSV numbers are retained even when flagged. A failure with
                # no numeric output remains None and cannot join adjacent segments.
                points.append({"native_frame": frame["native_frame"], "time_s": frame["time_s"],
                               "value": _finite(row.get(column)), "status": row.get("status") or frame["status"],
                               "source_row_index": index})
            series.append({"name": " / ".join(key) or title, "points": points})
        curves.append({"id": f"curve-{len(curves)}", "title": title, "table": table_name, "y_column": column,
                       "x_label": "time (s)", "y_label": unit, "series": series})
        if time_basis == "frame_index":
            curves[-1].update(x_key="x_value", x_label="native frame (index; not time)")
            for series in curves[-1]["series"]:
                for point in series["points"]:
                    point["x_value"] = _finite(point["native_frame"])
    gauge_curve = next((curve for curve in curves if curve["table"] == "gauges" and curve["y_column"] == "engineering_strain"), None)
    summary = tables.get("summary", {})
    if gauge_curve and "stress_MPa" in summary.get("columns", []):
        stress_rows = {_native(row.get("native_frame")): (index, row) for index, row in enumerate(summary["rows"])}
        stress_series = []
        for series in gauge_curve["series"]:
            points = []
            for point in series["points"]:
                index, row = stress_rows.get(_native(point["native_frame"]), (None, {}))
                points.append({**point, "x_value": point["value"], "value": _finite(row.get("stress_MPa")),
                               "source_row_index": index, "x_source_row_index": point["source_row_index"]})
            stress_series.append({"name": series["name"], "points": points})
        curves.append({"id": f"curve-{len(curves)}", "title": "工程应力与逐标距工程应变", "table": "summary", "y_column": "stress_MPa",
                       "x_key": "x_value", "x_table": "gauges", "x_column": "engineering_strain",
                       "x_label": "gauge engineering strain (dimensionless)", "y_label": "MPa", "series": stress_series})
    return curves


def _geometry(config: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    rectangles = []
    analysis = plan.get("analysis_rectangle_candidate_px")
    if isinstance(analysis, (list, tuple)) and len(analysis) == 4:
        rectangles.append({"name": "DIC analysis rectangle", "xywh": list(analysis), "kind": "analysis"})
    calibration = config.get("calibration") or {}
    for region in config.get("regions") or []:
        bounds = region.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            continue
        x0, x1, y0, y1 = (float(item) for item in bounds)
        if region.get("coordinate_system", "image") == "calibrated":
            scale = _finite(calibration.get("mm_per_px"))
            origin = calibration.get("origin_px", [0, 0])
            origin_mm = calibration.get("origin_mm", [0, 0])
            signs = calibration.get("axis_sign", [1, 1])
            if scale is None or scale <= 0:
                continue
            x0, x1 = sorted(((x0 - origin_mm[0]) / scale / signs[0] + origin[0], (x1 - origin_mm[0]) / scale / signs[0] + origin[0]))
            y0, y1 = sorted(((y0 - origin_mm[1]) / scale / signs[1] + origin[1], (y1 - origin_mm[1]) / scale / signs[1] + origin[1]))
        rectangles.append({"name": region.get("name", "ROI"), "xywh": [x0, y0, x1 - x0, y1 - y0], "kind": "region"})
    for index, rect in enumerate(config.get("exclusions_px") or []):
        rectangles.append({"name": f"excluded {index + 1}", "xywh": list(rect), "kind": "excluded"})
    gauges = []
    configured = {gauge.get("name"): gauge for gauge in config.get("gauges") or []}
    for gauge in configured.values():
        for side in ("a", "b"):
            if gauge.get(f"{side}_roi_px"):
                rectangles.append({"name": f"{gauge.get('name')} {side.upper()}", "xywh": list(gauge[f"{side}_roi_px"]), "kind": "gauge_endpoint"})
    for gauge in plan.get("gauge_reference_geometry") or []:
        gauges.append({"name": gauge.get("name"), "axis": gauge.get("axis"), "a": (gauge.get("a") or {}).get("reference_xy"),
                       "b": (gauge.get("b") or {}).get("reference_xy"), "initial_length_px": gauge.get("initial_length_px"),
                       "reference_status": gauge.get("reference_status"), "endpoint_policy": gauge.get("endpoint_policy"),
                       "definition": configured.get(gauge.get("name"), {}), "resolved": gauge})
    return {"coordinate_system": "reference image pixels; x right, y down", "rectangles": rectangles, "gauges": gauges,
            "regions": config.get("regions") or [], "configured_gauges": config.get("gauges") or [], "calibration": calibration,
            "reference_native_frame": plan.get("reference_native_frame"), "reference_path": plan.get("reference_path"),
            "mask_file": config.get("mask_file"), "mask_sha256": config.get("mask_sha256"),
            "analysis_rectangle_status": plan.get("analysis_rectangle_status", "candidate_not_physical_specimen_confirmation")}


def _preview(stage: Path, geometry: dict[str, Any], source: Mapping[str, Any]) -> list[str]:
    warnings = []
    reference = geometry.get("reference_path")
    if not reference:
        return ["参考图路径未提供；空间图仅显示记录的几何位置。"]
    path = Path(reference)
    source_image = next((item for item in source.get("source_images", []) if _native(item.get("native_frame")) == _native(geometry["reference_native_frame"])), {})
    try:
        if not source_image.get("sha256") or _sha256(path) != source_image["sha256"]:
            return ["参考图 SHA-256 未与运行记录一致核实；没有嵌入图像预览。"]
        from PIL import Image, ImageOps
        with Image.open(path) as image:
            geometry["image_shape_yx"] = [image.height, image.width]
            preview = ImageOps.autocontrast(image.convert("L"))
            preview.thumbnail((1600, 1600))
            (stage / "assets").mkdir(exist_ok=True)
            preview.save(stage / "assets" / "reference.png")
            geometry["preview"] = "assets/reference.png"
        mask_path = geometry.get("mask_file")
        if mask_path:
            if not geometry.get("mask_sha256") or _sha256(Path(mask_path)) != geometry["mask_sha256"]:
                warnings.append("遮罩 SHA-256 未与运行记录一致核实；没有嵌入遮罩预览。")
            else:
                with Image.open(mask_path) as mask:
                    if [mask.height, mask.width] != geometry["image_shape_yx"]:
                        warnings.append("遮罩尺寸与参考图不一致；没有嵌入遮罩预览。")
                    else:
                        excluded = mask.convert("L").point(lambda value: 0 if value else 100)
                        overlay = Image.new("RGBA", mask.size, (220, 38, 38, 0))
                        overlay.putalpha(excluded)
                        overlay.thumbnail((1600, 1600))
                        overlay.save(stage / "assets" / "mask.png")
                        geometry["mask_preview"] = "assets/mask.png"
    except (OSError, ValueError, ImportError) as exc:
        warnings.append(f"参考图预览不可用：{type(exc).__name__}；几何坐标仍来自参数快照。")
    return warnings


def _geometry_svg(geometry: Mapping[str, Any]) -> str:
    rects, gauges = geometry["rectangles"], geometry["gauges"]
    shape = geometry.get("image_shape_yx")
    endpoints = [gauge[side] for gauge in gauges for side in ("a", "b") if gauge[side] is not None]
    width = float(shape[1]) if shape else max([r["xywh"][0] + r["xywh"][2] for r in rects] + [point[0] for point in endpoints] + [100]) * 1.1
    height = float(shape[0]) if shape else max([r["xywh"][1] + r["xywh"][3] for r in rects] + [point[1] for point in endpoints] + [100]) * 1.1
    font = max(width, height) / 55
    result = [f'<svg role="img" aria-label="ROI and gauge reference positions" viewBox="0 0 {width:g} {height:g}" xmlns="http://www.w3.org/2000/svg">',
              f'<rect width="{width:g}" height="{height:g}" fill="#e2e8f0"/>']

    def label(text: Any, x: float, y: float, color: str) -> str:
        # Label backgrounds improve contrast without moving measurement geometry.
        padding = font * .23
        text_width = font * sum(1 if ord(char) > 127 else .61 for char in str(text))
        box_width = min(width, text_width + 2 * padding)
        left = min(max(0, x), max(0, width - box_width))
        top = min(max(0, y - font - padding), max(0, height - font * 1.5))
        return (f'<g class="geometry-label"><rect x="{left:g}" y="{top:g}" width="{box_width:g}" height="{font * 1.5:g}" '
                f'rx="{padding:g}" fill="white" fill-opacity="0.96" stroke="{color}" stroke-width="0.7" vector-effect="non-scaling-stroke"/>'
                f'<text x="{left + padding:g}" y="{top + font + padding:g}" font-size="{font:g}" font-weight="600" fill="#172033">{_esc(text)}</text></g>')
    for key in ("preview", "mask_preview"):
        if geometry.get(key):
            result.append(f'<image href="{_esc(geometry[key])}" width="{width:g}" height="{height:g}"/>')
    for index, rect in enumerate(rects):
        x, y, w, h = rect["xywh"]
        color = "#ef4444" if rect["kind"] == "excluded" else _COLORS[index % len(_COLORS)]
        result.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"><title>{_esc(rect["name"])}</title></rect>')
        result.append(label(rect["name"], x, y, color))
    for index, gauge in enumerate(gauges):
        if gauge["a"] is None or gauge["b"] is None:
            continue
        ax, ay = gauge["a"]
        bx, by = gauge["b"]
        color = _COLORS[(index + 2) % len(_COLORS)]
        result.append(f'<line x1="{ax:g}" y1="{ay:g}" x2="{bx:g}" y2="{by:g}" stroke="{color}" stroke-width="3" vector-effect="non-scaling-stroke"><title>{_esc(gauge["name"])}</title></line>')
        for x, y in ((ax, ay), (bx, by)):
            result.append(f'<circle cx="{x:g}" cy="{y:g}" r="{font / 4:g}" fill="{color}" stroke="white"/>')
        result.append(label(gauge["name"], (ax + bx) / 2, (ay + by) / 2, color))
    return "".join(result) + "</svg>"


def _curve_svg(curve: Mapping[str, Any]) -> str:
    displayed_series = curve["series"][:_MAX_CURVE_SERIES]
    frame_limit = max(1, _MAX_CURVE_POINTS // max(len(displayed_series), 1))
    preview = [{**series, "points": series["points"][:frame_limit]} for series in displayed_series]
    points = [point for series in preview for point in series["points"]]
    total_frames = max((len(series["points"]) for series in curve["series"]), default=0)
    truncated = total_frames > frame_limit or len(curve["series"]) > _MAX_CURVE_SERIES
    notice = (f'<p class="notice">曲线预览：前 {min(frame_limit, total_frames)} / {total_frames} 个帧位置，'
              f'{len(preview)} / {len(curve["series"])} 条序列；后续帧未显示。完整数值及缺失状态保留在 CSV 和报告 JSON 中。图例的缺失计数覆盖全序列。</p>') if truncated else ""
    x_key = curve.get("x_key", "time_s")
    xs = [point[x_key] for point in points if point[x_key] is not None]
    ys = [point["value"] for point in points if point["value"] is not None and point[x_key] is not None]
    if not xs or not ys:
        return notice + '<p class="notice">当前显示范围没有可绘制的有限值。请查看逐帧状态和完整来源表。</p>'
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    dx, dy = xmax - xmin or 1, ymax - ymin or max(abs(ymin) * .1, .001)
    ymin, ymax = ymin - .08 * dy, ymax + .08 * dy
    def xy(point: Mapping[str, Any]) -> tuple[float, float]:
        return 80 + 720 * (point[x_key] - xmin) / dx, 300 - 240 * (point["value"] - ymin) / (ymax - ymin)
    result = ['<svg class="curve" role="img" viewBox="0 0 860 365" xmlns="http://www.w3.org/2000/svg">',
              f'<title>{_esc(curve["title"])}</title><path d="M80 45 V300 H810" fill="none" stroke="#64748b"/>']
    for i in range(5):
        value = ymin + (ymax - ymin) * i / 4
        y = 300 - 240 * i / 4
        result.append(f'<path d="M80 {y:g} H810" stroke="#e2e8f0"/><text x="72" y="{y + 4:g}" text-anchor="end" font-size="12">{value:.5g}</text>')
    result.append(f'<text x="80" y="321" font-size="12">{xmin:g}</text><text x="800" y="321" text-anchor="end" font-size="12">{xmax:g}</text><text x="440" y="348" text-anchor="middle">{_esc(curve["x_label"])}</text><text x="80" y="25">{_esc(curve["y_label"])}</text>')
    for index, series in enumerate(preview):
        color = _COLORS[index % len(_COLORS)]
        previous = None
        for point in series["points"]:
            if point[x_key] is None or point["value"] is None:
                previous = None
                if point[x_key] is not None:
                    x = 80 + 720 * (point[x_key] - xmin) / dx
                    result.append(f'<path d="M{x - 3:.4f} 297 l6 6 m0 -6 l-6 6" stroke="{color}"><title>frame {_esc(point["native_frame"])}: missing / {_esc(point["status"])}</title></path>')
                continue
            x, y = xy(point)
            # Native-frame discontinuities are not assumed to be sampled data.
            if previous is not None:
                previous_point, px, py = previous
                a, b = _finite(previous_point["native_frame"]), _finite(point["native_frame"])
                increasing = (point["time_s"] is not None and previous_point["time_s"] is not None and point["time_s"] > previous_point["time_s"])
                if curve["x_label"] == "native frame (index; not time)":
                    increasing = point[x_key] > previous_point[x_key]
                if a is not None and b == a + 1 and increasing:
                    result.append(f'<path d="M{px:.4f} {py:.4f} L{x:.4f} {y:.4f}" fill="none" stroke="{color}"/>')
            result.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="3" fill="{color}" data-source-row="{point["source_row_index"]}" data-value="{point["value"]}"><title>{_esc(series["name"])}; frame {_esc(point["native_frame"])}; {_esc(point["value"])}; {_esc(point["status"])}</title></circle>')
            previous = (point, x, y)
    result.append("</svg><ul class=legend>")
    for index, series in enumerate(displayed_series):
        missing = sum(point["value"] is None or point[x_key] is None for point in series["points"])
        result.append(f'<li style="color:{_COLORS[index % len(_COLORS)]}">{_esc(series["name"])} · 缺失 {missing}/{len(series["points"])}</li>')
    return notice + "".join(result) + "</ul>"


def _table(columns: list[str], rows: list[Mapping[str, Any]], *, source: str = "") -> str:
    parts = []
    if len(rows) > _MAX_TABLE_ROWS:
        parts.append(f'<p class="notice">表格预览前 {_MAX_TABLE_ROWS} / {len(rows)} 行；完整记录保留在 CSV / 报告 JSON 中。</p>')
    parts.append(f'<div class="table-scroll"><table data-source="{_esc(source)}"><thead><tr>')
    parts.extend(f"<th>{_esc(column)}</th>" for column in columns)
    parts.append("</tr></thead><tbody>")
    for index, row in enumerate(rows[:_MAX_TABLE_ROWS]):
        parts.append(f'<tr data-source-row="{index}">')
        parts.extend(f'<td>{_esc(row.get(column)) if row.get(column) not in (None, "") else "<span class=missing>缺失</span>"}</td>' for column in columns)
        parts.append("</tr>")
    return "".join(parts) + "</tbody></table></div>"


def _has_plottable_values(curve: Mapping[str, Any]) -> bool:
    x_key = curve.get("x_key", "time_s")
    return any(point["value"] is not None and point.get(x_key) is not None for series in curve["series"] for point in series["points"])


def _unavailable_reason(curve: Mapping[str, Any], data: Mapping[str, Any]) -> str:
    if curve["table"] == "summary" and curve["y_column"] in {"force_N", "stress_MPa", "machine_engineering_strain"}:
        loads = data.get("mechanics_configuration")
        if not loads:
            return "未提供试验机载荷或位移数据。"
        if data.get("time_basis") == "frame_index":
            return "采集时间未知，不能与试验机通道同步。"
        if curve["y_column"] == "stress_MPa" and loads.get("area_mm2") is None:
            return "未提供试样原始截面积，不能将载荷换算为工程应力。"
        alignment = data["quality"].get("mechanics_alignment") or {}
        return "该通道已配置，但没有可绘制的有效配对；同步诊断：" + json.dumps(alignment, ensure_ascii=False) + "。逐帧原因保留在 summary.csv。"
    statuses = sorted({point["status"] for series in curve["series"] for point in series["points"]})
    return "没有可绘制的有限值或有效横轴；测量状态：" + (", ".join(statuses) or "未保存测量") + "。完整测量及原因保留在来源 CSV。"


def _html(data: Mapping[str, Any]) -> str:
    geometry = data["geometry"]
    source, quality = data["source_manifest"], data["quality"]
    postprocessing = quality.get("postprocessing_quality")
    quality_preview = {key: value for key, value in quality.items() if key != "frames"}
    provenance_preview = {key: value for key, value in source.items() if key not in {"source_images", "checkpoint_receipts", "artifacts"}}
    styles = """body{margin:0;color:#172033;background:#f1f5f9;font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif}main{max-width:1200px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.3}h2{font-size:23px;margin-top:0}h3{font-size:18px}section,header{background:white;padding:24px;border-radius:12px;margin-bottom:20px}nav a{margin-right:18px}a{color:#1754a7}.note,.notice{background:#fff7ed;border-left:4px solid #c2410c;padding:12px}.muted{color:#52627b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}svg{display:block;width:100%;height:auto;max-height:640px}.table-scroll{overflow:auto;max-height:560px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:7px 10px;border:1px solid #dbe3ec;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#eaf0f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace;background:#f8fafc;padding:12px}.missing{color:#a34117}.legend{display:flex;gap:18px;flex-wrap:wrap}.badge{display:inline-block;padding:4px 10px;background:#edf2f7;border-radius:5px;margin:4px}details{margin:12px 0}summary{cursor:pointer;font-weight:600}@media(max-width:600px){main{padding:10px}section,header{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:24px}}"""
    parts = ["<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
             f"<title>StrainTrace 应变分析报告 · {_esc(data['sequence_name'])}</title><style>{styles}</style><main><header>",
             "<p class=muted>STRAINTRACE · 离线分析报告</p><h1>StrainTrace 应变分析报告</h1>",
             f"<p>图像序列：{_esc(data['sequence_name'])} · 参考原始帧 {_esc(geometry['reference_native_frame'])}</p>",
             f"<span class=badge>执行 {_esc(source.get('run_status'))}</span><span class=badge>预期帧 {len(data['frames'])}</span>",
             f"<span class=badge>数值质量 {'通过' if quality.get('numerical_quality_passed') is True else '需检查'}</span><span class=badge>科学验收 未评估</span>",
             f"<p class=notice>本次后处理：{_esc(postprocessing.get('status'))}；原 DIC 数值质量与本次 ROI/标距提取状态分别记录。</p>" if postprocessing else "",
             f'<p class=note>{_NOTE}</p><nav><a href=#geometry>位置</a><a href=#parameters>参数</a><a href=#curves>曲线与对比</a><a href=#quality>质量与异常</a><a href=#data>数据</a><a href=#provenance>来源与复现</a></nav></header>',
             '<section id=geometry><h2>1. 测量位置与定义</h2><p>所有叠图位置位于固定参考图坐标：原点左上，+x 向右，+y 向下；单位 px。灰度预览仅用于定位，显示缩放不会改变原始坐标。矩形候选不表示已确认试样物理边界；红色区域为排除区域。</p>',
             _geometry_svg(geometry),
             '<details><summary>ROI、标距端点与标定的完整定义</summary><pre>' + _esc(json.dumps(geometry, ensure_ascii=False, indent=2)) + '</pre></details></section>',
             '<section id=parameters><h2>2. 处理方法与参数</h2><p>固定参考帧的平面二维 DIC。位移采用像素，标定后的位置/位移采用 mm；应变均为无量纲。局部 Green–Lagrange 应变 E = (FᵀF − I)/2，微小应变 e = (∇u + ∇uᵀ)/2；Exy/exy 是张量剪切，工程剪切为其两倍。张量分量始终采用图像基底；坐标标定不会自动旋转张量。</p>',
             '<p>虚拟标距工程应变 ε = (L − L₀)/L₀；x/y 标距使用该轴投影，distance 标距使用端点间欧氏距离。对数应变 ln(1+ε) 要求 1+ε &gt; 0。试验机位移换算应变单列，不能替代局部 DIC 应变。工程应力 = force_N / area_mm2；时钟未确认或面积缺失时不生成依赖值。</p>',
             '<pre>' + _esc(json.dumps(data['selected_parameters'], ensure_ascii=False, indent=2)) + '</pre>',
             '<p>局部应变拟合跨度为 (strain_window_poi − 1) × step_px；包含相关子区后的支持跨度再加 subset_size_px。二维测量不能从这些数据单独校正离面运动。</p></section>',
             '<section id=curves><h2>3. 历程与同分量比较</h2><p>圆点为原 CSV 数值；未知值以基线叉号提示并断开连线。原始帧号不连续时也断线。每条 ROI/标距序列单独保留；不会用全场均值代替标距，或用平均值掩盖缺失。</p>',
             '<p>横轴使用原生帧号，采集时间未知；帧号不代表秒数，不能据此计算应变率或同步载荷。</p>' if data.get("time_basis") == "frame_index" else '<p>横轴时间采用已记录的秒值。</p>',
             '<p>同一应变分量的 ROI 均值用于比较局部变形的演变；比较前应检查各 ROI 的有效测点数、空间支持及质量状态。局部差异本身不能确定材料机制。应力–应变关系按原始帧身份配对，载荷同步信息见 summary.csv。</p>']
    if not data["curves"]:
        parts.append('<p class=notice>未保存可绘制的曲线列。原始结果仍可从数据区读取。</p>')
    for curve in (item for item in data["curves"] if _has_plottable_values(item)):
        parts.extend([f'<article><h3>{_esc(curve["title"])}</h3>', _curve_svg(curve), f'<p class=muted>来源 {_esc(data["tables"][curve["table"]]["path"])} · {_esc(curve["y_column"])}</p></article>'])
    unavailable = [curve for curve in data["curves"] if not _has_plottable_values(curve)]
    if unavailable:
        parts.append(f'<details class="unavailable-channels"><summary>未提供或无有效数据的通道 · {len(unavailable)} 项</summary><ul>')
        for curve in unavailable:
            parts.append(f'<li><strong>{_esc(curve["title"])}</strong>：{_esc(_unavailable_reason(curve, data))} '
                         f'<a href="{_esc(data["tables"][curve["table"]]["path"])}">完整来源 CSV</a></li>')
        parts.append('</ul></details>')
    gallery = data["field_maps"]
    parts.extend(['</section><section id=field-maps><h2>4. 已保存场图</h2>',
                  f'<p>本报告显示 {len(gallery["selected"])} / {gallery["available_count"]} 幅已保存场图。选择规则：优先 Green–Lagrange Exx/Eyy/Exy 的首、中、末帧，其余按文件顺序补足，最多 {_MAX_FIELD_MAPS} 幅。<a href="data/field_maps.json">完整索引与选择记录</a>保留各图源路径和哈希。</p>'])
    if not gallery["selected"]:
        parts.append('<p class=notice>没有已保存的场图；本报告保留曲线、CSV 和检查点索引，不触发 DIC 重算。</p>')
    parts.append('<div class=grid>')
    for entry in gallery["selected"]:
        parts.append(f'<figure><a href="{_esc(entry["package_path"])}"><img loading="lazy" style="width:100%;height:auto" src="{_esc(entry["package_path"])}" alt="{_esc(entry["caption"])}"></a><figcaption>{_esc(entry["caption"])}</figcaption></figure>')
    parts.extend(['</div><p>场图的无效测点与色标保留原输出；不同图的色标范围须以各图标注为准，未声明共同色标时不能直接以颜色比较数值。</p></section><section id=quality><h2>5. 逐帧质量与异常</h2>',
                  _table(["native_frame", "time_s", "status", "quality_status", "failure_code", "failure_reason"], data["frames"]),
                  f'<p>全部 {len(data["frames"])} 帧中，执行失败/缺失/质量需检查 {sum(frame["status"] != "complete" or frame["quality_status"] not in {"pass", "valid"} for frame in data["frames"])} 帧。<a href="data/frame_status.csv">完整逐帧状态 CSV</a> · <a href="data/quality.json">完整质量诊断 JSON</a></p>',
                  '<details><summary>整体数值质量诊断</summary><pre>' + _esc(json.dumps(quality_preview, ensure_ascii=False, indent=2)) + '</pre></details>'])
    if postprocessing:
        parts.append(f'<h3>本次后处理：{_esc(postprocessing.get("status"))}</h3><p>下列计数为逐帧 ROI/分量测量行和逐帧标距测量行，不是独立试样数量。</p>')
        counts = [{"对象": label, **postprocessing.get(key, {})} for key, label in (("region_measurements", "ROI / 分量"), ("gauge_measurements", "标距"))]
        parts.append(_table(["对象", "total", "valid", "unavailable"], counts))
    for warning in data["warnings"]:
        parts.append(f'<p class=notice>{_esc(warning)}</p>')
    parts.append('</section><section id=data><h2>6. 完整数据与原始数值数组</h2><p>表格保留 CSV 原始文本精度；空白/NaN 不转为零。页面仅提供有界预览；完整 CSV / JSON 保留所有行。</p>')
    for name, table in data["tables"].items():
        parts.extend([f'<details><summary>{_esc(name)} · {len(table["rows"])} 行 · <a href="{_esc(table["path"])}">下载 CSV</a></summary>',
                      _table(table["columns"], table["rows"], source=table["path"]), '</details>'])
    mode = f"包含 {data['included_array_count']} 个检查点 NPZ 及其已保存 JSON。" if data['include_checkpoints'] else '轻量报告未复制检查点 NPZ；使用完整 export 生成包含数组的可搬移数据包。'
    parts.extend([f'<p>{mode} NPZ 使用 NumPy 数值数组；读取时设置 allow_pickle=False。逐点 valid/strain_valid 与缺失掩码必须随数组一起使用。</p>',
                  '<a href="data/report_data.json">报告结构化数据</a> · <a href="data/geometry.json">空间定义</a></section>',
                  '<section id=provenance><h2>7. 来源、完整性与复现</h2><p>manifest.json 逐文件记录字节数与 SHA-256；原始图像序列保留外部路径和源哈希，本包不复制原始图像。校验包内完整性不代表当前外部源文件已复核，也不证明实验精度。重新计算需要原始图像及记录的软件环境。</p>',
                  '<p><a href="manifest.json">包清单</a> · <a href="verify_report.py">独立校验脚本</a> · <a href="provenance/project.json">项目参数</a> · <a href="provenance/plan.json">计算计划</a> · <a href="provenance/source_manifest.json">源文件、软件环境与代码哈希</a> · <a href="README.txt">复现说明</a></p>',
                  '<pre>python verify_report.py .</pre><details><summary>运行来源与软件环境</summary><pre>' + _esc(json.dumps(provenance_preview, ensure_ascii=False, indent=2)) + '</pre></details></section></main></html>'])
    return "\n".join(parts) + "\n"


def _field_maps(stage: Path, run_root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for relative, expected in sorted(source.get("artifacts", {}).items()):
        if not relative.startswith("plots/") or not relative.lower().endswith(".png"):
            continue
        name = PurePosixPath(relative).name
        match = re.fullmatch(r"frame_native_(\d+)_(Exx|Eyy|Exy|exx_infinitesimal|eyy_infinitesimal|exy_infinitesimal|u|v|zncc)\.png", name)
        # History images and unknown suffixes are indexed, without guessing
        # their component, frame identity, units or coordinate basis.
        native = int(match.group(1)) if match else None
        component = match.group(2) if match else None
        caption = f"{name} · 图像类型/单位未从文件名确认"
        if component:
            unit = "px" if component in {"u", "v"} else "dimensionless"
            caption = f"原生帧 {native} · {component} · {unit} · 图像基底 +x 向右、+y 向下"
        entries.append({"source_relative_path": relative, "source_path": str(run_root / relative), "sha256": expected,
                        "native_frame": native, "component": component, "caption": caption})
    selected = []
    for component in ("Exx", "Eyy", "Exy"):
        matches = sorted((entry for entry in entries if entry["component"] == component), key=lambda item: item["native_frame"])
        if matches:
            for index in dict.fromkeys((0, len(matches) // 2, len(matches) - 1)):
                selected.append(matches[index])
    # Only recognized field maps are selected; unknown images remain indexed.
    selected.extend(entry for entry in entries if entry["component"] and entry not in selected)
    selected = selected[:_MAX_FIELD_MAPS]
    copied = []
    for index, entry in enumerate(selected):
        path = _safe_file(run_root, entry["source_relative_path"])
        destination = stage / "data" / "plots" / f"{index:02d}_{path.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if _sha256(destination) != entry["sha256"]:
            raise ValueError(f"Saved field map disagrees with source manifest: {path.name}")
        copied.append({**entry, "package_path": destination.relative_to(stage).as_posix()})
    return {"selection_rule": "first_middle_last_Exx_Eyy_Exy_then_recognized_fields_max_12", "available_count": sum(entry["component"] is not None for entry in entries),
            "selected": copied, "all_saved_pngs": entries}


def _build(stage: Path, run_root: Path, config: Mapping[str, Any], plan: Mapping[str, Any], regions_csv: Path | None, include_checkpoints: bool) -> dict[str, Any]:
    source = _json(run_root / "manifest.json")
    _validate_snapshot_binding(config, plan, source)
    source["artifacts"] = {key: value for key, value in source.get("artifacts", {}).items() if not key.startswith("report/")}
    quality = _json(run_root / "quality.json") if (run_root / "quality.json").is_file() else {}
    tables = {}
    for name, source_path in (("summary", run_root / "summary.csv"), ("gauges", run_root / "gauges.csv"), ("regions", regions_csv or run_root / "regions.csv")):
        if not source_path.is_file():
            continue
        relative = f"data/{name}.csv"
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        expected = source.get("artifacts", {}).get(source_path.relative_to(run_root).as_posix()) if source_path.is_relative_to(run_root) else None
        if expected and _sha256(destination) != expected:
            raise ValueError(f"Saved CSV disagrees with source manifest: {source_path.name}")
        tables[name] = _read_table(destination, relative)
    if "summary" not in tables:
        raise ValueError("Report requires saved summary.csv")
    for name, value in (("project", dict(config)), ("plan", dict(plan)), ("source_manifest", source)):
        _write_json(stage / "provenance" / f"{name}.json", value)
    _write_json(stage / "data" / "quality.json", quality)
    geometry = _geometry(config, plan)
    warnings = _preview(stage, geometry, source)
    _write_json(stage / "data" / "geometry.json", geometry)
    source_files = []
    receipts = source.get("checkpoint_receipts") or {}
    for path in sorted((run_root / "checkpoints").glob("*")):
        if path.suffix not in {".npz", ".json"} or not path.is_file():
            continue
        relative = "checkpoints/" + path.name
        receipt = receipts.get(path.with_suffix(".json").name) or {}
        expected = receipt.get("checkpoint_sha256" if path.suffix == ".npz" else "sha256")
        record = {"source_relative_path": relative, "bytes": path.stat().st_size, "sha256": expected, "included": include_checkpoints}
        if include_checkpoints:
            destination = stage / "data" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            copied_digest = _sha256(destination)
            if expected and copied_digest != expected:
                raise ValueError(f"Checkpoint disagrees with source manifest: {path.name}")
            record["sha256"] = copied_digest
            record["package_path"] = f"data/{relative}"
        source_files.append(record)
    frames = _frame_axis(source, tables["summary"])
    frame_columns = ["native_frame", "time_s", "status", "quality_status", "failure_code", "failure_reason"]
    with (stage / "data" / "frame_status.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, frame_columns)
        writer.writeheader()
        writer.writerows(frames)
    field_maps = _field_maps(stage, run_root, source)
    _write_json(stage / "data" / "field_maps.json", field_maps)
    time_basis = config.get("time_basis", "physical_time")
    included_array_count = sum(item["included"] and item["source_relative_path"].endswith(".npz") for item in source_files)
    if include_checkpoints and not included_array_count:
        warnings.append("本次报告目录中没有可复制的 NPZ 检查点；不能据此声明包含原始数值场。")
    data = {"schema": REPORT_SCHEMA, "project_id": config.get("project_id", source.get("project_id", "StrainTrace")),
            "sequence_name": PurePosixPath(str(config.get("image_dir") or "").replace("\\", "/").rstrip("/")).name or "未命名图像序列",
            "mechanics_configuration": config.get("loads"),
            "tables": tables, "frames": frames, "curves": _curves(tables, frames, time_basis), "geometry": geometry, "quality": quality,
            "selected_parameters": plan.get("selected_parameters") or {}, "source_manifest": source, "warnings": warnings,
            "include_checkpoints": include_checkpoints, "checkpoint_files": source_files, "included_array_count": included_array_count, "time_basis": time_basis,
            "field_maps": field_maps,
            "scientific_acceptance": "not_assessed", "experimental_uncertainty_calibrated": False}
    _write_json(stage / "data" / "report_data.json", data)
    (stage / "index.html").write_text(_html(data), encoding="utf-8")
    module_path = Path(__file__)
    if not module_path.is_file():
        module_path = Path(getattr(sys, "_MEIPASS", "")) / "sources" / "straintrace_agent" / "report.py"
    if not module_path.is_file():
        raise ValueError("Report verifier source is unavailable in this distribution")
    shutil.copyfile(module_path, stage / "verify_report.py")
    (stage / "README.txt").write_text(
        "StrainTrace offline report\n\nOpen index.html locally; no network or JavaScript is required.\n"
        "Verify with Python 3.11+: python verify_report.py .\n"
        "Verification checks package bytes, CSV/report-data equivalence, frame/curve values and HTML rendering.\n"
        "It does not validate experimental accuracy, authenticate the author, or re-hash external raw images.\n"
        "Source image paths and SHA-256, code hashes and environment are in provenance/source_manifest.json.\n"
        "Restore the original source inputs and recorded software environment before rerunning DIC.\n"
        "provenance/project.json and plan.json preserve exact configuration and reference identities.\n"
        "CSV tables preserve all saved rows; blank cells are unavailable values.\n"
        "NPZ arrays (when included) must be opened with numpy.load(path, allow_pickle=False).\n"
        "To regenerate this report, call straintrace_agent.report.build_report on the saved run directory.\n",
        encoding="utf-8")
    files = {path.relative_to(stage).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
             for path in sorted(stage.rglob("*")) if path.is_file()}
    _write_json(stage / "manifest.json", {"schema": PACKAGE_SCHEMA, "files": files,
                "include_checkpoints": include_checkpoints, "run_fingerprint": source.get("run_fingerprint"),
                "external_sources_checked": False, "manifest_self_hash": "excluded to avoid a recursive checksum"})
    return data


def build_report(run_root: str | os.PathLike[str], *, config: Mapping[str, Any], plan: Mapping[str, Any],
                 regions_csv: str | os.PathLike[str] | None = None, output_dir: str | os.PathLike[str] | None = None,
                 include_checkpoints: bool = False) -> dict[str, Any]:
    """Rebuild an offline report atomically from already-saved run outputs.

    Call under the workflow's project lock, after writing the source manifest.
    ``include_checkpoints=True`` copies numeric arrays for a portable delivery;
    raw image sequences remain externally referenced by path and SHA-256.
    Existing output directories are replaced only if they carry this schema.
    """
    root = Path(run_root).resolve()
    # Reject mismatched input snapshots before creating a staging directory.
    _validate_snapshot_binding(config, plan, _json(root / "manifest.json"))
    output = Path(output_dir).resolve() if output_dir else root / "report"
    if root == output or root.is_relative_to(output):
        raise ValueError("Report output must not replace the run or an ancestor")
    if output.exists() and (not (output / "manifest.json").is_file() or _json(output / "manifest.json").get("schema") != PACKAGE_SCHEMA):
        raise ValueError("Refusing to replace a directory not owned by the report builder")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-build-", dir=output.parent))
    backup = None
    try:
        _build(stage, root, config, plan, Path(regions_csv) if regions_csv else None, include_checkpoints)
        verification = verify_report(stage)
        if not verification["valid"]:
            raise ValueError("Generated report did not verify: " + "; ".join(verification["errors"]))
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-old-", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except BaseException:
            if backup is not None:
                os.replace(backup, output)
                backup = None
            raise
        verification["path"] = str(output)
        return {"report_html": str(output / "index.html"), "report_data": str(output / "data/report_data.json"),
                "report_manifest": str(output / "manifest.json"), "verification": verification,
                "artifacts": {"report_html": str(output / "index.html"), "report_package": str(output),
                              "report_manifest": str(output / "manifest.json")}}
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def verify_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Independently check package bytes and CSV-to-report correspondence."""
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    errors = []
    checked = 0
    try:
        manifest = _json(root / "manifest.json")
        if manifest.get("schema") != PACKAGE_SCHEMA or not isinstance(manifest.get("files"), dict):
            raise ValueError("Unsupported report manifest")
        expected_files = manifest["files"]
        required = {"index.html", "data/report_data.json", "data/summary.csv", "data/geometry.json", "data/quality.json", "data/frame_status.csv", "data/field_maps.json",
                    "provenance/project.json", "provenance/plan.json", "provenance/source_manifest.json", "verify_report.py", "README.txt"}
        for missing in sorted(required - expected_files.keys()):
            errors.append(f"Required file not listed: {missing}")
        for relative, expected in expected_files.items():
            file = _safe_file(root, relative)
            if not file.is_file():
                errors.append(f"Missing file: {relative}")
            elif file.stat().st_size != expected["bytes"] or _sha256(file) != expected["sha256"]:
                errors.append(f"Checksum mismatch: {relative}")
            checked += 1
        actual = {file.relative_to(root).as_posix() for file in root.rglob("*") if file.is_file()}
        for extra in sorted(actual - set(expected_files) - {"manifest.json"}):
            errors.append(f"Unlisted file: {extra}")
        data = _json(root / "data/report_data.json")
        if data.get("schema") != REPORT_SCHEMA:
            raise ValueError("Unsupported report data schema")
        for name, table in data["tables"].items():
            if table != _read_table(_safe_file(root, table["path"]), table["path"]):
                errors.append(f"CSV/report-data mismatch: {name}")
        source = _json(root / "provenance/source_manifest.json")
        if data["source_manifest"] != source:
            errors.append("Source manifest/report-data mismatch")
        if data["geometry"] != _json(root / "data/geometry.json") or data["quality"] != _json(root / "data/quality.json"):
            errors.append("Geometry or quality/report-data mismatch")
        config = _json(root / "provenance/project.json")
        plan = _json(root / "provenance/plan.json")
        _validate_snapshot_binding(config, plan, source)
        expected_sequence = PurePosixPath(str(config.get("image_dir") or "").replace("\\", "/").rstrip("/")).name or "未命名图像序列"
        if data.get("sequence_name") != expected_sequence or data.get("mechanics_configuration") != config.get("loads"):
            errors.append("Sequence/mechanics configuration mismatch")
        if data["quality"].get("postprocessing_quality") != source.get("postprocessing_quality"):
            errors.append("Postprocessing quality/source manifest mismatch")
        if data["time_basis"] != config.get("time_basis", "physical_time") or data["selected_parameters"] != (plan.get("selected_parameters") or {}):
            errors.append("Parameters/report-data mismatch")
        geometry = _geometry(config, plan)
        if any(data["geometry"].get(key) != value for key, value in geometry.items()):
            errors.append("Geometry/configuration mismatch")
        frames = _frame_axis(source, data["tables"]["summary"])
        if frames != data["frames"] or _curves(data["tables"], frames, data["time_basis"]) != data["curves"]:
            errors.append("Frame/curve source correspondence mismatch")
        saved_frames = _read_table(root / "data/frame_status.csv", "data/frame_status.csv")["rows"]
        if saved_frames != [{key: str(value) if value is not None else "" for key, value in frame.items()} for frame in frames]:
            errors.append("Frame status CSV/report-data mismatch")
        if data["field_maps"] != _json(root / "data/field_maps.json"):
            errors.append("Field map index/report-data mismatch")
        for entry in data["field_maps"]["selected"]:
            file = _safe_file(root, entry["package_path"])
            if entry["sha256"] != source.get("artifacts", {}).get(entry["source_relative_path"]) or _sha256(file) != entry["sha256"]:
                errors.append(f"Field map source checksum mismatch: {entry['source_relative_path']}")
        if (root / "index.html").read_text(encoding="utf-8") != _html(data):
            errors.append("HTML/report-data mismatch")
        for record in data["checkpoint_files"]:
            if record["included"]:
                file = _safe_file(root, record["package_path"])
                if not file.is_file() or _sha256(file) != record["sha256"]:
                    errors.append(f"Checkpoint source checksum mismatch: {record['source_relative_path']}")
        return {"valid": not errors, "path": str(root), "checked_files": checked, "errors": errors,
                "include_checkpoints": data["include_checkpoints"], "external_sources_checked": False,
                "scientific_acceptance": "not_assessed"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        return {"valid": False, "path": str(root), "checked_files": checked, "errors": errors,
                "external_sources_checked": False, "scientific_acceptance": "not_assessed"}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = verify_report(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)
