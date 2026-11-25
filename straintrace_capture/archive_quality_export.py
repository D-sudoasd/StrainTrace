"""Portable, read-only export of an already completed archive quality scan."""

import csv
from io import TextIOWrapper
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import zipfile

from .config import CaptureError
from .frame_search import format_receipt_utc
from .output import resolve_output_path
from .preview import QUALITY_ALERT_FRACTION
from .storage import _publish_path, json_text


QUALITY_REPORT_SCHEMA = "straintrace.capture.archive-quality/v1"
QUALITY_COVERAGE_LABELS = {
    "stopped": "已停止", "full": "已逐张检查", "sampled": "已抽查",
    "failed": "检查失败", "unchecked": "未检查", "no_frames": "无已登记帧",
}
_FALLBACK_LABELS = {"image_layout": "图像画幅变化", "sensor_window": "传感器范围变化"}
_COVERAGE_COLUMNS = (
    "阶段", "相机标识", "状态", "已登记帧数", "预定检查帧数", "已查帧数", "已核对帧数", "提醒帧数",
    "首条提醒帧", "计划框横坐标", "计划框纵坐标", "计划框宽", "计划框高", "改查全图帧数", "失败代码",
)
_FRAME_COLUMNS = (
    "阶段", "相机标识", "帧号", "台账序号", "帧标识", "原图相对路径", "台账 SHA-256", "主机收图 UTC",
    "图像宽", "图像高", "像素格式", "有效位深", "图像部件", "检查范围",
    "实际框横坐标", "实际框纵坐标", "实际框宽", "实际框高", "改查全图原因", "像素采样步长",
    "近黑比例", "近饱和比例", "亮度中位占满量程比例", "核对状态", "提醒代码",
)
_PAGE_SIZE = 256


def quality_report_destination(experiment, output):
    target = resolve_output_path(output)
    if target.suffix.lower() != ".zip":
        raise CaptureError("QUALITY_REPORT_SUFFIX", str(target))
    if target.is_relative_to(Path(experiment).resolve()):
        raise CaptureError("QUALITY_REPORT_INSIDE_ARCHIVE", str(target))
    if os.path.lexists(target):
        raise CaptureError("QUALITY_REPORT_EXISTS", str(target))
    if not target.parent.is_dir():
        raise CaptureError("QUALITY_REPORT_PARENT", str(target.parent))
    return target


def _bounds(value):
    return value if value is not None else (None, None, None, None)


def _spreadsheet_text(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _coverage_row(group):
    region = group["requested_region"]
    bounds = region.get("bounds") if isinstance(region, dict) else None
    bounds = bounds if isinstance(bounds, (list, tuple)) and len(bounds) == 4 else None
    return (group["phase"], group["camera"], QUALITY_COVERAGE_LABELS[group["status"]], group["available"],
            group["selected"], group["checked"], group["verified"], group["alerts"], group["first_alert_sequence"],
            *_bounds(bounds), group["whole_frame_fallbacks"], group["error_code"])


def _frame_row(phase, camera, row):
    metrics, layout = row["metrics"], row["layout"]
    height, width = layout["shape"][:2]
    stamp = format_receipt_utc(row["received_utc_ns"])
    stamp = stamp.replace(" ", "T", 1) + "Z" if stamp != "—" else None
    roi = row["roi"] if metrics is not None else None
    scope = "计划检查框" if roi is not None else "全图" if metrics is not None else None
    return (
        phase, camera, row["sequence"], row["ordinal"], row["frame_id"], row["path"], row["sha256"], stamp,
        width, height, layout["pixel_format"], layout["significant_bits"], layout["part"],
        scope, *_bounds(roi),
        _FALLBACK_LABELS.get(row["fallback"]), metrics["sample_stride_px"] if metrics else None,
        metrics["dark_fraction"] if metrics else None, metrics["near_saturation_fraction"] if metrics else None,
        metrics["brightness_fraction"][1] if metrics else None,
        "已核对" if metrics else "待复核", "|".join(row["flags"]),
    )


def _manifest(batch):
    groups = [batch.coverage(scope) for scope in batch.options["scopes"]]
    totals = {field: sum(row[field] or 0 for row in groups) for field in ("available", "checked", "verified", "alerts")}
    totals["failed_groups"] = sum(row["status"] == "failed" for row in groups)
    totals["unchecked_groups"] = sum(row["status"] == "unchecked" for row in groups)
    return {
        "schema": QUALITY_REPORT_SCHEMA,
        "experiment_id": batch.options["experiment_id"],
        "experiment_name": batch.options["experiment_name"],
        "started_utc_ns": batch.started_utc_ns,
        "finished_utc_ns": batch.finished_utc_ns,
        "mode": "all_frames" if batch.all_frames else "sample",
        "stopped": batch.stopped,
        "pixel_thresholds": {"near_black_full_scale_fraction": .01, "near_saturation_full_scale_fraction": .99,
                             "alert_pixel_fraction": QUALITY_ALERT_FRACTION},
        "totals": totals, "groups": groups,
    }


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise CaptureError("QUALITY_REPORT_CANCELLED", "quality report export stopped")


def export_archive_quality(batch, output, *, cancelled=None, progress=None):
    """Stream both CSVs to a new ZIP. The caller keeps the batch open until this returns."""
    target = quality_report_destination(batch.root, output)
    manifest = _manifest(batch)
    total = manifest["totals"]["checked"]
    partial = None
    try:
        _check_cancelled(cancelled)
        with NamedTemporaryFile(dir=target.parent, prefix=".capture-quality-", suffix=".part", delete=False) as stream:
            partial = Path(stream.name)
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("report.json", json_text(manifest) + "\n")
                with archive.open("coverage.csv", "w", force_zip64=True) as item:
                    with TextIOWrapper(item, encoding="utf-8-sig", newline="") as text:
                        writer = csv.writer(text)
                        writer.writerow(_COVERAGE_COLUMNS)
                        for group in manifest["groups"]:
                            _check_cancelled(cancelled)
                            writer.writerow(map(_spreadsheet_text, _coverage_row(group)))
                with archive.open("frames.csv", "w", force_zip64=True) as item:
                    with TextIOWrapper(item, encoding="utf-8-sig", newline="") as text:
                        writer = csv.writer(text)
                        writer.writerow(_FRAME_COLUMNS)
                        done = 0
                        if progress is not None:
                            progress(done, total)
                        for (camera, phase), report in batch.reports.items():
                            for start in range(0, report.checked, _PAGE_SIZE):
                                _check_cancelled(cancelled)
                                rows = report.page(start, _PAGE_SIZE)
                                for row in rows:
                                    writer.writerow(map(_spreadsheet_text, _frame_row(phase, camera, row)))
                                done += len(rows)
                                if progress is not None:
                                    progress(done, total)
            stream.flush()
            os.fsync(stream.fileno())
        _check_cancelled(cancelled)
        try:
            _publish_path(partial, target, write_through=True)
        except FileExistsError as exc:
            raise CaptureError("QUALITY_REPORT_EXISTS", str(target)) from exc
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return {"output": str(target), "experiment_id": manifest["experiment_id"], "mode": manifest["mode"],
            "stopped": manifest["stopped"], "scopes": len(manifest["groups"]), **manifest["totals"]}
