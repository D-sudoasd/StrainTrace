"""Append-only records and CSV snapshots of provisional desktop analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from .config import CaptureError
from .load_reconcile import late_load_for_frame, prepare_load_reconciliation, saved_load_settings


LIVE_RESULTS_DIR = "live-results"
RECORD_SCHEMA = "straintrace.live-result-record/v1"
LOG_IDENTITY_KEYS = ("reference_frame_id", "reference_sha256", "experiment_id", "camera_id")
CSV_COLUMNS = (
    "结果状态", "帧类型", "计算顺序", "本轮实验标识", "原图实验标识", "相机", "原图序号", "原图标识",
    "图像部件", "原图路径", "原图 SHA-256", "参考图实验标识", "参考图标识", "参考图 SHA-256",
    "收图时间基准", "相对收图时间 (s)", "标距", "工程应变 (%)", "标距应变有效", "标距跟踪状态",
    "全场有效应变点 (%)", "全场质量通过", "全场质量说明", "载荷 (N)", "位移 (mm)", "载荷对齐依据",
    "载荷对应状态", "力学数据 SHA-256", "已分析实验图 (累计)", "实时跳过实验图 (累计)",
    "载荷表匹配时间 (s)", "载荷表时间类型", "载荷同步帧标识", "载荷同步时间 (s)",
)
LOAD_COLUMNS = ("载荷值来源", "位移值来源", "实时载荷对应状态", "实时位移对应状态",
                "实时载荷表 SHA-256", "补齐载荷表 SHA-256", "补齐采样依据",
                "补齐核对载荷状态", "补齐核对位移状态")


def live_results_directory(experiment):
    root = Path(experiment).resolve()
    directory = root / LIVE_RESULTS_DIR
    if directory.resolve() != directory:
        raise CaptureError("UNSAFE_ANALYSIS_OUTPUT", "实时结果目录必须位于本轮实验内。")
    return directory


class LiveResultLog:
    """Write from the Analysis worker, with a separate file for each start."""

    def __init__(self, experiment, *, directory=None, replay=None):
        self.experiment = Path(experiment)
        self._directory = directory
        self._replay = replay
        self.path = None
        self.error = None
        self.test_frames = 0
        self._source = None

    def append(self, row, analyzer, policy):
        if self.error is not None:
            return
        try:
            first = self.path is None
            if row["analyzed_test_frames"] != (0 if first else self.test_frames + 1):
                raise CaptureError("LIVE_RESULT_SEQUENCE_GAP", "实时结果的顺序中断。可重置跟踪，从参考图重新保存。")
            if first:
                self._source, settings = analyzer.record_context()
                if self._replay is not None:
                    settings["replay"] = self._replay
                created = datetime.now().astimezone()
            record = {
                **row, **self._source, "record_schema": RECORD_SCHEMA, "follow_mode": policy,
                "role": "reference" if first else "test",
                "frame_experiment_id": (self._source["reference_experiment_id"] if first
                                        else self._source["experiment_id"]),
                "elapsed_receive_basis": row.get("elapsed_receive_basis", "reference_frame"),
            }
            if first:
                record.update(recorded_utc=created.astimezone(timezone.utc).isoformat(), settings=settings)
            line = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            if first:
                directory = self._directory or live_results_directory(self.experiment)
                directory.mkdir(exist_ok=True)
                path = directory / f"{created:%Y-%m-%d_%H-%M-%S-%f}-{uuid4().hex[:8]}.jsonl"
            else:
                path = self.path
            with path.open("xb" if first else "ab", buffering=0) as stream:
                if stream.write(line) != len(line):
                    raise OSError("写入的结果记录不完整。")
            self.path = path
            self.test_frames = row["analyzed_test_frames"]
        except (CaptureError, OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            self.error = exc


def _percent(value):
    return None if value is None else value * 100


def _csv_value(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    if isinstance(value, bool):
        return "是" if value else "否"
    return value


def _mechanics_fields(original, late):
    if not late:
        return original, {}
    updated = {**late, **original}
    filled = {}
    for key, support in (("force_N", "force_support"), ("displacement_mm", "displacement_support")):
        if original.get(key) is None and late[key] is not None:
            updated[key], updated[support] = late[key], late[support]
            filled[key] = late[support]
    if not filled:
        return original, {}
    if "force_N" in filled or not original.get("mechanics_source"):
        updated["mechanics_source"] = late["mechanics_source"]
    return updated, filled


def _csv_rows(record, mechanics, audit=(), *, replay=False):
    quality = record["field_quality"] or {}
    load_source = mechanics.get("mechanics_source") or {}
    load_origin = load_source.get("load_clock_origin") or {}
    basis = {"reference_frame": "收到参考图", "first_current_test_frame": "收到本轮首张实验图"}
    for gauge in record["gauges"] or ({},):
        row = (
            "按序重算" if replay else "实时预览", "参考图" if record["role"] == "reference" else "实验图",
            "优先看最新" if record["follow_mode"] == "latest" else "按序逐帧",
            record["experiment_id"], record["frame_experiment_id"], record["camera_id"],
            record["sequence"], record["frame_id"], record["image_part"] or "主图", record["image_path"],
            record["image_sha256"], record["reference_experiment_id"], record["reference_frame_id"],
            record["reference_sha256"], basis.get(record["elapsed_receive_basis"]), record["elapsed_receive_s"],
            gauge.get("name"), _percent(gauge.get("engineering_strain")), gauge.get("strain_valid"),
            gauge.get("quality_status"), _percent(quality.get("strain_valid_fraction")), quality.get("scientific_ok"),
            " / ".join(quality.get("scientific_reasons", ())), mechanics.get("force_N"), mechanics.get("displacement_mm"),
            mechanics.get("time_basis"), (mechanics.get("force_support") or {}).get("status", mechanics.get("error")),
            load_source.get("sha256"), record["analyzed_test_frames"], record["skipped_committed_frames"],
            mechanics.get("load_time_s"), {"relative": "相对计时", "unix_epoch": "Unix 时间"}.get(mechanics.get("load_time_domain")),
            load_origin.get("frame_id"), load_origin.get("load_time_s"),
        )
        yield tuple(map(_csv_value, (*row, *audit)))


def _load_audit(original, late, filled, *, replay=False):
    def source_label(key):
        if key in filled:
            return "后到采样"
        return ("重算匹配" if replay else "实时匹配") if original.get(key) is not None else None

    current = late or {}
    return (
        source_label("force_N"), source_label("displacement_mm"),
        (original.get("force_support") or {}).get("status"), (original.get("displacement_support") or {}).get("status"),
        (original.get("mechanics_source") or {}).get("sha256"),
        (current.get("mechanics_source") or {}).get("sha256"),
        json.dumps(filled, ensure_ascii=False, separators=(",", ":")) if filled else None,
        (current.get("force_support") or {}).get("status") if original.get("force_N") is None else None,
        (current.get("displacement_support") or {}).get("status") if original.get("displacement_mm") is None else None,
    )


def live_log_load_source(path):
    """Read the small settings record for the export file picker."""
    with Path(path).open("rb") as stream:
        first = next(_records(stream, os.fstat(stream.fileno()).st_size), None)
    if first is None:
        raise CaptureError("LIVE_RESULT_EMPTY", "这份记录中还没有可以导出的完整结果。")
    settings = saved_load_settings(first)
    if settings is None:
        raise CaptureError("LOAD_ARCHIVE_MISSING", "这份实时记录未配置载荷对齐。可直接导出实时结果。")
    return Path(settings["path"])


def _csv_options(output, fill_load, load_table, sequence_range):
    output = Path(output)
    if output.suffix.lower() != ".csv":
        raise CaptureError("LIVE_CSV_EXTENSION", "请使用 .csv 作为导出文件的扩展名。")
    if load_table is not None and not fill_load:
        raise CaptureError("LOAD_ARCHIVE_SETTINGS", "选择载荷表时请启用补齐载荷。")
    if sequence_range is not None:
        start, end = sequence_range
        if type(start) is not int or type(end) is not int or not 1 <= start <= end < 2**63:
            raise CaptureError("INVALID_FRAME_QUERY", "请使用有效的实验帧序号范围。")
    return output


def _protect_csv_inputs(output, source, first, load_table):
    target = output.resolve()
    if target == source.resolve():
        raise CaptureError("LIVE_CSV_OVERWRITES_RECORD", "导出位置与逐帧记录相同。请选择新的文件名。")
    load = saved_load_settings(first)
    inputs = (load["path"] if load else None, load_table)
    if any(path and target == Path(path).resolve() for path in inputs):
        raise CaptureError("LIVE_CSV_OVERWRITES_LOAD", "导出位置与载荷表相同。请选择新的结果文件名。")


def _selected_records(records, sequence_range, check):
    for record in records:
        if check is not None:
            check()
        if (sequence_range is None or record["role"] == "reference"
                or sequence_range[0] <= record["sequence"] <= sequence_range[1]):
            yield record


def _write_live_csv(output, records, channel, *, incomplete_tail, sequence_range=None, check=None,
                    progress=None, verify=None):
    temp = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=output.parent, prefix=".live-results-", suffix=".part", delete=False,
        ) as sink:
            temp = Path(sink.name)
            writer = csv.writer(sink)
            writer.writerow((*CSV_COLUMNS, *LOAD_COLUMNS) if channel else CSV_COLUMNS)
            summary = _export_records(_selected_records(records, sequence_range, check), writer, channel, progress=progress)
            if verify is not None:
                verify()
            if sequence_range is not None and not summary["test_frames"]:
                raise CaptureError("LIVE_CSV_EMPTY_RANGE", "这个范围内没有已读取的实验帧。")
            summary["incomplete_tail"] = incomplete_tail
            sink.flush()
            os.fsync(sink.fileno())
        if check is not None:
            check()
        os.replace(temp, output)
        return summary
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def export_live_csv(source, output, *, fill_load=False, load_table=None, sequence_range=None):
    """Export one log prefix and, if requested, fill missing load or displacement."""
    source = Path(source)
    output = _csv_options(output, fill_load, load_table, sequence_range)
    with source.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size:
            stream.seek(size - 1)
        incomplete_tail = size > 0 and stream.read(1) != b"\n"
        first = next(_records(stream, size), None)
        if first is None:
            raise CaptureError("LIVE_RESULT_EMPTY", "这份记录中还没有可以导出的完整结果。")
        _protect_csv_inputs(output, source, first, load_table)
        receipt, digest, channel = None, None, None
        if fill_load:
            receipt, digest = hashlib.sha256(), hashlib.sha256()
            channel = prepare_load_reconciliation(first, _records(stream, size, receipt.update), load_table)

        def verify():
            if receipt.digest() != digest.digest():
                raise CaptureError("LIVE_RECORD_CHANGED", "逐帧记录在导出时发生变化，请重试。")

        return _write_live_csv(
            output, _records(stream, size, digest.update if digest else None), channel,
            incomplete_tail=incomplete_tail, sequence_range=sequence_range, verify=verify if fill_load else None,
        )


def _records(stream, end, on_line=None, *, offset=0, first_index=0, identity=None):
    stream.seek(offset)
    count = first_index
    remaining = end - offset
    while remaining:
        line = stream.readline(remaining)
        if not line.endswith(b"\n"):
            if len(line) != remaining:
                raise CaptureError("LIVE_RECORD_CHANGED", "逐帧记录在读取时被截断。")
            break
        remaining -= len(line)
        try:
            record = json.loads(line)
            role = "reference" if count == 0 else "test"
            if (record["record_schema"] != RECORD_SCHEMA or record["schema"] != "straintrace.live-analysis/v1"
                    or record["provisional"] is not True or record["role"] != role
                    or record["follow_mode"] not in ("latest", "ordered")
                    or record["analyzed_test_frames"] != count
                    or record["frame_experiment_id"] != record["reference_experiment_id" if role == "reference" else "experiment_id"]
                    or record["frame_id"] != f"{record['frame_experiment_id']}/{record['camera_id']}/{record['sequence']}"):
                raise ValueError("unexpected result sequence")
            scope = tuple(record[key] for key in LOG_IDENTITY_KEYS)
            if identity is not None and scope != identity:
                raise ValueError("reference or experiment changed")
        except (UnicodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise CaptureError("LIVE_RESULT_INVALID", f"第 {count + 1} 行的实时结果记录无效。") from exc
        if on_line is not None:
            on_line(line)
        identity = scope
        count += 1
        yield record


def _export_records(records, writer, channel, *, progress=None):
    last, count = None, 0
    replay = False
    filled_frames = filled_force_frames = filled_displacement_frames = missing_force_frames = 0
    for count, record in enumerate(records, 1):
        if count == 1 and isinstance(settings := record.get("settings"), dict):
            replay = isinstance(settings.get("replay"), dict)
        original = record.get("mechanics") or {}
        try:
            late = late_load_for_frame(channel, record) if channel else None
            mechanics, filled = _mechanics_fields(original, late)
            audit = _load_audit(original, late, filled, replay=replay) if channel else ()
            writer.writerows(_csv_rows(record, mechanics, audit, replay=replay))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise CaptureError("LIVE_RESULT_INVALID", f"第 {count} 行的实时结果记录无效。") from exc
        filled_frames += bool(filled)
        filled_force_frames += "force_N" in filled
        filled_displacement_frames += "displacement_mm" in filled
        missing_force_frames += record["role"] == "test" and mechanics.get("force_N") is None
        last = record
        if progress is not None and count > 1:
            progress(count - 1)
    if last is None:
        raise CaptureError("LIVE_RESULT_EMPTY", "这份记录中还没有可以导出的完整结果。")
    summary = {"frames": count, "test_frames": count - 1, "skipped": last["skipped_committed_frames"]}
    if channel:
        summary.update(filled_frames=filled_frames, filled_force_frames=filled_force_frames,
                       filled_displacement_frames=filled_displacement_frames, missing_force_frames=missing_force_frames,
                       load_table_sha256=channel.provenance["sha256"])
    return summary
