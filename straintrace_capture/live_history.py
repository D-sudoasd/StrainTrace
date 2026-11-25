"""Read and seek verified pages in the saved live analysis log."""

from __future__ import annotations

import hashlib
import os
from bisect import bisect_left
from contextlib import closing
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path

from .config import CaptureError
from .continuation_analysis import _reference_directory
from .live_curve import CurvePoint, available_curve_modes
from .live_results import LOG_IDENTITY_KEYS, _csv_options, _protect_csv_inputs, _records, _write_live_csv
from .load_reconcile import prepare_load_reconciliation, saved_load_settings
from .storage import ExperimentReader


PAGE_SIZE = 500


@dataclass(frozen=True, slots=True)
class HistoryPage:
    offset: int
    end: int
    first_index: int
    count: int
    first_sequence: int | None
    last_sequence: int | None
    issues: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class HistoryIndex:
    experiment: Path
    path: Path
    file_id: tuple[int, int]
    identity: tuple[str, ...]
    reference_root: Path
    reference_sequence: int
    borrowed_reference: bool
    pages: tuple[HistoryPage, ...]
    modes: tuple[str, ...]
    frames: int
    skipped: int
    incomplete_tail: bool
    load_source: Path | None
    replay: bool

    @property
    def camera(self):
        return self.identity[3]

    def _check_file(self, stream):
        info = os.fstat(stream.fileno())
        if (info.st_dev, info.st_ino) != self.file_id or info.st_size < self.pages[-1].end:
            raise CaptureError("LIVE_RECORD_CHANGED", "保存的记录文件已替换或截断。请重新读取。")

    def _page_records(self, stream, page, on_line=None):
        page_digest = hashlib.sha256()

        def update(line):
            page_digest.update(line)
            if on_line is not None:
                on_line(line)

        rows = tuple(_records(stream, page.end, update, offset=page.offset,
                              first_index=page.first_index, identity=self.identity))
        if page_digest.hexdigest() != page.sha256 or len(rows) != page.count:
            raise CaptureError("LIVE_RECORD_CHANGED", "已读取的逐帧记录发生变化。请重新读取。")
        return rows

    def verified_records(self, *, page_numbers=None, on_line=None, check=None):
        """Read the selected prefix with page verification even if the live log grows."""
        pages = self.pages if page_numbers is None else (self.pages[number] for number in page_numbers)
        with self.path.open("rb") as stream:
            self._check_file(stream)
            for page in pages:
                if check is not None:
                    check()
                yield from self._page_records(stream, page, on_line)

    def read_page(self, number):
        with self.path.open("rb") as stream:
            self._check_file(stream)
            return tuple(_history_point(row, self.experiment, self.reference_root)
                         for row in self._page_records(stream, self.pages[number]))

    def find_sequence(self, sequence):
        index = bisect_left(self.pages, sequence, key=lambda page: page.last_sequence or 0)
        if not self.borrowed_reference and sequence == self.reference_sequence:
            index = 0
        if index >= len(self.pages):
            return None
        points = self.read_page(index)
        point = next((point for point in points if point.frame.sequence == sequence
                      and not (point.is_reference and self.borrowed_reference)), None)
        return (index, points, point) if point is not None else None

    def next_issue(self, page_index, mode, *, after_sequence=0):
        column = self.modes.index(mode)
        for number in range(page_index, len(self.pages)):
            if self.pages[number].issues[column]:
                points = self.read_page(number)
                point = next((point for point in points if point.frame.sequence > after_sequence and point.needs_review(mode)), None)
                if point is not None:
                    return number, points, point
        return None


def export_history_csv(index: HistoryIndex, output, *, sequence_range=None, fill_load=False, load_table=None,
                       cancelled=None, progress=None):
    """Keep the reference and export only rows in the verified snapshot shown in history."""
    output = _csv_options(output, fill_load, load_table, sequence_range)

    def check():
        if cancelled is not None and cancelled():
            raise CaptureError("LIVE_CSV_CANCELLED", "已取消导出。")

    check()
    page_numbers = None
    if sequence_range is not None:
        first_sequence, last_sequence = sequence_range
        page_numbers = (0, *(number for number, page in enumerate(index.pages[1:], 1)
                             if first_sequence <= page.last_sequence and page.first_sequence <= last_sequence))
    with closing(index.verified_records(page_numbers=page_numbers, check=check)) as records:
        first = next(records)
        _protect_csv_inputs(output, index.path, first, load_table)
        channel = None
        if fill_load:
            with closing(index.verified_records(check=check)) as known:
                channel = prepare_load_reconciliation(first, known, load_table)
        check()
        result = _write_live_csv(output, chain((first,), records), channel, incomplete_tail=index.incomplete_tail,
                                 sequence_range=sequence_range, check=check, progress=progress)
    return {**result, "source_frames": index.frames, "source_snapshot_bytes": index.pages[-1].end}


def _history_point(row, experiment, reference_root):
    root = experiment if row["frame_experiment_id"] == row["experiment_id"] else reference_root
    try:
        point = CurvePoint.from_result(row, root, row["camera_id"])
        if (type(point.frame.sequence) is not int or point.frame.sequence <= 0
                or type(point.receipt_utc_ns) is not int):
            raise ValueError("invalid recorded sequence or receipt clock")
        return point
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise CaptureError("LIVE_RESULT_INVALID", f"第 {row['analyzed_test_frames'] + 1} 行的曲线数据无效。") from error


def _saved_reference_root(first, experiment, settings):
    if first["frame_experiment_id"] == first["experiment_id"]:
        return experiment
    source = settings.get("external_reference")
    if (not isinstance(source, dict) or source.get("experiment_id") != first["frame_experiment_id"]
            or not isinstance(source.get("directory"), str) or not source["directory"]):
        raise CaptureError("LIVE_RESULT_INVALID", "缺少原参考图的实验来源。")
    try:
        return _reference_directory(experiment, {"relative_directory": None, **source})
    except CaptureError as error:
        if error.code != "CONTINUATION_REFERENCE_SOURCE":
            raise
        # The measurement remains reviewable even if the old reference was moved.
        return Path(source["directory"])


def read_history_index(experiment, path, *, expected_id=None, page_size=PAGE_SIZE):
    """Validate a fixed prefix and keep only page offsets, summaries and hashes."""
    if type(page_size) is not int or not 1 <= page_size <= PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {PAGE_SIZE}")
    experiment, path = Path(experiment).resolve(), Path(path).resolve()
    with ExperimentReader(experiment) as reader:
        experiment_id = reader.experiment["experiment_id"]
    if expected_id is not None and experiment_id != expected_id:
        raise CaptureError("FRAME_REVIEW_CHANGED", "工作台中的实验目录已变化。")

    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        size = info.st_size
        first = next(_records(stream, size), None)
        if first is None:
            raise CaptureError("LIVE_RESULT_EMPTY", "这份记录中还没有完整结果。")
        if first["experiment_id"] != experiment_id:
            raise CaptureError("LIVE_HISTORY_OTHER_EXPERIMENT", "请选择绑定当前采集实验的逐帧记录。")
        identity = tuple(first[key] for key in LOG_IDENTITY_KEYS)
        settings = first.get("settings") or {}
        if not isinstance(settings, dict):
            raise CaptureError("LIVE_RESULT_INVALID", "参考图的测量设置无效。")
        reference_root = _saved_reference_root(first, experiment, settings)
        modes = available_curve_modes(
            strain=bool(settings.get("gauges") or first.get("gauges")),
            force=bool(settings.get("mechanics") or first.get("mechanics")),
            field=bool(settings.get("field_roi") or first.get("field_quality")),
        )
        pages, count, offset, last, last_sequence = [], 0, 0, first, 0
        while offset < size:
            digest = hashlib.sha256()
            records = _records(stream, size, digest.update, offset=offset, first_index=count, identity=identity)
            issues = [0] * len(modes)
            first_test = last_test = None
            end, page_count = offset, 0
            for row in islice(records, page_size):
                point = _history_point(row, experiment, reference_root)
                if not point.is_reference:
                    sequence = point.frame.sequence
                    if sequence <= last_sequence:
                        raise CaptureError("LIVE_RESULT_INVALID", "实验帧序号没有按采集顺序递增。")
                    last_sequence = last_test = sequence
                    if first_test is None:
                        first_test = sequence
                    for i, mode in enumerate(modes):
                        issues[i] += point.needs_review(mode)
                last, end = row, stream.tell()
                page_count += 1
            if not page_count:
                break
            pages.append(HistoryPage(offset, end, count, page_count, first_test, last_test, tuple(issues), digest.hexdigest()))
            offset, count = end, count + page_count
    skipped = last.get("skipped_committed_frames")
    if type(skipped) is not int or skipped < 0:
        raise CaptureError("LIVE_RESULT_INVALID", "记录中的实时跳过数量无效。")
    load = saved_load_settings(first)
    return HistoryIndex(experiment, path, (info.st_dev, info.st_ino), identity, reference_root, first["sequence"],
                        first["frame_experiment_id"] != experiment_id, tuple(pages), modes, count,
                        skipped, offset < size, Path(load["path"]) if load else None,
                        isinstance(settings.get("replay"), dict))
