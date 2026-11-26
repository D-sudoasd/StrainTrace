"""Detect threshold crossings in new records of an append-only instrument CSV."""

from __future__ import annotations

import codecs
import copy
import csv
import hashlib
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

from straintrace_campaign._time_io import finite_number, sniff_csv_delimiter

from .cameras.base import receipt_times
from .config import CaptureError, CsvTrigger
from .synchronization import ThresholdTrigger


CSV_TRIGGER_EVENT = "csv_trigger"
_READ_BYTES = 64 * 1024
_ANCHOR_BYTES = 256
_POLL_S = .05


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _line_end(raw: bytes, newline: bytes, start=0) -> int:
    index = raw.find(newline, start)
    while len(newline) == 2 and index >= 0 and index % 2:
        index = raw.find(newline, index + 1)
    return index + len(newline) if index >= 0 else -1


def _row(text, delimiter):
    return next(csv.reader([text.rstrip("\r\n")], delimiter=delimiter, strict=True))


@dataclass(frozen=True)
class CsvHeader:
    raw: bytes
    columns: tuple[str, ...]
    encoding: str
    delimiter: str
    newline: bytes

    @property
    def sha256(self) -> str:
        return _digest(self.raw)

    def check(self, rule: CsvTrigger) -> None:
        if rule.header_sha256 and rule.header_sha256 != self.sha256:
            raise CaptureError("CSV_TRIGGER_HEADER_CHANGED", "CSV 表头已变化，请在阶段设置中重新选择列。")
        for field in (rule.value_column, rule.time_column):
            if field is not None and field not in self.columns:
                raise CaptureError("CSV_TRIGGER_COLUMN_MISSING", f"CSV 表头中找不到列 {field}。")


def _read_header(stream) -> CsvHeader:
    raw = stream.read(_READ_BYTES)
    encoding, encodings = "utf-8", ("utf-8", "gb18030")
    if raw.startswith(codecs.BOM_UTF16_LE):
        encoding, encodings = "utf-16-le", ("utf-16-le",)
    elif raw.startswith(codecs.BOM_UTF16_BE):
        encoding, encodings = "utf-16-be", ("utf-16-be",)
    elif raw.startswith(codecs.BOM_UTF8):
        encodings = ("utf-8",)
    newline = "\n".encode(encoding)
    end = _line_end(raw, newline)
    if end < 0:
        raise CaptureError("CSV_TRIGGER_HEADER_INVALID", "CSV 首行必须是完整表头，并以换行符结束。")
    raw = raw[:end]
    try:
        for encoding in encodings:
            try:
                text = raw.decode(encoding).removeprefix("\ufeff")
                break
            except UnicodeError:
                continue
        else:
            raise UnicodeError("unsupported CSV encoding")
        delimiter = sniff_csv_delimiter(text)
        columns = tuple(value.strip() for value in _row(text, delimiter))
    except (UnicodeError, csv.Error) as exc:
        raise CaptureError("CSV_TRIGGER_HEADER_INVALID", "无法读取 CSV 表头，请检查编码、分隔符和引号。") from exc
    if not columns or len(set(columns)) != len(columns) or any(not key or "\x00" in key for key in columns):
        raise CaptureError("CSV_TRIGGER_HEADER_INVALID", "CSV 首行需要填写各列的名称，且列名不能重复。")
    return CsvHeader(raw, columns, encoding, delimiter, newline)


def _source(path) -> Path:
    path = Path(path).expanduser()
    try:
        if stat.S_ISREG(path.stat().st_mode):
            return path
    except OSError as exc:
        raise CaptureError("CSV_TRIGGER_SOURCE_UNAVAILABLE", f"无法读取试验机 CSV：{path}（{exc.strerror or exc}）") from exc
    raise CaptureError("CSV_TRIGGER_SOURCE_UNAVAILABLE", f"请选择试验机正在追加的 CSV 文件：{path}")


def inspect_csv_trigger(path) -> CsvHeader:
    path = _source(path)
    try:
        with path.open("rb") as stream:
            return _read_header(stream)
    except OSError as exc:
        raise CaptureError("CSV_TRIGGER_SOURCE_UNAVAILABLE", f"无法打开试验机 CSV：{exc.strerror or exc}") from exc


def check_csv_triggers(phases) -> list[dict]:
    issues = []
    for index, phase in enumerate(phases, 1):
        if rule := phase.csv_trigger:
            try:
                inspect_csv_trigger(rule.path).check(rule)
            except CaptureError as exc:
                issues.append({**exc.as_dict(), "severity": "block", "phase": index, "path": rule.path})
    return issues


class _Tail:
    def __init__(self, rule: CsvTrigger):
        self.path = _source(rule.path)
        with self.path.open("rb") as stream:
            initial = os.fstat(stream.fileno())
            self.identity = (initial.st_dev, initial.st_ino)
            self.header = _read_header(stream)
            self.header.check(rule)
            size = initial.st_size
            newline = self.header.newline
            # Let the next read restore a partially written UTF-16 code unit.
            self.offset = size - size % len(newline)
            if self.offset < len(self.header.raw):
                raise CaptureError("CSV_TRIGGER_SOURCE_CHANGED", "读取起点时，CSV 内容发生了变化。")
            self.anchor = self._checkpoint(stream)
            self.discard_partial = self.offset != size or not self.anchor.endswith(newline)
        self.pending = b""

    def _checkpoint(self, stream):
        stream.seek(max(0, self.offset - _ANCHOR_BYTES))
        return stream.read(min(self.offset, _ANCHOR_BYTES))

    def read(self):
        with self.path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if ((info.st_dev, info.st_ino) != self.identity or info.st_size < self.offset
                    or stream.read(len(self.header.raw)) != self.header.raw or self._checkpoint(stream) != self.anchor):
                raise CaptureError("CSV_TRIGGER_SOURCE_CHANGED", "已监看部分被截断、改写或替换，阈值监看已停用。")
            stream.seek(self.offset)
            extra = stream.read(_READ_BYTES)
        if not extra:
            return
        raw, base = self.pending + extra, self.offset - len(self.pending)
        self.anchor = (self.anchor + extra)[-_ANCHOR_BYTES:]
        self.offset += len(extra)
        cursor = 0
        while (end := _line_end(raw, self.header.newline, cursor)) >= 0:
            if end - cursor > _READ_BYTES and not self.discard_partial:
                raise CaptureError("CSV_TRIGGER_ROW_INVALID", "CSV 数据行超过 64 KiB，阈值监看已停用。")
            if not self.discard_partial:
                yield base + cursor, raw[cursor:end]
            self.discard_partial, cursor = False, end
        self.pending = raw[cursor:]
        if len(self.pending) > _READ_BYTES and not self.discard_partial:
            raise CaptureError("CSV_TRIGGER_ROW_INVALID", "CSV 数据行超过 64 KiB，阈值监看已停用。")
        if self.discard_partial:
            leftover = len(self.pending) % len(self.header.newline)
            self.pending = self.pending[len(self.pending) - leftover:]


class CsvTriggerWatch:
    """Own the reader for one armed phase. The controller remains the trigger arbiter."""

    def __init__(self, rule: CsvTrigger, request, *, repeat=False):
        self.rule, self.request = rule, request
        self.repeat = repeat
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = {"state": "opening", "path": rule.path, "value_column": rule.value_column,
                       "time_column": rule.time_column, "quantity": rule.quantity, "unit": rule.unit,
                       "threshold": rule.threshold, "direction": rule.direction,
                       "samples": 0, "invalid_rows": 0, "early_crossings": 0}
        if repeat:
            self._state.update(events_accepted=0, busy_crossings=0)
        self._thread = threading.Thread(target=self._run, name="capture-csv-trigger", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=1)
        with self._lock:
            if self._state["state"] in ("opening", "watching"):
                self._state["state"] = "finished"

    def status(self):
        with self._lock:
            return copy.deepcopy(self._state)

    def _update(self, **changes):
        with self._lock:
            if not self._stop.is_set():
                self._state.update(changes)

    def _dispatch(self, source, *, payload):
        if self._stop.is_set():
            raise CaptureError("NOT_ARMED", "phase finished")
        utc, monotonic = receipt_times()
        evidence = {**payload, "detected_utc_ns": utc, "detected_monotonic_ns": monotonic,
                    "basis": "host_csv_observation", "path": self.rule.path,
                    "value_column": self.rule.value_column, "time_column": self.rule.time_column,
                    "header_sha256": self._header_sha256, "previous_sample": self._previous, "sample": self._current}
        signal_id = self.request(source, payload=evidence)
        with self._lock:
            if self.repeat:
                self._state["events_accepted"] += 1
            else:
                self._state.update(state="triggered", signal_id=signal_id, crossing=evidence)
        return signal_id

    def _run(self):
        rule = self.rule
        try:
            tail = _Tail(rule)
            header = tail.header
            self._header_sha256 = header.sha256
            value_index = header.columns.index(rule.value_column)
            time_index = header.columns.index(rule.time_column) if rule.time_column is not None else None
            self._previous = self._current = None
            threshold = ThresholdTrigger(self._dispatch, threshold=rule.threshold, direction=rule.direction,
                                         quantity=rule.quantity, unit=rule.unit, source="instrument_csv_threshold")
            self._update(state="watching", header_sha256=header.sha256, start_offset=tail.offset)
            samples = invalid_rows = early = busy = 0
            while not self._stop.wait(_POLL_S):
                for offset, raw in tail.read():
                    if self._stop.is_set():
                        return
                    row = _row(raw.decode(header.encoding), header.delimiter)
                    value = finite_number(row[value_index]) if len(row) == len(header.columns) else None
                    if value is None:
                        invalid_rows += 1
                        self._update(invalid_rows=invalid_rows)
                        self._previous = self._current = None
                        threshold.observe(float("nan"))
                        continue
                    sample = {"offset": offset, "length": len(raw), "sha256": _digest(raw), "value": value}
                    source_time = None
                    if time_index is not None:
                        sample["source_time_text"] = row[time_index]
                        source_time = {"column": rule.time_column, "text": row[time_index], "basis": "instrument_csv_text"}
                    self._previous, self._current = self._current, sample
                    samples += 1
                    self._update(samples=samples, last_value=value)
                    try:
                        if threshold.observe(value, source_time=source_time) is not None and not self.repeat:
                            return
                    except CaptureError as exc:
                        if exc.code in ("PREBUFFER_NOT_READY", "PHASE_PAUSE_TRANSITION"):
                            early += 1
                            self._update(early_crossings=early, last_rejection=exc.code)
                        elif exc.code == "EVENT_ALREADY_RECEIVED" and self.repeat:
                            busy += 1
                            self._update(busy_crossings=busy, last_rejection=exc.code)
                        elif exc.code in ("NOT_ARMED", "EVENT_ALREADY_RECEIVED", "SIGNAL_PHASE_CHANGED",
                                          "SIGNAL_PHASE_EXPIRED", "EVENT_TARGET_REACHED"):
                            self._update(state="finished")
                            return
                        else:
                            raise
        except CaptureError as exc:
            self._update(state="fault", error=exc.as_dict())
        except (UnicodeError, csv.Error) as exc:
            self._update(state="fault", error={"code": "CSV_TRIGGER_ROW_INVALID",
                                               "message": f"CSV 数据行无法读取，阈值监看已停用（{exc}）。"})
        except OSError as exc:
            self._update(state="fault", error={"code": "CSV_TRIGGER_SOURCE_UNAVAILABLE",
                                               "message": f"试验机 CSV 已无法读取（{exc.strerror or exc}）。"})
