"""Inspect a load table and preview its alignment against committed images."""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from statistics import median

from straintrace_campaign._time_io import classify_time_value, finite_number, sniff_csv_delimiter
from straintrace_campaign.models import CampaignError

from .config import CaptureError
from .handoff import select_camera
from .storage import DATABASE, ExperimentReader
from .synchronization import LOAD_TIME_UNITS, LoadAlignment, LoadClockOrigin, capture_axis


_TIME_HINTS = {"秒": "s", "毫秒": "ms", "微秒": "us", "纳秒": "ns", **{unit: unit for unit in LOAD_TIME_UNITS}}
_FORCE_HINTS = {"n": "N", "kn": "kN", "牛": "N", "牛顿": "N", "千牛": "kN", "千牛顿": "kN"}
_DISPLACEMENT_HINTS = {"mm": "mm", "um": "um", "m": "m", "毫米": "mm", "微米": "um", "米": "m"}
_COLUMN_NAMES = {
    "time": {"t", "time", "relative_time", "elapsed_time", "时间", "相对时间", "试验时间", "测试时间", "运行时间"},
    "absolute": {"timestamp", "timestamp_unix", "unix", "epoch", "unix_time", "utc", "timestamp_utc",
                 "time_utc", "iso8601", "absolute_time", "绝对时间", "时间戳"},
    "force": {"force", "load", "load_force", "力", "载荷", "负荷", "试验力", "拉力", "压力"},
    "displacement": {"displacement", "extension", "stroke", "travel", "crosshead", "位移", "伸长", "横梁位移"},
}
_UNIT_SUFFIX = re.compile(r"(?:\s*\((?P<paren>[^()]+)\)|\s*\[(?P<bracket>[^\[\]]+)\]|(?P<sep>[_/\s][^_/\s()\[\]]+))$")


def column_hint(header):
    text = unicodedata.normalize("NFKC", header).strip().casefold().replace("μ", "u")
    suffix = _UNIT_SUFFIX.search(text)
    unit = None
    if suffix:
        candidate = (suffix["paren"] or suffix["bracket"] or suffix["sep"][1:]).strip()
        if candidate in _TIME_HINTS or candidate in _FORCE_HINTS or candidate in _DISPLACEMENT_HINTS:
            text, unit = text[:suffix.start()].rstrip(), candidate
    text = re.sub(r"[\s-]+", "_", text)
    kind = next((name for name, synonyms in _COLUMN_NAMES.items() if text in synonyms), None)
    units = {"time": _TIME_HINTS, "absolute": _TIME_HINTS, "force": _FORCE_HINTS, "displacement": _DISPLACEMENT_HINTS}
    return kind, units.get(kind, {}).get(unit)


@dataclass(frozen=True)
class LoadCsvInfo:
    path: Path
    encoding: str
    headers: tuple[str, ...]
    samples: tuple[dict, ...]

    def force_unit_hint(self, field):
        return column_hint(field)[1] or (
            "column" if {"force_unit", "load_unit", "unit"}.intersection(self.headers) else "")

    def suggest(self):
        hints = {field: column_hint(field) for field in self.headers}

        def only(*kinds):
            fields = [field for field, (kind, _) in hints.items() if kind in kinds]
            return fields[0] if len(fields) == 1 else ""

        time, force, displacement = only("time", "absolute"), only("force"), only("displacement")
        form, unit, gap = self.time_settings(time)
        return {"time": time, "time_format": form, "time_unit": unit, "gap": gap, "force": force,
                "force_unit": self.force_unit_hint(force), "displacement": displacement,
                "displacement_unit": column_hint(displacement)[1] or ""}

    def time_settings(self, field):
        if not field:
            return "", "", None
        kind, unit = column_hint(field)
        form = "absolute" if kind == "absolute" else "relative" if kind == "time" and unit else ""
        values = [row[field] for row in self.samples if row.get(field) not in ("", None)]
        times = [finite_number(value) for value in values]
        if times and all(value is not None for value in times):
            if not form or not unit:
                return form, unit or "", None
            seconds = [value * LOAD_TIME_UNITS[unit] for value in times]
        elif values:
            seconds = [classify_time_value(value)["timestamp_unix_s"] for value in values]
            if any(value is None for value in seconds):
                return "", "", None
            form, unit = "absolute", "s"
        else:
            return form, unit or "", None
        steps = [right - left for left, right in zip(seconds, seconds[1:]) if right > left]
        gap = float(f"{median(steps) * 3:.3g}") if len(steps) >= 2 else None
        return form, unit, gap


def inspect_load_csv(path):
    """Read a small sample and detect a supported text encoding."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        bom = stream.read(3)
    encodings = ("utf-16",) if bom.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            with path.open(encoding=encoding, newline="") as stream:
                sample = stream.read(8192)
                stream.seek(0)
                reader = csv.DictReader(stream, delimiter=sniff_csv_delimiter(sample))
                headers = tuple(reader.fieldnames or ())
                rows = tuple(dict(row) for row in islice(reader, 16))
        except UnicodeError:
            continue
        if (len(headers) < 2 or any(not header.strip() or "\x00" in header for header in headers)
                or len({header.strip() for header in headers}) != len(headers)):
            raise CaptureError("LOAD_HEADER_INVALID", "首行需要不同的列名，至少包含时间列和载荷列。")
        return LoadCsvInfo(path, encoding, headers, rows)
    raise CaptureError("LOAD_ENCODING_INVALID", "请将载荷表保存为 UTF-8、GB18030 或带字节顺序标记的 UTF-16 文本。")


def suggested_capture_basis(experiment, camera):
    if not (Path(experiment) / DATABASE).is_file():
        return "exposure-midpoint"
    with ExperimentReader(experiment) as reader:
        if not reader.cameras():
            return "exposure-midpoint"
        found = reader.frames(camera_id=select_camera(reader, camera), newest_first=True, limit=3)
    return "host-receive" if found and not any(capture_axis(frame)["timestamp_unix_ns"] is not None
                                               for frame in found) else "exposure-midpoint"


def table_alignment(info, experiment, camera, *, time_column, force_column, force_unit, displacement_column=None,
                    displacement_unit=None, time_format, time_unit, basis, max_gap_s, sync_frame="", sync_time=""):
    selected = [time_column, force_column] + ([displacement_column] if displacement_column else [])
    if any(column not in info.headers for column in selected) or len(set(selected)) != len(selected):
        raise CaptureError("LOAD_COLUMNS_REQUIRED", "请分别选择时间列、载荷列，以及可选的位移列。")
    if time_format not in ("absolute", "relative") or time_unit not in LOAD_TIME_UNITS:
        raise CaptureError("LOAD_TIME_FORMAT_INVALID", "请选择时间格式和数字时间的单位。")
    if force_unit not in ("N", "kN", "column"):
        raise CaptureError("LOAD_FORCE_UNIT_REQUIRED", "请选择 N、kN，或使用载荷表中的单位列。")
    if displacement_column and displacement_unit not in ("mm", "um", "m"):
        raise CaptureError("DISPLACEMENT_UNIT_UNSUPPORTED", "请选择位移列的单位。")
    if basis not in ("exposure-midpoint", "host-receive"):
        raise CaptureError("CLOCK_BASIS_INVALID", "请选择曝光中点或主机收图时刻。")
    gap = finite_number(max_gap_s)
    if gap is None or gap <= 0:
        raise CaptureError("LOAD_MAX_GAP_INVALID", "最大插值间隔应为大于零的秒数。")
    origin = None
    if time_format == "relative":
        try:
            sequence = int(sync_frame)
        except (TypeError, ValueError):
            sequence = 0
        load_value = finite_number(sync_time)
        if not (0 < sequence < 2**63) or load_value is None:
            raise CaptureError("LOAD_SYNC_FRAME_REQUIRED", "请输入两个记录所对应的同一事件帧号和载荷表时间。")
        if not (Path(experiment) / DATABASE).is_file():
            raise CaptureError("LOAD_SYNC_FRAME_REQUIRED", "请先让采集保存同步事件对应的图像，再输入帧号。")
        with ExperimentReader(experiment) as reader:
            frame = reader.frame_at_sequence(select_camera(reader, camera), sequence)
        if frame is None:
            raise CaptureError("LOAD_SYNC_FRAME_REQUIRED", f"所选相机尚无序号 {sequence} 的已保存图像。")
        origin = LoadClockOrigin.from_frame(frame, basis, load_value * LOAD_TIME_UNITS[time_unit])
    return LoadAlignment(info.path, time_column=time_column, force_column=force_column,
                         force_unit=None if force_unit == "column" else force_unit,
                         displacement_column=displacement_column, displacement_unit=displacement_unit, basis=basis,
                         max_gap_s=gap, time_format=time_format, time_unit=time_unit, encoding=info.encoding, load_origin=origin)


def preview_mechanics(channel, experiment, camera, *, reference_sequence=None, linked_reference=False):
    try:
        channel.refresh()
    except CampaignError as error:
        if error.code != "LOAD_SOURCE_EMPTY" or channel.time_format != "absolute":
            raise
        return {"rows": 0, "valid_force_rows": 0}, []
    summary = channel.summary()
    if channel.load_origin and not summary["first_time_s"] <= channel.load_origin.load_time_s <= summary["last_time_s"]:
        raise CaptureError("LOAD_SYNC_TIME_OUTSIDE", "同步事件的时间不在载荷表记录范围内。请核对数值和时间单位。")
    if not (Path(experiment) / DATABASE).is_file():
        return summary, []
    with ExperimentReader(experiment) as reader:
        if not reader.cameras():
            return summary, []
        camera = select_camera(reader, camera)
        references = [] if linked_reference else reader.reference_frames(camera, sequence=reference_sequence)
        reference = references[0] if len(references) == 1 else None
        after = reference["ordinal"] if reference is not None else 0
        first, _ = reader.next_test_frame(camera, after=after, latest=False)
        last, _ = reader.next_test_frame(camera, after=after, latest=True)
        sync = reader.frame_at_sequence(camera, int(channel.load_origin.frame_id.rsplit("/", 1)[1])) if channel.load_origin else None
    frames = {frame["ordinal"]: frame for frame in (reference, first, last, sync) if frame is not None}
    return summary, [channel.at(frame, refresh=False) for _, frame in sorted(frames.items())]
