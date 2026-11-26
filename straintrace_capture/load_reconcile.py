"""Match late load samples using the clock and source evidence in a live log."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path

from straintrace_campaign.models import CampaignError

from .config import CaptureError
from .synchronization import LoadClockOrigin, load_alignment_settings


_SHA256 = re.compile(r"[a-f0-9]{64}")


def saved_load_settings(first):
    setup = first.get("settings")
    settings = setup.get("mechanics") if isinstance(setup, dict) else None
    return settings if isinstance(settings, dict) and isinstance(settings.get("path"), str) and settings["path"] else None


def _load_channel(settings, source, run_id, camera):
    try:
        origin = None
        if data := settings.get("load_clock_origin"):
            origin = LoadClockOrigin(**data)
            identity, _, sequence = origin.frame_id.rpartition("/")
            if (identity != f"{run_id}/{camera}" or not sequence.isdecimal() or int(sequence) < 1
                    or type(origin.capture_value) is not int or type(origin.capture_rate_hz) is not int
                    or origin.capture_rate_hz <= 0 or origin.capture_unit not in ("ns", "ticks")
                    or (origin.capture_unit == "ns" and origin.capture_rate_hz != 10**9)
                    or not math.isfinite(origin.load_time_s)):
                raise ValueError("invalid saved load clock origin")
        return load_alignment_settings({
            "source": str(source), "time_column": settings["time_column"], "force_column": settings["force_column"],
            "force_unit": settings.get("force_unit_override"), "basis": settings["time_basis"],
            "max_gap_s": settings["max_gap_s"], "mode": settings["mode"], "anchors": settings.get("anchors", ()),
            "displacement_column": settings.get("displacement_column"), "displacement_unit": settings.get("displacement_unit"),
            "time_format": settings.get("time_format"), "time_unit": settings.get("time_unit", "s"),
            "encoding": settings.get("encoding", "utf-8-sig"), "load_origin": origin,
        }, base_dir=source.parent)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise CaptureError("LOAD_ARCHIVE_SETTINGS", "保存的载荷设置无效。可导出原实时结果。") from error


def _observed_prefixes(sources):
    normalized, legacy = {}, set()
    for source in sources:
        if not source:
            continue
        if not isinstance(source, dict):
            raise CaptureError("LOAD_ARCHIVE_SETTINGS", "载荷来源记录无效。")
        if prefix := source.get("sample_prefix"):
            if not isinstance(prefix, dict):
                raise CaptureError("LOAD_ARCHIVE_SETTINGS", "载荷采样校验记录无效。")
            count, digest = prefix.get("rows"), prefix.get("sha256")
            if (type(count) is not int or count <= 0 or not _SHA256.fullmatch(str(digest))
                    or normalized.setdefault(count, digest) != digest):
                raise CaptureError("LOAD_ARCHIVE_CONFLICT", "逐帧记录中的载荷表校验信息存在冲突。")
        elif digest := source.get("sha256"):
            if not _SHA256.fullmatch(str(digest)):
                raise CaptureError("LOAD_ARCHIVE_SETTINGS", "逐帧记录中的载荷表校验值无效。")
            legacy.add(digest)
    return normalized, legacy


def _verify_legacy_prefixes(channel, known):
    """Older real-time logs recorded the raw hash without the sample count."""
    pending = known - {channel.provenance["sha256"]}
    if not pending:
        return
    digest = hashlib.sha256()
    with channel.source.open("rb") as stream:
        previous_ended_line = False
        for line in stream:
            if pending and previous_ended_line and line.startswith(b"\x00"):
                # In little-endian UTF-16 the zero byte follows the newline byte.
                candidate = digest.copy()
                candidate.update(b"\x00")
                pending.discard(candidate.hexdigest())
            if pending and line.endswith(b"\n"):
                # Check endings with and without a final line break.
                for end in range(1, min(4, len(line)) + 1):
                    if line[-end] not in (0, 10, 13):
                        break
                    candidate = digest.copy()
                    candidate.update(line[:-end])
                    pending.discard(candidate.hexdigest())
            digest.update(line)
            if pending:
                pending.discard(digest.hexdigest())
            previous_ended_line = line.endswith(b"\n")
    if digest.hexdigest() != channel.provenance["sha256"]:
        raise CaptureError("MECHANICS_SOURCE_UPDATING", "载荷表在读取时发生变化。")
    if pending:
        raise CaptureError("LOAD_ARCHIVE_PREFIX_CHANGED", "当前载荷表无法核对早期记录使用的采样。请选同一轮实验的载荷表。")


def prepare_load_reconciliation(first, records, source_override=None):
    settings = saved_load_settings(first)
    if settings is None:
        raise CaptureError("LOAD_ARCHIVE_MISSING", "这份实时记录未配置载荷对齐。可直接导出实时结果。")
    source = Path(source_override or settings["path"]).resolve()
    if not source.is_file():
        raise CaptureError("LOAD_ARCHIVE_FILE_MISSING", "找不到载荷表。请选择同一轮实验更新后的载荷表。")
    run_id, camera = first["experiment_id"], first["camera_id"]
    channel = _load_channel(settings, source, run_id, camera)

    def observations():
        yield settings
        for record in records:
            observed = record.get("mechanics")
            if observed is not None and not isinstance(observed, dict):
                raise CaptureError("LOAD_ARCHIVE_SETTINGS", "实时载荷记录格式无效。")
            if observed:
                yield observed.get("mechanics_source")

    normalized, legacy = _observed_prefixes(observations())
    if not normalized and not legacy and (channel.load_origin or source != Path(settings["path"]).resolve()):
        raise CaptureError("LOAD_ARCHIVE_UNVERIFIED", "这份记录缺少可核对的载荷采样。请使用原记录的绝对时间载荷表。")
    try:
        channel.refresh()
    except CampaignError as error:
        raise CaptureError(error.code, str(error)) from error
    except (csv.Error, UnicodeError) as error:
        raise CaptureError("LOAD_ARCHIVE_TABLE_INVALID", "无法读取所选载荷表。请核对原表编码和列名。") from error
    if channel.sample_prefixes(normalized) != normalized:
        raise CaptureError("LOAD_ARCHIVE_PREFIX_CHANGED", "所选载荷表已改写或缺少之前用过的采样。请核对试验轮次。")
    _verify_legacy_prefixes(channel, legacy)
    return channel


def late_load_for_frame(channel, record):
    if record["frame_experiment_id"] != record["experiment_id"]:
        return None
    original = record.get("mechanics") or {}
    need_force = original.get("force_N") is None
    need_displacement = bool(channel.displacement_column) and original.get("displacement_mm") is None
    if not (need_force or need_displacement):
        return None
    return channel.at(record, refresh=False)
