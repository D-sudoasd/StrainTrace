"""Exact time and frame identity for the saved-frame search workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config import CaptureError

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_NANOSECONDS = 1_000_000_000
_UTC_TEXT = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2})"
                       r"(?:\.([0-9]{1,9}))?(Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])\Z")
_SEQUENCES = re.compile(r"([0-9]{1,19})(?:\s*[-–]\s*([0-9]{1,19}))?\Z")


def parse_receipt_utc(text: str) -> int:
    """Parse an ISO timestamp with an explicit offset without losing nanoseconds."""
    match = _UTC_TEXT.fullmatch(text.strip())
    if match is not None:
        base, fraction, offset = match.groups()
        try:
            elapsed = datetime.fromisoformat(base + offset) - _EPOCH
        except (ValueError, OverflowError):
            pass
        else:
            seconds = elapsed.days * 86400 + elapsed.seconds
            return seconds * _NANOSECONDS + int((fraction or "").ljust(9, "0"))
    raise CaptureError("INVALID_FRAME_QUERY", "收图时间须带时区，例如 2026-11-25T09:30:00.123Z 或 2026-11-25T17:30:00.123+08:00。")


def format_receipt_utc(value: int | None, *, precision=9) -> str:
    if value is None:
        return "—"
    seconds, nanoseconds = divmod(value, _NANOSECONDS)
    try:
        whole = _EPOCH + timedelta(seconds=seconds)
    except OverflowError:
        return "—"
    return whole.strftime("%Y-%m-%d %H:%M:%S") + "." + f"{nanoseconds:09d}"[:precision]


def parse_sequence_range(text: str) -> tuple[int | None, int | None]:
    text = text.strip()
    if not text:
        return None, None
    match = _SEQUENCES.fullmatch(text)
    if match:
        start, end = int(match[1]), int(match[2] or match[1])
        if 1 <= start <= end < (1 << 63):
            return start, end
    raise CaptureError("INVALID_FRAME_QUERY", "请输入正整数序号，例如 42。查询连续范围时可填 40-80。")


@dataclass(frozen=True, slots=True)
class FrameLocation:
    experiment_id: str
    camera_id: str
    sequence: int
    ordinal: int
    frame_id: str
    sha256: str

    def read(self, reader):
        """Resolve exactly the row displayed in the search results."""
        if reader.experiment["experiment_id"] != self.experiment_id:
            raise CaptureError("FRAME_REVIEW_CHANGED", "the selected path now contains a different experiment")
        row = reader.frame_at_sequence(self.camera_id, self.sequence)
        if row is None or (row["ordinal"], row["frame_id"], row["sha256"]) != (self.ordinal, self.frame_id, self.sha256):
            raise CaptureError("FRAME_REVIEW_CHANGED", f"the displayed frame record changed: {self.camera_id}/{self.sequence}")
        return row
