"""Measurements and source identity for the live curve views."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from .config import CaptureError
from .image_inputs import frame_part_selector, recorded_image, select_frame_image
from .storage import ExperimentReader


CURVE_MODES = {"strain-time": "应变–时间", "force-time": "载荷–时间", "force-strain": "载荷–应变",
               "field-time": "全场质量–时间"}
_VALID_FORCE = {"EXACT_TIME_SAMPLE", "LINEAR_TIME_INTERPOLATION", "EXPOSURE_WINDOW_MEAN"}


def _finite(value):
    return value if value is not None and isfinite(value) else None


def available_curve_modes(*, strain, force, field):
    flags = (strain, force, strain and force, field)
    return tuple(mode for mode, enabled in zip(CURVE_MODES, flags, strict=True) if enabled)


@dataclass(frozen=True, slots=True)
class CurveFrame:
    experiment: Path
    camera_id: str
    frame_id: str
    sequence: int
    image_part: str | None
    image_path: str
    image_sha256: str

    @property
    def camera_scope(self):
        return self.frame_id.rsplit("/", 1)[0]


@dataclass(frozen=True, slots=True)
class CurvePoint:
    frame: CurveFrame
    elapsed_s: float | None
    receipt_utc_ns: int
    strain_pct: float | None
    strain_status: str | None
    force_N: float | None
    force_status: str | None
    time_basis: str | None
    load_time_domain: str | None
    field_valid_pct: float | None = None
    field_ok: bool | None = None
    is_reference: bool = False

    @classmethod
    def from_result(cls, row, experiment, camera):
        gauge = row["gauges"][0] if row["gauges"] else {}
        mechanics = row.get("mechanics") or {}
        force_status = (mechanics.get("force_support") or {}).get("status")
        strain = gauge.get("engineering_strain") if gauge.get("strain_valid") else None
        force = mechanics.get("force_N") if force_status in _VALID_FORCE else None
        field = row.get("field_quality") or {}
        valid_fraction = _finite(field.get("strain_valid_fraction"))
        field_percentage = valid_fraction * 100 if valid_fraction is not None and 0 <= valid_fraction <= 1 else None
        source = CurveFrame(Path(experiment), camera, row["frame_id"], row["sequence"], row["image_part"],
                            row["image_path"], row["image_sha256"])
        return cls(source, _finite(row["elapsed_receive_s"]), row["timestamps"]["received_utc"]["value"],
                   _finite(strain * 100) if strain is not None else None, gauge.get("quality_status"),
                   _finite(force), force_status, mechanics.get("time_basis"), mechanics.get("load_time_domain"),
                   field_percentage, field.get("scientific_ok"),
                   row.get("analyzed_test_frames") == 0 or row.get("role") == "reference")

    def values(self, mode):
        if mode == "force-strain":
            return self.strain_pct, self.force_N
        if mode == "field-time":
            return self.elapsed_s, self.field_valid_pct
        return self.elapsed_s, self.force_N if mode == "force-time" else self.strain_pct

    def needs_review(self, mode):
        return not self.is_reference and (None in self.values(mode) or mode == "field-time" and self.field_ok is False)


def curve_segments(points, mode):
    """Connect only consecutive valid points, in acquisition order."""
    segment, previous = [], None
    for point in points:
        frame = point.frame
        x, y = point.values(mode)
        if segment and (x is None or y is None or frame.camera_scope != previous.camera_scope
                        or frame.sequence != previous.sequence + 1):
            yield segment
            segment = []
        if x is not None and y is not None:
            segment.append((point, x, y))
        previous = frame
    if segment:
        yield segment


def read_curve_frame(frame):
    """Read and verify the exact image bytes used for a curve point."""
    with ExperimentReader(frame.experiment) as reader:
        scope = f"{reader.experiment['experiment_id']}/{frame.camera_id}"
        if scope != frame.camera_scope:
            raise CaptureError("FRAME_REVIEW_CHANGED", "曲线点对应的实验目录已变化。")
        row = reader.frame_at_sequence(frame.camera_id, frame.sequence)
        if row is None or row["frame_id"] != frame.frame_id:
            raise CaptureError("FRAME_REVIEW_CHANGED", "未找到曲线点对应的图像记录。")
        if row["path"] != frame.image_path or frame_part_selector(row) != frame.image_part:
            row = select_frame_image(row, frame.image_part)
        if (row["path"], row["sha256"], frame_part_selector(row)) != (frame.image_path, frame.image_sha256, frame.image_part):
            raise CaptureError("FRAME_REVIEW_CHANGED", "图像部件或台账校验值已变化。")
        return recorded_image(reader.image_path(row), row, verify=True)
