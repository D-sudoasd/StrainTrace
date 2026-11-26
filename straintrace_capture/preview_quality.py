"""Short, read-only checks of raw images from selected preview cameras."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from .config import CaptureError
from .preview import QUALITY_ALERT_FRACTION, _image_layout, image_quality


PREVIEW_CHECK_FRAMES = 5


@dataclass(frozen=True, slots=True)
class PreviewQualityTarget:
    camera_id: str
    roi: tuple[int, int, int, int] | None
    binding: object | None = None
    layout: tuple = ()
    unavailable: str = ""

    def matches(self, other):
        return (other is not None and self.camera_id == other.camera_id and self.binding is not None
                and self.binding is other.binding and self.roi == other.roi and self.layout == other.layout)


@dataclass(frozen=True, slots=True)
class QualityReading:
    received_ns: int
    utc_ns: int
    dark: float
    saturated: float
    texture: str | None
    texture_measured: bool


@dataclass(frozen=True, slots=True)
class PreviewQualityResult:
    target: PreviewQualityTarget
    state: str
    readings: tuple[QualityReading, ...] = ()
    detail: str = ""

    def summary(self):
        return {
            "frames": len(self.readings),
            "sampled_utc_ns": self.readings[-1].utc_ns if self.readings else None,
            "roi": list(self.target.roi) if self.target.roi is not None else None,
            "dark_frames": sum(item.dark >= QUALITY_ALERT_FRACTION for item in self.readings),
            "saturated_frames": sum(item.saturated >= QUALITY_ALERT_FRACTION for item in self.readings),
            "maximum_dark_fraction": max((item.dark for item in self.readings), default=None),
            "maximum_saturated_fraction": max((item.saturated for item in self.readings), default=None),
            "texture_frames": sum(item.texture_measured for item in self.readings),
            "texture_alerts": tuple(dict.fromkeys(item.texture for item in self.readings if item.texture)),
        }


def _reading(frame, roi):
    metrics = image_quality(frame, roi)
    texture = metrics["texture"]
    return QualityReading(frame.received_monotonic_ns, frame.received_utc_ns, metrics["dark_fraction"],
                          metrics["bright_fraction"], texture["failure_code"] if texture else None, texture is not None)


def inspect_preview_quality(preview, targets, *, after, cancelled, timeout=8.0):
    """Read up to five distinct, newer frames per camera under the original binding."""
    rows = {target.camera_id: PreviewQualityResult(target, "unavailable", detail=target.unavailable)
            for target in targets if target.binding is None}
    pending = {target.camera_id: (target, []) for target in targets if target.binding is not None}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for identity, (target, readings) in tuple(pending.items()):
            if cancelled.is_set():
                return ()
            snapshot = preview.bound_image(identity)
            if snapshot is None or snapshot[0] is not target.binding or _image_layout(snapshot[1]) != target.layout:
                rows[identity] = PreviewQualityResult(target, "changed")
            else:
                frame = snapshot[1]
                latest = readings[-1].received_ns if readings else after
                if frame.received_monotonic_ns <= latest:
                    continue
                try:
                    readings.append(_reading(frame, target.roi))
                except (CaptureError, ValueError, cv2.error) as exc:
                    detail = exc.code if isinstance(exc, CaptureError) else str(exc)
                    rows[identity] = PreviewQualityResult(target, "failed", detail=detail)
                else:
                    if len(readings) < PREVIEW_CHECK_FRAMES:
                        continue
                    rows[identity] = PreviewQualityResult(target, "ready", tuple(readings))
            del pending[identity]
        if pending and cancelled.wait(max(0., min(.025, deadline - time.monotonic()))):
            return ()
    rows.update((identity, PreviewQualityResult(target, "timeout", tuple(readings)))
                for identity, (target, readings) in pending.items())
    return tuple(rows[target.camera_id] for target in targets)


def quality_has_alert(summary):
    return bool(summary["dark_frames"] or summary["saturated_frames"] or summary["texture_alerts"])


def with_preview_quality(report, rows):
    """Include completed checks as preflight guidance."""
    evidence = {row.target.camera_id: row.summary() for row in rows
                if row.state == "ready" and row.target.camera_id in report["cameras"]}
    if not evidence:
        return report
    findings = [{"code": "PREVIEW_IMAGE_QUALITY", "severity": "warning", "camera": identity, **summary}
                for identity, summary in evidence.items() if quality_has_alert(summary)]
    return {**report, "preview_quality": evidence, "issues": [*report["issues"], *findings]}
