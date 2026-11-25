"""Measure raw highlight headroom and verify a bounded manual exposure adjustment."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from statistics import median

import numpy as np

from .cameras.base import Feature
from .config import CaptureError
from .preview import PreviewSession, _image_layout, _inspection_pixels, _inspection_stride

SAMPLE_FRAMES = 5
_TARGET = .82
_BRIGHTNESS_BAND = (.70, .90)
_CLIPPED_LIMIT = .005


@dataclass(frozen=True, slots=True)
class ExposureMeasurement:
    highlight: float
    near_saturation: float
    highlight_spread: float
    saturation_spread: float

    @property
    def stable(self):
        return self.highlight_spread <= .08 and self.saturation_spread <= .03


@dataclass(frozen=True, slots=True)
class ExposureAdvice:
    maximum_us: float
    proposed_us: int | float | None
    reason: str


@dataclass(frozen=True, slots=True)
class ExposureScope:
    preview: PreviewSession
    camera_id: str
    binding: object
    roi: tuple[int, int, int, int] | None
    layout: tuple
    features: dict[str, Feature]
    settings: dict
    planned_rates: tuple[float, ...]

    def still_matches(self, other):
        return (self.preview is other.preview and self.camera_id == other.camera_id and self.binding is other.binding
                and self.roi == other.roi and self.layout == other.layout and self.planned_rates == other.planned_rates)

    def latest(self, _timeout):
        if self.preview.camera_binding(self.camera_id) is not self.binding:
            raise CaptureError("EXPOSURE_SCOPE_CHANGED", "the preview connection or camera settings changed")
        return self.preview.latest_image(self.camera_id)


def _positive(value) -> bool:
    return type(value) in (int, float) and 0 < value <= 1e15


def exposure_unavailable(features: dict[str, Feature]) -> str:
    for name in ("exposure_auto", "gain_auto"):
        auto = features.get(name)
        if auto is not None and (not auto.readable or auto.value != "Off"):
            return "先在相机参数中关闭自动曝光和自动增益，再测量新的画面。"
    node = features.get("exposure_us")
    if (node is None or not node.readable or not node.writable or node.unit not in ("us", "µs", "μs")
            or not all(_positive(v) for v in (node.value, node.maximum))
            or type(node.minimum) not in (int, float) or not 0 <= node.minimum <= node.maximum
            or (node.increment is not None and node.increment != 0 and not _positive(node.increment))
            or not node.minimum <= node.value <= node.maximum):
        return "相机需要提供可读写的曝光时间、微秒单位和允许范围。可在相机参数中查看设备支持的设置。"
    return ""


def _frame_values(image, roi):
    pixels, _, _ = _inspection_pixels(image, roi)
    if min(pixels.shape[:2]) < 8 or pixels.shape[0] * pixels.shape[1] < 100:
        raise CaptureError("EXPOSURE_REGION_TOO_SMALL", "at least 100 pixels in an area at least 8 pixels wide and high")
    stride = _inspection_stride(pixels, bayer=image.color_model == "bayer")
    samples = pixels[::stride, ::stride]
    if samples.ndim == 3:
        samples = samples[..., :3].max(axis=2)
    maximum = (1 << image.significant_bits) - 1
    if int(samples.max()) > maximum:
        raise CaptureError("EXPOSURE_CODE_RANGE", "sample values exceed the declared image bit depth")
    return float(np.percentile(samples, 99) / maximum), float(np.mean(samples >= .99 * maximum))


def sample_exposure(read, *, after, layout, roi=None, cancelled=None, timeout=8.0):
    """Read distinct raw frames newer than the measurement request."""
    deadline = time.monotonic() + timeout
    highlights, clipped = [], []
    latest = None
    while len(highlights) < SAMPLE_FRAMES:
        if cancelled is not None and cancelled():
            raise CaptureError("EXPOSURE_CANCELLED", "exposure measurement cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CaptureError("EXPOSURE_SAMPLE_TIMEOUT", f"received {len(highlights)} of {SAMPLE_FRAMES} new images")
        frame = read(min(.2, remaining))
        if frame is None or frame.received_monotonic_ns <= after:
            time.sleep(min(.025, remaining))
            continue
        if _image_layout(frame) != layout:
            raise CaptureError("EXPOSURE_SCOPE_CHANGED", "the image size, format or image part changed")
        highlight, saturation = _frame_values(frame, roi)
        highlights.append(highlight)
        clipped.append(saturation)
        after, latest = frame.received_monotonic_ns, frame
    return ExposureMeasurement(median(highlights), median(clipped), max(highlights) - min(highlights),
                               max(clipped) - min(clipped)), latest


def _bound_time(feature, proposal, upper):
    lower = feature.minimum
    value = min(upper, max(lower, proposal))
    if feature.increment:
        maximum_step = math.floor((upper - lower) / feature.increment)
        step = min(maximum_step, round((value - lower) / feature.increment))
        value = lower + step * feature.increment
        if value > upper and step > 0:
            value = lower + (step - 1) * feature.increment
    if type(feature.value) is int:
        value = max(math.ceil(lower), math.floor(value))
    if not 0 < value <= upper:
        raise CaptureError("EXPOSURE_LIMIT", f"no supported exposure fits the {upper:g} µs limit")
    feature.validate(value)
    return value


def recommend_exposure(measurement, features, planned_rates=(), *, operator_limit=None):
    if reason := exposure_unavailable(features):
        raise CaptureError("EXPOSURE_UNAVAILABLE", reason)
    if operator_limit is not None and not _positive(operator_limit):
        raise CaptureError("EXPOSURE_LIMIT", "请输入大于 0 的最大曝光时间，单位为微秒。")
    feature = features["exposure_us"]
    fps = features.get("fps")
    fps_enable = features.get("fps_enable")
    rates = [rate for rate in planned_rates if _positive(rate)]
    if fps and fps.readable and _positive(fps.value) and (fps_enable is None or fps_enable.value is True):
        rates.append(fps.value)
    upper = min(feature.maximum, 1e6 / max(rates) if rates else math.inf, operator_limit or math.inf)
    if upper < feature.minimum:
        raise CaptureError("EXPOSURE_LIMIT", f"exposure limit {upper:g} µs below device minimum {feature.minimum:g} µs")
    if not measurement.stable:
        return ExposureAdvice(upper, None, "连续画面的亮部变化较大。请固定试样、检查光源，再重新测量。")
    if measurement.highlight < .04:
        return ExposureAdvice(upper, None, "亮部也接近全黑。请检查镜头盖、光源和检查框，再重新测量。")
    clipped = measurement.near_saturation > _CLIPPED_LIMIT
    if not clipped and _BRIGHTNESS_BAND[0] <= measurement.highlight <= _BRIGHTNESS_BAND[1]:
        message = ("当前画面有足够的亮部余量。" if feature.value <= upper
                   else "当前曝光超过时限。缩短曝光后可增加照明，再重新测量。")
        return ExposureAdvice(upper, None, message)
    if clipped:
        ratio = .50 if measurement.near_saturation >= .10 else .72
    else:
        ratio = max(.5, min(2., _TARGET / measurement.highlight))
    candidate = _bound_time(feature, feature.value * ratio, upper)
    if (ratio > 1 and candidate <= feature.value) or (ratio < 1 and candidate >= feature.value):
        return ExposureAdvice(upper, None, "设备的曝光范围、步长或当前时限已限制调整。可先调整照明。")
    return ExposureAdvice(upper, candidate, "")


def confirm_exposure_improvement(before, after, *, increased):
    """Reject unstable or unhelpful changes so the preview transaction restores the old value."""
    if before.stable and after.stable:
        direction = 1 if increased else -1
        highlight_delta = after.highlight - before.highlight
        highlight_margin = max(.025, before.highlight_spread + after.highlight_spread)
        clip_margin = max(.01, before.saturation_spread + after.saturation_spread)
        improved_headroom = (
            direction * highlight_delta > highlight_margin and after.highlight <= .95
            and after.near_saturation <= _CLIPPED_LIMIT
            and abs(after.highlight - _TARGET) + .015 < abs(before.highlight - _TARGET)
        )
        lower_clipping = (
            not increased and before.near_saturation > _CLIPPED_LIMIT
            and before.near_saturation - after.near_saturation > clip_margin
            and .45 <= after.highlight <= before.highlight
        )
        if improved_headroom or lower_clipping:
            return
    raise CaptureError("EXPOSURE_NO_IMPROVEMENT",
                       f"亮部 {before.highlight:.1%} → {after.highlight:.1%}，"
                       f"近饱和像素 {before.near_saturation:.2%} → {after.near_saturation:.2%}。"
                       "变化不足以确认所设曝光改善了画面。")
