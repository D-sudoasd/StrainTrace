"""Sampled operator advisories for the primary images admitted to a recording."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from statistics import median

from .cameras._gentl_multipart import AUTO_PART, IMAGE_PART
from .config import SENSOR_WINDOW_FIELDS, QualityRegion
from .preview import QUALITY_ALERT_FRACTION, _image_layout, image_quality


NOTICE_EVENT = "image_quality_notice"
SAMPLE_INTERVAL_S = .20
PIXEL_LIMIT = 32_768
CONFIRMATIONS = 3
MAX_BRIGHTNESS_CHANGE = .15
MIN_REFERENCE_CONTRAST = .05
SHARPNESS_ALERT = .45
SHARPNESS_CLEAR = .65
EXPOSURE_CLEAR = .15


@dataclass
class _CameraQuality:
    context: tuple
    baseline: list[tuple[float, float]] = field(default_factory=list)
    active: set[str] = field(default_factory=set)
    pending: dict[str, int] = field(default_factory=dict)
    latest: dict = field(default_factory=dict)

    def compare(self, measure, *, compare_sharpness):
        dark = measure["near_black_fraction"]
        bright = measure["near_saturation_fraction"]
        low, brightness, high = measure["brightness_p5_p50_p95"]
        response = measure["sharpness_response"]
        usable = compare_sharpness and max(dark, bright) < QUALITY_ALERT_FRACTION and response is not None
        if usable and len(self.baseline) < CONFIRMATIONS and high - low >= MIN_REFERENCE_CONTRAST and response > 0:
            self.baseline.append((response, brightness))

        ratio = None
        if usable and len(self.baseline) == CONFIRMATIONS:
            base_brightness = median(value[1] for value in self.baseline)
            if abs(brightness - base_brightness) <= MAX_BRIGHTNESS_CHANGE:
                ratio = response / median(value[0] for value in self.baseline)

        candidates = {
            "near_black": dark >= (EXPOSURE_CLEAR if "near_black" in self.active else QUALITY_ALERT_FRACTION),
            "near_saturation": bright >= (EXPOSURE_CLEAR if "near_saturation" in self.active else QUALITY_ALERT_FRACTION),
            "sharpness_down": None if ratio is None else ratio <= (
                SHARPNESS_CLEAR if "sharpness_down" in self.active else SHARPNESS_ALERT),
        }
        started, cleared = [], []
        for kind, condition in candidates.items():
            active = kind in self.active
            if condition is None or condition == active:
                self.pending.pop(kind, None)
                continue
            attempts = self.pending.get(kind, 0) + 1
            if attempts < CONFIRMATIONS:
                self.pending[kind] = attempts
                continue
            self.pending.pop(kind, None)
            if condition:
                self.active.add(kind)
                started.append(kind)
            else:
                self.active.remove(kind)
                cleared.append(kind)
        return ratio, started, cleared


def sensor_window_values(metadata, values):
    fields = {name: values.get(name) for name in SENSOR_WINDOW_FIELDS}
    part = metadata.get(IMAGE_PART) or {}
    fields.update({name: part[name] for name in ("offset_x", "offset_y") if name in part})
    return {name: int(value) for name, value in fields.items()
            if type(value) in (int, float) and 0 <= value <= 1e12 and int(value) == value
            and (name.startswith("offset_") or value > 0)}


def quality_region_from_image(image, bounds, values):
    return QualityRegion.from_dict({
        "bounds": list(bounds), "image_size_px": list(image.pixels.shape[:2][::-1]),
        "pixel_format": image.pixel_format, "significant_bits": image.significant_bits,
        "image_part": (image.metadata.get(IMAGE_PART) or {}).get("selector"),
        "sensor_window": sensor_window_values(image.metadata, values),
    })


quality_region_from_preview = quality_region_from_image


def quality_region_settings_mismatch(region, settings):
    part = settings.get(IMAGE_PART, AUTO_PART)
    width, height = region.image_size_px
    layout = {"width": width, "height": height, "pixel_format": region.pixel_format}
    if part not in (AUTO_PART, region.image_part) or (region.image_part is None and any(
        name in settings and settings[name] != value for name, value in layout.items()
    )):
        return "image_layout"
    if any(name in settings and settings[name] != value for name, value in region.sensor_window.items()):
        return "sensor_window"
    return None


def quality_region_mismatch(region, image, values):
    return quality_region_layout_mismatch(region, image.pixels.shape, image.pixel_format, image.significant_bits,
                                          image.metadata, values)


def quality_region_layout_mismatch(region, shape, pixel_format, bits, metadata, values):
    if (tuple(shape[:2][::-1]) != region.image_size_px or pixel_format != region.pixel_format
            or bits != region.significant_bits or (metadata.get(IMAGE_PART) or {}).get("selector") != region.image_part):
        return "image_layout"
    if values is not None:
        current = sensor_window_values(metadata, values)
        if any(current.get(key) != value for key, value in region.sensor_window.items()):
            return "sensor_window"
    return None


def measure_primary_image(image, *, roi=None):
    """Analyze a bounded sample of the immutable original primary image."""
    result = image_quality(image, roi, pixel_limit=PIXEL_LIMIT, measure_texture=False)
    return {
        "near_black_fraction": result["dark_fraction"],
        "near_saturation_fraction": result["bright_fraction"],
        "brightness_p5_p50_p95": [value / result["code_max"] for value in result["percentiles_p5_p50_p95"]],
        "sharpness_response": result["focus_laplacian_variance"],
        "sample_stride_px": result["histogram_stride_px"],
    }


class ImageQualityWatch:
    """Keep a fixed-size reference and the most recent sample per camera."""

    def __init__(self):
        self._cameras: dict[str, _CameraQuality] = {}
        self.measurements = 0
        self.measurement_errors = 0
        self.ledger_errors = 0
        self.deferred = False
        self.deferred_intervals = 0

    def update(self, camera, phase, sequence, image, measure, *, compare_sharpness, region=None):
        layout = _image_layout(image)[0]
        context = phase, layout, image.pixels.dtype.name, region
        track = self._cameras.get(camera)
        changed = region is not None and (track is None or track.context[0] != phase or track.latest.get("region") != region)
        if track is None or track.context != context:
            track = self._cameras[camera] = _CameraQuality(context)
        ratio, started, cleared = track.compare(measure, compare_sharpness=compare_sharpness)
        self.measurements += 1
        track.latest = {
            "phase": phase, "camera_id": camera, "sequence": sequence,
            "received_utc_ns": image.received_utc_ns,
            "received_monotonic_ns": image.received_monotonic_ns,
            "pixel_format": image.pixel_format, "significant_bits": image.significant_bits,
            "shape": list(image.pixels.shape), "image_part": layout[-1],
            "metrics": {**measure, "sharpness_ratio": ratio},
            "reference_samples": len(track.baseline),
            "compare_sharpness": compare_sharpness,
            "alerts": sorted(track.active),
            **({"region": region} if region is not None else {}),
        }
        if started or cleared or changed:
            return {**track.latest, "started": started, "cleared": cleared,
                    **({"region_changed": True} if changed else {})}
        return None

    def snapshot(self):
        return {
            "measurements": self.measurements, "measurement_errors": self.measurement_errors,
            "ledger_errors": self.ledger_errors,
            "deferred": self.deferred, "deferred_intervals": self.deferred_intervals,
            "confirmations": CONFIRMATIONS, "pixel_limit": PIXEL_LIMIT,
            "near_limit_fraction": QUALITY_ALERT_FRACTION, "sharpness_alert_ratio": SHARPNESS_ALERT,
            "cameras": {camera: deepcopy(track.latest) for camera, track in self._cameras.items()},
        }
