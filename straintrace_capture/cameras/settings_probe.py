"""Validate planned writes against read-only camera feature evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..config import CaptureError
from . import simulated
from .base import Device
from ._genapi import NODES, PHASE_FEATURES, ROI_AXES, image_layout, setting_order
from ._gentl_multipart import IMAGE_PART, SAVE_ALL_2D, parse_image_part


@dataclass(frozen=True)
class SettingsCheck:
    basis: str
    checked: tuple[str, ...] = ()
    deferred: tuple[str, ...] = ()
    invalid: tuple[dict, ...] = ()
    error: str | None = None
    readback: dict | None = field(default=None, repr=False)
    device: Device | None = field(default=None, repr=False)


_SAMPLING = frozenset(("pixel_format", "binning_x", "binning_y", "decimation_x", "decimation_y"))
_GEOMETRY = _SAMPLING | {name for axis in ROI_AXES for name in axis}
_DEPENDENCIES = {
    **{size: _SAMPLING | {offset} for size, offset in ROI_AXES},
    **{offset: _SAMPLING | {size} for size, offset in ROI_AXES},
    "exposure_us": _GEOMETRY | {"exposure_auto", "fps", "fps_enable"},
    "exposure_raw": _GEOMETRY | {"exposure_auto", "fps", "fps_enable"},
    "gain_db": {"gain_auto"}, "gain_raw": {"gain_auto"},
    "fps": _GEOMETRY | {"fps_enable", "exposure_us", "exposure_raw", "exposure_auto"},
}


def _invalid(name, value, feature, error):
    return {
        "feature": name, "value": value, "reason": error.code, "detail": str(error),
        "minimum": feature.minimum if feature else None, "maximum": feature.maximum if feature else None,
        "increment": feature.increment if feature else None, "choices": feature.choices if feature else (),
        "unit": feature.unit if feature else None,
    }


def _current_change(feature, value):
    return feature is None or not feature.readable or feature.value is None or value != feature.value


def readbacks_affected_by_pending_writes(backend, settings, features):
    pending = {name for name, value in settings.items() if _current_change(features.get(name), value)}
    if backend in ("pylon", "gentl"):
        pending |= {field for field, dependencies in _DEPENDENCIES.items() if dependencies & pending}
    return pending


def check_camera_settings(backend, settings, features):
    """Check the current catalog and defer limits affected by preceding writes."""
    genapi = backend in ("pylon", "gentl")
    ordered = setting_order(settings) if genapi else tuple(settings)
    checked, deferred, invalid = [], [], []
    changed = set()
    if genapi:
        # The offset is reset before the size write when a plan changes both.
        for size, offset in ROI_AXES:
            feature = features.get(offset)
            if size in settings and offset in settings and feature and feature.minimum is not None:
                if _current_change(feature, feature.minimum):
                    try:
                        feature.validate(feature.minimum)
                    except CaptureError as exc:
                        invalid.append(_invalid(offset, settings[offset], feature, exc))
                    changed.add(offset)

    for name in ordered:
        feature = features.get(name)
        depends = genapi and bool(changed.intersection(_DEPENDENCIES.get(name, changed)))
        try:
            if genapi and name in PHASE_FEATURES:
                raise CaptureError("PHASE_OWNS_TRIGGER", f"configure {name} via the capture phase")
            if genapi and name == "pixel_format":
                image_layout(settings[name])
            if backend == "gentl" and name == IMAGE_PART:
                if parse_image_part(settings[name]) is not None:
                    deferred.append(name)
                    continue
            if backend == "gentl" and name == SAVE_ALL_2D and feature is not None:
                feature.validate(settings[name])
                (deferred if settings[name] else checked).append(name)
                continue
            if feature is None and (not depends or name not in NODES):
                raise CaptureError("FEATURE_UNAVAILABLE", f"camera does not expose {name}")
            if backend == "uvc" and name != "pixel_format" and type(settings[name]) not in (int, float):
                raise CaptureError("FEATURE_TYPE", f"{name} requires a number")
            if depends:
                if feature is not None:
                    # Value type is stable while access, bounds and choices may change.
                    typed = replace(feature, minimum=None, maximum=None, increment=None, choices=(), writable=True)
                    typed.validate(settings[name])
                deferred.append(name)
            else:
                feature.validate(settings[name])
                checked.append(name)
        except CaptureError as exc:
            invalid.append(_invalid(name, settings[name], None if depends else feature, exc))
        if _current_change(feature, settings[name]):
            changed.add(name)
    return SettingsCheck("device_features", tuple(checked), tuple(deferred), tuple(invalid))


def check_simulator_settings(camera):
    """Evaluate the configured writes on an in-memory simulator without acquiring."""
    model = simulated.camera(camera.id)
    checked, invalid = [], []
    for name, value in camera.settings.items():
        feature = model.features().get(name)
        try:
            model.set_feature(name, value)
        except CaptureError as exc:
            invalid.append(_invalid(name, value, feature, exc))
        else:
            checked.append(name)
    return SettingsCheck("simulator", tuple(checked), invalid=tuple(invalid))
