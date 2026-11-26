"""Verify common settings on idle cameras before updating a capture plan."""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass, field

from .cameras.settings_probe import check_camera_settings
from .config import CaptureConfig, CaptureError
from .continuation import same_camera_value


BATCH_FIELDS = ("exposure_us", "gain_db", "pixel_format", "width", "height")
_UNITS = {"exposure_us": "us", "gain_db": "db", "width": "px", "height": "px"}
_GEOMETRY = frozenset(("pixel_format", "binning_x", "binning_y", "decimation_x", "decimation_y",
                       "width", "height", "offset_x", "offset_y"))
_ALIASES = {"exposure_us": "exposure_raw", "gain_db": "gain_raw"}
_AUTO = {"exposure_us": "exposure_auto", "gain_db": "gain_auto"}
_MANUAL = frozenset((*_AUTO, *_ALIASES.values()))


@dataclass(frozen=True)
class SharedFeature:
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    blockers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraProbe:
    identity: str
    state: str
    before: dict = field(default_factory=dict)
    requested: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)
    detail: str = ""
    issues: tuple[dict, ...] = ()


@dataclass(frozen=True)
class BatchVerification:
    plan: CaptureConfig | None
    rows: tuple[CameraProbe, ...]


def _unit(feature):
    unit = (feature.unit or "").strip().lower()
    return "us" if unit in ("µs", "μs") else unit


def _feature_problem(name, features, planned):
    feature = features.get(name)
    if feature is None:
        return "相机未提供此参数。"
    if key := _AUTO.get(name):
        auto = features.get(key)
        if (auto and (not auto.readable or auto.value not in (False, "Off"))
                or key in planned and planned[key] not in (False, "Off")):
            return "请先在该相机的参数中关闭对应的自动调节。"
    if not feature.readable or not feature.writable or feature.value is None:
        return "当前状态下无法写入并回读此参数。"
    if name in _UNITS and _unit(feature) != _UNITS[name]:
        return "相机在此接口使用的单位无法与其他相机统一。"
    if name == "pixel_format" and not feature.choices:
        return "相机未给出可选的像素格式。"
    return ""


def shared_features(config, readings):
    """Show the intersection of the current catalogs, including per-camera blockers."""
    result = {}
    for name in BATCH_FIELDS:
        features = []
        blockers = {}
        for specification in config.cameras:
            camera_id = specification.id
            evidence = readings.get(camera_id)
            if not evidence or evidence.error or evidence.readback is None:
                blockers[camera_id] = "无法读取相机参数。"
            elif problem := _feature_problem(name, evidence.readback, specification.settings):
                blockers[camera_id] = problem
            else:
                features.append(evidence.readback[name])
        minimum = max((feature.minimum for feature in features if feature.minimum is not None), default=None)
        maximum = min((feature.maximum for feature in features if feature.maximum is not None), default=None)
        choices = tuple(value for value in features[0].choices
                        if all(value in feature.choices for feature in features[1:])) if features else ()
        if not blockers and ((name == "pixel_format" and not choices)
                             or (minimum is not None and maximum is not None and minimum > maximum)):
            blockers = {item.id: "所选相机的当前取值没有交集。" for item in config.cameras}
        result[name] = SharedFeature(minimum, maximum, choices, blockers)
    return result


def parse_batch_values(fields):
    changes = {}
    for name in BATCH_FIELDS:
        if not (raw := fields[name].strip()):
            continue
        if name == "pixel_format":
            changes[name] = raw
            continue
        try:
            value = int(raw) if name in ("width", "height") else float(raw)
            if abs(value) > 1e15 or not math.isfinite(value):
                raise ValueError
        except ValueError as exc:
            raise ValueError(name) from exc
        changes[name] = value
    return changes


def _prepare(camera, specification, changes, features):
    for name in changes:
        if problem := _feature_problem(name, features, specification.settings):
            raise CaptureError("BATCH_SETTING_UNAVAILABLE", f"{name} · {problem}")
    planned = dict(specification.settings)
    for name, value in changes.items():
        if alias := _ALIASES.get(name):
            planned.pop(alias, None)
        if type(features[name].value) is int and type(value) is float and value.is_integer():
            value = int(value)
        planned[name] = value
    # Check requested sizes and exposure under the framing already saved in the plan.
    include_geometry = bool(changes.keys() & (_GEOMETRY | {"exposure_us"}))
    writes = {name: value for name, value in planned.items()
              if name in changes or include_geometry and name in _GEOMETRY}
    watched = _GEOMETRY | _MANUAL
    baseline = {name: feature for name, feature in features.items()
                if name in watched and feature.readable and feature.value is not None}
    for name in writes:
        if name not in baseline or not baseline[name].writable:
            raise CaptureError("BATCH_SETTING_UNAVAILABLE", f"{name} · 无法保留此计划参数的原值。")
    check = check_camera_settings(camera.device.backend, writes, features)
    return planned, writes, baseline, check.invalid


def _values(features, names):
    return {name: features[name].value if name in features and features[name].readable else None for name in names}


def _mismatches(features, expected, baseline):
    values = _values(features, expected)
    return [name for name, value in expected.items()
            if not same_camera_value(values[name], value) or _unit(features[name]) != _unit(baseline[name])]


def _compare(features, expected, baseline):
    if mismatches := _mismatches(features, expected, baseline):
        values = _values(features, mismatches)
        detail = "、".join(f"{name}（应为 {expected[name]}，读到 {values[name]}）" for name in mismatches)
        raise CaptureError("FEATURE_READBACK_MISMATCH", detail)


def _restore(camera, baseline, original):
    current = camera.features()
    # Format changes can alter ROI access and bounds. Restore the format first.
    if "pixel_format" in original and "pixel_format" in _mismatches(
            current, {"pixel_format": original["pixel_format"]}, baseline):
        camera.set_features({"pixel_format": original["pixel_format"]})
        current = camera.features()
    if mismatches := _mismatches(current, original, baseline):
        # Raw and physical nodes can represent the same control. Prefer the physical node.
        restore = {name: original[name] for name in mismatches if baseline[name].writable}
        for name, alias in _ALIASES.items():
            if name in restore:
                restore.pop(alias, None)
        if restore:
            camera.set_features(restore)
            current = camera.features()
        _compare(current, original, baseline)


def _trial(camera, requested, writes, baseline):
    identity = camera.device.id
    original = {name: feature.value for name, feature in baseline.items()}
    before = {name: original[name] for name in requested}
    features = baseline
    changed = any(not same_camera_value(original[name], value) for name, value in writes.items())
    state, detail = "verified", ""
    try:
        if changed:
            camera.set_features(writes)
            features = camera.features()
        # Geometry may not shift silently. Raw aliases can legitimately track a manual write.
        expected = {name: value for name, value in original.items() if name in _GEOMETRY}
        _compare(features, {**expected, **writes}, baseline)
    except Exception as exc:
        state, detail = "rejected", str(exc)
        try:
            features = camera.features()
        except Exception:
            features = {}
    observed = _values(features, requested)
    if changed:
        try:
            _restore(camera, baseline, original)
        except Exception as exc:
            detail = "\n".join(filter(None, (detail, f"恢复原值时发生问题。{exc}")))
            state = "restore_failed"
    return CameraProbe(identity, state, before, requested, observed, detail)


def verify_batch_settings(config, changes, readings, registry):
    """Precheck every camera, then try and restore in order. Return a plan only on success."""
    if len(config.cameras) < 2 or not changes or changes.keys() - set(BATCH_FIELDS):
        raise CaptureError("BATCH_INVALID_SELECTION", "请至少选择两台相机并填写一项共同参数。")
    rows, prepared = {}, {}
    with ExitStack() as connections:
        for specification in config.cameras:
            identity = specification.id
            try:
                previous = readings.get(identity)
                if previous is None or previous.device is None:
                    raise CaptureError("BATCH_UNAVAILABLE", "无法确认相机身份，请刷新设备列表后重试。")
                camera = connections.enter_context(registry.camera(identity))
                if camera.device != previous.device:
                    raise CaptureError("BATCH_DEVICE_CHANGED", "相机身份已变化，请关闭窗口并重新选择设备。")
                settings, writes, baseline, issues = _prepare(camera, specification, changes, camera.features())
                requested = {name: settings[name] for name in changes}
                if issues:
                    rows[identity] = CameraProbe(identity, "blocked", _values(baseline, requested),
                                                 requested, issues=issues)
                else:
                    prepared[identity] = (camera, requested, writes, baseline, settings)
            except Exception as exc:
                rows[identity] = CameraProbe(identity, "blocked", requested=dict(changes), detail=str(exc))
        if not rows:
            for identity, (camera, requested, writes, baseline, _) in prepared.items():
                rows[identity] = _trial(camera, requested, writes, baseline)
                if rows[identity].state == "restore_failed":
                    break
    results = tuple(rows.get(item.id) or CameraProbe(item.id, "skipped", requested=dict(changes))
                    for item in config.cameras)
    if any(row.state != "verified" for row in results):
        return BatchVerification(None, results)
    data = config.to_dict()
    for item in data["cameras"]:
        item["settings"] = prepared[item["id"]][-1]
    return BatchVerification(CaptureConfig.from_dict(data), results)
