"""Bind a fresh acquisition plan to a previous terminal experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from .cameras._gentl_multipart import IMAGE_PART, SAVE_ALL_2D
from .cameras.base import Device, Feature
from .config import REPEAT_OF, REPEAT_PARENT, CaptureConfig, CaptureError
from .recovery import frame_write_records_need_review
from .storage import ExperimentReader, TERMINAL_STATES, sha256_file

CONTINUATION_REFERENCE = "continuation_reference"
_SCHEMA = "straintrace.capture.continuation-reference/v1"
_SETUP_FEATURES = (
    "width", "height", "offset_x", "offset_y", "binning_x", "binning_y", "decimation_x", "decimation_y",
    "pixel_format", IMAGE_PART, SAVE_ALL_2D, "exposure_auto", "exposure_us", "exposure_raw", "gain_auto",
    "gain_db", "gain_raw", "fps_enable", "fps", "driver_exposure", "driver_gain",
)
_TRIGGER_FEATURES = ("trigger_selector", "trigger_mode", "trigger_source", "trigger_activation")
_ALTERNATE = {"exposure_us": "exposure_raw", "exposure_raw": "exposure_us", "gain_db": "gain_raw", "gain_raw": "gain_db"}
_PHASE_EVENTS = ("phase_started", "phase_ended", "phase_confirmation_cancelled")
_ISSUE_EVENTS = ("schedule_miss", "trigger_incomplete", "pre_window_short", "error",
                 "native_gap_unattributed", "native_duplicate", "native_reset_or_wrap", "hardware_identity_unavailable")


def camera_readback(features, *, phase=False):
    """Keep comparable device readback and its unit, with no inferred defaults."""
    result = {}
    names = _SETUP_FEATURES + _TRIGGER_FEATURES if phase else _SETUP_FEATURES
    for name in names:
        item = features.get(name)
        if isinstance(item, Feature):
            item = {"value": item.value, "readable": item.readable, "writable": item.writable, "unit": item.unit}
        if not isinstance(item, Mapping) or not item.get("readable", True):
            continue
        value, unit = item.get("value"), item.get("unit")
        if type(value) not in (str, bool, int, float) or (type(value) is float and not math.isfinite(value)):
            continue
        result[name] = {"value": value, "unit": unit if isinstance(unit, str) else None,
                        "writable": item.get("writable") is True}
    for control, fields in {
        "exposure_auto": ("exposure_us", "exposure_raw"), "gain_auto": ("gain_db", "gain_raw"),
        "fps_enable": ("fps",),
    }.items():
        manual = True if control == "fps_enable" else "Off"
        if control in features and result.get(control, {}).get("value") != manual:
            for field in fields:
                result.pop(field, None)
    for physical, raw in (("exposure_us", "exposure_raw"), ("gain_db", "gain_raw")):
        if physical in result:
            result.pop(raw, None)
    if result.get(IMAGE_PART, {}).get("value") == "auto":
        result.pop(IMAGE_PART, None)
    return result


def _phase_settings(plan):
    return {name: value for name, value in plan.items() if name != "hardware_pairing"}


def _reference(reader, record, start):
    cameras = [item["id"] for item in record["config"]["cameras"]]
    saved_cameras = reader.cameras()
    initial = {row["device"]["id"]: camera_readback(row["features"]) for row in saved_cameras}
    phases = {}
    for event in reader.events(kind="phase_started"):
        if event["phase"] >= start:
            phases[str(event["phase"] - start + 1)] = {
                key: camera_readback(features, phase=True) for key, features in (event.get("camera_features") or {}).items()
            }
    return {
        "schema": _SCHEMA, "experiment_id": record["experiment_id"], "camera_ids": cameras, "initial": initial,
        "devices": {row["device"]["id"]: row["device"] for row in saved_cameras},
        "recorded_phases": phases, "source_start_phase": start,
        "phase_plans": [_phase_settings(plan) for plan in record["config"]["phases"][start - 1:]],
    }


def continuation_reference(config):
    reference = config.metadata.get(CONTINUATION_REFERENCE)
    if (isinstance(reference, dict) and reference.get("schema") == _SCHEMA
            and isinstance(reference.get("experiment_id"), str)
            and reference["experiment_id"] == config.metadata.get("continuation_of")
            and isinstance(reference.get("camera_ids"), list)
            and all(isinstance(key, str) for key in reference["camera_ids"])
            and isinstance(reference.get("initial"), dict) and isinstance(reference.get("recorded_phases"), dict)
            and all(isinstance(phase, dict) for phase in reference["recorded_phases"].values())
            and type(reference.get("source_start_phase", 1)) is int and reference.get("source_start_phase", 1) >= 1
            and isinstance(reference.get("phase_plans"), list)
            and all(isinstance(plan, dict) for plan in reference["phase_plans"])):
        return reference
    return None


def same_camera_value(first, second):
    if type(first) in (int, float) and type(second) in (int, float):
        if first == second:
            return True
        if type(first) is float or type(second) is float:
            return abs(first) <= 1e15 and abs(second) <= 1e15 and math.isclose(first, second, rel_tol=1e-7, abs_tol=1e-8)
    return type(first) is type(second) and first == second


def _unit(label):
    return "us" if label in ("µs", "μs", "us") else label


def comparable_readback(first, second):
    return second is not None and _unit(first.get("unit")) == _unit(second.get("unit"))


def _saved_features(items):
    if not isinstance(items, dict):
        return {}
    valid_fields = _SETUP_FEATURES + _TRIGGER_FEATURES
    return {name: readback for name, readback in items.items()
            if (name in valid_fields and isinstance(readback, dict) and type(readback.get("value")) in (bool, int, float, str)
                and (readback.get("unit") is None or isinstance(readback.get("unit"), str)))}


def _phase_plan_differences(old, new):
    return sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))


def _source_phase(reference, index):
    return reference.get("source_start_phase", 1) + index - 1 if index <= len(reference["phase_plans"]) else None


def camera_identity(previous_devices, camera_id: str, device: Device | None) -> str:
    previous = previous_devices.get(camera_id) if isinstance(previous_devices, dict) else None
    if not isinstance(previous, dict) or device is None:
        return "unverified"
    if device.id != camera_id or previous.get("backend") != device.backend:
        return "changed"
    serial = previous.get("serial")
    if serial and device.serial and serial != device.serial:
        return "changed"
    if previous.get("identity_scope") == device.identity_scope:
        if device.identity_scope == "host_device_path":
            return "path_matched"
        if device.identity_scope == "serial_number" and serial and serial == device.serial:
            return "matched"
    return "unverified"


def compare_continuation_setup(config, evidence):
    """Compare prior device readback to the draft, idle camera, or bound preview."""
    if (reference := continuation_reference(config)) is None:
        return None
    expected = set(reference["camera_ids"])
    selected = {camera.id for camera in config.cameras}
    report = {"source_experiment_id": reference["experiment_id"], "missing_cameras": sorted(expected - selected),
              "source_start_phase": reference.get("source_start_phase", 1), "added_cameras": sorted(selected - expected),
              "phase_changes": [], "readbacks": {}, "suggested_settings": {}}
    for index in range(max(len(reference["phase_plans"]), len(config.phases))):
        previous = reference["phase_plans"][index] if index < len(reference["phase_plans"]) else {}
        current = _phase_settings(config.phases[index].to_dict()) if index < len(config.phases) else {}
        if fields := _phase_plan_differences(previous, current):
            report["phase_changes"].append({"phase": index + 1, "source_phase": _source_phase(reference, index + 1),
                                            "fields": fields, "previous": previous, "current": current})
    for camera in config.cameras:
        if camera.id not in expected:
            continue
        recorded = _saved_features(reference["initial"].get(camera.id))
        checked = evidence.get(camera.id)
        actual = camera_readback((checked.readback or {}) if checked else {})
        basis = checked.basis if checked and checked.readback is not None else "not_probed"
        identity = camera_identity(reference.get("devices"), camera.id, checked.device if checked else None)
        rows, proposed = [], {}
        if not recorded:
            report["readbacks"][camera.id] = {"basis": basis, "identity": identity, "fields": []}
            continue
        planned = camera.settings
        phase_values = [_saved_features(phase.get(camera.id)) for phase in reference["recorded_phases"].values()]
        for name, old in recorded.items():
            current = actual.get(name)
            value = old["value"]
            explicit = name in planned
            differs = explicit and not same_camera_value(value, planned[name])
            if differs:
                status = "plan_changed"
            elif not comparable_readback(old, current):
                status = "unverified"
            elif same_camera_value(value, current["value"]):
                status = "matched"
            else:
                status = "will_apply" if explicit and basis == "device_features" else "readback_changed"
            rows.append({"field": name, "recorded": value, "unit": old.get("unit"), "planned": planned.get(name),
                         "current": current["value"] if current is not None else None,
                         "current_unit": current.get("unit") if current is not None else None, "status": status})
            if (not explicit and _ALTERNATE.get(name) not in planned and old.get("writable") is True
                    and comparable_readback(old, current) and current["writable"]
                    and phase_values and all(comparable_readback(old, item.get(name)) and same_camera_value(value, item[name]["value"])
                                             for item in phase_values)):
                proposed[name] = value
        if identity in ("matched", "path_matched") and proposed and not any(row["status"] == "plan_changed" for row in rows):
            report["suggested_settings"][camera.id] = proposed
        report["readbacks"][camera.id] = {"basis": basis, "identity": identity, "fields": rows}
    return report


def compare_phase_readbacks(camera_ids, previous, previous_devices, readbacks, devices):
    """Compare the readable, unit-compatible camera values at phase start."""
    report = {
        "matched_fields": 0,
        "missing_cameras": sorted(set(camera_ids) - readbacks.keys()),
        "added_cameras": sorted(readbacks.keys() - set(camera_ids)), "unverified_cameras": [],
        "identities": {key: camera_identity(previous_devices, key, devices.get(key)) for key in camera_ids if key in readbacks},
        "fields": [],
    }
    for camera, features in readbacks.items():
        old = _saved_features(previous.get(camera))
        if not old:
            if camera in camera_ids:
                report["unverified_cameras"].append(camera)
            continue
        actual = camera_readback(features, phase=True)
        for name, saved in old.items():
            reading = actual.get(name)
            if comparable_readback(saved, reading) and same_camera_value(saved["value"], reading["value"]):
                report["matched_fields"] += 1
            else:
                report["fields"].append({"camera": camera, "field": name, "recorded": saved["value"],
                                         "current": reading["value"] if reading is not None else None,
                                         "current_unit": reading.get("unit") if reading is not None else None,
                                         "unit": saved.get("unit"),
                                         "status": "changed" if comparable_readback(saved, reading) else "unverified"})
    return report


def compare_continuation_phase(config, index, readbacks, devices):
    """Report what was actually read after the new phase was armed."""
    if (reference := continuation_reference(config)) is None:
        return None
    comparison = compare_phase_readbacks(
        reference["camera_ids"], reference["recorded_phases"].get(str(index), {}), reference.get("devices"), readbacks, devices,
    )
    previous_plan = reference["phase_plans"][index - 1] if index <= len(reference["phase_plans"]) else {}
    return {"source_experiment_id": reference["experiment_id"], "phase": index, "source_phase": _source_phase(reference, index),
            **comparison,
            "phase_changes": _phase_plan_differences(previous_plan, _phase_settings(config.phases[index - 1].to_dict()))}


def _require_terminal(record, expected_experiment_id):
    if expected_experiment_id is not None and record["experiment_id"] != expected_experiment_id:
        raise CaptureError("RECOVERY_SCOPE_CHANGED", "实验目录中的标识已变化，请重新打开实验。")
    if record["state"] not in TERMINAL_STATES:
        raise CaptureError("CONTINUATION_ACTIVE", "请先结束采集或恢复中断台账，再准备续采。")


def inspect_continuation(root: str | Path, *, expected_experiment_id: str | None = None) -> dict:
    """Summarize evidence for choosing a complete phase to repeat in a new run."""
    with ExperimentReader(root) as reader:
        experiment = reader.experiment
        _require_terminal(experiment, expected_experiment_id)
        write_records_need_review = frame_write_records_need_review(reader, experiment)
        record, frame_counts, events = reader.phase_evidence(_PHASE_EVENTS + _ISSUE_EVENTS)
    _require_terminal(record, expected_experiment_id)
    plan = record["config"]
    cameras = [camera["id"] for camera in plan["cameras"]]
    rows = [{"phase": index, "mode": phase["mode"], "role": phase["role"],
             "confirmation": bool(phase.get("start_confirmation")), "hardware_pairing": bool(phase.get("hardware_pairing")),
             "saved_by_camera": dict.fromkeys(cameras, 0), "issues": {}}
            for index, phase in enumerate(plan["phases"], 1)]
    by_number = {row["phase"]: row for row in rows}
    for number, camera, count in frame_counts:
        if number in by_number and camera in by_number[number]["saved_by_camera"]:
            by_number[number]["saved_by_camera"][camera] = count
    unassigned_errors = 0
    for number, kind, stopped, count in events:
        row = by_number.get(number)
        if row is None:
            if kind == "error":
                unassigned_errors += count
        elif kind in _ISSUE_EVENTS:
            row["issues"][kind] = count
        elif kind == "phase_ended":
            row[kind] = True
            row["stop_requested"] = bool(stopped)
        else:
            row[kind] = True
    first_recheck = None
    has_phase_events = any(any(event in row for event in _PHASE_EVENTS) for row in rows)
    for row in rows:
        counts = row["saved_by_camera"]
        row["saved_frames"] = sum(counts.values())
        row["cameras_without_frames"] = ([camera for camera, count in counts.items() if count == 0]
                                          if row.get("phase_ended") and not row.get("stop_requested") else [])
        if row.get("stop_requested") or (row.get("phase_started") and not row.get("phase_ended")):
            state = "interrupted"
        elif row.get("phase_confirmation_cancelled"):
            state = "confirmation_cancelled"
        elif row["issues"] or row["cameras_without_frames"]:
            state = "needs_review"
        elif row.get("phase_ended"):
            state = "ended"
        elif not row["saved_frames"] and (has_phase_events or record["state"] != "completed"):
            state = "not_started"
        else:
            state = "unverified"
        row["state"] = state
        if first_recheck is None and state in ("interrupted", "confirmation_cancelled", "needs_review", "not_started"):
            first_recheck = row["phase"]
    return {"experiment_id": record["experiment_id"], "name": plan["name"], "state": record["state"],
            "cameras": cameras, "stages": rows, "suggested_phase": first_recheck or 1,
            "has_suggested_restart": first_recheck is not None, "unassigned_errors": unassigned_errors,
            "write_records_need_review": write_records_need_review}


def verified_archived_calibrations(root, record, code):
    files = []
    for attachment in record.get("calibrations", ()):
        path = (root / attachment["path"]).resolve()
        if (not path.is_relative_to(root / "calibration") or not path.is_file()
                or sha256_file(path) != attachment["sha256"]):
            raise CaptureError(code, f"已归档的标定文件缺失或已变化：{path}")
        files.append(str(path))
    return files


def continuation_plan(root: str | Path, config: CaptureConfig | None = None, *,
                      start_phase: int = 1, expected_experiment_id: str | None = None) -> CaptureConfig:
    """Keep the prior experiment immutable and identify the new run's parent.

    When reusing a stored plan, attach its archived calibration files and
    require fresh hardware counter anchors for the new run. A supplied plan
    may already carry independently measured anchors.
    """
    with ExperimentReader(root) as reader:
        record = reader.experiment
        _require_terminal(record, expected_experiment_id)
        if config is not None and type(start_phase) is int and start_phase == 1:
            reference = continuation_reference(config)
            if reference is not None and reference["experiment_id"] == record["experiment_id"]:
                start_phase = reference.get("source_start_phase", 1)
        elif config is not None:
            raise CaptureError("CONTINUATION_PHASE_INVALID", "请用原计划选择起始阶段，或直接提交已编辑的完整计划。")
        if type(start_phase) is not int or not 1 <= start_phase <= len(record["config"]["phases"]):
            raise CaptureError("CONTINUATION_PHASE_INVALID", "请选择原实验中的一个采集阶段作为起点。")
        if config is not None:
            data = config.to_dict()
        else:
            data = CaptureConfig.from_dict(record["config"]).to_dict()
            data["calibration_files"] = verified_archived_calibrations(reader.root, record, "CONTINUATION_CALIBRATION_INVALID")
            data["phases"] = data["phases"][start_phase - 1:]
            for phase in data["phases"]:
                phase.pop("hardware_pairing", None)
        data["metadata"]["continuation_of"] = record["experiment_id"]
        data["metadata"]["continuation_parent"] = {"directory": str(reader.root)}
        data["metadata"][CONTINUATION_REFERENCE] = _reference(reader, record, start_phase)
        data["metadata"].pop(REPEAT_OF, None)
        data["metadata"].pop(REPEAT_PARENT, None)
    return CaptureConfig.from_dict(data)
