"""Compare trial readbacks with the selected camera before formal acquisition."""

from __future__ import annotations

from .cameras.settings_probe import check_camera_settings, readbacks_affected_by_pending_writes
from .config import CaptureError
from .continuation import camera_identity, camera_readback, comparable_readback, same_camera_value


_REUSABLE = {"exposure_us": ("exposure_raw", ("us", "µs", "μs")), "gain_db": ("gain_raw", ("dB",))}


def compare_trial_setup(config, trial, settings):
    result = {}
    for camera in config.cameras:
        saved = trial["cameras"].get(camera.id, {})
        before = camera_readback(saved.get("setup_readback", {}))
        device = saved.get("device_identity")
        if not (before or device):
            continue
        checked = settings.get(camera.id)
        features = checked.readback if checked and checked.readback is not None else {}
        now = camera_readback(features)
        basis = checked.basis if checked and (checked.readback is not None or checked.error) else "not_probed"
        identity = camera_identity({camera.id: device}, camera.id, checked.device if checked else None)
        pending = (readbacks_affected_by_pending_writes(camera.id.partition(":")[0], camera.settings, features)
                   if basis == "device_features" else set())
        fields, matched = [], 0
        for field, old in before.items():
            live = now.get(field)
            comparable = comparable_readback(old, live)
            if field not in pending and comparable and same_camera_value(old["value"], live["value"]):
                matched += 1
                continue
            status = "pending" if field in pending else "changed" if comparable else "unverified"
            fields.append({"field": field, "recorded": old["value"], "unit": old["unit"], "status": status,
                           "current": live["value"] if live else None, "current_unit": live["unit"] if live else None})
        result[camera.id] = {"basis": basis, "identity": identity, "matched_fields": matched, "fields": fields}
    return result


def cameras_with_trial_changes(setup):
    return [key for key, row in setup.items()
            if row["identity"] == "changed" or any(item["status"] == "changed" for item in row["fields"])]


def suggest_trial_settings(config, matching, settings):
    """Offer unset manual values that agree across the compatible verified trials."""
    verified = [(row, trial) for row, trial in matching.values() if row["status"] == "matched" and row["observed"]]
    proposed = {}
    for camera in config.cameras:
        checked = settings.get(camera.id)
        if not checked or not checked.readback:
            continue
        reports = [row.get("camera_readbacks", {}).get(camera.id, {}) for row, _ in verified]
        if not reports or any(row.get("identity") not in ("matched", "path_matched") for row in reports):
            continue
        changes = {item["field"] for row in reports for item in row["fields"] if item["status"] == "changed"}
        readings = [camera_readback(trial["cameras"][camera.id].get("setup_readback", {})) for _, trial in verified]
        additions = {}
        for field, (alternate, units) in _REUSABLE.items():
            feature = checked.readback.get(field)
            if field not in changes or field in camera.settings or alternate in camera.settings or feature is None:
                continue
            saved = readings[0].get(field)
            if not saved or saved.get("unit") not in units or feature.unit not in units or type(saved["value"]) not in (int, float):
                continue
            value = saved["value"]
            if not all(comparable_readback(saved, item.get(field)) and same_camera_value(value, item[field]["value"])
                       for item in readings):
                continue
            try:
                feature.validate(value)
            except CaptureError:
                continue
            additions[field] = value
        backend = camera.id.partition(":")[0]
        if additions and not check_camera_settings(backend, {**camera.settings, **additions}, checked.readback).invalid:
            proposed[camera.id] = additions
    return proposed
