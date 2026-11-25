"""Rehearse a capture phase on the planned output volume."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import sqlite3
import time
from pathlib import Path
from statistics import median
from uuid import uuid4

from .cameras._gentl_multipart import IMAGE_PART
from .config import REPEAT_OF, REPEAT_PARENT, CaptureConfig, CaptureError, Phase, QualityRegion, inside_repeat_source, positive_number
from .output import output_ancestor, resolve_output_path
from .continuation import CONTINUATION_REFERENCE, camera_readback
from .hardware_counters import summarize_hardware_counters
from .storage import ExperimentReader, TERMINAL_STATES, frame_image_records
from .trial_quality import inspect_trial_images, trial_sensor_values


TRIAL_SCHEMA = "straintrace.capture.trial/v1"
DEFAULT_TRIAL_SECONDS = 8.0
READBACK_FIELDS = (
    "width", "height", "pixel_format", "exposure_us", "exposure_auto", "gain_db", "fps", "fps_enable", "trigger_source",
    "trigger_activation",
)
STORAGE_SETTINGS = ("durability", "queue_capacity", "max_buffer_bytes", "segment_frames", "min_free_bytes")


def volume_identity(directory: Path) -> str | None:
    try:
        device = directory.stat().st_dev
    except OSError:
        return None
    host = platform.node()
    return hashlib.sha256(f"{host}\0{device}".encode("utf-8")).hexdigest() if host else None


def _planned_phase(phase) -> dict:
    data = phase.to_dict()
    # The pulse anchors are specific to each run.
    data.pop("hardware_pairing", None)
    # A one-phase performance trial starts at the operator's trial command.
    data.pop("start_confirmation", None)
    return data


def _setup(config: dict) -> dict:
    return {
        "camera_settings": {camera["id"]: camera["settings"] for camera in config["cameras"]},
        "cti_files": config["cti_files"],
        "storage": {name: config[name] for name in STORAGE_SETTINGS},
    }


def _trial_phase(original, formal_phase, seconds):
    phase = dict(formal_phase)
    if phase["mode"] != "single":
        phase["duration_s"] = min(seconds, phase.get("duration_s", seconds))
    if original.repeats_events and original.event_count:
        trial_rounds = math.ceil(phase["duration_s"] * original.fps)
        if trial_rounds < original.minimum_event_rounds:
            phase["event_count"] = 0
    return phase


def prepare_trial(
    config: CaptureConfig, output: str | Path, *, phase_number: int | None = None,
    seconds: float = DEFAULT_TRIAL_SECONDS, rate_source_experiment_id: str | None = None,
    rate_source_directory: str | Path | None = None,
    preserve_start_confirmation: bool = False,
) -> tuple[CaptureConfig, Path]:
    """Keep the selected camera and write settings and run one phase separately."""
    seconds = positive_number(seconds, "trial duration")
    phases = config.phases
    if phase_number is None:
        phase_number = next((i for i, item in enumerate(phases, 1) if item.role == "test"), 1)
    if type(phase_number) is not int or not 1 <= phase_number <= len(phases):
        raise CaptureError("TRIAL_PHASE_INVALID", f"选择 1 到 {len(phases)} 之间的采集阶段。")

    formal = resolve_output_path(output)
    if inside_repeat_source(config, formal):
        raise CaptureError("OUTPUT_INSIDE_REPEAT_SOURCE", "请选择来源实验目录之外的新输出位置。")
    directory = formal.parent / f"trial-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    original = phases[phase_number - 1]
    formal_phase = _planned_phase(original)
    phase = _trial_phase(original, formal_phase, seconds)
    if preserve_start_confirmation and original.start_confirmation:
        phase["start_confirmation"] = original.start_confirmation
    data = config.to_dict()
    data["name"] = f"{config.name} · {'建议频率复测' if rate_source_experiment_id else '试采'}"
    data["phases"] = [phase]
    data["metadata"].pop("continuation_of", None)
    data["metadata"].pop("continuation_parent", None)
    data["metadata"].pop(CONTINUATION_REFERENCE, None)
    data["metadata"].pop(REPEAT_OF, None)
    data["metadata"].pop(REPEAT_PARENT, None)
    data["metadata"]["capture_trial"] = {
        "schema": TRIAL_SCHEMA,
        "formal_output": str(formal),
        "phase_number": phase_number,
        "mode": phase["mode"],
        "duration_limit_s": phase.get("duration_s"),
        "requires_fresh_hardware_anchors": original.hardware_pairing is not None,
        "scope": {"phase": formal_phase, "volume": volume_identity(output_ancestor(formal))},
    }
    if rate_source_experiment_id is not None:
        evidence = data["metadata"]["capture_trial"]
        evidence["rate_source_experiment_id"] = rate_source_experiment_id
        if rate_source_directory is not None:
            source = Path(rate_source_directory).absolute()
            hint = evidence["rate_source"] = {"directory": str(source)}
            try:
                hint["relative_directory"] = Path(os.path.relpath(source, directory)).as_posix()
            except ValueError:
                pass  # Windows volumes do not always share a relative path.
    return CaptureConfig.from_dict(data), directory


def trial_metadata(config: dict) -> dict | None:
    evidence = config.get("metadata", {}).get("capture_trial")
    phases = config.get("phases", [])
    if (isinstance(evidence, dict) and evidence.get("schema") == TRIAL_SCHEMA
        and len(phases) == 1 and phases[0]["mode"] == evidence.get("mode")
        and type(evidence.get("phase_number")) is int and evidence["phase_number"] >= 1
        and isinstance(evidence.get("formal_output"), str)
        and "duration_limit_s" in evidence and evidence["duration_limit_s"] == phases[0].get("duration_s")
        and type(evidence.get("requires_fresh_hardware_anchors")) is bool):
        return evidence
    return None


def _trial_findings(report: dict) -> list[dict]:
    findings = []
    verification = report["verification"]
    phase = report["phase"]
    if error := report["error"]:
        findings.append({"code": "TRIAL_CAPTURE_ERROR", "capture_code": error["code"]})
    if verification["issues"]:
        findings.append({"code": "TRIAL_FILE_ISSUES", "count": len(verification["issues"])})
    if report["cameras_without_frames"]:
        findings.append({"code": "TRIAL_NO_FRAMES", "cameras": report["cameras_without_frames"], "mode": phase["mode"]})
    changing = [identity for identity, row in report["cameras"].items() if row["committed_frames"] and row["geometry"] is None]
    if changing:
        findings.append({"code": "TRIAL_VARIABLE_GEOMETRY", "cameras": changing})
    losses = verification["reported_camera_losses"]
    if unsaved := sum(row.get("app_unsaved", 0) for row in losses.values()):
        findings.append({"code": "TRIAL_UNSAVED_FRAMES", "count": unsaved})
    camera_losses = [camera for camera, row in losses.items() if any(key != "app_unsaved" for key in row)]
    if camera_losses:
        findings.append({"code": "TRIAL_CAMERA_LOSSES", "cameras": camera_losses})
    if misses := verification["scheduled_misses"]:
        findings.append({"code": "TRIAL_SCHEDULE_MISSES", "count": misses})
    if short := verification["short_pre_windows"]:
        findings.append({"code": "TRIAL_SHORT_PRE_WINDOW", "count": short})
    peak = report["raw_buffer_high_water_bytes"]
    limit = report["raw_buffer_limit_bytes"]
    if peak is not None and peak >= .85 * limit:
        findings.append({"code": "TRIAL_HIGH_BUFFER", "fraction": peak / limit, "mode": phase["mode"]})
    queued = report["write_buffer"]
    high, capacity = queued.get("queue_high_water"), queued.get("queue_capacity")
    if high is not None and capacity and high >= .85 * capacity:
        findings.append({"code": "TRIAL_HIGH_WRITE_QUEUE", "peak": high, "capacity": capacity,
                         "queue_mode": "auto" if report["setup"]["storage"]["queue_capacity"] == "auto" else "fixed"})
    if report["sustained_write_lag"]:
        findings.append({"code": "TRIAL_SUSTAINED_WRITE_LAG"})
    if phase["mode"] in ("timed", "continuous") and report["observed"]:
        for identity, camera in report["cameras"].items():
            if phase["mode"] == "timed":
                target = phase["fps"]
            else:
                settings = camera["configured_features"]
                enabled = camera["phase_feature_readback"].get("fps_enable", {}).get("value", settings.get("fps_enable"))
                target = settings.get("fps") if enabled not in (False, "Off") else None
            measured = camera["host_receive_fps"]
            if (camera["committed_frames"] >= 10 and measured is not None
                and type(target) in (int, float) and target > 0 and measured < .90 * target):
                findings.append({"code": "TRIAL_LOW_RECEIPT_RATE", "camera": identity, "mode": phase["mode"],
                                 "observed_fps": measured, "target_fps": target})
    return findings


def _scope_describes_trial(scope: dict, phase: dict) -> bool:
    stored = scope.get("phase")
    if not isinstance(stored, dict):
        return False
    actual = dict(phase)
    actual.pop("start_confirmation", None)
    if phase["mode"] == "single":
        return stored == actual
    limit = phase.get("duration_s")
    if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
        return False
    try:
        original = Phase.from_dict(stored)
    except (CaptureError, ValueError):
        return False
    return _trial_phase(original, stored, limit) == actual


def _camera_setting_changes(before, after):
    changes = {}
    for camera in sorted(before.keys() & after.keys()):
        previous, current = before[camera], after[camera]
        fields = [{"field": name, "trial": previous.get(name), "plan": current.get(name)}
                  for name in sorted(previous.keys() | current.keys()) if previous.get(name) != current.get(name)]
        if fields:
            changes[camera] = fields
    return changes


_IMAGE_CHECK_FIELDS = ("mode", "available", "selected", "checked", "flagged")


def _region_checks(config, report, image_checks):
    result = {}
    for camera, full in image_checks.items():
        pair = report["cameras"][camera]["image_check"].get("region_check")
        if pair is None:
            continue
        check, fallback = pair["check"], pair["fallback"]
        requested = config.quality_regions.get(camera)
        if requested is None or requested.to_dict() != pair["region"]:
            fallback = "plan_changed"
        elif not (report["observed"] and report["verification"].get("file_hashes_checked")):
            fallback = "integrity"
        elif fallback is None and (not check or full["checked"] != full["selected"] or not check["selected"]
                                   or check["selected"] != full["selected"] or check["checked"] != check["selected"]):
            fallback = "incomplete"
        result[camera] = {"region": pair["region"], "fallback": fallback, "used_for_review": fallback is None,
                          "check": {key: check[key] for key in _IMAGE_CHECK_FIELDS} if check else None}
    return result


def trial_image_alerts(assessment):
    regional = assessment.get("region_checks", {})
    return sum(regional[camera]["check"]["flagged"] if regional.get(camera, {}).get("used_for_review") else check["flagged"]
               for camera, check in assessment.get("image_checks", {}).items())


def compare_trials(config: CaptureConfig, output_volume: str | None, reports) -> tuple[list[dict], dict[int, tuple[dict, dict]]]:
    """Associate the newest compatible receipt with each formal phase."""
    setup = _setup(config.to_dict())
    assessments, matching = [], {}
    for report in sorted(reports, key=lambda item: (item["created_utc_ns"], item["output"])):
        evidence = report["trial"]
        phase_number = evidence["phase_number"]
        scope = evidence.get("scope")
        entry = {
            "phase": phase_number, "output": report["output"], "experiment_id": report["experiment_id"],
            "observed": report["observed"], "duration_limit_s": evidence["duration_limit_s"],
            "trial_state": report.get("verification", {}).get("state"),
            "sustained_write_lag": report.get("sustained_write_lag", False),
            "findings": report["findings"], "changes": [],
        }
        image_checks = {
            camera: {key: check[key] for key in _IMAGE_CHECK_FIELDS}
            for camera, row in report["cameras"].items() if (check := row.get("image_check")) is not None
        }
        if image_checks:
            entry["image_checks"] = image_checks
        if checks := _region_checks(config, report, image_checks):
            entry["region_checks"] = checks
        assessments.append(entry)
        if not (isinstance(scope, dict) and scope.get("volume") and output_volume
                and _scope_describes_trial(scope, report["phase"])):
            entry["status"] = "unbound"
            continue
        changes = entry["changes"]
        if phase_number > len(config.phases) or scope["phase"] != _planned_phase(config.phases[phase_number - 1]):
            changes.append("phase")
        elif (recorded := report["phase"].get("start_confirmation")) is not None and recorded != config.phases[phase_number - 1].start_confirmation:
            changes.append("start_confirmation")
        if scope["volume"] != output_volume or report["volume"] != scope["volume"]:
            changes.append("volume")
        changes.extend(key for key in setup if report["setup"][key] != setup[key])
        if "camera_settings" in changes:
            entry["camera_setting_changes"] = _camera_setting_changes(report["setup"]["camera_settings"], setup["camera_settings"])
        entry["status"] = "changed" if changes else "matched"
        if not changes:
            matching[phase_number] = (entry, report)
    return assessments, matching


def trial_report(root: str | Path, *, inspect_images=True, cancelled=None, expected_experiment_id=None,
                 full_image_cameras=(), image_quality_regions=None) -> dict:
    """Read receipt cadence, camera readback and file integrity from a finished rehearsal."""
    with ExperimentReader(root) as reader:
        record = reader.experiment
        if expected_experiment_id is not None and record["experiment_id"] != expected_experiment_id:
            raise CaptureError("TRIAL_REFERENCE_MISMATCH", str(reader.root))
        metadata = trial_metadata(record["config"])
        if metadata is None:
            raise CaptureError("TRIAL_REQUIRED", "所选实验没有试采记录。")
        if record["state"] not in TERMINAL_STATES:
            raise CaptureError("TRIAL_ACTIVE", "试采仍在运行。结束写入后可查看报告。")
        verification = reader.verify()
        events = reader.events()
        started = next((event for event in events if event["kind"] == "phase_started"), {})
        event_windows = [event for event in events if event["kind"] == "event_window"]
        event_phase = next((event for event in events if event["kind"] == "event_phase_summary"), None)
        readbacks = started.get("camera_features", {})
        quality_regions = record["config"].get("quality_regions")
        if inspect_images and image_quality_regions is None:
            image_quality_regions = {key: QualityRegion.from_dict(region) for key, region in (quality_regions or {}).items()}
        summary = record.get("summary") or {}
        registered = {row["device"]["id"]: row for row in reader.cameras()}
        cameras = {}
        for item in record["config"]["cameras"]:
            identity = item["id"]
            rows = reader.frames(camera_id=identity)
            instants = [row["timestamps"]["received_monotonic"]["value"] for row in rows]
            span = (instants[-1] - instants[0]) / 1e9 if len(instants) >= 2 else 0
            selection_delays = []
            for row, instant in zip(rows, instants):
                requested = row.get("backend_metadata", {}).get("host_selection_request_monotonic_ns")
                if type(requested) is int and instant > requested:
                    selection_delays.append((instant - requested) / 1e6)
            images = [frame_image_records(row) for row in rows]
            layouts = {tuple((tuple(part["shape"]), part["pixel_format"], part["dtype"], part["significant_bits"],
                              part["color_model"], (part.get("backend_metadata") or {}).get(IMAGE_PART, {}).get("selector"))
                             for part in parts)
                       for parts in images}
            formats = sorted({(shape, fmt, bits) for layout in layouts for shape, fmt, _dtype, bits, _color, _selector in layout})
            geometry = None
            if len(layouts) == 1:
                layout = layouts.pop()
                shape, pixel_format, *_ = layout[0]
                if all(dtype in ("uint8", "uint16") for _, _, dtype, *_ in layout):
                    byte_count = sum(math.prod(dims) * (2 if dtype == "uint16" else 1) for dims, _, dtype, *_ in layout)
                    geometry = {"width_px": shape[1], "height_px": shape[0], "pixel_format": pixel_format,
                                "frame_bytes": byte_count,
                                "saved_parts": [
                                    {"shape": list(dims), "dtype": dtype, "pixel_format": fmt, "significant_bits": bits,
                                     "color_model": color, "selector": selector}
                                    for dims, fmt, dtype, bits, color, selector in layout
                                ]}
            actual = {key: value for key, value in readbacks.get(identity, {}).items() if key in READBACK_FIELDS}
            counters = sum(row["native_counter"] is not None for row in rows)
            registration = registered.get(identity, {})
            device = registration.get("device", {})
            backend = device.get("backend", identity.partition(":")[0])
            unselected = None if backend == "uvc" else (summary.get("cameras") or {}).get(identity, {}).get("host_unselected_frames")
            cameras[identity] = {
                "backend": backend,
                "device_support": device.get("support"),
                "committed_frames": len(rows),
                "host_receive_span_s": span if span > 0 else None,
                "host_receive_fps": (len(instants) - 1) / span if span > 0 else None,
                "capture_method": started.get("camera_methods", {}).get(identity),
                "host_unselected_frames": unselected,
                "host_selection_delay_ms": ({"median": median(selection_delays), "maximum": max(selection_delays)}
                                            if selection_delays else None),
                "frames_with_native_counter": counters,
                "formats": [{"shape": list(shape), "pixel_format": pixel_format, "significant_bits": bits}
                            for shape, pixel_format, bits in formats],
                "configured_features": item["settings"],
                "idle_fps_feature": registration.get("features", {}).get("fps"),
                "phase_feature_readback": actual,
                "setup_readback": camera_readback(readbacks.get(identity, {}), phase=True),
                "device_identity": {name: device.get(name) for name in ("backend", "serial", "identity_scope")} if device else None,
                "geometry": geometry,
            }
            if metadata["mode"] == "hardware":
                cameras[identity]["hardware_counters"] = summarize_hardware_counters(rows)
            if inspect_images:
                cameras[identity]["image_check"] = inspect_trial_images(
                    reader, rows, all_frames=identity in full_image_cameras, cancelled=cancelled,
                    quality_region=image_quality_regions.get(identity), sensor_values=trial_sensor_values(started, identity),
                )

        unobserved = [identity for identity, row in cameras.items() if not row["committed_frames"]]
        observed = verification["complete"] and not unobserved
        write_timing = summary.get("write_timing")
        report = {
            "schema": TRIAL_SCHEMA,
            "output": str(reader.root),
            "experiment_id": record["experiment_id"],
            "created_utc_ns": record["created_utc_ns"],
            "setup": _setup(record["config"]),
            "volume": volume_identity(reader.root),
            "trial": metadata,
            "phase": record["config"]["phases"][0],
            "event_window": event_windows[0] if event_windows else None,
            **({"event_windows": event_windows, "event_phase": event_phase} if event_phase is not None else {}),
            **({"csv_triggers": summary["csv_triggers"]} if summary.get("csv_triggers") else {}),
            "cameras": cameras,
            **({"quality_regions": quality_regions, "image_quality_watch": summary.get("image_quality_watch")}
               if quality_regions else {}),
            "cameras_without_frames": unobserved,
            "observed": observed,
            "verification": verification,
            "disk_bytes_per_s": summary.get("disk_bytes_per_s"),
            "elapsed_s": summary.get("elapsed_s"),
            "raw_buffer_high_water_bytes": summary.get("raw_buffer_high_water_bytes"),
            "raw_buffer_limit_bytes": record["config"]["max_buffer_bytes"],
            "write_buffer": verification["write_buffer"],
            "write_timing": write_timing,
            "sustained_write_lag": bool(observed and write_timing and write_timing["sustained_lag"]),
            "durability": record["config"]["durability"],
            "error": summary.get("error"),
        }
        report["findings"] = _trial_findings(report)
        return report


def refresh_trial_image_regions(config, output, reports, *, cancelled=None):
    """Recheck matched original TIFFs before a desktop plan relies on a saved inspection box."""
    if not config.quality_regions or not reports:
        return reports
    try:
        volume = volume_identity(output_ancestor(resolve_output_path(output)))
    except (OSError, RuntimeError, ValueError):
        return reports
    _, matching = compare_trials(config, volume, reports)
    matched_reports = {id(evidence) for _assessment, evidence in matching.values()}
    if not matched_reports:
        return reports
    refreshed = []
    for report in reports:
        if id(report) in matched_reports:
            full = {camera for camera, row in report["cameras"].items() if (row.get("image_check") or {}).get("mode") == "all"}
            try:
                fresh = trial_report(report["output"], expected_experiment_id=report["experiment_id"],
                                     full_image_cameras=full, image_quality_regions=config.quality_regions, cancelled=cancelled)
                if fresh["trial"]["phase_number"] != report["trial"]["phase_number"]:
                    raise CaptureError("TRIAL_REFERENCE_CHANGED", report["output"])
                report = fresh
            except (CaptureError, OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
                if getattr(exc, "code", None) == "TRIAL_IMAGE_CHECK_CANCELLED":
                    raise
                findings = [item for item in report["findings"] if item["code"] != "TRIAL_RECHECK_UNAVAILABLE"]
                report = {**report, "observed": False, "findings": [*findings, {"code": "TRIAL_RECHECK_UNAVAILABLE"}],
                          "verification": {**report["verification"], "ok": False, "file_hashes_checked": False}}
        refreshed.append(report)
    return tuple(refreshed)
