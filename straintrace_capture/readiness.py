"""Read-only checks and raw-payload estimates for a planned acquisition."""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from .cameras import CameraRegistry
from .cameras._genapi import image_layout
from .cameras._gentl_multipart import AUTO_PART, IMAGE_PART, SAVE_ALL_2D
from .cameras.base import HOST_NEXT_RECEIPT, Image, TriggerSupport, is_physical_trigger_input
from .cameras.settings_probe import SettingsCheck, check_camera_settings, check_simulator_settings
from .cameras.simulated import FEATURES as SIMULATOR_FEATURES
from .cameras.uvc import UVC_TRIGGER_SUPPORT
from .config import CameraConfig, CaptureConfig, CaptureError, Phase, automatic_queue_capacity, repeat_source_directory
from .continuation import compare_continuation_setup, continuation_reference
from .csv_trigger import check_csv_triggers
from .image_quality_watch import quality_region_mismatch, quality_region_settings_mismatch
from .output import inspect_output
from .rate_trial import suggest_trial_rate
from .trial import compare_trials, trial_image_alerts, volume_identity
from .trial_setup import cameras_with_trial_changes, compare_trial_setup, suggest_trial_settings


def _backend_trigger_support(identity: str) -> TriggerSupport | None:
    backend = identity.partition(":")[0]
    if backend == "uvc":
        return UVC_TRIGGER_SUPPORT
    if backend == "sim":
        return TriggerSupport(software=True, hardware=False, basis="backend")
    return None


@dataclass(frozen=True, slots=True)
class CameraConnectionCheck:
    state: Literal["available", "open_failed", "close_failed"]
    reason: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CameraSetupProbe:
    triggers: dict[str, TriggerSupport]
    settings: dict[str, SettingsCheck]
    connections: dict[str, CameraConnectionCheck]


def probe_camera_setup(config: CaptureConfig, registry: CameraRegistry | None = None, *,
                       triggers=True, settings=True, include_readback=False) -> tuple[dict[str, TriggerSupport], dict[str, SettingsCheck]]:
    """Read settings and trigger catalogs using at most one connection per selected camera."""
    result = _probe_camera_setup(config, registry, triggers=triggers, settings=settings,
                                 include_readback=include_readback, check_connections=False)
    return result.triggers, result.settings


def probe_preflight_cameras(config: CaptureConfig, registry: CameraRegistry | None = None, *,
                            triggers=True, include_readback=False) -> CameraSetupProbe:
    """Open and release each selected physical camera, reading catalogs when needed."""
    return _probe_camera_setup(config, registry, triggers=triggers, settings=True,
                               include_readback=include_readback, check_connections=True)


def _probe_camera_setup(config, registry, *, triggers, settings, include_readback, check_connections):
    registry = registry if registry is not None else CameraRegistry(cti_files=config.cti_files)
    include_readback = include_readback or continuation_reference(config) is not None
    trigger_evidence, setting_evidence, connections = {}, {}, {}
    for item in config.cameras:
        backend = item.id.partition(":")[0]
        known = _backend_trigger_support(item.id) if triggers else None
        if known:
            trigger_evidence[item.id] = known
        read_trigger = triggers and known is None
        read_settings = settings and (include_readback or (bool(item.settings) and backend != "sim"))
        check_connection = check_connections and backend != "sim"
        if not (read_trigger or read_settings or check_connection):
            continue
        opened = False
        try:
            with registry.camera(item.id) as device:
                opened = True
                if read_trigger:
                    try:
                        trigger_evidence[item.id] = device.trigger_support()
                    except Exception as exc:
                        trigger_evidence[item.id] = TriggerSupport(error=str(exc))
                if read_settings:
                    try:
                        features = device.features()
                        checked = check_camera_settings(backend, item.settings, features)
                        setting_evidence[item.id] = (replace(checked, readback=features, device=device.device)
                                                     if include_readback else checked)
                    except Exception as exc:
                        setting_evidence[item.id] = SettingsCheck("unavailable", error=str(exc))
        except Exception as exc:
            if check_connection:
                reason = exc.code if isinstance(exc, CaptureError) else ("CAMERA_CLOSE_FAILED" if opened else "CAMERA_OPEN_FAILED")
                connections[item.id] = CameraConnectionCheck("close_failed" if opened else "open_failed", reason, str(exc))
            if read_trigger:
                trigger_evidence.setdefault(item.id, TriggerSupport(error=str(exc)))
            if read_settings:
                setting_evidence[item.id] = SettingsCheck("unavailable", error=str(exc))
        else:
            if check_connection:
                connections[item.id] = CameraConnectionCheck("available")
    return CameraSetupProbe(trigger_evidence, setting_evidence, connections)


def probe_trigger_support(config: CaptureConfig, registry: CameraRegistry | None = None) -> dict[str, TriggerSupport]:
    """Read selected devices' trigger catalogs without writing settings or starting streams."""
    return probe_camera_setup(config, registry, settings=False)[0]


def trigger_evidence(identity, evidence):
    reported = evidence.get(identity)
    if reported is not None and reported.basis != "unknown":
        return reported
    return _backend_trigger_support(identity) or reported or TriggerSupport()


def apply_phase_suggestion(phase: Phase, checked: Phase, replacement: dict) -> dict:
    """Apply the suggested edits to the original plan outside a trial's temporary limits."""
    before = checked.to_dict()
    data = phase.to_dict()
    for key in before.keys() - replacement.keys():
        data.pop(key, None)
    data.update({key: value for key, value in replacement.items() if key not in before or value != before[key]})
    return Phase.from_dict(data).to_dict()


def _geometry(camera: CameraConfig, preview: Image | None) -> dict:
    settings = dict(camera.settings)
    basis = "configured"
    if camera.id.startswith("sim:"):
        for key in ("width", "height", "pixel_format", "fps"):
            settings.setdefault(key, SIMULATOR_FEATURES[key].value)
        basis = "simulator"
    rate = settings.get("fps")
    configured_fps = (
        rate if type(rate) in (float, int) and 0 < rate <= 1e12
        and settings.get("fps_enable") not in (False, "Off") else None
    )
    if preview is not None:
        height, width = preview.pixels.shape[:2]
        return {"width_px": width, "height_px": height, "pixel_format": preview.pixel_format,
                "frame_bytes": preview.raw_bytes, "basis": "preview", "configured_fps": configured_fps,
                "saved_parts": [
                    {"shape": list(image.pixels.shape), "dtype": image.pixels.dtype.name, "pixel_format": image.pixel_format,
                     "significant_bits": image.significant_bits, "color_model": image.color_model,
                     "selector": (image.metadata.get(IMAGE_PART) or {}).get("selector")}
                    for image in (preview, *preview.auxiliary_images)
                ]}
    width, height, pixel_format = (settings.get(key) for key in ("width", "height", "pixel_format"))
    frame_bytes = None
    needs_preview = camera.id.startswith("gentl:") and (
        settings.get(IMAGE_PART, AUTO_PART) != AUTO_PART or settings.get(SAVE_ALL_2D))
    if not needs_preview and type(width) is int and width > 0 and type(height) is int and height > 0:
        try:
            bits, model = image_layout(pixel_format)
        except CaptureError:
            pass
        else:
            channels = len(model) if model in ("rgb", "bgr", "rgba", "bgra") else 1
            frame_bytes = width * height * channels * (1 if bits == 8 else 2)
    return {"width_px": width, "height_px": height, "pixel_format": pixel_format,
            "frame_bytes": frame_bytes, "basis": basis if frame_bytes is not None else "unavailable",
            "configured_fps": configured_fps}


def _frame_limit(phase: Phase) -> int | None:
    if phase.mode == "event":
        per_event = phase.pre_frames + (phase.pre_frames > 0) + phase.post_frames
        rounds = math.ceil(phase.duration_s * phase.fps) if phase.duration_s is not None else None
        if not phase.event_count:
            return rounds
        limit = per_event * phase.event_count
        return min(limit, rounds) if rounds is not None and phase.repeats_events else limit
    if phase.mode == "timed" and phase.duration_s is not None:
        scheduled = math.ceil(phase.duration_s * phase.fps)
        return scheduled if phase.frame_count is None else min(scheduled, phase.frame_count)
    return phase.frame_count


def event_window_shortfall(phase: Phase) -> int | None:
    if phase.mode == "event" and phase.duration_s is not None:
        planned_rounds = math.ceil(phase.duration_s * phase.fps)
        if planned_rounds < phase.minimum_event_rounds:
            return planned_rounds
    return None


def plan_camera_geometry(cameras, preview_images=None):
    preview_images = preview_images or {}
    return {camera.id: _geometry(camera, preview_images.get(camera.id)) for camera in cameras}


def estimate_phase(phase: Phase, geometry: Mapping[str, dict]) -> dict:
    """Estimate receipts and raw pixels from a phase and known camera layouts."""
    cameras = list(geometry.values())
    sizes = [row["frame_bytes"] for row in cameras]
    round_bytes = sum(sizes) if sizes and None not in sizes else None
    limit = _frame_limit(phase)
    payload = limit * round_bytes if limit is not None and round_bytes is not None else None
    stream_rate = None
    estimated_frames = None
    if phase.mode == "continuous" and round_bytes is not None and all(row["configured_fps"] is not None for row in cameras):
        stream_rate = sum(row["frame_bytes"] * row["configured_fps"] for row in cameras)
        if limit is None and phase.duration_s is not None:
            counts = [math.ceil(row["configured_fps"] * phase.duration_s) for row in cameras]
            estimated_frames = sum(counts)
            payload = sum(count * size for count, size in zip(counts, sizes))
    batches = None
    if round_bytes is not None:
        if phase.mode in ("continuous", "hardware"):
            batches = max(sizes)
        else:
            window = phase.pre_frames + 1 if phase.mode == "event" else 1
            batches = round_bytes * window
    return {"frame_limit_per_camera": limit, "raw_payload_bytes": payload,
            "scheduled_raw_bytes_per_s": round_bytes * phase.fps if round_bytes is not None and phase.fps else None,
            "configured_stream_raw_bytes_per_s": stream_rate, "frames_at_configured_rate": estimated_frames,
            "one_batch_raw_bytes": batches}


def _phase_geometry(camera_rows, trial):
    if trial is None:
        return camera_rows
    assessment, evidence = trial
    if not evidence["observed"] or not evidence["verification"]["file_hashes_checked"]:
        return camera_rows
    measured = {key: row["geometry"] for key, row in evidence["cameras"].items() if row["geometry"] is not None}
    for key, geometry in measured.items():
        row = camera_rows[key]
        if row["basis"] == "preview" and any(row[field] != value for field, value in geometry.items()):
            assessment["status"] = "changed"
            assessment["changes"].append("preview_geometry")
            return camera_rows
    return {
        key: {**row, **measured[key], "basis": "trial"} if key in measured and row["basis"] != "preview" else row
        for key, row in camera_rows.items()
    }


def _check_quality_regions(config, preview_images, camera_settings):
    results = {}
    for camera in config.cameras:
        if (region := config.quality_regions.get(camera.id)) is None:
            continue
        reason = quality_region_settings_mismatch(region, camera.settings)
        frame = preview_images.get(camera.id)
        settings = camera_settings.get(camera.id)
        features = settings.readback if settings is not None else None
        if reason is None and frame is not None:
            values = {name: feature.value for name, feature in features.items()} if features is not None else None
            reason = quality_region_mismatch(region, frame, values)
        verified_preview = frame is not None and (not region.sensor_window or features is not None)
        results[camera.id] = {"bounds": list(region.bounds), "fallback": reason,
                              "verified_preview": reason is None and verified_preview}
    return results


def inspect_plan(
    config: CaptureConfig,
    output: str | Path,
    *,
    preview_images: Mapping[str, Image] | None = None,
    trigger_support: Mapping[str, TriggerSupport] | None = None,
    camera_settings: Mapping[str, SettingsCheck] | None = None,
    camera_connections: Mapping[str, CameraConnectionCheck] | None = None,
    discovered_ids: set[str] | None = None,
    trial_reports: Sequence[dict] = (),
) -> dict:
    """Inspect a plan without opening a camera or creating an output directory.

    Preview images come from the same camera configuration. Trigger evidence is
    tied to the selected device catalogs or built-in interfaces. Matching verified
    trials can supply frame geometry. Explicit connection checks can identify
    a camera that could not be opened or released. Byte figures cover uncompressed samples.
    """
    issues: list[dict] = check_csv_triggers(config.phases)

    def add(code, severity="warning", **details):
        issues.append({"code": code, "severity": severity, **details})

    destination = inspect_output(output, repeat_source=repeat_source_directory(config.metadata), suggest=True)
    root = destination.path
    ancestor = destination.volume_directory or root.parent
    issues.extend(destination.issues)
    disk_free = None
    output_volume = None
    try:
        if destination.volume_directory is not None:
            disk_free = shutil.disk_usage(ancestor).free
            if trial_reports:
                output_volume = volume_identity(ancestor)
    except OSError as exc:
        add("OUTPUT_INSPECTION_FAILED", "block", path=str(root), detail=str(exc))
    if disk_free is not None and disk_free <= config.min_free_bytes:
        add("DISK_RESERVE_UNAVAILABLE", "block", free_bytes=disk_free, reserve_bytes=config.min_free_bytes)
    for path in (*config.cti_files, *config.calibration_files):
        if not Path(path).is_file():
            add("DEPENDENCY_FILE_MISSING", "block", path=path)

    connections = {camera.id: camera_connections[camera.id] for camera in config.cameras
                   if camera_connections is not None and camera.id in camera_connections}
    failed_connections = {camera for camera, check in connections.items() if check.state != "available"}
    for camera, check in connections.items():
        if camera in failed_connections:
            add("CAMERA_CONNECTION_CHECK_FAILED", "block", camera=camera, state=check.state,
                reason=check.reason, detail=check.detail)

    camera_rows = plan_camera_geometry(config.cameras, preview_images)
    quality_regions = _check_quality_regions(config, preview_images or {}, camera_settings or {})
    for identity, check in quality_regions.items():
        if check["fallback"]:
            add("QUALITY_REGION_CHANGED", camera=identity, **check)
    settings_evidence = {}
    for camera in config.cameras:
        if not camera.settings and camera.id not in (camera_settings or {}):
            continue
        checked = (camera_settings or {}).get(camera.id)
        if checked is None and camera.id.startswith("sim:"):
            try:
                checked = check_simulator_settings(camera)
            except CaptureError as exc:
                checked = SettingsCheck("unavailable", error=str(exc))
        checked = checked or SettingsCheck("not_probed")
        summary = asdict(replace(checked, readback=None, device=None))
        summary.pop("readback")
        summary.pop("device")
        settings_evidence[camera.id] = summary
        for issue in checked.invalid:
            add("CAMERA_SETTING_INVALID", "block", camera=camera.id, **issue)
        if checked.error:
            if camera.id not in failed_connections:
                add("CAMERA_SETTINGS_PROBE_FAILED", camera=camera.id, detail=checked.error)
        elif checked.basis == "not_probed" or checked.deferred:
            fields = checked.deferred or tuple(camera.settings)
            add("CAMERA_SETTINGS_UNCONFIRMED", camera=camera.id, fields=fields, basis=checked.basis)
    support = {camera: trigger_evidence(camera, trigger_support or {}) for camera in camera_rows}
    if any(phase.camera_trigger != "free" for phase in config.phases):
        for camera, record in support.items():
            if record.error and camera not in failed_connections:
                add("TRIGGER_PROBE_FAILED", camera=camera, detail=record.error)
    assessments, matching = compare_trials(config, output_volume, trial_reports)
    if discovered_ids is not None:
        missing = [camera.id for camera in config.cameras if camera.id not in discovered_ids
                   and camera.id not in connections and camera.id not in (preview_images or {}) and not camera.id.startswith("sim:")]
        if missing:
            add("CAMERAS_NOT_DISCOVERED", cameras=missing)
    if any(camera.id.startswith("sim:") for camera in config.cameras):
        add("SIMULATED_CAMERAS", cameras=[camera.id for camera in config.cameras if camera.id.startswith("sim:")])
    if any(phase.role == "test" for phase in config.phases) and not any(phase.role == "reference" for phase in config.phases):
        add("NO_EXPLICIT_REFERENCE")

    phases, unknown_geometries, phase_suggestions, rate_suggestions = [], set(), [], []
    queue_capacity = config.effective_queue_capacity
    queue_mode = "auto" if config.queue_capacity == "auto" else "fixed"
    can_expand_queue = queue_mode == "fixed" and automatic_queue_capacity(config.max_buffer_bytes) > queue_capacity
    suggest_auto_queue = False
    verified_free_run = all(camera in (preview_images or {}) for camera in camera_rows)
    for index, phase in enumerate(config.phases, start=1):
        trial = matching.get(index)
        geometry = _phase_geometry(camera_rows, trial)
        setup_changed = False
        changed_cameras = ()
        if trial:
            assessment, evidence = trial
            if setup := compare_trial_setup(config, evidence, camera_settings or {}):
                assessment["camera_readbacks"] = setup
                if changed := cameras_with_trial_changes(setup):
                    setup_changed = True
                    changed_cameras = changed
                    geometry = camera_rows
                    add("TRIAL_CAMERA_READBACK_CHANGED", phase=index, cameras=changed)
            assessment, evidence = trial
            assessment["latest_match"] = True
            incomplete = (not assessment["observed"]
                          and (assessment["findings"] or evidence["verification"]["state"] != "stopped"))
            if assessment["status"] == "matched" and (incomplete or assessment["sustained_write_lag"]):
                add("TRIAL_NEEDS_REVIEW", phase=index)
            for camera, check in assessment.get("region_checks", {}).items():
                fallback = quality_regions.get(camera, {}).get("fallback")
                if check["used_for_review"] and (fallback or camera in changed_cameras):
                    check.update(fallback=fallback or "camera_changed", used_for_review=False)
            flagged_images = trial_image_alerts(assessment)
            if assessment["status"] == "matched" and flagged_images:
                add("TRIAL_IMAGES_NEED_REVIEW", phase=index, flagged_images=flagged_images)
            if not setup_changed and assessment["status"] == "matched" and (suggestion := suggest_trial_rate(config, index, evidence)):
                rate_suggestions.append(suggestion)
            trial_peak = evidence.get("raw_buffer_high_water_bytes")
            if (not setup_changed and can_expand_queue and assessment["status"] == "matched" and trial_peak is not None
                and trial_peak < .85 * config.max_buffer_bytes
                and any(item["code"] == "TRIAL_HIGH_WRITE_QUEUE" for item in evidence["findings"])):
                suggest_auto_queue = True
        unknown_geometries.update(key for key, row in geometry.items() if row["basis"] == "unavailable")
        estimate = estimate_phase(phase, geometry)
        stream_rate, scheduled_rate = estimate["configured_stream_raw_bytes_per_s"], estimate["scheduled_raw_bytes_per_s"]
        stall = None
        if stream_rate or (phase.mode == "timed" and scheduled_rate):
            batches_per_s = sum(row["configured_fps"] for row in geometry.values()) if stream_rate else phase.fps
            queue_s = queue_capacity / batches_per_s
            byte_s = config.max_buffer_bytes / (stream_rate or scheduled_rate)
            stall = {"seconds": min(queue_s, byte_s), "queue_seconds": queue_s, "byte_seconds": byte_s,
                     "first_limit": "queue" if queue_s < byte_s else "bytes"}
        limit, estimated_frames = estimate["frame_limit_per_camera"], estimate["frames_at_configured_rate"]
        batch_limit = limit
        if phase.mode == "continuous":
            batch_limit = limit * len(geometry) if limit is not None else estimated_frames
        if (can_expand_queue and stall and stall["first_limit"] == "queue"
            and (batch_limit is None or batch_limit > queue_capacity)):
            suggest_auto_queue = True
        buffer_round_bytes = estimate["one_batch_raw_bytes"]
        phases.append({"phase": index, "mode": phase.mode, "role": phase.role, **estimate,
                       "write_stall_buffer": stall, "camera_geometry": geometry,
                       "trial_geometry_cameras": [key for key, row in geometry.items() if row["basis"] == "trial"]})
        if phase.start_confirmation:
            phases[-1]["start_confirmation"] = phase.start_confirmation
        trigger = phase.camera_trigger
        if trigger == "software":
            by_driver = [camera for camera, record in support.items() if record.software_method == HOST_NEXT_RECEIPT]
            if by_driver:
                add("DRIVER_HOST_FRAME_SELECTION", "info", phase=index, cameras=by_driver)
        if trigger != "free":
            unsupported = [camera for camera, record in support.items() if getattr(record, trigger) is False]
            stereo_calibration = phase.role == "calibration" and len(camera_rows) > 1
            if unsupported and trigger == "software" and phase.software_trigger_fallback:
                add("HOST_FRAME_SELECTION", phase=index, cameras=unsupported)
            elif unsupported:
                independent_single = phase.mode == "single" and not stereo_calibration
                host_fallback = trigger == "software" and phase.mode != "single" and not stereo_calibration
                suggestion = None
                if verified_free_run:
                    if independent_single:
                        suggestion = "independent_receipts"
                    elif host_fallback:
                        suggestion = "host_receipt"
                add(f"{trigger.upper()}_TRIGGER_UNSUPPORTED", "block", phase=index, cameras=unsupported,
                    offer_preview=independent_single and not verified_free_run,
                    offer_host_preview=host_fallback and not verified_free_run,
                    stereo_calibration=stereo_calibration,
                    **({"suggested_frame_selection": suggestion} if suggestion else {}))
                if suggestion == "independent_receipts":
                    phase_suggestions.append({"phase": index,
                                              "replacement": replace(phase, mode="continuous", frame_count=1).to_dict(),
                                              "basis": "preview", "pairing": suggestion})
                elif suggestion == "host_receipt":
                    phase_suggestions.append({"phase": index,
                                              "replacement": replace(phase, software_trigger_fallback=True).to_dict(),
                                              "basis": "preview", "pairing": suggestion})
        if phase.mode == "hardware":
            inputs = {camera: phase.hardware_input(camera) for camera in camera_rows}
            phases[-1]["hardware_inputs"] = inputs
            for camera, line in inputs.items():
                record = support[camera]
                if record.hardware is False:
                    continue
                if camera.partition(":")[0] in ("pylon", "gentl") and not is_physical_trigger_input(line):
                    add("HARDWARE_INPUT_INVALID", "block", phase=index, camera=camera, input=line)
                elif record.physical_inputs is not None and line not in record.physical_inputs:
                    add("HARDWARE_INPUT_UNAVAILABLE", "block", phase=index, camera=camera, input=line,
                        available=list(record.physical_inputs))
            if len(config.cameras) > 1 and phase.hardware_pairing is None:
                add("HARDWARE_PAIRING_UNVERIFIED", phase=index)
        if phase.mode in ("hardware", "manual") and phase.duration_s is None:
            add("WAITING_FOR_EXTERNAL_INPUT", phase=index)
        if (planned_rounds := event_window_shortfall(phase)) is not None:
            add("EVENT_WINDOW_EXCEEDS_DURATION", "block", phase=index, planned_rounds=planned_rounds,
                pre_frames=phase.pre_frames, post_frames=phase.post_frames,
                **({"event_count": phase.event_count} if phase.repeats_events else {}))
        if limit is None and estimated_frames is None:
            add("FRAME_LIMIT_UNAVAILABLE", phase=index)
        if buffer_round_bytes is not None and buffer_round_bytes > config.max_buffer_bytes:
            observed_geometry = all(row["basis"] != "configured" for row in geometry.values())
            add("RAW_BUFFER_TOO_SMALL", "block" if observed_geometry else "warning",
                phase=index, estimated_bytes=buffer_round_bytes, buffer_bytes=config.max_buffer_bytes)
        if phase.fps:
            for camera in config.cameras:
                exposure = camera.settings.get("exposure_us")
                if (type(exposure) in (float, int) and exposure > 1e6 / phase.fps
                    and camera.settings.get("exposure_auto", "Off") == "Off"):
                    add("EXPOSURE_EXCEEDS_INTERVAL", phase=index, camera=camera.id, exposure_us=exposure,
                        interval_us=1e6 / phase.fps)

    if unknown_geometries:
        add("FRAME_GEOMETRY_UNAVAILABLE", cameras=sorted(unknown_geometries))
    bounded = all(row["raw_payload_bytes"] is not None for row in phases)
    payload = sum(row["raw_payload_bytes"] for row in phases) if bounded else None
    if payload is not None and disk_free is not None and payload + config.min_free_bytes > disk_free:
        add("PLAN_EXCEEDS_AVAILABLE_SPACE", estimated_bytes=payload,
            available_bytes=max(0, disk_free - config.min_free_bytes))
    continuation = compare_continuation_setup(config, camera_settings or {})
    if continuation is not None:
        for camera in config.cameras:
            if additions := continuation["suggested_settings"].get(camera.id):
                features = camera_settings[camera.id].readback
                candidate = {**camera.settings, **additions}
                backend = camera.id.partition(":")[0]
                if check_camera_settings(backend, candidate, features).invalid:
                    continuation["suggested_settings"].pop(camera.id)
        if (continuation["missing_cameras"] or continuation["added_cameras"] or continuation["phase_changes"]
                or any(camera["identity"] == "changed" for camera in continuation["readbacks"].values())
                or any(row["status"] in ("plan_changed", "readback_changed")
                       for camera in continuation["readbacks"].values() for row in camera["fields"])):
            add("CONTINUATION_SETUP_CHANGED")
    return {"schema": "straintrace.capture.preflight/v1", "ready": not any(i["severity"] == "block" for i in issues),
            "output": str(root), "volume_probe_path": str(ancestor), "disk_free_bytes": disk_free,
            **({"output_suggestion": str(destination.suggestion)} if destination.suggestion is not None else {}),
            "reserve_bytes": config.min_free_bytes, "buffer_limit_bytes": config.max_buffer_bytes,
            "write_queue": {"mode": queue_mode, "capacity": queue_capacity, "suggest_auto": suggest_auto_queue},
            "cameras": camera_rows, "phases": phases, "raw_payload_bytes": payload,
            "trial_assessments": assessments, "camera_settings": settings_evidence,
            **({"quality_regions": quality_regions} if quality_regions else {}),
            "trial_setting_suggestions": suggest_trial_settings(config, matching, camera_settings or {}),
            "rate_suggestions": rate_suggestions,
            "trigger_support": {key: asdict(record) for key, record in support.items()},
            **({"camera_connections": {key: asdict(check) for key, check in connections.items()}}
               if camera_connections is not None else {}),
            "phase_suggestions": phase_suggestions, "issues": issues,
            **({"continuation": continuation} if continuation is not None else {})}
