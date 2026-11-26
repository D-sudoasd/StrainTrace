"""Capture automation, independent of the desktop."""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
import time
from contextlib import closing, nullcontext
from dataclasses import asdict

from . import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan, recover_experiment
from .cli_arguments import build_parser
from .control import CaptureControl, send_command
from .continuation import continuation_plan, inspect_continuation
from .repeat import prepare_repeat, validate_repeat_destination
from .frame_search import parse_receipt_utc, parse_sequence_range
from .handoff import analysis_settings, analysis_snapshot, derive_analysis_inputs, write_snapshot
from .hardware_readiness import wait_for_hardware
from .readiness import probe_camera_setup, probe_preflight_cameras
from .related import inspect_related_experiments
from .sequence_export import SequenceSelection, export_sequence, inspect_sequence, verify_sequence
from .storage import TERMINAL_STATES
from .trial import prepare_trial, trial_report
from .transfer import copy_experiment, inspect_transfer, verify_transfer


def _collect(session: CaptureSession, args, *, announce_output=False) -> dict:
    identity = session.start()
    needs_confirmation = any(phase.start_confirmation for phase in session.config.phases)
    try:
        control = CaptureControl(session, allow_trigger=args.control) if args.control or needs_confirmation else nullcontext()
        with control:
            if announce_output or needs_confirmation:
                print(json.dumps({"event": "capture_opened", "experiment_id": identity, "output": str(session.output)}),
                      file=sys.stderr, flush=True)
            announced = None
            while True:
                try:
                    return session.wait(0.5)
                except TimeoutError:
                    if args.progress:
                        print(json.dumps(session.status(), ensure_ascii=True, allow_nan=False), file=sys.stderr, flush=True)
                    elif needs_confirmation:
                        gate = session.status().get("phase_confirmation")
                        if gate and gate["state"] == "waiting" and gate["confirmation_id"] != announced:
                            announced = gate["confirmation_id"]
                            print(json.dumps({"event": "phase_confirmation_required", "output": str(session.output), **gate}),
                                  file=sys.stderr, flush=True)
                except KeyboardInterrupt:
                    session.request_stop()
    except BaseException:
        session.request_stop()
        session.wait()
        raise


def _run(args) -> tuple[dict, int]:
    config = CaptureConfig.load(args.config)
    if args.continue_from:
        config = continuation_plan(args.continue_from, config)
    need_settings = any(camera.settings for camera in config.cameras)
    if need_settings or any(phase.csv_trigger for phase in config.phases):
        settings = probe_camera_setup(config, triggers=False)[1] if need_settings else None
        preflight = inspect_plan(config, args.output, camera_settings=settings)
        if any(issue["code"] == "CAMERA_SETTING_INVALID" or issue["code"].startswith("CSV_TRIGGER_")
               for issue in preflight["issues"]):
            return preflight, 3
    session = CaptureSession(config, args.output, watch_quality=args.watch_quality)
    status = _collect(session, args)
    has_losses = any(
        any(camera[key] for key in ("app_unsaved", "native_counter_gaps", "native_counter_duplicates",
                                    "hardware_identity_missing", "reported_driver_skips", "transport_errors"))
        for camera in status["cameras"].values()
    )
    result = {"output": str(session.output), **status}
    incomplete = (has_losses or status["scheduled_misses"] or status["short_pre_windows"]
                  or any(item["state"] == "fault" for item in status.get("csv_triggers", ()))
                  or any(item["partial_events"] or item["busy_signals"] or item["early_signals"]
                         for item in status.get("event_phases", ())))
    if status["state"] == "completed" and any(phase.hardware_pairing for phase in config.phases):
        with ExperimentReader(session.output) as reader:
            pairing = reader.pairing()
        result["hardware_counter_gap_ranges"] = pairing["hardware_counter_gap_ranges"]
        result["incomplete_trigger_groups"] = sum(group["status"] != "associated" for group in pairing["groups"])
        incomplete = incomplete or result["incomplete_trigger_groups"] or result["hardware_counter_gap_ranges"]
    return result, 3 if status["state"] == "failed" else (4 if incomplete else 0)


def _trial_result(root) -> tuple[dict, int]:
    result = trial_report(root)
    if result["verification"]["state"] == "failed":
        return result, 3
    return result, 4 if not result["observed"] or result["sustained_write_lag"] else 0


def _execute(args):
    action = args.command
    if action == "export-quality-report":
        from .archive_quality import scan_archive_quality_batch
        from .archive_quality_export import export_archive_quality, quality_report_destination

        output = quality_report_destination(args.experiment, args.output)
        identity = args.expect_experiment_id
        if identity is None:
            with ExperimentReader(args.experiment) as reader:
                identity = reader.experiment["experiment_id"]
        try:
            with closing(scan_archive_quality_batch(
                args.experiment, identity, all_frames=args.all_frames, scratch=output.parent,
            )) as batch:
                receipt = export_archive_quality(batch, output)
        except KeyboardInterrupt:
            return {"cancelled": True}, 130
        incomplete = receipt["failed_groups"] or receipt["verified"] < receipt["checked"]
        return receipt, 4 if incomplete else 0
    if action == "export-live-results":
        from .live_results import export_live_csv

        interval = parse_sequence_range(args.frame_range) if args.frame_range else (None, None)
        return export_live_csv(args.record, args.output, fill_load=args.fill_load or args.load_table is not None,
                               load_table=args.load_table, sequence_range=interval if interval[0] is not None else None), 0
    if action == "replay-live-results":
        from .live_history import read_history_index
        from .live_replay import replay_live_results

        try:
            index = read_history_index(args.experiment, args.record)
            return replay_live_results(index, images_only=args.images_only, load_table=args.load_table), 0
        except KeyboardInterrupt:
            raise CaptureError("LIVE_REPLAY_CANCELLED", "已取消重算。") from None
    if action == "discover":
        return CameraRegistry(cti_files=args.cti).discover(args.backends), 0
    if action == "features":
        with CameraRegistry(cti_files=args.cti).camera(args.camera) as camera:
            return {"device": asdict(camera.device), "features": {name: asdict(feature)
                    for name, feature in camera.features().items()}}, 0
    if action == "validate-config":
        return CaptureConfig.load(args.config).to_dict(), 0
    if action == "preflight":
        config = CaptureConfig.load(args.config)
        if args.continue_from:
            config = continuation_plan(args.continue_from, config)
        registry = CameraRegistry(cti_files=config.cti_files)
        identities = None
        if args.discover:
            names = [camera.id.partition(":")[0] for camera in config.cameras]
            discovery = registry.discover(names)
            identities = {device["id"] for device in discovery["devices"]}
        trials = [trial_report(path, inspect_images=bool(config.quality_regions), image_quality_regions=config.quality_regions)
                  for path in args.trial]
        probe = probe_preflight_cameras(config, registry, include_readback=bool(trials)) if args.probe_cameras else None
        report = inspect_plan(config, args.output, discovered_ids=identities, trial_reports=trials,
                              trigger_support=probe.triggers if probe else None,
                              camera_settings=probe.settings if probe else None,
                              camera_connections=probe.connections if probe else None)
        if args.discover:
            report["providers"] = discovery["providers"]
        return report, (0 if report["ready"] else 3)
    if action == "write-probe":
        from .write_probe import DEFAULT_PROBE_MIB, measure_write_path, preferred_probe_phase, write_probe_phases

        try:
            config = CaptureConfig.load(args.config)
            trials = [trial_report(path, inspect_images=False) for path in args.trial]
            report = inspect_plan(config, args.output, trial_reports=trials)
            phases = write_probe_phases(report)
            if args.phase is not None and not 1 <= args.phase <= len(phases):
                raise CaptureError("TRIAL_PHASE_INVALID", f"请选择阶段 1–{len(phases)}。")
            phase = phases[args.phase - 1] if args.phase is not None else preferred_probe_phase(phases)
            if phase is None:
                raise CaptureError("WRITE_PROBE_GEOMETRY", "没有可核对尺寸和像素格式的阶段。")
            result = measure_write_path(config, args.output, phase, expected_parent=report["volume_probe_path"],
                                        max_mib=DEFAULT_PROBE_MIB if args.max_mib is None else args.max_mib)
            return result, 0
        except KeyboardInterrupt:
            raise CaptureError("WRITE_PROBE_CANCELLED", "已取消写盘测试。") from None
    if action == "run":
        return _run(args)
    if action == "trial":
        config = CaptureConfig.load(args.config)
        prepared, output = prepare_trial(config, args.output, phase_number=args.phase, seconds=args.seconds)
        probe = probe_preflight_cameras(prepared, CameraRegistry(cti_files=prepared.cti_files))
        readiness = inspect_plan(prepared, output, trigger_support=probe.triggers, camera_settings=probe.settings,
                                 camera_connections=probe.connections)
        if not readiness["ready"]:
            return readiness, 3
        _collect(CaptureSession(prepared, output), args, announce_output=args.control or args.progress)
        return _trial_result(output)
    if action == "trial-report":
        return _trial_result(args.experiment)
    if action == "copy-experiment":
        if args.check:
            return inspect_transfer(args.experiment, args.output, attachments=args.attach), 0
        last_progress = float("-inf")

        def show(update):
            nonlocal last_progress
            now = time.monotonic()
            if now - last_progress >= .25 or update.stage == "publishing":
                print(json.dumps(asdict(update), ensure_ascii=True), file=sys.stderr, flush=True)
                last_progress = now

        try:
            return copy_experiment(args.experiment, args.output, attachments=args.attach,
                                   progress=show if args.progress else None), 0
        except KeyboardInterrupt:
            return {"error": {"code": "TRANSFER_CANCELLED", "message": "copy interrupted; retry with the same output to reuse verified files"}}, 130
    if action == "verify-copy":
        try:
            return verify_transfer(args.experiment), 0
        except CaptureError as exc:
            if exc.code in {"TRANSFER_MANIFEST", "TRANSFER_MISSING", "TRANSFER_CHANGED", "TRANSFER_PATH", "TRANSFER_SCOPE"}:
                return {"ok": False, "error": exc.as_dict()}, 4
            raise
    if action == "export-sequence":
        first, last = parse_sequence_range(args.sequence)
        selection = SequenceSelection(camera_id=args.camera, phase=args.phase, reference_sequence=args.reference_sequence,
                                      first_sequence=first, last_sequence=last, image_part=args.image_part,
                                      event_context_id=args.event_context)
        if args.check:
            return inspect_sequence(args.experiment, args.output, selection), 0
        try:
            return export_sequence(args.experiment, args.output, selection), 0
        except KeyboardInterrupt:
            return {"error": {"code": "SEQUENCE_CANCELLED", "message": "image sequence export interrupted"}}, 130
    if action == "verify-sequence":
        try:
            return verify_sequence(args.experiment), 0
        except CaptureError as exc:
            return {"ok": False, "error": exc.as_dict()}, 4
    if action == "recover":
        if args.check:
            from .recovery import inspect_recovery

            return inspect_recovery(args.experiment), 0
        return recover_experiment(args.experiment), 0
    if action in {"trigger", "confirm-phase", "pause-phase", "resume-phase", "stop", "mark"}:
        payload = {"label": args.label} if action == "mark" else getattr(args, "payload", None)
        result = send_command(args.experiment, action, source=getattr(args, "source", "operator"),
                              payload=payload, expected_phase=getattr(args, "phase", None),
                              expected_window=getattr(args, "event_window", None),
                              require_prebuffer=getattr(args, "require_prebuffer", False),
                              confirmation_id=getattr(args, "confirmation_id", None), pause_id=getattr(args, "pause_id", None))
        return result, 0 if result["ok"] else 3
    if action == "status" and args.live:
        result = send_command(args.experiment, "status")
        return result, 0 if result["ok"] else 3
    if action == "wait-hardware":
        result = wait_for_hardware(args.experiment, args.phase, args.timeout)
        return result, 0 if result["ok"] else 3
    if action == "prepare-analysis":
        if args.reference_experiment is None:
            snapshot = analysis_snapshot(args.experiment, args.camera, args.reference_sequence,
                                         image_part=args.image_part, event_context_id=args.event_context)
        else:
            from .continuation_analysis import continuation_snapshot

            snapshot = continuation_snapshot(args.experiment, args.reference_experiment, args.camera, args.reference_sequence,
                                             image_part=args.image_part, event_context_id=args.event_context)
        if bool(args.template) != bool(args.run_config):
            raise CaptureError("ANALYSIS_TEMPLATE_USAGE", "provide --template and --run-config together")
        if args.derive_color and args.derived_folder:
            raise CaptureError("DERIVED_FOLDER_CONFLICT", "supply one derived output folder")
        folder = args.derive_color or args.derived_folder
        if (args.lens_profile and not folder) or (args.derived_folder and not args.lens_profile):
            raise CaptureError("DERIVED_FOLDER_REQUIRED", "lens correction requires a lens profile and a new derived folder")
        if folder:
            snapshot = derive_analysis_inputs(snapshot, folder, debayer=args.derive_color is not None,
                                              drop_alpha=args.derive_color is not None,
                                              lens_profile=args.lens_profile)
        if args.template:
            from ezdic_cli import normalize_config

            template = json.loads(args.template.read_text(encoding="utf-8-sig"))
            settings = normalize_config(analysis_settings(snapshot, template), base_dir=args.template.resolve().parent)
            write_snapshot(settings, args.run_config)
        return {"snapshot": str(write_snapshot(snapshot, args.output)), "frames": len(snapshot["frames"])}, 0
    if action == "calibrate":
        from .calibration import calibrate

        if args.image_part and len(args.image_part) != len(args.camera):
            raise CaptureError("CALIBRATION_CAMERAS_INVALID", "provide one --image-part or 'primary' for each --camera")
        parts = {camera: part for camera, part in zip(args.camera, args.image_part) if part != "primary"}
        result = calibrate(args.experiment, args.camera, inner_columns=args.columns, inner_rows=args.rows,
                           square_mm=args.square_mm, min_views=args.min_views, image_parts=parts)
        return {"calibration": str(write_snapshot(result, args.output)), "cameras": list(result["cameras"])}, 0
    if action in ("export-timing", "align-load"):
        return _sync_command(args), 0
    if action == "link-detector":
        return _detector_command(args)
    if action == "audit-pulses":
        from .pulse_audit import audit_external_pulses, write_pulse_audit
        from .pulse_binding import PulseBinding

        binding = PulseBinding.load(args.anchor_file) if args.anchor_file else None
        report = audit_external_pulses(args.experiment, args.source, phase=args.phase, event_column=args.event_column,
                                       first_event=args.first_event, last_event=args.last_event, binding=binding)
        limits = args.max_skew_ns, args.latency_min_ns, args.latency_max_ns
        if args.timing_source:
            from .pulse_timing import ExposureLimits, add_exposure_audit

            report = add_exposure_audit(report, args.timing_source, limits=ExposureLimits(*limits))
        elif any(value is not None for value in limits):
            raise CaptureError("PULSE_TIMING_LIMITS", "时序限值需要配合 --timing-source 使用。")
        result = {"path": str(write_pulse_audit(report, args.output)), "complete": report["complete"], **report["summary"]}
        if timing := report.get("exposure_timing"):
            result.update(identity_complete=report["identity_complete"], exposure_timing=timing["summary"])
        return result, 0 if result["complete"] else 4
    if action == "related":
        return inspect_related_experiments(args.experiment, search_dirs=args.search), 0
    if action == "prepare-continuation":
        if args.inspect:
            return inspect_continuation(args.experiment), 0
        return continuation_plan(args.experiment, start_phase=args.start_phase).to_dict(), 0
    if action == "prepare-repeat":
        draft = prepare_repeat(args.experiment, name=args.name, expected_experiment_id=args.expect_experiment_id)
        if args.output is not None:
            validate_repeat_destination(draft.source, args.output)
        return draft.config.to_dict(), 0
    with ExperimentReader(args.experiment) as reader:
        if action == "status":
            return {"experiment": reader.experiment, "cameras": reader.cameras()}, 0
        if action == "frames":
            if args.after < 0 or args.limit < 1:
                raise CaptureError("INVALID_FRAME_QUERY", "--after must be nonnegative and --limit positive")
            return reader.frames(camera_id=args.camera, after=args.after, limit=args.limit), 0
        if action == "find-frames":
            start, stop = parse_sequence_range(args.sequence)
            return reader.frame_page(
                camera_id=args.camera, phase=args.phase, role=args.role, first_sequence=start, last_sequence=stop,
                received_from_ns=parse_receipt_utc(args.received_from) if args.received_from else None,
                received_to_ns=parse_receipt_utc(args.received_to) if args.received_to else None,
                before=args.before_ordinal, after=args.after_ordinal, snapshot=args.snapshot_ordinal,
                expected_experiment_id=args.expect_experiment_id, event_context_id=args.event_context, limit=args.limit,
            ), 0
        if action == "timeline":
            return reader.timeline(camera_id=args.camera, phase=args.phase, before=args.before_ordinal,
                                   after=args.after_ordinal, snapshot=args.snapshot_ordinal,
                                   expected_experiment_id=args.expect_experiment_id, limit=args.limit,
                                   markers_only=args.markers_only, event_windows_only=args.event_windows_only,
                                   incidents_only=args.incidents_only), 0
        if action == "marker-context":
            return reader.marker_context(args.marker_id, expected_experiment_id=args.expect_experiment_id), 0
        if action == "events":
            return reader.events(), 0
        if action == "trial-comparison":
            return reader.trial_comparison(), 0
        if action == "pairing":
            return reader.pairing(), 0
        if action == "verify":
            result = reader.verify(hash_files=not args.size_only)
            return result, 0 if result["complete"] else 4
    raise AssertionError(action)


def _detector_command(args):
    if args.pulse_report is None:
        from .synchronization import link_detector

        if args.daq_source is not None:
            raise CaptureError("DETECTOR_AUDIT_INVALID", "--daq-source 需要配合 --pulse-report 使用。")
        result = link_detector(args.experiment, args.source, trigger_source=args.trigger_source,
                               event_column=args.event_column, frame_column=args.frame_column)
        return {"path": str(write_snapshot(result, args.output)), "events": len(result["links"])}, 0

    from .detector_link import link_audited_detector, write_detector_links

    try:
        audit = json.loads(args.pulse_report.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError("DETECTOR_AUDIT_INVALID", "请选择可读取的脉冲核对 JSON 报告。") from exc
    result = link_audited_detector(args.experiment, args.source, audit, event_column=args.event_column,
                                   frame_column=args.frame_column, daq_source=args.daq_source)
    saved = {"path": str(write_detector_links(result, args.output)), "events": len(result["links"]),
             "complete": result["complete"], **result["summary"]}
    return saved, 0 if saved["complete"] else 4


def _sync_command(args):
    from .handoff import select_camera
    from .report_files import new_text_output
    from .synchronization import ClockAnchor, LoadAlignment, timing_rows

    anchors = ()
    if args.clock_anchors:
        anchors = tuple(ClockAnchor.from_dict(row) for row in json.loads(args.clock_anchors.read_text(encoding="utf-8-sig")))
    if args.command == "align-load":
        channel = LoadAlignment(
            args.load, time_column=args.time_column, force_column=args.force_column, force_unit=args.force_unit,
            max_gap_s=args.max_gap_s, anchors=anchors, basis=args.time_basis, mode=args.mode,
            displacement_column=args.displacement_column, displacement_unit=args.displacement_unit,
        )
        channel.refresh()
        with ExperimentReader(args.experiment) as reader:
            camera = select_camera(reader, args.camera)
            rows = [channel.at(row, refresh=False) for row in reader.frames(camera_id=camera)]
        result = {"schema": "straintrace.capture-load-alignment/v1", "source": channel.provenance, "rows": rows}
        return {"path": str(write_snapshot(result, args.output)), "frames": len(rows),
                "finite_force": sum(row["force_N"] is not None for row in rows)}
    rows = iter(timing_rows(args.experiment, args.camera, basis=args.time_basis, anchors=anchors))
    first = next(rows, None)
    if first is None:
        raise CaptureError("TIMING_INPUT_EMPTY", "no committed frames for timing export")
    from itertools import chain

    with new_text_output(args.output, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=first.keys())
        writer.writeheader()
        count = 0
        for row in chain((first,), rows):
            if row["conversion_evidence"]:
                row["conversion_evidence"] = json.dumps(row["conversion_evidence"], ensure_ascii=False)
            writer.writerow(row)
            count += 1
    return {"path": str(args.output.resolve()), "frames": count}


def _follow(args) -> int:
    from .config import positive_number
    from .live import LiveAnalysis
    from .synchronization import load_alignment_settings

    interval = positive_number(args.poll_s, "poll interval")
    data = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.image_part is not None:
        data["image_part"] = args.image_part
    if args.reference_experiment is not None:
        data["reference_experiment"] = args.reference_experiment
    if args.reference_binding is not None:
        data["reference_binding"] = args.reference_binding
    if load := data.pop("mechanics", None):
        data["load_alignment"] = load_alignment_settings(load, base_dir=args.config.resolve().parent)
    with LiveAnalysis(args.experiment, args.camera, reference_sequence=args.reference_sequence,
                      latest=not args.all_frames, **data) as analysis:
        try:
            while True:
                row = analysis.poll()
                if row:
                    print(json.dumps(row, ensure_ascii=True, allow_nan=False), flush=True)
                elif analysis.experiment_state in TERMINAL_STATES:
                    if analysis.reference is None:
                        raise CaptureError("REFERENCE_UNAVAILABLE", "capture ended without a usable reference")
                    break
                else:
                    time.sleep(interval)
        except KeyboardInterrupt:
            pass
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "follow":
            return _follow(args)
        result, code = _execute(args)
    except (RuntimeError, OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
        result = {"error": exc.as_dict() if isinstance(exc, CaptureError) else {
            "code": "CAPTURE_ERROR", "message": str(exc),
        }}
        code = 2 if isinstance(exc, CaptureError) and exc.code == "INVALID_CONFIG" else 3
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2))
    return code
