"""Capture automation, independent of the desktop."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from contextlib import closing, nullcontext
from dataclasses import asdict
from pathlib import Path

from . import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan, recover_experiment
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
from .trial import DEFAULT_TRIAL_SECONDS, prepare_trial, trial_report
from .transfer import copy_experiment, inspect_transfer, verify_transfer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="straintrace-capture", description="Acquire and inspect StrainTrace experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="discover cameras and report SDK availability")
    discover.add_argument("--backend", action="append", dest="backends", help="restrict discovery to this backend")
    discover.add_argument("--cti", action="append", default=[], help="trusted vendor GenTL producer (.cti)")
    feature = commands.add_parser("features", help="connect and list current camera features")
    feature.add_argument("camera")
    feature.add_argument("--cti", action="append", default=[])
    config = commands.add_parser("validate-config", help="validate a Capture JSON configuration")
    config.add_argument("config", type=Path)
    preflight = commands.add_parser("preflight", help="inspect capture readiness and estimate raw storage before recording")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--discover", action="store_true", help="also check the discoverable camera IDs")
    preflight.add_argument("--probe-cameras", "--probe-triggers", action="store_true",
                           help="check selected camera connections and read settings and trigger support without acquiring")
    preflight.add_argument("--continue-from", type=Path, help="compare a draft with the readbacks of a terminal experiment")
    preflight.add_argument("--trial", type=Path, action="append", default=[],
                           help="check a finished trial directory against the current plan; repeat for multiple phases")
    probe = commands.add_parser("write-probe", help="test the output filesystem with synthetic TIFFs and the Capture writer")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True, help="planned experiment path; the probe uses temporary files beside it")
    probe.add_argument("--phase", type=int, help="one-based phase; defaults to the most demanding measurable phase")
    probe.add_argument("--max-mib", type=int, help="maximum raw sample size, default 128 MiB")
    probe.add_argument("--trial", type=Path, action="append", default=[],
                       help="finished trial used to resolve image geometry; repeat for multiple phases")
    run = commands.add_parser("run", help="acquire to a new experiment directory")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--control", action="store_true", help="accept operator commands through the experiment's local folder")
    run.add_argument("--progress", action="store_true", help="write live JSON progress to stderr")
    run.add_argument("--watch-quality", action="store_true", help="sample admitted primary images for operator advisories")
    run.add_argument("--continue-from", type=Path, help="start a new linked experiment after a prior terminal run")
    trial = commands.add_parser("trial", help="rehearse one phase with the configured cameras and write policy")
    trial.add_argument("--config", type=Path, required=True)
    trial.add_argument("--output", type=Path, required=True, help="intended formal output; the trial uses a new sibling directory")
    trial.add_argument("--phase", type=int, help="one-based phase number; defaults to the first test phase")
    trial.add_argument("--seconds", type=float, default=DEFAULT_TRIAL_SECONDS, help="trial phase time limit, default 8 seconds")
    trial.add_argument("--control", action="store_true", help="accept local operator commands in the trial directory")
    trial.add_argument("--progress", action="store_true", help="write live JSON progress to stderr")
    quality = commands.add_parser("export-quality-report", help="check archived image quality and write a new portable ZIP report")
    quality.add_argument("experiment", type=Path, help="completed Capture experiment")
    quality.add_argument("--output", type=Path, required=True, help="new .zip file outside the experiment directory")
    quality.add_argument("--all-frames", action="store_true", help="check every saved primary image instead of sampling each group")
    quality.add_argument("--expect-experiment-id", help="reject a directory with a different experiment ID")
    for name, help_text in (
        ("status", "read experiment state and camera records"), ("frames", "list committed frame metadata"),
        ("find-frames", "search compact saved-frame records with stable pagination"),
        ("timeline", "summarize host receipt times and page through selected acquisition incidents"),
        ("events", "list acquisition events"), ("pairing", "report trigger-based associations and missing cameras"),
        ("verify", "verify frame files, hashes and ledger state"), ("recover", "restore verified frames and finalize an interrupted ledger"),
        ("related", "list original, repeated, continued and sibling Capture experiments by recorded IDs"),
        ("prepare-continuation", "inspect source stages or output a new linked acquisition plan"),
        ("prepare-repeat", "prepare the full plan for an independent repeat acquisition"),
        ("copy-experiment", "make a verified copy of a finished experiment in a new directory"),
        ("verify-copy", "verify a received experiment against its transfer manifest"),
        ("export-sequence", "export original TIFFs in a reference-first image sequence"),
        ("verify-sequence", "verify a standalone TIFF sequence and its frame index"),
        ("trial-report", "regenerate the report for a finished trial from its ledger and TIFFs"),
        ("trial-comparison", "read the formal run's archived trial evidence and actual phase settings"),
        ("trigger", "request a manual or external event from an armed session"),
        ("mark", "record an operator observation during an active acquisition"),
        ("marker-context", "find each camera's closest saved receipts before and after a marker"),
        ("confirm-phase", "confirm a waiting phase so that it can begin arming"),
        ("pause-phase", "pause a timed or manual phase after its current camera round"),
        ("resume-phase", "continue the same phase from a confirmed pause"),
        ("wait-hardware", "wait for fresh camera-reported FrameTriggerWait outputs before the first received frame"),
        ("stop", "request a graceful stop and drain of the raw queue"),
        ("prepare-analysis", "write the committed frame binding for existing Analysis"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("experiment", type=Path)
        if name == "status":
            command.add_argument("--live", action="store_true", help="query the running Capture process with --control enabled")
        if name == "prepare-continuation":
            mode = command.add_mutually_exclusive_group()
            mode.add_argument("--inspect", action="store_true", help="show saved frames, phase evidence and a suggested starting phase")
            mode.add_argument("--start-phase", type=int, default=1, help="first source phase to repeat, one-based")
        if name == "prepare-repeat":
            command.add_argument("--name", help="experiment name for the repeat")
            command.add_argument("--output", type=Path, help="check a proposed new destination")
            command.add_argument("--expect-experiment-id", help="bind the source to the recorded experiment identity")
        if name == "frames":
            command.add_argument("--camera")
            command.add_argument("--after", type=int, default=0)
            command.add_argument("--limit", type=int, default=100)
        if name == "find-frames":
            command.add_argument("--role", choices=("reference", "test", "calibration"))
            command.add_argument("--sequence", default="", help="inclusive range such as 40-80, or one exact sequence")
            command.add_argument("--received-from", help="host receipt time with explicit ISO timezone, inclusive")
            command.add_argument("--received-to", help="host receipt time with explicit ISO timezone, inclusive")
            command.add_argument("--event-context", help="exact signal ID for one event window")
        if name in ("find-frames", "timeline"):
            command.add_argument("--camera", help="camera ID")
            command.add_argument("--phase", type=int)
            cursor = command.add_mutually_exclusive_group()
            cursor.add_argument("--before-ordinal", type=int, help="older_before from a prior result")
            cursor.add_argument("--after-ordinal", type=int, help="newer_after from a prior result")
            command.add_argument("--snapshot-ordinal", type=int, help="reuse the high-water mark from the first page")
            command.add_argument("--expect-experiment-id", help="reject a moved or replaced directory with another experiment ID")
            command.add_argument("--limit", type=int, default=48, help="1 to 256 rows per page")
        if name == "timeline":
            filter_group = command.add_mutually_exclusive_group()
            filter_group.add_argument("--markers-only", action="store_true", help="page through operator markers")
            filter_group.add_argument("--event-windows-only", action="store_true", help="page through recorded event windows")
            filter_group.add_argument("--incidents-only", action="store_true", help="page through acquisition incidents")
        if name == "verify":
            command.add_argument("--size-only", action="store_true", help="skip SHA-256 checks of the frame files")
        if name == "copy-experiment":
            command.add_argument("--output", type=Path, required=True, help="new experiment directory under an existing parent")
            command.add_argument("--attach", type=Path, action="append", default=[], help="include a file such as a DAQ CSV; repeat as needed")
            command.add_argument("--check", action="store_true", help="list scope and estimate size without creating a copy")
            command.add_argument("--progress", action="store_true", help="write periodic JSON progress to stderr")
        if name == "export-sequence":
            command.add_argument("--output", type=Path, required=True, help="new directory under an existing parent")
            command.add_argument("--camera", help="camera ID; optional for a one-camera experiment")
            command.add_argument("--phase", type=int, help="include test frames from this one-based phase only")
            command.add_argument("--event-context", help="include experiment images assigned to this exact event")
            command.add_argument("--reference-sequence", type=int, help="use this committed frame as the reference")
            command.add_argument("--sequence", default="", help="inclusive test-frame range such as 40-80")
            command.add_argument("--image-part", help="exact saved GenTL part, such as 2/16/0/Mono8")
            command.add_argument("--check", action="store_true", help="list scope without writing the sequence")
        if name == "recover":
            command.add_argument("--check", action="store_true", help="inspect pending frame evidence without changing the experiment")
        if name == "related":
            command.add_argument("--search", type=Path, action="append", default=[],
                                 help="also look at this experiment or its immediate child directories; repeatable")
        if name == "trigger":
            command.add_argument("--source", default="operator")
            command.add_argument("--payload", type=json.loads, help="external event metadata as JSON")
            command.add_argument("--phase", type=int, help="accept only in this one-based phase number")
            command.add_argument("--event-window", type=int, help="accept only in this one-based event window")
            command.add_argument("--require-prebuffer", action="store_true", help="accept only when the event prebuffer is full")
        if name == "mark":
            command.add_argument("--label", default="", help="optional description of what was observed")
            command.add_argument("--source", default="operator")
        if name == "marker-context":
            command.add_argument("marker_id", help="marker ID returned by mark or timeline --markers-only")
            command.add_argument("--expect-experiment-id", help="bind the lookup to the recorded experiment identity")
        if name == "confirm-phase":
            command.add_argument("--phase", type=int, required=True, help="phase shown in the pending confirmation")
            command.add_argument("--confirmation-id", help="bind the action to the exact confirmation shown by status --live")
            command.add_argument("--source", default="operator", help="confirmation source recorded in the ledger")
        if name in {"pause-phase", "resume-phase"}:
            command.add_argument("--phase", type=int, required=True, help="the one-based phase shown by status --live")
            command.add_argument("--source", default="operator", help="source recorded in the ledger")
            if name == "resume-phase":
                command.add_argument("--pause-id", required=True, help="the pause_id shown by status --live")
        if name == "wait-hardware":
            command.add_argument("--phase", type=int, required=True, help="the planned hardware phase, numbered from one")
            command.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for all cameras")
        if name == "prepare-analysis":
            command.add_argument("--camera")
            command.add_argument("--reference-sequence", type=int)
            command.add_argument("--event-context", help="include experiment images assigned to this exact event")
            command.add_argument("--reference-experiment", type=Path,
                                 help="use the explicitly selected reference sequence from a linked earlier experiment")
            command.add_argument("--image-part", help="exact saved GenTL part, such as 2/16/0/Mono8")
            command.add_argument("--output", type=Path, required=True, help="new snapshot JSON file")
            command.add_argument("--template", type=Path, help="existing ezDIC run configuration")
            command.add_argument("--run-config", type=Path, help="new ezDIC config bound to these frames")
            command.add_argument("--derive-color", "--derive-bayer", dest="derive_color", type=Path,
                                 help="new folder for Bayer luminance and luminance from four-channel images")
            command.add_argument("--lens-profile", type=Path, help="optional checksum-recorded lens correction")
            command.add_argument("--derived-folder", type=Path, help="new folder for lens-corrected Analysis inputs")
    calibrate = commands.add_parser("calibrate", help="fit a single or stereo checkerboard calibration")
    calibrate.add_argument("experiment", type=Path)
    calibrate.add_argument("--camera", action="append", required=True, help="one camera, or left then right")
    calibrate.add_argument("--image-part", action="append", default=[],
                           help="exact saved GenTL part or 'primary' for each --camera, in the same order")
    calibrate.add_argument("--columns", type=int, required=True, help="inner corners across")
    calibrate.add_argument("--rows", type=int, required=True, help="inner corners down")
    calibrate.add_argument("--square-mm", type=float, required=True, help="measured distance between adjacent corners")
    calibrate.add_argument("--min-views", type=int, default=6)
    calibrate.add_argument("--output", type=Path, required=True)
    for name, help_text in (
        ("export-timing", "export an InSituLink-compatible time and identity table"),
        ("align-load", "align measured force/displacement to committed images"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("experiment", type=Path)
        command.add_argument("--camera")
        command.add_argument("--time-basis", choices=("exposure-midpoint", "host-receive"), default="exposure-midpoint")
        command.add_argument("--clock-anchors", type=Path, help="verified device-to-UTC mappings as a JSON array")
        command.add_argument("--output", required=True, type=Path)
        if name == "align-load":
            command.add_argument("--load", required=True, type=Path)
            command.add_argument("--time-column", default="timestamp_unix_s")
            command.add_argument("--force-column", default="force_N")
            command.add_argument("--force-unit")
            command.add_argument("--displacement-column")
            command.add_argument("--displacement-unit", default="mm")
            command.add_argument("--max-gap-s", type=float, default=2.5)
            command.add_argument("--mode", choices=("instantaneous", "window_mean"), default="instantaneous")
    detector = commands.add_parser("link-detector", help="join detector frames by an explicit common DAQ pulse ID")
    detector.add_argument("experiment", type=Path)
    detector.add_argument("--source", type=Path, required=True, help="external detector/DAQ CSV")
    detector_scope = detector.add_mutually_exclusive_group(required=True)
    detector_scope.add_argument("--trigger-source", help="matches the hardware pairing source_id for direct ledger lookup")
    detector_scope.add_argument("--pulse-report", type=Path, help="recheck a DAQ pulse audit JSON against the experiment and original DAQ CSV")
    detector.add_argument("--daq-source", type=Path, help="relocated DAQ CSV with the same content as the pulse report source")
    detector.add_argument("--event-column", default="event_number")
    detector.add_argument("--frame-column", default="detector_frame")
    detector.add_argument("--output", required=True, type=Path)
    audit = commands.add_parser("audit-pulses", help="compare a finished hardware phase to its independent DAQ pulse CSV")
    audit.add_argument("experiment", type=Path)
    audit.add_argument("--source", required=True, type=Path, help="CSV with the recorded DAQ pulse IDs")
    audit.add_argument("--phase", required=True, type=int, help="hardware phase with shared-counter anchors")
    audit.add_argument("--event-column", default="event_number")
    audit.add_argument("--first-event", type=int, help="inclusive lower pulse ID, when isolating part of the recorded phase")
    audit.add_argument("--last-event", type=int, help="inclusive upper pulse ID, when isolating part of the recorded phase")
    audit.add_argument("--anchor-file", type=Path, help="same-run two-pulse anchors for a phase recorded without a pulse association")
    audit.add_argument("--timing-source", type=Path, help="DAQ CSV with independently timed trigger and exposure-start edges")
    audit.add_argument("--max-skew-ns", type=int, help="maximum inter-camera exposure-start skew, in ns")
    audit.add_argument("--latency-min-ns", type=int, help="minimum accepted trigger-to-exposure delay, in ns")
    audit.add_argument("--latency-max-ns", type=int, help="maximum accepted trigger-to-exposure delay, in ns")
    audit.add_argument("--output", required=True, type=Path, help="new detailed JSON report or CSV pulse table")
    follow = commands.add_parser("follow", help="publish provisional DIC JSONL from committed images")
    follow.add_argument("experiment", type=Path)
    follow.add_argument("--config", required=True, type=Path, help="live ROI and solver settings JSON")
    follow.add_argument("--camera")
    follow.add_argument("--reference-sequence", type=int)
    reference = follow.add_mutually_exclusive_group()
    reference.add_argument("--reference-experiment", type=Path, help="linked earlier experiment with the explicit reference sequence")
    reference.add_argument("--reference-binding", type=Path, help="Capture live reference binding JSON")
    follow.add_argument("--image-part", help="exact saved GenTL part, such as 2/16/0/Mono8")
    follow.add_argument("--all-frames", action="store_true", help="process every committed frame in order")
    follow.add_argument("--poll-s", type=float, default=.4)
    export = commands.add_parser("export-live-results", help="export a desktop live result log to CSV")
    export.add_argument("record", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--fill-load", action="store_true", help="fill missing values from the current saved load table")
    export.add_argument("--load-table", type=Path, help="updated copy of the same load table, implies --fill-load")
    export.add_argument("--frame-range", help="test frame number or inclusive range, such as 501-525; keeps the reference")
    replay = commands.add_parser("replay-live-results", help="recalculate saved live settings over the committed images")
    replay.add_argument("experiment", type=Path)
    replay.add_argument("record", type=Path)
    replay_load = replay.add_mutually_exclusive_group()
    replay_load.add_argument("--images-only", action="store_true", help="replay image measurements without load alignment")
    replay_load.add_argument("--load-table", type=Path, help="updated copy of the same load table")
    return parser


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

    with args.output.open("x", encoding="utf-8", newline="") as stream:
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
    args = _parser().parse_args(argv)
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
