"""Capture automation, independent of the desktop."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from . import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, recover_experiment
from .control import CaptureControl, send_command
from .handoff import analysis_settings, analysis_snapshot, derive_analysis_inputs, write_snapshot
from .storage import TERMINAL_STATES


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
    run = commands.add_parser("run", help="acquire to a new experiment directory")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--control", action="store_true", help="accept operator commands through the experiment's local folder")
    run.add_argument("--progress", action="store_true", help="write live JSON progress to stderr")
    run.add_argument("--continue-from", type=Path, help="start a new linked experiment after a prior terminal run")
    for name, help_text in (
        ("status", "read experiment state and camera records"), ("frames", "list committed frame metadata"),
        ("events", "list acquisition events"), ("pairing", "report trigger-based associations and missing cameras"),
        ("verify", "verify frame files, hashes and ledger state"), ("recover", "finalize an interrupted ledger"),
        ("trigger", "request a manual or external event from an armed session"),
        ("stop", "request a graceful stop and drain of the raw queue"),
        ("prepare-analysis", "write the committed frame binding for existing Analysis"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("experiment", type=Path)
        if name == "frames":
            command.add_argument("--camera")
            command.add_argument("--after", type=int, default=0)
            command.add_argument("--limit", type=int, default=100)
        if name == "verify":
            command.add_argument("--size-only", action="store_true", help="skip SHA-256 checks of the frame files")
        if name == "trigger":
            command.add_argument("--source", default="operator")
            command.add_argument("--payload", type=json.loads, help="external event metadata as JSON")
        if name == "prepare-analysis":
            command.add_argument("--camera")
            command.add_argument("--reference-sequence", type=int)
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
    detector.add_argument("--trigger-source", required=True, help="matches the hardware pairing source_id")
    detector.add_argument("--event-column", default="event_number")
    detector.add_argument("--frame-column", default="detector_frame")
    detector.add_argument("--output", required=True, type=Path)
    follow = commands.add_parser("follow", help="publish provisional DIC JSONL from committed images")
    follow.add_argument("experiment", type=Path)
    follow.add_argument("--config", required=True, type=Path, help="live ROI and solver settings JSON")
    follow.add_argument("--camera")
    follow.add_argument("--reference-sequence", type=int)
    follow.add_argument("--all-frames", action="store_true", help="process every committed frame in order")
    follow.add_argument("--poll-s", type=float, default=.4)
    return parser


def _run(args) -> tuple[dict, int]:
    config = CaptureConfig.load(args.config)
    if args.continue_from:
        with ExperimentReader(args.continue_from) as reader:
            prior = reader.experiment
        if prior["state"] not in TERMINAL_STATES:
            raise CaptureError("CONTINUATION_ACTIVE", "the prior run must finish or be recovered before continuation")
        data = config.to_dict()
        data["metadata"]["continuation_of"] = prior["experiment_id"]
        config = CaptureConfig.from_dict(data)
    session = CaptureSession(config, args.output)
    session.start()
    try:
        with CaptureControl(session) if args.control else nullcontext():
            while True:
                try:
                    status = session.wait(0.5)
                    break
                except TimeoutError:
                    if args.progress:
                        print(json.dumps(session.status(), ensure_ascii=True, allow_nan=False), file=sys.stderr, flush=True)
                except KeyboardInterrupt:
                    session.request_stop()
    except BaseException:
        session.request_stop()
        session.wait()
        raise
    has_losses = any(
        any(camera[key] for key in ("app_unsaved", "native_counter_gaps", "native_counter_duplicates",
                                    "hardware_identity_missing", "reported_driver_skips", "transport_errors"))
        for camera in status["cameras"].values()
    )
    result = {"output": str(session.output), **status}
    incomplete = has_losses or status["scheduled_misses"] or status["short_pre_windows"]
    if status["state"] == "completed" and any(phase.hardware_pairing for phase in config.phases):
        with ExperimentReader(session.output) as reader:
            pairing = reader.pairing()
        result["hardware_counter_gap_ranges"] = pairing["hardware_counter_gap_ranges"]
        result["incomplete_trigger_groups"] = sum(group["status"] != "associated" for group in pairing["groups"])
        incomplete = incomplete or result["incomplete_trigger_groups"] or result["hardware_counter_gap_ranges"]
    return result, 3 if status["state"] == "failed" else (4 if incomplete else 0)


def _execute(args):
    action = args.command
    if action == "discover":
        return CameraRegistry(cti_files=args.cti).discover(args.backends), 0
    if action == "features":
        with CameraRegistry(cti_files=args.cti).camera(args.camera) as camera:
            return {"device": asdict(camera.device), "features": {name: asdict(feature)
                    for name, feature in camera.features().items()}}, 0
    if action == "validate-config":
        return CaptureConfig.load(args.config).to_dict(), 0
    if action == "run":
        return _run(args)
    if action == "recover":
        return recover_experiment(args.experiment), 0
    if action in {"trigger", "stop"}:
        result = send_command(args.experiment, action, source=getattr(args, "source", "operator"),
                              payload=getattr(args, "payload", None))
        return result, 0 if result["ok"] else 3
    if action == "prepare-analysis":
        snapshot = analysis_snapshot(args.experiment, args.camera, args.reference_sequence)
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

        result = calibrate(args.experiment, args.camera, inner_columns=args.columns, inner_rows=args.rows,
                           square_mm=args.square_mm, min_views=args.min_views)
        return {"calibration": str(write_snapshot(result, args.output)), "cameras": list(result["cameras"])}, 0
    if action in ("export-timing", "align-load", "link-detector"):
        return _sync_command(args), 0
    with ExperimentReader(args.experiment) as reader:
        if action == "status":
            return {"experiment": reader.experiment, "cameras": reader.cameras()}, 0
        if action == "frames":
            if args.after < 0 or args.limit < 1:
                raise CaptureError("INVALID_FRAME_QUERY", "--after must be nonnegative and --limit positive")
            return reader.frames(camera_id=args.camera, after=args.after, limit=args.limit), 0
        if action == "events":
            return reader.events(), 0
        if action == "pairing":
            return reader.pairing(), 0
        if action == "verify":
            result = reader.verify(hash_files=not args.size_only)
            return result, 0 if result["complete"] else 4
    raise AssertionError(action)


def _sync_command(args):
    from .handoff import select_camera
    from .synchronization import ClockAnchor, LoadAlignment, link_detector, timing_rows

    action = args.command
    if action == "link-detector":
        result = link_detector(args.experiment, args.source, trigger_source=args.trigger_source,
                               event_column=args.event_column, frame_column=args.frame_column)
        return {"path": str(write_snapshot(result, args.output)), "events": len(result["links"])}
    anchors = ()
    if args.clock_anchors:
        anchors = tuple(ClockAnchor.from_dict(row) for row in json.loads(args.clock_anchors.read_text(encoding="utf-8-sig")))
    if action == "align-load":
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
