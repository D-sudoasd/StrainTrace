"""Argument definitions shared by the Capture console and module entry points."""

import argparse
import json
from pathlib import Path

from .trial import DEFAULT_TRIAL_SECONDS


def build_parser() -> argparse.ArgumentParser:
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
