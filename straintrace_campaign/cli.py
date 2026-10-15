"""GUI-free command-line interface for campaign workflow operations."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

from .models import CampaignError
from . import workflow


EXIT_OK = 0
EXIT_ARGUMENT = 2
EXIT_RUNTIME = 3
EXIT_GATE = 4


def _emit(value: Any, stream: Any = None) -> None:
    output = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (stream or sys.stdout).write(output + "\n")
    (stream or sys.stdout).flush()


def _parse_root(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("root must be ALIAS=ABSOLUTE_PATH")
    alias, path = value.split("=", 1)
    if not alias.strip() or not path.strip():
        raise argparse.ArgumentTypeError("root must be ALIAS=ABSOLUTE_PATH")
    return alias.strip(), path.strip()


def _positive_finite(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _nonnegative_finite(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative finite number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="straintrace campaign", description="Manage reusable DIC acquisition campaigns")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a campaign project")
    init.add_argument("project", type=Path)
    init.add_argument("--name", required=True)
    init.add_argument("--root", action="append", default=[], type=_parse_root, metavar="ALIAS=PATH")

    scan = commands.add_parser("scan", help="discover acquisitions under a configured root")
    scan.add_argument("project", type=Path)
    scan.add_argument("root_alias")
    scan.add_argument("--adapter", default="generic", choices=("generic", "desy", "desy_delun", "desy_camera"))
    scan.add_argument("--pattern")
    scan.add_argument("--exclude", action="append", default=[])

    imported = commands.add_parser("import-keystone", help="import an accepted Keystone package")
    imported.add_argument("project", type=Path)
    imported.add_argument("source", type=Path)

    propose = commands.add_parser("propose-rois", help="propose free-section and gauge locations")
    propose.add_argument("project", type=Path)
    propose.add_argument("acquisition_id")
    propose.add_argument("--free-region", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))

    approve = commands.add_parser("approve-rois", help="approve the displayed ROI proposal")
    approve.add_argument("project", type=Path)
    approve.add_argument("acquisition_id")
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--specimen-confirmed", required=True, choices=("true", "false"))

    run = commands.add_parser("run", help="track one or all approved acquisitions")
    run.add_argument("project", type=Path)
    run.add_argument("--acquisition-id")

    align = commands.add_parser("align", help="align tracked frames to measured force")
    align.add_argument("project", type=Path)
    align.add_argument("acquisition_id")
    align.add_argument("load_path", type=Path)
    align.add_argument("--mode", choices=("instantaneous", "window_mean"), default="instantaneous")
    align.add_argument("--max-gap-s", type=_positive_finite, default=2.5)
    align.add_argument(
        "--measurement-time-field", "--time-column", dest="measurement_time_field",
        choices=("timestamp_unix_s", "estimated_timestamp_unix_s", "time_s"),
        default="timestamp_unix_s",
        help="explicit frame-axis time field; time_s is relative and requires a matching clock ID",
    )
    align.add_argument("--load-time-column", default="timestamp_unix_s")
    align.add_argument("--dic-max-gap-s", type=_positive_finite)
    align.add_argument("--force-column", default="force_N")
    align.add_argument("--force-unit", choices=("N", "kN"))

    export = commands.add_parser("export", help="build and optionally publish a verified package")
    export.add_argument("project", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--publish-to", type=Path)
    export.add_argument("--timeout-s", type=_positive_finite, default=60.0)

    verify = commands.add_parser("verify", help="verify a project or exported package")
    verify.add_argument("path", type=Path)
    return parser


def add_campaign_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Add a forwarding parser beneath the legacy CLI.

    The legacy entrypoint detects the leading ``campaign`` token and delegates
    the remaining argv to :func:`main`, whose parser owns all nine commands.
    This parser exists for consistent top-level help and accepts those tokens
    unchanged; it deliberately does not share argparse child-parser objects.
    """
    campaign = subparsers.add_parser("campaign", help="run DIC acquisition campaign workflows")
    campaign.add_argument("campaign_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return campaign


def _exit_for(error: CampaignError) -> int:
    code = error.code
    if code.startswith(("ROI_", "GATE_", "DIMENSION_", "ACQUISITION_IDENTITY")) or code in {"PROJECT_LOCKED", "CAMPAIGN_CANCELLED"}:
        return EXIT_GATE
    if code.startswith(("PROJECT_", "ROOT_", "ADAPTER_", "DISCOVERY_", "ACQUISITION_PATCH", "ALIGNMENT_OPTION", "EXPORT_TIMEOUT", "LOAD_SOURCE")):
        return EXIT_ARGUMENT
    return EXIT_RUNTIME


def _install_cancel_handler(cancel: threading.Event) -> tuple[Any, Any]:
    previous = signal.getsignal(signal.SIGINT)

    def request_cancel(_signum: int, _frame: Any) -> None:
        cancel.set()

    try:
        signal.signal(signal.SIGINT, request_cancel)
        return previous, request_cancel
    except ValueError:  # called off the main thread
        return previous, None


def _dispatch(args: argparse.Namespace) -> int:
    command = args.command
    if command == "init":
        roots = dict(args.root)
        result = workflow.init_project(args.project, args.name, roots)
        _emit({"event": "project_created", "project_id": result["project_id"], "path": str(args.project.resolve())})
        return EXIT_OK
    if command == "scan":
        result = workflow.scan(args.project, args.root_alias, args.adapter, args.pattern, args.exclude)
        _emit({"event": "scan_complete", "added_count": len(result), "acquisitions": result})
        return EXIT_OK
    if command == "import-keystone":
        result = workflow.import_keystone(args.project, args.source)
        _emit({"event": "import_complete", "added_count": len(result), "acquisitions": result})
        return EXIT_OK
    if command == "propose-rois":
        result = workflow.propose_rois(args.project, args.acquisition_id, args.free_region)
        _emit({"event": "roi_proposed", "acquisition_id": args.acquisition_id, "proposal": result})
        return EXIT_OK
    if command == "approve-rois":
        result = workflow.approve_rois(args.project, args.acquisition_id, args.reviewer, args.specimen_confirmed == "true")
        _emit({"event": "roi_approved", "acquisition_id": args.acquisition_id, "approval": result})
        return EXIT_OK
    if command == "run":
        cancellation = threading.Event()
        previous, handler = _install_cancel_handler(cancellation)
        try:
            result = workflow.run(
                args.project,
                acquisition_id=args.acquisition_id,
                progress=_emit,
                cancel=cancellation,
            )
        finally:
            if handler is not None:
                signal.signal(signal.SIGINT, previous)
        _emit({"event": "run_complete", "result": result})
        if isinstance(result, list) and any(item.get("status") == "BLOCKED" for item in result):
            return EXIT_GATE
        return EXIT_OK
    if command == "align":
        result = workflow.align(
            args.project,
            args.acquisition_id,
            args.load_path,
            mode=args.mode,
            max_gap_s=args.max_gap_s,
            measurement_time_field=args.measurement_time_field,
            load_time_column=args.load_time_column,
            dic_max_gap_s=args.dic_max_gap_s,
            force_column=args.force_column,
            force_unit=args.force_unit,
        )
        _emit({"event": "alignment_complete", "acquisition_id": args.acquisition_id, "result": result})
        return EXIT_OK
    if command == "export":
        result = workflow.export(args.project, args.output, args.publish_to, args.timeout_s)
        _emit({"event": "export_complete", "receipt": result})
        return EXIT_OK
    if command == "verify":
        result = workflow.verify(args.path)
        _emit({"event": "verification_complete", "report": result})
        return EXIT_OK if result.get("valid") else EXIT_GATE
    raise CampaignError("CLI_COMMAND_UNKNOWN", f"unknown campaign command: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return _dispatch(args)
    except CampaignError as exc:
        _emit({"error": exc.as_dict()}, stream=sys.stderr)
        return _exit_for(exc)
    except KeyboardInterrupt:
        _emit({"error": {"error_code": "CAMPAIGN_CANCELLED", "message": "campaign operation was cancelled"}}, stream=sys.stderr)
        return EXIT_GATE


if __name__ == "__main__":
    raise SystemExit(main())
