"""Finite-JSON command line boundary for ``straintrace agent``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import api


EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_RUNTIME = 3
EXIT_QUALITY = 4


def _emit(value: Any) -> None:
    # JSON escapes keep JSONL safe on Windows consoles using legacy code pages
    # while json.loads still restores the original Unicode values.
    sys.stdout.write(json.dumps(api._json_safe(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def _error(code: str, message: str, path: str = "$") -> None:
    _emit({"event": "error", "error_code": code, "message": message, "path": path})


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        _error("CLI_USAGE_ERROR", message)
        raise SystemExit(EXIT_CONFIG)


def _parse_rect(text: str, label: str, *, float_values: bool = False) -> list[int] | list[float]:
    try:
        values = [float(part.strip()) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be four comma-separated numbers") from exc
    if len(values) != 4 or not all(__import__("math").isfinite(value) for value in values):
        raise argparse.ArgumentTypeError(f"{label} must be four finite comma-separated numbers")
    if float_values:
        return values
    if any(value != int(value) for value in values):
        raise argparse.ArgumentTypeError(f"{label} values must be integers")
    return [int(value) for value in values]


def _json_file(path: str | None, expected: type, label: str) -> Any:
    if path is None:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON value: {token}")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise api.AgentError("JSON_INPUT_ERROR", f"cannot read {label}: {exc}", path=path) from exc
    if not isinstance(value, expected):
        raise api.AgentError("JSON_INPUT_TYPE", f"{label} must contain a JSON {expected.__name__}", path=path)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="straintrace agent", description="Headless, restartable DIC project workflow")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    init = sub.add_parser("init", help="index an image directory and create a sample project")
    init.add_argument("--project", type=Path, required=True)
    init.add_argument("--images", type=Path, required=True)
    init.add_argument("--frame-interval-s", type=float)
    init.add_argument("--frame-map-csv", type=Path)
    init.add_argument("--reference-native-frame", type=int)
    init.add_argument("--loads-csv", type=Path)
    init.add_argument("--time-column")
    init.add_argument("--load-time-unit", choices=("s", "ms", "min", "h"))
    init.add_argument("--time-offset-s", type=float)
    init.add_argument("--clock-confirmed", action="store_true")
    init.add_argument("--max-gap-s", type=float)
    init.add_argument("--force-column")
    init.add_argument("--force-unit", choices=("N", "kN"))
    init.add_argument("--area-mm2", type=float)
    init.add_argument("--displacement-column")
    init.add_argument("--displacement-unit", choices=("mm", "um"))
    init.add_argument("--displacement-source", choices=("crosshead", "machine_extensometer", "other_machine_channel"))
    init.add_argument("--machine-gauge-length-mm", type=float)
    init.add_argument("--mask-file", type=Path)
    init.add_argument("--roi", help="analysis rectangle as x,y,width,height in px")
    init.add_argument("--exclude", action="append", default=[], help="exclusion rectangle x,y,width,height; repeatable")
    init.add_argument("--gauges-json", type=Path, help="JSON array of virtual-gauge endpoint queries")
    init.add_argument("--calibration-json", type=Path, help="explicit mm_per_px/origin_px/origin_mm/axis_sign object")
    init.add_argument("--step-hint-px", type=int)
    init.add_argument("--plot-every", type=int, default=0, help="plot every Nth frame, capped at 24 frames; default selects first/middle/last/load/strain stages")

    for name, help_text in (
        ("inspect", "read-only inventory and project state"), ("plan", "detect a texture ROI candidate and test bounded solver settings"),
        ("run", "execute/resume frame checkpoints and produce outputs"), ("analyze", "plan and run in one operation"),
        ("status", "read-only run and lock state"), ("verify", "read-only provenance and checksum verification"),
        ("plot", "rebuild sparse field maps without recomputing DIC"), ("export", "rebuild tables/report without recomputing DIC"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--project", type=Path, required=True)
        if name in {"run", "analyze"}:
            command.add_argument("--no-plots", action="store_true")
            command.add_argument("--field-csv", action="store_true")
            command.add_argument("--progress-json", action="store_true", help="emit line-delimited JSON progress events")
        if name == "plot":
            command.add_argument("--every", type=int)
        if name == "export":
            command.add_argument("--field-csv", action="store_true")
        if name == "plan":
            command.add_argument("--progress-json", action="store_true")

    extract_parser = sub.add_parser("extract", help="extract a region or path from saved per-frame fields")
    extract_parser.add_argument("--project", type=Path, required=True)
    extract_parser.add_argument("--component", default="exx")
    extract_parser.add_argument("--roi", help="inclusive xmin,xmax,ymin,ymax coordinates")
    extract_parser.add_argument("--path-json", type=Path, help="JSON array of [x,y] points")
    extract_parser.add_argument("--max-distance", type=float)
    extract_parser.add_argument("--coordinate-system", choices=("image", "calibrated"), default="image")

    batch_parser = sub.add_parser("batch", help="run independent sample projects with bounded process concurrency")
    batch_parser.add_argument("--project", type=Path, action="append", required=True)
    batch_parser.add_argument("--max-workers", type=int, default=2)
    batch_parser.add_argument("--no-plots", action="store_true")
    batch_parser.add_argument("--field-csv", action="store_true")

    cancel_parser = sub.add_parser("cancel", help="request cancellation at the next DIC progress point")
    cancel_parser.add_argument("--project", type=Path, required=True)
    recover = sub.add_parser("recover-lock", help="explicitly recover a confirmed dead local lock owner")
    recover.add_argument("--project", type=Path, required=True)
    recover.add_argument("--expected-pid", type=int, required=True)
    return parser


def _progress(enabled: bool):
    if not enabled:
        return None
    return _emit


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.command == "init":
            result = api.init_project(
                args.project, args.images, frame_interval_s=args.frame_interval_s,
                frame_map_csv=args.frame_map_csv, reference_native_frame=args.reference_native_frame,
                loads_csv=args.loads_csv, load_time_column=args.time_column,
                load_time_unit=args.load_time_unit, load_time_offset_s=args.time_offset_s,
                clock_confirmed=args.clock_confirmed, max_gap_s=args.max_gap_s,
                load_force_column=args.force_column, force_unit=args.force_unit,
                area_mm2=args.area_mm2, displacement_column=args.displacement_column,
                displacement_unit=args.displacement_unit, displacement_source=args.displacement_source,
                machine_gauge_length_mm=args.machine_gauge_length_mm, mask_file=args.mask_file,
                roi_override_px=_parse_rect(args.roi, "roi") if args.roi else None,
                exclusions_px=[_parse_rect(item, "exclude") for item in args.exclude],
                gauges=_json_file(args.gauges_json, list, "gauges JSON"),
                calibration=_json_file(args.calibration_json, dict, "calibration JSON"),
                step_hint_px=args.step_hint_px, plot_every=args.plot_every,
            )
            _emit({"event": "project_initialized", **result})
        elif args.command == "inspect":
            _emit(api.inspect_project(args.project))
        elif args.command == "plan":
            result = api.plan_project(args.project, progress=_progress(args.progress_json))
            _emit({"event": "plan_complete", "plan": result})
            return EXIT_OK if result.get("status") == "ready" else EXIT_QUALITY
        elif args.command == "run":
            result = api.run(args.project, progress=_progress(args.progress_json), make_plots=not args.no_plots, field_csv=args.field_csv)
            _emit({"event": "run_complete", "result": result})
            return EXIT_OK if result.get("scientific_ok") else EXIT_QUALITY
        elif args.command == "analyze":
            result = api.analyze(args.project, progress=_progress(args.progress_json), make_plots=not args.no_plots, field_csv=args.field_csv)
            _emit({"event": "analysis_complete", "result": result})
            return EXIT_OK if result.get("run", {}).get("scientific_ok") else EXIT_QUALITY
        elif args.command == "status":
            _emit(api.status(args.project))
        elif args.command == "verify":
            result = api.verify(args.project)
            _emit(result)
            return EXIT_OK if result.get("valid") else EXIT_QUALITY
        elif args.command == "plot":
            _emit(api.plot(args.project, plot_every=args.every))
        elif args.command == "export":
            _emit(api.export(args.project, field_csv=args.field_csv))
        elif args.command == "extract":
            roi = _parse_rect(args.roi, "roi", float_values=True) if args.roi else None
            points = _json_file(args.path_json, list, "path JSON")
            result = api.extract(args.project, component=args.component, roi=roi, path_points=points, max_distance=args.max_distance, coordinate_system=args.coordinate_system)
            _emit(result)
        elif args.command == "batch":
            result = api.batch(args.project, max_workers=args.max_workers, field_csv=args.field_csv, make_plots=not args.no_plots)
            _emit(result)
            return EXIT_OK if result.get("status") == "complete" and result.get("scientific_status") == "pass" else EXIT_QUALITY
        elif args.command == "cancel":
            _emit(api.cancel(args.project))
        elif args.command == "recover-lock":
            _emit(api.recover_lock(args.project, expected_pid=args.expected_pid))
        else:
            raise api.AgentError("CLI_COMMAND_UNSUPPORTED", f"unsupported agent command: {args.command}")
        return EXIT_OK
    except api.AgentError as exc:
        _emit({"event": "error", **exc.as_dict()})
        if exc.code == "PLAN_REVIEW_REQUIRED":
            return EXIT_QUALITY
        return EXIT_CONFIG if exc.code.startswith("CONFIG") or exc.code.startswith("CLI_") else EXIT_RUNTIME
    except Exception as exc:
        _emit({"event": "error", "error_code": "AGENT_RUNTIME_ERROR", "message": f"{type(exc).__name__}: {exc}", "path": "$"})
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
