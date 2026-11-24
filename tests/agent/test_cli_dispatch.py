"""CLI subcommand dispatch and exit-code classification tests for straintrace_agent.cli."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import ezdic_core as core
from straintrace_agent import cli


def _write_sequence(directory: Path, frames=(1, 2)):
    directory.mkdir(parents=True, exist_ok=True)
    ref = core.generate_synthetic_speckle(112, 112, seed=17)
    for native in frames:
        img = ref if native == frames[0] else core.warp_image_translation(ref, 0.4 * (native - frames[0]), 0.0)
        assert cv2.imwrite(str(directory / f"frame_{native:04d}.png"), np.clip(img, 0, 255).astype(np.uint8))


def _run_cli(capsys, argv: list[str]) -> tuple[int, list[dict]]:
    code = cli.main(argv)
    out = capsys.readouterr().out
    events = [json.loads(line) for line in out.splitlines() if line.strip()]
    return code, events


def test_cli_exit_codes_for_rect_and_config_errors(tmp_path, capsys):
    images = tmp_path / "images"
    _write_sequence(images)

    # Bad --roi / empty --roi / bad --exclude -> EXIT_CONFIG (2) with CLI_USAGE_ERROR
    for bad_args in (
        ["init", "--project", str(tmp_path / "p1"), "--images", str(images), "--time-basis", "frame_index", "--roi", "bad"],
        ["init", "--project", str(tmp_path / "p2"), "--images", str(images), "--time-basis", "frame_index", "--roi", ""],
        ["init", "--project", str(tmp_path / "p3"), "--images", str(images), "--time-basis", "frame_index", "--exclude", "1,2"],
    ):
        code, events = _run_cli(capsys, bad_args)
        assert code == cli.EXIT_CONFIG
        assert events[-1]["error_code"] == "CLI_USAGE_ERROR"

    # Missing frame interval (FRAME_TIME_REQUIRED) -> EXIT_CONFIG (2)
    code, events = _run_cli(capsys, ["init", "--project", str(tmp_path / "p4"), "--images", str(images)])
    assert code == cli.EXIT_CONFIG
    assert events[-1]["error_code"] == "FRAME_TIME_REQUIRED"

    # Bad JSON file -> EXIT_CONFIG (2)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    code, events = _run_cli(capsys, ["init", "--project", str(tmp_path / "p5"), "--images", str(images), "--time-basis", "frame_index", "--gauges-json", str(bad_json)])
    assert code == cli.EXIT_CONFIG
    assert events[-1]["error_code"] == "JSON_INPUT_ERROR"

    # Non-canonical load column without unit -> EXIT_CONFIG (2)
    loads = tmp_path / "loads.csv"
    loads.write_text("Time,Force\n0,10\n1,20\n", encoding="utf-8")
    code, events = _run_cli(capsys, [
        "init", "--project", str(tmp_path / "p6"), "--images", str(images), "--frame-interval-s", "0.5",
        "--loads-csv", str(loads), "--time-column", "Time", "--force-column", "Force",
        "--time-offset-s", "0", "--max-gap-s", "1",
    ])
    assert code == cli.EXIT_CONFIG
    assert events[-1]["error_code"] == "LOAD_TIME_UNIT"

    # Invalid batch max_workers -> EXIT_CONFIG (2)
    code, events = _run_cli(capsys, ["batch", "--project", str(tmp_path / "p6"), "--max-workers", "99"])
    assert code == cli.EXIT_CONFIG
    assert events[-1]["error_code"] == "BATCH_WORKERS"


def test_cli_full_subcommand_dispatch_lifecycle(tmp_path, capsys):
    images = tmp_path / "exp" / "images"
    _write_sequence(images, frames=(1, 2))
    project = tmp_path / "project"

    # discover
    code, events = _run_cli(capsys, ["discover", "--folder", str(tmp_path / "exp")])
    assert code == cli.EXIT_OK and events[-1]["status"] == "ready"

    # init
    code, events = _run_cli(capsys, ["init", "--project", str(project), "--images", str(images), "--frame-interval-s", "0.5", "--step-hint-px", "8"])
    assert code == cli.EXIT_OK and events[-1]["event"] == "project_initialized"

    # inspect
    code, events = _run_cli(capsys, ["inspect", "--project", str(project)])
    assert code == cli.EXIT_OK and events[-1]["frame_count"] == 2

    # plan
    code, events = _run_cli(capsys, ["plan", "--project", str(project), "--progress-json"])
    assert code == cli.EXIT_OK and events[-1]["event"] == "plan_complete"

    # status
    code, events = _run_cli(capsys, ["status", "--project", str(project)])
    assert code == cli.EXIT_OK and events[-1]["locked"] is False

    # run
    code, events = _run_cli(capsys, ["run", "--project", str(project), "--no-plots"])
    assert code == cli.EXIT_OK and events[-1]["event"] == "run_complete"
    run_dir = Path(events[-1]["result"]["run_dir"])

    # verify
    code, events = _run_cli(capsys, ["verify", "--project", str(project)])
    assert code == cli.EXIT_OK and events[-1]["valid"] is True

    # plot
    code, events = _run_cli(capsys, ["plot", "--project", str(project), "--every", "1"])
    assert code == cli.EXIT_OK and events[-1]["status"] == "complete"

    # export
    code, events = _run_cli(capsys, ["export", "--project", str(project)])
    assert code == cli.EXIT_OK and events[-1]["status"] == "complete"

    # extract (including empty --roi rejection and valid --roi)
    code, events = _run_cli(capsys, ["extract", "--project", str(project), "--roi", ""])
    assert code == cli.EXIT_CONFIG and events[-1]["error_code"] == "CLI_USAGE_ERROR"
    code, events = _run_cli(capsys, ["extract", "--project", str(project), "--component", "invalid_comp", "--roi", "20,80,20,80"])
    assert code == cli.EXIT_CONFIG and events[-1]["error_code"] == "COMPONENT_INVALID"
    code, events = _run_cli(capsys, ["extract", "--project", str(project), "--component", "Exx", "--roi", "20,80,20,80"])
    assert code == cli.EXIT_OK and Path(events[-1]["roi_csv"]).is_file()

    # report & verify-report
    code, events = _run_cli(capsys, ["report", "--project", str(project), "--include-checkpoints"])
    assert code == cli.EXIT_OK and events[-1]["verification"]["valid"] is True
    code, events = _run_cli(capsys, ["verify-report", "--report", str(run_dir / "report")])
    assert code == cli.EXIT_OK and events[-1]["valid"] is True

    # reprocess
    regions_json = tmp_path / "regions.json"
    regions_json.write_text(json.dumps([{"name": "center", "bounds": [30, 80, 30, 80]}]), encoding="utf-8")
    code, events = _run_cli(capsys, ["reprocess", "--project", str(project), "--regions-json", str(regions_json)])
    assert code == cli.EXIT_OK and events[-1]["status"] == "complete"

    # cancel & recover-lock
    code, events = _run_cli(capsys, ["cancel", "--project", str(project)])
    assert code == cli.EXIT_OK and events[-1]["cancel_requested"] is True
    (project / ".agent.lock").write_bytes(b"")
    code, events = _run_cli(capsys, ["recover-lock", "--project", str(project), "--expected-pid", "99999999"])
    assert code == cli.EXIT_OK and events[-1]["recovered"] is True

    # update & analyze
    changes_json = tmp_path / "changes.json"
    changes_json.write_text(json.dumps({"step_hint_px": 9}), encoding="utf-8")
    code, events = _run_cli(capsys, ["update", "--project", str(project), "--changes-json", str(changes_json)])
    assert code == cli.EXIT_OK and events[-1]["status"] == "updated"
    code, events = _run_cli(capsys, ["analyze", "--project", str(project), "--no-plots"])
    assert code == cli.EXIT_OK and events[-1]["event"] == "analysis_complete"

    # auto on new project
    auto_proj = tmp_path / "auto-proj"
    code, events = _run_cli(capsys, ["auto", "--folder", str(tmp_path / "exp"), "--project", str(auto_proj), "--no-plots"])
    assert code == cli.EXIT_OK and events[-1]["status"] == "complete"
