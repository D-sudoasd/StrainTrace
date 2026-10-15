from __future__ import annotations

import argparse
import json
from pathlib import Path

from straintrace_campaign import cli


def test_campaign_cli_exposes_all_nine_commands_and_alignment_clock_choices():
    parser = cli.build_parser()
    action = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
    assert set(action.choices) == {
        "init", "scan", "import-keystone", "propose-rois", "approve-rois", "run", "align", "export", "verify",
    }
    args = parser.parse_args([
        "align", "project", "acq", "load.csv",
        "--measurement-time-field", "estimated_timestamp_unix_s",
        "--load-time-column", "clock_s", "--force-column", "load_n",
    ])
    assert args.measurement_time_field == "estimated_timestamp_unix_s"
    assert args.load_time_column == "clock_s"
    assert args.force_column == "load_n"


def test_legacy_campaign_forwarder_accepts_rest_of_argv():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    cli.add_campaign_parser(subparsers)
    args = parser.parse_args(["campaign", "init", "campaign_dir", "--name", "pilot"])
    assert args.command == "campaign"
    assert args.campaign_args == ["init", "campaign_dir", "--name", "pilot"]


def test_cli_init_emits_one_machine_readable_json_event(tmp_path: Path, capsys):
    project = tmp_path / "campaign"
    source = tmp_path / "source"
    source.mkdir()
    code = cli.main(["init", str(project), "--name", "synthetic", "--root", f"raw={source}"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_OK
    event = json.loads(captured.out)
    assert event["event"] == "project_created"
    assert event["project_id"]
    assert not captured.err


def test_cli_requires_explicit_specimen_confirmation(tmp_path: Path, capsys):
    code = cli.main([
        "approve-rois", str(tmp_path / "missing"), "acq", "--reviewer", "operator", "--specimen-confirmed", "false",
    ])
    captured = capsys.readouterr()
    assert code == cli.EXIT_GATE
    error = json.loads(captured.err)
    assert error["error"]["error_code"] == "GATE_SPECIMEN_UNCONFIRMED"
