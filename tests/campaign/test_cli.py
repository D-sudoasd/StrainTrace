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


def test_cli_rejects_duplicate_root_alias_and_invalid_timeout(tmp_path: Path, capsys):
    project = tmp_path / "campaign"
    source = tmp_path / "source"
    source.mkdir()
    code = cli.main(["init", str(project), "--name", "dup", "--root", f"raw={source}", "--root", f"raw={source}"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_ARGUMENT
    assert json.loads(captured.err)["error"]["error_code"] == "PROJECT_ROOT_DUPLICATE_ALIAS"

    assert cli.main(["init", str(project), "--name", "ok", "--root", f"raw={source}"]) == cli.EXIT_OK
    capsys.readouterr()
    try:
        cli.main(["export", str(project), str(tmp_path / "out"), "--timeout-s", "nan"])
    except SystemExit as exc:
        assert exc.code == cli.EXIT_ARGUMENT
    else:
        raise AssertionError("expected argparse SystemExit(2) for --timeout-s nan")
    capsys.readouterr()
    from straintrace_campaign.models import CampaignError

    assert cli._exit_for(CampaignError("EXPORT_TIMEOUT_INVALID", "bad")) == cli.EXIT_ARGUMENT


def test_cli_subcommands_verify_project_json_and_blocked_run_dict(tmp_path: Path, monkeypatch, capsys):
    from straintrace_campaign import workflow

    project = tmp_path / "campaign"
    source = tmp_path / "source"
    (source / "batch_01").mkdir(parents=True)
    (source / "batch_01" / "dic_000001.tif").write_bytes(b"img")
    assert cli.main(["init", str(project), "--name", "cli-test", "--root", f"raw={source}"]) == cli.EXIT_OK
    capsys.readouterr()

    assert cli.main(["scan", str(project), "raw", "--adapter", "generic"]) == cli.EXIT_OK
    scan_event = json.loads(capsys.readouterr().out)
    assert scan_event["event"] == "scan_complete"
    assert len(scan_event["acquisitions"]) == 1

    # Verify accepts both project directory and direct project.json path.
    assert cli.main(["verify", str(project / "project.json")]) == cli.EXIT_OK
    verify_event = json.loads(capsys.readouterr().out)
    assert verify_event["report"]["valid"] is True

    # Single-acquisition run returning dict with status=BLOCKED exits with EXIT_GATE.
    monkeypatch.setattr(workflow, "run", lambda *_a, **_k: {"status": "BLOCKED", "reason": "gate"})
    blocked_code = cli.main(["run", str(project), "--acquisition", scan_event["acquisitions"][0]["acquisition_id"]])
    blocked_out = json.loads(capsys.readouterr().out)
    assert blocked_code == cli.EXIT_GATE
    assert blocked_out["result"]["status"] == "BLOCKED"


def test_python_m_straintrace_campaign_help_succeeds():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "straintrace_campaign", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "straintrace campaign" in proc.stdout or "usage:" in proc.stdout.lower()
