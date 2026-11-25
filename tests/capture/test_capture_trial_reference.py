import json
import sqlite3
import time

import pytest

from straintrace_capture import CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan, prepare_trial, trial_report
from straintrace_capture.cli import main
from straintrace_capture.presentation import format_trial_comparison
from straintrace_capture.storage import DATABASE, sha256_file
from straintrace_capture.trial_quality import rescan_trial_images
from straintrace_capture.trial_reference import archive_trial_reference, referenced_trial_report

from test_capture_continuation_setup import _registry
from test_capture_trial_quality import make_trial, trial_plan


def test_formal_ledger_binds_the_trial_to_actual_readbacks_and_tracks_unstarted_phases(tmp_path, capsys):
    config = CaptureConfig.from_dict({
        "config_version": 1, "name": "试采与正式采集", "cameras": [{"id": "sim:0", "settings": {"width": 96, "height": 80}}],
        "phases": [{"mode": "single", "role": "reference"}, {"mode": "single", "start_confirmation": "核对夹具后开始。"}],
    })
    formal = tmp_path / "正式实验"
    reports = []
    for number in (1, 2):
        plan, output = prepare_trial(config, formal, phase_number=number)
        trial = CaptureSession(plan, output, registry=_registry(1200.0, "Mono12", []))
        trial.start()
        assert trial.wait(5)["state"] == "completed"
        reports.append(trial_report(output))

    plan_check = inspect_plan(config, formal, trial_reports=reports)
    reference = archive_trial_reference(plan_check, reports)
    assert reference and not formal.exists()
    formal_run = CaptureSession(config, formal, registry=_registry(2000.0, "Mono12", []), trial_reference=reference)
    formal_run.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        wait = formal_run.status().get("phase_confirmation") or {}
        if wait.get("state") == "waiting":
            break
        time.sleep(0.01)
    formal_run.request_stop()
    assert formal_run.wait(5)["state"] == "stopped" and wait.get("state") == "waiting"

    with ExperimentReader(formal) as reader:
        comparison = reader.trial_comparison()
        assert reader.verify()["trial_comparison"] == comparison
    first, second = comparison["stages"]
    assert first["experiment_id"] == reports[0]["experiment_id"]
    assert first["setup_check"]["trial_experiment_id"] == first["experiment_id"]
    assert first["setup_check"]["identities"] == {"sim:0": "matched"}
    changed, = first["setup_check"]["fields"]
    assert changed == {"camera": "sim:0", "field": "exposure_us", "recorded": 1200.0, "unit": "us",
                       "current": 2000.0, "current_unit": "us", "status": "changed"}
    assert first["setup_check"]["matched_fields"] > 0
    assert second["experiment_id"] == reports[1]["experiment_id"] and second["setup_check"] is None
    assert "试采 1200 µs → 正式 2000 µs" in format_trial_comparison(comparison)
    assert "正式阶段尚未记录相机回读" in format_trial_comparison(comparison)
    assert main(["trial-comparison", str(formal)]) == 0
    assert json.loads(capsys.readouterr().out) == comparison


def test_reference_replays_full_checks_and_can_follow_moved_trials(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    trial = make_trial(project, 105, clip=(12,))
    report = trial_report(trial)
    report["cameras"]["sim:0"]["image_check"] = rescan_trial_images(trial, report["experiment_id"], "sim:0", all_frames=True)
    formal = project / "formal"
    config = trial_plan(105, 1)
    preflight = inspect_plan(config, formal, trial_reports=[report])
    stage, = archive_trial_reference(preflight, [report], image_warnings_confirmed=True)["stages"]
    assert stage["image_warnings_confirmed"] and stage["image_checks"]["sim:0"]["flagged"] == 1
    assert archive_trial_reference(inspect_plan(trial_plan(105, 1, {"exposure_us": 1500}), formal, trial_reports=[report]),
                                   [report]) is None
    formal.mkdir()
    original = sha256_file(trial / DATABASE)
    relocated = project.rename(tmp_path / "moved")
    moved_trial = relocated / trial.name
    reopened = referenced_trial_report(relocated / "formal", stage)
    assert reopened["output"] == str(moved_trial) and sha256_file(moved_trial / DATABASE) == original
    evidence = reopened["cameras"]["sim:0"]["image_check"]
    assert evidence["checked"] == 105 and [row["sequence"] for row in evidence["rows"] if row["flags"]] == [12]

    elsewhere = moved_trial.rename(tmp_path / "separate-trial")
    replacement = make_trial(relocated, 4).rename(moved_trial)
    with pytest.raises(CaptureError) as missing:
        referenced_trial_report(relocated / "formal", stage)
    assert missing.value.code == "TRIAL_REFERENCE_UNAVAILABLE"
    with pytest.raises(CaptureError) as mismatch:
        referenced_trial_report(relocated / "formal", stage, directory=replacement)
    assert mismatch.value.code == "TRIAL_REFERENCE_MISMATCH"
    assert referenced_trial_report(relocated / "formal", stage, directory=elsewhere)["experiment_id"] == stage["experiment_id"]

    db = sqlite3.connect(elsewhere / DATABASE)
    try:
        with db:
            ordinal, record = db.execute("SELECT ordinal, record FROM frames WHERE sequence=12").fetchone()
            frame = json.loads(record)
            frame["timestamps"]["received_utc"]["value"] += 1
            db.execute("UPDATE frames SET record=? WHERE ordinal=?", (json.dumps(frame), ordinal))
    finally:
        db.close()
    with pytest.raises(CaptureError) as changed:
        referenced_trial_report(relocated / "formal", stage, directory=elsewhere)
    assert changed.value.code == "TRIAL_REFERENCE_CHANGED"
