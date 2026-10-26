"""Review regressions for one-run delivery, interruption and report recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

import ezdic_core as core
from straintrace_agent import api, reporting


def _experiment(tmp_path):
    folder = tmp_path / "experiment"
    images = folder / "images"
    images.mkdir(parents=True)
    reference = core.generate_synthetic_speckle(112, 112, seed=17)
    for native, stretch in ((1, 1.0), (2, 1.005), (4, 1.015)):
        image = reference if native == 1 else core.warp_image_deformation_gradient(reference, [[stretch, 0], [0, 1]])
        assert cv2.imwrite(str(images / f"frame_{native:04d}.png"), np.clip(image, 0, 255).astype(np.uint8))
    return folder, tmp_path / "project"


@pytest.fixture
def complete_run(tmp_path):
    folder, project = _experiment(tmp_path)
    result = api.auto_analyze(folder, project, make_plots=False)
    assert result["status"] == "complete", result
    return folder, project, result


def _tree_digest(folder):
    result = {}
    for path in folder.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(folder).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def test_cancelled_auto_returns_saved_outputs_and_can_resume(tmp_path):
    folder, project = _experiment(tmp_path)

    def cancel_after_start(event):
        if event.get("event") == "run_started":
            api.cancel(project)

    cancelled = api.auto_analyze(folder, project, progress=cancel_after_start, make_plots=False)
    assert cancelled["status"] == cancelled["run"]["status"] == "cancelled"
    assert cancelled["run"]["numerical_quality_passed"] is False
    assert Path(cancelled["report"]["report_html"]).is_file()
    assert cancelled["report"]["partial"] is True
    resumed = api.auto_analyze(folder, project, make_plots=False)
    assert resumed["status"] == "complete"
    assert resumed["run"]["run_id"] == cancelled["run"]["run_id"]


@pytest.mark.parametrize("boundary", ["run", "report"])
def test_auto_rejects_a_different_current_run_at_each_delivery_boundary(tmp_path, monkeypatch, boundary):
    folder, project = _experiment(tmp_path)
    real_run, real_report = api.run, api.report
    run_ids = {}

    def replace_current():
        api.update_project(project, {"step_hint_px": 8})
        api.plan_project(project)
        replacement = real_run(project, make_plots=False)
        assert replacement["status"] == "complete"
        run_ids["replacement"] = replacement["run_id"]

    if boundary == "run":
        def interrupted_run(*args, **kwargs):
            result = real_run(*args, **kwargs)
            run_ids["original"] = result["run_id"]
            replace_current()
            return result
        monkeypatch.setattr(api, "run", interrupted_run)
    else:
        def interrupted_report(*args, **kwargs):
            result = real_report(*args, **kwargs)
            run_ids["original"] = result["run_id"]
            replace_current()
            return result
        monkeypatch.setattr(api, "report", interrupted_report)

    with pytest.raises(api.AgentError) as error:
        api.auto_analyze(folder, project, make_plots=False)
    assert error.value.code == "AUTOMATION_STALE_RUN"
    assert run_ids["original"] != run_ids["replacement"]
    assert error.value.details["expected_run_id"] == run_ids["original"]
    assert error.value.details["current_run_id"] == run_ids["replacement"]
    assert api.verify(project)["valid"] is True


@pytest.mark.parametrize("failure_boundary", ["before_manifest_write", "after_manifest_write"])
def test_report_commit_failure_rolls_back_package_and_outer_manifest(complete_run, monkeypatch, failure_boundary):
    _, project, initial = complete_run
    run_root = Path(initial["run"]["run_dir"])
    before_package = _tree_digest(run_root / "report")
    before_manifest = (run_root / "manifest.json").read_bytes()
    original_refresh = api._refresh_manifest_artifacts

    def fail_refresh(run):
        if failure_boundary == "after_manifest_write":
            original_refresh(run)
        raise OSError("injected report manifest commit failure")

    with monkeypatch.context() as context:
        context.setattr(api, "_refresh_manifest_artifacts", fail_refresh)
        with pytest.raises(OSError, match="injected"):
            api.report(project, include_checkpoints=False)
    assert _tree_digest(run_root / "report") == before_package
    assert (run_root / "manifest.json").read_bytes() == before_manifest
    assert api.verify(project)["valid"] is True
    assert not list(run_root.glob(".report-candidate-*"))
    assert not list(run_root.glob(".report-backup-*"))
    assert api.report(project, include_checkpoints=False)["verification"]["valid"] is True
    assert api.verify(project)["valid"] is True


def test_report_can_recover_interrupted_hash_commit_without_recalculating(complete_run):
    _, project, initial = complete_run
    run_root = Path(initial["run"]["run_dir"])
    before_checkpoints = _tree_digest(run_root / "checkpoints")
    config = api.read_project_config(project)
    pointer = json.loads((project / "CURRENT.json").read_text(encoding="utf-8"))
    plan = json.loads((project / pointer["current_plan"]).read_text(encoding="utf-8"))
    # Simulate a process exit after the report rename but before outer hashes.
    with api._project_lock(project, purpose="simulate_interrupted_report_commit"):
        reporting.build_report(run_root, config=config, plan=plan, include_checkpoints=False)
    broken = api.verify(project)
    assert broken["valid"] is False
    assert broken["issues"] and all(issue.get("path", "").startswith("report/") for issue in broken["issues"])
    assert api.report(project, include_checkpoints=True)["verification"]["valid"] is True
    assert api.verify(project)["valid"] is True
    assert _tree_digest(run_root / "checkpoints") == before_checkpoints


def test_report_recovery_does_not_waive_tampered_source_csv(complete_run):
    _, project, initial = complete_run
    run_root = Path(initial["run"]["run_dir"])
    before_package = _tree_digest(run_root / "report")
    with (run_root / "summary.csv").open("ab") as stream:
        stream.write(b"tampered source\n")
    with pytest.raises(api.AgentError) as error:
        api.report(project)
    assert error.value.code == "REPORT_SOURCE_INVALID"
    assert _tree_digest(run_root / "report") == before_package


def test_public_report_function_remains_callable_after_loading_implementation():
    code = (
        "import importlib; import straintrace_agent as package; "
        "before=package.report; "
        "importlib.import_module('straintrace_agent.reporting'); "
        "from straintrace_agent.reporting import verify_report; "
        "from straintrace_agent import report; "
        "assert callable(report) and package.report is before is report"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_stress_objective_requires_load_support_even_when_dic_and_integrity_pass(tmp_path):
    from scripts.validate_agent_upgrade import build_experiment

    folder = tmp_path / "experiment"
    options, _ = build_experiment(folder, failed_frame=False)
    options["load_time_offset_s"] = 100.0
    result = api.auto_analyze(folder, tmp_path / "project", objective="stress_strain", options=options, make_plots=False)
    assert result["run"]["status"] == "complete"
    assert result["run"]["numerical_quality_passed"] is True
    assert result["verification"]["valid"] is True
    summary = pd.read_csv(Path(result["run"]["run_dir"]) / "summary.csv")
    assert summary["stress_MPa"].isna().all()
    assert result["status"] == "attention_required"
    assert result["objective_issue"]
