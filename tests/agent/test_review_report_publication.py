"""Fault-injection checks for preserving the last verified report package."""
import hashlib
import json
from pathlib import Path

import pytest

from straintrace_agent import reporting


def _tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def published_report(tmp_path):
    run = tmp_path / "项目" / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "summary.csv").write_text("native_frame,time_s,status,quality_status\n1,0,complete,pass\n", encoding="utf-8")
    config = {"project_id": "report-publication", "image_dir": str(tmp_path / "images"), "regions": [], "gauges": []}
    plan = {"configuration_sha256": reporting._json_sha256(config), "reference_native_frame": 1}
    plan["plan_sha256"] = reporting._json_sha256(plan)
    source = {"run_id": "run", "configuration_sha256": reporting._json_sha256(config),
              "plan_sha256": plan["plan_sha256"], "source_images": [], "checkpoint_receipts": {},
              "artifacts": {"summary.csv": hashlib.sha256((run / "summary.csv").read_bytes()).hexdigest()}}
    (run / "manifest.json").write_text(json.dumps(source), encoding="utf-8")
    result = reporting.build_report(run, config=config, plan=plan)
    package = Path(result["artifacts"]["report_package"])
    assert reporting.verify_report(package)["valid"]
    return run, config, plan, package, _tree_bytes(package)


def test_failed_publish_and_restore_preserve_verified_backup(published_report, monkeypatch):
    run, config, plan, output, before = published_report
    original_replace = reporting.os.replace
    publication_error = PermissionError("publish blocked by an open report handle")
    restore_error = PermissionError("restore still blocked")

    def blocked_replace(source, destination):
        source, destination = Path(source), Path(destination)
        if destination == output:
            if source.name.startswith(".report-build-"):
                raise publication_error
            if source.name.startswith(".report-old-"):
                raise restore_error
        return original_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", blocked_replace)
    with pytest.raises(PermissionError) as caught:
        reporting.build_report(run, config=config, plan=plan)
    backups = list(run.glob(".report-old-*"))
    assert len(backups) == 1
    assert _tree_bytes(backups[0]) == before
    assert reporting.verify_report(backups[0])["valid"]
    assert caught.value is publication_error
    assert caught.value.__cause__ is restore_error
    assert str(backups[0]) in " ".join(caught.value.__notes__)
    # Once the OS allows renames again, the retained package is recoverable.
    monkeypatch.setattr(reporting.os, "replace", original_replace)
    original_replace(backups[0], output)
    result = reporting.build_report(run, config=config, plan=plan)
    assert result["verification"]["valid"]
    assert not list(run.glob(".report-old-*"))


def test_failed_publish_restores_original_package_and_allows_retry(published_report, monkeypatch):
    run, config, plan, output, before = published_report
    original_replace = reporting.os.replace
    publication_error = PermissionError("publish temporarily blocked")

    def fail_publication(source, destination):
        if Path(source).name.startswith(".report-build-") and Path(destination) == output:
            raise publication_error
        return original_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", fail_publication)
    with pytest.raises(PermissionError) as caught:
        reporting.build_report(run, config=config, plan=plan)
    assert caught.value is publication_error
    assert _tree_bytes(output) == before
    assert reporting.verify_report(output)["valid"]
    monkeypatch.setattr(reporting.os, "replace", original_replace)
    assert reporting.build_report(run, config=config, plan=plan)["verification"]["valid"]


def test_cleanup_failure_does_not_hide_publication_error(published_report, monkeypatch):
    run, config, plan, output, before = published_report
    original_replace = reporting.os.replace
    original_rmtree = reporting.shutil.rmtree
    publication_error = PermissionError("publish temporarily blocked")

    def fail_publication(source, destination):
        if Path(source).name.startswith(".report-build-") and Path(destination) == output:
            raise publication_error
        return original_replace(source, destination)

    def fail_stage_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".report-build-"):
            raise PermissionError("staging cleanup blocked")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(reporting.os, "replace", fail_publication)
    monkeypatch.setattr(reporting.shutil, "rmtree", fail_stage_cleanup)
    with pytest.raises(PermissionError) as caught:
        reporting.build_report(run, config=config, plan=plan)
    assert caught.value is publication_error
    assert "staging cleanup blocked" in " ".join(caught.value.__notes__)
    assert _tree_bytes(output) == before
    assert reporting.verify_report(output)["valid"]
