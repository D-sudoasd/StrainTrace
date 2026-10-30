from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from straintrace_campaign import workflow
from straintrace_campaign.models import CampaignError, validate_project
from straintrace_campaign.storage import (
    acquire_project_lock,
    fingerprint_data,
    proposal_hash as shared_proposal_hash,
    sha256_file,
    strict_json_load,
    write_jsonl_atomic,
)


def _new_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    sample = source / "batch_01"
    sample.mkdir(parents=True)
    image = sample / "dic_000001.tif"
    image.write_bytes(b"synthetic-frame-one")
    project = tmp_path / "campaign"
    workflow.init_project(project, "synthetic", {"raw": source})
    return project, source, image


def _frame(image: Path, *, frame: int = 1, content: str | None = None) -> dict:
    return {
        "frame": frame,
        "native_frame": frame,
        "path": str(image.resolve()),
        "content_sha256": content or sha256_file(image),
        "frame_namespace": "dic_filename_counter",
        "identity_status": "NATIVE_COUNTER_PARSED",
    }


def _discovered(image: Path, *, measurement_frames: list[dict] | None = None) -> dict:
    return {
        "name": "batch_01",
        "relative_path": "batch_01",
        "source_kind": "generic",
        "dimensions": {"area_mm2": None, "width_mm": None, "thickness_mm": None, "status": "unknown", "source": None},
        "analysis_mode": "extensometer",
        "image_frames": [_frame(image)],
        "measurement_frames": measurement_frames or [],
        "metadata": {"discovery_issues": []},
    }


def _roi(proj: Path, image: Path) -> dict:
    return {
        "free_region": [0, 0, 100, 60],
        "exclusions": [],
        "groups": [{"name": "G01", "roi1": [5, 20, 10, 10], "roi2": [70, 20, 10, 10]}],
        "reference_frame": 1,
        "reference_path": str(image.resolve()),
        "reference_sha256": sha256_file(image),
        "image_shape": [60, 100],
        "diagnostics": {},
        "preview_paths": [],
    }


def _fake_roi_module():
    def approve(acq, reviewer, specimen_confirmed):
        return {
            "proposal_hash": acq["roi"]["proposal_hash"],
            "reference_sha256": acq["roi"]["reference_sha256"],
            "reviewer": reviewer,
            "approved_at": "2026-01-01T00:00:00Z",
            "specimen_confirmed": specimen_confirmed,
        }

    def validate_approval(acq):
        approval = acq.get("approval")
        if not approval or approval.get("proposal_hash") != acq.get("roi", {}).get("proposal_hash"):
            raise CampaignError("ROI_APPROVAL_STALE", "stale test approval")

    return SimpleNamespace(
        propose=lambda acq, root, free_region=None: _roi(root, Path(acq["image_frames"][0]["path"])),
        proposal_hash=shared_proposal_hash,
        validate_proposal=lambda acq, proposal, root: None,
        approve=approve,
        validate_approval=validate_approval,
    )


def _fake_numerics(root: Path, calls: list[str]):
    def source_images_fingerprint(acq):
        dependencies = []
        frames = sorted(acq["image_frames"], key=lambda item: (item["frame"], item["native_frame"], item["path"].casefold()))
        for frame in frames:
            dependencies.append({
                "frame": frame["frame"], "native_frame": frame["native_frame"], "path": str(Path(frame["path"]).resolve()),
                "sha256": sha256_file(frame["path"]), "counter_namespace": frame.get("counter_namespace"),
                "duplicate_identity": frame.get("duplicate_identity", frame.get("duplicate_of")),
            })
        return fingerprint_data("campaign_source_images_v1", dependencies)

    def fingerprint(acq):
        return fingerprint_data("fake_tracking", {
            "roi": workflow._roi_tracking_geometry(acq), "config": acq["analysis_config"],
            "source_images_fingerprint": source_images_fingerprint(acq),
        })

    def analyze(acq, project_dir, progress=None, cancel=None):
        calls.append(acq["acquisition_id"])
        out = root / "stages" / acq["acquisition_id"] / f"fake-{len(calls)}"
        out.mkdir(parents=True, exist_ok=True)
        rows_path = out / "rows.jsonl"
        group = acq["roi"]["groups"][0]
        shift = float(group["roi1"][0]) / 1000.0
        ordered_frames = list(acq["image_frames"])
        if len(calls) > 1:
            ordered_frames.reverse()
        rows = [{
            "frame": frame["frame"], "native_dic_frame": frame["native_frame"],
            "engineering_strain": shift + frame["frame"] / 100.0,
            "true_strain": shift + frame["frame"] / 200.0,
            "valid_gauge_count": 3, "strain_status": "TRACKED_MEAN",
        } for frame in ordered_frames]
        write_jsonl_atomic(rows_path, rows)
        rel = rows_path.relative_to(root).as_posix()
        reference = next(frame for frame in acq["image_frames"] if frame["frame"] == acq["roi"]["reference_frame"])
        return {
            "status": "COMPLETE",
            "fingerprint": fingerprint(acq),
            "rows_path": rel,
            "artifacts": [rel],
            "checksums": {rel: sha256_file(rows_path)},
            "provenance": {
                "engine": "synthetic-test", "analysis_mode": acq["analysis_mode"],
                "analysis_config": acq["analysis_config"], "engine_code_sha256": {"synthetic": "fixed"},
                "reference_frame": reference["frame"], "reference_native_dic_frame": reference["native_frame"],
                "reference_sha256": reference["content_sha256"],
                "reference_fingerprint": workflow._roi_reference_fingerprint(acq, acq["roi"]),
                "source_images_fingerprint": source_images_fingerprint(acq),
            },
            "summary": {"frame_count": 1},
        }

    return SimpleNamespace(fingerprint=fingerprint, analyze=analyze)


def test_project_strict_json_identity_and_atomic_lock(tmp_path: Path):
    project, _, _ = _new_project(tmp_path)
    assert workflow.load_project(project)["schema_version"] == 1
    invalid = workflow.load_project(project)
    invalid["metadata"]["bad"] = float("nan")
    with pytest.raises(CampaignError, match="non-finite"):
        validate_project(invalid)
    with acquire_project_lock(project):
        with pytest.raises(CampaignError, match="another writer") as exc:
            workflow.update_acquisition(project, "unknown", {"metadata": {"x": 1}})
    assert exc.value.code == "PROJECT_LOCKED"


def test_lock_initialization_failure_cleans_only_the_new_lock(tmp_path: Path, monkeypatch):
    import straintrace_campaign.storage as storage

    project, _, _ = _new_project(tmp_path)
    original_fsync = storage.os.fsync
    calls = 0

    def fail_once(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", fail_once)
    with pytest.raises(CampaignError) as exc:
        with acquire_project_lock(project):
            pytest.fail("lock body must not run when initialization failed")
    assert exc.value.code == "PROJECT_LOCK_ERROR"
    assert not (project / ".campaign.lock").exists()
    with acquire_project_lock(project):
        assert (project / ".campaign.lock").exists()


def test_scan_is_idempotent_and_only_invalidates_roi_when_image_identity_changes(tmp_path: Path, monkeypatch):
    project, source, image = _new_project(tmp_path)
    records = [_discovered(image)]
    adapter = SimpleNamespace(discover=lambda *args, **kwargs: records)
    monkeypatch.setattr(workflow, "_module", lambda name: adapter if name == "adapters" else None)
    first = workflow.scan(project, "raw")
    assert len(first) == 1
    acquisition_id = first[0]["acquisition_id"]
    assert workflow.scan(project, "raw") == []

    roi_module = _fake_roi_module()
    monkeypatch.setattr(workflow, "_module", lambda name: roi_module if name == "roi" else adapter)
    proposal = workflow.propose_rois(project, acquisition_id)
    workflow.approve_rois(project, acquisition_id, "reviewer", True)
    approved = workflow.load_project(project)["acquisitions"][0]["approval"]
    assert approved["proposal_hash"] == proposal["proposal_hash"]

    # Updating only an explicit measurement clock preserves image approval.
    records[:] = [_discovered(image, measurement_frames=[{
        "frame": 1, "native_cbf_frame": 8, "scan_id": 12, "scan_id_is_global_fio": True,
        "timestamp_unix_s": 1770000000.0, "timestamp_status": "RECORDED",
    }])]
    changed = workflow.scan(project, "raw")
    assert len(changed) == 1
    after_time_change = workflow.load_project(project)["acquisitions"][0]
    assert after_time_change["approval"] == approved
    assert after_time_change["measurement_frames"][0]["native_cbf_frame"] == 8

    # A content change updates the explicit inventory and clears approval.
    image.write_bytes(b"different-frame-bytes")
    records[:] = [_discovered(image)]
    updated = workflow.scan(project, "raw")
    assert len(updated) == 1
    after_image_change = workflow.load_project(project)["acquisitions"][0]
    assert after_image_change["approval"] is None
    assert after_image_change["roi"] is None
    assert after_image_change["acquisition_id"] == acquisition_id
    assert after_image_change["metadata"]["stale_roi_proposals"]


def test_run_requires_review_reuses_valid_stage_and_recovers_corrupt_stage(tmp_path: Path, monkeypatch):
    project, source, image = _new_project(tmp_path)
    adapter = SimpleNamespace(discover=lambda *args, **kwargs: [_discovered(image)])
    roi_module = _fake_roi_module()
    calls: list[str] = []
    numerics = _fake_numerics(project, calls)
    monkeypatch.setattr(workflow, "_module", lambda name: {
        "adapters": adapter, "roi": roi_module, "numerics": numerics,
    }[name])
    acq = workflow.scan(project, "raw")[0]
    with pytest.raises(CampaignError) as exc:
        workflow.run(project, acq["acquisition_id"])
    assert exc.value.code == "ROI_APPROVAL_REQUIRED"
    workflow.propose_rois(project, acq["acquisition_id"])
    workflow.approve_rois(project, acq["acquisition_id"], "test reviewer", True)
    first = workflow.run(project, acq["acquisition_id"])
    assert first["status"] == "COMPLETE"
    # Reviewer identity and approval time are audit metadata, not numerical dependencies.
    workflow.approve_rois(project, acq["acquisition_id"], "different reviewer", True)
    second = workflow.run(project, acq["acquisition_id"])
    assert second["fingerprint"] == first["fingerprint"]
    assert len(calls) == 1

    rows = project / first["rows_path"]
    rows.write_text('{"frame":999}\n', encoding="utf-8")
    third = workflow.run(project, acq["acquisition_id"])
    assert third["fingerprint"] == first["fingerprint"]
    assert len(calls) == 2


def test_roi_retry_archives_baseline_and_writes_explicit_frame_join_comparison(tmp_path: Path, monkeypatch):
    project, source, image = _new_project(tmp_path)
    second_image = image.parent / "dic_000002.tif"
    second_image.write_bytes(b"synthetic-frame-two")
    record = _discovered(image)
    record["image_frames"] = [_frame(image, frame=1), _frame(second_image, frame=2)]
    adapter = SimpleNamespace(discover=lambda *args, **kwargs: [record])
    roi_module = _fake_roi_module()
    calls: list[str] = []
    numerics = _fake_numerics(project, calls)
    monkeypatch.setattr(workflow, "_module", lambda name: {
        "adapters": adapter, "roi": roi_module, "numerics": numerics,
    }[name])

    acq = workflow.scan(project, "raw")[0]
    proposal = workflow.propose_rois(project, acq["acquisition_id"])
    workflow.approve_rois(project, acq["acquisition_id"], "reviewer A", True)
    baseline = workflow.run(project, acq["acquisition_id"])
    baseline_rows = project / baseline["rows_path"]
    baseline_bytes = baseline_rows.read_bytes()

    retry = json.loads(json.dumps(proposal))
    retry["groups"][0]["roi1"][0] += 4
    workflow.update_roi(project, acq["acquisition_id"], retry)
    after_edit = workflow.load_project(project)["acquisitions"][0]
    assert after_edit["result"]["fingerprint"] == baseline["fingerprint"]
    assert after_edit["metadata"]["pending_roi_sensitivity"]["baseline_result"]["fingerprint"] == baseline["fingerprint"]
    workflow.approve_rois(project, acq["acquisition_id"], "reviewer B", True)
    candidate = workflow.run(project, acq["acquisition_id"])

    current = workflow.load_project(project)["acquisitions"][0]
    assert current["result"]["fingerprint"] == candidate["fingerprint"]
    assert baseline_rows.read_bytes() == baseline_bytes
    archived = [item for item in current["metadata"]["workflow_stage_history"] if item["stage"] == "result"]
    assert any(item["record"]["fingerprint"] == baseline["fingerprint"] for item in archived)
    reports = current["metadata"]["roi_sensitivity_comparisons"]
    assert len(reports) == 1 and reports[0]["status"] == "COMPARABLE"
    report_path = project / reports[0]["json_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["join_key"] == ["frame", "native_dic_frame"]
    assert [row["frame"] for row in report["frames"]] == [1, 2]
    assert all(row["join_status"] == "MATCHED" for row in report["frames"])
    assert [row["delta_engineering_strain"] for row in report["frames"]] == pytest.approx([0.004, 0.004])
    assert reports[0]["csv_path"] in current["result"]["artifacts"]
    assert reports[0]["json_path"] in current["result"]["artifacts"]
    for rel, checksum in current["result"]["checksums"].items():
        assert sha256_file(project / rel) == checksum


def test_roi_sensitivity_requires_reference_provenance_from_baseline(tmp_path: Path, monkeypatch):
    project, source, image = _new_project(tmp_path)
    adapter = SimpleNamespace(discover=lambda *args, **kwargs: [_discovered(image)])
    roi_module = _fake_roi_module()
    calls: list[str] = []
    numerics = _fake_numerics(project, calls)
    monkeypatch.setattr(workflow, "_module", lambda name: {
        "adapters": adapter, "roi": roi_module, "numerics": numerics,
    }[name])
    acq = workflow.scan(project, "raw")[0]
    proposal = workflow.propose_rois(project, acq["acquisition_id"])
    workflow.approve_rois(project, acq["acquisition_id"], "reviewer", True)
    baseline = workflow.run(project, acq["acquisition_id"])
    project_data = workflow.load_project(project)
    result = project_data["acquisitions"][0]["result"]
    for key in ("reference_sha256", "reference_fingerprint", "reference_frame", "reference_native_dic_frame"):
        result["provenance"].pop(key, None)
    workflow._write_project(project, project_data)

    retry = json.loads(json.dumps(proposal))
    retry["groups"][0]["roi1"][0] += 3
    workflow.update_roi(project, acq["acquisition_id"], retry)
    workflow.approve_rois(project, acq["acquisition_id"], "reviewer", True)
    workflow.run(project, acq["acquisition_id"])
    current = workflow.load_project(project)["acquisitions"][0]
    report = json.loads((project / current["metadata"]["roi_sensitivity_comparisons"][0]["json_path"]).read_text(encoding="utf-8"))
    assert report["comparison_status"] == "BASELINE_REFERENCE_PROVENANCE_MISSING"
    assert report["differences"]["engineering_strain"]["count"] == 0
    assert all(row["delta_engineering_strain"] is None for row in report["frames"])
    assert any(item["record"]["fingerprint"] == baseline["fingerprint"] for item in current["metadata"]["workflow_stage_history"])


def test_roi_sensitivity_rejects_changed_nonreference_image_source(tmp_path: Path, monkeypatch):
    project, source, image = _new_project(tmp_path)
    second_image = image.parent / "dic_000002.tif"
    second_image.write_bytes(b"synthetic-frame-two-before")
    record = _discovered(image)
    record["image_frames"] = [_frame(image, frame=1), _frame(second_image, frame=2)]
    adapter = SimpleNamespace(discover=lambda *args, **kwargs: [record])
    roi_module = _fake_roi_module()
    calls: list[str] = []
    numerics = _fake_numerics(project, calls)
    monkeypatch.setattr(workflow, "_module", lambda name: {
        "adapters": adapter, "roi": roi_module, "numerics": numerics,
    }[name])
    acq = workflow.scan(project, "raw")[0]
    proposal = workflow.propose_rois(project, acq["acquisition_id"])
    workflow.approve_rois(project, acq["acquisition_id"], "reviewer", True)
    workflow.run(project, acq["acquisition_id"])

    second_image.write_bytes(b"synthetic-frame-two-after")
    retry = json.loads(json.dumps(proposal))
    retry["groups"][0]["roi1"][0] += 2
    workflow.update_roi(project, acq["acquisition_id"], retry)
    workflow.approve_rois(project, acq["acquisition_id"], "reviewer", True)
    workflow.run(project, acq["acquisition_id"])
    current = workflow.load_project(project)["acquisitions"][0]
    report = json.loads((project / current["metadata"]["roi_sensitivity_comparisons"][0]["json_path"]).read_text(encoding="utf-8"))
    assert report["comparison_status"] == "INPUT_IMAGES_MISMATCH"
    assert report["baseline"]["reference_fingerprint"] == report["candidate"]["reference_fingerprint"]
    assert report["baseline"]["source_images_fingerprint"] != report["candidate"]["source_images_fingerprint"]
    assert report["differences"]["engineering_strain"]["count"] == 0
    assert all(row["delta_engineering_strain"] is None for row in report["frames"])


def test_frozen_code_fingerprint_prefers_bundled_source_and_never_returns_none(tmp_path: Path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundled_source = bundle / "sources" / "straintrace_campaign" / "numerics.py"
    bundled_source.parent.mkdir(parents=True)
    bundled_source.write_text("# frozen campaign source\n", encoding="utf-8")
    module = SimpleNamespace(__name__="straintrace_campaign.numerics", __file__=str(tmp_path / "loader.pyc"))
    monkeypatch.setattr(workflow.sys, "frozen", True, raising=False)
    monkeypatch.setattr(workflow.sys, "_MEIPASS", str(bundle), raising=False)
    assert workflow._module_code_sha256(module) == sha256_file(bundled_source)
    bundled_source.unlink()
    with pytest.raises(CampaignError) as exc:
        workflow._module_code_sha256(module)
    assert exc.value.code == "CAMPAIGN_CODE_FINGERPRINT_UNAVAILABLE"


def test_cancel_event_and_immutable_imported_results(tmp_path: Path, monkeypatch):
    project, source, image = _new_project(tmp_path)
    adapter = SimpleNamespace(discover=lambda *args, **kwargs: [_discovered(image)])
    roi_module = _fake_roi_module()
    calls: list[str] = []
    numerics = _fake_numerics(project, calls)
    monkeypatch.setattr(workflow, "_module", lambda name: {
        "adapters": adapter, "roi": roi_module, "numerics": numerics,
    }[name])
    acq = workflow.scan(project, "raw")[0]
    workflow.propose_rois(project, acq["acquisition_id"])
    workflow.approve_rois(project, acq["acquisition_id"], "reviewer", True)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CampaignError) as exc:
        workflow.run(project, acq["acquisition_id"], cancel=cancel)
    assert exc.value.code == "CAMPAIGN_CANCELLED"

    # A tiny synthetic package pins import idempotency without touching a real
    # Keystone path or any original scientific data.
    version = source / "keystone" / "versions" / "synthetic_v1"
    version.mkdir(parents=True)
    (version / "catalog.json").write_text("[]", encoding="utf-8")
    (version / "CHECKSUMS.json").write_text("{}", encoding="utf-8")
    (source / "keystone" / "CURRENT.json").write_text(json.dumps({"version_relative_path": "versions/synthetic_v1"}), encoding="utf-8")
    imported_rows = {"frame": 1, "native_cbf_frame": 3, "stress_MPa": 10.0}
    delivered: list[str] = []

    def import_fake(project_root, source_path):
        delivered.append(str(source_path))
        rel = "imports/keystone/synthetic_v1/keystone-id/frames.jsonl"
        rows_file = Path(project_root) / rel
        rows_file.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(rows_file, [imported_rows])
        return [{
            "acquisition_id": "keystone-id", "name": "imported", "relative_path": "Grip/sample",
            "source_kind": "imported_keystone", "physical_specimen_id": None,
            "dimensions": {"area_mm2": 2.0, "width_mm": 1.0, "thickness_mm": 2.0, "status": "confirmed", "source": "synthetic record", "area_basis": "measured"},
            "analysis_mode": "extensometer", "image_frames": [],
            "measurement_frames": [{"frame": 1, "native_cbf_frame": 3, "scan_id": 9, "scan_id_is_global_fio": True}],
            "roi": None, "approval": None, "analysis_config": {},
            "result": {"status": "IMPORTED_ACCEPTED", "fingerprint": "a" * 64, "rows_path": rel, "artifacts": [rel], "checksums": {rel: sha256_file(rows_file)}, "provenance": {"origin": "synthetic"}, "summary": {"frame_count": 1}},
            "alignment": None, "metadata": {"keystone_import": {"source_version_id": "synthetic_v1"}},
        }]

    delivery = SimpleNamespace(import_keystone=import_fake)
    monkeypatch.setattr(workflow, "_module", lambda name: delivery if name == "delivery" else None)
    first = workflow.import_keystone(project, source / "keystone")
    assert first[0]["acquisition_id"] == "keystone-id"
    assert first[0]["result"]["status"] == "IMPORTED_ACCEPTED"
    imported_path = project / first[0]["result"]["rows_path"]
    original = imported_path.read_bytes()
    second = workflow.import_keystone(project, source / "keystone")
    assert second[0]["acquisition_id"] == "keystone-id"
    assert len(delivered) == 1
    workflow.update_acquisition(project, "keystone-id", {"analysis_config": {"search_radius_px": 5}})
    after_update = next(a for a in workflow.load_project(project)["acquisitions"] if a["acquisition_id"] == "keystone-id")
    assert after_update["result"]["status"] == "IMPORTED_ACCEPTED"
    assert after_update["metadata"]["workflow_reanalysis_required"] is True
    assert imported_path.read_bytes() == original
    with pytest.raises(CampaignError) as exc:
        workflow.update_acquisition(project, "keystone-id", {"metadata": {"keystone_import": {"source_version_id": "forged"}}})
    assert exc.value.code == "ACQUISITION_FIELD_IMMUTABLE"


def test_alignment_is_archived_on_area_or_roi_change_and_export_blocks_stale_tracking(tmp_path: Path, monkeypatch):
    project, source, image = _new_project(tmp_path)
    adapter = SimpleNamespace(discover=lambda *args, **kwargs: [_discovered(image)])
    roi_module = _fake_roi_module()
    calls: list[str] = []
    numerics = _fake_numerics(project, calls)
    captured_exports: list[dict] = []

    def align_fake(acq, project_root, load_path, **kwargs):
        out = Path(project_root) / "stages" / acq["acquisition_id"] / "fake-alignment"
        out.mkdir(parents=True, exist_ok=True)
        row_path = out / "alignment.jsonl"
        write_jsonl_atomic(row_path, [{"frame": 1, "force_N": 20.0, "stress_MPa": 10.0}])
        rel = row_path.relative_to(project_root).as_posix()
        return {
            "status": "COMPLETE", "fingerprint": "internal",
            "rows_path": rel, "artifacts": [rel], "checksums": {rel: sha256_file(row_path)},
            "provenance": {"mode": kwargs.get("mode", "instantaneous")}, "summary": {"frame_count": 1},
        }

    aligner = SimpleNamespace(align=align_fake)
    delivery = SimpleNamespace(export_project=lambda root, project_data, output, **kwargs: (
        captured_exports.append(json.loads(json.dumps(project_data))) or {"published": False}
    ))

    def modules(name):
        return {"adapters": adapter, "roi": roi_module, "numerics": numerics, "alignment": aligner, "delivery": delivery}[name]

    monkeypatch.setattr(workflow, "_module", modules)
    acq = workflow.scan(project, "raw")[0]
    workflow.propose_rois(project, acq["acquisition_id"])
    workflow.approve_rois(project, acq["acquisition_id"], "reviewer", True)
    workflow.run(project, acq["acquisition_id"])
    load = source / "force.csv"
    load.write_text("timestamp_unix_s,force_N\n1770000000,20\n", encoding="utf-8")
    workflow.align(project, acq["acquisition_id"], load, measurement_time_field="timestamp_unix_s", load_time_column="timestamp_unix_s")

    # A confirmed geometry update drops the prior stress alignment from the
    # active pointer while retaining its record and files in history.
    workflow.update_acquisition(project, acq["acquisition_id"], {
        "dimensions": {"area_mm2": 3.0, "width_mm": 1.5, "thickness_mm": 2.0, "status": "confirmed", "source": "synthetic measurement"},
    })
    snapshot = workflow.load_project(project)["acquisitions"][0]
    assert snapshot["alignment"] is None
    assert any(item["stage"] == "alignment" for item in snapshot["metadata"]["workflow_stage_history"])
    workflow.export(project, tmp_path / "package")
    exported = captured_exports[-1]["acquisitions"][0]
    assert exported["alignment"] is None
    assert exported["result"]["rows_path"] == snapshot["result"]["rows_path"]

    # Changing ROI after tracking makes the result inapplicable until rerun.
    proposal = _roi(project, image)
    proposal["groups"][0]["roi1"] = [8, 20, 10, 10]
    workflow.update_roi(project, acq["acquisition_id"], proposal)
    assert workflow.load_project(project)["acquisitions"][0]["alignment"] is None
    with pytest.raises(CampaignError) as exc:
        workflow.export(project, tmp_path / "stale-package")
    assert exc.value.code == "EXPORT_STALE_RESULT"


def test_current_pointer_variants_and_strict_jsonl(tmp_path: Path):
    root = tmp_path / "versions"
    version = root / "versions" / "v1"
    version.mkdir(parents=True)
    (version / "catalog.json").write_text("[]", encoding="utf-8")
    (version / "CHECKSUMS.json").write_text("{}", encoding="utf-8")
    current = {"version_relative_path": "versions/v1"}
    (root / "CURRENT.json").write_text(json.dumps(current), encoding="utf-8")
    assert workflow._keystone_source_info(root)[0] == "v1"
    assert workflow._keystone_source_info(root / "CURRENT.json")[0] == "v1"
    bad = tmp_path / "bad.json"
    bad.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(CampaignError):
        strict_json_load(bad)
    rows = tmp_path / "rows.jsonl"
    with pytest.raises(CampaignError):
        write_jsonl_atomic(rows, [{"strain": float("inf")}])
    assert not rows.exists()


@pytest.mark.parametrize("current", [
    {"version_relative_path": "versions/v1"},
    {"version_path": "versions/v1"},
    {"version_path": "versions\\v1"},
    {"version_relative_path": None, "version_path": "versions/v1"},
    {"version_relative_path": "unused/version", "version_path": "versions/v1"},
])
def test_keystone_preflight_matches_delivery_pointer_resolution(tmp_path: Path, current):
    from straintrace_campaign import delivery

    root = tmp_path / "keystone"
    version = root / "versions/v1"
    (version / "samples").mkdir(parents=True)
    (version / "catalog.json").write_text("[]", encoding="utf-8")
    checksums = version / "CHECKSUMS.json"
    checksums.write_text("{}", encoding="utf-8")
    pointer = root / "CURRENT.json"
    pointer.write_text(json.dumps(current), encoding="utf-8")
    for source in (root, pointer, version):
        resolved = delivery._resolve_keystone_version(source)
        assert resolved == version.resolve()
        assert workflow._keystone_source_info(source) == (resolved.name, sha256_file(checksums))


def test_keystone_explicit_version_keeps_same_identity_when_it_contains_pointer(tmp_path: Path):
    from straintrace_campaign import delivery

    version = tmp_path / "v1"
    (version / "samples").mkdir(parents=True)
    (version / "catalog.json").write_text("[]", encoding="utf-8")
    (version / "CURRENT.json").write_text(json.dumps({"version_path": "not-the-selected-version"}), encoding="utf-8")
    assert delivery._resolve_keystone_version(version) == version.resolve()
    assert workflow._keystone_source_info(version) == (version.name, sha256_file(version / "catalog.json"))


@pytest.mark.parametrize("relative", ["../outside", "..\\outside", "/outside", "C:\\outside"])
def test_keystone_current_rejects_escaping_version_paths(tmp_path: Path, relative):
    root = tmp_path / "keystone"
    root.mkdir()
    (root / "CURRENT.json").write_text(json.dumps({"version_path": relative}), encoding="utf-8")
    with pytest.raises(CampaignError) as exc:
        workflow._keystone_source_info(root)
    assert exc.value.code == "PROJECT_PATH_ESCAPE"
