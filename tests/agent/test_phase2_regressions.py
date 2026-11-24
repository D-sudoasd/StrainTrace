"""Targeted Phase 2 root-cause regression tests for straintrace_agent."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytest

import ezdic_core as core
from straintrace_agent import api, discovery, fields, mechanics, regions, reporting


def _speckle_sequence(image_dir: Path, *, frames=(1, 2, 4), size=112, seed=17):
    image_dir.mkdir(parents=True, exist_ok=True)
    ref = core.generate_synthetic_speckle(size, size, seed=seed)
    for native in frames:
        stretch = 1.0 + (native - frames[0]) * 0.005
        img = ref if native == frames[0] else core.warp_image_deformation_gradient(ref, [[stretch, 0.0], [0.0, 1.0]])
        assert cv2.imwrite(str(image_dir / f"frame_{native:04d}.png"), np.clip(img, 0, 255).astype(np.uint8))


def _make_link_dir(link: Path, target: Path) -> None:
    if os.name == "nt":
        res = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True, check=False)
        assert res.returncode == 0, res.stdout + res.stderr
    else:
        link.symlink_to(target, target_is_directory=True)


# 1. P0 _safe_file ancestor junction/reparse point + 8.3/case + _MEIPASS fallback
def test_safe_file_and_report_succeed_under_ancestor_junction(tmp_path):
    real_base = tmp_path / "real_storage"
    real_base.mkdir()
    linked_base = tmp_path / "linked_onedrive_like"
    _make_link_dir(linked_base, real_base)

    images = linked_base / "images"
    _speckle_sequence(images, frames=(1, 2))
    project = linked_base / "project"
    api.init_project(project, images, frame_interval_s=0.5, step_hint_px=8)
    plan = api.plan_project(project)
    assert plan["status"] == "ready"
    res = api.run(project, make_plots=False)
    assert res["status"] == "complete"
    assert api.verify(project)["valid"] is True
    rep = api.report(project, include_checkpoints=True)
    assert rep["verification"]["valid"] is True
    assert reporting.verify_report(Path(res["run_dir"]) / "report")["valid"] is True


def test_reporting_no_cwd_fallback_when_meipass_unset(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "summary.csv").write_text("native_frame,time_s,status,quality_status\n1,0,complete,pass\n", encoding="utf-8")
    config = {"project_id": "p", "image_dir": str(tmp_path / "img"), "regions": [], "gauges": []}
    plan = {"configuration_sha256": reporting._json_sha256(config), "reference_native_frame": 1}
    plan["plan_sha256"] = reporting._json_sha256(plan)
    source = {
        "run_id": "run", "configuration_sha256": reporting._json_sha256(config),
        "plan_sha256": plan["plan_sha256"], "source_images": [], "checkpoint_receipts": {},
        "artifacts": {"summary.csv": hashlib.sha256((run / "summary.csv").read_bytes()).hexdigest()},
    }
    (run / "manifest.json").write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(reporting, "__file__", str(tmp_path / "nonexistent" / "reporting.py"))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    # Even if CWD happens to have sources/straintrace_agent/reporting.py, must NOT read from CWD
    fake_cwd = tmp_path / "fake_cwd"
    (fake_cwd / "sources" / "straintrace_agent").mkdir(parents=True)
    (fake_cwd / "sources" / "straintrace_agent" / "reporting.py").write_text("# rogue", encoding="utf-8")
    monkeypatch.chdir(fake_cwd)
    with pytest.raises(ValueError, match="unavailable"):
        reporting.build_report(run, config=config, plan=plan)


# 2. P0 roi_override_px fallback on low full-frame texture + single _texture_roi call
def test_plan_project_allows_explicit_roi_override_on_small_textured_patch(tmp_path):
    images = tmp_path / "small_specimen"
    images.mkdir()
    patch = np.clip(core.generate_synthetic_speckle(72, 72, seed=23), 0, 255).astype(np.uint8)
    # Embed 72x72 specimen (< 2% area even after 21px boxFilter halo) inside a 700x700 blank background
    ref = np.full((700, 700), 120, dtype=np.uint8)
    ref[300:372, 300:372] = patch
    deformed_patch = np.clip(core.warp_image_translation(patch.astype(float), 0.5, 0.0), 0, 255).astype(np.uint8)
    deformed = np.full((700, 700), 120, dtype=np.uint8)
    deformed[300:372, 300:372] = deformed_patch
    assert cv2.imwrite(str(images / "frame_0001.png"), ref)
    assert cv2.imwrite(str(images / "frame_0002.png"), deformed)

    no_override = tmp_path / "no-override"
    api.init_project(no_override, images, time_basis="frame_index")
    with pytest.raises(api.AgentError) as err:
        api.plan_project(no_override)
    assert err.value.code == "REFERENCE_FRAME_UNAVAILABLE"

    with_override = tmp_path / "with-override"
    api.init_project(with_override, images, time_basis="frame_index", roi_override_px=[304, 304, 64, 64], step_hint_px=8)
    plan = api.plan_project(with_override)
    assert plan["status"] == "ready", plan
    assert plan["roi_detection"]["selected_by"] == "explicit_user_roi_override"
    res = api.run(with_override, make_plots=False)
    assert res["status"] == "complete"


def test_plan_project_calls_texture_roi_once(tmp_path, monkeypatch):
    images = tmp_path / "images"
    _speckle_sequence(images, frames=(1, 2))
    project = tmp_path / "proj"
    api.init_project(project, images, time_basis="frame_index")
    calls = []
    real_texture = api._texture_roi

    def counting_texture(*args, **kwargs):
        calls.append(1)
        return real_texture(*args, **kwargs)

    monkeypatch.setattr(api, "_texture_roi", counting_texture)
    plan = api.plan_project(project)
    assert plan["status"] == "ready"
    assert len(calls) == 1


# 3. P0 _inventory_fresh preserves dimensions & _pilot_frame skips dimension mismatch
def test_inventory_metadata_preserved_across_plan_and_run_and_pilot_skips_mismatch(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    ref = np.clip(core.generate_synthetic_speckle(112, 112, seed=17), 0, 255).astype(np.uint8)
    wrong_shape = np.clip(core.generate_synthetic_speckle(80, 80, seed=19), 0, 255).astype(np.uint8)
    deformed = np.clip(core.warp_image_translation(ref.astype(float), 0.4, 0.0), 0, 255).astype(np.uint8)
    assert cv2.imwrite(str(images / "frame_0001.png"), ref)
    assert cv2.imwrite(str(images / "frame_0002.png"), wrong_shape)
    assert cv2.imwrite(str(images / "frame_0003.png"), deformed)

    project = tmp_path / "proj"
    api.init_project(project, images, frame_interval_s=0.5, step_hint_px=8)
    plan = api.plan_project(project)
    assert plan["status"] == "ready", plan
    assert plan["pilot_native_frame"] == 3  # skipped mismatched frame 2!

    inspected = api.inspect_project(project)
    assert inspected["image_shape_yx"] == [112, 112]
    assert inspected["unreadable_frames"] == [2]

    result = api.run(project, make_plots=False)
    assert result["status"] == "partial"
    summary = pd.read_csv(Path(result["run_dir"]) / "summary.csv")
    row2 = summary.loc[summary["native_frame"] == 2].iloc[0]
    assert row2["failure_code"] == "FRAME_DIMENSION_MISMATCH"
    assert api.inspect_project(project)["image_shape_yx"] == [112, 112]


# 4. P0 verify() provenance & integrity parity
def test_verify_catches_manifest_tampering_unsafe_paths_orphans_and_corrupt_json(tmp_path):
    images = tmp_path / "images"
    _speckle_sequence(images, frames=(1, 2))
    project = tmp_path / "proj"
    api.init_project(project, images, time_basis="frame_index")
    api.plan_project(project)
    res = api.run(project, make_plots=False)
    run_root = Path(res["run_dir"])
    assert api.verify(project)["valid"] is True

    # (a) Orphaned frame_native_*.npz
    orphan = run_root / "checkpoints" / "frame_native_00000099.npz"
    np.savez(orphan, u=np.array([1.0]))
    checked = api.verify(project)
    assert checked["valid"] is False
    assert any(i["code"] == "UNREGISTERED_FRAME_CHECKPOINT" for i in checked["issues"])
    orphan.unlink()

    # (b) Unsafe artifact path traversal in manifest
    manifest_path = run_root / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    tampered = json.loads(original_manifest.decode("utf-8"))
    tampered["artifacts"]["../escape.txt"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    checked = api.verify(project)
    assert checked["valid"] is False
    assert any(i["code"] == "ARTIFACT_PATH_UNSAFE" for i in checked["issues"])

    # (c) Tampered manifest provenance fields
    for field, expected_code in (
        ("run_fingerprint", "MANIFEST_RUN_FINGERPRINT_MISMATCH"),
        ("configuration_sha256", "CONFIG_FINGERPRINT_MISMATCH"),
        ("plan_sha256", "PLAN_FINGERPRINT_MISMATCH"),
        ("run_id", "RUN_ID_MISMATCH"),
    ):
        tampered = json.loads(original_manifest.decode("utf-8"))
        tampered[field] = "tampered-value"
        manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
        checked = api.verify(project)
        assert checked["valid"] is False
        assert any(i["code"] == expected_code for i in checked["issues"]), (field, checked["issues"])
    manifest_path.write_bytes(original_manifest)

    # (d) Corrupt run_state.json / inventory.json / receipt JSON returns valid=False (no uncaught AgentError)
    state_path = run_root / "run_state.json"
    original_state = state_path.read_bytes()
    state_path.write_text("{corrupt", encoding="utf-8")
    checked = api.verify(project)
    assert checked["valid"] is False
    assert any(i["code"] == "RUN_STATE_INVALID" for i in checked["issues"])
    state_path.write_bytes(original_state)

    inv_path = project / "inventory.json"
    original_inv = inv_path.read_bytes()
    inv_path.write_text("{corrupt", encoding="utf-8")
    checked = api.verify(project)
    assert checked["valid"] is False
    assert any(i["path"] == "inventory.json" for i in checked["issues"])
    inv_path.write_bytes(original_inv)

    receipt_path = run_root / "checkpoints" / "frame_native_00000001.json"
    original_receipt = receipt_path.read_bytes()
    receipt_path.write_text("{corrupt", encoding="utf-8")
    checked = api.verify(project)
    assert checked["valid"] is False
    assert any(i["code"] == "RECEIPT_INVALID" for i in checked["issues"])
    receipt_path.write_bytes(original_receipt)

    # (e) current_run is null preserves earlier PROJECT_INVENTORY_STALE / PLAN_MISSING issues
    ptr_path = project / "CURRENT.json"
    ptr = json.loads(ptr_path.read_text(encoding="utf-8"))
    ptr["current_plan"] = None
    ptr["current_run"] = None
    ptr_path.write_text(json.dumps(ptr), encoding="utf-8")
    checked = api.verify(project)
    codes = {i["code"] for i in checked["issues"]}
    assert checked["valid"] is False
    assert {"PLAN_MISSING", "RUN_MISSING"}.issubset(codes)


# 5. P1 auto_analyze converts REFERENCE_FRAME_UNAVAILABLE & IMAGE_DECODE_FAILED into review_required
def test_auto_analyze_returns_review_required_for_unavailable_or_corrupt_reference(tmp_path):
    blank_dir = tmp_path / "blank_exp" / "images"
    blank_dir.mkdir(parents=True)
    for idx in (1, 2):
        assert cv2.imwrite(str(blank_dir / f"frame_{idx:04d}.png"), np.full((112, 112), 100, dtype=np.uint8))
    res = api.auto_analyze(blank_dir.parent, tmp_path / "proj-blank", make_plots=False)
    assert res["status"] == "review_required"
    assert res["error"]["error_code"] == "REFERENCE_FRAME_UNAVAILABLE"

    corrupt_dir = tmp_path / "corrupt_exp" / "images"
    _speckle_sequence(corrupt_dir, frames=(2,))
    (corrupt_dir / "frame_0001.png").write_bytes(b"corrupt bytes")
    res2 = api.auto_analyze(corrupt_dir.parent, tmp_path / "proj-corrupt", options={"reference_native_frame": 1}, make_plots=False)
    assert res2["status"] == "review_required"
    assert res2["error"]["error_code"] == "IMAGE_DECODE_FAILED"


# 6. P1 export() on cancelled run reports honest completed_frame_count & native_frames_match_expected
def test_export_on_cancelled_run_reports_honest_coverage(tmp_path):
    images = tmp_path / "images"
    _speckle_sequence(images, frames=(1, 2, 4))
    project = tmp_path / "proj"
    api.init_project(project, images, time_basis="frame_index")
    api.plan_project(project)

    def cancel_after_one(event):
        if event.get("event") == "frame_complete" and event.get("native_frame") == 1:
            api.cancel(project)

    stopped = api.run(project, make_plots=False, progress=cancel_after_one)
    assert stopped["status"] == "cancelled"
    exported = api.export(project)
    run_root = Path(stopped["run_dir"])
    quality = json.loads((run_root / "quality.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(run_root / "summary.csv")
    assert summary["status"].tolist() == ["complete", "pending", "pending"]
    assert summary["reference_native_frame"].tolist() == [1, 1, 1]
    assert quality["completed_frame_count"] == manifest["completed_frame_count"] == 1
    assert quality["expected_frame_count"] == manifest["expected_frame_count"] == 3
    assert quality["native_frames_match_expected"] is False
    assert manifest["native_frames_match_expected"] is False
    assert quality["frame_count"] == len(quality["frames"]) == 1
    assert exported["numerical_quality_passed"] is False


# 7. P1 run() does not clobber CURRENT.json or strand run_state.json as "running"
def test_run_preserves_current_pointer_and_never_strands_running_state(tmp_path, monkeypatch):
    images = tmp_path / "images"
    _speckle_sequence(images, frames=(1, 2))
    project = tmp_path / "proj"
    api.init_project(project, images, time_basis="frame_index")
    api.plan_project(project)
    first = api.run(project, make_plots=False)
    assert first["status"] == "complete"
    prev_current = json.loads((project / "CURRENT.json").read_text(encoding="utf-8"))["current_run"]

    # Force a new plan that points to a failing mask check (or unreadable reference) before run starts
    api.update_project(project, {"step_hint_px": 9})
    api.plan_project(project)
    # Restore prev_current to simulate an existing valid current_run when a new run fails pre-validation
    ptr = json.loads((project / "CURRENT.json").read_text(encoding="utf-8"))
    ptr["current_run"] = prev_current
    (project / "CURRENT.json").write_text(json.dumps(ptr), encoding="utf-8")

    monkeypatch.setattr(api, "_load_image", lambda _p: (_ for _ in ()).throw(api.AgentError("IMAGE_DECODE_FAILED", "boom")))
    with pytest.raises(api.AgentError) as err:
        api.run(project, make_plots=False)
    assert err.value.code == "REFERENCE_FRAME_UNREADABLE"
    assert json.loads((project / "CURRENT.json").read_text(encoding="utf-8"))["current_run"] == prev_current

    # Mid-run fatal error marks run_state.json as failed and rolls back CURRENT.json
    monkeypatch.undo()
    monkeypatch.setattr(api, "_build_outputs", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError, match="disk full"):
        api.run(project, make_plots=False)
    assert json.loads((project / "CURRENT.json").read_text(encoding="utf-8"))["current_run"] == prev_current
    new_runs = [d for d in (project / "runs").iterdir() if d.name != Path(prev_current).name]
    assert len(new_runs) == 1
    state = json.loads((new_runs[0] / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"


# 8. P1 Lock race, 0-byte lock recovery & POSIX PID reuse
def test_status_and_recover_lock_handle_empty_lock_and_posix_pid_reuse(tmp_path, monkeypatch):
    images = tmp_path / "images"
    _speckle_sequence(images, frames=(1, 2))
    project = tmp_path / "proj"
    api.init_project(project, images, time_basis="frame_index")

    lock = project / ".agent.lock"
    lock.write_bytes(b"")
    st = api.status(project)
    assert st["locked"] is True
    assert st["lock_owner"]["corrupt"] is True

    recovered = api.recover_lock(project, expected_pid=99999999)
    assert recovered["recovered"] is True
    assert not lock.exists()
    assert api.status(project)["locked"] is False

    # POSIX /proc/{pid}/stat with spaces & parens in comm + PID reuse check
    stat_fields = ["S"] + [str(i) for i in range(4, 22)] + ["987654321", "0"]
    fake_stat = f"1234 (python (worker) 1) {' '.join(stat_fields)}\n"

    class _FakeProcPath:
        def __init__(self, text):
            self._text = str(text)

        def read_text(self, *args, **kwargs):
            if self._text == "/proc/1234/stat":
                return fake_stat
            raise FileNotFoundError(self._text)

    with monkeypatch.context() as mp:
        mp.setattr(api.os, "name", "posix")
        mp.setattr(api, "Path", _FakeProcPath)
        mp.setattr(api.os, "kill", lambda pid, sig: None)
        token = api._process_start_token(1234)
        same_alive = api._pid_state(1234, "987654321")
        reused_dead = api._pid_state(1234, "111111111")
    assert token == "987654321"
    assert same_alive is True
    assert reused_dead is False  # PID reused!


# 9. P1 16-bit image preview & float64 _texture_roi
def test_16bit_image_preview_and_float64_texture_roi(tmp_path):
    rng = np.random.default_rng(42)
    img16 = np.clip(40000 + rng.normal(0.0, 10.0, (128, 128)), 0, 65535).astype(np.uint16)
    roi, diag = api._texture_roi(img16, 21)
    assert roi[2] > 0 and roi[3] > 0
    assert diag["local_std_quantiles"]["q80"] > 5.0

    # 16-bit TIFF preview in reporting._preview must not clip to solid white 255
    tif_path = tmp_path / "img16.tif"
    grad16 = np.linspace(1000, 40000, 64 * 64, dtype=np.uint16).reshape(64, 64)
    Image.fromarray(grad16).save(tif_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    sha = hashlib.sha256(tif_path.read_bytes()).hexdigest()
    geom = {"reference_path": str(tif_path), "reference_native_frame": 1}
    warnings = reporting._preview(stage, geom, {"source_images": [{"native_frame": 1, "sha256": sha}]})
    assert not warnings
    preview = np.asarray(Image.open(stage / "assets" / "reference.png"))
    assert int(preview.min()) < 20 and int(preview.max()) > 235


# 10. P1 Windows CRLF self-verification with \r\n inside CSV cells
def test_report_self_verifies_with_crlf_inside_csv_cells_and_writes_lf(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    summary_path = run / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        import csv
        w = csv.DictWriter(stream, ["native_frame", "time_s", "status", "quality_status", "failure_reason"])
        w.writeheader()
        w.writerow({"native_frame": 1, "time_s": "0", "status": "failed", "quality_status": "fail", "failure_reason": "line1\r\nline2"})
    config = {"project_id": "crlf", "image_dir": str(tmp_path / "img"), "regions": [], "gauges": []}
    plan = {"configuration_sha256": reporting._json_sha256(config), "reference_native_frame": 1}
    plan["plan_sha256"] = reporting._json_sha256(plan)
    source = {
        "run_id": "run", "configuration_sha256": reporting._json_sha256(config),
        "plan_sha256": plan["plan_sha256"], "source_images": [], "checkpoint_receipts": {},
        "artifacts": {"summary.csv": hashlib.sha256(summary_path.read_bytes()).hexdigest()},
    }
    (run / "manifest.json").write_text(json.dumps(source), encoding="utf-8")
    receipt = reporting.build_report(run, config=config, plan=plan)
    assert receipt["verification"]["valid"] is True
    assert reporting.verify_report(run / "report")["valid"] is True
    assert b"\r\n" not in (run / "report" / "README.txt").read_bytes()
    assert b"\r\n" not in (run / "report" / "manifest.json").read_bytes()


# 11. P1 Grid / ROI / Config / Discovery / Mechanics
def test_fields_jacobian_support_on_1d_and_empty_grids_and_quadmesh_nan_guard():
    x1d = np.array([[10.0, 20.0, 30.0]])
    y1d = np.array([[15.0, 15.0, 15.0]])
    field_1x3 = {
        "X": x1d, "Y": y1d, "roi": (0, 0, 40, 30), "u": np.zeros_like(x1d), "v": np.zeros_like(x1d),
        "Exx": np.full_like(x1d, 0.01), "Eyy": np.zeros_like(x1d), "Exy": np.zeros_like(x1d),
        "valid": np.ones_like(x1d, dtype=bool), "strain_valid": np.ones_like(x1d, dtype=bool),
    }
    assert fields._has_jacobian_support(field_1x3) is False
    assert np.allclose(fields.component_values(field_1x3, "Exx"), 0.01)
    rows = regions.summarize_regions(field_1x3, [{"name": "r", "bounds": [5, 35, 10, 20], "min_valid_pois": 1}], ["Exx"])
    assert rows[0]["status"] == "valid" and rows[0]["mean"] == pytest.approx(0.01)

    # u_raw/v_raw-only 2x2 grid is recognized as having Jacobian support
    x2, y2 = np.meshgrid([0.0, 10.0], [0.0, 10.0])
    assert fields._has_jacobian_support({"X": x2, "Y": y2, "u_raw": np.zeros((2, 2)), "v_raw": np.zeros((2, 2))}) is True

    # quadmesh rejects NaN X/Y without leaking figures
    x_nan = x2.copy()
    x_nan[0, 0] = np.nan
    before_figs = len(plt.get_fignums())
    with pytest.raises(ValueError, match="finite"):
        fields.plot_field_map({"X": x_nan, "Y": y2, "u": np.ones((2, 2)), "valid": np.ones((2, 2), dtype=bool)}, "u", method="quadmesh")
    assert len(plt.get_fignums()) == before_figs


def test_safe_roi_no_negative_dims_and_plan_rejects_collapsed_roi(tmp_path):
    assert api._safe_roi_for_image((0, 0, 10, 10), (100, 100), search_radius_px=15)[2:] == (0, 0)
    images = tmp_path / "images"
    _speckle_sequence(images, frames=(1, 2), size=80)
    project = tmp_path / "proj"
    # ROI touches border and collapses after margin crop (margin=13 for subset=21, width=36 inside x=0..36 -> x0=13..36=23 ok; test 0..10 -> ROI_TOO_SMALL)
    api.init_project(project, images, time_basis="frame_index", roi_override_px=[0, 0, 36, 36], step_hint_px=4)
    plan = api.plan_project(project)
    assert all(v > 0 for v in plan["analysis_rectangle_candidate_px"][2:])


def test_config_validation_symmetry_and_discovery_readiness_and_junction_prune(tmp_path):
    images = tmp_path / "images"
    _speckle_sequence(images, frames=(1, 2))
    loads = tmp_path / "loads.csv"
    loads.write_text("Time,Force\n0,10\n1,20\n", encoding="utf-8")

    # Empty image_dir / folder rejected
    with pytest.raises(api.AgentError) as err:
        api.init_project(tmp_path / "p1", "", time_basis="frame_index")
    assert err.value.code == "CONFIG_TYPE_ERROR"
    with pytest.raises(api.AgentError) as err:
        discovery.discover_experiment("")
    assert err.value.code == "CONFIG_TYPE_ERROR"

    # Invalid configs rejected at init_project (matching _read_config)
    with pytest.raises(api.AgentError):
        api.init_project(tmp_path / "p2", images, frame_interval_s=0.5, loads_csv=loads, load_time_column="", load_time_unit="s", load_force_column="Force", force_unit="N", load_time_offset_s=0, max_gap_s=1)
    with pytest.raises(api.AgentError):
        api.init_project(tmp_path / "p3", images, frame_interval_s=0.5, loads_csv=loads, load_time_column="Time", load_time_unit="s", force_unit="N", displacement_column="Force", displacement_unit="mm", displacement_source="crosshead", load_time_offset_s=0, max_gap_s=1)
    with pytest.raises(api.AgentError):
        api.init_project(tmp_path / "p4", images, time_basis="frame_index", paths=[123])

    # update_project validates mask_file before hashing
    proj = tmp_path / "p5"
    api.init_project(proj, images, time_basis="frame_index")
    with pytest.raises(api.AgentError) as err:
        api.update_project(proj, {"mask_file": str(tmp_path / "missing_mask.png")})
    assert err.value.code == "MASK_FILE_MISSING"
    with pytest.raises(api.AgentError) as err:
        api.update_project(proj, {"mask_file": 123})
    assert err.value.code == "CONFIG_TYPE_ERROR"

    # discover_experiment must return needs_input (not ready) when units missing or time_basis conflicts
    exp = tmp_path / "exp"
    _speckle_sequence(exp / "seq", frames=(1, 2))
    d1 = discovery.discover_experiment(exp, options={"frame_interval_s": 0.5, "time_basis": "frame_index"})
    assert d1["status"] == "needs_input"
    d2 = discovery.discover_experiment(exp, options={"frame_interval_s": 0.5, "loads_csv": str(loads), "load_time_column": "Time", "load_force_column": "Force", "load_time_offset_s": 0, "max_gap_s": 1})
    assert d2["status"] == "needs_input"
    assert {"load_time_unit", "force_unit"}.issubset({item["field"] for item in d2["required_inputs"]})
    d3 = discovery.discover_experiment(exp, options={"frame_interval_s": 0.5, "loads_csv": str(loads), "load_time_column": "Time", "load_time_unit": "s", "load_time_offset_s": 0, "max_gap_s": 1})
    assert d3["status"] == "needs_input"

    # discovery.source_files prunes Windows junctions / symlinks
    external = tmp_path / "external_secret"
    _speckle_sequence(external, frames=(10, 11))
    _make_link_dir(exp / "linked_seq", external)
    found = list(discovery.source_files(exp))
    assert not any("linked_seq" in p.parts for p in found)


def test_mechanics_monotonicity_status_displacement_only_and_distance_gauge(tmp_path):
    # Non-monotonic timestamps rejected (not silently sorted) and overall status is unavailable
    bad_loads = tmp_path / "nonmonotonic.csv"
    bad_loads.write_text("time_s,force_N\n0,10\n2,30\n1,20\n", encoding="utf-8")
    cfg = {"loads": {
        "path": str(bad_loads), "time_column": "time_s", "time_unit": "s", "time_offset_s": 0.0,
        "max_gap_s": 1.0, "clock_confirmed": True, "force_column": "force_N", "force_unit": "N",
        "displacement_column": None, "displacement_unit": None, "displacement_source": None,
        "machine_gauge_length_mm": None, "area_mm2": 2.0,
    }}
    vals, st = mechanics._load_mechanics(cfg)
    assert vals == {}
    assert st["status"] == "unavailable"
    assert st["force_status"] == "duplicate_or_nonincreasing_times"

    # Displacement-only mechanics_status
    disp_loads = tmp_path / "disp_only.csv"
    disp_loads.write_text("time_s,displacement_mm\n0,0.0\n1,0.5\n", encoding="utf-8")
    disp_cfg = {"loads": {
        "path": str(disp_loads), "time_column": "time_s", "time_unit": "s", "time_offset_s": 0.0,
        "max_gap_s": 1.0, "clock_confirmed": True, "force_column": None, "force_unit": None,
        "displacement_column": "displacement_mm", "displacement_unit": "mm", "displacement_source": "crosshead",
        "machine_gauge_length_mm": 10.0, "area_mm2": None,
    }}
    dvals, dst = mechanics._load_mechanics(disp_cfg)
    assert dst["status"] == "aligned"
    at = mechanics._mechanics_at(0.5, dvals, disp_cfg, displacement_reference_mm=0.0)
    assert at["mechanics_status"] == "displacement_aligned"
    assert at["machine_displacement_mm"] == pytest.approx(0.25)

    # _gauge_group_summary supports distance and blocks mixed x + distance
    dist_only = mechanics._gauge_group_summary([
        {"axis": "distance", "status": "valid", "engineering_strain": 0.02, "true_strain": np.log1p(0.02)},
    ])
    assert dist_only["dic_virtual_gauge_engineering_strain"] == pytest.approx(0.02)
    assert dist_only["dic_virtual_gauge_engineering_strain_distance"] == pytest.approx(0.02)
    assert dist_only["dic_virtual_gauge_valid_count"] == 1

    mixed = mechanics._gauge_group_summary([
        {"axis": "x", "status": "valid", "engineering_strain": 0.01, "true_strain": np.log1p(0.01)},
        {"axis": "distance", "status": "valid", "engineering_strain": 0.02, "true_strain": np.log1p(0.02)},
    ])
    assert mixed["dic_virtual_gauge_engineering_strain"] is None
    assert mixed["dic_virtual_gauge_engineering_strain_x"] == pytest.approx(0.01)
    assert mixed["dic_virtual_gauge_engineering_strain_distance"] == pytest.approx(0.02)


# 12. P2 robustness: plan ID collision, orphan npz cleanup on failure, region candidates inside ROI, extract component validation, plot cleanup
def test_p2_robustness_plan_version_orphan_cleanup_regions_extract_and_plots(tmp_path):
    images = tmp_path / "images"
    _speckle_sequence(images, frames=(1, 2, 4))
    project = tmp_path / "proj"
    api.init_project(project, images, frame_interval_s=0.5, step_hint_px=8)
    api.plan_project(project)
    p2 = api.plan_project(project)
    assert (project / "plans" / "plan_v001.json").is_file()
    assert (project / "plans" / "plan_v002.json").is_file()
    # Delete v001; next plan must be v003 (not overwrite v002)
    (project / "plans" / "plan_v001.json").unlink()
    p3 = api.plan_project(project)
    assert (project / "plans" / "plan_v003.json").is_file()
    assert p3["plan_id"] != p2["plan_id"]

    # Every recommended region candidate must lie inside analysis_rectangle_candidate_px
    ax, ay, aw, ah = p3["analysis_rectangle_candidate_px"]
    for cand in p3["region_recommendations"]["candidates"]:
        bx0, bx1, by0, by1 = cand["bounds"]
        assert bx0 >= ax and bx1 <= ax + aw and by0 >= ay and by1 <= ay + ah

    res = api.run(project, make_plots=True)
    run_root = Path(res["run_dir"])
    assert api.verify(project)["valid"] is True

    # extract() validates component upfront
    with pytest.raises(api.AgentError) as err:
        api.extract(project, component="not_a_real_component", roi=[20, 80, 20, 80])
    assert err.value.code == "COMPONENT_INVALID"

    # _frame_failure removes stale orphaned .npz before writing failed .json
    orphan_npz = run_root / "checkpoints" / "frame_native_00000099.npz"
    np.savez(orphan_npz, u=np.array([1.0]))
    api._frame_failure(
        run_root, {"native_frame": 99, "time_s": 5.0, "path": "x", "sha256": "0" * 64},
        "fp", "in", {}, api.AgentError("FRAME_UNREADABLE", "bad"),
    )
    assert not orphan_npz.exists()
    (run_root / "checkpoints" / "frame_native_00000099.json").unlink()

    # Re-running plot with plot_every=2 removes stale frame_native_00000002_*.png and keeps report valid
    api.plot(project, plot_every=2)
    assert not list((run_root / "plots").glob("frame_native_00000002_*.png"))
    checked_after_plot = api.verify(project)
    assert checked_after_plot["valid"] is True, checked_after_plot["issues"]
    assert reporting.verify_report(run_root / "report")["valid"] is True
