"""Regression tests for automation provenance, cancellation, and lock safety."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import ezdic_core as core
from straintrace_agent import api
from straintrace_agent import _io, _locks, _outputs
from straintrace_agent._inventory import parse_frame_map


@pytest.fixture
def completed_project(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    reference = core.generate_synthetic_speckle(112, 112, seed=17)
    for native, stretch in ((1, 1.0), (2, 1.005)):
        image = reference if native == 1 else core.warp_image_deformation_gradient(
            reference, [[stretch, 0], [0, 1]],
        )
        assert cv2.imwrite(str(images / f"frame_{native:04d}.png"), np.clip(image, 0, 255).astype(np.uint8))
    project = tmp_path / "project"
    api.init_project(project, images, time_basis="frame_index")
    assert api.plan_project(project)["status"] == "ready"
    result = api.run(project, make_plots=False)
    assert result["status"] == "complete"
    assert api.verify(project)["valid"]
    return project, Path(result["run_dir"])


def test_rerun_recomputes_receipt_changed_after_manifest(completed_project):
    project, run_root = completed_project
    receipt_path = run_root / "checkpoints/frame_native_00000002.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["summary_row"]["max_abs_local_exx"] = 999.0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert not api.verify(project)["valid"]

    events = []
    result = api.run(project, make_plots=False, progress=events.append)
    assert result["status"] == "complete"
    assert any(e.get("event") == "frame_reused" and e.get("native_frame") == 1 for e in events)
    assert any(e.get("event") == "frame_complete" and e.get("native_frame") == 2 for e in events)
    assert pd.read_csv(run_root / "summary.csv").loc[1, "max_abs_local_exx"] != 999.0
    assert api.verify(project)["valid"]


@pytest.mark.parametrize("remove_file", [False, True])
def test_missing_core_manifest_entry_is_invalid_and_not_reapproved(completed_project, remove_file):
    project, run_root = completed_project
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["quality.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if remove_file:
        (run_root / "quality.json").unlink()

    checked = api.verify(project)
    assert not checked["valid"]
    assert any(issue.get("code") == "ARTIFACT_UNREGISTERED" and issue.get("path") == "quality.json" for issue in checked["issues"])
    before = manifest_path.read_bytes()
    with pytest.raises(api.AgentError) as error:
        api.export(project)
    assert error.value.code == "RUN_ARTIFACT_INVALID"
    assert manifest_path.read_bytes() == before


def test_cancellation_is_observed_before_reusing_checkpoints(completed_project):
    project, _ = completed_project

    def request_cancel(event):
        if event.get("event") == "run_started":
            api.cancel(project)

    result = api.run(project, make_plots=False, progress=request_cancel)
    assert result["status"] == "cancelled"
    assert result["completed_frames"] == 0


def test_cancel_does_not_reapprove_a_changed_unprocessed_receipt(completed_project):
    project, run_root = completed_project
    receipt_path = run_root / "checkpoints/frame_native_00000002.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["summary_row"]["max_abs_local_exx"] = 999.0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    def request_cancel(event):
        if event.get("event") == "run_started":
            api.cancel(project)

    result = api.run(project, make_plots=False, progress=request_cancel)
    assert result["status"] == "cancelled"
    assert not api.verify(project)["valid"]
    events = []
    api.run(project, make_plots=False, progress=events.append)
    assert any(e.get("event") == "frame_complete" and e.get("native_frame") == 2 for e in events)
    assert pd.read_csv(run_root / "summary.csv").loc[1, "max_abs_local_exx"] != 999.0
    assert api.verify(project)["valid"]


def test_rerun_does_not_reapprove_a_changed_extraction(completed_project):
    project, _ = completed_project
    extracted = api.extract(project, component="u", roi=[30, 80, 30, 80])
    path = Path(extracted["roi_csv"])
    path.write_text("tampered\n999\n", encoding="utf-8")
    assert not api.verify(project)["valid"]
    try:
        api.run(project, make_plots=False)
    except api.AgentError as exc:
        assert exc.code == "RUN_ARTIFACT_INVALID"
    assert not api.verify(project)["valid"]


def test_rerun_rebuilds_a_changed_requested_field_csv(completed_project):
    project, run_root = completed_project
    api.export(project, field_csv=True)
    csv_path = run_root / "fields_csv/frame_native_00000002.csv"
    original = csv_path.read_bytes()
    csv_path.write_text("tampered\n999\n", encoding="utf-8")
    assert not api.verify(project)["valid"]
    result = api.run(project, make_plots=False, field_csv=True)
    assert result["status"] == "complete"
    assert csv_path.read_bytes() == original
    assert api.verify(project)["valid"]


@pytest.mark.parametrize("field_csv", [False, True])
def test_failed_frame_removes_its_previous_field_csv(completed_project, monkeypatch, field_csv):
    project, run_root = completed_project
    api.export(project, field_csv=True)
    csv_path = run_root / "fields_csv/frame_native_00000002.csv"
    assert csv_path.is_file()
    receipt_path = run_root / "checkpoints/frame_native_00000002.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["summary_row"]["max_abs_local_exx"] = 999.0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    def fail_dic(*_args, **_kwargs):
        raise RuntimeError("injected frame failure")

    monkeypatch.setattr(core, "run_2d_dic", fail_dic)
    result = api.run(project, make_plots=False, field_csv=field_csv)
    assert result["status"] == "partial"
    assert not csv_path.exists()
    assert api.verify(project)["valid"]


def test_recomputed_failure_marks_previous_extraction_stale_until_reextracted(completed_project, monkeypatch):
    project, run_root = completed_project
    extracted = api.extract(project, component="u", roi=[30, 80, 30, 80])
    api.extract(project, component="v", roi=[30, 80, 30, 80])
    old_csv = Path(extracted["roi_csv"])
    assert set(pd.read_csv(old_csv)["native_frame"]) == {1, 2}
    receipt_path = run_root / "checkpoints/frame_native_00000002.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["summary_row"]["max_abs_local_exx"] = 999.0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    def fail_dic(*_args, **_kwargs):
        raise RuntimeError("injected frame failure")

    monkeypatch.setattr(core, "run_2d_dic", fail_dic)
    result = api.run(project, make_plots=False)
    assert result["status"] == "partial"
    assert old_csv.is_file()  # Existing work is retained, but cannot be silently certified.
    checked = api.verify(project)
    assert not checked["valid"]
    assert any(issue.get("code") == "DERIVED_ARTIFACT_STALE" for issue in checked["issues"])
    with pytest.raises(api.AgentError) as error:
        api.export(project)
    assert error.value.code == "RUN_ARTIFACT_INVALID"
    replacement = api.extract(project, component="u", roi=[30, 80, 30, 80])
    assert set(pd.read_csv(replacement["roi_csv"])["native_frame"]) == {1}
    assert not api.verify(project)["valid"]  # The v extraction still needs renewal.
    automatic = api.auto_analyze(project.parent / "images", project, make_plots=False)
    assert automatic["status"] == "attention_required"
    assert not automatic["verification"]["valid"]
    api.extract(project, component="v", roi=[30, 80, 30, 80])
    assert api.verify(project)["valid"]


def test_recomputed_failure_removes_a_previous_frame_map_even_if_plots_not_requested(completed_project, monkeypatch):
    project, run_root = completed_project
    old_plot = run_root / "plots/frame_native_00000002_Exx.png"
    old_plot.write_bytes(b"previous frame plot")
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["plots/frame_native_00000002_Exx.png"] = hashlib.sha256(old_plot.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path = run_root / "checkpoints/frame_native_00000002.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["summary_row"]["max_abs_local_exx"] = 999.0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    def fail_dic(*_args, **_kwargs):
        raise RuntimeError("injected frame failure")

    monkeypatch.setattr(core, "run_2d_dic", fail_dic)
    result = api.run(project, make_plots=False)
    assert result["status"] == "partial"
    assert not old_plot.exists()
    assert api.verify(project)["valid"]


def test_empty_lock_cannot_be_recovered_when_pid_state_is_unknown(tmp_path, monkeypatch):
    lock = tmp_path / ".agent.lock"
    lock.write_bytes(b"")
    monkeypatch.setattr(_locks, "_pid_state", lambda *_: None)
    with pytest.raises(api.AgentError) as error:
        api.recover_lock(tmp_path, expected_pid=123456)
    assert error.value.code == "LOCK_OWNER_UNKNOWN"
    assert lock.is_file()


def test_live_pid_is_recognized_without_a_start_token():
    assert _locks._pid_state(os.getpid(), None) is True


def test_unavailable_history_does_not_leave_a_stale_previous_plot(tmp_path):
    summary = tmp_path / "summary.csv"
    pd.DataFrame({"time_s": [0.0, 1.0], "force_N": [2.0, 3.0]}).to_csv(summary, index=False)
    _outputs._plot_standard_histories(tmp_path)
    plot = tmp_path / "plots/time_force.png"
    assert plot.is_file()
    pd.DataFrame({"time_s": [0.0, 1.0], "force_N": [None, None]}).to_csv(summary, index=False)
    result = _outputs._plot_standard_histories(tmp_path)
    assert next(item for item in result["plots"] if item["plot"] == "time_force.png")["status"] == "unavailable"
    assert not plot.exists()


@pytest.mark.parametrize("bad_path", ["wrong/frame_0001.png", "../frame_0001.png"])
def test_frame_map_does_not_replace_an_unknown_path_by_matching_basename(tmp_path, bad_path):
    images = tmp_path / "images"
    images.mkdir()
    image_paths = [images / "frame_0001.png", images / "frame_0002.png"]
    for path in image_paths:
        path.touch()
    mapping = tmp_path / "map.csv"
    mapping.write_text(f"path,native_frame\n{bad_path},1\nframe_0002.png,2\n", encoding="utf-8")
    with pytest.raises(api.AgentError) as error:
        parse_frame_map(mapping, images, image_paths)
    assert error.value.code == "FRAME_MAP_PATH_UNKNOWN"


@pytest.mark.skipif(os.name != "nt", reason="Windows rename provides atomic no-replace semantics")
def test_windows_lock_fallback_publishes_complete_payload_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(_locks.os, "link", lambda *_: (_ for _ in ()).throw(OSError("no hardlinks")))
    original_rename = _locks.os.rename
    observed = []

    def inspect_before_publish(source, destination):
        observed.append((json.loads(Path(source).read_text(encoding="utf-8")), Path(destination).exists()))
        return original_rename(source, destination)

    monkeypatch.setattr(_locks.os, "rename", inspect_before_publish)
    with _locks._project_lock(tmp_path, purpose="test"):
        assert json.loads((tmp_path / ".agent.lock").read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert len(observed) == 1
    assert observed[0][0]["pid"] == os.getpid()
    assert observed[0][1] is False


@pytest.mark.skipif(os.name != "nt", reason="Windows rename provides atomic no-replace semantics")
def test_windows_lock_fallback_does_not_replace_existing_owner(tmp_path, monkeypatch):
    lock = tmp_path / ".agent.lock"
    payload = {"pid": 42, "owner_token": "other"}
    lock.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(_locks.os, "link", lambda *_: (_ for _ in ()).throw(OSError("no hardlinks")))
    with pytest.raises(api.AgentError) as error:
        with _locks._project_lock(tmp_path, purpose="test"):
            pytest.fail("existing lock must not be replaced")
    assert error.value.code == "PROJECT_LOCKED"
    assert json.loads(lock.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("kind", ["plan", "run"])
def test_current_pointer_refuses_parent_directory_escape(tmp_path, kind):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "plan.json").write_text("{}", encoding="utf-8")
    pointer = {"current_plan": "../outside/plan.json", "current_run": "../outside"}
    (project / "CURRENT.json").write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(api.AgentError) as error:
        if kind == "plan":
            _io._plan_path(project, pointer)
        else:
            _io._load_run_pointer(project)
    assert error.value.code == ("PLAN_PATH_UNSAFE" if kind == "plan" else "RUN_PATH_UNSAFE")
