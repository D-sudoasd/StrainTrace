"""Invalid or interrupted extractions preserve the last verified delivery."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

import ezdic_core as core
from straintrace_agent import api, fields


def _digests(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def extracted_project(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    reference = core.generate_synthetic_speckle(112, 112, seed=17)
    for native, stretch in ((1, 1.0), (2, 1.005)):
        frame = reference if native == 1 else core.warp_image_deformation_gradient(reference, [[stretch, 0], [0, 1]])
        assert cv2.imwrite(str(images / f"frame_{native:04d}.png"), np.clip(frame, 0, 255).astype(np.uint8))
    project = tmp_path / "project"
    api.init_project(project, images, time_basis="frame_index")
    api.plan_project(project)
    result = api.run(project, make_plots=False)
    assert result["status"] == "complete"
    api.extract(project, component="u", roi=[30, 80, 30, 80],
                path_points=[[40, 40], [64, 40]], max_distance=7)
    assert api.verify(project)["valid"]
    return project


@pytest.mark.parametrize("path_options", [
    {"path_points": [[40, 40]]},
    {"path_points": [[40, 40]], "max_distance": -1},
    {"path_points": [[40]], "max_distance": 7},
    {"path_points": [], "max_distance": 7},
])
def test_invalid_combined_request_preserves_previous_roi_and_path(extracted_project, path_options):
    project = extracted_project
    before = _digests(project)
    # This ROI has fewer points than the saved one, making early replacement
    # observable even when the later path validation rejects the request.
    with pytest.raises((api.AgentError, ValueError)):
        api.extract(project, component="u", roi=[35, 75, 35, 75], **path_options)
    assert _digests(project) == before
    assert api.verify(project)["valid"]


def test_late_render_failure_preserves_previous_extractions(extracted_project, monkeypatch):
    project = extracted_project
    before = _digests(project)

    def fail_render(*args, **kwargs):
        raise ValueError("injected path rendering failure")

    with monkeypatch.context() as context:
        context.setattr(fields, "plot_position_time", fail_render)
        with pytest.raises(ValueError, match="injected path rendering failure"):
            api.extract(project, component="u", roi=[35, 75, 35, 75],
                        path_points=[[48, 48], [64, 64]], max_distance=7)
    assert _digests(project) == before
    assert api.verify(project)["valid"]


@pytest.mark.parametrize("after_manifest_write", [False, True])
def test_extraction_manifest_failure_rolls_back_files_and_manifest(extracted_project, monkeypatch, after_manifest_write):
    project = extracted_project
    before = _digests(project)
    original_refresh = api._refresh_manifest_artifacts

    def fail_refresh(run_root):
        if after_manifest_write:
            original_refresh(run_root)
        raise OSError("injected extraction manifest failure")

    with monkeypatch.context() as context:
        context.setattr(api, "_refresh_manifest_artifacts", fail_refresh)
        with pytest.raises(OSError, match="injected extraction manifest failure"):
            api.extract(project, component="u", roi=[35, 75, 35, 75],
                        path_points=[[48, 48], [64, 64]], max_distance=7)
    assert _digests(project) == before
    assert api.verify(project)["valid"]
    # Recovery is an ordinary retry, without rerunning DIC or discarding data.
    api.extract(project, component="u", roi=[35, 75, 35, 75],
                path_points=[[48, 48], [64, 64]], max_distance=7)
    assert api.verify(project)["valid"]


def test_export_rejects_unregistered_frame_array_before_writes(extracted_project):
    project = extracted_project
    run_root = Path(api.status(project)["project"]) / api.status(project)["current_run"]
    np.savez(run_root / "checkpoints/frame_native_00000099.npz", u=np.array([1.0]))
    before = _digests(project)
    with pytest.raises(api.AgentError) as error:
        api.export(project)
    assert error.value.code == "RUN_ARTIFACT_INVALID"
    assert any(issue.get("path") == "frame_native_00000099.npz" for issue in error.value.details)
    assert _digests(project) == before


def test_unrelated_private_arrays_are_excluded_from_checkpoint_count(extracted_project):
    project = extracted_project
    status = api.status(project)
    run_root = Path(status["project"]) / status["current_run"]
    private = run_root / "checkpoints/private_notes.npz"
    np.savez(private, notes=np.array([42]))
    before = private.read_bytes()
    exported = api.export(project)
    assert exported["status"] == "complete"
    assert api.verify(project)["valid"]
    assert private.read_bytes() == before
    assert not (run_root / "report/data/checkpoints/private_notes.npz").exists()
