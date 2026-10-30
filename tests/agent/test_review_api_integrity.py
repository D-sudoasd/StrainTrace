"""Derived operations preserve source evidence and spatial coordinate units."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import ezdic_core as core
from straintrace_agent import api, fields


def _project(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    reference = core.generate_synthetic_speckle(112, 112, seed=17)
    for native, stretch in ((1, 1.0), (2, 1.005), (4, 1.015)):
        frame = reference if native == 1 else core.warp_image_deformation_gradient(reference, [[stretch, 0], [0, 1]])
        assert cv2.imwrite(str(images / f"frame_{native:04d}.png"), np.clip(frame, 0, 255).astype(np.uint8))
    project = tmp_path / "project"
    api.init_project(project, images, time_basis="frame_index", calibration={
        "mm_per_px": 0.05, "origin_px": [0, 0], "origin_mm": [0, 0], "axis_sign": [1, 1],
    })
    api.plan_project(project)
    return project


@pytest.fixture
def complete_run(tmp_path):
    project = _project(tmp_path)
    result = api.run(project, make_plots=False)
    assert result["status"] == "complete"
    return project, Path(result["run_dir"])


def _digest_tree(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("operation", ["extract", "plot", "export"])
def test_derived_operation_does_not_reapprove_changed_summary(complete_run, operation):
    project, run_root = complete_run
    summary = pd.read_csv(run_root / "summary.csv")
    summary["max_abs_local_exx"] = 999.0
    summary.to_csv(run_root / "summary.csv", index=False)
    assert api.verify(project)["valid"] is False
    before = _digest_tree(project)

    with pytest.raises(api.AgentError) as error:
        if operation == "extract":
            api.extract(project, component="u", roi=[30, 80, 30, 80])
        else:
            getattr(api, operation)(project)

    assert error.value.code == "RUN_ARTIFACT_INVALID"
    assert any(issue.get("path") == "summary.csv" for issue in error.value.details)
    assert _digest_tree(project) == before
    assert api.verify(project)["valid"] is False


@pytest.mark.parametrize("damage", ["receipt", "checkpoint", "missing_receipt"])
def test_export_refuses_changed_checkpoint_evidence_before_writes(complete_run, damage):
    project, run_root = complete_run
    receipt = run_root / "checkpoints/frame_native_00000002.json"
    if damage == "receipt":
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        metadata["summary_row"]["max_abs_local_exx"] = 999.0
        receipt.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "checkpoint":
        with receipt.with_suffix(".npz").open("ab") as stream:
            stream.write(b"changed bytes")
    else:
        receipt.unlink()
    before = _digest_tree(project)
    with pytest.raises(api.AgentError) as error:
        api.export(project)
    assert error.value.code == "RUN_ARTIFACT_INVALID"
    assert _digest_tree(project) == before


def test_calibrated_path_figure_uses_the_same_points_as_csv(complete_run, monkeypatch):
    project, _ = complete_run
    observed = {}
    original_plot = fields.plot_position_time

    def capture_plot(*args, **kwargs):
        result = original_plot(*args, **kwargs)
        observed["xlabel"] = result[1].get_xlabel()
        observed["values"] = np.asarray(result[2].get_array()).copy()
        return result

    monkeypatch.setattr(fields, "plot_position_time", capture_plot)
    result = api.extract(project, component="u", path_points=[[2.0, 2.0], [3.2, 2.0]],
                         max_distance=0.35, coordinate_system="calibrated")
    table = pd.read_csv(result["paths_csv"])
    valid = table.loc[table["status"].eq("valid"), "u"].to_numpy()
    assert len(valid) == 6
    assert result["position_time_summary"]["sampled_values"] == len(valid)
    np.testing.assert_allclose(observed["values"], valid)
    assert observed["xlabel"] == "Position along path (mm)"
    assert api.verify(project)["valid"] is True


def test_cancelled_run_can_export_completed_evidence_and_pending_frame_positions(tmp_path):
    project = _project(tmp_path)

    def cancel_after_first_frame(event):
        if event.get("event") == "frame_complete" and event.get("native_frame") == 1:
            api.cancel(project)

    stopped = api.run(project, make_plots=False, progress=cancel_after_first_frame)
    assert stopped["status"] == "cancelled"
    exported = api.export(project)
    assert exported["status"] == "cancelled"
    assert exported["numerical_quality_passed"] is False
    table = pd.read_csv(exported["artifacts"]["summary"])
    assert table["frame_position_1based"].tolist() == [1, 2, 3]
    assert table["native_frame"].tolist() == [1, 2, 4]
    assert table["status"].tolist() == ["complete", "pending", "pending"]
