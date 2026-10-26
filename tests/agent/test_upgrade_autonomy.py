"""Folder discovery, explicit clocks, validated revisions and saved-field reuse."""
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
import pytest

import ezdic_core as core
from straintrace_agent import api


def _sequence(directory):
    directory.mkdir(parents=True)
    reference = core.generate_synthetic_speckle(112, 112, seed=17)
    for native, stretch in ((1, 1.0), (2, 1.005), (4, 1.015)):
        frame = reference if native == 1 else core.warp_image_deformation_gradient(reference, [[stretch, 0], [0, 1]])
        assert cv2.imwrite(str(directory / f"frame_{native:04d}.png"), np.clip(frame, 0, 255).astype(np.uint8))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_discovery_is_read_only_prunes_outputs_and_uses_native_frame_axis(tmp_path, monkeypatch):
    experiment = tmp_path / "experiment"
    _sequence(experiment / "sample" / "DIC")
    _sequence(experiment / "outputs" / "old")
    _sequence(experiment / "prior-project")
    for name in ("project.json", "CURRENT.json"):
        (experiment / "prior-project" / name).write_text("{}")
    monkeypatch.setattr(api, "_sha256", lambda _: pytest.fail("discovery must not reread all image bytes"))
    result = api.discover_experiment(experiment)
    assert result["status"] == "ready"
    assert len(result["candidate_sequences"]) == 1
    assert result["candidate_sequences"][0]["missing_native_frame_ranges"] == [[3, 3]]
    assert result["init_options"]["time_basis"] == "frame_index"
    assert not (experiment / "project.json").exists()


def test_discovery_needs_specific_missing_conditions_and_rejects_unknown_options(tmp_path):
    _sequence(tmp_path / "a")
    _sequence(tmp_path / "b")
    discovery = api.discover_experiment(tmp_path, objective="find the best material")
    assert {entry["field"] for entry in discovery["required_inputs"]} >= {"objective", "image_dir"}
    result = api.auto_analyze(tmp_path, tmp_path / "out", objective="stress_strain", options={"image_dir": "a"})
    assert result["status"] == "needs_input"
    assert {entry["field"] for entry in result["discovery"]["required_inputs"]} >= {"gauges", "clock_confirmed", "frame_interval_s"}
    assert not (tmp_path / "out").exists()
    with pytest.raises(api.AgentError, match="unknown field"):
        api.discover_experiment(tmp_path, options={"invented_option": True})


def test_explicit_metadata_and_frame_map_are_used_without_inventing_time(tmp_path):
    images = tmp_path / "images"
    _sequence(images)
    mapping = tmp_path / "frame_map.csv"
    mapping.write_text("filename,native_frame,time_s\nframe_0001.png,1,0.1\nframe_0002.png,2,0.6\nframe_0004.png,4,1.6\n", encoding="utf-8")
    (tmp_path / "experiment.json").write_text(json.dumps({"schema": "straintrace.experiment/v1", "options": {"image_dir": "images"}}))
    result = api.discover_experiment(tmp_path)
    assert result["status"] == "ready"
    assert result["init_options"]["time_basis"] == "physical_time"
    assert result["init_options"]["frame_map_csv"] == str(mapping)
    assert "frame_interval_s" not in result["init_options"]


def test_init_hashes_each_image_once_and_retains_default_time_requirement(tmp_path, monkeypatch):
    images = tmp_path / "images"
    _sequence(images)
    real_hash = api._sha256
    seen = []
    def count(path):
        seen.append(path)
        return real_hash(path)
    monkeypatch.setattr(api, "_sha256", count)
    api.init_project(tmp_path / "project", images, frame_interval_s=0.5)
    assert all(seen.count(path) == 1 for path in images.glob("*.png"))
    with pytest.raises(api.AgentError) as missing:
        api.init_project(tmp_path / "missing-time", images)
    assert missing.value.code == "FRAME_TIME_REQUIRED"
    with pytest.raises(api.AgentError) as conflict:
        api.init_project(tmp_path / "conflict", images, frame_interval_s=1, time_basis="frame_index")
    assert conflict.value.code == "CONFIG_TIME_BASIS_CONFLICT"


def test_folder_to_report_multiaxis_gauges_reprocess_and_parameter_revision(tmp_path):
    experiment = tmp_path / "experiment"
    _sequence(experiment / "images")
    project = tmp_path / "project"
    gauges = [
        {"name": "horizontal", "axis": "x", "a_point_px": [35, 50], "b_point_px": [75, 50], "max_distance_px": 7},
        {"name": "vertical", "axis": "y", "a_point_px": [50, 35], "b_point_px": [50, 75], "max_distance_px": 7},
        {"name": "diagonal", "axis": "distance", "a_point_px": [35, 35], "b_point_px": [75, 75], "max_distance_px": 7},
    ]
    result = api.auto_analyze(experiment, project, objective="virtual_gauge", options={"gauges": gauges}, make_plots=False)
    assert result["status"] == "complete", result
    assert result["verification"]["valid"] is True
    assert result["report"]["verification"]["valid"] is True
    run_root = project / "runs" / result["run"]["run_id"]
    summary = pd.read_csv(run_root / "summary.csv")
    assert summary["time_s"].isna().all()
    assert summary["dic_virtual_gauge_engineering_strain"].isna().all()  # mixed directions are not averaged
    assert pd.read_csv(run_root / "gauges.csv")["status"].eq("valid").all()
    inventory = json.loads((project / "inventory.json").read_text())
    assert all(frame["time_s"] is None for frame in inventory["frames"])
    before = {path.name: _sha(path) for path in (run_root / "checkpoints").glob("*")}
    config_before = _sha(project / "project.json")
    regions = [{"name": "center", "bounds": [30, 75, 30, 75], "coordinate_system": "image"}]
    derived = api.reprocess(project, regions=regions, gauges=[gauges[-1]])
    assert derived["status"] == "complete", derived
    assert derived["solver_recomputed"] is False
    region_table = pd.read_csv(derived["artifacts"]["regions.csv"])
    assert len(region_table) == 9
    assert region_table["status"].eq("valid").all()
    with np.load(run_root / "checkpoints/frame_native_00000002.npz", allow_pickle=False) as saved:
        inside = (saved["X"] >= 30) & (saved["X"] <= 75) & (saved["Y"] >= 30) & (saved["Y"] <= 75)
        expected = saved["Exx"].ravel()[inside.ravel() & saved["strain_valid"].astype(bool).ravel()]
        assert region_table.loc[(region_table["native_frame"] == 2) & (region_table["component"] == "Exx"), "mean"].iloc[0] == pytest.approx(float(expected.mean()), abs=1e-14)
    assert _sha(project / "project.json") == config_before
    assert {path.name: _sha(path) for path in (run_root / "checkpoints").glob("*")} == before
    assert api.verify(project)["valid"] is True
    path_result = api.extract(project, component="u", path_points=[[40, 40], [64, 40]], max_distance=7)
    assert path_result["position_time_summary"]["axis_unit"] == "frame"
    with pytest.raises(api.AgentError):
        api.update_project(project, {"step_hint_px": -1})
    assert _sha(project / "project.json") == config_before
    revised = api.update_project(project, {"step_hint_px": 9})
    assert revised["status"] == "updated"
    assert api.status(project)["current_run"] is None
    repeated = api.auto_analyze(experiment, project, objective="virtual_gauge", make_plots=False)
    assert repeated["run"]["run_id"] != result["run"]["run_id"]
    assert run_root.is_dir()
    assert repeated["verification"]["valid"] is True


def test_folder_pipeline_preserves_corrupt_frame_failure(tmp_path):
    experiment = tmp_path / "experiment"
    _sequence(experiment / "images")
    (experiment / "images" / "frame_0005.png").write_bytes(b"not an image")
    result = api.auto_analyze(experiment, tmp_path / "project", make_plots=False)
    assert result["status"] == "attention_required"
    assert result["run"]["status"] == "partial"
    assert result["verification"]["valid"] is True
    summary = pd.read_csv(result["run"]["artifacts"]["summary"])
    failed = summary.loc[summary["native_frame"] == 5].iloc[0]
    assert failed["status"] == "failed"
    assert pd.isna(failed["max_abs_local_exx"])


def test_unavailable_reprocessing_support_is_preserved_in_portable_report(tmp_path):
    experiment = tmp_path / "experiment"
    _sequence(experiment / "images")
    project = tmp_path / "project"
    source = api.auto_analyze(experiment, project, make_plots=False)
    assert source["status"] == "complete"
    run_root = project / "runs" / source["run"]["run_id"]
    original_quality_sha = _sha(run_root / "quality.json")
    original_checkpoints = {path.name: _sha(path) for path in (run_root / "checkpoints").glob("*")}
    regions = [{"name": "outside", "bounds": [1000, 1100, 1000, 1100]}]
    gauges = [
        {"name": "supported", "axis": "x", "a_point_px": [35, 50], "b_point_px": [75, 50], "max_distance_px": 7},
        {"name": "unsupported", "axis": "x", "a_point_px": [1000, 50], "b_point_px": [1100, 50], "max_distance_px": 7},
    ]
    result = api.reprocess(project, regions=regions, gauges=gauges)
    assert result["status"] == "attention_required"
    quality = result["postprocessing_quality"]
    assert quality["status"] == "attention_required"
    assert quality["source_dic_numerical_quality_passed"] is True
    assert quality["source_run_id"] == run_root.name
    assert quality["region_measurements"] == {"total": 9, "valid": 0, "unavailable": 9}
    assert quality["gauge_measurements"] == {"total": 6, "valid": 3, "unavailable": 3}
    output = Path(result["output_dir"])
    for path in (output / "quality.json", output / "manifest.json", output / "report/data/quality.json"):
        recorded = json.loads(path.read_text(encoding="utf-8"))
        assert recorded["postprocessing_quality"] == quality
        assert recorded["numerical_quality_passed"] is True
    report_data = json.loads((output / "report/data/report_data.json").read_text(encoding="utf-8"))
    assert report_data["quality"]["postprocessing_quality"] == quality
    html = (output / "report/index.html").read_text(encoding="utf-8")
    assert "本次后处理" in html
    assert "attention_required" in html
    from straintrace_agent.reporting import verify_report
    relocated = tmp_path / "relocated-report"
    shutil.copytree(output / "report", relocated)
    assert verify_report(relocated)["valid"] is True
    assert json.loads((relocated / "data/quality.json").read_text(encoding="utf-8"))["postprocessing_quality"] == quality
    assert _sha(run_root / "quality.json") == original_quality_sha
    assert {path.name: _sha(path) for path in (run_root / "checkpoints").glob("*")} == original_checkpoints
    repeated = api.reprocess(project, regions=regions, gauges=gauges)
    assert repeated["status"] == "attention_required"
    assert repeated["postprocessing_quality"] == quality
    assert api.verify(project)["valid"] is True
