"""End-to-end automatic image-folder analysis on a deterministic affine sequence."""

import json

import cv2
import numpy as np

import ezdic_core as core
from straintrace_agent import extract, init_project, plan_project, run, verify


def _write_affine_sequence(image_dir):
    image_dir.mkdir()
    reference = core.generate_synthetic_speckle(112, 112, seed=17)
    frames = []
    for native_frame, stretch in ((1, 1.0), (2, 1.005), (4, 1.015)):
        image = reference if native_frame == 1 else core.warp_image_deformation_gradient(
            reference, [[stretch, 0.0], [0.0, 1.0]],
        )
        path = image_dir / f"frame_{native_frame:04d}.png"
        assert cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))
        frames.append(path)
    return frames


def test_auto_directory_to_traceable_outputs_without_gui(tmp_path):
    image_dir = tmp_path / "raw"
    _write_affine_sequence(image_dir)
    project = tmp_path / "sample-project"

    initialized = init_project(project, image_dir, frame_interval_s=0.5, step_hint_px=8)
    assert initialized["frame_count"] == 3
    plan = plan_project(project)
    assert plan["status"] == "ready"
    assert plan["reference_native_frame"] == 1
    assert plan["specimen_geometry_confirmed"] is False
    assert plan["selected_parameters"]["subset_size_px"] in {21, 31}

    result = run(project, field_csv=True, make_plots=True)
    run_root = project / "runs" / result["run_id"]
    assert result["status"] == "complete"
    assert result["numerical_quality_passed"] is True
    assert result["scientific_acceptance"] == "not_assessed"
    assert result["experimental_uncertainty_calibrated"] is False

    summary = __import__("pandas").read_csv(run_root / "summary.csv")
    assert summary["native_frame"].tolist() == [1, 2, 4]
    np.testing.assert_allclose(summary["time_s"].to_numpy(), [0.0, 0.5, 1.5])
    assert summary["scientific_ok"].tolist() == [True, True, True]
    assert summary["engineering_strain"].isna().all()  # no physical gauge/calibration was invented
    assert len(list((run_root / "fields_csv").glob("*.csv"))) == 3

    plots = [path.name for path in (run_root / "plots").glob("*.png")]
    assert len(plots) == 27  # 3 native frames × 9 core components
    assert len({name.casefold() for name in plots}) == 27
    assert (run_root / "plots" / "frame_native_00000002_Exx.png").is_file()
    assert (run_root / "plots" / "frame_native_00000002_exx_infinitesimal.png").is_file()
    history = json.loads((run_root / "plots" / "history_summary.json").read_text(encoding="utf-8"))
    history_by_name = {item["plot"]: item for item in history["plots"]}
    assert history_by_name["time_force.png"]["status"] == "unavailable"
    assert "confirmed clock" in history_by_name["time_force.png"]["reason"]
    assert "no full-field mean" in history_by_name["time_dic_virtual_gauge_engineering_strain.png"]["reason"]
    report = (run_root / "report.md").read_text(encoding="utf-8")
    assert "`time_force.png`: unavailable" in report
    assert "Scientific acceptance: not assessed" in report
    history = json.loads((run_root / "plots" / "history_summary.json").read_text(encoding="utf-8"))
    history_by_name = {item["plot"]: item for item in history["plots"]}
    assert history_by_name["time_force.png"]["status"] == "unavailable"
    assert "confirmed clock" in history_by_name["time_force.png"]["reason"]
    assert "no full-field mean" in history_by_name["time_dic_virtual_gauge_engineering_strain.png"]["reason"]
    report = (run_root / "report.md").read_text(encoding="utf-8")
    assert "`time_force.png`: unavailable" in report
    assert "Scientific acceptance: not assessed" in report

    roi = [20, 80, 20, 80]
    gl = extract(project, component="Exx", roi=roi)
    small = extract(project, component="exx", roi=roi)
    assert gl["roi_csv"] != small["roi_csv"]
    assert gl["roi_summary"] != small["roi_summary"]
    assert json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))["scientific_acceptance"] == "not_assessed"

    checked = verify(project)
    assert checked["valid"] is True, checked["issues"]
    assert checked["numerical_quality_passed"] is True
    assert checked["scientific_acceptance"] == "not_assessed"
