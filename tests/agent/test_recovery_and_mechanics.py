"""Failure continuation, resumability, and time/units gates for Agent runs."""

import json

import cv2
import numpy as np
import pandas as pd

import ezdic_core as core
from straintrace_agent import init_project, plan_project, run, verify
from straintrace_agent.api import _interpolate_bounded_detail


def _images(directory, *, native_frames=(1, 2, 4), bad_frames=()):
    directory.mkdir()
    reference = core.generate_synthetic_speckle(112, 112, seed=17)
    for native in sorted(set(native_frames) | set(bad_frames)):
        path = directory / f"frame_{native:04d}.png"
        if native in bad_frames:
            path.write_bytes(b"not an encoded image")
            continue
        stretch = 1.0 + (native - 1) * 0.005
        image = reference if native == 1 else core.warp_image_deformation_gradient(reference, [[stretch, 0], [0, 1]])
        assert cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))


def _init(project, image_dir, *, loads=None, clock_confirmed=False, gauges=None, max_gap_s=1.0):
    arguments = {"frame_interval_s": 0.5, "step_hint_px": 8, "gauges": gauges or []}
    if loads:
        arguments.update({
            "loads_csv": loads, "load_time_column": "time_s", "load_force_column": "force_N",
            "displacement_column": "displacement_mm", "displacement_source": "crosshead",
            "clock_confirmed": clock_confirmed, "load_time_offset_s": -0.5,
            "max_gap_s": max_gap_s, "area_mm2": 2, "machine_gauge_length_mm": 10,
        })
    return init_project(project, image_dir, **arguments)


def test_bad_frame_is_receipted_and_later_native_frame_runs(tmp_path):
    image_dir = tmp_path / "raw"
    _images(image_dir, native_frames=(1, 2, 4), bad_frames=(3,))
    project = tmp_path / "sample"
    _init(project, image_dir)
    assert plan_project(project)["status"] == "ready"

    result = run(project, make_plots=False)
    assert result["status"] == "partial"
    assert result["passed_frames"] == 3
    assert result["failed_frames"] == 1
    summary = pd.read_csv(project / "runs" / result["run_id"] / "summary.csv")
    assert summary["native_frame"].tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(summary["time_s"].to_numpy(), [0.0, 0.5, 1.0, 1.5])
    bad = summary.loc[summary["native_frame"] == 3].iloc[0]
    assert bad["failure_code"] in {"FRAME_UNREADABLE", "IMAGE_DECODE_FAILED"}
    assert summary.loc[summary["native_frame"] == 4, "status"].iloc[0] == "complete"
    assert verify(project)["valid"] is True
    assert verify(project)["numerical_quality_passed"] is False

    receipt = project / "runs" / result["run_id"] / "checkpoints" / "frame_native_00000004.json"
    receipt.unlink()
    damaged = verify(project)
    assert damaged["valid"] is False
    assert any(issue["code"] == "FRAME_RECEIPT_MISSING" and issue.get("native_frame") == 4 for issue in damaged["issues"])


def test_cancelled_run_resumes_matching_checkpoint(tmp_path):
    image_dir = tmp_path / "raw"
    _images(image_dir)
    project = tmp_path / "sample"
    _init(project, image_dir)
    plan_project(project)

    requested = False

    def cancel_after_first_frame(event):
        nonlocal requested
        if event.get("event") == "frame_complete" and event.get("native_frame") == 1 and not requested:
            from straintrace_agent import cancel
            cancel(project)
            requested = True

    interrupted = run(project, make_plots=False, progress=cancel_after_first_frame)
    assert interrupted["status"] == "cancelled"
    resumed_events = []
    resumed = run(project, make_plots=False, progress=resumed_events.append)
    assert resumed["status"] == "complete"
    assert any(event.get("event") == "frame_reused" and event.get("native_frame") == 1 for event in resumed_events)
    assert verify(project)["valid"] is True


def test_confirmed_mechanics_units_offset_and_unconfirmed_clock(tmp_path):
    image_dir = tmp_path / "raw"
    _images(image_dir)
    loads = tmp_path / "loads.csv"
    pd.DataFrame({
        "time_s": [0.0, 1.0, 2.0], "force_N": [100.0, 200.0, 400.0],
        "displacement_mm": [0.0, 0.05, 0.15],
    }).to_csv(loads, index=False)
    project = tmp_path / "confirmed"
    _init(project, image_dir, loads=loads, clock_confirmed=True)
    plan_project(project)
    result = run(project, make_plots=False)
    summary = pd.read_csv(project / "runs" / result["run_id"] / "summary.csv")
    np.testing.assert_allclose(summary["force_N"].to_numpy(), [150, 200, 400])
    np.testing.assert_allclose(summary["stress_MPa"].to_numpy(), [75, 100, 200])
    np.testing.assert_allclose(summary["machine_displacement_mm"].to_numpy(), [0, 0.025, 0.125])
    np.testing.assert_allclose(summary["machine_engineering_strain"].to_numpy(), [0, 0.0025, 0.0125])
    assert set(summary["machine_displacement_source"]) == {"crosshead"}
    assert summary["force_alignment_kind"].tolist() == ["bounded_linear_interpolation", "exact_sample", "exact_sample"]
    assert summary["displacement_alignment_kind"].tolist() == ["bounded_linear_interpolation", "exact_sample", "exact_sample"]
    np.testing.assert_allclose(summary["force_bracket_left_time_s"].to_numpy(), [-0.5, 0.5, 1.5])
    np.testing.assert_allclose(summary["force_bracket_right_time_s"].to_numpy(), [0.5, 0.5, 1.5])
    assert summary["force_bracket_left_source_row"].tolist() == [1, 2, 3]
    assert summary["force_bracket_right_source_row"].tolist() == [2, 2, 3]
    assert summary["displacement_reference_alignment_kind"].tolist() == ["bounded_linear_interpolation"] * 3
    config = json.loads((project / "project.json").read_text(encoding="utf-8"))
    assert config["loads"]["area_mm2"] == 2
    assert config["loads"]["time_offset_s"] == -0.5

    unconfirmed = tmp_path / "unconfirmed"
    _init(unconfirmed, image_dir, loads=loads, clock_confirmed=False)
    plan_project(unconfirmed)
    unconfirmed_run = run(unconfirmed, make_plots=False)
    unavailable = pd.read_csv(unconfirmed / "runs" / unconfirmed_run["run_id"] / "summary.csv")
    assert unavailable["force_N"].isna().all()
    assert unavailable["stress_MPa"].isna().all()
    assert unavailable["machine_engineering_strain"].isna().all()


def test_machine_alignment_marks_gap_rejection_without_extrapolation():
    times = [0.0, 1.0, 2.0]
    values = [100.0, 200.0, 400.0]
    source_rows = [4, 9, 12]
    exact = _interpolate_bounded_detail(0.0, times, values, 1.0, source_rows)
    interpolated = _interpolate_bounded_detail(0.5, times, values, 1.0, source_rows)
    too_wide = _interpolate_bounded_detail(0.5, times, values, 0.4, source_rows)
    outside = _interpolate_bounded_detail(2.1, times, values, 1.0, source_rows)

    assert exact == {"value": 100.0, "alignment_kind": "exact_sample", "left_time_s": 0.0, "right_time_s": 0.0, "left_source_row": 4, "right_source_row": 4}
    assert interpolated["value"] == 150.0
    assert interpolated["alignment_kind"] == "bounded_linear_interpolation"
    assert (interpolated["left_source_row"], interpolated["right_source_row"]) == (4, 9)
    assert too_wide["value"] is None and too_wide["alignment_kind"] == "gap_exceeds_max_gap"
    assert (too_wide["left_time_s"], too_wide["right_time_s"]) == (0.0, 1.0)
    assert outside["value"] is None and outside["alignment_kind"] == "outside_time_support"


def test_configured_virtual_gauges_drive_labeled_history_and_stress_curve(tmp_path):
    image_dir = tmp_path / "raw"
    _images(image_dir)
    loads = tmp_path / "loads.csv"
    pd.DataFrame({
        "time_s": [0.0, 1.0, 2.0], "force_N": [100.0, 200.0, 400.0],
        "displacement_mm": [0.0, 0.05, 0.15],
    }).to_csv(loads, index=False)
    gauges = [
        {"name": "axial_x", "axis": "x", "a_point_px": [40, 52], "b_point_px": [72, 52], "max_distance_px": 5},
        {"name": "transverse_y", "axis": "y", "a_point_px": [52, 40], "b_point_px": [52, 72], "max_distance_px": 5},
    ]
    project = tmp_path / "gauged"
    _init(project, image_dir, loads=loads, clock_confirmed=True, gauges=gauges)
    plan = plan_project(project)
    assert plan["status"] == "ready"
    assert all(item["reference_status"] == "resolved" for item in plan["gauge_reference_geometry"])
    geometry = {item["name"]: item for item in plan["gauge_reference_geometry"]}
    assert geometry["axial_x"]["endpoint_policy"] == "fixed_reference_poi_set_all_pois_must_be_valid"

    result = run(project, make_plots=True)
    run_root = project / "runs" / result["run_id"]
    summary = pd.read_csv(run_root / "summary.csv")
    assert summary["dic_virtual_gauge_engineering_strain_x"].notna().all()
    assert summary["dic_virtual_gauge_engineering_strain_y"].notna().all()
    assert summary["dic_virtual_gauge_engineering_strain"].isna().all()
    assert summary["engineering_strain"].isna().all()
    gauges_csv = pd.read_csv(run_root / "gauges.csv")
    assert gauges_csv.groupby("gauge_name")["initial_length_px"].nunique().to_dict() == {"axial_x": 1, "transverse_y": 1}
    plot_files = {path.name for path in (run_root / "plots").glob("*.png")}
    assert "time_force.png" in plot_files
    assert "time_machine_displacement.png" in plot_files
    assert "time_dic_virtual_gauge_engineering_strain.png" in plot_files
    assert "stress_vs_dic_virtual_gauge_engineering_strain.png" in plot_files
    history = json.loads((run_root / "plots" / "history_summary.json").read_text(encoding="utf-8"))
    assert all(item["status"] == "generated" for item in history["plots"])
    assert all(item["point_rendering"] == "scatter_no_interframe_lines" for item in history["plots"])
    report = (run_root / "report.md").read_text(encoding="utf-8")
    assert "stress_vs_dic_virtual_gauge_engineering_strain.png`" in report
