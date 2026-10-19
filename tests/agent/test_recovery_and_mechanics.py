"""Failure continuation, resumability, and time/units gates for Agent runs."""

import json
import shutil

import cv2
import numpy as np
import pandas as pd
import pytest

import ezdic_core as core
from straintrace_agent import AgentError, export, extract, init_project, plan_project, run, verify
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
    assert interrupted["numerical_quality_passed"] is False
    run_root = project / "runs" / interrupted["run_id"]
    quality = json.loads((run_root / "quality.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert quality["run_status"] == manifest["run_status"] == "cancelled"
    assert quality["expected_frame_count"] == manifest["expected_frame_count"] == 3
    assert quality["completed_frame_count"] == manifest["completed_frame_count"] == 1
    assert quality["numerical_quality_passed"] is manifest["numerical_quality_passed"] is False
    assert verify(project)["numerical_quality_passed"] is False

    # A valid frame-1 payload placed under the frame-2 filenames must not be
    # reused as frame 2. Resume should retain frame 1 and recalculate frame 2.
    checkpoint_dir = run_root / "checkpoints"
    shutil.copyfile(checkpoint_dir / "frame_native_00000001.npz", checkpoint_dir / "frame_native_00000002.npz")
    shutil.copyfile(checkpoint_dir / "frame_native_00000001.json", checkpoint_dir / "frame_native_00000002.json")
    resumed_events = []
    resumed = run(project, make_plots=False, progress=resumed_events.append)
    assert resumed["status"] == "complete"
    assert any(event.get("event") == "frame_reused" and event.get("native_frame") == 1 for event in resumed_events)
    assert not any(event.get("event") == "frame_reused" and event.get("native_frame") == 2 for event in resumed_events)
    assert any(event.get("event") == "frame_complete" and event.get("native_frame") == 2 for event in resumed_events)
    frame_two = json.loads((checkpoint_dir / "frame_native_00000002.json").read_text(encoding="utf-8"))
    assert frame_two["native_frame"] == frame_two["summary_row"]["native_frame"] == 2
    assert verify(project)["valid"] is True
    assert verify(project)["numerical_quality_passed"] is True


def test_runtime_frame_failure_retries_on_resume_and_reuses_other_checkpoints(tmp_path, monkeypatch):
    image_dir = tmp_path / "raw"
    _images(image_dir)
    project = tmp_path / "sample"
    _init(project, image_dir)
    plan_project(project)

    original = core.run_2d_dic
    target = core.read_gray_image(str(image_dir / "frame_0002.png"))
    injected = False

    def fail_once_on_frame_two(reference, deformed, *args, **kwargs):
        nonlocal injected
        if not injected and np.array_equal(deformed, target):
            injected = True
            raise PermissionError("[WinError 5] transient checkpoint access")
        return original(reference, deformed, *args, **kwargs)

    monkeypatch.setattr(core, "run_2d_dic", fail_once_on_frame_two)
    first = run(project, make_plots=False)
    assert first["status"] == "partial"
    run_root = project / "runs" / first["run_id"]
    failed_receipt = json.loads((run_root / "checkpoints" / "frame_native_00000002.json").read_text(encoding="utf-8"))
    assert failed_receipt["status"] == "failed"
    assert failed_receipt["retryable"] is True
    assert failed_receipt["failure_class"] == "runtime"

    monkeypatch.setattr(core, "run_2d_dic", original)
    events = []
    resumed = run(project, make_plots=False, progress=events.append)
    assert resumed["status"] == "complete"
    assert {event["native_frame"] for event in events if event.get("event") == "frame_reused"} == {1, 4}
    assert any(event.get("event") == "frame_complete" and event.get("native_frame") == 2 for event in events)
    assert verify(project)["numerical_quality_passed"] is True


def test_review_required_plan_blocks_run_analyze_and_cli_before_full_frame_work(tmp_path, monkeypatch, capsys):
    image_dir = tmp_path / "low_texture"
    image_dir.mkdir()
    reference = core.generate_synthetic_speckle(112, 112, seed=31)
    frames = (
        reference,
        np.zeros((112, 112), dtype=np.uint8),
        core.warp_image_deformation_gradient(reference, [[1.005, 0], [0, 1]]),
    )
    for native, frame in enumerate(frames, 1):
        assert cv2.imwrite(str(image_dir / f"frame_{native:04d}.png"), np.asarray(frame, dtype=np.uint8))
    project = tmp_path / "low_texture_project"
    init_project(project, image_dir, frame_interval_s=0.5, step_hint_px=8)
    plan = plan_project(project)
    assert plan["status"] == "review_required"

    dic_calls = []

    def forbid_full_frame_dic(*_args, **_kwargs):
        dic_calls.append(True)
        raise AssertionError("full-frame DIC must not run for a review_required plan")

    monkeypatch.setattr(core, "run_2d_dic", forbid_full_frame_dic)
    with pytest.raises(AgentError) as error:
        run(project, make_plots=False)
    assert error.value.code == "PLAN_REVIEW_REQUIRED"
    assert error.value.details["plan_status"] == "review_required"
    with pytest.raises(AgentError) as error:
        from straintrace_agent import analyze
        analyze(project, make_plots=False)
    assert error.value.code == "PLAN_REVIEW_REQUIRED"

    from straintrace_agent.cli import EXIT_QUALITY, main
    assert main(["run", "--project", str(project), "--no-plots"]) == EXIT_QUALITY
    cli_error = json.loads(capsys.readouterr().out.strip())
    assert cli_error["error_code"] == "PLAN_REVIEW_REQUIRED"
    assert dic_calls == []
    assert not list((project / "runs").glob("run-*"))
    assert not list((project / "runs").rglob("*.npz"))


def test_export_and_extract_reject_changed_inputs_before_writing(tmp_path):
    image_dir = tmp_path / "raw"
    _images(image_dir)
    loads = tmp_path / "loads.csv"
    pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "force_N": [10.0, 20.0, 30.0]}).to_csv(loads, index=False)
    project = tmp_path / "load-project"
    _init(project, image_dir, loads=loads, clock_confirmed=True)
    plan_project(project)
    result = run(project, make_plots=False)
    run_root = project / "runs" / result["run_id"]
    summary_before = (run_root / "summary.csv").read_bytes()
    manifest_before = (run_root / "manifest.json").read_bytes()

    pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "force_N": [10.0, 20.0, 99.0]}).to_csv(loads, index=False)
    with pytest.raises(AgentError) as error:
        export(project)
    assert error.value.code == "RUN_STALE_INPUT"
    assert (run_root / "summary.csv").read_bytes() == summary_before
    assert (run_root / "manifest.json").read_bytes() == manifest_before

    calibrated_project = tmp_path / "calibrated-project"
    init_project(
        calibrated_project, image_dir, frame_interval_s=0.5, step_hint_px=8,
        calibration={"mm_per_px": 0.05, "origin_px": [0, 0], "origin_mm": [0, 0], "axis_sign": [1, 1]},
    )
    plan_project(calibrated_project)
    calibrated_run = run(calibrated_project, make_plots=False)
    calibrated_root = calibrated_project / "runs" / calibrated_run["run_id"]
    calibrated_summary_before = (calibrated_root / "summary.csv").read_bytes()
    calibrated_manifest_before = (calibrated_root / "manifest.json").read_bytes()
    config_path = calibrated_project / "project.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["calibration"]["mm_per_px"] = 0.06
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    with pytest.raises(AgentError) as error:
        extract(calibrated_project, roi=[20, 80, 20, 80], coordinate_system="calibrated")
    assert error.value.code == "RUN_STALE_CONFIG"
    assert not (calibrated_root / "extractions").exists()
    assert (calibrated_root / "summary.csv").read_bytes() == calibrated_summary_before
    assert (calibrated_root / "manifest.json").read_bytes() == calibrated_manifest_before


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
