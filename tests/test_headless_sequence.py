"""Real image-file E2E checks for the GUI-independent sequence engines."""

from pathlib import Path
import json
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

import dic_virtual_extensometer_gui_v7_multi_roi_range as gui
import ezdic_core as core


@pytest.fixture(autouse=True)
def _isolated_operation_state(tmp_path, monkeypatch):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "operation-state"))


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", np.asarray(image, dtype=np.uint8))
    assert ok
    encoded.tofile(str(path))


def _sequence(tmp_path: Path, *, n=3):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    reference = core.generate_synthetic_speckle(128, 128, seed=17)
    paths = []
    for index in range(n):
        image = reference if index == 0 else core.warp_image_translation(reference, 0.8 * index, -0.25 * index)
        path = image_dir / f"frame_{index + 1:03d}.png"
        _write_png(path, image)
        paths.append(path)
    return paths


def test_collect_images_tie_order_is_stable_across_calls_and_processes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("frame_2.png", "frame_02.png", "frame_10.png", "frame_1.png"):
        _write_png(image_dir / name, np.full((8, 8), 120, dtype=np.uint8))
    expected = [str(path) for path in core.collect_images(image_dir)]
    assert expected == [str(path) for path in core.collect_images(image_dir)]
    script = "import json,ezdic_core; print(json.dumps(ezdic_core.collect_images(r'" + str(image_dir) + "')))"
    observed = []
    for _ in range(3):
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        observed.append(json.loads(completed.stdout))
    assert all(item == expected for item in observed)
    assert [Path(path).name for path in expected] == ["frame_1.png", "frame_02.png", "frame_2.png", "frame_10.png"]


def test_collect_images_includes_uppercase_suffixes_and_matches_gui(tmp_path: Path) -> None:
    image_dir = tmp_path / "mixed_case"
    image_dir.mkdir()
    _write_png(image_dir / "frame_1.PNG", np.full((8, 8), 110, dtype=np.uint8))
    _write_png(image_dir / "frame_2.jpg", np.full((8, 8), 130, dtype=np.uint8))
    (image_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    names = [Path(path).name for path in core.collect_images(image_dir)]
    assert names == ["frame_1.PNG", "frame_2.jpg"]
    assert gui.collect_images is core.collect_images
    assert [Path(path).name for path in gui.collect_images(image_dir)] == names


def test_write_corr_plot_alias_enables_correlation_plot_export(tmp_path: Path) -> None:
    options = core._extensometer_options({"export": {"write_corr_plot": True}})
    assert options["corr_plot"] is True
    paths = _sequence(tmp_path)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 3,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out_corr"),
        "roi_groups": [{"name": "G01", "roi1": (20, 45, 21, 21), "roi2": (85, 45, 21, 21), "strain_mode": "x"}],
        "enable_fb_check": False,
        "quality": {"enable_fb_check": False, "min_valid_frames": 1, "min_strain_valid_ratio": 0.0},
        "export": {"write_manifest": True, "write_qc": True, "write_full_csv": True, "write_corr_plot": True},
    }
    result = core.run_extensometer_sequence(settings)
    assert result["scientific_ok"] is True
    corr_dir = tmp_path / "out_corr" / "optional" / "correlation_plots"
    assert list(corr_dir.glob("correlation_scores_*.png"))


def test_extensometer_real_files_publish_and_verify(tmp_path: Path) -> None:
    paths = _sequence(tmp_path)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 3,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "roi_groups": [{"name": "G01", "roi1": (20, 45, 21, 21), "roi2": (85, 45, 21, 21), "strain_mode": "x"}],
        "enable_fb_check": False,
        "quality": {"enable_fb_check": False, "min_valid_frames": 1, "min_strain_valid_ratio": 0.0},
        "export": {"write_manifest": True, "write_qc": True, "write_full_csv": True, "write_origin_txt": True},
    }
    result = core.run_extensometer_sequence(settings)
    manifest_path = Path(result["manifest_path"])
    assert result["status"] == "completed"
    assert result["scientific_ok"] is True
    assert manifest_path == tmp_path / "out" / "run_manifest.json"
    assert (tmp_path / "out" / "core" / "strain_G01.txt").is_file()
    assert (tmp_path / "out" / "qc" / "qc_summary.txt").is_file()
    assert (tmp_path / "out" / "optional" / "full_csv" / "strain_results_all_groups.csv").is_file()
    assert all("\\" not in entry["path"] and not Path(entry["path"]).is_absolute() for entry in result["manifest"]["outputs"])
    output_paths = {entry["path"] for entry in result["manifest"]["outputs"]}
    required_paths = set(result["manifest"]["required_output_paths"])
    assert output_paths == required_paths
    assert {
        "core/strain_G01.txt",
        "core/strain_all_groups.txt",
        "core/strain_mean_groups.txt",
        "qc/qc_summary.txt",
        "optional/full_csv/strain_results_all_groups.csv",
        "optional/full_csv/per_group_results/strain_results_G01.csv",
    } <= output_paths
    assert all(entry["size"] > 0 and len(entry["sha256"]) == 64 for entry in result["manifest"]["outputs"])
    assert core.verify_run_manifest(manifest_path)["ok"] is True
    assert result["dataframe"]["strain_valid"].dtype == bool
    assert result["json_summary"]["manifest_path"] == str(manifest_path)
    gate_group = result["manifest"]["scientific_gate"]["groups"]["G01"]
    texture = result["manifest"]["texture_preflight"]
    assert texture["version"] == core.TEXTURE_PREFLIGHT_VERSION
    assert texture["metrics_version"] == core.TEXTURE_METRICS_VERSION
    assert texture["discriminator_version"] == core.TEXTURE_DISCRIMINATOR_VERSION
    assert texture["effective_thresholds"]["min_structure_ratio"] == core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO
    assert texture["groups"]["G01"]["roi1"]["structure_tensor_ratio"] == pytest.approx(
        result["dataframe"].iloc[0]["texture_structure_ratio_roi1"]
    )
    assert texture["groups"]["G01"]["roi2"]["rank_one_ratio"] == pytest.approx(
        result["dataframe"].iloc[0]["texture_rank_ratio_roi2"]
    )
    assert result["dataframe"].iloc[0]["texture_metrics_version"] == core.TEXTURE_METRICS_VERSION
    assert result["dataframe"].iloc[0]["texture_min_periodicity_score"] == pytest.approx(
        core.DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE
    )
    non_reference = result["dataframe"][result["dataframe"]["frame_global_1based"] > 1]
    assert gate_group["valid_frames"] == 2
    assert gate_group["strain_valid_frames"] == 2
    assert non_reference["strain_valid"].all()
    assert np.allclose(
        non_reference["used_roi1_center_x_px"].to_numpy() - result["dataframe"].iloc[0]["used_roi1_center_x_px"],
        [0.8, 1.6],
        atol=0.4,
    )
    assert float(non_reference["engineering_strain"].abs().max()) < 0.02


def test_fullfield_real_files_publish_pyramid_and_verify(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, n=2)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "field_roi": (20, 20, 88, 88),
        "solver": {"name": "IC-GN", "subset_size_px": 15, "step_px": 12, "strain_window_px": 5, "search_radius_px": 12},
        "pyramid": {"levels": 2, "scale": 0.5},
        "quality": {"zncc_min": 0.65, "min_correlation_valid_fraction": 0.0, "min_strain_valid_fraction": 0.0},
        "export": {"write_manifest": True, "write_overlays": False},
    }
    result = core.run_fullfield_sequence(settings)
    manifest_path = Path(result["manifest_path"])
    assert result["status"] == "completed"
    assert result["scientific_ok"] is True
    assert result["fields"]
    assert result["frames"][0]["status"] == "scientific_valid"
    assert (tmp_path / "out" / "dic" / "frame_0002.txt").is_file()
    assert (tmp_path / "out" / "dic" / "frame_0002.csv").is_file()
    assert result["manifest"]["outputs"]
    assert set(result["manifest"]["required_output_paths"]) == {entry["path"] for entry in result["manifest"]["outputs"]}
    assert {
        "dic/frame_0002.txt",
        "dic/frame_0002.csv",
        "dic/frame_0002_parameters.txt",
        "dic/frame_0002_u.png",
        "dic/frame_0002_v.png",
        "dic/frame_0002_Exx.png",
        "dic/frame_0002_Eyy.png",
        "dic/frame_0002_Exy.png",
    } <= {entry["path"] for entry in result["manifest"]["outputs"]}
    assert all(entry["size"] > 0 and len(entry["sha256"]) == 64 for entry in result["manifest"]["outputs"])
    assert result["manifest"]["scientific_gate"]["valid_frame_count"] == 1
    assert result["manifest"]["frames"][0]["correlation_valid_fraction"] >= 0.75
    assert result["manifest"]["frames"][0]["strain_valid_fraction"] >= 0.75
    valid = np.asarray(result["fields"][0]["valid"], dtype=bool)
    strain_valid = np.asarray(result["fields"][0]["strain_valid"], dtype=bool)
    assert valid.mean() >= .75
    assert strain_valid.mean() >= .75
    assert np.all(~strain_valid | valid)
    assert float(np.nanmean(np.asarray(result["fields"][0]["u"])[valid])) == pytest.approx(0.8, abs=0.25)
    assert float(np.nanmean(np.asarray(result["fields"][0]["v"])[valid])) == pytest.approx(-0.25, abs=0.25)
    assert core.verify_run_manifest(manifest_path)["ok"] is True
    assert result["manifest"]["solver"]["pyramid_levels"] == 2
    assert result["manifest"]["solver"]["max_residual_rms"] is None
    parameter_text = (tmp_path / "out" / "dic" / "frame_0002_parameters.txt").read_text(encoding="utf-8")
    assert "max_residual_rms = null" in parameter_text
    assert "+inf" not in parameter_text.lower()
    assert "max_residual_rms = inf" not in parameter_text.lower()


def test_headless_preflight_rejects_dimension_mismatch_before_output(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, n=2)
    _write_png(paths[1], np.zeros((127, 128), dtype=np.uint8))
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "roi_groups": [{"name": "G01", "roi1": (20, 45, 21, 21), "roi2": (85, 45, 21, 21), "strain_mode": "x"}],
    }
    with pytest.raises(core.CoreError, match="IMAGE_DIMENSION_MISMATCH"):
        core.run_extensometer_sequence(settings)
    assert not (tmp_path / "out").exists()


def test_headless_ambiguous_texture_fails_before_publication(tmp_path: Path) -> None:
    paths = []
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    xx = np.indices((128, 128))[1]
    stripe = (128 + 100 * np.sin(xx * 2 * np.pi / 12)).astype(np.uint8)
    for index in range(2):
        path = image_dir / f"frame_{index + 1:03d}.png"
        _write_png(path, stripe)
        paths.append(path)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "field_roi": (20, 20, 88, 88),
    }
    with pytest.raises(core.CoreError) as error:
        core.run_fullfield_sequence(settings)
    assert error.value.code == "AMBIGUOUS_TEXTURE"
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob("out/.staging_*"))


def test_extensometer_periodic_texture_fails_end_to_end_before_staging(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    xx = np.indices((128, 128))[1]
    stripe = (128 + 100 * np.sin(xx * 2 * np.pi / 12)).astype(np.uint8)
    paths = []
    for index in range(2):
        path = image_dir / f"frame_{index + 1:03d}.png"
        _write_png(path, stripe)
        paths.append(path)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "roi_groups": [{"name": "G01", "roi1": (20, 20, 40, 40), "roi2": (80, 20, 40, 40), "strain_mode": "x"}],
    }
    with pytest.raises(core.CoreError) as error:
        core.run_extensometer_sequence(settings)
    assert error.value.code == "AMBIGUOUS_TEXTURE"
    assert not (tmp_path / "out").exists()
    assert not list((tmp_path / "out").glob(".staging_*"))


def test_group_name_sanitization_collision_fails_before_staging(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, n=2)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "roi_groups": [
            {"name": "a/b", "roi1": (20, 45, 21, 21), "roi2": (85, 45, 21, 21), "strain_mode": "x"},
            {"name": r"a\b", "roi1": (20, 70, 21, 21), "roi2": (85, 70, 21, 21), "strain_mode": "x"},
        ],
    }
    with pytest.raises(core.CoreError) as error:
        core.run_extensometer_sequence(settings)
    assert error.value.code == "GROUP_NAME_COLLISION"
    assert error.value.details["collisions"][0]["sanitized_name"] == "a_b"
    assert not (tmp_path / "out").exists()


def test_nonfinite_api_fixture_fails_before_output(tmp_path: Path, monkeypatch) -> None:
    paths = _sequence(tmp_path, n=2)
    bad = np.ones((128, 128), dtype=np.float32)
    bad[4, 7] = np.nan
    monkeypatch.setattr(core, "read_gray_image", lambda _path: bad)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "roi_groups": [{"name": "G01", "roi1": (20, 45, 21, 21), "roi2": (85, 45, 21, 21), "strain_mode": "x"}],
    }
    with pytest.raises(core.CoreError) as error:
        core.run_extensometer_sequence(settings)
    assert error.value.code == "NONFINITE_IMAGE"
    assert not (tmp_path / "out").exists()
    assert not list((tmp_path / "out").glob(".staging_*"))


def test_scientific_gate_failure_is_retained_without_current_manifest(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, n=2)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "roi_groups": [{"name": "G01", "roi1": (20, 45, 21, 21), "roi2": (85, 45, 21, 21), "strain_mode": "x"}],
        "quality": {"enable_fb_check": False, "min_valid_frames": 99, "min_strain_valid_ratio": 1.0},
        "export": {"write_qc": True},
    }
    result = core.run_extensometer_sequence(settings)
    failed_manifest = Path(result["manifest_path"])
    assert result["status"] == "scientific_gate_failed"
    assert result["scientific_ok"] is False
    assert result["integrity_ok"] is True
    assert failed_manifest.is_file()
    assert not (tmp_path / "out" / "run_manifest.json").exists()
    assert core.verify_run_manifest(failed_manifest)["ok"] is True


def test_invalid_deformation_frame_cannot_pass_on_reference_row(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, n=2)
    _write_png(paths[1], np.full((128, 128), 80, dtype=np.uint8))
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "roi_groups": [{"name": "G01", "roi1": (20, 45, 21, 21), "roi2": (85, 45, 21, 21), "strain_mode": "x"}],
        "quality": {"enable_fb_check": False, "min_valid_frames": 1, "min_strain_valid_ratio": 0.0},
        "export": {"write_qc": True},
    }
    result = core.run_extensometer_sequence(settings)
    assert result["status"] == "scientific_gate_failed"
    gate = result["manifest"]["scientific_gate"]["groups"]["G01"]
    assert gate["frames"] == 1
    assert gate["valid_frames"] == 0
    assert gate["strain_valid_frames"] == 0
    assert not (tmp_path / "out" / "run_manifest.json").exists()


@pytest.mark.parametrize("field", ["enabled", "archive_previous", "retain_failed_staging"])
def test_transaction_policy_false_fails_before_staging(tmp_path: Path, field: str) -> None:
    paths = _sequence(tmp_path, n=2)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "roi_groups": [{"name": "G01", "roi1": (20, 45, 21, 21), "roi2": (85, 45, 21, 21), "strain_mode": "x"}],
        "transaction": {field: False},
    }
    with pytest.raises(core.CoreError) as error:
        core.run_extensometer_sequence(settings)
    assert error.value.code == "INVALID_TRANSACTION_POLICY"
    assert not (tmp_path / "out").exists()


def test_fullfield_parameter_provenance_opt_out_is_explicitly_rejected(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, n=2)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "field_roi": (20, 20, 88, 88),
        "export": {"write_parameters": False},
    }
    with pytest.raises(core.CoreError) as error:
        core.run_fullfield_sequence(settings)
    assert error.value.code == "INVALID_EXPORT_POLICY"
    assert not (tmp_path / "out").exists()


def test_optional_origin_failure_is_warning_unless_required(tmp_path: Path, monkeypatch) -> None:
    paths = _sequence(tmp_path, n=2)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "roi_groups": [{"name": "G01", "roi1": (20, 45, 21, 21), "roi2": (85, 45, 21, 21), "strain_mode": "x"}],
        "quality": {"enable_fb_check": False, "min_valid_frames": 1, "min_strain_valid_ratio": 0.0},
        "export": {"write_qc": True, "write_origin_opju": True},
    }
    monkeypatch.setattr(core, "write_origin_opju_project", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Origin unavailable")))
    warning_result = core.run_extensometer_sequence(settings)
    assert warning_result["status"] == "completed_with_warnings"
    assert warning_result["scientific_ok"] is True
    assert warning_result["manifest"]["optional_failures"][0]["required_for_scientific_gate"] is False

    required_settings = dict(settings)
    required_settings["output_dir"] = str(tmp_path / "out-required")
    required_settings["export"] = {"write_qc": True, "write_origin_opju": True, "origin_opju_required": True}
    failed = core.run_extensometer_sequence(required_settings)
    assert failed["status"] == "scientific_gate_failed"
    assert failed["scientific_ok"] is False
    assert not (tmp_path / "out-required" / "run_manifest.json").exists()


def test_fullfield_scientific_gate_failure_is_retained_without_current(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, n=2)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 1,
        "end_frame_1based": 2,
        "reference_frame_1based": 1,
        "output_dir": str(tmp_path / "out"),
        "field_roi": (20, 20, 88, 88),
        "solver": {"name": "IC-GN", "subset_size_px": 15, "step_px": 12, "strain_window_px": 5, "search_radius_px": 12},
        "quality": {"zncc_min": 0.65, "min_correlation_valid_fraction": 1.0, "min_strain_valid_fraction": 1.0},
        "export": {"write_manifest": True},
    }
    result = core.run_fullfield_sequence(settings)
    failed_manifest = Path(result["manifest_path"])
    assert result["status"] == "scientific_gate_failed"
    assert result["scientific_ok"] is False
    assert result["integrity_ok"] is True
    assert failed_manifest.is_file()
    assert not (tmp_path / "out" / "run_manifest.json").exists()
    assert core.verify_run_manifest(failed_manifest)["ok"] is True


def test_fullfield_reference_selection_is_explicit_and_recorded(tmp_path: Path) -> None:
    paths = _sequence(tmp_path, n=4)
    settings = {
        "image_paths": [str(path) for path in paths],
        "start_frame_1based": 2,
        "end_frame_1based": 4,
        "reference_frame_1based": 2,
        "output_dir": str(tmp_path / "out"),
        "field_roi": (20, 20, 88, 88),
        "solver": {"name": "IC-GN", "subset_size_px": 15, "step_px": 12, "strain_window_px": 5, "search_radius_px": 12},
        "quality": {"zncc_min": 0.65, "min_correlation_valid_fraction": 0.0, "min_strain_valid_fraction": 0.0},
        "export": {"write_manifest": True, "write_overlays": False},
    }
    result = core.run_fullfield_sequence(settings)
    assert result["manifest"]["reference_frame_1based"] == 2
    assert result["manifest"]["processing_order_frame_1based"] == [2, 3, 4]
    assert [item["frame_global_1based"] for item in result["manifest"]["frames"]] == [3, 4]


def test_gui_adapters_call_core_engines_and_complete_after_engine_return(monkeypatch, tmp_path: Path) -> None:
    import dic_virtual_extensometer_gui_v7_multi_roi_range as gui

    class _Var:
        def __init__(self, value=""):
            self.value = value

        def set(self, value):
            self.value = value

        def get(self):
            return self.value

    class _Progress:
        def __init__(self):
            self.values = []

        def config(self, **kwargs):
            self.values.append(kwargs)

    app = object.__new__(gui.MultiROIGUI)
    app.progress = _Progress()
    app.status_var = _Var()
    events = []
    app.post_to_ui = lambda callback: callback()
    app.log = lambda message: events.append(("log", message))
    app.update_qc_overview = lambda _summary: events.append(("qc", None))
    app.show_completion_and_open_output_folder = lambda *_args: events.append(("completion", None))
    app.show_results_viewer = lambda *_args: events.append(("viewer", None))
    app.show_field_viewer = lambda *_args, **_kwargs: events.append(("field_viewer", None))

    ext_df = pd.DataFrame({"group": ["G01"], "accepted": [True], "strain_valid": [True], "engineering_strain": [0.0]})
    ext_summary = {"overall": {"qc_level": "Good", "rejected_frames": 0, "adaptive_accepted_frames": 0}}
    ext_settings = {
        "image_paths": ["unused.png"], "start_idx": 0, "end_idx": 0,
        "output_dir": str(tmp_path / "ext-out"), "roi_groups": [{"name": "G01"}],
        "export_origin_txt": False,
    }

    def fake_ext(settings, progress_callback=None):
        assert settings["_gui_adapter"] is True
        assert settings["_code_paths"]
        events.append(("ext_engine", settings.get("min_valid_frames")))
        return {"status": "completed", "scientific_ok": True, "manifest_path": "sealed-ext.json", "outputs": [], "dataframe": ext_df, "summary": ext_summary}

    monkeypatch.setattr(gui, "validate_image_sequence_dimensions", lambda _paths: None)
    monkeypatch.setattr(gui._core, "run_extensometer_sequence", fake_ext)
    gui.MultiROIGUI.process_images(app, ext_settings)
    assert [event[0] for event in events].index("ext_engine") < [event[0] for event in events].index("completion")

    events.clear()
    field_settings = {
        "image_paths": ["unused.png"], "start_idx": 0, "end_idx": 0,
        "output_dir": str(tmp_path / "field-out"), "field_roi": (10, 10, 40, 40),
    }
    field = {"frame_global_1based": 1, "frame_filename": "unused.png"}

    def fake_field(settings, progress_callback=None):
        assert settings["_gui_adapter"] is True
        events.append(("field_engine", settings.get("_code_paths")))
        return {"status": "completed", "scientific_ok": True, "manifest_path": "sealed-field.json", "outputs": [], "last_field": field, "last_image": np.zeros((20, 20), dtype=np.uint8), "frames": [{"status": "scientific_valid"}], "manifest": {"reference_frame_1based": 1}}

    monkeypatch.setattr(gui, "archive_previous_fullfield_outputs", lambda _path: events.append(("legacy_archive", None)))
    app.validate_fullfield_before_processing = lambda: events.append(("fullfield_preflight", None))
    monkeypatch.setattr(gui._core, "run_fullfield_sequence", fake_field)
    gui.MultiROIGUI.process_fullfield(app, field_settings)
    assert [event[0] for event in events].index("fullfield_preflight") < [event[0] for event in events].index("legacy_archive") < [event[0] for event in events].index("field_engine")
    assert [event[0] for event in events].index("field_engine") < [event[0] for event in events].index("completion")
