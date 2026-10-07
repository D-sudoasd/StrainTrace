"""Gating tests for shipped 2D DIC (IC-GN / IC-LM) against known warps."""
import math
from pathlib import Path

import numpy as np
import pytest

import dic_virtual_extensometer_gui_v7_multi_roi_range as ezdic


def _speckle_pair_translation(tx, ty, *, seed=3, size=128):
    reference = ezdic.generate_synthetic_speckle(size, size, seed=seed)
    deformed = ezdic.warp_image_translation(reference, tx, ty)
    roi = (18, 18, size - 36, size - 36)
    return reference, deformed, roi


def _speckle_pair_stretch(lambda_x, lambda_y=1.0, *, seed=3, size=128):
    reference = ezdic.generate_synthetic_speckle(size, size, seed=seed)
    F = np.array([[lambda_x, 0.0], [0.0, lambda_y]], dtype=float)
    deformed = ezdic.warp_image_deformation_gradient(reference, F)
    roi = (20, 20, size - 40, size - 40)
    return reference, deformed, roi, F


@pytest.mark.parametrize("solver", [ezdic.DIC_SOLVER_ICGN, ezdic.DIC_SOLVER_ICLM])
def test_ic_solvers_recover_known_subpixel_translation(solver):
    tx, ty = 1.37, -0.62
    reference, deformed, roi = _speckle_pair_translation(tx, ty)
    field = ezdic.run_2d_dic(
        reference,
        deformed,
        roi,
        subset_size=21,
        step=10,
        solver=solver,
        search_radius=8,
        zncc_min=0.75,
        strain_window=5,
        smooth_sigma=0.0,
    )

    assert field["solver"] == solver
    assert field["u"] is not ezdic.run_2d_dic  # shipped entry, not a mock
    valid = np.asarray(field["valid"], dtype=bool)
    assert int(valid.sum()) >= 20
    mean_u = float(np.nanmean(field["u"]))
    mean_v = float(np.nanmean(field["v"]))
    assert mean_u == pytest.approx(tx, abs=0.05)
    assert mean_v == pytest.approx(ty, abs=0.05)
    assert float(np.nanmean(field["zncc"])) > 0.95


@pytest.mark.parametrize("solver", [ezdic.DIC_SOLVER_ICGN, ezdic.DIC_SOLVER_ICLM])
def test_green_lagrange_matches_applied_stretch(solver):
    lambda_x = 1.02
    reference, deformed, roi, F = _speckle_pair_stretch(lambda_x)
    oracle = ezdic.green_lagrange_from_F(F)
    field = ezdic.run_2d_dic(
        reference,
        deformed,
        roi,
        subset_size=21,
        step=8,
        solver=solver,
        search_radius=6,
        zncc_min=0.75,
        strain_window=5,
        smooth_sigma=0.0,
    )

    mean_exx = float(np.nanmean(np.asarray(field["Exx"], dtype=float)))
    mean_eyy = float(np.nanmean(np.asarray(field["Eyy"], dtype=float)))
    mean_exy = float(np.nanmean(np.asarray(field["Exy"], dtype=float)))
    assert mean_exx == pytest.approx(oracle["Exx"], abs=0.003)
    assert mean_eyy == pytest.approx(oracle["Eyy"], abs=0.003)
    assert mean_exy == pytest.approx(oracle["Exy"], abs=0.003)
    inf_exx = float(np.nanmean(np.asarray(field["exx"], dtype=float)))
    assert inf_exx == pytest.approx(lambda_x - 1.0, abs=0.004)


def test_failed_and_out_of_roi_points_stay_nonfinite():
    tx, ty = 0.55, 0.40
    reference, deformed, roi = _speckle_pair_translation(tx, ty, seed=11, size=128)
    wrecked = deformed.copy()
    wrecked[48:80, 48:80] = 28.0
    field = ezdic.run_2d_dic(
        reference,
        wrecked,
        roi,
        subset_size=21,
        step=10,
        solver=ezdic.DIC_SOLVER_ICGN,
        zncc_min=0.8,
        strain_window=5,
        smooth_sigma=1.5,
    )
    valid = np.asarray(field["valid"], dtype=bool)
    u = np.asarray(field["u"], dtype=float)
    v = np.asarray(field["v"], dtype=float)
    exx = np.asarray(field["Exx"], dtype=float).ravel()
    xs = np.asarray(field["x"], dtype=float)
    ys = np.asarray(field["y"], dtype=float)

    assert int(valid.sum()) >= 8
    assert int((~valid).sum()) >= 1
    assert not np.isfinite(u[~valid]).any()
    assert not np.isfinite(v[~valid]).any()
    assert not np.isfinite(exx[~valid]).any()
    assert np.isfinite(u[valid]).all()
    # POIs whose subset sits inside the destroyed patch must fail.
    half = 10
    in_wreck = (xs - half >= 48) & (xs + half <= 80) & (ys - half >= 48) & (ys + half <= 80)
    if in_wreck.any():
        assert not valid[in_wreck].any()

    # Points are only generated inside the ROI; a tiny ROI cannot host a subset.
    X, Y = ezdic.build_poi_grid((0, 0, 8, 8), subset_size=21, step=5, image_shape=reference.shape)
    assert X.size == 0
    assert Y.size == 0

    flat = np.full_like(reference, 40.0)
    all_fail = ezdic.run_2d_dic(flat, wrecked, roi, subset_size=21, step=12, zncc_min=0.8)
    assert not np.asarray(all_fail["valid"]).any()
    assert not np.isfinite(all_fail["u"]).any()


def test_dic_table_and_colormap_use_plot_presets(tmp_path):
    tx, ty = 0.80, -0.35
    reference, deformed, roi = _speckle_pair_translation(tx, ty, seed=5)
    field = ezdic.run_2d_dic(
        reference, deformed, roi, subset_size=21, step=12, solver=ezdic.DIC_SOLVER_ICGN
    )
    table = ezdic.dic_field_to_dataframe(field)
    legacy_columns = [
        "x",
        "y",
        "u",
        "v",
        "zncc",
        "valid",
        "Exx",
        "Eyy",
        "Exy",
        "exx",
        "eyy",
        "exy",
    ]
    assert list(table.columns[: len(legacy_columns)]) == legacy_columns
    assert {"u_raw", "v_raw", "residual_rms", "strain_valid"}.issubset(table.columns)
    txt_path = tmp_path / "field.txt"
    ezdic.write_dic_field_txt(field, txt_path)
    text = txt_path.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("x\ty\tu\tv")
    plot_path = tmp_path / "Exx.png"
    ezdic.plot_dic_field_map(field, plot_path, component="Exx", preset_name="publication")
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

    fig, ax, _preset = ezdic.create_plot_figure("publication")
    mesh = ezdic.render_dic_field_on_axes(ax, field, "u")
    cbar = ezdic.add_dic_colorbar(fig, ax, mesh, "u (px)", preset_name="publication")
    assert cbar is not None
    ezdic.plt.close(fig)


def test_field_export_includes_provenance_parameters_and_effective_odd_values(tmp_path):
    reference, deformed, roi = _speckle_pair_translation(0.6, -0.3, seed=13)
    field = ezdic.run_2d_dic(
        reference,
        deformed,
        roi,
        subset_size=20,
        step=10,
        strain_window=6,
    )
    field["provenance"] = {
        "reference_frame_1based": 1,
        "reference_filename": "ref_001.png",
        "frame_global_1based": 2,
        "frame_filename": "def_002.png",
        "field_roi": roi,
    }
    outputs = ezdic.export_dic_field_outputs(field, tmp_path, stem="frame_0002")
    parameters = outputs["parameters"]
    assert parameters.exists() and parameters.stat().st_size > 0
    text = parameters.read_text(encoding="utf-8")
    assert "reference_frame_1based = 1" in text
    assert "reference_filename = ref_001.png" in text
    assert "frame_global_1based = 2" in text
    assert "frame_filename = def_002.png" in text
    assert "field_roi = 18,18,92,92" in text
    assert "subset_size_px = 21" in text
    assert "strain_window = 7" in text


def test_write_image_checked_supports_unicode_windows_style_paths(tmp_path):
    path = tmp_path / "中文结果" / "全场图_0002.png"
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    written = ezdic.write_image_checked(path, image)
    assert written == path
    assert path.exists() and path.stat().st_size > 0
    decoded = ezdic.read_gray_image(path)
    assert decoded.shape == image.shape


def test_commit_staging_rolls_back_on_second_move_and_failed_archive_keeps_all_files(
    tmp_path, monkeypatch
):
    dic_dir = tmp_path / "dic"
    staging = dic_dir / ".staging_test"
    staging.mkdir(parents=True)
    current_files = [staging / "frame_0002.txt", staging / "frame_0002.csv"]
    for path in current_files:
        path.write_text("current run\n", encoding="utf-8")
    similar_user_file = dic_dir / "frame_0002_u.txt"
    similar_user_file.write_text("keep user file\n", encoding="utf-8")

    real_move = ezdic.shutil.move
    commit_moves = 0
    events = []

    def fail_second_commit_move(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        nonlocal commit_moves
        is_commit_move = source_path.parent == staging and destination_path.parent == dic_dir
        if is_commit_move:
            commit_moves += 1
            events.append((source_path, destination_path))
            if commit_moves == 2:
                raise OSError("forced commit move #2 failure")
        elif source_path.parent == dic_dir and destination_path.parent == staging:
            events.append((source_path, destination_path))
        return real_move(source, destination)

    monkeypatch.setattr(ezdic.shutil, "move", fail_second_commit_move)
    with pytest.raises(RuntimeError, match="提交"):
        ezdic.commit_fullfield_staging(staging, dic_dir)

    assert commit_moves == 2
    assert (staging / "frame_0002.csv", dic_dir / "frame_0002.csv") in events
    assert (dic_dir / "frame_0002.csv", staging / "frame_0002.csv") in events
    assert all(path.exists() for path in current_files)
    assert not (dic_dir / "frame_0002.txt").exists()
    assert not (dic_dir / "frame_0002.csv").exists()
    assert similar_user_file.exists()

    failed_dir = ezdic.archive_failed_fullfield_staging(staging, dic_dir)
    assert failed_dir is not None and failed_dir.exists()
    assert sorted(path.name for path in failed_dir.iterdir()) == [
        "frame_0002.csv",
        "frame_0002.txt",
    ]
    assert not staging.exists()
    assert similar_user_file.exists()


def test_previous_output_archive_rolls_back_on_second_move_failure(tmp_path, monkeypatch):
    dic_dir = tmp_path / "dic"
    dic_dir.mkdir()
    generated = [dic_dir / "frame_0002.txt", dic_dir / "frame_0002.csv"]
    original_bytes = {}
    for path in generated:
        original_bytes[path] = b"previous run\x00\xff"
        path.write_bytes(original_bytes[path])
    similar_user_file = dic_dir / "frame_0002_parameters.png"
    similar_user_file.write_text("keep user file\n", encoding="utf-8")

    real_move = ezdic._core._move_exact
    archive_moves = 0
    archive_failures = 0
    events = []

    def fail_second_archive_move(source, destination, *, root=None):
        nonlocal archive_moves
        nonlocal archive_failures
        source_path = Path(source)
        destination_path = Path(destination)
        is_archive_move = (
            source_path.parent == dic_dir
            and destination_path.parent.parent == dic_dir / "_previous_runs"
        )
        if is_archive_move:
            archive_moves += 1
            events.append((source_path, destination_path))
            if archive_moves == 2:
                archive_failures += 1
                raise OSError("forced archive move #2 failure")
        elif source_path.parent.parent == dic_dir / "_previous_runs" and destination_path.parent == dic_dir:
            events.append((source_path, destination_path))
        return real_move(source, destination, root=root)

    monkeypatch.setattr(ezdic._core, "_move_exact", fail_second_archive_move)
    with pytest.raises(RuntimeError, match="归档"):
        ezdic.archive_previous_fullfield_outputs(dic_dir)

    assert archive_moves == 2
    assert archive_failures == 1
    assert any(event[0] == dic_dir / "frame_0002.csv" for event in events)
    assert any(event[1] == dic_dir / "frame_0002.csv" for event in events)
    assert all(path.exists() and path.read_bytes() == original_bytes[path] for path in generated)
    assert similar_user_file.exists()
    assert sorted(path.name for path in dic_dir.iterdir() if path.is_file()) == sorted(
        [path.name for path in generated] + [similar_user_file.name]
    )
    assert not list((dic_dir / "_previous_runs").rglob("frame_0002.txt"))
    assert not list((dic_dir / "_previous_runs").rglob("frame_0002.csv"))


def test_run_2d_dic_sequence_matches_single_frame():
    reference, deformed, roi = _speckle_pair_translation(1.1, -0.4, seed=8)
    single = ezdic.run_2d_dic(reference, deformed, roi, subset_size=21, step=12)
    batch = ezdic.run_2d_dic_sequence(reference, [deformed], roi, subset_size=21, step=12)
    assert len(batch) == 1
    assert float(np.nanmean(batch[0]["u"])) == pytest.approx(float(np.nanmean(single["u"])), abs=1e-9)


def test_ic_refinement_beats_integer_guess_on_shipped_icgn():
    tx, ty = 1.37, -0.62
    # A quantized OpenCV warp embeds a sampling bias in the manufactured
    # target. Evaluate an independent continuous image for subpixel truth.
    from benchmarks.research_validation import analytic_pair
    reference, deformed = analytic_pair(u=tx, v=ty)
    x = y = 64.0
    u0, v0, cc = ezdic.integer_cc_guess(reference, deformed, x, y, 21, search_radius=8)
    assert cc > 0.5
    refined = ezdic.refine_subset_icgn(
        reference, deformed, x, y, 21, p0=[u0, 0.0, 0.0, v0, 0.0, 0.0]
    )
    assert refined is not None
    assert refined["zncc"] > 0.98
    assert abs(refined["u"] - tx) < 0.05
    assert abs(refined["v"] - ty) < 0.05
    assert abs(refined["u"] - tx) <= abs(u0 - tx) + 1e-9
    lm = ezdic.refine_subset_iclm(
        reference, deformed, x, y, 21, p0=[u0, 0.0, 0.0, v0, 0.0, 0.0]
    )
    assert lm is not None
    assert abs(lm["u"] - tx) < 0.05
    assert abs(lm["v"] - ty) < 0.05


def test_green_lagrange_oracle_matches_analytic_stretch():
    F = np.array([[1.02, 0.0], [0.0, 1.0]])
    gl = ezdic.green_lagrange_from_F(F)
    expected = 0.5 * (1.02 ** 2 - 1.0)
    assert math.isclose(gl["Exx"], expected, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(gl["Eyy"], 0.0, abs_tol=1e-12)
    assert math.isclose(gl["Exy"], 0.0, abs_tol=1e-12)


def test_normalized_correlation_rejects_zero_variance_template_and_search(monkeypatch):
    reference = np.full((40, 40), 50.0, dtype=np.float32)
    deformed = np.full((40, 40), 80.0, dtype=np.float32)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("TM_CCOEFF_NORMED must not see a constant input")

    monkeypatch.setattr(ezdic.cv2, "matchTemplate", fail_if_called)
    u, v, score = ezdic.integer_cc_guess(reference, deformed, 20, 20, 15, 5)
    assert (u, v, score) == (0.0, 0.0, -1.0)

    candidate, score = ezdic.match_template_candidate(
        deformed,
        (10, 10, 15, 15),
        np.ones((15, 15), dtype=np.float32),
        5,
    )
    assert tuple(candidate) == (10, 10, 15, 15)
    assert score == -1.0


def test_normalized_correlation_rejects_constant_candidate_patch():
    template = np.zeros((9, 9), dtype=np.float32)
    template[2:7, 2:7] = np.arange(25, dtype=np.float32).reshape(5, 5)
    search = np.full((20, 20), 30.0, dtype=np.float32)
    candidate, score = ezdic.match_template_candidate(
        search,
        (5, 5, 9, 9),
        template,
        4,
    )
    assert tuple(candidate) == (5, 5, 9, 9)
    assert score == -1.0


def test_fullfield_rejects_mismatched_dimensions_and_all_nan_field():
    reference = np.zeros((64, 64), dtype=np.float32)
    deformed = np.zeros((63, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="dimensions must match"):
        ezdic.run_2d_dic(reference, deformed, (10, 10, 40, 40), subset_size=15, step=8)

    field = ezdic.run_2d_dic(
        np.full((64, 64), 50.0, dtype=np.float32),
        np.full((64, 64), 80.0, dtype=np.float32),
        (10, 10, 40, 40),
        subset_size=15,
        step=8,
    )
    assert not ezdic.fullfield_field_has_finite_strain(field)


def test_rank_deficient_one_row_grid_stays_nonfinite_with_large_window():
    x = np.arange(3, dtype=float).reshape(1, 3)
    y = np.zeros((1, 3), dtype=float)
    u = np.array([[0.0, 0.2, 0.4]], dtype=float)
    v = np.zeros_like(u)
    strains = ezdic.compute_strain_fields(x, y, u, v, window=7, smooth_sigma=0.0)
    assert not np.isfinite(strains["Exx"]).any()
    assert not np.isfinite(strains["Eyy"]).any()
    assert not np.isfinite(strains["Exy"]).any()


def test_simple_shear_recovers_tensor_shear_and_green_lagrange_quadratic_term():
    coords = np.arange(5, dtype=float)
    x, y = np.meshgrid(coords, coords)
    gamma = 0.20
    u = gamma * y
    v = np.zeros_like(u)
    strains = ezdic.compute_strain_fields(x, y, u, v, window=3, smooth_sigma=0.0)
    assert np.nanmean(strains["exy"]) == pytest.approx(gamma / 2.0, abs=1e-12)
    assert np.nanmean(strains["Exy"]) == pytest.approx(gamma / 2.0, abs=1e-12)
    assert np.nanmean(strains["Eyy"]) == pytest.approx(gamma ** 2 / 2.0, abs=1e-12)


def test_run_2d_dic_valid_requires_finite_u_v_zncc_and_affine_parameters(monkeypatch):
    reference = ezdic.generate_synthetic_speckle(64, 64, seed=23)
    deformed = ezdic.warp_image_translation(reference, 0.4, -0.2)

    def fake_refine(*_args, **_kwargs):
        return {
            "u": 0.4,
            "v": np.nan,
            "zncc": 0.99,
            "p": np.zeros(6, dtype=float),
        }

    monkeypatch.setattr(ezdic, "refine_subset_icgn", fake_refine)
    field = ezdic.run_2d_dic(
        reference,
        deformed,
        (10, 10, 40, 40),
        subset_size=15,
        step=8,
        solver=ezdic.DIC_SOLVER_ICGN,
    )
    assert not np.asarray(field["valid"], dtype=bool).any()
