import csv

import numpy as np
import pytest

from straintrace_agent.fields import (
    calibrate_field,
    component_values,
    iter_roi_rows,
    jacobian_determinant,
    plot_position_time,
    principal_strains,
    sample_path,
    summarize_roi,
    write_rows_csv,
)


def _affine_field(F, *, size=4):
    coords = np.arange(size, dtype=float)
    X, Y = np.meshgrid(coords, coords)
    reference = np.stack((X, Y), axis=-1)
    deformed = reference @ np.asarray(F, dtype=float).T
    U = deformed[..., 0] - X
    V = deformed[..., 1] - Y
    C = np.asarray(F, dtype=float).T @ np.asarray(F, dtype=float)
    E = 0.5 * (C - np.eye(2))
    grad = np.asarray(F, dtype=float) - np.eye(2)
    small = 0.5 * (grad + grad.T)
    P = np.zeros(X.shape + (6,), dtype=float)
    P[..., 0] = U
    P[..., 1] = F[0, 0] - 1.0
    P[..., 2] = F[0, 1]
    P[..., 3] = V
    P[..., 4] = F[1, 0]
    P[..., 5] = F[1, 1] - 1.0
    valid = np.ones(X.shape, dtype=bool)
    return {
        "x": X.ravel(), "y": Y.ravel(), "u": U.ravel(), "v": V.ravel(),
        "u_raw": U.ravel(), "v_raw": V.ravel(), "valid": valid.ravel(),
        "strain_valid": valid.ravel(),
        "Exx": np.full(X.size, E[0, 0]), "Eyy": np.full(X.size, E[1, 1]),
        "Exy": np.full(X.size, E[0, 1]), "exx": np.full(X.size, small[0, 0]),
        "eyy": np.full(X.size, small[1, 1]), "exy": np.full(X.size, small[0, 1]),
        "X": X, "Y": Y, "U": U, "V": V, "P": P,
    }


def test_core_run_2d_dic_field_layout_supports_affine_jacobian_diagnostic(monkeypatch):
    import ezdic_core as core

    reference = core.generate_synthetic_speckle(72, 72, seed=41)

    def identity_refinement(*_args, **_kwargs):
        return {"u": 0.0, "v": 0.0, "zncc": 1.0, "p": np.zeros(6), "residual_rms": 0.0}

    monkeypatch.setattr(core, "refine_subset_icgn", identity_refinement)
    field = core.run_2d_dic(
        reference, reference, (12, 12, 48, 48), subset_size=15, step=12,
        initialization="local", reject_ambiguous_peaks=False, zncc_min=0.0,
        min_correlation_valid_fraction=0.0, min_strain_valid_fraction=0.0,
    )
    assert field["X"].ndim == 2
    assert field["P"].shape == field["X"].shape + (6,)
    assert "dudx" not in field
    det = jacobian_determinant(field)
    assert det.shape == field["X"].shape
    assert np.nanmean(det) == pytest.approx(1.0, abs=1e-12)


def test_principal_green_lagrange_is_objective_under_large_rigid_rotation():
    angle = np.deg2rad(123.0)
    F = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result = principal_strains(_affine_field(F), "green_lagrange")
    assert np.nanmax(np.abs(result["major"])) < 1e-14
    assert np.nanmax(np.abs(result["minor"])) < 1e-14
    assert np.isnan(result["theta_rad"]).all()  # Equal principal values: direction undefined.
    assert result["measure"] == "green_lagrange"


def test_affine_F_principal_values_shear_factor_and_axis_angle():
    angle = np.deg2rad(31.0)
    stretch = np.diag([1.2, 0.8])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    F = rotation @ stretch @ rotation.T
    field = _affine_field(F)
    result = principal_strains(field, "green_lagrange")
    expected = np.linalg.eigvalsh(0.5 * (F.T @ F - np.eye(2)))
    assert result["minor"].mean() == pytest.approx(expected[0])
    assert result["major"].mean() == pytest.approx(expected[1])
    assert result["theta_deg"].mean() == pytest.approx(31.0)
    assert np.allclose(jacobian_determinant(field), np.linalg.det(F))

    gamma = 0.24
    shear_F = np.array([[1.0, gamma], [0.0, 1.0]])
    shear = _affine_field(shear_F)
    gl = principal_strains(shear, "green_lagrange")
    infinitesimal = principal_strains(shear, "infinitesimal")
    assert gl["tensor_shear"].mean() == pytest.approx(gamma / 2)
    assert gl["engineering_shear"].mean() == pytest.approx(gamma)
    assert infinitesimal["tensor_shear"].mean() == pytest.approx(gamma / 2)
    assert infinitesimal["engineering_shear"].mean() == pytest.approx(gamma)
    assert component_values(shear, "green_lagrange_gamma_xy").mean() == pytest.approx(gamma)
    assert component_values(shear, "small_strain_gamma_xy").mean() == pytest.approx(gamma)


def test_negative_jacobian_and_invalid_strain_mask():
    field = _affine_field(np.eye(2))
    field["P"] = field["P"].copy()
    field["P"][..., 1] = -2.0  # Core P convention gives det(F) = -1.
    det = jacobian_determinant(field)
    assert np.all(det == pytest.approx(-1.0))
    result = principal_strains(field)
    assert not result["valid"].any()
    assert np.isnan(result["major"]).all()

    field["P"][..., 1] = -1.0  # det(F) = 0 is degenerate and must also be masked.
    assert np.all(jacobian_determinant(field) == pytest.approx(0.0))
    assert not principal_strains(field)["valid"].any()

    field = _affine_field(np.eye(2))
    field["strain_valid"] = field["strain_valid"].copy()
    field["strain_valid"][5] = False
    result = principal_strains(field)
    assert not result["valid"].ravel()[5]
    assert np.isnan(result["major"].ravel()[5])


def test_explicit_local_gradients_take_precedence_over_subset_P():
    field = _affine_field(np.eye(2))
    field["P"] = field["P"].copy()
    field["P"][..., 1] = -2.0
    for key in ("dudx", "dudy", "dvdx", "dvdy"):
        field[key] = np.zeros(field["X"].shape)
    assert np.allclose(jacobian_determinant(field), 1.0)


def test_calibration_requires_explicit_origin_and_signs_and_converts_units():
    field = _affine_field(np.eye(2))
    with pytest.raises(ValueError, match="origin_px"):
        calibrate_field(field, {"mm_per_px": 0.2})
    with pytest.raises(ValueError, match="mm_per_px"):
        calibrate_field(field, {
            "mm_per_px": True, "origin_px": (0, 0), "origin_mm": (0, 0), "axis_sign": (1, 1),
        })
    with pytest.raises(ValueError, match="booleans"):
        calibrate_field(field, {
            "mm_per_px": 0.2, "origin_px": (False, 0), "origin_mm": (0, 0), "axis_sign": (1, 1),
        })
    calibrated = calibrate_field(field, {
        "mm_per_px": 0.2,
        "origin_px": (1, 2),
        "origin_mm": (10, -3),
        "axis_sign": (1, -1),
    })
    assert calibrated["x_mm"][0, 0] == pytest.approx(9.8)
    assert calibrated["y_mm"][0, 0] == pytest.approx(-2.6)
    assert calibrated["u_mm"].mean() == pytest.approx(0.0)
    assert calibrated["calibration"]["mm_per_px"] == 0.2


def test_roi_summary_uses_only_valid_original_pois_and_reports_scatter_not_se(tmp_path):
    field = _affine_field(np.eye(2), size=3)
    field["Exx"] = np.arange(9, dtype=float)
    field["strain_valid"] = field["strain_valid"].copy()
    field["strain_valid"][4] = False
    field["Exx"][4] = np.nan
    summary = summarize_roi(field, "Exx", (0, 2, 0, 2))
    assert summary["n_poi_total"] == 9
    assert summary["n_poi_valid"] == 8
    assert summary["mean"] == pytest.approx(4.0)
    assert summary["sd_sample"] == pytest.approx(np.std([0, 1, 2, 3, 5, 6, 7, 8], ddof=1))
    assert "not_SE" in summary["sd_interpretation"]
    assert all(row["interpolation"] == "none" for row in iter_roi_rows(field, "Exx", (0, 2, 0, 2)))
    path = write_rows_csv(iter_roi_rows(field, "Exx", (0, 2, 0, 2)), tmp_path / "roi.csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert rows[4]["status"] == "invalid_poi"
    assert rows[4]["value"] == ""


def test_roi_principal_axis_summary_respects_180_degree_periodicity():
    field = _affine_field(np.eye(2), size=2)
    angles = np.deg2rad(np.array([89.0, -89.0, 89.0, -89.0]))
    field["Exx"] = np.cos(2 * angles)
    field["Eyy"] = -np.cos(2 * angles)
    field["Exy"] = np.sin(2 * angles)
    summary = summarize_roi(field, "principal_angle_deg", (0, 1, 0, 1))
    assert abs(summary["mean_axial_deg"]) > 89.0
    assert summary["sd_sample"] is None
    assert summary["sd_axial_circular_deg"] == pytest.approx(1.0, abs=0.1)


def test_path_nearest_point_hole_and_distance_tolerance_do_not_jump_or_interpolate():
    field = _affine_field(np.eye(2), size=3)
    field["valid"] = field["valid"].copy()
    field["strain_valid"] = field["strain_valid"].copy()
    field["valid"][4] = False
    field["strain_valid"][4] = False
    field["Exx"][4] = np.nan
    rows = sample_path(field, [(1, 1), (1.5, 1.5), (9, 9)], "Exx", 0.5)
    assert rows[0]["poi_index"] == 4
    assert rows[0]["status"] == "invalid_poi"  # It does not jump to a neighbor.
    assert rows[1]["status"] == "outside_tolerance"
    assert rows[2]["status"] == "outside_tolerance"
    assert all(row["interpolation"] == "none" for row in rows)


def test_calibrated_map_respects_explicit_physical_y_sign():
    import matplotlib.pyplot as plt

    field = _affine_field(np.eye(2), size=3)
    calibrated = calibrate_field(field, {
        "mm_per_px": 0.1, "origin_px": (0, 0), "origin_mm": (0, 0), "axis_sign": (1, -1),
    })
    field_mm = {**field, **calibrated}
    from straintrace_agent.fields import plot_field_map

    fig, ax, _ = plot_field_map(field_mm, "u_mm", coordinate_system="calibrated")
    assert not ax.yaxis_inverted()
    plt.close(fig)
    calibrated_down = calibrate_field(field, {
        "mm_per_px": 0.1, "origin_px": (0, 0), "origin_mm": (0, 0), "axis_sign": (1, 1),
    })
    fig, ax, _ = plot_field_map({**field, **calibrated_down}, "u_mm", coordinate_system="calibrated")
    assert ax.yaxis_inverted()
    plt.close(fig)


def test_common_map_limits_are_shared_and_invalid_pois_are_omitted():
    import matplotlib.pyplot as plt

    from straintrace_agent.fields import common_color_limits, plot_field_map

    first = _affine_field(np.eye(2), size=3)
    second = _affine_field(np.eye(2), size=3)
    first["Exx"] = np.linspace(-2, 1, 9)
    second["Exx"] = np.linspace(-1, 3, 9)
    first["strain_valid"] = first["strain_valid"].copy()
    first["strain_valid"][0] = False
    limits = common_color_limits((first, second), "Exx", symmetric=True)
    assert limits == pytest.approx((-3.0, 3.0))
    fig, ax, artist = plot_field_map(first, "Exx", color_limits=limits)
    assert artist.get_clim() == pytest.approx(limits)
    assert len(artist.get_offsets()) == 8
    plt.close(fig)


def test_position_time_plot_keeps_missing_frames_and_gaps_as_empty_scatter():
    import matplotlib.pyplot as plt

    field = _affine_field(np.eye(2), size=3)
    fig, ax, artist, summary = plot_position_time(
        (f for f in (field, None, field)), (0.0, 0.5, 2.0), [(0, 0)], "Exx", 0.0,
        max_time_gap_s=0.75,
    )
    offsets = artist.get_offsets()
    assert offsets.shape == (2, 2)
    assert list(offsets[:, 1]) == pytest.approx([0.0, 2.0])
    assert summary["missing_frames"] == 1
    assert summary["sampled_values"] == 2
    assert summary["large_time_gaps"] == 1
    assert summary["max_observed_time_gap_s"] == pytest.approx(1.5)
    assert summary["interpolation"] == "none"
    assert summary["line_connections"] is False
    assert not ax.lines
    plt.close(fig)
