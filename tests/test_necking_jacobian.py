"""Regression checks for the fitted 2-D deformation-gradient determinant."""

import numpy as np
import pytest

import ezdic_core as core
from benchmarks.synthetic_cases import analytic_texture


def _affine_displacements(F):
    Y, X = np.mgrid[0:9, 0:9].astype(float)
    U = (F[0, 0] - 1) * X + F[0, 1] * Y
    V = F[1, 0] * X + (F[1, 1] - 1) * Y
    return X, Y, U, V


@pytest.mark.parametrize(
    "F",
    [
        np.eye(2),
        np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]]),
        np.array([[1.2, 0.25], [-0.1, 0.8]]),
        np.diag([1e-9, 1.0]),
    ],
)
def test_positive_jacobian_preserves_affine_strain(F):
    field = core.compute_strain_fields(*_affine_displacements(F), degree=1, robust=False)
    assert field["strain_valid"].all()
    np.testing.assert_allclose(field["deformation_jacobian"], np.linalg.det(F), atol=3e-15)
    expected = 0.5 * (F.T @ F - np.eye(2))
    for key, value in (("Exx", expected[0, 0]), ("Eyy", expected[1, 1]), ("Exy", expected[0, 1])):
        np.testing.assert_allclose(field[key], value, atol=3e-15)


def test_reflection_rejects_strain_but_preserves_fitted_gradient():
    # Green--Lagrange strain for a pure reflection is zero, so testing its
    # magnitude alone cannot identify this nonphysical 2-D mapping.
    F = np.diag([-1.0, 1.0])
    field = core.compute_strain_fields(*_affine_displacements(F), degree=1, robust=False)
    assert not field["strain_valid"].any()
    assert np.all(field["strain_invalid_reason"] == "NONPOSITIVE_JACOBIAN")
    for key in ("Exx", "Eyy", "Exy", "exx", "eyy", "exy"):
        assert np.isnan(field[key]).all()
    np.testing.assert_allclose(field["dudx"], -2.0, atol=1e-14)
    np.testing.assert_allclose(field["deformation_jacobian"], -1.0, atol=1e-14)


def test_singular_fit_does_not_become_valid_through_roundoff():
    field = core.compute_strain_fields(
        *_affine_displacements(np.diag([0.0, 1.0])), degree=1, robust=False
    )
    assert not field["strain_valid"].any()
    assert set(field["strain_invalid_reason"].ravel()) <= {
        "NONPOSITIVE_JACOBIAN", "NEAR_SINGULAR_JACOBIAN"
    }
    assert np.nanmax(np.abs(field["deformation_jacobian"])) < 4e-15
    assert np.isnan(field["Exx"]).all()


def test_fitted_jacobian_and_gradients_are_exported(tmp_path, monkeypatch):
    y, x = np.indices((96, 96), dtype=float)
    reference = analytic_texture(x, y, seed=17)
    deformed = analytic_texture(x - 0.37, y + 0.24, seed=17)
    field = core.run_2d_dic(
        reference, deformed, (18, 18, 60, 60), subset_size=21, step=8,
        search_radius=5, max_iter=40, conv_tol=1e-5, initialization="local",
    )
    assert np.asarray(field["strain_valid"]).any()
    determinant = np.asarray(field["deformation_jacobian"])
    expected = ((1 + field["strain_fit_dudx"]) * (1 + field["strain_fit_dvdy"])
                - field["strain_fit_dudy"] * field["strain_fit_dvdx"])
    np.testing.assert_allclose(determinant, expected, equal_nan=True)

    monkeypatch.setattr(core, "plot_dic_field_map", lambda _field, path, **_kwargs: path)
    paths = core.export_dic_field_outputs(field, tmp_path, stem="test")
    table = core.pd.read_csv(paths["csv"])
    np.testing.assert_allclose(table["deformation_jacobian"], determinant, equal_nan=True)
    with np.load(paths["arrays"], allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["deformation_jacobian"], determinant, equal_nan=True)
        for key in ("strain_fit_dudx", "strain_fit_dudy", "strain_fit_dvdx", "strain_fit_dvdy"):
            np.testing.assert_allclose(arrays[key], field[key], equal_nan=True)

    # Fixed eight-decimal formatting would erase the sign of a determinant at
    # the near-singular threshold, so the TXT diagnostic retains precision.
    for diagnostic in (-1e-15, 1e-15):
        tiny = dict(field)
        tiny["deformation_jacobian"] = np.full(determinant.shape, diagnostic)
        text_path = core.write_dic_field_txt(tiny, tmp_path / "tiny.txt")
        text_table = core.pd.read_csv(text_path, sep="\t")
        np.testing.assert_allclose(text_table["deformation_jacobian"], diagnostic, rtol=0, atol=1e-30)
