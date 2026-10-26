"""Analytic regressions for reference coordinates and strain objectivity."""

import numpy as np
import pytest

import ezdic_core as core
from benchmarks.synthetic_cases import analytic_texture


def _affine_image_pair(F, translation, shape=(144, 168)):
    """Evaluate a continuous texture at the exact inverse affine mapping."""
    y, x = np.indices(shape, dtype=float)
    coordinates = np.stack([x, y]).reshape(2, -1)
    center = np.array([shape[1] / 2, shape[0] / 2])[:, None]
    original = np.linalg.solve(F, coordinates-center-translation[:, None])+center
    reference = analytic_texture(x, y)
    deformed = analytic_texture(*original.reshape(2, *shape))
    return reference, deformed, center[:, 0]


@pytest.mark.parametrize("solver", [core.refine_subset_icgn, core.refine_subset_iclm])
@pytest.mark.parametrize("F", [np.eye(2), np.array([[1.008, .006], [-.004, .997]])])
def test_fractional_reference_center_recovers_displacement_at_that_center(solver, F):
    translation = np.array([.37, -.24])
    reference, deformed, center = _affine_image_pair(F, translation)
    position = np.array([80.3, 70.4])
    expected = (F-np.eye(2))@(position-center)+translation
    fit = solver(reference, deformed, *position, 31, max_iter=40, tol=1e-5)

    assert fit is not None and fit["converged"]
    np.testing.assert_allclose([fit["u"], fit["v"]], expected, atol=1e-4, rtol=0)
    np.testing.assert_allclose(fit["p"][[1, 2, 4, 5]], (F-np.eye(2)).ravel(), atol=1e-5, rtol=0)
    assert fit["zncc"] > .99999


def test_fractional_reference_coefficients_are_reused_without_changing_integer_path(monkeypatch):
    reference, deformed, _ = _affine_image_pair(np.eye(2), np.array([.37, -.24]))
    prepared = core._prepare_dic_images(reference, deformed)
    assert "reference_coefficients" not in prepared
    original = core.ndimage.spline_filter
    calls = []

    def count_filter(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(core.ndimage, "spline_filter", count_filter)
    for position in ((80, 70), (80.3, 70.4), (82.7, 72.1)):
        fit = core.refine_subset_icgn(reference, deformed, *position, 31, _prepared=prepared)
        assert fit is not None
    assert len(calls) == 1


@pytest.mark.parametrize("solver", [core.refine_subset_icgn, core.refine_subset_iclm])
def test_fractional_reference_subset_outside_image_is_rejected(solver):
    reference, deformed, _ = _affine_image_pair(np.eye(2), np.array([.37, -.24]))
    # Rounding x to 15 would conceal the unsupported leftmost reference pixel.
    assert solver(reference, deformed, 14.8, 70.4, 31) is None


@pytest.mark.parametrize("sigma", [0., 1., 2.])
def test_smoothing_preserves_rigid_rotation_and_translation(sigma):
    y, x = np.mgrid[0:15, 0:17].astype(float)
    theta = .2
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    displacement = (rotation-np.eye(2))@np.stack([x, y]).reshape(2, -1)
    u, v = displacement.reshape(2, *x.shape)
    u, v = u+3.7, v-2.4
    field = core.compute_strain_fields(x, y, u, v, smooth_sigma=sigma)

    assert field["strain_valid"].all()
    np.testing.assert_allclose(field["U"], u, atol=1e-12, rtol=0)
    np.testing.assert_allclose(field["V"], v, atol=1e-12, rtol=0)
    for component in ("Exx", "Eyy", "Exy"):
        np.testing.assert_allclose(field[component], 0., atol=1e-12, rtol=0)
    # Infinitesimal strain is not rotation-objective at finite angles.
    np.testing.assert_allclose(field["exx"], np.cos(theta)-1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(field["eyy"], np.cos(theta)-1, atol=1e-12, rtol=0)


def test_smoothing_uses_physical_grid_coordinates_for_affine_strain():
    y, x = np.mgrid[0:15, 0:17].astype(float)
    x = 2*x+.02*x*x+1e6
    y = 3*y+.01*y*y+2e6
    F = np.array([[1.08, .12], [-.04, .97]])
    u = (F[0, 0]-1)*(x-1e6)+F[0, 1]*(y-2e6)+.37
    v = F[1, 0]*(x-1e6)+(F[1, 1]-1)*(y-2e6)-.24
    field = core.compute_strain_fields(x, y, u, v, smooth_sigma=1.)
    green = .5*(F.T@F-np.eye(2))
    infinitesimal = .5*(F+F.T)-np.eye(2)

    assert field["strain_valid"].all()
    for key, expected in (("Exx", green[0, 0]), ("Eyy", green[1, 1]), ("Exy", green[0, 1]),
                          ("exx", infinitesimal[0, 0]), ("eyy", infinitesimal[1, 1]), ("exy", infinitesimal[0, 1])):
        np.testing.assert_allclose(field[key], expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize("missing", ["u", "v", "x", "y"])
def test_smoothing_cannot_cross_gap_in_either_displacement_or_coordinate(missing):
    y, x = np.mgrid[0:15, 0:17].astype(float)
    u = .01*x+np.where(x > 8, 4., 0.)
    v = .002*y+np.where(x > 8, 2., 0.)
    arrays = {"x": x, "y": y, "u": u, "v": v}
    arrays[missing][:, 8] = np.nan
    field = core.compute_strain_fields(x, y, u, v, window=7, smooth_sigma=1.)

    assert not field["strain_valid"][:, 8].any()
    assert np.isnan(field["U"][:, 8]).all()
    assert np.isnan(field["V"][:, 8]).all()
    assert np.isnan(field["Exx"][:, 8]).all()
    np.testing.assert_allclose(field["exx"][:, [7, 9]], .01, atol=1e-11, rtol=0)
    np.testing.assert_allclose(field["eyy"][:, [7, 9]], .002, atol=1e-11, rtol=0)


def test_smoothing_does_not_create_strain_support_on_single_row():
    y, x = np.mgrid[0:7, 0:9].astype(float)
    u, v = .01*x, .002*y
    u[:3] = u[4:] = np.nan
    v[:3] = v[4:] = np.nan
    field = core.compute_strain_fields(x, y, u, v, window=7, smooth_sigma=1.)
    assert not field["strain_valid"].any()
    assert np.isnan(field["Exx"]).all()
    np.testing.assert_allclose(field["U"][3], u[3], atol=1e-12, rtol=0)


def test_two_roi_distance_and_axis_projection_have_distinct_rotation_response():
    theta = .2
    roi1 = (20., 30., 25, 25)
    roi2 = (120., 30., 25, 25)
    moved1 = (23., 27., 25, 25)
    moved2 = (23.+100*np.cos(theta), 27.+100*np.sin(theta), 25, 25)
    distance_strain = core.length_between(moved1, moved2, "distance")/core.length_between(roi1, roi2, "distance")-1
    projected_strain = core.length_between(moved1, moved2, "x")/core.length_between(roi1, roi2, "x")-1
    assert distance_strain == pytest.approx(0., abs=1e-14)
    assert projected_strain == pytest.approx(np.cos(theta)-1, abs=1e-14)
