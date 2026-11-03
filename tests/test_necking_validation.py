"""Check the manufactured deformation and its headless DIC measurements."""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erf

from benchmarks import necking_validation as bench
from benchmarks.synthetic_cases import analytic_texture


@pytest.mark.parametrize("params", list(bench.CASES.values()))
def test_green_lagrange_oracle_matches_numerical_deformation_gradient(params):
    x = np.array([45.0, 75.0, 113.0])
    y = np.array([66.0, 80.0, 93.0])
    h = 1e-4
    up, vp, _ = bench.truth(x, y + h, params)
    um, vm, _ = bench.truth(x, y - h, params)
    _, _, eyy = bench.truth(x, y, params)
    numerical = 0.5 * (((up - um) / (2 * h)) ** 2 +
                       (1 + (vp - vm) / (2 * h)) ** 2 - 1)
    np.testing.assert_allclose(eyy, numerical, atol=3e-10, rtol=0)


@pytest.mark.parametrize("params", list(bench.CASES.values()))
def test_target_pixels_match_independent_bracketed_inverse(params):
    _, target = bench.image_pair(params)
    cx, cy = bench.SHAPE[1] / 2, bench.SHAPE[0] / 2
    for px, py in [(34, 38), (81, 79), (117, 101)]:
        def y_residual(ry):
            d = ry - cy
            return (ry + params["axial"] * d +
                    params["peak"] * params["width"] * np.sqrt(np.pi / 2) *
                    erf(d / (np.sqrt(2) * params["width"])) - py)
        ry = brentq(y_residual, -1000, 1000, xtol=1e-12)
        gaussian = np.exp(-0.5 * ((ry - cy) / params["width"]) ** 2)
        rx = cx + (px - cx) / (1 + params["lateral"] + params["neck"] * gaussian)
        expected = analytic_texture(np.array(rx), np.array(ry), seed=bench.SEED)
        assert target[py, px] == pytest.approx(float(expected), abs=1e-9)


def test_uniform_and_localized_fields_produce_auditable_accuracy():
    uniform = bench.run_case(bench.CASES["uniform"])
    narrow = bench.run_case(bench.CASES["narrow_localization"])
    assert uniform["correlation_valid_count"] / uniform["point_count"] >= 0.9
    assert uniform["displacement_vector_error_px"]["rmse"] < 0.005
    assert uniform["eyy_error"]["rmse"] < 0.0002
    assert narrow["correlation_valid_count"] / narrow["point_count"] >= 0.9
    assert narrow["displacement_vector_error_px"]["rmse"] < 0.5
    assert narrow["eyy_error"]["rmse"] < 0.08
    assert narrow["exact_displacement_windowed_eyy_error"]["count"] > 0
    assert len(narrow["centerline"]) > 0


def test_intact_texture_and_corrupted_patches_are_distinguished():
    occluded = bench.run_case(bench.CASES["broad_localization"], corruption="occlusion")
    unrelated = bench.run_case(bench.CASES["uniform"], corruption="unrelated_texture")
    coverage = occluded["corruption_acceptance"]
    assert coverage["fully_covered"]["count"] > 0
    assert coverage["fully_covered"]["accepted"] == 0
    assert coverage["clear"]["accepted"] == coverage["clear"]["count"]
    assert unrelated["corruption_acceptance"]["no_physical_correspondence"]["accepted"] == 0
    assert unrelated["displacement_vector_error_px"] is None
