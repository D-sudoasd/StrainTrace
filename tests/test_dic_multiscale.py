"""Known-translation gates for the coarse-to-fine 2-D DIC path."""

import json

import numpy as np
import pytest

import ezdic_core as core


def test_single_level_is_numerically_identical_to_legacy_default():
    reference = core.generate_synthetic_speckle(128, 128, seed=3)
    deformed = core.warp_image_translation(reference, 1.37, -0.62)
    kwargs = {"subset_size": 21, "step": 10, "search_radius": 8}
    default = core.run_2d_dic(reference, deformed, (18, 18, 92, 92), **kwargs)
    explicit = core.run_2d_dic(
        reference,
        deformed,
        (18, 18, 92, 92),
        pyramid_levels=1,
        pyramid_scale=0.5,
        **kwargs,
    )
    for key in ("u_raw", "v_raw", "zncc", "valid", "strain_valid", "peak_margin"):
        assert np.array_equal(default[key], explicit[key], equal_nan=True)
    assert explicit["pyramid_levels_requested"] == 1
    assert explicit["pyramid_levels_used"] == 1
    assert len(explicit["pyramid_level_diagnostics"]) == len(explicit["x"])


def test_three_level_recovers_large_translation_inside_safe_roi():
    reference = core.generate_synthetic_speckle(256, 256, seed=17)
    deformed = core.warp_image_translation(reference, 25.0, -18.0)
    field = core.run_2d_dic(
        reference,
        deformed,
        (36, 36, 184, 184),
        subset_size=21,
        step=20,
        search_radius=10,
        pyramid_levels=3,
        pyramid_scale=0.5,
    )
    valid = np.asarray(field["valid"], dtype=bool)
    error = np.hypot(
        np.asarray(field["u_raw"], dtype=float) - 25.0,
        np.asarray(field["v_raw"], dtype=float) + 18.0,
    )
    assert valid.mean() >= 0.95
    assert np.sqrt(np.mean(error[valid] ** 2)) <= 0.05
    assert np.percentile(error[valid], 95) <= 0.10
    assert field["pyramid_levels_requested"] == 3
    assert field["pyramid_levels_used"] == 3
    assert len(field["pyramid_level_diagnostics"]) == len(field["x"])
    assert all(len(levels) == 3 for levels in field["pyramid_level_diagnostics"])
    assert all(level["pad"] == 0 for levels in field["pyramid_level_diagnostics"] for level in levels)
    json.dumps(field["pyramid_level_diagnostics"])


def test_one_level_does_not_fake_pass_large_translation():
    reference = core.generate_synthetic_speckle(256, 256, seed=17)
    deformed = core.warp_image_translation(reference, 25.0, -18.0)
    field = core.run_2d_dic(
        reference,
        deformed,
        (36, 36, 184, 184),
        subset_size=21,
        step=20,
        search_radius=10,
        pyramid_levels=1,
        initialization="local",
    )
    valid = np.asarray(field["valid"], dtype=bool)
    error = np.hypot(
        np.asarray(field["u_raw"], dtype=float) - 25.0,
        np.asarray(field["v_raw"], dtype=float) + 18.0,
    )
    assert valid.mean() < 0.95 or np.sqrt(np.mean(error[valid] ** 2)) > 0.05


@pytest.mark.parametrize(
    ("levels", "scale"),
    [(0, 0.5), (3, 0.0), (3, 1.0), (3, 1.2), (1.5, 0.5)],
)
def test_pyramid_parameter_validation(levels, scale):
    reference = core.generate_synthetic_speckle(64, 64, seed=7)
    deformed = core.warp_image_translation(reference, 0.5, -0.25)
    with pytest.raises(ValueError, match="pyramid"):
        core.run_2d_dic(
            reference,
            deformed,
            (12, 12, 40, 40),
            subset_size=15,
            step=8,
            pyramid_levels=levels,
            pyramid_scale=scale,
        )


def test_too_many_levels_fail_with_explicit_boundary_error():
    reference = core.generate_synthetic_speckle(32, 32, seed=2)
    deformed = core.warp_image_translation(reference, 1.0, -0.5)
    with pytest.raises(ValueError, match="pyramid level|subset_size"):
        core.run_2d_dic(
            reference,
            deformed,
            (8, 8, 16, 16),
            subset_size=15,
            step=4,
            pyramid_levels=6,
            pyramid_scale=0.5,
        )


def test_affine_seed_survives_coarse_to_fine_scaling(monkeypatch):
    reference = core.generate_synthetic_speckle(64, 64, seed=4)
    deformed = core.warp_image_translation(reference, 2.0, -1.0)
    captured_p0 = []

    def fake_integer(*_args, **_kwargs):
        return {
            "u": 0.0,
            "v": 0.0,
            "zncc": 1.0,
            "best_peak": 1.0,
            "second_peak": 0.2,
            "peak_margin": 0.8,
            "peak_ratio": 5.0,
        }

    def fake_refine(*_args, **kwargs):
        p0 = np.asarray(kwargs["p0"], dtype=float).copy()
        captured_p0.append(p0)
        p = p0.copy()
        p[[1, 2, 4, 5]] = [0.01, -0.02, 0.03, -0.04]
        return {
            "u": float(p[0]),
            "v": float(p[3]),
            "p": p,
            "zncc": 1.0,
            "residual_rms": 0.0,
            "hessian_condition_number": 10.0,
            "iterations": 2,
            "converged": True,
            "stop_reason": "converged_all_affine_increment",
        }

    monkeypatch.setattr(core, "integer_cc_guess", fake_integer)
    monkeypatch.setattr(core, "refine_subset_icgn", fake_refine)
    core.run_2d_dic(
        reference,
        deformed,
        (12, 12, 40, 40),
        subset_size=15,
        step=8,
        search_radius=8,
        pyramid_levels=3,
    )

    assert captured_p0
    assert any(np.max(np.abs(p0[[1, 2, 4, 5]])) > 0 for p0 in captured_p0)
