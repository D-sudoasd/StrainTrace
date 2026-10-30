"""Directed quality/validity tests for the core 2-D DIC solver."""

import numpy as np
import pytest

import ezdic_core as core


def _checkerboard(size=128, period=8):
    yy, xx = np.indices((size, size))
    return (((xx // period + yy // period) % 2) * 255).astype(np.float32)


def test_integer_guess_keeps_legacy_tuple_and_exposes_second_peak():
    reference = core.generate_synthetic_speckle(128, 128, seed=17)
    deformed = core.warp_image_translation(reference, 2.3, -1.2)
    legacy = core.integer_cc_guess(reference, deformed, 64, 64, 21, 8)
    diagnostic = core.integer_cc_guess(
        reference, deformed, 64, 64, 21, 8, return_diagnostics=True
    )

    assert len(legacy) == 3
    assert diagnostic["best_peak"] == pytest.approx(legacy[2])
    assert diagnostic["second_peak"] < diagnostic["best_peak"]
    assert diagnostic["peak_margin"] > 0
    assert diagnostic["best_to_second_peak_ratio"] > 1
    assert diagnostic["second_to_best_peak_ratio"] < 1


def test_checkerboard_repeated_peak_is_ambiguous_and_not_accepted():
    reference = _checkerboard()
    deformed = core.warp_image_translation(reference, 8.0, 0.0)
    field = core.run_2d_dic(
        reference,
        deformed,
        (20, 20, 88, 88),
        subset_size=21,
        step=8,
        search_radius=16,
        zncc_min=0.75,
    )

    assert np.asarray(field["peak_is_ambiguous"], dtype=bool).all()
    assert not np.asarray(field["valid"], dtype=bool).any()
    assert np.all(np.asarray(field["invalid_reason"], dtype=object) == "AMBIGUOUS_PEAK")
    assert field["quality_summary"]["ambiguous_count"] == len(field["valid"])
    assert field["quality_summary"]["scientific_ok"] is False


def test_real_speckle_quality_does_not_regress_and_has_audit_fields():
    reference = core.generate_synthetic_speckle(128, 128, seed=17)
    deformed = core.warp_image_translation(reference, 2.3, -1.2)
    field = core.run_2d_dic(
        reference,
        deformed,
        (18, 18, 92, 92),
        subset_size=21,
        step=10,
        search_radius=10,
        zncc_min=0.75,
    )
    valid = np.asarray(field["valid"], dtype=bool)
    assert valid.mean() >= 0.90
    for key in (
        "best_peak",
        "second_peak",
        "peak_margin",
        "peak_ratio",
        "residual_rms",
        "iterations",
        "hessian_condition_number",
        "converged",
        "stop_reason",
        "u_raw",
        "v_raw",
        "strain_valid",
    ):
        assert key in field
    assert np.isfinite(np.asarray(field["residual_rms"], dtype=float)[valid]).all()
    assert np.isfinite(np.asarray(field["hessian_condition_number"], dtype=float)[valid]).all()
    # A solver may converge numerically and then be rejected as ambiguous.
    summary = field["quality_summary"]
    assert 0 < summary["converged_count"] <= summary["convergence_known_count"]
    assert 0 < summary["accepted_converged_count"] <= int(valid.sum())
    assert summary["converged_count"] == summary["accepted_converged_count"] + summary["rejected_converged_count"]


def test_convergence_is_not_high_zncc_only_and_uses_affine_increment():
    reference = core.generate_synthetic_speckle(96, 96, seed=5)
    deformed = core.warp_image_translation(reference, 1.37, -0.62)
    result = core.refine_subset_icgn(
        reference,
        deformed,
        48,
        48,
        21,
        p0=[1.0, 0.02, 0.01, -0.8, -0.01, 0.02],
        max_iter=25,
        tol=1e-3,
    )

    assert result is not None
    assert result["iterations"] > 1
    assert result["converged"] is True
    assert "zncc" not in result["stop_reason"].lower()
    assert np.isfinite(result["increment_norm_px"])
    assert np.isfinite(result["hessian_condition_number"])


def test_strain_valid_is_distinct_from_correlation_valid_and_summary_gates():
    coords = np.arange(3, dtype=float)
    x, y = np.meshgrid(coords, coords)
    u = np.ones_like(x)
    v = np.ones_like(y)
    u[1, 1] = np.nan
    v[1, 1] = np.nan
    strains = core.compute_strain_fields(x, y, u, v, window=3)
    field = {
        "valid": np.ones(9, dtype=bool),
        "strain_valid": strains["strain_valid"].ravel(),
        "invalid_reason": np.full(9, "", dtype=object),
        "peak_is_ambiguous": np.zeros(9, dtype=bool),
    }
    summary = core.field_quality_summary(field)

    assert summary["correlation_valid_fraction"] == pytest.approx(1.0)
    assert summary["strain_valid_fraction"] < 1.0
    assert summary["scientific_ok"] is False
    assert "strain_valid_fraction_below_threshold" in summary["scientific_reasons"]


def test_nonconverged_is_diagnostic_by_default_and_explicit_gate_remains_available():
    reference = core.generate_synthetic_speckle(128, 128, seed=3)
    deformed = core.warp_image_translation(reference, 1.37, -0.62)
    default_field = core.run_2d_dic(
        reference, deformed, (18, 18, 92, 92), subset_size=21, step=10
    )
    strict_field = core.run_2d_dic(
        reference,
        deformed,
        (18, 18, 92, 92),
        subset_size=21,
        step=10,
        reject_nonconverged=True,
    )
    assert default_field["reject_nonconverged"] is False
    assert strict_field["reject_nonconverged"] is True
    assert np.asarray(default_field["valid"], dtype=bool).sum() >= np.asarray(
        strict_field["valid"], dtype=bool
    ).sum()
    assert "converged_fraction" in default_field["quality_summary"]
    assert "invalid_reason_histogram" in strict_field["quality_summary"]


def test_residual_threshold_is_explicit_and_reports_rejection_reason():
    reference = core.generate_synthetic_speckle(96, 96, seed=11)
    deformed = core.warp_image_translation(reference, 0.8, -0.4)
    field = core.run_2d_dic(
        reference,
        deformed,
        (18, 18, 60, 60),
        subset_size=15,
        step=8,
        max_residual_rms=1e-8,
    )
    assert field["quality_summary"]["residual_threshold_enabled"] is True
    assert field["quality_summary"]["max_residual_rms"] == pytest.approx(1e-8)
    assert np.all(np.asarray(field["invalid_reason"], dtype=object) == "RESIDUAL_ABOVE_THRESHOLD")


def test_rejected_ambiguous_points_do_not_fail_scientific_gate_by_themselves():
    valid = np.ones(20, dtype=bool)
    valid[0] = False
    strain_valid = valid.copy()
    field = {
        "valid": valid,
        "strain_valid": strain_valid,
        "invalid_reason": np.array(["AMBIGUOUS_PEAK"] + [""] * 19, dtype=object),
        "peak_is_ambiguous": np.array([True] + [False] * 19, dtype=bool),
    }
    summary = core.field_quality_summary(field)
    assert summary["ambiguous_rejected_count"] == 1
    assert summary["ambiguous_accepted_count"] == 0
    assert summary["scientific_ok"] is True
    assert "ambiguous_peaks_present" not in summary["scientific_reasons"]
