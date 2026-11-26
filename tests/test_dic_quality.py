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
    assert 0 < field["quality_summary"]["converged_count"] <= int(valid.sum())


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


@pytest.mark.parametrize("eligible", [np.ones(9, dtype=bool), np.zeros(9, dtype=bool)])
def test_zero_fraction_thresholds_never_accept_a_field_without_valid_points(eligible):
    field = {"eligible": eligible, "valid": np.zeros(9, dtype=bool), "strain_valid": np.zeros(9, dtype=bool)}
    summary = core.field_quality_summary(
        field, min_correlation_valid_fraction=0.0, min_strain_valid_fraction=0.0
    )
    assert summary["scientific_ok"] is False
    assert summary["correlation_valid_count"] == 0
    assert summary["strain_valid_count"] == 0
    assert "no_valid_strain_points" in summary["scientific_reasons"]


def test_strain_validity_requires_eligible_valid_correlation():
    field = {
        "eligible": np.array([True, True, False]),
        "valid": np.array([True, False, True]),
        "strain_valid": np.array([False, True, True]),
        "Exx": np.zeros(3), "Eyy": np.zeros(3), "Exy": np.zeros(3),
    }
    summary = core.field_quality_summary(
        field, min_correlation_valid_fraction=0.0, min_strain_valid_fraction=0.0
    )
    assert summary["correlation_valid_count"] == 1
    assert summary["strain_valid_count"] == 0
    assert summary["strain_valid_fraction"] == 0.0
    assert summary["scientific_ok"] is False
    assert not core.fullfield_field_has_finite_strain(field)


def test_measured_field_coverage_counts_complete_finite_vectors_and_tensors():
    field = dict(valid=[True] * 9, eligible=[True] * 9, strain_valid=[True] * 9,
                 u=[0.] * 9, v=[np.nan] + [0.] * 8, Exx=[0.] * 9,
                 Eyy=[0., 0.] + [np.nan] * 7, Exy=[0.] * 9)
    report = core.field_quality_summary(field)
    assert report["correlation_valid_count"] == 8
    assert report["strain_valid_count"] == 1
    assert report["scientific_ok"] is False
    assert core.fullfield_field_has_finite_strain(field)
    field["eligible"] = [True]
    assert core.field_quality_summary(field)["eligible_point_count"] == 0
    assert not core.fullfield_field_has_finite_strain(field)


def test_fullfield_finite_strain_check_retains_legacy_fallback():
    field = {"valid": np.array([True]), "Exx": [0.0], "Eyy": [0.0], "Exy": [0.0]}
    assert core.fullfield_field_has_finite_strain(field)
    field["strain_valid"] = np.array([False])
    assert not core.fullfield_field_has_finite_strain(field)


@pytest.mark.parametrize("threshold", ["min_correlation_valid_fraction", "min_strain_valid_fraction"])
@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1])
def test_invalid_quality_fraction_cannot_silently_pass(threshold, value):
    field = {"valid": np.ones(3, dtype=bool), "strain_valid": np.ones(3, dtype=bool)}
    with pytest.raises(ValueError, match=threshold):
        core.field_quality_summary(field, **{threshold: value})


@pytest.mark.parametrize("option", ["zncc_min", "min_correlation_valid_fraction", "min_strain_valid_fraction"])
def test_dic_rejects_nonfinite_quality_threshold_before_running(option):
    image = core.generate_synthetic_speckle(48, 48, seed=17)
    with pytest.raises(ValueError, match=option):
        core.run_2d_dic(image, image, (8, 8, 32, 32), subset_size=9, step=8,
                        initialization="local", **{option: float("nan")})


def test_gui_permitted_low_texture_cannot_pass_fullfield_scientific_gate(tmp_path):
    image = core.generate_synthetic_speckle(80, 80, seed=17).astype(np.uint8)
    patch = image[12:68, 12:68].astype(float)
    image[12:68, 12:68] = np.clip(100 + .035 * (patch - 100), 0, 255).astype(np.uint8)
    paths = [tmp_path / "reference.png", tmp_path / "unchanged.png"]
    for path in paths:
        core.write_image_checked(path, image)
    result = core.run_fullfield_sequence({
        "image_paths": paths, "output_dir": tmp_path / "output", "field_roi": (12, 12, 56, 56),
        "subset_size": 15, "step": 10, "search_radius": 5, "strain_window": 3,
        "initialization": "local", "_allow_low_texture_preflight": True,
    })
    manifest = result["manifest"]
    assert manifest["texture_preflight"]["field_roi"]["code"] == "LOW_TEXTURE"
    assert result["scientific_ok"] is False
    assert result["status"] == "scientific_gate_failed"
    assert "texture_preflight_failed" in manifest["scientific_gate"]["reasons"]
    assert manifest["frames"][0]["scientific_valid"] is False
    assert manifest["frames"][0]["scientific_ok"] is False


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
