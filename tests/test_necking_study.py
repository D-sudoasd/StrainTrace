"""Check known-truth comparisons retain their spatial and validity meaning."""
import copy
import json
import numpy as np
import pytest

import ezdic_core as core
from benchmarks import necking_study as study
from benchmarks import necking_validation as generator
from benchmarks import compare_necking_studies as comparison


def _truth_field_with_missing_peak():
    params = generator.CASES["narrow_localization"]
    X, Y = core.build_poi_grid(study.locked_roi(25), 25, 8, generator.SHAPE)
    u, v, eyy = generator.truth(X, Y, params)
    valid = Y != 80
    field = {
        "X": X, "Y": Y, "U": np.where(valid, u, np.nan), "V": np.where(valid, v, np.nan),
        "u_raw": u.ravel(), "v_raw": v.ravel(), "valid": valid.ravel(),
        "strain_valid": valid.ravel(), "eligible": np.ones(X.size, bool),
        "P": np.zeros((*X.shape, 6)), "Eyy_grid": np.where(valid, eyy, np.nan),
        "invalid_reason": np.where(valid, "", "LOST_TEXTURE").ravel(),
        "strain_invalid_reason": np.where(valid, "", "INVALID_CENTER").ravel(),
    }
    return params, field


def test_subset_comparison_has_exactly_the_same_material_points():
    grids = [core.build_poi_grid(study.locked_roi(size), size, 8, generator.SHAPE)
             for size in (15, 25, 31)]
    for X, Y in grids:
        assert X.shape == Y.shape == (12, 12)
        np.testing.assert_array_equal(X[0], np.arange(32, 121, 8))
        np.testing.assert_array_equal(Y[:, 0], np.arange(32, 121, 8))


def test_rejected_peak_is_reported_against_the_full_domain():
    params, field = _truth_field_with_missing_peak()
    summary, points = study.summarize_field(field, params, generator.PARAMETERS, core)
    whole = summary["regions"]["entire_poi_domain"]
    neck = summary["regions"]["neck_band"]
    assert whole["eligible_count"] == 144
    assert whole["correlation_valid_count"] == 132
    assert whole["true_peak_point_count"] == 12
    assert whole["true_peak_points_with_measured_strain"] == 0
    assert whole["true_peak_full_domain"]["value"] == pytest.approx(.3192)
    assert whole["true_peak_on_measured_support"]["value"] < .3192
    assert whole["measured_peak"]["value"] < .3192
    assert neck["strain_coverage"] == pytest.approx(2/3)
    assert whole["correlation_invalid_reasons"] == {"LOST_TEXTURE": 12}
    assert all(np.isnan(p["Eyy"]) for p in points if p["y_px"] == 80)
    # Full-grid oracle remains independently available across rejected peaks.
    assert all(np.isfinite(p["exact_displacement_full_Eyy"]) for p in points)
    assert all(np.isnan(p["exact_displacement_masked_Eyy"]) for p in points if p["y_px"] == 80)


def test_unknown_rejected_candidates_are_not_labelled_bad_displacements():
    params, field = _truth_field_with_missing_peak()
    summary, _ = study.summarize_field(field, params, generator.PARAMETERS, core)
    candidate = summary["refinement_candidate_diagnostic"]
    assert candidate["available"] is False
    assert candidate["rejected_candidate_error_gt_0_5_px"] is None
    assert candidate["rejected_with_unknown_candidate_count"] == 12
    field["u_candidate"] = field["u_raw"].copy()
    field["v_candidate"] = field["v_raw"].copy()
    summary, _ = study.summarize_field(field, params, generator.PARAMETERS, core)
    candidate = summary["refinement_candidate_diagnostic"]
    assert candidate["available"] is True
    assert candidate["rejected_candidate_error_le_0_5_px"] == 12
    assert candidate["rejected_candidate_error_gt_0_5_px"] == 0


def test_conditional_error_decomposition_uses_the_same_support():
    params, field = _truth_field_with_missing_peak()
    _, points = study.summarize_field(field, params, generator.PARAMETERS, core)
    for point in points:
        measured, oracle, truth = (point[key] for key in
                                  ("Eyy", "exact_displacement_masked_Eyy", "true_Eyy"))
        if np.isfinite(measured) and np.isfinite(oracle):
            assert measured-truth == pytest.approx((measured-oracle)+(oracle-truth), abs=1e-15)


def test_reference_composition_recovers_uniform_affine_displacement():
    result = study.reference_strategy_diagnostic(core)
    rows = [row for row in result["points"] if row["case"] == "uniform"]
    assert len(rows) == 4
    for row in rows:
        for route in ("fixed", "oracle_middle"):
            assert row[route]["candidate_exists"]
            assert row[route]["converged"]
            assert row[route]["total_displacement_vector_error_px"] < 1e-3


def test_research_output_cannot_overwrite_existing_evidence(tmp_path):
    output = tmp_path / "previous.json"
    output.write_text("previous bytes", encoding="utf-8")
    with pytest.raises(FileExistsError):
        study.run(output)
    assert output.read_text(encoding="utf-8") == "previous bytes"


def test_frozen_core_comparison_rejects_different_input_images(tmp_path):
    before = {"schema": study.SCHEMA, "benchmark_sha256": "script", "generator_sha256": "generator",
              "texture_source_sha256": "texture", "environment": {}, "deformation_reference": "fixed",
              "strain_definition": "Green-Lagrange Eyy", "diagnostic_error_cutoff_px": .5,
              "cases": {"narrow": {"input_sha256": "original images"}}}
    after = copy.deepcopy(before)
    after["cases"]["narrow"]["input_sha256"] = "different images"
    a, b = tmp_path/"before.json", tmp_path/"after.json"
    a.write_text(json.dumps(before), encoding="utf-8")
    b.write_text(json.dumps(after), encoding="utf-8")
    with pytest.raises(ValueError, match="input_sha256"):
        comparison.compare(a, b, tmp_path/"comparison.json")
    assert not (tmp_path/"comparison.json").exists()


def test_pointwise_comparison_rejects_changed_numeric_evidence(tmp_path):
    points = tmp_path/"points.csv"
    points.write_text("case,x_px,y_px\nnarrow,32,32\n", encoding="utf-8")
    report = {"pointwise_csv": {"filename": points.name, "sha256": comparison.digest(points), "row_count": 1}}
    points.write_text("case,x_px,y_px\nnarrow,32,40\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content differs"):
        comparison.load_points(tmp_path/"report.json", report)


def test_experimental_policy_difference_requires_explicit_permission():
    before = {"subset_size": 25, "strain_window": 5}
    after = {**before, "experimental_full_search": True}
    with pytest.raises(ValueError, match="explicit opt-in"):
        comparison._parameter_difference(before, after)
    allowed = comparison._parameter_difference(before, after, allow_experimental_policy_difference=True)
    assert allowed == {"experimental_full_search": {"before": False, "after": True}}


@pytest.mark.parametrize("unrelated_setting", ["subset_size", "strain_window", "zncc_min"])
def test_experimental_comparison_cannot_hide_other_parameter_changes(unrelated_setting):
    before = {"subset_size": 25, "strain_window": 5, "zncc_min": .8}
    after = {**before, "experimental_full_search": True, unrelated_setting: before[unrelated_setting]+1}
    with pytest.raises(ValueError, match="beyond experimental_full_search"):
        comparison._parameter_difference(before, after, allow_experimental_policy_difference=True)
