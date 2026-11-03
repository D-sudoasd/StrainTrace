"""Known-displacement regressions for search after global initialization."""

import numpy as np
import pytest

import ezdic_core as core
from benchmarks.necking_search_validation import (
    CENTER, SETTINGS, displacement_truth, evaluate_case, exact_global_seed,
)


@pytest.mark.parametrize("case_name", ["localized_shear_band", "uniform_affine"])
def test_global_initialization_preserves_local_search_and_truth(case_name, monkeypatch):
    # The predictor contains the background transform, not the local band.
    monkeypatch.setattr(core, "estimate_dic_initial_transform",
                        lambda *_a, **_k: exact_global_seed(case_name))
    field, metrics = evaluate_case(case_name, experimental_full_search=True)
    assert field["effective_search_radius_px"] == SETTINGS["search_radius"]
    expected_u, expected_v = displacement_truth(case_name, field["x"], field["y"])
    valid = np.asarray(field["valid"], dtype=bool)
    errors = np.hypot(field["u_raw"]-expected_u, field["v_raw"]-expected_v)
    assert valid.all()
    assert field["quality_summary"]["scientific_ok"] is False
    assert "experimental_search_unvalidated" in field["quality_summary"]["scientific_reasons"]
    assert metrics["accepted_error_over_0_5_px_count"] == 0
    if case_name == "localized_shear_band":
        band = np.abs(field["y"]-CENTER[1]) < 35.0
        assert int(band.sum()) == 24
        assert np.mean(errors[band]) < .2
        assert np.max(errors[band]) < .5
    else:
        assert np.mean(errors) < .01
        assert np.max(errors) < .02


def test_default_keeps_established_global_radius(monkeypatch):
    monkeypatch.setattr(core, "estimate_dic_initial_transform",
                        lambda *_a, **_k: exact_global_seed("localized_shear_band"))
    field, _ = evaluate_case("localized_shear_band")
    assert field["experimental_full_search"] is False
    assert field["effective_search_radius_px"] == 3
    assert field["competing_peak_policy"] == "disabled"


def test_default_identity_field_keeps_existing_coverage():
    image = core.generate_synthetic_speckle(128, 128, seed=17).astype("uint8")
    normalized = core.normalize_dic_float(image, core.compute_reference_normalization(image))
    field = core.run_2d_dic(normalized, normalized, (20, 20, 88, 88),
                            subset_size=15, step=12, strain_window=5, search_radius=12)
    assert field["valid"].all() and field["strain_valid"].all()
    assert field["quality_summary"]["scientific_ok"] is True
    assert field["competing_peak_policy"] == "disabled"


@pytest.mark.parametrize("value", ["false", 1, None])
def test_experimental_switch_requires_an_explicit_boolean(value):
    image = core.generate_synthetic_speckle(64, 64, seed=3)
    with pytest.raises(ValueError, match="experimental_full_search must be a boolean"):
        core.run_2d_dic(image, image, (10, 10, 40, 40), experimental_full_search=value)
