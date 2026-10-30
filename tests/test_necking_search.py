"""Regression coverage for local tracking after a global affine initialization."""

import numpy as np
import pytest

import ezdic_core as core
from benchmarks.necking_search_validation import (
    CENTER,
    SETTINGS,
    displacement_truth,
    evaluate_case,
    exact_global_seed,
)


@pytest.fixture(params=("localized_shear_band", "uniform_affine"))
def field_and_case(request, monkeypatch):
    case_name = request.param
    # Exercise auto selection deterministically. The localization is not part
    # of the global seed, as would happen with RANSAC background inliers.
    monkeypatch.setattr(
        core,
        "estimate_dic_initial_transform",
        lambda *_args, **_kwargs: exact_global_seed(case_name),
    )
    field, metrics = evaluate_case(case_name)
    return case_name, field, metrics


def test_global_initialization_keeps_requested_search_radius(field_and_case):
    _name, field, _metrics = field_and_case
    assert field["initial_transform"] is not None
    assert field["effective_search_radius_px"] == SETTINGS["search_radius"]


def test_localized_and_uniform_known_displacement_accuracy(field_and_case):
    case_name, field, metrics = field_and_case
    expected_u, expected_v = displacement_truth(case_name, field["x"], field["y"])
    valid = np.asarray(field["valid"], dtype=bool)
    errors = np.hypot(field["u_raw"] - expected_u, field["v_raw"] - expected_v)
    if case_name == "localized_shear_band":
        band = np.abs(np.asarray(field["y"]) - CENTER[1]) < 35.0
        assert band.sum() == 24
        assert (valid & band).sum() >= 22
        assert np.mean(errors[valid & band]) < 0.2
        assert np.max(errors[valid & band]) < 0.75
    else:
        assert valid.all()
        assert np.mean(errors[valid]) < 0.01
        assert np.max(errors[valid]) < 0.02
    assert metrics["valid_error_over_1_px_count"] == 0
