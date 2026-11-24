"""Scientific-state and provenance regression tests for the 1-D core."""

import numpy as np
import pandas as pd
import pytest

import ezdic_core as core
import dic_virtual_extensometer_gui_v7_multi_roi_range as gui


def _params(**overrides):
    params = {
        "search_radius_base": 30,
        "hard_corr": 0.55,
        "soft_corr": 0.35,
        "enable_adaptive": True,
        "use_prev_frame_template": False,
        "template_alpha": 0.7,
        "max_frame_jump": None,
        "enable_fb_check": True,
        "fb_tolerance": 12.0,
        "pixel_size_mm": None,
    }
    params.update(overrides)
    return params


def _group():
    return {
        "name": "G01",
        "roi1": (50, 100, 25, 25),
        "roi2": (150, 100, 25, 25),
        "role": "none",
        "selected_mode": "x",
        "actual_mode": "x",
    }


def test_hard_and_adaptive_paths_cannot_bypass_forward_backward_failure(monkeypatch):
    reference = core.generate_synthetic_speckle(256, 256, seed=17)
    deformed = core.warp_image_translation(reference, 2.0, -1.0)

    def fake_match(_image, rect, _template, _radius, **_kwargs):
        return {
            "candidate_rect": (float(rect[0] + 2.0), float(rect[1] - 1.0), rect[2], rect[3]),
            "score": 0.80,
            "best_peak": 0.80,
            "second_peak": 0.10,
            "peak_margin": 0.70,
            "peak_ratio": 8.0,
            "best_to_second_peak_ratio": 8.0,
            "second_to_best_peak_ratio": 0.125,
            "peak_is_ambiguous": False,
        }

    monkeypatch.setattr(core, "match_template_candidate", fake_match)
    monkeypatch.setattr(core, "forward_backward_error", lambda *_args, **_kwargs: (float("inf"), float("nan")))
    for hard_corr in (0.55, 0.95):
        state = core.initialize_extensometer_group_state(reference, _group())
        core.track_extensometer_group_frame(state, reference, 0, "ref", _params(hard_corr=hard_corr))
        snapshot = {
            "rect1": state["last_good_rect1"],
            "rect2": state["last_good_rect2"],
            "strain": state["last_valid_strain"],
            "image": state["last_good_img8"].copy(),
            "template1": state["template1"].copy(),
            "template2": state["template2"].copy(),
        }
        row, _ = core.track_extensometer_group_frame(
            state,
            deformed,
            1,
            "def",
            _params(hard_corr=hard_corr, soft_corr=0.35),
        )
        assert row["accepted"] is False
        assert row["accept_mode"] == "rejected"
        assert row["tracking_status_code"] == "FB_FAILED"
        assert row["strain_valid"] is False
        assert np.isnan(row["length_px"])
        assert np.isnan(row["engineering_strain"])
        assert state["last_good_rect1"] == snapshot["rect1"]
        assert state["last_good_rect2"] == snapshot["rect2"]
        assert state["last_valid_strain"] == snapshot["strain"]
        assert np.array_equal(state["last_good_img8"], snapshot["image"])
        assert np.array_equal(state["template1"], snapshot["template1"])
        assert np.array_equal(state["template2"], snapshot["template2"])


def test_fixed_template_rigid_translation_has_zero_gauge_strain():
    reference = core.generate_synthetic_speckle(256, 256, seed=17)
    deformed = core.warp_image_translation(reference, 2.0, -1.0)
    state = core.initialize_extensometer_group_state(reference, _group())
    params = _params(enable_fb_check=True)
    core.track_extensometer_group_frame(state, reference, 0, "ref", params)
    row, _ = core.track_extensometer_group_frame(state, deformed, 1, "def", params)
    assert row["accepted"] is True
    assert row["strain_valid"] is True
    assert row["engineering_strain"] == pytest.approx(0.0, abs=2e-3)
    assert row["template_policy"] == "fixed_reference"
    assert row["fb_status"] == "passed"


def test_known_ten_percent_uniform_stretch_is_recovered():
    reference = core.generate_synthetic_speckle(256, 256, seed=9)
    deformed = core.warp_image_deformation_gradient(reference, np.array([[1.10, 0.0], [0.0, 1.0]]))
    state = core.initialize_extensometer_group_state(reference, _group())
    params = _params(search_radius_base=35, enable_fb_check=True)
    core.track_extensometer_group_frame(state, reference, 0, "ref", params)
    row, _ = core.track_extensometer_group_frame(state, deformed, 1, "def", params)
    assert row["accepted"] is True
    assert row["strain_valid"] is True
    assert row["engineering_strain"] == pytest.approx(0.10, abs=2e-3)


def test_periodic_texture_is_rejected_but_random_speckle_is_allowed():
    xx = np.indices((128, 128))[1]
    stripe = (128.0 + 100.0 * np.sin(xx * 2.0 * np.pi / 12.0)).astype(np.float32)
    group = {**_group(), "roi1": (20, 20, 40, 40), "roi2": (80, 20, 40, 40)}
    with pytest.raises(core.CoreError) as exc_info:
        core.initialize_extensometer_group_state(stripe, group)
    assert exc_info.value.code == "AMBIGUOUS_TEXTURE"

    reference = core.generate_synthetic_speckle(128, 128, seed=17)
    good_group = {**_group(), "roi1": (20, 40, 25, 25), "roi2": (80, 40, 25, 25)}
    state = core.initialize_extensometer_group_state(reference, good_group)
    row, _ = core.track_extensometer_group_frame(state, reference, 0, "ref", _params(enable_fb_check=False))
    assert row["accepted"] is True
    assert row["strain_valid"] is True


def test_follow_template_is_explicit_and_uses_subpixel_registered_patch():
    reference = core.generate_synthetic_speckle(128, 128, seed=8)
    group = {**_group(), "roi1": (20, 40, 25, 25), "roi2": (80, 40, 25, 25)}
    state = core.initialize_extensometer_group_state(reference, group)
    assert state["template_policy"] == "fixed_reference"
    rect = (20.37, 40.41, 25, 25)
    registered = core.update_template_from_rect(reference, rect, state["template1"], 1.0)
    rounded = reference[40:65, 20:45].astype(np.float32)
    assert not np.array_equal(registered, rounded)

    row, _ = core.track_extensometer_group_frame(
        state,
        reference,
        0,
        "ref",
        _params(enable_fb_check=False, use_prev_frame_template=True),
    )
    assert row["template_policy"] == "experimental_follow"


def test_dimension_preflight_happens_before_output_creation(monkeypatch, tmp_path):
    app = object.__new__(gui.MultiROIGUI)
    output_dir = tmp_path / "out"
    settings = {
        "start_idx": 0,
        "end_idx": 1,
        "image_paths": ["frame_001.png", "frame_002.png"],
        "roi_groups": [],
        "output_dir": output_dir,
    }

    def fail_dimensions(_paths):
        raise RuntimeError("image dimensions differ")

    monkeypatch.setattr(gui, "validate_image_sequence_dimensions", fail_dimensions)
    with pytest.raises(RuntimeError, match="dimensions differ"):
        gui.MultiROIGUI.process_images(app, settings)
    assert not output_dir.exists()


def test_strain_aggregates_use_strain_valid_and_legacy_fallback():
    data = pd.DataFrame(
        {
            "frame_global_1based": [1, 2, 3],
            "engineering_strain": [0.0, 0.10, 0.20],
            "accepted": [True, True, False],
            "strain_valid": [True, False, True],
            "group": ["G01"] * 3,
        }
    )
    table = core.build_core_strain_table(data)
    assert table.loc[0, "EngineeringStrain"] == pytest.approx(0.0)
    assert np.isnan(table.loc[1, "EngineeringStrain"])
    assert table.loc[2, "EngineeringStrain"] == pytest.approx(0.20)
    legacy = data.drop(columns=["strain_valid"])
    legacy_table = core.build_core_strain_table(legacy)
    assert np.isnan(legacy_table.loc[2, "EngineeringStrain"])


def test_gui_methods_and_public_match_template_api_are_core_canonical():
    assert gui.initialize_extensometer_group_state is core.initialize_extensometer_group_state
    assert gui.track_extensometer_group_frame is core.track_extensometer_group_frame
    assert gui.match_template_candidate is core.match_template_candidate
    assert gui.match_template_candidate_diagnostic is core.match_template_candidate_diagnostic


def test_zero_variance_frame_is_not_misclassified_as_ambiguous_peak():
    reference = core.generate_synthetic_speckle(256, 256, seed=17)
    blank = np.full((256, 256), 120.0, dtype=np.float32)
    state = core.initialize_extensometer_group_state(reference, _group())
    core.track_extensometer_group_frame(state, reference, 0, "ref", _params())
    row, _ = core.track_extensometer_group_frame(state, blank, 1, "blank", _params())
    assert row["accepted"] is False
    assert row["tracking_status_code"] == "ZERO_VARIANCE"
    assert row["peak_is_ambiguous_roi1"] is False
    assert row["peak_is_ambiguous_roi2"] is False


def test_permissive_peak_thresholds_are_not_overridden_by_defaults(monkeypatch):
    reference = core.generate_synthetic_speckle(256, 256, seed=17)
    deformed = core.warp_image_translation(reference, 1.0, 0.0)

    def slightly_close_peak(_image, rect, _template, _radius, **_kwargs):
        return {
            "candidate_rect": (float(rect[0] + 1.0), float(rect[1]), rect[2], rect[3]),
            "score": 0.90,
            "best_peak": 0.90,
            "second_peak": 0.89,
            "peak_margin": 0.01,
            "peak_ratio": 1.011,
            "best_to_second_peak_ratio": 1.011,
            "second_to_best_peak_ratio": 0.89 / 0.90,
            "peak_is_ambiguous": True,
            "reason": "peak_found",
        }

    monkeypatch.setattr(core, "match_template_candidate", slightly_close_peak)
    state = core.initialize_extensometer_group_state(reference, _group())
    core.track_extensometer_group_frame(state, reference, 0, "ref", _params(enable_fb_check=False))
    row, _ = core.track_extensometer_group_frame(
        state,
        deformed,
        1,
        "def",
        _params(enable_fb_check=False, peak_margin_min=0.0, peak_ratio_min=1.0),
    )
    assert row["accepted"] is True
    assert row["peak_is_ambiguous_roi1"] is False
    assert row["peak_is_ambiguous_roi2"] is False


def test_adaptive_disabled_and_low_corr_report_correlation_below_threshold(monkeypatch):
    reference = core.generate_synthetic_speckle(256, 256, seed=17)
    deformed = core.warp_image_translation(reference, 1.0, 0.0)

    def mid_score(_image, rect, _template, _radius, **_kwargs):
        return {
            "candidate_rect": (float(rect[0] + 1.0), float(rect[1]), rect[2], rect[3]),
            "score": 0.45,
            "best_peak": 0.45,
            "second_peak": 0.10,
            "peak_margin": 0.35,
            "peak_ratio": 4.5,
            "best_to_second_peak_ratio": 4.5,
            "second_to_best_peak_ratio": 0.10 / 0.45,
            "peak_is_ambiguous": False,
            "reason": "peak_found",
        }

    monkeypatch.setattr(core, "match_template_candidate", mid_score)
    monkeypatch.setattr(core, "forward_backward_error", lambda *_a, **_k: (99.0, 0.1))
    state = core.initialize_extensometer_group_state(reference, _group())
    core.track_extensometer_group_frame(state, reference, 0, "ref", _params())
    row, _ = core.track_extensometer_group_frame(
        state,
        deformed,
        1,
        "def",
        _params(enable_adaptive=False, hard_corr=0.55, soft_corr=0.35, enable_fb_check=True),
    )
    assert row["accepted"] is False
    assert row["tracking_status_code"] == "CORRELATION_BELOW_THRESHOLD"
