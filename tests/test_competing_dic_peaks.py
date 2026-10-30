"""Ambiguous widely separated speckles must not be silently trusted."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import ezdic_core as core


def repeated_patch_case(noise_sigma=0.06):
    rng = np.random.default_rng(3)
    reference = gaussian_filter(rng.normal(size=(130, 130)), 1.2).astype("float32")
    target = np.roll(reference, 5, axis=1).copy()
    x = y = 60
    half = 10
    target[y-half:y+half+1, x+5-half:x+5+half+1] += rng.normal(size=(21, 21)).astype("float32") * noise_sigma
    target[y+17-half:y+17+half+1, x+5+17-half:x+5+17+half+1] = reference[y-half:y+half+1, x-half:x+half+1]
    return reference, target


def run_case(radius, *, seed=(5, 0), reject=True, noise_sigma=0.06, strain_window=3):
    reference, target = repeated_patch_case(noise_sigma=noise_sigma)
    transform = {
        "matrix": np.array([[1.0, 0.0, seed[0]], [0.0, 1.0, seed[1]]]),
        "median_reprojection_error_px": 0.1,
    }
    return core.run_2d_dic(
        reference, target, (20, 20, 81, 81), subset_size=21, step=30,
        search_radius=radius, strain_window=strain_window, max_iter=40, zncc_min=0.75,
        reject_nonconverged=True, reject_ambiguous_peaks=reject,
        _global_initial_transform=transform,
    )


def test_wide_search_rejects_distinct_plausible_correspondences():
    narrow = run_case(3)
    wide = run_case(18)
    assert narrow["valid"][4]
    assert narrow["u_raw"][4] == pytest.approx(5.0, abs=0.1)
    assert not wide["valid"][4]
    assert np.isnan(wide["u"][4])
    assert wide["invalid_reason"][4] == "AMBIGUOUS_PEAK"
    assert wide["stop_reason"][4] == "competing_correlation_peaks"
    assert wide["competing_peak_is_plausible"][4]
    assert wide["competing_peak_zncc"][4] > 0.95
    assert wide["competing_peak_u"][4] == pytest.approx(5.0, abs=0.1)
    assert wide["competing_peak_v"][4] == pytest.approx(0.0, abs=0.1)
    assert "competing_peak_zncc" in core.dic_field_to_dataframe(wide).columns


def test_noisier_competing_true_patch_is_still_rejected():
    field = run_case(18, noise_sigma=0.08)
    assert not field["valid"][4]
    assert field["invalid_reason"][4] == "AMBIGUOUS_PEAK"
    assert field["competing_peak_zncc_gap"][4] < core.COMPETING_PEAK_MAX_ZNCC_GAP


def test_prediction_conflict_rejects_noisy_copied_patch():
    field = run_case(18, noise_sigma=0.10, strain_window=5)
    assert not field["valid"][4]
    assert field["invalid_reason"][4] == "AMBIGUOUS_PEAK"
    assert field["stop_reason"][4] == "prediction_conflicting_peaks"
    assert field["competing_peak_status"][4] == "PREDICTION_CONFLICT"
    assert field["prediction_conflict"][4]
    assert field["prediction_candidate_distance_px"][4] <= field["prediction_conflict_radius_px"][4]
    assert field["prediction_primary_distance_px"][4] > field["prediction_conflict_radius_px"][4]


def test_wrong_global_prediction_rejects_instead_of_selecting_it():
    rng = np.random.default_rng(3)
    reference = gaussian_filter(rng.normal(size=(130, 130)), 1.2).astype("float32")
    target = np.roll(reference, 5, axis=1).copy()
    target[67:88, 72:93] = reference[50:71, 50:71] + rng.normal(size=(21, 21)).astype("float32") * .12
    seed = {"matrix": np.array([[1., 0., 22.], [0., 1., 17.]]), "median_reprojection_error_px": .1}
    settings = dict(subset_size=21, step=30, search_radius=18, strain_window=5,
                    max_iter=40, zncc_min=.75, reject_nonconverged=True,
                    _global_initial_transform=seed)
    diagnostic = core.run_2d_dic(reference, target, (20, 20, 81, 81), reject_ambiguous_peaks=False, **settings)
    guarded = core.run_2d_dic(reference, target, (20, 20, 81, 81), reject_ambiguous_peaks=True, **settings)
    assert diagnostic["u_raw"][4] == pytest.approx(5.0, abs=.02)
    assert diagnostic["v_raw"][4] == pytest.approx(0.0, abs=.02)
    assert diagnostic["prediction_conflict"][4]
    assert diagnostic["quality_summary"]["ambiguous_accept_proxy_count"] >= 1
    assert not guarded["valid"][4]
    assert guarded["stop_reason"][4] == "prediction_conflicting_peaks"


@pytest.mark.xfail(strict=False, reason="A wrong global prediction aligned with the copied patch can still hide the true correspondence.")
def test_wrong_prediction_on_copied_patch_should_reject_or_track_correctly():
    # The predictor is never truth; this failure records a counterexample to
    # the heuristic and becomes an XPASS when an actual robust fix exists.
    field = run_case(18, seed=(22, 17), noise_sigma=.10, strain_window=5)
    if field["valid"][4]:
        error = np.hypot(field["u_raw"][4] - 5.0, field["v_raw"][4])
        assert error <= .5


def test_prediction_conflict_uses_global_not_coarse_coordinates():
    transform = {"matrix": np.array([[1., 0., 10.], [0., 1., 4.]]), "median_reprojection_error_px": .1}
    primary = {"u": 10., "v": 5.}
    candidate = {"u": 5., "v": 2.}
    prediction = core._prediction_conflict(primary, candidate, transform, 100., 80., scale_x=.5, scale_y=.5)
    assert prediction["tested"] and prediction["conflict"]
    assert prediction["candidate_distance_px"] == pytest.approx(0.)
    assert prediction["primary_distance_px"] == pytest.approx(np.hypot(10., 6.))
    assert not core._prediction_conflict(primary, candidate, None, 100., 80.)["tested"]


def test_fine_conflict_distances_override_nonconflicting_coarse_diagnostic():
    reference, target = repeated_patch_case(noise_sigma=.10)
    seed = {"matrix": np.array([[1., 0., 5.], [0., 1., 0.]]), "median_reprojection_error_px": .1}
    coarse = {(1, 1): {"level_index": 0, "scale_x": .5, "scale_y": .5,
                        "prediction_conflict": False, "prediction_conflict_tested": True,
                        "prediction_conflict_radius_px": 3., "prediction_primary_distance_px": 1.,
                        "prediction_candidate_distance_px": 8.,
                        "competing_peak_status": "DISTINCT_VALID_CANDIDATE"}}
    field = core.run_2d_dic(
        reference, target, (20, 20, 81, 81), subset_size=21, step=30,
        search_radius=18, strain_window=5, max_iter=40, zncc_min=.75,
        reject_nonconverged=True, reject_ambiguous_peaks=False,
        _global_initial_transform=seed, _pyramid_conflicts=coarse,
    )
    assert field["prediction_conflict"][4]
    assert field["prediction_candidate_distance_px"][4] <= field["prediction_conflict_radius_px"][4]
    assert field["prediction_primary_distance_px"][4] > field["prediction_conflict_radius_px"][4]


def test_intermediate_prediction_conflict_is_not_hidden_by_earlier_ambiguity(monkeypatch):
    from benchmarks.synthetic_cases import analytic_texture

    y, x = np.indices((200, 200), dtype=float)
    image = analytic_texture(x, y, seed=593).astype("float32")
    transform = {"matrix": np.array([[1., 0., 0.], [0., 1., 0.]]), "median_reprojection_error_px": .1}

    # Exercise level selection with a nonconflicting coarse ambiguity followed
    # by a conflicting intermediate level and a nonconflicting fine level.
    monkeypatch.setattr(core, "_refine_competing_peak", lambda *_a, **_k: {
        "status": "DISTINCT_VALID_CANDIDATE", "result": {"u": 0., "v": 0., "zncc": .9},
    })
    def prediction(_primary, _candidate, _transform, _x, _y, *, scale_x=1., scale_y=1.):
        conflict = .4 < scale_x < .6
        return {"tested": True, "conflict": conflict, "radius_px": 3.,
                "primary_distance_px": 8. if conflict else 1.,
                "candidate_distance_px": 1. if conflict else 8.}
    monkeypatch.setattr(core, "_prediction_conflict", prediction)
    field = core.run_2d_dic(
        image, image, (40, 40, 61, 61), subset_size=21, step=20,
        search_radius=5, pyramid_levels=3, reject_ambiguous_peaks=False,
        _global_initial_transform=transform,
    )
    records = field["pyramid_level_diagnostics"][4]
    assert len(records) == 3
    assert records[0]["peak_is_ambiguous"] and not records[0]["prediction_conflict"]
    assert records[1]["prediction_conflict"]
    assert field["prediction_conflict"][4]
    assert field["pyramid_conflict_level"][4] == 1
    assert field["prediction_candidate_distance_px"][4] <= field["prediction_conflict_radius_px"][4]
    assert field["prediction_primary_distance_px"][4] > field["prediction_conflict_radius_px"][4]


def test_much_weaker_candidate_does_not_reject_clear_primary():
    rng = np.random.default_rng(22)
    reference = gaussian_filter(rng.normal(size=(180, 180)), 1.2).astype("float32")
    target = np.roll(reference, 5, axis=1).copy()
    x = y = 80
    half = 10
    target[y+25-half:y+25+half+1, x+30-half:x+30+half+1] = (
        reference[y-half:y+half+1, x-half:x+half+1]
        + rng.normal(size=(21, 21)).astype("float32") * 0.12
    )
    transform = {"matrix": np.array([[1., 0., 5.], [0., 1., 0.]]), "median_reprojection_error_px": .1}
    field = core.run_2d_dic(
        reference, target, (40, 40, 101, 101), subset_size=21, step=30,
        search_radius=30, zncc_min=.75, max_iter=40, _global_initial_transform=transform,
    )
    assert field["valid"][4]
    assert field["u_raw"][4] == pytest.approx(5., abs=.01)
    assert field["v_raw"][4] == pytest.approx(0., abs=.01)
    assert field["competing_peak_refined_zncc"][4] >= .75
    assert field["competing_peak_zncc_gap"][4] > core.COMPETING_PEAK_MAX_ZNCC_GAP
    assert field["competing_peak_status"][4] == "CANDIDATE_WEAKER_THAN_GAP"


def test_conflicting_candidate_is_not_resolved_using_global_seed_as_truth():
    # The same pair is ambiguous even when the global seed points at the copy.
    wide = run_case(18, seed=(22, 17))
    assert not wide["valid"][4]
    assert wide["stop_reason"][4] == "competing_correlation_peaks"


def test_accepted_ambiguity_is_explicitly_reported_as_proxy_only():
    wide = run_case(18, reject=False)
    assert wide["valid"][4]
    assert wide["peak_is_ambiguous"][4]
    quality = wide["quality_summary"]
    assert quality["ambiguous_accept_proxy_count"] >= 1
    assert quality["false_accept_count"] == quality["ambiguous_accept_proxy_count"]
    assert quality["false_accept_is_proxy"] is True
    assert quality["false_accept_ground_truth_assessed"] is False


def test_candidate_below_final_threshold_can_pass_after_affine_refinement():
    from benchmarks.necking_validation import CASES, image_pair

    reference, target = image_pair(CASES["narrow_localization"])
    guess = core.integer_cc_guess(reference, target, 123, 27, 15, 10, return_diagnostics=True)
    assert .70 < guess["competing_peak_zncc"] < .80
    p0 = [guess["u"], 0., 0., guess["v"], 0., 0.]
    prepared = core._prepare_dic_images(reference, target)
    primary = core.refine_subset_icgn(reference, target, 123, 27, 15, p0=p0, max_iter=50, tol=1e-4, _prepared=prepared)
    candidate = core._refine_competing_peak(
        guess, primary, p0, core.refine_subset_icgn, reference, target, 123, 27, 15,
        max_iter=50, conv_tol=1e-4, zncc_min=.8, max_condition_number=1e12,
        max_residual_rms=float("inf"), reject_nonconverged=True, prepared=prepared,
    )
    assert candidate["result"]["zncc"] >= .8
    assert candidate["status"] == "DISTINCT_VALID_CANDIDATE"


def test_competing_diagnostics_are_auditable_in_exports(tmp_path, monkeypatch):
    field = run_case(18)
    monkeypatch.setattr(core, "plot_dic_field_map", lambda _field, path, **_kwargs: path)
    paths = core.export_dic_field_outputs(field, tmp_path)
    text = paths["txt"].read_text(encoding="utf-8")
    assert "competing_peak_status" in text
    assert "DISTINCT_VALID_CANDIDATE" in text
    parameters = paths["parameters"].read_text(encoding="utf-8")
    assert "competing_peak_raw_margin = 0.1" in parameters
    assert "competing_peak_max_zncc_gap = 0.08" in parameters
    with np.load(paths["arrays"], allow_pickle=False) as data:
        assert data["competing_peak_status"][4] == "DISTINCT_VALID_CANDIDATE"
        assert data["competing_peak_is_plausible"][4]


def test_a_single_broad_peak_shoulder_is_not_a_competing_maximum():
    y, x = np.indices((31, 31), dtype=float)
    surface = np.exp(-((x-15)**2 + (y-15)**2) / 100.0).astype("float32")
    candidate = core._competing_correlation_peak(surface, (15, 15))
    assert candidate is None


@pytest.mark.parametrize("reject", [True, False])
def test_pyramid_does_not_hide_coarse_level_ambiguity(reject):
    rng = np.random.default_rng(11)
    reference = gaussian_filter(rng.normal(size=(240, 240)), 1.2).astype("float32")
    target = np.roll(reference, 4, axis=1)
    target[70:131, 74:135] += rng.normal(size=(61, 61)).astype("float32") * 0.06
    target[118:179, 122:183] = reference[70:131, 70:131]
    transform = {"matrix": np.array([[1., 0., 4.], [0., 1., 0.]]), "median_reprojection_error_px": .1}
    field = core.run_2d_dic(
        reference, target, (40, 40, 121, 121), subset_size=21, step=50,
        search_radius=18, strain_window=3, max_iter=40, zncc_min=.75,
        pyramid_levels=3, reject_ambiguous_peaks=reject,
        _global_initial_transform=transform,
    )
    index = 4
    assert field["peak_is_ambiguous"][index]
    assert field["pyramid_conflict_level"][index] == 0
    assert field["pyramid_competing_peak_zncc"][index] > .9
    assert field["pyramid_competing_peak_u"][index] == pytest.approx(4.0, abs=.1)
    assert field["pyramid_competing_peak_v"][index] == pytest.approx(0.0, abs=.1)
    assert field["pyramid_level_diagnostics"][index][0]["peak_is_ambiguous"]
    assert "pyramid_conflict_level" in core.dic_field_to_dataframe(field).columns
    if reject:
        assert not field["valid"][index]
        assert field["stop_reason"][index] == "pyramid_competing_correlation_peaks"
    else:
        assert field["valid"][index]
        assert field["quality_summary"]["ambiguous_accept_proxy_count"] >= 1
