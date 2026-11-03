"""Known correspondences remain distinguishable from strong copied speckles."""

import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import ezdic_core as core


def repeated_patch_case(noise_sigma=.10):
    rng = np.random.default_rng(3)
    reference = gaussian_filter(rng.normal(size=(130, 130)), 1.2).astype("float32")
    target = np.roll(reference, 5, axis=1).copy()
    # The true correspondence is (5, 0); the exact copied patch is a distractor.
    target[50:71, 55:76] += rng.normal(size=(21, 21)).astype("float32") * noise_sigma
    target[67:88, 72:93] = reference[50:71, 50:71]
    return reference, target


def run_case(*, seed=(5, 0), reject=True, solver="IC-GN", noise_sigma=.10):
    reference, target = repeated_patch_case(noise_sigma)
    transform = None if seed is None else {
        "matrix": np.array([[1., 0., seed[0]], [0., 1., seed[1]]]),
        "median_reprojection_error_px": .1,
    }
    return core.run_2d_dic(
        reference, target, (20, 20, 81, 81), subset_size=21, step=30,
        search_radius=30, strain_window=5, max_iter=40, zncc_min=.75,
        reject_nonconverged=True, reject_ambiguous_peaks=reject, solver=solver,
        initialization="local", _global_initial_transform=transform, experimental_full_search=True,
    )


@pytest.mark.parametrize("solver", ["IC-GN", "IC-LM"])
@pytest.mark.parametrize("seed", [(5, 0), (22, 17), None])
def test_copied_patch_rejected_independent_of_global_prediction(seed, solver):
    field = run_case(seed=seed, solver=solver)
    assert not field["valid"][4]
    assert field["invalid_reason"][4] == "AMBIGUOUS_PEAK"
    assert field["competing_peak_status"][4] == "DISTINCT_VALID_CANDIDATE"
    assert field["u_candidate"][4] == pytest.approx(22.0, abs=.02)
    assert field["v_candidate"][4] == pytest.approx(17.0, abs=.02)
    assert field["competing_peak_refined_u"][4] == pytest.approx(5.0, abs=.1)
    assert field["competing_peak_refined_v"][4] == pytest.approx(0.0, abs=.1)
    # Reject even though the wrong answer's ZNCC is materially higher.
    assert field["zncc_candidate"][4] - field["competing_peak_refined_zncc"][4] > .08
    assert np.isnan(field["u"][4]) and np.isnan(field["Eyy"][4])
    assert not field["strain_valid"][4]


def test_permissive_ambiguity_is_flagged_and_never_a_false_match_measurement():
    field = run_case(reject=False)
    assert field["valid"][4] and field["peak_is_ambiguous"][4]
    assert field["quality_summary"]["scientific_ok"] is False
    assert field["quality_summary"]["false_accept_is_proxy"] is True
    assert field["quality_summary"]["false_accept_ground_truth_assessed"] is False


def test_rejected_candidates_survive_csv_txt_and_safe_npz(tmp_path, monkeypatch):
    field = run_case()
    monkeypatch.setattr(core, "plot_dic_field_map", lambda _f, path, **_k: path)
    paths = core.export_dic_field_outputs(field, tmp_path)
    table = core.dic_field_to_dataframe(field)
    assert table.loc[4, "u_candidate"] == field["u_candidate"][4]
    assert np.isnan(table.loc[4, "u"])
    assert "DISTINCT_VALID_CANDIDATE" in paths["txt"].read_text(encoding="utf-8")
    with np.load(paths["arrays"], allow_pickle=False) as arrays:
        for key in ("u_candidate", "v_candidate", "zncc_candidate", "u", "v", "strain_valid"):
            np.testing.assert_equal(arrays[key], field[key])
        assert arrays["competing_peak_status"][4] == "DISTINCT_VALID_CANDIDATE"
        metadata = json.loads(str(arrays["metadata_json"]))
        assert metadata["parameters"]["competing_peak_policy"] == field["competing_peak_policy"]
        assert metadata["parameters"]["experimental_full_search"] is True
        assert arrays["experimental_full_search"].shape == ()
        assert bool(arrays["experimental_full_search"]) is True
        reloaded = {key: arrays[key] for key in arrays.files}
    assert table["experimental_full_search"].all()
    # Remove ordinary coverage reasons: the experimental policy alone must
    # still prevent a pass after loading the safe array archive.
    quality = core.field_quality_summary(reloaded, min_correlation_valid_fraction=0.,
                                          min_strain_valid_fraction=0.)
    assert quality["experimental_full_search"] is True
    assert quality["scientific_ok"] is False
    assert "experimental_search_unvalidated" in quality["scientific_reasons"]
    assert core.dic_field_to_dataframe(reloaded)["experimental_full_search"].all()
    exported_again = core.export_dic_field_outputs(reloaded, tmp_path / "again")
    with np.load(exported_again["arrays"], allow_pickle=False) as arrays:
        assert bool(arrays["experimental_full_search"]) is True
        assert json.loads(str(arrays["metadata_json"]))["parameters"]["experimental_full_search"] is True


def test_one_broad_peak_has_no_distinct_candidate():
    y, x = np.indices((31, 31), dtype=float)
    surface = np.exp(-((x-15)**2 + (y-15)**2)/100).astype("float32")
    assert core._competing_correlation_peak(surface, (15, 15)) is None


@pytest.mark.parametrize("condition", [np.inf, np.nan])
def test_explicit_nonfinite_hessian_cannot_pass_or_supply_strain(monkeypatch, condition):
    image = core.generate_synthetic_speckle(80, 80, seed=22)
    def failed_hessian(*_a, **_k):
        return {"u": 0., "v": 0., "p": np.zeros(6), "zncc": 1.,
                "hessian_condition_number": condition, "converged": False}
    monkeypatch.setattr(core, "refine_subset_icgn", failed_hessian)
    field = core.run_2d_dic(image, image, (20, 20, 41, 41), subset_size=15, step=10,
                            initialization="local", reject_nonconverged=False)
    assert not field["valid"].any()
    assert set(field["invalid_reason"]) == {"ILL_CONDITIONED_HESSIAN"}
    assert np.isfinite(field["u_candidate"]).all()
    assert not field["strain_valid"].any()
    assert np.isnan(field["Eyy"]).all()


@pytest.mark.parametrize("reject", [True, False])
def test_pyramid_retains_a_coarse_copied_patch_ambiguity(reject):
    rng = np.random.default_rng(11)
    reference = gaussian_filter(rng.normal(size=(240, 240)), 1.2).astype("float32")
    target = np.roll(reference, 4, axis=1)
    target[70:131, 74:135] += rng.normal(size=(61, 61)).astype("float32")*.06
    target[118:179, 122:183] = reference[70:131, 70:131]
    transform = {"matrix": np.array([[1., 0., 4.], [0., 1., 0.]]), "median_reprojection_error_px": .1}
    field = core.run_2d_dic(
        reference, target, (40, 40, 121, 121), subset_size=21, step=50,
        search_radius=18, strain_window=3, max_iter=40, zncc_min=.75,
        pyramid_levels=3, reject_ambiguous_peaks=reject,
        _global_initial_transform=transform, experimental_full_search=True,
    )
    assert field["pyramid_ambiguous"][4] and field["peak_is_ambiguous"][4]
    assert field["pyramid_level_diagnostics"][4][0]["peak_is_ambiguous"]
    if reject:
        assert not field["valid"][4]
        assert field["stop_reason"][4] == "pyramid_ambiguous_correlation_peaks"
    else:
        assert field["quality_summary"]["ambiguous_accept_proxy_count"] >= 1


def test_failed_coarse_refinement_does_not_move_the_next_search(monkeypatch):
    from benchmarks.synthetic_cases import analytic_texture

    y, x = np.indices((200, 200), dtype=float)
    reference = analytic_texture(x, y, seed=593).astype("float32")
    target = analytic_texture(x-5., y, seed=593).astype("float32")
    native_refine = core.refine_subset_icgn
    def failed_coarse(ref, target, *args, **kwargs):
        if ref.shape[0] < 200:
            return {"u": 31., "v": 0., "p": np.array([31., 0., 0., 0., 0., 0.]),
                    "zncc": .3, "hessian_condition_number": 10., "converged": False}
        return native_refine(ref, target, *args, **kwargs)
    monkeypatch.setattr(core, "refine_subset_icgn", failed_coarse)
    transform = {"matrix": np.array([[1., 0., 5.], [0., 1., 0.]]), "median_reprojection_error_px": .1}
    field = core.run_2d_dic(
        reference, target, (50, 50, 61, 61), subset_size=21, step=20,
        search_radius=6, pyramid_levels=2, reject_nonconverged=True,
        _global_initial_transform=transform,
    )
    assert field["valid"].all()
    np.testing.assert_allclose(field["u"], 5., atol=.01)
    for levels in field["pyramid_level_diagnostics"]:
        assert levels[0]["refine_u"] == 31.
        assert not levels[0]["accepted"]
        assert levels[1]["seed_u"] == 5.


def test_coarse_boundary_failure_retains_seed_in_full_resolution_coordinates():
    from benchmarks.synthetic_cases import analytic_texture

    y, x = np.indices((160, 160), dtype=float)
    reference = analytic_texture(x, y, seed=593).astype("float32")
    target = analytic_texture(x-20., y, seed=593).astype("float32")
    transform = {"matrix": np.array([[1., 0., 20.], [0., 1., 0.]]), "median_reprojection_error_px": .1}
    settings = dict(subset_size=21, step=20, search_radius=6,
                    reject_nonconverged=True, _global_initial_transform=transform)
    fine = core.run_2d_dic(reference, target, (0, 40, 81, 81), **settings)
    multi = core.run_2d_dic(reference, target, (0, 40, 81, 81), pyramid_levels=3, **settings)
    assert fine["valid"].all() and multi["valid"].all()
    np.testing.assert_allclose(multi["u"], 20., atol=.01)
    coarse = multi["pyramid_level_diagnostics"][0][0]
    assert coarse["stop_reason"] == "boundary_support_failure"
    assert not coarse["peak_is_ambiguous"]


def test_known_degraded_texture_miss_never_passes_experimental_scientific_gate():
    # The true patch is too damaged to pass correlation; the copied patch can
    # still fool local DIC. This is why the full-radius option is research-only.
    field = run_case(noise_sigma=.35)
    assert field["valid"].all()
    assert np.hypot(field["u"][4]-5., field["v"][4]) > 20.
    assert field["quality_summary"]["scientific_ok"] is False
    assert "experimental_search_unvalidated" in field["quality_summary"]["scientific_reasons"]
    repeated = core.field_quality_summary(field, min_correlation_valid_fraction=0.,
                                          min_strain_valid_fraction=0.)
    assert repeated["scientific_ok"] is False
