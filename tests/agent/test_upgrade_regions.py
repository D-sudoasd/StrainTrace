"""Analytic support, coordinate and strain checks; no experimental accuracy claim."""
import math

import numpy as np
import pytest

import ezdic_core as core
from straintrace_agent.fields import calibrate_field, plot_position_time, summarize_roi
from straintrace_agent.mechanics import _auto_gauge_candidates, _mechanics_at, _measure_gauges, _resolve_gauge_geometry
from straintrace_agent.regions import normalize_regions, recommend_regions, summarize_regions


ROI = (0, 0, 65, 65)
SHAPE = (65, 65)
CALIBRATION = {"mm_per_px": 0.02, "origin_px": [0, 0], "origin_mm": [2, 5], "axis_sign": [1, -1]}


def _field(F=((1, 0), (0, 1)), translation=(0, 0)):
    X, Y = core.build_poi_grid(ROI, 9, 10, SHAPE)
    F = np.asarray(F, float)
    current = np.stack([X, Y], axis=-1) @ F.T + translation
    U, V = current[..., 0] - X, current[..., 1] - Y
    E = (F.T @ F - np.eye(2)) / 2
    e = (F + F.T) / 2 - np.eye(2)
    P = np.zeros(X.shape + (6,))
    P[..., 0], P[..., 3] = U, V
    P[..., 1], P[..., 2], P[..., 4], P[..., 5] = F[0, 0] - 1, F[0, 1], F[1, 0], F[1, 1] - 1
    return {
        "X": X, "Y": Y, "x": X.ravel(), "y": Y.ravel(), "roi": ROI,
        "u_raw": U.ravel(), "v_raw": V.ravel(), "u": U.ravel(), "v": V.ravel(), "P": P,
        "valid": np.ones(X.size, bool), "strain_valid": np.ones(X.size, bool), "zncc": np.full(X.size, 0.99),
        **{key: np.full(X.size, value) for key, value in {
            "Exx": E[0, 0], "Eyy": E[1, 1], "Exy": E[0, 1], "exx": e[0, 0], "eyy": e[1, 1], "exy": e[0, 1],
        }.items()},
    }


def _region(name="left", bounds=(4, 24, 4, 54), **kwargs):
    return {"name": name, "bounds": list(bounds), **kwargs}


def _gauge(name="g", axis="distance", a=(14, 24), b=(44, 24), **kwargs):
    return {"name": name, "axis": axis, "a_point_px": list(a), "b_point_px": list(b), "max_distance_px": 0, **kwargs}


def _geometry(gauges, calibration=None, mask=None):
    return _resolve_gauge_geometry(gauges, ROI, SHAPE, 9, 10, mask, calibration=calibration)


def test_multiple_regions_and_physical_gauge_lengths_uniform_extension():
    field = _field([[1.08, 0], [0, 0.97]])
    regions = [_region(), _region("right", (34, 54, 4, 54))]
    rows = summarize_regions(field, regions)
    assert len(rows) == 6
    expected = {"Exx": (1.08**2 - 1) / 2, "Eyy": (0.97**2 - 1) / 2, "Exy": 0}
    for row in rows:
        assert row["status"] == "valid"
        assert row["n_poi_valid"] == row["n_poi_total"] == 18
        assert row["mean"] == pytest.approx(expected[row["component"]], abs=1e-14)
        assert row["sd_sample"] == pytest.approx(0, abs=1e-14)
    gauges = [_gauge(), _gauge("vertical", "y", (24, 14), (24, 44))]
    measured = _measure_gauges(field, _geometry(gauges, CALIBRATION), quality_ok=True)
    for row, strain in zip(measured, (0.08, -0.03)):
        assert row["status"] == "valid"
        assert row["engineering_strain"] == pytest.approx(strain, abs=1e-14)
        assert row["true_strain"] == pytest.approx(math.log1p(strain), abs=1e-14)
        assert row["initial_length_px"] == 30
        assert row["initial_length_mm"] == pytest.approx(0.6)
        assert row["extension_mm"] == pytest.approx(0.6 * strain, abs=1e-14)
        assert row["current_length_mm"] == pytest.approx(0.6 * (1 + strain), abs=1e-14)
        assert row["loading_axis"] is None


def test_simple_shear_distinguishes_line_extension_from_projected_separation():
    gamma = 0.3
    field = _field([[1, gamma], [0, 1]])
    gauges = [_gauge("distance", "distance", (24, 14), (24, 44)), _gauge("projected", "y", (24, 14), (24, 44))]
    rows = _measure_gauges(field, _geometry(gauges), quality_ok=True)
    assert rows[0]["engineering_strain"] == pytest.approx(math.sqrt(1 + gamma**2) - 1, abs=1e-14)
    assert rows[1]["engineering_strain"] == pytest.approx(0, abs=1e-14)
    assert rows[0]["measurement_kind"] == "euclidean_distance"
    assert rows[1]["measurement_kind"] == "projected_separation"
    summary = summarize_regions(field, [_region()], ("Exy", "Eyy"))
    assert summary[0]["mean"] == pytest.approx(gamma / 2, abs=1e-14)
    assert summary[1]["mean"] == pytest.approx(gamma**2 / 2, abs=1e-14)


@pytest.mark.parametrize("angle,translation", [(0, (5, -3)), (32, (0, 0)), (123, (5, -3))])
def test_distance_gauge_and_green_lagrange_are_objective_for_rigid_motion(angle, translation):
    theta = math.radians(angle)
    R = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    field = _field(R, translation)
    measured = _measure_gauges(field, _geometry([_gauge()]), quality_ok=True)[0]
    assert measured["engineering_strain"] == pytest.approx(0, abs=1e-14)
    assert measured["initial_length_mm"] is None and measured["extension_mm"] is None
    assert all(abs(row["mean"]) < 1e-14 for row in summarize_regions(field, [_region()]))


def test_nonuniform_field_multiple_lengths_and_rois_reuse_original_arrays():
    field = _field()
    alpha = 0.0005
    X = field["X"]
    field["u_raw"] = (alpha * X**2).ravel()
    field["u"] = field["u_raw"].copy()
    field["P"][..., 0], field["P"][..., 1] = alpha * X**2, 2 * alpha * X
    field["Exx"] = ((1 + 2 * alpha * X)**2 - 1).ravel() / 2
    field["exx"] = (2 * alpha * X).ravel()
    before = {key: value.copy() for key, value in field.items() if isinstance(value, np.ndarray)}
    regions = [_region(), _region("right", (34, 54, 4, 54))]
    summaries = summarize_regions(field, regions, ["Exx"])
    for region, row in zip(regions, summaries):
        xmin, xmax, _, _ = region["bounds"]
        points = X[(X >= xmin) & (X <= xmax)]
        expected = np.mean(2 * alpha * points + 2 * alpha**2 * points**2)
        assert row["mean"] == pytest.approx(expected, abs=1e-14)
        assert row["mean"] == pytest.approx(summarize_roi(field, "Exx", region["bounds"])["mean"], abs=1e-14)
    assert summaries[1]["mean"] > summaries[0]["mean"]
    gauges = [_gauge("short", b=(24, 24)), _gauge("long", b=(54, 24))]
    measured = _measure_gauges(field, _geometry(gauges), quality_ok=True)
    assert measured[0]["engineering_strain"] == pytest.approx(alpha * (14 + 24), abs=1e-14)
    assert measured[1]["engineering_strain"] == pytest.approx(alpha * (14 + 54), abs=1e-14)
    for key, expected in before.items():
        np.testing.assert_array_equal(field[key], expected)


def test_displacement_and_strain_support_are_counted_separately_and_holes_stay_missing():
    field = _field([[1.02, 0], [0, 1]])
    field["strain_valid"][:] = False
    rows = summarize_regions(field, [_region()], ["u_raw", "Exx"])
    assert rows[0]["status"] == "valid" and rows[0]["n_displacement_valid"] == 18
    assert rows[0]["n_strain_valid"] == 0
    assert rows[1]["status"] == "unavailable" and rows[1]["mean"] is None
    field["valid"][:] = False
    row = summarize_regions(field, [_region()], ["u_raw"])[0]
    assert row["n_poi_total"] == 18 and row["n_poi_valid"] == 0
    assert row["mean"] is None and row["interpolation"] == "none"


def test_region_thresholds_and_global_quality_suppress_values_but_keep_evidence():
    field = _field([[1.02, 0], [0, 1]])
    field["zncc"][:18] = 0.4
    region = _region(min_zncc=0.9, min_valid_fraction=0.8)
    row = summarize_regions(field, [region], ["Exx"])[0]
    assert row["n_poi_valid"] == 9 and row["valid_fraction"] == 0.5
    assert row["mean"] is None and "insufficient_valid_fraction" in row["reasons"]
    row = summarize_regions(field, [_region()], ["Exx"], quality_ok=False)[0]
    assert row["n_poi_valid"] == 18 and row["mean"] is None
    assert row["reasons"] == ["frame_quality_rejected"]
    del field["zncc"]
    row = summarize_regions(field, [region], ["Exx"])[0]
    assert "correlation_quality_unavailable" in row["reasons"]
    assert row["mean"] is None


def test_calibrated_region_selection_with_reversed_axis_and_no_invented_scale():
    field = _field([[1.02, 0], [0, 1]])
    region = _region("physical", (2.08, 2.48, 3.92, 4.92), coordinate_system="calibrated")
    missing = summarize_regions(field, [region], ["Exx"])[0]
    assert missing["mean"] is None and "calibration_unavailable" in missing["reasons"]
    calibrated = {**field, **calibrate_field(field, CALIBRATION)}
    row = summarize_regions(calibrated, [region], ["Exx"])[0]
    assert row["n_poi_total"] == 18 and row["status"] == "valid"
    assert row["coordinate_unit"] == "mm" and row["unit"] == "dimensionless"
    assert row["mean"] == pytest.approx(0.0202, abs=1e-14)
    assert row["field_support_bounds"] == pytest.approx([2, 3.3, 3.7, 5])


def test_physical_region_and_gauge_are_invariant_to_pixel_coordinate_scale():
    field = _field([[1.08, 0], [0, 0.97]])
    scaled = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in field.items()}
    for key in ("X", "Y", "x", "y", "u", "v", "u_raw", "v_raw"):
        scaled[key] = 2 * scaled[key]
    scaled["P"][..., 0] *= 2
    scaled["P"][..., 3] *= 2
    scaled["roi"] = tuple(2 * value for value in ROI)
    field["calibration"] = CALIBRATION
    scaled["calibration"] = {**CALIBRATION, "mm_per_px": 0.01}
    region = _region("physical", (2.08, 2.48, 3.92, 4.92), coordinate_system="calibrated")
    original = summarize_regions(field, [region], ["Exx"])[0]
    changed = summarize_regions(scaled, [region], ["Exx"])[0]
    assert changed["mean"] == pytest.approx(original["mean"], abs=1e-14)
    assert changed["n_poi_total"] == original["n_poi_total"] == 18
    geometry = _resolve_gauge_geometry(
        [_gauge(a=(28, 48), b=(88, 48))], scaled["roi"], (130, 130), 17, 20, None,
        calibration=scaled["calibration"],
    )
    measured = _measure_gauges(scaled, geometry, quality_ok=True)[0]
    assert measured["initial_length_px"] == 60
    assert measured["initial_length_mm"] == pytest.approx(0.6)
    assert measured["extension_mm"] == pytest.approx(0.048, abs=1e-14)


def test_partial_outside_field_and_missing_frames_are_not_reported_as_success():
    row = summarize_regions(_field(), [_region(bounds=(-5, 24, 4, 54))], ["Exx"])[0]
    assert row["n_poi_valid"] == 18
    assert row["mean"] is None and "outside_field_support" in row["reasons"]
    row = summarize_regions({}, [_region()], ["Exx"], quality_ok=False)[0]
    assert row["n_poi_total"] == 0 and row["mean"] is None
    assert "no_poi_support" in row["reasons"]


@pytest.mark.parametrize("change", [
    {"bounds": [0, 0, 0, 1]}, {"bounds": [False, 1, 0, 1]}, {"bounds": ["0", 1, 0, 1]},
    {"min_valid_pois": True}, {"min_valid_fraction": 1.1}, {"min_zncc": math.nan},
    {"coordinate_system": "physical"}, {"roi_px": [0, 0, 10, 10]},
])
def test_region_normalization_rejects_ambiguous_or_invalid_geometry(change):
    with pytest.raises(ValueError):
        normalize_regions([{**_region(), **change}])


def test_region_names_are_unique_and_defaults_are_explicit():
    with pytest.raises(ValueError, match="unique"):
        normalize_regions([_region("A"), _region("a")])
    normalized = normalize_regions([_region()])[0]
    assert normalized["min_valid_pois"] == 3 and normalized["min_valid_fraction"] == 0.5


def test_point_gauge_does_not_jump_across_mask_hole():
    mask = np.ones(SHAPE, dtype=bool)
    mask[22:27, 12:17] = False
    geometry = _geometry([_gauge(max_distance_px=11)], mask=mask)
    assert geometry[0]["a"]["poi_indices"] == []
    assert geometry[0]["reference_status"] == "reference_support_missing"
    measured = _measure_gauges(_field(), geometry, quality_ok=True)[0]
    assert measured["engineering_strain"] is None and measured["status"] == "invalid_endpoint_support"


def test_fixed_roi_endpoint_rejects_any_missing_or_low_quality_point():
    gauge = {"name": "roi", "axis": "distance", "a_roi_px": [4, 14, 10, 20], "b_roi_px": [44, 14, 10, 20], "min_zncc": 0.9}
    geometry = _geometry([gauge], CALIBRATION)
    field = _field([[1.04, 0], [0, 1]])
    original_a = list(geometry[0]["a"]["poi_indices"])
    assert len(original_a) == 6
    field["zncc"][original_a[0]] = 0.5
    measured = _measure_gauges(field, geometry, quality_ok=True)[0]
    assert measured["a_valid_poi_count"] == 5 and measured["a_expected_poi_count"] == 6
    assert measured["engineering_strain"] is None and measured["extension_mm"] is None
    assert geometry[0]["a"]["poi_indices"] == original_a
    field["zncc"][original_a[0]] = 1.0
    field["u_raw"][original_a[0]] = np.nan
    assert _measure_gauges(field, geometry, quality_ok=True)[0]["engineering_strain"] is None


def test_reference_grid_mismatch_overlapping_endpoints_and_quality_rejection():
    field = _field([[1.04, 0], [0, 1]])
    geometry = _geometry([_gauge()], CALIBRATION)
    row = _measure_gauges(field, geometry, quality_ok=False)[0]
    assert row["initial_length_mm"] == pytest.approx(0.6)
    assert row["current_length_mm"] is None and row["engineering_strain"] is None
    field["X"] = field["X"] + 1
    row = _measure_gauges(field, geometry, quality_ok=True)[0]
    assert row["reasons"] == ["reference_grid_mismatch"]
    overlapping = _geometry([_gauge(b=(14, 24))])[0]
    assert overlapping["reference_status"] == "overlapping_endpoint_support"


def test_projected_gauge_cannot_report_nonpositive_current_length_as_valid():
    row = _measure_gauges(_field([[-1, 0], [0, -1]]), _geometry([_gauge(axis="x")]), quality_ok=True)[0]
    assert row["status"] == "invalid_deformed_length"
    assert row["engineering_strain"] is None


def test_supported_foldover_never_becomes_a_valid_line_length_measurement():
    field = _field([[-1, 0], [0, 1]])
    row = _measure_gauges(field, _geometry([_gauge()]), quality_ok=True)[0]
    assert row["status"] == "invalid_endpoint_support" and row["engineering_strain"] is None
    assert summarize_regions(field, [_region()], ["Exx"])[0]["n_strain_valid"] == 0


def test_automatic_gauges_support_vertical_geometry_without_assigning_loading_axis():
    candidates = _auto_gauge_candidates((10, 20, 100, 300), 21, 8)
    assert len(candidates) == 3
    for item in candidates:
        assert item["axis"] == "distance" and item["loading_axis"] is None
        assert item["a_roi_px"][0] == item["b_roi_px"][0]
        assert item["a_roi_px"][1] < item["b_roi_px"][1]
        assert "unconfirmed" in item["geometry_status"]


def test_roi_recommendations_have_multiple_sizes_and_never_confirm_specimen():
    image = core.generate_synthetic_speckle(180, 240, seed=41)
    recommendation = recommend_regions(image, 15, 8)
    assert len(recommendation["candidates"]) == 3
    assert len({tuple(item["roi_px"][2:]) for item in recommendation["candidates"]}) == 3
    assert recommendation["specimen_geometry_confirmed"] is False and recommendation["loading_axis"] is None
    assert recommendation["evaluated_rectangle_count"] <= 27
    for candidate in recommendation["candidates"]:
        assert candidate["eligible_poi_count"] >= 9
        assert "unconfirmed" in candidate["geometry_status"]
    empty_mask = np.zeros(image.shape, bool)
    assert recommend_regions(image, 15, 8, specimen_mask=empty_mask)["candidates"] == []


def test_texture_only_stripes_are_not_recommended():
    x = np.arange(240)
    stripes = np.repeat((127 + 60 * np.sin(x / 3))[None, :], 180, axis=0)
    result = recommend_regions(stripes, 15, 8)
    assert not result["candidates"]
    assert any(item["reason"] == "AMBIGUOUS_TEXTURE" for item in result["rejected"])


def test_missing_time_never_aligns_loads_and_frame_axis_never_invents_seconds():
    mechanics = {"force_time_s": [0, 1], "force_N": [100, 200]}
    config = {"loads": {"clock_confirmed": True, "force_column": "force_N", "max_gap_s": 1, "area_mm2": 1}}
    unavailable = _mechanics_at(None, mechanics, config)
    assert unavailable["force_N"] is None and unavailable["stress_MPa"] is None
    assert unavailable["mechanics_status"] == unavailable["force_alignment_kind"] == "time_unavailable"
    assert _mechanics_at(None, None, {})["mechanics_status"] == "not_provided"
    figure, axes, artist, summary = plot_position_time(
        [_field(), _field()], [3, 8], [[14, 24]], "Exx", 0,
        axis_unit="frame", axis_label="Native frame",
    )
    assert axes.get_ylabel() == "Native frame"
    np.testing.assert_array_equal(np.asarray(artist.get_offsets())[:, 1], [3, 8])
    assert summary["max_observed_time_gap_s"] is None
    assert summary["max_observed_axis_gap"] == 5
    import matplotlib.pyplot as plt
    plt.close(figure)
    with pytest.raises(ValueError, match="seconds"):
        plot_position_time([], [], [[14, 24]], "Exx", 0, axis_unit="frame", max_time_gap_s=1)
