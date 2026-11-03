import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.compare_dic_exports import SCHEMA, compare


def _write_csv(path, rows, columns=None):
    columns = columns or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path):
    # The highest-strain point is deliberately missing in the comparison export.
    rows = [dict(x=i, y=0, u=i / 10, v=0, Exx=0.01*i, Eyy=0.02*i, Exy=0,
                 eligible=1, valid=1, strain_valid=1, zncc=0.99,
                 invalid_reason="", strain_invalid_reason="") for i in range(4)]
    vic = [dict(X=row["x"], Y=row["y"], U=row["u"]-0.1, V=row["v"],
                exx=row["Exx"]-0.001, eyy=row["Eyy"]-0.002, exy=0,
                allowed=1, matched=1, strain_ok=1, sigma=0.02,
                failure="", strain_failure="") for row in rows[:-1]]
    _write_csv(tmp_path / "left.csv", rows)
    _write_csv(tmp_path / "right.csv", vic)
    _write_csv(tmp_path / "grid.csv", [dict(x=i, y=0, eligible=1) for i in range(4)])
    metadata = dict(reference_image_sha256="a"*64, reference_frame="dir_000001",
                    coordinate_frame="reference_image_x_right_y_down", coordinate_unit="px",
                    displacement_unit="px", displacement_reference="total_from_reference",
                    strain_measure="green_lagrange", strain_unit="1", shear_convention="tensor",
                    roi_xywh_px=[0, 0, 3, 1])
    parameters = dict(subset_size_px=31, step_px=1, strain_window_points=5,
                      shape_function="affine", reference_strategy="fixed",
                      strain_filter="local_quadratic", displacement_smoothing=0,
                      quality_policy="original validated flags; no new threshold")
    left_columns = dict(x="x", y="y", u="u", v="v", strain_xx="Exx", strain_yy="Eyy", strain_xy="Exy",
                        eligible="eligible", correlation_valid="valid", strain_valid="strain_valid",
                        quality="zncc", invalid_reason="invalid_reason", strain_invalid_reason="strain_invalid_reason")
    right_columns = dict(x="X", y="Y", u="U", v="V", strain_xx="exx", strain_yy="eyy", strain_xy="exy",
                         eligible="allowed", correlation_valid="matched", strain_valid="strain_ok",
                         quality="sigma", invalid_reason="failure", strain_invalid_reason="strain_failure")
    sources = {
        "straintrace": dict(metadata=metadata, parameters=parameters, rows_representation="complete_grid",
                            columns=left_columns, quality=dict(name="ZNCC", unit="1", meaning="normalized correlation")),
        "comparison": dict(metadata=copy.deepcopy(metadata), parameters=copy.deepcopy(parameters), rows_representation="partial_rows",
                           columns=right_columns, quality=dict(name="sigma", unit="px", meaning="VIC-exported match confidence indicator")),
    }
    sources["comparison"]["parameters"]["strain_filter"] = "VIC selected filter"
    config = dict(schema=SCHEMA, coordinate_tolerance_px=1e-6,
                  comparison_grid=dict(csv="grid.csv", columns=dict(x="x", y="y", eligible="eligible")),
                  sources=sources, frames=[
                      dict(frame_id="5", stage="uniform", image_sha256="b"*64,
                           straintrace=dict(csv="left.csv", image_sha256="b"*64),
                           comparison=dict(csv="right.csv", image_sha256="b"*64)),
                      dict(frame_id="9", stage="late_necking", image_sha256="c"*64,
                           straintrace=dict(csv="left.csv", image_sha256="c"*64), comparison=None),
                  ])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


def _save(path, config):
    path.write_text(json.dumps(config), encoding="utf-8")


def test_same_coordinates_compare_case_mapped_strain_and_keep_missing_peak(tmp_path):
    path, _ = _fixture(tmp_path)
    source_bytes = {p.name: p.read_bytes() for p in tmp_path.glob("*.csv")}
    report = compare(path, tmp_path / "results")
    frame = report["frames"][0]
    assert report["equivalence_claim"] is False
    assert report["kind"] == "agreement_not_ground_truth"
    assert report["parameter_differences"] == {"strain_filter": {
        "straintrace": "local_quadratic", "comparison": "VIC selected filter"}}
    assert frame["common_correlation_n"] == frame["common_strain_n"] == 3
    assert frame["component_differences"]["u"]["rmse"] == pytest.approx(0.1)
    assert frame["component_differences"]["strain_yy"]["bias"] == pytest.approx(0.002)
    assert frame["native_roi"]["straintrace"]["strain_peaks"]["strain_yy"]["value"] == pytest.approx(0.06)
    assert frame["common_domain_peaks"]["strain_yy"]["straintrace"]["value"] == pytest.approx(0.04)
    assert frame["fixed_domain"]["comparison"]["strain_valid_fraction"] == 0.75
    assert frame["native_roi"]["comparison"]["strain_valid_fraction"] is None
    assert frame["native_roi"]["straintrace"]["quality"]["definition"]["name"] == "ZNCC"
    assert frame["native_roi"]["comparison"]["quality"]["definition"]["name"] == "sigma"
    missing_frame = report["frames"][1]
    assert missing_frame["frame_id"] == "9"
    assert missing_frame["status"] == "no_common_displacement"
    assert missing_frame["displacement_vector_rmse_px"] is None
    assert missing_frame["fixed_domain"]["comparison"]["strain_valid_fraction"] == 0
    assert missing_frame["native_roi"]["comparison"]["strain_peaks"]["strain_yy"] is None
    with (tmp_path / "results" / "frame_0001_points.csv").open() as stream:
        points = list(csv.DictReader(stream))
    assert points[-1]["comparison_status"] == "missing"
    assert points[-1]["difference_strain_yy"] == ""
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.csv")} == source_bytes


@pytest.mark.parametrize("key,value", [
    ("reference_image_sha256", "d"*64), ("reference_frame", "dir_000002"),
    ("strain_measure", "infinitesimal"), ("strain_unit", "%"),
    ("coordinate_frame", "deformed_image"), ("displacement_unit", "mm"),
    ("shear_convention", "engineering"), ("roi_xywh_px", [0, 0, 2, 1]),
    ("displacement_reference", "incremental"),
])
def test_rejects_noncomparable_measurements_before_writing(tmp_path, key, value):
    path, config = _fixture(tmp_path)
    config["sources"]["comparison"]["metadata"][key] = value
    _save(path, config)
    with pytest.raises(ValueError):
        compare(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_target_image_identity_cannot_be_paired_by_frame_label_alone(tmp_path):
    path, config = _fixture(tmp_path)
    config["frames"][0]["comparison"]["image_sha256"] = "d"*64
    _save(path, config)
    with pytest.raises(ValueError, match="same target image"):
        compare(path, tmp_path / "out")


def test_different_grids_are_not_interpolated_and_no_overlap_is_explicit(tmp_path):
    path, _ = _fixture(tmp_path)
    filename = tmp_path / "right.csv"
    with filename.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["X"] = float(row["X"]) + 0.5
    _write_csv(filename, rows)
    report = compare(path, tmp_path / "out")
    frame = report["frames"][0]
    assert frame["status"] == "no_common_displacement"
    assert frame["common_strain_n"] == 0
    assert frame["native_roi"]["comparison"]["present_n"] == 3
    assert frame["fixed_domain"]["comparison"]["missing_n"] == 4
    assert frame["common_domain_peaks"]["strain_yy"]["amplitude_difference"] is None


@pytest.mark.parametrize("change", ["duplicate", "ambiguous", "invalid_flag", "nonfinite_valid", "strain_without_correlation"])
def test_malformed_or_ambiguous_export_is_rejected_without_partial_publication(tmp_path, change):
    path, _ = _fixture(tmp_path)
    filename = tmp_path / "right.csv"
    with filename.open() as stream:
        rows = list(csv.DictReader(stream))
    if change in {"duplicate", "ambiguous"}:
        rows.append(dict(rows[0]))
        if change == "ambiguous":
            rows[-1]["X"] = 0.5e-6
    elif change == "invalid_flag":
        rows[0]["matched"] = ""
    elif change == "nonfinite_valid":
        rows[0]["eyy"] = "NaN"
    else:
        rows[0]["matched"] = "0"
    _write_csv(filename, rows)
    with pytest.raises(ValueError):
        compare(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".dic-compare-*"))


def test_invalid_centers_stay_missing_even_when_quality_is_high(tmp_path):
    path, _ = _fixture(tmp_path)
    filename = tmp_path / "right.csv"
    with filename.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[2].update(matched="0", strain_ok="0", failure="NO_CONVERGENCE",
                   strain_failure="INVALID_CENTER", sigma="0")
    _write_csv(filename, rows)
    frame = compare(path, tmp_path / "out")["frames"][0]
    assert frame["common_correlation_n"] == 2
    assert frame["fixed_domain"]["comparison"]["strain_valid_fraction"] == 0.5
    assert frame["native_roi"]["comparison"]["correlation_failure_reasons"] == {"NO_CONVERGENCE": 1}
    assert frame["common_domain_peaks"]["strain_yy"]["straintrace"]["value"] == pytest.approx(0.02)


def test_rounding_tolerance_does_not_allow_multi_pixel_rematching(tmp_path):
    path, config = _fixture(tmp_path)
    config["coordinate_tolerance_px"] = 1
    _save(path, config)
    with pytest.raises(ValueError, match="only for coordinate rounding"):
        compare(path, tmp_path / "out")


@pytest.mark.parametrize("axis,boundary,direction", [
    ("x", 0, -1), ("x", 3, 1), ("y", 0, -1), ("y", 1, 1),
])
def test_fixed_grid_matches_rounding_across_each_roi_boundary(tmp_path, axis, boundary, direction):
    path, config = _fixture(tmp_path)
    # The fixed grid lies exactly on one ROI edge. CSV rounding places the
    # comparison coordinates just outside it without moving the material POIs.
    other_axis = "y" if axis == "x" else "x"
    for filename, names in (("grid.csv", {"x": "x", "y": "y"}),
                            ("left.csv", {"x": "x", "y": "y"}),
                            ("right.csv", {"x": "X", "y": "Y"})):
        with (tmp_path / filename).open() as stream:
            rows = list(csv.DictReader(stream))
        for index, row in enumerate(rows):
            row[names[axis]] = boundary
            row[names[other_axis]] = index / 3 if axis == "x" else index
            if filename == "right.csv":
                row[names[axis]] += direction * config["coordinate_tolerance_px"] / 2
        _write_csv(tmp_path / filename, rows)
    frame = compare(path, tmp_path / "out")["frames"][0]
    assert frame["common_displacement_n"] == frame["common_strain_n"] == 3
    assert frame["fixed_domain"]["comparison"]["strain_valid_fraction"] == 0.75
    assert frame["component_differences"]["u"]["rmse"] == pytest.approx(0.1)
    # Native ROI membership still uses the actual exported coordinates.
    assert frame["native_roi"]["comparison"]["present_n"] == 0


def test_outside_roi_peak_cannot_enter_native_or_fixed_domain_by_rounding_tolerance(tmp_path):
    path, config = _fixture(tmp_path)
    with (tmp_path / "right.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    rows.append({**rows[0], "X": 0.5, "Y": -config["coordinate_tolerance_px"] / 2,
                 "eyy": 1000})
    _write_csv(tmp_path / "right.csv", rows)
    frame = compare(path, tmp_path / "out")["frames"][0]
    assert frame["common_strain_n"] == 3
    assert frame["native_roi"]["comparison"]["present_n"] == 3
    for region in ("native_roi", "fixed_domain"):
        assert frame[region]["comparison"]["strain_peaks"]["strain_yy"]["value"] == pytest.approx(0.038)


def test_roi_edge_match_does_not_expand_coordinate_tolerance(tmp_path):
    path, config = _fixture(tmp_path)
    with (tmp_path / "right.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["Y"] = -2 * config["coordinate_tolerance_px"]
    _write_csv(tmp_path / "right.csv", rows)
    frame = compare(path, tmp_path / "out")["frames"][0]
    assert frame["common_displacement_n"] == frame["common_strain_n"] == 0
    assert frame["fixed_domain"]["comparison"]["missing_n"] == 4
    assert frame["component_differences"]["u"]["rmse"] is None


def test_existing_output_is_preserved(tmp_path):
    path, _ = _fixture(tmp_path)
    compare(path, tmp_path / "out")
    original = (tmp_path / "out" / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        compare(path, tmp_path / "out")
    assert (tmp_path / "out" / "report.json").read_bytes() == original


def test_missing_sides_and_frames_cannot_be_implicitly_dropped(tmp_path):
    path, config = _fixture(tmp_path)
    del config["frames"][1]["comparison"]
    _save(path, config)
    with pytest.raises(ValueError, match="null for a missing export"):
        compare(path, tmp_path / "out")


def test_no_strain_retains_valid_displacement_and_null_strain_difference(tmp_path):
    path, _ = _fixture(tmp_path)
    filename = tmp_path / "right.csv"
    with filename.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row.update(strain_ok="0", exx="NaN", eyy="NaN", exy="NaN", strain_failure="INSUFFICIENT_SUPPORT")
    _write_csv(filename, rows)
    frame = compare(path, tmp_path / "out")["frames"][0]
    assert frame["status"] == "no_common_strain"
    assert frame["component_differences"]["u"]["rmse"] == pytest.approx(0.1)
    assert frame["component_differences"]["strain_yy"]["rmse"] is None
    assert frame["common_domain_peaks"]["strain_yy"]["straintrace"] is None


def test_excluded_reference_point_does_not_raise_the_fixed_domain_coverage(tmp_path):
    path, _ = _fixture(tmp_path)
    filename = tmp_path / "right.csv"
    with filename.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(allowed="0", matched="0", strain_ok="0")
    _write_csv(filename, rows)
    frame = compare(path, tmp_path / "out")["frames"][0]
    assert frame["fixed_domain"]["comparison"]["excluded_n"] == 1
    assert frame["fixed_domain"]["comparison"]["strain_valid_fraction"] == 0.5


def test_documented_cli_preserves_partial_frame_status(tmp_path):
    path, _ = _fixture(tmp_path)
    script = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_dic_exports.py"
    result = subprocess.run([sys.executable, str(script), "--manifest", str(path),
                             "--output", str(tmp_path / "out")],
                            text=True, capture_output=True, check=True)
    status = json.loads(result.stdout)
    assert status["comparison_status"] == "partial_comparison"
    assert status["frames"] == 2
    assert status["frames_with_common_strain"] == 1
    assert status["equivalence_claim"] is False


def test_actual_core_foldover_export_keeps_correlation_separate_from_displacement(tmp_path, monkeypatch):
    import numpy as np
    import ezdic_core as core

    path, config = _fixture(tmp_path)
    image = core.generate_synthetic_speckle(72, 72, seed=41)

    def folded_subset(*_args, **_kwargs):
        return {"u": 0., "v": 0., "zncc": 1.,
                "p": np.array([0., -2., 0., 0., 0., 0.]), "residual_rms": 0.}

    monkeypatch.setattr(core, "refine_subset_icgn", folded_subset)
    field = core.run_2d_dic(image, image, (12, 12, 48, 48), subset_size=15, step=8,
                            initialization="local", reject_ambiguous_peaks=False, zncc_min=0.)
    assert field["valid"].all() and np.isfinite(field["u_raw"]).all()
    assert np.isnan(field["u"]).all()
    table = core.dic_field_to_dataframe(field)
    table.to_csv(tmp_path / "left.csv", index=False)
    table[["x", "y", "eligible"]].to_csv(tmp_path / "grid.csv", index=False)
    config["sources"]["straintrace"]["metadata"]["roi_xywh_px"] = [0, 0, 72, 72]
    config["sources"]["comparison"] = copy.deepcopy(config["sources"]["straintrace"])
    config["frames"] = config["frames"][:1]
    config["frames"][0]["comparison"]["csv"] = "left.csv"
    _save(path, config)
    frame = compare(path, tmp_path / "out")["frames"][0]
    assert frame["common_correlation_n"] == len(table)
    assert frame["common_displacement_n"] == 0
    assert frame["fixed_domain"]["straintrace"]["accepted_correlation_without_displacement_n"] == len(table)
    assert frame["status"] == "no_common_displacement"
    assert frame["component_differences"]["u"]["rmse"] is None


def test_compression_and_negative_shear_report_minima_on_native_and_common_support(tmp_path):
    path, _ = _fixture(tmp_path)
    with (tmp_path / "left.csv").open() as stream:
        left = list(csv.DictReader(stream))
    with (tmp_path / "right.csv").open() as stream:
        right = list(csv.DictReader(stream))
    for i, row in enumerate(left):
        row.update(Exx=-0.1*i, Exy=-0.02*i)
    for i, row in enumerate(right):
        row.update(exx=-0.1*i + 0.001, exy=[0.001, -0.05, -0.03][i])
    _write_csv(tmp_path / "left.csv", left)
    _write_csv(tmp_path / "right.csv", right)
    frame = compare(path, tmp_path / "out")["frames"][0]
    own = frame["native_roi"]["straintrace"]
    assert own["strain_peaks"]["strain_xx"]["kind"] == "signed_maximum"
    assert own["strain_peaks"]["strain_xx"]["value"] == 0
    assert own["strain_minima"]["strain_xx"]["kind"] == "signed_minimum"
    assert own["strain_minima"]["strain_xx"]["value"] == pytest.approx(-0.3)
    assert own["strain_minima"]["strain_xx"]["x"] == 3
    # The strongest compression is missing on the right; common-point extrema
    # do not pretend that this point was measured by both sides.
    native = frame["native_roi_extrema_comparison"]["signed_minimum"]["strain_xx"]
    common = frame["common_domain_minima"]["strain_xx"]
    assert native["amplitude_difference"] == pytest.approx(-0.101)
    assert native["location_distance_px"] == 1
    assert common["straintrace"]["value"] == pytest.approx(-0.2)
    assert common["amplitude_difference"] == pytest.approx(-0.001)
    assert common["location_distance_px"] == 0
    shear = frame["common_domain_minima"]["strain_xy"]
    assert shear["straintrace"]["value"] == pytest.approx(-0.04)
    assert shear["comparison"]["value"] == pytest.approx(-0.05)
    assert shear["amplitude_difference"] == pytest.approx(0.01)
    assert shear["location_distance_px"] == 1
    assert frame["common_domain_peaks"]["strain_xy"]["straintrace"]["value"] == 0


def test_tied_negative_minimum_does_not_claim_unique_location(tmp_path):
    path, _ = _fixture(tmp_path)
    filename = tmp_path / "left.csv"
    with filename.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["Exy"] = -0.1
    _write_csv(filename, rows)
    frame = compare(path, tmp_path / "out")["frames"][0]
    assert frame["common_domain_minima"]["strain_xy"]["straintrace"]["tied_point_count"] == 3
    assert frame["common_domain_minima"]["strain_xy"]["location_distance_px"] is None
    assert frame["native_roi_extrema_comparison"]["signed_minimum"]["strain_xy"]["location_distance_px"] is None
