"""Check displayed units against the applied settings, without requiring Tk."""

import numpy as np
import pytest

import dic_virtual_extensometer_gui_v7_multi_roi_range as gui


class _Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Viewer:
    _rebuild_field_viewer_plot = gui.MultiROIGUI._rebuild_field_viewer_plot
    _on_field_component_change = gui.MultiROIGUI._on_field_component_change
    apply_dic_display_options = gui.MultiROIGUI.apply_dic_display_options

    def __init__(self, field):
        self.dic_last_field = field
        self.dic_field_component = _Variable("Exx")
        self.dic_colormap = _Variable("viridis")
        self.dic_percent = _Variable(False)
        self.panel_bg = "white"
        self.text_color = "black"
        self.muted_color = "black"
        self.figure = None

    def _dispose_viewer_plot(self):
        pass

    def _style_viewer_plot_fonts(self, axes):
        pass

    def _style_viewer_axes_dark(self, axes):
        pass

    def _attach_viewer_canvas(self, figure):
        self.figure = figure

    def _update_field_viewer_overlay(self):
        pass

    def dic_display_options(self):
        return {"percent": self.dic_percent.get(), "color_mode": "range", "style": "points"}


def test_competing_peak_soft_risk_counts_unique_eligible_points_across_levels():
    weak = "CANDIDATE_WEAKER_THAN_GAP"
    field = {
        "valid": np.array([True, True, False, True, True, False]),
        "eligible": np.array([True, True, True, False, True, True]),
        "competing_peak_status": np.array([weak, "NO_CANDIDATE", weak, "NOT_EVALUATED", "NO_CANDIDATE", "NOT_EVALUATED"]),
        "pyramid_level_diagnostics": [
            [{"competing_peak_status": weak}, {"competing_peak_status": weak}],
            [{"competing_peak_status": weak}, {"competing_peak_status": weak}],
            [{"competing_peak_status": weak}],
            [{"competing_peak_status": weak}],
            [{"competing_peak_status": "NO_CANDIDATE"}],
            [{"competing_peak_status": "NOT_EVALUATED"}],
        ],
    }
    before = field["valid"].copy()
    result = gui.MultiROIGUI.competing_peak_soft_risk(field)
    assert result == {"flagged": 3, "flagged_valid": 2, "prediction_conflicts": 0, "prediction_conflicts_valid": 0,
                      "evaluated": 4, "unfinished": 1, "eligible": 5, "incomplete": False}
    np.testing.assert_array_equal(field["valid"], before)


def test_competing_peak_soft_risk_reports_missing_or_incomplete_records():
    fn = gui.MultiROIGUI.competing_peak_soft_risk
    assert fn({"valid": [True], "competing_peak_status": ["NOT_EVALUATED"]}) == {
        "flagged": 0, "flagged_valid": 0, "prediction_conflicts": 0, "prediction_conflicts_valid": 0,
        "evaluated": 0, "unfinished": 1, "eligible": 1, "incomplete": False}
    assert fn({"valid": [True]}) is None
    assert fn({"valid": [True], "eligible": [True, False], "competing_peak_status": ["NO_CANDIDATE"]}) is None
    field = {"valid": [True, True], "competing_peak_status": ["NO_CANDIDATE", "CANDIDATE_WEAKER_THAN_GAP"],
             "pyramid_levels_used": 3, "pyramid_level_diagnostics": [[]]}
    assert fn(field) == {"flagged": 1, "flagged_valid": 1, "prediction_conflicts": 0, "prediction_conflicts_valid": 0,
                         "evaluated": 0, "unfinished": 2, "eligible": 2, "incomplete": True}


def test_prediction_conflicts_are_unique_per_poi_across_status_boolean_and_levels():
    conflict = "PREDICTION_CONFLICT"
    no_candidate = "NO_CANDIDATE"
    field = {
        "valid": [True, True, False, False, True],
        "eligible": [True, True, True, True, False],
        "competing_peak_status": [conflict, "DISTINCT_VALID_CANDIDATE", "PYRAMID_AMBIGUOUS_PEAK", no_candidate, conflict],
        "prediction_conflict": np.array([True, True, True, False, True]),
        "pyramid_levels_used": 2,
        "pyramid_level_diagnostics": [
            [{"level_index": 0, "competing_peak_status": conflict, "prediction_conflict": True},
             {"level_index": 1, "competing_peak_status": conflict, "prediction_conflict": True}],
            [{"level_index": 0, "competing_peak_status": "CANDIDATE_WEAKER_THAN_GAP"},
             {"level_index": 1, "competing_peak_status": no_candidate}],
            [{"level_index": 0, "competing_peak_status": conflict, "prediction_conflict": True}],
            [{"level_index": 0, "competing_peak_status": "DISTINCT_VALID_CANDIDATE", "prediction_conflict": True}],
            [{"level_index": 0, "competing_peak_status": conflict, "prediction_conflict": True}],
        ],
    }
    result = gui.MultiROIGUI.competing_peak_soft_risk(field)
    assert result["prediction_conflicts"] == 4  # Multiple levels and sources do not duplicate a POI.
    assert result["prediction_conflicts_valid"] == 2
    assert result["flagged"] == 1  # A weak candidate and a conflict can overlap at one POI.
    assert result["evaluated"] == 3 and result["unfinished"] == 1
    assert result["incomplete"] is False  # Only rejected/ineligible points stopped early.
    np.testing.assert_array_equal(field["valid"], [True, True, False, False, True])


def test_competing_peak_failures_and_unknown_states_are_unfinished():
    statuses = ["NO_CANDIDATE", "BELOW_RAW_CANDIDATE_THRESHOLD", "CANDIDATE_REFINE_FAILED",
                "CANDIDATE_BOUNDARY_FAILURE", "PRIMARY_ZNCC_UNAVAILABLE", "NOT_EVALUATED", "FUTURE_UNKNOWN"]
    field = {"valid": np.ones(len(statuses), dtype=bool), "competing_peak_status": statuses}
    result = gui.MultiROIGUI.competing_peak_soft_risk(field)
    assert result["evaluated"] == 2
    assert result["unfinished"] == 5
    assert result["flagged"] == 0


def test_competing_peak_incomplete_pyramid_distinguishes_rejected_points():
    fn = gui.MultiROIGUI.competing_peak_soft_risk
    field = {"valid": [True], "competing_peak_status": ["NO_CANDIDATE"],
             "pyramid_levels_used": 3, "pyramid_level_diagnostics": [[]]}
    result = fn(field)
    assert result["incomplete"] is True
    assert result["evaluated"] == 0 and result["unfinished"] == 1

    field["valid"] = [False]
    field["competing_peak_status"] = ["NOT_EVALUATED"]
    result = fn(field)
    assert result["incomplete"] is False  # Rejected points may stop early.
    assert result["evaluated"] == 0 and result["unfinished"] == 1

    field["valid"] = [True]
    field["competing_peak_status"] = ["NO_CANDIDATE"]
    field["pyramid_level_diagnostics"] = [[{"level_index": i, "competing_peak_status": "NO_CANDIDATE"} for i in range(3)]]
    assert fn(field)["evaluated"] == 1
    assert fn(field)["incomplete"] is False
    field["pyramid_level_diagnostics"][0][1]["competing_peak_status"] = "CANDIDATE_REFINE_FAILED"
    field["pyramid_level_diagnostics"][0][2]["competing_peak_status"] = "CANDIDATE_WEAKER_THAN_GAP"
    result = fn(field)
    assert result["flagged"] == 1 and result["evaluated"] == 0 and result["unfinished"] == 1
    assert result["incomplete"] is False  # All levels exist; one check failed.


def test_switching_components_before_apply_keeps_units_with_rendered_values():
    x, y = np.meshgrid(np.arange(4) * 10.0, np.arange(4) * 10.0)
    strain = np.linspace(0.001, 0.01, x.size).reshape(x.shape)
    field = {
        "X": x, "Y": y, "u": np.ones_like(x), "v": np.zeros_like(y),
        "Exx": strain, "Eyy": strain, "zncc": np.full_like(x, 0.95),
        "valid": np.ones_like(x, dtype=bool), "strain_valid": np.ones_like(x, dtype=bool),
        "display_options": {"percent": False, "color_mode": "range", "style": "points"},
    }
    viewer = _Viewer(field)

    viewer.dic_percent.set(True)  # This is pending until Apply.
    viewer.dic_field_component.set("Eyy")
    viewer._on_field_component_change()
    assert viewer.figure.axes[1].get_ylabel() == "Eyy (1)"
    np.testing.assert_allclose(viewer.figure.axes[0].collections[0].get_clim(), [0.001, 0.01])

    viewer.apply_dic_display_options()
    assert viewer.figure.axes[1].get_ylabel() == "Eyy (%)"
    np.testing.assert_allclose(viewer.figure.axes[0].collections[0].get_clim(), [0.1, 1.0])

    viewer.dic_percent.set(False)  # Reverse the pending change as well.
    viewer.dic_field_component.set("Exx")
    viewer._on_field_component_change()
    assert viewer.figure.axes[1].get_ylabel() == "Exx (%)"
    np.testing.assert_allclose(viewer.figure.axes[0].collections[0].get_clim(), [0.1, 1.0])

    viewer.apply_dic_display_options()
    assert viewer.figure.axes[1].get_ylabel() == "Exx (1)"
    np.testing.assert_allclose(viewer.figure.axes[0].collections[0].get_clim(), [0.001, 0.01])

    viewer.dic_percent.set(True)
    viewer.apply_dic_display_options()
    for component, unit in (("u", "px"), ("zncc", "1")):
        viewer.dic_field_component.set(component)
        viewer._on_field_component_change()
        assert viewer.figure.axes[1].get_ylabel() == f"{component} ({unit})"

    np.testing.assert_array_equal(field["Exx"], strain)


def test_sequence_viewer_receives_the_displayed_frames_gate(monkeypatch, tmp_path):
    app = object.__new__(gui.MultiROIGUI)
    app.status_var = _Variable("")
    app.post_to_ui = lambda callback: callback()
    app.validate_fullfield_before_processing = lambda: None
    app._update_progress = lambda value: None
    app.log = lambda message: None
    app.show_completion_and_open_output_folder = lambda *_args: None
    shown = {}
    app.show_field_viewer = lambda field, **kwargs: shown.update(field=field, **kwargs)
    field = {"frame_global_1based": 3, "frame_filename": "frame3.png"}
    frames = [
        {"frame_global_1based": 2, "status": "scientific_valid", "correlation_valid_fraction": 1.0},
        {"frame_global_1based": 3, "status": "computed_partial", "correlation_valid_fraction": 0.6,
         "strain_valid_fraction": 0.4},
    ]
    result = {"scientific_ok": True, "last_field": field, "last_image": np.zeros((5, 5)),
              "frames": frames, "manifest": {"reference_frame_1based": 1},
              "manifest_path": str(tmp_path / "manifest.json")}
    monkeypatch.setattr(gui._core, "run_fullfield_sequence", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(gui, "archive_previous_fullfield_outputs", lambda *_args: None)
    settings = {"image_paths": ["unused"], "start_idx": 0, "end_idx": 0,
                "field_roi": (0, 0, 1, 1), "output_dir": str(tmp_path)}

    gui.MultiROIGUI.process_fullfield(app, settings)

    assert shown["field"] is field
    assert shown["frame_quality"] == frames[1]
    assert "1/2 帧通过数值质量门" in app.status_var.get()


@pytest.mark.parametrize("last_field, integrity", [(None, True), ({"frame_global_1based": 2}, False)])
def test_failed_gate_without_verified_field_is_not_shown(monkeypatch, tmp_path, last_field, integrity):
    app = object.__new__(gui.MultiROIGUI)
    app.post_to_ui = lambda callback: callback()
    app.validate_fullfield_before_processing = lambda: None
    shown = []
    app.show_field_viewer = lambda *_args, **_kwargs: shown.append(True)
    result = {"status": "scientific_gate_failed", "scientific_ok": False,
              "integrity_ok": integrity, "last_field": last_field,
              "errors": ["no_valid_field_frames"], "manifest_path": str(tmp_path / "failed.json")}
    monkeypatch.setattr(gui._core, "run_fullfield_sequence", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(gui, "archive_previous_fullfield_outputs", lambda *_args: None)
    settings = {"image_paths": ["unused"], "start_idx": 0, "end_idx": 0,
                "field_roi": (0, 0, 1, 1), "output_dir": str(tmp_path)}

    with pytest.raises(RuntimeError, match="数值质量门未通过") as caught:
        gui.MultiROIGUI.process_fullfield(app, settings)
    message = str(caught.value)
    assert str(tmp_path / "failed.json") in message
    assert ("记录完整性核验未通过" in message) is (not integrity)
    assert ("有限应变" in message) is (last_field is None)
    assert not shown
