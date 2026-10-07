"""Exercise the real Tk workbench; skip only when no desktop display exists."""

import gc
import os
import time
import tkinter as tk

import numpy as np
import pandas as pd
import pytest

import dic_virtual_extensometer_gui_v7_multi_roi_range as gui
import ezdic_core as core


@pytest.fixture
def workbench(monkeypatch, tmp_path, desktop, request):
    gc.collect()  # Dispose earlier Tcl variables on the UI thread, before any worker starts.
    monkeypatch.setenv(gui.RECENT_CONFIG_ENV_VAR, str(tmp_path / "recent.json"))
    monkeypatch.setenv(gui.UI_SCALE_ENV_VAR, str(getattr(request, "param", "1.333333")))
    root = tk.Toplevel(desktop)
    errors, dialogs = [], []
    desktop.report_callback_exception = lambda kind, value, tb: errors.append(value)
    app = gui.MultiROIGUI(root)
    app._test_errors = errors
    app._test_dialogs = dialogs
    for name in ("showerror", "showwarning", "showinfo"):
        monkeypatch.setattr(gui.messagebox, name, lambda title, message: dialogs.append((title, message)))
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(app, "show_completion_and_open_output_folder",
                        lambda message, folder: app.remember_recent_paths(output_dir=folder))
    root.geometry("1280x800+0+0")
    root.update()
    yield app
    for job in root.tk.call("after", "info"):
        root.after_cancel(job)
    # Release variable callbacks as well as widgets before another workbench
    # uses this interpreter.  Otherwise native traces retain earlier apps.
    for variable in vars(app).values():
        if isinstance(variable, tk.Variable):
            for modes, callback in variable.trace_info():
                variable.trace_remove(modes, callback)
    root.destroy()
    assert not errors


@pytest.fixture(scope="module")
def desktop():
    """One Tcl interpreter, independent workbench windows per scenario."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        if os.name == "nt" or os.environ.get("DISPLAY"):
            raise
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    yield root
    gc.collect()
    root.destroy()


def load_sequence(app, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    reference = gui.generate_synthetic_speckle(160, 200, seed=48)
    for index, tx in enumerate((0.0, 0.5, 1.0), start=1):
        image = reference if tx == 0 else gui.warp_image_translation(reference, tx, -0.2 * tx)
        gui.write_image_checked(image_dir / f"frame_{index:03d}.png", np.clip(image, 0, 255).astype(np.uint8))
    app.image_folder.set(str(image_dir))
    app.output_folder.set(str(tmp_path / "output"))
    app.load_images_button.invoke()
    app.root.update()
    assert len(app.image_paths) == 3
    assert not app._test_dialogs


def draw_roi(app, rect):
    """Use actual canvas events, including centered/scrolled coordinates."""
    x, y, width, height = rect
    scale = app.display_scale
    origin_x, origin_y = app.canvas.canvasx(0), app.canvas.canvasy(0)
    x0, y0 = round(x * scale - origin_x), round(y * scale - origin_y)
    x1, y1 = round((x + width) * scale - origin_x), round((y + height) * scale - origin_y)
    app.canvas.event_generate("<ButtonPress-1>", x=x0, y=y0)
    app.canvas.event_generate("<B1-Motion>", x=x1, y=y1)
    app.canvas.event_generate("<ButtonRelease-1>", x=x1, y=y1)
    app.root.update()


def finish_run(app):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        app.root.update()
        if not app.is_processing and not app._completion_pending and app.ui_queue.empty():
            break
        time.sleep(0.01)
    assert not app.is_processing and not app._completion_pending
    assert not app._test_dialogs, app._test_dialogs
    assert "处理完成" in app.status_var.get()
    assert app.progress_percent_var.get() == "100%"
    assert not app.image_folder_entry.instate(["disabled"])
    assert app.tracking_preset_box.instate(["readonly"])
    manifest = core.verify_run_manifest(str(app.output_folder.get()) + "/run_manifest.json")
    assert manifest["ok"], manifest
    assert app.workspace_notebook.select() == str(app.results_page)
    assert app.viewer_toolbar is not None and app.viewer_toolbar.winfo_manager() == "grid"


@pytest.mark.parametrize("geometry", ["1440x900", "1040x680"])
def test_layout_keeps_run_action_visible_and_empty_states_accessible(workbench, geometry):
    app = workbench
    app.root.geometry(geometry + "+0+0")
    app.root.update()
    assert app.start_button.instate(["disabled"])
    assert app.roi1_button.instate(["disabled"])
    assert app.add_group_button.instate(["disabled"])
    assert app.canvas.find_withtag("empty_state")
    for widget in (app.run_frame, app.start_button, app.status_label, app.controls_canvas, app.canvas):
        assert widget.winfo_width() > 20 and widget.winfo_height() > 4
        assert widget.winfo_rootx() + widget.winfo_width() <= app.root.winfo_rootx() + app.root.winfo_width()
        assert widget.winfo_rooty() + widget.winfo_height() <= app.root.winfo_rooty() + app.root.winfo_height()
    app.workspace_notebook.select(app.results_page)
    app.root.update()
    assert app.viewer_placeholder.winfo_ismapped()
    assert app.viewer_export_btn.instate(["disabled"])


def test_centered_and_zoomed_roi_coordinates_and_selection_states(workbench, tmp_path):
    app = workbench
    load_sequence(app, tmp_path)
    app.root.geometry("1040x680+0+0")
    app.root.update()
    pad_x = (app.canvas.winfo_width() - app.display_img.shape[1]) // 2
    assert abs(app.canvas.canvasx(0) + pad_x) <= 2
    app.root.geometry("1280x800+0+0")
    app.root.update()
    assert app.prev_frame_button.instate(["disabled"])
    assert not app.next_frame_button.instate(["disabled"])
    # A press in the letterbox must not create an off-image ROI.
    app.canvas.event_generate("<ButtonPress-1>", x=1, y=1)
    app.root.update()
    assert app.drag_start is None
    draw_roi(app, (30, 60, 31, 31))
    assert app.roi1 == (30, 60, 31, 31)
    assert app.add_group_button.instate(["disabled"])
    draw_roi(app, (120, 60, 31, 31))
    assert not app.add_group_button.instate(["disabled"])
    app.add_group_button.invoke()
    assert len(app.roi_groups) == 1
    assert app.load_group_button.instate(["disabled"])
    app.group_tree.selection_set("0")
    app.root.update()
    assert not app.load_group_button.instate(["disabled"])
    app.next_frame_button.invoke()
    app.next_frame_button.invoke()
    assert app.current_preview_index == 2
    assert app.next_frame_button.instate(["disabled"])
    app.prev_frame_button.invoke()
    assert app.current_preview_index == 1
    app.load_preview_frame(0)
    app.zoom_image(5)
    app.canvas.xview_moveto(0.12)
    app.canvas.yview_moveto(0.12)
    app.set_roi_mode(1)
    app.root.update()
    draw_roi(app, (40, 60, 31, 31))
    assert app.roi1 == (40, 60, 31, 31)


@pytest.mark.parametrize("mode", [gui.ANALYSIS_MODE_EXTENSOMETER, gui.ANALYSIS_MODE_FULLFIELD])
def test_analysis_runs_from_real_widgets_exports_and_restores_controls(workbench, tmp_path, mode):
    app = workbench
    load_sequence(app, tmp_path)
    if mode == gui.ANALYSIS_MODE_FULLFIELD:
        app.mode_fullfield_radio.invoke()
        app.dic_step.set(12)
        draw_roi(app, (25, 25, 140, 110))
        assert not app.roi_group_frame.winfo_ismapped()
        assert not app.export_frame.winfo_ismapped()
    else:
        draw_roi(app, (30, 60, 31, 31))
        draw_roi(app, (120, 60, 31, 31))
        app.add_group_button.invoke()
        app.export_engineering_png.set(False)
    app.root.update()
    assert not app.start_button.instate(["disabled"]), app.preflight_summary_var.get()
    gc.collect()
    app.start_button.invoke()
    assert app.is_processing
    assert app.image_folder_entry.instate(["disabled"])
    assert app.mode_fullfield_radio.instate(["disabled"])
    finish_run(app)
    if mode == gui.ANALYSIS_MODE_FULLFIELD:
        assert "2/2" in app.qc_overview_var.get()
        assert "相关 100.0%" in app.qc_overview_var.get()
        component = "Exx"
        app.dic_field_component.set(component)
        app.dic_component_box.event_generate("<<ComboboxSelected>>")
        app.root.update()
        limits = app.viewer_figure.axes[0].collections[0].get_clim()
        assert limits[0] == -limits[1]
        assert "参考帧 1" in app.field_viewer_context_var.get()
        assert f"第 {app.dic_last_frame_1based} 帧" in app.image_context_var.get()
        app.fit_image_to_view()
        assert not app._canvas_shows_field_overlay
        assert np.array_equal(app.display_img[:, :, 0], app.display_img[:, :, 1])
        assert "第 1/3 帧" in app.image_context_var.get()
        assert (tmp_path / "output" / "dic" / "frame_0002.csv").is_file()
    else:
        assert app.results_df is not None
        assert (tmp_path / "output" / "core" / "strain_G01.txt").is_file()
    app.clear_viewer()
    app.root.update()
    assert app.viewer_placeholder.winfo_ismapped()
    assert app.viewer_export_btn.instate(["disabled"])
    app.clear_sequence_dependent_state()
    app.root.update()
    assert app.viewer_placeholder.winfo_ismapped()


def test_theme_preserves_result_mode_tab_and_nan_gaps(workbench):
    app = workbench
    groups = [{"name": "axial", "role": "axial", "actual_mode": "x"},
              {"name": "transverse", "role": "transverse", "actual_mode": "y"}]
    df = pd.DataFrame([
        {"group": g["name"], "frame_global_1based": frame, "engineering_strain": strain}
        for g, strains in zip(groups, ([0.0, np.nan, 0.1], [0.0, np.nan, -0.03]))
        for frame, strain in enumerate(strains, 1)
    ])
    app.show_results_viewer(df, groups)
    assert np.isnan(app.viewer_figure.axes[0].lines[0].get_ydata()[1])
    app._viewer_mode = "poisson"
    app.viewer_mode_var.set("poisson")
    app._rebuild_viewer_plot()
    app.workspace_notebook.select(app.quality_page)
    app.toggle_dark_mode()
    app.root.update()
    assert app._viewer_mode == "poisson"
    assert app.workspace_notebook.select() == str(app.quality_page)
    assert app.style.lookup("TCombobox", "fieldbackground", ("readonly",)) == app.card_bg
    assert app.log_text.cget("background") == app.panel_bg
    assert app.preflight_summary_label.cget("state") == "disabled"
    assert app.viewer_toolbar.cget("background") == app.card_bg
    app.toggle_dark_mode()
    assert app._viewer_mode == "poisson"


@pytest.mark.parametrize("workbench", [1.333333, 1.666667, 2.5], indirect=True)
def test_field_colorbar_and_canvas_fit_at_multiple_dpi_scales(workbench):
    app = workbench
    X, Y = np.meshgrid(np.arange(6) * 12, np.arange(5) * 12)
    values = np.linspace(-0.01, 0.01, X.size).reshape(X.shape)
    field = {"X": X, "Y": Y, "Exx": values, "valid": np.ones(X.shape, dtype=bool),
             "strain_valid": np.ones(X.shape, dtype=bool)}
    app.show_field_viewer(field, component="Exx")
    app.root.update()
    canvas = app.viewer_canvas.get_tk_widget()
    assert abs(app.viewer_figure.bbox.width - canvas.winfo_width()) <= 1
    assert abs(app.viewer_figure.bbox.height - canvas.winfo_height()) <= 1
    for ax in app.viewer_figure.axes:
        box = ax.get_tightbbox(app.viewer_canvas.get_renderer())
        assert box.x0 >= -1 and box.y0 >= -1
        assert box.x1 <= canvas.winfo_width() + 1
        assert box.y1 <= canvas.winfo_height() + 1
    app.toggle_dark_mode()
    app.root.update()
    assert abs(app.viewer_figure.bbox.width - app.viewer_canvas.get_tk_widget().winfo_width()) <= 1


def test_invalid_parameters_and_completion_gate_remain_blocking(workbench, tmp_path):
    app = workbench
    load_sequence(app, tmp_path)
    app.mode_fullfield_radio.invoke()
    draw_roi(app, (25, 25, 140, 110))
    app.dic_step.set("invalid")
    assert app.start_button.instate(["disabled"])
    assert "阻止" in app.preflight_summary_var.get()
    app.dic_step.set(12)
    assert not app.start_button.instate(["disabled"])
    app._completion_pending = True
    app.update_workflow_action_states()
    assert app.start_button.instate(["disabled"])
    assert app.dic_solver_box.instate(["disabled"])
    app.analysis_mode.set(gui.ANALYSIS_MODE_EXTENSOMETER)
    app.set_analysis_mode()
    assert app.is_fullfield_mode()
    app._completion_pending = False
    app.update_workflow_action_states()
    assert app.dic_solver_box.instate(["readonly", "!disabled"])
    assert not app.start_button.instate(["disabled"])


def test_failed_worker_has_visible_feedback_and_allows_retry(workbench, tmp_path, monkeypatch):
    app = workbench
    load_sequence(app, tmp_path)
    app.show_results_viewer(pd.DataFrame({"group": ["G01"], "frame_global_1based": [1],
                                          "engineering_strain": [0.0]}), [{"name": "G01"}])
    app.mode_fullfield_radio.invoke()
    draw_roi(app, (25, 25, 140, 110))

    def fail_engine(*_args, **_kwargs):
        # Mode switching must release the earlier figure's Tk resources on
        # the UI thread rather than leave them for a worker's collection.
        gc.collect()
        raise RuntimeError("Controlled solver failure")

    monkeypatch.setattr(gui._core, "run_fullfield_sequence", fail_engine)
    app.start_button.invoke()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.root.update()
        if not app.is_processing and not app._completion_pending and app.ui_queue.empty():
            break
        time.sleep(0.01)
    assert not app.is_processing and not app._completion_pending
    assert "分析失败" in app.status_var.get()
    assert "分析失败" in app.run_state_var.get()
    assert not app.start_button.instate(["disabled"])
    assert not app.image_folder_entry.instate(["disabled"])
    assert "Controlled solver failure" in app.log_text.get("1.0", tk.END)
