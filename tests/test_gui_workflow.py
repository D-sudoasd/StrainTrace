"""Exercise the real Tk workbench; skip only when no desktop display exists."""

import gc
import os
import time
import tkinter as tk
from tkinter import ttk

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
    root.geometry("650x820+0+0")
    app.visual_window.geometry("1040x800+680+0")
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


@pytest.mark.parametrize("geometry", ["650x900", "620x680"])
def test_layout_keeps_run_action_visible_and_empty_states_accessible(workbench, geometry):
    app = workbench
    app.root.geometry(geometry + "+0+0")
    app.root.update()
    assert app.start_button.instate(["disabled"])
    assert app.roi1_button.instate(["disabled"])
    assert app.add_group_button.instate(["disabled"])
    assert app.canvas.find_withtag("empty_state")
    for widget in (app.run_frame, app.start_button, app.status_label, app.controls_canvas):
        assert widget.winfo_width() > 20 and widget.winfo_height() > 4
        assert widget.winfo_rootx() + widget.winfo_width() <= app.root.winfo_rootx() + app.root.winfo_width()
        assert widget.winfo_rooty() + widget.winfo_height() <= app.root.winfo_rooty() + app.root.winfo_height()
    app.visual_window.geometry("820x680+680+0")
    app.root.update()
    assert app.canvas.winfo_toplevel() == app.visual_window
    assert app.canvas.winfo_width() >= 700
    app.workspace_notebook.select(app.results_page)
    app.root.update()
    assert app.viewer_placeholder.winfo_ismapped()
    assert app.viewer_export_btn.instate(["disabled"])


def test_centered_and_zoomed_roi_coordinates_and_selection_states(workbench, tmp_path):
    app = workbench
    load_sequence(app, tmp_path)
    app.visual_window.geometry("820x680+680+0")
    app.root.update()
    pad_x = (app.canvas.winfo_width() - app.display_img.shape[1]) // 2
    assert abs(app.canvas.canvasx(0) + pad_x) <= 2
    app.visual_window.geometry("1040x800+680+0")
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
        app.dic_color_mode.set("零点对称")
        app.apply_dic_display_options()
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
    app.control_notebook.select(app.quality_page)
    app.toggle_dark_mode()
    app.root.update()
    assert app._viewer_mode == "poisson"
    assert app.control_notebook.select() == str(app.quality_page)
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


def test_exclusion_drawing_and_sequence_reset_preserve_mask_scope(workbench, tmp_path):
    app = workbench
    load_sequence(app, tmp_path)
    app.mode_fullfield_radio.invoke()
    draw_roi(app, (25, 25, 140, 110))
    app.draw_dic_exclusion()
    draw_roi(app, (65, 55, 20, 25))
    assert len(app.dic_mask_exclusions) == 1
    assert app.field_roi == (25, 25, 140, 110)
    mask, _, _ = core._resolve_specimen_mask(app.first_img8, app.field_roi,
                                           {"mask": app.dic_mask_settings()})
    assert not mask[60, 70]
    app.dic_mask_path.set("previous-specimen.png")
    app.dic_mask_mode.set("导入遮罩")
    app.clear_sequence_dependent_state()
    assert app.dic_mask_exclusions == []
    assert app.dic_mask_path.get() == ""
    assert app.dic_mask_mode.get() == "矩形 ROI"


def test_field_display_restores_percent_manual_limits_and_image_coordinates(workbench):
    app = workbench
    X, Y = np.meshgrid(np.arange(6)*10+10, np.arange(5)*10+10)
    field = {"X": X, "Y": Y, "Exx": np.linspace(.001, .01, X.size),
             "u": np.full(X.size, 5.), "v": np.zeros(X.size),
             "valid": np.ones(X.size, bool), "strain_valid": np.ones(X.size, bool),
             "display_options": {"percent": True, "color_mode": "manual", "vmin": 0.,
                                 "vmax": 1.2, "background": "deformed", "cmap": "viridis"}}
    app.show_field_viewer(field, component="Exx", image=np.full((80, 100), 120, np.uint8))
    app.root.update()
    assert app.dic_percent.get() and app.dic_color_mode.get() == "手动范围"
    assert app.dic_view_background.get() == "变形图"
    assert app.viewer_figure.axes[0].collections[0].get_clim() == (0., 1.2)
    assert app.viewer_figure.axes[1].get_ylabel() == "Exx (%)"
    before = field["Exx"].copy()
    app.dic_color_mode.set("数据范围")
    app.apply_dic_display_options()
    np.testing.assert_array_equal(field["Exx"], before)


def test_windows_resize_independently_and_keep_roi_coordinates(workbench, tmp_path):
    app = workbench
    load_sequence(app, tmp_path)
    draw_roi(app, (30, 60, 31, 31))
    original_roi = app.roi1
    assert app.canvas.winfo_toplevel() == app.visual_window
    assert app.controls_canvas.winfo_toplevel() == app.root
    assert app.quality_page.winfo_toplevel() == app.root
    visual_size = (app.canvas.winfo_width(), app.canvas.winfo_height())
    app.root.geometry("620x680+0+0")
    app.root.update()
    assert (app.canvas.winfo_width(), app.canvas.winfo_height()) == visual_size
    control_size = (app.controls_canvas.winfo_width(), app.controls_canvas.winfo_height())
    app.visual_window.geometry("820x580+680+0")
    wait_for(app, lambda: app._resize_after_id not in app.root.tk.call("after", "info"))
    assert (app.controls_canvas.winfo_width(), app.controls_canvas.winfo_height()) == control_size
    assert app.roi1 == original_roi
    draw_roi(app, (120, 60, 31, 31))
    assert app.roi2 == (120, 60, 31, 31)


def test_visual_window_close_reopens_without_losing_roi_or_result(workbench, tmp_path):
    app = workbench
    load_sequence(app, tmp_path)
    draw_roi(app, (30, 60, 31, 31))
    original_roi = app.roi1
    show_result(app, "strain")
    figure = app.viewer_figure
    callback = app.visual_window.protocol("WM_DELETE_WINDOW")
    app.root.tk.call(callback)
    app.root.update()
    assert app.visual_window.state() == "withdrawn"
    assert app.viewer_figure is figure and app.roi1 == original_roi
    assert app.root.winfo_exists()
    app.open_visual_button.invoke()
    app.root.update()
    assert app.visual_window.state() == "normal"
    assert app.workspace_notebook.select() == str(app.results_page)
    assert app.viewer_figure is figure and app.roi1 == original_roi
    for key in ("<Control-f>", "<Control-Return>", "<Control-i>"):
        assert app.root.bind(key) and app.visual_window.bind(key)


def test_completed_worker_reopens_hidden_visual_window(workbench, tmp_path):
    app = workbench
    load_sequence(app, tmp_path)
    draw_roi(app, (30, 60, 31, 31))
    draw_roi(app, (120, 60, 31, 31))
    app.add_group_button.invoke()
    app.export_engineering_png.set(False)
    app.hide_visual_window()
    app.start_button.invoke()
    finish_run(app)
    assert app.visual_window.state() == "normal"


def test_tall_field_is_centered_with_adjacent_full_height_colorbar(workbench):
    app = workbench
    X, Y = np.meshgrid(np.arange(6)*10, np.arange(60)*10)
    field = {"X": X, "Y": Y, "Exx": np.linspace(-.01, .02, X.size),
             "valid": np.ones(X.size, bool), "strain_valid": np.ones(X.size, bool)}
    app.visual_window.geometry("1040x800+680+0")
    app.show_field_viewer(field, component="Exx")
    app.root.update()
    ax, colorbar = app.viewer_figure.axes
    box, cbox = ax.get_position(), colorbar.get_position()
    assert .3 < (box.x0 + cbox.x1)/2 < .7
    assert 0 < cbox.x0 - box.x1 < .05
    assert cbox.height == pytest.approx(box.height, abs=.01)


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


# Help coverage shares this module's desktop interpreter and workbench fixture.
def widgets_under(widget):
    yield widget
    for child in widget.winfo_children():
        yield from widgets_under(child)



def wait_for(app, condition, timeout=2):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.01)
    assert condition()



def show_result(app, kind):
    if kind == "field":
        app.mode_fullfield_radio.invoke()
        X, Y = np.meshgrid(np.arange(6) * 12, np.arange(5) * 12)
        values = np.linspace(-0.01, 0.01, X.size).reshape(X.shape)
        field = {"X": X, "Y": Y, "valid": np.ones(X.shape, dtype=bool),
                 "strain_valid": np.ones(X.shape, dtype=bool)}
        field.update({component: values.copy() for component in gui.DIC_FIELD_COMPONENTS})
        app.show_field_viewer(field, component="Exx")
    elif kind == "strain":
        app.show_results_viewer(
            pd.DataFrame({"group": ["G01", "G01"], "frame_global_1based": [1, 2],
                          "engineering_strain": [0.0, 0.01]}), [{"name": "G01"}])
    app.root.update()



@pytest.mark.parametrize("kind", ["setup", "strain", "field"])
def test_every_app_control_and_dropdown_choice_has_help(workbench, kind):
    app = workbench
    show_result(app, kind)
    controls = (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Checkbutton, ttk.Radiobutton,
                ttk.Treeview, ttk.Notebook, ttk.Scrollbar, tk.Button, tk.Checkbutton,
                tk.Text, tk.Canvas)
    checked = []
    for widget in widgets_under(app.root):
        if isinstance(widget, controls):
            assert hasattr(widget, "_tooltip"), f"Missing help: {widget} ({widget.winfo_class()})"
            help_text = widget._tooltip.resolve_text()
            assert help_text and len(help_text) <= 240, (str(widget), help_text)
            checked.append(widget)
            if isinstance(widget, ttk.Combobox):
                assert set(widget._tooltip_choices) == {str(value) for value in widget.cget("values")}
                assert all(widget._tooltip_choices.values())
    assert len(checked) >= 70
    if kind != "setup":
        assert len(app.viewer_toolbar._buttons) == 7



def test_disabled_button_hover_and_keyboard_help_close_without_action(workbench, monkeypatch):
    app = workbench
    button = app.start_button
    assert button.instate(["disabled"])
    tip = button._tooltip
    button.event_generate("<Enter>", x=4, y=4)
    wait_for(app, lambda: tip.tip_window is not None)
    assert "暂不可用" in tip.shown_text and "加载" in tip.shown_text
    assert tip.tip_window.winfo_width() < 450
    button.event_generate("<Leave>")
    app.root.update()
    assert tip.tip_window is None

    escaped = []
    monkeypatch.setattr(app, "clear_current_rois", lambda: escaped.append(True))
    button.focus_force()
    app.root.update()
    button.event_generate("<F1>")
    app.root.update()
    assert tip.tip_window is not None and app.root.focus_get() == button
    button.event_generate("<Escape>")
    app.root.update()
    assert tip.tip_window is None and not escaped
    button.event_generate("<Escape>")
    app.root.update()
    assert escaped == [True]


def test_start_help_explains_actual_preflight_failures_in_plain_language(workbench, tmp_path):
    app = workbench
    assert "必须设置" in app.output_folder_entry._tooltip.resolve_text()
    load_sequence(app, tmp_path)
    app.mode_fullfield_radio.invoke()

    def blocked_help():
        app.update_workflow_action_states()
        assert app.start_button.instate(["disabled"])
        text = app.start_button._tooltip.resolve_text()
        assert len(text) <= 240
        return text.split("暂不可用：", 1)[1]

    text = blocked_help()
    assert "第一张要分析的图片" in text and "矩形分析区域" in text
    assert "ROI" not in text and "参考帧" not in text

    draw_roi(app, (20, 20, 120, 100))
    app.dic_subset_size.set(20)
    text = blocked_help()
    assert "图片块" in text and "奇数" in text and "像素" in text

    app.dic_subset_size.set(21)
    app.dic_step.set(100)
    text = blocked_help()
    assert "测量点不足" in text and "3 行、3 列" in text and "POI" not in text

    app.dic_step.set(8)
    app.start_frame_1based.set(2)
    text = blocked_help()
    assert "第一张要分析的图片" in text and "重新画" in text and "参考帧" not in text

    app.start_frame_1based.set(1)
    app.output_folder.set("")
    text = blocked_help()
    assert "结果文件夹路径" in text and "选择输出" in text

    occupied_path = tmp_path / "already-a-file.txt"
    occupied_path.write_text("existing file", encoding="utf-8")
    app.output_folder.set(str(occupied_path))
    text = blocked_help()
    assert "指向已有文件" in text and "文件夹路径" in text



@pytest.mark.parametrize("dark", [False, True])
def test_tooltip_measured_geometry_and_colors_at_screen_edge(workbench, dark):
    app = workbench
    if dark:
        app.toggle_dark_mode()
    tip = app.select_image_button._tooltip
    event = tk.Event()
    event.x_root = app.root.winfo_screenwidth() - 2
    event.y_root = app.root.winfo_screenheight() - 2
    tip.event = event
    tip.show()
    app.root.update()
    window = tip.tip_window
    assert window is not None
    assert window.winfo_rootx() >= 0 and window.winfo_rooty() >= 0
    assert window.winfo_rootx() + window.winfo_width() <= app.root.winfo_screenwidth()
    assert window.winfo_rooty() + window.winfo_height() <= app.root.winfo_screenheight()
    label = window.winfo_children()[0]
    assert str(label.cget("foreground")) == "#1f2937" and str(label.cget("background")) == "#fff8dc"
    tip.hide()



def test_tooltip_stays_on_secondary_monitor_when_available(workbench):
    app = workbench
    if app.root.winfo_vrootx() >= 0:
        pytest.skip("No monitor with negative horizontal coordinates is available")
    tip = app.select_image_button._tooltip
    event = tk.Event()
    event.x_root, event.y_root = app.root.winfo_vrootx() + 100, 500
    left, top, width, height = tip.screen_bounds(event.x_root, event.y_root)
    if left >= 0:
        pytest.skip("This Tk platform does not expose individual monitor bounds")
    tip.event = event
    tip.show()
    app.root.update()
    window = tip.tip_window
    assert left <= window.winfo_rootx() and window.winfo_rootx() + window.winfo_width() <= left + width
    assert top <= window.winfo_rooty() and window.winfo_rooty() + window.winfo_height() <= top + height
    tip.hide()



def test_dropdown_hover_explains_candidate_without_selecting_it(workbench):
    app = workbench
    box = app.strain_mode_box
    tip = box._tooltip
    original = box.get()
    app.root.tk.call("ttk::combobox::Post", str(box))
    app.root.update()
    listbox = tip.listbox
    values = box.cget("values")
    target = len(values) - 1
    bbox = app.root.tk.call(listbox, "bbox", target)
    x, y = 5, int(bbox[1]) + int(bbox[3]) // 2
    root_x = int(app.root.tk.call("winfo", "rootx", listbox)) + x
    root_y = int(app.root.tk.call("winfo", "rooty", listbox)) + y
    # Tk routes synthetic pointer events to the grab window even when a
    # listbox child is named. Release the grab only while injecting motion.
    popup = app.root.tk.call("winfo", "toplevel", listbox)
    app.root.tk.call("grab", "release", popup)
    app.root.tk.call("event", "generate", listbox, "<Motion>", "-x", x, "-y", y,
                     "-rootx", root_x, "-rooty", root_y)
    wait_for(app, lambda: tip.tip_window is not None and str(values[target]) in tip.shown_text)
    assert "直线距离" in tip.shown_text
    assert box.get() == original
    window = tip.tip_window
    list_x = int(app.root.tk.call("winfo", "rootx", listbox))
    list_width = int(app.root.tk.call("winfo", "width", listbox))
    assert window.winfo_rootx() >= list_x + list_width or window.winfo_rootx() + window.winfo_width() <= list_x
    app.root.tk.call("grab", "set", "-global", popup)
    app.root.tk.call("event", "generate", listbox, "<KeyPress-Return>")
    app.root.update()
    assert box.get() == str(values[target]) and app.strain_mode.get() == "distance"
    assert tip.tip_window is None and tip.after_id is None and tip.popup_choice is None



def test_dropdown_keyboard_navigation_explains_option_before_confirmation(workbench):
    app = workbench
    box = app.tracking_preset_box
    tip = box._tooltip
    app.root.tk.call("ttk::combobox::Post", str(box))
    app.root.update()
    app.root.tk.call("event", "generate", tip.listbox, "<KeyPress-Down>")
    app.root.tk.call("event", "generate", tip.listbox, "<KeyRelease-Down>")
    wait_for(app, lambda: tip.shown_text is not None and "低质量图像" in tip.shown_text)
    assert box.get() == "标准"
    app.root.tk.call("event", "generate", tip.listbox, "<KeyPress-Return>")
    app.root.update()
    assert box.get() == "低质量图像" and app.hard_corr.get() == gui.TRACKING_PRESETS["低质量图像"]["hard_corr"]
    assert tip.tip_window is None



def test_destroyed_or_hidden_widgets_cancel_pending_and_visible_help(workbench):
    app = workbench
    before = len(app.tooltips)
    button = ttk.Button(app.main_frame, text="temporary")
    button.grid(row=3, column=0)
    app.add_tooltip(button, "点击查看说明。")
    app.root.update()
    tip = button._tooltip
    tip.schedule()
    job = tip.after_id
    button.grid_remove()
    app.root.update()
    assert tip.after_id is None and job not in app.root.tk.call("after", "info")
    button.grid()
    app.root.update()
    tip.show_help()
    assert tip.tip_window is not None
    button.destroy()
    app.root.update()
    assert tip.tip_window is None and len(app.tooltips) == before



def test_rebuilt_result_plots_release_their_old_help(workbench):
    app = workbench
    show_result(app, "field")
    expected = len(app.tooltips)
    for _ in range(3):
        old_button = app.viewer_toolbar._buttons["Home"]
        old_button._tooltip.show_help()
        app._rebuild_field_viewer_plot()
        app.root.update()
        assert not old_button.winfo_exists()
        assert old_button._tooltip.tip_window is None
        assert len(app.tooltips) == expected



def test_notebook_tabs_and_group_headers_explain_their_specific_meaning(workbench):
    app = workbench
    notebook = app.workspace_notebook
    help_by_tab = {}
    for x in range(0, notebook.winfo_width(), 5):
        event = tk.Event()
        event.x, event.y = x, 20
        try:
            index = notebook.index(f"@{x},20")
        except tk.TclError:
            continue
        help_by_tab[index] = notebook._tooltip.resolve_text(event)
    assert "画测量框" in help_by_tab[0]
    assert "曲线" in help_by_tab[1]
    app.control_notebook.select(app.quality_page)
    assert "缺少什么" in app._control_workspace_help()
    app.control_notebook.select(app.settings_page)
    app.root.update()
    tree = app.group_tree
    explanations = {}
    for x in range(0, tree.winfo_width(), 5):
        event = tk.Event()
        event.x, event.y = x, 8
        if tree.identify_region(x, 8) == "heading":
            explanations[tree.identify_column(x)] = tree._tooltip.resolve_text(event)
    assert "初始间距" in explanations["#4"]
    assert "左右" in explanations["#3"]



def test_right_click_actions_reuse_help_and_select_the_clicked_group(workbench, tmp_path, monkeypatch):
    app = workbench
    load_sequence(app, tmp_path)
    draw_roi(app, (30, 60, 31, 31))
    draw_roi(app, (120, 60, 31, 31))
    app.add_group_button.invoke()
    app.root.update()
    inspected = []

    def inspect_popup(menu, *_args):
        for index, expected in ((0, "填回"), (1, "替换"), (3, "确认删除")):
            assert menu.entrycget(index, "state") == "normal"
            menu.activate(index)
            assert expected in menu._tooltip.resolve_text()
        inspected.append(True)

    monkeypatch.setattr(tk.Menu, "tk_popup", inspect_popup)
    event = tk.Event()
    event.x_root, event.y_root = 100, 100
    event.y = app.group_tree.bbox("0")[1] + 3
    app._show_group_tree_context_menu(event)
    assert inspected and app.group_tree.selection() == ("0",)



def test_subplot_dialog_sliders_and_reset_have_contextual_help(workbench):
    app = workbench
    show_result(app, "strain")
    app.viewer_toolbar._buttons["Subplots"].invoke()
    tool = app.viewer_toolbar.subplot_tool
    app.root.update()
    canvas = tool.figure.canvas.get_tk_widget()
    assert tool.buttonreset.label.get_text() == "恢复初始值"
    for axes, expected in ((tool.sliderleft.ax, "左侧留白"),
                           (tool.sliderwspace.ax, "并排"),
                           (tool.buttonreset.ax, "恢复")):
        bbox = axes.get_window_extent()
        event = tk.Event()
        event.x, event.y = bbox.x0 + bbox.width / 2, canvas.winfo_height() - bbox.y0 - bbox.height / 2
        assert expected in canvas._tooltip.resolve_text(event)
    app.clear_viewer()
    app.root.update()
    assert canvas._tooltip.text == "" and canvas._tooltip not in app.tooltips
