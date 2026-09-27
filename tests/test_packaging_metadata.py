from pathlib import Path
import os
import shutil
import subprocess
import sys
import threading
import time
import tkinter.font as tkfont

import cv2
import numpy as np
import pandas as pd
import pytest

import dic_virtual_extensometer_gui_v7_multi_roi_range as ezdic


ROOT = Path(__file__).resolve().parents[1]
DOI = "10.5281/zenodo.20222465"
DOI_URL = f"https://doi.org/{DOI}"


@pytest.fixture(scope="module")
def gui_app(tmp_path_factory):
    import tkinter as tk

    old_recent_config = os.environ.get("EZDIC_RECENT_CONFIG")
    recent_config = tmp_path_factory.mktemp("ezdic_gui_state") / "recent_paths.json"
    os.environ["EZDIC_RECENT_CONFIG"] = str(recent_config)
    root = tk.Tk()
    root.withdraw()
    try:
        app = ezdic.MultiROIGUI(root)
        root.update_idletasks()
        yield root, app
    finally:
        root.destroy()
        if old_recent_config is None:
            os.environ.pop("EZDIC_RECENT_CONFIG", None)
        else:
            os.environ["EZDIC_RECENT_CONFIG"] = old_recent_config


def write_test_image(path, value=120, shape=(100, 140)):
    arr = np.full(shape, value, dtype=np.uint8)
    ok, data = cv2.imencode(".png", arr)
    assert ok
    data.tofile(str(path))


def reset_gui_app(app):
    app.image_folder.set("")
    app.output_folder.set("")
    app.search_radius.set(180)
    app.hard_corr.set(0.55)
    app.soft_corr.set(0.35)
    app.strain_mode.set("auto")
    app.sync_strain_mode_display()
    app.roi_role.set("none")
    app.sync_roi_role_display()
    app.tracking_preset.set("标准")
    app.preset_status_var.set("当前追踪模式：标准")
    app.enable_adaptive.set(True)
    app.use_prev_frame_template.set(False)
    app.template_alpha.set(0.70)
    app.max_frame_strain_jump.set("0.01")
    app.enable_fb_check.set(True)
    app.fb_tolerance_px.set(12.0)
    app.overlay_every.set(5)
    app.pixel_size_mm.set("")
    app.auto_align_roi2.set(True)
    app.min_texture_std.set(8.0)
    app.min_texture_contrast.set(25.0)
    app.max_saturated_frac.set(0.20)
    app.export_origin_txt.set(True)
    app.export_engineering_png.set(True)
    app.export_qc_summary.set(True)
    app.export_full_csv.set(False)
    app.export_corr_plot.set(False)
    app.export_overlays.set(False)
    app.export_parameters.set(False)
    if hasattr(app, "export_publication_figures"):
        app.export_publication_figures.set(False)
    if hasattr(app, "export_origin_opju"):
        app.export_origin_opju.set(False)
    if hasattr(app, "analysis_mode"):
        app.is_processing = False
        app._completion_pending = False
        app.analysis_mode.set(ezdic.ANALYSIS_MODE_EXTENSOMETER)
        app._committed_analysis_mode = ezdic.ANALYSIS_MODE_EXTENSOMETER
    if hasattr(app, "dic_subset_size"):
        app.dic_subset_size.set(21)
        app.dic_step.set(5)
        app.dic_solver.set(ezdic.DIC_SOLVER_ICGN)
        app.dic_strain_window.set(5)
        app.dic_smooth_sigma.set(0.0)
        app.dic_field_component.set("u")
        app.field_roi = None
        app.dic_last_field = None
        app.dic_last_image = None
        app.dic_last_frame_1based = None
        app.dic_last_filename = None
        app.dic_last_reference_frame_1based = None
        app.dic_last_reference_filename = None
        app.field_viewer_context_var.set("")
        app._canvas_shows_field_overlay = False
    app.image_paths = []
    app.loaded_image_folder = None
    app.loaded_image_sequence_fingerprint = None
    app.first_raw = None
    app.first_img8 = None
    app.current_fullres_img8 = None
    app.display_img = None
    app.display_scale = 1.0
    app.zoom_factor = 1.0
    app.auto_fit_enabled = True
    app.photo = None
    app.preview_frame_1based.set(1)
    app.start_frame_1based.set(1)
    app.end_frame_1based.set(1)
    app.current_preview_index = 0
    app.roi1 = None
    app.roi2 = None
    app.roi1_reference_frame_1based = None
    app.roi2_reference_frame_1based = None
    app.field_roi_reference_frame_1based = None
    app.roi_groups.clear()
    app.next_group_idx = 1
    app.group_name_var.set("")
    app.refresh_group_tree()
    app.canvas.delete("all")
    app.results_df = None
    app.results_groups = None
    app.last_qc_summary = None
    app._completion_pending = False
    app._active_run_token = None
    app.is_processing = False
    if hasattr(app, "analysis_mode"):
        app.set_analysis_mode()


def load_two_frame_sequence(app, folder, output_folder):
    folder.mkdir(parents=True, exist_ok=True)
    write_test_image(folder / "frame_001.png", value=100)
    write_test_image(folder / "frame_002.png", value=130)
    output_folder.mkdir(parents=True, exist_ok=True)
    app.image_folder.set(str(folder))
    app.output_folder.set(str(output_folder))
    app.load_first_image()


def add_basic_roi_group(app):
    app.roi1 = (10, 10, 30, 30)
    app.roi2 = (80, 10, 30, 30)
    app.strain_mode.set("x")
    app.sync_strain_mode_display()
    app.add_current_group()


def actual_font_size(root, font_spec):
    return abs(tkfont.Font(root=root, font=font_spec).actual("size"))


def copy_release_contract_inputs(project):
    """Create a minimal but complete metadata fixture for build smoke tests."""
    names = [
        "VERSION.txt",
        "dic_virtual_extensometer_gui_v7_multi_roi_range.py",
        ".zenodo.json",
        "CITATION.cff",
        "requirements.txt",
        "requirements-origin.txt",
        "requirements-build.txt",
        "ezDIC.spec",
        "README_使用说明.txt",
        "NOTICE_Attribution_and_Usage.txt",
        "LICENSE.txt",
    ]
    for name in names:
        shutil.copy2(ROOT / name, project / name)


def assert_widget_fully_visible_in(container, widget, *, min_width=20, min_height=10):
    assert widget.winfo_ismapped(), f"{widget} should be mapped"
    assert widget.winfo_width() >= min_width
    assert widget.winfo_height() >= min_height

    c_x0 = container.winfo_rootx()
    c_y0 = container.winfo_rooty()
    c_x1 = c_x0 + container.winfo_width()
    c_y1 = c_y0 + container.winfo_height()

    w_x0 = widget.winfo_rootx()
    w_y0 = widget.winfo_rooty()
    w_x1 = w_x0 + widget.winfo_width()
    w_y1 = w_y0 + widget.winfo_height()

    assert c_x0 <= w_x0 < c_x1
    assert c_y0 <= w_y0 < c_y1
    assert w_x1 <= c_x1
    assert w_y1 <= c_y1


def scroll_workflow_widget_into_view(root, app, widget):
    root.update()
    root.update_idletasks()
    bbox = app.controls_canvas.bbox("all")
    if not bbox:
        return
    content_height = max(bbox[3] - bbox[1], 1)
    visible_height = max(app.controls_canvas.winfo_height(), 1)
    widget_y = widget.winfo_rooty() - app.controls_panel.winfo_rooty()
    target_y = max(widget_y - 24, 0)
    fraction = 0.0 if content_height <= visible_height else target_y / content_height
    app.controls_canvas.yview_moveto(min(max(fraction, 0.0), 1.0))
    root.update()
    root.update_idletasks()


def test_gui_preserves_platform_dpi_scaling(gui_app):
    root, app = gui_app
    app_scaling = float(root.tk.call("tk", "scaling"))

    assert app.ui_scaling > 1.05
    assert app_scaling > 1.05
    assert root.winfo_fpixels("1i") > 76


def test_gui_uses_readable_base_fonts(gui_app):
    root, app = gui_app
    root.update_idletasks()

    assert actual_font_size(root, app.style.lookup("TLabel", "font")) >= 11
    assert actual_font_size(root, app.style.lookup("TButton", "font")) >= 11
    assert actual_font_size(root, app.style.lookup("TLabelframe.Label", "font")) >= 12
    assert int(app.style.lookup("Treeview", "rowheight")) >= 30
    assert actual_font_size(root, app.log_text.cget("font")) >= 10


def test_viewer_plot_uses_readable_legend_font(gui_app):
    root, app = gui_app
    df = pd.DataFrame(
        {
            "group": ["G01", "G01"],
            "frame_global_1based": [1, 2],
            "engineering_strain": [0.0, 0.012],
        }
    )
    groups = [{"name": "G01", "role": "none"}]

    try:
        app.show_results_viewer(df, groups)
        root.update_idletasks()

        ax = app.viewer_figure.axes[0]
        legend = ax.get_legend()

        assert legend is not None
        assert legend.get_texts()[0].get_fontsize() >= 9
        assert ax.xaxis.label.get_size() >= 10
        assert ax.yaxis.label.get_size() >= 10
    finally:
        app.clear_viewer(keep_placeholder=False)
        app.viewer_frame.grid_remove()


def test_app_metadata_and_usage_notice_are_explicit():
    assert ezdic.APP_NAME == "ezDIC"
    assert ezdic.APP_VERSION == "0.1.4"
    assert ezdic.APP_DEVELOPER == "Dr. Delun Gong"
    assert ezdic.APP_DOI == DOI
    assert ezdic.APP_DOI_URL == DOI_URL
    assert "developed by Dr. Delun Gong" in ezdic.USAGE_NOTICE
    assert DOI in ezdic.USAGE_NOTICE
    assert "Gong, D. (2026)." in ezdic.CITATION_TEXT
    assert DOI_URL in ezdic.CITATION_TEXT
    assert "claim that they developed this software" in ezdic.USAGE_NOTICE
    assert "redistribute, copy, forward, or share" in ezdic.USAGE_NOTICE


def test_window_title_contains_developer(gui_app):
    root, _app = gui_app
    title = root.title()

    assert title == "ezDIC v0.1.4 - Developed by Dr. Delun Gong - DOI: 10.5281/zenodo.20222465"


def test_gui_initializes_poisson_role_selection(gui_app):
    _root, app = gui_app
    assert app.roi_role.get() == "none"
    assert app.roi_role_display.get() == "普通"


def test_gui_emphasizes_start_analysis_action(gui_app):
    _root, app = gui_app
    assert app.start_button.cget("text") == "开始分析并导出结果"
    assert app.start_button.cget("style") == "Primary.TButton"
    assert "下一步" in app.workflow_hint_var.get()
    assert getattr(app, "workflow_hint_label", None) is not None
    assert getattr(app, "workflow_guide_frame", None) is not None


def test_gui_initial_layout_fits_default_window_height(gui_app):
    root, app = gui_app
    root.update_idletasks()

    assert root.winfo_reqheight() <= 980
    log_height = int(app.log_text.cget("height"))
    assert 3 <= log_height <= 10
    assert app.progress is not None
    assert str(app.progress.winfo_class())
    assert app.status_label is not None
    assert app.status_var.get().strip()


def test_gui_minimum_size_fits_research_laptop_width(gui_app):
    root, _app = gui_app
    root.update_idletasks()

    min_w, min_h = root.minsize()
    assert min_w >= root.winfo_reqwidth()
    assert min_w <= 1366
    assert min_h <= 768


def test_gui_layout_fits_research_laptop_viewport(gui_app):
    root, app = gui_app
    root.deiconify()
    root.geometry("1366x768+0+0")
    root.update()
    root.update_idletasks()
    app.controls_canvas.yview_moveto(0)
    root.update()
    root.update_idletasks()

    min_w, min_h = root.minsize()
    assert min_w <= 1366
    assert min_h <= 768

    for attr in [
        "project_frame",
        "controls_canvas",
        "controls_panel",
        "image_frame",
        "analysis_frame",
        "workflow_guide_frame",
        "workflow_hint_label",
        "progress",
        "status_label",
        "log_text",
    ]:
        assert getattr(app, attr, None) is not None

    assert_widget_fully_visible_in(app.controls_frame, app.workflow_guide_frame)
    assert_widget_fully_visible_in(app.controls_frame, app.workflow_hint_label)
    assert app.workflow_hint_var.get().strip()

    root_w = root.winfo_width()
    root_h = root.winfo_height()
    root_x = root.winfo_rootx()
    root_y = root.winfo_rooty()
    for widget in [app.canvas, app.controls_canvas, app.measure_frame, app.roi1_button, app.roi2_button]:
        assert widget.winfo_width() > 20
        assert widget.winfo_height() > 10
        x0 = widget.winfo_rootx() - root_x
        y0 = widget.winfo_rooty() - root_y
        x1 = x0 + widget.winfo_width()
        y1 = y0 + widget.winfo_height()
        assert 0 <= x0 < root_w, f"{widget} x0={x0}"
        assert 0 <= y0 < root_h, f"{widget} y0={y0} root_h={root_h}"
        assert x1 <= root_w, f"{widget} x1={x1} root_w={root_w}"
        assert y1 <= root_h, f"{widget} y1={y1} root_h={root_h}"

    assert_widget_fully_visible_in(root, app.roi1_button)
    assert_widget_fully_visible_in(root, app.roi2_button)

    assert app.controls_panel.winfo_reqheight() > app.controls_canvas.winfo_height()
    for widget in [app.group_tree, app.start_button, app.progress, app.log_text]:
        scroll_workflow_widget_into_view(root, app, widget)
        assert_widget_fully_visible_in(app.controls_canvas, widget)
    root.withdraw()


def test_export_preset_buttons_fit_research_laptop_viewport(gui_app):
    root, app = gui_app
    root.deiconify()
    root.geometry("1366x768+0+0")
    root.update()
    root.update_idletasks()
    app.controls_canvas.yview_moveto(0)
    root.update()
    root.update_idletasks()

    assert app.export_preset_label.cget("text") == "导出预设："

    for widget in [
        app.export_preset_label,
        app.export_research_preset_button,
        app.export_quick_preset_button,
        app.export_all_preset_button,
    ]:
        scroll_workflow_widget_into_view(root, app, widget)
        assert_widget_fully_visible_in(app.controls_canvas, widget)

    root.withdraw()


def test_measurement_settings_are_primary_and_visible_on_laptop_viewport(gui_app):
    root, app = gui_app
    root.deiconify()
    root.geometry("1366x768+0+0")
    root.update()
    root.update_idletasks()
    app.controls_canvas.yview_moveto(0)
    root.update()
    root.update_idletasks()

    assert getattr(app, "measure_frame", None) is not None
    assert getattr(app, "workflow_canvas", None) is app.controls_canvas
    assert getattr(app, "workflow_panel", None) is app.controls_panel
    assert app.controls_canvas.winfo_height() >= 500
    assert app.measure_frame.winfo_height() >= 220
    assert_widget_fully_visible_in(app.controls_frame, app.workflow_hint_label)

    for widget in [
        app.measure_frame,
        app.preview_frame_entry,
        app.start_frame_entry,
        app.end_frame_entry,
        app.strain_mode_box,
        app.tracking_preset_box,
        app.pixel_size_entry,
        app.auto_align_roi2_check,
        app.advanced_toggle_btn,
    ]:
        assert_widget_fully_visible_in(app.controls_canvas, widget)

    root.withdraw()


def test_minimum_view_keeps_measurement_panel_and_image_canvas_useful(gui_app):
    root, app = gui_app
    root.deiconify()
    root.geometry("1120x740+0+0")
    root.update()
    root.update_idletasks()
    app.controls_canvas.yview_moveto(0)
    root.update()
    root.update_idletasks()

    assert app.canvas.winfo_width() >= 560
    assert app.canvas.winfo_height() >= 360
    assert_widget_fully_visible_in(app.controls_canvas, app.measure_frame)
    assert_widget_fully_visible_in(app.controls_canvas, app.strain_mode_box)
    assert_widget_fully_visible_in(app.controls_canvas, app.tracking_preset_box)
    assert_widget_fully_visible_in(app.controls_canvas, app.advanced_toggle_btn)

    root.withdraw()


def test_expanded_advanced_settings_scroll_without_hiding_base_measurement_controls(gui_app):
    root, app = gui_app
    root.deiconify()
    root.geometry("1366x768+0+0")
    try:
        if not app.advanced_visible.get():
            app.toggle_advanced_settings()
        root.update()
        root.update_idletasks()
        app.controls_canvas.yview_moveto(0)
        root.update()
        root.update_idletasks()

        assert app.advanced_visible.get() is True
        assert app.advanced_frame.winfo_ismapped()
        assert app.controls_panel.winfo_reqheight() > app.controls_canvas.winfo_height()
        assert app.advanced_frame.winfo_width() <= app.controls_canvas.winfo_width()

        for widget in [
            app.preview_frame_entry,
            app.start_frame_entry,
            app.end_frame_entry,
            app.strain_mode_box,
            app.tracking_preset_box,
        ]:
            assert_widget_fully_visible_in(app.controls_canvas, widget)
    finally:
        if app.advanced_visible.get():
            app.toggle_advanced_settings()
        root.withdraw()


def test_workflow_panel_supports_mouse_wheel_scrolling(gui_app, monkeypatch):
    root, app = gui_app
    root.deiconify()
    root.geometry("1366x768+0+0")
    root.update()
    root.update_idletasks()

    assert app.controls_canvas.bind("<MouseWheel>")
    assert app.controls_canvas.bind("<Button-4>")
    assert app.controls_canvas.bind("<Button-5>")
    assert app.controls_panel.bind("<MouseWheel>")

    calls = []

    def fake_scroll(amount, unit):
        calls.append((amount, unit))

    monkeypatch.setattr(app.controls_canvas, "yview_scroll", fake_scroll)

    wheel_down = type("Event", (), {"delta": -120, "num": 0})()
    assert app._on_workflow_mouse_wheel(wheel_down) == "break"
    assert calls[-1] == (1, "units")

    wheel_up = type("Event", (), {"delta": 0, "num": 4})()
    assert app._on_workflow_mouse_wheel(wheel_up) == "break"
    assert calls[-1] == (-1, "units")

    root.withdraw()


def test_loaded_image_auto_fits_current_canvas(gui_app, tmp_path):
    root, app = gui_app
    reset_gui_app(app)
    root.deiconify()
    root.geometry("1366x768+0+0")
    root.update()
    root.update_idletasks()

    image_dir = tmp_path / "wide_images"
    output_dir = tmp_path / "out"
    image_dir.mkdir()
    output_dir.mkdir()
    write_test_image(image_dir / "frame_001.png", value=120, shape=(900, 1600))

    app.image_folder.set(str(image_dir))
    app.output_folder.set(str(output_dir))
    app.load_first_image()
    root.update()
    root.update_idletasks()

    canvas_w = app.canvas.winfo_width()
    canvas_h = app.canvas.winfo_height()
    disp_h, disp_w = app.display_img.shape[:2]

    assert app.auto_fit_enabled is True
    assert disp_w <= canvas_w
    assert disp_h <= canvas_h
    assert app.display_scale == pytest.approx(min((canvas_w - 8) / 1600, (canvas_h - 8) / 900, 1.0))
    root.withdraw()


def test_auto_fit_shrinks_when_canvas_becomes_smaller(gui_app, tmp_path):
    root, app = gui_app
    reset_gui_app(app)
    root.deiconify()
    root.geometry("1480x900+0+0")
    root.update()
    root.update_idletasks()

    image_dir = tmp_path / "large_images"
    output_dir = tmp_path / "out"
    image_dir.mkdir()
    output_dir.mkdir()
    write_test_image(image_dir / "frame_001.png", value=120, shape=(1000, 1800))

    app.image_folder.set(str(image_dir))
    app.output_folder.set(str(output_dir))
    app.load_first_image()
    root.update()
    root.update_idletasks()
    large_h, large_w = app.display_img.shape[:2]

    root.geometry("1120x740+0+0")
    root.update()
    root.update_idletasks()
    app.fit_image_to_view()
    root.update()
    root.update_idletasks()

    small_h, small_w = app.display_img.shape[:2]
    assert small_w <= app.canvas.winfo_width()
    assert small_h <= app.canvas.winfo_height()
    assert small_w <= large_w
    assert small_h <= large_h
    root.withdraw()


def test_gui_beginner_workflow_and_key_button_tooltips_are_available(gui_app):
    _root, app = gui_app

    workflow_text = "\n".join(getattr(app, "workflow_step_texts", []))
    assert "图像文件夹" in workflow_text
    assert "ROI" in workflow_text
    assert "开始分析" in workflow_text

    tooltip_targets = [
        ("load_images_button", getattr(app, "load_images_button", None), "加载"),
        ("add_group_button", getattr(app, "add_group_button", None), "ROI"),
        ("start_button", app.start_button, "开始分析"),
    ]
    for name, widget, expected_text in tooltip_targets:
        assert widget is not None, f"{name} should be available for GUI tests"
        assert expected_text in getattr(widget, "_tooltip_text", "")


def test_gui_primary_interactive_controls_have_scientific_tooltips(gui_app):
    _root, app = gui_app

    required_attrs = [
        "image_folder_entry",
        "select_image_button",
        "output_folder_entry",
        "select_output_button",
        "load_images_button",
        "preview_frame_entry",
        "show_preview_button",
        "prev_frame_button",
        "next_frame_button",
        "start_frame_entry",
        "end_frame_entry",
        "set_start_button",
        "set_end_button",
        "strain_mode_box",
        "tracking_preset_box",
        "pixel_size_entry",
        "auto_align_roi2_check",
        "advanced_toggle_btn",
        "search_radius_entry",
        "hard_corr_entry",
        "soft_corr_entry",
        "max_frame_strain_jump_entry",
        "fb_tolerance_entry",
        "template_alpha_entry",
        "min_texture_std_entry",
        "min_texture_contrast_entry",
        "max_saturated_frac_entry",
        "overlay_every_entry",
        "enable_adaptive_check",
        "use_prev_frame_template_check",
        "enable_fb_check_check",
        "roi1_button",
        "roi2_button",
        "align_x_button",
        "align_y_button",
        "group_name_entry",
        "roi_role_box",
        "add_group_button",
        "update_group_button",
        "load_group_button",
        "delete_group_button",
        "clear_rois_button",
        "group_tree",
        "workflow_hint_label",
        "canvas",
        "start_button",
        "viewer_export_btn",
        "viewer_clear_btn",
        "usage_notice_button",
    ]

    for attr in required_attrs:
        widget = getattr(app, attr, None)
        assert widget is not None, f"{attr} should be stored for GUI help verification"
        tooltip = getattr(widget, "_tooltip_text", "").strip()
        assert len(tooltip) >= 24, f"{attr} should have a practical tooltip"

    for checkbutton in app.export_checkbuttons:
        tooltip = getattr(checkbutton, "_tooltip_text", "").strip()
        assert len(tooltip) >= 24

    assert "调大" in app.search_radius_entry._tooltip_text
    assert "留空" in app.pixel_size_entry._tooltip_text
    assert "勾选" in app.auto_align_roi2_check._tooltip_text


def test_gui_key_settings_use_light_visual_emphasis(gui_app):
    _root, app = gui_app

    assert app.analysis_range_label.cget("style") == "Key.TLabel"
    assert app.strain_mode_label.cget("style") == "Key.TLabel"
    assert app.export_hint_label.cget("style") == "Warning.TLabel"
    assert app.start_button.cget("style") == "Primary.TButton"
    assert app.select_image_button.cget("style") == "Secondary.TButton"
    assert app.select_output_button.cget("style") == "Secondary.TButton"
    assert app.load_images_button.cget("style") == "Secondary.TButton"
    assert app.delete_group_button.cget("style") == "Danger.TButton"
    assert app.clear_rois_button.cget("style") == "Danger.TButton"


def test_gui_uses_clear_action_button_labels(gui_app):
    _root, app = gui_app

    expected_text = {
        "select_image_button": "选图像文件夹",
        "select_output_button": "选输出文件夹",
        "load_images_button": "加载/刷新序列",
        "show_preview_button": "显示预览帧",
        "set_start_button": "设为起始/参考",
        "set_end_button": "设为结束帧",
        "start_button": "开始分析并导出结果",
        "delete_group_button": "删除选中组",
        "clear_rois_button": "清除当前 ROI",
        "viewer_export_btn": "导出预览图",
        "viewer_clear_btn": "清除预览图",
    }
    for attr, text in expected_text.items():
        assert getattr(app, attr).cget("text") == text


def test_roi_group_tree_headings_are_chinese_task_labels(gui_app):
    _root, app = gui_app
    raw_ids = ("name", "role", "selected", "actual")
    columns = app.group_tree["columns"]
    if isinstance(columns, str):
        columns = tuple(columns.split())
    else:
        columns = tuple(columns)
    assert columns[:4] == raw_ids
    for col in columns:
        text = str(app.group_tree.heading(col, "text")).strip()
        assert text, f"{col} heading should not be empty"
        assert text not in raw_ids
    for col in raw_ids:
        text = str(app.group_tree.heading(col, "text"))
        assert text != col
        assert any("\u4e00" <= ch <= "\u9fff" for ch in text), text
    assert "组名" == app.group_tree.heading("name", "text")
    assert "角色" == app.group_tree.heading("role", "text")
    assert "所选方向" == app.group_tree.heading("selected", "text")
    assert "实际方向" == app.group_tree.heading("actual", "text")


def test_export_preset_buttons_are_named_and_explained(gui_app):
    _root, app = gui_app

    expected = {
        "export_research_preset_button": ("推荐导出", "核心"),
        "export_quick_preset_button": ("快速查看导出", "快速检查"),
        "export_all_preset_button": ("全量复核导出", "文件数量"),
    }
    for attr, (text, tooltip_keyword) in expected.items():
        button = getattr(app, attr, None)
        assert button is not None, f"{attr} should be stored for export preset help verification"
        assert button.cget("text") == text
        tooltip = getattr(button, "_tooltip_text", "")
        assert tooltip_keyword in tooltip
        assert len(tooltip.strip()) >= 24


def test_start_analysis_button_reflects_workflow_readiness(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)

    assert str(app.start_button.cget("state")) == "disabled"
    assert "加载" in app.workflow_hint_var.get()

    load_two_frame_sequence(app, tmp_path / "images_for_state", tmp_path / "out_for_state")
    assert str(app.start_button.cget("state")) == "disabled"
    assert "ROI" in app.workflow_hint_var.get()

    add_basic_roi_group(app)
    assert str(app.start_button.cget("state")) == "normal"
    assert "开始分析" in app.workflow_hint_var.get()

    app.is_processing = True
    try:
        app.update_workflow_action_states()
        assert str(app.start_button.cget("state")) == "disabled"
        assert "正在处理" in app.workflow_hint_var.get()
    finally:
        app.is_processing = False
        app.update_workflow_action_states()
    assert str(app.start_button.cget("state")) == "normal"


def test_preflight_panel_blocks_missing_inputs_and_export_options(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)

    initial_items = app.build_preflight_items()
    assert any(item["level"] == "block" and item["label"] == "图像序列" for item in initial_items)
    assert "图像序列" in app.preflight_summary_var.get()
    assert str(app.start_button.cget("state")) == "disabled"

    load_two_frame_sequence(app, tmp_path / "images_preflight", tmp_path / "out_preflight")
    loaded_items = app.build_preflight_items()
    assert any(item["level"] == "block" and item["label"] == "ROI 组" for item in loaded_items)
    assert "ROI 组" in app.preflight_summary_var.get()

    add_basic_roi_group(app)
    assert str(app.start_button.cget("state")) == "normal"

    for var in [
        app.export_origin_txt,
        app.export_origin_opju,
        app.export_engineering_png,
        app.export_publication_figures,
        app.export_qc_summary,
        app.export_full_csv,
        app.export_corr_plot,
        app.export_overlays,
        app.export_parameters,
    ]:
        var.set(False)

    app.update_workflow_action_states()
    no_export_items = app.build_preflight_items()
    assert any(item["level"] == "block" and item["label"] == "导出选项" for item in no_export_items)
    assert str(app.start_button.cget("state")) == "disabled"
    assert "导出选项" in app.workflow_hint_var.get()


def test_preflight_reports_small_l0_as_warning_not_blocking(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)

    load_two_frame_sequence(app, tmp_path / "images_small_l0", tmp_path / "out_small_l0")
    app.roi1 = (10, 10, 20, 20)
    app.roi2 = (40, 10, 20, 20)
    app.strain_mode.set("x")
    app.sync_strain_mode_display()
    app.add_current_group()

    items = app.build_preflight_items()
    assert any(item["level"] == "warn" and item["label"] == "L0" for item in items)
    assert str(app.start_button.cget("state")) == "normal"
    assert "L0" in app.preflight_summary_var.get()


def test_current_roi_summary_updates_after_roi_changes(gui_app, tmp_path):
    _root, app = gui_app
    reset_gui_app(app)
    load_two_frame_sequence(app, tmp_path / "images_roi_summary", tmp_path / "out_roi_summary")

    assert "尚未绘制" in app.current_roi_summary_var.get()

    app.roi1 = (10, 10, 30, 30)
    app.refresh_current_roi_summary()
    assert "ROI1" in app.current_roi_summary_var.get()
    assert "ROI2 未绘制" in app.current_roi_summary_var.get()

    app.roi2 = (90, 10, 30, 30)
    app.strain_mode.set("x")
    app.sync_strain_mode_display()
    app.refresh_current_roi_summary()

    summary = app.current_roi_summary_var.get()
    assert "ROI1 30×30 px" in summary
    assert "ROI2 30×30 px" in summary
    assert "L0=80.0 px" in summary
    assert "方向=x" in summary
    assert "纹理" in summary


def test_qc_overview_highlights_worst_group_and_review_frames(gui_app):
    _root, app = gui_app
    df = pd.DataFrame(
        {
            "group": ["G_good", "G_good", "G_bad", "G_bad", "G_bad"],
            "frame_global_1based": [1, 2, 1, 2, 3],
            "engineering_strain": [0.0, 0.01, 0.0, np.nan, np.nan],
            "accepted": [True, True, True, False, False],
            "accept_mode": ["initial", "hard", "initial", "rejected", "rejected"],
            "corr_score_roi1": [1.0, 0.96, 1.0, 0.2, 0.1],
            "corr_score_roi2": [1.0, 0.95, 1.0, 0.3, 0.2],
            "filename": ["a", "b", "c", "d", "e"],
            "reason": ["initial", "ok", "initial", "fail", "fail"],
        }
    )

    summary = ezdic.build_qc_summary(df)
    app.update_qc_overview(summary)
    text = app.qc_overview_var.get()

    assert "Poor" in text
    assert "G_bad" in text
    assert "拒绝帧比例" in text
    assert "建议复核帧" in text
    assert "2, 3" in text


def test_recent_output_button_and_shortcuts_are_available(gui_app, tmp_path, monkeypatch):
    root, app = gui_app
    reset_gui_app(app)
    opened = []
    monkeypatch.setattr(ezdic.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(ezdic, "open_output_folder", lambda path: opened.append(str(path)))
    app._recent_config_path = tmp_path / "recent_paths.json"

    out_dir = tmp_path / "recent_out"
    out_dir.mkdir()
    app.show_completion_and_open_output_folder("done", out_dir)

    assert app.recent_output_dir == str(out_dir)
    assert str(app.open_recent_output_button.cget("state")) == "normal"

    app.open_recent_output_folder()
    assert opened[-1] == str(out_dir)

    for sequence in ["<Control-l>", "<Control-f>", "<Control-Return>", "<Escape>"]:
        assert root.bind(sequence), f"{sequence} should be bound for common GUI actions"


def test_windows_launcher_bat_invokes_source_entry_in_smoke_mode():
    launcher = ROOT / "start_ezDIC.bat"

    assert launcher.exists()
    text = launcher.read_text(encoding="utf-8")
    assert 'pushd "%~dp0"' in text
    assert "dic_virtual_extensometer_gui_v7_multi_roi_range.py" in text
    assert "pause" in text.lower()

    if os.name != "nt":
        pytest.skip("Windows launcher smoke test requires cmd.exe")

    env = os.environ.copy()
    env["EZDIC_LAUNCHER_SMOKE_TEST"] = "1"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "EZDIC launcher smoke test" in output
    assert "dic_virtual_extensometer_gui_v7_multi_roi_range.py" in output


def test_windows_launcher_recreates_stale_venv_before_running_entry(tmp_path):
    if os.name != "nt":
        pytest.skip("Windows launcher stale-venv test requires cmd.exe")

    project = tmp_path / "launcher_project"
    project.mkdir()
    shutil.copy2(ROOT / "start_ezDIC.bat", project / "start_ezDIC.bat")
    (project / "dic_virtual_extensometer_gui_v7_multi_roi_range.py").write_text(
        "print('entry-ran')\n",
        encoding="utf-8",
    )

    stale_python = project / ".venv" / "Scripts" / "python.exe"
    stale_python.parent.mkdir(parents=True)
    stale_python.write_text("not a usable python executable\n", encoding="utf-8")

    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(project / "start_ezDIC.bat")],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=60,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "entry-ran" in output


def test_build_release_smoke_recreates_stale_build_venv(tmp_path):
    if os.name != "nt":
        pytest.skip("PowerShell build smoke test requires Windows")

    project = tmp_path / "build_project"
    project.mkdir()
    shutil.copy2(ROOT / "build_release.ps1", project / "build_release.ps1")
    copy_release_contract_inputs(project)

    stale_python = project / ".venv-build" / "Scripts" / "python.exe"
    stale_python.parent.mkdir(parents=True)
    stale_python.write_text("not a usable python executable\n", encoding="utf-8")

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(project / "build_release.ps1"),
            "-SmokeTest",
        ],
        cwd=project,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=90,
    )

    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output
    assert "ezDIC build smoke test" in output


def test_build_release_smoke_accepts_usable_venv_after_nonzero_create(tmp_path):
    if os.name != "nt":
        pytest.skip("PowerShell build smoke test requires Windows")

    project = tmp_path / "build_project"
    project.mkdir()
    shutil.copy2(ROOT / "build_release.ps1", project / "build_release.ps1")
    copy_release_contract_inputs(project)

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    (fake_bin / "py.cmd").write_text("@echo off\r\nexit /b 1\r\n", encoding="utf-8")
    (fake_bin / "python.cmd").write_text(
        "\r\n".join(
            [
                "@echo off",
                "if \"%1\"==\"--version\" (",
                "  echo Python 3.11.99",
                "  exit /b 0",
                ")",
                f"\"{sys.executable}\" %*",
                "if \"%1\"==\"-m\" if \"%2\"==\"venv\" exit /b 1",
                "exit /b %ERRORLEVEL%",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(project / "build_release.ps1"),
            "-SmokeTest",
        ],
        cwd=project,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=90,
    )

    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output
    assert "ezDIC build smoke test" in output


def test_gui_includes_optional_origin_and_publication_exports_disabled_by_default(gui_app):
    _root, app = gui_app

    assert app.export_origin_opju.get() is False
    assert app.export_publication_figures.get() is False
    export_texts = [button.cget("text") for button in app.export_checkbuttons]
    assert "Origin OPJU" in export_texts
    assert "论文图包" in export_texts


def test_loading_new_image_folder_clears_previous_roi_state(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)

    load_two_frame_sequence(app, tmp_path / "images_a", tmp_path / "out_a")
    add_basic_roi_group(app)
    assert len(app.roi_groups) == 1

    load_two_frame_sequence(app, tmp_path / "images_b", tmp_path / "out_b")

    assert app.roi_groups == []
    assert app.roi1 is None
    assert app.roi2 is None
    assert app.next_group_idx == 1


def test_failed_image_load_preserves_previous_valid_sequence(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    errors = []
    monkeypatch.setattr(ezdic.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    load_two_frame_sequence(app, tmp_path / "images_good", tmp_path / "out_good")
    old_paths = list(app.image_paths)
    old_preview = app.current_preview_index
    log_messages = []
    monkeypatch.setattr(app, "log", lambda message: log_messages.append(message))
    corrupt_dir = tmp_path / "images_bad"
    corrupt_dir.mkdir()
    (corrupt_dir / "bad.png").write_bytes(b"not an image")

    app.image_folder.set(str(corrupt_dir))
    app.load_first_image()

    assert errors
    assert app.image_paths == old_paths
    assert app.current_preview_index == old_preview
    assert app.first_img8 is not None
    assert any("加载图像序列失败" in message for message in log_messages)
    assert not any("Traceback" in message for message in log_messages)


def test_clear_current_rois_requires_confirmation(gui_app, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    app.roi1 = (10, 10, 30, 30)
    app.roi2 = (80, 10, 30, 30)

    answers = iter([False, True])
    prompts = []
    monkeypatch.setattr(
        ezdic.messagebox,
        "askyesno",
        lambda title, message: prompts.append((title, message)) or next(answers),
    )

    app.clear_current_rois()
    assert app.roi1 == (10, 10, 30, 30)
    assert app.roi2 == (80, 10, 30, 30)

    app.clear_current_rois()
    assert app.roi1 is None
    assert app.roi2 is None
    assert prompts
    assert prompts[0][0] == "清除当前 ROI"


def test_clear_current_rois_is_ignored_while_processing(gui_app, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    app.roi1 = (10, 10, 30, 30)
    app.roi2 = (80, 10, 30, 30)
    app.is_processing = True

    prompts = []
    monkeypatch.setattr(
        ezdic.messagebox,
        "askyesno",
        lambda title, message: prompts.append((title, message)) or True,
    )

    try:
        app.clear_current_rois()
        assert app.roi1 == (10, 10, 30, 30)
        assert app.roi2 == (80, 10, 30, 30)
        assert prompts == []
        assert "正在处理" in app.status_var.get()
    finally:
        app.is_processing = False


def test_validate_rejects_output_path_that_is_existing_file(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    load_two_frame_sequence(app, tmp_path / "images", tmp_path / "out")
    add_basic_roi_group(app)
    output_file = tmp_path / "not_a_directory.txt"
    output_file.write_text("occupied", encoding="utf-8")
    app.output_folder.set(str(output_file))

    with pytest.raises(RuntimeError, match="不是文件夹"):
        app.validate_before_processing()


def test_validate_requires_at_least_one_export_option(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    load_two_frame_sequence(app, tmp_path / "images", tmp_path / "out")
    add_basic_roi_group(app)
    app.export_origin_txt.set(False)
    app.export_engineering_png.set(False)
    app.export_qc_summary.set(False)
    app.export_full_csv.set(False)
    app.export_corr_plot.set(False)
    app.export_overlays.set(False)
    app.export_parameters.set(False)
    app.export_publication_figures.set(False)
    app.export_origin_opju.set(False)

    with pytest.raises(RuntimeError, match="至少选择一种导出内容"):
        app.validate_before_processing()
    app.export_publication_figures.set(True)
    app.validate_before_processing()


def test_validate_reports_invalid_numeric_inputs_in_chinese(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    load_two_frame_sequence(app, tmp_path / "images", tmp_path / "out")
    add_basic_roi_group(app)

    app.start_frame_1based.set("abc")
    with pytest.raises(RuntimeError, match="起始帧.*整数"):
        app.validate_before_processing()

    app.start_frame_1based.set(1)
    app.search_radius.set("abc")
    with pytest.raises(RuntimeError, match="搜索半径.*整数"):
        app.validate_before_processing()


def test_processing_settings_are_snapshotted_before_worker_thread(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    load_two_frame_sequence(app, tmp_path / "images", tmp_path / "out")
    add_basic_roi_group(app)

    settings = app.build_processing_settings()
    original_output = settings["output_dir"]
    original_paths = list(settings["image_paths"])
    original_groups = list(settings["roi_groups"])
    assert settings["export_origin_opju"] is False
    assert settings["export_publication_figures"] is False

    app.export_origin_opju.set(True)
    assert app.build_processing_settings()["export_origin_opju"] is True
    app.export_publication_figures.set(True)
    assert app.build_processing_settings()["export_publication_figures"] is True

    app.output_folder.set(str(tmp_path / "changed_out"))
    app.image_paths.clear()
    app.roi_groups.clear()

    assert original_output == tmp_path / "out"
    assert settings["output_dir"] == original_output
    assert settings["image_paths"] == original_paths
    assert settings["roi_groups"] == original_groups
    assert settings["roi_groups"] is not app.roi_groups


def test_background_processing_finishes_even_if_ui_queue_is_not_drained(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(ezdic.messagebox, "showerror", lambda *args, **kwargs: None)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    speckle = ezdic.generate_synthetic_speckle(100, 160, seed=71)
    for idx in range(3):
        arr = ezdic.warp_image_translation(speckle, float(idx), 0.0).astype(np.uint8)
        ok, data = cv2.imencode(".png", arr)
        assert ok
        data.tofile(str(image_dir / f"frame_{idx:03d}.png"))

    app.image_folder.set(str(image_dir))
    app.output_folder.set(str(tmp_path / "out"))
    app.load_first_image()
    app.roi1 = (20, 30, 30, 30)
    app.roi2 = (90, 30, 30, 30)
    app.strain_mode.set("x")
    app.sync_strain_mode_display()
    app.add_current_group()

    app.start_processing()
    deadline = time.time() + 5
    while app.is_processing and time.time() < deadline:
        time.sleep(0.05)

    assert app.is_processing is False
    assert (tmp_path / "out" / "core" / "strain_G01.txt").exists()


def test_publication_figure_export_writes_high_res_and_vector_outputs(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    monkeypatch.setattr(ezdic.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(ezdic, "open_output_folder", lambda path: None, raising=False)
    monkeypatch.setattr(app, "post_to_ui", lambda callback: callback())

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    speckle = ezdic.generate_synthetic_speckle(100, 160, seed=72)
    for idx in range(3):
        arr = ezdic.warp_image_translation(speckle, float(idx), 0.0).astype(np.uint8)
        ok, data = cv2.imencode(".png", arr)
        assert ok
        data.tofile(str(image_dir / f"frame_{idx:03d}.png"))

    output_dir = tmp_path / "out"
    app.image_folder.set(str(image_dir))
    app.output_folder.set(str(output_dir))
    app.load_first_image()
    app.roi1 = (20, 30, 30, 30)
    app.roi2 = (90, 30, 30, 30)
    app.strain_mode.set("x")
    app.sync_strain_mode_display()
    app.add_current_group()
    app.export_engineering_png.set(False)
    app.export_publication_figures.set(True)

    app.process_images(app.build_processing_settings())

    pub_dir = output_dir / "optional" / "publication_figures"
    for suffix in [".png", ".tiff", ".pdf", ".svg", ".eps"]:
        path = pub_dir / f"engineering_strain_G01{suffix}"
        assert path.exists()
        assert path.stat().st_size > 0
    assert (pub_dir / "engineering_strain_all_groups.pdf").exists()


def test_origin_opju_failure_does_not_cancel_existing_exports(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    speckle = ezdic.generate_synthetic_speckle(100, 160, seed=73)
    for idx in range(3):
        arr = ezdic.warp_image_translation(speckle, float(idx), 0.0).astype(np.uint8)
        ok, data = cv2.imencode(".png", arr)
        assert ok
        data.tofile(str(image_dir / f"frame_{idx:03d}.png"))

    app.image_folder.set(str(image_dir))
    app.output_folder.set(str(tmp_path / "out"))
    app.load_first_image()
    app.roi1 = (20, 30, 30, 30)
    app.roi2 = (90, 30, 30, 30)
    app.strain_mode.set("x")
    app.sync_strain_mode_display()
    app.add_current_group()
    app.export_origin_opju.set(True)

    failures = []

    def fail_origin_export(*args, **kwargs):
        failures.append((args, kwargs))
        raise RuntimeError("OriginPro unavailable")

    monkeypatch.setattr(ezdic, "write_origin_opju_project", fail_origin_export)

    app.process_images(app.build_processing_settings())

    assert failures
    assert (tmp_path / "out" / "core" / "strain_G01.txt").exists()
    assert not (tmp_path / "out" / "core" / "ezDIC_results.opju").exists()


def test_processing_completion_mentions_mean_export_and_opens_output_root(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(app, "post_to_ui", lambda callback: callback())
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)

    messages = []
    opened_paths = []
    monkeypatch.setattr(
        ezdic.messagebox,
        "showinfo",
        lambda title, message: messages.append((title, message)),
    )
    monkeypatch.setattr(
        ezdic,
        "open_output_folder",
        lambda path: opened_paths.append(Path(path)),
        raising=False,
    )

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    speckle = ezdic.generate_synthetic_speckle(100, 160, seed=74)
    for idx in range(3):
        arr = ezdic.warp_image_translation(speckle, float(idx), 0.0).astype(np.uint8)
        ok, data = cv2.imencode(".png", arr)
        assert ok
        data.tofile(str(image_dir / f"frame_{idx:03d}.png"))

    output_dir = tmp_path / "out"
    app.image_folder.set(str(image_dir))
    app.output_folder.set(str(output_dir))
    app.load_first_image()
    for name, y in [("G01", 30), ("G02", 64)]:
        app.group_name_var.set(name)
        app.roi1 = (20, y, 30, 30)
        app.roi2 = (90, y, 30, 30)
        app.strain_mode.set("x")
        app.sync_strain_mode_display()
        app.add_current_group()

    app.process_images(app.build_processing_settings())

    assert (output_dir / "core" / "strain_mean_groups.txt").exists()
    assert messages
    assert messages[-1][0] == "完成"
    assert "平均应变文件: " in messages[-1][1]
    assert "core\\strain_mean_groups.txt" in messages[-1][1]
    assert "strain_all_groups.txt" in messages[-1][1]
    assert opened_paths == [output_dir]


def test_completed_analysis_populates_qc_overview_and_curve_preview(gui_app, tmp_path, monkeypatch):
    root, app = gui_app
    reset_gui_app(app)
    idle_qc = "分析完成后显示 QC 总览。"
    monkeypatch.setattr(app, "post_to_ui", lambda callback: callback())
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    monkeypatch.setattr(ezdic.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(ezdic, "open_output_folder", lambda path: None, raising=False)

    app.qc_overview_var.set(idle_qc)
    app.clear_viewer(keep_placeholder=True)
    app.viewer_frame.grid_remove()

    image_dir = tmp_path / "images_qc_preview"
    image_dir.mkdir()
    speckle = ezdic.generate_synthetic_speckle(100, 160, seed=75)
    for idx in range(3):
        arr = ezdic.warp_image_translation(speckle, float(idx), 0.0).astype(np.uint8)
        ok, data = cv2.imencode(".png", arr)
        assert ok
        data.tofile(str(image_dir / f"frame_{idx:03d}.png"))

    output_dir = tmp_path / "out_qc_preview"
    app.image_folder.set(str(image_dir))
    app.output_folder.set(str(output_dir))
    app.load_first_image()
    app.roi1 = (20, 30, 30, 30)
    app.roi2 = (90, 30, 30, 30)
    app.strain_mode.set("x")
    app.sync_strain_mode_display()
    app.add_current_group()

    try:
        app.process_images(app.build_processing_settings())
        root.update_idletasks()

        qc_text = app.qc_overview_var.get()
        assert qc_text != idle_qc
        assert "QC 总览" in qc_text
        assert "处理完成" in app.status_var.get()
        assert float(app.progress.cget("value")) == 100.0
        log_text = app.log_text.get("1.0", "end")
        assert "第 " in log_text
        assert "帧" in log_text
        assert "组" in log_text
        assert app.results_df is not None
        assert not app.results_df.empty
        assert app.viewer_figure is not None
        assert str(app.viewer_export_btn.cget("state")) == "normal"
        assert app.viewer_frame.grid_info()
    finally:
        app.clear_viewer(keep_placeholder=False)
        app.viewer_frame.grid_remove()
        app.qc_overview_var.set(idle_qc)
        reset_gui_app(app)


def test_tracking_status_line_uses_chinese_task_wording():
    line = ezdic.format_tracking_status_line(2, 10, "G01", "hard", "0.001000", 0.95, 0.94)
    assert "第 2/10 帧" in line
    assert "组 G01" in line
    assert "硬接受" in line
    assert "应变=0.001000" in line
    assert "Frame " not in line


def test_completion_folder_open_failure_is_logged(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)

    messages = []
    log_messages = []
    monkeypatch.setattr(
        ezdic.messagebox,
        "showinfo",
        lambda title, message: messages.append((title, message)),
    )
    monkeypatch.setattr(
        ezdic,
        "open_output_folder",
        lambda path: (_ for _ in ()).throw(RuntimeError("explorer failed")),
        raising=False,
    )
    monkeypatch.setattr(app, "log", lambda message: log_messages.append(message))

    app.show_completion_and_open_output_folder("处理完成。", tmp_path / "out")

    assert messages == [("完成", "处理完成。")]
    assert len(log_messages) == 1
    assert "无法自动打开结果目录" in log_messages[0]
    assert "explorer failed" in log_messages[0]


def test_release_support_files_exist_and_include_usage_limits():
    notice = ROOT / "NOTICE_Attribution_and_Usage.txt"
    readme = ROOT / "README_使用说明.txt"
    github_readme = ROOT / "README.md"
    citation = ROOT / "CITATION.cff"
    version = ROOT / "VERSION.txt"
    zenodo = ROOT / ".zenodo.json"
    release_notes = ROOT / "RELEASE_NOTES_v0.1.4.md"

    assert notice.exists()
    assert readme.exists()
    assert github_readme.exists()
    assert citation.exists()
    assert version.exists()
    assert zenodo.exists()
    assert release_notes.exists()

    notice_text = notice.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")
    github_readme_text = github_readme.read_text(encoding="utf-8")
    citation_text = citation.read_text(encoding="utf-8")
    version_text = version.read_text(encoding="utf-8")
    zenodo_text = zenodo.read_text(encoding="utf-8")
    release_notes_text = release_notes.read_text(encoding="utf-8")

    assert "Developer:\nDr. Delun Gong" in notice_text
    assert "MIT" in notice_text
    assert "redistribution are permitted" in notice_text
    assert "Windows 10/11 x64" in readme_text
    assert "Do not copy ezDIC.exe alone" in readme_text
    assert "Dr. Delun Gong" in readme_text
    assert DOI in readme_text
    assert DOI_URL in readme_text
    assert "virtual extensometer" in github_readme_text.lower()
    assert "Origin-compatible TXT" in github_readme_text
    assert "OPJU" in github_readme_text
    assert "OriginPro 2021+" in readme_text
    assert "publication_figures" in github_readme_text
    assert "论文图包" in readme_text
    assert "optional/publication_figures" in readme_text
    assert "PNG/TIFF/PDF/SVG/EPS" in readme_text
    assert "full-field DIC" in github_readme_text
    assert "Dr. Delun Gong" in github_readme_text
    assert DOI in github_readme_text
    assert DOI_URL in github_readme_text
    assert DOI in citation_text and "Historical ezDIC archive" in citation_text
    assert "license: MIT" in citation_text
    assert "ezDIC v0.1.4" in version_text
    assert DOI in version_text
    assert '"upload_type": "software"' in zenodo_text
    assert '"access_right": "open"' in zenodo_text
    assert '"title": "StrainTrace: Virtual extensometry and in-plane digital image correlation"' in zenodo_text
    assert "Poisson ratio export and GUI workflow update" in release_notes_text
    assert "How to cite" in release_notes_text
    assert DOI_URL in release_notes_text


def test_pyinstaller_build_files_define_green_folder_release():
    requirements = ROOT / "requirements.txt"
    build_requirements = ROOT / "requirements-build.txt"
    spec = ROOT / "ezDIC.spec"
    build_script = ROOT / "build_release.ps1"

    assert requirements.exists()
    assert build_requirements.exists()
    assert spec.exists()
    assert build_script.exists()

    req_text = requirements.read_text(encoding="utf-8")
    build_req_text = build_requirements.read_text(encoding="utf-8")
    spec_text = spec.read_text(encoding="utf-8")
    script_text = build_script.read_text(encoding="utf-8")

    for package in ["opencv-python", "numpy", "pandas", "matplotlib", "pillow"]:
        assert package in req_text
    assert "originpro" not in req_text
    assert "originpro" in (ROOT / "requirements-origin.txt").read_text(encoding="utf-8")

    assert "pyinstaller" in build_req_text.lower()
    assert "name='ezDIC'" in spec_text
    assert "console=False" in spec_text
    assert "release" in script_text
    assert "ezDIC_Windows_x64" in script_text
    assert "Compress-Archive" in script_text


def test_gui_exposes_dual_mode_fullfield_controls_and_field_viewer(gui_app):
    root, app = gui_app
    assert app.analysis_mode.get() == ezdic.ANALYSIS_MODE_EXTENSOMETER
    assert app.start_button.cget("text") == "开始分析并导出结果"
    assert app.roi1_button is not None
    assert app.dic_subset_size_entry is not None
    assert app.dic_step_entry is not None
    assert app.dic_solver_box is not None
    values = app.dic_solver_box.cget("values")
    if isinstance(values, str):
        values = tuple(values.split())
    else:
        values = tuple(values)
    assert ezdic.DIC_SOLVER_ICGN in values
    assert ezdic.DIC_SOLVER_ICLM in values
    assert app.mode_extensometer_radio is not None
    assert app.mode_fullfield_radio is not None

    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    root.update_idletasks()
    assert str(app.analysis_mode.get()) == ezdic.ANALYSIS_MODE_FULLFIELD

    reference = ezdic.generate_synthetic_speckle(64, 64, seed=2)
    deformed = ezdic.warp_image_translation(reference, 0.7, -0.4)
    field = ezdic.run_2d_dic(
        reference,
        deformed,
        (12, 12, 40, 40),
        subset_size=15,
        step=8,
        solver=ezdic.DIC_SOLVER_ICGN,
    )
    try:
        app.show_field_viewer(field, component="u")
        root.update_idletasks()
        assert app.viewer_figure is not None
        assert len(app.viewer_figure.axes) >= 2  # field + colorbar
        app.toggle_dark_mode()
        root.update_idletasks()
        app.toggle_dark_mode()
        root.update_idletasks()
        assert app.viewer_figure is not None
    finally:
        if app.dark_mode.get():
            app.toggle_dark_mode()
        app.clear_viewer(keep_placeholder=False)
        app.viewer_frame.grid_remove()
        app.analysis_mode.set(ezdic.ANALYSIS_MODE_EXTENSOMETER)
        app.set_analysis_mode()


def test_mode_switch_clears_stale_viewer_and_locks_during_processing(gui_app):
    root, app = gui_app
    reset_gui_app(app)
    reference = ezdic.generate_synthetic_speckle(64, 64, seed=3)
    deformed = ezdic.warp_image_translation(reference, 0.6, -0.3)
    field = ezdic.run_2d_dic(reference, deformed, (12, 12, 40, 40), subset_size=15, step=8)
    app.current_fullres_img8 = np.zeros_like(reference, dtype=np.uint8)
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    app.show_field_viewer(
        field,
        component="u",
        image=deformed,
        frame_1based=2,
        filename="frame_002.png",
        reference_frame_1based=1,
        reference_filename="frame_001.png",
    )
    root.update_idletasks()
    assert app.dic_last_field is not None
    assert app._viewer_kind == "fullfield"

    app.analysis_mode.set(ezdic.ANALYSIS_MODE_EXTENSOMETER)
    app.set_analysis_mode()
    root.update_idletasks()
    assert app.dic_last_field is None
    assert app.results_df is None
    assert app._viewer_kind == "extensometer"
    assert app._committed_analysis_mode == ezdic.ANALYSIS_MODE_EXTENSOMETER

    app.is_processing = True
    app.update_workflow_action_states()
    assert str(app.mode_fullfield_radio.cget("state")) == "disabled"
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    assert str(app.analysis_mode.get()) == ezdic.ANALYSIS_MODE_EXTENSOMETER
    app.is_processing = False
    app.update_workflow_action_states()
    assert str(app.mode_fullfield_radio.cget("state")) == "normal"
    reset_gui_app(app)


def test_fullfield_pyramid_entry_is_not_covered_by_draw_roi_row(gui_app):
    _root, app = gui_app
    reset_gui_app(app)
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    levels_row = int(app.dic_pyramid_levels_entry.grid_info()["row"])
    draw_row = int(app.draw_field_roi_button.master.grid_info()["row"])
    assert levels_row != draw_row
    assert draw_row > int(app.dic_pyramid_scale_entry.grid_info()["row"])
    assert app.dic_search_radius_entry.winfo_manager() == "grid"
    assert app.dic_zncc_min_entry.winfo_manager() == "grid"
    app.dic_search_radius.set(32)
    app.dic_zncc_min.set(0.82)
    assert int(app.dic_search_radius.get()) == 32
    assert float(app.dic_zncc_min.get()) == pytest.approx(0.82)
    reset_gui_app(app)


def test_select_image_folder_does_not_overwrite_existing_output(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    chosen = tmp_path / "images_keep_output"
    chosen.mkdir()
    existing_output = tmp_path / "project_output"
    app.output_folder.set(str(existing_output))
    monkeypatch.setattr(ezdic.filedialog, "askdirectory", lambda **_kwargs: str(chosen))
    app.select_image_folder()
    assert app.image_folder.get() == str(chosen)
    assert app.output_folder.get() == str(existing_output)
    app.output_folder.set("")
    app.select_image_folder()
    assert Path(app.output_folder.get()).name == "ezDIC_extensometer_output"
    reset_gui_app(app)


def test_fullfield_overlay_does_not_accept_roi_drawing(gui_app):
    _root, app = gui_app
    reset_gui_app(app)
    reference = ezdic.generate_synthetic_speckle(64, 64, seed=5)
    deformed = ezdic.warp_image_translation(reference, 0.5, -0.2)
    field = ezdic.run_2d_dic(reference, deformed, (12, 12, 40, 40), subset_size=15, step=8)
    app.current_fullres_img8 = np.zeros((64, 64), dtype=np.uint8)
    app.first_img8 = app.current_fullres_img8
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    app.show_field_viewer(field, component="u", image=deformed, frame_1based=2, filename="frame_002.png")
    class _Event:
        x = 10
        y = 10
    assert app._canvas_shows_field_overlay is True
    app.on_mouse_down(_Event())
    assert app.drag_start is None
    assert "叠加图" in app.status_var.get()
    reset_gui_app(app)


def test_clear_viewer_blocks_roi_until_preview_or_1to1_restores_canvas(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "showwarning", lambda *_args, **_kwargs: None)
    folder = tmp_path / "images_overlay_roi"
    folder.mkdir()
    speckle = ezdic.generate_synthetic_speckle(100, 140, seed=11)
    for index in range(2):
        frame = speckle if index == 0 else ezdic.warp_image_translation(speckle, 0.8, -0.3)
        ok, data = cv2.imencode(".png", np.clip(frame, 0, 255).astype(np.uint8))
        assert ok
        data.tofile(str(folder / f"frame_{index + 1:03d}.png"))
    app.image_folder.set(str(folder))
    app.output_folder.set(str(tmp_path / "out_overlay_roi"))
    app.load_first_image()
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    reference = np.asarray(app.first_img8, dtype=np.float32)
    deformed = ezdic.warp_image_translation(reference, 0.8, -0.3)
    field = ezdic.run_2d_dic(reference, deformed, (12, 12, 80, 50), subset_size=15, step=8)
    app.show_field_viewer(
        field,
        component="u",
        image=deformed,
        frame_1based=2,
        filename="frame_002.png",
        reference_frame_1based=1,
        reference_filename="frame_001.png",
    )
    overlay = np.asarray(app.display_img).copy()
    assert app._canvas_shows_field_overlay is True

    class _Event:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    app.dic_last_field = None
    app._viewer_kind = "extensometer"
    app.auto_fit_enabled = True
    app.display_img = overlay
    app._canvas_shows_field_overlay = True
    app.field_roi = None
    app.on_mouse_down(_Event(10, 10))
    assert app.drag_start is None
    app.drag_start = (10, 10)
    app.on_mouse_up(_Event(80, 80))
    assert app.field_roi is None
    assert app.drag_start is None

    app.show_field_viewer(
        field,
        component="u",
        image=deformed,
        frame_1based=2,
        filename="frame_002.png",
        reference_frame_1based=1,
        reference_filename="frame_001.png",
    )
    leftover = np.asarray(app.display_img).copy()
    app.clear_viewer(keep_placeholder=True)
    assert app._canvas_shows_field_overlay is False
    assert app.current_fullres_img8 is not None
    restored = np.asarray(app.display_img)
    assert restored.ndim == 3
    assert restored.shape[2] == 3
    preview_rgb, _ = ezdic.get_display_image(app.current_fullres_img8, max_w=restored.shape[1], max_h=restored.shape[0])
    np.testing.assert_array_equal(restored, preview_rgb)

    app.field_roi = None
    app.on_mouse_down(_Event(12, 12))
    assert app.drag_start is not None
    app.on_mouse_up(_Event(70, 70))
    assert app.field_roi is not None
    preview_roi = app.field_roi
    del leftover

    app.show_field_viewer(
        field,
        component="u",
        image=deformed,
        frame_1based=2,
        filename="frame_002.png",
        reference_frame_1based=1,
        reference_filename="frame_001.png",
    )
    app.field_roi = None
    app.load_preview_frame(0)
    assert app._canvas_shows_field_overlay is False
    app.on_mouse_down(_Event(12, 12))
    assert app.drag_start is not None
    app.on_mouse_up(_Event(70, 70))
    assert app.field_roi is not None
    assert app.field_roi_reference_frame_1based == 1

    app.show_field_viewer(
        field,
        component="u",
        image=deformed,
        frame_1based=2,
        filename="frame_002.png",
        reference_frame_1based=1,
        reference_filename="frame_001.png",
    )
    app.field_roi = None
    app.show_image_1to1()
    assert app._canvas_shows_field_overlay is False
    app.on_mouse_down(_Event(12, 12))
    assert app.drag_start is not None
    app.on_mouse_up(_Event(70, 70))
    assert app.field_roi is not None
    assert preview_roi[2] >= 15 and preview_roi[3] >= 15
    reset_gui_app(app)


def test_fullfield_preflight_requires_field_roi_not_extensometer_groups(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    load_two_frame_sequence(app, tmp_path / "images_ff", tmp_path / "out_ff")
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()

    items = app.build_preflight_items()
    assert any(item["level"] == "block" and item["label"] == "全场 ROI" for item in items)
    assert str(app.start_button.cget("state")) == "disabled"

    app.roi1 = (12, 12, 50, 50)
    app.field_roi = app.roi1
    app.field_roi_reference_frame_1based = 1
    app.update_workflow_action_states()
    items = app.build_preflight_items()
    assert not any(item["level"] == "block" and item["label"] == "全场 ROI" for item in items)
    assert any(item["label"] == "DIC 参数" for item in items)
    assert str(app.start_button.cget("state")) == "normal"

    app.analysis_mode.set(ezdic.ANALYSIS_MODE_EXTENSOMETER)
    app.set_analysis_mode()


def test_fullfield_roi_is_separate_from_1d_roi_and_tracks_reference_frame(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    load_two_frame_sequence(app, tmp_path / "images_roi_separation", tmp_path / "out_roi_separation")

    app.roi1 = (10, 10, 30, 30)
    app.roi2 = (80, 10, 30, 30)
    app.roi1_reference_frame_1based = 1
    app.roi2_reference_frame_1based = 1
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()

    items = app.build_preflight_items()
    assert any(item["level"] == "block" and item["label"] == "全场 ROI" for item in items)
    assert app.build_processing_settings()["field_roi"] is None

    app.field_roi = (20, 20, 50, 50)
    app.field_roi_reference_frame_1based = 1
    items = app.build_preflight_items()
    assert not any(item["level"] == "block" and item["label"] == "全场 ROI" for item in items)
    assert app.build_processing_settings()["field_roi"] == (20, 20, 50, 50)

    app.start_frame_1based.set(2)
    items = app.build_preflight_items()
    assert any(item["level"] == "block" and item["label"] == "参考帧" for item in items)


def test_fullfield_hides_1d_controls_and_does_not_require_1d_exports(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    load_two_frame_sequence(app, tmp_path / "images_mode_visibility", tmp_path / "out_mode_visibility")
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()

    assert app.measure_core_frame.grid_info() == {}
    assert app.roi_group_frame.grid_info() == {}
    assert app.export_frame.grid_info() == {}
    assert app.fullfield_export_info_frame.grid_info()

    app.field_roi = (12, 12, 70, 70)
    app.field_roi_reference_frame_1based = 1
    for variable in app._export_option_vars():
        variable.set(False)
    items = app.build_preflight_items()
    assert not any(item["label"] == "导出选项" and item["level"] == "block" for item in items)


def test_analysis_range_rejects_out_of_range_without_clamping_and_load_handles_bad_intvar(
    gui_app, tmp_path, monkeypatch
):
    _root, app = gui_app
    reset_gui_app(app)
    errors = []
    monkeypatch.setattr(ezdic.messagebox, "showerror", lambda title, message: errors.append((title, message)))
    load_two_frame_sequence(app, tmp_path / "images_range_validation", tmp_path / "out_range_validation")

    app.start_frame_1based.set(0)
    with pytest.raises(RuntimeError, match="1 到 2"):
        app.get_analysis_indices()
    assert int(app.start_frame_1based.get()) == 0

    app.start_frame_1based.set(1)
    app.end_frame_1based.set(3)
    with pytest.raises(RuntimeError, match="1 到 2"):
        app.get_analysis_indices()
    assert int(app.end_frame_1based.get()) == 3

    app.start_frame_1based.set("bad")
    app.image_folder.set(str(tmp_path / "images_range_validation"))
    app.load_first_image()
    assert errors
    assert "整数" in errors[-1][1]


def test_same_folder_mutation_clears_all_sequence_dependent_and_viewer_state(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    folder = tmp_path / "images_mutation"
    load_two_frame_sequence(app, folder, tmp_path / "out_mutation")
    add_basic_roi_group(app)
    app.results_df = pd.DataFrame({"x": [1]})
    app.results_groups = [{"name": "G01"}]
    app.last_qc_summary = {"overall": {"qc_level": "Good"}, "groups": {}}
    app.dic_last_field = {"x": np.array([1.0])}
    app._viewer_kind = "fullfield"
    app.viewer_frame.grid()

    # Keep the path and frame count unchanged but mutate file metadata/content.
    write_test_image(folder / "frame_001.png", value=101)
    app.load_first_image()

    assert app.roi_groups == []
    assert app.roi1 is None and app.roi2 is None and app.field_roi is None
    assert app.results_df is None
    assert app.results_groups is None
    assert app.last_qc_summary is None
    assert app.dic_last_field is None
    assert app.viewer_figure is None


def test_same_folder_unchanged_refresh_preserves_roi_setup(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    folder = tmp_path / "images_unchanged"
    load_two_frame_sequence(app, folder, tmp_path / "out_unchanged")
    add_basic_roi_group(app)
    before = [dict(group) for group in app.roi_groups]
    app.load_first_image()
    assert app.roi_groups == before


def test_fullfield_all_nan_processing_fails_without_export_or_completion(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "showinfo", lambda *args, **kwargs: pytest.fail("must not show completion"))
    folder = tmp_path / "images_fullfield_all_nan"
    folder.mkdir()
    write_test_image(folder / "frame_001.png", value=50, shape=(100, 140))
    write_test_image(folder / "frame_002.png", value=80, shape=(100, 140))
    app.image_folder.set(str(folder))
    app.output_folder.set(str(tmp_path / "out_fullfield_all_nan"))
    app.load_first_image()
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    app.field_roi = (10, 10, 80, 70)
    app.field_roi_reference_frame_1based = 1
    settings = app.build_processing_settings()
    stale_dic = Path(settings["output_dir"]) / "dic"
    stale_dic.mkdir(parents=True, exist_ok=True)
    stale_file = stale_dic / "frame_0002.txt"
    stale_file.write_text("old run\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="有限应变|有效应变"):
        app.process_fullfield(settings)
    current_dic = tmp_path / "out_fullfield_all_nan" / "dic"
    assert not list(current_dic.glob("*.txt"))
    assert not list(current_dic.glob("*.csv"))
    assert not list(current_dic.glob("*.png"))
    archived = list((stale_dic / "_previous_runs").rglob("frame_0002.txt"))
    assert archived and archived[0].read_text(encoding="utf-8") == "old run\n"
    assert not stale_file.exists()


def test_fullfield_viewer_uses_final_image_context_and_component_switch_updates_overlay(gui_app):
    root, app = gui_app
    reset_gui_app(app)
    reference = ezdic.generate_synthetic_speckle(64, 64, seed=17)
    deformed = ezdic.warp_image_translation(reference, 0.7, -0.4)
    field = ezdic.run_2d_dic(reference, deformed, (12, 12, 40, 40), subset_size=15, step=8)
    app.current_fullres_img8 = np.zeros_like(reference, dtype=np.float32)
    app.show_field_viewer(
        field,
        component="u",
        image=deformed,
        frame_1based=2,
        filename="frame_002.png",
        reference_frame_1based=1,
        reference_filename="frame_001.png",
    )
    root.update_idletasks()
    assert np.array_equal(np.asarray(app.dic_last_image), np.asarray(deformed))
    assert app.dic_last_frame_1based == 2
    assert "frame_002.png" in app.field_viewer_context_var.get()
    first_overlay = np.asarray(app.display_img).copy()

    app.dic_field_component.set("Exx")
    app._on_field_component_change()
    root.update_idletasks()
    assert not np.array_equal(first_overlay, np.asarray(app.display_img))


def test_fullfield_previous_run_archive_moves_only_ezdic_generated_files(tmp_path):
    dic_dir = tmp_path / "dic"
    dic_dir.mkdir()
    generated = [
        "frame_0002.txt",
        "frame_0002.csv",
        "frame_0002_u.png",
        "frame_0002_parameters.txt",
        "frame_0003_Exx.png",
    ]
    user_named_like_output = ["frame_0002_u.txt", "frame_0002_parameters.png", "notes.txt"]
    for name in generated:
        (dic_dir / name).write_text("generated\n", encoding="utf-8")
    for name in user_named_like_output:
        (dic_dir / name).write_text("keep me\n", encoding="utf-8")

    archive = ezdic.archive_previous_fullfield_outputs(dic_dir)
    assert archive.parent == dic_dir / "_previous_runs"
    assert archive.exists()
    assert all((dic_dir / name).exists() for name in user_named_like_output)
    for name in generated:
        assert not (dic_dir / name).exists()
        assert (archive / name).exists()


def test_fullfield_previous_run_archive_rolls_back_on_core_move_failure(tmp_path, monkeypatch):
    dic_dir = tmp_path / "dic"
    dic_dir.mkdir()
    generated = [dic_dir / "frame_0002.txt", dic_dir / "frame_0002.csv"]
    original_bytes = {}
    for path in generated:
        original_bytes[path] = b"previous run\x00\xff"
        path.write_bytes(original_bytes[path])

    real_move = ezdic._core._move_exact
    archive_moves = 0
    rollback_events = []

    def fail_second_archive_move(source, destination, *, root=None):
        nonlocal archive_moves
        source_path = Path(source)
        destination_path = Path(destination)
        is_archive_move = (
            source_path.parent == dic_dir
            and destination_path.parent.parent == dic_dir / "_previous_runs"
        )
        if is_archive_move:
            archive_moves += 1
            if archive_moves == 2:
                raise OSError("forced archive move #2 failure")
        elif source_path.parent.parent == dic_dir / "_previous_runs" and destination_path.parent == dic_dir:
            rollback_events.append((source_path, destination_path))
        return real_move(source, destination, root=root)

    monkeypatch.setattr(ezdic._core, "_move_exact", fail_second_archive_move)
    with pytest.raises(RuntimeError, match="归档"):
        ezdic.archive_previous_fullfield_outputs(dic_dir)

    assert archive_moves == 2
    assert rollback_events
    assert all(path.read_bytes() == original_bytes[path] for path in generated)
    assert not list((dic_dir / "_previous_runs").rglob("frame_0002.txt"))
    assert not list((dic_dir / "_previous_runs").rglob("frame_0002.csv"))


def test_fullfield_fatal_frame_is_transactionally_archived_without_root_partial_files(
    gui_app, tmp_path, monkeypatch
):
    _root, app = gui_app
    reset_gui_app(app)
    folder = tmp_path / "images_fullfield_transaction"
    folder.mkdir()
    for idx in range(3):
        write_test_image(folder / f"frame_{idx + 1:03d}.png", value=50 + idx, shape=(100, 140))
    output_dir = tmp_path / "out_fullfield_transaction"
    app.image_folder.set(str(folder))
    app.output_folder.set(str(output_dir))
    app.load_first_image()
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    app.field_roi = (10, 10, 80, 70)
    app.field_roi_reference_frame_1based = 1
    settings = app.build_processing_settings()

    dic_dir = output_dir / "dic"
    dic_dir.mkdir(parents=True, exist_ok=True)
    old_root = dic_dir / "frame_0002.txt"
    old_root.write_text("previous run\n", encoding="utf-8")
    calls = []

    fake_field = {
        "valid": np.ones(9, dtype=bool),
        "u": np.zeros(9),
        "v": np.zeros(9),
        "Exx": np.zeros(9),
        "Eyy": np.zeros(9),
        "Exy": np.zeros(9),
    }

    def fake_run(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("fatal solver failure")
        return dict(fake_field)

    def fake_export(field, output_path, *, stem="dic_field", preset_name="publication"):
        del field, preset_name
        path = Path(output_path) / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("partial current run\n", encoding="utf-8")
        return {"txt": path}

    monkeypatch.setattr(ezdic, "run_2d_dic", fake_run)
    monkeypatch.setattr(ezdic, "export_dic_field_outputs", fake_export)

    with pytest.raises(RuntimeError, match="fatal solver failure"):
        app.process_fullfield(settings)

    previous = list((dic_dir / "_previous_runs").rglob("frame_0002.txt"))
    failed = list((dic_dir / "_failed_runs").rglob("frame_0002.txt"))
    assert previous and previous[0].read_text(encoding="utf-8") == "previous run\n"
    assert failed and failed[0].read_text(encoding="utf-8") == "partial current run\n"
    assert not old_root.exists()
    assert not list(dic_dir.glob("frame_*.txt"))


def test_fullfield_partial_valid_run_commits_only_valid_staged_frames(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    folder = tmp_path / "images_fullfield_partial_valid"
    folder.mkdir()
    for idx in range(3):
        write_test_image(folder / f"frame_{idx + 1:03d}.png", value=50 + idx, shape=(100, 140))
    output_dir = tmp_path / "out_fullfield_partial_valid"
    app.image_folder.set(str(folder))
    app.output_folder.set(str(output_dir))
    app.load_first_image()
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    app.field_roi = (10, 10, 80, 70)
    app.field_roi_reference_frame_1based = 1
    settings = app.build_processing_settings()

    calls = []
    fake_valid = {
        "valid": np.ones(9, dtype=bool),
        "u": np.zeros(9),
        "v": np.zeros(9),
        "Exx": np.zeros(9),
        "Eyy": np.zeros(9),
        "Exy": np.zeros(9),
    }
    fake_invalid = {
        "valid": np.zeros(9, dtype=bool),
        "u": np.full(9, np.nan),
        "v": np.full(9, np.nan),
        "Exx": np.full(9, np.nan),
        "Eyy": np.full(9, np.nan),
        "Exy": np.full(9, np.nan),
    }

    def fake_run(*_args, **_kwargs):
        calls.append(True)
        return dict(fake_invalid if len(calls) == 1 else fake_valid)

    def fake_export(field, output_path, *, stem="dic_field", preset_name="publication"):
        del field, preset_name
        path = Path(output_path) / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("committed valid frame\n", encoding="utf-8")
        return {"txt": path}

    monkeypatch.setattr(ezdic, "run_2d_dic", fake_run)
    monkeypatch.setattr(ezdic, "export_dic_field_outputs", fake_export)
    app.process_fullfield(settings)

    dic_dir = output_dir / "dic"
    assert (dic_dir / "frame_0003.txt").exists()
    assert not (dic_dir / "frame_0002.txt").exists()
    assert not list(dic_dir.glob(".staging*"))


def test_fullfield_preflight_blocks_narrow_roi_without_three_by_three_grid(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    load_two_frame_sequence(app, tmp_path / "images_narrow_ff", tmp_path / "out_narrow_ff")
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    app.field_roi = (10, 10, 30, 30)
    app.field_roi_reference_frame_1based = 1
    app.dic_subset_size.set(21)
    app.dic_step.set(10)

    items = app.build_preflight_items()
    assert any(item["level"] == "block" and item["label"] == "POI 网格" for item in items)
    with pytest.raises(RuntimeError, match="3×3|POI"):
        app.validate_fullfield_before_processing()


def _fullfield_snapshot(image_paths, output_dir, **overrides):
    settings = {
        "image_paths": [str(path) for path in image_paths],
        "start_idx": 0,
        "end_idx": len(image_paths) - 1,
        "reference_frame_1based": 1,
        "field_roi": (12, 12, 40, 40),
        "field_roi_reference_frame_1based": 1,
        "output_dir": str(output_dir),
        "min_texture_std": 8.0,
        "min_texture_contrast": 25.0,
        "max_saturated_frac": 0.20,
        "dic_subset_size": 15,
        "dic_step": 8,
        "dic_solver": ezdic.DIC_SOLVER_ICGN,
        "dic_strain_window": 5,
        "dic_smooth_sigma": 0.0,
        "dic_search_radius": 12,
        "dic_zncc_min": 0.75,
        "dic_pyramid_levels": 1,
        "dic_pyramid_scale": 0.5,
    }
    settings.update(overrides)
    return settings


@pytest.mark.skipif(os.name != "nt", reason="requires a real Windows junction")
def test_fullfield_legacy_migration_rejects_real_dic_junction_before_core(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "out_junction"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    outside_file = outside / "frame_0002_u.png"
    outside_bytes = b"outside legacy bytes\x00\xff"
    outside_file.write_bytes(outside_bytes)

    junction = output_root / "dic"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    image_dir = tmp_path / "images_junction"
    image_dir.mkdir()
    reference = ezdic.generate_synthetic_speckle(64, 64, seed=93)
    image_paths = []
    for index, shift in enumerate((0.0, 0.7)):
        path = image_dir / f"frame_{index + 1:03d}.png"
        image = ezdic.warp_image_translation(reference, shift, 0.0).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        encoded.tofile(str(path))
        image_paths.append(path)

    core_calls = []
    monkeypatch.setattr(
        ezdic._core,
        "run_fullfield_sequence",
        lambda *_args, **_kwargs: core_calls.append(True),
    )
    app = object.__new__(ezdic.MultiROIGUI)
    try:
        with pytest.raises(RuntimeError, match="REPARSE_PATH_REJECTED|junction|symlink"):
            app.process_fullfield(_fullfield_snapshot(image_paths, output_root))
    finally:
        # Remove the junction itself, never its outside target.
        subprocess.run(["cmd", "/c", "rmdir", str(junction)], check=False)

    assert core_calls == []
    assert outside_file.read_bytes() == outside_bytes
    assert outside_file.resolve() == outside_file
    assert not (outside / "_previous_runs").exists()
    assert not (output_root / "_previous_runs").exists()


def test_fullfield_worker_uses_snapshot_without_tk_reads(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    image_dir = tmp_path / "images_worker_snapshot"
    image_dir.mkdir()
    reference = ezdic.generate_synthetic_speckle(64, 64, seed=91)
    image_paths = []
    for index, shift in enumerate((0.0, 0.7)):
        path = image_dir / f"frame_{index + 1:03d}.png"
        image = ezdic.warp_image_translation(reference, shift, 0.0).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        encoded.tofile(str(path))
        image_paths.append(path)

    output_dir = tmp_path / "out_worker_snapshot"
    settings = _fullfield_snapshot(image_paths, output_dir)
    callbacks = []
    archive_events = []
    engine_events = []
    app.post_to_ui = lambda callback, **_kwargs: callbacks.append(callback)
    app.validate_fullfield_before_processing = lambda *args, **kwargs: pytest.fail(
        "worker must not call the Tk-bound fullfield validator"
    )

    class _ForbiddenTkVariable:
        def get(self):
            raise AssertionError("worker attempted to read a Tk variable")

    # If the worker accidentally consults GUI state, this replaces the values
    # it would be most likely to read after the main-thread snapshot exists.
    forbidden_names = (
        "output_folder",
        "dic_subset_size",
        "dic_step",
        "dic_solver",
        "dic_strain_window",
        "dic_smooth_sigma",
        "dic_search_radius",
        "dic_zncc_min",
        "dic_pyramid_levels",
        "dic_pyramid_scale",
    )
    original_variables = {name: getattr(app, name) for name in forbidden_names}
    for name in forbidden_names:
        setattr(app, name, _ForbiddenTkVariable())

    def fake_archive(path):
        archive_events.append(Path(path))

    field = {
        "frame_global_1based": 2,
        "frame_filename": image_paths[1].name,
        "valid": np.ones(9, dtype=bool),
        "Exx": np.zeros(9),
        "Eyy": np.zeros(9),
        "Exy": np.zeros(9),
    }

    def fake_engine(engine_settings, progress_callback=None):
        engine_events.append(engine_settings)
        return {
            "scientific_ok": True,
            "manifest_path": str(output_dir / "run_manifest.json"),
            "last_field": field,
            "last_image": np.zeros((64, 64), dtype=np.uint8),
            "frames": [{"status": "scientific_valid"}],
            "manifest": {"reference_frame_1based": 1, "reference_filename": image_paths[0].name},
        }

    monkeypatch.setattr(ezdic, "archive_previous_fullfield_outputs", fake_archive)
    monkeypatch.setattr(ezdic._core, "run_fullfield_sequence", fake_engine)
    app.show_field_viewer = lambda *_args, **_kwargs: None
    app.show_completion_and_open_output_folder = lambda *_args, **_kwargs: None
    app.log = lambda *_args, **_kwargs: None

    errors = []

    def run_worker():
        try:
            app.process_fullfield_thread(settings)
        except Exception as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    try:
        worker = threading.Thread(target=run_worker, name="ezdic-fullfield-test-worker")
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert errors == []
        assert len(engine_events) == 1
        assert len(archive_events) == 1
        assert callbacks
    finally:
        for name, variable in original_variables.items():
            setattr(app, name, variable)


def test_fullfield_invalid_snapshot_fails_before_archive_and_core(tmp_path, monkeypatch):
    image_dir = tmp_path / "images_invalid_snapshot"
    image_dir.mkdir()
    reference = ezdic.generate_synthetic_speckle(64, 64, seed=92)
    image_paths = []
    for index in range(2):
        path = image_dir / f"frame_{index + 1:03d}.png"
        image = ezdic.warp_image_translation(reference, float(index), 0.0).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        encoded.tofile(str(path))
        image_paths.append(path)

    archive_events = []
    engine_events = []
    monkeypatch.setattr(
        ezdic,
        "archive_previous_fullfield_outputs",
        lambda path: archive_events.append(Path(path)),
    )
    monkeypatch.setattr(
        ezdic._core,
        "run_fullfield_sequence",
        lambda *_args, **_kwargs: engine_events.append(True),
    )
    app = object.__new__(ezdic.MultiROIGUI)
    settings = _fullfield_snapshot(
        image_paths,
        tmp_path / "out_invalid_snapshot",
        dic_subset_size=7,
    )

    with pytest.raises(RuntimeError, match="子集尺寸"):
        app.process_fullfield(settings)
    assert archive_events == []
    assert engine_events == []


def test_fullfield_settings_ignore_invalid_hidden_1d_parameters(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    load_two_frame_sequence(app, tmp_path / "images_hidden_1d", tmp_path / "out_hidden_1d")
    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    app.field_roi = (10, 10, 80, 70)
    app.field_roi_reference_frame_1based = 1
    for variable, value in [
        (app.search_radius, "not-an-int"),
        (app.hard_corr, "not-a-float"),
        (app.soft_corr, "not-a-float"),
        (app.template_alpha, "not-a-float"),
        (app.fb_tolerance_px, "not-a-float"),
        (app.overlay_every, "not-an-int"),
        (app.max_frame_strain_jump, "not-a-float"),
        (app.pixel_size_mm, "not-a-float"),
    ]:
        variable.set(value)

    settings = app.build_processing_settings()
    assert settings["analysis_mode"] == ezdic.ANALYSIS_MODE_FULLFIELD
    assert settings["field_roi"] == (10, 10, 80, 70)
    assert settings["dic_subset_size"] == 21


def test_fullfield_overlay_checkbox_and_workflow_steps_are_visible_and_mode_specific(gui_app):
    root, app = gui_app
    reset_gui_app(app)
    checkbox = app.fullfield_export_overlays_checkbutton
    assert "overlay" in checkbox.cget("text").lower()
    assert app.fullfield_export_info_frame.grid_info() == {}

    app.analysis_mode.set(ezdic.ANALYSIS_MODE_FULLFIELD)
    app.set_analysis_mode()
    root.update_idletasks()
    assert app.fullfield_export_info_frame.grid_info()
    fullfield_text = app.workflow_steps_label.cget("text")
    assert "全场 ROI" in fullfield_text
    assert "u/v" in fullfield_text

    app.analysis_mode.set(ezdic.ANALYSIS_MODE_EXTENSOMETER)
    app.set_analysis_mode()
    root.update_idletasks()
    assert app.fullfield_export_info_frame.grid_info() == {}
    assert "ROI1/ROI2" in app.workflow_steps_label.cget("text")


def test_processing_rejects_duplicate_start_while_completion_is_pending(gui_app, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    app._completion_pending = True
    messages = []
    monkeypatch.setattr(ezdic.messagebox, "showinfo", lambda title, message: messages.append((title, message)))
    app.start_processing()
    assert messages
    assert "完成" in messages[-1][1] or "等待" in messages[-1][1]


def test_constant_texture_tracking_is_rejected_and_strain_stays_nan(gui_app):
    _root, app = gui_app
    reset_gui_app(app)
    app.first_img8 = np.full((100, 140), 80, dtype=np.uint8)
    app.current_preview_index = 0
    app.start_frame_1based.set(1)
    app.roi1 = (10, 10, 30, 30)
    app.roi2 = (80, 10, 30, 30)
    app.strain_mode.set("x")
    app.sync_strain_mode_display()
    group = app.make_group_from_current("flat")
    state = app.init_group_states(app.first_img8, [group])[0]
    params = {
        "search_radius_base": 10,
        "hard_corr": 0.55,
        "soft_corr": 0.35,
        "enable_adaptive": True,
        "use_prev_frame_template": True,
        "template_alpha": 0.7,
        "max_frame_jump": 0.01,
        "enable_fb_check": True,
        "fb_tolerance": 12.0,
        "pixel_size_mm": None,
    }

    row0, _ = app.process_one_group_one_frame(state, app.first_img8, 0, "flat_001.png", params)
    row1, _ = app.process_one_group_one_frame(state, app.first_img8, 1, "flat_002.png", params)
    assert row0["accepted"] is False
    assert row0["accept_mode"] == "rejected"
    assert np.isnan(row0["engineering_strain"])
    assert row1["accepted"] is False
    assert np.isnan(row1["engineering_strain"])


def test_overlay_write_failure_is_reported(gui_app, tmp_path, monkeypatch):
    _root, app = gui_app
    reset_gui_app(app)
    monkeypatch.setattr(ezdic.messagebox, "askyesno", lambda *args, **kwargs: True)
    folder = tmp_path / "images_write_failure"
    folder.mkdir()
    speckle = ezdic.generate_synthetic_speckle(100, 160, seed=76)
    for idx in range(2):
        arr = ezdic.warp_image_translation(speckle, float(idx), 0.0).astype(np.uint8)
        ok, data = cv2.imencode(".png", arr)
        assert ok
        data.tofile(str(folder / f"frame_{idx:03d}.png"))
    app.image_folder.set(str(folder))
    app.output_folder.set(str(tmp_path / "out_write_failure"))
    app.load_first_image()
    app.roi1 = (20, 30, 30, 30)
    app.roi2 = (90, 30, 30, 30)
    app.strain_mode.set("x")
    app.sync_strain_mode_display()
    app.add_current_group()
    app.export_origin_txt.set(False)
    app.export_engineering_png.set(False)
    app.export_qc_summary.set(False)
    app.export_overlays.set(True)
    app.overlay_every.set(1)
    monkeypatch.setattr(ezdic.cv2, "imencode", lambda *_args, **_kwargs: (False, None))

    with pytest.raises(RuntimeError, match="写入图像失败"):
        app.process_images(app.build_processing_settings())
