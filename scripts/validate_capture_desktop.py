"""Exercise the Windows Capture and live Analysis controls and save desktop screenshots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import ttk


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def screenshot(root, output: Path, name: str):
    from PIL import ImageGrab
    from straintrace_capture.viewport import refresh_window_surface

    root.lift()
    root.update_idletasks()
    refresh_window_surface(root)
    if sys.platform == "win32":
        rendered = ImageGrab.grab(window=int(root.wm_frame(), 0))
    else:
        x, y = root.winfo_rootx(), root.winfo_rooty()
        rendered = ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height()))
    rendered.save(output / f"{name}.png")


def trigger_when(root, check, action, timeout=15):
    started = time.monotonic()
    def probe():
        if check():
            action()
        elif time.monotonic() - started > timeout:
            raise TimeoutError(f"timed out waiting for {action.__name__}")
        else:
            root.after(120, probe)
    root.after(120, probe)


def controls(parent, kind):
    for child in parent.winfo_children():
        if isinstance(child, kind):
            yield child
        yield from controls(child, kind)


def live_child(root, output):
    from straintrace_capture import ExperimentReader
    from straintrace_capture.live_gui import LiveWorkbench

    run = output / "experiment"
    root.geometry("1170x750+72+73")
    app = LiveWorkbench(root, run, "sim:0")
    root.geometry("1170x750+72+73")
    def saved_count():
        with ExperimentReader(run) as read:
            return len(read.frames(camera_id="sim:0"))

    observed = {}

    def pause_then_resume():
        app.pause_button.invoke()
        assert app._paused
        observed["receipts_at_pause"] = saved_count()
        trigger_when(root, lambda: saved_count() >= observed["receipts_at_pause"] + 4, resume)

    def resume():
        observed["receipts_at_resume"] = saved_count()
        app.pause_button.invoke()
        trigger_when(root, lambda: len(app.plot.points) >= 10, finish)

    def finish():
        assert any(point.strain_pct is not None for point in app.plot.points)
        root.attributes("-topmost", True)
        screenshot(root, output, "capture-live-analysis")
        observed["preview_points"] = len(app.plot.points)
        (output / "live-evidence.json").write_text(json.dumps(observed, indent=2) + "\n", encoding="utf-8")
        root.destroy()

    def start():
        assert app.start_button.instate(["disabled"])
        assert "Shift" in app.start_button._tooltip.resolve_text()
        app.rois["Gauge ROI 1"].set("273，113，33，33")
        assert "两个端点" in app.start_button._tooltip.resolve_text()
        app.rois["Gauge ROI 2"].set("273,298,33,33")
        app.strain_mode.set("y")
        assert app.start_button.instate(["!disabled"])
        observed["roi_guidance"] = True
        app.start_button.invoke()
        trigger_when(root, lambda: len(app.plot.points) >= 3, pause_then_resume)
    trigger_when(root, lambda: app.view._image is not None, start)


def capture_parent(root, output, scaling=None):
    from straintrace_capture import CaptureConfig, ExperimentReader, trial_report
    from straintrace_capture.gui import CaptureWorkbench, demo_config

    data = demo_config()
    data["phases"][1] = {"mode": "continuous", "duration_s": 90.0}
    settings = {"width": 640, "height": 480, "pixel_format": "Mono12", "motion_px": 0.04,
                "strain_per_frame": 0.00012, "fps": 12.0}
    data["cameras"] = [{"id": f"sim:{index}", "settings": settings} for index in range(2)]
    run = output / "experiment"
    app = CaptureWorkbench(root, CaptureConfig.from_dict(data))
    root.geometry("1230x770+55+65")
    app.output_path.set(str(run))
    child = None
    confirmed_preview = {}
    trial_evidence = {}
    help_evidence = {}
    adjustment_evidence = {}

    def initial():
        assert app.record_button.cget("text") == "开始模拟采集"
        assert "模拟练习" in app.selection_text.get()
        assert app.storage_mode.get() == "系统缓冲"
        assert app.step_button.cget("text") == "连接并预览"
        assert app.camera_context.get() == "模拟练习"
        screenshot(root, output, "capture-start")
        app.show_guide()
        trigger_when(root, lambda: app._guide_window.top.winfo_viewable(), check_guide)

    def check_guide():
        guide = app._guide_window
        assert tuple(guide.tabs.tab(tab, "text") for tab in guide.tabs.tabs()) == (
            "连接相机", "检查画面", "检查计划", "开始采集", "复查分析",
        )
        guide.tabs.select(4)

        def navigate():
            assert guide.go.cget("text") == "转到复查与分析"
            screenshot(guide.top, output, "capture-guide")
            guide.go.invoke()
            assert app.tabs.select() == str(app.review_panel) and app.session is None and not run.exists()
            app.tabs.select(app.plan_panel)
            help_evidence["guide_navigates_without_starting_capture"] = True
            app.devices.selection_remove(*app.devices.selection())
            trigger_when(root, lambda: app.record_button.instate(["disabled"]), hover_start)

        root.after(240, navigate)

    def hover_start():
        app.name_entry.focus_force()
        button = app.record_button
        button.event_generate("<Enter>", x=10, y=10, rootx=button.winfo_rootx()+10, rooty=button.winfo_rooty()+10)
        trigger_when(root, lambda: button._tooltip.tip_window is not None, dismiss_start_help)

    def dismiss_start_help():
        help_card = app.record_button._tooltip
        assert "选择至少一台相机" in help_card.shown_text
        assert "当前不可用" in help_card._heading.cget("text")
        screenshot(help_card.tip_window, output, "capture-disabled-help")
        app.devices.selection_set("sim:0", "sim:1")
        trigger_when(root, lambda: help_card.shown_text and "先检查计划" in help_card.shown_text, updated_start_help)

    def updated_start_help():
        help_card = app.record_button._tooltip
        help_evidence["hover_refresh_after_camera_selection"] = True
        assert "当前不可用" not in help_card._heading.cget("text")
        app.name_entry.event_generate("<Shift-F1>")
        assert help_card.tip_window is not None and app.name_entry._tooltip.tip_window is None
        app.record_button.event_generate("<Leave>")
        app.preflight_button.event_generate("<Enter>", x=4, y=4)
        assert app.preflight_button._tooltip.after_id is None and help_card.tip_window is not None
        root.after(220, dismiss_pinned_help)

    def dismiss_pinned_help():
        help_card = app.record_button._tooltip
        assert help_card.tip_window is not None
        screenshot(help_card.tip_window, output, "capture-pinned-help")
        app.name_entry.event_generate("<Escape>")
        assert help_card.tip_window is None and app.session is None
        help_evidence["disabled_reason_and_escape"] = True
        help_evidence["pointer_help_can_be_pinned_and_dismissed"] = True
        app.devices.focus("sim:0")
        app.devices.focus_force()
        app.devices.event_generate("<F1>")
        assert "设备编号 sim:0" in app.devices._tooltip.shown_text
        app.devices.event_generate("<Escape>")
        assert app.devices._tooltip.tip_window is None
        help_evidence["camera_keyboard_help"] = True
        trigger_when(root, lambda: app.settings_button.instate(["!disabled"]), camera_settings)

    def camera_settings():
        app.settings_button.invoke()
        trigger_when(root, lambda: next((widget for widget in root.winfo_children()
                                        if isinstance(widget, tk.Toplevel) and widget.title().startswith("相机参数")), None),
                     check_camera_settings)

    def check_camera_settings():
        dialog = next(widget for widget in root.winfo_children()
                      if isinstance(widget, tk.Toplevel) and widget.title().startswith("相机参数"))
        formats = next(widget for widget in controls(dialog, ttk.Combobox) if "Mono12" in widget.get())
        assert "12 位灰度" in formats.get()
        assert any("2 字节" in text for text in formats._tooltip_choices.values())
        exposure = next(widget for widget in controls(dialog, ttk.Entry) if widget._tooltip.title == "曝光时间")
        assert "微秒" in exposure._tooltip.resolve_text()

        def finish():
            screenshot(dialog, output, "capture-camera-settings")
            help_evidence["exposure_units_and_pixel_format_choices"] = True
            dialog.destroy()
            app.plan_buttons[1].invoke()
            trigger_when(root, lambda: next((widget for widget in root.winfo_children()
                                            if isinstance(widget, tk.Toplevel) and widget.title() == "采集阶段"), None),
                         check_phase_editor)

        root.after(300, finish)

    def check_phase_editor():
        dialog = next(widget for widget in root.winfo_children()
                      if isinstance(widget, tk.Toplevel) and widget.title() == "采集阶段")
        modes = next(widget for widget in controls(dialog, ttk.Combobox) if "硬件触发" in widget.cget("values"))
        assert "物理脉冲" in modes._tooltip_choices["硬件触发"]
        modes.set("事件前后")
        modes.event_generate("<<ComboboxSelected>>")
        fields = {widget._tooltip.title: widget._tooltip.resolve_text() for widget in controls(dialog, ttk.Entry)}
        assert "填 0" in fields["事件前张数"] and "内存" in fields["事件前张数"]
        assert "至少 1 张" in fields["事件后张数"]
        assert "每秒计划请求 10 张" in fields["定时频率（张/秒）"]

        def finish():
            screenshot(dialog, output, "capture-phase-editor")
            modes.set("硬件触发")
            modes.event_generate("<<ComboboxSelected>>")
            wiring = next(button for button in controls(dialog, ttk.Button) if button.cget("text") == "物理输入与脉冲编号…")
            wiring.invoke()
            trigger_when(root, lambda: next((window for window in dialog.winfo_children()
                                            if isinstance(window, tk.Toplevel) and window.winfo_viewable()), None),
                         lambda: check_hardware_editor(dialog))

        root.after(300, finish)

    def check_hardware_editor(dialog):
        hardware = next(window for window in dialog.winfo_children() if isinstance(window, tk.Toplevel))
        counter = next(widget for widget in controls(hardware, ttk.Combobox) if "相机帧计数" in widget.cget("values"))
        assert counter.instate(["disabled"])
        assert "先勾选" in counter._tooltip.resolve_text()
        pair = next(widget for widget in controls(hardware, ttk.Checkbutton) if "关联图像" in widget.cget("text"))
        pair.invoke()
        assert counter.instate(["!disabled"])
        assert "设备字段 ChunkFramecounter" in counter._tooltip_choices["相机帧计数"]

        def finish():
            screenshot(hardware, output, "capture-hardware-help")
            hardware.destroy()
            help_evidence["hardware_counter_choices_and_prerequisites"] = True
            dialog.destroy()
            assert len(app.config["phases"]) == 2 and app.config["phases"][0]["mode"] == "single"
            app.template_button.invoke()
            template = app._template_window
            assert template.menu._tooltip_choices["参考帧和定时实验"] and len(template.table.get_children()) == 2
            template.top.destroy()
            help_evidence["phase_fields_and_templates"] = True
            preview()

        root.after(240, finish)

    def preview():
        assert app.step_button.cget("text") == "连接并预览"
        app.step_button.invoke()
        trigger_when(root, lambda: app.preview and all(app.preview.latest_image(camera) is not None for camera in app.views),
                     adjust_preview)

    def adjust_preview():
        source = app.preview
        app.select_view("sim:0")
        assert app.settings_button.instate(["!disabled"])
        assert "收到新画面" in app.settings_button._tooltip.resolve_text()
        peer_before = source.latest_image("sim:1").received_monotonic_ns
        app.settings_button.invoke()
        dialog = next(widget for widget in root.winfo_children()
                      if isinstance(widget, tk.Toplevel) and widget.title().startswith("相机参数"))
        exposure = next(widget for widget in controls(dialog, ttk.Entry) if widget._tooltip.title == "曝光时间")
        assert float(exposure.get()) == 2000.
        exposure.delete(0, tk.END)
        exposure.insert(0, "1200")
        button = next(widget for widget in controls(dialog, ttk.Button) if widget.cget("text") == "应用并更新预览")
        assert "短暂停止这台相机" in button._tooltip.resolve_text()
        button.invoke()

        def reviewed():
            assert app.preview is source and app.tabs.select() == str(app.quality_panel)
            assert source.status()["sim:0"]["features"]["exposure_us"] == 1200.
            assert source.status()["sim:1"]["features"]["exposure_us"] == 2000.
            assert source.latest_image("sim:1").received_monotonic_ns > peer_before
            assert app._read_config().cameras == source.cameras
            adjustment_evidence.update(exposure_us={"sim:0": 1200., "sim:1": 2000.},
                                       comparison=app._preview_adjustment["note"], peer_advanced=True)
            screenshot(root, output, "capture-preview-adjustment")
            inspect()

        trigger_when(root, lambda: "调参前后" in app.quality_text.get(), reviewed)

    def inspect():
        for camera in app.views:
            confirmed_preview[camera] = app.preview.latest_image(camera).pixels.shape
        focus = app.views["sim:0"]
        canvas = focus.canvas
        zoom, left, top = focus._project
        x, y = round(left + 155 * zoom), round(top + 98 * zoom)
        end_x, end_y = round(left + 355 * zoom), round(top + 265 * zoom)
        canvas.event_generate("<ButtonPress-1>", x=x, y=y, state=1)
        canvas.event_generate("<B1-Motion>", x=end_x, y=end_y, state=257)
        canvas.event_generate("<ButtonRelease-1>", x=end_x, y=end_y, state=1)
        assert focus.roi is not None
        canvas.event_generate("<Motion>", x=end_x, y=end_y)
        px, py = focus._probe_coordinate
        assert focus._caption.cget("text") == f"({px},{py}) {focus._image.pixels[py, px]}"
        canvas.event_generate("<Leave>")
        trigger_when(root, lambda: any(
            isinstance(window, tk.Toplevel) and window.title() == "输入检查区域" and window.winfo_viewable()
            for window in root.winfo_children()
        ), enter_region)
        app.set_region()
        assert focus.roi == (155, 98, 200, 167)
        app.guide.set("Crosshair")
        app.apply_preview_options()
        trigger_when(root, lambda: "局部纹理" in app.quality_text.get(), check_plan)

    def enter_region():
        dialog = next(window for window in root.winfo_children()
                      if isinstance(window, tk.Toplevel) and window.title() == "输入检查区域")
        entries = list(controls(dialog, ttk.Entry))
        entries[0].delete(0, tk.END)
        entries[0].insert(0, "-1")
        dialog.ok()
        assert dialog.winfo_exists() and "完整落在原图内" in dialog.error.cget("text")
        for entry, value in zip(entries, ("155", "98", "200", "167"), strict=True):
            entry.delete(0, tk.END)
            entry.insert(0, value)
        assert "像素数" in entries[0]._tooltip.resolve_text()
        screenshot(dialog, output, "capture-region-input")
        dialog.ok()
        help_evidence["localized_roi_fields_and_validation"] = True

    def check_plan():
        screenshot(root, output, "capture-preview")
        app.tabs.select(app.plan_panel)
        assert app.step_button.cget("text") == "检查计划"
        app.step_button.invoke()
        trigger_when(root, lambda: getattr(app, "_plan_report_window", None), record)

    def record():
        assert "采前检查通过" in app.plan_summary.get()
        assert not run.exists()
        assert "待估算" not in app.plan_summary.get()
        screenshot(app._plan_report_window, output, "capture-preflight")
        app.output_path.set(str(output / "another-output"))
        assert "已修改" in app.plan_summary.get()
        assert "已修改" in app._plan_report_window.title()
        app.output_path.set(str(run))
        help_evidence["invalidated_preflight_after_edit"] = True
        app.storage_mode.set("逐张刷盘")
        app.storage_mode.event_generate("<<ComboboxSelected>>")
        assert app._read_config().durability == "write_through"
        storage_help = app.storage_mode._tooltip.resolve_text()
        assert "刷盘后登记台账" in storage_help and "停采后的等待时间" in storage_help
        help_evidence["storage_mode_selection"] = True
        app._plan_report_window.destroy()
        root.geometry("1020x665+60+76")
        root.after(1200, compact)

    def compact():
        controls = [app.preview_heading, app.storage_mode, *app.preview_controls, *app.plan_buttons,
                    app.preflight_button, app.trial_button,
                    *(view._caption for view in app.views.values())]
        screenshot(root, output, "capture-compact")
        for widget in controls:
            assert widget.winfo_ismapped() and widget.winfo_width() >= widget.winfo_reqwidth(), (
                str(widget), widget.winfo_width(), widget.winfo_reqwidth()
            )
        app.preflight_button.event_generate("<Enter>", x=15, y=10)
        trigger_when(root, lambda: app.preflight_button._tooltip.tip_window is not None, hover_verified)

    def hover_verified():
        assert "磁盘空间" in app.preflight_button._tooltip.shown_text
        app.preflight_button.event_generate("<Leave>")
        assert app.preflight_button._tooltip.tip_window is None
        app.phases.selection_set("1")
        trigger_when(root, lambda: any(
            isinstance(window, tk.Toplevel) and window.title().startswith("试采阶段") and window.winfo_viewable()
            for window in root.winfo_children()
        ), fill_trial)
        app.trial_button.invoke()
        trigger_when(root, lambda: getattr(app, "_trial_report_window", None) is not None, trial_done)

    def fill_trial():
        dialog = next(window for window in root.winfo_children() if isinstance(window, tk.Toplevel)
                      and window.title().startswith("试采阶段"))
        entry = next(controls(dialog, ttk.Entry))
        assert "时限" in entry._tooltip.title
        entry.delete(0, tk.END)
        entry.insert(0, "0")
        dialog.ok()
        assert dialog.winfo_exists() and "大于零" in dialog.error.cget("text")
        entry.delete(0, tk.END)
        entry.insert(0, "1")
        dialog.ok()
        help_evidence["localized_trial_limit_validation"] = True

    def trial_done():
        assert app.session.finished and not run.exists()
        assert app.output_path.get() == str(run)
        assert len(app.config["phases"]) == 2
        assert app.session.output.parent == run.parent and app.session.output != run
        trial_evidence.update(trial_report(app.session.output))
        assert trial_evidence["observed"] and trial_evidence["trial"]["phase_number"] == 2
        assert trial_evidence["durability"] == "write_through"
        assert all(camera["host_receive_fps"] is not None for camera in trial_evidence["cameras"].values())
        screenshot(app._trial_report_window, output, "capture-trial-report")
        check = next(button for button in controls(app._trial_report_window, ttk.Button)
                     if button.cget("text") == "核对当前计划")
        dialog = app._trial_report_window
        dialog.geometry("520x360")

        def compact_report():
            assert check.winfo_viewable() and check.winfo_height() >= check.winfo_reqheight()
            assert check.winfo_rooty() + check.winfo_height() <= dialog.winfo_rooty() + dialog.winfo_height()
            screenshot(dialog, output, "capture-trial-compact-report")
            check.invoke()
            trigger_when(root, plan_report_visible, verified_trial_match)

        root.after(300, compact_report)

    def plan_report_visible():
        window = getattr(app, "_plan_report_window", None)
        return window is not None and window.winfo_exists()

    def complete_plan_report():
        window = app._plan_report_window
        text = next(controls(window, tk.Text))
        if window.readiness_overview is not None:
            window.readiness_overview.master.select(text.master)
        return text.get("1.0", tk.END)

    def verified_trial_match():
        assert app.tabs.select() == str(app.plan_panel)
        assert "已匹配 1 个阶段的试采" in app.plan_summary.get()
        content = complete_plan_report()
        assert "使用已校验的试采尺寸" in content and "计划设置匹配 · TIFF 校验通过" in content
        screenshot(app._plan_report_window, output, "capture-trial-matched")
        app._plan_report_window.destroy()
        app.durability.set("buffered")
        app.check_plan()
        trigger_when(root, plan_report_visible, verified_trial_stale)

    def verified_trial_stale():
        assert "缺少匹配" in app.plan_summary.get()
        content = complete_plan_report()
        assert "设置已变化 · 存储设置" in content
        assert "使用已校验的试采尺寸" not in content
        screenshot(app._plan_report_window, output, "capture-trial-changed")
        help_evidence["trial_settings_match_and_stale"] = True
        app._plan_report_window.destroy()
        app.durability.set("write_through")
        root.geometry("1230x770+55+65")
        app.tabs.select(app.plan_panel)
        app.record_button.invoke()
        trigger_when(root, lambda: app.session and app.session.output == run
                     and app.session.status()["cameras"]["sim:0"]["saved"] >= 8, launch)

    def launch():
        nonlocal child
        assert app.storage_mode.instate(["disabled"])
        screenshot(root, output, "capture-recording")
        logs = (output / "live-application.log").open("w", encoding="utf-8")
        try:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            command = [sys.executable, str(Path(__file__).resolve()), "--child", str(output)]
            if scaling is not None:
                command.extend(("--scaling", str(scaling)))
            child = subprocess.Popen(command,
                                     stdout=logs, stderr=logs, creationflags=flags)
        finally:
            logs.close()
        root.attributes("-topmost", False)
        trigger_when(root, lambda: child.poll() is not None, finish_recording, timeout=45)

    def finish_recording():
        app.stop_button.invoke()
        trigger_when(root, lambda: app.session.finished and app._run_reviewed, review, timeout=15)

    def review():
        if child.returncode:
            raise RuntimeError((output / "live-application.log").read_text(encoding="utf-8"))
        root.attributes("-topmost", True)
        assert app.tabs.select() == str(app.review_panel)
        trigger_when(root, lambda: app.review_image is not None, finish_review)

    def finish_review():
        expected = app.session.status()
        proof = {"geometry": confirmed_preview, "status": expected, "live": child.returncode,
                 "preflight": True, "hover_help": help_evidence, "compact_controls_visible": True,
                 "trial": trial_evidence, "preview_adjustment": adjustment_evidence}
        with ExperimentReader(run) as reader:
            assert reader.experiment["config"]["durability"] == "write_through"
            exposures = {row["device"]["id"]: row["features"]["exposure_us"]["value"] for row in reader.cameras()}
            assert exposures == adjustment_evidence["exposure_us"]
            proof["verification"] = reader.verify()
            parent_id = reader.experiment["experiment_id"]
            assert proof["verification"]["state"] == "stopped"
            assert proof["verification"]["ok"] and not proof["verification"]["reported_camera_losses"]

        def verify_continuation():
            assert app.config["metadata"]["continuation_of"] == parent_id
            assert app.storage_mode.get() == "逐张刷盘"
            sequel = Path(app.output_path.get())
            assert sequel.parent == run.parent and sequel != run and not sequel.exists()
            proof["continuation_prepared"] = True
            (output / "capture-evidence.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
            app.close()

        def check_saved():
            assert app._image("sim:0") is None
            assert app._image("sim:1") is app.review_image[1]
            assert app.preview_heading.cget("text") == "已保存"
            assert app._next_step.target is app.timeline_button
            assert app.step_button.cget("text") == "采集时间线…"
            assert "文件待校验" in app.review_result.cget("text")
            help_evidence["saved_image_and_camera_selection"] = True
            canvas = app.review_scroll.canvas

            def visible_in_review():
                menu = app.calibration_menu
                return (menu.winfo_viewable() and
                        canvas.winfo_rooty() <= menu.winfo_rooty() and
                        menu.winfo_rooty() + menu.winfo_height() <= canvas.winfo_rooty() + canvas.winfo_height())

            app.calibration_menu.focus_force()
            trigger_when(root, visible_in_review, check_review_scroll)

        def check_review_scroll():
            canvas = app.review_scroll.canvas
            assert app.calibration_menu.winfo_rooty() + app.calibration_menu.winfo_height() <= (
                canvas.winfo_rooty() + canvas.winfo_height()
            )
            app.calibration_menu.event_generate("<F1>")
            assert app.calibration_menu._tooltip.shown_text and "棋盘格图像" in app.calibration_menu._tooltip.shown_text
            app.calibration_menu.event_generate("<Escape>")
            assert app.calibration_menu._tooltip.tip_window is None
            help_evidence["review_tools_keyboard_scroll"] = True
            screenshot(root, output, "capture-review-tools")
            app.verify()
            trigger_when(root, lambda: getattr(app, "_integrity_report_window", None) is not None, verify_review)

        def verify_review():
            assert "校验通过" in app.review_result.cget("text")
            assert app._review_integrity["ok"] and app._review_integrity["file_hashes_checked"]
            assert "序号" in app.review_text.get()
            assert app.camera_context.get() == "实验复查"
            assert any(button.cget("text") == "复制报告" for button in controls(app._integrity_report_window, ttk.Button))
            screenshot(app._integrity_report_window, output, "capture-integrity-report")
            app._integrity_report_window.destroy()
            app.continue_button.invoke()
            trigger_when(root, lambda: getattr(app, "_continuation_window", None) is not None, select_continuation)

        def select_continuation():
            window = app._continuation_window
            assert window.table.selection()
            window.table.selection_set("1")
            window.load.invoke()
            trigger_when(root, lambda: "已载入续采计划" in app.plan_summary.get(), verify_continuation)

        screenshot(root, output, "capture-review")
        app.select_view("sim:1")
        trigger_when(root, lambda: (app.review_image is not None and app.review_image[0] == "sim:1"
                                    and not app._review_busy and app._next_step.target is app.timeline_button), check_saved)

    trigger_when(root, lambda: getattr(app, "_providers", None), initial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new QA directory")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--scaling", type=float, help="Tk pixels per point, normally 1.333 at 100%% scaling")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.child:
        output.mkdir(parents=True, exist_ok=False)

    from ezdic_gui.theme import enable_windows_dpi_awareness

    enable_windows_dpi_awareness()
    root = tk.Tk()
    if args.scaling is not None:
        root.tk.call("tk", "scaling", args.scaling)
    root.attributes("-topmost", True)
    failure = []

    def failed(exc, value, tb):
        traceback.print_exception(exc, value, tb)
        failure.append(str(value))
        root.destroy()

    root.report_callback_exception = failed
    if args.child:
        live_child(root, output)
    else:
        capture_parent(root, output, scaling=args.scaling)
    root.mainloop()
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
