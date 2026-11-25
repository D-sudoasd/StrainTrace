import sys
import threading
import time
import tkinter as tk
from contextlib import closing
from dataclasses import asdict, replace

import numpy as np
import pytest

from straintrace_capture import CaptureConfig, gui as capture_ui
from straintrace_capture.cameras import CameraRegistry
from straintrace_capture.cameras.base import Image, receipt_times
from straintrace_capture.cameras.simulated import SimulatedCamera, simulated_device
from straintrace_capture.draft import capture_draft
from straintrace_capture.gui import CaptureWorkbench, demo_config
from straintrace_capture.storage import ExperimentWriter
from straintrace_capture.viewport import ImageViewport


def destroy_window(root):
    for callback in root.tk.splitlist(root.tk.call("after", "info")):
        root.after_cancel(callback)
    root.destroy()


@pytest.fixture(scope="module")
def desktop_root():
    # The workbench uses one Tk interpreter per process. Give each test a
    # separate window without repeatedly loading the Windows ttk resources.
    root = tk.Tk()
    root.withdraw()
    try:
        yield root
    finally:
        destroy_window(root)


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises the Windows Tk viewport")
def test_closing_the_viewport_cancels_its_scheduled_redraw(desktop_root):
    parent = tk.Toplevel(desktop_root)
    try:
        view = ImageViewport(parent, "sim:0", "模拟相机", lambda *_args: None, lambda *_args: None)
        queued = set(desktop_root.tk.splitlist(desktop_root.tk.call("after", "info")))
        view.image(None, placeholder="等待归档图像")
        redraw, = set(desktop_root.tk.splitlist(desktop_root.tk.call("after", "info"))) - queued

        view.destroy()

        assert redraw not in desktop_root.tk.splitlist(desktop_root.tk.call("after", "info"))
        desktop_root.update_idletasks()
    finally:
        parent.destroy()


@pytest.mark.skipif(sys.platform != "win32", reason="Uses Windows Tk hit testing at high DPI")
def test_compact_capture_controls_are_reachable_and_restore_the_wide_layout(tmp_path, monkeypatch, desktop_root):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "state"))
    discover = CameraRegistry.discover
    monkeypatch.setattr(CameraRegistry, "discover", lambda registry: discover(registry, ("sim",)))
    root = tk.Toplevel(desktop_root)
    previous_scaling = root.tk.call("tk", "scaling")
    errors = []
    root.report_callback_exception = lambda _kind, error, _traceback: errors.append(error)
    try:
        root.tk.call("tk", "scaling", 2.6666667)
        root.attributes("-topmost", True)
        config = CaptureConfig.from_dict(demo_config())
        app = CaptureWorkbench(root, config)
        app._draft_tracking = False
        output = tmp_path / ("钛合金原位拉伸实验" * 4) / "第二批次" / "试样 B03"
        app.output_path.set(str(output))
        root.geometry("900x560+30+30")

        def settle(predicate):
            deadline = time.monotonic() + 8
            while not predicate():
                root.update()
                assert not errors, errors
                assert time.monotonic() < deadline
                time.sleep(.01)
            root.update()

        settle(lambda: app._compact_workspace and not app._discovering)
        app._draft_loaded(capture_draft(config, str(output), 0, ()))
        settle(lambda: app.output_entry.xview()[1] == 1.0)
        assert app.output_entry.xview()[0] > 0
        assert app.output_entry.winfo_width() >= 220
        assert app.restore_draft_button.winfo_width() >= app.restore_draft_button.winfo_reqwidth()
        assert set(app.tabs.tabs()) == {
            str(page) for page in (app._device_pane, app.plan_panel, app.quality_panel, app.review_panel)
        }
        app.tabs.select(app.plan_panel)
        selected = app.devices.selection()
        app.devices.selection_remove(*selected)
        settle(lambda: app._next_step.title == "选择相机")
        assert app.step_button.winfo_ismapped() and app.step_button.cget("text") == "定位到相机列表"
        assert "选择要采集的设备" in app.step_button._tooltip.resolve_text()
        root.focus_force()
        app.step_button.invoke()
        settle(lambda: root.focus_get() is app.devices)
        assert app.tabs.select() == str(app._device_pane) and not app.devices.selection()
        app.devices.selection_set(*selected)

        app.tabs.select(app._device_pane)
        settle(app.preview_button.winfo_ismapped)
        app.preview_button.focus_force()
        settle(lambda: app._device_scroll.canvas.yview()[1] == 1.0)
        x = app.preview_button.winfo_rootx() + app.preview_button.winfo_width() // 2
        y = app.preview_button.winfo_rooty() + app.preview_button.winfo_height() // 2
        assert root.winfo_containing(x, y) is app.preview_button

        app.tabs.select(app.plan_panel)
        settle(app.phases.winfo_ismapped)
        app.preflight_button.focus_force()
        settle(lambda: app.plan_editor.canvas.canvasy(0) > 0)
        y = app.preflight_button.winfo_rooty()
        top = app.plan_editor.canvas.winfo_rooty()
        assert top <= y and y + app.preflight_button.winfo_height() <= top + app.plan_editor.canvas.winfo_height()

        app.tabs.select(app._device_pane)
        settle(app.preview_button.winfo_ismapped)
        app.preview_button.focus_force()
        settle(lambda: root.focus_get() is app.preview_button)
        root.geometry("1280x760+30+30")
        settle(lambda: not app._compact_workspace)
        assert str(app._device_pane) not in app.tabs.tabs()
        assert app.tabs.select() == str(app.plan_panel)
        assert root.focus_get() is app.preview_button
        assert app.output_path.get() == str(output)
        assert not errors
    finally:
        root.tk.call("tk", "scaling", previous_scaling)
        destroy_window(root)


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises the Windows Capture workbench")
def test_review_shows_its_own_experiment_and_preserves_the_draft_and_selected_image(tmp_path, monkeypatch, desktop_root):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "state"))
    discover = CameraRegistry.discover
    monkeypatch.setattr(CameraRegistry, "discover", lambda registry: discover(registry, ("sim",)))
    source = tmp_path / "昨日采集" / "试样 A01"
    source.parent.mkdir()
    draft_path = tmp_path / "下一次采集"
    saved = CaptureConfig.from_dict({
        "config_version": 1, "name": "试样 A01 · 已完成", "cameras": [{"id": "sim:0"}],
        "phases": [{"mode": "continuous", "frame_count": 2}],
    })
    draft = CaptureConfig.from_dict({
        "config_version": 1, "name": "试样 B02 · 下次采集", "cameras": [{"id": "sim:1"}],
        "phases": [{"mode": "single"}],
    })
    with closing(ExperimentWriter(source, saved)) as writer:
        writer.register_camera(simulated_device(0), {})
        for number in (1, 2):
            image = Image(np.full((16, 24), number, dtype=np.uint8), "Mono8", 8, "mono", *receipt_times())
            writer.commit_frames([writer.prepare_frame("sim:0", number, 1, "test", image, None, None, None, 0)])
        writer.set_state("completed", {})

    root = tk.Toplevel(desktop_root)
    previous_scaling = root.tk.call("tk", "scaling")
    errors = []
    root.report_callback_exception = lambda _kind, error, _traceback: errors.append(error)
    try:
        root.tk.call("tk", "scaling", 2.0)
        root.geometry("900x560+30+30")
        app = CaptureWorkbench(root, draft, experiment=source)
        root.geometry("900x560+30+30")
        app._draft_tracking = False
        app.output_path.set(str(draft_path))

        def settle(predicate):
            deadline = time.monotonic() + 8
            while True:
                root.update()
                assert not errors, errors
                if predicate():
                    return
                assert time.monotonic() < deadline
                time.sleep(.01)

        settle(lambda: not app._discovering and app.review_image and not app._review_busy)
        assert app.tabs.select() == str(app.review_panel)
        assert not app.name_entry.winfo_ismapped() and not app.record_button.winfo_ismapped()
        assert app.current_name_entry.get() == saved.name
        assert app.current_path_entry.get() == str(source)
        assert app.current_path_entry.instate(["readonly"])
        assert not app.save_plan_button.winfo_ismapped()
        app.browse_frame(-1)
        settle(lambda: app.review_image[1].pixels[0, 0] == 1)
        chosen = app.review_image
        assert app._review_ordinal == 1 and tuple(app.views) == ("sim:0",)

        app.return_to_plan_button.invoke()
        settle(lambda: app.name_entry.winfo_ismapped() and app.record_button.winfo_ismapped())
        assert app.tabs.select() == str(app.plan_panel)
        assert not app.current_path_entry.winfo_ismapped()
        assert app.name_entry.get() == draft.name and app.output_entry.get() == str(draft_path)
        assert app.save_plan_button.winfo_ismapped() and not app.return_to_plan_button.winfo_ismapped()
        assert tuple(app.views) == ("sim:1",) and app._image("sim:1") is None
        assert app.session is None and app.preview is None and not draft_path.exists()

        app.tabs.select(app.review_panel)
        settle(lambda: app.current_path_entry.winfo_ismapped() and tuple(app.views) == ("sim:0",))
        assert app.review_image is chosen and app._review_ordinal == 1
        assert app._image("sim:0") is chosen[1]
        assert not app.record_button.winfo_ismapped()
        root.geometry("1280x760+30+30")
        settle(lambda: not app._compact_workspace and str(app._device_pane) not in app.tabs.tabs())
        assert not app._device_pane.winfo_ismapped() and app.review_image is chosen
        root.geometry("900x560+30+30")
        settle(lambda: app._compact_workspace and str(app._device_pane) in app.tabs.tabs())
        assert app.tabs.select() == str(app.review_panel) and app.review_image is chosen
    finally:
        root.tk.call("tk", "scaling", previous_scaling)
        destroy_window(root)


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises the Windows Capture workbench")
def test_output_warning_discards_an_old_probe_and_fills_a_new_path_without_creating_it(tmp_path, monkeypatch, desktop_root):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "state"))
    discover = CameraRegistry.discover
    monkeypatch.setattr(CameraRegistry, "discover", lambda registry: discover(registry, ("sim",)))
    original = capture_ui.inspect_output
    occupied, typed = tmp_path / "试样 B03", tmp_path / "另一个实验"
    occupied.mkdir()
    expected = tmp_path / "试样 B03-run-02"
    entered, release = threading.Event(), threading.Event()

    def delayed(output, **options):
        if output == str(typed):
            entered.set()
            assert release.wait(8)
        return original(output, **options)

    monkeypatch.setattr(capture_ui, "inspect_output", delayed)
    root = tk.Toplevel(desktop_root)
    errors = []
    root.report_callback_exception = lambda _kind, error, _traceback: errors.append(error)
    try:
        config = CaptureConfig.from_dict(demo_config())
        app = CaptureWorkbench(root, config)
        app._draft_tracking = False
        root.geometry("900x560+30+30")

        def settle(predicate):
            deadline = time.monotonic() + 8
            while True:
                root.update()
                assert not errors, errors
                if predicate():
                    return
                assert time.monotonic() < deadline
                time.sleep(.01)

        settle(lambda: not app._discovering)
        app.output_path.set(str(typed))
        settle(entered.is_set)
        app.output_path.set(str(occupied))
        release.set()
        settle(lambda: app._next_step.suggested_output == expected)
        assert app.record_button.instate(["disabled"])
        assert not app.preflight_button.instate(["disabled"])
        assert str(expected) in app.output_entry._tooltip.resolve_text()
        app.step_button.invoke()
        settle(lambda: app._output_check is not None and not app.record_button.instate(["disabled"]))
        assert app.output_path.get() == str(expected)
        assert app._read_config() == config and not expected.exists() and app.session is None
    finally:
        release.set()
        destroy_window(root)


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises the Windows Capture workbench")
def test_trigger_guide_follows_the_active_phase_when_the_monitor_is_hidden(tmp_path, monkeypatch, desktop_root):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "state"))
    discover = CameraRegistry.discover
    monkeypatch.setattr(CameraRegistry, "discover", lambda registry: discover(registry, ("sim",)))
    csv_path = tmp_path / "load.csv"
    csv_path.write_text("time,load\n0,2\n", encoding="utf-8")
    config = CaptureConfig.from_dict({
        "config_version": 1, "name": "单次事件与手动触发", "min_free_bytes": 0,
        "cameras": [{"id": "sim:0", "settings": {"width": 64, "height": 48}}],
        "phases": [{"mode": "event", "pre_frames": 0, "post_frames": 3, "fps": 4, "duration_s": 10,
                    "csv_trigger": {"path": str(csv_path), "value_column": "load", "threshold": 5, "direction": "rising",
                                    "quantity": "载荷", "unit": "kN"}},
                   {"mode": "manual", "frame_count": 1, "duration_s": 10}],
    })
    root = tk.Toplevel(desktop_root)
    errors = []
    root.report_callback_exception = lambda _kind, error, _traceback: errors.append(error)
    app = CaptureWorkbench(root, config)
    app._draft_tracking = False
    app.output_path.set(str(tmp_path / "run"))

    def settle(predicate):
        deadline = time.monotonic() + 8
        while True:
            root.update()
            assert not errors, errors
            if predicate():
                return
            assert time.monotonic() < deadline
            time.sleep(.01)

    try:
        settle(lambda: not app._discovering and not app.record_button.instate(["disabled"]))
        app.record_button.invoke()
        settle(lambda: app._trigger_phase == 1 and app.run_monitor._event_status is not None)
        app.tabs.select(app.quality_panel)
        settle(lambda: app._next_step.title == "事件前后采集" and "正在等待" in app._next_step.hint)
        settle(lambda: app.session.status()["phase_progress"]["csv_trigger"]["state"] == "watching")
        csv_path.write_text("time,load\n", encoding="utf-8")
        settle(lambda: app._next_step.title == "CSV 监看已停用" and app._next_step.target is app.trigger_button)
        app.step_button.invoke()
        settle(lambda: "事件后已收到" in app._next_step.hint)
        settle(lambda: app._trigger_phase == 2 and app._next_step.title == "等待手动触发")
        assert app.tabs.select() == str(app.quality_panel) and app.run_monitor._event_status is not None
        app.step_button.invoke()
        settle(lambda: app.session and app.session.finished)
        assert app.session.status()["state"] == "completed"
    finally:
        if app.session and not app.session.finished:
            app.session.request_stop()
            app.session.wait(8)
        destroy_window(root)


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises the Windows camera selection and preview controls")
def test_camera_selection_previews_new_identity_before_repairing_hardware_mapping(tmp_path, monkeypatch, desktop_root):
    monkeypatch.setenv("EZDIC_STATE_DIR", str(tmp_path / "state"))
    first, second, emulator = "pylon:left", "pylon:right", "pylon:emulator"
    actual = [replace(simulated_device(number), id=identity, backend="pylon", model="Fixture camera",
                      support="industrial") for number, identity in enumerate((first, second))]
    emulated = replace(simulated_device(7), id=emulator, backend="pylon", support="vendor_emulator")
    opened = []

    def camera(_registry, identity, *, refresh=False):
        opened.append(identity)
        device = actual[[first, second].index(identity)]
        instance = SimulatedCamera(0)
        instance.device = device
        return instance

    inventory = [asdict(device) for device in (simulated_device(0), *actual, emulated)]
    monkeypatch.setattr(CameraRegistry, "discover", lambda _registry: {"devices": inventory, "providers": {}})
    monkeypatch.setattr(CameraRegistry, "camera", camera)
    output = tmp_path / "正式采集"
    monkeypatch.setattr(CaptureWorkbench, "_new_path", lambda _self: str(output))
    root = tk.Toplevel(desktop_root)
    errors = []
    root.report_callback_exception = lambda _kind, error, _traceback: errors.append(error)
    app = CaptureWorkbench(root)
    app._draft_tracking = False

    def settle(condition):
        deadline = time.monotonic() + 8
        while True:
            root.update()
            assert not errors, errors
            if condition():
                return
            assert time.monotonic() < deadline
            time.sleep(.01)

    try:
        settle(lambda: app._next_step.target is app.camera_choice_button)
        assert app.devices.selection() == ("sim:0",) and not opened and not output.exists()
        app.step_button.invoke()
        dialog = app._camera_choice_dialog
        assert tuple(dialog.checks) == (first, second) and dialog.connect.instate(["disabled"])
        dialog.checks[first].invoke()
        dialog.top.destroy()
        assert app.devices.selection() == ("sim:0",) and not opened
        app.step_button.invoke()
        dialog = app._camera_choice_dialog
        dialog.checks[first].invoke()
        dialog.connect.invoke()
        settle(lambda: app.preview is not None and app._image(first) is not None)
        assert app.devices.selection() == (first,) and opened == [first] and not output.exists()
        assert app.name.get() == "新建采集"

        app.toggle_preview()
        settle(lambda: app.preview is None and not app._releasing_preview)
        plan = CaptureConfig.from_dict({
            "config_version": 1, "name": "按原接线采集", "cameras": [{"id": first}],
            "phases": [{"mode": "single", "role": "reference"},
                       {"mode": "hardware", "hardware_source": {first: "Line1"}, "frame_count": 3}],
        })
        app._set_config(plan)
        root.update()
        app.show_camera_choice()
        dialog = app._camera_choice_dialog
        assert dialog.choices[first].get() and not dialog.choices[second].get()
        dialog.checks[first].invoke()
        dialog.checks[second].invoke()
        dialog.connect.invoke()
        settle(lambda: app.preview is not None and app._image(second) is not None)
        assert app.devices.selection() == (second,) and opened == [first, second]
        assert app.config["phases"] == plan.to_dict()["phases"]
        app.tabs.select(app.quality_panel)
        settle(lambda: app._next_step.title == "核对物理触发输入")
        repaired = []
        monkeypatch.setattr(app, "edit_phase", lambda **options: repaired.append((app._selected_phase(), options)))
        app.step_button.invoke()
        assert repaired == [(1, {"wiring": True})] and app.preview is not None
        assert app.session is None and not output.exists()
    finally:
        if app.preview:
            app.preview.stop()
            app.preview.wait(3)
        destroy_window(root)
