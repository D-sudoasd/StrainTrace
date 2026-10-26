"""Real Tk paths for the researcher-editable automatic analysis workbench."""
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import tkinter as tk

import cv2
import numpy as np
import pytest

import dic_virtual_extensometer_gui_v7_multi_roi_range as legacy
import ezdic_core as core
from straintrace_agent import api


@pytest.fixture(scope="module")
def desktop():
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


@pytest.fixture
def workbench(desktop, monkeypatch, tmp_path):
    gc.collect()
    monkeypatch.setenv(legacy.RECENT_CONFIG_ENV_VAR, str(tmp_path / "recent.json"))
    monkeypatch.setenv(legacy.UI_SCALE_ENV_VAR, "1.666667")
    root = tk.Toplevel(desktop)
    errors = []
    desktop.report_callback_exception = lambda _kind, value, _trace: errors.append(value)
    app = legacy.MultiROIGUI(root)
    app.root.geometry("620x680+0+0")
    app.visual_window.geometry("820x680+0+0")
    app.control_notebook.select(app.agent_page)
    app.root.update()
    yield app
    if app.agent_page._worker is not None:
        app.agent_page._worker.join(10)
    for job in root.tk.call("after", "info"):
        root.after_cancel(job)
    for value in vars(app).values():
        if isinstance(value, tk.Variable):
            for modes, callback in value.trace_info():
                value.trace_remove(modes, callback)
    root.destroy()
    assert not errors


def wait(page, timeout=60):
    deadline = time.monotonic() + timeout
    while page._busy and time.monotonic() < deadline:
        page.update()
        time.sleep(0.01)
    page.update()
    assert not page._busy, page.status_var.get()


def create_project(app, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    reference = core.generate_synthetic_speckle(128, 128, seed=17)
    for index, shift in enumerate((0.0, 0.5), 1):
        image = reference if not shift else core.warp_image_translation(reference, shift, 0.0)
        assert cv2.imwrite(str(images / f"frame_{index:04d}.png"), np.clip(image, 0, 255).astype(np.uint8))
    page = app.agent_page
    page.source_var.set(str(images))
    page.project_var.set(str(tmp_path / "project"))
    page.interval_var.set("0.5")
    page.step_var.set("8")
    page.create_button.invoke()
    wait(page)
    assert page.project_dir == tmp_path / "project", page.status_var.get()
    assert page._image.size == (128, 128)
    return page


def draw(page, rect, target):
    page.show_image()
    page.update()
    page.draw_var.set(target)
    x, y, width, height = rect
    ox, oy = page._offset
    s = page._scale
    page.image_canvas.event_generate("<ButtonPress-1>", x=round(ox + x * s), y=round(oy + y * s))
    page.image_canvas.event_generate("<B1-Motion>", x=round(ox + (x + width) * s), y=round(oy + (y + height) * s))
    page.image_canvas.event_generate("<ButtonRelease-1>", x=round(ox + (x + width) * s), y=round(oy + (y + height) * s))
    page.update()


def test_narrow_short_workbench_keeps_actions_visible_and_help(workbench):
    app, page = workbench, workbench.agent_page
    for geometry in ("620x680", "620x560"):
        app.root.geometry(geometry)
        app.root.update()
        for widget in (page.save_button, page.plan_button, page.run_button, page.cancel_button, page.status_label):
            assert widget.winfo_ismapped()
            assert widget.winfo_rootx() + widget.winfo_width() <= app.root.winfo_rootx() + app.root.winfo_width()
            assert widget.winfo_rooty() + widget.winfo_height() <= app.root.winfo_rooty() + app.root.winfo_height()
            if widget is not page.status_label:
                assert widget._tooltip.resolve_text()
        assert page.input_page._canvas.winfo_height() > 20
        page.input_page._canvas.yview_moveto(1)
        page.update()
        assert page.create_button.winfo_ismapped()
    assert page.run_button.instate(["disabled"])
    assert page.report_button.instate(["disabled"])
    assert app.workspace_notebook.tab(page.visual_page, "text") == "自动分析图像"


def test_real_canvas_multi_roi_gauge_edit_save_and_reopen(workbench, tmp_path):
    page = create_project(workbench, tmp_path)
    for index, rect in enumerate(((25, 25, 30, 40), (65, 25, 30, 40)), 1):
        page.new_item()
        draw(page, rect, "统计 ROI")
        page.name_var.set(f"Region {index}")
        page.apply_button.invoke()
    for index, y in enumerate((30, 60), 1):
        page.new_item()
        draw(page, (25, y, 25, 25), "标距 A")
        draw(page, (70, y, 25, 25), "标距 B")
        page.name_var.set(f"Gauge {index}")
        page.axis_var.set("x")
        page.apply_button.invoke()
    assert len(page.items) == 4, page.status_var.get()
    page.geometry_tree.selection_set("0")
    page.update()
    page.name_var.set("Upper region")
    page.a_var.set("24, 26, 32, 38")
    page.apply_button.invoke()
    page.save_button.invoke()
    wait(page)
    saved = api.read_project_config(page.project_dir)
    assert len(saved["regions"]) == len(saved["gauges"]) == 2
    assert saved["regions"][0]["name"] == "Upper region"
    assert saved["regions"][0]["bounds"] == [24, 56, 26, 64]
    assert saved["gauges"][0]["a_roi_px"] == [25, 30, 25, 25]
    assert saved["gauges"][0]["axis"] == "x"
    assert api.status(page.project_dir)["current_plan"] is None
    page.open_project(page.project_dir)
    wait(page)
    assert len(page.items) == 4
    assert not page._dirty and not page._editor_dirty
    assert page.run_button.instate(["disabled"])
    page.reference_var.set("2")
    before = page.analysis_roi_var.get()
    draw(page, (20, 20, 60, 60), "计算区域")
    assert page.analysis_roi_var.get() == before
    assert "先保存" in page.status_var.get()


class RuntimeSpy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, threading.get_ident()))
            if name == "run":
                kwargs["make_plots"] = False
            return getattr(api, name)(*args, **kwargs)
        return call


def test_real_plan_run_report_integrity_and_reprocess(workbench, tmp_path, monkeypatch):
    page = create_project(workbench, tmp_path)
    spy = RuntimeSpy()
    page._api = spy
    main_thread = threading.get_ident()
    page.plan_button.invoke()
    wait(page)
    assert page.plan and page.plan["status"] == "ready", page.status_var.get()
    assert not page.run_button.instate(["disabled"])
    page.run_button.invoke()
    wait(page)
    assert "计算完成" in page.status_var.get(), page.status_var.get()
    assert "实验验收：未评估" in page.status_var.get()
    assert not page.report_button.instate(["disabled"])
    assert page.artifacts["report"].is_file()
    opened = []
    monkeypatch.setattr(os, "startfile", lambda path: opened.append(Path(path)), raising=False)
    if os.name == "nt":
        page.report_button.invoke()
        assert opened == [page.artifacts["report"]]
    page.verify_button.invoke()
    wait(page)
    assert "完整性核验：通过" in page.status_var.get(), page.status_var.get()
    checkpoint = next((page.run_dir / "checkpoints").glob("*.npz"))
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    page.new_item()
    page.kind_var.set("统计 ROI")
    page.name_var.set("Review region")
    page.a_var.set("30, 30, 50, 50")
    page.apply_button.invoke()
    page.reprocess_button.invoke()
    wait(page)
    assert "未重算 DIC" in page.status_var.get(), page.status_var.get()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    assert not api.read_project_config(page.project_dir)["regions"]
    assert all(thread != main_thread for _, thread in spy.calls)


def test_worker_responsive_cancel_and_failure_are_visible(workbench, tmp_path):
    page = create_project(workbench, tmp_path)
    started, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    calls = []
    class SlowAPI:
        def run(self, project, progress):
            calls.append(threading.get_ident())
            started.set()
            release.wait(3)
            raise RuntimeError("visible worker failure")
        def cancel(self, project):
            calls.append(threading.get_ident())
            release.set()
            return {"cancel_requested": True}
    page._api = SlowAPI()
    page.plan = {"status": "ready"}
    page._dirty = False
    page._update_actions()
    page.run_button.invoke()
    assert started.wait(1)
    ticks = []
    page.after(10, lambda: ticks.append(True))
    until = time.monotonic() + 0.1
    while time.monotonic() < until:
        page.update()
        time.sleep(0.005)
    assert ticks and page._busy
    assert page.save_button.instate(["disabled"])
    page.cancel_button.invoke()
    wait(page)
    assert "visible worker failure" in page.status_var.get()
    assert "计算完成" not in page.status_var.get()
    assert all(thread != main_thread for thread in calls)
    assert page.cancel_button.instate(["disabled"])


def test_image_only_input_uses_frame_identity_without_invented_seconds(workbench, tmp_path):
    page = create_project(workbench, tmp_path)
    page.interval_var.set("")
    page.save_button.invoke()
    wait(page)
    config = api.read_project_config(page.project_dir)
    assert config["time_basis"] == "frame_index"
    assert config["frame_interval_s"] is None
    inventory = json.loads((page.project_dir / "inventory.json").read_text(encoding="utf-8"))
    assert all(row["time_s"] is None for row in inventory["frames"])
    assert "原生帧号" in page.status_var.get()


def test_discovery_preserves_metadata_geometry_and_time(workbench, tmp_path):
    page = create_project(workbench, tmp_path)
    image_dir = Path(page.config["image_dir"])
    metadata = {"schema": "straintrace.experiment/v1", "options": {
        "frame_interval_s": 0.25,
        "regions": [{"name": "From record", "bounds": [20, 80, 20, 80], "coordinate_system": "image", "min_valid_fraction": 0.5, "min_valid_pois": 3}],
    }}
    (image_dir / "experiment.json").write_text(json.dumps(metadata), encoding="utf-8")
    page.interval_var.set("")
    page.discover_button.invoke()
    wait(page)
    assert page.interval_var.get() == "0.25", page.discovery_text.get("1.0", "end")
    assert page.items[0]["value"]["name"] == "From record"
    assert page._init_options()["regions"][0]["name"] == "From record"


def test_aggregate_attention_does_not_report_success(workbench):
    page = workbench.agent_page
    page._accept_run(({"status": "attention_required", "run": {"status": "complete", "numerical_quality_passed": True},
                       "objective_issue": "No supported gauge"}, None))
    assert "结果需要检查" in page.status_var.get()
    assert "计算完成" not in page.status_var.get()
