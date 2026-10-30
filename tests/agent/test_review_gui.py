"""Regression tests for project identity, reference geometry and Tk navigation."""
import gc
import json
import os
import threading
import time
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import pytest

import ezdic_core as core
from straintrace_agent import api
from straintrace_agent.gui import AgentPage, _snapshot


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
def page(desktop, monkeypatch, tmp_path):
    gc.collect()
    monkeypatch.setenv("EZDIC_RECENT_CONFIG", str(tmp_path / "recent.json"))
    host = tk.Toplevel(desktop)
    host.withdraw()
    host.geometry("620x460")
    errors = []
    desktop.report_callback_exception = lambda _kind, value, _trace: errors.append(value)
    result = AgentPage(host, api=api)
    # Explicit placement exercises real Tk geometry while every window remains
    # withdrawn. Hidden toplevels otherwise leave packed descendants at 1 px.
    result.place(x=0, y=0, width=620, height=460)
    visual = tk.Toplevel(desktop)
    visual.withdraw()
    notebook = ttk.Notebook(visual)
    notebook.place(x=0, y=0, width=640, height=460)
    result.attach_visual(notebook)
    host.update()
    yield result
    if result._worker is not None:
        result._worker.join(10)
    result.destroy()
    visual.destroy()
    host.destroy()
    assert not errors


def wait(page, timeout=45):
    deadline = time.monotonic() + timeout
    while page._busy and time.monotonic() < deadline:
        page.update()
        time.sleep(0.01)
    page.update()
    assert not page._busy, page.status_var.get()


def project(page, tmp_path, *, reference=None, regions=(), blank_first=False):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    reference_image = core.generate_synthetic_speckle(128, 128, seed=17)
    for index in ((1, 2, 3) if blank_first else (1, 2)):
        frame = np.zeros((128, 128), dtype=np.uint8) if blank_first and index == 1 else np.clip(reference_image, 0, 255).astype(np.uint8)
        assert cv2.imwrite(str(images / f"frame_{index:04d}.png"), frame)
    destination = tmp_path / "project"
    api.init_project(destination, images, reference_native_frame=reference,
                     step_hint_px=8, regions=list(regions), time_basis="frame_index")
    page.open_project(destination)
    wait(page)
    assert page.project_dir == destination
    return destination


def region(name, offset=20):
    return {"name": name, "bounds": [offset, offset + 30, 20, 60],
            "coordinate_system": "image", "min_valid_fraction": 0.5, "min_valid_pois": 3}


@pytest.mark.parametrize("choose_directory", (False, True))
def test_changed_project_directory_cannot_operate_on_previous_project(page, tmp_path, monkeypatch, choose_directory):
    a_dir = project(page, tmp_path)
    b_dir = tmp_path / "project-b"
    api.init_project(b_dir, tmp_path / "images", step_hint_px=8, time_basis="frame_index")
    a_before = (a_dir / "project.json").read_bytes()
    b_before = (b_dir / "project.json").read_bytes()
    if choose_directory:
        monkeypatch.setattr("straintrace_agent.gui.filedialog.askdirectory", lambda **_kwargs: str(b_dir))
        page.choose_project()
    else:
        page.project_var.set(str(b_dir))
    page.interval_var.set("0.25")
    page.plan = {"status": "ready"}
    for action in (page.save_project, page.make_plan, page.run_project,
                   page.verify_project, page.reprocess, lambda: page.open_artifact("folder")):
        action()
        assert not page._busy
        assert "项目目录已修改" in page.status_var.get()
        assert (a_dir / "project.json").read_bytes() == a_before
        assert (b_dir / "project.json").read_bytes() == b_before
    page.open_project(b_dir)
    wait(page)
    page.interval_var.set("0.25")
    page.save_project()
    wait(page)
    assert api.read_project_config(b_dir)["frame_interval_s"] == 0.25
    assert (a_dir / "project.json").read_bytes() == a_before


def test_reference_clear_blocks_drawing_and_manual_geometry_until_reloaded(page, tmp_path):
    project(page, tmp_path, reference=2)
    assert page._record["native_frame"] == 2
    page.reference_var.set("")
    before = page.analysis_roi_var.get()
    page._draw_start(SimpleNamespace(x=100, y=100))
    assert page._drag is None
    assert "先保存" in page.status_var.get()
    assert "先保存" in page.geometry_reference_var.get()
    page.name_var.set("New region")
    page.a_var.set("20,20,30,40")
    page.apply_item()
    assert not page.items
    assert "先保存" in page.status_var.get()
    page.new_item()
    page.analysis_roi_var.set("20,20,40,40")
    page.save_project()
    assert not page._busy
    assert "先还原计算区域" in page.status_var.get()
    page.analysis_roi_var.set(before)
    page.save_project()
    wait(page)
    assert page.config["reference_native_frame"] is None
    assert page._record["native_frame"] == 1
    assert "参考原生帧 1" in page.geometry_reference_var.get()
    page.name_var.set("On frame 1")
    page.a_var.set("20,20,30,40")
    page.apply_item()
    assert page.items[0]["value"]["name"] == "On frame 1"


def test_reference_change_during_drag_cannot_apply_old_image_coordinates(page, tmp_path):
    project(page, tmp_path, reference=2)
    page._drag = (20, 20)
    page.reference_var.set("")
    page._draw_end(SimpleNamespace(x=100, y=100))
    assert not page.analysis_roi_var.get()


def test_new_plan_uses_and_displays_resolved_automatic_reference(page, tmp_path):
    project(page, tmp_path, reference=2)
    page.make_plan()
    wait(page)
    assert page.plan["status"] == "ready", page.status_var.get()
    assert page.plan["reference_native_frame"] == page._record["native_frame"] == 2
    page.reference_var.set("")
    page._draw_start(SimpleNamespace(x=100, y=100))
    assert page._drag is None
    page.make_plan()
    wait(page)
    assert page.plan["status"] == "ready", page.status_var.get()
    assert page.plan["reference_native_frame"] == page._record["native_frame"] == 1
    assert "参考原生帧 1" in page.image_info_var.get()
    assert "参考原生帧 1" in page.geometry_reference_var.get()
    ox, oy = page._offset
    page._draw_start(SimpleNamespace(x=ox + 40 * page._scale, y=oy + 40 * page._scale))
    assert page._drag is not None


def test_saved_automatic_preview_skips_untextured_first_frame_and_matches_plan(page, tmp_path, monkeypatch):
    project(page, tmp_path, reference=2, blank_first=True)
    assert page._record["native_frame"] == 2
    page.reference_var.set("")
    loaded = []
    load_image = api._load_image

    def record_load(path):
        loaded.append((str(path), threading.get_ident()))
        return load_image(path)

    monkeypatch.setattr(api, "_load_image", record_load)
    page.save_project()
    wait(page)
    assert page.config["reference_native_frame"] is None
    assert page.plan is None
    assert page._record["native_frame"] == 2
    assert [path.rsplit("frame_", 1)[1] for path, _thread in loaded] == ["0001.png", "0002.png"]
    assert all(worker != threading.get_ident() for _path, worker in loaded)
    assert "参考原生帧 2" in page.image_info_var.get()
    assert "参考原生帧 2" in page.geometry_reference_var.get()
    ox, oy = page._offset
    page._draw_start(SimpleNamespace(x=ox + 40 * page._scale, y=oy + 40 * page._scale))
    assert page._drag == (40, 40)
    page.name_var.set("Defined on frame 2")
    page.a_var.set("20,20,30,40")
    page.apply_item()
    assert page.items[0]["value"]["name"] == "Defined on frame 2"
    page.make_plan()
    wait(page)
    assert page.plan is not None, page.status_var.get()
    assert page.plan["reference_native_frame"] == page._record["native_frame"] == 2
    assert "参考原生帧 2" in page.image_info_var.get()
    assert "参考原生帧 2" in page.geometry_reference_var.get()


@pytest.mark.parametrize("reference_source", ("config", "plan"))
def test_snapshot_never_substitutes_another_frame_for_missing_reference(page, tmp_path, reference_source):
    destination = project(page, tmp_path)
    config = api.read_project_config(destination)
    status = {"current_plan": None, "current_run": None}
    if reference_source == "config":
        config["reference_native_frame"] = 99
    else:
        (destination / "missing-reference-plan.json").write_text(
            json.dumps({"reference_native_frame": 99}), encoding="utf-8")
        status["current_plan"] = "missing-reference-plan.json"
    snapshot_api = SimpleNamespace(read_project_config=lambda _project: config,
                                   status=lambda _project: status,
                                   _automatic_reference=api._automatic_reference,
                                   _reference_record=api._reference_record,
                                   _load_image=api._load_image)
    with pytest.raises(api.AgentError) as error:
        _snapshot(snapshot_api, destination)
    assert error.value.code == "REFERENCE_FRAME_UNKNOWN"


def test_busy_selection_stays_aligned_with_editor(page, tmp_path):
    project(page, tmp_path, regions=(region("R1"), region("R2", 60)))
    page.geometry_tree.selection_set("0")
    page.update()
    assert page._selected == 0 and page.name_var.get() == "R1"
    started, release = threading.Event(), threading.Event()

    class SlowVerify:
        def verify(self, _project):
            started.set()
            release.wait(3)
            return {"valid": True}

    page._api = SlowVerify()
    try:
        page.verify_project()
        assert started.wait(1)
        page.geometry_tree.selection_set("1")
        page.update()
        assert page.geometry_tree.selection() == ("0",)
        assert page._selected == 0 and page.name_var.get() == "R1"
    finally:
        release.set()
        wait(page)
    page.name_var.set("Updated R1")
    page.apply_item()
    assert page.items[0]["value"]["name"] == "Updated R1"
    assert page.items[1]["value"]["name"] == "R2"


@pytest.mark.parametrize("discard", ("new", "select"))
def test_discarding_editor_changes_restores_ready_run_button(page, tmp_path, discard):
    project(page, tmp_path, regions=(region("R1"),))
    page.plan = {"status": "ready"}
    page._update_actions()
    assert not page.run_button.instate(["disabled"])
    page.name_var.set("Unapplied")
    assert page.run_button.instate(["disabled"])
    if discard == "new":
        page.new_item()
    else:
        page.geometry_tree.selection_set("0")
        page.update()
    assert not page._editor_dirty
    assert not page.run_button.instate(["disabled"])


def assert_in_view(widget, canvas):
    top, ancestor = 0, widget
    while ancestor.master is not canvas:
        top += ancestor.winfo_y()
        ancestor = ancestor.master
    top += canvas.coords(ancestor._canvas_window)[1] - canvas.canvasy(0)
    assert top >= 0
    assert top + widget.winfo_height() <= canvas.winfo_height()


def test_keyboard_traversal_reveals_controls_in_short_page(page):
    canvas = page.input_page._canvas
    assert canvas.winfo_height() > 20
    assert page.create_button.winfo_rooty() > canvas.winfo_rooty() + canvas.winfo_height()
    # Tk normally excludes withdrawn widgets from traversal. Custom takefocus
    # callbacks enable the real Tk traversal algorithm without mapping a window.
    source_entry = next(widget for widget in page.input_page.winfo_children()
                        if isinstance(widget, ttk.Entry) and str(widget.cget("textvariable")) == str(page.source_var))
    targets = (source_entry, page.preview_button, page.create_button, page.auto_button)
    for widget in targets:
        widget.configure(takefocus=lambda *_args: True)
    page.update()
    assert page.preview_button.tk_focusNext() == page.create_button
    page.tk.call("tk::TabToWindow", str(page.preview_button.tk_focusNext()))
    page.update()
    assert canvas.yview()[0] > 0
    assert_in_view(page.create_button, canvas)
    page.tk.call("tk::TabToWindow", str(page.create_button.tk_focusNext()))
    page.update()
    assert_in_view(page.auto_button, canvas)
    assert page.auto_button.tk_focusPrev() == page.create_button
    page.tk.call("tk::TabToWindow", str(page.auto_button.tk_focusPrev()))
    page.update()
    assert_in_view(page.create_button, canvas)
    assert page.create_button.tk_focusPrev() == page.preview_button
    page.tk.call("tk::TabToWindow", str(page.create_button.tk_focusPrev()))
    page.update()
    assert_in_view(page.preview_button, canvas)
    assert page.preview_button.tk_focusPrev() == source_entry
    canvas.yview_moveto(1)
    bottom = canvas.yview()[0]
    page.tk.call("tk::TabToWindow", str(page.preview_button.tk_focusPrev()))
    page.update()
    assert canvas.yview()[0] < bottom
    assert_in_view(source_entry, canvas)
    canvas.yview_moveto(1)
    source_entry.event_generate("<FocusIn>")
    page.update()
    assert_in_view(source_entry, canvas)
    assert page._scheduler.state() == "withdrawn"


def test_identity_only_frame_map_remains_usable_when_saving_and_planning(page, tmp_path):
    destination = project(page, tmp_path)
    mapping = tmp_path / "identity.csv"
    mapping.write_text("filename,native_frame\nframe_0001.png,1\nframe_0002.png,2\n", encoding="utf-8")
    api.update_project(destination, {"frame_map_csv": str(mapping)})
    page.open_project(destination)
    wait(page)
    before = (destination / "project.json").read_bytes()
    page.save_project()
    wait(page)
    assert "操作未完成" not in page.status_var.get(), page.status_var.get()
    assert (destination / "project.json").read_bytes() == before
    page.make_plan()
    wait(page)
    assert page.plan is not None, page.status_var.get()
    assert page.config["time_basis"] == "frame_index"
    assert all(row["time_s"] is None for row in _snapshot(api, destination)["inventory"])


def test_new_project_resolves_frame_map_time_without_reading_on_tk_thread(page, tmp_path, monkeypatch):
    project(page, tmp_path)
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("filename,native_frame\nframe_0001.png,1\nframe_0002.png,2\n", encoding="utf-8")
    workers = []
    read_map = api._explicit_frame_map

    def record_read(*args, **kwargs):
        workers.append(threading.get_ident())
        return read_map(*args, **kwargs)

    monkeypatch.setattr(api, "_explicit_frame_map", record_read)
    destination = tmp_path / "mapped-project"
    page.project_var.set(str(destination))
    page.map_var.set(str(mapping))
    page.create_project()
    wait(page)
    assert page.project_dir == destination, page.status_var.get()
    assert page.config["time_basis"] == "frame_index"
    # Replacing that map with actual recorded times must promote the time basis.
    timed = tmp_path / "timed.csv"
    timed.write_text("filename,native_frame,time_s\nframe_0001.png,1,0.5\nframe_0002.png,2,1.5\n", encoding="utf-8")
    page.map_var.set(str(timed))
    page.save_project()
    wait(page)
    assert page.config["time_basis"] == "physical_time", page.status_var.get()
    assert [row["time_s"] for row in _snapshot(api, destination)["inventory"]] == [0.5, 1.5]
    page.map_var.set(str(mapping))
    page.save_project()
    wait(page)
    assert page.config["time_basis"] == "frame_index", page.status_var.get()
    assert workers and all(worker != threading.get_ident() for worker in workers)


def test_typing_new_source_does_not_reuse_another_experiments_hidden_calibration(page, tmp_path):
    project(page, tmp_path)
    images = tmp_path / "images"
    calibration = {"mm_per_px": 0.1, "origin_px": [0, 0], "origin_mm": [0, 0], "axis_sign": [1, 1]}
    (images / "experiment.json").write_text(json.dumps({"schema": "straintrace.experiment/v1", "options": {"calibration": calibration}}), encoding="utf-8")
    page.discover()
    wait(page)
    assert page._init_options()["calibration"] == calibration
    second = tmp_path / "second-images"
    second.mkdir()
    for source in images.glob("*.png"):
        (second / source.name).write_bytes(source.read_bytes())
    page.source_var.set(str(second))
    page.project_var.set(str(tmp_path / "second-project"))
    page.create_project()
    wait(page)
    assert page.config["image_dir"] == str(second), page.status_var.get()
    assert page.config["calibration"] is None


@pytest.mark.parametrize("key,kind,value", [
    ("regions", "统计 ROI", region("Metadata ROI")),
    ("gauges", "虚拟标距", {"name": "Metadata gauge", "axis": "x", "a_point_px": [30, 30], "b_point_px": [90, 30]}),
    ("exclusions_px", "排除区", [20, 20, 10, 10]),
])
def test_discovery_does_not_restore_geometry_explicitly_deleted_in_editor(page, tmp_path, key, kind, value):
    project(page, tmp_path)
    images = tmp_path / "images"
    (images / "experiment.json").write_text(json.dumps({"schema": "straintrace.experiment/v1", "options": {key: [value]}}), encoding="utf-8")
    page.discover()
    wait(page)
    assert page.items == [{"kind": kind, "value": value}]
    page.geometry_tree.selection_set("0")
    page.update()
    page.delete_item()
    assert not page.items
    page.discover()
    wait(page)
    assert not page.items
    assert page._discovered_options[key] == []


def test_auto_analysis_requires_replacement_after_user_deletes_metadata_gauge(page, tmp_path):
    project(page, tmp_path)
    gauge = {"name": "Metadata gauge", "axis": "x", "a_point_px": [30, 30], "b_point_px": [90, 30]}
    (tmp_path / "images" / "experiment.json").write_text(json.dumps({
        "schema": "straintrace.experiment/v1", "options": {"gauges": [gauge]},
    }), encoding="utf-8")
    page.objective_var.set("虚拟标距应变")
    page.discover()
    wait(page)
    page.geometry_tree.selection_set("0")
    page.update()
    page.delete_item()
    destination = tmp_path / "automatic-project"
    page.project_var.set(str(destination))
    page.auto_analyze()
    wait(page)
    assert "需要补充输入" in page.status_var.get(), page.status_var.get()
    assert not (destination / "project.json").exists()
    assert '"field": "gauges"' in page.result_text.get("1.0", "end")


@pytest.mark.parametrize("pending", (False, True))
def test_open_project_during_manual_work_reports_actionable_error(page, tmp_path, pending):
    page.app = SimpleNamespace(is_processing=not pending, _completion_pending=pending)
    page.open_project(tmp_path / "project")
    assert not page._busy
    assert "手动分析仍在运行" in page.status_var.get()
