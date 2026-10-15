from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import tkinter as tk
from pathlib import Path

import pytest

from straintrace_campaign.gui import CampaignPage


def _tk_root():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    return root


@pytest.fixture(scope="module")
def tk_root():
    root = _tk_root()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


class FakeWorkflow:
    def __init__(self, project):
        self.project = copy.deepcopy(project)
        self.calls = []
        self.ui_thread = threading.get_ident()

    def init_project(self, path, name, roots):
        self.calls.append(("init_project", threading.get_ident()))
        self.project.update({"name": name, "roots": roots, "project_dir": str(path)})
        return self.project

    def load_project(self, _path):
        self.calls.append(("load_project", threading.get_ident()))
        return copy.deepcopy(self.project)

    def scan(self, _path, root_alias, adapter="generic", pattern=None, exclude=()):
        self.calls.append(("scan", root_alias, adapter, pattern, threading.get_ident()))
        return self.project["acquisitions"]

    def update_acquisition(self, _path, acquisition_id, patch):
        self.calls.append(("update_acquisition", threading.get_ident()))
        acq = next(a for a in self.project["acquisitions"] if a["acquisition_id"] == acquisition_id)
        acq.update(copy.deepcopy(patch))
        return acq

    def propose_rois(self, _path, acquisition_id, free_region=None):
        self.calls.append(("propose_rois", threading.get_ident()))
        acq = next(a for a in self.project["acquisitions"] if a["acquisition_id"] == acquisition_id)
        image_path = acq["image_frames"][0]["path"]
        digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        proposal = {"free_region": [2, 2, 80, 45], "exclusions": [],
                    "groups": [{"name": "G01", "roi1": [10, 12, 8, 8], "roi2": [65, 12, 8, 8]}],
                    "reference_frame": 1, "reference_path": image_path,
                    "reference_sha256": digest, "proposal_hash": "fake-proposal-hash",
                    "diagnostics": {"texture": "synthetic"}}
        acq["roi"] = copy.deepcopy(proposal)
        return proposal

    def update_roi(self, _path, acquisition_id, proposal):
        self.calls.append(("update_roi", threading.get_ident()))
        acq = next(a for a in self.project["acquisitions"] if a["acquisition_id"] == acquisition_id)
        proposal = copy.deepcopy(proposal)
        proposal["proposal_hash"] = "updated-fake-hash"
        acq["roi"] = proposal
        acq["approval"] = None
        return proposal

    def approve_rois(self, _path, acquisition_id, reviewer, specimen_confirmed):
        self.calls.append(("approve_rois", reviewer, specimen_confirmed, threading.get_ident()))
        assert specimen_confirmed is True
        acq = next(a for a in self.project["acquisitions"] if a["acquisition_id"] == acquisition_id)
        acq["approval"] = {"reviewer": reviewer, "specimen_confirmed": True,
                            "proposal_hash": acq["roi"]["proposal_hash"]}
        return acq["approval"]

    def run(self, _path, acquisition_id=None, progress=None, cancel=None):
        self.calls.append(("run", acquisition_id, threading.get_ident()))
        assert cancel is not None and hasattr(cancel, "is_set")
        if progress:
            progress({"message": "tracking"})
        result = {"status": "COMPLETE", "rows_path": "results/rows.jsonl"}
        for acq in self.project["acquisitions"]:
            if acquisition_id is None or acq["acquisition_id"] == acquisition_id:
                acq["result"] = copy.deepcopy(result)
        return result

    def align(self, _path, acquisition_id, load_path, **kwargs):
        self.calls.append(("align", acquisition_id, load_path, kwargs, threading.get_ident()))
        result = {"status": "COMPLETE"}
        acq = next(a for a in self.project["acquisitions"] if a["acquisition_id"] == acquisition_id)
        acq["alignment"] = result
        return result

    def export(self, _path, output, publish_to=None):
        self.calls.append(("export", output))
        return {"status": "COMPLETE"}

    def verify(self, _path):
        self.calls.append(("verify",))
        return {"status": "PASS"}

    def import_keystone(self, _path, source):
        self.calls.append(("import_keystone", source))
        return []


@pytest.fixture
def campaign_case(tmp_path, monkeypatch, tk_root):
    root = tk_root
    monkeypatch.setattr("straintrace_campaign.gui.messagebox.showwarning", lambda *a, **k: None)
    monkeypatch.setattr("straintrace_campaign.gui.messagebox.showerror", lambda *a, **k: None)
    monkeypatch.setattr("straintrace_campaign.gui.messagebox.showinfo", lambda *a, **k: None)
    from PIL import Image

    image_paths = []
    for index in range(3):
        path = tmp_path / f"dic_{index + 1:06d}.tif"
        Image.new("RGB", (100, 60), color=(80 + index, 90, 100)).save(path)
        image_paths.append(str(path))
    acquisition = {
        "acquisition_id": "acq-synthetic-001", "name": "SYNTHETIC_001", "relative_path": "group/SYNTHETIC_001",
        "source_kind": "generic", "physical_specimen_id": None,
        "dimensions": {"area_mm2": None, "width_mm": None, "thickness_mm": None,
                       "status": "unknown", "source": None},
        "analysis_mode": "extensometer",
        "image_frames": [{"frame": i + 1, "native_frame": i + 1, "path": path} for i, path in enumerate(image_paths)],
        "measurement_frames": [{"frame": i + 1, "native_cbf_frame": i + 1, "scan_id": 20 + i,
                                 "scan_id_is_global_fio": True, "timestamp_unix_s": 1_700_000_100.0 + i,
                                 "estimated_timestamp_unix_s": 1_700_000_200.0 + i, "time_s": float(i)}
                                for i in range(3)],
        "roi": None, "approval": None, "result": None, "alignment": None, "metadata": {},
    }
    project = {"schema_version": 1, "kind": "straintrace_campaign", "project_id": "synthetic",
               "name": "Synthetic", "roots": {"data": str(tmp_path)},
               "acquisitions": [acquisition], "metadata": {}}
    wf = FakeWorkflow(project)
    page = CampaignPage(root, workflow=wf)
    page._accept_project(tmp_path / "campaign", project)
    page.acquisition_tree.selection_set(acquisition["acquisition_id"])
    page._on_select_acquisition()
    yield root, page, wf, tmp_path
    try:
        page.destroy()
    except tk.TclError:
        pass


def _wait(root, page, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and page._busy:
        root.update()
        time.sleep(0.01)
    root.update()
    assert not page._busy, "campaign operation did not finish before timeout"


def test_campaign_page_scans_edits_reviews_and_runs_on_worker(campaign_case):
    root, page, wf, tmp_path = campaign_case
    page.scan_root()
    _wait(root, page)
    assert page.acquisition_tree.exists("acq-synthetic-001")
    assert any(call[0] == "scan" and call[1] == "data" for call in wf.calls), (
        f"calls={wf.calls!r}; status={page.status_var.get()!r}; alias={page.root_alias_var.get()!r}; "
        f"roots={page.project.get('roots')!r}; project_dir={page.project_dir!r}")

    page.identity_vars["physical_specimen_id"].set("Coupon-7")
    page.identity_vars["width_mm"].set("2.0")
    page.identity_vars["thickness_mm"].set("1.5")
    page.identity_vars["area_mm2"].set("3.0")
    page.identity_vars["dimension_source"].set("synthetic logged dimensions")
    page.dimension_confirmed_var.set(True)
    page.save_identity()
    _wait(root, page)
    updated = page._selected_acquisition()
    assert updated["physical_specimen_id"] == "Coupon-7"
    assert updated["dimensions"]["status"] == "confirmed"
    assert updated["dimensions"]["area_mm2"] == pytest.approx(3.0)

    page.propose_rois()
    _wait(root, page)
    assert page._proposal["reference_sha256"]
    page.roi_entries[("group", 0, "roi1")][0].set("11")
    page.save_roi_edits()
    _wait(root, page)
    assert page._selected_acquisition()["approval"] is None
    assert page._selected_acquisition()["roi"]["groups"][0]["roi1"][0] == 11.0

    page.reviewer_var.set("Reviewer A")
    page.specimen_confirm_var.set(False)
    page.approve_rois()
    assert not any(call[0] == "approve_rois" for call in wf.calls)
    page.specimen_confirm_var.set(True)
    page.approve_rois()
    _wait(root, page)
    assert page._selected_acquisition()["approval"]["reviewer"] == "Reviewer A"

    page.run_selected()
    _wait(root, page)
    run_calls = [call for call in wf.calls if call[0] == "run"]
    assert run_calls and run_calls[-1][1] == "acq-synthetic-001"
    assert run_calls[-1][2] != wf.ui_thread
    assert page._selected_acquisition()["result"]["status"] == "COMPLETE"


def test_alignment_preview_is_read_only_and_uses_explicit_clock(campaign_case):
    root, page, wf, tmp_path = campaign_case
    load = tmp_path / "load.csv"
    load.write_text("timestamp_unix_s,force_N\n1700000099,9\n1700000101,11\n1700000103,13\n", encoding="utf-8")
    page.load_path_var.set(str(load))
    page.time_column_var.set("timestamp_unix_s")
    page.force_column_var.set("force_N")
    page.measurement_time_field_var.set("timestamp_unix_s")
    page.max_gap_var.set("2.5")
    page.preview_alignment()
    assert "尚未调用 align" in page.alignment_preview.get("1.0", tk.END)
    assert not any(call[0] == "align" for call in wf.calls)
    rows = page._preview_alignment_rows(page._selected_acquisition(), page._load_force_rows(), 2.5)
    assert [row[-1] for row in rows] == ["可对齐"] * 3

    page.measurement_time_field_var.set("estimated_timestamp_unix_s")
    assert str(page.align_button["state"]) == tk.DISABLED
    page.preview_alignment()
    rows = page._preview_alignment_rows(page._selected_acquisition(), page._load_force_rows(), 2.5)
    assert [row[-1] for row in rows] == ["超出载荷时间范围"] * 3
    assert not any(call[0] == "align" for call in wf.calls)

    load.write_text("timestamp_unix_s,force_N\n1700000099,9\n1700000100,\n1700000101,11\n1700000103,13\n",
                    encoding="utf-8")
    page.measurement_time_field_var.set("timestamp_unix_s")
    page.preview_alignment()
    rows = page._preview_alignment_rows(page._selected_acquisition(), page._load_force_rows(), 2.5)
    assert "载荷支撑点无效" in rows[0][-1]
    assert not any(call[0] == "align" for call in wf.calls)

    page.load_path_var.set(str(tmp_path / "relative-load.csv"))
    (tmp_path / "relative-load.csv").write_text(
        "time_s,clock_id,force_N\n0,Deben-clock,0\n1,Deben-clock,1\n2,Deben-clock,2\n",
        encoding="utf-8")
    for item in page._selected_acquisition()["measurement_frames"]:
        item["clock_id"] = "DIC-clock"
    page.time_column_var.set("time_s")
    page.measurement_time_field_var.set("time_s")
    page.preview_alignment()
    rows = page._preview_alignment_rows(page._selected_acquisition(), page._load_force_rows(), 2.5)
    assert all(row[-1] == "相对时钟 ID 缺失或不一致" for row in rows)
    assert not any(call[0] == "align" for call in wf.calls)

    page.load_path_var.set(str(load))
    page.time_column_var.set("timestamp_unix_s")
    page.measurement_time_field_var.set("timestamp_unix_s")
    load.write_text("timestamp_unix_s,force_N\n1700000099,9\n1700000101,11\n1700000103,13\n",
                    encoding="utf-8")
    page.preview_alignment()
    page.align()
    _wait(root, page)
    align_calls = [call for call in wf.calls if call[0] == "align"]
    assert len(align_calls) == 1, f"calls={wf.calls!r}; status={page.status_var.get()!r}; preview_key={page._alignment_preview_key!r}"
    assert align_calls[0][3]["measurement_time_field"] == "timestamp_unix_s"
    assert align_calls[0][-1] != wf.ui_thread


def test_roi_approval_requires_reviewer_and_explicit_specimen_confirmation(campaign_case):
    root, page, wf, _tmp_path = campaign_case
    page.propose_rois()
    _wait(root, page)
    page.reviewer_var.set(" ")
    page.specimen_confirm_var.set(True)
    page.approve_rois()
    assert not any(call[0] == "approve_rois" for call in wf.calls)
    page.reviewer_var.set("Reviewer B")
    page.specimen_confirm_var.set(False)
    page.approve_rois()
    assert not any(call[0] == "approve_rois" for call in wf.calls)


def test_legacy_workbench_hosts_campaign_preview_in_visual_window(tmp_path, monkeypatch, tk_root):
    import dic_virtual_extensometer_gui_v7_multi_roi_range as legacy_gui
    from PIL import Image

    monkeypatch.setenv(legacy_gui.RECENT_CONFIG_ENV_VAR, str(tmp_path / "recent.json"))
    host_root = tk.Toplevel(tk_root)
    app = legacy_gui.MultiROIGUI(host_root)
    image_path = tmp_path / "reference.png"
    Image.new("RGB", (160, 90), color=(120, 130, 140)).save(image_path)
    proposal = {"free_region": [5, 5, 140, 75], "exclusions": [],
                "groups": [{"name": "G01", "roi1": [20, 30, 10, 10], "roi2": [115, 30, 10, 10]}],
                "reference_frame": 1, "reference_path": str(image_path),
                "reference_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "proposal_hash": "visual-test", "diagnostics": {}}
    acq = {"acquisition_id": "visual-acq", "name": "VISUAL_SYNTHETIC", "relative_path": "VISUAL_SYNTHETIC",
           "dimensions": {}, "image_frames": [{"frame": 1, "native_frame": 1, "path": str(image_path)}],
           "measurement_frames": [], "roi": proposal, "approval": None, "result": None, "alignment": None}
    app.campaign_page._accept_project(tmp_path / "campaign", {"name": "visual", "roots": {}, "acquisitions": [acq]})
    app.campaign_page.acquisition_tree.selection_set("visual-acq")
    app.campaign_page._on_select_acquisition()
    app.campaign_page.render_roi_preview()
    app.campaign_page.show_campaign_image()
    tk_root.update()
    assert str(app.workspace_notebook.select()) == str(app.campaign_visual_page)
    assert len(app.campaign_preview_canvas.find_all()) >= 3  # reference image plus the free-region/gauge rectangles
    assert "reference.png" in app.campaign_preview_info_var.get()
    app.campaign_page._result_plot_points = [(1, 0.0), (2, 0.01)]
    app.campaign_page._result_plot_labels = ("frame", "engineering_strain")
    app.campaign_page._draw_result_curve()
    app.campaign_page.show_campaign_results()
    tk_root.update()
    assert str(app.workspace_notebook.select()) == str(app.campaign_result_visual_page)
    assert len(app.campaign_result_canvas.find_all()) >= 4
    try:
        for job in host_root.tk.call("after", "info"):
            host_root.after_cancel(job)
        host_root.destroy()
        app.visual_window.destroy()
    except tk.TclError:
        pass


def test_roi_edit_save_and_approval_use_real_workflow(tmp_path, monkeypatch, tk_root):
    import numpy as np
    from PIL import Image
    from ezdic_core import generate_synthetic_speckle
    from straintrace_campaign import workflow

    monkeypatch.setattr("straintrace_campaign.gui.messagebox.showwarning", lambda *a, **k: None)
    monkeypatch.setattr("straintrace_campaign.gui.messagebox.showerror", lambda *a, **k: None)
    data = tmp_path / "data"
    image_dir = data / "SYNTHETIC_COUPON" / "DIC"
    image_dir.mkdir(parents=True)
    reference = np.asarray(generate_synthetic_speckle(160, 320, seed=123), dtype=np.uint8)
    for frame in range(1, 4):
        Image.fromarray(reference).save(image_dir / f"dic_{frame:06d}.png")

    project_dir = tmp_path / "campaign"
    workflow.init_project(project_dir, "real ROI test", {"data": str(data)})
    discovered = workflow.scan(project_dir, "data", pattern={"acquisition_parent_levels": 1})
    assert len(discovered) == 1
    acquisition_id = discovered[0]["acquisition_id"]

    page = CampaignPage(tk_root, workflow=workflow)
    page._accept_project(project_dir, workflow.load_project(project_dir))
    page.acquisition_tree.selection_set(acquisition_id)
    page._on_select_acquisition()
    for var, value in zip(page.free_region_seed_vars, (20, 20, 280, 120)):
        var.set(str(value))
    page.propose_rois()
    _wait(tk_root, page)
    assert page._proposal and page._proposal["groups"], (
        f"ROI proposal failed: status={page.status_var.get()!r}; result={page.result_text.get('1.0', tk.END)!r}")

    original_hash = page._proposal["proposal_hash"]
    endpoint = page.roi_entries[("group", 0, "roi1")]
    endpoint[0].set(str(int(endpoint[0].get()) + 1))
    page.save_roi_edits()
    _wait(tk_root, page)
    edited = workflow.load_project(project_dir)["acquisitions"][0]
    assert edited["roi"]["groups"][0]["roi1"][0] == int(endpoint[0].get())
    assert edited["roi"]["proposal_hash"] != original_hash
    assert edited["approval"] is None

    page.reviewer_var.set("Synthetic Reviewer")
    page.specimen_confirm_var.set(True)
    page.approve_rois()
    _wait(tk_root, page)
    approved = workflow.load_project(project_dir)["acquisitions"][0]
    assert approved["approval"]["proposal_hash"] == approved["roi"]["proposal_hash"]
    assert approved["approval"]["reference_sha256"] == approved["roi"]["reference_sha256"]

    # Exercise the real campaign workflow/numerical backend through the Tk
    # action; an approved ROI must produce recoverable frame rows.
    page.run_selected()
    worker = page._worker
    assert worker is not None and worker is not threading.current_thread()
    _wait(tk_root, page, timeout=15.0)
    analyzed = workflow.load_project(project_dir)["acquisitions"][0]
    stage = analyzed["result"]
    assert stage and stage["status"] == "COMPLETED"
    rows_path = project_dir / stage["rows_path"]
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["frame"] for row in rows] == [1, 2, 3]
    assert all(row["native_dic_frame"] == row["frame"] for row in rows)
    assert all(len(row["gauge_results"]) == len(analyzed["roi"]["groups"]) for row in rows)
    assert rows[0]["engineering_strain"] == pytest.approx(0.0)
    page.destroy()
