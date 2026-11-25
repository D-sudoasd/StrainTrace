"""Review the recorded associations and exact TIFFs from a capture trigger."""

from __future__ import annotations

import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import ttk

import cv2

from ezdic_gui.tooltips import ToolTip

from .capture_review import check_review_identity
from .cameras.base import HOST_NEXT_RECEIPT
from .config import CaptureError
from .hardware_counters import frame_hardware_counters
from .image_inputs import recorded_image
from .presentation import ROLE_LABELS, ui_error
from .storage import ExperimentReader
from .viewport import ImageViewport, repaint_after_resize


_GROUP_STATUS = {
    "associated": "齐全", "incomplete": "缺帧", "pending_or_incomplete": "未齐",
    "ambiguous": "有冲突",
}
_TIMING = {
    "unverified": "曝光同步待验证",
    "host_receipt_only": "含主机选帧 · 曝光同步待核实",
    "hardware_association_clock_unverified": "相机曝光时钟待验证",
}
_METHOD = {HOST_NEXT_RECEIPT: "按主机收图选帧", "software_frame_start": "相机软件触发"}


def _receipt_span(rows, group):
    if group["status"] != "associated" or len(rows) < 2:
        return None
    stamps = [row.get("timestamps", {}).get("received_monotonic") or {} for row in rows]
    clock = stamps[0].get("clock_id")
    if not clock or any(stamp.get("clock_id") != clock or stamp.get("unit") != "ns"
                        or type(stamp.get("value")) is not int for stamp in stamps):
        return None
    values = [stamp["value"] for stamp in stamps]
    return max(values) - min(values)


def read_trigger_review(root, event_id, *, expected_experiment_id=None):
    """Decode verified images only for cameras with exactly one associated frame."""
    with ExperimentReader(root) as reader:
        if expected_experiment_id is not None:
            check_review_identity(reader.experiment, expected_experiment_id)
        group, rows = reader.trigger_group(event_id)
        return _read_group_images(reader, group, rows, {})


def _read_group_images(reader, group, rows, errors):
    images, frames = {}, defaultdict(list)
    for row in rows:
        frames[row["camera_id"]].append(row)
    for camera in group["expected_cameras"]:
        records = frames[camera]
        if camera in errors or len(records) != 1:
            continue
        row, = records
        try:
            images[camera] = recorded_image(reader.image_path(row), row, verify=True)
        except (CaptureError, OSError, ValueError, cv2.error) as exc:
            errors[camera] = ui_error(exc)
    span = None if group["basis"] == "anchor_preview" or ("references" in group and errors) else _receipt_span(rows, group)
    return group, dict(frames), images, errors, span


def counter_review_snapshot(experiment_id, phase, source_id, number, cameras, refs, counter_feature, *, status, preview=False):
    return {"experiment_id": experiment_id, "phase": phase, "external_source_id": source_id,
            "external_event_number": number, "event_id": f"selected:{phase}:{number}", "expected_cameras": tuple(cameras),
            "counter_feature": counter_feature, "references": refs, "status": status,
            "unexpected_cameras": [], "camera_methods": {}, "synchronization": "hardware_association_clock_unverified",
            "basis": "anchor_preview" if preview else "post_capture_counter_anchors"}


def read_counter_review(root, snapshot):
    rows, errors = [], {}
    with ExperimentReader(root) as reader:
        if reader.experiment["experiment_id"] != snapshot["experiment_id"]:
            raise CaptureError("PULSE_BINDING_SCOPE", "当前实验与所选原图组不一致。")
        for ref in snapshot["references"]:
            camera, sequence = ref["camera_id"], ref["sequence"]
            try:
                row = reader.frame_at_sequence(camera, sequence)
                if row is None or row["phase"] != snapshot["phase"] or row.get("trigger_event_id") is not None:
                    raise CaptureError("PULSE_BINDING_FRAME", f"{camera} 的序号 {sequence} 无法在本阶段读取。")
                if "frame_id" in ref and (
                    any(row.get(key) != ref[key] for key in ("frame_id", "path", "sha256")) or
                    frame_hardware_counters(row.get("backend_metadata")).get(snapshot["counter_feature"]) != ref["counter"]
                ):
                    raise CaptureError("PULSE_BINDING_FRAME", f"{camera} 的序号 {sequence} 已变化，请重新核对。")
                rows.append(row)
            except (CaptureError, KeyError, ValueError) as exc:
                errors[camera] = ui_error(exc)
        return _read_group_images(reader, snapshot, rows, errors)


def _trigger_label(group):
    number = group["external_event_number"]
    if number is not None:
        return f"脉冲 {number}"
    return f"软件 · 台账 {group['ordinal']}"


class TriggerReview:
    PAGE_SIZE = 24

    def __init__(self, parent, experiment, camera_names, submit, *, event_id=None, timing_note="", snapshots=(),
                 expected_experiment_id=None):
        self.experiment = Path(experiment)
        self.experiment_id = expected_experiment_id
        self.names = dict(camera_names)
        self.submit = submit
        self._snapshots = {group["event_id"]: group for group in snapshots}
        if event_id is None and self._snapshots:
            event_id = next(iter(self._snapshots))
        title = "计数器原图复查" if snapshots else "触发组复查"
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title(f"{title} · {self.experiment.name}")
        self.top.geometry("1150x690")
        self.top.minsize(835, 490)
        repaint_after_resize(self.top)
        self._page = []
        self._page_busy = self._image_busy = False
        self._older = self._newer = False
        self._wanted = None
        self._timing_note = event_id, timing_note
        self._views = {}
        self._view_details = {}
        self._group_rows = {}
        self._group_note = ""
        self._rows_of_views = 0

        layout = ttk.Frame(self.top, padding=12)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(1, weight=1)
        layout.rowconfigure(1, weight=1)
        header = ttk.Frame(layout)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        ttk.Label(header, text=title, style="CaptureSection.TLabel").pack(side=tk.LEFT)
        self.problems = tk.BooleanVar(self.top)
        self.problem_filter = ttk.Checkbutton(header, text="仅看缺帧或冲突", variable=self.problems,
                                              command=lambda: self._request_page())
        self.problem_filter.pack(side=tk.RIGHT)
        ToolTip(self.problem_filter, "筛选台账中缺少相机图像、同一相机出现多帧或记录到额外相机的触发组。")
        if snapshots:
            self.problem_filter.pack_forget()
        self.refresh = ttk.Button(header, text="刷新", command=self._refresh, style="Small.TButton")
        self.refresh.pack(side=tk.RIGHT, padx=(0, 10))
        ToolTip(self.refresh, "重新校验本组原图。" if snapshots else "重读台账及当前触发组。正在采集时，可检查新提交的图像。")

        sidebar = ttk.Frame(layout)
        sidebar.grid(row=1, column=0, sticky="ns", padx=(0, 10))
        sidebar.rowconfigure(0, weight=1)
        self.groups = ttk.Treeview(sidebar, columns=("phase", "trigger", "status"), show="headings", selectmode="browse")
        self.groups.grid(row=0, column=0, sticky="nsew")
        for column, text, width in (("phase", "阶段", 48),
                                    ("trigger", "所选脉冲" if snapshots else "触发记录（台账顺序）", 182 if snapshots else 143),
                                    ("status", "状态", 64)):
            self.groups.heading(column, text=text)
            self.groups.column(column, width=width, stretch=False, anchor="center" if column != "trigger" else "w")
        scroll = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=self.groups.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.groups.configure(yscrollcommand=scroll.set)
        self.groups.bind("<<TreeviewSelect>>", self._select_group)
        ToolTip(self.groups, "选择一组即可查看对应的原图。可用方向键切换记录。", title="触发记录")
        paging = ttk.Frame(sidebar)
        paging.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.newer = ttk.Button(paging, text="较新一页", style="Small.TButton", command=lambda: self._request_page("newer"))
        self.older = ttk.Button(paging, text="较早一页", style="Small.TButton", command=lambda: self._request_page("older"))
        self.latest = ttk.Button(paging, text="最新一页", style="Small.TButton", command=lambda: self._request_page())
        for widget, text in ((self.newer, "查看本页之后登记的触发组。"), (self.older, "查看本页之前登记的触发组。"),
                             (self.latest, "跳到台账中最新登记的触发组。")):
            widget.pack(side=tk.LEFT, padx=(0, 5))
            ToolTip(widget, text)
        if snapshots:
            paging.grid_remove()

        view_panel = ttk.Frame(layout)
        view_panel.grid(row=1, column=1, sticky="nsew")
        view_panel.columnconfigure(0, weight=1)
        view_panel.rowconfigure(1, weight=1)
        self.summary = tk.StringVar(self.top, "正在读取触发记录…")
        banner = ttk.Label(view_panel, textvariable=self.summary, justify=tk.LEFT, style="CaptureMuted.TLabel")
        banner.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 7))
        banner.bind("<Configure>", lambda event: banner.configure(wraplength=max(1, event.width - 4)))
        ToolTip(banner, lambda _event: self._group_note or self.summary.get(), title="配对依据")
        self.canvas = tk.Canvas(view_panel, highlightthickness=0, background=ttk.Style(parent).lookup("TFrame", "background"))
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.image_scroll = ttk.Scrollbar(view_panel, orient=tk.VERTICAL, command=self.canvas.yview)
        self.image_scroll.grid(row=1, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.image_scroll.set)
        self.wall = ttk.Frame(self.canvas)
        self.wall_window = self.canvas.create_window((0, 0), window=self.wall, anchor="nw")
        self.canvas.bind("<Configure>", self._resize_wall)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self._request_page(through_event=event_id)

    def _alive(self):
        return self.top.winfo_exists()

    def reopen(self, event_id, *, timing_note=""):
        if event_id and self._page_busy:
            return False
        self.top.lift()
        self.groups.focus_set()
        if event_id:
            if self.groups.selection() != (event_id,):
                self._clear_group_page()
            self._timing_note = event_id, timing_note
            self.problems.set(False)
            self._request_page(through_event=event_id)
        return True

    def _refresh(self):
        current = self.groups.selection()
        self._request_page(through_event=current[0] if current else None)

    def _controls(self):
        for widget, enabled in ((self.newer, self._newer), (self.older, self._older), (self.latest, True),
                                (self.refresh, True), (self.problem_filter, True)):
            widget.state(["!disabled" if enabled and not self._page_busy else "disabled"])

    def _request_page(self, direction="latest", *, through_event=None):
        if self._page_busy:
            return
        filters = {"limit": self.PAGE_SIZE + 1, "problems_only": self.problems.get(), "through_event": through_event}
        if direction == "older":
            filters["before"] = self._page[-1]["ordinal"]
        elif direction == "newer":
            filters.update(after=self._page[0]["ordinal"], newest_first=False)
        current_summary, current_note = self.summary.get(), self._group_note
        self._page_busy = True
        self._controls()
        self._group_note = ""
        self.summary.set("正在读取触发记录…")

        def read():
            if self._snapshots:
                return list(self._snapshots.values())
            with ExperimentReader(self.experiment) as reader:
                if self.experiment_id is not None:
                    check_review_identity(reader.experiment, self.experiment_id)
                return reader.trigger_group_page(**filters)

        def done(rows):
            if not self._alive():
                return
            self._page_busy = False
            if not rows and direction != "latest":
                if direction == "older":
                    self._older = False
                else:
                    self._newer = False
                self.summary.set(current_summary)
                self._group_note = current_note
                self._controls()
                return
            more = len(rows) > self.PAGE_SIZE
            self._older, self._newer = more, bool(through_event) or direction == "older"
            page = rows[:self.PAGE_SIZE]
            if direction == "newer":
                page.reverse()
                self._older, self._newer = True, more
            self._page = page
            self._wanted = None
            self._group_rows = {item["event_id"]: item for item in page}
            self.groups.delete(*self.groups.get_children())
            for item in page:
                self.groups.insert("", "end", iid=item["event_id"],
                                   values=(item["phase"], _trigger_label(item), _GROUP_STATUS[item["status"]]))
            if page:
                selected = through_event if through_event in self._group_rows else page[0]["event_id"]
                self.groups.selection_set(selected)
                self.groups.focus(selected)
                self._select_group()
            else:
                self._arrange_views(())
                self.summary.set("未找到缺帧或冲突组。" if self.problems.get() else
                                 "当前实验还没有可复查的触发组。已保存原图可在主窗口按相机查看。")
            self._controls()

        def failed(exc):
            if self._alive():
                self._page_busy = False
                if through_event is not None:
                    self._clear_group_page()
                self.summary.set(ui_error(exc, details=False))
                self._group_note = ui_error(exc)
                self._controls()

        self.submit(read, done, failed)

    def _clear_group_page(self):
        self._wanted = None
        self._page = []
        self._group_rows.clear()
        self._older = self._newer = False
        self.groups.delete(*self.groups.get_children())
        self._arrange_views(())

    def _resize_wall(self, _event=None):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        available = max(height, self._rows_of_views * 235)
        self.canvas.itemconfigure(self.wall_window, width=max(1, width), height=max(1, available))
        self.canvas.configure(scrollregion=(0, 0, width, available))

    def _arrange_views(self, cameras):
        if tuple(self._views) == tuple(cameras):
            return
        self.wall.destroy()
        self.wall = ttk.Frame(self.canvas)
        self.canvas.itemconfigure(self.wall_window, window=self.wall)
        self._views.clear()
        self._view_details.clear()
        columns = min(2, len(cameras))
        self._rows_of_views = (len(cameras) + columns - 1) // columns if columns else 0
        for index, camera in enumerate(cameras):
            row, column = divmod(index, columns)
            self.wall.columnconfigure(column, weight=1, uniform="cameras")
            self.wall.rowconfigure(row, weight=1, uniform="camera_rows")
            view = ImageViewport(self.wall, camera, self.names.get(camera, camera), lambda _key: None, lambda *_args: None)
            view.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
            view.region_editable = False
            ToolTip(view.canvas, lambda _event, key=camera: self._view_details.get(key, "等待读取图像。"), title="本组原图")
            self._views[camera] = view
        self.canvas.yview_moveto(0)
        self._resize_wall()

    def _select_group(self, _event=None):
        if self._page_busy or not (selection := self.groups.selection()):
            return
        identity = selection[0]
        if self._wanted is not None and self._wanted[1] == identity:
            return
        ToolTip.dismiss_window(self.top)
        self._wanted = (object(), identity)
        group = self._group_rows[identity]
        self._arrange_views(group["expected_cameras"])
        for camera, view in self._views.items():
            view.set_title(self.names.get(camera, camera))
            view.image(None, placeholder="正在核对本组原图…", empty_caption="正在核对")
        self._view_details.clear()
        self._group_note = ""
        self.summary.set(f"阶段 {group['phase']} · {_trigger_label(group)} · 正在核对本组原图…")
        self._read_selected()

    def _read_selected(self):
        if self._image_busy or self._wanted is None:
            return
        request = self._wanted
        self._image_busy = True

        def finish(result):
            if not self._alive():
                return
            self._image_busy = False
            if self._wanted != request:
                self._read_selected()
            elif isinstance(result, Exception):
                self.summary.set(ui_error(result, details=False))
                self._group_note = ui_error(result)
            else:
                self._show_group(*result)

        snapshot = self._snapshots.get(request[1])
        self.submit(lambda: read_counter_review(self.experiment, snapshot) if snapshot else
                    read_trigger_review(self.experiment, request[1], expected_experiment_id=self.experiment_id), finish, finish)

    def _show_group(self, group, frames, images, errors, span_ns):
        self._arrange_views(group["expected_cameras"])
        phase = f"阶段 {group['phase']} · {_trigger_label(group)}"
        basis = {"explicit_shared_counter_anchor": "外部脉冲编号", "common_capture_request": "同一软件触发请求",
                 "post_capture_counter_anchors": "两次已确认脉冲之间的采后补配", "anchor_preview": "待确认的锚点原图"}.get(
                     group["basis"], group["basis"])
        timing = _TIMING.get(group["synchronization"], group["synchronization"])
        shown = f"{len(images)}/{len(group['expected_cameras'])} 张可查看"
        status = _GROUP_STATUS[group["status"]] + (" · 原图校验失败" if errors else "")
        receipt = f" · 主机接收跨度 {span_ns / 1e6:.3f} ms" if span_ns is not None else ""
        summary = f"{phase} · {status} · {shown}\n{basis} · {timing}{receipt}"
        annotated_event, note = self._timing_note
        if note and annotated_event == group["event_id"]:
            summary += f"\n{note}"
        self.summary.set(summary)
        self._group_note = self.summary.get() if self._snapshots else f"{self.summary.get()}\n台账事件 {group['event_id']}"
        if source := group["external_source_id"]:
            self._group_note += f"\n外部信号源 {source}"
        if group["unexpected_cameras"]:
            names = (self.names.get(key, key) for key in group["unexpected_cameras"])
            self._group_note += f"\n额外相机 {'、'.join(names)}"
        if span_ns is not None:
            self._group_note += "\n接收跨度取同一主机时钟上最早和最晚的收图时刻。"
        for camera, view in self._views.items():
            records = frames[camera]
            name = self.names.get(camera, camera)
            if image := images.get(camera):
                row, = records
                view.set_title(f"{name} · 序号 {row['sequence']}")
                view.image(image)
                detail = f"{ROLE_LABELS.get(row['role'], row['role'])} · {row['pixel_format']}\n{row['frame_id']}"
                if feature := group.get("counter_feature"):
                    value = frame_hardware_counters(row.get("backend_metadata")).get(feature, "未记录")
                    detail += f"\n{feature} · {value}"
                method = group["camera_methods"].get(camera)
                if method:
                    detail += f"\n{_METHOD.get(method, method)}"
                    if (method == HOST_NEXT_RECEIPT and (row.get("backend_metadata") or {}).get("frame_selection")
                            == "first_host_receipt_after_request"):
                        detail += "\n本帧是请求后主机收到的第一张。"
                part = image.metadata.get("gentl_image_part")
                if isinstance(part, dict) and part.get("selector"):
                    detail += f"\nGenTL 部件 {part['selector']}"
                if additional := row.get("auxiliary_images"):
                    detail += f"\n此帧还有 {len(additional)} 张原图，可在复查页的“本帧图像”中切换。"
                detail += "\n滚轮缩放，拖动平移，双击适应窗口。"
            else:
                if camera in errors:
                    detail, caption = errors[camera], "原图校验失败"
                elif records:
                    sequences = "、".join(str(row["sequence"]) for row in records)
                    detail = f"该相机有 {len(records)} 张关联图像，无法唯一配对。\n候选图像序号 {sequences}"
                    caption = "关联冲突"
                elif group["status"] == "pending_or_incomplete":
                    detail, caption = "等待此相机提交图像。", "尚未保存"
                else:
                    detail, caption = "该触发组缺少此相机图像。", "缺帧"
                view.image(None, placeholder=detail.split("\n", 1)[0], empty_caption=caption)
            self._view_details[camera] = detail
