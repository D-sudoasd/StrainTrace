"""Browse checked trial frames and inspect the same region over time."""

from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

from ezdic_gui.controls import localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .frame_search import format_receipt_utc
from .presentation import CAPTURE_ERROR_LABELS, TEXTURE_GUIDANCE, ui_error
from .trial_quality import QUALITY_ALERT_FRACTION, open_trial_image, rescan_trial_images
from .viewport import ImageViewport, repaint_after_resize


_PAGE = 40
_ALERT_LABELS = {
    "TRIAL_IMAGE_NEAR_BLACK": "近黑",
    "TRIAL_IMAGE_NEAR_SATURATION": "近饱和",
    "TRIAL_IMAGE_REGION_CHANGED": "画幅变化",
}


@dataclass
class _Scan:
    stop: threading.Event
    completed: int = 0
    total: int = 0


def _flag_text(flag):
    return _ALERT_LABELS.get(flag) or ("纹理需复核" if flag in TEXTURE_GUIDANCE else "原图需复核")


def _flag_help(flag):
    threshold = f"{QUALITY_ALERT_FRACTION:.0%}"
    if flag == "TRIAL_IMAGE_NEAR_BLACK":
        return f"至少 {threshold} 像素的亮度落在满量程的最低 1%。可核对照明及检查框位置。"
    if flag == "TRIAL_IMAGE_NEAR_SATURATION":
        return f"至少 {threshold} 像素的亮度或某个颜色通道进入最高 1%。可核对曝光和反光。"
    return TEXTURE_GUIDANCE.get(flag) or CAPTURE_ERROR_LABELS.get(flag) or flag


class TrialImageWindow:
    def __init__(self, parent, report, names, active_camera, submit, capture_busy, on_full_scan=None, *,
                 on_use_region=None, plan_region_block=None):
        self.report = report
        self.output = Path(report["output"])
        self.experiment_id = report["experiment_id"]
        self.submit, self.capture_busy = submit, capture_busy
        self.on_full_scan = on_full_scan
        self.on_use_region, self.plan_region_block = on_use_region, plan_region_block
        self.automatic = {key: row["image_check"] for key, row in report["cameras"].items() if "image_check" in row}
        self._verified = report["observed"] and report["verification"].get("file_hashes_checked", False)
        default = next((key for key, data in self.automatic.items()
                        if (self._default_plan_check(data) or data)["flagged"]),
                       active_camera if active_camera in self.automatic else next(iter(self.automatic)))
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("检查试采原图")
        self.top.geometry("970x615")
        self.top.minsize(745, 500)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.camera = tk.StringVar(self.top, default)
        self.only_flags = tk.BooleanVar(self.top, bool(self.automatic[default]["flagged"]))
        self.summary = tk.StringVar(self.top)
        self.status = tk.StringVar(self.top)
        self.page_text = tk.StringVar(self.top)
        self._page = 0
        self._shown = {}
        self._result = self.automatic[default]
        self._displayed = None
        self._opening = None
        self._scan = None
        self._poll_id = None
        self._roi = self._region_layout = None
        self._region_frame = self._adopting = None
        self._using_plan_region = False
        self._detail = ""
        self._build({key: names.get(key, key) for key in self.automatic})
        self.camera.trace_add("write", self._camera_changed)
        self.top.bind("<Destroy>", self._destroyed, add="+")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self._reset_region(prefer_plan=True)

    def _default_plan_check(self, full):
        pair = full.get("region_check")
        return pair["check"] if self._verified and pair and pair["fallback"] is None else None

    def _build(self, names):
        form = ttk.Frame(self.top, padding=(14, 12))
        form.pack(fill=tk.BOTH, expand=True)
        form.columnconfigure(0, weight=1)
        form.rowconfigure(2, weight=1)
        tools = ttk.Frame(form)
        tools.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(tools, text="相机").pack(side=tk.LEFT, padx=(0, 6))
        menu = localized_combobox(tools, self.camera, names, width=14)
        menu.pack(side=tk.LEFT, padx=(0, 12))
        ToolTip(menu, "按相机查看原图检查结果。换相机后显示该相机的计划检查框或全图。")
        flags = ttk.Checkbutton(tools, text="只看提醒", variable=self.only_flags, command=self._filter)
        flags.pack(side=tk.LEFT, padx=(0, 6))
        ToolTip(flags, "只列出达到近黑、近饱和比例提醒，或需要复核纹理和原图的帧。")
        self.clear_button = ttk.Button(tools, text="清除检查框", command=self._reset_region, style="Small.TButton")
        self.clear_button.pack(side=tk.RIGHT)
        ToolTip(self.clear_button, lambda _event: ("用“查看全图结果”切换检查范围。" if self._using_plan_region else
                "当前检查范围是全图。" if self._roi is None else "清除临时框选，回到最近一次全图检查结果。"))
        self.scan_button = ttk.Button(tools, text="检查全部", command=self._scan_button, style="Small.TButton")
        self.scan_button.pack(side=tk.RIGHT, padx=(0, 6))
        ToolTip(self.scan_button, self._scan_help)

        summary_row = ttk.Frame(form)
        summary_row.grid(row=1, column=0, sticky="ew", pady=(0, 7))
        summary_row.columnconfigure(0, weight=1)
        summary = ttk.Label(summary_row, textvariable=self.summary, justify=tk.LEFT, style="CaptureMuted.TLabel")
        summary.grid(row=0, column=0, sticky="ew")
        wrap_label_on_resize(summary)
        ToolTip(summary, "抽查按原帧序号均匀取样，包含首末帧。核验张数是可读取并计算指标的原图数。")
        self.scope_button = ttk.Button(summary_row, style="Small.TButton", command=self._toggle_scope)
        ToolTip(self.scope_button, lambda _event: (
            "查看同一批已核验原图的全图检查结果。" if self._using_plan_region else
            "查看这些原图在试采时保存的计划框内的亮度结果。打开原图时会标出该框。"))
        body = ttk.Frame(form)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)
        table_panel = ttk.Frame(body, padding=(0, 0, 6, 0))
        table_panel.columnconfigure(0, weight=1)
        table_panel.rowconfigure(0, weight=1)
        table_panel.grid(row=0, column=0, sticky="nsew")
        tree = self.tree = ttk.Treeview(table_panel, columns=("sequence", "black", "bright", "flags"),
                                       show="headings", selectmode="browse", height=8)
        total_width = 0
        for key, title, width in (("sequence", "序号", 53), ("black", "近黑", 65),
                                  ("bright", "近饱和", 82), ("flags", "提醒", 108)):
            tree.heading(key, text=title)
            minimum = tree.tk.call("font", "measure", "TkDefaultFont", title)
            actual = max(width, int(minimum) + 18)
            tree.column(key, width=actual, minwidth=int(minimum) + 12,
                        anchor="w" if key == "flags" else "center", stretch=key == "flags")
            total_width += actual
        tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table_panel, orient=tk.VERTICAL, command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        body.columnconfigure(0, weight=2, minsize=total_width + scroll.winfo_reqwidth() + 8)
        tree.configure(yscrollcommand=scroll.set)
        tree.tag_configure("flag", foreground="#ffca83")
        tree.bind("<<TreeviewSelect>>", self._open_selected)
        ToolTip(tree, self._table_help, title="检查结果")
        ToolTip(scroll, "滚动查看这一页。用下方按钮翻页。")
        pages = ttk.Frame(table_panel)
        pages.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        self.previous = ttk.Button(pages, text="上一页", command=lambda: self._step(-1), style="Small.TButton")
        self.previous.pack(side=tk.LEFT)
        ToolTip(self.previous, "查看更早的已检查帧。")
        self.next = ttk.Button(pages, text="下一页", command=lambda: self._step(1), style="Small.TButton")
        self.next.pack(side=tk.RIGHT)
        ToolTip(self.next, "查看更晚的已检查帧。")
        ttk.Label(pages, textvariable=self.page_text).pack(side=tk.LEFT, padx=(10, 0))
        self.viewport = ImageViewport(body, "trial", "原图", lambda _key: None, self._region_drawn)
        self.viewport.grid(row=0, column=1, sticky="nsew")
        ToolTip(self.viewport.canvas, "单击左侧一行打开对应原图。滚轮缩放，拖动平移。"
                "按住 Shift 拖出检查框，可在同一位置复查试采图像。")

        footer = ttk.Frame(form)
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)
        status = ttk.Label(footer, textvariable=self.status, justify=tk.LEFT)
        status.grid(row=0, column=0, sticky="ew")
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.status.get(), title="原图检查详情")
        if self.on_use_region is not None:
            self.use_region_button = ttk.Button(footer, text="用于采集抽查", style="Small.TButton",
                                                command=self._use_region)
            self.use_region_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
            ToolTip(self.use_region_button, lambda _event: self._use_region_block() or
                    "把显示的检查框保存到当前计划的这台相机。可离线设置，随后在采前检查中核对新范围。")

    def _use_region_block(self):
        if self._adopting is not None:
            return "正在重新核对原图和当前计划。"
        if self._scan is not None:
            return "等待原图检查结束后再保存检查框。"
        if self._roi is None:
            hint = "在原图上按住 Shift 拖动框住试样。"
            if self.scope_button.winfo_manager():
                hint += "也可先点击“查看检查框结果”。"
            return hint
        if self._displayed is None or self.viewport.roi != self._roi:
            return "请打开一张能显示检查框的已核验原图。"
        return self.plan_region_block(self.camera.get())

    def sync_plan_region_action(self):
        if self.on_use_region is not None:
            blocked = bool(self._use_region_block())
            if blocked != self.use_region_button.instate(["disabled"]):
                self.use_region_button.state(["disabled" if blocked else "!disabled"])

    def _use_region(self):
        if reason := self._use_region_block():
            self.status.set(reason)
            return
        camera = self.camera.get()
        item = self._region_frame or self._displayed
        request = self._adopting = object()
        self._detail = ""
        self.status.set("正在重新核对原图和当前计划…")
        self.sync_plan_region_action()

        def active():
            return self._adopting is request and self.top.winfo_exists()

        def finished():
            if active():
                self._adopting = None
                self.status.set(f"已将原帧 {item['sequence']} 的检查框保存到当前采集计划。")
                self.sync_plan_region_action()

        def failed(exc):
            if active():
                self._adopting = None
                self._detail = ui_error(exc)
                self.status.set(self._detail.split("\n", 1)[0])
                self.sync_plan_region_action()

        self.on_use_region(self.report, camera, item, self._roi, active, finished, failed)

    def _scan_help(self, _event=None):
        if self._scan is not None:
            return "停止本次检查，回到最近一次全图检查结果。"
        if self._result["mode"] == "all":
            return "已遍历当前相机的全部已登记主图像。可换相机或重新框选。"
        return "依次核对并检查本相机的全部已登记主图像，使用当前检查框。运行时可停止。"

    def _table_help(self, event):
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            key = self.tree.identify_row(event.y)
        else:
            key = next(iter(self.tree.selection()), None)
        if item := self._shown.get(key):
            stamp = format_receipt_utc(item["received_utc_ns"])
            lines = [f"原帧 {item['sequence']} · 主机收图 UTC {stamp}"]
            if metrics := item["metrics"]:
                p5, p50, p95 = metrics["brightness_fraction"]
                lines.append(f"亮度 P5 / P50 / P95  {p5:.1%} / {p50:.1%} / {p95:.1%}，按有效位深的满量程归一。")
                if metrics["focus_response"] is not None:
                    lines.append(f"对焦响应 {metrics['focus_response']:.2f} · 可在同一检查框内作比较。")
                if metrics["texture_measured"]:
                    lines.append(f"检查框内对比度 P95−P5 为 {metrics['contrast_8bit']:.1f} 个 8 位等效灰度级。")
                if metrics["texture_error"]:
                    lines.append("检查框较大，可缩小至 100 万像素以内查看纹理指标。")
            return "\n".join((*lines, *(_flag_help(flag) for flag in item["flags"])))
        return "近黑按亮度统计，近饱和按原始像素或任一颜色通道统计。选择一行可查看原图。"

    def _filter(self):
        self._page = 0
        self._table()

    def _step(self, delta):
        self._page += delta
        self._table()

    def _controls(self, pages):
        scanning = self._scan is not None
        self.viewport.region_editable = not scanning
        self.previous.state(["!disabled" if self._page > 0 and not scanning else "disabled"])
        self.next.state(["!disabled" if self._page < pages - 1 and not scanning else "disabled"])
        self.clear_button.state(["!disabled" if self._roi is not None and not self._using_plan_region else "disabled"])
        candidate = self.automatic[self.camera.get()].get("region_check")
        if candidate and candidate["check"] is not None:
            label = "查看全图结果" if self._using_plan_region else (
                "查看试采计划框" if self._roi is not None else "查看检查框结果"
            )
            self.scope_button.configure(text=label)
            self.scope_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
            self.scope_button.state(["disabled" if scanning else "!disabled"])
        else:
            self.scope_button.grid_remove()
        self.scan_button.configure(text="停止" if scanning else "检查全部")
        enabled = scanning or self._result["mode"] != "all"
        self.scan_button.state(["!disabled" if enabled else "disabled"])
        self.sync_plan_region_action()

    def _table(self, preferred=None):
        data = self._result
        rows = [item for item in data["rows"] if not self.only_flags.get() or item["flags"]]
        pages = max(1, (len(rows) + _PAGE - 1) // _PAGE)
        if preferred is not None:
            self._page = next((index // _PAGE for index, item in enumerate(rows) if item["sequence"] == preferred), 0)
        self._page = min(max(0, self._page), pages - 1)
        self._shown = {}
        self.tree.delete(*self.tree.get_children())
        if self._scan is None:
            start = self._page * _PAGE
            for item in rows[start:start + _PAGE]:
                key = str(item["sequence"])
                self._shown[key] = item
                metrics = item["metrics"]
                dark = f"{metrics['dark_fraction']:.1%}" if metrics else "—"
                bright = f"{metrics['near_saturation_fraction']:.1%}" if metrics else "—"
                flags = "、".join(dict.fromkeys(_flag_text(flag) for flag in item["flags"])) or "—"
                self.tree.insert("", tk.END, iid=key, values=(key, dark, bright, flags),
                                 tags=("flag",) if item["flags"] else ())
        mode = "全查" if data["mode"] == "all" else "均匀抽查"
        roi = data["roi"]
        scope = f"{'计划' if self._using_plan_region else ''}检查框 {roi[2]} × {roi[3]} px" if roi else "全图"
        self.summary.set(f"主图像 · {scope} · {mode} {data['selected']}/{data['available']} 张"
                         f" · 已核验 {data['checked']} 张 · 提醒 {data['flagged']} 张")
        self.page_text.set(f"{self._page + 1} / {pages}")
        self._controls(pages)
        if self._scan is None:
            if self._shown:
                first = str(preferred) if str(preferred) in self._shown else next(iter(self._shown))
                self.tree.selection_set(first)
            else:
                self._opening = self._displayed = self._adopting = None
                message = "当前筛选暂无图像。" if data["selected"] else "这台相机尚无已登记图像。"
                self.viewport.image(None, placeholder=message, empty_caption="暂无原图")
                self.status.set(message)
                self.sync_plan_region_action()

    def _open_selected(self, _event=None):
        selection = self.tree.selection()
        item = self._shown.get(selection[0]) if selection and self._scan is None else None
        if item is None or item is self._displayed or item is self._opening:
            return
        self._displayed = self._adopting = None
        self._opening = item
        camera = self.camera.get()
        self.viewport.roi = None
        self.viewport.image(None, placeholder="正在核验并读取原图…", empty_caption="读取中")
        self._detail = ""
        self.status.set(f"正在读取原帧 {item['sequence']}…")
        self.sync_plan_region_action()

        def opened(image):
            if self.top.winfo_exists() and self._opening is item and self.camera.get() == camera:
                self._opening = None
                self._displayed = item
                self.viewport.roi = self._roi if self._using_plan_region or item["layout"] == self._region_layout else None
                self.viewport.image(image, frame_caption=f"原帧 {item['sequence']} · {image.pixel_format}")
                self.viewport.fit()
                hint = " · 按 Shift 拖动可框住试样" if self.on_use_region is not None and self._roi is None else ""
                self.status.set(f"原帧 {item['sequence']} · 已核对 TIFF 校验值{hint}")
                self.sync_plan_region_action()

        def failure(exc):
            if self.top.winfo_exists() and self._opening is item and self.camera.get() == camera:
                self._opening = None
                self._detail = ui_error(exc)
                note = self._detail.split("\n", 1)[0]
                self.viewport.image(None, placeholder=note, empty_caption="原图待复核")
                self.status.set(note)
                self.sync_plan_region_action()

        self.submit(lambda: open_trial_image(self.output, self.experiment_id, camera, item), opened, failure)

    def _region_drawn(self, _camera, roi):
        if self._displayed is not None and roi:
            self._using_plan_region = False
            self._roi, self._region_layout = tuple(roi), self._displayed["layout"]
            self._region_frame = self._displayed
            self._start_scan(all_frames=self._result["mode"] == "all")

    def _scan_button(self):
        if self._scan:
            self._scan.stop.set()
            self.scan_button.state(["disabled"])
        else:
            self._start_scan(all_frames=True)

    def _start_scan(self, *, all_frames):
        if self.capture_busy():
            self._reset_region()
            self.status.set("请等当前采集写完，再检查试采图像。")
            return
        preferred = self._displayed["sequence"] if self._displayed is not None else None
        self._cancel()
        camera, roi, layout = self.camera.get(), self._roi, self._region_layout
        use_plan = self._using_plan_region
        if use_plan:
            roi = layout = None
        pair = self.automatic[camera].get("region_check")
        region = pair["region"] if pair is not None and roi is None else None
        scan = self._scan = _Scan(threading.Event())
        self._page = 0
        self._table()

        def progress(done, total):
            scan.completed, scan.total = done, total

        def finished(result):
            if self.top.winfo_exists() and self._scan is scan:
                self._scan = None
                self._result, self._displayed = result, None
                if result["roi"] is None and result["mode"] == "all":
                    self.automatic[camera] = result
                    if self.on_full_scan is not None:
                        self.on_full_scan(self.report, camera, result)
                    if use_plan:
                        self._set_saved_scope(result.get("region_check", {}).get("check"))
                self.only_flags.set(bool(self._result["flagged"]))
                self._table(preferred)

        def failed(exc):
            if self.top.winfo_exists() and self._scan is scan:
                self._reset_region()
                self._detail = ui_error(exc)
                self.status.set(self._detail.split("\n", 1)[0])

        self.submit(lambda: rescan_trial_images(
            self.output, self.experiment_id, camera, roi=roi, region_layout=layout, all_frames=all_frames,
            cancelled=lambda: scan.stop.is_set() or self.capture_busy(), progress=progress,
            quality_region=region,
        ), finished, failed)
        self._poll(scan)

    def _poll(self, scan):
        self._poll_id = None
        if self._scan is scan:
            self.status.set("正在停止检查…" if scan.stop.is_set() else
                            f"正在检查原图 · {scan.completed}/{scan.total} 张" if scan.total else "正在读取试采台账…")
            self._poll_id = self.top.after(125, lambda: self._poll(scan))

    def _camera_changed(self, *_args):
        self._reset_region(prefer_plan=True)

    def _set_saved_scope(self, regional):
        self._using_plan_region = regional is not None
        self._result = regional if self._using_plan_region else self.automatic[self.camera.get()]
        self._roi = tuple(regional["roi"]) if self._using_plan_region else None
        self._region_layout = self._region_frame = self.viewport.roi = None
        self.viewport.draw()

    def _toggle_scope(self):
        selected = self.tree.selection()
        sequence = int(selected[0]) if selected else None
        self._reset_region(prefer_plan=not self._using_plan_region, explicit=True, preferred=sequence)

    def _reset_region(self, *, prefer_plan=False, explicit=False, preferred=None):
        self._cancel()
        self._detail = ""
        full = self.automatic[self.camera.get()]
        regional = None
        if prefer_plan:
            regional = full.get("region_check", {}).get("check") if explicit else self._default_plan_check(full)
        self._set_saved_scope(regional)
        show_flags = self._result["flagged"] and (preferred is None or any(
            row["sequence"] == preferred and row["flags"] for row in self._result["rows"]
        ))
        self.only_flags.set(bool(show_flags))
        self._page = 0
        self._table(preferred)

    def _cancel(self):
        if self._poll_id is not None:
            self.top.after_cancel(self._poll_id)
            self._poll_id = None
        if self._scan:
            self._scan.stop.set()
        self._scan = self._opening = self._displayed = self._adopting = None

    def _destroyed(self, event):
        if event.widget is self.top:
            self._cancel()
