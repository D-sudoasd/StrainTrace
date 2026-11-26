"""Inspect archived phase images and open their exact source frames in Review."""

from math import ceil
import threading
import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .archive_quality import scan_archive_quality
from .archive_quality_batch_gui import ArchiveQualityOverview
from .frame_search import FrameLocation, format_receipt_utc
from .presentation import ui_error
from .trial_quality import open_quality_frame
from .trial_quality_gui import _Scan, _flag_help, _flag_text
from .viewport import ImageViewport, repaint_after_resize


_PAGE = 40
_FALLBACK_LABELS = {"image_layout": "画幅变化", "sensor_window": "传感器范围变化"}


class ArchiveQualityWindow:
    def __init__(self, parent, experiment, experiment_id, options, cameras, phases, preferred_camera,
                 submit, unavailable, capture_busy, open_in_review, open_comparison, scratch):
        self.experiment, self.experiment_id = experiment, experiment_id
        self.options, self.submit, self.unavailable = options, submit, unavailable
        self.capture_busy = capture_busy
        self.open_in_review, self.open_comparison, self.scratch = open_in_review, open_comparison, scratch
        seen = options["counts"]
        camera = next((key for key in (preferred_camera, *cameras) if any((key, phase) in seen for phase in phases)),
                      preferred_camera if preferred_camera in cameras else next(iter(cameras)))
        phase = next((key for key in (*options["test_phases"], *phases) if (camera, key) in seen), next(iter(phases)))
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("检查原图画质")
        self.top.geometry("960x600")
        self.top.minsize(730, 485)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.top.bind("<Destroy>", self._destroyed, add="+")
        self.camera, self.phase = tk.StringVar(self.top, camera), tk.StringVar(self.top, phase)
        self.use_region = tk.BooleanVar(self.top, camera in options["regions"])
        self.only_alerts = tk.BooleanVar(self.top, False)
        self.summary, self.status, self.page_label = (tk.StringVar(self.top) for _ in range(3))
        self.report = self._scan = self._poll_id = self._opening = self._displayed = None
        self._borrowed_report = self._selecting_scope = False
        self._page = 0
        self._shown = {}
        self._detail = ""
        self._build(cameras, phases)
        self.camera.trace_add("write", self._camera_changed)
        self.phase.trace_add("write", lambda *_args: self._start())
        self._start()

    def _build(self, cameras, phases):
        self.tabs = ttk.Notebook(self.top)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self.detail_page = ttk.Frame(self.tabs)
        self.overview_page = ttk.Frame(self.tabs)
        self.tabs.add(self.detail_page, text="相机与阶段")
        self.tabs.add(self.overview_page, text="整场检查")
        form = ttk.Frame(self.detail_page, padding=12)
        form.pack(fill=tk.BOTH, expand=True)
        form.columnconfigure(0, weight=1)
        form.rowconfigure(3, weight=1)
        filters = ttk.Frame(form)
        filters.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for name, choices, variable in (("相机", cameras, self.camera), ("阶段", phases, self.phase)):
            ttk.Label(filters, text=name).pack(side=tk.LEFT, padx=(0, 5))
            menu = localized_combobox(filters, variable, choices, width=13)
            menu.pack(side=tk.LEFT, padx=(0, 10))
            ToolTip(menu, f"切换{name}后，均匀抽查该相机和阶段的已登记主图像。")
        self.scan_button = ttk.Button(filters, style="Small.TButton", command=self._scan_action)
        self.scan_button.pack(side=tk.RIGHT)
        ToolTip(self.scan_button, lambda _event: "停止扫描，保留已经完成的检查。" if self._scan else
                self._scan_block() or "逐张核对所选相机和阶段的全部主图像。运行时可停止。")

        scope = ttk.Frame(form)
        scope.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        self.region_box = ttk.Checkbutton(scope, text="使用计划检查框", variable=self.use_region, command=self._start)
        self.region_box.pack(side=tk.LEFT, padx=(0, 14))
        ToolTip(self.region_box, lambda _event: (
            "已归档画幅相符时检查框内。发生画幅或传感器范围变化的帧会改查全图并标明原因。"
            if self.camera.get() in self.options["regions"] else "此相机的原采集计划未保存检查框。"))
        self.flags_box = ttk.Checkbutton(scope, text="只看提醒", variable=self.only_alerts, command=lambda: self._table(0))
        self.flags_box.pack(side=tk.LEFT)
        ToolTip(self.flags_box, "查看含近黑、近饱和、画幅变化或 TIFF 读取问题的帧。")
        summary = ttk.Label(form, textvariable=self.summary, style="CaptureMuted.TLabel", width=1)
        summary.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        wrap_label_on_resize(summary)
        ToolTip(summary, "抽查按已登记帧号均匀分布，包含首末帧。已核对指通过 TIFF 校验并成功计算亮度的帧。")

        wall = ttk.Frame(form)
        wall.grid(row=3, column=0, sticky="nsew")
        wall.rowconfigure(0, weight=1)
        wall.columnconfigure(1, weight=1)
        table = ttk.Frame(wall, padding=(0, 0, 8, 0))
        table.grid(row=0, column=0, sticky="nsew")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        tree = self.tree = ttk.Treeview(table, columns=("frame", "black", "bright", "flag"), show="headings",
                                       selectmode="browse", height=7)
        for key, label, width in (("frame", "帧号", 52), ("black", "近黑", 62), ("bright", "近饱和", 76), ("flag", "提醒", 140)):
            tree.heading(key, text=label)
            tree.column(key, width=width, minwidth=width, stretch=key == "flag", anchor="w" if key == "flag" else "center")
        tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table, orient=tk.VERTICAL, command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)
        tree.tag_configure("flag", foreground="#ffca83")
        tree.bind("<<TreeviewSelect>>", self._open_selected)
        ToolTip(tree, self._row_help)
        ToolTip(scroll, "在本页滚动。页码旁的按钮可查看其他结果。")
        pages = ttk.Frame(table)
        pages.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.previous = ttk.Button(pages, text="上一页", style="Small.TButton", command=lambda: self._table(self._page - 1))
        self.previous.pack(side=tk.LEFT)
        self.next = ttk.Button(pages, text="下一页", style="Small.TButton", command=lambda: self._table(self._page + 1))
        self.next.pack(side=tk.RIGHT)
        ttk.Label(pages, textvariable=self.page_label).pack(side=tk.LEFT, padx=(8, 0))
        ToolTip(self.previous, "查看更早的帧。")
        ToolTip(self.next, "查看更晚的帧。")

        self.viewport = ImageViewport(wall, "archive_quality", "原图", lambda _camera: None, lambda *_args: None)
        self.viewport.grid(row=0, column=1, sticky="nsew")
        self.viewport.region_editable = False
        ToolTip(self.viewport.canvas, "选择左侧结果可核对并打开原始 TIFF。滚轮缩放，拖动平移。框线显示该帧实际统计的范围。")
        footer = ttk.Frame(form)
        footer.grid(row=4, column=0, sticky="ew", pady=(7, 0))
        footer.columnconfigure(0, weight=1)
        status = ttk.Label(footer, textvariable=self.status, justify=tk.LEFT, width=1)
        status.grid(row=0, column=0, sticky="ew")
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.status.get())
        self.review_button = ttk.Button(footer, text="在复查页打开", style="Small.TButton", command=self._open_review)
        self.review_button.grid(row=0, column=1, padx=(7, 0))
        ToolTip(self.review_button, lambda _event: "将这张主图像带到本实验的复查页，可继续浏览前后帧。"
                if self._displayed else "请在结果中选择能核验并打开的原图。")
        self.compare_button = ttk.Button(footer, text="对照当前预览…", style="Small.TButton", command=self._open_comparison)
        self.compare_button.grid(row=0, column=2, padx=(7, 0))
        ToolTip(self.compare_button, lambda _event: (
            "核对这张 TIFF 后，与同一台相机的当前原始画面对照。画幅和传感器范围相符时沿用本帧检查框。"
            if self._displayed else "请在结果中选择能核验并打开的原图。"))
        self.overview = ArchiveQualityOverview(
            self.overview_page, self.experiment, self.experiment_id, cameras, phases, self.submit,
            self.unavailable, self.capture_busy, self.scratch, self._prepare_batch, self._show_batch_scope,
        )
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        ToolTip(self.tabs, "逐组查看相机与阶段，或在“整场检查”中自动抽查各组并打开结果。")

    def _tab_changed(self, _event=None):
        if self.tabs.select() == str(self.overview_page):
            self.overview.visit()

    def _scan_block(self):
        if self.overview.running:
            return "请等整场检查完成，或停止后检查单组。"
        return self.unavailable()

    def _prepare_batch(self):
        if self._scan or self._borrowed_report:
            self._stop()
            self._clear_report()
            self.summary.set("")
            self._table(0, "可从“整场检查”打开本次检查的组。")

    def _show_batch_scope(self, camera, phase, result, use_region):
        self._stop()
        self._clear_report()
        self._selecting_scope = True
        try:
            self.camera.set(camera)
            self.phase.set(phase)
            self.use_region.set(use_region)
        finally:
            self._selecting_scope = False
        self._present_result(result, "计划检查框" if use_region else "全图", borrowed=True)
        self.tabs.select(self.detail_page)

    def _present_result(self, result, scope, *, borrowed=False):
        self.report = result
        self._borrowed_report = borrowed
        self._detail = ""
        mode = "逐张检查" if result.selected == result.available else "均匀抽查"
        tag = ("整场结果 · " if borrowed else "") + ("已停止 · " if result.stopped else "")
        summary = f"{tag}{scope} · {mode} {result.checked}/{result.available} 帧 · 已核对 {result.verified} · 提醒 {result.flagged}"
        if result.fallbacks:
            summary += f" · {result.fallbacks} 帧改查全图"
        self.summary.set(summary)
        self.only_alerts.set(bool(result.flagged))
        self._table(0)

    def _row_help(self, event):
        hovered = event and event.type in (tk.EventType.Enter, tk.EventType.Motion)
        key = self.tree.identify_row(event.y) if hovered else next(iter(self.tree.selection()), None)
        if (row := self._shown.get(key)) is None:
            return "选择一行查看该帧的原始 TIFF。近黑和近饱和的阈值与试采画质检查相同。"
        seconds = format_receipt_utc(row["received_utc_ns"], precision=3)
        lines = [f"原帧 {row['sequence']} · 主机收图 UTC {seconds}"]
        roi = row["roi"]
        lines.append(f"检查框 {roi[2]} × {roi[3]} px" if roi else "全图")
        lines.extend(_flag_help(flag) for flag in row["flags"])
        return "\n".join(lines)

    def _camera_changed(self, *_args):
        self.use_region.set(self.camera.get() in self.options["regions"])
        self._start()

    def _scan_action(self):
        if self._scan:
            self._scan.stop.set()
        else:
            self._start(all_frames=True)

    def _start(self, *, all_frames=False):
        if self._selecting_scope:
            return
        self._stop()
        self._detail = ""
        self._clear_report()
        if reason := self._scan_block():
            self.summary.set("")
            self._table(0, reason)
            return
        camera, phase, use_region = self.camera.get(), int(self.phase.get()), self.use_region.get()
        scope = "计划检查框" if use_region else "全图"
        self.summary.set(f"主图像 · {scope} · 等待读取台账")
        task = self._scan = _Scan(threading.Event())
        self._table(0, "正在读取阶段台账…")

        def progress(done, total):
            task.completed, task.total = done, total

        def finished(result):
            if not self.top.winfo_exists() or self._scan is not task:
                result.close()
                return
            self._stop()
            self._present_result(result, scope)

        def failed(exc):
            if self.top.winfo_exists() and self._scan is task:
                self._stop()
                self._detail = ui_error(exc)
                self.summary.set("检查未完成")
                self._table(0, self._detail.split("\n", 1)[0])

        self.submit(lambda: scan_archive_quality(
            self.experiment, self.experiment_id, camera, phase, use_region=use_region, all_frames=all_frames,
            cancelled=lambda: task.stop.is_set() or self.capture_busy(), progress=progress, scratch=self.scratch,
        ), finished, failed)
        self._progress(task)

    def _progress(self, task):
        self._poll_id = None
        if self._scan is task:
            self.status.set("正在停止…" if task.stop.is_set() else
                            f"正在核对原图 · {task.completed}/{task.total} 帧" if task.total else "正在读取阶段台账…")
            self._poll_id = self.top.after(125, lambda: self._progress(task))

    def _table(self, page, notice=""):
        self._opening = self._displayed = None
        report = self.report
        only = self.only_alerts.get()
        count = (report.flagged if only else report.checked) if report else 0
        pages = max(1, ceil(count / _PAGE))
        self._page = max(0, min(page, pages - 1))
        self.tree.delete(*self.tree.get_children())
        self._shown = {}
        rows = report.page(self._page * _PAGE, _PAGE, alerts_only=only) if report else ()
        for row in rows:
            key = str(row["ordinal"])
            self._shown[key] = row
            metrics = row["metrics"]
            black = f"{metrics['dark_fraction']:.1%}" if metrics else "—"
            bright = f"{metrics['near_saturation_fraction']:.1%}" if metrics else "—"
            flags = [_flag_text(flag) for flag in row["flags"] if not flag.startswith("ARCHIVE_QUALITY_")]
            if row["fallback"]:
                flags.append(_FALLBACK_LABELS[row["fallback"]])
            self.tree.insert("", tk.END, iid=key, values=(row["sequence"], black, bright, "、".join(dict.fromkeys(flags)) or "—"),
                             tags=("flag",) if flags else ())
        self.page_label.set(f"{self._page + 1} / {pages}")
        self.previous.state(["!disabled" if self._page > 0 else "disabled"])
        self.next.state(["!disabled" if self._page < pages - 1 else "disabled"])
        self.scan_button.configure(text="停止" if self._scan else "检查全部")
        self.flags_box.state(["!disabled" if report and report.checked else "disabled"])
        self.region_box.state(["!disabled" if self.camera.get() in self.options["regions"] else "disabled"])
        self._update_frame_actions()
        self.viewport.roi = None
        placeholder = notice or ("当前没有提醒。" if only else "当前阶段没有可显示的检查结果。")
        self.viewport.image(None, placeholder=placeholder)
        self.status.set(placeholder)
        if self._shown:
            first = next(iter(self._shown))
            self.tree.selection_set(first)
            self.tree.focus(first)

    def _open_selected(self, _event=None):
        selection = self.tree.selection()
        row = self._shown.get(selection[0]) if selection else None
        if row is None or row is self._opening or row is self._displayed:
            return
        self._displayed = None
        self._update_frame_actions()
        self._opening = row
        self._detail = ""
        camera = self.camera.get()
        self.viewport.roi = None
        self.viewport.image(None, placeholder="正在核对 TIFF…")
        self.status.set(f"正在读取原帧 {row['sequence']}…")

        def loaded(image):
            if self.top.winfo_exists() and self._opening is row:
                self._opening, self._displayed = None, row
                self.viewport.roi = tuple(row["roi"]) if row["roi"] else None
                self.viewport.image(image, frame_caption=f"原帧 {row['sequence']} · {image.pixel_format}")
                self.viewport.fit()
                self.status.set(f"原帧 {row['sequence']} · 已核对 TIFF 校验值")
                self._update_frame_actions()

        def failed(exc):
            if self.top.winfo_exists() and self._opening is row:
                self._opening = None
                self._detail = ui_error(exc)
                self.status.set(self._detail.split("\n", 1)[0])
                self.viewport.image(None, placeholder=self.status.get())

        self.submit(lambda: open_quality_frame(self.experiment, self.experiment_id, camera, row), loaded, failed)

    def _update_frame_actions(self):
        state = ["!disabled"] if self._displayed else ["disabled"]
        self.review_button.state(state)
        self.compare_button.state(state)

    def _open_comparison(self):
        row = self._displayed
        if row is not None:
            reason = self.open_comparison(self.experiment, self.experiment_id, self.camera.get(), int(self.phase.get()), row)
            if reason:
                self.status.set(reason)
            else:
                self.top.withdraw()

    def _open_review(self):
        if (row := self._displayed) is None:
            return
        camera = self.camera.get()
        location = FrameLocation(self.experiment_id, camera, row["sequence"], row["ordinal"], row["frame_id"], row["sha256"])

        def failed(exc):
            if self.top.winfo_exists():
                self._detail = ui_error(exc)
                self.status.set(self._detail.split("\n", 1)[0])

        self.open_in_review(self.experiment, location, self.top.withdraw, failed)

    def _stop(self):
        if self._scan:
            self._scan.stop.set()
        self._scan = None
        if self._poll_id is not None:
            self.top.after_cancel(self._poll_id)
            self._poll_id = None

    def _clear_report(self):
        if self.report is not None:
            if not self._borrowed_report:
                self.report.close()
            self.report = None
        self._borrowed_report = False

    def _destroyed(self, event):
        if event.widget is self.top:
            self._stop()
            self._clear_report()
            self.overview.close()
