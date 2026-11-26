"""Whole-experiment coverage and navigation to the saved phase results."""

from dataclasses import dataclass
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .archive_quality import scan_archive_quality_batch
from .archive_quality_export import QUALITY_COVERAGE_LABELS, export_archive_quality
from .presentation import ui_error


@dataclass
class _BatchScan:
    stop: threading.Event
    current: tuple | None = None


@dataclass
class _Export:
    stop: threading.Event
    progress: tuple[int, int] | None = None


class ArchiveQualityOverview:
    def __init__(self, page, experiment, experiment_id, cameras, phases, submit, unavailable, capture_busy,
                 scratch, prepare, show_scope):
        self.page, self.experiment, self.experiment_id = page, experiment, experiment_id
        self.cameras, self.phases, self.scratch = cameras, phases, scratch
        self.submit, self.unavailable, self.capture_busy = submit, unavailable, capture_busy
        self.prepare, self.show_scope = prepare, show_scope
        self.started = False
        self.result = self._scan = self._poll_id = None
        self._export = self._export_poll = None
        self._visible = {}
        self.only_alerts = tk.BooleanVar(page, False)
        self.summary, self.status = (tk.StringVar(page) for _ in range(2))
        self._detail = ""
        self._build()
        self.summary.set("按相机和阶段汇总原图检查结果。")
        self.status.set("可以先抽查整场，再查看有提醒的一组。")
        self._controls()

    @property
    def running(self):
        return self._scan is not None

    @property
    def exporting(self):
        return self._export is not None

    def _build(self):
        box = ttk.Frame(self.page, padding=12)
        box.pack(fill=tk.BOTH, expand=True)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(2, weight=1)
        tools = ttk.Frame(box)
        tools.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.sample = ttk.Button(tools, text="抽查整场", style="Small.TButton", command=self.start)
        self.full = ttk.Button(tools, text="逐张查整场", style="Small.TButton", command=lambda: self.start(all_frames=True))
        self.stop_button = ttk.Button(tools, text="停止", style="Small.TButton", command=self._request_stop)
        for button in (self.sample, self.full, self.stop_button):
            button.pack(side=tk.LEFT, padx=(0, 6))
        ToolTip(self.sample, "按相机和阶段均匀选帧，包含各组首末帧。每组最多 96 帧。")
        ToolTip(self.full, "逐张核对各相机、各阶段已登记的全部主图像。大型实验所需时间取决于文件量。可随时停止。")
        ToolTip(self.stop_button, "保留已查的组，以及当前组已查的帧。汇总会标出未开始的组。")
        self.filter = ttk.Checkbutton(tools, text="只看提醒或失败", variable=self.only_alerts, command=self._table)
        self.filter.pack(side=tk.RIGHT)
        ToolTip(self.filter, "保留发现提醒或读取失败的相机与阶段。总覆盖数仍显示在上方。")
        summary = ttk.Label(box, textvariable=self.summary, style="CaptureMuted.TLabel", width=1)
        summary.grid(row=1, column=0, sticky="ew", pady=(0, 7))
        wrap_label_on_resize(summary)
        ToolTip(summary, "已查含读图失败。已核对指校验值匹配且已成功计算亮度的主图像。")

        table = ttk.Frame(box)
        table.grid(row=2, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(table, columns=("phase", "camera", "coverage", "verified", "alerts", "first", "state"),
                                 show="headings", selectmode="browse", height=7)
        for key, label, width in (
            ("phase", "阶段", 102), ("camera", "相机", 112), ("coverage", "已查 / 登记", 104),
            ("verified", "已核对", 68), ("alerts", "提醒", 52), ("first", "首条提醒帧", 106), ("state", "状态", 102),
        ):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, minwidth=width, stretch=key in ("phase", "camera"),
                             anchor="w" if key in ("phase", "camera", "state") else "center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table, orient=tk.VERTICAL, command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.tag_configure("alert", foreground="#ffca83")
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.tree.bind("<Double-1>", lambda _event: self._open_selected())
        self.tree.bind("<Return>", lambda _event: self._open_selected())
        ToolTip(self.tree, self._row_help)
        ToolTip(scroll, "滚动查看所有相机和阶段。")

        footer = ttk.Frame(box)
        footer.grid(row=3, column=0, sticky="ew", pady=(7, 0))
        footer.columnconfigure(0, weight=1)
        status = ttk.Label(footer, textvariable=self.status, justify=tk.LEFT, width=1)
        status.grid(row=0, column=0, sticky="ew")
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.status.get())
        self.copy_button = ttk.Button(footer, text="复制汇总", style="Small.TButton", command=self._copy)
        self.copy_button.grid(row=0, column=1, padx=(7, 0))
        ToolTip(self.copy_button, "复制实验标识、所有组的覆盖数、提醒数和首条提醒帧。")
        self.detail_button = ttk.Button(footer, text="查看该组原图", style="Small.TButton", command=self._open_selected)
        self.detail_button.grid(row=0, column=2, padx=(6, 0))
        ToolTip(self.detail_button, "打开该组已有结果。选择提醒可重新核对并打开对应原图。")
        self.export_button = ttk.Button(footer, text="导出报告…", style="Small.TButton", command=self._export_action)
        self.export_button.grid(row=0, column=3, padx=(6, 0))
        ToolTip(self.export_button, lambda _event: (
            "取消本次导出并清理尚未发布的文件。" if self.exporting else
            "保存 ZIP，包含覆盖汇总、每帧的画质和原图相对路径、台账校验值。" if self.result else
            "完成或停止整场检查后，即可导出已有结果。"))

    def visit(self):
        if not self.started:
            self.start()

    def start(self, *, all_frames=False):
        if self.running or self.exporting:
            return
        if reason := self.unavailable():
            self.status.set(reason)
            return
        self.started = True
        old = self.result
        self.prepare()
        self.result = None
        if old is not None:
            old.close()
        self._detail = ""
        self.summary.set("整场逐张检查" if all_frames else "整场抽查")
        task = self._scan = _BatchScan(threading.Event())
        self._table()

        def progress(number, total, camera, phase, done, selected):
            task.current = number, total, camera, phase, done, selected

        def finished(result):
            if not self.page.winfo_exists() or self._scan is not task:
                result.close()
                return
            self._end_poll()
            self.result = result
            totals = result.reports.values()
            count = sum(result.options["counts"].get(scope, 0) for scope in result.options["scopes"])
            mode = "整场逐张检查" if result.all_frames else "整场抽查"
            self.summary.set(
                f"{'已停止 · ' if result.stopped else ''}{mode} · 已查 {sum(x.checked for x in totals)}/{count} 帧"
                f" · 已核对 {sum(x.verified for x in totals)} · 提醒 {sum(x.flagged for x in totals)}"
                f"{f' · {len(result.errors)} 组检查失败' if result.errors else ''}"
            )
            self.only_alerts.set(False)
            self._table()

        def failed(exc):
            if self.page.winfo_exists() and self._scan is task:
                self._end_poll()
                self.summary.set("整场检查未完成")
                self._detail = ui_error(exc)
                self.status.set(self._detail.split("\n", 1)[0])
                self._controls()

        self.submit(lambda: scan_archive_quality_batch(
            self.experiment, self.experiment_id, all_frames=all_frames,
            cancelled=lambda: task.stop.is_set() or self.capture_busy(), progress=progress, scratch=self.scratch,
        ), finished, failed)
        self._poll(task)

    def _request_stop(self):
        if self._scan is not None:
            self._scan.stop.set()
            self._controls()

    def _poll(self, task):
        self._poll_id = None
        if self._scan is task:
            if task.stop.is_set():
                hint = "正在停止…"
            elif task.current is None:
                hint = "正在读取实验台账…"
            else:
                number, total, camera, phase, done, selected = task.current
                hint = f"第 {number}/{total} 组 · 阶段 {phase} · {self.cameras.get(camera, camera)} · {done}/{selected} 帧"
            self.status.set(hint)
            self._poll_id = self.page.after(125, lambda: self._poll(task))

    def _values(self, scope):
        camera, phase = scope
        group = self.result.coverage(scope)
        numbers = [group[name] if group[name] is not None else "—" for name in ("verified", "alerts", "first_alert_sequence")]
        return (self.phases.get(phase, phase), self.cameras.get(camera, camera),
                f"{group['checked']} / {group['available']}", *numbers, QUALITY_COVERAGE_LABELS[group["status"]])

    def _table(self):
        self.tree.delete(*self.tree.get_children())
        self._visible = {}
        flagged = []
        if self.result is not None:
            for index, scope in enumerate(self.result.options["scopes"]):
                report = self.result.reports.get(scope)
                alert = scope in self.result.errors or bool(report and report.flagged)
                if self.only_alerts.get() and not alert:
                    continue
                key = str(index)
                self._visible[key] = scope
                self.tree.insert("", tk.END, iid=key, values=self._values(scope), tags=("alert",) if alert else ())
                if alert:
                    flagged.append(key)
        if self._visible:
            first = flagged[0] if flagged else next(iter(self._visible))
            self.tree.selection_set(first)
            self.tree.focus(first)
        self._selection_changed()

    def _selected_scope(self):
        return self._visible.get(next(iter(self.tree.selection()), ""))

    def _scope_help(self, scope):
        if not scope or self.result is None:
            return "选择一个相机和阶段，查看这一组的检查范围与结果。"
        if scope in self.result.errors:
            return ui_error(self.result.errors[scope])
        camera, phase = scope
        report = self.result.reports.get(scope)
        values = self._values(scope)
        lines = [f"阶段 {phase} · {self.cameras.get(camera, camera)}", f"检查 {values[2]} 帧 · {values[-1]}"]
        if report is not None:
            text = "计划检查框" if camera in self.result.options["regions"] else "全图"
            if report.fallbacks:
                text += f" · {report.fallbacks} 帧改查全图"
            lines.append(text)
            if report.first_alert is not None:
                lines.append(f"首条提醒位于原帧 {report.first_alert}。")
        return "\n".join(lines)

    def _row_help(self, event):
        hovered = event and event.type in (tk.EventType.Enter, tk.EventType.Motion)
        scope = self._visible.get(self.tree.identify_row(event.y)) if hovered else self._selected_scope()
        return self._scope_help(scope)

    def _selection_changed(self, _event=None):
        scope = self._selected_scope()
        if self.result is not None and not self.exporting:
            self._detail = self._scope_help(scope)
            if scope in self.result.errors:
                self.status.set(self._detail.split("\n", 1)[0])
            elif scope and (report := self.result.reports.get(scope)) and report.fallbacks:
                self.status.set(f"该组有 {report.fallbacks} 帧按全图检查。可打开查看原因。")
            elif scope and self.result.reports.get(scope):
                self.status.set("可打开该组原图，查看每帧结果。")
            else:
                self.status.set("当前筛选暂无已查原图。")
        self._controls()

    def _controls(self):
        scope = self._selected_scope()
        for widget, enabled in (
            (self.sample, not (self.running or self.exporting)), (self.full, not (self.running or self.exporting)),
            (self.stop_button, self.running and not self._scan.stop.is_set()),
            (self.filter, self.result is not None), (self.copy_button, self.result is not None),
            (self.detail_button, self.result is not None and scope in self.result.reports
             and self.result.reports[scope].checked > 0),
            (self.export_button, self.result is not None and (self._export is None or not self._export.stop.is_set())),
        ):
            widget.state(["!disabled" if enabled else "disabled"])

    def _open_selected(self):
        scope = self._selected_scope()
        if self.result is not None and (report := self.result.reports.get(scope)) and report.checked:
            self.show_scope(*scope, report, scope[0] in self.result.options["regions"])

    def _copy(self):
        if self.result is not None:
            lines = [f"实验标识 {self.experiment_id}", self.summary.get()]
            for scope in self.result.options["scopes"]:
                phase, camera, coverage, verified, flags, first, state = self._values(scope)
                lines.append(f"{phase} · {camera} · 已查 {coverage} · 已核对 {verified} · 提醒 {flags} · 首条提醒帧 {first} · {state}")
            self.page.clipboard_clear()
            self.page.clipboard_append("\n".join(lines))
            if not self.exporting:
                self.status.set("已复制整场汇总。")

    def _export_action(self):
        if self.exporting:
            self.cancel_export()
            return
        batch = self.result
        if batch is None:
            return
        output = filedialog.asksaveasfilename(
            parent=self.page.winfo_toplevel(), title="导出采集画质报告", initialfile="采集画质报告.zip",
            defaultextension=".zip", filetypes=[("ZIP 压缩报告", "*.zip")], confirmoverwrite=False,
        )
        if not output or not self.page.winfo_exists():
            return
        batch.retain()
        request = self._export = _Export(threading.Event())
        self._detail = str(output)
        self.export_button.configure(text="取消导出")
        self._controls()

        def progress(done, total):
            request.progress = done, total

        def write():
            try:
                return export_archive_quality(batch, output, cancelled=request.stop.is_set, progress=progress)
            finally:
                batch.release()

        def finished(receipt):
            self._export_finished(request, f"已导出 {receipt['checked']} 帧的检查记录。", receipt["output"])

        def failed(exc):
            detail = ui_error(exc)
            self._export_finished(request, detail.split("\n", 1)[0], detail)

        try:
            # Finish removing the temporary ZIP if this window closes before the app exits.
            self.submit(write, finished, failed, daemon=False)
        except RuntimeError as exc:
            batch.release()
            failed(exc)
            return
        self._poll_export(request)

    def _poll_export(self, request):
        self._export_poll = None
        if self._export is request:
            if request.stop.is_set():
                self.status.set("正在取消导出…")
            elif request.progress is not None:
                done, total = request.progress
                self.status.set(f"正在导出逐帧记录 · {done}/{total}")
            else:
                self.status.set("正在准备导出报告…")
            self._export_poll = self.page.after(125, lambda: self._poll_export(request))

    def _export_finished(self, request, text, detail):
        if self._export is request:
            self._export = None
            if not self.page.winfo_exists():
                return
            if self._export_poll is not None:
                self.page.after_cancel(self._export_poll)
                self._export_poll = None
            self.export_button.configure(text="导出报告…")
            self.status.set(text)
            self._detail = detail
            self._controls()

    def cancel_export(self):
        if self._export is not None:
            self._export.stop.set()
            if self.page.winfo_exists():
                self._controls()

    def _end_poll(self):
        if self._scan is not None:
            self._scan.stop.set()
        self._scan = None
        if self._poll_id is not None:
            self.page.after_cancel(self._poll_id)
            self._poll_id = None

    def close(self):
        self._end_poll()
        if self._export is not None:
            self._export.stop.set()
            if self._export_poll is not None:
                self.page.after_cancel(self._export_poll)
                self._export_poll = None
        if self.result is not None:
            self.result.close()
            self.result = None
