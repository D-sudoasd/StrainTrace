"""Open and inspect saved live curves without restarting analysis."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.controls import localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .config import CaptureError
from .gui import BG
from .live_curve import CURVE_MODES
from .live_curve_gui import CURVE_HELP, CurveFrameWindow, LiveCurve, point_description
from .live_history import read_history_index
from .live_results import live_results_directory
from .presentation import LOAD_SUPPORT_LABELS, TRACKING_LABELS, ui_error
from .viewport import repaint_after_resize


class LiveHistoryWindow:
    def __init__(self, parent, experiment, *, expected_id=None, preferred_record=None):
        self.experiment = Path(experiment).resolve()
        self.expected_id = expected_id
        self.directory = live_results_directory(self.experiment)
        self.top = tk.Toplevel(parent, bg=BG)
        self.top.title("StrainTrace · 历史曲线")
        self.top.geometry("910x620")
        self.top.minsize(730, 500)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.top.bind("<Destroy>", self._destroy, add="+")
        self.index = None
        self.page = 0
        self.selected = None
        self._reader_busy = False
        self._poll = None
        self._error = None
        self._viewer = None
        self._replay_window = None
        self._export_window = None
        self._pending_replay = None
        self._path = None
        self.mode = tk.StringVar(self.top, "strain-time")
        self.path_label = tk.StringVar(self.top, "未选择逐帧记录")
        self.summary = tk.StringVar(self.top)
        self.page_label = tk.StringVar(self.top, "— / —")
        self.axis = tk.StringVar(self.top)
        self.status = tk.StringVar(self.top, "选择本实验的逐帧记录后可查看曲线。")
        self.sequence = tk.StringVar(self.top)
        self.selection = tk.StringVar(self.top)
        self._build()
        self.mode.trace_add("write", self._mode_changed)
        self._controls()
        latest = preferred_record or max(self.directory.glob("*.jsonl"), default=None)
        if latest is not None:
            self.open_record(Path(latest))

    def _build(self):
        panel = ttk.Frame(self.top, padding=12)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(3, weight=1)
        header = ttk.Frame(panel)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        filename = ttk.Label(header, textvariable=self.path_label, style="CaptureSection.TLabel", width=1)
        filename.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ToolTip(filename, lambda _event: str(self._path) if self._path else "选择实验内已有的逐帧记录。")
        self.pick_button = ttk.Button(header, text="选择记录…", command=self.choose_record)
        self.pick_button.grid(row=0, column=1, padx=(0, 6))
        ToolTip(self.pick_button, "每次开始跟踪会保存一份新记录。可选同一实验中较早的那一份。")
        self.refresh_button = ttk.Button(header, text="重新读取", command=self.refresh, style="Small.TButton")
        self.refresh_button.grid(row=0, column=2, padx=(0, 6))
        ToolTip(self.refresh_button, "读取记录当前已写入的部分。正在跟踪时，可用它更新本窗口的结果。")
        self.replay_button = ttk.Button(header, text="按序重算…", command=self.show_replay, style="Small.TButton")
        self.replay_button.grid(row=0, column=3)
        ToolTip(
            self.replay_button,
            lambda _event: "沿用这份记录的参考图和测量区，逐张分析已保存的实验图。实时跳过的帧也会纳入。"
            if self.index else "先选择本实验的逐帧记录，再沿用其中的测量设置。",
            title="按序重算",
        )

        details = ttk.Frame(panel)
        details.grid(row=1, column=0, sticky="ew", pady=(7, 6))
        details.columnconfigure(0, weight=1)
        overview = ttk.Label(details, textvariable=self.summary, width=1, style="CaptureMuted.TLabel", justify=tk.LEFT)
        overview.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        wrap_label_on_resize(overview)
        ToolTip(overview, lambda _event: self.summary.get(), title="记录范围")
        self.export_button = ttk.Button(details, text="导出表格…", command=self.show_export, style="Small.TButton")
        self.export_button.grid(row=0, column=1, sticky="e")
        ToolTip(self.export_button, lambda _event: "把当前记录保存为 CSV。可选整份记录、当前段或指定帧号。"
                if self.index else "先选择本实验的逐帧记录，再导出表格。", title="导出曲线表格")

        navigation = ttk.Frame(panel)
        navigation.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.mode_picker = localized_combobox(navigation, self.mode, CURVE_MODES, width=14)
        self.mode_picker.pack(side=tk.LEFT, padx=(0, 9))
        ChoiceToolTip(self.mode_picker, "选择历史曲线的测量量。", CURVE_HELP, title="曲线视图")
        self.previous = ttk.Button(navigation, text="上一段", command=lambda: self.turn(-1), style="Small.TButton")
        self.previous.pack(side=tk.LEFT)
        ToolTip(self.previous, "查看按采集顺序排列的上一段结果。聚焦曲线后也可按 Page Up。")
        page = ttk.Label(navigation, textvariable=self.page_label, anchor=tk.CENTER)
        page.pack(side=tk.LEFT, padx=9)
        ToolTip(page, "每段最多显示 500 张。页码覆盖整份记录。")
        self.next = ttk.Button(navigation, text="下一段", command=lambda: self.turn(1), style="Small.TButton")
        self.next.pack(side=tk.LEFT)
        ToolTip(self.next, "查看后一段结果。聚焦曲线后也可按 Page Down。")
        self.seek_button = ttk.Button(navigation, text="定位", command=self.seek, style="Small.TButton")
        self.seek_button.pack(side=tk.RIGHT)
        ToolTip(self.seek_button, "按本轮实验的图像序号找记录。实时跳过的图像可在采集复查中查看。")
        self.seek_entry = ttk.Entry(navigation, textvariable=self.sequence, width=8)
        self.seek_entry.pack(side=tk.RIGHT, padx=5)
        self.seek_entry.bind("<Return>", lambda _event: self.seek())
        ToolTip(self.seek_entry, "输入原图的帧序号，例如 1204。")
        ttk.Label(navigation, text="帧序号").pack(side=tk.RIGHT)

        self.plot = LiveCurve(panel, self._plot_selection, self.open_image, history=True)
        self.plot.grid(row=3, column=0, sticky="nsew")
        self.plot.bind("<Prior>", lambda _event: self.turn(-1), add="+")
        self.plot.bind("<Next>", lambda _event: self.turn(1), add="+")

        actions = ttk.Frame(panel)
        actions.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(actions, textvariable=self.axis, style="CaptureMuted.TLabel").pack(side=tk.LEFT)
        self.inspect_button = ttk.Button(actions, text="核对原图", command=self.open_image, style="Small.TButton")
        self.inspect_button.pack(side=tk.RIGHT)
        ToolTip(self.inspect_button, lambda _event: point_description(self.selected) if self.selected
                else "点击曲线点、缺测短线，或按帧序号定位。")
        self.issue_button = ttk.Button(actions, text="下一处缺测", command=self.next_issue, style="Small.TButton")
        self.issue_button.pack(side=tk.RIGHT, padx=6)
        ToolTip(self.issue_button, "从当前选中帧向后找当前视图的缺测。全场质量视图同时查找质量未通过的帧。")
        selected = ttk.Label(actions, textvariable=self.selection, width=1, anchor="e", style="CaptureMuted.TLabel")
        selected.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        ToolTip(selected, lambda _event: point_description(self.selected) if self.selected else None)
        status = ttk.Label(panel, textvariable=self.status, style="CaptureMuted.TLabel", width=1, justify=tk.LEFT)
        status.grid(row=5, column=0, sticky="ew", pady=(7, 0))
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: ui_error(self._error) if self._error else self.status.get())
        self._mode_changed()

    def _set_selection(self, point, *, drawn=True):
        self.selected = point
        title = f"帧 {point.frame.sequence}" if point is not None else ""
        detail = ""
        if point is not None:
            mode = self.mode.get()
            if mode == "force-time" and point.force_N is None:
                detail = LOAD_SUPPORT_LABELS.get(point.force_status, "载荷缺测")
            elif mode == "strain-time" and point.strain_pct is None:
                detail = TRACKING_LABELS.get(point.strain_status, "应变缺测")
            elif mode == "field-time" and point.needs_review(mode):
                detail = "全场质量需复查"
            elif not drawn:
                detail = "本视图无曲线坐标"
        self.selection.set(detail)
        self.inspect_button.configure(text=f"核对{title}" if title else "核对原图")
        self._controls()

    def _plot_selection(self, point):
        self._set_selection(point)

    def _select_point(self, point):
        drawn = self.plot.select_point(point)
        self._set_selection(point, drawn=drawn)

    def _mode_changed(self, *_args):
        selected = self.selected
        mode = self.mode.get()
        self.plot.configure_mode(mode, enabled=bool(self.index and self.index.modes),
                                 empty_message="可按帧序号核对原图" if self.index else "选择已有逐帧记录")
        if selected is not None:
            self._select_point(selected)
        if mode == "force-strain":
            self.axis.set("工程应变 (%)")
        elif self.index and self.index.modes:
            origin = "本轮首图" if self.index.borrowed_reference else "参考图"
            self.axis.set(f"相对{origin}收图时间 (s)")
        else:
            self.axis.set("")
        self.issue_button.configure(text="下一处质量异常" if mode == "field-time" else "下一处缺测")
        self._controls()

    def _controls(self):
        index = self.index
        modes = index.modes if index else ()
        can_find_issue = False
        if modes and self.mode.get() in modes:
            column = modes.index(self.mode.get())
            can_find_issue = any(page.issues[column] for page in index.pages[self.page:])
        for widget, ready in ((self.pick_button, True), (self.refresh_button, self._path), (self.replay_button, index),
                              (self.export_button, index),
                              (self.mode_picker, len(modes) > 1), (self.previous, index and self.page > 0),
                              (self.next, index and self.page + 1 < len(index.pages)), (self.seek_button, index),
                              (self.seek_entry, index), (self.inspect_button, self.selected), (self.issue_button, can_find_issue)):
            widget.state(["!disabled" if ready and not self._reader_busy else "disabled"])

    def _work(self, action, complete, progress):
        if self._reader_busy:
            return
        self._reader_busy, self._error = True, None
        self.status.set(progress)
        self._controls()
        result = queue.SimpleQueue()

        def work():
            try:
                result.put((action(), None))
            except Exception as error:
                result.put((None, error))

        def poll():
            self._poll = None
            try:
                value, error = result.get_nowait()
            except queue.Empty:
                self._poll = self.top.after(50, poll)
                return
            self._reader_busy, self._error = False, error
            if error is not None:
                self.status.set(ui_error(error, details=False))
            else:
                complete(value)
            self._controls()
            if self._pending_replay is not None:
                path, self._pending_replay = self._pending_replay, None
                self._replay_saved(path)

        threading.Thread(target=work, name="capture-history-read", daemon=True).start()
        self._poll = self.top.after(50, poll)

    def _destroy(self, event):
        if event.widget is self.top and self._poll is not None:
            self.top.after_cancel(self._poll)

    def choose_record(self):
        name = filedialog.askopenfilename(
            parent=self.top, title="选择历史曲线记录",
            initialdir=self.directory if self.directory.is_dir() else self.experiment,
            initialfile=self._path.name if self._path else "", filetypes=[("实时分析逐帧记录", "*.jsonl")],
        )
        if name:
            self.open_record(Path(name))

    def refresh(self):
        if self._path is not None:
            self.open_record(self._path, preserve=True)

    def show_replay(self):
        if self.index is None or self._reader_busy:
            return
        if self._replay_window is not None and self._replay_window.top.winfo_exists():
            self._replay_window.top.lift()
            return
        from .live_replay_gui import LiveReplayWindow

        self._replay_window = LiveReplayWindow(self.top, self.index, self._replay_saved)

    def show_export(self):
        if self.index is None or self._reader_busy:
            return
        ToolTip.dismiss_window(self.top)
        if (window := self._export_window) is not None and window.top.winfo_exists():
            if window.working or (window.index is self.index and window.page == self.page):
                window.top.lift()
                return
            window.top.destroy()
        from .live_export_gui import LiveExportWindow

        self._export_window = LiveExportWindow(self.top, self.index, self.page, self.path_label.get())

    def _replay_saved(self, path):
        if self._reader_busy:
            self._pending_replay = path
        else:
            self.top.lift()
            self.open_record(path)

    def open_record(self, path, *, preserve=False):
        selected = self.selected.frame.sequence if preserve and self.selected and not self.selected.is_reference else None
        current_page = self.page if preserve else 0

        def read():
            index = read_history_index(self.experiment, path, expected_id=self.expected_id)
            found = index.find_sequence(selected) if selected is not None else None
            page = min(current_page, len(index.pages) - 1)
            return index, found or (page, index.read_page(page), None)

        self._work(read, self._opened, "正在核对并读取完整逐帧记录")

    def _opened(self, result):
        self.index, page = result
        self._path = self.index.path
        try:
            when = datetime.strptime(self._path.stem[:19], "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            label = self._path.stem
        else:
            label = when.strftime("%Y-%m-%d %H:%M:%S")
        self.path_label.set(label)
        tail = " · 略过未写完的末行" if self.index.incomplete_tail else ""
        kind = "按序重算" if self.index.replay else f"实时跳过 {self.index.skipped:,} 张"
        self.summary.set(f"{self.index.camera} · 1 张参考图 · {self.index.frames - 1:,} 张实验图"
                         f" · {kind}{tail}")
        self.mode_picker.configure(values=tuple(CURVE_MODES[mode] for mode in self.index.modes))
        if self.index.modes and self.mode.get() not in self.index.modes:
            self.mode.set(self.index.modes[0])
        self._show_page(page)

    def _show_page(self, result):
        self.page, points, selected = result
        ToolTip.dismiss_window(self.top)
        self.plot.clear()
        self.plot.points.extend(points)
        self._mode_changed()
        self._select_point(selected)
        self.page_label.set(f"{self.page + 1:,} / {len(self.index.pages):,}")
        page = self.index.pages[self.page]
        extent = f"实验帧 {page.first_sequence}–{page.last_sequence}" if page.last_sequence is not None else "参考图"
        self.status.set(f"当前段 · {extent} · {page.count:,} 条记录")

    def turn(self, direction):
        target = self.page + direction
        if self.index is not None and 0 <= target < len(self.index.pages):
            index = self.index
            self._work(lambda: (target, index.read_page(target), None), self._show_page, "正在读取所选段")
        return "break"

    def seek(self):
        if self.index is None:
            return
        try:
            sequence = int(self.sequence.get())
        except ValueError:
            sequence = 0
        if not 0 < sequence < 2**63:
            self.status.set("请输入大于零的图像帧序号。")
            return
        index = self.index

        def completed(found):
            if found is None:
                self.status.set(f"逐帧记录中没有本轮帧 {sequence} 的结果。可在采集复查中查找原图。")
            else:
                self._show_page(found)

        self._work(lambda: index.find_sequence(sequence), completed, "正在定位图像序号")

    def next_issue(self):
        if not self.index or not self.index.modes:
            return
        index, page, mode = self.index, self.page, self.mode.get()
        after = self.selected.frame.sequence if self.selected and not self.selected.is_reference else 0

        def completed(found):
            if found is None:
                self.status.set("当前选择之后的记录中没有此视图的缺测或质量异常。")
            else:
                self._show_page(found)

        self._work(lambda: index.next_issue(page, mode, after_sequence=after), completed, "正在查找下一处缺测或质量异常")

    def open_image(self, point=None):
        point = point or self.selected
        if point is None:
            return
        ToolTip.dismiss_window(self.top)
        if self._viewer is None or not self._viewer.top.winfo_exists():
            self._viewer = CurveFrameWindow(self.top)
        self._viewer.show(point)
