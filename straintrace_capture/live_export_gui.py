"""Export the saved history currently selected in Capture."""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.controls import localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .config import CaptureError
from .frame_search import parse_sequence_range
from .live_history import export_history_csv
from .presentation import ui_error
from .viewport import repaint_after_resize


SCOPE_HELP = {
    "整份记录": "导出打开窗口时已读取的全部实验帧和参考图。采集中可先重新读取，再导出后续结果。",
    "当前段": "导出打开本窗口时曲线显示的那一段。参考图会一并导出。",
    "指定帧号": "选择本轮的一个原图序号或连续范围。只导出范围内已经计算并保存的实验图。",
}


class LiveExportWindow:
    def __init__(self, parent, index, page, record_label):
        self.index, self.page = index, page
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("StrainTrace · 导出曲线表格")
        self.top.transient(parent)
        self.top.geometry("610x440" if index.load_source else "610x350")
        self.top.minsize(525, 420 if index.load_source else 326)
        repaint_after_resize(self.top)
        self.top.protocol("WM_DELETE_WINDOW", self._close)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self._close())
        self.top.bind("<Destroy>", self._destroy, add="+")

        current = index.pages[page]
        self._scopes = {"all": "整份记录"}
        self._current_range = None
        if current.first_sequence is not None:
            self._scopes["page"] = "当前段"
            self._current_range = current.first_sequence, current.last_sequence
        if index.frames > 1:
            self._scopes["range"] = "指定帧号"
        extent_text = f"{current.first_sequence}-{current.last_sequence}" if self._current_range else ""
        self.scope = tk.StringVar(self.top, "all")
        self.range_text = tk.StringVar(self.top, extent_text)
        self.extent = tk.StringVar(self.top)
        self.fill_load = tk.BooleanVar(self.top, False)
        self.table_label = tk.StringVar(self.top)
        self.status = tk.StringVar(self.top, "点击“保存 CSV…”选择文件位置。")
        self._table = None
        self._saved = None
        self._detail = None
        self._working = False
        self._valid = True
        self._cancel = threading.Event()
        self._result = queue.SimpleQueue()
        self._after = None
        self._written = 0
        self._build(record_label)
        self.scope.trace_add("write", self._controls)
        self.range_text.trace_add("write", self._controls)
        self._controls()
        target = self.scope_picker if len(self._scopes) > 1 else self.start_button
        target.focus_set()

    @property
    def working(self):
        return self._working

    def _build(self, record_label):
        panel = ttk.Frame(self.top, padding=14)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(0, weight=1)
        kind = "按序重算" if self.index.replay else "实时记录"
        source = ttk.Label(
            panel, width=1, text=f"{kind} · {record_label} · {self.index.camera}",
            style="CaptureSection.TLabel",
        )
        source.grid(row=0, sticky="ew", pady=(0, 11))
        ToolTip(source, f"{self.index.path}\n从当前已读取的记录导出。若采集仍在继续，可重新读取后再导出。", title="源记录")

        scope = ttk.Frame(panel)
        scope.grid(row=1, sticky="ew")
        scope_label = ttk.Label(scope, text="图像范围")
        scope_label.pack(side=tk.LEFT, padx=(0, 8))
        ToolTip(scope_label, "决定哪些实验帧进入表格。三种范围都包含原参考图。")
        self.scope_picker = localized_combobox(scope, self.scope, self._scopes, width=12)
        self.scope_picker.pack(side=tk.LEFT)
        ChoiceToolTip(self.scope_picker, "按图像帧序号选择范围。表格会保留所有已记录的测量量。",
                      SCOPE_HELP, title="图像范围")
        self.range_entry = ttk.Entry(scope, textvariable=self.range_text, width=14)
        self.range_entry.pack(side=tk.LEFT, padx=(8, 0))
        ToolTip(self.range_entry, "输入单个帧号，例如 42，或范围 40-80，包含两端。图像序号可在历史曲线中查看。")
        extent = ttk.Label(panel, width=1, textvariable=self.extent, style="CaptureMuted.TLabel", justify=tk.LEFT)
        extent.grid(row=2, sticky="ew", pady=(6, 9))
        wrap_label_on_resize(extent)
        ToolTip(extent, "表格中每个标距单独占一行。同一张实验图可对应多行。", title="表格范围")

        if self.index.load_source is not None:
            load = ttk.Frame(panel)
            load.grid(row=3, sticky="ew", pady=(2, 5))
            self.fill_toggle = ttk.Checkbutton(load, text="补齐缺失的载荷与位移", variable=self.fill_load, command=self._controls)
            self.fill_toggle.pack(anchor="w")
            ToolTip(self.fill_toggle, "已匹配的载荷和位移取自原记录。启用后，程序用同一轮试验的新增采样补齐缺测，并在表中标出依据。")
            line = ttk.Frame(load)
            line.pack(fill=tk.X, pady=(4, 0))
            table = ttk.Label(line, width=1, textvariable=self.table_label, style="CaptureMuted.TLabel")
            table.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))
            ToolTip(table, lambda _event: str(self._table or self.index.load_source) if self.fill_load.get()
                    else "按逐帧记录里的数值导出。缺测值在表格中留空。", title="载荷来源")
            self.table_button = ttk.Button(line, text="选择载荷表…", command=self._pick_table, style="Small.TButton")
            self.table_button.pack(side=tk.RIGHT)
            ToolTip(self.table_button, "可选择同一轮试验的更新表。程序会核对原先用过的采样。")

        self.bar = ttk.Progressbar(panel, mode="determinate", maximum=1, style="CaptureCopy.Horizontal.TProgressbar")
        self.bar.grid(row=4, sticky="ew", pady=(7, 5))
        status = ttk.Label(panel, width=1, textvariable=self.status, style="CaptureMuted.TLabel", justify=tk.LEFT)
        status.grid(row=5, sticky="new", pady=(2, 0))
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.status.get(), title="导出详情")
        panel.rowconfigure(5, weight=1, minsize=status.winfo_reqheight())

        actions = ttk.Frame(panel)
        actions.grid(row=6, sticky="ew", pady=(9, 0))
        self.close_button = ttk.Button(actions, text="关闭", command=self._close)
        self.close_button.pack(side=tk.LEFT)
        ToolTip(self.close_button, lambda _event: "取消当前导出，清理未写完的表格。" if self._working else "关闭导出窗口。")
        self.open_button = ttk.Button(actions, text="打开文件夹", command=self._open_folder, style="Small.TButton")
        ToolTip(self.open_button, lambda _event: str(self._saved.parent) if self._saved else None)
        self.start_button = ttk.Button(actions, text="保存 CSV…", command=self.start, style="Record.TButton")
        self.start_button.pack(side=tk.RIGHT)
        ToolTip(self.start_button, lambda _event: "用保存对话框选择文件名和位置。可用 Excel、Origin 或 Python 打开表格。"
                if self._valid else "请先输入正整数帧号或范围，例如 40-80。")

    def _interval(self):
        if self.scope.get() == "range":
            start, end = parse_sequence_range(self.range_text.get())
            if start is None:
                raise CaptureError("INVALID_FRAME_QUERY", "请输入帧号或范围。")
            return start, end
        return self._current_range if self.scope.get() == "page" else None

    def _controls(self, *_args):
        custom = self.scope.get() == "range"
        if custom:
            self.range_entry.pack(side=tk.LEFT, padx=(8, 0))
        else:
            self.range_entry.pack_forget()
        try:
            interval = self._interval()
        except CaptureError:
            self._valid = False
            label = "输入单个帧号或范围，例如 40-80。"
        else:
            self._valid = True
            if self.scope.get() == "page":
                section = self.index.pages[self.page]
                count = section.count - (section.first_index == 0)
                label = f"第 {self.page + 1} 段 · {count:,} 张实验图及参考图"
            elif interval:
                start, end = interval
                label = f"帧 {start:,}–{end:,} · 范围内已读取的实验图及参考图"
            else:
                label = f"已读取的 {self.index.frames - 1:,} 张实验图及参考图"
        self.extent.set(label)
        self.scope_picker.state(["disabled" if self._working or len(self._scopes) == 1 else "!disabled"])
        self.range_entry.state(["disabled" if self._working else "!disabled"])
        if self.index.load_source is not None:
            self.fill_toggle.state(["disabled" if self._working else "!disabled"])
            self.table_button.state(["!disabled" if self.fill_load.get() and not self._working else "disabled"])
            self.table_label.set((self._table or self.index.load_source).name if self.fill_load.get()
                                 else "保留逐帧记录的载荷和位移")
        if self._saved is not None and hasattr(os, "startfile"):
            self.open_button.pack(side=tk.LEFT, padx=(6, 0))
        self.open_button.state(["disabled" if self._working else "!disabled"])
        self.start_button.state(["!disabled" if self._valid and not self._working else "disabled"])
        self.close_button.configure(text="取消导出" if self._working else "关闭")
        self.close_button.state(["disabled" if self._working and self._cancel.is_set() else "!disabled"])

    def _pick_table(self):
        current = self._table or self.index.load_source
        name = filedialog.askopenfilename(
            parent=self.top, title="选择同一轮实验的载荷表", initialdir=current.parent,
            filetypes=[("载荷表", "*.csv *.tsv *.txt"), ("所有文件", "*.*")],
        )
        if name:
            self._table = Path(name)
            self._controls()

    def start(self):
        if self._working or not self._valid:
            return
        interval = self._interval()
        fill_load = self.fill_load.get()
        prefix = "补齐载荷" if fill_load else "曲线"
        scope = f"-帧{interval[0]}-{interval[1]}" if interval else ""
        name = filedialog.asksaveasfilename(
            parent=self.top, title="保存曲线表格", initialdir=self._saved.parent if self._saved else self.index.path.parent,
            initialfile=f"{prefix}-{self.index.path.stem[:55]}{scope}.csv", defaultextension=".csv",
            filetypes=[("CSV 表格", "*.csv")],
        )
        if not name:
            return
        target, table = Path(name), self._table if fill_load else None
        self._working, self._detail, self._written = True, None, 0
        self._cancel.clear()
        self.status.set("正在核对记录和载荷表" if fill_load else "正在核对记录")
        self.bar.configure(mode="indeterminate", maximum=100, value=0)
        self.bar.start(18)
        self._controls()

        def progress(count):
            self._written = count

        def work():
            try:
                result = export_history_csv(self.index, target, sequence_range=interval, fill_load=fill_load,
                                            load_table=table, cancelled=self._cancel.is_set, progress=progress)
            except Exception as error:
                self._result.put((target, None, error))
            else:
                self._result.put((target, result, None))

        threading.Thread(target=work, name="capture-history-export", daemon=True).start()
        self._after = self.top.after(75, self._poll)

    def _poll(self):
        self._after = None
        try:
            path, result, error = self._result.get_nowait()
        except queue.Empty:
            if self._written and not self._cancel.is_set():
                self.status.set(f"正在写入表格 · {self._written:,} 张实验图")
            self._after = self.top.after(75, self._poll)
            return
        self.bar.stop()
        self.bar.configure(mode="determinate", maximum=1, value=0 if error is not None else 1)
        self._working = False
        if error is not None:
            self.status.set(ui_error(error, details=False))
            self._detail = ui_error(error)
        else:
            self._saved = path
            caption = f"已保存 {result['test_frames']:,} 张实验图及参考图"
            if "filled_frames" in result:
                caption += f" · 补齐 {result['filled_frames']:,} 张图的力学数据"
            self.status.set(f"{caption}\n{path.name}")
            self._detail = f"{caption}\n{path}"
        self._controls()

    def _open_folder(self):
        if self._saved is not None and hasattr(os, "startfile"):
            try:
                os.startfile(self._saved.parent)
            except OSError as error:
                self.status.set(ui_error(error, details=False))
                self._detail = ui_error(error)

    def _close(self):
        if not self._working:
            self.top.destroy()
            return
        self._cancel.set()
        self.status.set("正在取消导出")
        self._controls()

    def _destroy(self, event):
        if event.widget is self.top:
            self._cancel.set()
            if self._after is not None:
                self.top.after_cancel(self._after)
