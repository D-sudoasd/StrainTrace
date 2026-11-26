"""Choose and explain the external detector identity columns in the Capture desktop."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.tooltips import ToolTip

from .detector_link import link_audited_detector
from .presentation import ui_error
from .pulse_audit import pulse_csv_columns
from .viewport import repaint_after_resize


class DetectorLinkDialog:
    def __init__(self, parent, experiment, audit, submit, accept, *, previous=None):
        self.experiment, self.audit = experiment, audit
        self.submit, self.accept = submit, accept
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("关联探测器帧")
        self.top.transient(parent)
        self.top.geometry("630x325")
        self.top.minsize(530, 320)
        repaint_after_resize(self.top)
        self.source = ""
        self.busy = False
        self.path_name = tk.StringVar(self.top)
        self.event_column = tk.StringVar(self.top)
        self.frame_column = tk.StringVar(self.top)
        self.status = tk.StringVar(self.top, "请选择探测器的记录文件。")
        self._build()
        for variable in (self.event_column, self.frame_column):
            variable.trace_add("write", self._controls)
        self._controls()
        self.top.grab_set()
        if previous:
            self._load(previous["path"], previous["event_column"], previous["frame_column"])

    def _build(self):
        layout = ttk.Frame(self.top, padding=14)
        layout.pack(fill=tk.BOTH, expand=True)
        actions = ttk.Frame(layout)
        actions.pack(side=tk.BOTTOM, anchor=tk.E, pady=(8, 0))
        area = ttk.Frame(layout)
        area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        area.columnconfigure(1, weight=1)
        ttk.Label(area, text=f"阶段 {self.audit['phase']} · {self.audit['source_id']}",
                  style="CaptureSection.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        tips = "CSV 每行记录共享脉冲编号与探测器帧标识。"
        intro = ttk.Label(area, text=tips, style="CaptureMuted.TLabel", justify=tk.LEFT)
        intro.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 10))
        intro.bind("<Configure>", lambda event: intro.configure(wraplength=max(1, event.width - 5)))
        ttk.Label(area, text="探测器 CSV").grid(row=2, column=0, sticky="w", padx=(0, 12))
        source = ttk.Entry(area, textvariable=self.path_name, state="readonly")
        source.grid(row=2, column=1, sticky="ew")
        ToolTip(source, lambda _event: self.source or tips)
        self.choose = ttk.Button(area, text="选择文件…", command=self._choose, style="Small.TButton")
        self.choose.grid(row=2, column=2, sticky="ew", padx=(8, 0))
        ToolTip(self.choose, "选择 UTF-8 CSV。首行为列名，可用 source_id 和 phase 列筛选当前信号源与阶段。")
        self.combo = []
        for row, label, variable, help_text in (
            (3, "脉冲编号列", self.event_column, "选择与 DAQ 记录对应的整数编号列，常用列名是 event_number。"),
            (4, "探测器帧标识列", self.frame_column, "选择能区分各帧的列，例如图像文件名或 HDF5 中的帧标识。"),
        ):
            ttk.Label(area, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
            menu = ttk.Combobox(area, textvariable=variable, state="readonly")
            menu.grid(row=row, column=1, columnspan=2, sticky="ew")
            self.combo.append(menu)
            ToolTip(menu, lambda _event, help_text=help_text: help_text if self.source else "先选择探测器 CSV。", title=label)

        status = ttk.Label(area, textvariable=self.status, justify=tk.LEFT, style="CaptureMuted.TLabel")
        status.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(7, 0))
        status.bind("<Configure>", lambda event: status.configure(wraplength=max(1, event.width - 5)))
        ToolTip(status, lambda _event: self.status.get())
        cancel = ttk.Button(actions, text="取消", command=self.top.destroy, style="Small.TButton")
        cancel.pack(side=tk.LEFT, padx=(0, 8))
        ToolTip(cancel, "返回 DAQ 核对窗口，保留已完成的结果。")
        self.run = ttk.Button(actions, text="开始关联", command=self._confirm, style="Record.TButton")
        self.run.pack(side=tk.LEFT)
        ToolTip(self.run, self._run_help, title="关联同一脉冲下的帧")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.choose.focus_set()

    def _run_help(self, _event):
        if self.busy:
            return "正在读取文件，请等本次操作完成。"
        if not self.source:
            return "先选择探测器 CSV。"
        if self.run.instate(["disabled"]):
            return "分别选择脉冲编号列和探测器帧标识列。"
        return "按共享脉冲编号关联。结果会列出各端缺帧、DAQ 无记录和编号冲突。"

    def _controls(self, *_args):
        self.choose.state(["disabled" if self.busy else "!disabled"])
        for menu in self.combo:
            menu.state(["!disabled" if self.source and not self.busy else "disabled"])
        event_column, frame_column = self.event_column.get(), self.frame_column.get()
        ready = self.source and event_column and frame_column and event_column != frame_column and not self.busy
        self.run.state(["!disabled" if ready else "disabled"])

    def _status(self, message, *, busy):
        self.busy = busy
        self.status.set(message)
        self._controls()

    def _choose(self):
        path = filedialog.askopenfilename(parent=self.top, filetypes=[("探测器 CSV", "*.csv"), ("所有文件", "*.*")])
        if path:
            self._load(path)

    def _load(self, path, event_column="event_number", frame_column="detector_frame"):
        self._status("正在读取 CSV 表头…", busy=True)

        def loaded(fields):
            if not self.top.winfo_exists():
                return
            fields = tuple(field for field in fields if field not in ("phase", "source_id"))
            if len(fields) < 2:
                self._status("需要至少两列，分别记录共享脉冲编号和探测器帧标识。", busy=False)
                return
            self.source = path
            self.path_name.set(Path(path).name)
            for menu in self.combo:
                menu.configure(values=fields)
            self.event_column.set(event_column if event_column in fields else "")
            self.frame_column.set(frame_column if frame_column in fields else "")
            self._status("确认两列后点“开始关联”。", busy=False)

        self.submit(lambda: pulse_csv_columns(path, code="DETECTOR_CSV_INVALID"), loaded, self._failed)

    def _failed(self, exc):
        if self.top.winfo_exists():
            self._status(str(exc) if getattr(exc, "code", None) else ui_error(exc), busy=False)

    def _confirm(self):
        self._status("正在复核 DAQ 来源和相机记录，并查找对应的探测器帧…", busy=True)
        source, event_column, frame_column = self.source, self.event_column.get(), self.frame_column.get()

        def saved(report):
            if self.top.winfo_exists():
                self.top.destroy()
                self.accept(report)

        self.submit(lambda: link_audited_detector(self.experiment, source, self.audit, event_column=event_column,
                                                  frame_column=frame_column), saved, self._failed)
