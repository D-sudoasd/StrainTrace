"""Reopen a recently used experiment, including one moved to another drive."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from ezdic_gui.tooltips import ToolTip

from .presentation import ui_error
from .recent import recent_experiments
from .viewport import repaint_after_resize


class RecentExperimentsDialog:
    def __init__(self, parent, submit, open_experiment):
        self.open_experiment = open_experiment
        self.rows = {}
        self.busy = True
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("最近实验")
        self.top.transient(parent)
        self.top.geometry("745x425")
        self.top.minsize(620, 350)
        repaint_after_resize(self.top)
        self.status = tk.StringVar(self.top, "正在读取最近实验…")
        self.directory = tk.StringVar(self.top)
        self._build()
        self._controls()
        self.top.grab_set()
        submit(recent_experiments, self._loaded, self._history_failed)

    def _build(self):
        area = ttk.Frame(self.top, padding=14)
        area.pack(fill=tk.BOTH, expand=True)
        area.columnconfigure(0, weight=1)
        area.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(area, columns=("name", "directory"), show="headings", selectmode="browse", takefocus=True)
        self.table.heading("name", text="实验名称")
        self.table.heading("directory", text="实验目录")
        self.table.column("name", width=230, minwidth=170, stretch=False)
        self.table.column("directory", minwidth=235)
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(area, orient=tk.VERTICAL, command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        ToolTip(scroll, "滚动查看其余历史记录。", title="最近实验")
        self.table.bind("<<TreeviewSelect>>", self._selected)
        self.table.bind("<Return>", lambda _event: self._open())
        self.table.bind("<Double-1>", self._double_click)
        ToolTip(self.table, self._table_help, title="打开最近使用的实验")
        path = ttk.Entry(area, textvariable=self.directory, state="readonly")
        path.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 3))
        ToolTip(path, lambda _event: self.directory.get() or "选中一行可查看上次使用的完整目录。")

        status = ttk.Label(area, textvariable=self.status, justify=tk.LEFT, style="CaptureMuted.TLabel")
        status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(3, 8))
        status.bind("<Configure>", lambda event: status.configure(wraplength=max(1, event.width - 5)))
        ToolTip(status, lambda _event: self.status.get())
        actions = ttk.Frame(area)
        actions.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.relocate = ttk.Button(actions, text="定位已移动的实验…", command=self._relocate, style="Small.TButton")
        self.relocate.pack(side=tk.LEFT)
        ToolTip(self.relocate, lambda _event: self._selection_help(
            "如果改了目录名或移到其他磁盘，选择新位置。程序核对实验标识后才更新路径。"))
        self.open_button = ttk.Button(actions, text="打开实验", command=self._open, style="Record.TButton")
        self.open_button.pack(side=tk.RIGHT)
        ToolTip(self.open_button, lambda _event: self._selection_help(
            "读取所选实验的台账，打开复查页并显示最近保存的图像。"))
        cancel = ttk.Button(actions, text="关闭", command=self.top.destroy, style="Small.TButton")
        cancel.pack(side=tk.RIGHT, padx=(0, 8))
        ToolTip(cancel, "返回当前工作台。")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.table.focus_set()

    def _table_help(self, event):
        if event is not None and getattr(event, "keysym", "??") in ("??", ""):
            if self.table.identify_region(event.x, event.y) == "heading":
                return ("显示采集时填写的实验名称。" if self.table.identify_column(event.x) == "#1" else
                        "显示上次使用的目录。实验移过位置时，可选中记录后点击“定位已移动的实验”。")
            row = self.rows.get(self.table.identify_row(event.y))
        else:
            row = self._selected_row()
        if row:
            return f"{row.name}\n{row.directory}\n双击或按 Enter 打开实验。"
        return "从最近采集或打开的实验中选择一项。可按 Enter 打开。"

    def _selection_help(self, text):
        if self.busy or not self.rows:
            return self.status.get()
        return text if self._selected_row() else "先在上方选中一个实验。"

    def _loaded(self, rows):
        if not self.top.winfo_exists():
            return
        for index, row in enumerate(rows):
            key = str(index)
            self.rows[key] = row
            self.table.insert("", tk.END, iid=key, values=(row.name, str(row.directory)))
        if rows:
            self.table.selection_set("0")
            self.table.focus("0")
        self.busy = False
        self.status.set("双击实验可打开复查。" if rows else "已有采集或打开过的实验会显示在这里。可返回工作台选择实验目录。")
        self._selected()

    def _history_failed(self, _exc):
        if self.top.winfo_exists():
            self.busy = False
            self.status.set("无法读取最近实验记录。可返回工作台，使用“打开实验”选择目录。")
            self._controls()

    def _selected_row(self):
        selection = self.table.selection()
        return self.rows.get(selection[0]) if selection else None

    def _selected(self, _event=None):
        row = self._selected_row()
        self.directory.set(str(row.directory) if row else "")
        self._controls()

    def _controls(self):
        state = "!disabled" if not self.busy and self._selected_row() else "disabled"
        self.open_button.state([state])
        self.relocate.state([state])

    def _double_click(self, event):
        if self.table.identify_row(event.y):
            self._open()

    def _relocate(self):
        if not self.busy and self._selected_row():
            chosen = filedialog.askdirectory(parent=self.top, title="定位实验的新目录", mustexist=True)
            if chosen:
                self._open(chosen)

    def _open(self, folder=None):
        if self.busy or not (row := self._selected_row()):
            return
        self.busy = True
        self._controls()
        self.status.set("正在读取实验台账并核对实验标识…")
        if not self.open_experiment(folder or row.directory, expected_id=row.experiment_id, on_opened=self.top.destroy,
                                    on_error=self._open_failed, wanted=self.top.winfo_exists):
            self._open_failed("当前操作还在进行。结束后可重新打开实验。")

    def _open_failed(self, exc):
        self.busy = False
        self.status.set(ui_error(exc, details=False))
        self._controls()
