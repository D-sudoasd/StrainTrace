"""Review recorded relationships between acquisition ledgers."""

from __future__ import annotations

import sqlite3
import tkinter as tk
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .presentation import STATE_LABELS, ui_error
from .recent import recent_experiments
from .related import inspect_related_experiments
from .viewport import repaint_after_resize

_RELATION = {
    "current": "当前实验",
    "current_copy": "同标识目录",
    "parent": "上一轮",
    "repeat_source": "复拍来源",
    "ancestor": "更早实验",
    "continuation": "后续实验",
    "repeat": "复拍实验",
    "branch": "同源分支",
    "linked": "关联待核对",
}
_COLUMNS = {
    "relation": "关系", "name": "实验名称", "state": "状态", "frames": "已登记帧", "images": "TIFF 张数",
}
_COLUMN_HELP = {
    "relation": "依据台账中的续采和复拍记录显示关联。共享来源的其他实验会显示为同源分支。",
    "name": "从实验台账读取的名称。重名实验通过标识和目录区分。",
    "state": "该目录台账中记录的采集状态。",
    "frames": "台账中登记的相机帧总数，各台相机分别计数。",
    "images": "台账中登记的 TIFF 总数。同一帧可以保存多张图像。",
}


def _issue_help(issue):
    code = issue["code"]
    if code == "PARENT_UNAVAILABLE":
        paths = "\n".join(issue["mismatched_directories"])
        note = f"原实验尚未定位 · {issue['experiment_id']}"
        return f"{note}\n以下路径对应其他实验\n{paths}" if paths else note
    if code == "ID_MULTIPLE_DIRECTORIES":
        return f"同一实验标识出现在 {len(issue['directories'])} 个目录。选中各行可查看完整位置。"
    if code == "PARENT_CONFLICT":
        return f"同标识目录中的原实验记录互相冲突 · {issue['experiment_id']}"
    if code == "LINEAGE_CYCLE":
        return "实验关联形成循环 · " + "、".join(issue["experiment_ids"])
    return f"无法读取目录 · {issue['directory']}"


class RelatedExperimentsWindow:
    def __init__(self, parent, root, identity, submit, open_experiment):
        self.root, self.identity = root, identity
        self.submit, self.open_experiment = submit, open_experiment
        self.search_dirs = []
        self.rows = {}
        self.busy = False
        self.status_detail = ""
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("关联实验")
        self.top.transient(parent)
        self.top.geometry("875x480")
        self.top.minsize(710, 390)
        repaint_after_resize(self.top)
        self.summary = tk.StringVar(self.top)
        self.directory = tk.StringVar(self.top)
        self.status = tk.StringVar(self.top)
        self._build()
        self.top.grab_set()
        self._refresh()

    def _build(self):
        area = ttk.Frame(self.top, padding=14)
        area.pack(fill=tk.BOTH, expand=True)
        area.columnconfigure(0, weight=1)
        area.rowconfigure(2, weight=1)
        ttk.Label(area, text="关联实验", style="CaptureSection.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(area, textvariable=self.summary, style="CaptureMuted.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        self.table = ttk.Treeview(area, columns=tuple(_COLUMNS), show="headings", selectmode="browse", takefocus=True)
        for (column, label), width in zip(_COLUMNS.items(), (112, 221, 97, 92, 101), strict=True):
            self.table.heading(column, text=label)
            self.table.column(column, width=width, minwidth=110 if column == "name" else width,
                              stretch=column == "name", anchor="e" if column in ("frames", "images") else "w")
        self.table.grid(row=2, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(area, orient=tk.VERTICAL, command=self.table.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<<TreeviewSelect>>", self._selected)
        self.table.bind("<Return>", lambda _event: self._open())
        self.table.bind("<Double-1>", self._double_click)
        ToolTip(self.table, self._table_help, title="关联实验")

        path = ttk.Entry(area, textvariable=self.directory, state="readonly")
        path.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        ToolTip(path, lambda _event: self.directory.get() or "选择一行即可查看完整的实验目录。", title="实验目录")
        status = ttk.Label(area, textvariable=self.status, width=1, justify=tk.LEFT, style="CaptureMuted.TLabel",
                           takefocus=True)
        status.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self.status_detail or self.status.get(), title="查找结果")

        actions = ttk.Frame(area)
        actions.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.search_button = ttk.Button(actions, text="添加搜索目录…", command=self._add_search, style="Small.TButton")
        self.search_button.pack(side=tk.LEFT)
        ToolTip(self.search_button, "选择已移动的实验，或选择存放实验的文件夹。程序会核对台账标识。")
        self.refresh_button = ttk.Button(actions, text="重新查找", command=self._refresh, style="Small.TButton")
        self.refresh_button.pack(side=tk.LEFT, padx=(6, 0))
        ToolTip(self.refresh_button, "在当前目录附近、已知位置和所选搜索目录查找，再读一次登记的图像数量。")
        self.open_button = ttk.Button(actions, text="打开所选实验", command=self._open, style="Record.TButton")
        self.open_button.pack(side=tk.RIGHT)
        ToolTip(self.open_button, self._open_help)
        close = ttk.Button(actions, text="关闭", command=self.top.destroy, style="Small.TButton")
        close.pack(side=tk.RIGHT, padx=(0, 7))
        ToolTip(close, "回到当前实验。")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.table.focus_set()

    def _table_help(self, event):
        item = ""
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            if self.table.identify_region(event.x, event.y) == "heading":
                column = self.table.identify_column(event.x)
                if column.startswith("#") and column[1:].isdigit():
                    index = int(column[1:]) - 1
                    if 0 <= index < len(_COLUMN_HELP):
                        return tuple(_COLUMN_HELP.values())[index]
            item = self.table.identify_row(event.y)
        elif selection := self.table.selection():
            item = selection[0]
        row = self.rows.get(item)
        if row is None:
            return "选中一行查看目录和相机计数。双击或按回车可打开另一个实验。"
        help_lines = [row["name"], row["directory"], f"实验标识 · {row['experiment_id']}"]
        if row["continuation_of"]:
            help_lines.append(f"原实验标识 · {row['continuation_of']}")
        if source := row.get("repeat_of"):
            help_lines.append(f"复拍来源标识 · {source}")
        if (stamp := row["created_utc_ns"]) is not None:
            try:
                help_lines.append(datetime.fromtimestamp(stamp // 1_000_000_000).strftime("创建时间 · %Y-%m-%d %H:%M:%S"))
            except (OSError, OverflowError, ValueError):
                pass
        cameras = row["cameras"]
        help_lines.extend(f"{key} · {count['frames']} 帧 · {count['images']} 张 TIFF"
                          for key, count in tuple(cameras.items())[:5])
        if len(cameras) > 5:
            help_lines.append(f"共 {len(cameras)} 台相机。")
        return "\n".join(help_lines)

    def _selected_row(self):
        selection = self.table.selection()
        return self.rows.get(selection[0]) if selection else None

    def _selected(self, _event=None):
        row = self._selected_row()
        self.directory.set(row["directory"] if row else "")
        self.open_button.state(["disabled" if self.busy or not row or row["relation"] == "current" else "!disabled"])
        for button in (self.search_button, self.refresh_button):
            button.state(["disabled" if self.busy else "!disabled"])

    def _refresh(self, *, searched=None):
        if self.busy:
            return
        selected = self.directory.get()
        self.busy = True
        self._selected()
        self.status_detail = ""
        self.status.set("正在查找关联实验并读取台账计数…")
        directories = tuple(self.search_dirs)

        def read():
            try:
                recent = tuple(row.directory for row in recent_experiments())
            except (OSError, sqlite3.Error):
                recent = ()
            return inspect_related_experiments(self.root, search_dirs=directories, recent_paths=recent,
                                               expected_experiment_id=self.identity)

        self.submit(read, lambda report: self._loaded(report, selected, searched), self._lookup_failed)

    def _loaded(self, report, selected, searched):
        if not self.top.winfo_exists():
            return
        previous_dirs = {row["directory"] for row in self.rows.values()}
        self.table.delete(*self.table.get_children())
        self.rows.clear()
        wanted, current = None, None
        for index, row in enumerate(report["runs"]):
            key = str(index)
            self.rows[key] = row
            if row["directory"] == selected:
                wanted = key
            if row["relation"] == "current":
                current = key
            self.table.insert("", tk.END, iid=key, values=(
                _RELATION[row["relation"]], row["name"], STATE_LABELS.get(row["state"].upper(), "状态未知"),
                row["committed_frames"], row["committed_images"],
            ))
        if (choose := wanted or current) is not None:
            self.table.selection_set(choose)
            self.table.focus(choose)
            self.table.see(choose)
        self.busy = False
        unique = len({row["experiment_id"] for row in self.rows.values()})
        self.summary.set(f"{unique} 个实验 · {len(self.rows)} 个目录 · 按创建时间排列")
        errors = report["issues"]
        kinds = Counter(item["code"] for item in errors)
        hints = []
        if kinds["PARENT_UNAVAILABLE"]:
            hints.append(f"{kinds['PARENT_UNAVAILABLE']} 个原实验未定位，可添加搜索目录。")
        if kinds["ID_MULTIPLE_DIRECTORIES"]:
            hints.append("存在同标识目录，请选择要打开的实际位置。")
        if kinds["PARENT_CONFLICT"] or kinds["LINEAGE_CYCLE"]:
            hints.append("部分关联记录需要核对。")
        if kinds["SEARCH_UNAVAILABLE"] or kinds["EXPERIMENT_UNAVAILABLE"]:
            hints.append("部分目录无法读取。")
        if searched and not any(row["directory"] not in previous_dirs for row in self.rows.values()):
            hints.append("此次搜索未新增关联实验。")
        self.status.set(" ".join(hints) or "可双击打开另一次采集，各实验分别复查。")
        details = [_issue_help(item) for item in errors[:4]]
        if len(errors) > 4:
            details.append(f"共 {len(errors)} 项，完整信息可用相关实验的命令行报告查看。")
        self.status_detail = "\n\n".join(details)
        self._selected()

    def _lookup_failed(self, exc):
        if not self.top.winfo_exists():
            return
        self.rows.clear()
        self.table.delete(*self.table.get_children())
        self.summary.set("")
        self._open_failed(exc)

    def _add_search(self):
        if self.busy:
            return
        directory = filedialog.askdirectory(parent=self.top, title="选择移动后的实验或存放实验的目录", mustexist=True)
        if directory:
            path = Path(directory)
            if path not in self.search_dirs:
                self.search_dirs.append(path)
            self._refresh(searched=path)

    def _open_help(self, _event=None):
        if self.busy:
            return "请等待台账查询结束。"
        row = self._selected_row()
        if row is None:
            return "先在列表中选择一个实验。"
        if row["relation"] == "current":
            return "工作台当前正在复查此目录。请选择其他实验。"
        return "核对所选目录的实验标识，打开该实验的复查页。"

    def _double_click(self, event):
        if key := self.table.identify_row(event.y):
            self.table.selection_set(key)
            self._open()

    def _open(self):
        if self.busy or not (row := self._selected_row()) or row["relation"] == "current":
            return "break"
        self.busy = True
        self.status_detail = ""
        self.status.set("正在核对所选目录的实验标识…")
        self._selected()
        if not self.open_experiment(row["directory"], expected_id=row["experiment_id"], on_opened=self.top.destroy,
                                    on_error=self._open_failed, wanted=self.top.winfo_exists):
            self._open_failed("当前操作结束后可再次打开。")
        return "break"

    def _open_failed(self, exc):
        if self.top.winfo_exists():
            self.busy = False
            self.status.set(ui_error(exc, details=False))
            self.status_detail = ui_error(exc)
            self._selected()
