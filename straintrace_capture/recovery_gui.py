"""Review interrupted image writes and explicitly apply their verified recovery."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ezdic_gui.tooltips import ToolTip

from .presentation import CAPTURE_ERROR_LABELS, INTEGRITY_LABELS, STATE_LABELS, integrity_issues, ui_error
from .recovery import inspect_recovery, recover
from .storage import TERMINAL_STATES
from .viewport import repaint_after_resize


_JOURNAL = {"present": "已读取帧写入记录", "missing": "找不到帧写入记录", "unavailable": "无法读取帧写入记录",
            "not_recorded": "这份台账没有帧写入记录", "not_applicable": "本实验已结束"}


class RecoveryWindow:
    PAGE_SIZE = 40

    def __init__(self, parent, experiment, camera_names, submit, on_recovered, on_continue=None, *, experiment_id=None):
        self.experiment = experiment
        self.names = camera_names
        self.submit, self.on_recovered, self.on_continue = submit, on_recovered, on_continue
        self.report = None
        self.experiment_id = experiment_id
        self.working = None
        self.result = False
        self.rows, self.page = [], 0
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title(f"检查中断 · {experiment.name}")
        self.top.transient(parent)
        self.top.geometry("850x535")
        self.top.minsize(715, 435)
        repaint_after_resize(self.top)
        self.headline, self.status, self.detail, self.page_text = (tk.StringVar(self.top) for _ in range(4))
        self._build()
        self.top.grab_set()
        self._check()

    def _build(self):
        area = ttk.Frame(self.top, padding=14)
        area.pack(fill=tk.BOTH, expand=True)
        area.columnconfigure(0, weight=1)
        area.rowconfigure(2, weight=1)
        headline = ttk.Label(area, textvariable=self.headline, style="CaptureSection.TLabel")
        headline.grid(row=0, column=0, sticky="ew")
        ToolTip(headline, self._summary_help, title="实验台账")
        note = ttk.Label(area, textvariable=self.status, justify=tk.LEFT, style="CaptureMuted.TLabel")
        note.grid(row=1, column=0, sticky="ew", pady=(5, 9))
        note.bind("<Configure>", lambda event: note.configure(wraplength=max(1, event.width - 4)))
        ToolTip(note, self._status_help)

        table_area = ttk.Frame(area)
        table_area.grid(row=2, column=0, sticky="nsew")
        table_area.columnconfigure(0, weight=1)
        table_area.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(table_area, columns=("state", "file", "reason"), show="headings", selectmode="browse")
        self.table.grid(row=0, column=0, sticky="nsew")
        for key, label, width in (("state", "状态", 105), ("file", "文件名或写入记录", 282), ("reason", "说明", 220)):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=78, stretch=key != "state")
        scroll = ttk.Scrollbar(table_area, orient=tk.VERTICAL, command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        ToolTip(scroll, "滚动查看本页的图像和写入记录。用下方按钮可翻页。", title="中断检查结果")
        self.table.bind("<<TreeviewSelect>>", self._selected)
        ToolTip(self.table, self._table_help)

        nav = ttk.Frame(area)
        nav.grid(row=3, column=0, sticky="ew", pady=(8, 3))
        self.previous = ttk.Button(nav, text="上一页", command=lambda: self._move(-1), style="Small.TButton")
        self.next = ttk.Button(nav, text="下一页", command=lambda: self._move(1), style="Small.TButton")
        self.previous.pack(side=tk.LEFT)
        self.next.pack(side=tk.LEFT, padx=5)
        ttk.Label(nav, textvariable=self.page_text, style="CaptureMuted.TLabel").pack(side=tk.LEFT, padx=(3, 10))
        detail = ttk.Entry(nav, textvariable=self.detail, state="readonly")
        detail.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(detail, lambda _event: self.detail.get() or "选中一行可查看完整路径或写入记录行号。")

        actions = ttk.Frame(area)
        actions.grid(row=4, column=0, sticky="ew", pady=(7, 0))
        self.refresh = ttk.Button(actions, text="重新检查", command=self._check, style="Small.TButton")
        self.refresh.pack(side=tk.LEFT)
        ToolTip(self.refresh, "重新读取台账和遗留文件。原采集退出或文件位置调整后可再检查。")
        self.restore = ttk.Button(actions, text="恢复台账", command=self._restore, style="Record.TButton")
        self.restore.pack(side=tk.RIGHT)
        ToolTip(self.restore, self._restore_help, title="登记能够核实的图像")
        self.close = ttk.Button(actions, text="关闭", command=self._close, style="Small.TButton")
        self.close.pack(side=tk.RIGHT, padx=(0, 7))
        ToolTip(self.close, "回到复查页。")
        self.continue_button = ttk.Button(actions, text="选择续采起点…", command=self._continue, style="Small.TButton")
        if self.on_continue is not None:
            self.continue_button.pack(side=tk.RIGHT, padx=(0, 7))
        ToolTip(self.continue_button, self._continue_help, title="准备一个新的关联实验")
        self.top.protocol("WM_DELETE_WINDOW", self._close)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self._close())

    def _summary_help(self, _event):
        paths = [str(self.experiment)]
        if self.report and "cameras" in self.report:
            paths.insert(0, self.report["name"])
            paths.extend(f"{self.names.get(camera, camera)} · 已登记 {row['frames']} 帧／{row['images']} 张 TIFF"
                         for camera, row in self.report["cameras"].items())
        return "\n".join(paths)

    def _status_help(self, _event):
        if self.report and (error := self.report.get("capture_error")):
            return f"{self.status.get()}\n{error['code']}  {error.get('message', '')}"
        return self.status.get()

    def _restore_help(self, _event):
        if not self.report:
            return "先读取台账和写入记录。"
        if not self._can_restore():
            return "当前检查没有需要写入的恢复操作。"
        return "再次校验遗留 TIFF，将证据齐全的帧登记回原实验。未结束的台账会标记为已中断。"

    def _continue_help(self, _event):
        if not self.report or self.working:
            return "先完成本次中断检查。"
        if self.report["needs_finalize"]:
            return "这份台账仍记录为采集中。先恢复台账，将状态标记为已中断。"
        if not self.result and (frames := self.report["recovery"]["recoverable"]):
            return (f"当前台账有 {self.report['committed_frames']} 帧，另有 {len(frames)} 帧已校验、可补登记。\n"
                    "可先点“恢复台账”。\n此按钮按当前已登记的帧选择续采起点。")
        if integrity_issues(self.report):
            return "上方仍有待复查项。此按钮按当前台账选择续采起点，并准备新的实验目录。"
        return "按原阶段查看保存帧和异常记录。选中起点后，载入并检查在新目录执行的采集计划。"

    def _selected_index(self):
        selection = self.table.selection()
        return int(selection[0]) if selection else None

    def _selected(self, _event=None):
        index = self._selected_index()
        self.detail.set(self.rows[index][1] if index is not None and index < len(self.rows) else "")

    def _table_help(self, event):
        if event is not None and getattr(event, "keysym", "??") in ("??", ""):
            if self.table.identify_region(event.x, event.y) == "heading":
                return {
                    "#1": "区分可以补登记的图像、需要人工复查的图像和其他记录。",
                    "#2": "对应实验目录中的文件，或采集过程中的写入记录。选中一行可查看完整位置。",
                    "#3": "显示检查结果，说明为何能够恢复或仍需复查。",
                }.get(self.table.identify_column(event.x), "")
            index = self.table.identify_row(event.y)
            index = int(index) if index else None
        else:
            index = self._selected_index()
        if index is None or not 0 <= index < len(self.rows):
            return "选择一行查看文件路径。悬停可查看为何能够恢复或仍需复查。"
        state, path, reason, context = self.rows[index]
        return "\n".join(filter(None, (f"{state} · {reason}", path, context)))

    def _move(self, offset):
        if 0 <= self.page + offset and (self.page + offset) * self.PAGE_SIZE < len(self.rows):
            self.page += offset
            self._render_page()

    def _render_page(self):
        ToolTip.dismiss_window(self.top)
        self.table.delete(*self.table.get_children())
        start = self.page * self.PAGE_SIZE
        for index, row in enumerate(self.rows[start:start + self.PAGE_SIZE], start):
            state, path, reason, _ = row
            self.table.insert("", tk.END, iid=str(index), values=(state, path.rsplit("/", 1)[-1], reason))
        if self.table.get_children():
            self.table.selection_set(str(start))
        pages = (len(self.rows) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        self.page_text.set(f"第 {self.page + 1}/{pages} 页 · {len(self.rows)} 项" if pages else "暂无检查项")
        self._selected()
        self._controls()

    def _can_restore(self):
        return bool(self.report and not self.result and (self.report["needs_finalize"] or self.report["recovery"]["recoverable"]))

    def _controls(self):
        def toggle(widget, enabled):
            widget.state(["!disabled" if enabled else "disabled"])

        can_restore = self._can_restore()
        can_continue = not self.working and self.report and self.report["state"] in TERMINAL_STATES
        needs_review = can_continue and (can_restore or integrity_issues(self.report))
        toggle(self.refresh, not self.working)
        toggle(self.restore, not self.working and can_restore)
        self.restore.configure(style="Record.TButton" if can_restore else "Small.TButton")
        self.continue_button.configure(text="按当前台账续采…" if needs_review else "选择续采起点…",
                                       style="Record.TButton" if can_continue and not needs_review else "Small.TButton")
        toggle(self.continue_button, can_continue)
        toggle(self.close, self.working != "restore")
        toggle(self.previous, self.page > 0)
        toggle(self.next, (self.page + 1) * self.PAGE_SIZE < len(self.rows))

    def _check(self):
        self.working = "check"
        self.report = None
        self.rows, self.page = [], 0
        self.headline.set("正在核对遗留图像…")
        self.status.set("读取实验台账、帧写入记录和遗留文件，核对图像标识与校验值。")
        self._render_page()
        identity = self.experiment_id
        self.submit(lambda: inspect_recovery(self.experiment, expected_experiment_id=identity), self._display, self._failed)

    def _restore(self):
        if self.working or not self._can_restore():
            return
        self.working = "restore"
        self.status.set("正在登记验证通过的帧，并校验台账中的原图…")
        self._controls()
        identity = self.experiment_id
        self.submit(lambda: recover(self.experiment, expected_experiment_id=identity), self._display, self._failed)

    def _display(self, report):
        if not self.top.winfo_exists():
            return
        self.result = self.working == "restore"
        self.working = None
        self.report = report
        if not self.result:
            self.experiment_id = report["experiment_id"]
        recovery = report["recovery"]
        frames = recovery["recovered" if self.result else "recoverable"]
        issues = integrity_issues(report)
        state = "已恢复" if self.result else "可恢复"
        self.rows = [(state, row["path"], f"写入记录第 {row['line']} 行", f"来源文件 {row['source_path']}\n{row['frame_id']}")
                     for row in frames]
        for row in issues:
            path = row.get("source_path") or row.get("image_path") or row.get("path") or row.get("event_id") or ""
            if row.get("line"):
                path = f"{path} · 写入记录第 {row['line']} 行".strip(" ·")
            reason = INTEGRITY_LABELS.get(row["code"], "触发组关联不完整" if row["code"].startswith("TRIGGER_PAIRING_") else row["code"])
            self.rows.append(("待复查", path, reason, row.get("message", "")))
        self.headline.set(f"已登记 {report['committed_frames']} 帧 · {report['committed_images']} 张 TIFF · "
                          f"{state} {len(frames)} 帧 · 待复查 {len(issues)} 项")
        prior = report["state"].upper()
        state_caption = "上次采集未正常结束" if report.get("needs_finalize") else STATE_LABELS.get(prior, prior)
        note = "已完成原图和台账校验" if self.result else _JOURNAL[recovery["journal"]]
        if error := report.get("capture_error"):
            note += f"\n{CAPTURE_ERROR_LABELS.get(error['code'], '采集遇到异常，可悬停查看记录。')}"
        self.status.set(f"{state_caption} · {note}")
        self.page = 0
        self._render_page()
        if self.result:
            self.on_recovered(report)

    def _failed(self, exc):
        if self.top.winfo_exists():
            self.working = None
            self.report = None
            self.rows, self.page = [], 0
            self.headline.set("本次检查未完成")
            self.status.set(ui_error(exc, details=False))
            self._render_page()

    def _continue(self):
        if self.working or not self.report or self.report["state"] not in TERMINAL_STATES or self.on_continue is None:
            return
        identity = self.experiment_id
        self._close()
        self.on_continue(identity)

    def _close(self):
        if self.working != "restore":
            self.top.destroy()
