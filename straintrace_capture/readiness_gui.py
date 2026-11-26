"""Selection and contextual help for the preflight findings."""

import tkinter as tk
from collections import Counter
from tkinter import ttk

from ezdic_gui.controls import ScrollableForm, wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .presentation import (
    ISSUE_LABELS, READINESS_CONFIRMATION_ISSUES, readiness_issue_detail, readiness_issue_location, readiness_issue_text,
)


_LEVELS = ("阻塞", "需复核", "提示")


def _level(issue):
    if issue["severity"] == "block":
        return 0
    return 1 if issue["code"] in READINESS_CONFIRMATION_ISSUES else 2


class ReadinessOverview(ttk.Frame):
    """Keep each explanation and action bound to the same finding."""

    def __init__(self, parent, issues, labels, phase_offset, issue_actions):
        super().__init__(parent, padding=(9, 8))
        self.issues = {str(number): issue for number, issue in enumerate(issues)}
        self.labels, self.phase_offset = labels, phase_offset
        self.issue_actions = issue_actions
        self.stale_reason = ""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=2, minsize=83)
        self.rowconfigure(2, weight=1, minsize=78)
        counts = Counter(_level(issue) for issue in issues)
        summary = [f"{title} {counts[level]} 项" for level, title in enumerate(_LEVELS) if counts[level]]
        if not counts[0]:
            summary.insert(0, "检查通过" if not counts[1] else "检查完成")
        self.summary = ttk.Label(self, text=" · ".join(summary), style="CaptureSection.TLabel")
        self.summary.grid(row=0, sticky="ew", pady=(0, 7))
        ToolTip(self.summary, lambda _event: self.stale_reason or
                "阻塞项须处理后才能开始。需复核的事项会在开始前请你确认。", title="检查结果")

        table_area = ttk.Frame(self)
        table_area.grid(row=1, sticky="nsew")
        table_area.columnconfigure(0, weight=1)
        table_area.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(table_area, columns=("level", "scope", "finding"), show="headings",
                                  selectmode="browse", height=4)
        for column, title, width, minimum in (("level", "结果", 79, 73), ("scope", "位置", 151, 95),
                                               ("finding", "检查事项", 360, 180)):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, minwidth=minimum, stretch=column != "level")
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table_area, orient=tk.VERTICAL, command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        ToolTip(scroll, "滚动查看其余检查事项。也可聚焦列表后用方向键浏览。", title="检查事项")
        ToolTip(self.table, self._table_help, title="检查事项")
        for key, issue in sorted(self.issues.items(), key=lambda item: _level(item[1])):
            location = readiness_issue_location(issue, phase_offset, labels, include_path=False)
            context = location or ("文件或目录" if issue.get("path") else "采集计划")
            caption = ISSUE_LABELS[issue["code"]].partition("。")[0].partition("，")[0]
            self.table.insert("", tk.END, iid=key, values=(_LEVELS[_level(issue)], context, caption))
        self.table.bind("<<TreeviewSelect>>", self._select)

        self.details = ScrollableForm(self, padding=(0, 7))
        self.details.grid(row=2, sticky="nsew")
        self.detail = ttk.Label(self.details.body, width=1, justify=tk.LEFT)
        self.detail.pack(fill=tk.X)
        wrap_label_on_resize(self.detail)
        self.details.bind_controls()
        ToolTip(self.detail, lambda _event: self._describe(self._selected()), title="所选事项")
        ToolTip(self.details.scrollbar, "滚动阅读所选事项的完整说明。")
        self.action = ttk.Button(self, style="Small.TButton", command=self._apply)
        self.action.grid(row=3, sticky="e", pady=(3, 0))
        ToolTip(self.action, lambda _event: self.stale_reason or self._action_spec()[2], title="处理所选事项")
        first = self.table.get_children()[0]
        self.table.selection_set(first)
        self.table.focus(first)
        self._select()

    def _selected(self):
        selected = self.table.selection()
        return selected[0] if selected else ""

    def _action_spec(self):
        return self.issue_actions.get(self._selected(), ("", None, ""))

    def _describe(self, key):
        if key not in self.issues:
            return "选中一行可阅读原因。悬停或按 F1 查看该项的完整说明。"
        text = readiness_issue_text(self.issues[key], self.phase_offset, self.labels)
        if spec := self.issue_actions.get(key):
            return f"{text}\n\n{self.stale_reason or spec[2]}"
        return text

    def _table_help(self, event):
        key = self._selected()
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            key = self.table.identify_row(event.y)
        return self._describe(key)

    def _select(self, _event=None):
        issue = self.issues.get(self._selected())
        text = ""
        if issue is not None:
            detail = readiness_issue_detail(issue)
            if issue["code"] == "CAMERA_SETTING_INVALID":
                text = detail
            else:
                parts = (detail, ISSUE_LABELS[issue["code"]], issue.get("path"))
                text = "\n".join(part for part in parts if part)
        self.detail.configure(text=text)
        self.details.canvas.yview_moveto(0)
        label, command, _hint = self._action_spec()
        if command is None:
            self.action.grid_remove()
        else:
            self.action.configure(text=label)
            self.action.state(["disabled" if self.stale_reason else "!disabled"])
            self.action.grid()

    def _apply(self):
        _label, command, _hint = self._action_spec()
        if command is not None and not self.stale_reason:
            command(self.winfo_toplevel())

    def invalidate(self, reason):
        self.stale_reason = reason
        self.summary.configure(text="设置已修改 · 上次检查结果")
        self._select()
