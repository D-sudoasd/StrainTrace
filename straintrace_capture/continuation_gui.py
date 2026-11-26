"""Choose the first source phase for a new linked acquisition."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .config import CaptureError
from .presentation import CAPTURE_EVENT_LABELS, MODE_LABELS, ROLE_LABELS, ui_error
from .viewport import repaint_after_resize


_STATES = {
    "ended": ("有结束记录", "台账记录了该阶段结束。"),
    "interrupted": ("中途停止", "台账记载了停止请求，或此阶段有开始记录、缺少结束记录。"),
    "confirmation_cancelled": ("确认前结束", "本次采集在等待操作员确认时结束。"),
    "needs_review": ("有记录需核对", "可在时间线查看该阶段的事件和已保存图像。"),
    "not_started": ("尚未启动", "台账缺少该阶段的开始记录，也没有登记图像。"),
    "unverified": ("阶段记录不足", "可用保存帧和原实验记录核对该阶段的执行情况。"),
}


class ContinuationWindow:
    def __init__(self, parent, path, report, camera_name, submit, show_timeline):
        self.path, self.experiment_id = path, report["experiment_id"]
        self.camera_name, self.submit, self.show_timeline = camera_name, submit, show_timeline
        self._rows = {row["phase"]: row for row in report["stages"]}
        self._busy = False
        self._error = ""
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("选择续采起点")
        self.top.geometry("770x510")
        self.top.minsize(580, 402)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.protocol("WM_DELETE_WINDOW", self._close)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self._close())

        body = ttk.Frame(self.top, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        heading = ttk.Label(body, text=f"原实验 · {report['name']}", style="CaptureSection.TLabel", width=1, justify="left")
        heading.grid(row=0, column=0, sticky="ew")
        wrap_label_on_resize(heading)
        ToolTip(heading, f"{report['name']}\n{path}\n实验标识 {self.experiment_id}")
        suggestion = (f"建议先核对原阶段 {report['suggested_phase']}，再选择续采起点。" if report["has_suggested_restart"]
                      else "请选择从哪个原阶段开始。")
        if report["unassigned_errors"]:
            suggestion += f" 有 {report['unassigned_errors']} 条采集错误未标明阶段。"
        hint = ttk.Label(body, text=suggestion, style="CaptureMuted.TLabel", width=1, justify="left")
        hint.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        wrap_label_on_resize(hint)
        ToolTip(hint, "建议依据写入台账的阶段边界、已保存帧及异常事件。可选择任一原阶段。")

        table_area = ttk.Frame(body)
        table_area.grid(row=2, column=0, sticky="nsew")
        table_area.rowconfigure(0, weight=1)
        table_area.columnconfigure(0, weight=1)
        columns = ("phase", "mode", "role", "status", "frames")
        self.table = ttk.Treeview(table_area, columns=columns, show="headings", selectmode="browse", height=5)
        self.table.grid(row=0, column=0, sticky="nsew")
        for column, title, width, minimum in zip(columns, ("原阶段", "方式", "用途", "台账状态", "已保存帧"),
                                                 (72, 122, 106, 170, 104), (66, 101, 82, 126, 94)):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, minwidth=minimum, stretch=column in ("mode", "status"))
        scroll = ttk.Scrollbar(table_area, orient=tk.VERTICAL, command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        for index, row in self._rows.items():
            self.table.insert("", "end", iid=str(index), values=(
                index, MODE_LABELS[row["mode"]], ROLE_LABELS[row["role"]], _STATES[row["state"]][0], row["saved_frames"],
            ))
        ToolTip(self.table, self._table_help, title="原采集阶段")
        self.table.bind("<<TreeviewSelect>>", self._on_selected)
        self.table.bind("<Return>", lambda _event: self._load())
        initial = str(report["suggested_phase"])
        self.table.selection_set(initial)
        self.table.focus(initial)
        self.table.see(initial)
        self.table.focus_set()

        self.selection_text = tk.StringVar(self.top)
        detail = ttk.Label(body, textvariable=self.selection_text, style="CaptureMuted.TLabel", width=1, justify="left")
        detail.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        wrap_label_on_resize(detail)
        ToolTip(detail, lambda _event: self._error or self.selection_text.get())
        footer = ttk.Frame(body)
        footer.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.timeline = ttk.Button(footer, text="查看阶段时间线…", command=self._open_timeline, style="Small.TButton")
        self.timeline.pack(side=tk.LEFT)
        ToolTip(self.timeline, "核对所选阶段的收图、触发和异常记录。可从时间线打开对应的原图。")
        self.load = ttk.Button(footer, text="载入续采计划", command=self._load, style="Record.TButton")
        self.load.pack(side=tk.RIGHT)
        ToolTip(self.load, lambda _event: "正在读取原记录，请稍候。" if self._busy else
                "从选中的原阶段开始准备新计划。检查设置并点开始采集后，再在新目录写入图像。")
        self.cancel = ttk.Button(footer, text="取消", command=self._close, style="Small.TButton")
        self.cancel.pack(side=tk.RIGHT, padx=8)
        ToolTip(self.cancel, "关闭续采起点选择。")
        self._on_selected()

    def _selected(self):
        selected = self.table.selection()
        return int(selected[0]) if selected else None

    def _on_selected(self, _event=None):
        if self._busy:
            return
        self._error = ""
        phase = self._selected()
        for button in (self.timeline, self.load):
            button.state(["!disabled" if phase else "disabled"])
        if phase is None:
            self.selection_text.set("选中一个原阶段。")
            return
        text = (f"新计划从原阶段 {phase} 开始，包含 {len(self._rows) - phase + 1} 个阶段。"
                f"原阶段 {phase} 在本轮编号为 1。")
        chosen = [row for number, row in self._rows.items() if number >= phase]
        if any(row["hardware_pairing"] for row in chosen):
            text += "\n需为本轮硬件触发重新核对脉冲编号和相机计数。"
        if any(row["role"] == "test" for row in chosen) and all(row["role"] != "reference" for row in chosen):
            text += "\n应变分析需要与加载状态对应的参考图。请在采集计划中核对参考图安排。"
        self.selection_text.set(text)

    def _table_help(self, event):
        phase = self._selected()
        if getattr(event, "keysym", None) in ("??", ""):
            if self.table.identify_region(event.x, event.y) == "heading":
                return "已保存帧统计本阶段各相机登记的图像帧。同一帧可包含多个图像部件。"
            key = self.table.identify_row(event.y)
            phase = int(key) if key else None
        if phase not in self._rows:
            return "选中一个原阶段，查看它的时间线或载入续采计划。"
        row = self._rows[phase]
        sections = [f"原阶段 {phase} · {_STATES[row['state']][0]}", _STATES[row["state"]][1]]
        sections.extend(f"{self.camera_name(camera)} · 已保存 {count} 帧" for camera, count in row["saved_by_camera"].items())
        sections.extend(f"{CAPTURE_EVENT_LABELS[kind]} · {count} 条记录" for kind, count in row["issues"].items())
        if row["cameras_without_frames"]:
            names = "、".join(self.camera_name(camera) for camera in row["cameras_without_frames"])
            sections.append(f"未登记图像的相机有 {names}。")
        if row["confirmation"]:
            sections.append("原计划要求在本阶段开始前人工确认。")
        return "\n".join(sections)

    def _open_timeline(self):
        if self._busy or (phase := self._selected()) is None:
            return
        try:
            self.show_timeline(phase)
        except CaptureError as exc:
            self._error = ui_error(exc)
            self.selection_text.set(ui_error(exc, details=False))

    def _load(self):
        if self._busy or (phase := self._selected()) is None:
            return
        self._busy = True
        ToolTip.dismiss_window(self.top)
        self.selection_text.set("正在核对原实验和归档标定…")
        for button in (self.timeline, self.load, self.cancel):
            button.state(["disabled"])
        self.top.grab_set()

        def finished():
            if self.top.winfo_exists():
                self.top.destroy()

        def failed(exc):
            if not self.top.winfo_exists():
                return
            self.top.grab_release()
            self._busy = False
            self.cancel.state(["!disabled"])
            self._on_selected()
            self._error = ui_error(exc)
            self.selection_text.set(ui_error(exc, details=False))

        try:
            self.submit(phase, finished, failed)
        except CaptureError as exc:
            failed(exc)

    def _close(self):
        if not self._busy:
            self.top.destroy()
