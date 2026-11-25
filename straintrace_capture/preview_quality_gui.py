"""Compare raw preview checks and return to the affected camera."""

import threading
import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import ScrollableForm, wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .presentation import preview_quality_detail, ui_error
from .preview_quality import PREVIEW_CHECK_FRAMES, quality_has_alert
from .viewport import repaint_after_resize


_STATES = {
    "unavailable": "未检查",
    "changed": "画面已变更",
    "timeout": "新画面不足",
    "failed": "无法检查",
}


class PreviewQualityWindow:
    def __init__(self, parent, measure, open_camera):
        self.measure = measure
        self.open_camera = open_camera
        self.working = False
        self.cancelled = threading.Event()
        self.rows = {}
        self.summaries = {}
        self.names = {}
        self._stale = frozenset()
        self._cameras = frozenset()
        self._blocked = ""

        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("多机画质检查")
        self.top.geometry("635x465")
        scaling = float(self.top.tk.call("tk", "scaling"))
        self.top.minsize(420, max(315, round(145 * scaling)))
        self.top.transient(parent)
        self.top.protocol("WM_DELETE_WINDOW", self.top.destroy)
        self.top.bind("<Escape>", self._escape)
        self.top.bind("<Destroy>", self._destroyed)
        repaint_after_resize(self.top)

        layout = ttk.Frame(self.top, padding=(13, 10))
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(2, weight=1)
        self.status = ttk.Label(layout, text="正在检查预览画面…", style="CaptureSection.TLabel")
        self.status.grid(row=0, sticky="w", pady=(0, 6))
        ToolTip(self.status, f"每台取 {PREVIEW_CHECK_FRAMES} 张新预览图。选中相机可看最大比例、出现次数和检查范围。")
        listing = ttk.Frame(layout)
        listing.grid(row=1, sticky="ew")
        self.table = ttk.Treeview(listing, columns=("state",), show="tree headings", selectmode="browse", height=2)
        self.table.heading("#0", text="相机")
        self.table.heading("state", text="检查结果")
        self.table.column("#0", width=220, minwidth=100)
        self.table.column("state", width=190, minwidth=142)
        self.table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll = ttk.Scrollbar(listing, orient=tk.VERTICAL, command=self.table.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<<TreeviewSelect>>", self._select)
        self.table.bind("<Return>", lambda _event: self._open())
        ToolTip(self.table, self._table_help, title="检查所选相机")
        ToolTip(scroll, "滚动查看其他相机。聚焦表格后可按方向键选择。")

        detail_area = self.detail_area = ScrollableForm(layout, padding=(0, 5, 0, 0))
        detail_area.grid(row=2, sticky="nsew")
        self.detail = ttk.Label(detail_area.body, width=1, justify=tk.LEFT, takefocus=True)
        self.detail.pack(fill=tk.X)
        wrap_label_on_resize(self.detail)
        detail_area.bind_controls()
        ToolTip(self.detail, lambda _event: self.detail.cget("text"), title="检查依据")

        actions = ttk.Frame(layout)
        actions.grid(row=3, sticky="ew", pady=(6, 0))
        self.open_button = ttk.Button(actions, text="查看画面", command=self._open, style="Small.TButton")
        self.open_button.pack(side=tk.RIGHT)
        self.measure_button = ttk.Button(actions, text="重新检查", command=self.measure, style="Small.TButton")
        self.measure_button.pack(side=tk.RIGHT, padx=(0, 6))
        ToolTip(self.open_button, self._open_help, title="查看画面")
        ToolTip(self.measure_button, lambda _event: self._blocked or (
            "等待本次检查结束。" if self.working else "从各台相机重新取新画面。调整过照明或参数后可复测。"))

    def begin(self, names):
        self.cancelled.clear()
        self.working = True
        self.names = names
        self.rows = {}
        self.summaries = {}
        self._stale = frozenset()
        self.status.configure(text="正在检查预览画面…")
        self.detail.configure(text="")
        self.table.delete(*self.table.get_children())
        for identity, name in names.items():
            self.table.insert("", tk.END, iid=identity, text=name, values=("待检查",))
        self._controls()

    def finish(self, result):
        if not result:
            self.fail("当前没有可检查的相机，请检查预览。")
            return
        self.working = False
        self.rows = {row.target.camera_id: row for row in result}
        self.summaries = {identity: row.summary() for identity, row in self.rows.items()}
        self.status.configure(text=f"已检查 {sum(row.state == 'ready' for row in result)}/{len(result)} 台")
        for identity in self.rows:
            self.table.set(identity, "state", self._state(identity))
        first = next((identity for identity, row in self.rows.items() if
                      row.state != "ready" or quality_has_alert(self.summaries[identity])), next(iter(self.rows)))
        self.table.selection_set(first)
        self.table.focus(first)
        self.table.see(first)
        self.table.focus_set()
        self._select()

    def fail(self, error):
        self.working = False
        self.status.configure(text="画质检查未完成")
        self.detail.configure(text=ui_error(error))
        self._controls()

    def refresh(self, current, cameras, blocked):
        stale = frozenset(identity for identity, row in self.rows.items() if
                          row.target.binding is not None and identity not in current)
        self._cameras, self._blocked = cameras, blocked
        if stale != self._stale:
            changed, self._stale = stale ^ self._stale, stale
            for identity in changed:
                self.table.set(identity, "state", self._state(identity))
            self._select()
        self._controls()

    def _state(self, identity):
        if identity in self._stale:
            return "需重新检查"
        row = self.rows[identity]
        if row.state != "ready":
            return _STATES[row.state]
        summary = self.summaries[identity]
        labels = [name for key, name in (("saturated_frames", "近饱和"), ("dark_frames", "近黑"),
                                        ("texture_alerts", "纹理")) if summary[key]]
        if len(labels) > 1:
            return "多项提醒"
        return f"{labels[0]}提醒" if labels else "已检查"

    def _describe(self, identity):
        if identity not in self.rows:
            return "选择相机可查看检查结果。"
        row, summary = self.rows[identity], self.summaries[identity]
        lines = []
        if identity in self._stale or row.state == "changed":
            lines.append("连接、参数或检查框已变化。请重新检查。")
        elif row.state == "timeout":
            lines.append(f"等到 {summary['frames']}/{PREVIEW_CHECK_FRAMES} 张新画面。请确认相机仍在送图。")
        elif row.state == "failed" and row.detail == "ROI_OUT_OF_BOUNDS":
            lines.append("检查框超出当前画面。请清除或重新绘制。")
        elif row.detail:
            lines.append(row.detail)
        if details := preview_quality_detail(summary):
            lines.append(details)
        return "\n".join(lines)

    def _table_help(self, event):
        identity = self._selected()
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            identity = self.table.identify_row(event.y)
        return self._describe(identity)

    def _selected(self):
        return next(iter(self.table.selection()), "")

    def _select(self, _event=None):
        self.detail.configure(text=self._describe(self._selected()))
        self.detail_area.canvas.yview_moveto(0)
        self._controls()

    def _controls(self):
        self.measure_button.state(["disabled" if self.working or self._blocked else "!disabled"])
        self.open_button.state(["!disabled" if self._selected() in self._cameras else "disabled"])

    def _open_help(self, _event):
        identity = self._selected()
        if not identity:
            return "先在表格里选择相机。"
        if identity not in self._cameras:
            return "请先连接这台相机的预览。"
        return "打开所选相机的画质页，可调整照明、检查框，或使用曝光建议。"

    def _open(self):
        identity = self._selected()
        if identity in self._cameras and self.open_camera(identity):
            self.top.withdraw()

    def _escape(self, _event):
        if not ToolTip.dismiss_window(self.top):
            self.top.destroy()
        return "break"

    def _destroyed(self, event):
        if event.widget is self.top:
            self.cancelled.set()
