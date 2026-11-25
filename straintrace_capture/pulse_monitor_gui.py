"""Show which cameras have received and registered each anchored hardware pulse."""

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .presentation import WAIT_OUTPUT_LABELS
from .viewport import repaint_after_resize


def pulse_notice(pulses):
    if not pulses or not pulses["accepting"]:
        return None
    if any(row["unidentified"] or row["backtracks"] for row in pulses["cameras"].values()):
        return "核对脉冲编号", "有图像无法关联编号，或收到较小编号。可逐台核对计数器与接线。"
    for group in pulses["groups"]:
        if group["duplicates"]:
            return "脉冲编号重复", f"脉冲 {group['number']} 有相机收到多帧。可逐台核对。"
        if group["advanced"]:
            return "脉冲组需复核", f"有相机已收到比 {group['number']} 更大的编号，这一组尚未收齐。"
    return None


def pulse_help(pulses):
    if not pulses:
        return "在硬件触发阶段配置共享脉冲编号和各台相机的计数器锚点后可查看。"
    if notice := pulse_notice(pulses):
        return notice[1]
    return "逐个脉冲查看哪些相机已收图，以及 TIFF 是否已写入并登记。"


def _group_state(group, pulses):
    if group["duplicates"]:
        return "编号重复"
    if group["advanced"]:
        return "更大编号已到"
    if group["saved_cameras"] == len(pulses["cameras"]):
        return "已登记齐"
    if group["received_cameras"] == len(pulses["cameras"]):
        return "已收齐"
    return "等收图" if pulses["accepting"] else "未收齐"


def _camera_state(camera, group, pulses, wait_state):
    if counts := group["cameras"].get(camera):
        if counts["received"] > 1:
            return f"重复收到 {counts['received']} 帧"
        return "已登记" if counts["saved"] else "已收 · 未登记"
    if camera in group["advanced"]:
        return "更大编号已到"
    if not pulses["accepting"]:
        return "未见此编号"
    status = {"ready": "已就绪 · 等收图", "not_ready": "相机报告未就绪"}.get(wait_state)
    return status or "等待收图"


def _table(panel, row, columns, *, height):
    frame = ttk.Frame(panel)
    frame.grid(row=row, sticky="nsew", pady=(0, 6))
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    keys = [col[0] for col in columns]
    table = ttk.Treeview(frame, columns=keys, show="headings", height=height, selectmode="browse")
    for key, heading, width, minimum in columns:
        table.heading(key, text=heading)
        table.column(key, width=width, minwidth=minimum)
    table.grid(row=0, column=0, sticky="nsew")
    scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=table.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    table.configure(yscrollcommand=scroll.set)
    ToolTip(scroll, "滚动查看其余记录。")
    return table


def _target_row(table, event):
    if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
        return table.identify_row(event.y)
    selected = table.selection()
    return selected[0] if selected else ""


class PulseMonitorWindow:
    def __init__(self, parent, camera_title):
        self.camera_title = camera_title
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("StrainTrace · 脉冲监看")
        self.top.geometry("690x540")
        self.top.minsize(580, 425)
        self.top.transient(parent)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        repaint_after_resize(self.top)
        self._pulses = None
        self._wait = {}
        self._rows = {}
        self.heading = tk.StringVar(self.top)
        self.note = tk.StringVar(self.top)
        panel = ttk.Frame(self.top, padding=12)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=3, minsize=93)
        panel.rowconfigure(3, weight=2, minsize=89)
        heading = ttk.Label(panel, width=1, textvariable=self.heading, style="CaptureSection.TLabel")
        heading.grid(row=0, sticky="ew", pady=(0, 8))
        ToolTip(heading, self._scope_help, title="关联依据")
        self.groups = _table(panel, 1, (
            ("number", "脉冲编号", 226, 155), ("received", "已收图", 98, 72),
            ("saved", "已登记", 98, 72), ("state", "状态", 160, 135),
        ), height=5)
        ToolTip(self.groups, self._group_help, title="按脉冲查看")
        self.groups.bind("<<TreeviewSelect>>", self._select)
        detail = ttk.Label(panel, textvariable=self.note, width=1, justify=tk.LEFT, style="CaptureMuted.TLabel")
        detail.grid(row=2, sticky="ew", pady=(0, 5))
        wrap_label_on_resize(detail)
        ToolTip(detail, lambda _event: self.note.get(), title="所选脉冲")
        self.members = _table(panel, 3, (
            ("camera", "相机", 163, 95), ("state", "本组状态", 213, 155), ("latest", "已收最大编号", 216, 138),
        ), height=3)
        ToolTip(self.members, self._camera_help, title="逐台收图依据")
        close = ttk.Button(panel, text="关闭", style="Small.TButton", command=self.top.destroy)
        close.grid(row=4, sticky="e", pady=(2, 0))
        ToolTip(close, "返回采集工作台。")
        self.groups.focus_set()

    def render(self, pulses, wait):
        if pulses == self._pulses and wait == self._wait:
            return
        old_phase = self._pulses["phase"] if self._pulses else None
        selected = _target_row(self.groups, None) if old_phase == pulses["phase"] else ""
        self._pulses = pulses
        self._wait = wait
        suffix = "监看中" if pulses["accepting"] else "收图已结束"
        self.heading.set(f"阶段 {pulses['phase']} · {pulses['source_id']} · {suffix}")
        rows = self._rows = {str(group["number"]): group for group in pulses["groups"]}
        for key in self.groups.get_children():
            if key not in rows:
                self.groups.delete(key)
        total = len(pulses["cameras"])
        for index, (key, group) in enumerate(rows.items()):
            label = f"{key} · 起始" if group["number"] == pulses["first_number"] else key
            values = (label, f"{group['received_cameras']}/{total}", f"{group['saved_cameras']}/{total}",
                      _group_state(group, pulses))
            if self.groups.exists(key):
                if self.groups.item(key, "values") != values:
                    self.groups.item(key, values=values)
                if self.groups.index(key) != index:
                    self.groups.move(key, "", index)
            else:
                self.groups.insert("", index, iid=key, values=values)
        item = selected if selected in rows else next(iter(rows))
        if item != _target_row(self.groups, None):
            self.groups.selection_set(item)
            self.groups.focus(item)
            self.groups.see(item)
        self._select()

    def _select(self, _event=None):
        key = _target_row(self.groups, None)
        if (group := self._rows.get(key)) is None:
            return
        pulses = self._pulses
        caption = _group_state(group, pulses)
        self.note.set(f"脉冲 {key} · {caption}")
        for camera, row in pulses["cameras"].items():
            largest = "待收图" if row["highest_number"] is None else str(row["highest_number"])
            values = (self.camera_title(camera), _camera_state(camera, group, pulses, self._wait.get(camera)), largest)
            if self.members.exists(camera):
                if self.members.item(camera, "values") != values:
                    self.members.item(camera, values=values)
            else:
                self.members.insert("", tk.END, iid=camera, values=values)
        for camera in self.members.get_children():
            if camera not in pulses["cameras"]:
                self.members.delete(camera)

    def _scope_help(self, _event):
        pulses = self._pulses
        return (f"使用 {pulses['counter_feature']} 的已配置锚点关联图像。"
                f"保留计划起始编号及后续观测中最大的 {pulses['recent_limit']} 个编号。"
                "物理漏拍和曝光同步用独立脉冲或曝光记录核实。")

    def _group_help(self, event):
        if (group := self._rows.get(_target_row(self.groups, event))) is None:
            return "选中脉冲，查看下方每台相机的收图和登记情况。"
        lines = [f"脉冲 {group['number']} · {_group_state(group, self._pulses)}",
                 "已收图按收到该编号的相机数统计。已登记表示这台相机至少一帧已保存并写入台账。"]
        if group["advanced"]:
            names = "、".join(self.camera_title(key) for key in group["advanced"])
            lines.append(f"{names}已收到更大的编号，尚未看到本组图像。")
        if group["duplicates"]:
            names = "、".join(self.camera_title(key) for key in group["duplicates"])
            lines.append(f"{names}在本编号已收到多帧。")
        return "\n".join(lines)

    def _camera_help(self, event):
        camera = _target_row(self.members, event)
        group = self._rows.get(_target_row(self.groups, None))
        if not camera or group is None:
            return "选中相机可查看计数器锚点、已收编号和本组登记情况。"
        row = self._pulses["cameras"][camera]
        counts = group["cameras"].get(camera, {})
        lines = [self.camera_title(camera),
                 f"本组收图 {counts.get('received', 0)} 帧 · 登记 {counts.get('saved', 0)} 帧",
                 f"起始计数器值 {row['first_counter']}"]
        if row["latest_number"] is not None:
            lines.append(f"最近收到的关联编号 {row['latest_number']}")
        if (wait := self._wait.get(camera)) and wait != "received":
            lines.append(f"相机等待输出 · {WAIT_OUTPUT_LABELS[wait]}")
        if row["unidentified"] or row["backtracks"]:
            lines.append(f"本阶段未关联编号 {row['unidentified']} 帧 · 收到比上一帧更小的编号 {row['backtracks']} 次")
        return "\n".join(lines)
