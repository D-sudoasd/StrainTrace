"""Preview and edit a capture template before replacing the planned phases."""

import math
import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import ScrollableForm, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .cameras.base import HOST_NEXT_RECEIPT
from .config import CaptureError, Phase
from .hardware_dialog import HardwareTriggerDialog
from .presentation import (
    DRIVER_HOST_FRAME_HELP, MODE_HELP, MODE_LABELS, PHASE_FIELD_HELP, PHASE_FIELDS, PHASE_FIELDS_BY_MODE, PHASE_PRESETS, ROLE_HELP,
    ROLE_LABELS, format_bytes, phase_limit_text, ui_error,
)
from .readiness import estimate_phase, event_window_shortfall
from .viewport import repaint_after_resize


_TEMPLATES = {**PHASE_PRESETS, "硬件触发记录": ({"mode": "hardware", "role": "test", "frame_count": 100},)}
TEMPLATE_HELP = {
    "参考帧和定时实验": "加载前先拍参考图，再由程序按设定频率请求实验图像。",
    "参考图后确认": "参考图登记完成后暂停。由操作者确认夹具和加载准备，再定时拍摄。",
    "连续记录": "持续保存相机送来的图像。实际帧率由各相机设置和传输条件决定。",
    "手动逐次记录": "由操作者逐次触发，每次请求一轮图像。适合分步加载或间隔不固定的实验。",
    "事件前后记录": "等待事件时滚动缓存图像。发出触发后保存事件前后两段。",
    "棋盘格标定": "每次摆稳棋盘格再触发。改变位置和倾角，随后可拟合镜头参数。",
    "硬件触发记录": "每次物理脉冲由接线传给相机。请指定实际输入，再用试采核对。",
}
_GEOMETRY_BASIS = {"preview": "打开时的预览", "simulator": "模拟相机参数", "configured": "计划设置",
                   "unavailable": "尺寸或格式待核对"}


def _signature(phases):
    return tuple((row["mode"], row["role"], bool(row.get("start_confirmation"))) for row in phases)


class PlanTemplateWindow:
    def __init__(self, parent, *, current, cameras, geometry, support, buffer_limit, external, read_ports, apply):
        self.cameras, self.geometry, self.support = cameras, geometry, support
        self.buffer_limit, self.external, self.read_ports, self.apply = buffer_limit, external, read_ports, apply
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("图像采集 · 采集模板")
        self.top.transient(parent)
        self.top.geometry("760x570")
        self.top.minsize(640, 480)
        repaint_after_resize(self.top)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())

        self._drafts = {name: [dict(phase) for phase in phases] for name, phases in _TEMPLATES.items()}
        name = next((key for key, phases in self._drafts.items() if _signature(phases) == _signature(current)), "")
        if name:
            self._drafts[name] = [dict(phase) for phase in current]
        self.preset = tk.StringVar(self.top, name or next(iter(_TEMPLATES)))
        self._values = {
            key: {field: tk.StringVar(self.top, str(phases[-1].get(field, "")))
                  for field in PHASE_FIELDS_BY_MODE[phases[-1]["mode"]]}
            for key, phases in self._drafts.items()
        }
        self._candidate = None
        self._blocked = ""
        self.estimate = tk.StringVar(self.top)
        self.note = tk.StringVar(self.top)
        self.entries = {}
        self._build()
        self.preset.trace_add("write", self._show_template)
        for fields in self._values.values():
            for field in fields.values():
                field.trace_add("write", self._evaluate)
        self._show_template()
        self.top.grab_set()
        self.menu.focus_set()

    def _build(self):
        layout = ttk.Frame(self.top, padding=14)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(0, weight=1)
        self.form = ScrollableForm(layout)
        self.form.grid(row=0, column=0, sticky="nsew")
        panel = self.form.body
        panel.columnconfigure(0, weight=1)
        chooser = ttk.Frame(panel)
        chooser.grid(row=0, column=0, sticky="ew")
        chooser.columnconfigure(1, weight=1)
        label = ttk.Label(chooser, text="按任务选择")
        label.grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.menu = ttk.Combobox(chooser, textvariable=self.preset, values=tuple(_TEMPLATES), state="readonly", width=22)
        self.menu.grid(row=0, column=1, sticky="ew")
        ChoiceToolTip(self.menu, "比较各模板的触发方式和阶段顺序。应用后将这一方案写入采集计划。",
                      TEMPLATE_HELP, title="采集模板")
        ToolTip(label, self.menu._tooltip.resolve_text)
        self.description = ttk.Label(panel, width=1, justify=tk.LEFT)
        self.description.grid(row=1, column=0, sticky="ew", pady=(8, 9))
        wrap_label_on_resize(self.description)
        self.fields = ttk.Frame(panel)
        self.fields.grid(row=2, column=0, sticky="ew")
        self.fields.columnconfigure((1, 3), weight=1)
        self.table = ttk.Treeview(panel, columns=("number", "mode", "role", "limits"), show="headings",
                                  selectmode="browse", height=2)
        for key, label, width, minimum in (("number", "阶段", 63, 63), ("mode", "拍摄方式", 180, 150),
                                            ("role", "用途", 85, 70), ("limits", "计划参数", 290, 180)):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=minimum, stretch=key in ("mode", "limits"))
        self.table.grid(row=3, column=0, sticky="ew", pady=(12, 6))
        ToolTip(self.table, self._stage_help, title="实际执行顺序")
        summary = ttk.Label(panel, textvariable=self.estimate, width=1, justify=tk.LEFT, style="CaptureSection.TLabel")
        summary.grid(row=4, column=0, sticky="ew", pady=(4, 3))
        wrap_label_on_resize(summary)
        ToolTip(summary, self._estimate_help, title="预计保存量")
        note = ttk.Label(panel, textvariable=self.note, width=1, justify=tk.LEFT, style="CaptureMuted.TLabel")
        note.grid(row=5, column=0, sticky="ew", pady=(2, 0))
        wrap_label_on_resize(note)
        ToolTip(note, lambda _event: self.note.get(), title="当前方案")
        ToolTip(self.form.scrollbar, "滚动查看阶段顺序、估算和提醒。")

        actions = ttk.Frame(layout)
        actions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        close = ttk.Button(actions, text="关闭", command=self.top.destroy, style="Small.TButton")
        close.pack(side=tk.RIGHT)
        ToolTip(close, "关闭模板窗口，保留当前采集计划。")
        self.apply_button = ttk.Button(actions, text="应用并替换阶段", command=self._apply, style="Record.TButton")
        self.apply_button.pack(side=tk.RIGHT, padx=(0, 7))
        ToolTip(self.apply_button, lambda _event: self._blocked or
                "用上方列出的阶段替换采集计划。然后可检查磁盘、触发条件或先试采。")

    def _show_template(self, *_args):
        name = self.preset.get()
        self.description.configure(text=TEMPLATE_HELP[name])
        for child in self.fields.winfo_children():
            child.destroy()
        self.entries.clear()
        for index, (key, variable) in enumerate(self._values[name].items()):
            row, column = divmod(index, 2)
            label = ttk.Label(self.fields, text=PHASE_FIELDS[key][0])
            label.grid(row=row, column=2 * column, sticky="w", padx=(0 if not column else 14, 7), pady=3)
            entry = ttk.Entry(self.fields, width=8, textvariable=variable)
            entry.grid(row=row, column=2 * column + 1, sticky="ew", pady=3)
            self.entries[key] = entry
            optional = {"frame_count": "留空时不设张数上限。", "duration_s": "留空时不设时长上限。"}.get(key, "")
            ToolTip(entry, PHASE_FIELD_HELP[key] + optional, title=PHASE_FIELDS[key][0])
            ToolTip(label, entry._tooltip.resolve_text)
        if self._drafts[name][-1]["mode"] == "hardware":
            button = ttk.Button(self.fields, text="物理输入与脉冲编号…", command=self._edit_hardware, style="Small.TButton")
            button.grid(row=2, column=0, columnspan=4, pady=(6, 0), sticky="w")
            button.state(["!disabled" if self.cameras else "disabled"])
            ToolTip(button, "填写实际接线和已确认的脉冲配对设置。" if self.cameras else "请先选择要接收物理脉冲的相机。")
        self.form.bind_controls()
        self._evaluate()

    def _read_phases(self, rows):
        last = rows[-1]
        for key, variable in self._values[self.preset.get()].items():
            value = variable.get().strip()
            label, parse = PHASE_FIELDS[key]
            if not value:
                last.pop(key, None)
                continue
            try:
                number = parse(value)
                valid = number >= 0 if key == "pre_frames" else number > 0
                if not valid or (parse is float and not math.isfinite(number)):
                    raise ValueError()
            except ValueError:
                requirement = "非负整数" if key == "pre_frames" else "正整数" if parse is int else "大于零的有限数值"
                raise ValueError(f"“{label}”请填写{requirement}。") from None
            last[key] = number
        mode = last["mode"]
        if mode in ("timed", "event") and "fps" not in last:
            raise ValueError("请填写定时频率。")
        if mode == "timed" and not (last.get("duration_s") or last.get("frame_count")):
            raise ValueError("请填写本阶段的张数或最长时长。")
        if mode == "event" and "post_frames" not in last:
            raise ValueError("请填写事件后张数。")
        if mode == "hardware" and not last.get("hardware_source"):
            raise ValueError("请先设置相机实际使用的物理输入。")
        if isinstance(last.get("hardware_source"), dict) and last["hardware_source"].keys() != self.cameras.keys():
            raise ValueError("逐台输入与所选相机不一致。请重新设置物理输入。")
        if (pairing := last.get("hardware_pairing")) and pairing["first_counter"].keys() != self.cameras.keys():
            raise ValueError("已设置的脉冲计数与所选相机不一致。请重新核对。")
        return [Phase.from_dict(row) for row in rows]

    def _evaluate(self, *_args):
        rows = [dict(row) for row in self._drafts[self.preset.get()]]
        self._candidate = None
        self._blocked = ""
        try:
            phases = self._read_phases(rows)
            if (rounds := event_window_shortfall(phases[-1])) is not None:
                raise ValueError(f"按时长和频率只能安排 {rounds} 轮。请增加时限、减少事件次数或减少每次前后张数。")
            budgets = [estimate_phase(phase, self.geometry) for phase in phases]
            self.estimate.set(self._estimate_label(budgets))
            self._candidate = [phase.to_dict() for phase in phases]
        except (CaptureError, ValueError, OverflowError) as exc:
            self._blocked = ("频率、时长或张数过大，请核对填写值。" if isinstance(exc, OverflowError) else
                             ui_error(exc) if isinstance(exc, CaptureError) else str(exc))
            self.estimate.set("")
        self.apply_button.state(["disabled" if self._blocked else "!disabled"])
        self.note.set(self._blocked or self._plan_note(phases, budgets))
        selected = self.table.selection()
        self.table.delete(*self.table.get_children())
        display = self._candidate or rows
        for number, row in enumerate(display, 1):
            mode = MODE_LABELS[row["mode"]] + (" · 等确认" if row.get("start_confirmation") else "")
            limits = "核对上方参数" if self._blocked and number == len(display) else phase_limit_text(row)
            self.table.insert("", tk.END, iid=str(number), values=(number, mode, ROLE_LABELS[row["role"]], limits))
        item = selected[0] if selected and self.table.exists(selected[0]) else self.table.get_children()[0]
        self.table.selection_set(item)
        self.table.focus(item)

    def _estimate_label(self, budgets):
        if not self.geometry:
            return "选择相机后可估算保存量"
        counts, bounded = [], True
        for row in budgets:
            if (limit := row["frame_limit_per_camera"]) is not None:
                counts.append(limit * len(self.geometry))
            else:
                bounded = False
                counts.append(row["frames_at_configured_rate"])
        parts = [f"{len(self.geometry)} 台相机"]
        if None not in counts:
            unit = "帧" if any(len(row.get("saved_parts", ())) > 1 for row in self.geometry.values()) else "张"
            parts.append(f"{'最多' if bounded else '约'} {sum(counts):,} {unit}")
        payloads = [row["raw_payload_bytes"] for row in budgets]
        parts.append("预计原始像素量待核对" if None in payloads else
                     f"原始像素 {format_bytes(sum(payloads), small_units=True)}")
        return " · ".join(parts)

    def _estimate_help(self, _event):
        lines = [f"{self.cameras[key]} · {_GEOMETRY_BASIS[row['basis']]}"
                 + (f" · {row['width_px']} × {row['height_px']} · {row['pixel_format']}" if row["frame_bytes"] else "")
                 for key, row in self.geometry.items()]
        lines.append("按每台相机的帧数估算。一帧可能包含多个保存的图像部件。TIFF 和台账另占空间。"
                     "实际收图与写入能力可通过试采确认。")
        return "\n".join(lines)

    def _stage_help(self, event):
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            item = self.table.identify_row(event.y)
        else:
            item = self.table.focus()
        if not item:
            return "阶段从上到下执行。选中一行后按 F1 查看该阶段的触发条件。"
        row = (self._candidate or self._drafts[self.preset.get()])[int(item) - 1]
        text = [MODE_HELP[row["mode"]], ROLE_HELP[row["role"]], self.table.item(item, "values")[3]]
        if row["mode"] == "event" and row.get("pre_frames"):
            text.append("保存量预留一张可能跨过触发时刻的边界图像。")
        if confirm := row.get("start_confirmation"):
            text.append(f"开始本阶段前需要确认。{confirm}")
        return "\n".join(text)

    def _plan_note(self, phases, budgets):
        notes = []
        if any(phase.camera_trigger == "software" for phase in phases):
            driver_inputs = [self.cameras[key] for key, evidence in self.support.items()
                             if evidence.software_method == HOST_NEXT_RECEIPT]
            if driver_inputs:
                notes.append(f"{'、'.join(driver_inputs)}按主机收图取帧。{DRIVER_HOST_FRAME_HELP}")
        for trigger, method in (("software", "软件"), ("hardware", "物理")):
            if not any(phase.camera_trigger == trigger and not phase.software_trigger_fallback for phase in phases):
                continue
            unsupported = [self.cameras[key] for key, evidence in self.support.items() if getattr(evidence, trigger) is False]
            if unsupported:
                notes.append(f"{'、'.join(unsupported)}不支持{method}触发。")
        if any(row["one_batch_raw_bytes"] is not None and row["one_batch_raw_bytes"] > self.buffer_limit for row in budgets):
            notes.append("预计一批图像或事件前缓存超过字节缓冲。请核对图像尺寸和事件前张数。")
        mode = phases[-1].mode
        if mode in ("event", "manual"):
            notes.append("用“触发一次”" + ("或外部软件命令触发。" if self.external else "触发。"))
        if mode == "hardware":
            source = phases[-1].hardware_source
            ports = "已指定逐台物理输入。" if not isinstance(source, str) else f"物理输入 {source}。"
            notes.insert(0, ports + "应用后请检查接线与相机回读。")
        if phases[-1].start_confirmation:
            notes.append("参考图登记后等待操作者确认，确认后开始实验阶段。")
        return "\n".join(notes)

    def _edit_hardware(self):
        row = self._drafts[self.preset.get()][-1]

        def accept(source, pairing):
            row["hardware_source"] = source
            if pairing is None:
                row.pop("hardware_pairing", None)
            else:
                row["hardware_pairing"] = pairing
            self._evaluate()

        HardwareTriggerDialog(self.top, self.cameras, row.get("hardware_source"), row.get("hardware_pairing"),
                              self.read_ports, accept)

    def _apply(self):
        if self._candidate is not None and self.apply(self._candidate):
            self.top.destroy()
