"""Edit the physical trigger input and shared pulse anchor for each camera."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import focus_canvas_child, localized_combobox
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .cameras.base import is_physical_trigger_input
from .config import HARDWARE_COUNTER_FEATURES, CaptureError, HardwarePairing, nonempty_text
from .presentation import ui_error
from .pulse_audit import pulse_integer
from .viewport import repaint_after_resize


_COUNTER_HELP = {
    "ChunkCounterValue": "使用相机在每张图像中附带的所选计数器值。请在设备中确认计数事件来自对应脉冲。",
    "ChunkFramecounter": "使用相机附带的帧计数器。先用独立脉冲记录核实帧计数与脉冲编号的对应关系。",
    "ChunkFrameID": "使用相机附带的帧标识。确认在本次阶段中随共享脉冲逐一递增。",
}
_COUNTER_LABELS = {"ChunkCounterValue": "相机触发计数值", "ChunkFramecounter": "相机帧计数", "ChunkFrameID": "相机帧标识"}
_INPUT_HELP = "按实际接线选择输入。如果需要关联图像，请填入同一次已知脉冲在各台相机上的计数值。"
_PORT_OPTION_HELP = "这是相机报告的触发输入。请选择实际接线使用的端口，再用试采确认能收到图像。"


class HardwareTriggerDialog:
    def __init__(self, parent, cameras, source, pairing, read_ports, accept):
        self.parent, self.cameras = parent, cameras
        self.read_ports, self.accept = read_ports, accept
        self.support = {}
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("物理触发与脉冲编号")
        self.top.transient(parent)
        self.top.geometry("720x530")
        self.top.minsize(575, 475)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        repaint_after_resize(self.top)
        self.per_camera = tk.BooleanVar(self.top, isinstance(source, dict))
        self.same_source = tk.StringVar(self.top, source if isinstance(source, str) else "")
        self.sources = {camera: tk.StringVar(self.top, source.get(camera, "") if isinstance(source, dict) else "")
                        for camera in cameras}
        self.pair = tk.BooleanVar(self.top, pairing is not None)
        data = pairing or {}
        self.source_id = tk.StringVar(self.top, data.get("source_id", ""))
        self.event = tk.StringVar(self.top, str(data.get("first_event_number", "")))
        self.counter_feature = tk.StringVar(self.top, data.get("counter_feature", HARDWARE_COUNTER_FEATURES[0]))
        self.counters = {camera: tk.StringVar(self.top, str(data.get("first_counter", {}).get(camera, ""))) for camera in cameras}
        self.note = tk.StringVar(self.top, _INPUT_HELP)
        self._port_entries, self._counter_entries = {}, {}
        self._build()
        self._refresh()
        for variable in (self.same_source, self.per_camera, self.pair, self.event, self.source_id, self.counter_feature,
                         *self.sources.values(), *self.counters.values()):
            variable.trace_add("write", self._input_changed)
        self.top.grab_set()

    def _build(self):
        layout = ttk.Frame(self.top, padding=14)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(2, weight=1)
        wiring = ttk.Frame(layout)
        wiring.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        wiring.columnconfigure(1, weight=1)
        option = ttk.Checkbutton(wiring, text="逐台指定输入", variable=self.per_camera, command=self._toggle_ports)
        option.grid(row=0, column=0, sticky="w", pady=(0, 5))
        ToolTip(option, "各台相机使用不同的输入时勾选。例如第一台接 Line1，第二台接 Line2。")
        self.probe = ttk.Button(wiring, text="读取设备端口…", command=self._probe, style="Small.TButton")
        self.probe.grid(row=0, column=2, sticky="e", pady=(0, 5))
        ToolTip(self.probe, lambda _event: "正在读取端口，请等待结果。" if self.probe.instate(["disabled"]) else
                "读取相机支持的帧开始触发（FrameStart）输入，便于按现场接线选择端口。")
        self.shared_label = ttk.Label(wiring, text="全部相机使用的输入")
        self.shared_label.grid(row=1, column=0, sticky="w", padx=(0, 10))
        self.shared_input = ttk.Combobox(wiring, textvariable=self.same_source, width=22)
        self.shared_input.grid(row=1, column=1, columnspan=2, sticky="ew")
        shared_help = ChoiceToolTip(self.shared_input, self._shared_help, {}, title="全部相机使用的输入")
        ToolTip(self.shared_label, shared_help.resolve_text)

        pairing = ttk.Frame(layout)
        pairing.grid(row=1, column=0, sticky="ew")
        pairing.columnconfigure(1, weight=1)
        pairing.columnconfigure(3, weight=1)
        option = ttk.Checkbutton(pairing, text="按共享脉冲编号关联图像", variable=self.pair, command=self._refresh)
        option.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))
        ToolTip(option, "多个相机接收同一个物理脉冲时，可用数据采集设备（DAQ）记录的脉冲编号关联图像。"
                "请先确认同一脉冲在各台相机图像中的计数值。")
        source_label = ttk.Label(pairing, text="信号源编号")
        source_label.grid(row=1, column=0, padx=(0, 8), sticky="w")
        self.source_entry = ttk.Entry(pairing, textvariable=self.source_id, width=15)
        self.source_entry.grid(row=1, column=1, sticky="ew", padx=(0, 12))
        source_tip = ToolTip(self.source_entry, lambda _event: self._pairing_help(
            "填入记录这组脉冲的数据采集设备或信号的编号，例如 DAQ-A。请和脉冲 CSV 中的编号保持一致。"), title="信号源编号")
        ToolTip(source_label, source_tip.resolve_text)
        event_label = ttk.Label(pairing, text="已知脉冲编号")
        event_label.grid(row=1, column=2, padx=(0, 8), sticky="w")
        self.event_entry = ttk.Entry(pairing, textvariable=self.event, width=19)
        self.event_entry.grid(row=1, column=3, sticky="ew")
        event_tip = ToolTip(self.event_entry, lambda _event: self._pairing_help(
            "填入本阶段开始时已知的共享脉冲编号，使用独立脉冲记录中的非负整数。"), title="已知脉冲编号")
        ToolTip(event_label, event_tip.resolve_text)
        counter_label = ttk.Label(pairing, text="帧附带的计数器")
        counter_label.grid(row=2, column=0, padx=(0, 8), pady=6, sticky="w")
        self.counter_menu = localized_combobox(pairing, self.counter_feature, _COUNTER_LABELS, width=22)
        self.counter_menu.grid(row=2, column=1, columnspan=3, sticky="ew", pady=6)
        counter_tip = ChoiceToolTip(self.counter_menu, lambda _event: self._pairing_help(
            "选择每台相机都随图像记录、且能与共享脉冲对应的计数器。"),
            {label: f"{_COUNTER_HELP[key]}\n设备字段 {key}" for key, label in _COUNTER_LABELS.items()})
        ToolTip(counter_label, counter_tip.resolve_text)

        table = ttk.Frame(layout)
        table.grid(row=2, column=0, sticky="nsew", pady=(5, 0))
        canvas = tk.Canvas(table, height=94, highlightthickness=0, background=ttk.Style(table).lookup("TFrame", "background"))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar = ttk.Scrollbar(table, orient=tk.VERTICAL, command=canvas.yview)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=bar.set)
        content = ttk.Frame(canvas, padding=(5, 0, 5, 0))
        win = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(win, width=event.width))
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        content.columnconfigure(0, weight=1)
        for column, label in enumerate(("相机", "物理输入", "该脉冲的计数值")):
            ttk.Label(content, text=label, style="CaptureMuted.TLabel").grid(row=0, column=column, sticky="w",
                                                                            padx=6 if column == 1 else 0, pady=(0, 4))

        def scroll(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        canvas.bind("<MouseWheel>", scroll)
        for row, (camera, label) in enumerate(self.cameras.items(), 1):
            name = ttk.Label(content, text=label, wraplength=205, justify=tk.LEFT)
            name.grid(row=row, column=0, sticky="w", pady=4)
            ToolTip(name, f"相机编号 {camera}")
            port = ttk.Combobox(content, textvariable=self.sources[camera], width=14)
            port.grid(row=row, column=1, padx=(6, 12), sticky="ew")
            self._port_entries[camera] = port
            ChoiceToolTip(port, lambda _event, key=camera: self._port_help(key), {}, title="这台相机的输入")
            counter = ttk.Entry(content, textvariable=self.counters[camera], width=19)
            counter.grid(row=row, column=2, sticky="ew")
            self._counter_entries[camera] = counter
            ToolTip(counter, lambda _event: "填写这台相机在上方已知脉冲的原图中附带的计数值。" if self.pair.get() else
                    "启用“按共享脉冲编号关联图像”后，可填写这台相机在已知脉冲上的计数值。", title="相机计数值")
            for widget in (name, port, counter):
                widget.bind("<MouseWheel>", scroll)
                widget.bind("<FocusIn>", lambda _event, item=widget: focus_canvas_child(item, canvas, content), add="+")

        note = ttk.Label(layout, textvariable=self.note, justify=tk.LEFT, style="CaptureMuted.TLabel")
        note.grid(row=3, column=0, sticky="ew", pady=(8, 6))
        note.bind("<Configure>", lambda event: note.configure(wraplength=max(1, event.width - 4)))
        actions = ttk.Frame(layout)
        actions.grid(row=4, column=0, sticky="e")
        cancel = ttk.Button(actions, text="取消", command=self.close, style="Small.TButton")
        cancel.pack(side=tk.LEFT, padx=(0, 8))
        ToolTip(cancel, "回到采集阶段，保留打开此窗口前的设置。")
        self.apply = ttk.Button(actions, text="应用外触发设置", command=self._apply, style="Record.TButton")
        self.apply.pack(side=tk.LEFT)
        ToolTip(self.apply, lambda _event: "设备端口读取完成后即可应用。" if self.apply.instate(["disabled"]) else
                "将输入和已填写的脉冲编号带回阶段编辑器。点击“应用阶段”后写入当前计划。")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.close())

    def _toggle_ports(self):
        if self.per_camera.get():
            for variable in self.sources.values():
                if not variable.get():
                    variable.set(self.same_source.get())
        elif not self.same_source.get():
            values = {variable.get() for variable in self.sources.values()}
            if len(values) == 1:
                self.same_source.set(next(iter(values)))
        self._refresh()

    def _refresh(self):
        for widget in (self.shared_label, self.shared_input):
            if self.per_camera.get():
                widget.grid_remove()
            else:
                widget.grid()
        for entry in self._port_entries.values():
            entry.state(["!disabled" if self.per_camera.get() else "disabled"])
        for entry in (self.source_entry, self.event_entry, self.counter_menu, *self._counter_entries.values()):
            entry.state(["!disabled" if self.pair.get() else "disabled"])

    def _input_changed(self, *_args):
        if self.apply.instate(["!disabled"]):
            self.note.set(_INPUT_HELP)

    def _pairing_help(self, description):
        return description if self.pair.get() else "如需关联图像，先勾选“按共享脉冲编号关联图像”。"

    def _shared_help(self, _event):
        values = self.shared_input.cget("values")
        return ("填写各台相机共同使用的输入名称。" +
                (f"已读取的共同选项 {'、'.join(values)}。" if values else "可读取设备目录，或按现场接线填写。"))

    def _port_help(self, camera):
        if not self.per_camera.get():
            return "当前使用上方的统一输入。勾选“逐台指定输入”后可修改这一台。"
        if camera in self.support:
            record = self.support[camera]
            if record.hardware is False:
                return "设备报告缺少物理触发支持。可检查设备、驱动或采集模式。"
            if record.physical_inputs is not None:
                choices = "、".join(record.physical_inputs) or "未列出物理输入"
                return f"当前 FrameStart 目录 {choices}。请按接线选择。"
        return "填写实际接线使用的输入名称。可点“读取设备端口”查询可用项。"

    def _probe(self):
        self.probe.state(["disabled"])
        self.apply.state(["disabled"])
        self.note.set("正在读取相机的 FrameStart 输入目录…")
        self.read_ports(self._ports_loaded, self._ports_failed)

    def _ports_loaded(self, support):
        if not self.top.winfo_exists():
            return
        self.support = support
        known = []
        for camera, entry in self._port_entries.items():
            record = support.get(camera)
            values = record.physical_inputs if record else None
            entry.configure(values=values or ())
            entry._tooltip.set_choices(dict.fromkeys(values or (), _PORT_OPTION_HELP))
            if values is not None:
                known.append(set(values))
        shared = sorted(set.intersection(*known)) if known else ()
        self.shared_input.configure(values=shared)
        self.shared_input._tooltip.set_choices(dict.fromkeys(shared, _PORT_OPTION_HELP))
        self.probe.state(["!disabled"])
        self.apply.state(["!disabled"])
        self.note.set(f"已读到 {len(known)}/{len(self.cameras)} 台的 FrameStart 输入目录。其余按接线填写，并用试采核对。")

    def _ports_failed(self, exc):
        if self.top.winfo_exists():
            self.probe.state(["!disabled"])
            self.apply.state(["!disabled"])
            self.note.set(ui_error(exc))

    def _apply(self):
        try:
            requested = {camera: self.sources[camera] if self.per_camera.get() else self.same_source for camera in self.cameras}
            sources = {}
            for camera, variable in requested.items():
                field = self._port_entries[camera] if self.per_camera.get() else self.shared_input
                source = variable.get().strip()
                label = self.cameras[camera]
                if not source or (camera.partition(":")[0] in ("pylon", "gentl") and not is_physical_trigger_input(source)):
                    field.focus_set()
                    raise CaptureError("HARDWARE_INPUT_INVALID", f"请填写 {label} 实际接线的物理输入，例如 Line1。")
                sources[camera] = nonempty_text(source, "hardware trigger input")
            selected = sources if self.per_camera.get() else next(iter(sources.values()))
            pairing = self._pairing() if self.pair.get() else None
            self.accept(selected, pairing)
            self.close()
        except CaptureError as exc:
            self.note.set(str(exc))

    def _pairing(self):
        if not (source_id := self.source_id.get().strip()):
            self.source_entry.focus_set()
            raise CaptureError("HARDWARE_ANCHOR_INVALID", "请填写用于记录此脉冲的 DAQ 信号源编号。")
        number = pulse_integer(self.event.get())
        if number is None:
            self.event_entry.focus_set()
            raise CaptureError("HARDWARE_ANCHOR_INVALID", "已知脉冲编号须为非负整数。")
        counters = {}
        for camera, variable in self.counters.items():
            value = pulse_integer(variable.get())
            if value is None:
                self._counter_entries[camera].focus_set()
                raise CaptureError("HARDWARE_ANCHOR_INVALID", f"请填写 {self.cameras[camera]} 在该脉冲的非负整数计数值。")
            counters[camera] = value
        return HardwarePairing.from_dict({"source_id": source_id, "counter_feature": self.counter_feature.get(),
                                          "first_event_number": number, "first_counter": counters}).to_dict()

    def close(self):
        self.top.destroy()
        if self.parent.winfo_exists():
            self.parent.grab_set()
