"""Select and preview the two confirmed pulse anchors of a saved hardware phase."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from ezdic_gui.controls import focus_canvas_child, localized_combobox
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .config import HARDWARE_COUNTER_FEATURES, CaptureError
from .presentation import ui_error
from .pulse_audit import pulse_integer
from .pulse_binding import PULSE_BINDING_SCHEMA, PulseBinding
from .trigger_review import TriggerReview, counter_review_snapshot
from .viewport import repaint_after_resize


_INPUT_HINT = "填写后可预览两组原图，并核对各台相机的原始计数值。"


class PulseBindingDialog:
    def __init__(self, parent, experiment, experiment_id, phase, camera_names, submit, accept, *, previous=None):
        self.experiment, self.experiment_id, self.phase = experiment, experiment_id, phase
        self.cameras, self.submit, self.accept = camera_names, submit, accept
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("采后补配锚点")
        self.top.transient(parent)
        self.top.geometry("735x530")
        self.top.minsize(600, 465)
        repaint_after_resize(self.top)
        self.values = {key: tk.StringVar(self.top) for key in ("source_id", "counter_feature", "first_event_number", "last_event_number", "evidence")}
        self.values["counter_feature"].set(HARDWARE_COUNTER_FEATURES[0])
        self.sequences = {camera: (tk.StringVar(self.top), tk.StringVar(self.top)) for camera in self.cameras}
        self.note = tk.StringVar(self.top, _INPUT_HINT)
        self._build()
        if previous is not None:
            self._fill(previous)
        for variable in (*self.values.values(), *(value for pair in self.sequences.values() for value in pair)):
            variable.trace_add("write", lambda *_args: self.note.set(_INPUT_HINT))
        self.top.grab_set()

    def _build(self):
        panel = ttk.Frame(self.top, padding=14)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)
        ttk.Label(panel, text=f"阶段 {self.phase} · 按现场记录确认的两个脉冲和对应原图填写。",
                  style="CaptureMuted.TLabel", wraplength=555).grid(row=0, column=0, sticky="w", pady=(0, 8))
        fields = ttk.Frame(panel)
        fields.grid(row=1, column=0, sticky="ew")
        fields.columnconfigure(1, weight=1)
        fields.columnconfigure(3, weight=1)
        ttk.Label(fields, text="DAQ 信号源").grid(row=0, column=0, sticky="w", padx=(0, 8))
        source = ttk.Entry(fields, textvariable=self.values["source_id"], width=12)
        source.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        ToolTip(source, "填写独立 DAQ 记录中的信号源编号。", title="DAQ 信号源")
        ttk.Label(fields, text="帧附带计数器").grid(row=0, column=2, sticky="w", padx=(0, 8))
        counter = localized_combobox(fields, self.values["counter_feature"], {key: key for key in HARDWARE_COUNTER_FEATURES}, width=20)
        counter.grid(row=0, column=3, sticky="ew")
        ChoiceToolTip(counter, "选择各台在原图元数据中记录、且经现场确认随同一共享脉冲递增的计数器。",
                      {key: "可从锚点原图的悬停信息核对这个字段的值。" for key in HARDWARE_COUNTER_FEATURES})
        for index, (label, key) in enumerate((("起点脉冲", "first_event_number"), ("终点脉冲", "last_event_number"))):
            ttk.Label(fields, text=label).grid(row=1, column=2 * index, sticky="w", padx=(0, 8), pady=7)
            entry = ttk.Entry(fields, textvariable=self.values[key], width=22)
            entry.grid(row=1, column=2 * index + 1, sticky="ew", padx=(0, 10) if index == 0 else 0)
            ToolTip(entry, "填写 DAQ 记录中一个独立确认的整数编号。核对范围含两个端点。", title=label)

        table = ttk.Frame(panel)
        table.grid(row=2, column=0, sticky="nsew", pady=(5, 10))
        canvas = tk.Canvas(table, height=105, highlightthickness=0, background=ttk.Style(self.top).lookup("TFrame", "background"))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(table, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        rows = ttk.Frame(canvas, padding=(3, 0, 5, 0))
        view = canvas.create_window((0, 0), window=rows, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(view, width=event.width))
        rows.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        rows.columnconfigure(0, weight=1)

        def scroll(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        canvas.bind("<MouseWheel>", scroll)
        labels = ("相机", "起点帧序号", "终点帧序号")
        for column, label in enumerate(labels):
            ttk.Label(rows, text=label, style="CaptureMuted.TLabel").grid(row=0, column=column, sticky="w", padx=4, pady=(0, 5))
        for index, (camera, label) in enumerate(self.cameras.items(), 1):
            name = ttk.Label(rows, text=label, wraplength=230, justify=tk.LEFT)
            name.grid(row=index, column=0, sticky="w", pady=4)
            name.bind("<MouseWheel>", scroll)
            ToolTip(name, f"相机编号 {camera}")
            for column, variable in enumerate(self.sequences[camera], 1):
                entry = ttk.Entry(rows, textvariable=variable, width=17)
                entry.grid(row=index, column=column, padx=4, pady=4, sticky="ew")
                ToolTip(entry, f"填写 {label} 在{labels[column][:2]}脉冲对应的已保存图像序号。", title=labels[column])
                entry.bind("<FocusIn>", lambda _event, item=entry: focus_canvas_child(item, canvas, rows), add="+")
                entry.bind("<MouseWheel>", scroll)

        evidence = ttk.Frame(panel)
        evidence.grid(row=3, column=0, sticky="ew")
        evidence.columnconfigure(1, weight=1)
        ttk.Label(evidence, text="确认依据").grid(row=0, column=0, sticky="w", padx=(0, 8))
        note = ttk.Entry(evidence, textvariable=self.values["evidence"])
        note.grid(row=0, column=1, sticky="ew")
        ToolTip(note, "说明如何独立确认两次脉冲及各台对应原图，例如 DAQ 标记和原图中的同一标记变化。", title="确认依据")
        hint = ttk.Label(panel, textvariable=self.note, style="CaptureMuted.TLabel", justify=tk.LEFT)
        hint.grid(row=4, column=0, sticky="ew", pady=(8, 7))
        hint.bind("<Configure>", lambda event: hint.configure(wraplength=max(1, event.width - 4)))
        actions = ttk.Frame(panel)
        actions.grid(row=5, column=0, sticky="ew")
        buttons = (("载入 JSON…", self._load, "可使用本实验已导出的核对报告或独立锚点文件。"),
                   ("预览两组原图…", self._preview, "校验并打开指定序号的原图，可比对现场标记和帧计数器。"),
                   ("取消", self.top.destroy, "返回核对窗口。"),
                   ("使用两组锚点", self._apply, "将当前输入用于核对。开始核对时检查两端原图和计数变化。"))
        for index, (label, command, help_text) in enumerate(buttons):
            action = ttk.Button(actions, text=label, command=command, style="Record.TButton" if index == 3 else "Small.TButton")
            action.pack(side=tk.RIGHT if index == 3 else tk.LEFT, padx=(0, 5))
            ToolTip(action, help_text)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())

    def _read(self, *, preview=False):
        data = {key: widget.get().strip() for key, widget in self.values.items()}
        if not data["source_id"]:
            raise CaptureError("PULSE_BINDING_INVALID", "请填写 DAQ 记录使用的信号源编号。")
        for key in ("first_event_number", "last_event_number"):
            data[key] = pulse_integer(data[key])
            if data[key] is None:
                raise CaptureError("PULSE_BINDING_INVALID", "起点和终点脉冲须用非负整数填写。")
        if not data["evidence"]:
            if not preview:
                raise CaptureError("PULSE_BINDING_INVALID", "请简述确认两次脉冲与各台原图所用的现场记录。")
            data["evidence"] = "原图预览"
        return PulseBinding.from_dict({**data, "schema": PULSE_BINDING_SCHEMA, "experiment_id": self.experiment_id, "phase": self.phase,
                                       "camera_frames": {camera: [pulse_integer(value.get()) for value in pair]
                                                         for camera, pair in self.sequences.items()}})

    def _fill(self, binding):
        if (binding.experiment_id, binding.phase, binding.camera_frames.keys()) != (self.experiment_id, self.phase, self.sequences.keys()):
            raise CaptureError("PULSE_BINDING_SCOPE", "文件须对应本实验、本阶段和全部相机。")
        payload = binding.to_dict()
        for key, variable in self.values.items():
            variable.set(str(payload[key]))
        for camera, pair in self.sequences.items():
            for variable, value in zip(pair, binding.camera_frames[camera]):
                variable.set(str(value))

    def _load(self):
        path = filedialog.askopenfilename(parent=self.top, filetypes=[("补配锚点或核对报告", "*.json")])
        if path:
            try:
                self._fill(PulseBinding.load(path))
                self.note.set("已载入，可预览原图后用于核对。")
            except (CaptureError, OSError, ValueError) as exc:
                self.note.set(ui_error(exc))

    def _preview(self):
        try:
            binding = self._read(preview=True)
        except CaptureError as exc:
            self.note.set(ui_error(exc))
            return
        groups = []
        for index, number in enumerate((binding.first_event_number, binding.last_event_number)):
            refs = [{"camera_id": camera, "sequence": pair[index]} for camera, pair in binding.camera_frames.items()]
            groups.append(counter_review_snapshot(binding.experiment_id, binding.phase, binding.source_id, number, self.cameras,
                                                  refs, binding.counter_feature, status="associated", preview=True))
        viewer = TriggerReview(self.top, self.experiment, self.cameras, self.submit, snapshots=groups)
        viewer.top.transient(self.top)
        viewer.top.grab_set()
        viewer.top.bind("<Destroy>", lambda event: self.top.grab_set() if event.widget == viewer.top and self.top.winfo_exists() else None, add="+")

    def _apply(self):
        try:
            binding = self._read()
        except CaptureError as exc:
            self.note.set(ui_error(exc))
            return
        self.top.destroy()
        self.accept(binding)
