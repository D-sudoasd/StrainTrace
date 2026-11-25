"""Capture dialogs for phase configuration and camera features."""

import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from ezdic_gui.controls import ScrollableForm, focus_canvas_child, localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .calibration import CheckerboardTarget
from .cameras._genapi import PHASE_FEATURES, image_layout, is_packed_mono_or_bayer
from .cameras.base import Feature
from .cameras.settings_probe import check_camera_settings, check_simulator_settings
from .config import CameraConfig, CaptureError, Phase, hardware_mapping_error
from .csv_trigger_gui import CsvTriggerDialog
from .hardware_dialog import HardwareTriggerDialog
from .presentation import (
    EVENT_OVERLAP_HELP, FEATURE_HELP, MODE_HELP, MODE_LABELS, PHASE_FIELD_HELP, PHASE_FIELDS, PHASE_FIELDS_BY_MODE,
    ROLE_HELP, ROLE_LABELS,
    csv_trigger_help, csv_trigger_rule_text, ui_error,
)
from .viewport import repaint_after_resize


class _CaptureDialog(simpledialog.Dialog):
    def __init__(self, parent, title, prompt, *, confirm_text):
        self.prompt = prompt
        self.confirm_text = confirm_text
        super().__init__(parent, title=title)

    def body(self, master):
        background = ttk.Style(master).lookup("TFrame", "background")
        self.configure(background=background)
        master.configure(background=background)
        ttk.Label(master, text=self.prompt, wraplength=440, justify=tk.LEFT).pack(anchor="w", padx=15, pady=(12, 8))

    def buttonbox(self):
        row = ttk.Frame(self, padding=(15, 0, 15, 12))
        row.pack(fill=tk.X)
        confirm = ttk.Button(row, text=self.confirm_text, command=self.ok, style="Record.TButton")
        confirm.pack(side=tk.RIGHT)
        ToolTip(confirm, "确认当前输入或选项并继续。也可按回车。")
        cancel = ttk.Button(row, text="取消", command=self.cancel)
        cancel.pack(side=tk.RIGHT, padx=(0, 6))
        ToolTip(cancel, "回到上一步。也可按 Esc。")
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", lambda event: ToolTip.dismiss_window(self) or self.cancel(event))


class CaptureChoiceDialog(_CaptureDialog):
    def __init__(self, parent, title, prompt, choices, *, confirm_text="打开"):
        self.choices = choices
        super().__init__(parent, title, prompt, confirm_text=confirm_text)

    def body(self, master):
        super().body(master)
        self.menu = ttk.Combobox(master, values=tuple(self.choices), state="readonly", width=40)
        self.menu.current(0)
        self.menu.pack(fill=tk.X, padx=15, pady=(0, 8))
        ChoiceToolTip(self.menu, self.prompt, self.choices, title=self.title())
        return self.menu

    def apply(self):
        self.result = self.menu.get()


class CaptureFieldDialog(_CaptureDialog):
    def __init__(self, parent, title, prompt, fields, parse, error, *, confirm_text="应用"):
        self.fields = fields
        self.parse = parse
        self.error_text = error
        super().__init__(parent, title, prompt, confirm_text=confirm_text)

    def body(self, master):
        super().body(master)
        form = ttk.Frame(master, padding=(15, 0, 15, 8))
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)
        self.entries = []
        self.error = ttk.Label(form, style="CaptureMuted.TLabel", wraplength=440, justify=tk.LEFT)
        for row, (name, initial, help_text) in enumerate(self.fields):
            label = ttk.Label(form, text=name)
            label.grid(row=row, column=0, sticky="w", padx=(0, 10))
            entry = ttk.Entry(form, width=16)
            entry.insert(0, initial)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            ToolTip(entry, help_text, title=name)
            ToolTip(label, help_text)
            self.entries.append(entry)
        self.error.grid(row=len(self.fields), column=0, columnspan=2, sticky="ew")
        self.entries[0].selection_range(0, tk.END)
        return self.entries[0]

    def validate(self):
        try:
            self.result = self.parse(*(entry.get() for entry in self.entries))
        except (TypeError, ValueError):
            self.error.configure(text=self.error_text)
            return False
        return True


class CorrectedAnalysisDialog:
    def __init__(self, parent, camera_label, references, preferred_ordinal, accept):
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("校正后分析")
        self.top.transient(parent)
        self.top.minsize(460, 210)
        repaint_after_resize(self.top)
        self.top.grab_set()
        panel = ttk.Frame(self.top, padding=16)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text=camera_label, style="CaptureSection.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(panel, text="选择参考图。程序会校正这张图及之后的实验帧，然后打开离线分析。",
                  wraplength=455, justify=tk.LEFT, style="CaptureMuted.TLabel").grid(
                      row=1, column=0, columnspan=4, sticky="w", pady=(6, 12))

        choices = {str(row["sequence"]): f"序号 {row['sequence']} · 阶段 {row['phase']} · {ROLE_LABELS[row['role']]}"
                   for row in references}
        preferred = next((str(row["sequence"]) for row in references if row["ordinal"] == preferred_ordinal), next(iter(choices)))
        self.reference = tk.StringVar(self.top, preferred)
        self.reference_combo = localized_combobox(panel, self.reference, choices, width=30)
        ttk.Label(panel, text="参考图").grid(row=2, column=0, sticky="w", padx=(0, 10))
        self.reference_combo.grid(row=2, column=1, columnspan=3, sticky="ew")
        ChoiceToolTip(self.reference_combo, "选择加载前或基准状态的一帧。",
                      {label: "用作校正序列的第 1 张，后续图像与它比较。" for label in choices.values()}, title="参考图")

        self.lens = None
        self.lens_name = tk.StringVar(self.top, "自动匹配")
        ttk.Label(panel, text="镜头标定").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=(8, 0))
        lens = ttk.Label(panel, textvariable=self.lens_name, wraplength=220, justify=tk.LEFT)
        lens.grid(row=3, column=1, sticky="w", pady=(8, 0))
        ToolTip(lens, lambda _event: str(self.lens) if self.lens else
                "优先使用实验归档中与所选图像匹配的标定，也会检查已载入的文件。", title="镜头标定")
        select = ttk.Button(panel, text="选择文件…", command=self._choose_profile, style="Small.TButton")
        select.grid(row=3, column=2, padx=(8, 4), pady=(8, 0))
        ToolTip(select, "指定已拟合的镜头标定 JSON。所选文件需匹配这一组图像。")
        reset = ttk.Button(panel, text="恢复自动", command=lambda: self._set_profile(None), style="Small.TButton")
        reset.grid(row=3, column=3, pady=(8, 0))
        ToolTip(reset, "恢复从实验归档和已载入文件中自动匹配。")

        actions = ttk.Frame(panel)
        actions.grid(row=4, column=0, columnspan=4, sticky="e", pady=(14, 0))
        cancel = ttk.Button(actions, text="取消", command=self.top.destroy)
        cancel.pack(side=tk.LEFT, padx=(0, 6))
        ToolTip(cancel, "返回复查。也可按 Esc。")
        self.start_button = ttk.Button(actions, text="生成并打开分析", style="Record.TButton", command=lambda: self._start(accept))
        self.start_button.pack(side=tk.LEFT)
        ToolTip(self.start_button, "在实验的 analysis-inputs 目录内保存新的图像序列和标定副本。")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.reference_combo.focus_set()

    def _choose_profile(self):
        if name := filedialog.askopenfilename(parent=self.top, title="选择匹配的镜头标定", filetypes=[("镜头标定文件", "*.json")]):
            self._set_profile(Path(name))

    def _set_profile(self, path):
        self.lens = path
        self.lens_name.set(path.name if path else "自动匹配")

    def _start(self, accept):
        accept(int(self.reference.get()), self.lens)
        self.top.destroy()


class CheckerboardDialog:
    def __init__(self, parent, camera_ids, accept, *, capturing=False):
        background = ttk.Style(parent).lookup("TFrame", "background")
        self.top = tk.Toplevel(parent, bg=background)
        self.top.title("棋盘格标定采图" if capturing else "棋盘格标定板")
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.grab_set()
        self.camera_ids = tuple(camera_ids)
        self.accept = accept
        self.error = tk.StringVar()

        panel = ttk.Frame(self.top, padding=16)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(1, weight=1)
        self.cameras = tk.StringVar()
        label = ttk.Label(panel, textvariable=self.cameras, wraplength=420, justify=tk.LEFT)
        label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ToolTip(label, "双相机组按左相机到右相机标定。设备编号用于把结果和原始图像对应起来。", title="参与标定的相机")
        self._show_cameras()
        if len(self.camera_ids) == 2:
            swap = ttk.Button(panel, text="交换左右", command=self.swap_cameras, style="Small.TButton")
            swap.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 6))
            ToolTip(swap, "如果相机的实际位置与上面的顺序相反，点击交换。")

        self.values = []
        for row, (name, initial, hint) in enumerate((
            ("列内角点数", "9", "数棋盘格一行中黑白方格交界的内部交点。横向 10 个方格对应 9 个内角点。"),
            ("行内角点数", "6", "数棋盘格一列中的内部交点。双相机需要一个方向为奇数，另一方向为偶数。"),
            ("实测方格边长（mm）", "", "测量打印或加工后的棋盘格单个方格边长。这个数值决定双相机平移量的毫米尺度。"),
        ), 2):
            text = ttk.Label(panel, text=name)
            text.grid(row=row, column=0, sticky="w", padx=(0, 14), pady=4)
            value = tk.StringVar(value=initial)
            value.trace_add("write", lambda *_args: self.error.set(""))
            entry = ttk.Entry(panel, textvariable=value, width=18)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            ToolTip(entry, hint, title=name)
            ToolTip(text, hint, title=name)
            self.values.append(value)

        instructions = ("让整块棋盘格出现在所有相机画面里。每次摆稳后触发，保存后会显示角点检查结果。"
                        if capturing else "使用拍摄时的棋盘格尺寸。可以悬停查看每项的填写方法。")
        ttk.Label(panel, text=instructions, style="CaptureMuted.TLabel", wraplength=390, justify=tk.LEFT).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(7, 10))
        ttk.Label(panel, textvariable=self.error, wraplength=390, justify=tk.LEFT).grid(
            row=6, column=0, columnspan=2, sticky="ew")
        actions = ttk.Frame(panel)
        actions.grid(row=7, column=0, columnspan=2, sticky="e", pady=(7, 0))
        cancel = ttk.Button(actions, text="取消", command=self.top.destroy)
        cancel.pack(side=tk.LEFT, padx=(0, 6))
        ToolTip(cancel, "返回工作台，保留原先的采集计划和标定。")
        submit = ttk.Button(actions, text="开始标定采图" if capturing else "继续拟合", command=self.save, style="Record.TButton")
        submit.pack(side=tk.LEFT)
        ToolTip(submit, "检查相机和存储条件，并在所选输出位置旁保存标定图像。" if capturing else "选好保存位置后从实验原图拟合镜头参数。")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.top.bind("<Return>", lambda _event: self.save())
        entry.focus_set()

    def _show_cameras(self):
        if len(self.camera_ids) == 2:
            self.cameras.set(f"左相机  {self.camera_ids[0]}\n右相机  {self.camera_ids[1]}")
        else:
            self.cameras.set(f"单相机  {self.camera_ids[0]}")

    def swap_cameras(self):
        self.camera_ids = self.camera_ids[::-1]
        self._show_cameras()

    def save(self):
        try:
            columns, rows, size = (value.get().strip() for value in self.values)
            target = CheckerboardTarget(self.camera_ids, int(columns), int(rows), float(size))
        except ValueError:
            self.error.set("请填写两个整数内角点数及实测方格边长。")
            return
        except CaptureError as exc:
            self.error.set("每个方向至少有 3 个内角点。方格边长须为正数。" if exc.code == "INVALID_CONFIG"
                           else ui_error(exc, details=False))
            return
        self.top.destroy()
        self.accept(target)


class PhaseDialog:
    _fields = PHASE_FIELDS
    _settings = PHASE_FIELDS_BY_MODE

    def __init__(self, parent, phase, accept, *, cameras, read_ports, submit):
        background = ttk.Style(parent).lookup("TFrame", "background")
        self.top = tk.Toplevel(parent, bg=background)
        self.top.title("采集阶段")
        self.top.transient(parent)
        self.top.geometry("550x500")
        self.top.minsize(430, 320)
        repaint_after_resize(self.top)
        self.top.grab_set()
        self.accept = accept
        self.cameras, self.read_ports = cameras, read_ports
        self.submit = submit
        self.hardware_source = phase.get("hardware_source")
        self.hardware_pairing = phase.get("hardware_pairing")
        self.csv_trigger = phase.get("csv_trigger")
        self.csv_note = tk.StringVar(self.top)
        self.mode = tk.StringVar(value=phase["mode"])
        self.role = tk.StringVar(value=phase["role"])
        self.software_trigger_fallback = tk.BooleanVar(value=phase.get("software_trigger_fallback", False))
        self.monitor_trigger_wait = tk.BooleanVar(value=phase.get("monitor_trigger_wait", False))
        self.overlap_events = tk.BooleanVar(self.top, phase.get("overlap_events", False))
        self.confirm_start = tk.BooleanVar(self.top, bool(phase.get("start_confirmation")))
        self.start_instruction = tk.StringVar(self.top, phase.get("start_confirmation", ""))
        self.mode_note = tk.StringVar()
        self.hardware_note = tk.StringVar(self.top)
        self.entries = {}
        self.values = {key: tk.StringVar(value=str(phase.get(key, 1 if key == "event_count" else "")))
                       for key in self._fields}
        self.scroll = ScrollableForm(self.top, padding=(15, 12, 10, 2))
        self.scroll.pack(fill=tk.BOTH, expand=True)
        for widget in (self.scroll.canvas, self.scroll.scrollbar):
            ToolTip(widget, "向下滚动可查看阶段的其余设置。按 Tab 时会自动滚动到当前设置。", title="采集阶段")
        panel = self.scroll.body
        panel.columnconfigure(1, weight=1)
        selected_mode = None
        for index, (label, var, labels, descriptions) in enumerate((
            ("拍摄方式", self.mode, MODE_LABELS, MODE_HELP), ("图像用途", self.role, ROLE_LABELS, ROLE_HELP),
        )):
            heading = ttk.Label(panel, text=label)
            heading.grid(row=index, column=0, sticky="w", padx=(0, 15), pady=5)
            menu = localized_combobox(panel, var, labels, width=19)
            menu.grid(row=index, column=1, sticky="ew")
            ChoiceToolTip(menu, f"选择{label}。展开后悬停可查看每项含义。",
                          {label: descriptions[key] for key, label in labels.items()})
            ToolTip(heading, menu._tooltip.resolve_text)
            if index == 0:
                menu.bind("<<ComboboxSelected>>", self.show_fields, add="+")
                selected_mode = menu
        mode_note = ttk.Label(panel, textvariable=self.mode_note, width=1, wraplength=360, justify=tk.LEFT,
                              style="CaptureMuted.TLabel")
        mode_note.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 8))
        wrap_label_on_resize(mode_note)
        ToolTip(mode_note, lambda _event: MODE_HELP[self.mode.get()], title="拍摄方式")
        self.form = ttk.Frame(panel)
        self.form.grid(row=3, column=0, columnspan=2, sticky="ew")
        confirmation = ttk.Checkbutton(panel, text="本阶段开始前等待确认", variable=self.confirm_start,
                                       command=self._show_start_instruction)
        confirmation.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ToolTip(confirmation, "上一阶段的图像登记完成后，等操作者确认。确认后启动本阶段的相机和计时。")
        self.instruction_label = ttk.Label(panel, text="确认提示")
        self.instruction_label.grid(row=5, column=0, sticky="w", padx=(0, 15), pady=(5, 0))
        self.instruction_entry = ttk.Entry(panel, textvariable=self.start_instruction, width=26)
        self.instruction_entry.grid(row=5, column=1, sticky="ew", pady=(5, 0))
        for widget in (self.instruction_label, self.instruction_entry):
            ToolTip(widget, "写明需要确认的实验条件，例如“核对夹具和加载准备”。此提示会显示在主界面并记入台账。",
                    title="确认提示")
        self._show_start_instruction()
        actions = ttk.Frame(self.top, padding=(15, 8))
        actions.pack(fill=tk.X)
        apply = ttk.Button(actions, text="应用阶段", command=self.save, style="Record.TButton")
        apply.pack(side=tk.RIGHT)
        ToolTip(apply, "保存对这个阶段的修改。回到采集计划后可检查空间和触发条件。")
        cancel = ttk.Button(actions, text="取消", command=self.top.destroy)
        cancel.pack(side=tk.RIGHT, padx=(0, 6))
        ToolTip(cancel, "关闭窗口，保留原计划。")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.show_fields()
        selected_mode.focus_set()

    def _show_start_instruction(self):
        for widget in (self.instruction_label, self.instruction_entry):
            if self.confirm_start.get():
                widget.grid()
            else:
                widget.grid_remove()

    def show_fields(self, _event=None):
        for widget in self.form.winfo_children():
            widget.destroy()
        self.entries.clear()
        mode = self.mode.get()
        note = {
            "timed": "填写定时频率，并设置张数或时长。",
            "event": "填写频率、每次事件的前后张数和事件次数。可手动触发，也可添加 CSV 阈值。",
            "hardware": "先核对相机接线。掌握共享脉冲编号和各台计数值时，可按脉冲关联图像。",
        }.get(mode, "")
        self.mode_note.set(note or MODE_HELP[mode])
        for line, key in enumerate((*self._settings[mode], "frame_timeout_s")):
            label = ttk.Label(self.form, text=self._fields[key][0])
            label.grid(row=line, column=0, sticky="w", padx=(0, 15))
            entry = ttk.Entry(self.form, textvariable=self.values[key], width=20)
            entry.grid(row=line, column=1, sticky="ew", pady=5)
            self.entries[key] = entry
            ToolTip(entry, PHASE_FIELD_HELP[key], title=self._fields[key][0])
            ToolTip(label, entry._tooltip.resolve_text)
        if mode in ("single", "timed", "manual", "event"):
            option = ttk.Checkbutton(self.form, text="缺少软件触发时按主机到达选帧", variable=self.software_trigger_fallback)
            option.grid(row=len(self._settings[mode]) + 1 + (mode == "event"), column=0, columnspan=2, sticky="w",
                        pady=(7, 0))
            ToolTip(option, "相机缺少软件触发时连续送图。每次请求选择请求后主机收到的第一张。"
                    "同一请求可关联多台相机的图像，曝光同步需独立核实。")
        if mode == "event":
            index = len(self._settings[mode]) + 1
            overlap = ttk.Checkbutton(self.form, text="允许事件窗口重叠", variable=self.overlap_events)
            overlap.grid(row=index, column=0, columnspan=2, sticky="w", pady=(7, 0))
            ToolTip(overlap, EVENT_OVERLAP_HELP, title="短间隔连续触发")
            index += 2
            setup = ttk.Button(self.form, text="试验机 CSV 阈值…", command=self.open_csv_trigger, style="Small.TButton")
            setup.grid(row=index, column=0, columnspan=2, sticky="w", pady=(8, 0))
            note = ttk.Label(self.form, textvariable=self.csv_note, width=1, justify=tk.LEFT, style="CaptureMuted.TLabel")
            note.grid(row=index + 1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
            wrap_label_on_resize(note)
            for control in (setup, note):
                ToolTip(control, lambda _event: csv_trigger_help(self.csv_trigger) if self.csv_trigger else
                        "指定持续更新的 CSV、触发量列和阈值。事件前缓存满后，量值按指定方向越线即可自动触发。",
                        title="试验机 CSV 阈值")
            self._set_csv_trigger(self.csv_trigger)
        if mode == "hardware":
            index = len(self._settings[mode]) + 1
            setup = ttk.Button(self.form, text="物理输入与脉冲编号…", command=self.open_hardware, style="Small.TButton")
            setup.grid(row=index, column=0, columnspan=2, sticky="w", pady=(7, 0))
            setup.state(["!disabled" if self.cameras else "disabled"])
            ToolTip(setup, "填写每台相机实际使用的物理输入及已确认的计数器锚点。" if self.cameras else
                    "请先关闭阶段窗口，在设备列表中选择相机。")
            ttk.Label(self.form, textvariable=self.hardware_note, wraplength=355, justify=tk.LEFT,
                      style="CaptureMuted.TLabel").grid(row=index + 1, column=0, columnspan=2, sticky="w", pady=(4, 0))
            self._describe_hardware()
            option = ttk.Checkbutton(self.form, text="查看相机触发等待输出", variable=self.monitor_trigger_wait)
            option.grid(row=index + 2, column=0, columnspan=2, sticky="w", pady=(7, 0))
            ToolTip(option, "首帧前读取相机已配置的 FrameTriggerWait 输出。界面会逐台显示回读结果。"
                    "请先在厂商工具中设置等待输出。连续触发请使用现场接线的等待信号。", title="外触发前查看相机状态")
        self.form.columnconfigure(1, weight=1)
        self.scroll.bind_controls()

    def _camera_mapping_current(self):
        counters = self.hardware_pairing["first_counter"] if self.hardware_pairing else None
        return not hardware_mapping_error(self.cameras.keys(), self.hardware_source, counters)

    def _describe_hardware(self):
        if not self._camera_mapping_current():
            self.hardware_note.set("相机选择已变化。请核对逐台输入和计数值。")
            return
        source = self.hardware_source
        label = f"已指定 {len(source)} 台相机的输入" if isinstance(source, dict) else f"统一输入 {source}" if source else "物理输入待设置"
        if pairing := self.hardware_pairing:
            label += f" · {pairing['source_id']} 脉冲 {pairing['first_event_number']}"
        self.hardware_note.set(label)

    def open_hardware(self):
        if self.mode.get() == "hardware" and self.cameras:
            HardwareTriggerDialog(self.top, self.cameras, self.hardware_source, self.hardware_pairing, self.read_ports, self._set_hardware)

    def open_csv_trigger(self):
        if self.mode.get() == "event":
            CsvTriggerDialog(self.top, self.csv_trigger, self._set_csv_trigger, self.submit)

    def _set_csv_trigger(self, rule):
        self.csv_trigger = rule
        self.csv_note.set(csv_trigger_rule_text(rule) if rule else "可选。也可在采集时点“触发一次”。")

    def _set_hardware(self, source, pairing):
        self.hardware_source, self.hardware_pairing = source, pairing
        self._describe_hardware()

    def save(self):
        try:
            mode = self.mode.get()
            data = {"mode": mode, "role": self.role.get()}
            if self.confirm_start.get():
                data["start_confirmation"] = self.start_instruction.get().strip() or "请确认已准备好开始本阶段。"
            if mode in ("single", "timed", "manual", "event") and self.software_trigger_fallback.get():
                data["software_trigger_fallback"] = True
            if mode == "event":
                if self.csv_trigger:
                    data["csv_trigger"] = self.csv_trigger
                if self.overlap_events.get():
                    data["overlap_events"] = True
            if mode == "hardware":
                if not self.hardware_source or not self._camera_mapping_current():
                    raise CaptureError("HARDWARE_INPUT_INVALID", "请先打开“物理输入与脉冲编号”，核对当前相机。")
                data["hardware_source"] = self.hardware_source
                if self.hardware_pairing is not None:
                    data["hardware_pairing"] = self.hardware_pairing
                if self.monitor_trigger_wait.get():
                    data["monitor_trigger_wait"] = True
            for key in (*self._settings[mode], "frame_timeout_s"):
                if value := self.values[key].get().strip():
                    try:
                        data[key] = self._fields[key][1](value)
                    except ValueError:
                        self.entries[key].focus_set()
                        raise ValueError(f"请核对“{self._fields[key][0]}”的填写格式。") from None
            validated = Phase.from_dict(data).to_dict()
            self.accept(validated)
            self.top.destroy()
        except (CaptureError, ValueError) as exc:
            messagebox.showerror("无法应用采集阶段", ui_error(exc), parent=self.top)


class FeatureDialog:
    _first = ("exposure_us", "exposure_auto", "gain_db", "gain_auto", "fps", "fps_enable",
              "pixel_format", "gentl_image_part", "gentl_save_all_2d", "width", "height", "offset_x", "offset_y")
    _units = {"px": "像素", "us": "微秒", "px/frame": "像素／帧", "1/frame": "每帧", "frames": "张"}

    def __init__(self, parent, device, features, settings, accept, *, preview=False, focus_feature=None, plan_only=False):
        background = ttk.Style(parent).lookup("TFrame", "background")
        self.top = tk.Toplevel(parent, bg=background)
        self.top.title(f"相机参数 · {device.id}")
        self.top.geometry("600x560")
        self.top.minsize(490, 390)
        self.top.transient(parent)
        self.top.columnconfigure(0, weight=1)
        self.top.rowconfigure(1, weight=1)
        repaint_after_resize(self.top)
        self.top.grab_set()
        self.device, self.settings, self.accept = device, settings, accept
        self.features = {name: replace(feature, writable=False) if name in PHASE_FEATURES else feature
                         for name, feature in features.items()}
        self.preview = preview
        self.removed = set()
        if not preview:
            for key in settings.keys() - features.keys():
                self.features[key] = Feature(key, None, readable=False, writable=False)
        self.fields = {}
        self.widgets = {}
        header = ttk.Frame(self.top, padding=13)
        header.grid(row=0, column=0, sticky="ew")
        name = "模拟散斑相机" if device.backend == "sim" else device.model
        ttk.Label(header, text=name, style="CaptureSection.TLabel").pack(anchor="w")
        description = ("保存后重新检查计划" if plan_only else
                       "应用后会更新这台相机的画面" if preview else "应用后可连接预览核对画面")
        ttk.Label(header, text=f"{device.id} · {description}", style="CaptureMuted.TLabel", wraplength=440,
                  justify=tk.LEFT).pack(anchor="w", pady=(4, 0))
        middle = ScrollableForm(self.top, padding=(12, 1, 12, 1))
        middle.grid(row=1, column=0, sticky="nsew")
        content = middle.body
        for widget in (middle.canvas, middle.scrollbar):
            ToolTip(widget, "滚动查看全部相机参数。用 Tab 聚焦下一项时会自动滚动。")
        content.columnconfigure(1, weight=1)
        ordered = [key for key in self._first if key in self.features] + sorted(self.features.keys() - self._first)
        for index, key in enumerate(ordered):
            feature = self.features[key]
            baseline = feature.value if preview else settings.get(key, feature.value)
            var = tk.StringVar(self.top, value="" if baseline is None else str(baseline))
            self.fields[key] = var, var.get()
            label, description = FEATURE_HELP.get(key, (key, "相机驱动提供的参数。请按显示的范围和单位填写。"))
            if key not in features:
                description = "这是旧计划中的字段，当前设备不提供这个参数。可从计划移除。"
            elif key in PHASE_FEATURES:
                description = "采集时由阶段控制。请在采集计划中编辑阶段。"
            field_label = ttk.Label(content, text=label)
            field_label.grid(row=index, column=0, sticky="w", padx=(0, 8))
            choices = feature.choices or (("False", "True") if type(feature.value) is bool else ())
            constraints = self.bounds(feature)
            unit = self._units.get(feature.unit, feature.unit or "")
            suffix = ("预览时只读" if preview else "当前不可设置") if not feature.writable else f"{constraints} {unit}".strip()
            detail = "\n" + suffix if suffix else ""
            if feature.writable and feature.increment:
                detail += f" · 步长 {feature.increment:g}" if detail else f"\n步长 {feature.increment:g}"
            if choices and feature.writable:
                labels, help_text = self.choice_labels(key, choices)
                widget = localized_combobox(content, var, labels, width=16)
                ChoiceToolTip(widget, description + detail, help_text, title=label)
            else:
                widget = ttk.Entry(content, textvariable=var, width=16, state="normal" if feature.writable else "readonly")
                ToolTip(widget, description + detail, title=label)
            widget.grid(row=index, column=1, sticky="ew", pady=3)
            self.widgets[key] = widget
            if not preview and key in settings and not feature.writable:
                remove = ttk.Button(content, text="移除设置", style="Small.TButton")
                remove.configure(command=lambda name=key, value=var, button=remove: self._remove_setting(name, value, button))
                remove.grid(row=index, column=2, sticky="w", padx=(9, 0))
                ToolTip(remove, "保存后，从计划移除此项。采集时使用设备提供的设置。")
            else:
                bounds = ttk.Label(content, text=suffix, style="CaptureMuted.TLabel")
                bounds.grid(row=index, column=2, sticky="w", padx=(9, 0))
                if suffix:
                    ToolTip(bounds, widget._tooltip.resolve_text, title=label)
            ToolTip(field_label, widget._tooltip.resolve_text)
        middle.bind_controls()
        actions = ttk.Frame(self.top, padding=(14, 8))
        actions.grid(row=2, column=0, sticky="ew")
        apply_label = "保存并重新检查" if plan_only else "应用并更新预览" if preview else "应用相机参数"
        apply = ttk.Button(actions, text=apply_label, command=self.save, style="Record.TButton")
        apply.pack(side=tk.RIGHT)
        ToolTip(apply, "将修改保存到计划并重新检查。连接预览或开始采集时才会写入相机。" if plan_only else
                "短暂停止这台相机的取图。回读参数并收到新画面后，将修改写入当前采集计划。" if preview else
                "让相机核对并应用修改，再将设置保存到当前采集计划。")
        cancel = ttk.Button(actions, text="取消", command=self.top.destroy)
        cancel.pack(side=tk.RIGHT, padx=(0, 6))
        ToolTip(cancel, "关闭窗口，保留现有相机参数。")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        if self.widgets:
            widget = self.widgets.get(focus_feature) or next(iter(self.widgets.values()))
            widget.focus_set()
            self.top.after_idle(lambda: focus_canvas_child(widget, middle.canvas, content))

    def _remove_setting(self, name, variable, button):
        self.removed.add(name)
        variable.set("")
        button.state(["disabled"])

    @staticmethod
    def bounds(feature):
        if feature.minimum is not None and feature.maximum is not None:
            return f"{feature.minimum:,}–{feature.maximum:,}"
        if feature.minimum is not None:
            return f"≥ {feature.minimum:,}"
        if feature.maximum is not None:
            return f"≤ {feature.maximum:,}"
        return ""

    @staticmethod
    def choice_labels(key, choices):
        labels, help_text = {}, {}
        for raw in choices:
            value = str(raw)
            label, tip = value, "这是相机提供的值。具体行为可参照厂商参数说明。"
            if value in ("True", "False") or key in ("exposure_auto", "gain_auto") and value == "Off":
                label = "开启" if value == "True" else "关闭"
                tip = f"{label}此功能。"
            elif key in ("exposure_auto", "gain_auto") and value in ("Once", "Continuous"):
                label = "自动调整一次" if value == "Once" else "持续自动调整"
                tip = "由相机根据画面调节。"
            elif key == "pixel_format" and is_packed_mono_or_bayer(value):
                bits, color = image_layout(value)
                kind = "灰度打包" if color == "mono" else " Bayer 打包"
                label = f"{bits} 位{kind}（{value}）"
                byte_count = 1 if bits <= 8 else 2
                tip = f"相机以打包格式发送数据。原始 TIFF 每个样本占 {byte_count} 字节，保留 {bits} 位有效数据。"
                if color == "bayer":
                    tip += "保存原始 Bayer 排列。"
            elif key == "pixel_format" and (bits := value.removeprefix("Mono")).isdigit() and value.startswith("Mono"):
                label = f"{bits} 位灰度（{value}）"
                byte_count = 1 if int(bits) <= 8 else 2
                tip = f"每个像素保留 {bits} 位灰度，原始 TIFF 中占 {byte_count} 字节。"
            labels[value] = label
            help_text[label] = tip
        return labels, help_text

    def save(self):
        name = None
        try:
            changes = {key: value for key, value in self.settings.items() if key not in self.removed}
            for name, (variable, initial) in self.fields.items():
                feature = self.features[name]
                text = variable.get().strip()
                if not feature.writable or text == initial:
                    continue
                if not text:
                    changes.pop(name, None)
                    continue
                original = feature.value
                if type(original) is bool:
                    value = text == "True"
                elif type(original) is int:
                    value = int(text)
                elif type(original) is float or feature.unit in ("Hz", "dB", "us"):
                    value = float(text)
                else:
                    value = text
                if self.preview:
                    feature.validate(value)
                changes[name] = value
            if not self.preview:
                check = (check_simulator_settings(CameraConfig(self.device.id, changes)) if self.device.backend == "sim" else
                         check_camera_settings(self.device.backend, changes, self.features))
                if check.invalid:
                    problem = check.invalid[0]
                    name = problem["feature"]
                    raise CaptureError(problem["reason"], problem["detail"])
            self.accept(changes)
            self.top.destroy()
        except (CaptureError, ValueError) as exc:
            label = FEATURE_HELP.get(name, (name,))[0]
            if name in self.widgets:
                self.widgets[name].focus_set()
            message = ui_error(exc) if isinstance(exc, CaptureError) else "请按提示输入数字。"
            messagebox.showerror("无法应用相机参数", f"{label}\n{message}" if label else message, parent=self.top)
