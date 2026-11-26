"""Run a storage-only measurement from a Capture plan report."""

from __future__ import annotations

import json
import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ezdic_gui.controls import localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .presentation import DURABILITY_LABELS, MODE_LABELS, ROLE_LABELS, format_bytes, ui_error
from .viewport import repaint_after_resize
from .write_probe import DEFAULT_PROBE_MIB, MAX_PROBE_MIB, MIB, measure_write_path, preferred_probe_phase, write_probe_phases


_COMPARISONS = {
    "at_or_above_plan": "本次写盘速率达到计划估算",
    "below_plan": "本次写盘速率低于计划估算",
    "no_planned_rate": "本阶段未设置固定采集频率",
    "short_sample": "测试轮次较少，请用阶段试采核对",
}


def _rate(value):
    return "待估算" if value is None else f"{value / MIB:,.2f} MiB/s"


class WriteProbeWindow:
    def __init__(self, parent, config, report, authorize, start_trial):
        self.config, self.output = config, report["output"]
        self.parent_path = report["volume_probe_path"]
        self._authorize, self._start_trial = authorize, start_trial
        phases = write_probe_phases(report)
        self._phases = {str(phase.number): phase for phase in phases}
        preferred = preferred_probe_phase(phases) or phases[0]
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("StrainTrace · 写盘实测")
        self.top.transient(parent)
        self.top.geometry("690x445")
        self.top.minsize(605, 424)
        repaint_after_resize(self.top)
        self.top.protocol("WM_DELETE_WINDOW", self._close)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self._close())
        self.top.bind("<Destroy>", self._destroy, add="+")
        self.phase = tk.StringVar(self.top, str(preferred.number))
        self.scope = tk.StringVar(self.top)
        self.status = tk.StringVar(self.top)
        self.result_text = tk.StringVar(self.top)
        self._working = False
        self._result = None
        self._detail = ""
        self._latest = ("preparing", 0, 0)
        self._cancel = threading.Event()
        self._completed = queue.SimpleQueue()
        self._after = None
        self._build()
        self.phase.trace_add("write", self._select)
        self._select()

    @property
    def working(self):
        return self._working

    def _selected(self):
        return self._phases[self.phase.get()]

    def _budget(self, phase):
        return min(MAX_PROBE_MIB, max(DEFAULT_PROBE_MIB, math.ceil(phase.round_bytes * 3 / MIB)))

    def _geometry_block(self, phase):
        if phase.unresolved:
            return f"图像格式待核对 · {'、'.join(phase.unresolved)}。可连接预览后重新检查计划。"
        if phase.round_bytes > MAX_PROBE_MIB * MIB:
            return "一轮图像超过测试上限。可在实际阶段试采中核对该设置。"
        return ""

    def _build(self):
        panel = ttk.Frame(self.top, padding=14)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(0, weight=1)
        heading = ttk.Label(panel, width=1, style="CaptureSection.TLabel",
                            text=f"写入方式 · {DURABILITY_LABELS[self.config.durability]}")
        heading.grid(row=0, sticky="ew", pady=(0, 5))
        directory = Path(self.parent_path)
        location_text = str(directory) if len(str(directory)) <= 35 else str(Path(directory.anchor) / "…" / directory.name)
        location = ttk.Label(panel, width=1, text=f"测试位置 · {location_text}", style="CaptureMuted.TLabel")
        location.grid(row=1, sticky="ew", pady=(0, 9))
        ToolTip(location, f"在此位置临时写入测试 TIFF 和台账，并在结束时清理。\n{self.parent_path}", title="目标卷")

        selection = ttk.Frame(panel)
        selection.grid(row=2, sticky="ew")
        ttk.Label(selection, text="采集阶段").pack(side=tk.LEFT, padx=(0, 10))
        labels = {key: f"阶段 {phase.number} · {MODE_LABELS[phase.mode]} · {ROLE_LABELS[phase.role]}"
                  for key, phase in self._phases.items()}
        choices = {caption: self._geometry_block(self._phases[key]) or
                   f"一轮 {format_bytes(self._phases[key].round_bytes)} · 计划数据率 {_rate(self._phases[key].planned_raw_bytes_per_s)}"
                   for key, caption in labels.items()}
        self.picker = localized_combobox(selection, self.phase, labels, width=26)
        self.picker.pack(side=tk.LEFT)
        ChoiceToolTip(self.picker, "默认选择计划数据率最高、图像格式已知的阶段。", choices, title="采集阶段")
        scope = ttk.Label(panel, width=1, textvariable=self.scope, justify=tk.LEFT, style="CaptureMuted.TLabel")
        scope.grid(row=3, sticky="ew", pady=(7, 8))
        wrap_label_on_resize(scope)
        ToolTip(scope, "用相同尺寸、位深和部件组成的合成图像测试 TIFF 和台账写入。"
                "实际收图速度和持续写入仍可通过阶段试采核对。", title="测量范围")

        self.bar = ttk.Progressbar(panel, maximum=1, style="CaptureCopy.Horizontal.TProgressbar")
        self.bar.grid(row=4, sticky="ew", pady=(3, 5))
        status = ttk.Label(panel, width=1, textvariable=self.status, style="CaptureSection.TLabel", justify=tk.LEFT)
        status.grid(row=5, sticky="ew", pady=(0, 4))
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.status.get(), title="写盘状态")
        result = ttk.Label(panel, width=1, textvariable=self.result_text, style="CaptureMuted.TLabel", justify=tk.LEFT)
        result.grid(row=6, sticky="new")
        wrap_label_on_resize(result)
        ToolTip(result, "登记速度按 TIFF 和台账写入的用时计算。含刷盘的速率还计入请求系统刷盘的用时，"
                "可与计划数据率对照。P95 表示约 95% 的写入轮次用时不超过该值。", title="速率和耗时")
        panel.rowconfigure(6, weight=1, minsize=44)

        actions = ttk.Frame(panel)
        actions.grid(row=7, sticky="ew", pady=(7, 0))
        self.close_button = ttk.Button(actions, text="关闭", command=self._close)
        self.close_button.pack(side=tk.LEFT)
        ToolTip(self.close_button, lambda _event: "取消测试并清理临时文件。" if self.working else "关闭写盘实测窗口。")
        self.copy_button = ttk.Button(actions, text="复制结果", command=self._copy, style="Small.TButton")
        self.copy_button.pack(side=tk.LEFT, padx=(6, 0))
        ToolTip(self.copy_button, "复制本次测量的 JSON，包含目标卷、像素格式、样本量和测量值。")
        self.start_button = ttk.Button(actions, text="开始写盘测试", command=self.start, style="Record.TButton")
        self.start_button.pack(side=tk.RIGHT)
        ToolTip(self.start_button, self._start_help)
        self.trial_button = ttk.Button(actions, text="试采此阶段…", command=self._trial, style="Small.TButton")
        self.trial_button.pack(side=tk.RIGHT, padx=(0, 6))
        ToolTip(self.trial_button, "进入此阶段的相机试采，核对收图、触发和连续写入。")

    def _start_help(self, _event):
        if self._working:
            return "正在测试。如需停止，请点击“取消”。"
        return self._geometry_block(self._selected()) or self._authorize() or (
            "用当前选择的图像规格写入、刷盘并校验。结果用于了解本次短时写入条件。")

    def _controls(self):
        self.picker.state(["disabled" if self._working or len(self._phases) == 1 else "!disabled"])
        self.start_button.state(["disabled" if self._working or self._geometry_block(self._selected()) else "!disabled"])
        for widget in (self.copy_button, self.trial_button):
            widget.state(["!disabled" if self._result is not None and not self._working else "disabled"])
        self.close_button.configure(text="取消" if self._working else "关闭")
        self.close_button.state(["disabled" if self._working and self._cancel.is_set() else "!disabled"])

    def _select(self, *_args):
        phase = self._selected()
        self._result, self._detail = None, ""
        self.scope.set(self._geometry_block(phase) or
                       f"一轮 {format_bytes(phase.round_bytes)} · 计划数据率 {_rate(phase.planned_raw_bytes_per_s)}\n"
                       f"合成图像写入上限 {self._budget(phase)} MiB，结束后清理测试文件。")
        self.status.set("")
        self.result_text.set("")
        self.bar.configure(value=0)
        self._controls()

    def start(self):
        if self._working or self._geometry_block(self._selected()):
            return
        if reason := self._authorize():
            self.status.set(reason)
            return
        phase = self._selected()
        self._working = True
        self._cancel.clear()
        self._latest, self._result, self._detail = ("preparing", 0, 0), None, ""
        self.result_text.set("")
        self.status.set("正在准备合成图像")
        self.bar.configure(mode="indeterminate", maximum=100, value=0)
        self.bar.start(18)
        self._controls()

        def progress(stage, raw, count):
            self._latest = stage, raw, count

        def work():
            try:
                result = measure_write_path(self.config, self.output, phase, expected_parent=self.parent_path,
                                            max_mib=self._budget(phase), cancelled=self._cancel.is_set, progress=progress)
            except Exception as error:
                self._completed.put((None, error))
            else:
                self._completed.put((result, None))

        threading.Thread(target=work, name="capture-write-probe", daemon=True).start()
        self._after = self.top.after(90, self._poll)

    def _poll(self):
        self._after = None
        try:
            result, error = self._completed.get_nowait()
        except queue.Empty:
            stage, raw, count = self._latest
            if not self._cancel.is_set():
                stage_text = {"preparing": "正在准备合成图像", "writing": "正在写入", "flushing": "正在刷盘",
                              "verifying": "正在校验 TIFF 和台账"}[stage]
                self.status.set(f"{stage_text} · {format_bytes(raw)} · {count:,} 帧" if count else stage_text)
            self._after = self.top.after(90, self._poll)
            return
        self.bar.stop()
        self.bar.configure(mode="determinate", maximum=1, value=0 if error else 1)
        self._working, self._result = False, result
        if error:
            self.status.set(ui_error(error, details=False))
            self._detail = ui_error(error)
        else:
            self.status.set(_COMPARISONS[result["comparison"]])
            rates = f"含刷盘 {_rate(result['flushed_raw_bytes_per_s'])}"
            if self.config.durability == "buffered":
                rates += f" · 登记 {_rate(result['committed_raw_bytes_per_s'])}"
            self.result_text.set(
                f"{rates} · {result['flush_inclusive_seconds']:.2f} 秒\n"
                f"校验 {result['tiffs']:,} 张 TIFF · 原始像素 {format_bytes(result['raw_bytes'])}\n"
                f"每轮写入耗时 P95 {result['p95_round_ms']:.1f} ms"
            )
        self._controls()

    def _copy(self):
        if self._result is not None:
            self.top.clipboard_clear()
            self.top.clipboard_append(json.dumps(self._result, ensure_ascii=False, indent=2))
            self.status.set(f"已复制结果 · {_COMPARISONS[self._result['comparison']]}")

    def _trial(self):
        if reason := self._authorize():
            self.status.set(reason)
            return
        self.top.withdraw()
        if not self._start_trial(self._selected().number):
            self.top.deiconify()

    def cancel(self):
        if self._working:
            self._cancel.set()
            self.status.set("正在取消并清理测试文件")
            self._controls()

    def _close(self):
        if self._working:
            self.cancel()
        else:
            self.top.destroy()

    def _destroy(self, event):
        if event.widget is self.top:
            self._cancel.set()
            if self._after is not None:
                self.top.after_cancel(self._after)
