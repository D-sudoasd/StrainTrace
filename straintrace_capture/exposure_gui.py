"""Show and verify a manual exposure suggestion from the active preview."""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .config import CaptureError
from .exposure import SAMPLE_FRAMES, confirm_exposure_improvement, recommend_exposure, sample_exposure
from .presentation import ui_error
from .viewport import repaint_after_resize


def _summary(result):
    return f"亮部码值 {result.highlight:.1%} · 近饱和像素 {result.near_saturation:.2%}"


class ExposureWindow:
    def __init__(self, parent, current_scope, initial_scope, name_for, submit, apply_adjustment):
        self.current_scope = current_scope
        self.name_for = name_for
        self.submit = submit
        self.apply_adjustment = apply_adjustment
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("曝光建议")
        self.top.geometry("665x361")
        self.top.minsize(510, 350)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.limit = tk.StringVar(self.top)
        self.source = tk.StringVar(self.top)
        self.metrics = tk.StringVar(self.top)
        self.status = tk.StringVar(self.top, "固定光源和试样后点击“测量”。")
        self._working = ""
        self._stop = threading.Event()
        self._scope = initial_scope
        self._measurement = None
        self._advice = None
        self._detail = ""
        self._build()
        self._show_scope(initial_scope)
        self.limit.trace_add("write", self._limit_changed)
        self.top.protocol("WM_DELETE_WINDOW", self._close)
        self.top.bind("<Destroy>", self._destroyed, add="+")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top), add="+")
        self._controls()
        self.measure_button.focus_set()

    def _build(self):
        panel = ttk.Frame(self.top, padding=(18, 12))
        panel.pack(fill=tk.BOTH, expand=True)
        footer = ttk.Frame(panel)
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=(9, 0))
        self.measure_button = ttk.Button(footer, text="测量", command=self.measure, style="Small.TButton")
        self.measure_button.pack(side=tk.LEFT)
        ToolTip(self.measure_button, f"读取 {SAMPLE_FRAMES} 张新画面的原始像素，给出曝光时间建议。")
        self.close_button = ttk.Button(footer, text="关闭", command=self._close, style="Small.TButton")
        self.close_button.pack(side=tk.RIGHT)
        ToolTip(self.close_button, lambda _event: "停止测量或试设。试设已写入时会验证并恢复原参数。"
                if self._working else "返回采集工作台。")
        self.apply_button = ttk.Button(footer, text="试设并复测", command=self.apply,
                                       style="Confirm.Record.TButton", width=0)
        self.apply_button.pack(side=tk.RIGHT, padx=(0, 7))
        ToolTip(self.apply_button, "复核当前画面后，只对所选相机试设曝光。新画面改善才采用到本次计划，"
                "亮部没有改善或近饱和像素过多时恢复原设置。")

        head = ttk.Label(panel, textvariable=self.source, style="CaptureSection.TLabel", justify=tk.LEFT)
        head.pack(fill=tk.X, pady=(0, 7))
        wrap_label_on_resize(head)
        ToolTip(head, "使用当前相机的原始画面。若已在图上按住 Shift 画出检查框，只测量框内。")
        row = ttk.Frame(panel)
        row.pack(fill=tk.X, pady=(0, 10))
        label = ttk.Label(row, text="最大曝光（微秒，可选）")
        label.pack(side=tk.LEFT, padx=(0, 9))
        self.limit_entry = ttk.Entry(row, textvariable=self.limit, width=13)
        self.limit_entry.pack(side=tk.LEFT)
        limit_help = "可为运动试样输入允许的最长曝光时间。留空时使用设备范围和已设定的帧周期，实际收图速率可用试采检查。"
        ToolTip(label, limit_help, title="最大曝光时间")
        ToolTip(self.limit_entry, limit_help, title="最大曝光时间")
        metrics = ttk.Label(panel, textvariable=self.metrics, justify=tk.LEFT, style="CaptureMuted.TLabel")
        metrics.pack(fill=tk.X, pady=(0, 7))
        wrap_label_on_resize(metrics)
        ToolTip(metrics, "亮部码值取第 99 百分位，再除以图像位深允许的最大码值。彩色图像用每个像素最亮的颜色通道。"
                "近饱和像素占比表示快达到原图上限的像素。")
        status = ttk.Label(panel, textvariable=self.status, justify=tk.LEFT)
        status.pack(fill=tk.X)
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.status.get(), title="测量详情")

    def _show_scope(self, scope):
        text = self.name_for(scope.camera_id)
        area = f"检查框 {scope.roi[2]} × {scope.roi[3]} 像素" if scope.roi else "整幅画面"
        self.source.set(f"{text}  ·  {area}")
        us = scope.features["exposure_us"].value
        self.metrics.set(f"当前曝光 {us:g} 微秒")

    def _controls(self):
        self.measure_button.state(["disabled" if self._working else "!disabled"])
        self.limit_entry.state(["disabled" if self._working else "!disabled"])
        ready = self._advice is not None and self._advice.proposed_us is not None and not self._working
        self.apply_button.state(["!disabled" if ready else "disabled"])
        self.close_button.configure(text="取消" if self._working else "关闭")
        self.close_button.state(["disabled" if self._working and self._stop.is_set() else "!disabled"])

    def _limit_changed(self, *_args):
        if self._advice is not None:
            self._advice = None
            self._detail = ""
            self.status.set("最长曝光已变更，请重新测量。")
        self._controls()

    def _sample(self, scope):
        observed, _ = sample_exposure(scope.latest, after=time.perf_counter_ns(), layout=scope.layout,
                                      roi=scope.roi, cancelled=self._stop.is_set)
        return observed

    def measure(self):
        if self._working:
            return
        try:
            scope = self.current_scope()
            field = self.limit.get().strip()
            limit = float(field) if field else None
            if limit is not None and (not math.isfinite(limit) or limit <= 0):
                raise ValueError
        except ValueError:
            self._failed(CaptureError("EXPOSURE_LIMIT", "最大曝光时间需填写微秒数。"))
            return
        except CaptureError as exc:
            self._failed(exc)
            return
        self._scope = scope
        self._show_scope(scope)
        self._working = "measure"
        self._advice = self._measurement = None
        self._detail = ""
        self._stop.clear()
        self.status.set(f"正在测量 {SAMPLE_FRAMES} 张新画面…")
        self._controls()

        def work():
            measurement = self._sample(scope)
            return measurement, recommend_exposure(measurement, scope.features, scope.planned_rates, operator_limit=limit)

        self.submit(work, self._measured, self._failed)

    def _validate_scope(self):
        if not self._scope.still_matches(self.current_scope()):
            raise CaptureError("EXPOSURE_SCOPE_CHANGED", "相机、检查框或计划帧率已经变更。")

    def _measured(self, result):
        if not self.top.winfo_exists():
            return
        try:
            self._validate_scope()
        except CaptureError as exc:
            self._failed(exc)
            return
        self._measurement, self._advice = result
        current = self._scope.features["exposure_us"].value
        self.metrics.set(f"当前曝光 {current:g} 微秒\n{_summary(self._measurement)}")
        advice = self._advice
        self.status.set(advice.reason if advice.proposed_us is None else (
            f"建议试设 {advice.proposed_us:g} 微秒。当前时限 {advice.maximum_us:g} 微秒。"))
        self._working = ""
        self._controls()

    def apply(self):
        if self._working or self._advice is None or self._advice.proposed_us is None:
            return
        try:
            self._validate_scope()
        except CaptureError as exc:
            self._failed(exc)
            return
        self._stop.clear()
        self._working = "verify"
        self._detail = ""
        self.status.set("正在复核当前亮度，随后试设并读取新画面…")
        self._controls()
        scope, original, proposed = self._scope, self._measurement, self._advice.proposed_us

        def fresh():
            baseline = self._sample(scope)
            if (not baseline.stable or abs(original.highlight - baseline.highlight) > .05
                    or abs(original.near_saturation - baseline.near_saturation) > .01):
                raise CaptureError("EXPOSURE_SCOPE_CHANGED", "当前亮度与测量时不同。")
            return baseline

        def begin(baseline):
            if not self.top.winfo_exists():
                return
            if self._stop.is_set():
                self._failed(CaptureError("EXPOSURE_CANCELLED", "exposure adjustment cancelled"))
                return
            measured = None

            def verify(first, read):
                nonlocal measured
                measured, last = sample_exposure(read, after=first.received_monotonic_ns, layout=scope.layout,
                                                 roi=scope.roi, cancelled=self._stop.is_set)
                confirm_exposure_improvement(baseline, measured, increased=proposed > scope.features["exposure_us"].value)
                if self._stop.is_set():
                    raise CaptureError("EXPOSURE_CANCELLED", "exposure adjustment cancelled")
                return last

            def completed():
                if self.top.winfo_exists():
                    self._working = ""
                    self._measurement = self._advice = None
                    self.metrics.set(f"复测前  {_summary(baseline)}\n复测后  {_summary(measured)}")
                    self.status.set(f"已采用 {proposed:g} 微秒，并保存到当前采集计划。")
                    self._controls()

            self.apply_adjustment(scope, proposed, verify, completed, self._failed)

        self.submit(fresh, begin, self._failed)

    def _failed(self, exc):
        if self.top.winfo_exists():
            self._working = ""
            self._advice = None
            self._detail = ui_error(exc)
            status = ("原曝光已恢复。悬停可看复测详情。" if getattr(exc, "code", None) == "PREVIEW_ADJUST_RESTORED"
                      else self._detail.split("\n", 1)[0])
            self.status.set(status)
            self._controls()

    def _close(self):
        if self._working:
            self._stop.set()
            self._controls()
        else:
            self.top.destroy()

    def _destroyed(self, event):
        if event.widget is self.top:
            self._stop.set()
