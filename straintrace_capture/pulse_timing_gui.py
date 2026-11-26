"""Enter DAQ exposure timing inputs and limits in the Capture desktop."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.tooltips import ToolTip

from .config import CaptureError
from .pulse_audit import pulse_integer
from .pulse_timing import ExposureLimits
from .viewport import repaint_after_resize


def microseconds(value):
    if value is None:
        return "—"
    whole, fraction = divmod(abs(value), 1000)
    sign = "-" if value < 0 else ""
    return f"{sign}{whole}.{fraction:03d}".rstrip("0").rstrip(".")


def _nanoseconds(entry):
    text = entry.get().strip()
    if not text:
        return None
    whole, dot, fraction = text.removeprefix("-").partition(".")
    integral = pulse_integer(whole)
    fractional = pulse_integer(fraction) if dot else 0
    if integral is None or fractional is None or len(fraction) > 3 or any(character.isspace() for character in text):
        raise CaptureError("PULSE_TIMING_LIMITS", "限值用微秒填写，最多三位小数，例如 0.250。")
    value = integral * 1000 + fractional * 10 ** (3 - len(fraction))
    return -value if text.startswith("-") else value


class ExposureTimingDialog:
    def __init__(self, parent, cameras, accept, *, previous=None):
        self.cameras, self.accept = cameras, accept
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("核对曝光时序")
        self.top.transient(parent)
        self.top.minsize(520, 340)
        repaint_after_resize(self.top)
        previous = previous or {}
        self.source = previous.get("input", {}).get("path")
        self.source_name = tk.StringVar(self.top, Path(self.source).name if self.source else "")
        self.note = tk.StringVar(self.top)
        limits = previous.get("limits_ns", {})
        self.values = {name: tk.StringVar(self.top, microseconds(limits[name]) if limits.get(name) is not None else "")
                       for name in ("max_skew_ns", "latency_min_ns", "latency_max_ns")}
        self._build()
        for variable in self.values.values():
            variable.trace_add("write", self._controls)
        self._controls()
        self.top.grab_set()

    def _build(self):
        area = ttk.Frame(self.top, padding=15)
        area.pack(fill=tk.BOTH, expand=True)
        area.columnconfigure(1, weight=1)
        ttk.Label(area, text="选择同一 DAQ 时钟记录的触发沿和各相机曝光起始沿。", wraplength=490,
                  justify=tk.LEFT, style="CaptureMuted.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Label(area, text="时序 CSV").grid(row=1, column=0, sticky="w", padx=(0, 12))
        source = ttk.Entry(area, textvariable=self.source_name, state="readonly")
        source.grid(row=1, column=1, sticky="ew")
        ToolTip(source, lambda _event: self.source or "先选择时序 CSV。")
        choose = ttk.Button(area, text="选择文件…", command=self._choose, style="Small.TButton")
        choose.grid(row=1, column=2, sticky="ew", padx=(8, 0))
        ToolTip(choose, "表头包含 event_number、clock_id、edge、camera_id、time_ns 和 uncertainty_ns。"
                        "trigger 行的相机编号留空。exposure_start 行填相机编号。时间和误差界均用整数纳秒。")

        for row, (key, label, help_text) in enumerate((
            ("max_skew_ns", "相机间起始偏差上限（µs）", "同一次脉冲下各相机最早和最晚曝光起始时刻之差。可留空。"),
            ("latency_min_ns", "触发至曝光延迟下限（µs）", "各相机曝光起始沿减去共享触发沿。检查延迟须同时填写下限与上限。"),
            ("latency_max_ns", "触发至曝光延迟上限（µs）", "逐台检查延迟。测量区间碰到限值时，只有整个区间在范围内才判定符合。"),
        ), 2):
            ttk.Label(area, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
            entry = ttk.Entry(area, textvariable=self.values[key], width=15)
            entry.grid(row=row, column=1, sticky="w")
            ToolTip(entry, ("单相机可核对触发至曝光的延迟。" if key == "max_skew_ns" and len(self.cameras) < 2 else help_text), title=label)
            if key == "max_skew_ns" and len(self.cameras) < 2:
                entry.state(["disabled"])

        copy_ids = ttk.Button(area, text="复制相机编号", command=self._copy_cameras, style="Small.TButton")
        copy_ids.grid(row=5, column=0, sticky="w", pady=(5, 0))
        ToolTip(copy_ids, "将本阶段的相机编号复制到剪贴板，可粘贴到时序 CSV 的 camera_id 列。")
        ttk.Label(area, textvariable=self.note, wraplength=490, justify=tk.LEFT,
                  style="CaptureMuted.TLabel").grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 6))
        actions = ttk.Frame(area)
        actions.grid(row=7, column=0, columnspan=3, sticky="e")
        cancel = ttk.Button(actions, text="取消", command=self.top.destroy, style="Small.TButton")
        cancel.pack(side=tk.LEFT, padx=(0, 7))
        ToolTip(cancel, "返回脉冲核对，保留已导入的结果。")
        self.run = ttk.Button(actions, text="核对时序", command=self._confirm, style="Record.TButton")
        self.run.pack(side=tk.LEFT)
        ToolTip(self.run, lambda _event: "先选择时序 CSV，再填写相机间偏差上限或延迟范围。" if self.run.instate(["disabled"]) else
                "计算逐个脉冲的相机间偏差和逐台延迟，使用测量误差界判断是否符合所填限值。")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        choose.focus_set()

    def _choose(self):
        path = filedialog.askopenfilename(parent=self.top, filetypes=[("DAQ 时序 CSV", "*.csv"), ("所有文件", "*.*")])
        if path:
            self.source = path
            self.source_name.set(Path(path).name)
            self._controls()

    def _copy_cameras(self):
        self.top.clipboard_clear()
        self.top.clipboard_append("\n".join(self.cameras))
        self.note.set(f"已复制 {len(self.cameras)} 台相机的编号。")

    def _controls(self, *_args):
        ready = self.source and any(variable.get().strip() for variable in self.values.values())
        self.run.state(["!disabled" if ready else "disabled"])
        self.note.set("为本次实验填写至少一项限值。CSV 中误差界未知时留空。")

    def _confirm(self):
        try:
            limits = ExposureLimits(**{key: _nanoseconds(value) for key, value in self.values.items()})
        except CaptureError as exc:
            self.note.set(str(exc))
            return
        self.top.destroy()
        self.accept(self.source, limits)
