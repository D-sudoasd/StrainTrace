"""Desktop detail for the active Capture writer."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .presentation import (
    RAW_BUFFER_HELP, WRITE_PRESSURE_GUIDANCE, WRITE_PRESSURE_LABELS, format_bytes, write_pressure_estimate, write_pressure_tip,
)
from .viewport import repaint_after_resize


class WritePressureWindow:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("StrainTrace · 写入趋势")
        self.top.geometry("650x485")
        self.top.minsize(550, 454)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.heading = tk.StringVar(self.top)
        self.byte_text = tk.StringVar(self.top)
        self.queue_text = tk.StringVar(self.top)
        self.guidance = tk.StringVar(self.top)
        self._pressure = None
        panel = ttk.Frame(self.top, padding=14)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(6, weight=1)
        heading = ttk.Label(panel, width=1, textvariable=self.heading, style="CaptureSection.TLabel")
        heading.grid(row=0, sticky="ew", pady=(0, 8))
        ToolTip(heading, lambda _event: write_pressure_tip(self._pressure), title="写入状态")
        self.bytes_bar = self._usage(panel, 1, self.byte_text, RAW_BUFFER_HELP)
        self.queue_bar = self._usage(panel, 3, self.queue_text,
                                     "一批可以包含一台或多台相机的帧，队列显示等待交给写入线程的批次。")
        self.rates = ttk.Treeview(panel, columns=("window", "accepted", "committed"), show="headings",
                                  height=2, selectmode="browse")
        for name, title, width in (("window", "观测区间", 136), ("accepted", "入队速率", 132), ("committed", "登记速率", 132)):
            self.rates.heading(name, text=title)
            self.rates.column(name, width=width, anchor=tk.CENTER)
        self.rates.grid(row=5, sticky="ew", pady=(7, 6))
        ToolTip(self.rates, "两列都按原始像素字节统计。入队表示图像交给写入线程，登记表示 TIFF 已保存并写入采集台账。",
                title="近期速率")
        guidance = ttk.Label(panel, width=1, textvariable=self.guidance, anchor="nw", justify=tk.LEFT,
                             style="CaptureMuted.TLabel")
        guidance.grid(row=6, sticky="nsew", pady=(2, 8))
        wrap_label_on_resize(guidance)
        ToolTip(guidance, lambda _event: write_pressure_tip(self._pressure), title="估算依据")
        close = ttk.Button(panel, text="关闭", command=self.top.destroy, style="Small.TButton")
        close.grid(row=7, sticky="e")
        ToolTip(close, "回到采集工作台。")
        self.rates.focus_set()

    def _usage(self, panel, row, variable, help_text):
        label = ttk.Label(panel, textvariable=variable, width=1)
        label.grid(row=row, sticky="ew")
        bar = ttk.Progressbar(panel, mode="determinate", style="CaptureCopy.Horizontal.TProgressbar")
        bar.grid(row=row + 1, sticky="ew", pady=(3, 8))
        for widget in (label, bar):
            ToolTip(widget, help_text)
        return bar

    def render(self, status):
        pressure = self._pressure = status["write_pressure"]
        phase = f"阶段 {status['phase']} · " if status["phase_progress"] else ""
        caption = WRITE_PRESSURE_LABELS[pressure["state"]]
        self.heading.set(phase + caption if status["state"] in ("opening", "running", "draining") else "本次采集已结束")
        buffered, maximum = status["buffered_raw_bytes"], status["raw_buffer_limit_bytes"]
        batches, capacity = status["queue_size"], status["queue_capacity"]
        self.byte_text.set(f"字节缓冲 {format_bytes(buffered)} / {format_bytes(maximum)}"
                           f" · 预缓存 {format_bytes(pressure['prebuffer_raw_bytes'])}")
        self.bytes_bar.configure(maximum=maximum, value=buffered)
        self.queue_text.set(f"待写批次 {batches:,} / {capacity:,}")
        self.queue_bar.configure(maximum=capacity, value=batches)
        windows = pressure["windows"]
        for index, item in enumerate(windows):
            text = "最近" if index == len(windows) - 1 else "前一段"
            values = (f"{text} {item['duration_s']:.1f} 秒", f"{format_bytes(item['accepted_raw_bytes_per_s'])}/s",
                      f"{format_bytes(item['committed_raw_bytes_per_s'])}/s")
            key = str(index)
            if self.rates.exists(key):
                self.rates.item(key, values=values)
            else:
                self.rates.insert("", "end", iid=key, values=values)
        for key in self.rates.get_children()[len(windows):]:
            self.rates.delete(key)
        guidance = WRITE_PRESSURE_GUIDANCE[pressure["state"]]
        if estimate := write_pressure_estimate(pressure["estimate"]):
            guidance = f"{estimate}\n{guidance}"
        if not windows and pressure["state"] == "sampling":
            guidance = "观察满约 2 秒后可显示近期速率。"
        self.guidance.set(guidance)
