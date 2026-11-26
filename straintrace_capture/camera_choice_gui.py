"""Select detected experiment cameras before opening their preview."""

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import ScrollableForm, wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .viewport import repaint_after_resize


class CameraPreviewDialog:
    def __init__(self, parent, cameras, selected, accept):
        self.accept = accept
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("选择实验相机")
        height = min(540, max(320, 228 + len(cameras) * 38))
        self.top.geometry(f"640x{height}")
        self.top.minsize(500, 320)
        self.top.transient(parent)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.top.bind("<Return>", lambda _event: self._confirm())
        repaint_after_resize(self.top)

        layout = ttk.Frame(self.top, padding=14)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(1, weight=1)
        instruction = ttk.Label(layout, text="勾选相机再连接预览。同型号可核对序列号。",
                                width=1, justify=tk.LEFT)
        instruction.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        wrap_label_on_resize(instruction)

        self.form = ScrollableForm(layout, padding=(0, 0, 7, 0))
        self.form.grid(row=1, column=0, sticky="nsew")
        self.form.body.columnconfigure(0, weight=1)
        self.choices = {}
        self.checks = {}
        for row, (identity, (name, interface)) in enumerate(cameras.items()):
            choice = tk.BooleanVar(self.top, len(cameras) == 1 or identity in selected)
            self.choices[identity] = choice
            check = ttk.Checkbutton(self.form.body, text=name, variable=choice, command=self._selection_changed)
            check.grid(row=row, column=0, sticky="ew", padx=(0, 8), pady=4)
            self.checks[identity] = check
            source = ttk.Label(self.form.body, text=interface, style="CaptureMuted.TLabel")
            source.grid(row=row, column=1, sticky="e", pady=4)
            help_text = f"{name} · {interface}\n设备编号 {identity}\n勾选后按“连接预览”查看实时画面。"
            ToolTip(check, help_text, title="本次预览相机")
            ToolTip(source, help_text)
        self.form.bind_controls()

        self.status = tk.StringVar(self.top)
        status = ttk.Label(layout, textvariable=self.status, style="CaptureMuted.TLabel", width=1, justify=tk.LEFT)
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        wrap_label_on_resize(status)
        footer = ttk.Frame(layout)
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.connect = ttk.Button(footer, text="连接预览", command=self._confirm, style="Record.TButton")
        self.connect.pack(side=tk.RIGHT)
        ToolTip(self.connect, lambda _event: "先勾选至少一台相机。" if self.connect.instate(["disabled"]) else
                "用勾选的相机更新设备选择，打开实时画面。保存图像时再点“开始采集”。")
        cancel = ttk.Button(footer, text="取消", command=self.top.destroy, style="Small.TButton")
        cancel.pack(side=tk.RIGHT, padx=(0, 8))
        ToolTip(cancel, "关掉选择窗口，保留原来的设备选择。")
        self._selection_changed()
        self.top.grab_set()
        next(iter(self.checks.values())).focus_set()

    def _selected(self):
        return tuple(identity for identity, choice in self.choices.items() if choice.get())

    def _selection_changed(self):
        count = len(self._selected())
        self.status.set(f"已选 {count} 台 · 预览时可检查光照和对焦。" if count else "至少勾选一台相机。")
        self.connect.state(["!disabled" if count else "disabled"])

    def _confirm(self):
        if not (selected := self._selected()):
            return
        if problem := self.accept(selected):
            self.status.set(problem)
        else:
            self.top.destroy()
