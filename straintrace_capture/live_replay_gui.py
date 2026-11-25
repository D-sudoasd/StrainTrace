"""Start and monitor an ordered replay from saved live measurements."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .live_replay import replay_live_results
from .presentation import ui_error
from .viewport import repaint_after_resize


class LiveReplayWindow:
    def __init__(self, parent, index, on_saved):
        self.index, self.on_saved = index, on_saved
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("StrainTrace · 按序重算")
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.geometry("610x396" if index.load_source else "610x310")
        self.top.minsize(485, 384 if index.load_source else 292)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self._close())
        self.top.bind("<Destroy>", self._destroy, add="+")
        self.top.protocol("WM_DELETE_WINDOW", self._close)
        self.use_load = tk.BooleanVar(self.top, index.load_source is not None)
        self.load_label = tk.StringVar(self.top)
        self.status = tk.StringVar(self.top, "可点击“开始重算”。完成后将打开新的历史曲线。")
        self._load_table = None
        self._cancel = threading.Event()
        self._result = queue.SimpleQueue()
        self._latest = (0, 0)
        self._after = None
        self._working = False
        self._detail = None
        self._build()
        self._controls()
        self.start_button.focus_set()

    def _build(self):
        panel = ttk.Frame(self.top, padding=14)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="用原测量设置重算已保存的图像", style="CaptureSection.TLabel").grid(row=0, sticky="w")
        scope = ttk.Label(
            panel, width=1, justify=tk.LEFT,
            text="沿用原参考图、标距和全场测量区，按相机序号逐张计算。实时跳过的实验图也会纳入。",
        )
        scope.grid(row=1, sticky="ew", pady=(6, 3))
        wrap_label_on_resize(scope)
        source = ttk.Label(
            panel, width=1, style="CaptureMuted.TLabel",
            text=f"{self.index.camera} · 原记录已计算 {self.index.frames - 1:,} 张 · 实时跳过 {self.index.skipped:,} 张",
        )
        source.grid(row=2, sticky="ew", pady=(0, 8))
        ToolTip(source, self.index.path.name + "\n图像范围以点击开始重算后读取到的台账为准。", title="原逐帧记录")

        if self.index.load_source is not None:
            options = ttk.Frame(panel)
            options.grid(row=3, sticky="ew", pady=(0, 6))
            self.load_toggle = ttk.Checkbutton(options, text="按原设置对齐载荷和位移",
                                               variable=self.use_load, command=self._controls)
            self.load_toggle.pack(anchor="w")
            ToolTip(self.load_toggle, "默认沿用原载荷表、时间基准和同步设置。取消勾选后，新曲线仅包含图像测量结果。")
            line = ttk.Frame(options)
            line.pack(fill=tk.X, pady=(3, 0))
            label = ttk.Label(line, textvariable=self.load_label, width=1, style="CaptureMuted.TLabel")
            label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
            ToolTip(label, lambda _event: str(self._load_table or self.index.load_source), title="当前载荷表")
            self.pick_button = ttk.Button(line, text="更换载荷表…", command=self._pick, style="Small.TButton")
            self.pick_button.pack(side=tk.RIGHT)
            ToolTip(self.pick_button, "原表移位或需要补入后续采样时，可选择同一轮实验的更新表。程序会核对原先用过的采样。")

        self.bar = ttk.Progressbar(panel, mode="determinate", maximum=1, style="CaptureCopy.Horizontal.TProgressbar")
        self.bar.grid(row=4, sticky="ew", pady=(7, 5))
        ToolTip(self.bar, "先核对原记录，再统计已完成计算的实验图。参考图单独处理。")
        status = ttk.Label(panel, width=1, textvariable=self.status, justify=tk.LEFT, style="CaptureMuted.TLabel")
        status.grid(row=5, sticky="new", pady=(2, 0))
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.status.get(), title="重算状态")
        panel.rowconfigure(5, weight=1, minsize=status.winfo_reqheight())

        actions = ttk.Frame(panel)
        actions.grid(row=6, sticky="ew", pady=(9, 0))
        self.close_button = ttk.Button(actions, text="关闭", command=self._close)
        self.close_button.pack(side=tk.LEFT)
        ToolTip(self.close_button, lambda _event: "处理完当前图像后取消这次重算。" if self._working else "关闭重算窗口。")
        self.start_button = ttk.Button(actions, text="开始重算", command=self.start, style="Record.TButton")
        self.start_button.pack(side=tk.RIGHT)
        ToolTip(self.start_button, "读取此时已保存的图像并逐张计算。完成后在实验目录内保存一份新的逐帧记录。")

    def _pick(self):
        current = self._load_table or self.index.load_source
        name = filedialog.askopenfilename(
            parent=self.top, title="选择同一轮实验的载荷表", initialdir=current.parent,
            filetypes=[("载荷表", "*.csv *.tsv *.txt"), ("所有文件", "*.*")],
        )
        if name:
            self._load_table = Path(name)
            self._controls()

    def _controls(self):
        if self.index.load_source is not None:
            self.load_toggle.state(["disabled" if self._working else "!disabled"])
            self.pick_button.state(["!disabled" if self.use_load.get() and not self._working else "disabled"])
            source = self._load_table or self.index.load_source
            self.load_label.set(source.name if self.use_load.get() else "这次只重算图像测量")
        self.start_button.state(["disabled" if self._working else "!disabled"])
        self.close_button.configure(text="取消重算" if self._working else "关闭")
        self.close_button.state(["disabled" if self._working and self._cancel.is_set() else "!disabled"])

    def start(self):
        if self._working:
            return
        self._working, self._detail, self._latest = True, None, (0, 0)
        self._cancel.clear()
        self.bar.configure(value=0)
        self.status.set("正在读取重算范围")
        self._controls()
        images_only = self.index.load_source is not None and not self.use_load.get()
        table = self._load_table if self.use_load.get() else None

        def progress(done, total):
            self._latest = done, total

        def work():
            try:
                result = replay_live_results(self.index, images_only=images_only, load_table=table,
                                             cancelled=self._cancel.is_set, progress=progress)
            except Exception as error:
                self._result.put((None, error))
            else:
                self._result.put((Path(result["record"]), None))

        threading.Thread(target=work, name="capture-live-replay", daemon=True).start()
        self._after = self.top.after(50, self._poll)

    def _poll(self):
        self._after = None
        try:
            path, error = self._result.get_nowait()
        except queue.Empty:
            done, total = self._latest
            self.bar.configure(value=done, maximum=max(1, total))
            if total and not self._cancel.is_set():
                stage = "正在逐张计算" if done else "正在核对原记录和测量设置"
                self.status.set(f"{stage} · {done:,} / {total:,} 张实验图")
            self._after = self.top.after(75, self._poll)
            return
        self._working = False
        if error is not None:
            self._detail = ui_error(error)
            self.status.set(ui_error(error, details=False))
            self._controls()
        else:
            self.top.destroy()
            self.on_saved(path)

    def _close(self):
        if not self._working:
            self.top.destroy()
            return
        self._cancel.set()
        self.status.set("正在取消重算")
        self._controls()

    def _destroy(self, event):
        if event.widget is self.top:
            self._cancel.set()
            if self._after is not None:
                self.top.after_cancel(self._after)
