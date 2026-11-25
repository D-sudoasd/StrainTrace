"""Desktop workflow for producing a verified experiment copy."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .config import CaptureError
from .presentation import format_bytes, ui_error
from .transfer import TransferProgress, copy_experiment
from .viewport import repaint_after_resize


class ExperimentCopyWindow:
    def __init__(self, parent, source, experiment_id, submit, open_experiment):
        self.source = Path(source)
        self.experiment_id = experiment_id
        self.submit, self.open_experiment = submit, open_experiment
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("制作交接副本")
        self.top.geometry("690x450")
        self.top.minsize(530, 400)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self._default_name = f"{self.source.name}-交接"
        self._source_text = tk.StringVar(self.top, str(self.source))
        self.destination = tk.StringVar(self.top, str(self.source.parent / self._default_name))
        self.attachment_text = tk.StringVar(self.top, "未添加附件")
        self.status = tk.StringVar(self.top, "请选择目标位置，再点“复制并校验”。")
        self._attachments = ()
        self._cancel = threading.Event()
        self._latest = TransferProgress("preparing")
        self._working = False
        self._result = None
        self._detail = ""
        self._build()
        self.destination.trace_add("write", self._controls)
        self._controls()
        self.top.protocol("WM_DELETE_WINDOW", self._close)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self._close())
        self.top.bind("<Destroy>", self._destroyed, add="+")

    def _build(self):
        area = ttk.Frame(self.top, padding=16)
        area.pack(fill=tk.BOTH, expand=True)
        area.columnconfigure(1, weight=1)
        area.rowconfigure(5, weight=1)
        scope = ttk.Label(area, width=1, wraplength=600, justify=tk.LEFT,
                          text="打开副本即可查看原图、台账和归档标定。\n可以附上载荷、脉冲和探测器记录。")
        scope.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 13))
        wrap_label_on_resize(scope)

        source_label = ttk.Label(area, text="源实验")
        source_label.grid(row=1, column=0, sticky="w", padx=(0, 12), pady=(0, 6))
        source = ttk.Entry(area, state="readonly", textvariable=self._source_text)
        source.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(0, 6))
        for widget in (source_label, source):
            ToolTip(widget, str(self.source), title="当前源实验目录")

        target_label = ttk.Label(area, text="目标目录")
        target_label.grid(row=2, column=0, sticky="w", padx=(0, 12), pady=4)
        self.target = ttk.Entry(area, textvariable=self.destination)
        self.target.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=4)
        self.choose = ttk.Button(area, text="选择位置…", command=self._choose_destination, style="Small.TButton")
        self.choose.grid(row=2, column=2, sticky="ew")
        for widget in (self.target, target_label):
            ToolTip(widget, lambda _event: "填写尚未创建的目标目录。上一级目录须已存在。"
                    "中断后，可使用相同的目标目录继续。"
                    + (f"\n\n当前目标\n{self.destination.get()}" if self.destination.get().strip() else ""), title="目标目录")
        ToolTip(self.choose, "选择另一块磁盘或共享位置。程序会在所选位置下填写一个新的实验目录名。")

        attachments = ttk.Frame(area)
        attachments.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 10))
        attachments.columnconfigure(0, weight=1)
        summary = ttk.Label(attachments, textvariable=self.attachment_text, width=1, style="CaptureMuted.TLabel")
        summary.grid(row=0, column=0, sticky="ew")
        ToolTip(summary, lambda _event: "\n".join(map(str, self._attachments)) or "原图、台账和归档标定已在复制范围内。实验的额外记录可在这里添加。",
                title="随副本保存的附件")
        self.add_files = ttk.Button(attachments, text="添加附件…", command=self._choose_attachments, style="Small.TButton")
        self.add_files.grid(row=0, column=1, padx=(6, 4))
        ToolTip(self.add_files, "可多选外部记录文件。它们会按原文件名保存在副本的附件目录。")
        self.clear = ttk.Button(attachments, text="清空", command=lambda: self._set_attachments(()), style="Small.TButton")
        self.clear.grid(row=0, column=2)
        ToolTip(self.clear, "只清除本次选择的附件列表。")

        self.bar = ttk.Progressbar(area, mode="determinate", maximum=1000, style="CaptureCopy.Horizontal.TProgressbar")
        self.bar.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 5))
        ToolTip(self.bar, "进度按副本中的文件字节数计算。每个文件核验后才计入已完成文件数。")
        status = ttk.Label(area, textvariable=self.status, wraplength=600, width=1, justify=tk.LEFT,
                           style="CaptureMuted.TLabel")
        status.grid(row=5, column=0, columnspan=3, sticky="new", pady=(4, 0))
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.status.get(), title="复制状态")

        actions = ttk.Frame(area)
        actions.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.close_button = ttk.Button(actions, text="关闭", command=self._close)
        self.close_button.pack(side=tk.LEFT)
        ToolTip(self.close_button, lambda _event: "请求取消复制。再次使用同一位置时可继续。" if self._working else "关闭交接窗口。")
        self.start_button = ttk.Button(actions, text="复制并校验", command=self.start, style="Record.TButton")
        self.start_button.pack(side=tk.RIGHT)
        ToolTip(self.start_button, self._start_help)
        self.open_button = ttk.Button(actions, text="打开副本", command=self._open, style="Record.TButton")
        ToolTip(self.open_button, "用当前工作台打开新副本。此后可以查图、核验交接清单或进入应变分析。")
        self.target.focus_set()

    def _choose_destination(self):
        parent = filedialog.askdirectory(parent=self.top, title="选择交接副本的保存位置", mustexist=True)
        if parent:
            self.destination.set(str(Path(parent) / self._default_name))

    def _choose_attachments(self):
        files = filedialog.askopenfilenames(parent=self.top, title="选择随副本保存的记录文件",
                                            filetypes=[("常见记录", "*.csv *.tsv *.json *.txt"), ("所有文件", "*.*")])
        if files:
            self._set_attachments(tuple(dict.fromkeys((*self._attachments, *files))))

    def _set_attachments(self, files):
        self._attachments = files
        self.attachment_text.set(f"已选 {len(files)} 个附加文件" if files else "未添加附件")
        self._controls()

    def _controls(self, *_args):
        state = "disabled" if self._working or self._result is not None else "!disabled"
        for widget in (self.choose, self.add_files):
            widget.state([state])
        self.target.state(["disabled", "!readonly"] if self._working else
                          ["!disabled", "readonly" if self._result is not None else "!readonly"])
        self.clear.state([state if self._attachments else "disabled"])
        self.start_button.state([state if self.destination.get().strip() else "disabled"])
        self.close_button.configure(text="取消复制" if self._working else "关闭")
        self.close_button.state(["disabled" if self._working and self._cancel.is_set() else "!disabled"])

    def _start_help(self, _event):
        if self._working:
            return "正在处理本次复制。如需暂停，可点击“取消复制”。"
        if not self.destination.get().strip():
            return "请先选择目标位置，或填写新目录的完整路径。"
        return "先读取源台账，再逐项复制和核对文件。通过后会创建可打开的目标目录。"

    def start(self):
        if self._working or self._result is not None or not (target := self.destination.get().strip()):
            return
        self._working = True
        self._cancel.clear()
        self._latest = TransferProgress("preparing")
        self._detail = ""
        self.bar.configure(value=0)
        self._controls()
        attachments = self._attachments

        def progress(update):
            self._latest = update

        self.submit(lambda: copy_experiment(self.source, target, attachments=attachments, expected_id=self.experiment_id,
                                            cancelled=self._cancel.is_set, progress=progress), self._finished, self._failed)
        self._poll()

    def _poll(self):
        if not self._working or not self.top.winfo_exists():
            return
        update = self._latest
        if self._cancel.is_set():
            self.status.set("正在取消复制。已核验的文件会留在目标盘，供下次继续。")
        elif update.stage == "preparing":
            self.status.set("正在读取台账、核对复制范围…")
        else:
            labels = {"copying": "正在复制", "checking": "正在核验", "publishing": "正在生成交接目录"}
            self.status.set(f"{labels[update.stage]} · 已核验 {update.files_done}/{update.files_total} 项"
                            + (f" · 已复用 {update.reused} 项" if update.reused else "")
                            + (f"\n{Path(update.path).name}" if update.path else ""))
            self._detail = (f"{self.status.get()}\n{format_bytes(update.bytes_done)} / {format_bytes(update.bytes_total)}"
                            + (f"\n{update.path}" if update.path else ""))
            self.bar.configure(value=1000 * update.bytes_done / max(update.bytes_total, 1))
        self.top.after(150, self._poll)

    def _finished(self, result):
        if not self.top.winfo_exists():
            return
        self._working = False
        self._result = result
        self.bar.configure(value=1000)
        self.status.set(f"交接副本已通过校验\n原图 {result['registered_images']} 张 · 附件 {result['attachment_files']} 个"
                        f" · {format_bytes(result['bytes'])}")
        self._detail = f"{self.status.get()}\n{result['output']}"
        self.start_button.pack_forget()
        self.open_button.pack(side=tk.RIGHT)
        self._controls()
        self.open_button.focus_set()

    def _failed(self, exc):
        if not self.top.winfo_exists():
            return
        self._working = False
        self._detail = ui_error(exc)
        self.status.set(ui_error(exc, details=False) if isinstance(exc, CaptureError) else
                        "复制未完成。悬停查看详细信息，检查目标位置的可用空间和权限后可重试。")
        self._controls()

    def _open(self):
        if self._result is None:
            return
        self.open_button.state(["disabled"])

        def failed(exc):
            if self.top.winfo_exists():
                self._detail = ui_error(exc)
                self.status.set("无法打开副本。悬停可查看详细信息。")
                self.open_button.state(["!disabled"])

        if not self.open_experiment(self._result["output"], expected_id=self.experiment_id,
                                    on_opened=self.top.destroy, on_error=failed,
                                    wanted=lambda: bool(self.top.winfo_exists())):
            failed("请等待采集工作台当前操作结束，再打开副本。")

    def _close(self):
        if self._working:
            self._cancel.set()
            self._controls()
        else:
            self.top.destroy()

    def _destroyed(self, event):
        if event.widget is self.top:
            self._cancel.set()
