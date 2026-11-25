"""Chinese desktop workflow for exporting a reference-first TIFF sequence."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.controls import ScrollableForm, localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .config import CaptureError
from .frame_search import parse_sequence_range
from .presentation import format_bytes, ui_error
from .sequence_export import SequenceSelection, export_sequence, inspect_sequence
from .transfer import TransferProgress
from .viewport import repaint_after_resize


class SequenceExportWindow:
    def __init__(self, parent, source, experiment_id, cameras, active_camera, phases, active_part, submit,
                 *, event=None, reference_sequence=None):
        self.source = Path(source)
        self.experiment_id = experiment_id
        self.submit = submit
        self._part_camera = active_camera
        self._active_part = active_part
        self.event = event
        suffix = f"-阶段{event['phase']}-事件{event['event_number']}" if event is not None else ""
        self._default_name = f"{self.source.name}{suffix}-图像序列"
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("导出图像序列")
        self.top.transient(parent)
        self.top.geometry("740x470")
        self.top.minsize(540, 425)
        repaint_after_resize(self.top)
        initial_camera = active_camera if active_camera in cameras else next(iter(cameras))
        self.camera = tk.StringVar(self.top, initial_camera)
        self.phase = tk.StringVar(self.top, str(event["phase"]) if event is not None else "")
        self.reference = tk.StringVar(self.top, "" if reference_sequence is None else str(reference_sequence))
        self.sequence = tk.StringVar(self.top)
        self.destination = tk.StringVar(self.top, str(self.source.parent / self._default_name))
        self.part = tk.StringVar(self.top)
        self.status = tk.StringVar(self.top, "可先检查范围，查看选中的参考图、实验图数量和所需空间。")
        self._cancel = threading.Event()
        self._latest = TransferProgress("preparing")
        self._working = ""
        self._report = None
        self._detail = ""
        self._inputs = []
        self._build(cameras, phases)
        self._refresh_part()
        self.camera.trace_add("write", self._refresh_part)
        for value in (self.camera, self.phase, self.reference, self.sequence, self.destination, self.part):
            value.trace_add("write", self._edited)
        self.top.protocol("WM_DELETE_WINDOW", self._close)
        self.top.bind("<Destroy>", self._destroyed, add="+")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top), add="+")
        self._sync_controls()
        self.camera_box.focus_set()

    def _build(self, cameras, phases):
        panel = ttk.Frame(self.top, padding=(20, 13))
        panel.pack(fill=tk.BOTH, expand=True)
        footer = ttk.Frame(panel)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        source = ttk.Label(panel, text=f"源实验  {self.source.name}", style="CaptureMuted.TLabel")
        source.pack(fill=tk.X, pady=(1, 8))
        wrap_label_on_resize(source)
        ToolTip(source, str(self.source), title="当前源实验")
        scroll = ScrollableForm(panel)
        scroll.pack(fill=tk.BOTH, expand=True)
        form = scroll.body
        form.columnconfigure(1, weight=1)

        self.camera_box = localized_combobox(form, self.camera, cameras)
        self._field(form, 0, "相机", self.camera_box,
                    "每次导出一台相机。若两台相机同时拍摄，请分别导出，再按索引中的触发事件编号配对。")
        if self.event is None:
            self._field(form, 1, "实验阶段", localized_combobox(form, self.phase, {"": "全部实验阶段", **phases}),
                        "筛选用于记录变形的实验图。参考图独立选择，可以来自更早的阶段。")
        else:
            title = f"阶段 {self.event['phase']} · 事件 {self.event['event_number']}"
            self._field(form, 1, "导出范围", ttk.Label(form, text=title), "按触发编号读取这台相机在本次事件中的原图。")
        self._field(form, 2, "参考帧序号", ttk.Entry(form, textvariable=self.reference),
                    "更换基准时可输入事件内的原帧序号，或本事件之前登记的实验参考帧序号。"
                    if self.event is not None else
                    "留空时优先使用记录的参考图。无参考阶段时用首张实验图。有多张参考图时，请在“查找图像”中核对"
                    "变形前的图像，再输入对应的原帧序号。")
        if self.event is None:
            self._field(form, 3, "实验帧范围", ttk.Entry(form, textvariable=self.sequence),
                        "可输入单个原帧序号，如 42，或连续范围 40-80，包含两端。留空时取该阶段中晚于参考图的全部实验帧。")
        self._part_box = ttk.Combobox(form, state="readonly", textvariable=self.part)
        self._field(form, 4, "图像部件", self._part_box,
                    "一般使用主图像。如果同一帧保存多个图像部件，可先在复查区选中所需部件，再打开本窗口。")
        self._field(form, 5, "目标目录", ttk.Entry(form, textvariable=self.destination),
                    "输入要新建的目录完整路径。上一级目录需已存在。输出是连续编号的 TIFF、帧索引及核验清单。",
                    columns=1)
        self.choose = ttk.Button(form, text="选择位置…", command=self._choose, style="Small.TButton")
        self.choose.grid(row=5, column=2, sticky="ew", padx=(8, 2), pady=3)
        self._inputs.append(self.choose)
        ToolTip(self.choose, "选择上一级目录，工作台会填入新的图像序列目录名。")
        scroll.bind_controls()

        self.bar = ttk.Progressbar(footer, mode="determinate", style="CaptureCopy.Horizontal.TProgressbar")
        self.bar.pack(fill=tk.X, pady=(10, 5))
        status = ttk.Label(footer, textvariable=self.status, justify="left")
        status.pack(fill=tk.X)
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.status.get(), title="导出详情")
        actions = ttk.Frame(footer)
        actions.pack(fill=tk.X, pady=(9, 1))
        self.check_button = ttk.Button(actions, text="检查范围", command=lambda: self.start("check"), style="Small.TButton")
        self.check_button.pack(side=tk.LEFT)
        ToolTip(self.check_button, "读取台账、检查图像尺寸与像素格式，并估算输出大小。可根据结果调整筛选。")
        self.open_button = ttk.Button(actions, text="打开文件夹", command=self._open_folder, style="Small.TButton")
        self.open_button.pack(side=tk.LEFT, padx=(7, 0))
        ToolTip(self.open_button, "在资源管理器中打开图像序列，供其他软件使用。")
        self.close_button = ttk.Button(actions, text="关闭", command=self._close, style="Small.TButton")
        self.close_button.pack(side=tk.RIGHT)
        ToolTip(self.close_button, lambda _event: "停止本次操作并清理临时文件。" if self._working else "返回采集工作台。")
        self.start_button = ttk.Button(actions, text="导出并核验", command=lambda: self.start("export"),
                                       style="Confirm.Record.TButton", width=0)
        self.start_button.pack(side=tk.RIGHT, padx=(0, 7))
        ToolTip(self.start_button, "参考图排在第 0 张，后续实验帧依次编号。逐个核对 TIFF 校验值后生成新目录。")

    def _field(self, form, row, title, widget, help_text, *, columns=2):
        label = ttk.Label(form, text=title)
        label.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
        widget.grid(row=row, column=1, columnspan=columns, sticky="ew", pady=3)
        self._inputs.append(widget)
        ToolTip(label, help_text, title=title)
        ToolTip(widget, help_text, title=title)

    def _refresh_part(self, *_args):
        choices = ["主图像"]
        if self.camera.get() == self._part_camera and self._active_part:
            choices.append(self._active_part)
        self._part_box.configure(values=choices)
        self.part.set(choices[-1])

    def _choose(self):
        chosen = filedialog.askdirectory(parent=self.top, title="选择图像序列的上一级目录",
                                         initialdir=self.source.parent, mustexist=True)
        if chosen:
            self.destination.set(str(Path(chosen) / self._default_name))

    def _edited(self, *_args):
        if self._working:
            return
        self._report = None
        self._detail = ""
        self.bar.configure(value=0)
        self.status.set("可先检查范围，查看选中的参考图、实验图数量和所需空间。")
        self._sync_controls()

    def _selection(self):
        ref, ref_end = parse_sequence_range(self.reference.get())
        if ref != ref_end:
            raise CaptureError("INVALID_FRAME_QUERY", "参考帧只接受一个帧序号。")
        first, last = parse_sequence_range(self.sequence.get())
        part = self.part.get()
        return SequenceSelection(camera_id=self.camera.get(), phase=int(self.phase.get()) if self.phase.get() else None,
                                 reference_sequence=ref, first_sequence=first, last_sequence=last,
                                 image_part=None if part == "主图像" else part,
                                 event_context_id=self.event["signal_id"] if self.event is not None else None)

    def _sync_controls(self):
        available = bool(self.destination.get().strip()) and not self._working
        for widget in self._inputs:
            widget.state(["disabled" if self._working else "!disabled"])
        for button in (self.check_button, self.start_button):
            button.state(["!disabled" if available and not (self._report or {}).get("ok") else "disabled"])
        self.open_button.state(["!disabled" if (self._report or {}).get("ok") and hasattr(os, "startfile") else "disabled"])
        self.close_button.configure(text="取消" if self._working else "关闭")
        self.close_button.state(["disabled" if self._working and self._cancel.is_set() else "!disabled"])

    def start(self, mode):
        target = self.destination.get().strip()
        if not target or self._working or (self._report or {}).get("ok"):
            return
        try:
            selection = self._selection()
        except CaptureError as exc:
            self._failed(exc)
            return
        self._working = mode
        self._detail = ""
        self._cancel.clear()
        self._latest = TransferProgress("preparing")
        self.bar.configure(value=0)
        self._sync_controls()

        def progress(update):
            self._latest = update

        args = {"expected_id": self.experiment_id, "cancelled": self._cancel.is_set}
        if mode == "check":
            self.submit(lambda: inspect_sequence(self.source, target, selection, **args), self._finished, self._failed)
        else:
            self.submit(lambda: export_sequence(self.source, target, selection, progress=progress, **args),
                        self._finished, self._failed)
        self._poll()

    def _poll(self):
        if not self._working or not self.top.winfo_exists():
            return
        update = self._latest
        if self._cancel.is_set():
            self.status.set("正在取消并清理临时文件…")
        elif update.stage == "preparing":
            self.status.set("正在读取台账、检查所选图像…")
        elif update.stage == "publishing":
            self.status.set("图像已核验，正在创建目标目录…")
        else:
            label = "正在复制原始 TIFF" if update.stage == "copying" else "正在核验 TIFF"
            self.status.set(f"{label}  {update.files_done}/{update.files_total} 张 · {format_bytes(update.bytes_done)}")
            share = update.bytes_done / max(1, update.bytes_total) * 50
            self.bar.configure(value=share + (50 if update.stage == "checking" else 0))
        self.top.after(100, self._poll)

    def _finished(self, result):
        if not self.top.winfo_exists():
            return
        self._working = ""
        self._report = result
        summary = (f"参考图为原帧 {result['reference_sequence']}，另有 {result['test_frames']} 张实验图。"
                   f" TIFF 共 {format_bytes(result['image_bytes'])}。")
        if result.get("ok"):
            self.status.set(f"导出并核验完成。{summary}")
            self.bar.configure(value=100)
        else:
            available = result["available_bytes"]
            space = "目标盘空间不足。" if available < result["image_bytes"] else f"目标盘可用 {format_bytes(available)}。"
            self.status.set(summary + "\n" + space)
        self._detail = summary + "\n目标目录\n" + result["output"]
        self._sync_controls()

    def _failed(self, exc):
        if not self.top.winfo_exists():
            return
        self._working = ""
        self._report = None
        self._detail = ui_error(exc)
        self.status.set(self._detail.split("\n", 1)[0])
        self.bar.configure(value=0)
        self._sync_controls()

    def _open_folder(self):
        if self._report and self._report.get("ok") and hasattr(os, "startfile"):
            try:
                os.startfile(self._report["output"])
            except OSError as exc:
                self._detail = ui_error(exc)
                self.status.set("无法打开目标目录。悬停可查看详细信息。")

    def _close(self):
        if self._working:
            self._cancel.set()
            self._sync_controls()
        else:
            self.top.destroy()

    def _destroyed(self, event):
        if event.widget is self.top:
            self._cancel.set()
