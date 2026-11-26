"""Configure and inspect load alignment without hand-editing a settings file."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.controls import ScrollableForm, localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .mechanics_setup import column_hint, inspect_load_csv, preview_mechanics, suggested_capture_basis, table_alignment
from .presentation import LOAD_TIME_BASIS_LABELS, LOAD_SUPPORT_LABELS, ui_error
from .synchronization import load_alignment_settings
from .viewport import repaint_after_resize


TIME_FORMATS = {"absolute": "绝对时间", "relative": "相对计时"}
TIME_UNITS = {"s": "秒 (s)", "ms": "毫秒 (ms)", "us": "微秒 (µs)", "ns": "纳秒 (ns)"}
FORCE_UNITS = {"N": "牛 (N)", "kN": "千牛 (kN)", "column": "用表内单位列"}
DISPLACEMENT_UNITS = {"mm": "毫米 (mm)", "um": "微米 (µm)", "m": "米 (m)"}


class MechanicsDialog:
    def __init__(self, parent, experiment, camera, reference_sequence, linked_reference, current, accept):
        self.experiment, self.camera = experiment, camera
        self.reference_sequence, self.linked_reference = reference_sequence, linked_reference
        self.current, self.accept = current, accept
        self.info = self.config_path = self.candidate = None
        self.busy = False
        self._poll = None
        self._error = None
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("载荷对齐")
        self.top.geometry("820x640")
        self.top.minsize(680, 515)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.grab_set()
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.top.bind("<Destroy>", self._destroy, add="+")
        self.source_label = tk.StringVar(self.top, current.source.name if current else "未选择载荷表")
        self.status = tk.StringVar(self.top, "先选择试验机导出的载荷表，再核对列名、时间单位和测得的帧载荷。")
        if current:
            self.status.set("当前已配置载荷对齐。可选用新的载荷表，或清除当前设置。")
        self.values = {key: tk.StringVar(self.top) for key in (
            "time_column", "force_column", "displacement_column", "time_format", "time_unit", "force_unit",
            "displacement_unit", "basis", "max_gap_s", "sync_frame", "sync_time",
        )}
        self.values["max_gap_s"].set("2.5")
        self.values["basis"].set(current.basis if current else "exposure-midpoint")
        self.fields = []
        self._build()
        for field in self.values.values():
            field.trace_add("write", self._edited)
        self._controls()

    def _build(self):
        panel = ttk.Frame(self.top, padding=12)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=3)
        panel.rowconfigure(2, weight=1)

        source = ttk.Frame(panel)
        source.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        source.columnconfigure(0, weight=1)
        name = ttk.Label(source, textvariable=self.source_label, width=1, style="CaptureSection.TLabel")
        name.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ToolTip(name, self._source_help, title="载荷数据来源")
        self.pick_button = ttk.Button(source, text="选择载荷表…", command=self.choose_table)
        self.pick_button.grid(row=0, column=1, padx=(0, 6))
        ToolTip(self.pick_button, "选试验机导出的 CSV、TSV 或带表头的文本。程序会建议可识别的时间、载荷和位移列。")
        self.import_button = ttk.Button(source, text="导入 JSON…", command=self.choose_config, style="Small.TButton")
        self.import_button.grid(row=0, column=2)
        ToolTip(self.import_button, "读取已有的载荷对齐设置。可使用曝光时钟映射和曝光窗口平均等配置。")

        self.scroller = ScrollableForm(panel)
        self.scroller.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        form = self.scroller.body
        form.columnconfigure(1, weight=3)
        form.columnconfigure(3, weight=2)
        self.time_column = self._select(form, 0, 0, "时间列", "time_column", None,
                                        "选载荷表里记录时间的列。可用带时区的日期时间或以数字记录的时间。")
        self._select(form, 0, 2, "时间格式", "time_format", TIME_FORMATS, "按试验机记录的时间来源选择。", {
            "绝对时间": "带时区的日期时间或 Unix 时间。使用之前请核对两侧绝对时钟的同步。",
            "相对计时": "试验机的计时器读数。填写两侧都记录到的同一同步事件。",
        })
        self.force_column = self._select(form, 1, 0, "载荷列", "force_column", None, "选记录力值的列。")
        self._select(form, 1, 2, "载荷单位", "force_unit", FORCE_UNITS,
                     "核对表头或试验机设置。对齐后统一以 N 显示；kN 会乘以 1000。")
        self.displacement_column = self._select(form, 2, 0, "位移列", "displacement_column", None,
                                               "选记录横梁位移或伸长的列。没有位移记录时留空。")
        self._select(form, 2, 2, "位移单位", "displacement_unit", DISPLACEMENT_UNITS,
                     "位移列有值时确认原单位，结果统一以 mm 显示。")
        self._select(form, 3, 0, "图像时刻", "basis", LOAD_TIME_BASIS_LABELS, "选择用哪个图像时刻匹配载荷。", {
            "曝光中点": "用相机记录的曝光中点。绝对时间需要曝光时钟已换算到 Unix 时间。",
            "主机收图": "用图像到达电脑的时刻。传输和缓冲会影响它与曝光时刻的差值。",
        })
        self._select(form, 3, 2, "数字时间单位", "time_unit", TIME_UNITS,
                     "与时间列表头的数字单位一致。带时区的日期时间会直接解析。")
        self._entry(form, 4, 0, "最大插值间隔 (s)", "max_gap_s",
                    "两条载荷记录之间超过此间隔时留空。识别到采样间隔时，默认建议其三倍，也可按实验修改。")
        self.relative = ttk.Frame(form)
        self.relative.grid(row=5, column=0, columnspan=4, sticky="ew")
        self.relative.columnconfigure(1, weight=1)
        self.relative.columnconfigure(3, weight=1)
        self._entry(self.relative, 0, 0, "同步帧序号", "sync_frame",
                    "填写本轮所选相机已保存、能用来确认同步事件的一帧序号。")
        self._entry(self.relative, 0, 2, "表内对应时间", "sync_time",
                    "填写同一事件在载荷表上的原始时间读数，单位与时间列一致。这个对应关系只校正时间零点。")
        self.relative.grid_remove()
        self.scroller.bind_controls()
        ToolTip(self.scroller.scrollbar, "窗口较小时可滚动查看插值间隔和同步事件设置。")
        self.config_note = ttk.Label(panel, text="使用导入配置中的列名、单位和时钟设置。可用“预览匹配”重新读取。",
                                     style="CaptureMuted.TLabel", justify=tk.LEFT, width=1)
        self.config_note.grid(row=1, column=0, sticky="new", pady=5)
        wrap_label_on_resize(self.config_note)
        self.config_note.grid_remove()

        for box in (self.time_column, self.force_column, self.displacement_column):
            box.bind("<<ComboboxSelected>>", self._suggest_column, add="+")
        preview_box = ttk.Frame(panel)
        preview_box.grid(row=2, column=0, sticky="nsew")
        preview_box.columnconfigure(0, weight=1)
        preview_box.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(preview_box, columns=("frame", "time", "force", "displacement", "support"),
                                  show="headings", height=3, selectmode="browse")
        for key, label, width in (
            ("frame", "帧号", 55), ("time", "表内时刻 (s)", 144), ("force", "载荷 (N)", 115),
            ("displacement", "位移 (mm)", 112), ("support", "匹配依据", 180),
        ):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=width // 2, anchor=tk.W if key == "support" else tk.E)
        self.table.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(preview_box, orient="vertical", command=self.table.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scrollbar.set)
        ToolTip(self.table, "核对参考图、首末实验图和所选同步帧。时间超出记录或缺少有效采样点时，载荷留空。",
                title="载荷匹配预览")
        info = ttk.Label(panel, textvariable=self.status, style="CaptureMuted.TLabel", width=1, justify=tk.LEFT)
        info.grid(row=3, column=0, sticky="ew", pady=7)
        wrap_label_on_resize(info)
        ToolTip(info, lambda _event: ui_error(self._error) if self._error else self.status.get())

        actions = ttk.Frame(panel)
        actions.grid(row=4, column=0, sticky="ew")
        self.clear_button = ttk.Button(actions, text="清除对齐", command=lambda: self._accept(None), style="Small.TButton")
        self.clear_button.pack(side=tk.LEFT)
        ToolTip(self.clear_button, "解除当前载荷设置。随后开始跟踪时只计算图像应变。")
        close = ttk.Button(actions, text="取消", command=self.top.destroy, style="Small.TButton")
        close.pack(side=tk.RIGHT)
        ToolTip(close, "关闭窗口并保留之前应用的设置。")
        self.apply_button = ttk.Button(actions, text="应用到实时分析", command=lambda: self._accept(self.candidate),
                                       style="Record.TButton")
        self.apply_button.pack(side=tk.RIGHT, padx=6)
        ToolTip(self.apply_button, lambda _event: "请先预览匹配，确认列、单位和同步事件。"
                if not self.candidate else "使用本次预览的设置。实时跟踪和结果导出会带上对应载荷。")
        self.preview_button = ttk.Button(actions, text="预览匹配", command=self.preview, style="Small.TButton")
        self.preview_button.pack(side=tk.RIGHT)
        ToolTip(self.preview_button, "读取完整载荷表，核对时间顺序、单位及已保存图像的匹配结果。")
        self.pick_button.focus_set()

    def _select(self, parent, row, start, label, key, choices, help_text, tips=None):
        value = self.values[key]
        if choices:
            widget = localized_combobox(parent, value, choices, width=10)
        else:
            widget = ttk.Combobox(parent, textvariable=value, state="readonly", width=10)
        self._position(parent, row, start, label, widget, help_text, tips)
        return widget

    def _entry(self, parent, row, start, label, key, help_text):
        widget = ttk.Entry(parent, textvariable=self.values[key], width=10)
        self._position(parent, row, start, label, widget, help_text, None)

    def _position(self, parent, row, start, label, widget, help_text, tips):
        caption = ttk.Label(parent, text=label)
        caption.grid(row=row, column=start, sticky="w", padx=(0 if start == 0 else 10, 5))
        widget.grid(row=row, column=start + 1, sticky="ew", padx=(0, 5), pady=3)
        tooltip = ChoiceToolTip(widget, help_text, tips, title=label) if tips else ToolTip(widget, help_text, title=label)
        ToolTip(caption, tooltip.resolve_text)
        self.fields.append(widget)

    def _source_help(self, _event=None):
        if self.info:
            return f"{self.info.path}\n文件编码 {self.info.encoding}"
        path = self.config_path or (self.current.source if self.current else None)
        return str(path) if path else "选择带表头的载荷表，或读取之前配置的 JSON。"

    def _suggest_column(self, event):
        if event.widget is self.time_column:
            form, unit, gap = self.info.time_settings(self.values["time_column"].get())
            for key, value in (("time_format", form), ("time_unit", unit), ("max_gap_s", gap or "2.5"), ("sync_time", "")):
                self.values[key].set(value)
        else:
            key = "force" if event.widget is self.force_column else "displacement"
            field = self.values[f"{key}_column"].get()
            unit = self.info.force_unit_hint(field) if key == "force" else column_hint(field)[1]
            self.values[f"{key}_unit"].set(unit or "")

    def _invalidate(self):
        self.candidate = None
        self._error = None
        self.table.delete(*self.table.get_children())

    def _edited(self, *_args):
        if self.busy:
            return
        self._invalidate()
        self.status.set("设置已修改。预览匹配后可应用。")
        self._controls()

    def _controls(self):
        if self.values["time_format"].get() == "relative":
            self.relative.grid()
        else:
            self.relative.grid_remove()
        for widget in self.fields:
            widget.state(["disabled" if self.busy or self.info is None else "!disabled"])
        for button, available in ((self.pick_button, True), (self.import_button, True),
                                  (self.preview_button, self.info or self.config_path),
                                  (self.apply_button, self.candidate), (self.clear_button, self.current)):
            button.state(["!disabled" if available and not self.busy else "disabled"])

    def _work(self, operation, finished, progress):
        results = queue.SimpleQueue()
        self.busy = True
        self._invalidate()
        self.status.set(progress)
        self._controls()

        def run():
            try:
                results.put((operation(), None))
            except Exception as error:
                results.put((None, error))

        def poll():
            self._poll = None
            try:
                value, error = results.get_nowait()
            except queue.Empty:
                self._poll = self.top.after(50, poll)
                return
            if error:
                self._error = error
                self.status.set(ui_error(error, details=False))
            else:
                finished(value)
            self.busy = False
            self._controls()

        threading.Thread(target=run, name="preview-live-load", daemon=True).start()
        self._poll = self.top.after(50, poll)

    def _destroy(self, event):
        if event.widget is self.top and self._poll is not None:
            self.top.after_cancel(self._poll)

    def choose_table(self):
        path = filedialog.askopenfilename(parent=self.top, title="选择试验机载荷表",
                                          filetypes=[("载荷表", "*.csv *.tsv *.txt"), ("所有文件", "*.*")])
        if path:
            self._work(lambda: (inspect_load_csv(path), suggested_capture_basis(self.experiment, self.camera)),
                       self._show_columns, "正在读取表头和采样间隔")

    def _show_columns(self, value):
        self.info, basis = value
        self.config_path = None
        self.config_note.grid_remove()
        self.scroller.grid()
        self.scroller.canvas.yview_moveto(0)
        self.source_label.set(self.info.path.name)
        for box in (self.time_column, self.force_column, self.displacement_column):
            box.configure(values=(("",) if box is self.displacement_column else ()) + self.info.headers)
        suggestions = self.info.suggest()
        for key in ("time", "force", "displacement"):
            self.values[f"{key}_column"].set(suggestions[key])
        for key in ("time_format", "time_unit", "force_unit", "displacement_unit"):
            self.values[key].set(suggestions[key])
        self.values["max_gap_s"].set(suggestions["gap"] or "2.5")
        self.values["basis"].set(basis)
        self.values["sync_frame"].set("")
        self.values["sync_time"].set("")
        self.status.set("已读取列名。核对建议的列和单位，随后点“预览匹配”。")

    def choose_config(self):
        path = filedialog.askopenfilename(parent=self.top, title="导入载荷对齐配置",
                                          filetypes=[("力学信号对齐配置", "*.json")])
        if path:
            self.config_path, self.info = Path(path), None
            self.source_label.set(self.config_path.name)
            self.scroller.grid_remove()
            self.config_note.grid()
            self.preview()

    def preview(self):
        scope = {"reference_sequence": self.reference_sequence, "linked_reference": self.linked_reference}
        info, path = self.info, self.config_path
        options = {key: value.get() for key, value in self.values.items()}

        def inspect():
            if path:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                channel = load_alignment_settings(data.get("mechanics", data), base_dir=path.parent)
            else:
                channel = table_alignment(info, self.experiment, self.camera, **options)
            summary, rows = preview_mechanics(channel, self.experiment, self.camera, **scope)
            return channel, summary, rows

        self._work(inspect, self._show_preview, "正在核对时间、单位及已保存图像")

    def _show_preview(self, value):
        channel, summary, rows = value

        def number(value):
            return f"{value:.9g}" if value is not None else "—"

        for row in rows:
            when = row["load_time_s"]
            code = row["force_support"]["status"]
            self.table.insert("", "end", values=(row["frame"], f"{when:.6f}" if when is not None else "—",
                              number(row["force_N"]), number(row["displacement_mm"]), LOAD_SUPPORT_LABELS.get(code, code)))
        valid = summary["valid_force_rows"]
        support = sum(row["force_N"] is not None for row in rows)
        if summary["rows"] == 0:
            self.candidate = channel
            self.status.set("已读取列名 · 已设置对齐参数，可等待试验机写入载荷。")
        elif valid:
            self.candidate = channel
            detail = f"预览帧匹配 {support}/{len(rows)} 张" if rows else "等待相机保存图像"
            if rows and not support and all(row["load_time_s"] is None for row in rows):
                detail = "当前图像时钟不可用 · 检查图像时刻或时钟设置"
            self.status.set(f"可用载荷记录 {valid:,}/{summary['rows']:,} 行 · {detail}")
        else:
            self.status.set("所选列没有有效载荷。请核对原始数值、单位及有效性标记。")

    def _accept(self, channel):
        self.accept(channel)
        self.top.destroy()
