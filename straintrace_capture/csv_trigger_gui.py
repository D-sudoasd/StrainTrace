"""Set a file, data column and crossing direction for event-window capture."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.controls import ScrollableForm, localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .config import CaptureError, CsvTrigger
from .csv_trigger import inspect_csv_trigger
from .mechanics_setup import column_hint
from .presentation import CSV_DIRECTIONS, ui_error
from .viewport import repaint_after_resize


class CsvTriggerDialog:
    def __init__(self, parent, rule, accept, submit):
        self.parent, self.accept, self.submit = parent, accept, submit
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.transient(parent)
        self.top.title("试验机 CSV 阈值触发")
        self.top.geometry("550x480")
        self.top.minsize(390, 300)
        self.top.grab_set()
        repaint_after_resize(self.top)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.close())
        self.path = None
        self.header = None
        previous = rule or {}
        self.file_label = tk.StringVar(self.top, "尚未选择")
        self.value = tk.StringVar(self.top)
        self.source_time = tk.StringVar(self.top)
        self.quantity = tk.StringVar(self.top)
        self.unit = tk.StringVar(self.top)
        self.threshold = tk.StringVar(self.top, previous.get("threshold", ""))
        self.direction = tk.StringVar(self.top, previous.get("direction", "rising"))
        self.status = tk.StringVar(self.top)
        self.details = ""
        self.busy = False

        form = ScrollableForm(self.top, padding=(14, 10, 10, 6))
        form.pack(fill=tk.BOTH, expand=True)
        panel = form.body
        panel.columnconfigure(1, weight=1)
        lead = ttk.Label(panel, text="选择试验机持续写入的 CSV。开始本阶段后，新增数据越过设定值时自动保存事件前后的原图。",
                         width=1, justify=tk.LEFT)
        lead.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        wrap_label_on_resize(lead)
        self.choose_button = ttk.Button(panel, text="选择 CSV…", command=self.choose, style="Small.TButton")
        self.choose_button.grid(row=1, column=0, sticky="w", padx=(0, 8))
        file_name = ttk.Label(panel, textvariable=self.file_label, width=1)
        file_name.grid(row=1, column=1, sticky="ew")
        file_help = ("选择仪器每写完一条记录就追加一行的文件。首行填写列名。开始后已存在的数据会被跳过。"
                     "请用试采确认文件会在试验过程中更新。")
        ToolTip(self.choose_button, file_help, title="试验机 CSV")
        ToolTip(file_name, lambda _event: f"{self.path}\n{file_help}" if self.path else file_help, title="试验机 CSV")
        self.value_box = ttk.Combobox(panel, textvariable=self.value, state="disabled", width=23)
        self.value_box.bind("<<ComboboxSelected>>", self._suggest)
        self.time_box = ttk.Combobox(panel, textvariable=self.source_time, state="disabled", width=23)
        self.threshold_entry = ttk.Entry(panel, textvariable=self.threshold, width=12)
        direction = localized_combobox(panel, self.direction, CSV_DIRECTIONS, width=14)
        hints = {"上穿": "相邻两个有效值从低于阈值变为等于或高于阈值。",
                 "下穿": "相邻两个有效值从高于阈值变为等于或低于阈值。"}
        direction_tip = ChoiceToolTip(direction, "选择量值越过阈值的方向。", hints, title="触发方向")
        for row, (name, control, help_text) in enumerate((
            ("触发量列", self.value_box, "选择用于比较的数值列。优先提示表头中能识别的载荷或位移列。"),
            ("量名称", ttk.Entry(panel, textvariable=self.quantity, width=18), "填写量的名称，例如载荷或位移。会写入台账。"),
            ("原值单位", ttk.Entry(panel, textvariable=self.unit, width=12), "按 CSV 中的单位填写，例如 kN 或 mm。阈值也要使用这个单位。"),
            ("阈值", self.threshold_entry, "例如载荷列用 kN，填 5 表示 5 kN。可以用负数或小数。"),
            ("方向", direction, direction_tip.resolve_text),
            ("时间列（可选）", self.time_box, "可保留试验机时间的原始字段。回看同时显示主机发现越线的时刻。"),
        ), 2):
            label = ttk.Label(panel, text=name)
            label.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            control.grid(row=row, column=1, sticky="ew", pady=4)
            ToolTip(label, help_text, title=name)
            if control is not direction:
                ToolTip(control, help_text, title=name)
        message = ttk.Label(panel, width=1, justify=tk.LEFT, textvariable=self.status, style="CaptureMuted.TLabel")
        message.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        wrap_label_on_resize(message)
        ToolTip(message, lambda _event: self.details or self.status.get(), title="CSV 设置状态")
        form.bind_controls()

        actions = ttk.Frame(self.top, padding=(12, 7))
        actions.pack(fill=tk.X)
        self.apply = ttk.Button(actions, text="使用此阈值", command=self.save, style="Record.TButton", state="disabled")
        self.apply.pack(side=tk.RIGHT)
        ToolTip(self.apply, "保存此条件。回到阶段窗口点“应用阶段”，即可在计划检查中核对文件。")
        cancel = ttk.Button(actions, text="取消", command=self.close)
        cancel.pack(side=tk.RIGHT, padx=(0, 6))
        ToolTip(cancel, "关闭阈值设置，保留原先的阶段条件。")
        if rule:
            manual = ttk.Button(actions, text="移除 CSV 条件", command=self.remove)
            manual.pack(side=tk.LEFT)
            ToolTip(manual, "本阶段继续接受手动触发，或由已启用的外部命令触发。")
            self._load(rule["path"], previous)
        self.choose_button.focus_set()

    def choose(self):
        if self.busy:
            return
        file = filedialog.askopenfilename(parent=self.top, title="选择试验机 CSV",
                                         initialdir=str(self.path.parent) if self.path else None,
                                         filetypes=(("CSV 数据", "*.csv"), ("全部文件", "*.*")))
        if file:
            self._load(file, {})

    def _load(self, path, previous):
        self.busy = True
        self.path = Path(path).expanduser()
        self.header = None
        self.file_label.set(self.path.name)
        self.status.set("正在读取表头…")
        self.details = ""
        for widget in (self.choose_button, self.apply, self.value_box, self.time_box):
            widget.state(["disabled"])
        self.submit(lambda: inspect_csv_trigger(path), lambda header: self._loaded(header, previous), self._failed)

    def _loaded(self, header, previous):
        if not self.top.winfo_exists():
            return
        self.busy = False
        self.header = header
        self.choose_button.state(["!disabled"])
        self.apply.state(["!disabled"])
        self.value_box.configure(values=header.columns, state="readonly")
        self.time_box.configure(values=("", *header.columns), state="readonly")
        old_column = previous.get("value_column")
        old_time = previous.get("time_column")
        if old_column in header.columns:
            self.value.set(old_column)
            self.quantity.set(previous.get("quantity", ""))
            self.unit.set(previous.get("unit", ""))
        else:
            recognized = [column for column in header.columns if column_hint(column)[0] in ("force", "displacement")]
            self.value.set(recognized[0] if len(recognized) == 1 else "")
            self._suggest()
        times = [column for column in header.columns if column_hint(column)[0] in ("time", "absolute")]
        self.source_time.set(old_time if old_time in header.columns else times[0] if not previous and len(times) == 1 else "")
        changed = previous.get("header_sha256") not in (None, header.sha256)
        self.status.set("表头已变化，请确认触发量列、时间列和单位。" if changed else
                        "按原值单位填写阈值。缓存满后，新的越线才会触发。")
        self.details = "可将指针停在字段上，或按 F1 查看该项说明。"
        (self.threshold_entry if self.value.get() else self.value_box).focus_set()

    def _failed(self, exc):
        if not self.top.winfo_exists():
            return
        self.busy = False
        self.choose_button.state(["!disabled"])
        self.status.set(ui_error(exc, details=False))
        self.details = ui_error(exc)

    def _suggest(self, _event=None):
        column = self.value.get()
        kind, unit = column_hint(column)
        self.quantity.set({"force": "载荷", "displacement": "位移"}.get(kind, column))
        self.unit.set(unit or "")
        if self.source_time.get() == column:
            self.source_time.set("")

    def save(self):
        if not self.header or self.busy:
            return
        try:
            if not self.value.get():
                raise ValueError("请选择触发量列。")
            if not self.unit.get().strip():
                raise ValueError("请填写 CSV 数值列的单位。")
            if self.source_time.get() == self.value.get():
                raise ValueError("时间列需要与触发量列不同。")
            try:
                threshold = float(self.threshold.get())
            except ValueError:
                raise ValueError("阈值需要填写数字，例如 5 或 -0.2。") from None
            rule = CsvTrigger.from_dict({
                "path": str(self.path.absolute()), "value_column": self.value.get(), "quantity": self.quantity.get(),
                "unit": self.unit.get(), "threshold": threshold, "direction": self.direction.get(),
                "header_sha256": self.header.sha256,
                **({"time_column": self.source_time.get()} if self.source_time.get() else {}),
            })
            self.header.check(rule)
        except (CaptureError, ValueError) as exc:
            self.status.set(str(exc))
            self.details = str(exc)
            return
        self.accept(rule.to_dict())
        self.close()

    def remove(self):
        self.accept(None)
        self.close()

    def close(self):
        self.top.destroy()
        if self.parent.winfo_exists():
            self.parent.grab_set()
