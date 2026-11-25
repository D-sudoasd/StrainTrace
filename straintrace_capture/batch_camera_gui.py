"""Desktop form for checking shared camera settings."""

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import ScrollableForm, localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .batch_camera import BATCH_FIELDS, parse_batch_values, shared_features
from .dialogs import FeatureDialog
from .presentation import FEATURE_HELP, camera_setting_detail, ui_error
from .viewport import repaint_after_resize


_UNIT_LABELS = {"exposure_us": "微秒", "gain_db": "dB", "width": "像素", "height": "像素"}
_ENTRY_HELP = "只填写要统一的参数"
_RESULT_HELP = "选中相机，查看参数回读"
_ROW_STATES = {
    "verified": "通过 · 原值已恢复",
    "rejected": "未通过 · 原值已恢复",
    "restore_failed": "恢复原值未确认",
    "blocked": "检查未通过 · 未试设",
    "skipped": "未试设",
}


class BatchCameraDialog:
    def __init__(self, parent, config, readings, camera_labels, verify):
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("多机参数")
        width = min(640, self.top.winfo_screenwidth() - 48)
        height = min(545, self.top.winfo_screenheight() - 88)
        self.top.geometry(f"{width}x{height}")
        scaling = float(self.top.tk.call("tk", "scaling"))
        self.top.minsize(420, max(310, round(150 * scaling)))
        self.top.transient(parent)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.top.bind("<Escape>", self._escape)
        self.top.columnconfigure(0, weight=1)
        self.top.rowconfigure(1, weight=1)
        repaint_after_resize(self.top)
        self.working = False
        self.verify = verify
        self.camera_labels = camera_labels
        self.fields, self.widgets = {}, {}
        self._available = set()
        self._rows = {}
        header = ttk.Frame(self.top, padding=(14, 10))
        header.grid(row=0, column=0, sticky="ew")
        self.selection_title = f"已选 {len(config.cameras)} 台相机"
        self.heading = ttk.Label(header, text=self.selection_title, takefocus=True, style="CaptureSection.TLabel")
        self.heading.pack(anchor="w")
        device_details = []
        for item in config.cameras:
            evidence = readings.get(item.id)
            error = evidence.error if evidence else "未收到设备报告。"
            device_details.append(f"{camera_labels[item.id]} · {item.id}" + (f"\n读取失败 · {error}" if error else ""))
        ToolTip(self.heading, "\n".join(device_details), title="本次检查的设备")
        self.description = self._label(header, _ENTRY_HELP)

        self.scroll = ScrollableForm(self.top, padding=(14, 0, 14, 0))
        self.scroll.grid(row=1, column=0, sticky="nsew")
        form = self.scroll.body
        form.columnconfigure(1, weight=1)
        for row, (name, shared) in enumerate(shared_features(config, readings).items()):
            self._field(form, row, name, shared, config, readings)
        self.scroll.bind_controls()
        self._build_results(self.top)

        footer = ttk.Frame(self.top, padding=(14, 8))
        footer.grid(row=2, column=0, sticky="ew")
        self.status = tk.StringVar(self.top, value=(
            "悬停或按 F1 查看各台参数" if self._available else "悬停字段查看不可用原因"))
        self._label(footer, textvariable=self.status)
        self.actions = ttk.Frame(footer)
        self.actions.pack(fill=tk.X, pady=(6, 0))
        self.actions.columnconfigure(0, weight=1)
        self.apply_button = ttk.Button(self.actions, text="试设并写入计划", command=self.submit, style="Record.TButton")
        self.apply_button.grid(row=0, column=2)
        ToolTip(self.apply_button, "逐台写入并回读已填写的参数，每台试设后都恢复原值。"
                "全部通过后才更新计划。下次连接预览或开始采集时应用新计划。")
        self.back_button = ttk.Button(self.actions, text="返回修改", command=self.show_form, style="Record.TButton")
        self.back_button.grid(row=0, column=2)
        self.back_button.grid_remove()
        ToolTip(self.back_button, "返回参数表单。上次输入会保留，可修改后重新试设。")
        self.close_button = ttk.Button(self.actions, text="关闭", command=self.close)
        self.close_button.grid(row=0, column=1, sticky="e", padx=(0, 7))
        ToolTip(self.close_button, "查看完结果后关闭。试设期间须等相机回读和恢复完成。")
        if self._available:
            self.widgets[next(name for name in BATCH_FIELDS if name in self._available)].focus_set()
        else:
            self.apply_button.state(["disabled"])
            self.close_button.focus_set()
        self.top.grab_set()

    @staticmethod
    def _label(parent, text="", **options):
        label = ttk.Label(parent, text=text, style="CaptureMuted.TLabel",
                          width=1, wraplength=460, justify=tk.LEFT, **options)
        label.pack(fill=tk.X, anchor="w", pady=(4, 0))
        wrap_label_on_resize(label)
        return label

    def _field(self, parent, row, name, shared, config, readings):
        label, explanation = FEATURE_HELP[name]
        unit = _UNIT_LABELS.get(name, "")
        label_widget = ttk.Label(parent, text=label, takefocus=bool(shared.blockers))
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 12))
        var = self.fields[name] = tk.StringVar(self.top)
        if not shared.blockers:
            self._available.add(name)
        hint = [explanation]
        for item in config.cameras:
            title = self.camera_labels[item.id]
            evidence = readings.get(item.id)
            feature = (evidence.readback or {}).get(name) if evidence else None
            if reason := shared.blockers.get(item.id):
                hint.append(f"{title} · {reason}")
            elif feature and feature.readable:
                line = f"{title} · 当前 {feature.value} {unit}".rstrip()
                if name in item.settings:
                    line += f" · 计划 {item.settings[name]}"
                if bounds := FeatureDialog.bounds(feature):
                    line += f" · 范围 {bounds}"
                if feature.increment:
                    line += f" · 步长 {feature.increment:g}"
                hint.append(line)
        help_text = "\n".join(hint)
        if name == "pixel_format" and not shared.blockers:
            labels, tips = FeatureDialog.choice_labels(name, shared.choices)
            keep = "保留各台设置"
            widget = localized_combobox(parent, var, {"": keep, **labels}, width=18)
            ChoiceToolTip(widget, help_text, {keep: "继续使用各台目前的计划设置。", **tips}, title=label)
        else:
            widget = ttk.Entry(parent, textvariable=var, width=18)
            ToolTip(widget, help_text, title=label)
        widget.grid(row=row, column=1, sticky="ew", pady=3)
        self.widgets[name] = widget
        if shared.blockers:
            widget.state(["disabled"])
            caption = "查看不可用原因"
        elif name == "pixel_format":
            caption = f"{len(shared.choices)} 种共同格式"
        else:
            caption = " ".join(value for value in (FeatureDialog.bounds(shared), unit) if value)
        caption_widget = ttk.Label(parent, text=caption, style="CaptureMuted.TLabel")
        caption_widget.grid(row=row, column=2, sticky="w", padx=(10, 0))
        for target in (label_widget, caption_widget):
            ToolTip(target, help_text, title=label)

    def _build_results(self, parent):
        self.results = ttk.Frame(parent, padding=(14, 0, 14, 0))
        self.results.grid(row=1, column=0, sticky="nsew")
        self.results.columnconfigure(0, weight=1)
        self.results.rowconfigure(1, weight=1)
        self.results.grid_remove()
        listing = ttk.Frame(self.results)
        listing.grid(row=0, column=0, sticky="ew")
        self.table = ttk.Treeview(listing, columns=("state",), show="tree headings", selectmode="browse", height=2)
        self.table.heading("#0", text="相机")
        self.table.heading("state", text="核验状态")
        self.table.column("#0", width=170, minwidth=110)
        self.table.column("state", width=170, minwidth=135)
        self.table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar = ttk.Scrollbar(listing, orient="vertical", command=self.table.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.bind("<<TreeviewSelect>>", self._select_row)
        ToolTip(self.table, self._table_help, title="核验结果")
        self.detail_scroll = ScrollableForm(self.results, padding=(0, 4, 0, 0))
        self.detail_scroll.grid(row=1, column=0, sticky="nsew")
        self.detail = tk.StringVar(self.top)
        detail = self._label(self.detail_scroll.body, textvariable=self.detail, takefocus=True)
        ToolTip(detail, lambda _event: self.detail.get(), title="设备读数")
        self.detail_scroll.bind_controls()

    def _row_detail(self, row, *, include_heading=False):
        lines = [f"{self.camera_labels[row.identity]} · {_ROW_STATES[row.state]}"] if include_heading else []
        lines.extend(camera_setting_detail(issue) for issue in row.issues)
        if row.detail:
            lines.append(row.detail)
        for name, target in row.requested.items():
            if name in row.before:
                unit = f"（{_UNIT_LABELS[name]}）" if name in _UNIT_LABELS else ""
                baseline = row.before[name]
                old = f"{baseline:g}" if type(baseline) in (int, float) else str(baseline)
                goal = f"{target:g}" if type(target) in (int, float) else str(target)
                line = f"{FEATURE_HELP[name][0]}{unit} · 原值 {old} · 目标 {goal}"
                observed = row.observed.get(name)
                if row.state != "blocked":
                    reading = f"{observed:g}" if type(observed) in (int, float) else observed
                    line += f" · 回读 {reading if reading is not None else '未读到'}"
                lines.append(line)
        return "\n".join(lines) or "请先查看其他相机的检查结果。"

    def _table_help(self, event):
        identity = self.table.identify_row(event.y) if event and event.type in (
            tk.EventType.Enter, tk.EventType.Motion) else self.table.focus()
        row = self._rows.get(identity)
        return self._row_detail(row, include_heading=True) if row else _RESULT_HELP

    def _select_row(self, _event=None):
        selected = self.table.selection()
        self.detail.set(self._row_detail(self._rows[selected[0]]) if selected else "")
        self.detail_scroll.canvas.yview_moveto(0)

    def _set_working(self, working):
        self.working = working
        for name in self._available:
            self.widgets[name].state(["disabled" if working else "!disabled"])
        for button in (self.apply_button, self.close_button):
            button.state(["disabled" if working else "!disabled"])

    def submit(self):
        if self.working or not self.apply_button.winfo_manager() or self.apply_button.instate(["disabled"]):
            return
        try:
            changes = parse_batch_values({name: var.get() for name, var in self.fields.items()})
        except ValueError as exc:
            name = str(exc)
            kind = "整数" if name in ("width", "height") else "数值"
            self.status.set(f"请为“{FEATURE_HELP[name][0]}”输入有效{kind}。")
            self.widgets[name].focus_set()
            return
        if not changes:
            self.status.set("请至少填写一项要统一的参数。")
            return
        self._set_working(True)
        self.status.set("正在逐台核验、试设并恢复原值…")
        self.verify(self, changes)

    def show_form(self):
        self.results.grid_remove()
        self.back_button.grid_remove()
        self.heading.configure(text=self.selection_title)
        self.description.pack(fill=tk.X, anchor="w", pady=(4, 0))
        self.scroll.grid()
        self.apply_button.grid()
        issue = next((issue["feature"] for row in self._rows.values() for issue in row.issues
                      if issue["feature"] in self._available), None)
        target = issue or next(name for name in BATCH_FIELDS if name in self._available)
        self.widgets[target].focus_set()
        self.status.set("调整后可再次试设。")

    def finish(self, result, applied):
        self._set_working(False)
        self._rows = {row.identity: row for row in result.rows}
        self.table.delete(*self.table.get_children())
        for row in result.rows:
            self.table.insert("", "end", iid=row.identity, text=self.camera_labels[row.identity],
                              values=(_ROW_STATES[row.state],))
        first = next((row.identity for row in result.rows if row.state not in ("verified", "skipped")),
                     result.rows[0].identity)
        self.heading.configure(text="逐台回读")
        self.description.pack_forget()
        self.scroll.grid_remove()
        self.apply_button.grid_remove()
        self.results.grid()
        self.table.selection_set(first)
        self.table.focus(first)
        self.table.see(first)
        self._select_row()
        self.table.focus_set()
        if applied:
            self.status.set("全部通过，已写入计划")
        elif result.plan:
            self.status.set("计划已变化，请重新核验")
        elif any(row.state == "restore_failed" for row in result.rows):
            self.status.set("相机未确认恢复，请核对原值")
        else:
            self.status.set("查看原因后可返回修改")
            self.back_button.grid()

    def fail(self, error):
        self._set_working(False)
        self.status.set(ui_error(error))

    def close(self):
        if not self.working:
            self.top.destroy()

    def _escape(self, _event):
        if not ToolTip.dismiss_window(self.top):
            self.close()
        return "break"
