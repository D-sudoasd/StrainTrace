"""Filter the saved-frame ledger without decoding images."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .config import CaptureError
from .frame_search import FrameLocation, format_receipt_utc, parse_receipt_utc, parse_sequence_range
from .presentation import ROLE_LABELS, ui_error
from .storage import ExperimentReader
from .viewport import repaint_after_resize


_PAGE = 48
_TIME_HELP = "用主机收到图像的时间筛选。请输入带时区的日期和时间，例如 2026-11-25T09:30:00Z。起止时间均包含在内。"
_HEADINGS = {
    "#1": "同一台相机的图像序号。上面的序号框可查一张或一段已有记录。",
    "#2": "该图像所属的采集阶段，从 1 开始。",
    "#3": "采集计划登记的图像用途，可用于定位参考图或标定图。",
    "#4": "主机收到图像的 UTC 时间。悬停在行上可查看到纳秒。需要曝光时间时，请用时序核对或时间导出。",
    "#5": "主图像像素格式。同一帧有多路图像时，可打开后切换部件。",
    "#6": "显示是否有关联的触发记录，或是否处于事件前后窗口。",
}
_EVENT_POSITIONS = {"pre": "前图", "boundary": "边界图", "post": "后图"}


class FrameSearchWindow:
    def __init__(self, parent, experiment, experiment_id, camera_names, active_camera, phase_names, submit,
                 show_frame, show_trigger, *, scope=None, on_opened=None, use_event=None):
        self.experiment, self.experiment_id = experiment, experiment_id
        self.camera_names = camera_names
        self.submit, self.show_frame, self.show_trigger = submit, show_frame, show_trigger
        self.on_opened = on_opened
        self.use_event = use_event
        self._event_buttons = []
        self._event_pending = False
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("查找已保存图像")
        self.top.geometry("850x590")
        self.top.minsize(710, 470)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.camera = tk.StringVar(self.top, active_camera if active_camera in camera_names else next(iter(camera_names)))
        self.phase, self.role, self.sequence, self.time_from, self.time_to = (tk.StringVar(self.top) for _ in range(5))
        self.use_time = tk.BooleanVar(self.top)
        self.event_context = scope.get("event_context_id") if scope else None
        self.event_number = scope.get("event_number") if scope else None
        self.use_event_context = tk.BooleanVar(self.top, bool(self.event_context))
        if scope:
            self.camera.set(scope["camera"])
            phase = str(scope["phase"])
            self.phase.set(phase if phase in phase_names else "")
            self.sequence.set(str(scope.get("sequence", "")))
            if scope.get("start") is not None:
                self.time_from.set(format_receipt_utc(scope["start"]).replace(" ", "T") + "Z")
                self.time_to.set(format_receipt_utc(scope["end"]).replace(" ", "T") + "Z")
                self.use_time.set(True)
        self.summary = tk.StringVar(self.top)
        self._last_criteria = None
        self._filter = {}
        self._page = None
        self._rows = {}
        self._busy = False
        self._detail = ""
        self._build(phase_names)
        for variable in (self.camera, self.phase, self.role, self.sequence, self.time_from, self.time_to,
                         self.use_time, self.use_event_context):
            variable.trace_add("write", self._criteria_changed)
        self.top.protocol("WM_DELETE_WINDOW", self.top.destroy)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self._query()

    def _build(self, phase_names):
        area = ttk.Frame(self.top, padding=14)
        area.pack(fill=tk.BOTH, expand=True)
        area.columnconfigure(0, weight=1)
        area.rowconfigure(2, weight=1)
        filters = ttk.Frame(area)
        filters.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        names = {key: f"{name} · {key}" for key, name in self.camera_names.items()}
        for index, (label, variable, options, help_text) in enumerate((
            ("相机", self.camera, names, "序号按相机分别登记。跨相机关联请用触发组。"),
            ("阶段", self.phase, {"": "全部", **phase_names}, "按采集阶段筛选。"),
            ("用途", self.role, {"": "全部", **ROLE_LABELS}, "按图像用途筛选，例如只找参考帧。"),
        )):
            col = index * 2
            heading = ttk.Label(filters, text=label)
            heading.grid(row=0, column=col, padx=(0, 6), sticky="w")
            box = localized_combobox(filters, variable, options, width=(14, 9, 7)[index])
            box.grid(row=0, column=col + 1, padx=(0, 12 if index < 2 else 0), sticky="ew")
            filters.columnconfigure(col + 1, weight=2 if index == 0 else 1)
            tip = ChoiceToolTip(box, help_text, {name: help_text for name in options.values()}, title=label)
            ToolTip(heading, tip.resolve_text)
        sequence = ttk.Entry(filters, textvariable=self.sequence, width=12)
        sequence_label = ttk.Label(filters, text="序号或范围")
        sequence_label.grid(row=1, column=0, padx=(0, 7), pady=(8, 0), sticky="w")
        sequence.grid(row=1, column=1, columnspan=2, padx=(0, 12), pady=(8, 0), sticky="ew")
        sequence_help = "输入 42 可查这一张，输入 40-80 可查一段。留空查询所有序号。"
        for control in (sequence, sequence_label):
            ToolTip(control, sequence_help, title="图像序号")
        sequence.bind("<Return>", lambda _event: self._query())
        time_option = ttk.Checkbutton(filters, text="按收图时间", variable=self.use_time, command=self._toggle_time)
        time_option.grid(row=1, column=3, columnspan=2, pady=(8, 0), sticky="w")
        ToolTip(time_option, _TIME_HELP, title="主机收图时间")
        self.query_button = ttk.Button(filters, text="查询", command=self._query, style="Small.TButton")
        self.query_button.grid(row=1, column=5, pady=(8, 0), sticky="ew")
        ToolTip(self.query_button, lambda _event: "正在查询，请等待本次读取完成。" if self._busy else
                "按当前条件重新查询并纳入最新已登记的图像。")
        self.time_form = ttk.Frame(filters)
        self.time_form.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        self.time_form.columnconfigure(1, weight=1)
        for index, (label, text) in enumerate((("起始时间", self.time_from), ("截止时间", self.time_to))):
            heading = ttk.Label(self.time_form, text=label)
            heading.grid(row=index, column=0, padx=(0, 8), pady=(0, 4), sticky="w")
            edit = ttk.Entry(self.time_form, textvariable=text, width=34)
            edit.grid(row=index, column=1, sticky="ew", pady=(0, 4))
            edit.bind("<Return>", lambda _event: self._query())
            for control in (heading, edit):
                ToolTip(control, _TIME_HELP + " 单边留空时只按另一个时间边界筛选。", title=label)
        self._toggle_time()
        if self.event_context:
            exact = ttk.Checkbutton(filters, text="仅看本次事件", variable=self.use_event_context)
            exact.grid(row=3, column=0, columnspan=6, sticky="w", pady=(6, 0))
            ToolTip(exact, "只显示所选事件的前图、边界图和后图。换相机可查同次事件的另一台相机。"
                    "窗口重叠时，同一张原图可出现在多次事件中。取消勾选可查询其他图像。")

        status = ttk.Label(area, textvariable=self.summary, width=1, justify=tk.LEFT, style="CaptureMuted.TLabel")
        status.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._detail or self.summary.get(), title="查询状态")
        table = ttk.Frame(area)
        table.grid(row=2, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        columns = ("sequence", "phase", "role", "time", "format", "trigger")
        self.table = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse", height=7)
        self.table.grid(row=0, column=0, sticky="nsew")
        for key, title, width in zip(columns, ("序号", "阶段", "用途", "收图时间（UTC）", "像素格式", "触发"),
                                     (78, 62, 85, 291, 100, 85)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=width if key in ("sequence", "phase") else 78,
                              stretch=key in ("time", "format"))
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        ToolTip(scroll, "浏览本页。用下方按钮切换到更早或更新的登记记录。")
        ToolTip(self.table, self._table_help, title="已保存图像")
        self.table.bind("<<TreeviewSelect>>", self._controls)
        self.table.bind("<Double-1>", self._double_click)
        self.table.bind("<Return>", lambda _event: self._open_frame())
        self.table.bind("<Control-c>", self._copy_time)
        sequence.focus_set()

        footer = ttk.Frame(area)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.older_button = ttk.Button(footer, text="较早一页", style="Small.TButton", command=lambda: self._paginate("older_before"))
        self.older_button.pack(side=tk.LEFT, padx=(0, 6))
        ToolTip(self.older_button, lambda _event: self._page_help("older_before", "较早"))
        self.newer_button = ttk.Button(footer, text="较新一页", style="Small.TButton", command=lambda: self._paginate("newer_after"))
        self.newer_button.pack(side=tk.LEFT)
        ToolTip(self.newer_button, lambda _event: self._page_help("newer_after", "较新"))
        self.open_button = ttk.Button(footer, text="打开原图", style="Record.TButton", command=self._open_frame)
        self.open_button.pack(side=tk.RIGHT)
        ToolTip(self.open_button, lambda _event: self._blocked() or "先按台账核对所选原图，再在采集工作台显示。")
        self.trigger_button = ttk.Button(footer, text="查看触发组…", style="Small.TButton", command=self._open_trigger)
        self.trigger_button.pack(side=tk.RIGHT, padx=(0, 8))
        ToolTip(self.trigger_button, lambda _event: self._blocked() or (
            "按相同触发标识并排查看各台相机的原图。" if self._selected().get("trigger_event_id") else
            "所选图像没有关联触发记录。"))
        if self.event_context and self.use_event:
            actions = ttk.Frame(area)
            actions.grid(row=4, column=0, sticky="ew", pady=(6, 0))
            for action, name, hint in (
                ("analysis", "分析本次事件…", "先选择参考图，再打开这台相机在本次事件中的图像进行离线分析。"),
                ("export", "导出本次事件…", "先选择参考图，再按事件导出 TIFF 原图和帧索引。"),
            ):
                button = ttk.Button(actions, text=name, command=lambda mode=action: self._use_event(mode), style="Small.TButton")
                button.pack(side=tk.RIGHT, padx=(6, 0))
                self._event_buttons.append(button)
                ToolTip(button, lambda _event, text=hint: self._event_block() or text)

    def _toggle_time(self):
        if self.use_time.get():
            self.time_form.grid()
        else:
            self.time_form.grid_remove()

    def _criteria(self):
        return (self.camera.get(), self.phase.get(), self.role.get(), self.sequence.get().strip(),
                self.time_from.get().strip() if self.use_time.get() else "", self.time_to.get().strip() if self.use_time.get() else "",
                self.event_context if self.use_event_context.get() else None)

    def _selected(self):
        selection = self.table.selection()
        return self._rows.get(selection[0], {}) if selection else {}

    def _blocked(self):
        if self._busy:
            return "正在读取数据，请稍候。"
        if self._criteria() != self._last_criteria:
            return "条件已修改。请先点“查询”更新结果。"
        return "请先在结果列表中选中一张图像。" if not self._selected() else ""

    def _controls(self, *_args):
        changed = self._criteria() != self._last_criteria
        ready = not self._busy and not changed and self._page is not None
        self.query_button.state(["disabled" if self._busy else "!disabled"])
        self.older_button.state(["!disabled" if ready and self._page["older_before"] is not None else "disabled"])
        self.newer_button.state(["!disabled" if ready and self._page["newer_after"] is not None else "disabled"])
        self.open_button.state(["disabled" if self._blocked() else "!disabled"])
        self.trigger_button.state(["!disabled" if ready and self._selected().get("trigger_event_id") else "disabled"])
        for button in self._event_buttons:
            button.state(["disabled" if self._event_block() else "!disabled"])

    def _event_block(self):
        if self._event_pending:
            return "正在读取本次事件及参考图。"
        if not self.use_event_context.get():
            return "勾选“仅看本次事件”，查询后可使用这次事件的图像。"
        if self._busy or self._criteria() != self._last_criteria:
            return "请先等这台相机的查询完成。"
        return "" if self._rows else "这个范围内没有已保存的原图。"

    def _use_event(self, action):
        if self._event_block():
            return
        self._event_pending = True
        self._controls()
        self.summary.set("正在读取本次事件及参考图…")

        def finished():
            if self.top.winfo_exists():
                self._event_pending = False
                self._criteria_changed()

        def failed(exc):
            finished()
            self._set_error(exc)

        try:
            self.use_event(self.top, self.experiment, self.experiment_id, self._page["camera_id"],
                           self.event_context, action, finished, failed)
        except CaptureError as exc:
            failed(exc)

    def _page_help(self, cursor, label):
        if self._busy:
            return "正在读取台账，请稍候。"
        if self._page is None or self._criteria() != self._last_criteria:
            return "请先按当前条件查询。"
        if self._page[cursor] is None:
            return f"本次查询范围内没有{label}的记录。重新查询可纳入新登记的图像。"
        return f"保留当前筛选条件，查看{label}一页的记录。"

    def _criteria_changed(self, *_args):
        if not self._busy and self._criteria() != self._last_criteria:
            self.summary.set("条件已修改。点“查询”查看结果。")
            self._detail = ""
        elif not self._busy and self._page is not None:
            self._show_summary(self._page)
        self._controls()

    def _query(self):
        if self._busy:
            return
        criteria = self._criteria()
        camera, phase, role, span, start, end, context = criteria
        try:
            first, last = parse_sequence_range(span)
            first_time = parse_receipt_utc(start) if start else None
            last_time = parse_receipt_utc(end) if end else None
        except CaptureError as exc:
            self._set_error(exc)
            self.summary.set(str(exc))
            return
        self._filter = {"camera_id": camera, "phase": int(phase) if phase else None, "role": role or None,
                        "first_sequence": first, "last_sequence": last, "received_from_ns": first_time, "received_to_ns": last_time,
                        "event_context_id": context}
        self._last_criteria = criteria
        self._page = None
        self._rows.clear()
        self.table.delete(*self.table.get_children())
        self._load()

    def _paginate(self, cursor):
        if self._busy or self._page is None or self._criteria() != self._last_criteria or self._page[cursor] is None:
            return
        direction = "before" if cursor == "older_before" else "after"
        self._load(snapshot=self._page["snapshot_ordinal"], **{direction: self._page[cursor]})

    def _load(self, **cursor):
        self._busy = True
        self._detail = ""
        self.summary.set("正在查询已登记的图像…")
        self._controls()
        filters = self._filter.copy()

        def read():
            with ExperimentReader(self.experiment) as reader:
                return reader.frame_page(**filters, **cursor, expected_experiment_id=self.experiment_id, limit=_PAGE)

        self.submit(read, self._loaded, self._set_error)

    def _show_summary(self, page):
        self._detail = "列表按台账登记顺序排列。新写入的图像可通过“查询”纳入。"
        if page["frames"]:
            scope = ""
            if self._filter.get("event_context_id"):
                scope = f"事件 {self.event_number} · " if self.event_number is not None else "本次事件 · "
            self.summary.set(f"{scope}本页 {len(page['frames'])} 帧 · 按登记顺序从新到旧")
        else:
            self.summary.set("这个范围内没有已保存的图像。可放宽条件后重新查询。")

    def _loaded(self, page):
        if not self.top.winfo_exists():
            return
        self._busy = False
        self._page = page
        self._rows = {str(row["ordinal"]): row for row in page["frames"]}
        self.table.delete(*self.table.get_children())
        exact_event = bool(self._filter.get("event_context_id"))
        self.table.heading("trigger", text="位置" if exact_event else "触发")
        for key, row in self._rows.items():
            position = _EVENT_POSITIONS.get(row["event_position"])
            trigger = "已关联" if row["trigger_event_id"] else "—"
            if position:
                trigger = position if exact_event else f"事件{position}"
            self.table.insert("", "end", iid=key, values=(row["sequence"], row["phase"], ROLE_LABELS[row["role"]],
                              format_receipt_utc(row["received_utc_ns"], precision=3), row["pixel_format"], trigger))
        self.table.yview_moveto(0)
        if self._rows:
            self.table.selection_set(next(iter(self._rows)))
            self.table.focus(next(iter(self._rows)))
        self._criteria_changed()

    def _set_error(self, exc):
        if not self.top.winfo_exists():
            return
        self._busy = False
        self._detail = ui_error(exc)
        self.summary.set(ui_error(exc, details=False))
        self._controls()

    def _table_help(self, event):
        row = self._selected()
        if event is not None and getattr(event, "keysym", "??") in ("??", ""):
            if self.table.identify_region(event.x, event.y) == "heading":
                column = self.table.identify_column(event.x)
                if column == "#6" and self._filter.get("event_context_id"):
                    return "显示前图、边界图或后图。位置按本次事件的触发时间判断。"
                return _HEADINGS.get(column, "")
            row = self._rows.get(self.table.identify_row(event.y), {})
        if not row:
            return "选中一行后可打开原图，或按 Ctrl+C 复制它的精确收图时间。"
        stamp = format_receipt_utc(row["received_utc_ns"])
        position = _EVENT_POSITIONS.get(row["event_position"])
        if position:
            position = f"在本次事件中用作{position}" if self._filter.get("event_context_id") else f"事件{position}"
        return (f"{self._page['camera_id']} · 序号 {row['sequence']} · 阶段 {row['phase']}\n"
                f"主机收图（UTC）{stamp}\n{row['images']} 张原图同属这一帧"
                + (f" · {position}" if position else "")
                + (f"\n拍摄请求 {row['trigger_event_id']}" if row["trigger_event_id"] else "")
                + "\n双击打开原图。Ctrl+C 复制收图时间。")

    def _copy_time(self, _event):
        value = self._selected().get("received_utc_ns")
        if value is not None and not self._blocked():
            self.top.clipboard_clear()
            self.top.clipboard_append(format_receipt_utc(value).replace(" ", "T") + "Z")
        return "break"

    def _double_click(self, event):
        if key := self.table.identify_row(event.y):
            self.table.selection_set(key)
            self._open_frame()

    def _open_frame(self):
        if self._blocked():
            return
        row = self._selected()
        location = FrameLocation(self.experiment_id, self._page["camera_id"], row["sequence"], row["ordinal"],
                                 row["frame_id"], row["sha256"])
        self.summary.set("正在核验并读取原图…")
        self._busy = True
        self._controls()

        def opened():
            if self.top.winfo_exists():
                self._busy = False
                self._criteria_changed()
                self.top.withdraw()
                if self.on_opened:
                    self.on_opened()

        self.show_frame(self.experiment, location, opened, self._set_error)

    def _open_trigger(self):
        if not self._blocked() and (event_id := self._selected()["trigger_event_id"]):
            try:
                self.show_trigger(self.experiment, self.experiment_id, event_id)
            except CaptureError as exc:
                self._set_error(exc)
            else:
                self.top.withdraw()

    def reopen(self):
        self.top.deiconify()
        self.top.lift()
