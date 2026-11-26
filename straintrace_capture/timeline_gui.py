"""Review recorded phase boundaries, saved images, and acquisition incidents."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import localized_combobox, wrap_label_on_resize
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .config import CaptureError
from .csv_trigger import CSV_TRIGGER_EVENT
from .frame_search import format_receipt_utc
from .image_quality_watch import NOTICE_EVENT
from .markers import MARKER_EVENT
from .presentation import (CAPTURE_EVENT_LABELS, CSV_STATES, csv_crossing_lines, csv_trigger_help,
                           image_quality_alert_text, image_quality_sample_help, ui_error)
from .storage import ExperimentReader
from .viewport import repaint_after_resize


_BASES = {"scheduled": "计划时刻", "host_receipt": "主机收图", "host": "主机操作",
          "ledger": "台账登记", "trigger_request": "触发请求", "host_csv": "CSV 检测", "host_signal": "主机收信"}
_PLOT_HELP = ("蓝柱为各时段已保存的图像帧数。橙点为待核对事件，紫色菱形为实验标记。"
              "绿色条为阶段，灰色条为人工确认等待，黄线为台账记录的暂停。"
              "横轴按主机 UTC 计时。点击蓝柱可查原图，或用左右方向键选择时段后按 Enter。")
_UTC_LIMIT = (1 << 63) - 1
_WINDOW_RESULTS = {"complete": "已收齐", "short_pre": "前图不足", "short_post": "后图不足"}
_EVENT_FILTER_LABELS = {"": "全部事件", "incidents_only": "仅看异常",
                        "event_windows_only": "仅看事件窗口", "markers_only": "仅看实验标记"}
_EVENT_FILTER_HELP = {
    "": "查看阶段和采集事件。上方时间图始终按相机和阶段显示已保存帧。",
    "incidents_only": "显示漏拍、计数变化、触发缺帧、CSV 故障和采集错误。选中一行可查看详情和对应操作。",
    "event_windows_only": "按触发请求查找已保存的事件前后原图。选中一行后可按相机查看。",
    "markers_only": "查看采集时提交的实验标记。选中一行可查各相机在标记前后的原图。",
}
_EVENT_IMAGE_KINDS = ("event_window", CSV_TRIGGER_EVENT, "pre_window_short")
_COMPACT_HEIGHT = 555


def _window_signal(row):
    if row.get("kind") in _EVENT_IMAGE_KINDS:
        signal = row["details"].get("signal_id")
        return signal if isinstance(signal, str) and signal.strip() else None


class TimelineWindow:
    def __init__(self, parent, experiment, identity, cameras, active_camera, phases, submit, open_frames, open_trigger,
                 *, phase=None, all_cameras=False, open_marker=None, event_filter="", summary_notes=()):
        self.experiment, self.experiment_id = experiment, identity
        self.cameras = cameras
        self.active_camera = active_camera if active_camera in cameras else next(iter(cameras), None)
        self.submit, self.open_frames, self.open_trigger = submit, open_frames, open_trigger
        self.open_marker = open_marker
        self.summary_notes = summary_notes
        style = ttk.Style(parent)
        self.top = tk.Toplevel(parent, bg=style.lookup("TFrame", "background"))
        self.top.title("采集时间线")
        self.top.geometry("920x630")
        self.top.minsize(730, 505)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.camera = tk.StringVar(self.top, "" if phase is not None or all_cameras else self.active_camera or "")
        self.phase = tk.StringVar(self.top, "" if phase is None else str(phase))
        self.event_filter = tk.StringVar(self.top, event_filter)
        self.summary = tk.StringVar(self.top)
        self.selection_note = tk.StringVar(self.top)
        self._busy = False
        self._overview = self._page = None
        self._rows = {}
        self._plot_bounds = None
        self._keyboard_bucket = None
        self._error_detail = ""
        self._filters = []
        self._build(phases, style)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self._query()

    def _build(self, phases, style):
        body = ttk.Frame(self.top, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)
        filters = ttk.Frame(body)
        filters.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for index, (title, variable, values, hint) in enumerate((
            ("相机", self.camera, {"": "全部相机", **self.cameras},
             "选择相机可单独查看其收图情况。阶段和整组触发事件会保留。查看全部相机时，点蓝柱将优先打开当前复查相机。"),
            ("阶段", self.phase, {"": "全部阶段", **phases}, "选择要复查的阶段，可缩小时间范围。"),
        )):
            column = index * 2
            label = ttk.Label(filters, text=title)
            label.grid(row=0, column=column, padx=(0, 6))
            combo = localized_combobox(filters, variable, values, width=(14, 11)[index])
            combo.grid(row=0, column=column + 1, sticky="ew", padx=(0, 11))
            combo.bind("<<ComboboxSelected>>", self._query, add="+")
            self._filters.append(combo)
            filters.columnconfigure(column + 1, weight=1)
            ToolTip(label, hint)
            ChoiceToolTip(combo, hint, {value: hint for value in values.values()}, title=title)
        self.refresh = ttk.Button(filters, text="刷新", command=self._query, style="Small.TButton")
        self.refresh.grid(row=0, column=6)
        ToolTip(self.refresh, "重新读取台账，纳入已登记的新图像和事件。")
        label = ttk.Label(filters, text="事件类型")
        label.grid(row=0, column=4, padx=(0, 6))
        self.event_combo = localized_combobox(filters, self.event_filter, _EVENT_FILTER_LABELS, width=11)
        self.event_combo.grid(row=0, column=5, sticky="ew", padx=(0, 10))
        filters.columnconfigure(5, weight=1)
        self.event_combo.bind("<<ComboboxSelected>>", self._query, add="+")
        self._filters.append(self.event_combo)
        filter_help = "选择列表要显示的事件类型。"
        ToolTip(label, filter_help)
        ChoiceToolTip(self.event_combo, filter_help, {
            caption: _EVENT_FILTER_HELP[value] for value, caption in _EVENT_FILTER_LABELS.items()
        }, title="事件类型")

        status = ttk.Label(body, textvariable=self.summary, style="CaptureMuted.TLabel", width=1, justify=tk.LEFT)
        status.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        wrap_label_on_resize(status)
        ToolTip(status, self._summary_help)
        self._plot_text = style.lookup("TLabel", "foreground") or "#25344a"
        plot_bg = style.lookup("Treeview", "background") or "#ffffff"
        self.plot = tk.Canvas(body, height=142, bg=plot_bg, highlightthickness=1, highlightbackground="#8293a7",
                              takefocus=True)
        self.plot.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.top.bind("<Configure>", self._resize_plot, add="+")
        self.plot.bind("<Configure>", self._draw)
        self.plot.bind("<Button-1>", self._click_plot)
        self.plot.bind("<Left>", lambda _event: self._step_bucket(-1))
        self.plot.bind("<Right>", lambda _event: self._step_bucket(1))
        self.plot.bind("<Return>", lambda _event: self._open_bucket(self._keyboard_bucket))
        ToolTip(self.plot, self._plot_tip, title="采集时序")
        label = ttk.Label(body, text="采集事件", style="CaptureSection.TLabel")
        label.grid(row=3, column=0, sticky="w", pady=(0, 4))
        ToolTip(label, "事件按台账登记顺序从新到旧。悬停可查看到纳秒及其时间来源。")
        table_area = ttk.Frame(body)
        table_area.grid(row=4, column=0, sticky="nsew")
        table_area.columnconfigure(0, weight=1)
        table_area.rowconfigure(0, weight=1)
        columns = ("time", "basis", "phase", "camera", "event")
        self.table = ttk.Treeview(table_area, columns=columns, show="headings", selectmode="browse", height=6)
        self.table.grid(row=0, column=0, sticky="nsew")
        for key, title, size, minimum in zip(columns, ("时间（UTC）", "时间来源", "阶段", "相机", "事件"),
                                              (226, 94, 58, 122, 155), (205, 82, 48, 109, 134)):
            self.table.heading(key, text=title)
            self.table.column(key, width=size, minwidth=minimum, stretch=key in ("time", "event"))
        bar = ttk.Scrollbar(table_area, orient="vertical", command=self.table.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=bar.set)
        self.table.bind("<<TreeviewSelect>>", self._selected_changed)
        self.table.bind("<Return>", lambda _event: self._open_selected())
        self.table.bind("<Double-1>", self._double_click)
        self.table.bind("<Control-c>", self._copy_time)
        ToolTip(self.table, self._table_tip, title="采集事件")

        self.selection_label = ttk.Label(body, textvariable=self.selection_note, style="CaptureMuted.TLabel",
                                         width=1, justify=tk.LEFT)
        self.selection_label.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        self.selection_label.grid_remove()
        wrap_label_on_resize(self.selection_label)

        footer = ttk.Frame(body)
        footer.grid(row=6, column=0, sticky="ew", pady=(9, 0))
        self.older = ttk.Button(footer, text="较早一页", style="Small.TButton", command=lambda: self._paginate("before"))
        self.newer = ttk.Button(footer, text="较新一页", style="Small.TButton", command=lambda: self._paginate("after"))
        self.older.pack(side=tk.LEFT, padx=(0, 6))
        self.newer.pack(side=tk.LEFT)
        for button, direction in ((self.older, "before"), (self.newer, "after")):
            ToolTip(button, lambda _event, key=direction: "沿当前台账快照继续翻页。" if self._cursor(key) else
                    "当前范围内没有更多登记记录。")
        self.frames_button = ttk.Button(footer, text="查附近原图…", command=self._open_selected_frames, style="Record.TButton")
        self.frames_button.pack(side=tk.RIGHT)
        ToolTip(self.frames_button, lambda _event: self._frames_help())
        self.trigger_button = ttk.Button(footer, text="查看触发组…", command=self._open_selected_trigger, style="Small.TButton")
        self.trigger_button.pack(side=tk.RIGHT, padx=(0, 8))
        ToolTip(self.trigger_button, lambda _event: "按台账中的触发标识并排查看本组原图。缺帧的相机会单独标出。" if
                self._selected().get("trigger_event_id") else "选中含触发组标识的事件后，可直接查看这一组。")

    def _criteria(self):
        return (self.camera.get() or None, int(self.phase.get()) if self.phase.get() else None,
                self.event_filter.get())

    def _query(self, _event=None):
        if self._busy:
            return
        self._page = self._overview = None
        self._keyboard_bucket = None
        self._rows.clear()
        self.table.delete(*self.table.get_children())
        self._draw()
        self._load()

    def _cursor(self, direction):
        if not self._page:
            return None
        page_filter = next((key for key in _EVENT_FILTER_LABELS if self._page.get(key)), "")
        if self._criteria() == (self._page["camera_id"], self._page["phase"], page_filter):
            return self._page["older_before" if direction == "before" else "newer_after"]
        return None

    def _paginate(self, direction):
        if not self._busy and (cursor := self._cursor(direction)):
            self._load(snapshot=self._page["snapshot_ordinal"], **{direction: cursor})

    def _load(self, **cursor):
        camera, phase, event_filter = self._criteria()
        filters = {event_filter: True} if event_filter else {}
        self._busy = True
        self._error_detail = ""
        self.summary.set("正在读取已登记的图像和事件…")
        self._update_controls()

        def read():
            with ExperimentReader(self.experiment) as reader:
                return reader.timeline(camera_id=camera, phase=phase, expected_experiment_id=self.experiment_id,
                                       **filters, **cursor)

        def finished(result, failed=False):
            if not self.top.winfo_exists():
                return
            self._busy = False
            if failed:
                self._error_detail = ui_error(result)
                self.summary.set(ui_error(result, details=False))
            else:
                self._show_page(result)
            self._update_controls()

        self.submit(read, finished, lambda error: finished(error, True))

    def _show_page(self, page):
        self._page = page
        if "overview" in page:
            self._overview = page["overview"]
        self._rows = {str(row["ordinal"]): row for row in page["events"]}
        self.table.delete(*self.table.get_children())
        for key, row in self._rows.items():
            camera_id = row["camera_id"]
            camera_name = self.cameras.get(camera_id or "", camera_id or "—")
            event = CAPTURE_EVENT_LABELS[row["kind"]]
            if row["kind"] == MARKER_EVENT and (label := row["details"].get("label")):
                event = f"{event} · {label}"
            elif row["kind"] == NOTICE_EVENT:
                detail = row["details"]
                if codes := detail.get("started") or detail.get("cleared"):
                    label = image_quality_alert_text(codes)
                    event = f"{'画质提醒' if detail.get('started') else '提醒解除'} · {label}"
                elif scope := detail.get("region"):
                    area = "全图" if scope.get("fallback") else "检查框"
                    event = f"画质抽查范围 · {area}"
            elif row["kind"] == CSV_TRIGGER_EVENT:
                details = row["details"]
                state = details.get("state")
                if state in CSV_STATES:
                    if details.get("signal_id"):
                        event = "CSV 阈值触发"
                    elif state == "finished" and details.get("events_accepted"):
                        event = f"CSV · 触发 {details['events_accepted']} 次"
                    else:
                        event = f"CSV · {CSV_STATES[state]}"
            elif row["kind"] == "event_window":
                detail = row["details"]
                event = f"事件 {detail['event_number']} · {_WINDOW_RESULTS[detail['result']]}"
            elif row["kind"] == "error" and self._failed_frame_time(row):
                event = f"写入错误 · 帧 {row['details'].get('sequence', '—')}"
            elif row["kind"] == "trigger_incomplete" and (failed := row["details"].get("failed_cameras")):
                camera_name = "、".join(self.cameras.get(camera, camera) for camera in failed)
            elif row["kind"] == "event_phase_summary":
                event = "触发请求未接收"
            self.table.insert("", "end", iid=key, values=(
                format_receipt_utc(row["utc_ns"], precision=3), _BASES.get(row["time_basis"], "—"),
                row["phase"] or "—", camera_name, event,
            ))
        if self._rows:
            first = next(iter(self._rows))
            self.table.selection_set(first)
            self.table.focus(first)
            self.table.yview_moveto(0)
        self._draw()
        chart = self._overview
        missing = chart["frame_count"] - chart["timed_frame_count"]
        text = f"已保存 {chart['frame_count']:,} 帧  ·  采集事件 {chart['event_count']:,} 条  ·  本页 {len(self._rows)} 条"
        if page.get("markers_only"):
            text = f"实验标记 · 本页 {len(self._rows)} 条" if self._rows else "当前范围内没有实验标记。"
        if page.get("event_windows_only"):
            text = f"事件窗口 · 本页 {len(self._rows)} 条" if self._rows else "当前范围内没有收到事件触发。"
        if page.get("incidents_only"):
            text = (f"异常记录 {chart['incident_count']:,} 条 · 本页 {len(self._rows)} 条 · "
                    f"已保存 {chart['frame_count']:,} 帧")
            if not self._rows and self.summary_notes:
                text += " · 悬停查看全实验计数"
        if missing:
            text += f"  ·  {missing:,} 帧缺少收图时间"
        self.summary.set(text)

    def _summary_help(self, _event):
        if self._error_detail:
            return self._error_detail
        if self.event_filter.get() == "incidents_only" and self.summary_notes:
            return "\n".join((self.summary.get(), "", "全实验计数汇总", *self.summary_notes))
        return self.summary.get()

    def _selected(self):
        selection = self.table.selection()
        return self._rows.get(selection[0], {}) if selection else {}

    def _selected_changed(self, _event):
        self._update_controls()
        self._draw()

    def _update_controls(self, *_args):
        row = self._selected()
        has_trigger = bool(self.cameras and row.get("trigger_event_id"))
        caption = ("查看事件前后原图…" if row.get("kind") in _EVENT_IMAGE_KINDS else
                   "查看前后原图…" if row.get("marker_id") and self.open_marker else
                   "定位抽查帧…" if row.get("kind") == NOTICE_EVENT else "查附近原图…")
        self.frames_button.configure(text=caption, style="Small.TButton" if has_trigger else "Record.TButton")
        self.trigger_button.configure(style="Record.TButton" if has_trigger else "Small.TButton")
        show_note = bool(row and not self._busy and self.event_filter.get() == "incidents_only")
        if show_note:
            self.selection_note.set(self._incident_action(row))
            self.selection_label.grid()
        else:
            self.selection_label.grid_remove()
        for combo in self._filters:
            combo.state(["disabled" if self._busy else "!disabled"])
        for button, enabled in ((self.refresh, True), (self.older, self._cursor("before")),
                                (self.newer, self._cursor("after")),
                                (self.frames_button, self._has_frame_action(row)), (self.trigger_button, has_trigger)):
            button.state(["!disabled" if enabled and not self._busy else "disabled"])

    def _resize_plot(self, event):
        if event.widget is self.top:
            requested = 110 if event.height <= _COMPACT_HEIGHT else 142
            if int(self.plot.cget("height")) != requested:
                self.plot.configure(height=requested)

    def _draw(self, _event=None):
        canvas = self.plot
        canvas.delete("all")
        self._plot_bounds = None
        chart = self._overview
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if not chart or not chart["buckets"]:
            if not self._busy and chart is not None:
                canvas.create_text(width / 2, height / 2, text="当前范围内没有可绘制的时间记录", fill=self._plot_text)
            return
        left, right, top, bottom = 61, width - 15, 61, height - 27
        if right <= left or bottom <= top:
            return
        self._plot_bounds = left, right, top, bottom
        count = len(chart["buckets"])
        step = (right - left) / count
        start, period = chart["first_utc_ns"], chart["bucket_ns"]
        maximum = max(1, max(bucket["frames"] for bucket in chart["buckets"]))
        labels = [(20, "阶段"), (40, "事件"), ((top + bottom) / 2, "帧数")]
        if bottom - top >= 48:
            labels.extend(((top, str(maximum)), (bottom, "0")))
        for y, label in labels:
            canvas.create_text(left - 9, y, text=label, fill=self._plot_text, anchor="e")
        canvas.create_line(left, bottom, right, bottom, fill="#8293a7")
        for index, bucket in enumerate(chart["buckets"]):
            x1, x2 = left + index * step, left + (index + 1) * step
            if bucket["frames"]:
                bar_height = max(1, (bottom - top) * bucket["frames"] / maximum)
                canvas.create_rectangle(x1, bottom - bar_height, max(x1 + 1, x2 - 2), bottom, fill="#347dc7", width=0)
            if bucket["incidents"]:
                mid = (x1 + x2) / 2
                canvas.create_oval(mid - 3, 37, mid + 3, 43, fill="#c46a27", width=0)
            if bucket.get("markers"):
                mid = (x1 + x2) / 2
                canvas.create_polygon(mid, 44, mid + 4, 48, mid, 52, mid - 4, 48, fill="#ae83ed")
            if index == self._keyboard_bucket:
                canvas.create_rectangle(x1, top, x2, bottom, outline=self._plot_text, dash=(2, 2))
        self._draw_phases(chart["phase_marks"], start, period, step, left, right)
        for index, bucket in enumerate(chart["buckets"]):
            if bucket.get("paused"):
                canvas.create_line(left + index * step, 29, left + (index + 1) * step, 29, fill="#ecbd54", width=2)
        row = self._selected()
        if row.get("locatable") and start <= row["utc_ns"] <= chart["last_utc_ns"]:
            x = left + (row["utc_ns"] - start) / period * step
            canvas.create_line(x, 10, x, bottom, fill=self._plot_text, dash=(3, 3))
        if chart["first_utc_ns"] == chart["last_utc_ns"]:
            canvas.create_text(left, bottom + 13, text="0 s", fill=self._plot_text, anchor="w")
        else:
            span = period * count
            for index in range(5):
                shift = span * index // 4
                x = left + (right - left) * index / 4
                canvas.create_text(x, bottom + 13, text=f"{shift / 1e9:.5g} s", fill=self._plot_text,
                                   anchor="w" if index == 0 else "e" if index == 4 else "center")

    def _draw_phases(self, marks, start, period, step, left, right):
        phases = {}
        for mark in marks:
            phases.setdefault(mark["phase"], {})[mark["kind"]] = mark["utc_ns"]

        def x_at(stamp):
            return max(left, min(right, left + (stamp - start) / period * step))

        for number, times in phases.items():
            end_wait = times.get("phase_confirmed", times.get("phase_confirmation_cancelled"))
            for begin, end, color in ((times.get("phase_confirmation_waiting"), end_wait, "#8293a7"),
                                      (times.get("phase_started"), times.get("phase_ended"), "#458970")):
                if begin is None and end is None:
                    continue
                begin_x, end_x = x_at(end if begin is None else begin), x_at(begin if end is None else end)
                if begin_x >= end_x:
                    self.plot.create_line(begin_x, 14, begin_x, 26, fill=color, width=2)
                    if begin_x > end_x:
                        self.plot.create_line(end_x, 14, end_x, 26, fill=color, width=2)
                else:
                    self.plot.create_rectangle(begin_x, 14, end_x, 26, fill=color, width=0)
                    if color == "#458970" and end_x - begin_x > 22:
                        self.plot.create_text((begin_x + end_x) / 2, 20, text=str(number), fill="#ffffff")

    def _plot_index(self, event, *, include_incidents=False):
        if not self._plot_bounds:
            return None
        left, right, top, bottom = self._plot_bounds
        if left <= event.x < right and (27 if include_incidents else top) <= event.y <= bottom:
            count = len(self._overview["buckets"])
            return min(count - 1, int((event.x - left) * count / (right - left)))
        return None

    def _interval(self, index):
        start = self._overview["first_utc_ns"] + index * self._overview["bucket_ns"]
        return start, min(self._overview["last_utc_ns"], start + self._overview["bucket_ns"] - 1)

    def _plot_tip(self, event):
        if getattr(event, "keysym", None) in ("??", ""):
            index = self._plot_index(event, include_incidents=True)
        else:
            index = self._keyboard_bucket
        if not self._overview or self._overview["first_utc_ns"] is None:
            return _PLOT_HELP
        if index is None:
            return f"{_PLOT_HELP}\n横轴起点（UTC）{format_receipt_utc(self._overview['first_utc_ns'])}"
        start, end = (format_receipt_utc(value) for value in self._interval(index))
        bucket = self._overview["buckets"][index]
        parts = [f"UTC {start} 至 {end}", f"已保存 {bucket['frames']:,} 帧，待核对事件 {bucket['incidents']:,} 条。"]
        if bucket.get("markers"):
            parts.append(f"实验标记 {bucket['markers']} 条。")
        if bucket.get("paused"):
            parts.append("此时段含有操作员暂停记录。")
        if not self.camera.get():
            parts.extend(f"{self.cameras.get(camera, camera)} · {count:,} 帧"
                         for camera, count in bucket["camera_counts"].items())
        return "\n".join((*parts, _PLOT_HELP))

    def _click_plot(self, event):
        index = self._plot_index(event)
        if index is not None:
            self._keyboard_bucket = index
            self.plot.focus_set()
            self._draw()
            self._open_bucket(index)

    def _step_bucket(self, change):
        if self._overview and self._overview["buckets"]:
            count = len(self._overview["buckets"])
            index = self._keyboard_bucket
            self._keyboard_bucket = 0 if index is None else min(count - 1, max(0, index + change))
            self._draw()
        return "break"

    def _open_bucket(self, index):
        if self._busy or index is None:
            return
        counts = self._overview["buckets"][index]["camera_counts"]
        if counts:
            preferred = self.camera.get() or self.active_camera
            camera = preferred if preferred in counts else next(iter(counts))
            self._jump(*self._interval(index), camera=camera)

    def _frames_help(self):
        row = self._selected()
        kind = row.get("kind")
        if row.get("marker_id") and self.open_marker:
            return "逐台查找标记时刻及之前最近的原图，以及标记后最近的原图。按主机收图时间计算时间差。"
        if not self.cameras:
            return "本次实验没有登记相机。可悬停在事件行查看采集错误。"
        if not row:
            return "先选中一条事件。"
        if kind in _EVENT_IMAGE_KINDS:
            if _window_signal(row):
                text = "按本次触发请求，查找选中相机已保存的全部事件前后原图。"
                if row["details"].get("overlap"):
                    text += "若与其他事件共用原图，会按本次触发的时间标记图像位置。"
                return text
            if kind == CSV_TRIGGER_EVENT:
                return "这条 CSV 状态记录没有生成触发请求。悬停或按 F1 查看监看结果。"
            return "这条记录缺少触发标识。可在“事件窗口”中查看有编号的记录。"
        if kind == "event_phase_summary":
            return "这条记录汇总整个阶段的请求次数。将光标移到事件行可查看各原因的计数。"
        if kind == "signal_unused":
            return "阶段结束时，这次触发请求仍在等待处理。"
        if not row["locatable"]:
            return ("旧台账集中登记漏拍范围，缺少计划 UTC 时刻。" if kind == "schedule_miss" else
                    "这条记录缺少可定位的时间。")
        if self._failed_frame_time(row):
            return "以待写帧的主机收图时间为中心，查找已成功保存的附近原图。"
        if kind == "schedule_miss":
            return "按漏发请求的计划时间范围查找附近已保存的原图。"
        if kind == NOTICE_EVENT:
            return "按相机、阶段、序号和收图时刻定位该张原图。"
        return "以该事件标记的时间为中心查找已保存的原图，起止条件可在查找窗口修改。"

    def _has_frame_action(self, row):
        if row.get("marker_id") and self.open_marker:
            return True
        if not self.cameras or row.get("kind") in ("event_phase_summary", "signal_unused"):
            return False
        return bool(_window_signal(row) if row.get("kind") in _EVENT_IMAGE_KINDS else row.get("locatable"))

    def _incident_action(self, row):
        kind = row["kind"]
        if row.get("trigger_event_id") and self.cameras:
            return "双击或按 Enter 查看本组各相机的原图和缺帧情况。"
        if kind == "event_phase_summary":
            details = row["details"]
            return (f"本阶段有 {details.get('early_signals', 0)} 次请求早于缓存就绪，"
                    f"{details.get('busy_signals', 0)} 次请求遇到正在进行的收图。")
        if kind == CSV_TRIGGER_EVENT and row["details"].get("state") == "fault":
            return "悬停在此行或按 F1，可查看 CSV 路径和监看中断原因。"
        if _window_signal(row) and self.cameras:
            return "双击或按 Enter，按这次触发查看已保存的事件前后原图。"
        if self._failed_frame_time(row):
            return "用待写帧的收图时刻查附近已保存的原图。悬停可查看写入错误。"
        if kind == "schedule_miss" and row["locatable"]:
            return "按漏发请求的计划时段查附近原图。悬停可查看计划序号和时刻。"
        if self._has_frame_action(row):
            return "悬停或按 F1 查看这条记录的依据。双击可查附近已保存的原图。"
        if kind == "error" and not self.cameras:
            return "悬停或按 F1 查看采集错误详情。"
        return self._frames_help()

    def _failed_frame_time(self, row):
        utc = row["details"].get("frame_received_utc_ns")
        return utc if type(utc) is int and 1 <= utc <= _UTC_LIMIT else None

    def _double_click(self, event):
        if key := self.table.identify_row(event.y):
            self.table.selection_set(key)
            self._open_selected()

    def _open_selected(self):
        if self._selected().get("trigger_event_id") and self.cameras:
            self._open_selected_trigger()
        else:
            self._open_selected_frames()

    def _open_selected_frames(self):
        row = self._selected()
        if not self._busy and self.open_marker and (marker := row.get("marker_id")):
            try:
                self.open_marker(self.experiment, self.experiment_id, marker)
            except CaptureError as exc:
                self._error_detail = ui_error(exc)
                self.summary.set(ui_error(exc, details=False))
            return
        if self._busy or not self._has_frame_action(row):
            return
        if context := _window_signal(row):
            self._jump(None, None, phase=row["phase"], event_context=context, event_number=row["details"].get("event_number"))
            return
        if row["kind"] == NOTICE_EVENT and (sequence := row["details"].get("sequence")):
            self._jump(row["utc_ns"], row["utc_ns"], row["camera_id"], row["phase"], sequence=sequence)
            return
        margin = max(500_000_000, self._overview["bucket_ns"] or 0)
        stamp = self._failed_frame_time(row) or row["utc_ns"]
        last_due = row["details"].get("last_due_utc_ns")
        end = last_due if row["kind"] == "schedule_miss" and type(last_due) is int and stamp <= last_due <= _UTC_LIMIT else stamp
        failed = row["details"].get("failed_cameras", ())
        camera = row["camera_id"] or (failed[0] if len(failed) == 1 and failed[0] in self.cameras and not self.camera.get() else None)
        self._jump(max(1, stamp - margin), min(_UTC_LIMIT, end + margin), camera, row["phase"])

    def _jump(self, start, end, camera=None, phase=None, *, sequence=None, event_context=None, event_number=None):
        try:
            self.open_frames(self.experiment, self.experiment_id, {
                "camera": camera or self.camera.get() or self.active_camera,
                "phase": phase or self.phase.get(), "start": start, "end": end,
                **({"sequence": sequence} if sequence is not None else {}),
                **({"event_context_id": event_context} if event_context is not None else {}),
                **({"event_number": event_number} if event_number is not None else {}),
            })
        except CaptureError as exc:
            self._error_detail = ui_error(exc)
            self.summary.set(ui_error(exc, details=False))

    def _open_selected_trigger(self):
        if not self._busy and self.cameras and (event := self._selected().get("trigger_event_id")):
            try:
                self.open_trigger(self.experiment, self.experiment_id, event)
            except CaptureError as exc:
                self._error_detail = ui_error(exc)
                self.summary.set(ui_error(exc, details=False))

    def _copy_time(self, _event):
        if (stamp := self._selected().get("utc_ns")) is not None:
            self.top.clipboard_clear()
            self.top.clipboard_append(format_receipt_utc(stamp).replace(" ", "T") + "Z")
        return "break"

    def _table_tip(self, event):
        row = self._selected()
        if getattr(event, "keysym", None) in ("??", ""):
            if self.table.identify_region(event.x, event.y) == "heading":
                return "显示 UTC 时间及其来源。台账登记时间可能晚于事件本身。"
            row = self._rows.get(self.table.identify_row(event.y), {})
        if not row:
            return "选中事件可查附近原图。Ctrl+C 复制所选行的时间。"
        kind, detail = row["kind"], row["details"]
        basis = "主机发现越线" if row["time_basis"] == "host_csv" else _BASES.get(row["time_basis"], "时间缺失")
        text = f"{CAPTURE_EVENT_LABELS[kind]}\n{basis}（UTC）{format_receipt_utc(row['utc_ns'])}"
        if kind == "schedule_miss":
            text += f"\n计划拍摄序号 {detail.get('first_tick', '—')} 至 {detail.get('last_tick', '—')}，从 0 开始。"
            if row["locatable"]:
                text += f"\n计划末次（UTC）{format_receipt_utc(detail.get('last_due_utc_ns'))}。按阶段起点换算。"
            else:
                text += "\n旧台账集中登记了漏拍范围，缺少计划 UTC 时刻。"
        elif kind == "phase_paused":
            text += "\n这一轮拍摄已结束，计划拍摄计时在此暂停。"
        elif kind == "phase_resume_requested":
            seconds = detail.get("pause_to_request_ns", 0) / 1e9
            text += f"\n从暂停到收到继续请求为 {seconds:.3f} 秒。"
            text += "\n请求写入台账后恢复计划计时。"
        elif kind == "phase_pause_cancelled":
            if duration := detail.get("pause_duration_ns"):
                text += f"\n暂停持续 {duration / 1e9:.3f} 秒，随后结束采集。"
            else:
                text += "\n本次暂停请求随阶段结束而撤销。"
        elif kind == "phase_ended" and (duration := detail.get("paused_duration_ns")):
            text += f"\n本阶段累计暂停 {duration / 1e9:.3f} 秒。"
        elif kind == NOTICE_EVENT and detail.get("metrics"):
            if raised := detail.get("started"):
                text += f"\n提醒 {image_quality_alert_text(raised)}。"
            if cleared := detail.get("cleared"):
                text += f"\n提醒解除 {image_quality_alert_text(cleared)}。"
            text += "\n" + image_quality_sample_help({**detail, "phase": row["phase"]})
        elif kind == CSV_TRIGGER_EVENT:
            hint = "\n双击查看本次触发对应的事件前后原图。" if detail.get("signal_id") else ""
            return text + "\n" + csv_trigger_help(detail) + hint
        elif kind == "event_window":
            pre = detail.get("pre_retained_rounds")
            post = detail["post_received_rounds"]
            lines = [text, f"第 {detail['event_number']} 次事件 · {_WINDOW_RESULTS[detail['result']]}",
                     f"每台相机收到事件前 {pre or 0}/{detail['pre_required_rounds']} 张，"
                     f"事件后 {post}/{detail['post_required_rounds']} 张。"]
            if detail.get("boundary_rounds"):
                lines.append(f"跨越主机收信时刻的收图轮次 {detail['boundary_rounds']}。")
            if crossing := detail.get("csv_crossing"):
                lines.extend(csv_crossing_lines(crossing))
            else:
                source = detail["source"]
                lines.append(f"触发来源 {'操作者' if source == 'operator' else source}")
            return "\n".join((*lines, "双击查看这次事件已保存的前后原图。"))
        elif "previous" in detail and "current" in detail:
            text += f"\n相机计数 {detail['previous']} → {detail['current']}。可结合触发日志核对。"
        elif failed := detail.get("failed_cameras"):
            text += "\n缺少 " + "、".join(self.cameras.get(camera, camera) for camera in failed) + " 的图像。"
            for camera, message in detail.get("failure_messages", {}).items():
                text += f"\n{self.cameras.get(camera, camera)} · {message}"
        elif kind == "pre_window_short":
            text += f"\n事件前要求 {detail.get('required_rounds', '—')} 组，缓冲区有 {detail.get('observed_rounds', '—')} 组。"
        elif kind == "event_phase_summary":
            return f"{text}\n{self._incident_action(row)}\n台账时间为阶段汇总的登记时刻。"
        elif kind == "signal_unused":
            text += "\n阶段结束时，这次触发请求仍在等待处理。"
        elif kind == "hardware_identity_unavailable":
            text += f"\n计数器 {detail.get('counter_feature', '—')}。请核对相机计数与外部脉冲记录。"
        elif kind == "error":
            if self._failed_frame_time(row):
                text += (f"\n待写帧序号 {detail.get('sequence', '—')}。"
                         f"\n错误登记（UTC）{format_receipt_utc(row['ledger_utc_ns'])}。")
            text += "\n" + ui_error(CaptureError(detail.get("code", ""), detail.get("message", "")))
        elif kind == MARKER_EVENT:
            label = detail.get("label") or "现场观察"
            return f"{text}\n{label}\n双击查看各相机在标记前后的原图和收图时间差。"
        if row.get("trigger_event_id") and self.cameras:
            text += "\n双击或按 Enter 查看这一组的原图和缺帧相机。"
        elif self._has_frame_action(row):
            action = "查看这次事件的前后原图" if _window_signal(row) else "查看附近原图"
            text += f"\n双击或按 Enter {action}。"
        return text

    def reopen(self):
        self.top.deiconify()
        self.top.lift()
