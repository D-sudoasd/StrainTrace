"""Per-camera progress and write monitoring for the active Capture session."""

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .presentation import (
    MODE_LABELS, PHASE_PAUSE_DISPLAY, RAW_BUFFER_HELP, WRITE_PRESSURE_LABELS, csv_trigger_caption,
    csv_trigger_help, event_window_position, finishing_phase_writes, phase_display, wait_output_camera_text,
    write_pressure_tip,
)
from .pulse_monitor_gui import PulseMonitorWindow, pulse_help


_RECEIPT_NOTICE_S = 1.5
_EVIDENCE_COUNTS = {
    "app_unsaved": "未保存",
    "native_counter_gaps": "计数跳号",
    "native_counter_duplicates": "计数重复",
    "native_counter_resets": "计数回退或绕回",
    "hardware_identity_missing": "缺少触发标识",
    "reported_driver_skips": "驱动报告跳帧",
    "transport_errors": "传输错误",
}


@dataclass(frozen=True)
class CameraMonitorRow:
    received: int
    saved: int
    pending: int
    caption: str
    guidance: str
    attention: bool
    age_s: float | None
    fps: float | None
    evidence: tuple[str, ...]
    hardware_input: str | None = None
    wait_output: str | None = None


_HARDWARE_WAIT = {
    "checking": ("读取中", "正在读取相机的等待输出。"),
    "ready": ("等触发", "相机报告可接收触发，主机正在等首帧。"),
    "not_ready": ("未就绪", "先等相机报告就绪。持续未就绪时请核对相机设置和线路。"),
    "stale": ("回读过时", "请核对现场接线的等待输出。"),
    "unavailable": ("待核实", "未能核实等待输出。可核对相机设置或现场状态信号。"),
}


def _receipt_state(status, progress, camera, count, age_s, fps, recent, wait):
    if status["state"] == "opening":
        return "连接中", "正在打开设备并应用采集计划。", False
    if not status["error"] and (pause := status.get("phase_pause")):
        state = pause["state"]
        label = {"requested": "暂停中", "paused": "已暂停", "resuming": "继续中", "stopping": "收尾中"}[state]
        return label, PHASE_PAUSE_DISPLAY[state][1], False
    if status["state"] == "draining" or status["error"] or (progress and not progress["accepting_images"]):
        return "收尾中", "等待已入队的图像写入并登记。", False
    if gate := status.get("phase_confirmation"):
        label = {"saving_previous": "收尾中", "waiting": "待确认", "arming": "启动中", "stopping": "收尾中"}
        return label[gate["state"]], "可在顶部查看阶段交接或确认要求。", False
    if not progress:
        if finishing_phase_writes(status):
            return "保存中", "本阶段收图已结束，正在完成图像写入和台账登记。", False
        return "准备中", "正在准备采集阶段。", False
    if (limit := progress["frame_limit_per_camera"]) and count >= limit:
        return "已收齐", "已收到本阶段计划的帧数，待存帧会继续写入。", False
    mode = status["mode"]
    window = progress["event_window"]
    active_events = window.get("active_events", 0) if window else 0
    if (mode == "manual" and not progress["manual_signal_id"]) or (
        window and window["state"] == "waiting" and not window["pre_required_rounds"] and not active_events
    ):
        return "等触发", "当前允许等待操作员或软件触发命令。", False

    if mode == "hardware":
        age = progress["elapsed_s"] if age_s is None else age_s
        recent_peer = len(recent) - int(camera in recent) > 0
        if age >= _RECEIPT_NOTICE_S and recent_peer:
            return "无新图", "其他相机近期已有新图。请核对这台相机的触发和连接，漏拍需用触发记录确认。", True
        if age_s is None or age_s >= _RECEIPT_NOTICE_S:
            if wait and (state := wait["state"]) in _HARDWARE_WAIT:
                caption, guidance = _HARDWARE_WAIT[state]
                return caption, guidance, age >= _RECEIPT_NOTICE_S and state in {"not_ready", "stale", "unavailable"}
            return "等触发", "等待外部触发后送图。可核对外部脉冲与接线。", False
    else:
        rate = progress.get("planned_fps") or fps
        threshold = max(_RECEIPT_NOTICE_S, 2 / rate) if rate else _RECEIPT_NOTICE_S
        age = progress["elapsed_s"] if age_s is None else age_s
        if (mode in ("continuous", "timed", "event") and age >= threshold
                and not (window and not window["pre_required_rounds"] and window["state"] == "trigger_queued")):
            return "无新图", "主机暂未收到新图像。请核对相机连接和本阶段的拍摄设置。", True

    if window and window["state"] in ("buffering", "waiting") and not active_events:
        return "预缓存", "触发前图像按窗口长度滚动保留。", False
    if age_s is None:
        return "等首图", "等待本阶段第一帧。", False
    return "收图中", "已收到本阶段的新图像。", False


def camera_monitor_rows(status):
    progress = status["phase_progress"]
    inputs = (progress or {}).get("hardware_inputs", {})
    waits = ((progress or {}).get("hardware_readiness") or {}).get("cameras", {})
    recent = set()
    if progress and status["mode"] == "hardware":
        recent = {camera for camera, row in progress["camera_receipts"].items()
                  if row["last_receipt_age_s"] is not None and row["last_receipt_age_s"] < _RECEIPT_NOTICE_S}
    rows = {}
    for camera, stats in status["cameras"].items():
        receipt = progress["camera_receipts"][camera] if progress else {}
        received = progress["received_by_camera"][camera] if progress else stats["received"]
        saved = receipt["saved"] if progress else stats["saved"]
        age, fps = receipt.get("last_receipt_age_s"), receipt.get("host_receive_fps")
        wait = waits.get(camera)
        caption, guidance, attention = _receipt_state(status, progress, camera, received, age, fps, recent, wait)
        evidence = tuple(f"{label} {stats[key]}" for key, label in _EVIDENCE_COUNTS.items() if stats[key])
        if stats["app_unsaved"]:
            caption, guidance = "有未存", "已有图像未保存。请查看采集状态，结束后校验文件和台账。"
        elif evidence and not attention:
            caption, guidance = "需核对", "设备或台账记录到异常。结束后可逐项复查。"
        rows[camera] = CameraMonitorRow(received, saved, stats["pending_write"], caption, guidance,
                                        bool(evidence) or attention, age, fps, evidence, inputs.get(camera),
                                        wait_output_camera_text(wait) if wait else None)
    return rows


_COLUMN_HELP = {
    "#0": "相机。选中一行可切换左侧预览和下方详情。",
    "#1": "收到是主机已收到的相机帧数。已存是图像已写入且台账已登记的帧数。一帧可包含多张 TIFF。",
    "#2": "待存是本次实验中已入队、尚未登记的帧数，包含正在写入的帧。事件前滚动缓存另计。",
    "#3": "根据阶段所需的触发、主机收图和已观测的异常显示当前状态。",
}


class CaptureMonitor(ttk.Frame):
    def __init__(self, parent, camera_title, select_camera):
        super().__init__(parent)
        self.camera_title = camera_title
        self.select_camera = select_camera
        self.rows = {}
        self._active = None
        self._in_phase = False
        self._write_status = None
        self._pressure_window = None
        self._quality_window = None
        self._quality_status = None
        self._csv_status = None
        self._event_status = None
        self._pulse_status = None
        self._pulse_wait = {}
        self._pulse_window = None
        self._pulse_experiment = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.phase = ttk.Label(self, style="CaptureSection.TLabel", width=1)
        self.phase.grid(row=0, column=0, sticky="ew", pady=(0, 3))
        self.scope = ttk.Label(self, width=1, style="CaptureMuted.TLabel", justify=tk.LEFT, takefocus=True)
        self.scope.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        wrap_label_on_resize(self.scope)
        ToolTip(self.scope, self._scope_help, title="采集监看")
        table = ttk.Frame(self)
        table.grid(row=2, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.cameras = ttk.Treeview(table, columns=("counts", "pending", "status"),
                                    show="tree headings", height=2, selectmode="browse", takefocus=True)
        for column, heading, width, minimum, anchor in (
            ("#0", "相机", 104, 60, tk.W),
            ("counts", "收／存", 85, 62, tk.CENTER),
            ("pending", "待存", 51, 44, tk.CENTER),
            ("status", "状态", 76, 70, tk.CENTER),
        ):
            self.cameras.heading(column, text=heading, anchor=anchor)
            self.cameras.column(column, width=width, minwidth=minimum, anchor=anchor, stretch=column == "#0")
        self.cameras.tag_configure("attention", foreground="#ffda87")
        self.cameras.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(2, minsize=self.cameras.winfo_reqheight())
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.cameras.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.cameras.configure(yscrollcommand=scroll.set)
        self.cameras.bind("<<TreeviewSelect>>", self._select_row)
        ToolTip(self.cameras, self._table_help, title="相机收图与保存")
        ToolTip(scroll, "滚动查看其余相机。也可聚焦表格后用方向键查看。")
        self.detail = ttk.Label(self, width=1, justify=tk.LEFT)
        self.detail.grid(row=3, column=0, sticky="ew", pady=(8, 6))
        wrap_label_on_resize(self.detail)
        ToolTip(self.detail, lambda _event: self.describe(self._active), title="选中相机")
        self.buffer = ttk.Progressbar(self, orient="horizontal", maximum=100, mode="determinate",
                                      style="CaptureCopy.Horizontal.TProgressbar")
        self.buffer.grid(row=4, column=0, sticky="ew", pady=(0, 4))
        self.buffer_text = ttk.Label(self, width=1, style="CaptureMuted.TLabel", justify=tk.LEFT)
        self.buffer_text.grid(row=5, column=0, sticky="ew")
        wrap_label_on_resize(self.buffer_text)
        for widget in (self.buffer, self.buffer_text):
            ToolTip(widget, RAW_BUFFER_HELP)
        actions = ttk.Frame(self)
        actions.grid(row=6, column=0, sticky="ew", pady=(5, 0))
        actions.columnconfigure(0, weight=1)
        self.trend_button = ttk.Button(actions, text="写入趋势…", command=self.show_write_pressure, style="Small.TButton")
        self.trend_button.grid(row=0, column=0, sticky="ew")
        ToolTip(self.trend_button, lambda _event: write_pressure_tip((self._write_status or {}).get("write_pressure")),
                title="写入趋势")
        self.pulse_button = ttk.Button(actions, text="脉冲监看…", command=self.show_pulses, style="Small.TButton")
        ToolTip(self.pulse_button, lambda _event: pulse_help(self._pulse_status), title="脉冲监看")
        self.quality_button = ttk.Button(actions, text="画质抽查…", command=self.show_quality, style="Small.TButton")
        ToolTip(self.quality_button, "查看各相机最近一次抽查的原图指标、帧号和提醒。")

    def write_window_open(self):
        return self._pressure_window is not None and self._pressure_window.top.winfo_exists()

    def _scope_help(self, _event):
        lines = []
        if window := self._event_status:
            lines.append(f"{event_window_position(window)} · 已完成 {window['completed_events']} 次")
            if window.get("overlap"):
                lines.append(f"{window['active_events']} 次事件正在收图 · 已接收 {window['accepted_events']} 次触发")
                lines.append(f"缓存未满时收到 {window['early_signals']} 次 · 达到同时收图上限时收到 {window['busy_signals']} 次")
                lines.append("同一张原图可以属于多个事件，在采集时间线中可分别查看。")
            elif window["event_count"] == 1:
                if window["pre_required_rounds"]:
                    lines.append("缓存未满时触发会保留当时已有的前图，台账会记录缺少的张数。")
            else:
                lines.append(f"缓存未满时收到 {window['early_signals']} 次 · 正在收图时收到 {window['busy_signals']} 次")
                lines.append("每次事件使用新收到的前图。未接收的请求需要重新发送或重新越线。")
            lines.append("收图结束后，事件记录和原图还需写入。保存完成后可在采集时间线复查。")
        if self._csv_status:
            lines.append(csv_trigger_help(self._csv_status))
        if any(row.hardware_input for row in self.rows.values()):
            lines.append("选中相机可核对计划触发输入和首帧收图。启用等待输出监测时，还可查看相机回读。")
        return "\n".join(lines) or "收到和已存随当前阶段呈现。待存统计整次采集中已入队且尚未登记的帧。"

    def quality_window_open(self):
        return self._quality_window is not None and self._quality_window.top.winfo_exists()

    def update_quality_watch(self, status):
        report = (status or {}).get("image_quality_watch")
        if self.quality_window_open() and (
                report is None or status["experiment_id"] != self._quality_status["experiment_id"]):
            self._quality_window.top.destroy()
        self._quality_status = status
        if report is None:
            self.quality_button.grid_remove()
            return
        if not self.quality_button.winfo_manager():
            self.quality_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        if self.quality_window_open():
            self._quality_window.render(status)

    def show_quality(self):
        status = self._quality_status
        if not status or "image_quality_watch" not in status:
            return
        if self.quality_window_open():
            self._quality_window.top.deiconify()
            self._quality_window.top.lift()
            return
        from .image_quality_watch_gui import ImageQualityWatchWindow

        candidates = status["image_quality_watch"]["cameras"].values()
        camera = next((row["camera_id"] for row in candidates if row["phase"] == status["phase"] and row["alerts"]),
                      self._active)
        self._quality_window = ImageQualityWatchWindow(self.winfo_toplevel(), self.camera_title, self.select_camera)
        self._quality_window.render(status, preferred=camera)

    def show_write_pressure(self):
        if not self._write_status or "write_pressure" not in self._write_status:
            return
        if self.write_window_open():
            self._pressure_window.top.lift()
            return
        from .write_pressure_gui import WritePressureWindow

        self._pressure_window = WritePressureWindow(self.winfo_toplevel())
        self._pressure_window.render(self._write_status)

    def update_write_pressure(self, status):
        self._write_status = status
        if pressure := status.get("write_pressure"):
            label = WRITE_PRESSURE_LABELS[pressure["state"]]
            caption = "写入趋势…" if self.pulse_button.winfo_manager() else f"写入趋势 · {label}…"
            if caption != self.trend_button.cget("text"):
                self.trend_button.configure(text=caption)
            if self.write_window_open():
                self._pressure_window.render(status)

    def pulse_window_open(self):
        return self._pulse_window is not None and self._pulse_window.top.winfo_exists()

    def show_pulses(self):
        if self._pulse_status is None:
            return
        if self.pulse_window_open():
            self._pulse_window.top.lift()
            return
        self._pulse_window = PulseMonitorWindow(self.winfo_toplevel(), self.camera_title)
        self._pulse_window.render(self._pulse_status, self._pulse_wait)

    def update_pulses(self, status):
        experiment_id = status["experiment_id"]
        if self._pulse_experiment != experiment_id and self.pulse_window_open():
            self._pulse_window.top.destroy()
        self._pulse_experiment = experiment_id
        self._pulse_status = pulses = status.get("hardware_groups")
        visible = pulses is not None and pulses["phase"] == status["phase"]
        progress = status["phase_progress"] if visible else None
        wait = ((progress or {}).get("hardware_readiness") or {}).get("cameras", {})
        self._pulse_wait = {camera: row["state"] for camera, row in wait.items()}
        if visible and not self.pulse_button.winfo_manager():
            self.pulse_button.grid(row=0, column=1, padx=(5, 0), sticky="e")
        elif not visible and self.pulse_button.winfo_manager():
            self.pulse_button.grid_remove()
        if pulses is not None and self.pulse_window_open():
            self._pulse_window.render(pulses, self._pulse_wait)

    def _select_row(self, _event):
        if (selection := self.cameras.selection()) and selection[0] != self._active:
            self.select_camera(selection[0])

    def _table_help(self, event):
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            if camera := self.cameras.identify_row(event.y):
                return self.describe(camera)
            if column := self.cameras.identify_column(event.x):
                return _COLUMN_HELP.get(column, _COLUMN_HELP["#0"])
        return self.describe(self.cameras.focus() or self._active)

    def describe(self, camera):
        if (row := self.rows.get(camera)) is None:
            return "选中相机后可查看主机收图、台账登记和待存数量。"
        scope = "本阶段" if self._in_phase else "累计"
        details = [
            self.camera_title(camera),
            f"{scope}收到 {row.received} 帧 · 已存 {row.saved} 帧 · 本次采集待存 {row.pending} 帧",
        ]
        if self._in_phase:
            details.append(self._timing(row))
            if row.hardware_input:
                details.append(f"计划触发输入 {row.hardware_input}")
            if row.wait_output:
                details.append(row.wait_output)
            if row.fps is not None:
                details.append("帧率按本阶段首尾两帧的主机接收时间计算。")
        details.append(row.guidance)
        if row.evidence:
            details.append("本次采集记录 · " + "，".join(row.evidence))
        return "\n".join(details)

    @staticmethod
    def _timing(row):
        age = "本阶段未收图" if row.age_s is None else f"距上次收图 {row.age_s:.1f} 秒"
        return age if row.fps is None else f"{age} · {row.fps:.2f} 帧/s"

    def render(self, status, active_camera):
        self.rows = rows = camera_monitor_rows(status)
        self._active = active_camera if active_camera in rows else next(iter(rows), None)
        progress = status["phase_progress"]
        self._in_phase = progress is not None
        stage = f"阶段 {status['phase']} · {MODE_LABELS[status['mode']]}" if progress else "本次采集"
        self.phase.configure(text=stage)
        self._csv_status = (progress or {}).get("csv_trigger")
        self._event_status = (progress or {}).get("event_window")
        if self._event_status or (progress and status["mode"] == "hardware"):
            _state, message = phase_display(status["mode"], progress)
            if self._csv_status:
                message += f" · {csv_trigger_caption(self._csv_status)}"
        elif self._csv_status:
            message = csv_trigger_caption(self._csv_status)
        else:
            message = "本阶段收／存 · 待存按整次采集计" if progress else "本次采集累计帧数"
        self.scope.configure(text=message)
        for camera in self.cameras.get_children():
            if camera not in rows:
                self.cameras.delete(camera)
        for camera, row in rows.items():
            values = (f"{row.received}/{row.saved}", row.pending, row.caption)
            tags = ("attention",) if row.attention else ()
            if not self.cameras.exists(camera):
                self.cameras.insert("", tk.END, iid=camera, text=self.camera_title(camera), values=values, tags=tags)
            elif self.cameras.item(camera, "values") != values or self.cameras.item(camera, "tags") != tags:
                self.cameras.item(camera, values=values, tags=tags)
        if self._active and self.cameras.selection() != (self._active,):
            self.cameras.selection_set(self._active)
            self.cameras.focus(self._active)
            self.cameras.see(self._active)
        row = rows.get(self._active)
        if row:
            short_guidance = row.guidance.partition("。")[0] + "。"
            detail = self._timing(row) if progress else short_guidance
            if row.hardware_input:
                detail += f" · 计划输入 {row.hardware_input}"
            if progress and not self.pulse_button.winfo_manager() and (row.attention or row.caption != "收图中"):
                detail += f"\n{short_guidance}"
            self.detail.configure(text=detail)
        used = status["buffered_raw_bytes"] / status["raw_buffer_limit_bytes"]
        self.buffer["value"] = min(100, 100 * used)
        occupancy = "不足 1%" if 0 < used < .01 else f"{used:.0%}"
        self.buffer_text.configure(text=f"字节缓冲 {occupancy} · 待存 {sum(row.pending for row in rows.values())} 帧")
