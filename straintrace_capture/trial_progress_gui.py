"""Desktop navigation between the planned phases and their saved trials."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .config import CaptureError
from .presentation import CAPTURE_ERROR_LABELS, MODE_LABELS, ROLE_LABELS, csv_trigger_caption, ui_error
from .viewport import repaint_after_resize


TRIAL_STATUS = {
    "matched": "试采匹配",
    "review": "需要复核",
    "changed": "设置已变化",
    "unbound": "来源待核对",
    "missing": "尚未试采",
}
TRIAL_CHANGES = {"phase": "阶段参数", "volume": "保存卷", "camera_settings": "相机设置",
                 "cti_files": "相机驱动", "storage": "存储设置", "preview_geometry": "当前预览",
                 "start_confirmation": "开始确认说明"}
BATCH_STAGE = {"queued": "等待试采", "checking": "正在检查", "capturing": "正在试采", "verifying": "正在核验",
               "confirmation": "等待确认", "manual": "等待操作员触发", "buffering": "正在缓存事件前图像",
               "csv": "等待 CSV 触发", "hardware": "等待硬件信号",
               "completed": "试采完成", "review": "完成 · 有提醒", "stopped": "已停止", "failed": "需复核",
               "no_images": "时限内未保存图像", "preflight": "请查看采前检查"}


def _batch_live_help(live):
    if confirmation := live.status.get("phase_confirmation"):
        if confirmation["state"] == "waiting":
            return f"{confirmation['instruction']}\n确认后开始本阶段计时。"
        return "正在准备本阶段。"
    progress = live.status.get("phase_progress")
    if progress is None:
        return "正在准备相机。"
    if not progress["accepting_images"]:
        return "正在保存已收到的图像。"
    mode = live.plan.mode
    remaining = progress.get("duration_s")
    timer = f" · 剩余 {max(0., remaining - progress['elapsed_s']):.1f} 秒" if remaining is not None else ""
    if mode == "hardware":
        inputs = "、".join(dict.fromkeys(progress.get("hardware_inputs", {}).values()))
        line = f"等待外部设备向 {inputs} 发送脉冲。" if inputs else "等待外部设备发送硬件脉冲。"
        readiness = progress.get("hardware_readiness")
        if readiness and not readiness["ready"] and not any(progress["received_by_camera"].values()):
            line = "等待就绪回读。请在设备上核对相机的触发等待状态。"
    elif mode == "event":
        window = progress["event_window"] or {}
        if window.get("state") == "buffering":
            line = f"正在缓存事件前图像（{window['pre_buffered_rounds']}/{window['pre_required_rounds']} 轮）。"
        elif status := progress.get("csv_trigger"):
            line = csv_trigger_caption(status)
            if error := status.get("error"):
                line += f" · {error['message']}"
        elif live.action:
            line = "事件前图像已就绪。观察到事件时点“触发一次”。"
        else:
            line = "正在记录或保存本次事件图像。"
    elif mode == "manual":
        line = "点“触发一次”，各相机各取一张。" if live.action else "正在处理本轮触发。"
    else:
        return "正在试采。"
    if live.external and live.action and mode in ("manual", "event"):
        line += "也可用本次试采目录中的外部控制接口。"
    return line + timer


class TrialProgressWindow:
    def __init__(self, parent, load, start_trial, report, check_plan, start_batch, stop_batch, show_batch_check):
        self._load, self._start_trial, self._report, self._check_plan = load, start_trial, report, check_plan
        self._start_batch, self._stop_batch, self._show_batch_check = start_batch, stop_batch, show_batch_check
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("StrainTrace · 阶段试采")
        self.top.geometry("750x480")
        self.top.minsize(615, 440)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.protocol("WM_DELETE_WINDOW", self._dismiss)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self._dismiss())
        self.top.bind("<Destroy>", self._destroy, add="+")
        self.summary = tk.StringVar(self.top)
        self.detail = tk.StringVar(self.top)
        self.batch_status = tk.StringVar(self.top)
        self.batch_guided = tk.BooleanVar(self.top, False)
        self._progress = None
        self._phase_rows = {}
        self._blocked = ""
        self._ready = False
        self._revision = 0
        self._pending = None
        self._advance = False
        self.batch_active = self._batch_stopping = False
        self._batch_rows = {}
        self._batch_check = None
        self._batch_live = None
        self._build()
        self.refresh()

    def _build(self):
        area = ttk.Frame(self.top, padding=14)
        area.pack(fill=tk.BOTH, expand=True)
        area.columnconfigure(0, weight=1)
        area.rowconfigure(1, weight=1)
        summary = ttk.Label(area, width=1, textvariable=self.summary, justify=tk.LEFT, style="CaptureSection.TLabel")
        summary.grid(row=0, sticky="ew", pady=(0, 8))
        wrap_label_on_resize(summary)
        ToolTip(summary, lambda _event: self.summary.get(), title="阶段试采进度")

        body = ttk.Frame(area)
        body.grid(row=1, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(body, columns=("phase", "role", "mode", "status"), show="headings",
                                  selectmode="browse", height=5)
        for key, title, width in (("phase", "阶段", 98), ("role", "用途", 108),
                                  ("mode", "拍摄方式", 130), ("status", "试采状态", 205)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=73)
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<<TreeviewSelect>>", lambda _event: self._controls())
        ToolTip(self.table, self._row_help, title="试采状态")
        ToolTip(scroll, "按正式计划的顺序查看其余阶段。", title="阶段列表")

        detail = ttk.Label(area, width=1, textvariable=self.detail, justify=tk.LEFT, style="CaptureMuted.TLabel")
        detail.grid(row=2, sticky="ew", pady=(8, 5))
        wrap_label_on_resize(detail)
        ToolTip(detail, lambda _event: self.detail.get(), title="选中阶段")
        self.batch_note = ttk.Label(area, width=1, textvariable=self.batch_status, justify=tk.LEFT, style="CaptureMuted.TLabel")
        self.batch_note.grid(row=3, sticky="ew", pady=(0, 5))
        self.batch_note.grid_remove()
        wrap_label_on_resize(self.batch_note)
        ToolTip(self.batch_note, lambda _event: self.batch_status.get(), title="连续试采")
        actions = ttk.Frame(area)
        actions.grid(row=4, sticky="ew", pady=(5, 0))
        self.close_button = ttk.Button(actions, text="关闭", command=self._dismiss, style="Small.TButton")
        self.close_button.pack(side=tk.LEFT)
        ToolTip(self.close_button, lambda _event: "隐藏进度窗口。可从“阶段试采”重新打开。"
                if self.batch_active else "返回采集工作台。可再次打开查看试采状态。")
        self.report_button = ttk.Button(actions, text="试采报告…", command=self._view_report, style="Small.TButton")
        self.report_button.pack(side=tk.LEFT, padx=(6, 0))
        ToolTip(self.report_button, self._report_help)
        self.start_button = ttk.Button(actions, text="试采此阶段…", command=self._start, style="Record.TButton")
        self.start_button.pack(side=tk.RIGHT)
        ToolTip(self.start_button, self._start_help)
        self.check_button = ttk.Button(actions, text="检查计划", command=self._check, style="Small.TButton")
        self.check_button.pack(side=tk.RIGHT, padx=(0, 6))
        ToolTip(self.check_button, lambda _event: self._blocked or "检查整份计划的触发、相机回读和磁盘空间。")
        self.batch_actions = ttk.Frame(area)
        self.batch_actions.grid(row=5, sticky="ew", pady=(6, 0))
        self.batch_actions.grid_remove()
        self.batch_button = ttk.Button(self.batch_actions, text="连续试采…", command=self._batch_action, style="Small.TButton")
        self.batch_button.pack(side=tk.LEFT)
        ToolTip(self.batch_button, self._batch_help, title="连续试采")
        self.batch_option = ttk.Checkbutton(self.batch_actions, text="纳入需确认或触发的阶段", variable=self.batch_guided,
                                            command=self._controls)
        ToolTip(self.batch_option, "仍按计划顺序。程序在需要确认的阶段等待，手动和事件阶段提供触发提示，"
                "配置 CSV 或硬件输入时等待相应信号。")
        self.batch_step_button = ttk.Button(self.batch_actions, style="Record.TButton", command=self._request_batch_step)
        ToolTip(self.batch_step_button, lambda _event: _batch_live_help(self._batch_live) if self._batch_live else "")
        self.batch_check_button = ttk.Button(self.batch_actions, text="查看本次采前检查…", style="Small.TButton",
                                             command=lambda: self._show_batch_check(self._batch_check))
        ToolTip(self.batch_check_button, "查看连续试采停下时这一阶段的检查事项。")
        self.table.focus_set()

    def refresh(self, *, follow_recommendation=False):
        if self.batch_active:
            return
        self._advance |= follow_recommendation
        self._revision += 1
        self._ready = False
        self.summary.set("正在对照当前计划与试采记录")
        self._controls()
        if self._pending is None:
            self._pending = self.top.after_idle(self._request)

    def _request(self):
        self._pending = None
        version = self._revision

        def completed(progress):
            if self.top.winfo_exists() and version == self._revision:
                self._populate(progress)

        def failed(error):
            if self.top.winfo_exists() and version == self._revision:
                self.summary.set(ui_error(error, details=False))
                self.detail.set(ui_error(error))

        self._load(completed, failed)

    def _populate(self, progress):
        selected = self.table.selection()
        current = int(selected[0]) if selected else None
        self._progress = progress
        self._phase_rows = {row.number: row for row in progress.phases}
        self._ready = True
        self.table.delete(*self.table.get_children())
        for row in progress.phases:
            status = "可选试采" if row.status == "missing" and row.role != "test" else TRIAL_STATUS[row.status]
            if row.status == "missing" and row.batch_exclusion in ("confirmation", "trigger"):
                status = "需确认" if row.batch_exclusion == "confirmation" else "需触发"
            if row.number == progress.recommended:
                status += " · 建议处理"
            self.table.insert("", "end", iid=str(row.number), values=(
                f"阶段 {row.number}", ROLE_LABELS[row.role], MODE_LABELS[row.mode], status,
            ))
        if progress.recommended and (current not in self._phase_rows or self._advance):
            current = progress.recommended
        default = current if current in self._phase_rows else progress.phases[0].number
        self.table.selection_set(str(default))
        self.table.focus(str(default))
        self.table.see(str(default))
        self._advance = False
        experiments = [row for row in progress.phases if row.role == "test"]
        matched = sum(row.status == "matched" for row in experiments)
        scope = (f"实验图阶段 {matched}/{len(experiments)} 试采匹配且无提醒"
                 if experiments else f"共 {len(progress.phases)} 个阶段")
        if progress.blockers:
            next_step = f"{progress.blockers} 项采前阻塞"
        elif progress.recommended:
            next_step = f"建议处理阶段 {progress.recommended}"
        elif progress.checked_plan:
            next_step = "已检查整份计划"
        else:
            next_step = "可检查整份计划"
        self.summary.set(f"{scope} · {next_step}")
        self._controls()

    def _selection(self):
        chosen = self.table.selection()
        return self._phase_rows.get(int(chosen[0])) if self._ready and chosen else None

    def _description(self, row):
        if row is None:
            return ""
        blockers = tuple(item for item in row.issues if item["severity"] == "block")
        if blockers:
            return "。".join(CAPTURE_ERROR_LABELS.get(item["code"], "请查看采前检查报告").rstrip("。")
                            for item in blockers[:2]) + "。"
        trigger = {"manual": "开始后用“触发一次”发送请求。", "event": "预缓存就绪后用“触发一次”。",
                   "hardware": "相机就绪后等外部设备发送物理触发。", "single": "每台相机拍摄一张。"}.get(row.mode, "")
        if row.status == "missing":
            if row.batch_exclusion == "confirmation":
                return "此阶段要求人工确认。可将它纳入连续试采或单独启动。" + trigger
            return "可按需使用当前相机和保存设置单独试采。" + trigger
        if row.status == "changed":
            fields = "、".join(TRIAL_CHANGES.get(key, key) for key in row.record["changes"])
            return f"{fields}与原试采不同。可查看原报告，再按当前设置试采。"
        if row.status == "unbound":
            return "原试采缺少可比的参数或保存卷信息。可按当前设置试采。"
        if row.status == "review":
            notices = [f"{row.image_alerts} 张原图有提醒"] if row.image_alerts else []
            notices.extend(CAPTURE_ERROR_LABELS.get(item["code"], "请在采前检查中核对该阶段。") for item in row.issues[:2])
            if row.record["findings"] or row.record.get("sustained_write_lag") or not row.record["observed"]:
                notices.append("请从试采报告核对收图、写入和校验结果")
            return "。".join(text.rstrip("。") for text in notices) + "。"
        checked = sum(item["checked"] for item in row.record.get("image_checks", {}).values())
        pictures = f"已检查 {checked} 张试采原图。" if checked else ""
        return f"试采与当前计划匹配。{pictures}\n可用“检查计划”核对相机回读和磁盘空间。"

    def _row_help(self, event):
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            if self.table.identify_region(event.x, event.y) == "heading":
                return {
                    "#1": "阶段编号与正式采集计划一致，试采的图像保存在独立目录。",
                    "#2": "标明该阶段的图像用于实验、参考或镜头标定。",
                    "#3": "决定怎样开始取图。硬件触发阶段等外部设备发送物理信号。",
                    "#4": "匹配依据包括相机设置、阶段参数和保存卷。已运行采前检查时也显示回读提醒。",
                }.get(self.table.identify_column(event.x))
            line = self.table.identify_row(event.y)
            if line and self._ready:
                row = self._phase_rows[int(line)]
            else:
                row = None
        else:
            row = self._selection()
        if row:
            if self.batch_active and row.number in self._batch_rows:
                return f"阶段 {row.number} · {BATCH_STAGE[self._batch_rows[row.number]]}\n{self.batch_status.get()}"
            return self._description(row)
        return "按计划顺序显示。建议优先复核异常，再试采尚无匹配记录的实验图阶段。"

    def _report_help(self, _event=None):
        row = self._selection()
        return self._blocked or ("重新核对所选阶段的试采台账和原图，可查看收图、写入及校验结果。"
                                 if row and row.record else "选中有试采记录的阶段后可查看报告。")

    def _start_help(self, _event=None):
        row = self._selection()
        return self._blocked or (f"按现有设置打开阶段 {row.number} 的试采确认。图像保存在单独目录。"
                                 if row else "请先选中需要试采的阶段。")

    def set_block(self, reason):
        if reason != self._blocked:
            self._blocked = reason
            self._controls()

    def _controls(self):
        row = self._selection()
        available = self._ready and not self._blocked and not self.batch_active
        usable = row is not None and available
        self.report_button.state(["!disabled" if usable and row.record else "disabled"])
        self.start_button.state(["!disabled" if usable else "disabled"])
        report_first = row is not None and row.status == "review"
        check_first = row is not None and row.status == "matched"
        self.report_button.configure(style="Record.TButton" if report_first else "Small.TButton")
        self.start_button.configure(text="重新试采…" if row and row.record else "试采此阶段…",
                                    style="Small.TButton" if self.batch_active or report_first or check_first else "Record.TButton")
        self.check_button.configure(style="Record.TButton" if check_first else "Small.TButton")
        self.check_button.state(["!disabled" if available else "disabled"])
        if self.batch_active:
            self.detail.set(_batch_live_help(self._batch_live) if self._batch_live and not self._batch_stopping
                            else "停止后会保存并核验已收到的图像。")
        elif row:
            self.detail.set(self._blocked or self._description(row))
        elif not self._ready:
            self.detail.set(self._blocked)
        batch_visible = self.batch_active or self._batch_rows or (self._progress and len(self._progress.phases) > 1)
        if batch_visible and not self.batch_actions.winfo_manager():
            self.batch_actions.grid()
        elif not batch_visible and self.batch_actions.winfo_manager():
            self.batch_actions.grid_remove()
        guided_pending = bool(self._progress and self._progress.trial_batch_phases(guided=True) != self._progress.batch_phases)
        option_visible = self._ready and not self.batch_active and guided_pending and batch_visible
        if option_visible and not self.batch_option.winfo_manager():
            self.batch_option.pack(side=tk.LEFT, padx=(8, 0))
        elif not option_visible and self.batch_option.winfo_manager():
            self.batch_option.pack_forget()
        self.batch_option.state(["!disabled" if available else "disabled"])
        start_batch = available and bool(self._batch_numbers())
        button_ready = not self._batch_stopping if self.batch_active else start_batch
        self.batch_button.state(["!disabled" if button_ready else "disabled"])
        step = self._batch_live.action if self._batch_live is not None else None
        step_label = self._batch_step_label()
        self.batch_button.configure(text="停止连续试采" if self.batch_active else "连续试采…",
                                    style="Record.TButton" if (self.batch_active and not step_label) or start_batch else "Small.TButton")
        if step_label:
            if not self.batch_step_button.winfo_manager():
                self.batch_step_button.pack(side=tk.RIGHT)
            self.batch_step_button.configure(text=step_label)
            self.batch_step_button.state(["!disabled" if step else "disabled"])
        elif self.batch_step_button.winfo_manager():
            self.batch_step_button.pack_forget()
        self.close_button.configure(text="隐藏" if self.batch_active else "关闭")

    def _batch_numbers(self):
        return self._progress.trial_batch_phases(guided=self.batch_guided.get()) if self._progress else ()

    def _batch_step_label(self):
        live = self._batch_live
        if not self.batch_active or self._batch_stopping or live is None:
            return ""
        if live.status.get("phase_confirmation"):
            return "确认并试采"
        if live.plan.mode in ("manual", "event") and live.plan.csv_trigger is None:
            return "触发一次"
        return ""

    def _batch_help(self, _event=None):
        if self.batch_active:
            return "结束当前试采，等已收图像存完。后面的阶段留待下次运行。"
        if self._blocked:
            return self._blocked
        if not self._ready:
            return "等待当前计划和试采记录核对完成。"
        if not self._batch_numbers():
            if self._progress.trial_batch_phases(guided=True):
                return "勾选“纳入需确认或触发的阶段”即可继续试采。"
            return "目前没有待连续试采的阶段。可选中一个阶段单独试采或复核。"
        return ("按计划顺序试采缺少匹配记录的实验图阶段。"
                "勾选选项后可纳入需要人工确认或触发的阶段。")

    def _batch_action(self):
        if self.batch_active:
            self._stop_batch()
        elif self._ready and not self._blocked and (numbers := self._batch_numbers()):
            self._start_batch(numbers, guided=self.batch_guided.get())

    def _set_batch_note(self, text):
        self.batch_status.set(text)
        if text:
            self.batch_note.grid()
        else:
            self.batch_note.grid_remove()

    def begin_batch(self, numbers):
        self._revision += 1
        self.batch_active, self._batch_stopping = True, False
        self._batch_rows = dict.fromkeys(numbers, "queued")
        self._batch_check = None
        self._batch_live = None
        self.batch_check_button.pack_forget()
        for number in numbers:
            self.table.set(str(number), "status", BATCH_STAGE["queued"])
        self.summary.set(f"连续试采 · 已核验 0/{len(self._batch_rows)} 个阶段")
        self._set_batch_note("正在准备连续试采…")
        self._controls()

    def update_batch(self, number, state, saved=None):
        if not self.batch_active:
            return
        self._batch_rows[number] = state
        status = BATCH_STAGE[state]
        checked = sum(value in ("completed", "review") for value in self._batch_rows.values())
        title = f"连续试采 · 已核验 {checked}/{len(self._batch_rows)} 个阶段"
        if self.summary.get() != title:
            self.summary.set(title)
        self.table.set(str(number), "status", status)
        self.table.selection_set(str(number))
        self.table.see(str(number))
        if not self._batch_stopping:
            order = tuple(self._batch_rows).index(number) + 1
            count = f" · 已存 {saved} 张" if saved is not None else ""
            self._set_batch_note(f"第 {order}/{len(self._batch_rows)} 项 · 阶段 {number} · {status}{count}")

    def stopping_batch(self):
        self._batch_stopping = True
        self._set_batch_note("正在停止连续试采。请等已收到的图像保存和核验。")
        self._controls()

    def update_batch_live(self, live):
        if self.batch_active and not self._batch_stopping:
            self._batch_live = live
            self._controls()

    def _request_batch_step(self):
        live = self._batch_live
        if not self.batch_active or self._batch_stopping or live is None:
            return
        self.batch_step_button.state(["disabled"])
        try:
            live.request()
        except CaptureError as exc:
            self.update_batch_live(live.refreshed())
            self.detail.set(ui_error(exc, details=False))
        else:
            self.update_batch_live(live.refreshed())

    def finish_batch(self, result, text):
        self.batch_active = self._batch_stopping = False
        self._batch_live = None
        if result.preflight is not None:
            self._batch_check = (result.phase, result.preflight)
            self.batch_check_button.pack(side=tk.LEFT, padx=(8, 0))
        self._set_batch_note(text)
        self.refresh(follow_recommendation=result.state == "completed")

    def _dismiss(self):
        if self.batch_active:
            self.top.withdraw()
        else:
            self.top.destroy()

    def _start(self):
        if (row := self._selection()) and not self._blocked:
            self.top.withdraw()
            try:
                self._start_trial(row.number)
            finally:
                if self.top.winfo_exists():
                    self.top.deiconify()

    def capture_started(self):
        self._advance = True
        self._set_batch_note("")
        self.batch_check_button.pack_forget()
        self.top.withdraw()

    def _view_report(self):
        if (row := self._selection()) and row.record and not self._blocked:
            self._report(row.record)

    def _check(self):
        if not self._blocked:
            self.top.withdraw()
            self._check_plan()

    def _destroy(self, event):
        if event.widget is self.top and self._pending is not None:
            self.top.after_cancel(self._pending)
