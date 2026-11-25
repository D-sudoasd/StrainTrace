"""Windows desktop review of an external DAQ pulse audit."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

from ezdic_gui.controls import localized_combobox
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .config import CaptureError
from .detector_link import write_detector_links
from .detector_link_gui import DetectorLinkDialog
from .presentation import ui_error
from .pulse_audit import audit_external_pulses, pulse_csv_columns, pulse_integer, write_pulse_audit
from .pulse_binding_gui import PulseBindingDialog
from .pulse_timing import add_exposure_audit
from .pulse_timing_gui import ExposureTimingDialog, microseconds
from .trigger_review import TriggerReview, counter_review_snapshot
from .viewport import repaint_after_resize


_LABELS = {"matched": "关联齐全", "missed": "全部相机缺帧", "partial": "部分相机缺帧",
           "conflict": "标识冲突", "capture_only": "仅台账有记录"}
_CONFLICTS = {"duplicate_daq_rows": "DAQ 编号重复", "duplicate_capture_events": "台账有多个触发组",
              "duplicate_camera_frames": "同一相机有多帧", "unexpected_cameras": "台账记录了额外相机",
              "invalid_camera_counters": "帧记录或计数器与锚点不一致"}
_VIEWS = {"differences": "脉冲差异", "all": "全部脉冲", "unassociated": "未关联图像", "events": "异常台账事件",
          "timing": "曝光时序", "gaps": "DAQ 编号缺口", "detector_differences": "探测器关联差异", "detector": "全部探测器关联"}
_DETECTOR_VIEWS = ("detector_differences", "detector")
_DETECTOR_LABELS = {"identity_match": "关联齐全", "partial_capture": "部分相机缺帧", "missing_cameras": "全部相机缺帧",
                    "missing_detector": "探测器无记录", "missing_both": "两端均无记录", "daq_unlisted": "DAQ 无记录", "conflict": "标识冲突"}
_DETECTOR_CONFLICTS = {"duplicate_detector_pulses": "一个脉冲有多条探测器记录", "reused_detector_frames": "探测器帧标识重复"}
_DETECTOR_COLUMN_HELP = {"#1": "由 DAQ 和探测器分别记录的共享脉冲编号。",
                         "#2": "关联齐全需要一条 DAQ 记录、一条探测器记录，以及每台相机各一张对应原图。",
                         "#3": "本脉冲对应的相机台账帧数。选中记录后可打开原图复查。",
                         "#4": "显示探测器帧标识和缺帧或冲突原因。选择一行查看原 CSV 行号。"}
_TIMING_LABELS = {"within_limits": "符合限值", "outside_limits": "超出限值", "inconclusive": "无法判定",
                  "missing_edges": "缺少时序信号", "conflicting_edges": "时序信号冲突"}
_UNASSOCIATED = {"no_event": "缺少触发关联", "invalid_event": "触发记录与本阶段不符", "wrong_phase": "帧所属阶段与触发组不符",
                 "unusable_counter": "该帧的计数或台账记录无法用于补配", "outside_config": "原图与本阶段的相机配置不一致"}
_COLUMN_HELP = {"#1": "DAQ 的脉冲编号。查看未关联图像时显示相机名称。",
                "#2": "显示关联齐全、全部缺帧、部分缺帧或标识冲突。DAQ 未列出的台账编号也会单独显示。",
                "#3": "台账记录的帧数。点“查看此触发组”可核对原图是否可读。",
                "#4": "列出缺帧相机和冲突原因。选中脉冲可查看对应的 CSV 行号。"}
_TIMING_COLUMN_HELP = {"#1": "来自 DAQ 脉冲文件的整数编号。",
                       "#2": "测量区间完全位于限值内才判为符合。横跨限值或缺少误差界时无法判定。",
                       "#3": "最早与最晚的相机曝光起始沿之差，单位为微秒。",
                       "#4": "选择脉冲可查看相机延迟、测量区间及缺少的信号。"}


class PulseAuditWindow:
    PAGE_SIZE = 40

    def __init__(self, parent, experiment, phases, camera_names, submit, show_group, *, experiment_id):
        self.experiment = Path(experiment)
        self.names = dict(camera_names)
        self.experiment_id, self.phases = experiment_id, phases
        self._bindings = {}
        self._counter_review = None
        self.submit, self.show_group = submit, show_group
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title(f"核对 DAQ 脉冲 · {self.experiment.name}")
        self.top.geometry("990x655")
        self.top.minsize(755, 530)
        repaint_after_resize(self.top)
        self._busy = False
        self._report = None
        self._detector = None
        self._rows = []
        self._visible = {}
        self._page = 0
        self.source = tk.StringVar(self.top)
        self.source_name = tk.StringVar(self.top)
        self.phase = tk.StringVar(self.top, str(next(iter(phases))))
        self.event_column = tk.StringVar(self.top)
        self.first, self.last = tk.StringVar(self.top), tk.StringVar(self.top)
        self.view = tk.StringVar(self.top, "differences")
        self.summary = tk.StringVar(self.top, "选择 DAQ 记录的脉冲 CSV，再核对所选阶段。")
        self.details = tk.StringVar(self.top)
        self.page_text = tk.StringVar(self.top)
        self._build(phases)
        for variable in (self.phase, self.event_column, self.first, self.last):
            variable.trace_add("write", self._inputs_changed)
        self._inputs_changed()

    def _build(self, phases):
        layout = ttk.Frame(self.top, padding=14)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(3, weight=1)
        form = ttk.Frame(layout)
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="DAQ 脉冲 CSV").grid(row=0, column=0, sticky="w", padx=(0, 10))
        path_entry = ttk.Entry(form, textvariable=self.source_name, state="readonly")
        path_entry.grid(row=0, column=1, columnspan=3, sticky="ew")
        ToolTip(path_entry, lambda _event: self.source.get() or "选择保存实际发出脉冲编号的 CSV。")
        self.choose = ttk.Button(form, text="选择文件…", command=self._choose, style="Small.TButton")
        self.choose.grid(row=0, column=4, padx=(8, 0), sticky="ew")
        ToolTip(self.choose, "选择 DAQ 独立记录的脉冲列表。带 phase 或 source_id 列时按所选阶段和信号源筛选。")
        ttk.Label(form, text="采集阶段").grid(row=1, column=0, sticky="w")
        labels = {str(number): f"阶段 {number} · {item['source_id'] if item else '采后补配'}" for number, item in phases.items()}
        self.phase_menu = localized_combobox(form, self.phase, labels, width=19)
        self.phase_menu.grid(row=1, column=1, sticky="ew", pady=7)
        ChoiceToolTip(self.phase_menu, "选择要核对的硬件触发阶段。",
                      {labels[str(number)]: (f"计数器 {item['counter_feature']} · 首个脉冲锚点 {item['first_event_number']}" if item else
                                             "填写两次已确认脉冲及各台相机对应的原图序号，可在采后补配。")
                       for number, item in phases.items()})
        ttk.Label(form, text="脉冲编号列").grid(row=1, column=2, sticky="w", padx=(12, 8))
        self.event_menu = ttk.Combobox(form, textvariable=self.event_column, state="readonly", width=22)
        self.event_menu.grid(row=1, column=3, columnspan=2, sticky="ew", pady=7)
        ToolTip(self.event_menu, self._column_help, title="选择脉冲编号列")
        ttk.Label(form, text="编号范围").grid(row=2, column=0, sticky="w")
        bounds = ttk.Frame(form)
        bounds.grid(row=2, column=1, columnspan=3, sticky="ew")
        first = ttk.Entry(bounds, textvariable=self.first, width=14)
        last = ttk.Entry(bounds, textvariable=self.last, width=14)
        first.pack(side=tk.LEFT)
        ttk.Label(bounds, text="至").pack(side=tk.LEFT, padx=(12, 8))
        last.pack(side=tk.LEFT)
        for field in (first, last):
            ToolTip(field, "可留空。填写时包含边界编号。确认范围覆盖希望检查的首尾脉冲。")
        self.run = ttk.Button(form, text="开始核对", command=self._audit, style="Record.TButton")
        self.run.grid(row=2, column=4, sticky="ew", padx=(8, 0))
        ToolTip(self.run, self._run_help, title="核对 DAQ 和采集台账")
        self._inputs = (self.choose, self.phase_menu, self.event_menu, first, last)

        banner = ttk.Label(layout, textvariable=self.summary, justify=tk.LEFT, style="CaptureMuted.TLabel")
        banner.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        banner.bind("<Configure>", lambda event: banner.configure(wraplength=event.width - 5))
        ToolTip(banner, self._timing_summary_help, title="核对范围与证据")
        toolbar = ttk.Frame(layout)
        toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(1, weight=1)
        filters, actions = ttk.Frame(toolbar), ttk.Frame(toolbar)
        filters.grid(row=0, column=0, sticky="w")
        actions.grid(row=0, column=1, sticky="e")
        self.view_menu = localized_combobox(filters, self.view, _VIEWS, width=14)
        self.view_menu.pack(side=tk.LEFT)
        self.view_menu.bind("<<ComboboxSelected>>", lambda _event: self._set_view(), add="+")
        ChoiceToolTip(self.view_menu, "核对完成后，可按结果类型筛选。", {
            "脉冲差异": "查看全漏、部分缺帧、标识冲突和只在台账出现的脉冲。",
            "全部脉冲": "同时查看关联齐全和存在差异的脉冲。",
            "未关联图像": "查看当前阶段未能匹配到脉冲编号的图像及其保存路径。",
            "异常台账事件": "查看来源、计数器锚点或相机列表与本阶段记录不一致的事件。",
            "曝光时序": "按脉冲检查 DAQ 实测的相机曝光延迟和相机间偏差。选中脉冲可查看测量区间。",
            "DAQ 编号缺口": "查看核对区间内 CSV 未列出的连续编号。核对 DAQ 记录是否完整。",
            "探测器关联差异": "查看缺帧、编号冲突和 DAQ 未列出的探测器编号。",
            "全部探测器关联": "查看本次关联的每个脉冲，包括完整关联和差异。",
        })
        self.bind_button = ttk.Button(filters, text="采后补配…", command=self._binding_setup, style="Small.TButton")
        ToolTip(self.bind_button, "选择两个独立确认的 DAQ 脉冲，填写每台相机在两个脉冲的原图序号。")
        self.export = ttk.Button(actions, text="导出核对结果…", style="Small.TButton", command=self._export)
        self.export.pack(side=tk.RIGHT)
        ToolTip(self.export, self._export_help)
        self.timing_button = ttk.Button(actions, text="核对曝光时序…", style="Small.TButton", command=self._timing_setup)
        self.timing_button.pack(side=tk.RIGHT, padx=(0, 8))
        ToolTip(self.timing_button, lambda _event: "先点“开始核对”载入本阶段的 DAQ 脉冲编号。" if self._report is None else
                "导入同一台 DAQ 记录的触发和曝光起始沿。可按实验限值检查各台相机的触发延迟或相机间偏差。")
        self.detector_button = ttk.Button(actions, text="关联探测器帧…", style="Small.TButton", command=self._detector_setup)
        self.detector_button.pack(side=tk.RIGHT, padx=(0, 8))
        ToolTip(self.detector_button, lambda _event: "先选择 DAQ 文件并点“开始核对”。" if self._report is None else
                "选择记录共享脉冲编号和探测器帧标识的 CSV。结果会指出哪一端缺帧以及哪些编号有冲突。", title="关联探测器和相机帧")

        def fit_toolbar(_event):
            wrap = toolbar.winfo_width() < filters.winfo_reqwidth() + actions.winfo_reqwidth() + 8
            if wrap and actions.grid_info()["row"] == 0:
                actions.grid_configure(row=1, column=0, columnspan=2, pady=(4, 0))
            elif not wrap and actions.grid_info()["row"] == 1:
                actions.grid_configure(row=0, column=1, columnspan=1, pady=0)

        for widget in (toolbar, filters, actions):
            widget.bind("<Configure>", fit_toolbar, add="+")
        table = ttk.Frame(layout)
        table.grid(row=3, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(table, columns=("record", "status", "frames", "description"), show="headings", selectmode="browse")
        self.table.grid(row=0, column=0, sticky="nsew")
        font = tkfont.Font(self.top, font=ttk.Style(self.top).lookup("Treeview", "font"))
        record_width = max(187, font.measure("9" * 20) + 12)
        for key, title, width in (("record", "脉冲编号或相机", record_width), ("status", "状态", 138),
                                  ("frames", "台账帧数", 80), ("description", "详情", 360)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, stretch=key == "description", anchor="center" if key == "frames" else "w")
        scroll = ttk.Scrollbar(table, orient=tk.VERTICAL, command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        ToolTip(scroll, "滚动查看本页核对结果。用下方按钮可翻页。", title="脉冲核对结果")
        self.table.bind("<<TreeviewSelect>>", self._selected)
        ToolTip(self.table, self._table_help)
        caption = ttk.Label(layout, textvariable=self.details, justify=tk.LEFT, style="CaptureMuted.TLabel")
        caption.grid(row=4, column=0, sticky="ew", pady=(8, 5))
        caption.bind("<Configure>", lambda event: caption.configure(wraplength=event.width - 5))
        footer = ttk.Frame(layout)
        footer.grid(row=5, column=0, sticky="ew")
        self.previous = ttk.Button(footer, text="上一页", command=lambda: self._move(-1), style="Small.TButton")
        self.next = ttk.Button(footer, text="下一页", command=lambda: self._move(1), style="Small.TButton")
        self.previous.pack(side=tk.LEFT)
        self.next.pack(side=tk.LEFT, padx=5)
        ToolTip(self.previous, lambda _event: "当前没有上一页。" if self.previous.instate(["disabled"]) else
                "返回同一筛选条件下的上一页。")
        ToolTip(self.next, lambda _event: "当前没有下一页。" if self.next.instate(["disabled"]) else
                "查看同一筛选条件下的下一页。")
        ttk.Label(footer, textvariable=self.page_text).pack(side=tk.LEFT, padx=5)
        self.inspect = ttk.Button(footer, text="查看此触发组…", command=self._inspect, style="Small.TButton")
        self.inspect.pack(side=tk.RIGHT)
        ToolTip(self.inspect, self._inspect_help, title="查看此脉冲对应的原图")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())

    def _column_help(self, _event):
        if not self.source.get():
            return "先点“选择文件…”，再从 CSV 的表头中选择脉冲编号列。"
        return "选择记录实际发出脉冲编号的列，常用列名为 event_number。编号用非负整数。"

    def _run_help(self, event):
        if self._busy:
            return "本窗口正在处理请求，完成后可继续核对。"
        if self._needs_binding() and int(self.phase.get()) not in self._bindings:
            return "请用“采后补配”填写两次已确认脉冲及各台相机对应的原图。"
        if not self.source.get() or not self.event_column.get():
            return self._column_help(event)
        return "按编号对照 CSV 和台账，找出全部漏拍、部分缺帧和编号冲突。编号范围可留空。"

    def _table_help(self, event):
        view = self.view.get()
        if view == "gaps":
            return "显示 DAQ 文件在核对区间内未列出的编号。选中一行查看起止编号。"
        if event is not None and getattr(event, "keysym", "??") in ("??", ""):
            region = self.table.identify_region(event.x, event.y)
            if region == "heading":
                headings = _DETECTOR_COLUMN_HELP if view in _DETECTOR_VIEWS else _TIMING_COLUMN_HELP if view == "timing" else _COLUMN_HELP
                return headings.get(self.table.identify_column(event.x), "")
            if region == "cell":
                row = self._visible.get(self.table.identify_row(event.y))
                if row and view in _DETECTOR_VIEWS:
                    return f"脉冲 {row['event_number']}\n{self._detector_details(row)}"
                if row and view == "timing":
                    return self._timing_details(row)
                if row and view in ("differences", "all") and (issues := self._pulse_issues(row)):
                    return f"脉冲 {row['event_number']} · {_LABELS[row['status']]}\n{issues}"
        if view == "timing":
            return "选择一行查看该脉冲的曝光时序。测量区间按输入文件提供的误差界计算。"
        if view in _DETECTOR_VIEWS:
            row = self._selected_row()
            return f"脉冲 {row['event_number']}\n{self._detector_details(row)}" if row else "选择一行查看探测器帧标识和来源行号。"
        return "选择一行可查看 DAQ 原行号、缺帧相机或未关联原因。"

    def _timing_summary_help(self, _event):
        if self._report is None:
            return ""
        lines = []
        if association := self._report.get("association"):
            binding = association["binding"]
            lines.extend((f"计数器 {binding['counter_feature']} · 确认脉冲 {binding['first_event_number']} 至 {binding['last_event_number']}",
                          f"确认依据 {binding['evidence']}", "起止锚点外的原图保持未核对，可在完整报告查看逐台张数。"))
        if self._detector:
            evidence = self._detector["detector_input"]
            lines.append(f"探测器文件 {Path(evidence['path']).name} · 筛除 {sum(evidence['excluded_rows'].values())} 行")
            counts = self._detector["summary"]
            lines.append(" · ".join(f"{label} {counts[key]}" for key, label in _DETECTOR_LABELS.items() if counts[key]))
        if (timing := self._report.get("exposure_timing")) is None:
            return "\n".join(lines)
        limits, extras = timing["limits_ns"], timing["unmatched"]
        skew = limits["max_skew_ns"]
        low, high = limits["latency_min_ns"], limits["latency_max_ns"]
        lines.append(f"时序文件 {Path(timing['input']['path']).name} · 时基 {timing['input']['clock_id'] or '无相关记录'}")
        if skew is not None:
            lines.append(f"相机间起始偏差上限 {microseconds(skew)} µs")
        if low is not None:
            lines.append(f"触发至曝光延迟范围 {microseconds(low)} 至 {microseconds(high)} µs")
        if extras:
            numbers = "、".join(str(row["event_number"]) for row in extras[:10])
            lines.append(f"时序文件中的额外编号 {numbers}，共 {len(extras)} 个")
        return "\n".join(lines)

    def _inspect_help(self, _event):
        if self.view.get() in _DETECTOR_VIEWS and self._selected_row() and not self._capture_row():
            return "相机台账没有对应脉冲的记录。可以在探测器 CSV 查到所选行。"
        if not (row := self._capture_row()) or self.view.get() in ("unassociated", "events", "gaps"):
            return "先在脉冲列表中选择一条记录。"
        if self._report and "association" in self._report:
            return ("校验原图后并排打开。重复计数的相机会列出候选序号。" if row.get("frames") else
                    "此脉冲在锚点之间没有对应的已保存图像。")
        identities = row["capture_event_ids"]
        if not identities:
            return "台账没有登记此脉冲的触发组。"
        if len(identities) > 1:
            return "台账有多个触发组对应此脉冲。打开完整报告可查看各组标识。"
        return "打开此触发组并校验已保存的原图。窗口会标出缺帧相机。"

    def _alive(self):
        return self.top.winfo_exists()

    def _set_busy(self, busy):
        self._busy = busy
        self._controls()

    def _choose(self):
        path = filedialog.askopenfilename(parent=self.top, filetypes=[("DAQ 脉冲 CSV", "*.csv"), ("所有文件", "*.*")])
        if not path:
            return
        self._set_busy(True)

        def loaded(fields):
            if not self._alive():
                return
            fields = tuple(name for name in fields if name not in ("source_id", "phase"))
            self.source.set(path)
            self.source_name.set(Path(path).name)
            self.event_menu.configure(values=fields)
            self._set_busy(False)
            self.event_column.set("event_number" if "event_number" in fields else fields[0] if len(fields) == 1 else "")
            self._inputs_changed()

        self.submit(lambda: pulse_csv_columns(path), loaded, self._failed)

    def _inputs_changed(self, *_args):
        if self._busy:
            return
        self._report = None
        self._detector = None
        if self._counter_review and self._counter_review.top.winfo_exists():
            self._counter_review.top.destroy()
        self._counter_review = None
        binding = self._bindings.get(int(self.phase.get()))
        if self._needs_binding() and binding is None:
            self.summary.set("此阶段可用两次已确认的脉冲建立采后关联。请打开“采后补配”。")
        elif binding is not None:
            self.summary.set(f"采后补配 · 已选择脉冲 {binding.first_event_number} 至 {binding.last_event_number}。选择 DAQ 文件后开始核对。")
        else:
            self.summary.set("设置就绪，点击“开始核对”。" if self.source.get() and self.event_column.get() else
                             "选择 DAQ 脉冲文件及保存脉冲编号的列。")
        self._set_view()

    def _needs_binding(self):
        return self.phases[int(self.phase.get())] is None

    def _binding_setup(self):
        phase = int(self.phase.get())

        def accept(binding):
            self._bindings[phase] = binding
            self._inputs_changed()

        PulseBindingDialog(self.top, self.experiment, self.experiment_id, phase, self.names, self.submit, accept,
                           previous=self._bindings.get(phase))

    def _bound(self, variable):
        if not (text := variable.get().strip()):
            return None
        value = pulse_integer(text)
        if value is None:
            raise CaptureError("PULSE_AUDIT_RANGE", "请填写非负整数编号，或留空使用文件中的全部相关脉冲。")
        return value

    def _audit(self):
        try:
            first, last = self._bound(self.first), self._bound(self.last)
        except CaptureError as exc:
            self._failed(exc)
            return
        path, phase, column = self.source.get(), int(self.phase.get()), self.event_column.get()
        binding = self._bindings.get(phase)
        self._inputs_changed()
        self._set_busy(True)
        self.summary.set("正在核对 DAQ 和台账的脉冲编号…")
        self.submit(lambda: audit_external_pulses(self.experiment, path, phase=phase, event_column=column,
                                                 first_event=first, last_event=last, binding=binding), self._display, self._failed)

    def _display(self, report, *, focus=None):
        if not self._alive():
            return
        self._report = report
        counts = report["summary"]
        if focus is None:
            focus = ("timing" if "exposure_timing" in report else "differences" if counts["matched"] < len(report["pulses"]) else
                     "unassociated" if counts["unassociated_frames"] else "events" if counts["invalid_events"] else
                     "gaps" if counts.get("daq_unlisted_pulses") else "all")
        self.view.set(focus)
        self._busy = False
        self._set_view()

    def _show_summary(self, report, view):
        if view in _DETECTOR_VIEWS:
            counts = self._detector["summary"]
            matched = counts["identity_match"]
            review = len(self._detector["links"]) - matched
            text = (f"阶段 {report['phase']} · {report['source_id']} · 关联齐全 {matched} · "
                    f"需复查 {review} · 逐相机关联 {counts['camera_frame_links']} 条")
            if gaps := self._detector["capture_summary"].get("daq_unlisted_pulses"):
                text += f" · DAQ 编号缺口 {gaps}"
            self.summary.set(text)
            return
        counts = report["summary"]
        stats = (f"DAQ 脉冲 {counts['expected_pulses']} · 关联齐全 {counts['matched']} · 全部缺帧 {counts['missed']} · "
                 f"部分缺帧 {counts['partial']} · 冲突 {counts['conflict']} · DAQ 无记录 {counts['not_in_daq']}")
        outside = sum(report["input"]["excluded_rows"].values())
        each = (f"阶段 {report['phase']} · {report['source_id']} · 未关联图像 {counts['unassociated_frames']} · "
                f"异常台账事件 {counts['invalid_events']} · 筛除 CSV 行 {outside} · 范围外台账脉冲 {counts['capture_pulses_outside_range']}")
        if "association" in report:
            each += (f"\n采后补配 · 核对 {report['first_event']} 至 {report['last_event']} · "
                     f"锚点外原图 {counts['outside_anchor_frames']} 张 · DAQ 编号缺口 {counts['daq_unlisted_pulses']}")
        if timing := report.get("exposure_timing"):
            summary = timing["summary"]
            stats_timing = " · ".join(f"{label} {summary[key]}" for key, label in _TIMING_LABELS.items())
            each += f"\n曝光时序 · {stats_timing} · 额外时序编号 {summary['unmatched_timing_pulses']}"
        self.summary.set(f"{stats}\n{each}")

    def _set_view(self):
        ToolTip.dismiss_window(self.top)
        view, report = self.view.get(), self._report
        has_timing = report is not None and "exposure_timing" in report
        gaps = (report or {}).get("association", {}).get("daq_unlisted_ranges", ())
        available = {key: label for key, label in _VIEWS.items() if key not in ("timing", "gaps", *_DETECTOR_VIEWS)}
        if has_timing:
            available["timing"] = _VIEWS["timing"]
        if gaps:
            available["gaps"] = _VIEWS["gaps"]
        if self._detector:
            available.update((key, _VIEWS[key]) for key in _DETECTOR_VIEWS)
        self.view_menu.configure(values=tuple(available.values()))
        if view not in available:
            view = "differences"
            self.view.set(view)
        if report:
            self._show_summary(report, view)
        self.table.heading("frames", text={"timing": "起始偏差（µs）", "gaps": "缺少编号数"}.get(view, "台账帧数"))
        self.table.column("frames", width=130 if view in ("timing", "gaps") else 80)
        self.table.heading("status", text="时序判断" if view == "timing" else "状态")
        self.table.heading("description", text="探测器关联详情" if view in _DETECTOR_VIEWS else "详情")
        self.export.configure(text="导出探测器关联…" if view in _DETECTOR_VIEWS else "导出核对结果…")
        if not report:
            self._rows = []
        elif view == "unassociated":
            self._rows = report["unassociated_frames"]
        elif view == "events":
            self._rows = report["invalid_events"]
        elif view == "timing":
            self._rows = [row for row in report["pulses"] if "exposure_timing" in row]
        elif view == "gaps":
            self._rows = gaps
        elif view in _DETECTOR_VIEWS:
            links = self._detector["links"]
            self._rows = links if view == "detector" else [row for row in links if row["status"] != "identity_match"]
        else:
            self._rows = report["pulses"] if view == "all" else [row for row in report["pulses"] if row["status"] != "matched"]
        self._page = 0
        self._fill_page()

    def _move(self, offset):
        pages = (len(self._rows) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        if 0 <= self._page + offset < pages:
            self._page += offset
            self._fill_page()

    def _pulse_issues(self, row):
        missing = "、".join(self.names.get(camera, camera) for camera in row["missing_cameras"])
        messages = [_CONFLICTS[code] for code in row["conflicts"]]
        if row["conflicts"] and not row["daq_lines"]:
            messages.insert(0, "DAQ 未记录此编号")
        if missing:
            messages.insert(0, f"缺帧 {missing}")
        return " · ".join(messages)

    def _timing_issues(self, row):
        timing = row["exposure_timing"]
        issues = []
        if timing["trigger_count"] != 1:
            issues.append("缺少触发沿" if timing["trigger_count"] == 0 else "触发沿重复")
        for key, label in (("missing_cameras", "缺少曝光沿"), ("duplicate_cameras", "曝光沿重复"), ("unexpected_cameras", "额外相机")):
            if cameras := timing[key]:
                issues.append(f"{label} {'、'.join(self.names.get(camera, camera) for camera in cameras)}")
        return " · ".join(issues)

    def _timing_details(self, row):
        timing = row["exposure_timing"]
        lines = [f"脉冲 {row['event_number']} · {_TIMING_LABELS[timing['status']]} · 台账图像 {_LABELS[row['status']]}"]
        if issues := self._timing_issues(row):
            lines.append(issues)
        if timing["skew_ns"] is not None:
            lines.append(f"相机间起始偏差 {self._timing_metric(timing['skew_ns'], timing['skew_bounds_ns'])} µs")
        delays = [f"{self.names.get(camera, camera)} {self._timing_metric(item['latency_ns'], item['latency_bounds_ns'])}"
                  for camera, item in timing["cameras"].items() if item["latency_ns"] is not None]
        if delays:
            lines.append("延迟（µs） " + " · ".join(delays[:6]) + (f" · 另有 {len(delays) - 6} 台见完整报告" if len(delays) > 6 else ""))
        return "\n".join(lines)

    def _detector_issues(self, row):
        issues = [_DETECTOR_CONFLICTS[code] for code in row["conflicts"] if code in _DETECTOR_CONFLICTS]
        if (capture := row["capture"]) and (capture_issues := self._pulse_issues(capture)):
            issues.append(capture_issues)
        return " · ".join(issues)

    @staticmethod
    def _detector_description(row):
        records = row["detector_rows"]
        if not records:
            return "无探测器帧记录"
        return records[0]["frame_id"] if len(records) == 1 else f"{len(records)} 条探测器记录"

    def _detector_details(self, row):
        def brief(values):
            if not values:
                return "无"
            visible = "、".join(map(str, values[:6]))
            return visible + (f" 等 {len(values)} 项" if len(values) > 6 else "")

        records = row["detector_rows"]
        daq = f"DAQ CSV 第 {brief(row['daq_lines'])} 行" if row["daq_lines"] else "DAQ 未记录此编号"
        detector = (f"探测器 CSV 第 {brief([item['line'] for item in records])} 行 · 探测器帧 {brief([item['frame_id'] for item in records])}"
                    if records else "探测器 CSV 无记录")
        issues = self._detector_issues(row)
        detail = f"{daq} · {detector}"
        return f"{detail}\n{issues}" if issues else detail

    @staticmethod
    def _timing_metric(value, bounds):
        interval = f"[{microseconds(bounds[0])}, {microseconds(bounds[1])}]" if bounds else "误差界缺失"
        return f"{microseconds(value)}（{interval}）"

    def _fill_page(self):
        self.table.delete(*self.table.get_children())
        self._visible.clear()
        start = self._page * self.PAGE_SIZE
        for index, row in enumerate(self._rows[start:start + self.PAGE_SIZE], start):
            key = str(index)
            self._visible[key] = row
            if self.view.get() == "unassociated":
                cells = (self.names.get(row["camera_id"], row["camera_id"]), "未关联", 1, _UNASSOCIATED[row["reason"]])
            elif self.view.get() == "events":
                cells = (row["event_number"], "台账异常", "", row["event_id"])
            elif self.view.get() == "timing":
                timing = row["exposure_timing"]
                cells = (row["event_number"], _TIMING_LABELS[timing["status"]], microseconds(timing["skew_ns"]),
                         self._timing_issues(row) or f"台账图像 {_LABELS[row['status']]} · 选中查看逐台延迟")
            elif self.view.get() == "gaps":
                cells = (row["first_event"], "DAQ 未列出", row["last_event"] - row["first_event"] + 1, f"至 {row['last_event']}")
            elif self.view.get() in _DETECTOR_VIEWS:
                description = self._detector_description(row)
                if issues := self._detector_issues(row):
                    description += f" · {issues}"
                cells = (row["event_number"], _DETECTOR_LABELS[row["status"]], len((row["capture"] or {}).get("frames", ())), description)
            else:
                cells = (row["event_number"], _LABELS[row["status"]], len(row["frames"]), self._pulse_issues(row))
            self.table.insert("", tk.END, iid=key, values=cells)
        pages = (len(self._rows) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        self.page_text.set(f"第 {self._page + 1}/{pages} 页 · {len(self._rows)} 条" if pages else "当前筛选没有记录")
        if self._visible:
            self.table.selection_set(next(iter(self._visible)))
        self._selected()

    def _selected_row(self):
        selected = self.table.selection()
        return self._visible.get(selected[0]) if selected else None

    def _capture_row(self):
        row = self._selected_row()
        return row["capture"] if row and self.view.get() in _DETECTOR_VIEWS else row

    def _selected(self, _event=None):
        row = self._selected_row()
        detail = ""
        if row:
            if self.view.get() == "unassociated":
                detail = f"序号 {row['sequence']} · 计数器 {row['counter'] if row['counter'] is not None else '缺失'}\n{row['path']}"
            elif self.view.get() == "events":
                detail = f"台账事件 {row['event_id']} · 记录信号源 {row['source_id']}"
            elif self.view.get() == "timing":
                detail = self._timing_details(row)
            elif self.view.get() == "gaps":
                detail = f"DAQ CSV 未列出 {row['first_event']} 至 {row['last_event']}。请核对信号源和记录范围。"
            elif self.view.get() in _DETECTOR_VIEWS:
                detail = self._detector_details(row)
            else:
                lines = "、".join(map(str, row["daq_lines"][:12]))
                if len(row["daq_lines"]) > 12:
                    lines += f" 等 {len(row['daq_lines'])} 行"
                origin = (f"采后补配原图 {len(row['frames'])} 张" if "association" in self._report else
                          f"台账触发组 {len(row['capture_event_ids'])} 个")
                detail = f"DAQ CSV 行号 {lines or '无记录'} · {origin}"
                if issues := self._pulse_issues(row):
                    detail += f"\n{issues}"
        self.details.set(detail)
        self._controls()

    def _inspect(self):
        if self._inspectable(row := self._capture_row()):
            note = ""
            if timing := row.get("exposure_timing"):
                source = Path(self._report["exposure_timing"]["input"]["path"]).name
                note = f"DAQ 时序（{source}） · {_TIMING_LABELS[timing['status']]}"
                if timing["skew_ns"] is not None:
                    note += f" · 起始偏差 {self._timing_metric(timing['skew_ns'], timing['skew_bounds_ns'])} µs"
            if self.view.get() in _DETECTOR_VIEWS:
                link = self._selected_row()
                note = f"探测器关联 · {_DETECTOR_LABELS[link['status']]} · {self._detector_description(link)}"
            if "association" in self._report:
                self._show_counter_group(row, note)
            else:
                self.show_group(row["capture_event_ids"][0], note)

    def _inspectable(self, row):
        return bool(row and (len(row.get("capture_event_ids", ())) == 1 or
                             (self._report is not None and "association" in self._report and row.get("frames"))))

    def _show_counter_group(self, row, note):
        report = self._report
        status = "ambiguous" if row["status"] == "conflict" else "incomplete" if row["missing_cameras"] else "associated"
        group = counter_review_snapshot(report["experiment_id"], report["phase"], report["source_id"], row["event_number"],
                                        report["pairing"]["first_counter"], row["frames"], report["pairing"]["counter_feature"], status=status)
        if self._counter_review and self._counter_review.top.winfo_exists():
            self._counter_review.top.destroy()
        self._counter_review = TriggerReview(self.top, self.experiment, self.names, self.submit, snapshots=(group,), timing_note=note)

    def _timing_setup(self):
        if self._report is not None:
            cameras = tuple(self._report["pairing"]["first_counter"])
            ExposureTimingDialog(self.top, cameras, self._audit_timing, previous=self._report.get("exposure_timing"))

    def _audit_timing(self, source, limits):
        if self._report is not None:
            report = self._report
            self._set_busy(True)
            self.submit(lambda: add_exposure_audit(report, source, limits=limits), self._display, self._failed)

    def _detector_setup(self):
        if self._report is not None:
            DetectorLinkDialog(self.top, self.experiment, self._report, self.submit, self._display_detector,
                               previous=self._detector["detector_input"] if self._detector else None)

    def _display_detector(self, report):
        if self._alive() and self._report is not None:
            self._detector = report
            view = "detector" if all(row["status"] == "identity_match" for row in report["links"]) else "detector_differences"
            self._display(self._report, focus=view)

    def _export_help(self, _event):
        if self._report is None:
            return "先核对 DAQ 文件，再导出结果。"
        if self.view.get() in _DETECTOR_VIEWS:
            return "JSON 含缺帧、冲突和原始行号。CSV 可供后续分析，每行对应一个能唯一关联到探测器帧的相机帧。"
        return "JSON 保留核对依据、来源校验值和行号。已核对的曝光时序会同时写入逐脉冲 CSV。"

    def _export(self):
        if self.view.get() in _DETECTOR_VIEWS:
            report, writer, filename = self._detector, write_detector_links, "detector-link"
            filetypes = [("完整探测器关联报告", "*.json"), ("逐相机关联表", "*.csv")]
        else:
            report, writer, filename = self._report, write_pulse_audit, "pulse-audit"
            filetypes = [("完整核对报告", "*.json"), ("脉冲列表", "*.csv")]
        if report is None:
            return
        path = filedialog.asksaveasfilename(parent=self.top, defaultextension=".json", initialfile=filename, filetypes=filetypes)
        if path:
            self._set_busy(True)

            def saved(result):
                if self._alive():
                    self._set_busy(False)
                    self.details.set(f"已导出到 {result}")

            self.submit(lambda: writer(report, path), saved, self._failed)

    def _failed(self, exc):
        if self._alive():
            self._set_busy(False)
            if self._report is None:
                self.summary.set(ui_error(exc, details=False))
            messagebox.showerror("无法完成脉冲核对", ui_error(exc), parent=self.top)

    def _controls(self):
        def enable(widget, value):
            widget.state(["!disabled" if value and not self._busy else "disabled"])

        for control in self._inputs:
            enable(control, True)
        needs_binding = self._needs_binding()
        if needs_binding and not self.bind_button.winfo_manager():
            self.bind_button.pack(side=tk.LEFT, padx=(8, 0), after=self.view_menu)
        elif not needs_binding:
            self.bind_button.pack_forget()
        enable(self.bind_button, needs_binding)
        enable(self.run, self.source.get() and self.event_column.get() and (not needs_binding or int(self.phase.get()) in self._bindings))
        enable(self.view_menu, self._report is not None)
        enable(self.timing_button, self._report is not None)
        enable(self.detector_button, self._report is not None)
        enable(self.export, self._report is not None)
        enable(self.previous, self._page > 0)
        enable(self.next, (self._page + 1) * self.PAGE_SIZE < len(self._rows))
        enable(self.inspect, self._inspectable(self._capture_row()))
