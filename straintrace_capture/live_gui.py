"""Analysis desktop mode for following a Capture ledger."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from time import monotonic
from tkinter import filedialog, messagebox, ttk

from ezdic_gui.controls import ScrollableForm, localized_combobox, wrap_label_on_resize
from ezdic_gui.theme import create_tk_root
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from .cameras.base import Image
from .config import CaptureError
from .continuation_analysis import offline_from_live_reference, read_live_reference
from .gui import ACCENT, BG, set_theme
from .handoff import launch_analysis, launch_capture, launch_corrected_analysis
from .image_inputs import (
    analysis_part_options, frame_part_selector, read_analysis_image, select_frame_image, validate_analysis_part, verified_image_path,
)
from .live import LiveAnalysis, read_linked_reference_image
from .live_curve import CURVE_MODES, CurvePoint, available_curve_modes
from .live_curve_gui import CURVE_HELP, CurveFrameWindow, LiveCurve
from .live_results import LiveResultLog, export_live_csv, live_log_load_source, live_results_directory
from .mechanics_gui import MechanicsDialog
from .presentation import LOAD_SUPPORT_LABELS, LOAD_TIME_BASIS_LABELS, STATE_LABELS, TRACKING_LABELS, ui_error
from .storage import TERMINAL_STATES, ExperimentReader
from .viewport import ImageViewport, repaint_after_resize

ROI_LABELS = {"Gauge ROI 1": "标距端点 1", "Gauge ROI 2": "标距端点 2", "2D field ROI": "全场区域"}
AXIS_LABELS = {"auto": "自动判断", "x": "水平投影", "y": "竖直投影", "distance": "直线距离"}
AXIS_HELP = {
    "自动判断": "按两个端点的相对位置选择标距方向。已知加载方向时可以直接指定。",
    "水平投影": "用两个端点水平间距的变化计算应变。适合试样水平放置。",
    "竖直投影": "用两个端点竖直间距的变化计算应变。适合试样竖直放置。",
    "直线距离": "用两个端点之间的直线长度变化计算应变。适合斜向放置的标距。",
}
ROI_HELP = {
    "Gauge ROI 1": "框住第一个标距端点的清晰散斑。两个端点是用来计算长度变化的两块小区域。",
    "Gauge ROI 2": "框住另一个标距端点。两个端点要有间距，并沿所选测量方向布置。",
    "2D field ROI": "框住要计算二维位移和应变分布的区域。全场分析可单独使用。",
}
DRAW_HELP = "按住 Shift 拖出选定的测量区。滚轮缩放，直接拖动平移。按 0 恢复适应窗口。"
FOLLOW_MODES = {"latest": "优先看最新", "ordered": "按序逐帧"}
FOLLOW_HELP = {
    "优先看最新": "有多张待分析图像时，先计算最新的图像。中间省略的帧计入“实时跳过”。",
    "按序逐帧": "从当前进度接着按台账登记顺序分析。要补算此前跳过的图像，可在历史曲线中选“按序重算”。",
}
_POLL_IDLE_S = .15
_STATUS_S = .5
_PAINT_S = .10
FIELD_REASON_LABELS = {
    "correlation_valid_fraction_below_threshold": "通过相关性检查的点太少",
    "strain_valid_fraction_below_threshold": "能计算应变的点太少",
    "ambiguous_peaks_accepted": "部分点存在多个可能的匹配位置",
    "experimental_search_unvalidated": "当前搜索设置尚未通过验证",
}


def _rectangle(text):
    if not text.strip():
        return None
    try:
        rect = tuple(int(value) for value in text.replace("，", ",").split(","))
    except ValueError:
        rect = ()
    if len(rect) != 4 or min(rect[:2]) < 0 or min(rect[2:]) < 1:
        raise ValueError("请输入 x、y、宽、高四个整数。原点坐标应大于等于零，宽和高应大于零。")
    return rect


class LiveWorkbench:
    def __init__(self, root, experiment, camera=None, reference_sequence=None, *, image_part=None, reference_binding=None):
        self.root, self.experiment = root, Path(experiment).resolve()
        self.reference_binding = reference_binding
        self._bound = self._bound_digest = None
        if reference_binding is not None:
            self._bound, self._bound_digest = read_live_reference(reference_binding)
            if Path(self._bound["experiment_root"]) != self.experiment:
                raise CaptureError("CAPTURE_BINDING_INVALID", "参考图绑定对应另一项采集实验。")
            camera, image_part = self._bound["camera_id"], self._bound["image_part"]
            reference_sequence = self._bound["external_reference"]["sequence"]
        root.title("StrainTrace Analysis · 实时分析")
        root.geometry("1160x760")
        root.minsize(860, 620)
        set_theme(root)
        repaint_after_resize(root)
        self.camera = tk.StringVar(value=camera or "")
        validate_analysis_part(image_part)
        self.image_part = image_part
        self._part_camera = camera
        self._part_choices = {}
        self.reference = tk.StringVar(value=str(reference_sequence) if reference_sequence else "")
        self.rois = {key: tk.StringVar() for key in ROI_LABELS}
        self.strain_mode = tk.StringVar(value="auto")
        self.debayer = tk.BooleanVar()
        self.draw_target = tk.StringVar(value="Gauge ROI 1")
        self.state = tk.StringVar(value="等待已保存图像")
        self.result = tk.StringVar(value="在已保存的参考图上画两个标距端点，或画一个全场分析区域。")
        self.progress = tk.StringVar(value="")
        self.record_status = tk.StringVar(value="跟踪开始后自动保存")
        self.follow_mode = tk.StringVar(value="latest")
        self.curve_mode = tk.StringVar(value="strain-time")
        self._curve_modes = ()
        self._frame_window = None
        self._paused = False
        self._working = False
        self._reset_pending = False
        self._issue = None
        self._jobs = queue.SimpleQueue()
        self._tasks = queue.SimpleQueue()
        self._worker = None
        self._offline_jobs = queue.SimpleQueue()
        self._offline_pending = False
        self._recording = None
        self._record_problem = None
        self._last_record = None
        self._export_jobs = queue.SimpleQueue()
        self._export_pending = False
        self._export_fills_load = False
        self._export_result = None
        try:
            directory = live_results_directory(self.experiment)
            self._last_record = max(directory.glob("*.jsonl"), default=None)
        except (CaptureError, OSError) as exc:
            self._record_problem = exc
        self._update_record_status()
        self._analyzer = self._display_key = None
        self._load_alignment = None
        self._reference_request = None
        self._reference_problem = ""
        self._camera_ids = ()
        self._metrics = None
        self._next_poll = self._next_status = self._next_paint = 0.
        self._pending_paint = None
        self._tick_after = None
        self._build()
        for variable in (self.reference, *self.rois.values()):
            variable.trace_add("write", self._controls)
        root.bind("<Escape>", lambda _event: ToolTip.dismiss_window(root))
        root.bind("<Destroy>", self._destroy, add="+")
        self.refresh()
        self._tick_after = root.after(250, self._tick)

    def _build(self):
        layout = ttk.Frame(self.root, padding=14)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(1, weight=1)
        banner = ttk.Frame(layout)
        banner.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        banner.columnconfigure(0, weight=1)
        ttk.Label(banner, text="实时应变分析", style="CaptureTitle.TLabel").grid(row=0, column=0, sticky="w")
        capture = ttk.Button(banner, text="打开采集实验 ↗", command=self.open_capture)
        capture.grid(row=0, column=1, padx=5)
        ToolTip(capture, "在采集工作台中查看原始图像、实验记录和校验结果。")
        self.offline_button = ttk.Button(banner, text="打开离线分析 ↗", command=self.offline)
        self.offline_button.grid(row=0, column=2)
        ToolTip(self.offline_button, "采集结束后，以这张原参考图打开本轮的离线分析。" if self._bound else
                "用所选相机和参考帧打开离线工作台。可设置测量区、逐帧计算并导出数据。")
        path_label = ttk.Label(banner, text=str(self.experiment), style="CaptureMuted.TLabel", width=1)
        path_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 5))
        ToolTip(path_label, str(self.experiment), title="当前实验目录")
        history = ttk.Button(banner, text="历史曲线…", command=self.show_history, style="Small.TButton")
        history.grid(row=1, column=2, sticky="e", pady=(3, 0))
        ToolTip(history, "复查本实验整份已保存的逐帧记录。可按帧号定位并核对原图。")
        dock = tk.PanedWindow(layout, orient=tk.HORIZONTAL, bg=BG, sashwidth=8, borderwidth=0)
        dock.grid(row=1, column=0, sticky="nsew")
        view_panel = ttk.Frame(dock)
        data_panel = ttk.Frame(dock)
        dock.add(view_panel, minsize=420, stretch="always")
        dock.add(data_panel, minsize=355, width=420, stretch="never")
        data_panel.columnconfigure(0, weight=1)
        self.view = ImageViewport(view_panel, "analysis", "已保存的参考图", lambda _camera: None, self._draw_region)
        ToolTip(self.view.canvas, lambda _event: "重置跟踪后可重新画测量区。" if self._analyzer else DRAW_HELP,
                title="图像操作")
        view_actions = ttk.Frame(view_panel)
        view_actions.pack(side=tk.BOTTOM, fill=tk.X, pady=(7, 0))
        self.view.pack(fill=tk.BOTH, expand=True)
        fit = ttk.Button(view_actions, text="适应", style="Small.TButton", command=self.view.fit)
        fit.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(fit, "把完整图像缩放到窗口内。双击图像也可恢复。")
        one_to_one = ttk.Button(view_actions, text="1:1", width=3, style="Small.TButton", command=self.view.one_to_one)
        one_to_one.pack(side=tk.LEFT)
        ToolTip(one_to_one, "按原始像素比例显示，便于检查散斑边缘。拖动可浏览其他位置。")
        draw_label = ttk.Label(view_actions, text="Shift 拖动绘制", style="CaptureMuted.TLabel")
        draw_label.pack(side=tk.LEFT, padx=(10, 6))
        ToolTip(draw_label, DRAW_HELP, title="绘制测量区域")
        self.target = localized_combobox(view_actions, self.draw_target, ROI_LABELS, width=12)
        self.target.pack(side=tk.LEFT)
        ChoiceToolTip(self.target, lambda _event: self._editable_help(
            "选择要绘制的区域。画完第一个标距端点后，程序会自动切到第二个端点。"),
            {label: ROI_HELP[key] for key, label in ROI_LABELS.items()}, title="绘制区域")
        self.settings_form = ScrollableForm(data_panel)
        self.settings_form.grid(row=0, column=0, sticky="nsew")
        ToolTip(self.settings_form.scrollbar, "窗口较小时，可滚动查看相机、测量区及图像转换选项。")
        form = self.settings_form.body
        self.tracking_scope = tk.StringVar()
        self.scope_label = ttk.Label(data_panel, textvariable=self.tracking_scope, style="CaptureMuted.TLabel",
                                     width=1, justify=tk.LEFT)
        self.scope_label.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.scope_label.grid_remove()
        wrap_label_on_resize(self.scope_label)
        ToolTip(self.scope_label, lambda _event: self.tracking_scope.get() + "\n重置跟踪后可重新绘制测量区。")
        form.columnconfigure(1, weight=1)
        camera_label = ttk.Label(form, text="相机")
        camera_label.grid(row=0, column=0, sticky="w", pady=3)
        self.options = ttk.Combobox(form, textvariable=self.camera, state="readonly")
        self.options.grid(row=0, column=1, sticky="ew", padx=(9, 5))
        self.options.bind("<<ComboboxSelected>>", self._selection_changed)
        self.camera_help = ChoiceToolTip(self.options, lambda _event: self._editable_help(
            "选择实验中的相机。切换相机后，请在新参考图上核对测量区。"), {}, title="相机")
        ToolTip(camera_label, self.camera_help.resolve_text)
        self.refresh_button = ttk.Button(form, text="刷新", style="Small.TButton", command=self.refresh)
        self.refresh_button.grid(row=0, column=2)
        ToolTip(self.refresh_button, lambda _event: self._editable_help("重新从实验台账读取相机和参考帧。"))
        reference_label = ttk.Label(form, text="原参考图序号" if self._bound else "参考图序号")
        reference_label.grid(row=1, column=0, sticky="w", pady=3)
        reference_entry = ttk.Entry(form, textvariable=self.reference)
        reference_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(9, 0))
        reference_entry.bind("<Return>", self._selection_changed)
        reference_tip = ToolTip(reference_entry, lambda _event: self._editable_help(
            (f"已绑定原实验《{self._bound['external_reference']['name']}》的参考图。"
             "要换参考图，可从采集工作台重新打开实时分析。") if self._bound else
            self._reference_problem or "参考图是所有变形计算的基准。有多张时填写序号并按回车，只有一张可用时可留空。"),
            title="参考图序号")
        ToolTip(reference_label, reference_tip.resolve_text)
        self.part_row = ttk.Frame(form)
        self.part_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(3, 5))
        self.part_row.grid_remove()
        part_label = ttk.Label(self.part_row, text="图像部件")
        part_label.pack(side=tk.LEFT, padx=(0, 9))
        self.part_menu = ttk.Combobox(self.part_row, state="readonly", width=16)
        self.part_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.part_menu.bind("<<ComboboxSelected>>", self._part_selected)
        self.part_help = ChoiceToolTip(self.part_menu, lambda _event: self._editable_help(
            "选择参考帧和后续实验帧使用的一路图像。切换部件后，请在新参考图上重画测量区。"), {}, title="分析所用的图像")
        ToolTip(part_label, self.part_help.resolve_text)
        self.inputs = [self.options, reference_entry, self.target, self.refresh_button, self.part_menu]
        self._fixed_inputs = (self.options, reference_entry, self.part_menu) if self._bound else ()
        for line, (name, variable) in enumerate(self.rois.items(), 3):
            label = ttk.Label(form, text=ROI_LABELS[name])
            label.grid(row=line, column=0, sticky="w")
            entry = ttk.Entry(form, textvariable=variable)
            entry.grid(row=line, column=1, columnspan=2, sticky="ew", pady=2, padx=(9, 0))
            tip = ToolTip(entry, lambda _event, region=name: self._editable_help(
                f"{ROI_HELP[region]}\n可填写 x、y、宽、高四个像素值，例如 120,80,40,40。也可按住 Shift 拖出方框。"),
                title=ROI_LABELS[name])
            ToolTip(label, tip.resolve_text)
            self.inputs.append(entry)
        axis_label = ttk.Label(form, text="标距方向")
        axis_label.grid(row=6, column=0, sticky="w")
        axis = localized_combobox(form, self.strain_mode, AXIS_LABELS)
        axis.grid(row=6, column=1, columnspan=2, sticky="ew", pady=3, padx=(9, 0))
        axis_tip = ChoiceToolTip(axis, lambda _event: self._editable_help("选择计算两端点间距的方向。"), AXIS_HELP, title="标距方向")
        ToolTip(axis_label, axis_tip.resolve_text)
        self.inputs.append(axis)
        bayer = ttk.Checkbutton(form, text="将 Bayer 图像转换为亮度", variable=self.debayer)
        bayer.grid(row=7, column=0, columnspan=3, sticky="w", pady=3)
        ToolTip(bayer, lambda _event: self._editable_help(
            "Bayer 是彩色传感器的原始像素排列。此选项生成用于跟踪的亮度图，原始图像保留在实验目录。"))
        self.inputs.append(bayer)
        self.settings_form.bind_controls()

        actions = ttk.Frame(data_panel)
        actions.grid(row=1, column=0, sticky="ew", pady=8)
        self.start_button = ttk.Button(actions, text="开始跟踪", width=0, style="Record.TButton", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(self.start_button, lambda _event: self._start_problem() or
                "用参考图中的测量区域跟踪已保存的实验图。每张算出的结果会自动保存，可导出为表格。")
        self.pause_button = ttk.Button(actions, text="暂停", command=self.pause, style="Small.TButton")
        self.pause_button.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(self.pause_button, lambda _event: (
            "请先开始跟踪。" if self._analyzer is None else "继续跟踪新保存的图像。" if self._paused else
            "暂停更新应变曲线。采集继续写入图像。"))
        self.end_button = ttk.Button(actions, text="重置", command=self.reset, style="Small.TButton")
        self.end_button.pack(side=tk.LEFT)
        ToolTip(self.end_button, lambda _event: (
            "当前帧计算结束后将重置。" if self._reset_pending else "请先开始跟踪。" if self._analyzer is None else
            "结束本次跟踪并清空画面上的曲线。已保存的结果仍可导出。重新开始会生成一份新记录。"))
        self.mechanics_button = ttk.Button(actions, text="载荷对齐…", command=self.mechanics, style="Small.TButton")
        self.mechanics_button.pack(side=tk.RIGHT)
        ToolTip(self.mechanics_button, self._mechanics_help)

        plot_heading = ttk.Frame(data_panel)
        plot_heading.grid(row=2, column=0, sticky="ew", pady=(3, 5))
        self.plot_caption = ttk.Label(plot_heading, text="计算顺序", style="CaptureSection.TLabel")
        self.plot_caption.pack(side=tk.LEFT)
        self.curve_picker = localized_combobox(plot_heading, self.curve_mode, CURVE_MODES, width=11)
        ChoiceToolTip(self.curve_picker, "切换图上显示的量。曲线点仍对应同一张图像。",
                      CURVE_HELP, title="曲线视图")
        policy = localized_combobox(plot_heading, self.follow_mode, FOLLOW_MODES, width=11)
        policy.pack(side=tk.RIGHT)
        policy.bind("<<ComboboxSelected>>", self._mode_changed, add="+")
        ChoiceToolTip(policy, "选择实时计算顺序。切换时从当前进度继续。重置后可从参考图重新跟踪。",
                      FOLLOW_HELP, title="实时计算顺序")
        self.plot = LiveCurve(data_panel, self._curve_selection, self._open_curve_image)
        self.plot.grid(row=3, column=0, sticky="nsew")
        self.chart_actions = ttk.Frame(data_panel)
        self.chart_actions.grid(row=4, column=0, sticky="ew", pady=(4, 2))
        self.inspect_curve_button = ttk.Button(self.chart_actions, text="核对原图", command=self._open_curve_image,
                                               style="Small.TButton", state="disabled")
        self.inspect_curve_button.pack(side=tk.RIGHT)
        ToolTip(self.inspect_curve_button, lambda _event: (
            f"读取并校验帧 {self.plot.selected.frame.sequence} 的已保存原图。" if self.plot.selected else
            "先点击曲线点或底部短线选中一帧。也可聚焦曲线后按方向键。"))
        self.time_axis = ttk.Label(self.chart_actions, style="CaptureMuted.TLabel", width=1)
        self.time_axis.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ToolTip(self.time_axis, self._curve_axis_help)
        self.curve_mode.trace_add("write", self._curve_mode_changed)
        self.result_label = ttk.Label(data_panel, textvariable=self.result, justify=tk.LEFT, width=1, wraplength=340)
        self.result_label.grid(row=5, column=0, sticky="ew", pady=(5, 0))
        wrap_label_on_resize(self.result_label)
        ToolTip(self.result_label, lambda _event: self.result.get(), title="实时测量结果")
        footer = ttk.Frame(layout)
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(1, weight=1)
        ttk.Label(footer, text="实时预览结果", foreground=ACCENT).grid(row=0, column=0, sticky="w", padx=(0, 12))
        state = ttk.Label(footer, textvariable=self.state, width=1, justify=tk.LEFT)
        state.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        wrap_label_on_resize(state)
        ToolTip(state, lambda _event: self._reference_problem or self.state.get(), title="跟踪状态")
        progress = ttk.Label(footer, textvariable=self.progress, style="CaptureMuted.TLabel")
        progress.grid(row=0, column=2, sticky="e")
        ToolTip(progress, "统计所选相机在参考图之后的实验帧。实时跳过的帧保存在原实验，仍可离线分析。",
                title="实时分析进度")
        record_label = ttk.Label(footer, textvariable=self.record_status, style="CaptureMuted.TLabel", width=1)
        record_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 6), pady=(5, 0))
        wrap_label_on_resize(record_label)
        ToolTip(record_label, self._record_help, title="实时结果记录")
        export_actions = ttk.Frame(footer)
        export_actions.grid(row=1, column=2, sticky="e", pady=(5, 0))
        self.fill_load_button = ttk.Button(export_actions, text="补齐载荷…",
                                           command=lambda: self.export_results(fill_load=True), style="Small.TButton")
        self.fill_load_button.pack(side=tk.LEFT, padx=(0, 6))
        ToolTip(self.fill_load_button, lambda _event: (
            "跟踪前设置载荷对齐，才可按同一轮载荷表补齐缺失值。" if self._analyzer and not self._load_alignment else
            "选择同一轮更新后的载荷表。核对之前的采样后，将能匹配的缺失载荷和位移补入新 CSV。"
            "应变和已匹配的载荷来自原逐帧记录。"))
        self.export_button = ttk.Button(export_actions, text="导出实时结果…", command=self.export_results, style="Small.TButton")
        self.export_button.pack(side=tk.LEFT)
        ToolTip(self.export_button, lambda _event: (
            "请先开始跟踪，等参考图分析完成。" if self._last_record is None else
            "将原图标识、时间、应变和跟踪质量保存到 CSV 表格。重置跟踪后可选择以前的记录。"))
        self._controls()

    def _record_help(self, _event=None):
        lines = []
        if self._record_problem is not None:
            lines.append(f"逐帧保存遇到问题。\n{ui_error(self._record_problem)}")
        if self._last_record is not None:
            lines.append(f"逐帧记录保存在实验的 live-results 文件夹。\n{self._last_record}")
        else:
            lines.append("参考图算完后开始保存。暂停或重置仍可保留已经算出的结果。")
        if self._export_result is not None:
            name, summary, error = self._export_result
            if error is not None:
                lines.append(f"表格导出失败。\n{ui_error(error)}")
            else:
                lines.append(f"最近导出的表格包含参考图和 {summary['test_frames']:,} 张实验图的结果。\n{name}")
                if summary["incomplete_tail"]:
                    lines.append("记录末尾有一行不完整，导出时已略过。")
                if summary["skipped"]:
                    lines.append(f"这次跟踪实时跳过 {summary['skipped']:,} 张实验图。")
                if "filled_frames" in summary:
                    lines.append(f"后到采样补齐 {summary['filled_frames']:,} 张图。其中 {summary['filled_force_frames']:,} 张补齐载荷，"
                                 f"{summary['filled_displacement_frames']:,} 张补齐位移。"
                                 f"仍有 {summary['missing_force_frames']:,} 张实验图缺少有效载荷。")
        return "\n\n".join(lines)

    def _update_record_status(self):
        recording = self._recording
        if self._export_pending:
            status = "正在核对载荷并导出" if self._export_fills_load else "正在导出表格"
        elif self._record_problem is not None:
            status = "后续结果保存失败" if recording and recording.path else "实时结果保存失败"
        elif recording is not None:
            if recording.path is None:
                status = "等待参考图结果"
            else:
                count = recording.test_frames
                status = f"已保存 {count:,} 张实验图的结果" if count else "已保存参考图结果"
        else:
            status = "可导出以往的实时结果" if self._last_record else "跟踪开始后自动保存"
        if not self._export_pending and self._export_result is not None:
            _, summary, error = self._export_result
            if error is not None:
                status += " · 表格导出失败"
            elif "filled_frames" in summary:
                status += f" · 已导出 {summary['test_frames']:,} 张 · 补齐 {summary['filled_frames']:,} 张"
            else:
                status += f" · 已导出 {summary['test_frames']:,} 张"
        if status != self.record_status.get():
            self.record_status.set(status)

    def _editable_help(self, text):
        return "重置实时跟踪后可修改。" if self._analyzer else text

    def _reference_number(self):
        if not (value := self.reference.get().strip()):
            return None
        try:
            number = int(value)
        except ValueError:
            number = 0
        if number <= 0:
            raise ValueError("参考帧序号应为大于零的整数。")
        return number

    def _reference_key(self):
        return self.camera.get(), self.reference.get().strip(), self.image_part

    def _start_problem(self):
        if self._analyzer:
            return "已开始跟踪。可暂停曲线更新，或重置后修改参考帧和测量区。"
        if self._camera_ids and self.camera.get() not in self._camera_ids:
            return "请在“相机”中选择要分析的设备。"
        origin = self._load_alignment.load_origin if self._load_alignment else None
        if origin and self.camera.get() != origin.camera_id:
            return "载荷同步帧属于另一台相机。请切回该相机，或在“载荷对齐”里重新设置。"
        if self._reference_request == self._reference_key() and self._reference_problem:
            return self._reference_problem
        try:
            self._reference_number()
            first, second, field = (_rectangle(var.get()) for var in self.rois.values())
        except ValueError as exc:
            return str(exc)
        if (first is None) != (second is None):
            return "标距需要两个端点。请补画另一个端点，或清空标距输入后仅做全场分析。"
        if not first and not field:
            return "按住 Shift 在参考图上画两个标距端点，或选“全场区域”画一个方框。"
        return ""

    def _controls(self, *_args):
        active = self._analyzer is not None
        if active:
            row = self._analyzer.reference
            source = f"原实验参考图 {self.reference.get()}" if self._bound else (
                f"参考图 {row['sequence']}" if row else "等待参考图")
            detail = f"相机 {self.camera.get()}  ·  {source}"
            if detail != self.tracking_scope.get():
                self.tracking_scope.set(detail)
            if self.settings_form.winfo_manager():
                ToolTip.dismiss_window(self.root)
                self.settings_form.grid_remove()
                self.scope_label.grid()
        elif not self.settings_form.winfo_manager():
            self.scope_label.grid_remove()
            self.settings_form.grid()
        if active != bool(self.plot.winfo_manager()):
            self.plot.master.rowconfigure(0, weight=int(not active))
            self.plot.master.rowconfigure(3, weight=int(active))
            self._configure_curves(active)
            for widget in (self.plot, self.chart_actions, self.result_label):
                if active:
                    widget.grid()
                else:
                    widget.grid_remove()
        problem = self._start_problem()
        self.start_button.state(["disabled" if problem else "!disabled"])
        self.pause_button.state(["!disabled" if active and not self._reset_pending else "disabled"])
        self.pause_button.configure(text="继续" if self._paused else "暂停")
        self.end_button.state(["!disabled" if active and not self._reset_pending else "disabled"])
        self.mechanics_button.state(["disabled" if active else "!disabled"])
        for item in self.inputs:
            item.state(["disabled" if active or item in self._fixed_inputs else "!disabled"])
        self.offline_button.state(["disabled" if self._offline_pending else "!disabled"])
        self.export_button.state(["disabled" if self._export_pending or self._last_record is None else "!disabled"])
        self.fill_load_button.state(["disabled" if self._export_pending or self._last_record is None
                                     or active and self._load_alignment is None else "!disabled"])
        self.view.region_editable = not active
        if not active:
            self.result.set(problem or "测量区域已设置。点击“开始跟踪”查看应变预览。")

    def refresh(self):
        if self._analyzer is not None:
            return
        try:
            with ExperimentReader(self.experiment) as reader:
                cameras = [item["device"] for item in reader.cameras()]
            identities = tuple(device["id"] for device in cameras)
            if identities != self._camera_ids:
                self._camera_ids = identities
                self.options.configure(values=identities)
                self.camera_help.set_choices({
                    device["id"]: ("模拟散斑相机" if device["backend"] == "sim" else device["model"])
                    + (f" · 序列号 {device['serial']}" if device.get("serial") else "")
                    for device in cameras
                })
            if self.camera.get() not in identities and len(identities) == 1:
                self.camera.set(identities[0])
            self._display_reference()
        except Exception as exc:
            self.state.set(ui_error(exc, details=False) if (self.experiment / "capture.sqlite3").is_file() else "等待采集台账")
        self._controls()

    def _selection_changed(self, _event=None):
        camera = self.camera.get()
        if self._part_camera and camera != self._part_camera:
            self.image_part = None
            self._clear_regions()
        self._part_camera = camera
        self._set_part_options(None)
        self._display_key = None
        self._reference_problem = ""
        self.view.roi = None
        self.view.set_title("已保存的参考图")
        self.view.image(None, placeholder="等待这台相机的参考帧")
        self._display_reference()
        self._controls()

    def _clear_regions(self):
        for variable in self.rois.values():
            variable.set("")

    def _set_part_options(self, frame):
        self._part_choices.clear()
        if frame is None:
            self.part_row.grid_remove()
            return
        primary = frame_part_selector(frame)
        options = analysis_part_options(frame)
        if len(options) == 1 and self.image_part is None:
            self.part_row.grid_remove()
            return
        self._part_choices = {label: selector for label, (selector, _) in options.items()}
        chosen = next((label for label, selector in self._part_choices.items()
                       if self.image_part == (selector or primary)), None)
        if self.image_part is None:
            chosen = next(iter(options))
        self.part_menu.configure(values=tuple(options))
        self.part_menu.set(chosen or f"参考帧缺少部件 {self.image_part}")
        self.part_help.set_choices({label: detail for label, (_, detail) in options.items()})
        self.part_row.grid()

    def _part_selected(self, _event=None):
        label = self.part_menu.get()
        if label in self._part_choices and self._part_choices[label] != self.image_part:
            self.image_part = self._part_choices[label]
            self._clear_regions()
            self._selection_changed()

    def _display_reference(self):
        if self._working or self._analyzer or not self.camera.get() or self._display_key:
            return
        self._reference_request = self._reference_key()
        self._reference_problem = ""
        try:
            if self._bound:
                row = self._bound["frames"][0]
                self._set_part_options(row)
                pixels = read_linked_reference_image(row, debayer=row["color_model"] == "bayer")
                self._update_image(row, pixels)
                self._display_key = row["frame_id"]
                self.view.set_title(f"原实验参考图 {row['sequence']}")
                self.state.set("原参考图已就绪")
                return
            sequence = self._reference_number()
            with ExperimentReader(self.experiment) as reader:
                frames = reader.reference_frames(self.camera.get(), sequence=sequence)
                if not frames:
                    if reader.experiment["state"] in TERMINAL_STATES:
                        self._reference_problem = "实验中找不到这台相机的参考帧。请核对帧序号，或切换相机。"
                        self.state.set("找不到参考帧")
                    else:
                        self.state.set("等待参考帧" if sequence is None else f"等待已保存的参考帧 {sequence}")
                    return
                if len(frames) > 1:
                    sequences = "、".join(str(frame["sequence"]) for frame in frames[:6])
                    self._reference_problem = f"有多张参考图（如 {sequences}）。请填写序号并按回车，也可在采集工作台查看原图。"
                    self.state.set("请选择参考帧")
                    return
                self._set_part_options(frames[0])
                row = select_frame_image(frames[0], self.image_part)
                path = verified_image_path(reader, row)
            pixels = read_analysis_image(path, row, debayer=row["color_model"] == "bayer")
            self._update_image(row, pixels)
            self._display_key = row["frame_id"]
            self.view.set_title(f"参考帧 {row['sequence']}" + (f" · {self.image_part}" if self.image_part else ""))
            self.state.set(f"参考帧 {row['sequence']}")
        except ValueError as exc:
            self._reference_problem = str(exc)
            self.state.set("请检查参考图序号")
        except Exception as exc:
            self._reference_problem = ui_error(exc, details=False)
            self.state.set(self._reference_problem)

    def _update_image(self, row, pixels, *, display_only=False):
        timestamps = row["timestamps"]
        label = ("Bayer 亮度" if row.get("image_conversion") == "derived_bayer_luminance"
                 else row.get("pixel_format", "相机亮度"))
        self.view.image(Image(
            pixels, "DisplayRGB8" if display_only else label,
            pixels.dtype.itemsize * 8 if display_only else row.get("significant_bits", pixels.dtype.itemsize * 8),
            "rgb" if pixels.ndim == 3 else "mono",
            timestamps["received_utc"]["value"], timestamps["received_monotonic"]["value"],
        ))

    def _draw_region(self, _camera, region):
        if not self._analyzer:
            key = self.draw_target.get()
            self.rois[key].set(", ".join(map(str, region)))
            if key == "Gauge ROI 1":
                self.draw_target.set("Gauge ROI 2")

    def start(self):
        if self._analyzer:
            return
        if self._reference_request != self._reference_key():
            self._selection_changed()
        if problem := self._start_problem():
            self.state.set(problem)
            return
        try:
            first, second, field = (_rectangle(var.get()) for var in self.rois.values())
            groups = [{"name": "G01", "roi1": first, "roi2": second, "strain_mode": self.strain_mode.get()}] if first else []
            sequence = self._reference_number()
            self._analyzer = LiveAnalysis(self.experiment, self.camera.get() or None,
                                          reference_sequence=None if self._bound else sequence,
                                          image_part=self.image_part, gauges=groups, field_roi=field, load_alignment=self._load_alignment,
                                          debayer_luminance=self.debayer.get(), reference_binding=self.reference_binding,
                                          reference_binding_digest=self._bound_digest)
            self._recording = LiveResultLog(self.experiment)
            self._record_problem = None
            self._export_result = None
            self._update_record_status()
            self._paused = False
            self.state.set("正在跟踪已保存图像" if self._display_key else "等待已保存的参考帧")
            self._controls()
        except Exception as exc:
            messagebox.showerror("无法开始实时分析", ui_error(exc), parent=self.root)

    def pause(self):
        if self._reset_pending:
            return
        self._paused = not self._paused
        if not self._paused:
            self._issue = None
            self._next_poll = 0.
        self._show_analysis_state()
        self._controls()

    def reset(self):
        if self._analyzer is None:
            return
        if self._working:
            self._reset_pending = True
            self.state.set("当前帧结束后重置")
            self._controls()
        else:
            self._finish_reset()

    def _finish_reset(self):
        self._analyzer.close()
        self._analyzer = self._metrics = self._pending_paint = self._issue = None
        self._recording = None
        self._update_record_status()
        self.plot.clear()
        self._paused = self._reset_pending = False
        self._next_poll = self._next_status = self._next_paint = 0.
        self.progress.set("")
        self._selection_changed()
        self._controls()
        self.refresh()

    def _mode_changed(self, _event=None):
        self._next_poll = 0.

    def _configure_curves(self, active):
        modes = available_curve_modes(strain=bool(self._analyzer.groups), force=bool(self._load_alignment),
                                      field=self._analyzer.field_roi is not None) if active else ()
        self._curve_modes = modes
        if len(modes) > 1:
            self.plot_caption.pack_forget()
            self.curve_picker.configure(values=tuple(CURVE_MODES[mode] for mode in modes))
            self.curve_picker.pack(side=tk.LEFT)
        else:
            self.curve_picker.pack_forget()
            caption = CURVE_MODES[modes[0]] if modes else "全场预览" if active else "计算顺序"
            self.plot_caption.configure(text=caption)
            self.plot_caption.pack(side=tk.LEFT)
        if modes and self.curve_mode.get() not in modes:
            self.curve_mode.set(modes[0])
        else:
            self._curve_mode_changed()

    def _curve_mode_changed(self, *_args):
        mode = self.curve_mode.get()
        self.plot.configure_mode(mode, enabled=bool(self._curve_modes))
        if mode == "force-strain":
            caption = "工程应变 (%)"
        else:
            caption = "相对本轮首图收图时间 (s)" if self._bound else "相对参考图收图时间 (s)"
        self.time_axis.configure(text=caption if self._curve_modes else "")

    def _curve_axis_help(self, _event=None):
        if self.curve_mode.get() == "force-strain":
            return "横轴是同一帧的工程应变。按采集顺序绘制，保留加载和卸载的方向变化。"
        origin = "本轮首张实验图" if self._bound else "参考图"
        return f"以收到{origin}的时刻为零。载荷值按载荷对齐中设置的时钟取值。"

    def _curve_selection(self, point):
        self.inspect_curve_button.state(["!disabled" if point else "disabled"])
        self.inspect_curve_button.configure(text=f"核对帧 {point.frame.sequence}" if point else "核对原图")

    def _open_curve_image(self, point=None):
        point = point or self.plot.selected
        if point is None:
            return
        ToolTip.dismiss_window(self.root)
        if self._frame_window is None or not self._frame_window.top.winfo_exists():
            self._frame_window = CurveFrameWindow(self.root)
        self._frame_window.show(point)

    def _destroy(self, event):
        if event.widget is self.root:
            if self._tick_after is not None:
                self.root.after_cancel(self._tick_after)
            if self._analyzer is not None and not self._working:
                self._analyzer.close()
            if self._worker is not None:
                self._tasks.put(None)

    def offline(self):
        if self._bound:
            if self._offline_pending:
                return
            self._offline_pending = True
            self._controls()

            def prepare():
                try:
                    file = offline_from_live_reference(self.reference_binding, expected_digest=self._bound_digest)
                except Exception as exc:
                    self._offline_jobs.put((None, exc))
                else:
                    self._offline_jobs.put((file, None))
            threading.Thread(target=prepare, name="prepare-live-offline", daemon=True).start()
            return
        try:
            sequence = self._reference_number()
            launch_analysis(self.experiment, camera_id=self.camera.get() or None, reference_sequence=sequence,
                            image_part=self.image_part)
        except (CaptureError, OSError, ValueError) as exc:
            messagebox.showerror("无法打开离线分析", ui_error(exc), parent=self.root)

    def open_capture(self):
        try:
            launch_capture(self.experiment)
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法打开采集实验", ui_error(exc), parent=self.root)

    def show_history(self):
        from .live_history_gui import LiveHistoryWindow

        window = getattr(self, "_live_history_window", None)
        if window is not None and window.top.winfo_exists():
            window.top.lift()
            return
        try:
            self._live_history_window = LiveHistoryWindow(self.root, self.experiment, preferred_record=self._last_record)
        except (CaptureError, OSError) as error:
            messagebox.showerror("无法打开历史曲线", ui_error(error), parent=self.root)

    def _mechanics_help(self, _event=None):
        if channel := self._load_alignment:
            basis = LOAD_TIME_BASIS_LABELS[channel.basis]
            origin = " · 按同步帧对齐" if channel.load_origin else ""
            help_text = f"已选 {channel.source.name} · {basis}{origin}\n开始跟踪后，载荷会随匹配到的图像一起保存。"
            return help_text + ("\n重置跟踪后可更换载荷表。" if self._analyzer else "")
        return self._editable_help("直接选择试验机导出的载荷表，先预览帧与载荷的匹配。只测图像应变时可跳过。")

    def mechanics(self):
        try:
            sequence = self._reference_number()
        except ValueError as exc:
            self.state.set(str(exc))
            return
        MechanicsDialog(self.root, self.experiment, self.camera.get() or None, sequence, bool(self._bound),
                        self._load_alignment, self._set_mechanics)

    def _set_mechanics(self, channel):
        self._load_alignment = channel
        self.mechanics_button.configure(text="载荷已设置…" if channel else "载荷对齐…")
        self.state.set(f"已设置载荷对齐 · {LOAD_TIME_BASIS_LABELS[channel.basis]}" if channel else "已清除载荷对齐")
        self._controls()

    def export_results(self, *, fill_load=False):
        if self._export_pending or self._last_record is None:
            return
        source = self._recording.path if self._recording is not None else None
        if source is None:
            name = filedialog.askopenfilename(
                parent=self.root, title="选择要导出的实时记录", initialdir=self._last_record.parent,
                initialfile=self._last_record.name, filetypes=[("实时分析逐帧记录", "*.jsonl")],
            )
            if not name:
                return
            source = Path(name)
        load_table = None
        if fill_load:
            try:
                original_table = live_log_load_source(source)
            except (CaptureError, OSError) as error:
                self._export_result = (str(source), None, error)
                self._update_record_status()
                return
            name = filedialog.askopenfilename(
                parent=self.root, title="选择同一轮更新后的载荷表",
                initialdir=original_table.parent if original_table.parent.is_dir() else source.parent,
                initialfile=original_table.name, filetypes=[("载荷表", "*.csv *.tsv *.txt"), ("所有文件", "*.*")],
            )
            if not name:
                return
            load_table = Path(name)
        prefix = "补齐载荷" if fill_load else "实时结果"
        name = filedialog.asksaveasfilename(
            parent=self.root, title="补齐载荷并导出" if fill_load else "导出实时结果表格", initialdir=source.parent,
            initialfile=f"{prefix}-{source.stem}.csv", defaultextension=".csv", filetypes=[("CSV 表格", "*.csv")],
        )
        if not name:
            return
        self._export_pending = True
        self._export_fills_load = fill_load
        self._export_result = None
        self._update_record_status()
        self._controls()

        def export():
            try:
                summary = export_live_csv(source, name, fill_load=fill_load, load_table=load_table)
            except Exception as exc:
                self._export_jobs.put((name, None, exc))
            else:
                self._export_jobs.put((name, summary, None))

        threading.Thread(target=export, name="live-results-export").start()

    def _analysis_worker(self):
        analyzer = None
        for analyzer, recording, policy, measure in iter(self._tasks.get, None):
            row = image = legend = error = metrics = None
            if policy is not None:
                try:
                    row = analyzer.poll(latest=policy == "latest")
                    image = analyzer.image
                    if row is not None:
                        recording.append(row, analyzer, policy)
                    if row and (quality := row["field_quality"]) and quality.get("scientific_ok"):
                        core = analyzer._core
                        image8 = core.normalize_with_bounds(image, analyzer._norm)
                        image = core.overlay_dic_field_on_image(image8, analyzer.field, component="Eyy")
                        lo, hi, scale = core.dic_color_limits(analyzer.field, "Eyy")
                        legend = f"Eyy 色标范围  {lo / scale:.3%} … {hi / scale:.3%}"
                except Exception as exc:
                    error = exc
            if measure:
                try:
                    metrics = analyzer.progress()
                except Exception as exc:
                    error = error or exc
            self._jobs.put((recording, policy, row, image, legend, metrics, error))
        if analyzer is not None:
            analyzer.close()

    def _show_analysis_state(self):
        if self._issue is not None:
            self.state.set(ui_error(self._issue, details=False))
            return
        metrics = self._metrics
        if self._paused:
            self.state.set("分析已暂停")
        elif not metrics:
            self.state.set("正在跟踪已保存图像")
        elif metrics["experiment_state"] in TERMINAL_STATES:
            captured = STATE_LABELS.get(metrics["experiment_state"].upper(), "采集已结束")
            if not metrics["reference_ready"]:
                self.state.set(f"{captured} · 缺少参考图")
            else:
                status = "正在分析待处理图像" if metrics["pending_committed_frames"] else "已处理到末帧"
                self.state.set(f"{captured} · {status}")
        elif not metrics["reference_ready"]:
            self.state.set("等待已保存的参考图")
        elif not metrics["analyzed_test_frames"]:
            self.state.set(("原参考图" if self._bound else "参考图") + "已就绪 · 等待本轮实验图")
        else:
            self.state.set(STATE_LABELS.get(metrics["experiment_state"].upper(), "状态未知"))

    def _refresh_progress(self, row, metrics):
        if metrics is not None:
            self._metrics = metrics
        elif row is not None and self._metrics is not None:
            consumed = 0
            for key in ("analyzed_test_frames", "skipped_committed_frames"):
                consumed += row[key] - self._metrics[key]
                self._metrics[key] = row[key]
            pending = self._metrics["pending_committed_frames"]
            if pending is not None:
                self._metrics["pending_committed_frames"] = max(0, pending - consumed)
        if self._metrics is not None:
            pending = self._metrics["pending_committed_frames"]
            backlog = f"{pending:,}" if pending is not None else "—"
            self.progress.set(f"已分析 {self._metrics['analyzed_test_frames']:,} 张 · 待分析 {backlog} 张"
                              f" · 实时跳过 {self._metrics['skipped_committed_frames']:,} 张")

    def _tick(self):
        self._tick_after = None
        now = monotonic()
        if not self._offline_jobs.empty():
            path, error = self._offline_jobs.get_nowait()
            self._offline_pending = False
            if error is None:
                try:
                    launch_corrected_analysis(path)
                except OSError as exc:
                    error = exc
            if error is not None:
                messagebox.showerror("无法打开离线分析", ui_error(error), parent=self.root)
            self._controls()
        if not self._jobs.empty():
            recording, policy, row, image, legend, metrics, error = self._jobs.get_nowait()
            self._working = False
            if row is not None:
                if recording.path is not None:
                    self._last_record = recording.path
                self._record_problem = recording.error
                self._update_record_status()
            if self._reset_pending:
                self._finish_reset()
            else:
                self._refresh_progress(row, metrics)
                if policy is not None:
                    self._next_poll = now if row is not None else now + _POLL_IDLE_S
                if row is not None:
                    self._render_result(row, image, legend)
                if error is not None:
                    self._paused, self._issue = True, error
                    self.result.set(ui_error(error))
                self._show_analysis_state()
                self._controls()
        if not self._export_jobs.empty():
            self._export_result = self._export_jobs.get_nowait()
            self._export_pending = False
            self._update_record_status()
            self._controls()
        analyzer = self._analyzer
        drained = (self._metrics is not None and self._metrics["experiment_state"] in TERMINAL_STATES
                   and self._metrics["pending_committed_frames"] == 0)
        if analyzer is not None and not self._working:
            measure = now >= self._next_status
            follow = not self._paused and not drained and now >= self._next_poll
            if follow or measure:
                if self._worker is None:
                    self._worker = threading.Thread(target=self._analysis_worker, name="live-analysis-read", daemon=True)
                    self._worker.start()
                self._working = True
                self._tasks.put((analyzer, self._recording, self.follow_mode.get() if follow else None, measure))
                if measure:
                    self._next_status = now + _STATUS_S
        elif not analyzer and not self._display_key:
            self.refresh()
        if self._pending_paint is not None and now >= self._next_paint:
            self._paint_latest()
            self._next_paint = now + _PAINT_S
        interval = 400 if analyzer is None else (100 if self._paused or drained else 30)
        self._tick_after = self.root.after(interval, self._tick)

    def _render_result(self, row, image, legend):
        gauge = row["gauges"][0] if row["gauges"] else None
        quality = row["field_quality"]
        self._pending_paint = row, image, legend is not None
        experiment = self._bound["external_reference"]["directory"] if row.get("is_reference") else self.experiment
        self.plot.points.append(CurvePoint.from_result(row, experiment, self._analyzer.camera_id))
        if gauge:
            strain = gauge["engineering_strain"]
            tracking_status = gauge["quality_status"]
            result = f"{gauge['name']}  ·  {TRACKING_LABELS.get(tracking_status, '请复查跟踪结果')}"
            if strain is not None:
                result += f"  ·  {strain:.4%}"
        else:
            result = ""
        if quality:
            reasons = "、".join(FIELD_REASON_LABELS.get(reason, "请复查应变质量") for reason in quality["scientific_reasons"])
            result += ("\n" if result else "") + (
                f"全场 DIC  ·  有效应变点 {quality['strain_valid_fraction']:.0%}\n{legend}" if legend
                else f"全场质量检查  ·  {reasons}"
            )
        if mechanics := row.get("mechanics"):
            force = mechanics.get("force_N")
            support = (mechanics.get("force_support") or {}).get("status")
            if force is None:
                result += f"\n载荷  ·  {LOAD_SUPPORT_LABELS.get(support, '无可用数据')}"
            else:
                basis = LOAD_TIME_BASIS_LABELS[mechanics["time_basis"]]
                if mechanics.get("load_time_domain") == "relative":
                    basis += " · 同步帧"
                result += f"\n载荷  {force:.4g} N  ·  {basis}"
                if (displacement := mechanics.get("displacement_mm")) is not None:
                    result += f"\n位移  {displacement:.4g} mm"
        if self._bound and not row.get("is_reference") and row["elapsed_receive_s"] is None:
            result += "\n本帧缺少可对照的本轮收图时间。"
        self.result.set(f"最新帧 {row['sequence']}  ·  {result.lstrip()}" if result else "")

    def _paint_latest(self):
        row, image, field_overlay = self._pending_paint
        self._pending_paint = None
        title = f"原实验参考图 {row['sequence']}" if row.get("is_reference") else f"已保存图像 {row['sequence']}"
        self.view.set_title(title + (" · Eyy" if field_overlay else ""))
        self._update_image(row, image, display_only=field_overlay)
        self.plot.draw()


def run_live(experiment, camera=None, reference_sequence=None, *, image_part=None, reference_binding=None):
    root = create_tk_root()
    try:
        LiveWorkbench(root, experiment, camera, reference_sequence, image_part=image_part, reference_binding=reference_binding)
    except (CaptureError, OSError) as exc:
        messagebox.showerror("无法打开实时分析", ui_error(exc), parent=root)
        root.destroy()
        return 2
    root.mainloop()
    return 0
