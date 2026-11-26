"""Independent StrainTrace Capture desktop workbench."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sqlite3
import sys
import threading
import time
import tkinter as tk
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.font import Font
from uuid import uuid4

from ezdic_core import _operation_state_dir
from ezdic_gui.controls import ScrollableForm, focus_canvas_child, localized_combobox, notebook_help, wrap_label_on_resize
from ezdic_gui.theme import choose_ui_font_family, create_tk_root
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from . import CameraRegistry, CaptureConfig, CaptureError, CaptureSession, ExperimentReader, inspect_plan
from .archive_quality import archive_quality_options, open_archive_comparison
from .archive_quality_gui import ArchiveQualityWindow
from .batch_camera import verify_batch_settings
from .batch_camera_gui import BatchCameraDialog
from .camera_choice_gui import CameraPreviewDialog
from .calibration import (
    CHECKERBOARD_PHASE, DEFAULT_MIN_VIEWS, CheckerboardScan, archived_profile_path, calibrate, calibration_geometry,
    checkerboard_target, matching_calibration, phase_sensor_geometry, prepare_checkerboard_capture, read_profile,
)
from .capture_review import check_review_identity, inspect_capture_review
from .cameras._gentl_multipart import IMAGE_PART, SAVE_ALL_2D
from .cameras.base import HOST_NEXT_RECEIPT
from .config import CameraConfig, Phase, QualityRegion, hardware_mapping_error, repeat_source_directory
from .continuation import CONTINUATION_REFERENCE, continuation_plan, continuation_reference, inspect_continuation
from .continuation_analysis_gui import ContinuationReferenceWindow
from .continuation_gui import ContinuationWindow
from .control import CaptureControl
from .dialogs import CaptureChoiceDialog, CaptureFieldDialog, CheckerboardDialog, CorrectedAnalysisDialog, FeatureDialog, PhaseDialog
from .event_images import event_image_choices
from .draft import capture_draft, load_capture_draft, load_draft_trials, save_capture_draft
from .exposure import ExposureScope, exposure_unavailable
from .exposure_gui import ExposureWindow
from .guide_gui import CaptureGuide
from .hardware_counters import frame_hardware_counters
from .image_quality_watch import quality_region_from_image, quality_region_mismatch
from .handoff import (
    analysis_snapshot, corrected_reference_choices, launch_analysis, launch_capture, launch_corrected_analysis,
    launch_live_reference, prepare_corrected_analysis, write_snapshot,
)
from .frame_search_gui import FrameSearchWindow
from .timeline_gui import TimelineWindow
from .image_inputs import frame_part_selector, recorded_image
from .marker_gui import MarkerReviewWindow
from .monitor_gui import CaptureMonitor
from .output import inspect_output, output_ancestor, resolve_output_path
from .plan_template_gui import PlanTemplateWindow
from .practice import PracticeCameraRegistry, practice_config, require_simulated_cameras
from .presentation import (
    CAPTURE_ERROR_LABELS, CSV_STATES, DRIVER_HOST_FRAME_HELP, DURABILITY_HELP, DURABILITY_LABELS,
    ISSUE_LABELS, MODE_HELP, MODE_LABELS, PHASE_PAUSE_DISPLAY,
    PHASE_PRESETS, PREVIEW_STATES, PROVIDER_LABELS, RATE_TRIAL_LABELS, RAW_BUFFER_HELP, READINESS_CONFIRMATION_ISSUES,
    ROLE_HELP, ROLE_LABELS, STATE_LABELS, TEXTURE_GUIDANCE, TRIGGER_ERROR_LABELS,
    WAIT_OUTPUT_LABELS, WRITE_PRESSURE_GUIDANCE, WRITE_PRESSURE_LABELS,
    buffer_peak_lines, capture_incident_lines, csv_trigger_caption, csv_trigger_help, csv_trigger_rule_text, format_bytes, format_integrity,
    event_window_position, finishing_phase_writes, format_provider_report, format_readiness, format_trial,
    format_trial_comparison, hardware_trigger_guidance, image_quality_alert_text, integrity_issues, phase_display, phase_limit_text,
    provider_diagnostic, quality_region_summary, ui_error, wait_output_camera_text, wait_output_help, write_pressure_estimate,
)
from .preview import PART_REQUIRED, QUALITY_ALERT_FRACTION, FocusComparison, PreviewSession, _image_layout, image_quality
from .preview_compare_gui import PreviewCompareWindow
from .preview_image import preview_original, preview_region_mismatch, save_preview_image
from .preview_quality import PreviewQualityTarget, inspect_preview_quality, with_preview_quality
from .preview_quality_gui import PreviewQualityWindow
from .pulse_audit import pulse_audit_context
from .pulse_audit_gui import PulseAuditWindow
from .pulse_monitor_gui import pulse_notice
from .rate_trial import assess_rate_trial, read_rate_trial_source, with_trial_rate
from .readiness import (
    apply_phase_suggestion, plan_camera_geometry, probe_camera_setup,
    probe_preflight_cameras, probe_trigger_support, trigger_evidence,
)
from .readiness_gui import ReadinessOverview
from .recent import RecentExperiment, recent_experiments, remember_experiment
from .recent_gui import RecentExperimentsDialog
from .recovery_gui import RecoveryWindow
from .related_gui import RelatedExperimentsWindow
from .repeat import prepare_repeat, validate_repeat_destination
from .repeat_gui import RepeatPlanWindow
from .storage import TERMINAL_STATES, TRIAL_REFERENCE_SCHEMA, frame_image_records, reported_camera_losses
from .transfer import TRANSFER_MANIFEST, verify_transfer
from .transfer_gui import ExperimentCopyWindow
from .sequence_gui import SequenceExportWindow
from .trial import (
    DEFAULT_TRIAL_SECONDS, compare_trials, prepare_trial, refresh_trial_image_regions, trial_image_alerts,
    trial_metadata, trial_report, volume_identity,
)
from .trial_batch import TrialBatchEnd, run_trial_batch
from .trial_progress import inspect_trial_progress, summarize_trial_progress
from .trial_progress_gui import TrialProgressWindow
from .trial_quality import read_trial_quality_region
from .trial_quality_gui import TrialImageWindow
from .trial_reference import archive_trial_reference, referenced_trial_report
from .trigger_review import TriggerReview
from .viewport import ImageViewport, repaint_after_resize
from .workspace_layout import fit_line, wrap_toolbar


BG, CARD, ENTRY = "#121c25", "#1a2a36", "#111e29"
FG, MUTED, ACCENT, BORDER = "#eef4f7", "#b4c7d1", "#66dec0", "#375060"
GOOD, WARN, BAD = "#a6edce", "#ffda87", "#ffacab"


@dataclass(slots=True)
class _TrialBatchTask:
    window: TrialProgressWindow
    stop: threading.Event
    total: int
    running: bool = False
    verified: int = 0


@dataclass(frozen=True)
class _RateTrial:
    original: CaptureConfig
    output: str
    revised: CaptureConfig
    suggestion: dict
    source: dict


@dataclass(frozen=True)
class _NextStep:
    title: str
    hint: str
    target: ttk.Button | None = None
    tab: ttk.Frame | None = None
    phase_confirmation: tuple[int, str] | None = None
    button_label: str | None = None
    timeline_filter: str = ""
    focus: tk.Misc | None = None
    suggested_output: Path | None = None
    action: Callable[[], object] | None = None


@dataclass(frozen=True)
class _QualityScope:
    source: PreviewSession | CaptureSession | None
    view: ImageViewport
    saved_frame: tuple[int, int | None] | None
    roi: tuple[int, int, int, int] | None
    image_layout: tuple


def _positive_number(text, convert):
    number = convert(text)
    if number <= 0 or (isinstance(number, float) and not math.isfinite(number)):
        raise ValueError
    return number


TAB_HELP = (
    "安排采集顺序、拍摄方式和结束条件。正式开始前可检查预计磁盘空间，或先试采一个阶段。",
    "检查光照、曝光和散斑。按住 Shift 在图像上拖出方框，可单独检查这一小块。",
    "浏览写入台账的图像，校验文件，打开实时或离线分析，也可准备续采。",
)
QUALITY_HELP = {
    "调参前后": "使用应用前的画面和收到的第一张新画面。取景和照明一致时可对照同一区域的亮度变化。",
    "亮度分位数": "P5、P50 和 P95 是亮度排序后位于 5%、50% 和 95% 位置的值。有检查框时，三个值都来自框内。",
    "近黑像素": "亮度落在最低 1% 码值范围的像素占比。比例偏高时可检查照明不足和曝光过短。",
    "近饱和像素": "码值落在最高 1% 范围的像素占比。彩色图像的任一颜色通道达到这一范围就会计入。"
               "比例偏高时可检查过曝和局部反光。",
    "对焦响应": "根据原图里相邻像素的变化估计细节强弱。固定相机、试样和照明，只转动对焦环，才能对照此值。",
    "对比度": "检查区域里亮和暗的差距。程序取第 95 与第 5 百分位的差值，按 8 位灰度计算。",
    "结构张量特征值比": "用于判断纹理是否集中在同一方向。接近 1 表示两个方向都有纹理。接近 0 时，"
    "纹理可能主要沿一个方向，另一个方向较难跟踪。",
}


def set_theme(root):
    root.configure(bg=BG)
    family = choose_ui_font_family(root)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=(family, 10), foreground=FG, background=BG)
    style.configure("CaptureCard.TFrame", background=CARD)
    style.configure("CaptureCard.TLabel", background=CARD, foreground=FG)
    style.configure("CaptureMuted.TLabel", foreground=MUTED, background=BG)
    style.configure("CaptureMono.TLabel", font=("Consolas", 9), foreground=MUTED, background=CARD)
    style.configure("CaptureTitle.TLabel", font=(family, 18, "bold"), foreground=FG)
    style.configure("CaptureSection.TLabel", font=(family, 12, "bold"), foreground=FG)
    style.configure("CaptureSelected.TLabel", font=(family, 10, "bold"), foreground=ACCENT, background=CARD)
    style.configure("CaptureAttention.TLabel", font=(family, 10, "bold"), foreground=WARN, background=CARD)
    style.configure("CaptureContext.TLabel", font=(family, 9, "bold"), foreground=ACCENT,
                    background=CARD, padding=(8, 4))
    style.configure("CaptureState.TLabel", font=(family, 9, "bold"), background=CARD, padding=(7, 4))
    style.configure("CaptureNext.TLabel", font=(family, 9), foreground=FG, background=CARD)
    style.configure("Help.TLabel", background="#233b49", foreground=FG, bordercolor="#5c889a")
    style.configure("HelpTitle.TLabel", font=(family, 10, "bold"))
    style.configure("HelpKey.TLabel", font=(family, 9), foreground="#b8d3df")
    style.configure("TButton", padding=(11, 6), bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD, background=CARD)
    style.map("TButton", background=[("pressed", "#304857"), ("active", "#283d4a")],
              foreground=[("disabled", "#889ba6")])
    style.configure("Small.TButton", padding=(7, 3), width=0)
    style.configure("Next.TButton", padding=(8, 3), width=0, foreground=ACCENT)
    style.configure("TCheckbutton", background=BG, indicatorcolor=ENTRY, foreground=FG)
    style.map("TCheckbutton", background=[("disabled", BG), ("active", CARD)],
              foreground=[("disabled", MUTED)], indicatorcolor=[("selected", ACCENT), ("disabled", BORDER)])
    style.configure("Record.TButton", font=(family, 10, "bold"), padding=(15, 8),
                    foreground="#112932", background=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT, bordercolor=ACCENT)
    style.map("Record.TButton", background=[("disabled", "#375060"), ("pressed", "#48bd9f"), ("active", "#a4f0dc")],
              foreground=[("disabled", MUTED)])
    style.configure("Confirm.Record.TButton", padding=(8, 3))
    style.configure("TEntry", fieldbackground=ENTRY, bordercolor=BORDER, insertcolor=FG, padding=6)
    style.configure("TCombobox", fieldbackground=ENTRY, arrowcolor=FG, bordercolor=BORDER, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", ENTRY)], selectbackground=[("readonly", ENTRY)],
              selectforeground=[("readonly", FG)])
    style.configure("TNotebook", background=BG, bordercolor=BORDER)
    style.configure("TNotebook.Tab", background=CARD, foreground=MUTED, bordercolor=BORDER, padding=(9, 6))
    style.map("TNotebook.Tab", background=[("selected", ENTRY), ("active", "#263b49")],
              foreground=[("selected", ACCENT)])
    row_font = Font(root=root, family=family, size=9)
    style.configure("Treeview", background=ENTRY, fieldbackground=ENTRY, foreground=FG, bordercolor=BORDER,
                    rowheight=max(26, row_font.metrics("linespace") + 6), font=(family, 9))
    style.map("Treeview", background=[("selected", "#325968")], foreground=[("selected", "#ffffff")])
    style.configure("Treeview.Heading", background=CARD, foreground=MUTED, padding=(4, 4), bordercolor=BORDER)
    style.configure("TScrollbar", background=BORDER, troughcolor=ENTRY, arrowcolor=FG,
                    bordercolor=ENTRY, lightcolor=BORDER, darkcolor=BORDER)
    style.map("TScrollbar", background=[("active", "#547483")])
    style.configure("CaptureCopy.Horizontal.TProgressbar", background=ACCENT, troughcolor=ENTRY,
                    bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
    root.option_add("*TCombobox*Listbox.background", ENTRY)
    root.option_add("*TCombobox*Listbox.foreground", FG)


_DEMO_CAPTURE_NAME = "散斑采集练习"


def demo_config() -> dict:
    return {
        "config_version": 1, "name": _DEMO_CAPTURE_NAME,
        "cameras": [{"id": "sim:0", "settings": {"width": 640, "height": 480, "pixel_format": "Mono12"}}],
        "phases": [dict(phase) for phase in PHASE_PRESETS["参考帧和定时实验"]],
    }


class CaptureWorkbench:
    def __init__(self, root, config=None, experiment=None, *, practice=False):
        self.root = root
        self._practice = practice
        root.title("StrainTrace Capture · 模拟练习" if practice else "StrainTrace Capture · 图像采集")
        width = max(900, min(1280, root.winfo_screenwidth() - 32))
        height = max(560, min(815, root.winfo_screenheight() - 88))
        root.geometry(f"{width}x{height}")
        root.minsize(900, 560)
        set_theme(root)
        repaint_after_resize(root)
        self.config = {}
        self.inventory = {}
        self.views = {}
        self.active_camera = None
        self.experiment = None
        self._experiment_id = None
        self._experiment_cameras = ()
        self.session = self.preview = self.control = None
        self.review_image = None
        self._review_sensor_geometry = {}
        self._calibration_profiles = {}
        self._jobs = queue.SimpleQueue()
        self._quality_work = False
        self._quality_due = 0.
        self._monitor_visible = False
        self._monitor_due = 0.
        self._write_pressure = None
        self._image_quality_alert = None
        self._pulse_notice = None
        self._quality_key = None
        self._quality_serial = 0
        self._quality_checked_preview = None
        self._preview_quality_results = ()
        self._preview_regions_pending = {}
        self._saving_preview_image = False
        self._preview_save_summary = self._preview_save_detail = ""
        self._last_preview_original = self._preview_compare_window = None
        self._preview_quality_window = None
        self._focus_comparison = None
        self._board_visible = False
        self._board_scanning = None
        self._releasing_preview = False
        self._preview_revision = None
        self._checking_plan = False
        self._preflight_result = None
        self._camera_settings_busy = False
        self._batch_settings_dialog = None
        self._camera_choice_dialog = None
        self._preview_adjustment = None
        self._discovering = False
        self._selected_devices = ()
        self._start_stop = None
        self._output_check = self._output_check_ticket = self._output_check_job = None
        self._output_check_active = False
        self._trigger_phase = None
        self._trigger_window = None
        self._phase_confirmation = None
        self._phase_pause = None
        self._pausable_phase = None
        self._pause_available = False
        self._run_reviewed = False
        self._trial_reads = set()
        self._trial_results = {}
        self._plan_trial_progress = None
        self._trial_progress_window = None
        self._trial_batch = None
        self._write_probe_window = None
        self._checking_rate_trial = False
        self._review_trial = None
        self._review_parent = None
        self._has_trial_reference = False
        self._discovered_ids = None
        self._review_ordinal = None
        self._review_trigger = None
        self._review_revision = 0
        self._review_phases = ()
        self._review_phase_roles = ()
        self._trigger_review = None
        self._pulse_audit = None
        self._review_busy = False
        self._checking_experiment = False
        self._preparing_analysis = ""
        self._review_deferred = None
        self._review_state = None
        self._review_is_copy = False
        self._review_name = ""
        self._review_outcome = self._review_integrity = self._review_read_ticket = None
        self._review_issue_count = 0
        self._review_read_error = ""
        self._review_frame_error = ""
        self._opening_experiment = False
        self._closing = False
        self._draft_tracking = False
        self._draft_after = None
        self._draft_saving = 0
        self._stored_draft = None
        self._restoring_draft = False
        self._offer_startup_draft = not practice and config is None and experiment is None
        self._offer_camera_start = self._offer_startup_draft
        self._initial_draft_pending = self._offer_startup_draft
        self.name = tk.StringVar()
        self.output_path = tk.StringVar(value=self._new_path())
        self.state = tk.StringVar(value="READY")
        self.state_caption = tk.StringVar(value=STATE_LABELS["READY"])
        self.activity = tk.StringVar(value="正在准备相机列表。")
        self.counter = tk.StringVar()
        self.marker_note = tk.StringVar(root)
        self.marker_status = tk.StringVar(root)
        self.provider_text = tk.StringVar()
        self.selection_text = tk.StringVar()
        self.camera_context = tk.StringVar(root, "正在查找相机")
        self.footer_detail = tk.StringVar()
        self.quality_text = tk.StringVar(value="连接预览或打开一张已保存的图像。")
        self.board_progress = tk.StringVar()
        self.board_detail = tk.StringVar()
        self.board_note = tk.StringVar()
        self._set_checkerboard_scan(None)
        self.footer_roi = tk.StringVar(value="Shift + 拖动检查局部图像")
        self.external = tk.BooleanVar()
        self.watch_quality = tk.BooleanVar(value=True)
        self.durability = tk.StringVar()
        self.guide = tk.StringVar(value="Off")
        self.palette = tk.StringVar(value="Native")
        self.review_text = tk.StringVar(value="打开已有实验，或先完成一次采集。")
        self._review_counters = {}
        self.plan_summary = tk.StringVar(value="检查计划可估算空间。")
        self._build()
        self._set_config(config or (practice_config() if practice else CaptureConfig.from_dict(demo_config())))
        self._draft_last = self._current_draft()
        self._draft_tracking = not practice
        self.phases.bind("<<TreeviewSelect>>", self._schedule_draft_save, add="+")
        for variable in (self.name, self.output_path, self.external, self.durability):
            variable.trace_add("write", self._invalidate_preflight)
        self.output_path.trace_add("write", self._schedule_output_check)
        root.bind("<Escape>", lambda _event: ToolTip.dismiss_window(root))
        root.bind("<KeyRelease-F8>", self._mark_shortcut)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(90, self._tick)
        initial_draft = self._draft_last

        def initial_draft_loaded(draft):
            self._initial_draft_pending = False
            if self._draft_last is initial_draft:
                self._draft_loaded(draft)

        if not practice:
            self._job(load_capture_draft, initial_draft_loaded, lambda _exc: initial_draft_loaded(None))
        if experiment:
            def failed(exc):
                self.discover()
                self._error(exc)

            self.open_experiment(experiment, on_opened=self.discover, on_error=failed)
        elif practice:
            self.discover(on_finished=self.toggle_preview)
        else:
            self.discover()

    def _new_path(self, parent=None):
        folder = Path(parent) if parent else Path.home() / "Documents" / "StrainTrace Capture"
        if self._practice:
            if parent is None:
                folder /= "模拟练习"
            return str(folder / (time.strftime("练习-%Y%m%d-%H%M%S") + f"-{uuid4().hex[:8]}"))
        path = folder / time.strftime("Capture-%Y%m%d-%H%M%S")
        return str(path.with_name(f"{path.name}-{uuid4().hex[:8]}") if path.exists() else path)

    def _build(self):
        layout = ttk.Frame(self.root, padding=(14, 10))
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(3, weight=1)

        title = ttk.Frame(layout)
        title.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        heading = ttk.Frame(title)
        app_title = ttk.Label(heading, text="图像采集", style="CaptureTitle.TLabel")
        app_title.pack(side=tk.LEFT)
        context = ttk.Label(heading, textvariable=self.camera_context, style="CaptureContext.TLabel")
        context.pack(side=tk.LEFT, padx=(12, 0))
        ToolTip(context, self._camera_context_help, title="当前工作区")
        menu = ttk.Frame(title)
        wrap_toolbar(title, heading, menu, gap=4)
        self._plan_menu = ttk.Frame(menu)
        self.setup_buttons = []
        for label, command, help_text in (
            ("新手指南", self.show_guide, "按步骤学习选相机、检查画面和计划、采集与复查。可从指南直接定位到对应操作。"),
            ("载入计划", self.load_plan, lambda _event: self._setup_help(
                "读取已保存的相机参数和采集阶段。正式采集时请选择新的保存目录。", needs_idle_camera=True)),
            ("恢复上次计划", self.restore_capture_draft, self._draft_help),
            ("保存计划", self.save_plan, "把当前采集设置保存为 JSON，供下次实验或 CLI 复用。"),
            ("最近实验", self.show_recent_experiments, lambda _event: self._setup_help(
                "重新打开最近采集或复查的实验。目录已移动时，可在列表内选择新位置。", needs_idle_camera=True)),
            ("打开实验", self.open_experiment, lambda _event: self._setup_help(
                "选择含采集台账的实验总目录。打开后可浏览相机图像、校验文件或进入分析。", needs_idle_camera=True)),
        ):
            if self._practice and command == self.load_plan:
                continue
            parent = self._plan_menu if command in (self.load_plan, self.restore_capture_draft, self.save_plan) else menu
            button = ttk.Button(parent, text=label, command=command, style="Small.TButton")
            button.pack(side=tk.LEFT, padx=(5, 0))
            ToolTip(button, help_text)
            if command in (self.load_plan, self.open_experiment, self.show_recent_experiments):
                self.setup_buttons.append(button)
            if command == self.restore_capture_draft:
                self.restore_draft_button = button
                button.pack_forget()
            if command == self.save_plan:
                self.save_plan_button = button
            if command == self.show_recent_experiments:
                self._recent_button = button
            if command == self.open_experiment:
                self.open_experiment_button = button
        self._plan_menu.pack(side=tk.LEFT, before=self._recent_button)

        def fit_title(_event=None):
            show_title = title.winfo_width() >= app_title.winfo_reqwidth() + context.winfo_reqwidth() + menu.winfo_reqwidth() + 20
            visible = bool(app_title.winfo_manager())
            if show_title == visible:
                return
            if show_title:
                app_title.pack(side=tk.LEFT, before=context)
            else:
                app_title.pack_forget()
            context.pack_configure(padx=(12, 0) if show_title else 0)

        for widget in (title, context, menu):
            widget.bind("<Configure>", fit_title, add="+")

        self._build_experiment_settings(layout)
        self._build_current_experiment(layout)
        self._build_next_step(layout)
        self._dock = tk.PanedWindow(layout, orient=tk.HORIZONTAL, background=BG, borderwidth=0, sashwidth=8,
                                    sashrelief=tk.FLAT, opaqueresize=True)
        self._dock.grid(row=3, column=0, sticky="nsew")
        self._device_pane, self._preview_pane, self._detail_pane = (ttk.Frame(self._dock) for _ in range(3))
        for pane, minimum, width, stretch in (
            (self._device_pane, 190, 225, "never"),
            (self._preview_pane, 390, 400, "always"),
            (self._detail_pane, 340, 340, "never"),
        ):
            self._dock.add(pane, minsize=minimum, width=width, stretch=stretch)
        devices = self._device_scroll = ScrollableForm(self._device_pane, fill_height=True)
        devices.pack(fill=tk.BOTH, expand=True)
        self._device_panel(devices.body)
        ToolTip(devices.scrollbar, "向下滚动可查看相机参数、驱动和预览操作。", title="相机设置")
        devices.bind_controls(keep_wheel=(ttk.Treeview,))
        self._preview_panel(self._preview_pane)
        self._compact_workspace = False
        self._dock_layout = None
        self._detail_panel(self._detail_pane)
        self._last_detail_tab = str(self.plan_panel)
        self._dock.bind("<Configure>", self._reflow_workspace, add="+")
        self._build_footer(layout)

    def _build_experiment_settings(self, layout):
        experiment = self._plan_settings = ttk.Frame(layout)
        experiment.grid(row=1, column=0, sticky="ew", pady=(0, 7))
        experiment.columnconfigure(3, weight=1)
        name_label = ttk.Label(experiment, text="实验名称")
        name_label.grid(row=0, column=0, sticky="w", padx=(0, 7))
        self.name_entry = ttk.Entry(experiment, textvariable=self.name, width=17)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=(0, 14))
        ToolTip(self.name_entry, lambda _event: self._setup_help(
            "填写本次计划所用的名称，例如“试样 A03 室温拉伸”。开始采集后将写入台账。"), title="实验名称")
        ToolTip(name_label, self.name_entry._tooltip.resolve_text)
        path_label = ttk.Label(experiment, text="输出目录")
        path_label.grid(row=0, column=2, sticky="w", padx=(0, 7))
        self.output_entry = ttk.Entry(experiment, textvariable=self.output_path)
        self.output_entry.grid(row=0, column=3, sticky="ew", padx=(0, 7))
        ToolTip(self.output_entry, self._output_help, title="输出目录")
        ToolTip(path_label, self.output_entry._tooltip.resolve_text)
        self.choose_output_button = ttk.Button(experiment, text="选择位置…", command=self.choose_output)
        self.choose_output_button.grid(row=0, column=4)
        ToolTip(self.choose_output_button, lambda _event: self._setup_help(
            "选择父文件夹后，程序会为本次实验生成一个新目录。检查计划前可修改目录名。"))
        storage_label = ttk.Label(experiment, text="存储模式")
        storage_label.grid(row=0, column=5, sticky="w", padx=(12, 6))
        self.storage_mode = localized_combobox(experiment, self.durability, DURABILITY_LABELS, width=8)
        self.storage_mode.grid(row=0, column=6, sticky="e")
        ChoiceToolTip(self.storage_mode, lambda _event: self._setup_help(
            "正式采集和试采使用同一设置。"),
            {label: DURABILITY_HELP[key] for key, label in DURABILITY_LABELS.items()}, title="存储模式")
        ToolTip(storage_label, self.storage_mode._tooltip.resolve_text)

        stacked = False
        path_font = Font(root=self.root, font=ttk.Style(self.root).lookup("TEntry", "font") or "TkDefaultFont")

        def fit_settings(_event=None):
            nonlocal stacked
            width = experiment.winfo_width()
            if width <= 1:
                return
            fixed = sum(widget.winfo_reqwidth() for widget in (
                name_label, self.name_entry, path_label, self.choose_output_button, storage_label, self.storage_mode,
            ))
            leaf = Path(self.output_path.get()).name
            path_width = max(220, min(360, path_font.measure(leaf) + 22), self.name_entry.winfo_reqwidth())
            wrap = width < fixed + 56 + path_width
            if wrap == stacked:
                return
            stacked = wrap
            self.name_entry.grid(columnspan=3 if wrap else 1)
            path_label.grid(row=int(wrap), column=0 if wrap else 2, pady=(4, 0) if wrap else 0)
            self.output_entry.grid(row=int(wrap), column=1 if wrap else 3,
                                   columnspan=5 if wrap else 1, pady=(4, 0) if wrap else 0)
            self.choose_output_button.grid(row=int(wrap), column=6 if wrap else 4, pady=(4, 0) if wrap else 0)
            storage_label.grid(column=4 if wrap else 5)
            self.storage_mode.grid(column=5 if wrap else 6, columnspan=2 if wrap else 1)

        def show_destination(_event=None):
            if self.output_entry.winfo_ismapped() and self.root.focus_get() is not self.output_entry:
                fit_settings()
                self.output_entry.icursor(tk.END)
                self.output_entry.xview_moveto(1.)

        experiment.bind("<Configure>", fit_settings, add="+")
        for sequence in ("<Map>", "<Configure>", "<FocusOut>"):
            self.output_entry.bind(sequence, lambda _event: self.root.after_idle(show_destination), add="+")
        self.output_path.trace_add("write", lambda *_args: self.root.after_idle(show_destination))

    def _build_current_experiment(self, layout):
        row = self._current_experiment_row = ttk.Frame(layout)
        row.grid(row=1, column=0, sticky="ew", pady=(0, 7))
        row.grid_remove()
        row.columnconfigure(1, weight=1)
        row.columnconfigure(3, weight=2)
        self._current_name = tk.StringVar(self.root)
        self._current_path = tk.StringVar(self.root)
        name_label = ttk.Label(row, text="实验名称")
        name_label.grid(row=0, column=0, sticky="w", padx=(0, 7))
        self.current_name_entry = ttk.Entry(row, textvariable=self._current_name, state="readonly", width=20)
        self.current_name_entry.grid(row=0, column=1, sticky="ew", padx=(0, 14))
        path_label = ttk.Label(row, text="实验目录")
        path_label.grid(row=0, column=2, sticky="w", padx=(0, 7))
        self.current_path_entry = ttk.Entry(row, textvariable=self._current_path, state="readonly", width=25)
        self.current_path_entry.grid(row=0, column=3, sticky="ew")
        ToolTip(self.current_name_entry, lambda _event: self._current_name.get(), title="台账中的实验名称")
        ToolTip(self.current_path_entry, lambda _event: (
            f"{self._current_path.get()}\n\n选中文字后按 Ctrl+C，可复制实验路径。"), title="实际保存的实验目录")
        ToolTip(name_label, self.current_name_entry._tooltip.resolve_text, title="台账中的实验名称")
        ToolTip(path_label, self.current_path_entry._tooltip.resolve_text, title="实际保存的实验目录")

        def show_folder_name(*_args):
            if self.root.focus_get() is not self.current_path_entry:
                self.current_path_entry.xview_moveto(1.)

        self.current_path_entry.bind("<Configure>", show_folder_name, add="+")
        self.current_path_entry.bind("<FocusOut>", show_folder_name, add="+")
        self._current_path.trace_add("write", show_folder_name)

    def _reflow_workspace(self, event):
        if event.height < 215:
            self._preview_hint.grid_remove()
        elif not self._preview_hint.winfo_manager():
            self._preview_hint.grid()
        device_width = max(225, self.cti_button.winfo_reqwidth())
        detail_width = max(340, self._plan_checks.winfo_reqwidth() + 36, self._region_actions.winfo_reqwidth() + 36)
        compact = event.width < device_width + 390 + detail_width + 16
        signature = (compact, device_width, detail_width)
        if signature == self._dock_layout:
            return
        self._dock_layout = signature
        if compact != self._compact_workspace:
            focus = self.root.focus_get()
            device_focused = focus is not None and str(focus).startswith(f"{self._device_pane}.")
            self._compact_workspace = compact
            self.restore_draft_button.configure(text="恢复计划" if compact else "恢复上次计划")
            self._recent_button.configure(text="最近" if compact else "最近实验")
            if compact:
                if self._device_pane.winfo_manager() == "panedwindow":
                    self._dock.forget(self._device_pane)
                self.tabs.insert(0, self._device_pane, text="相机", padding=8)
                if device_focused:
                    self.tabs.select(self._device_pane)
            else:
                if self.tabs.select() == str(self._device_pane):
                    self.tabs.select(self.plan_panel if device_focused else self._last_detail_tab)
                self.tabs.forget(self._device_pane)
            if device_focused:
                self.root.after_idle(focus.focus_set)
            self._update_workspace_tabs()
        self._dock.paneconfigure(self._detail_pane, width=detail_width, minsize=detail_width)
        if not compact:
            busy = self._busy()
            self._render_workspace(reviewing=not busy and self._saved_panel(), busy=busy)
            if self._device_pane.winfo_manager() == "panedwindow":
                self._dock.paneconfigure(self._device_pane, width=device_width, minsize=device_width)

    def _build_next_step(self, layout):
        row = ttk.Frame(layout, style="CaptureCard.TFrame", padding=(9, 3))
        row.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        row.columnconfigure(1, weight=1)
        self._next_step = _NextStep("相机与预览", "正在搜索采集设备。")
        self.step_title = ttk.Label(row, text=self._next_step.title, style="CaptureSelected.TLabel")
        self.step_title.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.step_hint = ttk.Label(row, text=self._next_step.hint, style="CaptureNext.TLabel", width=1, takefocus=True)
        self.step_hint.grid(row=0, column=1, sticky="ew")
        hint_font = Font(root=self.root, font=ttk.Style(self.root).lookup("CaptureNext.TLabel", "font"))

        def render_hint(_event=None):
            width = max(1, self.step_hint.winfo_width() - 3)
            hint = " ".join(self._next_step.hint.split())
            short_window = self.root.winfo_height() < 640
            if short_window:
                text, wrap = fit_line(hint, hint_font, width), 0
            else:
                text = hint if len(hint) <= 105 else f"{hint[:105].rstrip()}…（悬停查看全文）"
                wrap = width
            if self.step_hint.cget("text") != text or int(self.step_hint.cget("wraplength") or 0) != wrap:
                self.step_hint.configure(text=text, wraplength=wrap)

        self._render_step_hint = render_hint
        self.step_hint.bind("<Configure>", render_hint, add="+")
        self.root.bind("<Configure>", lambda event: render_hint() if event.widget is self.root else None, add="+")
        self.step_button = ttk.Button(row, style="Next.TButton", command=self._follow_next_step)
        self.step_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        row.rowconfigure(0, minsize=self.step_button.winfo_reqheight())
        self.step_button.grid_remove()
        for widget in (self.step_title, self.step_hint):
            ToolTip(widget, lambda _event: self._next_step.hint, title="下一步")
        ToolTip(self.step_button, self._next_step_help, title="下一步")

    def _device_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        ttk.Label(header, text="采集设备", style="CaptureSection.TLabel").pack(side=tk.LEFT)
        self.refresh_button = ttk.Button(header, text="刷新", style="Small.TButton", command=self.discover)
        self.refresh_button.pack(side=tk.RIGHT)
        ToolTip(self.refresh_button, lambda _event: self._setup_help(
            "重新查找已连接的设备。接入或重连相机后可刷新列表。", needs_idle_camera=True))
        device_list = ttk.Frame(parent)
        device_list.grid(row=1, column=0, sticky="nsew")
        device_list.columnconfigure(0, weight=1)
        device_list.rowconfigure(0, weight=1)
        self.devices = ttk.Treeview(device_list, columns=("source", "discovery"), height=3, selectmode="extended",
                                    show="tree headings")
        self.devices.heading("#0", text="相机")
        self.devices.heading("source", text="来源")
        self.devices.heading("discovery", text="查找")
        self.devices.column("#0", width=110, minwidth=66, stretch=True)
        self.devices.column("source", width=72, minwidth=58, stretch=False)
        self.devices.column("discovery", width=68, minwidth=64, stretch=False)
        self.devices.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(device_list, orient="vertical", command=self.devices.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.devices.configure(yscrollcommand=scroll.set)
        ToolTip(scroll, "滚动查看其余相机。也可聚焦列表后用方向键浏览。", title="相机列表")
        self.devices.bind("<<TreeviewSelect>>", self._camera_selection)
        ToolTip(self.devices, self._device_help, title="相机列表")
        summary = ttk.Label(parent, textvariable=self.selection_text, style="CaptureMuted.TLabel", width=1)
        summary.grid(row=2, column=0, sticky="ew", padx=2, pady=(7, 4))
        ToolTip(summary, lambda _event: self.provider_text.get(), title="相机发现结果")
        self.provider_button = ttk.Button(parent, text="驱动状态…", command=self.show_providers, style="Small.TButton")
        self.provider_button.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        ToolTip(self.provider_button, "查看各相机接口和已发现设备。实际相机没有出现时，先看哪一项缺少驱动或接口文件。")
        self.settings_button = ttk.Button(parent, text="相机参数…", command=self.edit_camera, style="Small.TButton")
        self.settings_button.grid(row=4, column=0, sticky="ew", pady=(0, 5))
        self.batch_settings_button = ttk.Button(parent, text="多机参数…", command=self.edit_camera_batch,
                                                style="Small.TButton")
        self.batch_settings_button.grid(row=5, column=0, sticky="ew", pady=(0, 5))
        self.batch_settings_button.grid_remove()
        self.cti_button = ttk.Button(parent, text="选择厂商 .cti…", command=self.choose_cti, style="Small.TButton")
        self.cti_button.grid(row=6, column=0, sticky="ew", pady=(0, 5))
        self.camera_choice_button = ttk.Button(parent, text="选择实验相机…", command=self.show_camera_choice,
                                               style="Small.TButton")
        self.camera_choice_button.grid(row=7, column=0, sticky="ew", pady=(0, 5))
        self.camera_choice_button.grid_remove()
        ToolTip(self.camera_choice_button, lambda _event: self._setup_help(
            "勾选本次想用的相机，再连接实时预览。可按型号、序列号和接口核对设备。", needs_idle_camera=True))
        connections = ttk.Frame(parent)
        connections.grid(row=8, column=0, sticky="ew")
        connections.columnconfigure(0, weight=1)
        self.preview_button = ttk.Button(connections, text="连接并预览", command=self.toggle_preview, style="Small.TButton")
        self.preview_button.grid(row=0, column=0, sticky="ew")
        self.retry_preview_button = ttk.Button(connections, text="重连相机", command=self._retry_preview, style="Small.TButton")
        self.retry_preview_button.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.retry_preview_button.grid_remove()
        ToolTip(self.retry_preview_button, self._retry_preview_help)
        ToolTip(self.settings_button, lambda _event: self._camera_settings_block() or (
            "调整当前画面的相机。回读参数并收到新画面后，修改会写入采集计划。" if self.preview else
            "读取所选相机的曝光、增益、帧率和图像尺寸。应用后可连接预览核对画面。"))
        ToolTip(self.batch_settings_button, lambda _event: self._batch_settings_block() or
                "为所选相机统一曝光、增益、像素格式或图像尺寸。逐台核验通过后才更新采集计划。",
                title="多机参数")
        ToolTip(self.cti_button, lambda _event: self._cti_block() or
            "部分工业相机要通过厂商提供的 .cti 文件连接。请从驱动原安装目录选择。"
            "刷新时也会搜索系统已经登记的文件。")
        ToolTip(self.preview_button, self._preview_button_help)

    def _preview_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        self.preview_heading = ttk.Label(header, text="画面", style="CaptureSection.TLabel")
        controls = ttk.Frame(header)
        wrap_toolbar(header, self.preview_heading, controls, gap=3, collapse_heading=True)
        ToolTip(self.preview_heading, lambda _event: {
            "上阶段": "等待确认期间显示上一阶段收到的图像。确认后才启动本阶段。",
            "待启动": "第一阶段正在等待确认。确认后才会启动相机并显示图像。",
            "已保存": "显示已登记并读取的原图。可在复查与分析页切换图像。",
            "预览": "显示相机当前画面，可用于对焦和调整光照。",
            "监看": "显示采集过程中最近一次收到的图像。",
        }.get(self.preview_heading.cget("text"), "选中相机并连接预览后可查看画面。"))
        guides = localized_combobox(controls, self.guide, {"Off": "无辅助线", "Crosshair": "十字线", "Grid": "网格"}, width=7)
        guides.pack(side=tk.RIGHT, padx=(2, 0))
        guides.bind("<<ComboboxSelected>>", self.apply_preview_options, add="+")
        ChoiceToolTip(guides, "辅助线只显示在预览上。", {
            "无辅助线": "只查看图像。", "十字线": "以中心十字辅助对中。", "网格": "用网格检查试样方向和构图。",
        })
        palette = localized_combobox(controls, self.palette, {"Native": "原图", "Inferno": "伪彩色"}, width=6)
        palette.pack(side=tk.RIGHT, padx=(2, 0))
        palette.bind("<<ComboboxSelected>>", self.apply_preview_options, add="+")
        ChoiceToolTip(palette, "色带只改变预览的显示效果。", {
            "原图": "按相机色彩或灰度显示。", "伪彩色": "用颜色强调灰度变化，方便检查光照。",
        })
        self.preview_controls = [guides, palette]
        for label, mode, description in (
            ("1:1", "one_to_one", "按原始像素比例检查纹理和边缘。拖动可浏览局部。"),
            ("适应", "fit", "把完整图像缩放到当前窗口内。"),
        ):
            button = ttk.Button(controls, text=label, style="Small.TButton",
                                command=lambda method=mode: self._viewport_action(method))
            button.pack(side=tk.RIGHT, padx=(2, 0))
            ToolTip(button, description)
            self.preview_controls.append(button)
        self.preview_area = ttk.Frame(parent)
        self.preview_area.grid(row=1, column=0, sticky="nsew")
        self.empty_preview = ttk.Label(self.preview_area, text="请选择相机，再连接预览", anchor="center",
                                       style="CaptureCard.TLabel")
        self.empty_preview.grid(row=0, column=0, sticky="nsew")
        ToolTip(self.empty_preview, lambda _event: self._workflow_hint(), title="图像预览")
        self.image_part_row = ttk.Frame(parent)
        self.image_part_row.grid(row=2, column=0, sticky="ew", pady=(7, 0))
        self.image_part_row.grid_remove()
        self.image_part_row.columnconfigure(1, weight=1)
        part_label = ttk.Label(self.image_part_row, text="主图像")
        part_label.grid(row=0, column=0, padx=(0, 8))
        self.image_part_menu = ttk.Combobox(self.image_part_row, state="readonly", width=12)
        self.image_part_menu.grid(row=0, column=1, sticky="ew")
        self.image_part_menu.bind("<<ComboboxSelected>>", self._choose_preview_part)
        self._part_menu_signature = None
        self._part_selectors = {}
        self._part_help = ChoiceToolTip(
            self.image_part_menu,
            "选择预览和默认分析使用的主图像。收到新画面后，所选部件会写入采集计划。",
            {}, title="选择 GenTL 主图像",
        )
        ToolTip(part_label, self._part_help.resolve_text)
        self._save_all_2d = tk.BooleanVar(self.root)
        self.archive_parts = ttk.Checkbutton(self.image_part_row, text="保存全部二维部件",
                                             variable=self._save_all_2d, command=self._choose_part_storage)
        self.archive_parts.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ToolTip(self.archive_parts, self._part_storage_help, title="同一帧保存多张原图")
        footer = ttk.Frame(parent)
        footer.grid(row=3, column=0, sticky="ew", pady=(7, 0))
        footer.columnconfigure(0, weight=1)
        region = self._preview_hint = ttk.Label(footer, textvariable=self.footer_roi, width=1,
                                                style="CaptureMuted.TLabel")
        region.grid(row=0, column=0, sticky="ew")
        ToolTip(region, lambda _event: (
            "绿色圆点只显示在同一张已保存且识别成功的棋盘格原图上。" if self._board_visible else
            "x、y 是方框左上角的坐标，以原图左上角为原点。宽、高和坐标都用原图像素表示。"))
        self.save_preview_button = ttk.Button(footer, text="保存预览原图…", command=self._save_preview_image,
                                              style="Small.TButton")
        self.compare_preview_button = ttk.Button(footer, text="原图对照…", command=self._open_preview_compare,
                                                 style="Small.TButton")
        self.compare_preview_button.grid(row=0, column=1, padx=(6, 0), sticky="e")
        self.save_preview_button.grid(row=0, column=2, padx=(6, 0), sticky="e")
        self.save_preview_button.state(["disabled"])
        ToolTip(self.save_preview_button, self._preview_save_help)
        ToolTip(self.compare_preview_button, lambda _event: (
            "采集结束后可打开预览原图进行对照。" if self._busy() else
            "打开刚保存的预览原图，与同一相机的当前帧或另一张原图对照。" if self._last_preview_original else
            "打开之前保存的预览 TIFF。可对照相机当前帧，也可离线打开两张图。"))

    def _detail_panel(self, parent):
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.plan_panel = ttk.Frame(self.tabs, padding=8)
        self.quality_panel = ttk.Frame(self.tabs, padding=8)
        self.review_panel = ttk.Frame(self.tabs, padding=(8, 8, 5, 8))
        self.tabs.add(self.plan_panel, text="采集计划")
        self.tabs.add(self.quality_panel, text="图像质量")
        self.tabs.add(self.review_panel, text="复查与分析")
        self.tabs.enable_traversal()
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        ToolTip(self.tabs, self._workspace_tab_help)
        self.plan_panel.columnconfigure(0, weight=1)
        self.plan_panel.rowconfigure(0, weight=1)
        self.plan_editor = ScrollableForm(self.plan_panel, fill_height=True)
        self.plan_editor.grid(row=0, column=0, sticky="nsew")
        self._build_plan(self.plan_editor.body)
        ToolTip(self.plan_editor.scrollbar, "向下滚动可查看阶段编辑、检查计划和试采。", title="采集计划")
        self.plan_editor.bind_controls(keep_wheel=(ttk.Treeview,))
        self._monitor_scroll = ScrollableForm(self.plan_panel, fill_height=True)
        self._monitor_scroll.grid(row=0, column=0, sticky="nsew")
        self._monitor_scroll.grid_remove()
        self.run_monitor = CaptureMonitor(self._monitor_scroll.body, self._view_title, self.select_view)
        self.run_monitor.pack(fill=tk.BOTH, expand=True)
        markers = ttk.Frame(self._monitor_scroll.body, padding=(0, 8, 0, 2))
        markers.pack(fill=tk.X)
        ttk.Label(markers, text="标记说明（选填）").pack(anchor="w")
        note = ttk.Entry(markers, textvariable=self.marker_note, width=1)
        note.pack(fill=tk.X, pady=(3, 0))
        ToolTip(note, "可预填最多 160 个字。看到变化时按 F8 或点“标记事件”，采集继续进行。")
        feedback = ttk.Label(markers, textvariable=self.marker_status, width=1, style="CaptureMuted.TLabel")
        feedback.pack(fill=tk.X)
        wrap_label_on_resize(feedback)
        ToolTip(feedback, lambda _event: self.marker_status.get() or "这里只显示本窗口提交的标记。")
        ToolTip(self._monitor_scroll.scrollbar, "向下滚动可查看各相机的接收、待存和写入趋势。", title="采集监看")
        self._monitor_scroll.bind_controls(keep_wheel=(ttk.Treeview,))
        self._build_quality()
        self._build_review()

    def _workspace_tab_help(self, event):
        help_by_tab = {
            str(self._device_pane): "选择相机、连接预览或检查设备参数。窗口加宽后，这些操作会回到左侧。",
            str(self.plan_panel): ("查看各台相机的收图、已保存和待存帧数。选中一行可看时间和状态原因。"
                                   if self._monitor_visible else TAB_HELP[0]),
            str(self.quality_panel): ("核对保存后的棋盘格角点、不同摆位和双相机配对。"
                                      if self._board_visible else TAB_HELP[1]),
            str(self.review_panel): TAB_HELP[2],
        }
        return notebook_help(self.tabs, tuple(help_by_tab[tab] for tab in self.tabs.tabs()), event)

    def _update_workspace_tabs(self):
        if self._compact_workspace:
            plan, quality, review = ("监看" if self._monitor_visible else "计划",
                                     "标定" if self._board_visible else "画质", "复查")
        else:
            plan, quality, review = ("采集监看" if self._monitor_visible else "采集计划",
                                     "棋盘格检查" if self._board_visible else "图像质量", "复查与分析")
        self.tabs.tab(self.plan_panel, text=plan)
        self.tabs.tab(self.quality_panel, text=quality)
        self.tabs.tab(self.review_panel, text=review)

    def _build_plan(self, panel):
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        heading = ttk.Frame(panel)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(heading, text="采集阶段", style="CaptureSection.TLabel").pack(side=tk.LEFT)
        self.template_button = ttk.Button(heading, text="采集模板…", command=self.show_template, style="Small.TButton")
        self.template_button.pack(side=tk.RIGHT)
        ToolTip(self.template_button, lambda _event: self._setup_help(
            "按任务比较不同模板。应用前可改频率和时长，查看阶段顺序与预计保存量。"), title="采集模板")
        phase_list = ttk.Frame(panel)
        phase_list.grid(row=1, column=0, sticky="nsew")
        phase_list.columnconfigure(0, weight=1)
        phase_list.rowconfigure(0, weight=1)
        self.phases = ttk.Treeview(phase_list, columns=("role", "limits"), show="tree headings",
                                  height=2, selectmode="browse")
        self.phases.heading("#0", text="拍摄方式")
        self.phases.heading("role", text="用途")
        self.phases.heading("limits", text="计划参数")
        self.phases.column("#0", width=104, minwidth=75)
        self.phases.column("role", width=88, minwidth=62)
        self.phases.column("limits", width=107, minwidth=80)
        self.phases.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(phase_list, orient="vertical", command=self.phases.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.phases.configure(yscrollcommand=scroll.set)
        ToolTip(scroll, "滚动查看其余阶段。阶段按由上到下的顺序执行。", title="采集阶段")
        self.phases.bind("<Double-Button-1>", self._edit_clicked_phase)
        self.phases.bind("<Return>", lambda _event: self.edit_phase())
        ToolTip(self.phases, self._phase_help)
        style = ttk.Style(self.root)
        self._phase_font = Font(root=self.root, font=style.lookup("Treeview", "font") or "TkDefaultFont")
        self._phase_heading_font = Font(root=self.root, font=style.lookup("Treeview.Heading", "font") or "TkDefaultFont")
        self._phase_layout = None
        self.phase_details = ttk.Label(panel, style="CaptureMuted.TLabel", width=1, justify=tk.LEFT, takefocus=True)
        self.phase_details.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.phase_details.grid_remove()
        wrap_label_on_resize(self.phase_details)
        ToolTip(self.phase_details, lambda _event: self._phase_help(None), title="选中阶段")
        self.phases.bind("<Configure>", self._fit_phase_columns, add="+")
        self.phases.bind("<<TreeviewSelect>>", self._fit_phase_columns, add="+")
        actions = ttk.Frame(panel)
        actions.grid(row=3, column=0, sticky="ew", pady=(5, 4))
        self.plan_buttons = []
        for text, action, help_text in (
            ("添加", lambda: self.edit_phase(add=True), "新增一个阶段，例如在加载前拍一张参考图。"),
            ("编辑", self.edit_phase, "修改选中阶段的拍摄方式、图像用途和结束条件。"),
            ("删除", self.remove_phase, "从待运行计划中移除选中的阶段。"),
            ("↑", lambda: self.move_phase(-1), "把选中的阶段移到前面执行。"),
            ("↓", lambda: self.move_phase(1), "把选中的阶段移到后面执行。"),
        ):
            button = ttk.Button(actions, text=text, style="Small.TButton", command=action)
            button.pack(side=tk.LEFT, padx=(0, 3))
            self.plan_buttons.append(button)
            ToolTip(button, lambda _event, label=text, tip=help_text: self._phase_block(label) or tip,
                    title={"↑": "提前执行", "↓": "延后执行"}.get(text, text))
        trial_row = ttk.Frame(panel)
        trial_row.grid(row=4, column=0, sticky="ew", pady=(1, 3))
        acquisition_options = ttk.Frame(trial_row)
        self.external_box = ttk.Checkbutton(acquisition_options, text="接收软件触发", variable=self.external)
        ToolTip(self.external_box, lambda _event: self._setup_help(
            "已配置数据采集系统时，可通过实验目录接收本机软件发来的信号。单击“触发一次”无需勾选。"
            "相机接线上的物理触发请使用“硬件触发”阶段。"))
        self.watch_quality_box = ttk.Checkbutton(acquisition_options, text="画质提醒", variable=self.watch_quality)
        self.external_box.pack(side=tk.LEFT, padx=(0, 8))
        self.watch_quality_box.pack(side=tk.LEFT)
        ToolTip(self.watch_quality_box,
                "正式采集中抽查已经入队的主图。\n"
                "可在画质页按相机保存检查框，作为抽查范围。\n"
                "画面持续偏暗、近饱和或清晰度下降时，提示对应相机。\n"
                "图像继续保存，采后可在时间线定位抽查帧。", title="画质提醒")
        self.trial_progress_button = ttk.Button(trial_row, text="阶段试采…", command=self.show_trial_progress,
                                                style="Small.TButton")
        wrap_toolbar(trial_row, acquisition_options, self.trial_progress_button)
        ToolTip(self.trial_progress_button, lambda _event: self._plan_input_block() or
                "按阶段查看已有试采、设置变化和复核提醒。可直接进入建议阶段的试采确认。", title="阶段试采")
        checks = self._plan_checks = ttk.Frame(panel)
        checks.grid(row=5, column=0, sticky="ew", pady=(3, 5))
        self.preflight_button = ttk.Button(checks, text="检查计划", command=self.check_plan, style="Small.TButton")
        self.preflight_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ToolTip(self.preflight_button, lambda _event: self._plan_input_block() or (
            "核对相机的触发能力、输出目录和磁盘空间。预览中的图像可用于估算数据量。"
            "检查只读取设备信息并生成报告。"))
        self.trial_button = ttk.Button(checks, text="试采…", command=self.start_trial, style="Small.TButton")
        self.trial_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ToolTip(self.trial_button, self._trial_help)
        self.board_button = ttk.Button(checks, text="标定采图…", command=self.start_checkerboard, style="Small.TButton")
        self.board_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(self.board_button, lambda _event: self._checkerboard_setup_block() or (
            "选择一台或两台相机，填写棋盘格尺寸。每次触发保存原图，随后检查角点和摆位。标定图像存入独立目录。"))
        summary = ttk.Label(panel, textvariable=self.plan_summary, style="CaptureMuted.TLabel", width=1)
        summary.grid(row=6, column=0, sticky="ew", pady=(0, 4))
        wrap_label_on_resize(summary)
        ToolTip(summary, lambda _event: self.plan_summary.get(), title="采前检查状态")

    def _build_quality(self):
        quality = self._quality_scroll = ScrollableForm(self.quality_panel, fill_height=True)
        quality.pack(fill=tk.BOTH, expand=True)
        ToolTip(quality.scrollbar, "向下滚动可查看指标、检查各相机和打开曝光建议。", title="图像质量")
        panel = self.quality_metrics = ttk.Frame(quality.body)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(3, weight=1)
        header = ttk.Frame(panel)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.hist_heading = ttk.Label(header, text="亮度直方图", style="CaptureSection.TLabel")
        self.focus_button = ttk.Button(header, text="开始对焦比较", command=self._toggle_focus_comparison, style="Small.TButton")
        wrap_toolbar(header, self.hist_heading, self.focus_button)
        ToolTip(self.focus_button, self._focus_help, title="对焦比较")
        self.hist = tk.Canvas(panel, background=ENTRY, height=82, highlightthickness=1, highlightbackground=BORDER)
        self.hist.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.hist.configure(takefocus=True)
        ToolTip(self.hist, "横轴是原图中从暗到亮的码值，纵轴是各区间的相对像素数。有检查框时只统计框内。"
                "彩色直方图按亮度统计，单个颜色通道的饱和可看下方比例。")
        self.focus_feedback = ttk.Label(panel, width=1, style="CaptureMuted.TLabel")
        self.focus_feedback.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.focus_feedback.grid_remove()
        ToolTip(self.focus_feedback, "当前值取最近三张新画面的中位数。最高值是本轮已观测的最高响应。"
                "调整时保持试样、照明、曝光和增益不变。", title="对焦比较结果")
        detail = ttk.Frame(panel)
        detail.grid(row=3, column=0, sticky="nsew")
        text = tk.Text(detail, height=1, width=1, bg=BG, fg=FG, font=(choose_ui_font_family(self.root), 9), borderwidth=0,
                       highlightthickness=0, wrap="word", state="disabled")

        def metric_help(event):
            if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
                index = text.index(f"@{event.x},{event.y}")
                line = text.get(f"{index} linestart", f"{index} lineend")
                for name, help_text in QUALITY_HELP.items():
                    if line.startswith(name):
                        return help_text
            return "停在指标名称上可查看含义。散斑是试样表面用于追踪变形的明暗纹理，按住 Shift 拖出方框可检查局部。"

        ToolTip(text, metric_help, title="图像质量")
        scrollbar = ttk.Scrollbar(detail, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ToolTip(scrollbar, "滚动查看亮度、对焦和局部纹理数据。悬停在指标名称上可了解它的含义。", title="图像质量")
        text.pack(fill=tk.BOTH, expand=True)

        def update_text(*_args):
            scroll = text.yview()[0]
            text.configure(state="normal")
            text.delete("1.0", tk.END)
            text.insert("1.0", self.quality_text.get())
            text.configure(state="disabled")
            text.yview_moveto(scroll)

        self.quality_text.trace_add("write", update_text)
        update_text()
        self.hist.bind("<Configure>", lambda _event: setattr(self, "_quality_due", 0.))
        region_buttons = self._region_actions = ttk.Frame(panel)
        region_buttons.grid(row=4, column=0, sticky="ew", pady=(7, 0))
        self.region_buttons = []
        for label, command, description in (
            ("输入区域", self.set_region, "填写左上角 x、y 和宽、高，单位均为像素。也可按住 Shift 拖出矩形。"),
            ("清除区域", self.clear_region, "移除检查框，改为查看全图的曝光和对焦数据。"),
            ("复制坐标", self.copy_region, "把左上角 x、y 和宽、高复制到剪贴板，供应变分析复用。"),
        ):
            button = ttk.Button(region_buttons, text=label, command=command, style="Small.TButton")
            button.pack(side=tk.LEFT, padx=(0, 4))
            self.region_buttons.append((label, button))
            ToolTip(button, lambda _event, action=label, tip=description: self._region_help(action, tip))
        region_plan = ttk.Frame(panel)
        region_plan.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        region_actions = ttk.Frame(region_plan)
        region_actions.pack(fill=tk.X)
        self.quality_region_button = ttk.Button(region_actions, text="用于采集抽查", style="Small.TButton",
                                                command=self._save_quality_region)
        self.quality_region_button.pack(side=tk.LEFT)
        ToolTip(self.quality_region_button, lambda _event: self._quality_region_candidate()[1] or
                "将当前相机的检查框保存到采集计划。试采和正式采集统计框内像素，并保存相机原图。")
        self.quality_region_clear = ttk.Button(region_actions, text="恢复全图抽查", style="Small.TButton",
                                              command=self._clear_quality_region)
        ToolTip(self.quality_region_clear, lambda _event: self._setup_block() or
                "移除这台相机的计划检查框。随后采集使用整幅原图做画质抽查。")
        self.quality_region_note = ttk.Label(region_plan, width=1, style="CaptureMuted.TLabel", justify=tk.LEFT)
        wrap_label_on_resize(self.quality_region_note)
        ToolTip(self.quality_region_note, lambda _event: self.quality_region_note.cget("text"))
        self.preview_quality_button = ttk.Button(panel, text="检查各相机…", command=self.show_preview_quality,
                                                 style="Small.TButton")
        self.preview_quality_button.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        self.preview_quality_button.grid_remove()
        self.preview_quality_button.state(["disabled"])
        ToolTip(self.preview_quality_button, lambda _event: self._preview_quality_block() or (
            "每台相机读取 5 张新预览图，列出近黑、近饱和和框内纹理提醒。选中相机后点“查看画面”。"),
            title="多机画质检查")
        self.exposure_button = ttk.Button(panel, text="曝光建议…", command=self.show_exposure_advice, style="Small.TButton")
        self.exposure_button.grid(row=7, column=0, sticky="ew", pady=(6, 0))
        ToolTip(self.exposure_button, lambda _event: self._exposure_block() or (
            "测量全图或检查框里的原始亮度，建议曝光时间。可在试设后对照新旧画面，再将通过复测的曝光写入计划。"),
            title="曝光建议")
        self._build_checkerboard_panel()
        quality.bind_controls(keep_wheel=(tk.Text,))

    def _build_checkerboard_panel(self):
        panel = self.board_panel = ttk.Frame(self._quality_scroll.body)
        self.board_fit_button = ttk.Button(panel, text="拟合镜头参数…", command=self.fit_calibration, style="Small.TButton")
        self.board_fit_button.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        ToolTip(self.board_fit_button, lambda _event: self._fit_block() or "使用已记录的标定板尺寸和原始图像拟合，另存镜头标定文件。")
        for variable, style, help_text in (
            (self.board_progress, "CaptureSection.TLabel", "按默认要求至少拍摄 6 个不同摆位。双相机按有效配对中两侧较少的不同摆位数计。"),
            (self.board_detail, "TLabel", "相机下方显示识别张数和保存张数。角点跨度是可用于拟合的角点在原图宽、高方向的分布范围。"),
            (self.board_note, "CaptureMuted.TLabel", "绿色圆点标出最新已检查图像上的内角点。可把棋盘格移到其他区域并改变倾角，补充新的视图。"),
        ):
            label = ttk.Label(panel, textvariable=variable, style=style, width=1, justify=tk.LEFT)
            label.pack(fill=tk.X, pady=(0, 6))
            wrap_label_on_resize(label)
            help_tip = (lambda _event, tip=help_text: f"{tip}\n\n{self.board_detail.get()}") if variable is self.board_detail else help_text
            ToolTip(label, help_tip, title="棋盘格检查")

    def _build_review(self):
        self.review_scroll = ScrollableForm(self.review_panel)
        self.review_scroll.pack(fill=tk.BOTH, expand=True)
        panel = self.review_scroll.body
        for widget in (self.review_scroll.canvas, self.review_scroll.scrollbar):
            ToolTip(widget, "向下滚动可查看台账校验、续采和镜头标定。聚焦后可按 Page Up 或 Page Down 翻页。",
                    title="复查与分析")
        heading = ttk.Frame(panel)
        heading.pack(fill=tk.X, pady=(0, 5))
        self.review_heading = ttk.Label(heading, text="已保存图像", style="CaptureSection.TLabel")
        self.review_heading.pack(side=tk.LEFT)
        self.related_button = ttk.Button(heading, text="关联实验…", command=self.show_related_experiments,
                                         style="Small.TButton")
        self.related_button.pack(side=tk.RIGHT)
        ToolTip(self.related_button, lambda _event: self._review_help(
            "查看原实验、后续实验和各自保存的帧数。实验移过位置时，可添加搜索目录找到它。", needs_idle=True))
        self.review_result = ttk.Label(panel, style="CaptureMuted.TLabel", width=1, justify=tk.LEFT, takefocus=True)
        wrap_label_on_resize(self.review_result)
        ToolTip(self.review_result, self._review_result_help, title="采集结果")
        self.review_report_button = ttk.Button(panel, text="查看校验报告…", command=self.show_review_integrity,
                                               style="Small.TButton")
        ToolTip(self.review_report_button, lambda _event: self._review_help(
            "阅读本次校验指出的图像、标定或触发记录问题。", needs_idle=True))
        detail = self.review_image_detail = ttk.Label(
            panel, textvariable=self.review_text, justify=tk.LEFT, width=1, wraplength=245, takefocus=True,
        )
        detail.pack(fill=tk.X, pady=(0, 6))
        wrap_label_on_resize(detail)
        ToolTip(detail, self._saved_frame_help, title="图像信息与实验目录")
        self.repeat_button = ttk.Button(panel, text="准备复拍…", command=self.show_repeat_plan, style="Small.TButton")
        self.repeat_button.pack(fill=tk.X, pady=(0, 6))
        ToolTip(self.repeat_button, lambda _event: self._repeat_block() or
                "沿用这次采集的完整计划，为另一组试样填写实验名称和新的保存位置。", title="准备复拍")
        row = ttk.Frame(panel)
        row.pack(fill=tk.X)
        self.review_navigation = []
        for label, command, description in (
            ("上一张", lambda: self.browse_frame(-1), "查看同一台相机在此之前保存的一张图像。"),
            ("下一张", lambda: self.browse_frame(1), "查看同一台相机的下一张已保存图像。"),
            ("最新", lambda: self.browse_frame(latest=True), "跳到选中相机的最新已保存图像。"),
        ):
            button = ttk.Button(row, text=label, command=command, style="Small.TButton")
            button.pack(side=tk.LEFT, padx=(0, 4))
            self.review_navigation.append(button)
            ToolTip(button, lambda _event, tip=description: self._review_help(tip, needs_camera=True, browsing=True))
        self.frame_search_button = ttk.Button(panel, text="查找图像…", command=self.show_frame_search, style="Small.TButton")
        self.frame_search_button.pack(fill=tk.X, pady=(5, 0))
        ToolTip(self.frame_search_button, lambda _event: self._frame_search_block() or
                "按相机、采集阶段、用途、序号或主机收图时间查找，选定后可校验并打开原图。")
        self._review_part = 0
        self._review_part_choices = {}
        self.review_part_row = ttk.Frame(panel)
        part_label = ttk.Label(self.review_part_row, text="本帧图像")
        part_label.pack(anchor="w")
        self.review_part_menu = ttk.Combobox(self.review_part_row, state="readonly", width=14)
        self.review_part_menu.pack(fill=tk.X, pady=(2, 0))
        self.review_part_menu.bind("<<ComboboxSelected>>", self._choose_review_part)
        self._review_part_help = ChoiceToolTip(
            self.review_part_menu,
            lambda _event: self._analysis_help(
                "切换同一帧中保存的原图。往前或往后浏览时保留所选部件。参考帧和实验帧都使用这一路。"),
            {}, title="复查这一帧的图像部件",
        )
        ToolTip(part_label, self._review_part_help.resolve_text)
        self.analysis_buttons = []
        ttk.Separator(panel).pack(fill=tk.X, pady=(9, 5))
        ttk.Label(panel, text="应变分析", style="CaptureSection.TLabel").pack(anchor="w")
        analysis = ttk.Frame(panel)
        analysis.pack(fill=tk.X, pady=(5, 0))
        analysis.columnconfigure((0, 1), weight=1, uniform="analysis")
        for column, (label, live, description) in enumerate((
            ("离线分析 ↗", False, "打开分析工作台，选择要测量的区域。用变形前的参考图和后续实验图逐帧计算应变，可导出结果。"),
            ("实时分析 ↗", True, "打开新窗口，跟踪当前相机已保存的较新图像。在该窗口暂停跟踪时，采集仍继续。"),
        )):
            button = ttk.Button(analysis, text=label, command=lambda enabled=live: self.open_analysis(enabled),
                                style="Small.TButton")
            button.grid(row=0, column=column, sticky="ew", padx=(0, 5) if column == 0 else 0)
            self.analysis_buttons.append((button, True))
            ToolTip(button, lambda _event, tip=description, follow=live: self._analysis_help(
                ("续采可先载入原实验参考图，在新实验图保存后接续计算。" if follow else
                 "续采可并排查看原参考图和本轮首张实验图，核对后选择应变基准。")
                if self._review_parent else tip))
        self.history_button = ttk.Button(analysis, text="历史曲线…", command=self.show_live_history, style="Small.TButton")
        self.history_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ToolTip(self.history_button, lambda _event: self._review_help("") or
                "打开本实验保存的应变、载荷和质量曲线。可按帧号定位缺测或核对原图。")
        self.offline_analysis_button = self.analysis_buttons[0][0]
        self.corrected_analysis_button = ttk.Button(panel, text="校正后分析…", command=self.open_corrected_analysis,
                                                    style="Small.TButton")
        self.corrected_analysis_button.pack(fill=tk.X, pady=(5, 0))
        ToolTip(self.corrected_analysis_button, lambda _event: self._analysis_help(
            "用匹配所选相机和图像部件的镜头标定消除畸变，再打开应变分析。可在窗口中选择参考图和标定文件。",
            needs_idle=True), title="校正后分析")
        self.sequence_export_button = ttk.Button(panel, text="导出图像序列…", command=self.show_sequence_export,
                                                style="Small.TButton")
        self.sequence_export_button.pack(fill=tk.X, pady=(5, 0))
        ToolTip(self.sequence_export_button, lambda _event: self._sequence_export_block() or (
            "将参考图与所选实验帧按顺序导出，供其他 DIC 软件使用。保留原始 TIFF、帧序号和收图时间。"),
            title="导出图像序列")

        ttk.Separator(panel).pack(fill=tk.X, pady=(9, 5))
        ttk.Label(panel, text="数据检查与恢复", style="CaptureSection.TLabel").pack(anchor="w")
        self.timeline_button = ttk.Button(panel, text="采集时间线…", command=self.show_timeline, style="Small.TButton")
        self.timeline_button.pack(fill=tk.X, pady=(5, 0))
        ToolTip(self.timeline_button, lambda _event: self._review_help("") or
                "查看各阶段和原图的时间分布。将“事件类型”设为“仅看异常”，再按相机和阶段查漏拍、缺帧及写入错误。",
                title="采集时间线")
        self.verify_button = ttk.Button(panel, text="校验图像和台账", command=self.verify, style="Small.TButton")
        self.verify_button.pack(fill=tk.X, pady=(5, 0))
        self.analysis_buttons.append((self.verify_button, False))
        ToolTip(self.verify_button, lambda _event: self._review_help(
            "逐张核对文件大小和校验值，找出丢失、损坏或未登记的图像。有交接清单时，同时核验副本及附件。",
            needs_idle=True))
        self.archive_quality_button = ttk.Button(panel, text="检查原图画质…", command=self.show_archive_quality,
                                                  style="Small.TButton")
        self.archive_quality_button.pack(fill=tk.X, pady=(5, 0))
        ToolTip(self.archive_quality_button, lambda _event: self._archive_quality_block() or
                "按相机和阶段抽查已保存的主图像。逐张检查可以定位短暂的近黑或近饱和，并打开对应原图。")
        self.trial_comparison_button = ttk.Button(panel, text="试采与正式设置…", command=self.show_trial_comparison,
                                                  style="Small.TButton")
        ToolTip(self.trial_comparison_button, lambda _event: self._review_help(
            "查看采集前关联的试采及各阶段实际回读。可按实验编号找回试采原图。", needs_idle=True))
        self.copy_button = ttk.Button(panel, text="制作交接副本…", command=self.show_experiment_copy, style="Small.TButton")
        self.copy_button.pack(fill=tk.X, pady=(5, 0))
        ToolTip(self.copy_button, lambda _event: self._copy_block() or (
            "把原始 TIFF、采集台账和归档标定复制到新目录。可附上载荷或触发记录。核验通过后可直接打开副本。"))
        self.trigger_actions = ttk.Frame(panel)
        self.trigger_actions.pack(fill=tk.X, pady=(5, 0))
        self.trigger_actions.columnconfigure((0, 1), weight=1, uniform="triggers")
        self.trigger_review_button = ttk.Button(self.trigger_actions, text="触发组复查…", command=self.open_trigger_review,
                                                style="Small.TButton")
        self.trigger_review_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ToolTip(self.trigger_review_button, lambda _event: self._review_help(
            "并排查看同一次触发对应的各相机原图，检查缺帧、重复图像和电脑收到图像的时间。",
            browsing=True), title="触发组复查")
        self.pulse_audit_button = ttk.Button(self.trigger_actions, text="核对脉冲…", command=self.open_pulse_audit,
                                             style="Small.TButton")
        self.pulse_audit_button.grid(row=0, column=1, sticky="ew")
        ToolTip(self.pulse_audit_button, lambda _event: self._review_help(
            "使用数据采集设备（DAQ）的脉冲记录核对硬件触发。报告可找出缺帧和编号冲突，并可用脉冲编号关联探测器图像。",
            needs_idle=True), title="核对独立脉冲记录")
        recovery = self.review_recovery = ttk.Frame(panel)
        recovery.pack(fill=tk.X, pady=(7, 0))
        self.recover_button = ttk.Button(recovery, text="检查中断…", command=self.recover_run, style="Small.TButton")
        self.continue_button = ttk.Button(recovery, text="准备续采", command=self.prepare_continuation, style="Small.TButton")
        for button, description in (
            (self.recover_button, lambda _event: self._review_help(
                "先只读核对帧写入记录与遗留 TIFF，显示哪些帧可以补登记。点“恢复台账”才会写入原实验。", needs_idle=True)),
            (self.continue_button, lambda _event: self._review_help(
                "查看试采实测帧率、缓冲使用量和图像校验结果。" if self._review_trial else
                (self._fit_block() or "使用本实验记录的标定板尺寸和设备顺序拟合镜头参数。") if self._board_scan else
                "查看各阶段的保存情况并选择续采起点。帧写入记录有待核对时，会先打开中断检查。", needs_idle=True)),
        ):
            button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            ToolTip(button, description)
        actions = {
            "用棋盘格图像标定…": (self.fit_calibration, "从所选图像部件的棋盘格帧拟合镜头参数。标定采图时登记的尺寸会自动带入。"),
            "载入标定文件…": (self.load_calibration, "把已有标定加入下次采集，程序会随新实验归档一份。"),
            "切换畸变校正预览": (self.toggle_calibration, "按当前相机、尺寸和图像部件查找匹配的标定。再次选择可切回原图。"),
            "从计划移除标定": (self.clear_calibration, "下次采集将不引用已加载的标定，磁盘上的标定文件仍可重新载入。"),
        }
        calibration = ttk.Combobox(panel, values=tuple(actions), state="readonly")
        calibration.set("镜头标定…")
        calibration.pack(fill=tk.X, pady=(8, 0))

        def choose_calibration(_event):
            action = actions.get(calibration.get())
            calibration.set("镜头标定…")
            if action:
                action[0]()

        calibration.bind("<<ComboboxSelected>>", choose_calibration)
        ChoiceToolTip(calibration, lambda _event: self._setup_help(
            "从棋盘格图像求镜头参数，或载入已有标定检查畸变校正预览。"),
            {label: tip for label, (_, tip) in actions.items()}, title="镜头标定")
        self.calibration_menu = calibration
        self.review_scroll.bind_controls()

    def _build_footer(self, layout):
        bottom = ttk.Frame(layout)
        bottom.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(1, weight=1)
        self.state_label = ttk.Label(bottom, textvariable=self.state_caption, style="CaptureState.TLabel")
        self.state_label.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 8))
        ToolTip(self.state_label, self._activity_help, title="当前采集状态")
        activity = ttk.Label(bottom, width=1, justify=tk.LEFT, takefocus=True)
        activity.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        activity_font = Font(root=self.root, font=ttk.Style(self.root).lookup("TLabel", "font"))

        def show_activity(*_args):
            caption = fit_line(" ".join(self.activity.get().split()), activity_font, activity.winfo_width() - 3)
            if activity.cget("text") != caption:
                activity.configure(text=caption)

        activity.bind("<Configure>", show_activity, add="+")
        self.activity.trace_add("write", show_activity)
        ToolTip(activity, self._activity_help)
        statistics = self._footer_statistics = ttk.Label(bottom, textvariable=self.footer_detail, takefocus=True,
                                                         style="CaptureMuted.TLabel", width=1)
        statistics.grid(row=1, column=1, columnspan=5, sticky="ew")
        ToolTip(statistics, self._statistics_help)
        self.trigger_button = ttk.Button(bottom, text="触发一次", width=0, command=self.trigger)
        self.pause_button = ttk.Button(bottom, text="暂停采集", width=0, command=self.toggle_phase_pause)
        self.stop_button = ttk.Button(bottom, text="停止并保存", width=0, command=self.stop)
        self.record_button = ttk.Button(bottom, text="开始采集", width=0, style="Record.TButton", command=self.start)
        for identifier, button in enumerate((self.trigger_button, self.pause_button, self.stop_button, self.record_button), 2):
            button.grid(row=0, column=identifier, padx=(6, 0), sticky="ns")
        self.pause_button.grid_remove()
        self.return_to_plan_button = ttk.Button(
            bottom, text="返回采集计划", style="Next.TButton", command=self._return_to_plan,
        )
        self.return_to_plan_button.grid(row=0, column=5, padx=(6, 0), sticky="ns")
        self.return_to_plan_button.grid_remove()
        self.marker_button = ttk.Button(bottom, text="标记事件", width=0, command=self.mark_event, style="Record.TButton")
        self.marker_button.grid(row=0, column=5, padx=(6, 0), sticky="ns")
        self.marker_button.grid_remove()
        self.marker_button.state(["disabled"])
        ToolTip(self.marker_button, lambda _event: self._marker_block() or
                "点“标记事件”或按 F8，记录当前时刻。\n"
                "采集结束后，在“采集时间线”选择标记。\n"
                "可查看各相机在标记前后的原图。", title="标记事件")
        ToolTip(self.trigger_button, self._trigger_help)
        ToolTip(self.pause_button, self._pause_help)
        ToolTip(self.stop_button, lambda _event: (
            "结束采集，已入队的图像会继续保存。\n"
            "可在“采集监看”的“待存”栏查看尚未登记的帧数。" if self._busy() else "开始采集后才能停止。"))
        ToolTip(self.record_button, self._start_help)
        ToolTip(self.return_to_plan_button, lambda _event: (
            "正在读取实验台账，请稍候。" if self._opening_experiment else
            "回到当前采集计划，检查相机和保存位置。回到复查页时，可继续查看已打开的图像。"))

    def _activity_help(self, _event):
        if status := self._displayed_session_status():
            if error := status["error"]:
                return f"{self.activity.get()}\n\n{ui_error(CaptureError(error['code'], error['message']))}"
            progress = status["phase_progress"] or {}
            if readiness := progress.get("hardware_readiness"):
                names = {camera: self._view_title(camera) for camera in readiness["cameras"]}
                detail = wait_output_help(readiness, names, inputs=progress.get("hardware_inputs"))
                return f"{self.activity.get()}\n\n{detail}"
            if csv_watch := progress.get("csv_trigger"):
                return f"{self.activity.get()}\n\n{csv_trigger_help(csv_watch)}"
            if faults := [item for item in status.get("csv_triggers", ()) if item["state"] == "fault"]:
                return f"{self.activity.get()}\n\n阶段 {faults[-1]['phase']} · {csv_trigger_help(faults[-1])}"
        if self.preview:
            issues = []
            for camera, info in self.preview.status().items():
                if info["error"]:
                    issues.append(f"{self._view_title(camera)}\n{self._preview_error(info)}")
            if issues:
                return "\n\n".join((self.activity.get(), *issues))
        return self.activity.get()

    @staticmethod
    def _preview_error(info):
        code, detail = info.get("error_code"), info["error"]
        if code == "CAMERA_DISCONNECTED":
            return f"预览时与相机的连接中断。请检查接线、电源或网络。\n{detail}"
        return ui_error(CaptureError(code, detail)) if code else detail

    def _trigger_help(self, _event=None):
        status = self._displayed_session_status() if self._busy() else None
        if status and status.get("phase_pause"):
            return "请先点“继续采集”，再触发下一轮拍摄。"
        if status and status.get("phase_confirmation"):
            return "这一阶段尚未启动。请看顶部的确认提示，随后按该阶段的触发方式采集。"
        progress = status["phase_progress"] if status else None
        if progress and status["mode"] == "hardware":
            return "当前阶段接收相机线路上的物理触发。可用顶部的“查看触发状态”核对各相机的输入和收图。"
        if progress and (window := progress["event_window"]):
            if window.get("overlap"):
                if window["event_target_reached"]:
                    return "已接收计划中的全部触发。待最后几次事件收图并保存后，可在采集时间线逐次查看。"
                if window["state"] == "buffering":
                    return "先等第一次前图缓存满。就绪后可在上一事件收图期间继续触发。"
                if not progress["trigger_available"]:
                    return "请等正在收图的事件完成后，再发送下一次触发。"
                text = "保存下一次事件的前后原图。相邻事件可以共用同一张原图，每次事件均可单独复查。"
                return text + ("\n\n" + csv_trigger_help(csv_watch) if (csv_watch := progress.get("csv_trigger")) else "")
            if window["state"] == "buffering":
                if window.get("event_count", 1) != 1:
                    return f"{event_window_position(window)}的前图正在缓存。缓存满后才接收新的触发。"
                return "等事件前图像达到设定的张数后，再发送信号。提前触发会保存已有图像，并在台账记录缺少的张数。"
            if window["signal_id"]:
                return "当前事件已有一条触发请求。请查看事件后采集进度。"
            if csv_watch := progress.get("csv_trigger"):
                return ("发送一次操作员信号，立即保存事件前后的原图。\n\n" + csv_trigger_help(csv_watch))
        if not progress or status["mode"] not in ("manual", "event"):
            return "“手动触发”和“事件前后”阶段启动后，此按钮才可发送触发信号。"
        if self._board_scan:
            return "摆稳棋盘格后触发。待右侧显示本次角点检查结果，再调整棋盘格的位置或倾角。"
        if progress.get("host_sampling_cameras"):
            return "发送一次采集请求。连续送图的相机按请求后的首张主机收图取帧，曝光同步需独立核实。"
        return "发送一次操作员信号。手动模式会拍一轮，事件模式会保留事件前后图像。"

    def _statistics_help(self, _event):
        if self._preview_save_summary and self.footer_detail.get() == self._preview_save_summary:
            return self._preview_save_detail
        if not self.session or not self.counter.get():
            return "采集时可查看收到、已保存的图像张数和写入情况。悬停在控件上可查看对应说明。"
        status = self.session.status()
        mode = "自动" if status["queue_mode"] == "auto" else "固定"
        return "\n".join((
            "收到是电脑已接收的图像数。已保存是文件写入并登记后的图像数。",
            f"待写队列 {status['queue_size']} / {status['queue_capacity']} 批 · {mode}",
            RAW_BUFFER_HELP,
            *buffer_peak_lines(status),
            "异常包含设备报告、帧计数和未保存图像。可在复查页核对。",
        ))

    def _set_config(self, config):
        if self._practice:
            require_simulated_cameras(camera.id for camera in config.cameras)
        if self.config:
            self._offer_camera_start = False
        self.config = config.to_dict()
        self._calibration_profiles = {}
        self._invalidate_preflight()
        self.name.set(config.name)
        self.durability.set(config.durability)
        for device in config.cameras:
            self.inventory.setdefault(device.id, {"id": device.id, "model": device.id, "backend": device.id.split(":")[0]})
        self._populate_devices([device.id for device in config.cameras])
        self._populate_phases()
        self._schedule_output_check()

    def _registry(self):
        if self._practice:
            return PracticeCameraRegistry()
        return CameraRegistry(cti_files=self.config.get("cti_files", []))

    def _selected_camera_rows(self):
        saved = {device["id"]: device["settings"] for device in self.config["cameras"]}
        selected = self.devices.selection()
        if not selected:
            raise CaptureError("CAMERA_REQUIRED", "请从设备列表选择至少一台相机。")
        return [{"id": identity, "settings": saved.get(identity, {})} for identity in selected]

    def _read_config(self, *, phases=None):
        data = {**self.config, "name": self.name.get(), "durability": self.durability.get(),
                "cameras": self._selected_camera_rows()}
        regions = data.get("quality_regions", {})
        if regions:
            data["quality_regions"] = {item["id"]: regions[item["id"]] for item in data["cameras"] if item["id"] in regions}
        if phases is not None:
            data["phases"] = phases
        return CaptureConfig.from_dict(data)

    def _busy(self):
        return self._start_stop is not None or (self.session is not None and not self.session.finished)

    def _saved_panel(self):
        tab = self.tabs.select()
        return tab == str(self.review_panel) or (
            tab == str(self.quality_panel) and self.preview is None
            and (self.review_image is not None or self._board_scan is not None and not self._busy()))

    def _render_workspace(self, *, reviewing, busy):
        show_experiment = self.experiment is not None and (reviewing or busy and self._start_stop is None)
        if show_experiment:
            name, path = self._review_name, str(self.experiment)
            if self._current_name.get() != name:
                self._current_name.set(name)
            if self._current_path.get() != path:
                self._current_path.set(path)
        for row, visible in ((self._current_experiment_row, show_experiment),
                             (self._plan_settings, not show_experiment and not reviewing)):
            if visible and not row.winfo_manager():
                row.grid()
            elif not visible and row.winfo_manager():
                row.grid_remove()
        if reviewing and self._plan_menu.winfo_manager():
            self._plan_menu.pack_forget()
        elif not reviewing and not self._plan_menu.winfo_manager():
            self._plan_menu.pack(side=tk.LEFT, before=self._recent_button)
        if not self._compact_workspace:
            device_pane_visible = self._device_pane.winfo_manager() == "panedwindow"
            if reviewing and device_pane_visible:
                self._dock.forget(self._device_pane)
            elif not reviewing and not device_pane_visible:
                width = max(225, self.cti_button.winfo_reqwidth())
                self._dock.add(self._device_pane, before=self._preview_pane, minsize=width, width=width, stretch="never")
        timed = self._pausable_phase is not None and self._pausable_phase[1] == "timed"
        for button, visible in ((self.trigger_button, not reviewing and not timed),
                                (self.stop_button, not reviewing),
                                (self.record_button, not busy and not reviewing),
                                (self.marker_button, busy), (self.return_to_plan_button, reviewing)):
            if visible and not button.winfo_manager():
                button.grid()
            elif not visible and button.winfo_manager():
                button.grid_remove()

    def _displayed_session_status(self):
        if self.session is None or self._start_stop is not None:
            return None
        saved_session_selected = self.preview is None and self.tabs.select() == str(self.review_panel)
        if self._run_reviewed and not self._busy() and not saved_session_selected:
            return None
        return self.session.status()

    def _setup_block(self, *, needs_camera=False, needs_idle_camera=False):
        if self._busy():
            return "请等待本次采集结束，或点击“停止并保存”。"
        if self._trial_batch is not None:
            return "正在连续试采。可在“阶段试采”查看进度或停止。"
        if self._write_probe_window is not None and self._write_probe_window.working:
            return "正在测试目标盘写入。测试完成后可开始采集或修改计划。"
        if self._opening_experiment:
            return "正在打开实验，请等台账读取完成。"
        if self._restoring_draft:
            return "正在恢复上次计划并核对试采，请稍候。"
        if self._checking_plan:
            return "正在检查计划。请等待检查结果。"
        if self._checking_rate_trial:
            return "正在重新核验原试采和频率复测。"
        if self._preparing_analysis:
            return self._preparing_analysis
        if self._releasing_preview:
            return "正在断开相机预览，请稍候。"
        if self._camera_settings_busy:
            return "正在读取或应用相机参数，请稍候。"
        if self._discovering:
            return "正在查找相机。请等待设备列表更新。"
        if needs_idle_camera and self.preview:
            return "请先点击“断开预览”，让程序释放相机。"
        if needs_camera and not self.devices.selection():
            return "请先在相机列表中选择至少一台相机。"
        return ""

    def _setup_help(self, description, **requirements):
        return self._setup_block(**requirements) or description

    def _cti_block(self):
        if self._practice:
            return "练习窗口使用内置模拟相机。厂商接口可在图像采集主窗口设置。"
        if reason := self._setup_block(needs_idle_camera=True):
            return reason
        state = getattr(self, "_providers", {}).get("gentl", {})
        if state.get("code") in ("GENTL_UNAVAILABLE", "GENTL_SDK_LOAD_FAILED"):
            return provider_diagnostic("gentl", state, frozen=getattr(sys, "frozen", False))[1]
        return ""

    def _plan_input_block(self, preview_states=None):
        if reason := self._setup_block(needs_camera=True):
            return reason
        states = self.preview.status() if preview_states is None and self.preview else (preview_states or {})
        if any(info["state"] == PART_REQUIRED for info in states.values()):
            return "请先在预览下方为等待的相机选择图像部件。"
        if any(info["state"] == "failed" for info in states.values()):
            return "仍有相机预览失败。请恢复相机画面，再检查计划或开始采集。"
        if any(info["state"] != "preview" for info in states.values()):
            return "请等所选相机都收到预览图像，再检查计划或开始采集。"
        return "" if self.output_path.get().strip() else "请先选择新的保存位置。"

    def _checkerboard_setup_block(self):
        if reason := self._plan_input_block():
            return reason
        selected = self.devices.selection()
        if len(selected) > 2:
            return "标定采图需要一台或两台相机。按住 Ctrl 可更改选择。"
        if any(key.startswith("sim:") for key in selected):
            return "当前选中了模拟散斑。标定采图请选择能拍到棋盘格的相机。"
        return ""

    def _start_help(self, _event=None):
        if reason := self._plan_input_block() or self._output_problem():
            return reason
        kind = self._camera_kind()
        notice = {"sim": "将使用模拟图像练习。", "mixed": "当前选择包含模拟相机。"}.get(kind, "")
        return f"{notice}先检查计划，再按阶段采集。若已连接预览，程序会先切换相机连接。"

    def _trial_help(self, _event=None):
        if reason := self._plan_input_block():
            return reason
        index = self._selected_phase()
        if index is None:
            return "先在“采集计划”选择一行，再试采这一阶段。"
        mode = self.config["phases"][index]["mode"]
        return (f"短时试采第 {index + 1} 阶段（{MODE_LABELS[mode]}）。图像保存到单独目录，"
                "结束后可查看实测帧率、缓冲用量和文件校验。"
                + ("连续测量时间足够时，还可检查写入是否持续落后。" if mode in ("continuous", "timed") else ""))

    def _preview_button_help(self, _event=None):
        description = (
            "断开预览并释放所选相机。预览中可点“相机参数”调整当前画面。" if self.preview else
            "显示所选相机的实时图像，便于调整画面和光照。正式保存图像时点“开始采集”。"
        )
        return self._setup_help(description, needs_camera=True)

    def _retry_target(self, states):
        candidates = [camera for camera, info in states.items()
                      if info.get("retryable") and info["state"] in ("failed", "retry_wait")]
        return self.active_camera if self.active_camera in candidates else next(iter(candidates), None)

    def _retry_preview_help(self, _event=None):
        if reason := self._setup_block():
            return reason
        states = self.preview.status() if self.preview else {}
        camera = self._retry_target(states)
        if camera is None:
            return "当前没有可手动重连的相机。可悬停在异常画面上查看原因。"
        info = states[camera]
        hint = ("\n此设备按系统视频输入序号识别。收到画面后，请核对是否是所需相机。"
                if info.get("identity_scope") == "host_index" or
                (camera.startswith("uvc:") and camera[4:].isdecimal()) else "")
        return f"重新打开 {self._view_title(camera)}。其他相机继续预览。{hint}"

    def _retry_preview(self):
        if self._setup_block() or self.preview is None:
            return
        if camera := self._retry_target(self.preview.status()):
            try:
                self.preview.retry(camera)
                self.select_view(camera)
            except CaptureError as exc:
                self.activity.set(ui_error(exc, details=False))

    def _camera_kind(self):
        selected = self.devices.selection()
        simulated = sum(self._is_simulated(self.inventory[camera]) for camera in selected)
        if simulated:
            return "sim" if simulated == len(selected) else "mixed"
        return "physical" if selected else ""

    def _camera_context_help(self, _event=None):
        if self.camera_context.get() == "实验复查" and not self.experiment:
            return "打开一次已保存的实验，可浏览原图、校验台账或进入分析。"
        return {
            "模拟练习": "当前只选了模拟相机。可以实际保存练习图像，再试用复查和应变分析。",
            "实验相机": "当前只选了实验相机。正式采集前可检查画面、触发条件和磁盘空间。",
            "模拟与实验相机": "当前同时选了模拟和实验相机。若本次只需保存实验图像，请按住 Ctrl 取消选中模拟相机。",
            "实验复查": "当前在查看已写入实验目录的图像和台账。可以切换相机、校验原图或进入分析。",
            "交接副本": "当前目录带有交接清单。可在复查页点击“校验图像和台账”，核对文件是否仍与交接时一致。",
            "正在查找相机": "正在读取系统设备和厂商相机接口，列表更新后即可选择相机。",
        }.get(self.camera_context.get(), "在相机列表中选择设备。没有接相机时可选“模拟散斑”练习。")

    @staticmethod
    def _is_simulated(camera):
        return camera["backend"] == "sim" or camera.get("support") == "vendor_emulator"

    def _invalidate_preflight(self, *_args, reason="设置已修改，可重新检查磁盘空间和触发条件。"):
        self._schedule_draft_save()
        self._plan_trial_progress = None
        self._refresh_trial_progress()
        if self._preflight_result is not None:
            self._preflight_result = None
            self.plan_summary.set(reason)
        window = getattr(self, "_plan_report_window", None)
        if window is not None and window.winfo_exists() and not window.readiness_stale:
            self._mark_plan_report_stale(window, reason)

    @staticmethod
    def _mark_plan_report_stale(window, reason):
        window.readiness_stale = reason
        window.title(window.title().split(" · ", 1)[0] + " · 设置已修改")
        if window.readiness_overview is not None:
            window.readiness_overview.invalidate(reason)
        for control in window.report_actions:
            control.state(["disabled"])

    def _workflow_hint(self):
        return self._next_step.hint

    def _guide_next_step(self, preview_states, status):
        if self._trial_batch is not None:
            return _NextStep("查看连续试采", "查看各阶段收图和核验进度，或停止剩余试采。",
                             self.trial_progress_button, self.plan_panel)
        if self._busy():
            if self._start_stop is not None or status is None or status["state"] == "opening":
                return _NextStep("准备采集", "正在打开相机和保存目录。")
            if self.state.get() == "DRAINING":
                return _NextStep("正在保存", "等候队列中的图像写完，随后可复查。")
            if gate := self._phase_confirmation:
                number = gate["phase"]
                if gate["state"] == "waiting":
                    return _NextStep("等待阶段确认", f"阶段 {number} · {gate['instruction']}",
                                     phase_confirmation=(number, gate["confirmation_id"]))
                if gate["state"] == "saving_previous":
                    return _NextStep("完成上一阶段", f"等待图像登记完成，随后可确认开始阶段 {number}。")
                return _NextStep("启动阶段" if gate["state"] == "arming" else "正在保存",
                                 f"阶段 {number} 正在启动，请等相机就绪。" if gate["state"] == "arming" else "正在结束本次采集。")
            if self._phase_pause:
                state, hint = PHASE_PAUSE_DISPLAY[self._phase_pause["state"]]
                if self._phase_pause["state"] == "paused":
                    return _NextStep("暂停中", "调整完成后点“继续采集”，或用“停止并保存”结束。")
                return _NextStep(STATE_LABELS[state], hint)
            if self._write_pressure and (state := self._write_pressure["state"]) in ("sustained", "queue_growth", "near_limit"):
                hint = write_pressure_estimate(self._write_pressure["estimate"]) or WRITE_PRESSURE_GUIDANCE[state]
                return _NextStep(WRITE_PRESSURE_LABELS[state], hint, self.run_monitor.trend_button, self.plan_panel,
                                 button_label="查看写入趋势…")
            if self._pulse_notice:
                title, hint = self._pulse_notice
                return _NextStep(title, hint, self.run_monitor.pulse_button, self.plan_panel)
            if self._image_quality_alert:
                camera, codes = self._image_quality_alert[0]
                hint = (f"{len(self._image_quality_alert)} 台相机有画质提醒，可按相机查看抽查数据。"
                        if len(self._image_quality_alert) > 1 else
                        f"{self._view_title(camera)} · 最近抽查发现{image_quality_alert_text(codes)}。")
                return _NextStep("画质提醒", hint,
                                 self.run_monitor.quality_button, self.plan_panel, button_label="查看抽查…")
            progress = status["phase_progress"]
            if not progress:
                return _NextStep("准备本阶段", "正在应用相机设置。阶段启动后会显示收图或触发条件。")
            if not progress["accepting_images"]:
                return _NextStep("正在结束阶段", "本阶段已停止接收新请求，正在完成收图和保存。")
            csv_watch = progress.get("csv_trigger")
            window = progress.get("event_window")
            event_in_flight = window and (window.get("active_events") or window.get("event_target_reached")
                                          or window["state"] in ("trigger_queued", "recording_post", "complete"))
            if csv_watch and csv_watch["state"] == "fault" and not event_in_flight:
                hint = "悬停在底部采集状态查看文件读取失败的原因。"
                if self._trigger_phase is not None:
                    return _NextStep("CSV 监看已停用", hint + "可点“触发一次”保存事件原图。", self.trigger_button)
                return _NextStep("CSV 监看已停用", hint)
            if window:
                title = (event_window_position(window) if window.get("overlap") or window["event_count"] != 1
                         else "事件前后采集")
                pre = f"{window['pre_buffered_rounds']}/{window['pre_required_rounds']}"
                if window.get("overlap"):
                    if window["event_target_reached"]:
                        return _NextStep(title, "正在完成最后的事件图像。保存后可在采集时间线逐次复查。")
                    if window["state"] == "buffering":
                        return _NextStep(title, f"事件前缓存已累计 {pre} 张/相机。缓存满后可触发。")
                    if self._trigger_phase is None:
                        return _NextStep(title, "同时收图的事件已达到上限。等其中一次完成后可继续触发。")
                    if window["active_events"]:
                        return _NextStep(title, f"{window['active_events']} 次事件正在收图。可继续点“触发一次”。",
                                         self.trigger_button)
                    if csv_watch:
                        return _NextStep(title, f"正在等待 {csv_trigger_rule_text(csv_watch)}。也可点“触发一次”。")
                    return _NextStep(title, "前图已就绪。点“触发一次”保存这次事件前后的原图。", self.trigger_button)
                if window["state"] == "buffering":
                    return _NextStep(title, f"事件前缓存已累计 {pre} 张/相机。请等缓存齐再触发。")
                if window["state"] == "trigger_queued":
                    return _NextStep(title, "已收到触发请求，正在等待相机执行。")
                if window["state"] == "recording_post":
                    post = f"{window['post_received_rounds']}/{window['post_required_rounds']}"
                    return _NextStep(title, f"已接收触发请求。事件后已收到 {post} 张/相机。")
                if window["state"] == "complete":
                    return _NextStep(title, "事件收图已完成。等待原图和事件记录保存后可复查。")
                if csv_watch:
                    return _NextStep(title, f"正在等待 {csv_trigger_rule_text(csv_watch)}。也可点“触发一次”。")
                return _NextStep(title, "事件前图像已就绪。点“触发一次”保存这次事件的前后原图。", self.trigger_button)
            if status["mode"] == "hardware":
                title, hint = hardware_trigger_guidance(progress)
                return _NextStep(title, hint, tab=self.plan_panel, focus=self.run_monitor.cameras,
                                 button_label="查看触发状态")
            if status["mode"] == "manual":
                if progress["manual_signal_id"]:
                    return _NextStep("正在执行手动触发", "已收到触发请求，正在等待相机送图。")
                if pending := progress["queued_signals"]:
                    return _NextStep("触发请求已入队", f"还有 {pending} 个请求等待执行。")
                if self._trigger_phase is not None:
                    return _NextStep("等待手动触发", "点“触发一次”拍摄一轮，也可通过已启用的外部命令触发。",
                                     self.trigger_button)
            return _NextStep("正在采集", "可在“采集监看”查看收图，或按 F8 记录现场变化。")
        if reason := self._setup_block():
            return _NextStep("正在处理", reason)
        if self._saved_panel():
            if self.preview:
                return _NextStep("切换到实验复查", "相机仍在预览。断开后可打开实验或查看已保存图像。",
                                 self.preview_button, self.review_panel)
            if not self.experiment:
                return _NextStep("复查实验", "打开已保存的实验，可查看原图和采集记录。", self.open_experiment_button)
            if self._checking_experiment or self._review_busy:
                return _NextStep("复查实验", "正在读取图像或检查台账。")
            if self._review_state in ("INTERRUPTED", "FAILED"):
                outcome, verification = self._review_outcome, self._review_integrity
                if verification and verification.get("recovery"):
                    if self._review_issue_count:
                        return self._guide_review_result()
                    if verification["ok"]:
                        return _NextStep("复查已恢复的实验", "台账恢复和原图校验已完成。可选择接下来要采集的阶段。",
                                         self.continue_button)
                if outcome and outcome.write_records_need_review:
                    return _NextStep("核对帧写入记录", "这次采集有待核对的写入记录。打开中断检查可确认原图能否补登记。",
                                     self.recover_button)
                if outcome and outcome.recovery_recorded:
                    return _NextStep("准备续采", "台账已记录恢复操作。可查看各阶段的保存情况并选择续采起点。",
                                     self.continue_button)
                if not (self._review_state == "FAILED" and outcome and (outcome.has_incidents or outcome.event_phases)):
                    return _NextStep("复查中断的实验", "先核对已登记图像及遗留的帧写入记录。", self.recover_button)
            if self._review_trial:
                if self._review_trial.get("rate_source_experiment_id"):
                    return _NextStep("核对建议频率", "打开复测报告，可对照原试采和当前计划。",
                                     self.continue_button)
                if (len(self.config["phases"]) > 1 and not self._plan_input_block(preview_states)
                        and any(row["output"] == str(self.experiment) for row in self._trial_results.values())):
                    return _NextStep("复查试采", "查看各阶段的试采记录，选择接下来要复核或试采的阶段。",
                                     self.trial_progress_button)
                return _NextStep("复查试采", "查看实测帧率、缓冲使用量和文件校验结果。", self.continue_button)
            if self._board_scan:
                return _NextStep("复查标定图像", "用保存的棋盘格图像拟合镜头参数。", self.continue_button)
            return self._guide_review_result()
        if self._offer_startup_draft and self._stored_draft and self.preview is None:
            return _NextStep("恢复上次计划", "可找回相机设置、保存位置和关联试采。",
                             self.restore_draft_button, self.plan_panel)
        if not self.devices.selection():
            return _NextStep("选择相机", "打开相机列表，选择要采集的设备。没有接相机时，可选“模拟散斑”练习。",
                             button_label="定位到相机列表", focus=self.devices)
        if (self._offer_camera_start and self._discovered_ids is not None and self.preview is None
                and self._preflight_result is None and self.devices.selection() == ("sim:0",)):
            if self._initial_draft_pending:
                return _NextStep("准备采集", "正在查看是否有上次保存的计划。")
            if found := len(self._detected_experiment_cameras()):
                return _NextStep("选择实验相机", f"发现 {found} 台实际相机。选择后可检查实时画面。",
                                 self.camera_choice_button)
            return _NextStep("开始使用", "暂未发现实际相机。可打开独立模拟练习，或接上相机后点“刷新”。",
                             button_label="打开模拟练习", action=self.open_practice)
        if self.preview is None and self._preflight_result is None:
            missing = [camera for camera in self.devices.selection() if self._camera_not_discovered(camera)]
            if missing:
                subject = self._view_title(missing[0]) if len(missing) == 1 else f"{len(missing)} 台计划相机"
                return _NextStep("核对相机连接", f"上次未找到 {subject}。接回后可刷新并检查。",
                                 self.refresh_button, button_label="刷新并检查")
        if any(info["state"] == PART_REQUIRED for info in preview_states.values()):
            return _NextStep("选择主图像", "点选等待选择的相机画面，再从下方选择要保存的图像。")
        if camera := self._retry_target(preview_states):
            info = preview_states[camera]
            if info["state"] == "retry_wait":
                note = f"等待第 {info['retry_number']}／{info['retry_limit']} 次自动重连。也可立即重连。"
            else:
                note = "检查电源和接线后可重连。悬停在异常画面上可查看详情。"
            return _NextStep("恢复相机预览", f"{self._view_title(camera)} · {note}", self.retry_preview_button)
        if any(info["state"] == "failed" for info in preview_states.values()):
            return _NextStep("预览异常", "请检查驱动和设备连接。悬停在异常画面上可查看详情。", self.provider_button)
        if any(info["state"] in ("releasing", "reconnecting") for info in preview_states.values()):
            return _NextStep("恢复相机预览", "正在释放旧连接或打开相机。其他相机会继续预览。")
        if not self.output_path.get().strip():
            return _NextStep("设置保存位置", "请选择本次实验要保存到哪个文件夹。", self.choose_output_button)
        if problem := self._output_problem():
            if suggestion := self._output_check.suggestion:
                return _NextStep("输出目录已占用", "可在同一父目录使用新的实验路径。采用后请核对实验名称。",
                                 button_label="使用新路径", suggested_output=suggestion)
            return _NextStep("检查保存位置", problem, self.choose_output_button)
        if self.preview:
            if any(info["state"] == "connecting" for info in preview_states.values()):
                return _NextStep("等待预览", "正在等相机送来第一张图像。")
        if self._preflight_result == "ready":
            if self._plan_trial_progress and (number := self._plan_trial_progress.recommended):
                return _NextStep("核对阶段试采", f"计划已检查。建议先查看阶段 {number}，核对试采记录或用当前设置试采。",
                                 self.trial_progress_button, self.plan_panel, button_label=f"查看阶段 {number} 试采…")
            return _NextStep("可以采集", "计划已检查。确认现场状态后，可用右下角的按钮开始采集。")
        if self._preflight_result == "blocked":
            return _NextStep("处理检查结果", "在检查结果中选择阻塞项，查看原因和处理方式。",
                             self.preflight_button, self.plan_panel, button_label="查看阻塞项…")
        if self._preflight_result == "review":
            return _NextStep("核对采前提醒", "在检查结果中查看需要复核的事项。",
                             self.preflight_button, self.plan_panel, button_label="查看复核事项…")
        if self.preview:
            if self._quality_checked_preview is not self.preview and self.tabs.select() != str(self.quality_panel):
                if len(preview_states) > 1 and not self._preview_quality_block(preview_states):
                    return _NextStep("检查画面", "可一次核对各台相机的画面，定位照明或纹理问题。",
                                     self.preview_quality_button, self.quality_panel)
                return _NextStep("检查画面", "先选中相机画面，核对试样的光照和清晰度。",
                                 tab=self.quality_panel, button_label="查看图像质量", focus=self.hist)
            if (index := self._hardware_mapping_phase()) is not None:
                return _NextStep("核对物理触发输入", f"阶段 {index + 1} 的接线或脉冲计数对应此前的相机，请按当前设备核对。",
                                 button_label=f"核对阶段 {index + 1}…", action=self._edit_hardware_mapping)
            if guide := self._preview_trigger_guide():
                return guide
            return _NextStep("检查采集计划", "确认画面的光照和对焦后，核对触发条件和所需磁盘空间。",
                             self.preflight_button, self.plan_panel)
        hint = ("当前选择为模拟相机。连接预览可先查看模拟散斑。"
                if self._camera_kind() == "sim" else "连接预览后可检查光照和对焦。")
        return _NextStep("相机与预览", hint, self.preview_button)

    def _preview_trigger_guide(self):
        evidence = {camera: self.preview.trigger_support(camera) for camera in self.devices.selection()}
        if not evidence or any(item is None for item in evidence.values()):
            return None
        support = [trigger_evidence(camera, evidence) for camera in evidence]
        phases = self.config["phases"]
        if any(phase["mode"] == "hardware" for phase in phases) and any(item.hardware is False for item in support):
            return _NextStep("核对物理触发", "部分相机无法响应本计划的物理触发。检查结果会定位受影响的阶段。",
                             self.preflight_button, self.plan_panel, button_label="核对触发阶段…")
        requested = [phase for phase in phases if phase["mode"] not in ("continuous", "hardware")]
        if not requested:
            return None
        missing_software = any(item.software is False for item in support)
        if missing_software and any(not phase.get("software_trigger_fallback") for phase in requested):
            return _NextStep("核对取图方式", "部分相机连续送图。采前检查会列出能否改用主机选帧及受影响的阶段。",
                             self.preflight_button, self.plan_panel, button_label="查看取图建议…")
        if missing_software or any(item.software_method == HOST_NEXT_RECEIPT for item in support):
            return _NextStep("检查主机取图计划", "这些阶段含主机选帧。用试采核对实际保存帧率和请求到收图的时间。",
                             self.preflight_button, self.plan_panel)
        return None

    def _guide_review_result(self):
        outcome, verification = self._review_outcome, self._review_integrity
        if self._review_state and self._review_state.lower() not in TERMINAL_STATES:
            return _NextStep("核对实验状态", "台账尚未登记结束。如果采集已停止，可检查中断和遗留图像。",
                             self.recover_button)
        if self._review_issue_count:
            return _NextStep("处理校验问题", f"本次校验发现 {self._review_issue_count} 项问题，可查看详情。",
                             self.review_report_button)
        if self._review_frame_error and verification is None:
            return _NextStep("核对图像文件", "当前图像无法打开。校验整组文件可定位缺失或受损的图像。", self.verify_button)
        if outcome is None:
            if self._review_read_error:
                return _NextStep("复查记录", "采集结果暂不可读，可校验台账和已保存图像。", self.verify_button)
            return _NextStep("复查实验", "正在读取采集结果。")
        if any(state == "fault" for _number, state in outcome.csv_triggers):
            return _NextStep("复查 CSV 阈值触发", "试验机 CSV 监看曾中断。选中故障记录可查看阶段和原因。",
                             self.timeline_button, button_label="查看异常记录…", timeline_filter="incidents_only")
        if outcome.has_incidents:
            notes = capture_incident_lines(outcome.camera_losses, outcome.scheduled_misses, outcome.short_pre_windows,
                                           self._view_title)
            hint = (f"{notes[0]}。可查看台账中的异常记录。" if notes else
                    "打开异常列表，选中一行可核对阶段、相机和相关原图。")
            return _NextStep("复查采集异常", hint, self.timeline_button, button_label="查看异常记录…",
                             timeline_filter="incidents_only")
        if not outcome.frames:
            if outcome.event_phases:
                return _NextStep("本次未保存事件原图", "在时间线核对是否收到触发及结束阶段。", self.timeline_button)
            return _NextStep("未登记图像", "查看各阶段的开始、结束和异常记录。", self.timeline_button)
        if any(row["completed_events"] for row in outcome.event_phases) and verification is None:
            return _NextStep("复查事件前后原图", "按事件编号查看这次保存的各相机图像。",
                             self.timeline_button, button_label="看事件前后…", timeline_filter="event_windows_only")
        if self._review_state == "STOPPED":
            return _NextStep("复查已停止的实验", "查看采集停在哪个阶段，以及此前登记的图像。", self.timeline_button)
        if verification is None:
            if any(state == "triggered" for _number, state in outcome.csv_triggers):
                return _NextStep("复查 CSV 阈值触发", "在时间线选中越线记录，可查看这次保存的事件前后原图。",
                                 self.timeline_button)
            return _NextStep("核对已保存数据", "先校验整组 TIFF、归档标定和台账中的触发关联。", self.verify_button)
        if self.review_image is not None:
            camera_frames = outcome.cameras.get(self.active_camera, {}).get("frames", 0)
            if camera_frames >= 2 or self._review_parent:
                return _NextStep("复查原图", "可切换已保存的图像。需要逐帧计算应变时，打开离线分析。",
                                 self.offline_analysis_button)
            return _NextStep("复查原图", "当前相机只有一帧。应变分析需要参考图和后续实验图。",
                             self.frame_search_button)
        return _NextStep("复查原图", "选择本实验中已登记图像的相机，或按阶段查找图像。", self.frame_search_button)

    def _update_next_step(self, preview_states, status):
        step = self._guide_next_step(preview_states, status)
        choosing_camera = step.target is self.camera_choice_button or step.action == self.open_practice
        if step != self._next_step:
            self._next_step = step
            style = ("CaptureAttention.TLabel" if step.suggested_output is not None or step.target in (
                         self.run_monitor.trend_button, self.run_monitor.pulse_button, self.run_monitor.quality_button,
                         self.review_report_button, self.timeline_button)
                     else "CaptureSelected.TLabel")
            self.step_title.configure(text=step.title, style=style)
            self._render_step_hint()
            self.record_button.configure(style="TButton" if choosing_camera else "Record.TButton")
        if step.phase_confirmation is not None:
            label = f"确认开始阶段 {step.phase_confirmation[0]}"
            style = "Confirm.Record.TButton"
        elif step.target is not None and not step.target.instate(["disabled"]):
            label = step.button_label or step.target.cget("text")
            style = "Confirm.Record.TButton" if choosing_camera else "Next.TButton"
        elif step.focus is not None or step.suggested_output is not None or step.action is not None:
            label = step.button_label
            style = "Confirm.Record.TButton" if choosing_camera else "Next.TButton"
        else:
            label = ""
        if label:
            if self.step_button.cget("text") != label or self.step_button.cget("style") != style:
                self.step_button.configure(text=label, style=style)
            if not self.step_button.winfo_manager():
                self.step_button.grid()
        elif self.step_button.winfo_manager():
            self.step_button.grid_remove()

    def _follow_next_step(self):
        step = self._next_step
        if step.phase_confirmation is not None:
            phase, identity = step.phase_confirmation
            try:
                self.session.request_phase_confirmation(phase, confirmation_id=identity)
                self.activity.set(f"已确认阶段 {phase}，正在启动相机。")
                self.step_button.grid_remove()
            except CaptureError as exc:
                self.activity.set(ui_error(exc, details=False))
        elif step.target is not None and not step.target.instate(["disabled"]):
            if step.tab is not None:
                self.tabs.select(step.tab)
            elif self._compact_workspace and step.target in (
                self.preview_button, self.retry_preview_button, self.provider_button, self.refresh_button,
                self.camera_choice_button,
            ):
                self.tabs.select(self._device_pane)
            report = getattr(self, "_plan_report_window", None)
            if (step.target is self.preflight_button and self._preflight_result in ("blocked", "review")
                    and report is not None and report.winfo_exists() and not report.readiness_stale):
                report.lift()
            elif step.target is self.timeline_button:
                self.show_timeline(all_cameras=True, event_filter=step.timeline_filter)
            elif step.target is self.trial_progress_button:
                self.show_trial_progress(follow_recommendation=True)
            elif step.target is self.refresh_button:
                self._refresh_and_check_plan()
            else:
                step.target.invoke()
        elif step.suggested_output is not None:
            self._use_suggested_output(step.suggested_output)
        elif step.action is not None:
            step.action()
        elif step.focus is not None:
            if step.focus is self.run_monitor.cameras and self._busy() and (status := self._displayed_session_status()):
                self.run_monitor.render(status, self.active_camera)
                self._monitor_scroll.canvas.yview_moveto(0)
            self._focus_guide_target(step.tab, step.focus)

    def _next_step_help(self, event):
        step = self._next_step
        if step.focus is self.run_monitor.cameras:
            return f"{step.hint}\n\n打开采集监看。选中相机后可查看计划输入、收图和已保存的帧数。"
        if step.suggested_output is not None:
            return f"将新路径填入采集计划。开始采集时才会创建目录。\n\n{step.suggested_output}"
        if step.action == self.open_practice:
            return f"{step.hint}\n独立练习会连接内置模拟散斑，使用专用练习目录。"
        if step.action == self._edit_hardware_mapping:
            return f"{step.hint}\n打开该阶段的接线与计数设置。按现场接线填写，应用后再检查计划。"
        if step.phase_confirmation is not None:
            phase = self.session.config.phases[step.phase_confirmation[0] - 1]
            note = "确认后启动相机并开始阶段计时。"
            if phase.mode == "hardware":
                note += "请等相机就绪后再开始发送物理触发。"
            return f"{step.hint}\n{note}"
        if step.target is self.timeline_button:
            if step.timeline_filter == "incidents_only":
                return (f"{step.hint}\n\n打开全部相机的异常列表。悬停或按 F1 查看单条记录的原因和下一步。"
                        "可按相机和阶段缩小范围。")
            if step.timeline_filter == "event_windows_only":
                return "打开采集时间线并显示各次事件。选中一条，可按相机查看这次事件的全部已保存原图。"
            return "打开全部相机的时间线，查看各阶段的图像和已记录的异常。打开后可按相机和阶段筛选。"
        if step.target is self.preflight_button and step.title == "检查主机取图计划":
            return f"{step.hint}\n{DRIVER_HOST_FRAME_HELP}"
        if step.target is self.preflight_button and step.title in ("核对取图方式", "核对物理触发"):
            return f"{step.hint}\n检查会列出各台相机的触发能力。可从报告修改对应阶段。"
        if step.target is self.trial_progress_button:
            return "打开阶段试采。当前有建议阶段时，程序会选中该阶段，可查看报告或打开试采确认。"
        if step.target is self.refresh_button:
            return "重新查找相机，再按当前计划检查设备连接和保存位置。"
        return step.target._tooltip.resolve_text(event) if step.target else step.hint

    def _device_help(self, event):
        item = ""
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            if self.devices.identify_region(event.x, event.y) == "heading":
                return {
                    "#0": "相机型号或识别名称。鼠标停在相机行上可查看完整名称与编号。",
                    "#1": "程序通过哪种接口连接相机。“模拟”使用软件生成的图像。",
                    "#2": "上次查找的结果。可用“检查计划”验证程序是否能按计划里的编号打开设备。",
                }.get(self.devices.identify_column(event.x), "")
            item = self.devices.identify_row(event.y)
        elif selected := self.devices.selection():
            item = self.devices.focus() if self.devices.focus() in selected else selected[0]
        if item not in self.inventory:
            return "单击选择相机，按住 Ctrl 可多选。选择两台以上时会显示“多机参数”。“模拟散斑”可用来练习。"
        device = self.inventory[item]
        status = "软件生成散斑图，可用来练习。" if device["backend"] == "sim" else ""
        if self._camera_not_discovered(item):
            status = "上次查找未找到此设备。检查接线和驱动后刷新，或检查计划以验证连接。"
        return f"{self._view_title(item)}\n设备编号 {item}\n{status or '单击选择。按住 Ctrl 可保留已选相机。'}"

    def _region_help(self, action, description):
        view = self.views.get(self.active_camera)
        if not view or view._image is None:
            return "请先连接预览，或在“复查与分析”打开一张已保存图像。"
        if action == "输入区域" and view.corrected:
            return "请先切回原始图像，再输入原图像素坐标。"
        if action != "输入区域" and view.roi is None:
            return "请先按住 Shift 拖出检查区域，或点击“输入区域”。"
        return description

    def _review_help(self, description, *, needs_camera=False, needs_idle=False, browsing=False):
        if self._opening_experiment:
            return "正在打开实验，请等台账读取完成。"
        if not self.experiment:
            return "先从顶部打开已有实验，或先完成一次采集。"
        if needs_idle and (reason := self._setup_block(needs_idle_camera=True)):
            return reason
        if needs_idle and self._checking_experiment:
            return "正在检查实验台账，请等待报告。"
        if needs_camera and self.preview:
            return "请先断开预览，再选择实验中要复查的相机。"
        if needs_camera and self.active_camera not in self._experiment_cameras:
            return "请选择本实验的相机画面。"
        if browsing and self._review_busy:
            return "正在读取已保存的图像，请稍候。"
        return description

    def _analysis_help(self, description, *, needs_idle=False):
        return self._preparing_analysis or self._review_help(
            description, needs_camera=True, needs_idle=needs_idle, browsing=True)

    def _tab_changed(self, _event=None):
        selected = self.tabs.select()
        if selected == str(self._device_pane):
            # The shared pane is still a child of the dock, so raise it above the notebook.
            self._device_pane.lift(self._detail_pane)
        elif selected:
            self._last_detail_tab = selected
        self._quality_due = 0.
        if selected == str(self.plan_panel):
            self._monitor_due = 0.
        if selected != str(self.quality_panel):
            self._sync_quality_scope(None)
        saved = self._saved_panel()
        busy = self._busy()
        self._render_workspace(reviewing=saved and not busy, busy=busy)
        if saved:
            self._show_saved_workspace_images()
        elif not busy and not self.preview:
            self._show_views(self.devices.selection(), browse_saved=False)

    def _show_saved_workspace_images(self):
        if self.preview or not self.experiment or self._start_stop is not None:
            return
        previous = self.active_camera
        preferred = self.review_image[0] if self.review_image is not None else previous
        self._show_views(self._experiment_cameras, preferred=preferred, browse_saved=False)
        if self.review_image is None and (not self._review_busy or self.active_camera != previous):
            self.browse_frame(latest=True)

    def _return_to_plan(self):
        if self._opening_experiment or self._busy():
            return
        self.tabs.select(self.plan_panel)
        self._schedule_output_check()
        self.activity.set("已返回计划。可核对相机和保存位置。")
        self.root.after_idle(self.name_entry.focus_set)

    def _job(self, action, report, failure=None, *, daemon=True):
        def work():
            try:
                self._jobs.put((report, action()))
            except Exception as exc:
                self._jobs.put((failure or self._error, exc))
        threading.Thread(target=work, name="capture-desktop-task", daemon=daemon).start()

    def _error(self, exc):
        message = ui_error(exc)
        self.activity.set(message.split("\n", 1)[0])
        messagebox.showerror("采集操作未完成", message, parent=self.root)

    def show_guide(self):
        guide = getattr(self, "_guide_window", None)
        if guide is not None and guide.top.winfo_exists():
            guide.top.lift()
            guide.tabs.focus_set()
            return
        self._guide_window = CaptureGuide(self.root, (
            (None, self.devices), (self.quality_panel, self.hist), (self.plan_panel, self.template_button),
            (self.plan_panel, self.record_button), (self.review_panel, self.review_scroll.canvas),
        ), self._focus_guide_target, open_practice=None if self._practice else self.open_practice)

    def open_practice(self):
        try:
            launch_capture(practice=True)
        except OSError as exc:
            self._error(exc)
            return False
        return True

    def _focus_guide_target(self, panel, control):
        if panel is not None:
            self.tabs.select(panel)
        elif control is self.devices:
            if self._compact_workspace:
                self.tabs.select(self._device_pane)
            elif self._saved_panel():
                self.tabs.select(self.plan_panel)
        self.root.lift()
        self.root.after_idle(control.focus_set)

    def _refresh_and_check_plan(self, dialog=None, *, check=None):
        if reason := self._setup_block(needs_idle_camera=True):
            self.activity.set(reason)
            return
        if dialog is not None:
            dialog.destroy()
        self.discover(on_finished=check or self.check_plan)

    def discover(self, *, on_finished=None):
        if self._setup_block(needs_idle_camera=True):
            return
        self._discovering = True
        self.refresh_button.state(["disabled"])
        self.provider_text.set("正在搜索采集设备…")
        registry = self._registry()
        def failed(exc):
            self._discovering = False
            self.provider_text.set("查找设备失败。可检查驱动后点“刷新”。")
            self._error(exc)

        def completed(result):
            self._discovered(result)
            if on_finished is not None and not self._closing:
                on_finished()

        self._job(registry.discover, completed, failed)

    def _discovered(self, result):
        self._discovering = False
        selected = self.devices.selection()
        configured = {entry["id"] for entry in self.config["cameras"]}
        retained = configured | set(selected) | set(self._experiment_cameras)
        self.inventory = {key: value for key, value in self.inventory.items() if key in retained}
        self.inventory.update({item["id"]: item for item in result["devices"]})
        discovered = {item["id"] for item in result["devices"]}
        if discovered != self._discovered_ids:
            self._invalidate_preflight()
        self._discovered_ids = discovered
        self._populate_devices(selected)
        for identity, view in self.views.items():
            view.set_title(self._view_title(identity))
        providers = result["providers"]
        detected = [f"{PROVIDER_LABELS.get(name, name)} {state['count']}"
                    for name, state in providers.items() if state.get("count", 0)]
        self.provider_text.set("　".join(detected) or "未发现可用设备")
        self._providers = providers
        self.refresh_button.state(["!disabled"])
        practice = sum(self._is_simulated(item) for item in result["devices"])
        actual = len(result["devices"]) - practice
        if not self._saved_panel():
            self.activity.set(f"实际相机 {actual} 台 · 模拟相机 {practice} 台")

    def show_providers(self):
        providers = getattr(self, "_providers", {})
        text = format_provider_report(providers, frozen=getattr(sys, "frozen", False))
        gentl = providers.get("gentl", {})
        action = None
        if gentl.get("code") in ("GENTL_NO_CTI", "GENTL_CTI_INVALID", "GENTL_CTI_LOAD_FAILED", "DUPLICATE_CAMERA_ID"):
            action = ("选择厂商 .cti…", self.choose_cti, "选择后刷新相机列表，可以直接连接预览。")
        self._providers_window = self._open_text_report(
            "采集驱动状态", text, getattr(self, "_providers_window", None), action=action,
        )

    def _camera_not_discovered(self, identity):
        return self._discovered_ids is not None and identity not in self._discovered_ids and not identity.startswith("sim:")

    def _detected_experiment_cameras(self):
        discovered = self._discovered_ids or ()
        return {identity: (self._view_title(identity), PROVIDER_LABELS.get(device["backend"], device["backend"]))
                for identity, device in self.inventory.items()
                if identity in discovered and not self._is_simulated(device)}

    def show_camera_choice(self):
        if reason := self._setup_block(needs_idle_camera=True):
            self.activity.set(reason)
            return
        cameras = self._detected_experiment_cameras()
        if not cameras:
            self.activity.set("暂未发现实际相机。接上设备后请刷新相机列表。")
            self._focus_guide_target(None, self.devices)
            return
        previous = self._camera_choice_dialog
        if previous is not None and previous.top.winfo_exists():
            previous.top.lift()
            return
        self._camera_choice_dialog = CameraPreviewDialog(
            self.root, cameras, self.devices.selection(), self._apply_camera_choice,
        )

    def _apply_camera_choice(self, selected):
        if reason := self._setup_block(needs_idle_camera=True):
            return reason
        if not set(selected) <= self._detected_experiment_cameras().keys():
            return "设备列表已更新。请关闭窗口，再重新选择相机。"
        if (self._offer_camera_start and self._stored_draft is None and not self._initial_draft_pending
                and self.name.get() == _DEMO_CAPTURE_NAME):
            self.name.set("新建采集")
        self.devices.selection_set(*selected)
        self.devices.focus(selected[0])
        self.devices.see(selected[0])
        self._camera_selection()

        def preview():
            if not self._closing and self.preview is None and self.devices.selection() == selected:
                self.toggle_preview()

        self.root.after_idle(preview)
        return ""

    def _populate_devices(self, selected):
        self.devices.delete(*self.devices.get_children())
        for identity, item in self.inventory.items():
            title = f"模拟散斑 {identity.rsplit(':', 1)[-1]}" if item["backend"] == "sim" else item["model"]
            if item["backend"] == "sim":
                discovery = "内置"
            elif self._discovered_ids is None:
                discovery = "待查找"
            else:
                discovery = "未找到" if self._camera_not_discovered(identity) else "已找到"
            self.devices.insert("", "end", iid=identity, text=title,
                                values=(PROVIDER_LABELS.get(item["backend"], item["backend"]), discovery))
        self.devices.selection_set(*selected)
        if selected:
            self.devices.focus(selected[0])
        self._camera_selection()

    def _camera_selection(self, _event=None):
        selected = self.devices.selection()
        if selected != self._selected_devices:
            if self._setup_block(needs_idle_camera=True):
                self.devices.selection_set(*self._selected_devices)
                return
            self._selected_devices = selected
            self._invalidate_preflight()
            if self._saved_panel():
                self.tabs.select(self.plan_panel)
            self._show_views(selected)
        if len(selected) >= 2:
            self.batch_settings_button.grid()
        else:
            self.batch_settings_button.grid_remove()
        kind = {"sim": "模拟练习", "mixed": "含模拟相机", "physical": "实际相机"}.get(self._camera_kind())
        self.selection_text.set(f"已选 {len(selected)} 台 · {kind}" if selected else "请在列表中选择相机")

    def _show_views(self, identities, *, preferred=None, browse_saved=True):
        if list(self.views) == list(identities):
            for identity in identities:
                self.views[identity].set_title(self._view_title(identity))
            if preferred in self.views:
                self.select_view(preferred, browse_saved=browse_saved)
            return
        self.empty_preview.grid_remove()
        for view in self.views.values():
            view.destroy()
        self.views.clear()
        cols = 2 if len(identities) > 1 else 1
        for idx in range(2):
            self.preview_area.columnconfigure(idx, weight=int(idx < cols), uniform="view-cols" if idx < cols else "")
        for idx in range(max(getattr(self, "_view_rows", 1), (len(identities) + cols - 1) // cols)):
            self.preview_area.rowconfigure(idx, weight=int(idx < (len(identities) + cols - 1) // cols))
        self._view_rows = (len(identities) + cols - 1) // cols
        for idx, identity in enumerate(identities):
            pane = ImageViewport(self.preview_area, identity, self._view_title(identity), self.select_view, self._inspect_region)
            pane.grid(row=idx // cols, column=idx % cols, sticky="nsew", padx=3, pady=3)
            pane.canvas.configure(takefocus=True)
            ToolTip(pane.canvas, lambda _event, view=pane: self._viewport_help(view), title="图像操作")
            self.views[identity] = pane
        self.select_view(preferred if preferred in self.views else next(iter(self.views), None),
                         browse_saved=browse_saved)
        if not self.views:
            self.empty_preview.grid(row=0, column=0, sticky="nsew")

    def _view_title(self, identity):
        item = self.inventory.get(identity, {})
        if item.get("backend") == "sim":
            return f"模拟散斑 {identity.rsplit(':', 1)[-1]}"
        serial = f" · {item['serial']}" if item.get("serial") else ""
        return f"{item.get('model', identity)}{serial}"

    def _viewport_help(self, view):
        if (view._image is None or view._frame_caption == "上阶段图像") and self._busy() and not self._saved_panel():
            status = self._displayed_session_status()
            if status and status["mode"] == "hardware" and (progress := status["phase_progress"]):
                camera = view.camera_id
                if source := progress.get("hardware_inputs", {}).get(camera):
                    lines = [self._view_title(camera), f"计划触发输入 {source}"]
                    if count := progress["received_by_camera"][camera]:
                        lines.append(f"主机已收到本阶段 {count} 帧")
                    else:
                        lines.append("主机尚未收到本阶段首帧")
                    if reading := (progress.get("hardware_readiness") or {}).get("cameras", {}).get(camera):
                        lines.append(wait_output_camera_text(reading))
                    return "\n".join(lines)
        if view._image is None:
            info = self.preview.status().get(view.camera_id, {}) if self.preview else {}
            if info.get("state") == PART_REQUIRED:
                return "单击选中这台相机，然后在画面下方选择要保存的图像部件。"
            if info.get("error"):
                return f"{self._view_title(view.camera_id)} · {PREVIEW_STATES[info['state']]}\n\n{self._preview_error(info)}"
            if not self.preview and self.experiment and self._saved_panel():
                return "单击相机卡片查看这台相机。右侧可逐张浏览，也可用“查找图像”定位特定帧。"
            return self._workflow_hint()
        if view.corrected:
            return "当前为畸变校正预览。切回原始图像后可绘制检查区域。滚轮缩放，拖动平移。"
        if view._frame_caption == "上阶段图像":
            return "显示这台相机上一阶段的最后一帧。到“采集监看”可查看本阶段的收图和待存状态。"
        if self._board_visible:
            return "绿色圆点标出这张图像里的棋盘格内角点。滚轮缩放，拖动平移。"
        return "单击可选中这台相机。滚轮缩放，拖动平移。按住 Shift 拖出方框可检查局部曝光、对焦和散斑。"

    def select_view(self, identity, *, browse_saved=True):
        if identity != self.active_camera:
            self.active_camera = identity
            self._monitor_due = 0.
            self._review_revision += 1
            self._sync_quality_scope(None)
            self._update_region_caption()
            if (not self.preview and self._saved_panel() and identity in self._experiment_cameras
                    and (self.review_image is None or self.review_image[0] != identity)):
                self.review_image = None
                self._review_ordinal = None
                self._review_trigger = None
                self._set_review_parts(None)
                if browse_saved:
                    self.browse_frame(latest=True)
        for key, view in self.views.items():
            view.set_active(key == identity)

    def _update_region_caption(self):
        if self._board_visible:
            self.footer_roi.set("绿色圆点标出棋盘格内角点")
            return
        view = self.views.get(self.active_camera)
        roi = view.roi if view else None
        self.footer_roi.set(f"检查区域  {roi[0]}, {roi[1]}   {roi[2]} × {roi[3]} px" if roi else
                            "Shift + 拖动检查局部图像")

    def _inspect_region(self, camera, roi):
        self.select_view(camera)
        self._update_region_caption()
        self.tabs.select(self.quality_panel)
        self._sync_quality_scope(None)
        self._quality_due = 0.

    def clear_region(self):
        if view := self.views.get(self.active_camera):
            view.roi = None
            view.draw()
            self._update_region_caption()
            self._sync_quality_scope(None)
            self._quality_due = 0.

    def set_region(self):
        if not (view := self.views.get(self.active_camera)) or (image := self._image(self.active_camera)) is None:
            self.activity.set("先连接预览或打开一张图像，再选择检查区域。")
            return
        if view.corrected:
            self.activity.set("先切回原始图像预览，再输入像素坐标。")
            return
        ih, iw = image.pixels.shape[:2]

        def parse(*fields):
            x, y, width, height = map(int, fields)
            if min(x, y) < 0 or min(width, height) < 1 or x + width > iw or y + height > ih:
                raise ValueError
            return x, y, width, height

        x, y, width, height = view.roi or (0, 0, iw, ih)
        dialog = CaptureFieldDialog(
            self.root, "输入检查区域", f"原图尺寸为 {iw} × {ih} 像素。坐标从原图左上角开始。",
            (("左边 x", x, "从原图左边缘到检查框左边的像素数。填零表示与左边缘对齐。"),
             ("上边 y", y, "从原图上边缘到检查框上边的像素数。填零表示与上边缘对齐。"),
             ("宽度", width, "检查框横向包含的像素数，至少为 1。"),
             ("高度", height, "检查框纵向包含的像素数，至少为 1。")),
            parse, "请填写四个整数，并让检查框完整落在原图内。",
        )
        if dialog.result is None:
            return
        view.roi = dialog.result
        view.draw()
        self._inspect_region(self.active_camera, view.roi)

    def copy_region(self):
        view = self.views.get(self.active_camera)
        if not view or not view.roi:
            self.activity.set("先绘制一个检查区域，再复制坐标。")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(", ".join(map(str, view.roi)))
        self.activity.set("已复制原始像素坐标。")

    def _viewport_action(self, function):
        if view := self.views.get(self.active_camera):
            getattr(view, function)()

    def _preview_save_candidate(self, camera=None):
        if self._saving_preview_image:
            return None, "正在保存这张预览原图。"
        if self._busy() or self._closing:
            return None, "采集结束后可在复查页查看已保存的原图。"
        if self.preview is None or self._releasing_preview:
            return None, "请先选择相机，再点“连接并预览”。"
        camera = self.active_camera if camera is None else camera
        view = self.views.get(camera)
        image = view.displayed_image if view is not None else None
        binding = self.preview.camera_binding(camera)
        if binding is None or image is None or image.received_monotonic_ns < binding.first_receipt_ns:
            return None, "请选中正在显示预览的相机画面。首次连接时请等待新图像。"
        return (binding, image), ""

    def _preview_save_help(self, _event=None):
        sample, reason = self._preview_save_candidate()
        if reason:
            return reason
        binding, image = sample
        height, width = image.pixels.shape[:2]
        return (f"保存 {self._view_title(binding.device.id)} 当前显示这一帧的原始像素。"
                f"完整画幅 {width} × {height}，有效位深 {image.significant_bits} 位。\n"
                "选择文件名后保存为无损 TIFF，并继续预览。")

    def _show_preview_save_note(self, text, detail=None):
        self._preview_save_detail = detail or text
        self._preview_save_summary = text

    def _save_preview_image(self):
        sample, reason = self._preview_save_candidate()
        if reason:
            self._show_preview_save_note(reason)
            return
        binding, image = sample
        title = self._view_title(binding.device.id)
        seconds, nanoseconds = divmod(image.received_utc_ns, 1_000_000_000)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(seconds))
        self._saving_preview_image = True

        def failed(exc):
            self._saving_preview_image = False
            detail = ui_error(exc)
            self._show_preview_save_note(detail.split("\n", 1)[0], detail)

        try:
            target = filedialog.asksaveasfilename(
                parent=self.root, title=f"保存预览原图 · {title}", confirmoverwrite=False,
                initialfile=f"预览原图-{stamp}-{nanoseconds // 1_000_000:03d}.tiff",
                defaultextension=".tiff", filetypes=[("TIFF 原图", ("*.tif", "*.tiff"))],
            )
        except tk.TclError as exc:
            failed(exc)
            return
        if not target or self._closing:
            self._saving_preview_image = False
            return
        self._show_preview_save_note("正在写入原图…", target)

        def saved(path):
            self._saving_preview_image = False
            self._last_preview_original = path
            self._show_preview_save_note(f"已保存 {title} · {path.name}", str(path))

        self._job(lambda: save_preview_image(Path(target), binding.device, image, readback=binding.readback), saved, failed)

    def _comparison_preview_frame(self, camera, *, archive=False):
        if archive:
            if self._busy() or self._closing:
                return None, "请等本次采集结束后再对照预览。"
            if self._camera_settings_busy or self._releasing_preview:
                return None, "请等相机设置更新并收到新画面。"
            if self.preview is None:
                return None, "点“定位相机”连接预览，也可打开右图离线对照。"
            sample = self.preview.bound_image(camera)
            if sample is None:
                return None, f"等待 {self._view_title(camera)} 的新画面。可用“定位相机”检查连接。"
            binding, image = sample
            return preview_original(binding.device, image, binding.readback), ""
        sample, reason = self._preview_save_candidate(camera)
        if sample is None:
            if self.preview is not None and not self._busy() and not self._releasing_preview:
                reason = f"请等待 {self._view_title(camera)} 的预览画面，或打开已保存的右图。"
            return None, reason
        binding, image = sample
        return preview_original(binding.device, image, binding.readback), ""

    def _locate_archive_comparison_camera(self, original, roi):
        if reason := self._setup_block():
            return reason
        identity = original.device.id
        title = self._view_title(identity)
        if self.preview:
            current, _reason = self._comparison_preview_frame(identity, archive=True)
            if current is not None:
                mismatch = preview_region_mismatch(original, current)
                if mismatch in ("camera", "identity_unknown"):
                    return "归档记录与当前预览无法核对为同一台相机。请检查设备身份。"
                if self._open_preview_quality_camera(identity):
                    view = self.views[identity]
                    note = ""
                    if roi and mismatch is None and not view.corrected and not self._board_visible:
                        view.roi = roi
                        view.draw()
                        self._inspect_region(identity, roi)
                        note = "已在画质页使用对照检查框。"
                    self.activity.set(note + "调整后，可从预览下方的“原图对照”返回并取新画面。")
                    return ""
            self._focus_guide_target(None, self.devices)
            self.activity.set(f"等待 {title} 的预览。若未连接这台相机，可断开预览后到列表中选择。")
            return ""
        self._focus_guide_target(None, self.devices)
        if self.devices.exists(identity) and not self._camera_not_discovered(identity):
            self.devices.focus(identity)
            self.devices.see(identity)
            self.activity.set(f"已定位 {title}。选中这台相机，再点“连接并预览”。")
        else:
            self.activity.set(f"请连接归档中的相机 {identity}，再点“刷新”。")
        return ""

    def _comparison_window(self):
        window = self._preview_compare_window
        if window is None or not window.top.winfo_exists():
            window = self._preview_compare_window = PreviewCompareWindow(
                self.root, self._job, self._comparison_preview_frame, self._locate_archive_comparison_camera,
            )
        window.top.deiconify()
        window.top.lift()
        return window

    def _open_preview_compare(self):
        if self._busy() or self._closing:
            return
        prior = self._preview_compare_window
        window = self._comparison_window()
        if prior is window:
            return
        if self._last_preview_original is not None:
            window.open_file(0, self._last_preview_original)
        else:
            window.top.after_idle(lambda: window.choose_file(0) if window.top.winfo_exists() else None)

    def apply_preview_options(self, _event=None):
        for view in self.views.values():
            view.guide = self.guide.get()
            view.palette = "inferno" if self.palette.get() == "Inferno" else "native"
            self._display_image(view, self._image(view.camera_id))
            view.draw()

    def _populate_phases(self, index=0):
        self.phases.delete(*self.phases.get_children())
        modes = self._phase_heading_font.measure("拍摄方式") + 12
        roles = self._phase_heading_font.measure("用途") + 12
        limits = self._phase_heading_font.measure("计划参数") + 12
        widest_limit = limits
        for idx, row in enumerate(self.config["phases"]):
            mode = MODE_LABELS[row["mode"]]
            if row.get("start_confirmation"):
                mode = f"{mode[:2]} · 确认"
            role, parameters = ROLE_LABELS[row["role"]], phase_limit_text(row)
            self.phases.insert("", "end", iid=str(idx), text=mode,
                               values=(role, parameters))
            modes = max(modes, self._phase_font.measure(mode) + 22)
            roles = max(roles, self._phase_font.measure(role) + 12)
            widest_limit = max(widest_limit, self._phase_font.measure(parameters) + 12)
        self._phase_widths = modes, roles, limits, widest_limit
        if self.phases.get_children():
            selected = str(min(index, len(self.config["phases"]) - 1))
            self.phases.selection_set(selected)
            self.phases.see(selected)
        self.root.after_idle(self._fit_phase_columns)

    def _fit_phase_columns(self, _event=None):
        available = self.phases.winfo_width() - 4
        if available < 2:
            return
        modes, roles, heading, widest_limit = self._phase_widths
        room_for_limits = available - modes - roles
        show_limits = room_for_limits >= heading
        if not show_limits:
            modes = max(1, available - roles)
        signature = (modes, roles, room_for_limits, show_limits)
        if signature != self._phase_layout:
            self._phase_layout = signature
            self.phases.configure(displaycolumns=("role", "limits") if show_limits else ("role",))
            self.phases.column("#0", width=modes, minwidth=0, stretch=False)
            self.phases.column("role", width=roles, minwidth=0, stretch=False)
            self.phases.column("limits", width=max(1, room_for_limits), minwidth=0, stretch=False)
        index = self._selected_phase()
        if index is not None and (not show_limits or widest_limit > room_for_limits):
            parameters = self.phases.set(str(index), "limits")
            caption = f"阶段 {index + 1} · {parameters}"
            if self.phase_details.cget("text") != caption:
                self.phase_details.configure(text=caption)
            if not self.phase_details.winfo_manager():
                self.phase_details.grid()
        elif self.phase_details.winfo_manager():
            self.phase_details.grid_remove()

    def _selected_phase(self):
        return int(self.phases.selection()[0]) if self.phases.selection() else None

    def _phase_help(self, event):
        item = ""
        column = ""
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            column = self.phases.identify_column(event.x)
            if self.phases.identify_region(event.x, event.y) == "heading":
                return {
                    "#0": "按列表顺序执行。模式决定相机何时拍摄，例如定时、连续或收到触发。",
                    "#1": "标明图像用于实验、参考还是镜头标定。此信息会写入台账。",
                    "#2": "显示定时频率、张数或时长。事件模式的两项张数分别对应事件前和事件后。",
                }.get(column, "")
            item = self.phases.identify_row(event.y)
        index = int(item) if item else self._selected_phase()
        if index is None:
            return "采集阶段由上到下执行。双击一行可调整触发方式、帧数和时长。"
        phase = self.config["phases"][index]
        instruction = f"\n开始前等待确认 · {phase['start_confirmation']}" if phase.get("start_confirmation") else ""
        if rule := phase.get("csv_trigger"):
            instruction += "\n\n" + csv_trigger_help(rule)
        limits = f"计划 · {self.phases.set(str(index), 'limits')}"
        if phase.get("frame_count") and phase.get("duration_s"):
            limits += "\n先达到张数或时长上限时结束。"
        mode_help, role_help = MODE_HELP[phase["mode"]], ROLE_HELP[phase["role"]]
        rows = (role_help, mode_help, limits) if column == "#1" else (
            (limits, mode_help, role_help) if column == "#2" else (mode_help, role_help, limits))
        heading = f"阶段 {index + 1} · {MODE_LABELS[phase['mode']]}"
        reference = self.config.get("metadata", {}).get(CONTINUATION_REFERENCE)
        if isinstance(reference, dict):
            start, stages = reference.get("source_start_phase"), reference.get("phase_plans")
            if type(start) is int and start > 1 and isinstance(stages, list) and index < len(stages):
                heading += f" · 对照原阶段 {start + index}"
        return "\n".join((heading, *rows)) + instruction + "\n双击或按 Enter 编辑。"

    def _phase_block(self, action):
        if reason := self._setup_block():
            return reason
        if action == "添加":
            return ""
        index = self._selected_phase()
        if index is None:
            return "请先在上方选择一个阶段。"
        count = len(self.config["phases"])
        if action == "删除" and count == 1:
            return "采集计划至少需要一个阶段。可先添加另一个阶段，再删除此阶段。"
        if action == "↑" and index == 0:
            return "当前阶段已经在第一行。"
        if action == "↓" and index == count - 1:
            return "当前阶段已经在最后一行。"
        return ""

    def _edit_clicked_phase(self, event):
        if item := self.phases.identify_row(event.y):
            self.phases.selection_set(item)
            self.edit_phase()

    def _hardware_mapping_phase(self):
        camera_ids = set(self.devices.selection())
        if camera_ids:
            for index, phase in enumerate(self.config["phases"]):
                pairing = phase.get("hardware_pairing")
                counters = pairing["first_counter"] if pairing else None
                if hardware_mapping_error(camera_ids, phase.get("hardware_source"), counters):
                    return index
        return None

    def _edit_hardware_mapping(self):
        if (index := self._hardware_mapping_phase()) is None:
            return False
        if reason := self._setup_block(needs_camera=True):
            self.activity.set(reason)
            return True
        self.tabs.select(self.plan_panel)
        row = str(index)
        self.phases.selection_set(row)
        self.phases.focus(row)
        self.phases.see(row)
        self.activity.set(f"请核对阶段 {index + 1} 对应当前相机的物理输入和脉冲计数。")
        self.edit_phase(wiring=True)
        return True

    def show_template(self):
        if self._setup_block():
            return
        cameras = {identity: self._view_title(identity) for identity in self.devices.selection()}
        geometry, support = {}, {}
        if cameras:
            try:
                config = self._read_config(phases=[{"mode": "continuous"}])
                samples, evidence, _settings = self._preview_evidence(config)
                geometry = plan_camera_geometry(config.cameras, samples)
                support = {identity: trigger_evidence(identity, evidence) for identity in cameras}
            except CaptureError as exc:
                self._error(exc)
                return
        self._template_window = PlanTemplateWindow(
            self.root, current=self.config["phases"], cameras=cameras, geometry=geometry, support=support,
            buffer_limit=self.config["max_buffer_bytes"], external=self.external.get(), read_ports=self._read_phase_ports,
            apply=self._install_template,
        )

    def _install_template(self, phases):
        if reason := self._setup_block():
            self._template_window.note.set(reason)
            return False
        if phases != self.config["phases"]:
            self.config["phases"] = phases
            self._invalidate_preflight()
            self._populate_phases(len(phases) - 1)
            self.plan_summary.set("已应用模板。可检查触发条件、磁盘空间，也可先试采。")
        return True

    def edit_phase(self, *, add=False, wiring=False, csv_source=False, recheck=False):
        if self._phase_block("添加" if add else "编辑"):
            return
        index = None if add else self._selected_phase()
        if index is None and not add:
            return
        phase = {"mode": "timed", "role": "test", "fps": 10., "frame_count": 20} if add else self.config["phases"][index]
        cameras = {device: self._view_title(device) for device in self.devices.selection()}

        def save(value):
            self._replace_phase(index, value)
            if recheck:
                self.check_plan()

        dialog = PhaseDialog(self.root, phase, save, cameras=cameras, read_ports=self._read_phase_ports, submit=self._job)
        if wiring:
            dialog.top.after_idle(dialog.open_hardware)
        if csv_source:
            dialog.top.after_idle(dialog.open_csv_trigger)

    def _read_phase_ports(self, done, failed):
        if reason := self._setup_block(needs_camera=True):
            failed(CaptureError("TRIGGER_PROBE_FAILED", reason))
            return
        if preview := self.preview:
            done({identity: support for identity in self.devices.selection() if (support := preview.trigger_support(identity))})
            return
        try:
            config = self._read_config(phases=[{"mode": "continuous"}])
            registry = self._registry()
        except CaptureError as exc:
            failed(exc)
            return
        self._camera_settings_busy = True

        def finished(result):
            self._camera_settings_busy = False
            done(result)

        def error(exc):
            self._camera_settings_busy = False
            failed(exc)

        self._job(lambda: probe_trigger_support(config, registry), finished, error)

    def _replace_phase(self, index, phase):
        if index is None:
            self.config["phases"].append(phase)
            index = len(self.config["phases"]) - 1
        else:
            self.config["phases"][index] = phase
        self._invalidate_preflight()
        self._populate_phases(index)

    def remove_phase(self):
        if self._phase_block("删除"):
            return
        if (index := self._selected_phase()) is not None:
            self.config["phases"].pop(index)
            self._invalidate_preflight()
            self._populate_phases(index)

    def move_phase(self, offset):
        if self._phase_block("↑" if offset < 0 else "↓"):
            return
        index = self._selected_phase()
        rows = self.config["phases"]
        if index is not None and 0 <= index + offset < len(rows):
            rows.insert(index + offset, rows.pop(index))
            self._invalidate_preflight()
            self._populate_phases(index + offset)

    def choose_output(self, *, parent=None):
        if self._setup_block():
            return False
        if folder := filedialog.askdirectory(parent=parent or self.root, title="选择新实验的保存位置"):
            self.output_path.set(self._new_path(folder))
            return True
        return False

    def _schedule_output_check(self, *_args):
        if self._output_check_job is not None:
            self.root.after_cancel(self._output_check_job)
            self._output_check_job = None
        self._output_check = None
        self._output_check_ticket = object()
        if not self._closing and self.output_path.get().strip():
            self._output_check_job = self.root.after(180, self._check_planned_output)

    def _check_planned_output(self):
        self._output_check_job = None
        if self._closing or self._output_check_active:
            return
        path = self.output_path.get().strip()
        repeat_source = repeat_source_directory(self.config.get("metadata", {}))
        ticket = self._output_check_ticket
        self._output_check_active = True

        def finished(result):
            self._output_check_active = False
            if self._closing:
                return
            if ticket is self._output_check_ticket:
                self._output_check = result
            elif self._output_check_job is None:
                self._schedule_output_check()

        self._job(lambda: inspect_output(path, repeat_source=repeat_source, suggest=True), finished,
                  lambda _exc: finished(None))

    def _output_problem(self):
        if self._output_check is not None and self._output_check.issues:
            return ISSUE_LABELS[self._output_check.issues[0]["code"]]
        return ""

    def _output_help(self, _event=None):
        if reason := self._setup_block():
            return reason
        if not (path := self.output_path.get().strip()):
            return "点“选择位置…”，或输入本次实验的新目录。"
        lines = [path, self._output_problem() or "开始采集后，原始图像和台账会保存到这个目录。"]
        if self._output_check is not None and (suggestion := self._output_check.suggestion):
            lines.append(f"建议使用\n{suggestion}")
        return "\n\n".join(lines)

    def _use_suggested_output(self, suggestion):
        if self._setup_block() or self._saved_panel() or self._output_check is None:
            return
        if self._output_check.suggestion != suggestion:
            return
        self._fill_output_suggestion(suggestion)

    def _fill_output_suggestion(self, suggestion):
        self.tabs.select(self.plan_panel)
        self.output_path.set(str(suggestion))
        self.activity.set("已填写新路径。请核对实验名称。")

    def choose_cti(self, dialog=None):
        if reason := self._cti_block():
            self.activity.set(reason)
            return
        paths = filedialog.askopenfilenames(parent=dialog or self.root, title="选择厂商 GenTL 接口文件",
                                            filetypes=[("GenTL 接口文件", "*.cti")])
        if paths:
            self.config["cti_files"] = list(paths)
            if dialog is not None:
                dialog.destroy()
            self._invalidate_preflight()
            self.discover()

    def load_plan(self):
        if reason := self._setup_block(needs_idle_camera=True):
            self.activity.set(reason)
            return
        if path := filedialog.askopenfilename(parent=self.root, title="载入采集计划", filetypes=[("采集计划", "*.json")]):
            try:
                self._set_config(CaptureConfig.load(path))
                self.discover()
            except Exception as exc:
                self._error(exc)

    def save_plan(self):
        try:
            config = self._read_config()
            path = filedialog.asksaveasfilename(parent=self.root, title="保存采集计划", defaultextension=".json",
                                                filetypes=[("采集计划", "*.json")])
            if path:
                Path(path).write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.activity.set("采集计划已保存。")
        except Exception as exc:
            self._error(exc)

    def _current_draft(self):
        try:
            return capture_draft(self._read_config(), self.output_path.get().strip(), self._selected_phase(),
                                 self._trial_results.values())
        except (CaptureError, OSError, ValueError):
            return None

    def _draft_help(self, _event=None):
        if reason := self._setup_block(needs_idle_camera=True):
            return reason
        if draft := self._stored_draft:
            output = draft.output or "待选择保存位置"
            return (f"{draft.config.name} · {len(draft.config.phases)} 个阶段 · {len(draft.trials)} 个关联试采\n{output}\n"
                    "载入本机保存的计划和输出位置，重新核对关联试采。恢复后可检查计划或连接预览。")
        return "首次修改有效计划后，工作台会自动保存草稿。"

    def _draft_loaded(self, draft):
        if self._closing:
            return
        self._stored_draft = draft
        if draft is None:
            self.restore_draft_button.pack_forget()
        elif not self.restore_draft_button.winfo_manager():
            self.restore_draft_button.pack(side=tk.LEFT, padx=(6, 0), before=self.save_plan_button)

    def _cancel_draft_timer(self):
        if self._draft_after is not None:
            self.root.after_cancel(self._draft_after)
            self._draft_after = None

    def _schedule_draft_save(self, _event=None):
        if self._draft_tracking and not self._restoring_draft and not self._closing:
            self._cancel_draft_timer()
            self._draft_after = self.root.after(600, self._save_draft)

    def _save_draft(self):
        self._draft_after = None
        if not self._draft_tracking:
            return
        draft = self._current_draft()
        if draft is None or draft == self._draft_last:
            return
        previous, self._draft_last = self._draft_last, draft
        self._offer_startup_draft = False
        edit_ns = time.time_ns()
        self._draft_saving += 1

        def completed(saved):
            self._draft_saving -= 1
            if self._draft_last is draft:
                self._draft_loaded(saved)

        def failed(_exc):
            self._draft_saving -= 1
            if self._draft_last is draft:
                self._draft_last = previous
                self.activity.set("草稿未能保存。可使用“保存计划”导出当前设置。")

        self._job(lambda: draft if save_capture_draft(draft, edit_ns) else load_capture_draft(), completed, failed)

    def restore_capture_draft(self):
        if reason := self._setup_block(needs_idle_camera=True):
            self.activity.set(reason)
            return
        self._cancel_draft_timer()
        self._restoring_draft = True
        self.activity.set("正在读取上次计划并核对关联试采…")

        def read():
            draft = load_capture_draft()
            if draft is None:
                return None
            loaded, missing = load_draft_trials(draft, cancelled=lambda: self._closing)
            return draft, loaded, missing

        def applied(result):
            self._restoring_draft = False
            if self._closing:
                return
            if result is None:
                self._draft_loaded(None)
                self.activity.set("暂时没有可恢复的计划。")
                return
            draft, loaded, missing = result
            self._offer_startup_draft = False
            self._trial_results = {item["trial"]["phase_number"]: item for item in loaded}
            self._set_config(draft.config)
            self.output_path.set(draft.output)
            self.external.set(False)
            self._populate_phases(draft.selected_phase)
            self.tabs.select(self.plan_panel)
            self._draft_last = self._current_draft()
            self._draft_loaded(draft)
            summary = f"已恢复计划 · 已核对试采 {len(loaded)} 次"
            if missing:
                stages = "、".join(map(str, missing))
                summary += f"\n阶段 {stages} 的试采待定位，可从“最近实验”查找。"
            self.plan_summary.set(summary)
            self.discover()

        def failed(exc):
            self._restoring_draft = False
            if not self._closing:
                self._error(exc)

        self._job(read, applied, failed)

    def _settings_camera(self):
        selected = self.devices.selection()
        return self.active_camera if self.active_camera in selected else next(iter(selected), None)

    def _camera_settings_block(self):
        if reason := self._setup_block(needs_camera=True):
            return reason
        if self.preview is not None:
            info = self.preview.status().get(self._settings_camera(), {})
            state = info.get("state")
            if state == "failed":
                return ("这台相机的预览失败。请检查连接，再使用“重连相机”。" if info.get("retryable") else
                        "旧连接尚未释放。请悬停在异常画面上查看详情。")
            if state == PART_REQUIRED:
                return "先在预览下方选择图像部件，再调整这台相机的其他参数。"
            if state != "preview":
                return "等这台相机出现预览图像后再调参。"
        return ""

    def _batch_settings_block(self):
        return self._setup_block(needs_idle_camera=True) or (
            "请按住 Ctrl 选择两台或更多相机。" if len(self.devices.selection()) < 2 else "")

    def edit_camera_batch(self):
        if reason := self._batch_settings_block():
            self.activity.set(reason)
            return
        if (dialog := self._batch_settings_dialog) is not None and dialog.top.winfo_exists():
            dialog.top.lift()
            return
        try:
            config = self._read_config()
        except CaptureError as exc:
            self._error(exc)
            return
        registry = self._registry()
        self._camera_settings_busy = True
        self.activity.set("正在读取所选相机的当前值与取值范围…")

        def ready(data):
            self._camera_settings_busy = False
            if self._closing:
                return
            readings = data[1]
            labels = {item.id: self._view_title(item.id) for item in config.cameras}
            available = sum(evidence.error is None for evidence in readings.values())
            self.activity.set(f"已读取 {available}/{len(config.cameras)} 台相机的参数。")
            self._batch_settings_dialog = BatchCameraDialog(
                self.root, config, readings, labels,
                lambda dialog, changes: self._verify_camera_batch(dialog, config, changes, readings, registry))

        self._job(lambda: probe_camera_setup(config, registry, triggers=False, include_readback=True),
                  ready, self._camera_settings_failed)

    def _same_batch_plan(self, config):
        if self._closing or self._batch_settings_block():
            return False
        try:
            return self._read_config() == config
        except CaptureError:
            return False

    def _verify_camera_batch(self, dialog, config, changes, readings, registry):
        if not self._same_batch_plan(config):
            dialog.fail("当前计划已变化，请关闭窗口后重新核验。")
            return
        self._camera_settings_busy = True
        self.activity.set("正在逐台核验参数并恢复相机原值…")

        def finished(result):
            self._camera_settings_busy = False
            if self._closing or not dialog.top.winfo_exists():
                return
            applied = result.plan is not None and self._same_batch_plan(config)
            if applied:
                updated = {item["id"]: item for item in result.plan.to_dict()["cameras"]}
                cached = {item["id"]: item for item in self.config["cameras"]}
                self.config["cameras"] = list({**cached, **updated}.values())
                reason = "多机参数已更新，可连接预览或重新检查计划。"
                self._invalidate_preflight(reason=reason)
                self.activity.set(reason)
            elif any(row.state == "restore_failed" for row in result.rows):
                reason = "有相机未确认恢复原值。请核对参数后重新检查计划。"
                self._invalidate_preflight(reason=reason)
                self.activity.set(reason)
            else:
                self.activity.set("多机参数未写入计划，请查看逐台结果。")
            dialog.finish(result, applied)

        def failed(exc):
            self._camera_settings_busy = False
            if not self._closing and dialog.top.winfo_exists():
                reason = "多机参数核验未完成。请检查相机参数后重新检查计划。"
                self._invalidate_preflight(reason=reason)
                self.activity.set(reason)
                dialog.fail(exc)

        self._job(lambda: verify_batch_settings(config, changes, readings, registry), finished, failed)

    def edit_camera(self, identity=None, *, focus_feature=None, recheck=False):
        if reason := self._camera_settings_block():
            self.activity.set(reason)
            return
        identity = identity or self._settings_camera()
        if preview := self.preview:
            if parameters := preview.parameters(identity):
                device, features, settings = parameters
                FeatureDialog(self.root, device, features, settings,
                              lambda changes: self._adjust_preview(preview, identity, settings, changes), preview=True)
            return
        registry = self._registry()
        settings = next((item["settings"] for item in self.config["cameras"] if item["id"] == identity), {})

        def features():
            with registry.camera(identity) as camera:
                return camera.device, camera.features()

        self._camera_settings_busy = True
        self.activity.set(f"正在读取相机 {identity} 的参数…")
        self._job(features, lambda data: self._camera_dialog(data, registry, settings, focus_feature, recheck),
                  self._camera_settings_failed)

    def _camera_settings_failed(self, exc):
        self._camera_settings_busy = False
        if not self._closing:
            self._error(exc)

    def _camera_dialog(self, data, registry, settings, focus_feature, recheck):
        self._camera_settings_busy = False
        if self._closing:
            return
        device, features = data
        self.activity.set(f"已读取相机 {device.id} 的参数。")

        def accept(changes):
            if recheck:
                self._set_camera_settings(device.id, changes, recheck=True)
                return
            self._camera_settings_busy = True
            self.activity.set(f"正在向相机 {device.id} 应用参数…")

            def apply():
                with registry.camera(device.id) as camera:
                    camera.set_features(changes)
                return changes
            self._job(apply, lambda applied: self._set_camera_settings(device.id, applied), self._camera_settings_failed)
        FeatureDialog(self.root, device, features, settings, accept, focus_feature=focus_feature, plan_only=recheck)

    def _update_part_menu(self, states):
        identity = self._settings_camera()
        status = states.get(identity, {})
        options = status.get("image_parts", ())
        if not options:
            self._part_menu_signature = None
            self._part_selectors.clear()
            self.image_part_row.grid_remove()
            return
        signature = identity, options
        if self._part_menu_signature != signature:
            self._part_menu_signature = signature
            self._part_selectors = {}
            tips = {}
            for option in options:
                source, x, y, fmt = option.selector.split("/")
                label = f"{fmt} · {option.width} × {option.height} · 源 {source} · ({x}, {y})"
                self._part_selectors[label] = option.selector
                detail = "括号内是图像左上角的横向和纵向偏移，单位为像素。"
                if "-" in (source, x, y):
                    detail += "\n短横线表示相机未报告该项。"
                tips[label] = f"{detail}\n计划中的编号为 {option.selector}。"
            self.image_part_menu.configure(values=tuple(self._part_selectors))
            self._part_help.set_choices(tips)
        if status["state"] == "adjusting":
            value = "正在读取所选部件…"
        else:
            chosen = status.get("image_part")
            value = next((label for label, selector in self._part_selectors.items() if selector == chosen), None)
            value = value or (f"当前部件 {chosen}" if chosen else "请选择主图像")
        if self.image_part_menu.get() != value:
            self.image_part_menu.set(value)
        busy = self._setup_block() or status["state"] == "adjusting"
        self.image_part_menu.state(["disabled" if busy else "!disabled"])
        saved = status.get("features", {}).get(SAVE_ALL_2D, False)
        if status["state"] != "adjusting":
            self._save_all_2d.set(saved)
        if len(options) > 1 or saved:
            self.archive_parts.grid()
            self.archive_parts.state(["!disabled" if not busy and status["state"] == "preview" else "disabled"])
        else:
            self.archive_parts.grid_remove()
        self.image_part_row.grid()

    def _part_storage_help(self, _event=None):
        status = self.preview.status().get(self._settings_camera(), {}) if self.preview else {}
        if status.get("state") == PART_REQUIRED:
            return "请先选主图像，收到预览后即可开启。"
        detail = "为同一 GenTL 缓冲区中的每个可解码二维部件保存一张原始 TIFF。采前检查会按所有部件计算空间和缓冲。"
        if status.get("features", {}).get(SAVE_ALL_2D):
            detail += f"\n预览确认每帧可保存 {status['archived_parts']} 张图像。"
        return detail

    def _choose_part_storage(self):
        identity = self._settings_camera()
        if self.preview is None or self._setup_block() or self.preview.status().get(identity, {}).get("state") != "preview":
            return
        current = next(camera for camera in self.preview.cameras if camera.id == identity)
        self._adjust_preview(self.preview, identity, current.settings, {**current.settings, SAVE_ALL_2D: self._save_all_2d.get()})

    def _choose_preview_part(self, _event=None):
        preview = self.preview
        identity = self._settings_camera()
        if preview is None or not self._part_menu_signature or self._part_menu_signature[0] != identity or self._setup_block():
            return
        selector = self._part_selectors.get(self.image_part_menu.get())
        if selector is None:
            return
        current = next((camera for camera in preview.cameras if camera.id == identity), None)
        if current is not None:
            self._adjust_preview(preview, identity, current.settings, {**current.settings, IMAGE_PART: selector})

    def _adjust_preview(self, preview, identity, previous, settings):
        if self.preview is not preview:
            return
        status = preview.status()[identity]
        if previous == settings and status["state"] != PART_REQUIRED:
            return
        image = preview.latest_image(identity)
        before = status.get("features", {})
        view = self.views.get(identity)
        roi = view.roi if view else None
        try:
            future = preview.reconfigure(identity, settings)
        except CaptureError as exc:
            self._error(exc)
            return
        self._preview_adjustment = None
        self._camera_settings_busy = True
        self._sync_quality_scope(None)
        self.activity.set(f"正在更新相机 {identity} 的参数和画面…")

        def finished(result):
            config, frame = result
            self._set_camera_settings(identity, dict(config.settings))
            if self._closing:
                return
            after = preview.status()[identity].get("features", {})
            height, width = frame.pixels.shape[:2]
            geometry = ("width", "height", "offset_x", "offset_y", "binning_x", "binning_y", "decimation_x", "decimation_y")
            old_part = image.metadata.get(IMAGE_PART, {}).get("selector") if image is not None else None
            new_part = frame.metadata.get(IMAGE_PART, {}).get("selector")
            moved = old_part != new_part or any(before.get(key) != after.get(key) for key in geometry)
            changed_size = image is not None and image.pixels.shape[:2] != (height, width)
            archive_changed = previous.get(SAVE_ALL_2D, False) != config.settings.get(SAVE_ALL_2D, False)
            if (moved or changed_size) and (view := self.views.get(identity)):
                view.roi = None
                view.show_calibration(None)
                self._update_region_caption()
            note = f"已收到新画面  {width} × {height} 像素 · {frame.pixel_format}"
            if archive_changed:
                archive_note = (f"预览确认每帧可保存 {1 + len(frame.auxiliary_images)} 张图像。"
                                if config.settings.get(SAVE_ALL_2D) else "采集计划已设为只保存主图像。")
                note += f"\n{archive_note}"
                self.activity.set(archive_note)
            if moved or changed_size:
                note += "\n采集区域已更新，可重新绘制检查框。"
            snapshot = self._preview_adjustment = {"preview": preview, "camera": identity, "note": note}
            self.tabs.select(self.quality_panel)
            self._quality_due = 0.
            if image is not None and not archive_changed and not moved and not changed_size and image.pixel_format == frame.pixel_format:
                def sample():
                    return image_quality(image, roi), image_quality(frame, roi)

                def compared(results):
                    if self._preview_adjustment is snapshot and self.preview is preview and not self._closing:
                        old, new = results
                        p50 = [row["percentiles_p5_p50_p95"][1] / row["code_max"] for row in results]
                        snapshot["compared_roi"] = roi
                        where = "选区" if roi else "全图"
                        snapshot["note"] = (f"调参前后  ·  {where}中位亮度 {p50[0]:.1%} → {p50[1]:.1%}\n"
                                            f"近饱和像素 {old['bright_fraction']:.1%} → {new['bright_fraction']:.1%}")
                        self._quality_due = 0.

                self._job(sample, compared, lambda _exc: None)

        self._job(future.result, finished, self._camera_settings_failed)

    def _set_camera_settings(self, identity, settings, *, recheck=False):
        self._camera_settings_busy = False
        updated = {"id": identity, "settings": settings}
        cameras = self.config["cameras"]
        if any(item["id"] == identity for item in cameras):
            self.config["cameras"] = [updated if item["id"] == identity else item for item in cameras]
        else:
            cameras.append(updated)
        self._invalidate_preflight()
        if recheck:
            self.activity.set("相机参数已保存到计划，正在重新检查。")
            self.root.after_idle(self.check_plan)
        else:
            self.activity.set(f"相机 {identity} 已接受并回读参数。")

    def toggle_preview(self):
        if reason := self._setup_block(needs_camera=True):
            self.activity.set(reason)
            return
        if self.preview:
            self._preview_adjustment = None

            def released():
                self.activity.set("预览已断开。")
                if self._saved_panel():
                    self._show_saved_workspace_images()

            self._after_preview(released)
            return
        try:
            cameras = tuple(CameraConfig.from_dict(row) for row in self._selected_camera_rows())
            self._preview_save_summary = self._preview_save_detail = ""
            boxes = self.config.get("quality_regions", {})
            self._preview_regions_pending = {camera.id: QualityRegion.from_dict(boxes[camera.id])
                                             for camera in cameras if camera.id in boxes}
            self.session = None
            self.review_image = None
            self.counter.set("")
            self._show_views([camera.id for camera in cameras])
            if self.tabs.select() == str(self.review_panel):
                self.tabs.select(self.quality_panel)
            self._invalidate_preflight(reason="已开始连接预览。收到图像后可重新检查计划。")
            self.preview = PreviewSession(cameras, self._registry())
            self.preview.start()
            self._offer_camera_start = False
            self.activity.set("正在连接预览相机…")
        except Exception as exc:
            self._error(exc)

    def _after_preview(self, action):
        if self._releasing_preview:
            return
        if self.preview is None:
            action()
            return
        self._releasing_preview = True
        self._quality_checked_preview = None
        self._preview_quality_results = ()
        self._preview_regions_pending.clear()
        self._sync_quality_scope(None)
        self.preview.stop()
        self.activity.set("正在释放预览连接…")

        def ready():
            if self.preview is not None and not self.preview.stopped:
                self.root.after(80, ready)
            else:
                self.preview = None
                self._preview_revision = None
                self._releasing_preview = False
                if not self._closing:
                    action()
        self.root.after(50, ready)

    def start(self):
        if not self._busy() and self._saved_panel():
            self._return_to_plan()
            return
        if reason := self._output_problem():
            self.activity.set(reason)
            return
        self._inspect_plan(start=True)

    def check_plan(self):
        self._inspect_plan(start=False)

    def start_checkerboard(self):
        if reason := self._checkerboard_setup_block():
            self.activity.set(reason)
            return
        CheckerboardDialog(self.root, self.devices.selection(),
                           lambda target: self._inspect_plan(start=True, checkerboard=target), capturing=True)

    def show_trial_progress(self, *, follow_recommendation=False):
        if (task := self._trial_batch) is not None:
            task.window.top.deiconify()
            task.window.top.lift()
            return
        if reason := self._plan_input_block():
            self.activity.set(reason)
            return
        if self._edit_hardware_mapping():
            return
        window = self._trial_progress_window
        if window is not None and window.top.winfo_exists():
            window.top.deiconify()
            window.top.lift()
            window.refresh(follow_recommendation=follow_recommendation)
            return
        self._trial_progress_window = TrialProgressWindow(
            self.root, self._load_trial_progress, self._start_progress_trial,
            self._review_progress_trial, self.check_plan, self._start_trial_batch, self._stop_trial_batch,
            self._show_trial_batch_check,
        )

    def _refresh_trial_progress(self):
        window = self._trial_progress_window
        if window is not None and window.top.winfo_exists() and window.top.state() != "withdrawn":
            window.refresh()

    def _load_trial_progress(self, complete, fail):
        if self._plan_trial_progress is not None:
            complete(self._plan_trial_progress)
            return
        try:
            config = self._read_config()
            output = self.output_path.get().strip()
            if not output:
                raise CaptureError("OUTPUT_REQUIRED", "请选择本次实验的保存位置。")
        except CaptureError as error:
            fail(error)
            return
        reports = tuple(self._trial_results.values())
        self._job(lambda: inspect_trial_progress(config, output, reports), complete, fail)

    def _start_progress_trial(self, number):
        phase = str(number - 1)
        if not self.phases.exists(phase) or self._plan_input_block():
            return
        self.tabs.select(self.plan_panel)
        self.phases.selection_set(phase)
        self.phases.see(phase)
        self.start_trial()

    def _review_progress_trial(self, record):
        self._read_trial_report(Path(record["output"]), expected_id=record["experiment_id"])

    def _start_trial_batch(self, numbers, *, guided=False):
        window = self._trial_progress_window
        if reason := self._plan_input_block():
            self.activity.set(reason)
            return
        try:
            config = self._read_config()
        except CaptureError as exc:
            self._error(exc)
            return
        output = self.output_path.get().strip()
        listed = "、".join(str(number) for number in numbers[:12])
        scope = f"阶段 {listed}" if len(numbers) <= 12 else f"阶段 {listed} 等 {len(numbers)} 个阶段"
        operation = ("\n需要确认或触发时，可在阶段列表操作或查看应等待的信号。" if guided else "")
        confirm = CaptureFieldDialog(
            window.top, "开始连续试采", f"按计划顺序试采 {scope}。每个阶段在正式输出位置旁单独保存并核验。"
            "\n如采前检查需要复核，序列会停在该阶段。" + operation,
            (("每阶段试采时限（秒）", f"{DEFAULT_TRIAL_SECONDS:g}",
              "单张阶段每台拍一张。其它阶段受本时限和原计划的张数、时长限制。"),),
            lambda value: _positive_number(value, float), "请输入大于零的秒数。", confirm_text="开始连续试采",
        )
        seconds = confirm.result
        if seconds is None or self._closing or not window.top.winfo_exists():
            return
        try:
            plan_matches = config == self._read_config() and output == self.output_path.get().strip()
        except CaptureError:
            plan_matches = False
        if not plan_matches or self._plan_input_block():
            self.activity.set("相机、计划或保存位置发生变化。请重新查看试采范围。")
            return

        task = self._trial_batch = _TrialBatchTask(window, threading.Event(), len(numbers))
        registry, discovered, watch = self._registry(), self._discovered_ids, self.watch_quality.get()
        external = self.external.get()
        window.begin_batch(numbers)
        self.tabs.select(self.plan_panel)
        self.activity.set("正在准备连续试采…")

        def show(update):
            if self._trial_batch is task and not self._closing:
                window.update_batch(*update)

        def show_live(live):
            if self._trial_batch is task and not self._closing:
                window.update_batch_live(live)

        def report_result(update):
            number, state, report = update
            if self._trial_batch is not task:
                return
            task.verified += state in ("completed", "review")
            self._remember_trial(report, plan=config)
            show((number, state, report["verification"]["committed_frames"]))

        def finish(result):
            self._finish_trial_batch(task, result)

        def launch():
            if self._trial_batch is not task or task.stop.is_set():
                return
            task.running = True
            self.activity.set("正在连续试采。可在阶段列表查看进度。")

            def work():
                return run_trial_batch(
                    config, output, numbers, seconds, registry=registry, discovered_ids=discovered, stop=task.stop,
                    watch_quality=watch, progress=lambda *update: self._jobs.put((show, update)),
                    opened=lambda identity, name, path: remember_experiment(RecentExperiment(identity, name, path)),
                    completed=lambda *update: self._jobs.put((report_result, update)),
                    guided=guided, observe=lambda live: self._jobs.put((show_live, live)), external=external,
                )

            try:
                self._job(work, finish, lambda exc: finish(TrialBatchEnd("failed", error=exc)), daemon=False)
            except RuntimeError as exc:
                finish(TrialBatchEnd("failed", error=exc))

        self._after_preview(launch)

    def _stop_trial_batch(self):
        if (task := self._trial_batch) is not None:
            task.stop.set()
            if not self._closing:
                task.window.stopping_batch()
            if not task.running:
                self._finish_trial_batch(task, TrialBatchEnd("stopped"))

    def _finish_trial_batch(self, task, result):
        if self._trial_batch is not task:
            return
        self._trial_batch = None
        scope = f"已核验 {task.verified}/{task.total} 个阶段"
        phase = f"阶段 {result.phase} " if result.phase is not None else ""
        message = {
            "completed": f"连续试采完成 · {scope}",
            "stopped": f"已停止连续试采 · {scope}。已收到的图像已保留。",
            "preflight": f"{phase}需复核采前检查 · {scope}",
            "failed": f"{phase}试采需复核 · {scope}",
            "no_images": f"{phase}到达试采时限，未保存图像 · {scope}",
        }[result.state]
        if result.error is not None:
            message += f"\n{ui_error(result.error, details=False)}"
        self._cancel_draft_timer()
        self._save_draft()
        if not self._closing:
            self.activity.set(message)
            task.window.finish_batch(result, message)

    def _show_trial_batch_check(self, finding):
        if finding is not None and self._trial_batch is None:
            number, report = finding
            self._show_plan_report(report, phase_offset=number - 1, title=f"阶段 {number} · 试采前检查")

    def start_trial(self):
        if reason := self._plan_input_block():
            self.activity.set(reason)
            return
        if self._edit_hardware_mapping():
            return
        index = self._selected_phase()
        if index is None:
            self.activity.set("请先在采集计划中选择要试采的阶段。")
            return
        phase = self.config["phases"][index]
        mode = phase["mode"]
        title = f"试采阶段 {index + 1} · {MODE_LABELS[mode]}"
        location = "相机采集得到的图像会保存到正式输出位置旁的新目录。"
        seconds = DEFAULT_TRIAL_SECONDS
        if mode == "single":
            if not messagebox.askyesno(title, f"每台相机拍摄一张。{location}", parent=self.root):
                return
        else:
            instruction = ""
            if mode in ("manual", "event"):
                instruction = "开始后可点击“触发一次”。\n"
            elif mode == "hardware":
                instruction = "开始后等待相机收到物理触发信号。\n"
            elif mode in ("continuous", "timed"):
                instruction = "希望检查持续写入时，可保留默认试采时限。实测时间受原阶段的结束条件限制。\n"
            if mode == "event":
                nominal = (phase.get("pre_frames", 0) + phase["post_frames"]) / phase["fps"]
                seconds = max(seconds, nominal + 4)
                instruction += f"事件前后窗口约需 {nominal:.2f} 秒，请在试采时限内留出等待触发的时间。\n"
            if span := phase.get("duration_s"):
                seconds = min(span, seconds)
                instruction += f"原阶段时限为 {span:g} 秒。\n"
            dialog = CaptureFieldDialog(
                self.root, title, f"{location}\n{instruction.rstrip()}",
                (("试采时限（秒）", f"{seconds:g}", "填写最多采集多少秒。原阶段的张数或时长先达到时会提前结束。"),),
                lambda value: _positive_number(value, float), "请输入大于零的秒数。", confirm_text="检查并试采",
            )
            seconds = dialog.result
            if seconds is None:
                return
        self._inspect_plan(start=True, trial=(index + 1, seconds))

    def _inspect_plan(self, *, start, trial=None, checkerboard=None, rate_trial=None):
        if reason := self._plan_input_block():
            self.activity.set(reason)
            return
        if checkerboard is None and self._edit_hardware_mapping():
            return
        phases = [CHECKERBOARD_PHASE] if checkerboard else None
        try:
            config = self._read_config(phases=phases)
            output = self.output_path.get().strip()
            if not output:
                raise CaptureError("OUTPUT_REQUIRED", "请选择一个新输出目录。")
            if trial:
                phase_number, seconds = trial
                if rate_trial is not None and (config != rate_trial.original or output != rate_trial.output):
                    self.activity.set("计划或输出位置已变化。请重新检查建议频率。")
                    return
                measures = rate_trial.suggestion if rate_trial is not None else {}
                run_config, run_output = prepare_trial(
                    rate_trial.revised if rate_trial is not None else config, output,
                    phase_number=phase_number, seconds=seconds, rate_source_experiment_id=measures.get("source_experiment_id"),
                    rate_source_directory=rate_trial.source["output"] if rate_trial is not None else None,
                )
                if rate_trial is not None and trial_metadata(run_config.to_dict())["scope"]["volume"] != measures["source_volume"]:
                    self.activity.set("输出卷已变化。请在当前输出位置重新试采。")
                    return
            elif checkerboard:
                run_config, run_output = prepare_checkerboard_capture(config, output, checkerboard)
            else:
                run_config, run_output = config, output
        except Exception as exc:
            self._error(exc)
            return
        external = self.external.get()
        phase_offset = trial[0] - 1 if trial else 0
        discovered = self._discovered_ids
        trial_reports = () if trial or checkerboard else tuple(self._trial_results.values())
        preview = self.preview
        revision = preview.evidence_revision if preview else None
        if preview and revision is None:
            self.activity.set("相机正在恢复预览。请等所有相机送来新图像。")
            return
        samples, support, settings = self._preview_evidence(run_config, include_readback=bool(trial_reports))
        need_trigger = any(phase.camera_trigger != "free" for phase in run_config.phases)
        registry = self._registry() if preview is None else None
        self._checking_plan = True
        self._preflight_result = None
        self._plan_trial_progress = None
        self.plan_summary.set("正在核对采集计划及试采原图…" if trial_reports and run_config.quality_regions else
                              "正在检查相机参数、触发条件和输出位置…")

        def failed(exc):
            self._checking_plan = False
            if not self._closing:
                self.plan_summary.set("无法完成采前检查，请检查路径或计划。")
                self._error(exc)

        def preview_is_current():
            if self.preview is preview and (preview is None or preview.evidence_revision == revision):
                return True
            note = "检查期间相机连接已变化。请确认预览并重新检查计划。"
            self._invalidate_preflight(reason=note)
            self.plan_summary.set(note)
            return False

        def trials_are_current():
            if trial or checkerboard:
                return True
            current = self._trial_results.values()
            return len(current) == len(trial_reports) and all(old is new for old, new in zip(trial_reports, current))

        def finding_is_current(dialog):
            if not self._plan_report_is_current(dialog, config, output, phases=phases) or not preview_is_current():
                return False
            if not trials_are_current() or self.external.get() != external:
                self._mark_plan_report_stale(dialog, "试采记录或软件触发设置已变化，请重新检查计划。")
                return False
            return True

        def refresh_cameras(dialog):
            self._refresh_and_check_plan(
                dialog, check=lambda: self._inspect_plan(start=False, trial=trial, checkerboard=checkerboard,
                                                         rate_trial=rate_trial),
            )

        def launch(report, dialog=None):
            if dialog is not None and not self._plan_report_is_current(dialog, config, output, phases=phases):
                return
            if not trials_are_current() or self.external.get() != external:
                if dialog is not None:
                    dialog.destroy()
                self._inspect_plan(start=True, trial=trial, checkerboard=checkerboard, rate_trial=rate_trial)
                return
            if not preview_is_current():
                return
            if dialog is not None:
                dialog.destroy()
            reference = archive_trial_reference(report, trial_reports, image_warnings_confirmed=dialog is not None,
                                                readback_warnings_confirmed=dialog is not None)
            self._after_preview(lambda: self._start_capture(run_config, run_output, external=external and not checkerboard,
                                                          trial_reference=reference))

        def checked(result):
            nonlocal trial_reports
            report, checked_trials = result
            self._checking_plan = False
            if self._closing or not preview_is_current():
                return
            try:
                same_plan = self._read_config(phases=phases) == config and self.output_path.get().strip() == output
            except CaptureError:
                same_plan = False
            if not same_plan or self.external.get() != external:
                self.plan_summary.set("计划或输出位置已修改，请重新检查。")
                return
            if not trials_are_current():
                self._inspect_plan(start=start)
                return
            if checked_trials is not trial_reports:
                for previous, fresh in zip(trial_reports, checked_trials):
                    if previous is not fresh:
                        self._trial_results[previous["trial"]["phase_number"]] = fresh
                trial_reports = checked_trials
                self._schedule_draft_save()
            if preview is not None and not checkerboard:
                report = with_preview_quality(report, self._current_preview_quality(self._preview_quality_results))
            run_label = "标定采图" if checkerboard else "试采" if trial else "采前"
            prefix = f"{run_label}检查通过" if report["ready"] else "请先处理采前阻塞项"
            trial_needs_review = any(issue["code"] == "TRIAL_NEEDS_REVIEW" for issue in report["issues"])
            images_need_review = any(issue["code"] == "TRIAL_IMAGES_NEED_REVIEW" for issue in report["issues"])
            trial_setup_changed = any(issue["code"] == "TRIAL_CAMERA_READBACK_CHANGED" for issue in report["issues"])
            continuation_changed = any(issue["code"] == "CONTINUATION_SETUP_CHANGED" for issue in report["issues"])
            needs_review = any(issue["code"] in READINESS_CONFIRMATION_ISSUES for issue in report["issues"])
            if report["ready"] and needs_review:
                prefix = f"{run_label}检查完成 · 有事项需复核"
            ready_state = "checkerboard" if checkerboard else "trial" if trial else "review" if needs_review else "ready"
            self._preflight_result = ready_state if report["ready"] else "blocked"
            if not trial and not checkerboard:
                self._schedule_output_check()
                self._plan_trial_progress = summarize_trial_progress(config.phases, report["trial_assessments"],
                                                                     issues=report["issues"])
                self._refresh_trial_progress()
            summary = f"{prefix}\n计划像素量 {format_bytes(report['raw_payload_bytes'])}"
            if affected := sum(issue["code"] == "PREVIEW_IMAGE_QUALITY" for issue in report["issues"]):
                summary += f"\n{affected} 台相机有画质提醒"
            if continuation_changed:
                summary += "\n本轮与原实验存在设置差异"
            if trial_setup_changed:
                summary += "\n相机回读与试采有差异"
            if report["trial_assessments"]:
                matches = sum(item.get("latest_match", False) and item["status"] == "matched" and item["observed"]
                              for item in report["trial_assessments"])
                if images_need_review:
                    summary += "\n试采原图需要复核"
                elif trial_needs_review:
                    summary += "\n试采结果需要复查"
                elif not trial_setup_changed:
                    summary += f"\n已匹配 {matches} 个阶段的试采" if matches else "\n当前设置缺少匹配的已校验试采"
            self.plan_summary.set(summary)
            trial_action = None
            if report["ready"] and (trial_values := report.get("trial_setting_suggestions")):
                def apply_trial_values(dialog):
                    if not trials_are_current():
                        self.activity.set("试采报告已更新。请重新检查计划。")
                    elif preview_is_current():
                        self._apply_reference_settings(dialog, config, output, phases, trial_values,
                                                       notice="已将试采回读值补入计划。可在新设置下重新试采。")

                trial_action = ("补入试采参数", apply_trial_values,
                                "把所列曝光或增益补入当前计划。这些值用于整次采集。保存后可重新试采。")
            if not start or not report["ready"]:
                self.tabs.select(self.plan_panel)
                suggestions = report["phase_suggestions"] if not checkerboard and rate_trial is None else ()
                replacements, trigger_actions = {}, {}

                def apply_trigger_suggestions(dialog, index=None):
                    if finding_is_current(dialog):
                        changes = replacements if index is None else {index: replacements[index]}
                        self._apply_phase_suggestions(dialog, config, output, preview, changes)

                for row in suggestions:
                    checked_index = row["phase"] - 1
                    index = checked_index + phase_offset
                    replacements[index] = apply_phase_suggestion(
                        config.phases[index], run_config.phases[checked_index], row["replacement"],
                    )
                    if row["pairing"] == "host_receipt":
                        label = "按主机收图选帧并检查"
                        detail = "保留该阶段的请求方式和参数。先修改这一个阶段，再重新检查整份计划。"
                    else:
                        label = "每台取 1 张并检查"
                        detail = "该阶段由各台相机分别保存第一张连续送图。保留阶段用途，再重新检查整份计划。"
                    trigger_actions[row["phase"]] = (
                        label, lambda dialog, phase=index: apply_trigger_suggestions(dialog, phase), detail,
                    )
                issue_actions = self._plan_finding_actions(
                    report, phase_offset=phase_offset, checkerboard=checkerboard,
                    current=finding_is_current, trials=trial_reports, refresh_cameras=refresh_cameras,
                    formal_output=not (trial or checkerboard), trigger_actions=trigger_actions,
                )
                if rate_trial is not None:
                    self._show_plan_report(report, phase_offset=phase_offset, issue_actions=issue_actions)
                    return
                action = None
                if suggestions:
                    host = any(row["pairing"] == "host_receipt" for row in suggestions)
                    if len(suggestions) > 1:
                        label = "应用全部取图建议"
                        detail = f"应用报告列出的 {len(suggestions)} 项建议，再检查整份计划。也可在上方只修改选中的阶段。"
                    elif host:
                        label = "应用取图建议"
                        detail = "保留该阶段的请求方式和参数，启用主机选帧，再重新检查计划。"
                    else:
                        label = "改为连续采集 1 张"
                        detail = "该阶段改为每台相机各保存收到的第一张图像。应用后重新检查计划。"
                    action = (label, apply_trigger_suggestions, detail)
                if (action is None and report["ready"] and
                        (additions := report.get("continuation", {}).get("suggested_settings"))):
                    action = ("补入原实验参数",
                              lambda dialog: preview_is_current() and self._apply_reference_settings(
                                  dialog, config, output, phases, additions),
                              "将报告中列出的原实验回读值补入续采计划。计划会重新检查，连接预览时可验证应用结果。")
                action = action or trial_action
                if action is None and report["ready"] and report["rate_suggestions"]:
                    selected = self._selected_phase()
                    rates = report["rate_suggestions"]
                    suggestion = next((row for row in rates if selected == row["phase"] - 1), rates[0])
                    action = (f"试采阶段 {suggestion['phase']} 的建议频率",
                              lambda dialog: self._try_suggested_rate(dialog, config, output, suggestion),
                              "在独立目录试采报告中的建议频率。验证后可选择采用，再开始正式采集。")
                if action is None and report["write_queue"]["suggest_auto"]:
                    action = ("改用自动队列",
                              lambda dialog: self._apply_automatic_queue(dialog, config, output, phases),
                              "按当前字节缓冲上限设定待写队列容量。保存后重新检查计划。")
                probe_action = None
                if not trial and not checkerboard:
                    def probe_block():
                        if reason := self._plan_input_block():
                            return reason
                        try:
                            same = self._read_config() == config and self.output_path.get().strip() == output
                        except CaptureError:
                            same = False
                        if (not same or not trials_are_current() or self.preview is not preview
                                or (preview is not None and preview.evidence_revision != revision)):
                            return "计划或预览已变化。请重新检查计划后测试。"
                        return ""

                    probe_action = ("写盘实测…", lambda: self._show_write_probe(config, report, probe_block),
                                    "按所选阶段的图像规格测试目标盘。用合成 TIFF 和台账测量写入和刷盘。")
                self._show_plan_report(report, phase_offset=phase_offset, action=action, trials=trial_reports,
                                       probe_action=probe_action, issue_actions=issue_actions)
                return
            if needs_review:
                title = "请确认标定采图" if checkerboard else "请确认试采" if trial else "请确认采集计划"
                issue_actions = self._plan_finding_actions(report, phase_offset=phase_offset, checkerboard=checkerboard,
                                                           current=finding_is_current, trials=trial_reports,
                                                           refresh_cameras=refresh_cameras,
                                                           formal_output=not (trial or checkerboard))
                self._show_plan_report(report, phase_offset=phase_offset, trials=trial_reports, title=title, action=(
                    "仍按当前设置开始", lambda dialog: launch(report, dialog),
                    "请先查看需要复核的事项。点击后会按当前计划开始。"), secondary_action=trial_action,
                    issue_actions=issue_actions)
                return
            launch(report)

        def inspect():
            triggers, parameters, connections = support, settings, None
            if registry is not None:
                probe = probe_preflight_cameras(run_config, registry, triggers=need_trigger, include_readback=bool(trial_reports))
                triggers, parameters, connections = probe.triggers, probe.settings, probe.connections
            checked_trials = refresh_trial_image_regions(run_config, run_output, trial_reports, cancelled=lambda: self._closing)
            report = inspect_plan(run_config, run_output, preview_images=samples, trigger_support=triggers,
                                  camera_settings=parameters, camera_connections=connections,
                                  discovered_ids=discovered, trial_reports=checked_trials)
            return report, checked_trials

        self._job(inspect, checked, failed)

    def _preview_evidence(self, config, *, include_readback=False):
        samples, support, settings = {}, {}, {}
        if (preview := self.preview) is None:
            return samples, support, settings
        opened = {camera.id: camera for camera in preview.cameras}
        include_readback = include_readback or bool(config.quality_regions) or continuation_reference(config) is not None
        now = time.perf_counter_ns()
        for camera in config.cameras:
            if camera != opened.get(camera.id):
                continue
            if evidence := preview.trigger_support(camera.id):
                support[camera.id] = evidence
                frame = preview.latest_image(camera.id)
                if frame is not None and now - frame.received_monotonic_ns < 4_000_000_000:
                    samples[camera.id] = frame
                    if result := preview.settings_check(camera, include_readback=include_readback):
                        settings[camera.id] = result
        return samples, support, settings

    def _plan_report_is_current(self, dialog, config, output, *, phases=None):
        if reason := getattr(dialog, "readiness_stale", "") or self._plan_input_block():
            self.activity.set(reason)
            return False
        try:
            same_plan = self._read_config(phases=phases) == config and self.output_path.get().strip() == output
        except CaptureError:
            same_plan = False
        if not same_plan:
            reason = "计划或保存位置已变化。请重新检查计划。"
            self._mark_plan_report_stale(dialog, reason)
            self.activity.set(reason)
        return same_plan

    def _edit_plan_report_phase(self, dialog, index, *, wiring=False, csv_source=False):
        dialog.destroy()
        self.tabs.select(self.plan_panel)
        self._populate_phases(index)
        self.edit_phase(wiring=wiring, csv_source=csv_source, recheck=True)

    def _edit_plan_report_camera(self, dialog, camera, feature):
        dialog.destroy()
        self._after_preview(lambda: self.edit_camera(camera, focus_feature=feature, recheck=True))

    def _plan_finding_actions(self, report, *, phase_offset, checkerboard, current, trials, refresh_cameras,
                              formal_output, trigger_actions=None):
        trials_by_id = {trial["experiment_id"]: trial for trial in trials}
        matched = {row["phase"]: trials_by_id.get(row["experiment_id"])
                   for row in report["trial_assessments"] if row.get("latest_match")}
        output_codes = {"OUTPUT_EXISTS", "OUTPUT_INSIDE_REPEAT_SOURCE", "OUTPUT_PARENT_NOT_DIRECTORY", "OUTPUT_INSPECTION_FAILED",
                        "DISK_RESERVE_UNAVAILABLE", "PLAN_EXCEEDS_AVAILABLE_SPACE"}
        input_codes = {"HARDWARE_INPUT_INVALID", "HARDWARE_INPUT_UNAVAILABLE", "HARDWARE_PAIRING_UNVERIFIED"}
        phase_codes = input_codes | {"SOFTWARE_TRIGGER_UNSUPPORTED", "HARDWARE_TRIGGER_UNSUPPORTED",
                                     "EVENT_WINDOW_EXCEEDS_DURATION", "WAITING_FOR_EXTERNAL_INPUT",
                                     "FRAME_LIMIT_UNAVAILABLE", "RAW_BUFFER_TOO_SMALL", "EXPOSURE_EXCEEDS_INTERVAL"}

        def change_output(dialog):
            if self.choose_output(parent=dialog):
                dialog.destroy()
                self.check_plan()

        def use_output(dialog, suggestion):
            dialog.destroy()
            self._fill_output_suggestion(suggestion)
            self.check_plan()

        def review_trial(trial):
            self._read_trial_report(Path(trial["output"]), expected_id=trial["experiment_id"])

        def preview_cameras(dialog):
            dialog.destroy()
            self.toggle_preview()

        def action_for(issue):
            code = issue["code"]
            if code == "OUTPUT_EXISTS" and formal_output and (suggestion := report.get("output_suggestion")):
                return ("使用新路径并检查", lambda dialog: use_output(dialog, suggestion),
                        f"将建议路径填入计划，重新检查保存位置和相机。\n\n{suggestion}")
            if code in output_codes:
                return ("选择输出位置…", change_output, "选择新的父文件夹后，会重新检查保存位置和可用空间。")
            if code == "CAMERA_SETTING_INVALID":
                return ("修改相机参数…",
                        lambda dialog: self._edit_plan_report_camera(dialog, issue["camera"], issue["feature"]),
                        "打开受影响的相机参数。应用后会重新检查计划。")
            if code == "PREVIEW_IMAGE_QUALITY":
                return ("查看该相机画面", lambda dialog: self._open_preview_quality_camera(issue["camera"], dialog),
                        "定位到这台相机的画质页，可检查照明和反光，或调整检查框后复测。")
            if code == "QUALITY_REGION_CHANGED":
                return ("核对抽查范围", lambda dialog: self._open_quality_region_camera(issue["camera"], dialog),
                        "定位到这台相机的画质页。可按当前原图重新框选，也可恢复全图抽查。")
            if not checkerboard and code.startswith("CSV_TRIGGER_"):
                phase = issue["phase"] + phase_offset
                return (f"修改阶段 {phase} 的试验机 CSV…",
                        lambda dialog: self._edit_plan_report_phase(dialog, phase - 1, csv_source=True),
                        "选择正确的 CSV、触发量列和单位。保存阶段后会重新检查。")
            if not checkerboard and code == "SOFTWARE_TRIGGER_UNSUPPORTED":
                if trigger_actions and (action := trigger_actions.get(issue["phase"])):
                    return action
                if self.preview is None and (issue.get("offer_preview") or issue.get("offer_host_preview")):
                    return ("连接并预览", preview_cameras,
                            "确认所选相机都能送图后，可按阶段查看可用的取图方式。")
            if not checkerboard and code in phase_codes:
                phase = issue["phase"] + phase_offset
                wiring = code in input_codes
                return (f"设置阶段 {phase} 的物理输入…" if wiring else f"编辑阶段 {phase}…",
                        lambda dialog: self._edit_plan_report_phase(dialog, phase - 1, wiring=wiring),
                        "打开该阶段的物理输入和脉冲配对设置。保存后会重新检查。"
                        if wiring else "打开该阶段的拍摄方式、频率和结束条件。保存后会重新检查。")
            if code in {"CAMERAS_NOT_DISCOVERED", "CAMERA_CONNECTION_CHECK_FAILED"}:
                return ("刷新并检查", refresh_cameras, "接回相机或关闭占用设备的程序后，重新查找并检查当前计划。")
            if code in {"TRIGGER_PROBE_FAILED", "CAMERA_SETTINGS_PROBE_FAILED"}:
                return ("查看驱动状态…", lambda _dialog: self.show_providers(),
                        "核对设备是否已发现，以及相机接口和驱动的状态。")
            if code == "FRAME_GEOMETRY_UNAVAILABLE" and self.preview is None:
                return ("连接并预览", lambda _dialog: self.toggle_preview(),
                        "预览送图后可用实际尺寸和像素格式重新估算空间。")
            if code in {"TRIAL_IMAGES_NEED_REVIEW", "TRIAL_NEEDS_REVIEW"} and (trial := matched.get(issue["phase"])):
                if code == "TRIAL_IMAGES_NEED_REVIEW":
                    return ("打开该阶段试采原图…", lambda _dialog: self._show_trial_images(trial),
                            "打开这个阶段的试采图像和对应提醒。")
                return ("打开该阶段试采报告…", lambda _dialog: review_trial(trial),
                        "查看这个阶段实测的写入、收图和文件校验结果。")
            return None

        actions = {}
        for index, issue in enumerate(report["issues"]):
            if spec := action_for(issue):
                label, callback, hint = spec
                actions[str(index)] = (label, lambda dialog, run=callback: current(dialog) and run(dialog), hint)
        return actions

    def _apply_automatic_queue(self, dialog, config, output, phases):
        if self._plan_report_is_current(dialog, config, output, phases=phases):
            self.config["queue_capacity"] = "auto"
            dialog.destroy()
            self._invalidate_preflight()
            self.check_plan()

    def _apply_reference_settings(self, dialog, config, output, phases, additions, *, notice=None):
        if not self._plan_report_is_current(dialog, config, output, phases=phases):
            return

        def apply():
            self.config["cameras"] = [
                {**item, "settings": {**additions[item["id"]], **item["settings"]}}
                if item["id"] in additions else item
                for item in self.config["cameras"]
            ]
            self._invalidate_preflight()
            if notice:
                self.activity.set(notice)
            self.check_plan()

        dialog.destroy()
        self._after_preview(apply)

    def _try_suggested_rate(self, dialog, config, output, suggestion):
        if not self._plan_report_is_current(dialog, config, output):
            return
        number = suggestion["phase"]
        source = self._trial_results.get(number)
        if source is None or source["experiment_id"] != suggestion["source_experiment_id"]:
            dialog.title("采前检查 · 试采结果已更新")
            self.activity.set("已有新的试采结果。请重新检查计划。")
            return
        rate_trial = _RateTrial(config, output, with_trial_rate(config, suggestion), suggestion, source)
        dialog.destroy()
        self._after_preview(lambda: self._inspect_plan(start=True, trial=(number, suggestion["trial_seconds"]), rate_trial=rate_trial))

    def _apply_verified_rate(self, dialog, scope, report, reviewed):
        config, output = scope
        if not self._plan_report_is_current(dialog, config, output):
            return
        self._checking_rate_trial = True
        for control in dialog.report_actions:
            control.state(["disabled"])
        dialog.title("试采报告 · 正在重新核验")
        self.activity.set("正在重新核验原试采和频率复测。")
        known = self._rate_source_directories()
        full = {camera for camera, row in report["cameras"].items() if row.get("image_check", {}).get("mode") == "all"}

        def current():
            return (not self._closing and dialog.winfo_exists()
                    and self._plan_report_is_current(dialog, config, output))

        def read():
            fresh = trial_report(report["output"], expected_experiment_id=report["experiment_id"],
                                 full_image_cameras=full, cancelled=lambda: self._closing)
            if fresh["trial"].get("rate_source_experiment_id") != report["trial"].get("rate_source_experiment_id"):
                return fresh, ("setup", None)
            return fresh, self._read_rate_trial_check(scope, fresh, known)

        def finished(values):
            self._checking_rate_trial = False
            if not current():
                return
            fresh, (status, revised) = values
            if status != "verified" or revised != reviewed:
                status = "setup" if status == "verified" else status
                self.activity.set(RATE_TRIAL_LABELS[status])
                self._display_trial_report(fresh, status)
                return

            def apply():
                if not current():
                    return
                try:
                    volume = volume_identity(output_ancestor(resolve_output_path(output)))
                except (OSError, RuntimeError, ValueError):
                    volume = None
                if volume != fresh["volume"]:
                    self.activity.set(RATE_TRIAL_LABELS["volume"])
                    self._display_trial_report(fresh, "volume")
                    return
                data = revised.to_dict()
                if revised.cameras != config.cameras:
                    entries = {item["id"]: item for item in self.config["cameras"]}
                    entries.update({item["id"]: item for item in data["cameras"]})
                    self.config["cameras"] = list(entries.values())
                self.config["phases"] = data["phases"]
                self._populate_phases(fresh["trial"]["phase_number"] - 1)
                self._remember_trial(fresh, plan=revised)
                dialog.destroy()
                self.activity.set("已采用本次复测验证的频率。")
                self.check_plan()

            if revised.cameras != config.cameras:
                self._after_preview(apply)
            else:
                apply()

        def failed(error):
            self._checking_rate_trial = False
            if current():
                self._mark_plan_report_stale(dialog, "复测原图或台账未能重新核验。请重开试采报告。")
                self._error(error)

        self._job(read, finished, failed)

    def _apply_phase_suggestions(self, dialog, config, output, preview, replacements):
        if not self._plan_report_is_current(dialog, config, output):
            return
        samples, _, _ = self._preview_evidence(config)
        if self.preview is not preview or len(samples) != len(config.cameras):
            dialog.title("采前检查 · 设置已修改")
            self.activity.set("预览状态已变化。请重新检查后应用修改。")
            return
        index = self._selected_phase()
        self.config["phases"] = [replacements.get(i, phase.to_dict()) for i, phase in enumerate(config.phases)]
        dialog.destroy()
        self._invalidate_preflight()
        self._populate_phases(0 if index is None else index)
        self.check_plan()

    def _show_write_probe(self, config, report, authorize):
        if reason := authorize():
            self.activity.set(reason)
            return
        from .write_probe_gui import WriteProbeWindow

        window = self._write_probe_window
        if window is not None and window.top.winfo_exists():
            window.top.destroy()
        self._write_probe_window = WriteProbeWindow(self.root, config, report, authorize, self._start_progress_trial)

    def _show_plan_report(self, report, *, phase_offset=0, action=None, secondary_action=None, trials=(),
                          probe_action=None, issue_actions=None, title="采前检查"):
        detail = None
        if trials:
            assessments = {row["experiment_id"]: row for row in report["trial_assessments"]}
            available = [(trial, row) for trial in trials if (row := assessments.get(trial["experiment_id"]))
                         and any(check["selected"] for check in row.get("image_checks", {}).values())]
            if available:
                detail = ("检查试采原图", lambda: self._choose_trial_images(available),
                          "打开试采原图及对应提醒。计划有修改时也可查看之前的图像。")
        labels = self._report_camera_labels(report)
        self._plan_report_window = self._open_text_report(
            title, format_readiness(report, phase_offset=phase_offset, camera_labels=labels),
            getattr(self, "_plan_report_window", None), action=action, detail_action=detail, secondary_action=secondary_action,
            extra_detail_action=probe_action, readiness=(report["issues"], labels, phase_offset, issue_actions or {}),
        )
        checks = report.get("preview_quality", {})
        self._plan_report_window.preview_quality_rows = tuple(
            row for row in self._preview_quality_results if row.state == "ready" and row.target.camera_id in checks)

    def _choose_trial_images(self, available):
        if len(available) == 1:
            self._show_trial_images(available[0][0])
            return
        choices, reports = {}, {}
        for trial, row in sorted(available, key=lambda item: (item[1]["status"] != "matched", item[1]["phase"])):
            flagged = trial_image_alerts(row)
            scope = "计划设置匹配" if row["status"] == "matched" else "历史试采"
            label = f"阶段 {row['phase']} · {scope} · 提醒 {flagged} 张"
            choices[label] = f"打开该阶段的试采原图。\n{trial['output']}"
            reports[label] = trial
        selected = CaptureChoiceDialog(self._plan_report_window, "选择试采阶段", "选择要查看原图的试采阶段。", choices)
        if selected.result:
            self._show_trial_images(reports[selected.result])

    def _report_camera_labels(self, report):
        labels = {key: self._view_title(key) for key in report["cameras"]}
        counts = Counter(labels.values())
        return {key: label for key, label in labels.items() if counts[label] == 1}

    def _open_text_report(self, title, contents, previous, *, offer_plan_check=False, action=None, detail_action=None,
                          secondary_action=None, extra_detail_action=None, readiness=None):
        if previous is not None and previous.winfo_exists():
            previous.destroy()
        dialog = tk.Toplevel(self.root, bg=BG)
        dialog.title(title)
        dialog.geometry("720x545")
        dialog.minsize(500, 340)
        dialog.transient(self.root)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        panel = ttk.Frame(dialog, padding=12)
        panel.grid(row=0, column=0, sticky="nsew")
        dialog.readiness_overview = None
        dialog.readiness_stale = ""
        dialog.report_actions = []
        text_parent = panel
        if readiness and readiness[0]:
            repaint_after_resize(dialog)
            dialog.minsize(595, 440)
            notebook = ttk.Notebook(panel)
            notebook.pack(fill=tk.BOTH, expand=True)
            overview = dialog.readiness_overview = ReadinessOverview(notebook, *readiness)
            text_parent = ttk.Frame(notebook, padding=(4, 4))
            notebook.add(overview, text="检查结果")
            notebook.add(text_parent, text="完整报告")
            notebook.enable_traversal()
            ToolTip(notebook, lambda event: notebook_help(notebook, (
                "选择检查事项，查看原因和可执行的操作。", "查看本次检查的完整依据、估算和试采对照。",
            ), event), title="采前检查")
        text = tk.Text(text_parent, bg=ENTRY, fg=FG, borderwidth=0, wrap="word", padx=10, pady=8,
                       font=(choose_ui_font_family(dialog), 10))
        scroll = ttk.Scrollbar(text_parent, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", contents)
        text.configure(state="disabled")
        ToolTip(text, "报告可滚动。拖动选中文字，再按 Ctrl+C 可复制选中的部分。", title="报告")
        ToolTip(scroll, "滚动查看报告其余部分。", title="报告")
        actions = ttk.Frame(dialog, padding=(12, 0, 12, 10))
        actions.grid(row=1, column=0, sticky="ew")

        def copy_report():
            self.root.clipboard_clear()
            self.root.clipboard_append(contents)
            copy.configure(text="已复制报告")

        copy = ttk.Button(actions, text="复制报告", command=copy_report, style="Small.TButton")
        if readiness:
            copy.configure(width=9)
        copy.pack(side=tk.LEFT, padx=(0, 8))
        ToolTip(copy, "把完整报告复制到剪贴板，可粘贴到实验记录。")
        close = ttk.Button(actions, text="关闭", command=dialog.destroy)
        close.pack(side=tk.RIGHT)
        ToolTip(close, "关闭报告，回到工作台。也可按 Esc。")
        followups = actions
        for spec, plan_only in ((detail_action, False), (extra_detail_action, bool(readiness))):
            if spec is None:
                continue
            label, command, hint = spec
            button = ttk.Button(actions, text=label, command=command, style="Small.TButton")
            button.pack(side=tk.LEFT, padx=(0, 4))
            if plan_only:
                dialog.report_actions.append(button)
                ToolTip(button, lambda _event, note=hint: dialog.readiness_stale or note)
            else:
                ToolTip(button, hint)
        if secondary_action or ((detail_action or extra_detail_action) and (action or offer_plan_check)):
            followups = ttk.Frame(dialog, padding=(12, 0, 12, 10))
            followups.grid(row=2, column=0, sticky="ew")
        for spec, style in ((action, "Record.TButton"), (secondary_action, "Small.TButton")):
            if spec:
                label, command, hint = spec
                button = ttk.Button(followups, text=label, command=lambda cmd=command: cmd(dialog), style=style)
                button.pack(side=tk.RIGHT, padx=(0, 7))
                dialog.report_actions.append(button)
                ToolTip(button, lambda _event, note=hint: dialog.readiness_stale or note)
        if offer_plan_check:
            def check():
                dialog.destroy()
                progress = self._trial_progress_window
                if progress is not None and progress.top.winfo_exists():
                    progress.top.withdraw()
                self.tabs.select(self.plan_panel)
                self.check_plan()

            button = ttk.Button(followups, text="核对当前计划", command=check,
                                style="TButton" if action else "Record.TButton")
            button.pack(side=tk.RIGHT, padx=(0, 7))
            ToolTip(button, "对照试采与当前阶段、相机和存储设置。此操作会打开采前检查报告。")
        dialog.bind("<Escape>", lambda _event: ToolTip.dismiss_window(dialog) or dialog.destroy())
        focus = dialog.readiness_overview.table if dialog.readiness_overview is not None else text
        focus.focus_set()
        return dialog

    def _start_capture(self, config, output, *, external=False, trial_reference=None):
        self._show_views([camera.id for camera in config.cameras])
        if self.tabs.select() == str(self.review_panel):
            self.tabs.select(self.plan_panel)
        stop_event = self._start_stop = threading.Event()
        self.marker_note.set("")
        self.marker_status.set("")
        previous_control, self.control = self.control, None
        registry = self._registry() if self._practice else None
        watch_quality = self.watch_quality.get()
        self.activity.set("正在创建实验目录和台账…")

        def open_session():
            if previous_control is not None:
                previous_control.__exit__(None, None, None)
            if stop_event.is_set():
                return None
            session = CaptureSession(config, output, registry=registry, trial_reference=trial_reference,
                                     watch_quality=watch_quality)
            identity = session.start(stop_event=stop_event)
            control = None
            control_error = None
            if external and not stop_event.is_set():
                try:
                    control = CaptureControl(session).__enter__()
                except Exception as exc:
                    session.request_stop()
                    control_error = exc
            remember_experiment(RecentExperiment(identity, config.name, session.output))
            return session, control, control_error, identity

        def opened(result):
            self._start_stop = None
            if result is None:
                self.activity.set("已取消启动。")
                return
            session, self.control, control_error, self._experiment_id = result
            self.session = session
            self.experiment = session.output
            self._schedule_output_check()
            self._review_revision += 1
            self._review_is_copy = False
            self._experiment_cameras = tuple(camera.id for camera in config.cameras)
            self._set_review_action(config.to_dict(), trial_reference=trial_reference)
            window = self._trial_progress_window
            if self._review_trial and window is not None and window.top.winfo_exists():
                window.capture_started()
            self._run_reviewed = False
            self.review_image = None
            self._review_ordinal = None
            self._set_review_parts(None)
            if self._board_scan:
                self._update_quality_panel()
                self.tabs.select(self.quality_panel)
            elif self.tabs.select() == str(self.review_panel):
                self.browse_frame(latest=True)
            else:
                self.tabs.select(self.plan_panel)
            self.activity.set("正在连接采集设备…")
            if control_error is not None and not self._closing:
                self._error(control_error)

        def failed(exc):
            self._start_stop = None
            if not self._closing:
                if getattr(exc, "code", None) == "OUTPUT_EXISTS":
                    self._schedule_output_check()
                self._error(exc)

        self._job(open_session, opened, failed)

    def _marker_block(self):
        if self._start_stop is not None:
            return "请等待采集启动。"
        if self.session is None or not self.session.marker_available:
            return "请在采集中记录标记。"
        return ""

    def _mark_shortcut(self, _event):
        self.mark_event()
        return "break"

    def mark_event(self):
        if reason := self._marker_block():
            self.activity.set(reason)
            return
        session = self.session
        label = self.marker_note.get()
        try:
            _identity, receipt = session.request_marker(label)
        except CaptureError as exc:
            self.marker_status.set(ui_error(exc, details=False))
            return
        self.marker_note.set("")
        self.marker_status.set("正在登记标记…")

        def finish(result):
            if self.session is not session or self._closing:
                return
            try:
                result.result()
            except CaptureError as exc:
                self.marker_status.set(ui_error(exc, details=False))
                if not self.marker_note.get():
                    self.marker_note.set(label)
            else:
                self.marker_status.set("标记已保存，可在采集时间线复查。")

        receipt.add_done_callback(lambda future: self._jobs.put((finish, future)))

    def trigger(self):
        phase = self._trigger_phase
        if self.session is None or phase is None or self._start_stop is not None:
            return
        try:
            self.session.request_trigger(expected_phase=phase, expected_window=self._trigger_window)
            self.activity.set("操作员触发请求已入队。")
            status = self.session.status()
            progress = status["phase_progress"] or {}
            window = progress.get("event_window") or {}
            if status["phase"] == phase and window.get("overlap") and progress["trigger_available"]:
                self._trigger_window = window["event_number"]
            elif self._board_scan or self._trigger_window is not None or status["phase"] != phase:
                self._trigger_phase = None
                self.trigger_button.state(["disabled"])
        except CaptureError as exc:
            self._error(TRIGGER_ERROR_LABELS.get(exc.code, str(exc)))

    def _pause_help(self, _event=None):
        if self._phase_pause:
            return PHASE_PAUSE_DISPLAY[self._phase_pause["state"]][1]
        if not self._pause_available:
            return "当前有触发请求待执行，或本阶段即将结束。"
        return ("本轮拍摄结束后暂停，已入队的图像继续保存。\n"
                "暂停期间停止本阶段的拍摄计时。\n"
                "点“继续采集”可从本阶段接着拍摄。")

    def toggle_phase_pause(self):
        if self.session is None or self._pausable_phase is None:
            return
        phase, _ = self._pausable_phase
        try:
            if (pause := self._phase_pause) and pause["state"] == "paused":
                self.session.request_phase_resume(phase, pause["pause_id"])
            elif self._pause_available:
                self.session.request_phase_pause(phase)
            else:
                return
            self.pause_button.state(["disabled"])
        except CaptureError as exc:
            self.activity.set(ui_error(exc, details=False))

    def stop(self):
        if self._start_stop is not None:
            self._start_stop.set()
            self.activity.set("正在结束启动，已写入的记录会保留。")
        elif self.session:
            self.session.request_stop()

    def _set_review_action(self, config, *, trial_reference=None):
        self._review_outcome = self._review_integrity = self._review_read_ticket = None
        self._review_read_error = ""
        self._review_frame_error = ""
        self._review_name = config["name"]
        self._render_review_result()
        self._review_phases = tuple(phase["mode"] for phase in config["phases"])
        self._review_phase_roles = tuple(phase["role"] for phase in config["phases"])
        self._review_trial = trial_metadata(config)
        parent = config.get("metadata", {}).get("continuation_of")
        self._review_parent = parent if isinstance(parent, str) and parent else None
        self._has_trial_reference = isinstance(trial_reference, dict) and trial_reference.get("schema") == TRIAL_REFERENCE_SCHEMA
        if self._has_trial_reference:
            self.trial_comparison_button.pack(fill=tk.X, pady=(5, 0), before=self.copy_button)
        else:
            self.trial_comparison_button.pack_forget()
        self._set_checkerboard_scan(checkerboard_target(config))
        for button, needs_camera in self.analysis_buttons:
            if needs_camera:
                if self._board_scan:
                    button.pack_forget()
                elif not button.winfo_manager():
                    button.pack(fill=tk.X, pady=(5, 0), before=self.review_recovery)
        if self._review_trial:
            name = "频率复测报告" if self._review_trial.get("rate_source_experiment_id") else "试采报告"
            self.continue_button.configure(text=name, command=self.show_trial_report)
        elif self._board_scan:
            self.continue_button.configure(text="拟合镜头参数…", command=self.fit_calibration)
        else:
            self.continue_button.configure(text="准备续采", command=self.prepare_continuation)

    def _remember_trial(self, report, *, plan=None):
        phase = report["trial"]["phase_number"]
        previous = self._trial_results.get(phase)
        if previous and previous["created_utc_ns"] > report["created_utc_ns"]:
            if plan is None or phase in compare_trials(plan, report["volume"], [previous])[1]:
                return
        if (previous and previous["experiment_id"] == report["experiment_id"] and previous["output"] == report["output"]
                and report["verification"]["ok"]):
            for camera, row in report["cameras"].items():
                prior = previous["cameras"].get(camera, {}).get("image_check")
                fresh = row.get("image_check")
                prior_region = (prior or {}).get("region_check", {})
                fresh_region = (fresh or {}).get("region_check", {})
                if (prior and fresh and prior["mode"] == "all" and fresh["mode"] == "sampled"
                        and prior["checked"] == prior["available"]
                        and prior_region.get("region") == fresh_region.get("region")
                        and prior_region.get("fallback") == fresh_region.get("fallback")
                        and prior["series_sha256"] == fresh["series_sha256"]):
                    row["image_check"] = prior
        self._trial_results[phase] = report
        self._invalidate_preflight()
        self.plan_summary.set("已读取试采结果。检查计划可核对当前设置。")

    def _rate_review_scope(self):
        if output := self.output_path.get().strip():
            try:
                return self._read_config(), output
            except CaptureError:
                pass
        return None

    def _rate_source_directories(self):
        return {row["experiment_id"]: row["output"] for row in self._trial_results.values()}

    @staticmethod
    def _read_rate_trial_check(scope, report, known):
        if scope is None:
            return "original_plan", None
        config, output = scope
        try:
            volume = volume_identity(output_ancestor(resolve_output_path(output)))
        except (OSError, RuntimeError, ValueError):
            volume = None
        assessed = assess_rate_trial(config, volume, report, None)
        if assessed[0] != "source_unavailable":
            return assessed
        identity = report["trial"]["rate_source_experiment_id"]
        directories = [known[identity]] if identity in known else []
        try:
            directories.extend(row.directory for row in recent_experiments() if row.experiment_id == identity)
        except (OSError, sqlite3.Error):
            pass
        source = read_rate_trial_source(report, directories)
        return assess_rate_trial(config, volume, report, source)

    def _trial_matches_current_plan(self, report):
        try:
            config = self._read_config()
            output = self.output_path.get().strip()
            if output:
                volume = volume_identity(output_ancestor(resolve_output_path(output)))
                _, matches = compare_trials(config, volume, [report])
                return bool(matches)
        except (CaptureError, OSError):
            pass
        return False

    def show_trial_report(self):
        if not self._review_trial or not self.experiment or self._busy() or self._closing:
            return
        self._read_trial_report(self.experiment, expected_id=self._experiment_id, current_only=True)

    def _read_trial_report(self, path, *, expected_id, current_only=False):
        if path in self._trial_reads or self._busy() or self._closing or self._trial_batch is not None:
            return
        self._trial_reads.add(path)
        self.activity.set("正在核对试采台账与图像…")
        scope, sources = self._rate_review_scope(), self._rate_source_directories()

        def allowed():
            return not self._busy() and not self._closing and (not current_only or self.experiment == path)

        def read():
            report = trial_report(path, expected_experiment_id=expected_id, cancelled=lambda: self._busy() or self._closing)
            rate = (None, None)
            if report["trial"].get("rate_source_experiment_id"):
                rate = self._read_rate_trial_check(scope, report, sources)
            return report, rate

        def show(result):
            self._trial_reads.remove(path)
            if allowed():
                report, (status, revised) = result
                if not report["trial"].get("rate_source_experiment_id") or self._trial_matches_current_plan(report):
                    self._remember_trial(report)
                self.activity.set("已读取试采报告。")
                action = None
                if revised is not None:
                    if scope == self._rate_review_scope():
                        action = ("重新核验并采用频率", lambda dialog: self._apply_verified_rate(dialog, scope, report, revised),
                                  "重新校验两次试采的原图和台账。\n计划和输出卷匹配后，更新频率并打开采前检查。")
                    else:
                        status = "original_plan"
                self._display_trial_report(report, status, action)

        def failed(exc):
            self._trial_reads.remove(path)
            if getattr(exc, "code", None) == "TRIAL_IMAGE_CHECK_CANCELLED":
                return
            for phase, report in tuple(self._trial_results.items()):
                if report["output"] == str(path):
                    del self._trial_results[phase]
                    self._invalidate_preflight()
            if allowed():
                self._error(exc)

        self._job(read, show, failed)

    def _display_trial_report(self, report, status=None, action=None):
        detail = None
        if any(row.get("image_check", {}).get("selected") for row in report["cameras"].values()):
            detail = ("检查试采原图", lambda: self._show_trial_images(report),
                      "按相机查看需要复核的帧。可打开原图、框选试样，并把检查框保存到采集计划。")
        progress_action = None
        if len(self.config["phases"]) > 1:
            def show_progress(dialog):
                dialog.destroy()
                self.show_trial_progress()

            progress_action = ("阶段试采…", show_progress, "回到阶段列表，查看接下来要复核或试采的阶段。")
        self._trial_report_window = self._open_text_report(
            "试采报告", format_trial(report, camera_labels=self._report_camera_labels(report), rate_check=status),
            getattr(self, "_trial_report_window", None), offer_plan_check=True, action=action, detail_action=detail,
            secondary_action=progress_action,
        )

    def _show_trial_images(self, report, *, use_current_plan=True):
        if self._busy() or self._closing:
            self.activity.set("请等当前采集结束，再查看试采原图。")
            return
        if use_current_plan:
            latest = self._trial_results.get(report["trial"]["phase_number"])
            if latest and latest["experiment_id"] == report["experiment_id"] and latest["output"] == report["output"]:
                report = latest
        window = getattr(self, "_trial_image_window", None)
        if window is not None and window.top.winfo_exists():
            if window.report is report and (window.on_use_region is not None) == use_current_plan:
                window.top.lift()
                return
            window.top.destroy()
        self._trial_image_window = TrialImageWindow(
            self.root, report, self._report_camera_labels(report), self.active_camera, self._job, self._busy,
            self._store_full_trial_check if use_current_plan else None,
            on_use_region=self._use_trial_region if use_current_plan else None,
            plan_region_block=self._trial_region_block if use_current_plan else None,
        )

    def _trial_region_block(self, camera):
        if self._closing:
            return "采集窗口正在关闭。"
        return self._setup_block() or (
            "" if camera in self.devices.selection() else "请先在当前计划的相机列表中选中这台相机。"
        )

    def _use_trial_region(self, report, camera, frame, bounds, active, done, failed):
        if reason := self._trial_region_block(camera):
            failed(reason)
            return
        original_config = self.config
        try:
            plan = self._read_config()
        except CaptureError as exc:
            failed(exc)
            return
        settings = next(row.settings for row in plan.cameras if row.id == camera)

        def finished(region):
            if not active():
                return
            if reason := self._trial_region_block(camera):
                failed(reason)
                return
            try:
                same_plan = self.config is original_config and self._read_config() == plan
            except CaptureError:
                same_plan = False
            if not same_plan:
                failed(CaptureError("TRIAL_REGION_PLAN_CHANGED", camera))
                return
            if sample := self.preview.bound_image(camera) if self.preview is not None else None:
                binding, image = sample
                if reason := quality_region_mismatch(region, image, binding.readback):
                    failed(CaptureError(f"TRIAL_REGION_{reason.upper()}", camera))
                    return
                self._preview_regions_pending[camera] = region
            self._set_plan_quality_region(camera, region, "已保存试采检查框。可核对当前计划，复核新范围。")
            done()

        self._job(lambda: read_trial_quality_region(
            Path(report["output"]), report["experiment_id"], camera, frame, bounds, settings,
        ), finished, failed)

    def _store_full_trial_check(self, report, camera, result):
        phase = report["trial"]["phase_number"]
        current = self._trial_results.get(phase)
        if current is None or current["experiment_id"] != report["experiment_id"] or current["output"] != report["output"]:
            return
        previous = current["cameras"][camera]["image_check"]
        if result["series_sha256"] != previous["series_sha256"]:
            del self._trial_results[phase]
            self._invalidate_preflight()
            self.activity.set("试采台账已变化，请重新打开对应的试采报告。")
            return
        cameras = {**current["cameras"], camera: {**current["cameras"][camera], "image_check": result}}
        self._remember_trial({**current, "cameras": cameras})

    def show_recent_experiments(self):
        if reason := self._setup_block(needs_idle_camera=True):
            self.activity.set(reason)
            return
        RecentExperimentsDialog(self.root, self._job, self.open_experiment)

    def show_related_experiments(self):
        if reason := self._review_help("", needs_idle=True):
            self.activity.set(reason)
            return
        RelatedExperimentsWindow(self.root, self.experiment, self._experiment_id, self._job, self.open_experiment)

    def _copy_block(self):
        if reason := self._review_help("", needs_idle=True):
            return reason
        if (self._review_state or "").lower() not in TERMINAL_STATES:
            return "请等采集结束并写完台账。若原采集异常退出，可先用“检查中断”核对实验。"
        return ""

    def show_experiment_copy(self):
        window = getattr(self, "_experiment_copy_window", None)
        if window is not None and window.top.winfo_exists():
            window.top.lift()
            return
        if reason := self._copy_block():
            self.activity.set(reason)
            return
        self._experiment_copy_window = ExperimentCopyWindow(
            self.root, self.experiment, self._experiment_id, self._job, self.open_experiment,
        )

    def _sequence_export_block(self):
        if reason := self._copy_block():
            return reason
        if not self._experiment_cameras:
            return "这个实验的台账还没有相机记录。"
        if "test" not in self._review_phase_roles:
            return "这个实验没有记录变形的实验阶段。"
        return ""

    def show_sequence_export(self):
        if reason := self._sequence_export_block():
            self.activity.set(reason)
            return
        window = getattr(self, "_sequence_export_window", None)
        if window is not None and window.top.winfo_exists():
            if (window.source == Path(self.experiment) and window.experiment_id == self._experiment_id
                    and (window.event is None or window._working)):
                window.top.lift()
                return
            window.top.destroy()
        cameras = {camera: self._view_title(camera) for camera in self._experiment_cameras}
        phases = {str(number): f"{number} · {MODE_LABELS[mode]}"
                  for number, (mode, role) in enumerate(zip(self._review_phases, self._review_phase_roles), 1)
                  if role == "test"}
        active_part = None
        if self.review_image and self.review_image[0] == self.active_camera and self._review_part:
            try:
                active_part = self._analysis_image_part()
            except CaptureError as exc:
                self._error(exc)
                return
        self._sequence_export_window = SequenceExportWindow(
            self.root, self.experiment, self._experiment_id, cameras, self.active_camera, phases, active_part, self._job,
        )

    def open_experiment(self, folder=None, *, expected_id=None, on_opened=None, on_error=None, wanted=None):
        if reason := self._setup_block(needs_idle_camera=True):
            self.activity.set(reason)
            return False
        folder = folder or filedialog.askdirectory(parent=self.root, title="打开采集实验", mustexist=True)
        if not folder:
            return False
        self._opening_experiment = True
        self.activity.set("正在打开实验，读取采集台账…")

        def load():
            try:
                with ExperimentReader(folder) as reader:
                    experiment = reader.experiment
                    if self._practice:
                        require_simulated_cameras(camera["id"] for camera in experiment["config"]["cameras"])
                    if expected_id is not None and experiment["experiment_id"] != expected_id:
                        raise CaptureError("RECENT_EXPERIMENT_MISMATCH", "所选目录的实验标识与最近记录不一致。")
                    manifest = reader.root / TRANSFER_MANIFEST
                    return reader.root, experiment, reader.cameras(), manifest.exists() or manifest.is_symlink()
            except (OSError, sqlite3.Error) as exc:
                raise CaptureError("EXPERIMENT_UNAVAILABLE", f"无法读取实验台账。\n{folder}\n{exc}") from exc

        def loaded(result):
            self._opening_experiment = False
            if self._closing or (wanted is not None and not wanted()):
                return
            self.experiment, experiment, camera_rows, self._review_is_copy = result
            self._review_revision += 1
            self._experiment_id = experiment["experiment_id"]
            self._review_state = experiment["state"].upper()
            self.session = None
            self.counter.set("")
            self._set_review_action(experiment["config"], trial_reference=experiment.get("trial_reference"))
            self._review_ordinal = None
            self._review_trigger = None
            self._review_sensor_geometry = {}
            self._review_deferred = None
            self.review_image = None
            self._set_review_parts(None)
            self.inventory.update({row["device"]["id"]: row["device"] for row in camera_rows})
            self._experiment_cameras = tuple(row["device"]["id"] for row in camera_rows)
            self._show_views(self._experiment_cameras)
            self.review_scroll.canvas.yview_moveto(0)
            self.tabs.select(self.review_panel)
            state = STATE_LABELS.get(self._review_state, "状态未知")
            self.review_text.set(state)
            self.activity.set("已打开交接副本。可核验交接清单或进入分析。" if self._review_is_copy else
                              "已打开实验。请查看采集结果和已保存图像。")
            self._read_review_result()
            self.browse_frame(latest=True)
            history = RecentExperiment(experiment["experiment_id"], experiment["config"]["name"], self.experiment)
            self._job(lambda: remember_experiment(history), lambda _result: None)
            if on_opened is not None:
                on_opened()

        def failed(exc):
            self._opening_experiment = False
            if self._closing or (wanted is not None and not wanted()):
                return
            self.activity.set(ui_error(exc, details=False))
            (on_error or self._error)(exc)

        self._job(load, loaded, failed)
        return True

    def _read_review_result(self):
        if self._review_trial or self._board_scan:
            return
        path, identity = self.experiment, self._experiment_id
        ticket = self._review_read_ticket = object()
        self._review_read_error = ""
        self._render_review_result()

        def current():
            return not self._closing and self._review_read_ticket is ticket

        def loaded(outcome):
            if current():
                self._review_outcome = outcome
                self._render_review_result()

        def failed(exc):
            if current():
                self._review_read_error = ui_error(exc)
                self._render_review_result()

        self._job(lambda: inspect_capture_review(path, expected_id=identity), loaded, failed)

    def _render_review_result(self):
        outcome, verification = self._review_outcome, self._review_integrity
        issue_count = self._review_issue_count = len(integrity_issues(verification)) if verification else 0
        if self._review_read_ticket is None:
            self.review_heading.configure(text="已保存图像")
            self.review_result.pack_forget()
            self.review_report_button.pack_forget()
            return
        pending_writes = outcome and outcome.write_records_need_review and not (verification and verification.get("recovery"))
        if verification:
            frames, images = verification["committed_frames"], verification["committed_images"]
            if issue_count:
                check = f"校验发现 {issue_count} 项问题"
            elif pending_writes:
                check = "帧写入记录待核对"
            else:
                check = "本次文件与台账校验通过" if images else "台账已检查 · 无 TIFF"
            text = f"已登记 {frames:,} 帧 · {images:,} 张 TIFF\n{check}"
        elif outcome:
            check = "帧写入记录待核对" if pending_writes else "文件待校验"
            text = f"已登记 {outcome.frames:,} 帧 · {outcome.images:,} 张 TIFF\n{check}"
        else:
            text = "采集结果暂不可读" if self._review_read_error else "正在读取采集结果…"
        self.review_heading.configure(text="采集结果")
        self.review_result.configure(text=text)
        self.review_result.pack(fill=tk.X, pady=(0, 6), before=self.review_image_detail)
        if issue_count:
            self.review_report_button.pack(fill=tk.X, pady=(0, 6), before=self.review_image_detail)
        else:
            self.review_report_button.pack_forget()

    def _review_result_help(self, _event):
        if self._review_read_error:
            return self._review_read_error
        if (outcome := self._review_outcome) is None:
            return "正在从实验台账读取各台相机登记的图像数量。"
        lines = ["帧数来自采集台账。一帧含多个图像部件时，会保存多张 TIFF。"]
        lines.extend(f"{self._view_title(camera)} · {counts['frames']:,} 帧 · {counts['images']:,} 张 TIFF"
                     for camera, counts in outcome.cameras.items())
        lines.extend(f"阶段 {phase} · CSV · {CSV_STATES[state]}" for phase, state in outcome.csv_triggers)
        lines.extend(f"阶段 {row['phase']} · 完成 {row['completed_events']} 次 · 部分收图 {row['partial_events']} 次"
                     f" · 缓存未满时收到 {row['early_signals']} 次 · 正在收图时收到 {row['busy_signals']} 次"
                     for row in outcome.event_phases)
        if incidents := capture_incident_lines(outcome.camera_losses, outcome.scheduled_misses, outcome.short_pre_windows,
                                              self._view_title):
            lines.extend(("", *incidents))
        if not outcome.runtime_recorded:
            lines.append("台账缺少完整的运行期统计。")
        if outcome.write_records_need_review and not (self._review_integrity and self._review_integrity.get("recovery")):
            lines.append("点“检查中断”核对帧写入记录与遗留 TIFF。")
        elif self._review_integrity is None:
            lines.append("点“校验图像和台账”核对每张原图、归档标定和台账触发记录。")
        else:
            lines.append("文件校验针对本次检查时的存档。曝光同步可通过独立触发或计时记录核对。")
        return "\n".join(lines)

    def _set_review_parts(self, frame, selected=0):
        self._review_part = selected
        self._review_counters = frame_hardware_counters(frame.get("backend_metadata")) if frame else {}
        self._review_part_choices = {}
        if frame and frame.get("auxiliary_images"):
            details = {}
            for index, asset in enumerate(frame_image_records(frame)):
                part = (asset.get("backend_metadata") or {}).get(IMAGE_PART, {}).get("selector")
                label = f"部件 {index + 1} · {part or asset['pixel_format']}" if index else f"主图像 · {asset['pixel_format']}"
                self._review_part_choices[label] = index
                height, width = asset["shape"][:2]
                details[label] = f"{width} × {height} 像素 · {asset['significant_bits']} 位有效数据\n{asset['path']}"
            self.review_part_menu.configure(values=tuple(self._review_part_choices))
            self._review_part_help.set_choices(details)
            self._restore_review_part_label()
            self.review_part_row.pack(fill=tk.X, pady=(6, 3), before=self.trigger_actions)
        else:
            self.review_part_row.pack_forget()

    def _saved_frame_help(self, _event):
        if self.experiment is None:
            return "从顶部打开已有实验，或先完成一次采集。"
        lines = [f"当前实验\n{self.experiment}"]
        if self._review_frame_error:
            lines.extend(("", self._review_frame_error))
        if self._review_counters:
            lines.extend(("此帧的相机计数值", *(f"{field} · {value}" for field, value in self._review_counters.items())))
        return "\n".join(lines)

    def _restore_review_part_label(self):
        for label, index in self._review_part_choices.items():
            if index == self._review_part:
                self.review_part_menu.set(label)
                break

    def _choose_review_part(self, _event=None):
        if (part := self._review_part_choices.get(self.review_part_menu.get())) is not None and part != self._review_part:
            self.browse_frame(part=part)

    def _frame_search_block(self):
        if reason := self._review_help(""):
            return reason
        if self.preview:
            return "请先断开相机预览，再查找已保存图像。"
        if not self._experiment_cameras:
            return "当前台账还没有可复查的相机。"
        return ""

    def _review_scope_labels(self):
        cameras = {camera: self._view_title(camera) for camera in self._experiment_cameras}
        phases = {str(number): f"{number} · {MODE_LABELS[mode]}" for number, mode in enumerate(self._review_phases, 1)}
        return cameras, phases

    def _archive_quality_block(self):
        if reason := self._review_help("") or self._setup_block():
            return reason
        if self._checking_experiment:
            return "正在检查实验台账，请等待报告。"
        if self._review_trial:
            return "请在试采报告中点“检查试采原图”。"
        if not self._experiment_cameras or not self._review_phases:
            return "请选择有相机和阶段记录的实验。"
        if (self._review_state or "").lower() not in TERMINAL_STATES:
            return "请在采集完成或中断恢复后检查已保存的原图。"
        return ""

    def show_archive_quality(self):
        if reason := self._archive_quality_block():
            self.activity.set(reason)
            return
        experiment, identity = self.experiment, self._experiment_id
        window = getattr(self, "_archive_quality_window", None)
        if window is not None and window.top.winfo_exists():
            if (window.experiment, window.experiment_id) == (experiment, identity):
                window.top.deiconify()
                window.top.lift()
                return
            window.top.destroy()

        def current():
            return (self.experiment, self._experiment_id) == (experiment, identity) and not self._closing

        def show(options):
            if current():
                cameras, phases = self._review_scope_labels()
                self._archive_quality_window = ArchiveQualityWindow(
                    self.root, experiment, identity, options, cameras, phases, self.active_camera, self._job,
                    lambda: self._archive_quality_block() if current() else "请先回到此窗口对应的实验。",
                    lambda: not current() or self._busy(), self._open_archive_quality_frame,
                    self._open_archive_comparison, _operation_state_dir(),
                )

        self._experiment_job(lambda: archive_quality_options(experiment, identity), show)

    def _open_archive_comparison(self, experiment, experiment_id, camera, phase, row):
        if self._busy() or self._closing:
            return "请等本次采集结束后再对照。"
        if (experiment, experiment_id) != (self.experiment, self._experiment_id):
            return "请先回到此窗口对应的实验。"
        self._comparison_window().open_archive(lambda: open_archive_comparison(experiment, experiment_id, camera, phase, row))
        return ""

    def _open_archive_quality_frame(self, experiment, location, on_opened, on_error):
        try:
            self._check_search_scope(experiment, location.experiment_id)
        except CaptureError as exc:
            on_error(exc)
        else:
            self.browse_frame(part=0, location=location, on_opened=on_opened, on_error=on_error)

    def show_live_history(self):
        if reason := self._review_help(""):
            self.activity.set(reason)
            return
        from .live_history_gui import LiveHistoryWindow

        previous = getattr(self, "_live_history_window", None)
        if previous is not None and previous.top.winfo_exists():
            if (previous.experiment, previous.expected_id) == (self.experiment, self._experiment_id):
                previous.top.lift()
                return
            previous.top.destroy()
        try:
            self._live_history_window = LiveHistoryWindow(self.root, self.experiment, expected_id=self._experiment_id)
        except (CaptureError, OSError) as error:
            self._error(error)

    def show_timeline(self, *, phase=None, all_cameras=False, event_filter=""):
        if reason := self._review_help(""):
            self.activity.set(reason)
            return
        window = getattr(self, "_timeline_window", None)
        if window is not None and window.top.winfo_exists():
            if (phase is None and not all_cameras and not event_filter and window.experiment == self.experiment
                    and window.experiment_id == self._experiment_id):
                window.reopen()
                return
            window.top.destroy()
        cameras, phases = self._review_scope_labels()
        outcome = self._review_outcome
        notes = capture_incident_lines(outcome.camera_losses, outcome.scheduled_misses, outcome.short_pre_windows,
                                       self._view_title) if outcome else ()
        self._timeline_window = TimelineWindow(
            self.root, self.experiment, self._experiment_id, cameras, self.active_camera, phases,
            self._job, self._open_timeline_frames, self._show_search_trigger, phase=phase, all_cameras=all_cameras,
            open_marker=self._open_timeline_marker, event_filter=event_filter, summary_notes=notes,
        )

    def _open_timeline_marker(self, experiment, identity, marker_id):
        self._check_search_scope(experiment, identity)
        previous = getattr(self, "_marker_review_window", None)
        if previous is not None and previous.top.winfo_exists():
            if (previous.experiment, previous.experiment_id, previous.marker_id) == (experiment, identity, marker_id):
                previous.reopen()
                return
            previous.top.destroy()
        timeline = self._timeline_window
        cameras, _phases = self._review_scope_labels()

        def open_frame(location, opened, failed):
            def hide_windows():
                if timeline.top.winfo_exists():
                    timeline.top.withdraw()
                opened()

            self._show_search_frame(experiment, location, hide_windows, failed)

        self._marker_review_window = MarkerReviewWindow(
            self.root, experiment, identity, marker_id, cameras, self._job, open_frame)

    def _open_timeline_frames(self, experiment, identity, scope):
        self._check_search_scope(experiment, identity)
        timeline = self._timeline_window

        def hide_timeline():
            if timeline.top.winfo_exists():
                timeline.top.withdraw()

        self.show_frame_search(scope=scope, on_opened=hide_timeline)

    def show_frame_search(self, *, scope=None, on_opened=None):
        if reason := self._frame_search_block():
            self.activity.set(reason)
            return
        window = getattr(self, "_frame_search_window", None)
        if window is not None and window.top.winfo_exists():
            if scope is None and window.experiment == self.experiment and window.experiment_id == self._experiment_id:
                window.reopen()
                return
            window.top.destroy()
        cameras, phases = self._review_scope_labels()
        self._frame_search_window = FrameSearchWindow(
            self.root, self.experiment, self._experiment_id, cameras, self.active_camera, phases,
            self._job, self._show_search_frame, self._show_search_trigger, scope=scope, on_opened=on_opened,
            use_event=self._use_event_images,
        )

    def _use_event_images(self, parent, experiment, identity, camera, context, action, finished, failed):
        self._check_search_scope(experiment, identity)
        part = None
        if self.review_image and self.review_image[0] == camera and self._review_part:
            part = self._analysis_image_part()

        def read():
            with ExperimentReader(experiment) as reader:
                record = reader.experiment
                check_review_identity(record, identity)
                return event_image_choices(reader, record, camera, context)

        def launch(_snapshot):
            if self._closing or not parent.winfo_exists():
                return
            try:
                self._check_search_scope(experiment, identity)
                launch_analysis(experiment, camera_id=camera, reference_sequence=sequence, image_part=part,
                                event_context_id=context, expected_id=identity)
            except (CaptureError, OSError) as exc:
                failed(exc)
            else:
                self.activity.set("已在独立窗口中打开本次事件的离线分析。")
                finished()

        sequence = None

        def choose(data):
            nonlocal sequence
            if self._closing or not parent.winfo_exists():
                return
            event, counts, references = data
            try:
                self._check_search_scope(experiment, identity)
                candidates = {}
                guidance = {}
                positions = {"pre": "事件前图", "boundary": "边界图", "post": "事件后图"}
                for row in references:
                    if part is not None and not any(frame_part_selector(asset) == part for asset in frame_image_records(row)):
                        continue
                    role = "实验参考图" if row["role"] == "reference" else positions.get(row.get("event_position"), "事件图")
                    title = f"原帧 {row['sequence']} · {role}"
                    candidates[title] = row["sequence"]
                    guidance[title] = ("以已登记的实验参考图作为基准。保留本次事件的全部实验图。"
                                       if row["role"] == "reference" else
                                       "用这张事件图作为基准。只使用本次事件中比它晚的图像。")
                if not candidates:
                    raise CaptureError("EVENT_REFERENCE_REQUIRED", "当前图像部件没有可用的参考图。")
                prompt = (f"{self._view_title(camera)} · 阶段 {event['phase']} · 事件 {event['event_number']}\n"
                          f"前图 {counts['pre']} 张 · 边界 {counts['boundary']} 张 · 后图 {counts['post']} 张。"
                          "请选择计算或导出所用的基准图。")
                picked = CaptureChoiceDialog(parent, "选择本次事件的参考图", prompt, guidance,
                                             confirm_text="继续导出" if action == "export" else "打开分析").result
                if self._closing or not parent.winfo_exists():
                    return
                if picked is None:
                    finished()
                    return
                self._check_search_scope(experiment, identity)
                sequence = candidates[picked]
                if action == "export":
                    self._show_event_export(experiment, identity, camera, part, event, sequence)
                    finished()
                else:
                    self._job(lambda: analysis_snapshot(experiment, camera, sequence, image_part=part,
                                                        event_context_id=context, expected_id=identity), launch, failed)
            except CaptureError as exc:
                failed(exc)

        self._job(read, choose, failed)

    def _show_event_export(self, experiment, identity, camera, part, event, sequence):
        if reason := self._sequence_export_block():
            raise CaptureError("EVENT_IMAGES_ACTIVE", reason)
        prior = getattr(self, "_sequence_export_window", None)
        if prior is not None and prior.top.winfo_exists():
            if prior._working:
                prior.top.lift()
                raise CaptureError("EVENT_IMAGES_ACTIVE", "请让当前导出完成后再导出另一次事件。")
            prior.top.destroy()
        self._sequence_export_window = SequenceExportWindow(
            self.root, experiment, identity, {camera: self._view_title(camera)}, camera, {}, part, self._job,
            event=event, reference_sequence=sequence,
        )

    def _check_search_scope(self, experiment, identity):
        if self.experiment != experiment or self._experiment_id != identity:
            raise CaptureError("FRAME_REVIEW_CHANGED", "the search window belongs to a different experiment")
        if reason := self._frame_search_block():
            raise CaptureError("FRAME_REVIEW_UNAVAILABLE", reason)

    def _show_search_frame(self, experiment, location, on_opened, on_error):
        try:
            self._check_search_scope(experiment, location.experiment_id)
        except CaptureError as exc:
            on_error(exc)
        else:
            self.browse_frame(location=location, on_opened=on_opened, on_error=on_error)

    def _show_search_trigger(self, experiment, identity, event):
        self._check_search_scope(experiment, identity)
        names = {camera: self._view_title(camera) for camera in self._experiment_cameras}
        self._show_trigger_review(experiment, names, event)

    def browse_frame(self, direction=0, *, latest=False, part=None, location=None, on_opened=None, on_error=None):
        if reason := self._review_help("", needs_camera=location is None):
            (on_error or self.activity.set)(reason)
            return
        if self._review_busy:
            if location is not None:
                (on_error or self.activity.set)("正在读取另一张原图。请稍候再打开这张。")
            else:
                self._review_deferred = (direction, latest, part)
            return
        path, identity, revision = self.experiment, self._experiment_id, self._review_revision
        requested_tab = self.tabs.select()
        camera = location.camera_id if location is not None else self.active_camera
        ordinal = self._review_ordinal
        wanted_part = self._review_part if part is None else part
        if camera != self.active_camera:
            wanted_part = 0
        self._review_busy = True

        def read():
            newest = latest or ordinal is None
            after = ordinal if not newest and direction > 0 else 0
            before = ordinal if not newest and direction < 0 else None
            if not newest and direction == 0:
                after, before = ordinal - 1, ordinal + 1
            with ExperimentReader(path) as reader:
                if location is None:
                    rows = reader.frames(camera_id=camera, after=after, before=before, limit=1,
                                         newest_first=newest or before is not None)
                    if not rows:
                        return None
                    row = rows[0]
                else:
                    row = location.read(reader)
                assets = frame_image_records(row)
                index = wanted_part if 0 < wanted_part < len(assets) else 0
                chosen = assets[index]
                image = recorded_image(reader.image_path(chosen), {**chosen, "timestamps": row["timestamps"]},
                                       verify=location is not None)
                geometry = phase_sensor_geometry(reader, camera).get(str(row["phase"]))
                if geometry is None:
                    features = next(item["features"] for item in reader.cameras() if item["device"]["id"] == camera)
                    geometry = {name: feature["value"] for name, feature in features.items()}
                return row, image, geometry, index

        def still_selected():
            return (not self._closing and self.experiment == path and self._experiment_id == identity
                    and self._review_revision == revision and self.preview is None
                    and (location is None or self.tabs.select() == requested_tab))

        def changed():
            if on_error and not self._closing:
                on_error("工作台中的实验、相机或页面已切换。请回到对应实验后重新查找。")

        def finished(result):
            if still_selected():
                self._review_frame_error = ""
                if result is not None:
                    if location is not None:
                        self.tabs.select(self.review_panel)
                        self._show_views(self._experiment_cameras, browse_saved=False)
                        self.select_view(camera, browse_saved=False)
                    row, image, self._review_sensor_geometry, index = result
                    if index != self._review_part and (view := self.views.get(camera)):
                        view.roi = None
                        view.show_calibration(None)
                        self._update_region_caption()
                    self._set_review_parts(row, index)
                    self._review_ordinal = row["ordinal"]
                    self._review_trigger = row.get("trigger_event_id")
                    self.review_image = (camera, image)
                    self._quality_due = 0.
                    caption = (f"{self._view_title(camera)}  ·  序号 {row['sequence']}\n"
                               f"{ROLE_LABELS[row['role']]}  ·  阶段 {row['phase']}  ·  {image.pixel_format}")
                    part = image.metadata.get(IMAGE_PART)
                    if isinstance(part, dict) and part.get("selector"):
                        caption += f"\nGenTL 部件 {part['selector']}"
                    if self._review_counters:
                        caption += "\n悬停可查看相机计数值"
                    self.review_text.set(caption)
                    if on_opened:
                        on_opened()
                elif ordinal is None:
                    self.review_text.set("这台相机还没有已保存的图像。")
                else:
                    self.activity.set("当前相机已经显示第一张已保存图像。" if direction < 0 else
                                      "当前相机已经显示最新的已保存图像。")
            else:
                changed()
            release()

        def failed(exc):
            if still_selected():
                self._restore_review_part_label()
                if on_error:
                    on_error(exc)
                elif isinstance(exc, (CaptureError, OSError, ValueError)):
                    self._review_frame_error = ui_error(exc)
                    if self._review_integrity and self._review_integrity["ok"]:
                        self._review_integrity = None
                        self._render_review_result()
                    if self.review_image is None:
                        self.review_text.set("图像无法打开 · 悬停查看原因")
                    self.activity.set("无法打开所选图像。可用“校验图像和台账”检查存档。")
                else:
                    self._error(exc)
            else:
                changed()
            release()

        def release():
            self._review_busy = False
            pending, self._review_deferred = self._review_deferred, None
            if pending is not None and not self._closing and self._saved_panel():
                self.browse_frame(pending[0], latest=pending[1], part=pending[2])

        self._job(read, finished, failed)

    def open_trigger_review(self):
        if reason := self._review_help("", browsing=True):
            self.activity.set(reason)
            return
        event = self._review_trigger if self._review_ordinal is not None and self.review_image else None
        names = {key: self._view_title(key) for key in self._experiment_cameras}
        self._show_trigger_review(self.experiment, names, event)

    def _show_trigger_review(self, experiment, names, event, timing_note="", *, expected_experiment_id=None):
        identity = expected_experiment_id if expected_experiment_id is not None else self._experiment_id
        window = self._trigger_review
        if window is not None and window.top.winfo_exists():
            if (window.experiment == experiment and window.experiment_id == identity
                    and window.reopen(event, timing_note=timing_note)):
                return
            window.top.destroy()
        self._trigger_review = TriggerReview(self.root, experiment, names, self._job, event_id=event,
                                             timing_note=timing_note, expected_experiment_id=identity)

    def open_pulse_audit(self):
        if reason := self._review_help("", needs_idle=True):
            self.activity.set(reason)
            return
        experiment = self.experiment
        window = self._pulse_audit
        if window is not None and window.top.winfo_exists():
            if window.experiment == experiment:
                window.top.lift()
                return
            window.top.destroy()
        names = {key: self._view_title(key) for key in self._experiment_cameras}

        def show(context):
            if self._closing or self.experiment != experiment:
                return
            experiment_id, phases, cameras = context
            if not phases:
                self.activity.set("实验需要有已完成采集的硬件触发阶段。")
                messagebox.showinfo("核对 DAQ 脉冲", "当前实验计划中没有硬件触发阶段。", parent=self.root)
                return
            self._pulse_audit = PulseAuditWindow(self.root, experiment, phases, {camera: names.get(camera, camera) for camera in cameras},
                                                self._job, lambda event, note: self._show_trigger_review(
                                                    experiment, names, event, note, expected_experiment_id=experiment_id),
                                                experiment_id=experiment_id)

        self._experiment_job(lambda: pulse_audit_context(experiment), show)

    def recover_run(self):
        if self._review_help("", needs_idle=True):
            return
        path = self.experiment
        identity = self._experiment_id
        names = {camera: self._view_title(camera) for camera in self._experiment_cameras}

        def applied(report):
            if (self.experiment, self._experiment_id) != (path, identity) or self._closing:
                return
            state = report["state"].upper()
            recovered = len(report["recovery"]["recovered"])
            self.session = None
            self._review_state = state
            self.counter.set("")
            self.review_text.set(f"{STATE_LABELS.get(state, state)}  ·  已保存 {report['committed_images']} 张 TIFF\n"
                                 f"本次恢复 {recovered} 帧")
            self.activity.set("已完成台账恢复和原图校验。可复查图像或检查续采计划。")
            if self._board_scan:
                self._set_checkerboard_scan(self._board_scan.target)
            elif regular_experiment:
                self._review_integrity = report
                self._read_review_result()
            if recovered:
                self.browse_frame(latest=True)

        def check_continuation(identity):
            if self.experiment == path and not self._closing:
                self.prepare_continuation(expected_experiment_id=identity, check_plan=True)

        regular_experiment = not (self._board_scan or self._review_trial)
        self._recovery_window = RecoveryWindow(self.root, path, names, self._job, applied,
                                               check_continuation if regular_experiment else None,
                                               experiment_id=self._experiment_id)

    def _repeat_block(self):
        if reason := self._review_help("", needs_idle=True):
            return reason
        if self._review_trial or self._board_scan:
            return "请在复查页打开一次正式采集作为来源。"
        if (self._review_state or "").lower() not in TERMINAL_STATES:
            return "请先结束采集或恢复中断台账，再准备复拍。"
        return ""

    def show_repeat_plan(self):
        if reason := self._repeat_block():
            self.activity.set(reason)
            return
        source, identity = self.experiment, self._experiment_id

        def current():
            return (self.experiment, self._experiment_id) == (source, identity) and not self._closing

        def show(draft):
            if not current():
                return
            if (prior := getattr(self, "_repeat_window", None)) is not None and prior.top.winfo_exists():
                prior.top.destroy()

            def load(name, destination, finished, failed):
                if not current():
                    failed(CaptureError("REPEAT_SCOPE_CHANGED", "当前复查的实验已变化，请重新打开复拍计划。"))
                    return
                if reason := self._repeat_block():
                    failed(reason)
                    return

                def read():
                    updated = prepare_repeat(source, name=name, expected_experiment_id=identity)
                    if (replace(updated.config, name=draft.config.name) != draft.config
                            or updated.cleared_anchors != draft.cleared_anchors):
                        raise CaptureError("REPEAT_SCOPE_CHANGED", "来源计划已变化，请重新打开复拍计划。")
                    return updated.config, validate_repeat_destination(updated.source, destination)

                def install(result):
                    if not window.top.winfo_exists():
                        return
                    if not current():
                        failed(CaptureError("REPEAT_SCOPE_CHANGED", "当前复查的实验已变化，请重新打开复拍计划。"))
                        return
                    if reason := self._repeat_block():
                        failed(reason)
                        return
                    config, output = result
                    self.session = None
                    self.counter.set("")
                    self._trial_results.clear()
                    self._set_config(config)
                    self.output_path.set(str(output))
                    self.tabs.select(self.plan_panel)
                    self.plan_summary.set("已载入复拍计划。正在重新查找相机。")
                    self.activity.set("已填写新的实验名称和输出位置。")
                    finished()
                    self.discover(on_finished=self.check_plan)

                self._experiment_job(read, install, failure=failed)

            window = self._repeat_window = RepeatPlanWindow(
                self.root, draft, self._new_path(draft.source.parent), self._view_title, self._new_path, load,
            )

        self.activity.set("正在读取来源计划和已归档的标定文件…")
        self._experiment_job(lambda: prepare_repeat(source, expected_experiment_id=identity), show)

    def prepare_continuation(self, *, expected_experiment_id=None, check_plan=False):
        if reason := self._review_help("", needs_idle=True):
            self.activity.set(reason)
            return
        if not check_plan and self._review_state and self._review_state.lower() not in TERMINAL_STATES:
            self.recover_run()
            return
        path = self.experiment
        experiment_id = expected_experiment_id or self._experiment_id

        def current():
            return self.experiment == path and self._experiment_id == experiment_id and not self._closing

        def show_timeline(phase):
            if not current():
                raise CaptureError("RECOVERY_SCOPE_CHANGED", "请选择当前实验的续采起点。")
            self.show_timeline(phase=phase)

        def load_plan(phase, finished, failed):
            if not current():
                raise CaptureError("RECOVERY_SCOPE_CHANGED", "请选择当前实验的续采起点。")
            if reason := self._review_help("", needs_idle=True):
                failed(reason)
                return

            def applied(config):
                if not current():
                    failed(CaptureError("RECOVERY_SCOPE_CHANGED", "请选择当前实验的续采起点。"))
                    return
                if reason := self._setup_block(needs_idle_camera=True):
                    failed(reason)
                    return
                self.session = None
                self.counter.set("")
                self._set_config(config)
                self.output_path.set(self._new_path(path.parent))
                self.plan_summary.set(f"已载入续采计划 · 从原阶段 {phase} 开始。检查计划可对照原阶段和相机回读。")
                self.activity.set("已准备新实验目录。检查相机、阶段和参考图安排后即可采集。")
                self.tabs.select(self.plan_panel)
                finished()
                if check_plan:
                    self.check_plan()

            self._experiment_job(lambda: continuation_plan(path, start_phase=phase, expected_experiment_id=experiment_id),
                                 applied, failure=failed)

        def show_selection(report):
            if not current():
                return
            if not check_plan and report["write_records_need_review"]:
                self.recover_run()
                return
            window = getattr(self, "_continuation_window", None)
            if window is not None and window.top.winfo_exists():
                window.top.destroy()
            self.activity.set("已读取原阶段记录。请选择续采起点。")
            self._continuation_window = ContinuationWindow(
                self.root, path, report, self._view_title, load_plan, show_timeline,
            )

        self.activity.set("正在读取原实验各阶段的保存情况…")
        self._experiment_job(lambda: inspect_continuation(path, expected_experiment_id=experiment_id), show_selection)

    def verify(self):
        if reason := self._review_help("", needs_idle=True):
            self.activity.set(reason)
            return
        path = self.experiment
        identity, ticket = self._experiment_id, self._review_read_ticket
        expect_manifest = self._review_is_copy
        self.activity.set("正在校验已保存图像和触发记录…")

        def action():
            copy_report = None
            manifest = path / TRANSFER_MANIFEST
            if expect_manifest or manifest.exists() or manifest.is_symlink():
                try:
                    copy_report = verify_transfer(path)
                except (CaptureError, OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
                    copy_report = {"ok": False, "detail": ui_error(exc)}
            with ExperimentReader(path) as reader:
                check_review_identity(reader.experiment, identity)
                report = reader.verify(hash_files=not (copy_report and copy_report["ok"]))
            if copy_report is not None:
                report["transfer_copy"] = copy_report
                if not copy_report["ok"]:
                    report["issues"].append({"code": "TRANSFER_MANIFEST_MISMATCH"})
                    report["ok"] = report["complete"] = False
            return report

        def current():
            return not self._closing and (self.experiment, self._experiment_id, self._review_read_ticket) == (
                path, identity, ticket,
            )

        def report(data):
            if not current():
                return
            self._review_integrity = data
            if self._review_read_error:
                self._read_review_result()
            self._render_review_result()
            self.activity.set("校验完成，发现文件或台账问题。" if self._review_issue_count else "图像文件和台账校验完成。")
            self._show_integrity_report("图像和台账校验", data)

        def failed(exc):
            if current():
                self._error(exc)

        self._experiment_job(action, report, failure=failed)

    def _experiment_job(self, action, report, *, failure=None):
        self._checking_experiment = True

        def finished(data):
            self._checking_experiment = False
            report(data)

        def failed(exc):
            self._checking_experiment = False
            if not self._closing:
                (failure or self._error)(exc)

        self._job(action, finished, failed)

    def _show_integrity_report(self, title, report):
        self._integrity_report_window = self._open_text_report(
            title, format_integrity(report), getattr(self, "_integrity_report_window", None),
        )

    def show_review_integrity(self):
        if self._review_integrity and not self._review_help("", needs_idle=True):
            self._show_integrity_report("图像和台账校验", self._review_integrity)

    def show_trial_comparison(self):
        if reason := self._review_help("", needs_idle=True):
            self.activity.set(reason)
            return
        formal, identity = self.experiment, self._experiment_id

        def read():
            with ExperimentReader(formal) as reader:
                if reader.experiment["experiment_id"] != identity:
                    raise CaptureError("FRAME_REVIEW_CHANGED", str(formal))
                return reader.trial_comparison()

        def show(comparison):
            if self._closing or (formal, identity) != (self.experiment, self._experiment_id):
                return
            if not comparison:
                self.activity.set("本实验暂无已关联的试采记录。")
                return
            labels = self._report_camera_labels({"cameras": self._experiment_cameras})
            stages = [stage for stage in comparison["stages"] if stage["committed_images"]]
            detail = ("打开关联试采原图", lambda: self._open_referenced_trial(formal, identity, stages),
                      "按阶段打开采前记录的试采原图。目录迁移后可手动定位并核对实验编号。") if stages else None
            self._trial_comparison_window = self._open_text_report(
                "试采与正式设置", format_trial_comparison(comparison, camera_labels=labels),
                getattr(self, "_trial_comparison_window", None), detail_action=detail,
            )

        self._experiment_job(read, show)

    def _open_referenced_trial(self, formal, identity, stages):
        dialog = self._trial_comparison_window

        def current():
            return (formal, identity) == (self.experiment, self._experiment_id) and not self._closing and dialog.winfo_exists()

        if not current():
            self.activity.set("请打开报告对应的正式实验，再查看试采原图。")
            return
        if reason := self._review_help("", needs_idle=True):
            self.activity.set(reason)
            return
        stage = stages[0]
        if len(stages) > 1:
            choices = {f"阶段 {row['phase']} · {row['committed_images']} 张 TIFF": row for row in stages}
            paths = {label: row["source"]["directory"] for label, row in choices.items()}
            chosen = CaptureChoiceDialog(dialog, "选择试采阶段", "选择要查看的试采原图。", paths)
            if not chosen.result:
                return
            stage = choices[chosen.result]

        def load(directory=None):
            self._checking_experiment = True
            self.activity.set("正在核对试采编号与对应原图…")

            def result(report):
                self._checking_experiment = False
                if current():
                    self.activity.set("已打开关联试采原图。")
                    self._show_trial_images(report, use_current_plan=False)

            def failed(exc):
                self._checking_experiment = False
                if not current() or getattr(exc, "code", None) == "TRIAL_IMAGE_CHECK_CANCELLED":
                    return
                if directory is None and getattr(exc, "code", None) in {"TRIAL_REFERENCE_UNAVAILABLE", "TRIAL_REFERENCE_CHANGED"}:
                    self.activity.set(ui_error(exc, details=False))
                    folder = filedialog.askdirectory(parent=dialog, title="定位原试采目录", initialdir=formal.parent, mustexist=True)
                    if folder and current():
                        load(folder)
                else:
                    self._error(exc)

            self._job(lambda: referenced_trial_report(formal, stage, directory=directory,
                                                      cancelled=lambda: self._closing or self._busy()), result, failed)

        load()

    def _fit_block(self):
        reason = self._review_help("", needs_camera=self._board_scan is None or bool(self._review_part),
                                   needs_idle=True, browsing=True)
        if reason or self._board_scan is None or self._review_part:
            return reason
        result = self._board_result
        if self._board_scanning is self._board_scan or result is None:
            return self._board_error or "正在检查保存的棋盘格图像，请稍候。"
        if result["read_errors"]:
            return "部分原图无法校验或读取。请在复查页校验图像和台账。"
        if result["mixed_image_sizes"]:
            return "标定图像宽高不一致。请核对相机尺寸设置后重新采图。"
        if result["distinct_poses"] < DEFAULT_MIN_VIEWS:
            return f"目前有 {result['distinct_poses']} 个不同摆位。拟合至少需要 {DEFAULT_MIN_VIEWS} 个。"
        return ""

    def fit_calibration(self):
        if reason := self._fit_block():
            self.activity.set(reason)
            return
        try:
            part = self._analysis_image_part()
        except CaptureError as exc:
            self._error(exc)
            return
        parts = {self.active_camera: part} if part is not None else {}
        if self._board_scan:
            self._save_calibration(self._board_scan.target, image_parts=parts)
            return
        camera_ids = [self.active_camera]
        peers = [camera for camera in self.views if camera != self.active_camera]
        if peers and messagebox.askyesno("双相机标定", "以当前相机为左相机进行双相机标定？",
                                         parent=self.root):
            options = {f"{self._view_title(camera)} · {camera}": camera for camera in peers}
            dialog = CaptureChoiceDialog(
                self.root, "双相机标定", "选择拍摄时位于右侧的相机。",
                {name: f"选为右相机。请按拍摄时的相机位置核对设备编号 {camera}。" for name, camera in options.items()},
                confirm_text="使用所选相机",
            )
            if dialog.result is None:
                return
            camera_ids.append(options[dialog.result])
        CheckerboardDialog(self.root, camera_ids, lambda target: self._save_calibration(target, image_parts=parts))

    def _save_calibration(self, target, *, image_parts=None):
        output = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json", filetypes=[("镜头标定文件", "*.json")])
        if not output:
            return
        root = self.experiment

        def run():
            profile = calibrate(root, target.camera_ids, image_parts=image_parts, **target.parameters())
            write_snapshot(profile, output)
            return profile

        def finished(profile):
            if root != self.experiment or self._closing:
                return
            self._calibration_profiles[output] = profile
            labels = ("左相机", "右相机") if target.stereo else ("相机",)
            rms = "\n".join(f"{label}  {profile['cameras'][key]['reprojection_rms_px']:.3f} px"
                            for label, key in zip(labels, target.camera_ids))
            self.tabs.select(self.review_panel)
            self.review_text.set(f"标定已保存\n重投影均方根误差\n{rms}")
            self.activity.set("标定文件已保存。可从“镜头标定”载入文件，随下次实验归档。")

        self.activity.set("正在从已保存的棋盘格图像拟合镜头参数…")
        if image_parts:
            self.activity.set(f"正在拟合图像部件 {next(iter(image_parts.values()))} 的镜头参数…")
        self._experiment_job(run, finished)

    def load_calibration(self):
        if self._setup_block():
            return
        if path := filedialog.askopenfilename(parent=self.root, filetypes=[("镜头标定文件", "*.json")]):
            try:
                profile = read_profile(path)
                self._calibration_profiles.pop(path, None)
                self._calibration_profiles[path] = profile
                if path not in self.config["calibration_files"]:
                    self.config["calibration_files"].append(path)
                    self._invalidate_preflight()
                self.activity.set("下次采集会随实验保存此标定文件。")
            except Exception as exc:
                self._error(exc)

    def clear_calibration(self):
        if self._setup_block():
            return
        self.config["calibration_files"] = []
        self._invalidate_preflight()
        self._calibration_profiles.clear()
        for view in self.views.values():
            view.show_calibration(None)
        self.activity.set("已从下一次采集计划中移除标定文件。")

    def toggle_calibration(self):
        view = self.views.get(self.active_camera)
        if not view:
            self.activity.set("先选择要查看校正预览的相机。")
            return
        if view.corrected:
            view.show_calibration(None)
            self.activity.set("已切回原始图像。")
            return
        try:
            identity = self.active_camera
            image = self._image(identity)
            features = self._calibration_features(identity, image)
            if features is None:
                raise CaptureError("CALIBRATION_IMAGE_UNAVAILABLE", "请选择已连接的预览图像或一张保存的原图。")
            geometry = calibration_geometry(image.pixels.shape, features, image.metadata)
            archived = []
            if not self.preview and self.experiment:
                with ExperimentReader(self.experiment) as reader:
                    for record in reader.experiment.get("calibrations", ()):
                        if identity in record["camera_ids"]:
                            profile_path = archived_profile_path(reader.root, record)
                            profile = read_profile(profile_path, expected_hash=record["sha256"])
                            archived.append(profile["cameras"][identity])
            try:
                model = matching_calibration(archived, geometry)
            except CaptureError:
                for path in self.config["calibration_files"]:
                    if path not in self._calibration_profiles:
                        self._calibration_profiles[path] = read_profile(path)
                loaded = [profile["cameras"][identity] for profile in self._calibration_profiles.values()
                          if identity in profile["cameras"]]
                model = matching_calibration([*loaded, *archived], geometry)
            view.show_calibration(model, image=image, features=features)
            self.activity.set("已切换到校正预览。保存的 TIFF 仍使用相机的原始像素。")
        except Exception as exc:
            self._error(exc)

    def _calibration_features(self, identity, image, preview_states=None):
        if image is None:
            return None
        if self.preview:
            states = self.preview.status() if preview_states is None else preview_states
            status = states.get(identity, {})
            if status.get("state") == "preview":
                return status.get("features", {})
        elif self.review_image and self.review_image[0] == identity and self.review_image[1] is image:
            return self._review_sensor_geometry
        return None

    def _display_image(self, view, image, *, placeholder=None, preview_states=None, empty_caption="等待图像",
                       frame_caption=None):
        features = self._calibration_features(view.camera_id, image, preview_states) if view.corrected else None
        view.image(image, placeholder=placeholder, features=features, empty_caption=empty_caption, frame_caption=frame_caption)

    def _analysis_image_part(self):
        if not self._review_part:
            return None
        image = self.review_image[1]
        selector = image.metadata.get(IMAGE_PART, {}).get("selector")
        if selector is None:
            raise CaptureError("ANALYSIS_PART_SELECTOR", "该图像没有记录可用于系列分析的部件编号。")
        return selector

    def open_corrected_analysis(self):
        if reason := self._analysis_help("", needs_idle=True):
            self.activity.set(reason)
            return
        experiment, camera, ordinal = self.experiment, self.active_camera, self._review_ordinal
        try:
            part = self._analysis_image_part()
        except CaptureError as exc:
            self._error(exc)
            return
        label = self._view_title(camera)
        if part:
            label += f" · {part}"

        def offer(references):
            if self._closing or self.experiment != experiment or self.active_camera != camera:
                return
            self.activity.set("可选择参考图和镜头标定。")
            CorrectedAnalysisDialog(self.root, label, references, ordinal,
                                    lambda sequence, profile: self._prepare_corrected_analysis(experiment, camera, sequence, profile, part))

        self.activity.set("正在读取本实验的参考图…")
        self._experiment_job(lambda: corrected_reference_choices(experiment, camera, image_part=part), offer)

    def _prepare_corrected_analysis(self, experiment, camera, sequence, profile, image_part=None):
        if reason := self._review_help("", needs_camera=True, needs_idle=True):
            self.activity.set(reason)
            return
        if self.experiment != experiment or self.active_camera != camera:
            return
        loaded = tuple(dict.fromkeys((*self.config["calibration_files"], *self._calibration_profiles)))
        self._preparing_analysis = "正在生成校正图像，完成后可打开分析。"
        self.activity.set(f"正在从序号 {sequence} 生成校正图像…")

        def finished(binding):
            self._preparing_analysis = ""
            if not self._closing:
                try:
                    launch_corrected_analysis(binding)
                    self.activity.set("校正序列已保存，正在打开离线分析。")
                except OSError as exc:
                    self._error(CaptureError("ANALYSIS_LAUNCH_FAILED", f"{binding}\n{exc}"))

        def failed(exc):
            self._preparing_analysis = ""
            if not self._closing:
                self._error(exc)

        self._job(lambda: prepare_corrected_analysis(experiment, camera, sequence, lens_profile=profile,
                                                     loaded_profiles=loaded, image_part=image_part), finished, failed)

    def _open_continuation_reference(self, experiment, camera, part, *, live=False):
        identity = self._experiment_id
        window = getattr(self, "_continuation_reference_window", None)
        if window is not None and window.top.winfo_exists():
            if (window.root, window.identity, window.camera, window.part, window.live) == (
                experiment, identity, camera, part, live,
            ):
                window.top.lift()
                return
            window.top.destroy()

        def still_current():
            return (not self._closing and self.experiment == experiment and self._experiment_id == identity and
                    self.active_camera == camera and self._analysis_image_part() == part)

        def use_binding(binding):
            if not still_current():
                raise CaptureError("CONTINUATION_REFERENCE_CHANGED", "工作台已切换相机、实验或部件。请重新选择参考图。")
            if live:
                launch_live_reference(binding)
                self.activity.set("已绑定原实验参考图，正在打开实时分析。")
            else:
                launch_corrected_analysis(binding)
                self.activity.set("已保存参考图及本轮实验图的来源，正在打开离线分析。")

        def use_current():
            if still_current():
                self.open_analysis(live, offer_previous=False)

        title = self._view_title(camera)
        if part:
            title += f" · {part}"
        self._continuation_reference_window = ContinuationReferenceWindow(
            self.root, experiment, identity, camera, title, part, self._job, use_binding, use_current, live=live,
        )

    def open_analysis(self, live, *, offer_previous=True):
        if reason := self._analysis_help(""):
            self.activity.set(reason)
            return
        experiment, camera = self.experiment, self.active_camera
        try:
            part = self._analysis_image_part()
        except Exception as exc:
            self._error(exc)
            return
        if offer_previous and self._review_parent:
            if reason := self._analysis_help("", needs_idle=not live):
                self.activity.set(reason)
                return
            self._open_continuation_reference(experiment, camera, part, live=live)
            return

        def launch(selection=None):
            try:
                launch_analysis(experiment, camera_id=camera, live=live, reference_sequence=selection, image_part=part)
                self.activity.set(f"已打开部件 {part} 的分析。" if part else "已在独立窗口中打开分析。")
            except Exception as exc:
                self._error(exc)

        if live:
            launch()
            return
        self._preparing_analysis = "正在检查离线分析所需的参考图和实验图。"
        self.activity.set(self._preparing_analysis)

        def check():
            try:
                analysis_snapshot(experiment, camera, image_part=part)
            except CaptureError as exc:
                if exc.code != "CHOOSE_REFERENCE":
                    raise
                try:
                    return corrected_reference_choices(experiment, camera)
                except CaptureError as missing:
                    if missing.code != "TOO_FEW_FRAMES":
                        raise
                    return []
            return None

        def ready(rows):
            try:
                if self._closing:
                    return
                if self.experiment != experiment or self.active_camera != camera:
                    self.activity.set("已切换复查对象。可重新打开当前图像的分析。")
                    return
                selection = None
                if rows is not None:
                    selection = self._choose_analysis_reference(rows, part)
                    if selection is None:
                        self.activity.set("可继续复查图像。")
                        return
                launch(selection)
            finally:
                self._preparing_analysis = ""

        def failed(exc):
            self._preparing_analysis = ""
            if not self._closing:
                self.activity.set(ui_error(exc, details=False))
                self._error(exc)

        self._job(check, ready, failed)

    def _choose_analysis_reference(self, rows, part):
        manual = "按序号输入其他图像…"
        options = {f"第 {row['sequence']} 张 · 第 {row['phase']} 阶段": row["sequence"] for row in rows
                   if part is None or sum(frame_part_selector(asset) == part for asset in frame_image_records(row)) == 1}
        if options:
            guidance = {label: "用这一张作为基准，计算其后实验图像的应变。" for label in options}
            guidance[manual] = "手动输入已保存图像的序号。分析时会校验该图及其后的实验图。"
            chosen = CaptureChoiceDialog(self.root, "选择参考图", "这台相机保存了多张参考图。选择加载前的基准图像。",
                                          guidance, confirm_text="继续").result
            if chosen != manual:
                return options.get(chosen)
        return CaptureFieldDialog(
            self.root, "输入参考图序号", "可在复查页查看图像序号。",
            (("图像序号", "", "填写当前相机已保存图像的序号。应变会以这一张作为基准。"),),
            lambda value: _positive_number(value, int), "请输入大于零的整数序号。", confirm_text="使用此图像",
        ).result

    def _image(self, camera):
        if self.preview:
            return self.preview.latest_image(camera)
        if self._saved_panel():
            return self.review_image[1] if self.review_image and camera == self.review_image[0] else None
        return self.session.latest_image(camera) if self.session is not None and not self.session.finished else None

    def _set_checkerboard_scan(self, target):
        self._board_scan = CheckerboardScan(self.experiment, target) if target else None
        self._board_signature = None
        self._board_result = None
        self._board_error = ""
        if self._board_scan:
            self.board_progress.set("正在检查已保存的图像…")
            self.board_detail.set("")
            self.board_note.set("摆稳棋盘格后点击“触发一次”。")

    def _update_quality_panel(self):
        visible = self._board_scan is not None and self.preview is None
        if visible == self._board_visible:
            return
        self._board_visible = visible
        previous = self.quality_metrics if visible else self.board_panel
        current = self.board_panel if visible else self.quality_metrics
        previous.pack_forget()
        current.pack(fill=tk.BOTH, expand=True)
        self._quality_scroll.canvas.yview_moveto(0)
        self._update_workspace_tabs()
        self._update_region_caption()
        self._quality_due = 0.

    def _tick_checkerboard(self, saved, active):
        scan = self._board_scan
        signature = (saved, active)
        if scan is None or self._board_scanning is scan or self._board_signature == signature:
            return
        self._board_signature = signature
        self._board_scanning = scan

        def finished(report):
            if self._board_scanning is scan:
                self._board_scanning = None
            if self._board_scan is scan and not self._closing:
                self._board_result = report
                self._board_error = ""
                self._show_checkerboard_result(report, active)

        def failed(exc):
            if self._board_scanning is scan:
                self._board_scanning = None
            if self._board_scan is scan and not self._closing:
                self._board_result = None
                self._board_error = ui_error(exc, details=False)
                self.board_progress.set("无法检查棋盘格")
                self.board_detail.set("")
                self.board_note.set(self._board_error)

        self._job(scan.scan, finished, failed)

    def _show_checkerboard_result(self, report, active):
        target = self._board_scan.target
        self.board_progress.set(f"不同摆位 {report['distinct_poses']} / {DEFAULT_MIN_VIEWS}")
        detail = [f"{target.inner_columns} × {target.inner_rows} 内角点 · 方格边长 {target.square_mm:g} mm"]
        if target.stereo:
            detail.append(f"可用于拟合的配对 {report['usable_pairs']} 组")
        for index, camera in enumerate(report["cameras"].values()):
            name = ("左相机", "右相机")[index] if target.stereo else "相机"
            latest = {"waiting": "等待拍摄", "detected": "已识别", "not_detected": "未识别", "read_error": "读取失败"}
            detail.append(f"{name}  {camera['detected']}/{camera['saved']} 张 · {latest[camera['latest']]}")
            if span := camera["corner_span_fraction_xy"]:
                detail.append(f"角点跨度  横向 {span[0]:.0%}  纵向 {span[1]:.0%}")
        self.board_detail.set("\n".join(detail))

        last_step = {
            "waiting": "让整块棋盘格进入画面，摆稳后点击“触发一次”。",
            "new_pose": "本次识别到新的摆位。下一张可调整到画面的其他区域，并改变倾角。",
            "similar_pose": "本次与已有摆位接近。请移动棋盘格或改变倾角后再拍。",
            "not_detected": "本次未识别到角点。请核对内角点数、对焦和光照，确保棋盘格完整入镜。",
            "unpaired": "本次尚无可用配对。请核对两侧是否都已保存清晰的棋盘格图像。",
            "read_error": "本次原图无法读取或校验。",
        }[report["latest"]]
        if report["read_errors"]:
            note = f"有 {report['read_errors']} 张原图无法读取或校验。请在复查页校验图像和台账。"
        elif report["mixed_image_sizes"]:
            note = "标定图像宽高不一致。请核对相机尺寸设置后重新采图。"
        elif report["distinct_poses"] >= DEFAULT_MIN_VIEWS:
            note = ("已达到拟合所需的摆位数。可补拍或点击“停止并保存”。" if active else
                    "可点击“拟合镜头参数”生成标定文件。")
        else:
            note = last_step if active else "可用摆位数不足。请重新标定采图，覆盖更多位置和倾角。"
        self.board_note.set(note)

    def _quality_input(self):
        view = self.views.get(self.active_camera)
        if (self._board_visible or not view or self.tabs.select() != str(self.quality_panel)
                or self._camera_settings_busy or self._releasing_preview):
            return None
        image = self._image(self.active_camera)
        if image is None:
            return None
        source = self.preview or self.session
        review = (self._review_revision, self._review_ordinal) if self.preview is None and self.review_image is not None else None
        geometry = (image.pixels.shape, image.color_model, image.pixel_format, image.significant_bits,
                    image.metadata.get(IMAGE_PART, {}).get("selector"))
        return _QualityScope(source, view, review, view.roi, geometry), image

    def _show_focus_feedback(self, text):
        if text != self.focus_feedback.cget("text"):
            self.focus_feedback.configure(text=text)
            if text:
                self.focus_feedback.grid()
            else:
                self.focus_feedback.grid_remove()

    def _sync_quality_scope(self, scope):
        if scope == self._quality_key:
            return
        self._quality_key = scope
        self._quality_serial += 1
        self._quality_due = 0.
        self.hist.delete("all")
        title = "亮度直方图" if scope is None else ("选区亮度" if scope.roi else "全图亮度")
        self.hist_heading.configure(text=title)
        self.quality_text.set("正在读取当前画面的亮度和细节…" if scope else "连接预览或打开一张已保存的图像。")
        if self._focus_comparison is not None:
            self._focus_comparison = None
            self._show_focus_feedback("画面或设置已变更，可重新比较。")

    def _focus_block(self, preview_states=None):
        if self.preview is None:
            return "对焦比较需要实时画面。先连接相机预览。"
        if self._camera_settings_busy or self._releasing_preview:
            return "等相机参数更新并收到新画面后再比较。"
        states = self.preview.status() if preview_states is None else preview_states
        state = states.get(self.active_camera, {})
        if state.get("state") != "preview" or self.preview.latest_image(self.active_camera) is None:
            return "选中已有实时画面的相机，再开始比较。"
        features = state.get("features", {})
        if any(features.get(name) in ("Once", "Continuous") for name in ("exposure_auto", "gain_auto")):
            return "相机正在自动调整曝光或增益。可在相机参数中切到手动后再比较。"
        return ""

    def _preview_quality_block(self, preview_states=None):
        if reason := self._setup_block():
            return reason
        if self.preview is None:
            return "请先连接相机预览。收到画面后即可检查。"
        states = self.preview.status() if preview_states is None else preview_states
        if not any(info["state"] == "preview" for info in states.values()):
            return "请等相机送出预览画面。可悬停在异常画面上查看连接状态。"
        return ""

    def _preview_quality_target(self, preview, identity, state=None):
        view = self.views.get(identity)
        roi = view.roi if view else None
        if view and (sample := preview.bound_image(identity)):
            binding, frame = sample
            return PreviewQualityTarget(identity, roi, binding, _image_layout(frame))
        if state == PART_REQUIRED:
            reason = "请先选择这台相机的主图像部件。"
        elif state in ("failed", "releasing", "retry_wait", "reconnecting"):
            reason = "这台相机正在重连或连接异常。恢复预览后可重新检查。"
        else:
            reason = "这台相机尚未送出可检查的画面。"
        return PreviewQualityTarget(identity, roi, unavailable=reason)

    def _current_preview_quality(self, rows):
        preview = self.preview
        if preview is None:
            return ()
        selected = set(self.devices.selection())
        return tuple(row for row in rows if row.target.binding is not None and row.target.camera_id in selected
                     and row.target.matches(self._preview_quality_target(preview, row.target.camera_id)))

    def _refresh_preview_quality_window(self, preview_states=None):
        window = self._preview_quality_window
        if window is not None and window.top.winfo_exists():
            current = {row.target.camera_id for row in self._current_preview_quality(window.rows.values())}
            cameras = frozenset(self.views) if self.preview is not None else frozenset()
            window.refresh(current, cameras, self._preview_quality_block(preview_states))
        if (report := getattr(self, "_plan_report_window", None)) is not None and report.winfo_exists():
            rows = getattr(report, "preview_quality_rows", ())
            if rows and not report.readiness_stale and len(self._current_preview_quality(rows)) != len(rows):
                self._mark_plan_report_stale(report, "预览参数或检查框已变化，请重新检查画质和计划。")

    def show_preview_quality(self):
        window = self._preview_quality_window
        if window is not None and window.top.winfo_exists():
            window.top.deiconify()
            window.top.lift()
            return
        if reason := self._preview_quality_block():
            self.activity.set(reason)
            return
        window = self._preview_quality_window = PreviewQualityWindow(
            self.root, self._measure_preview_quality, self._open_preview_quality_camera)
        self._measure_preview_quality()

    def _measure_preview_quality(self):
        window = self._preview_quality_window
        if window.working or self._preview_quality_block():
            return
        after = time.perf_counter_ns()
        preview = self.preview
        states = preview.status()
        targets = tuple(self._preview_quality_target(preview, identity, states.get(identity, {}).get("state"))
                        for identity in self.devices.selection())
        window.begin({target.camera_id: self._view_title(target.camera_id) for target in targets})
        self._refresh_preview_quality_window(states)

        def finished(result):
            if not self._closing and window.top.winfo_exists():
                if self.preview is preview:
                    self._preview_quality_results = result
                    report = getattr(self, "_plan_report_window", None)
                    if (report is not None and report.winfo_exists() and not report.readiness_stale
                            and getattr(report, "preview_quality_rows", ())):
                        self._mark_plan_report_stale(report, "画质复测结果已更新，请重新检查计划。")
                window.finish(result)
                self._refresh_preview_quality_window()

        def failed(error):
            if not self._closing and window.top.winfo_exists():
                window.fail(error)

        self._job(lambda: inspect_preview_quality(preview, targets, after=after, cancelled=window.cancelled),
                  finished, failed)

    def _open_preview_quality_camera(self, identity, report=None):
        if self.preview is None or identity not in self.views:
            self.activity.set("请先连接这台相机的预览。")
            return False
        if report is not None:
            report.destroy()
        self.select_view(identity)
        self.tabs.select(self.quality_panel)
        self.root.lift()
        self.hist.focus_set()
        return True

    def _quality_region_candidate(self):
        if reason := self._setup_block(needs_camera=True):
            return None, reason
        camera = self.active_camera
        view = self.views.get(camera)
        if self.preview is None or not view or camera not in self.devices.selection():
            return None, "先选中相机并连接实时预览。"
        if view.corrected:
            return None, "请切回原始图像后框选。"
        if not view.roi:
            return None, "按住 Shift 拖动框住试样，也可点“输入区域”填写原图像素坐标。"
        sample = self.preview.bound_image(camera)
        if sample is None or view._image is None:
            return None, "等相机收到并显示新画面后再设置。"
        binding, image = sample
        if time.monotonic_ns() - image.received_monotonic_ns > 4_000_000_000:
            return None, "预览较久未更新。等收到新画面后再设置。"
        try:
            region = quality_region_from_image(image, view.roi, binding.readback)
        except CaptureError:
            return None, "检查框超出当前原图。请重新框住试样。"
        if quality_region_mismatch(region, view._image, binding.readback):
            return None, "画幅刚发生变化。等画面更新后重新框选。"
        return region, ""

    def _save_quality_region(self):
        region, reason = self._quality_region_candidate()
        if reason:
            self.activity.set(reason)
            return
        self._set_plan_quality_region(self.active_camera, region, "已保存这台相机的采集抽查范围。")

    def _set_plan_quality_region(self, camera, region, notice):
        self.config.setdefault("quality_regions", {})[camera] = region.to_dict()
        self._quality_region_edited(notice)

    def _clear_quality_region(self):
        if reason := self._setup_block(needs_camera=True):
            self.activity.set(reason)
            return
        if self.config.get("quality_regions", {}).pop(self.active_camera, None) is not None:
            self._quality_region_edited("这台相机已恢复全图抽查。")

    def _quality_region_edited(self, notice):
        self._invalidate_preflight()
        self._schedule_draft_save()
        self._refresh_quality_region_controls()
        self.activity.set(notice)

    def _refresh_quality_region_controls(self):
        _region, reason = self._quality_region_candidate()
        self.quality_region_button.state(["disabled" if reason else "!disabled"])
        saved = self.config.get("quality_regions", {}).get(self.active_camera)
        if saved is None:
            self.quality_region_note.pack_forget()
            self.quality_region_clear.pack_forget()
            return
        if not self.quality_region_clear.winfo_manager():
            self.quality_region_clear.pack(side=tk.LEFT, padx=(5, 0))
            self.quality_region_note.pack(fill=tk.X, pady=(4, 0))
        self.quality_region_clear.state(["disabled" if self._setup_block(needs_camera=True) else "!disabled"])
        note = f"计划抽查 · {quality_region_summary(saved)}"
        if not self.watch_quality.get():
            note += " · 需在计划中开启画质提醒"
        if self.quality_region_note.cget("text") != note:
            self.quality_region_note.configure(text=note)

    def _show_saved_quality_region(self, camera, view, image):
        if self.preview is None or image is None or view.corrected or camera not in self._preview_regions_pending:
            return
        if binding := self.preview.camera_binding(camera):
            region = self._preview_regions_pending.pop(camera)
            if quality_region_mismatch(region, image, binding.readback) is None:
                view.roi = region.bounds
                if camera == self.active_camera:
                    self._update_region_caption()

    def _open_quality_region_camera(self, identity, report):
        report.destroy()
        self._show_views(self.devices.selection(), preferred=identity, browse_saved=False)
        self.tabs.select(self.quality_panel)
        self._refresh_quality_region_controls()
        form = self._quality_scroll
        self.root.update_idletasks()
        focus_canvas_child(self.quality_region_note, form.canvas, form.body)
        self.quality_region_clear.focus_set()

    def _exposure_block(self):
        if reason := self._setup_block():
            return reason
        preview = self.preview
        if preview is None:
            return "曝光建议需要实时画面。请先连接相机预览。"
        parameters = preview.parameters(self.active_camera)
        if parameters is None or preview.latest_image(self.active_camera) is None:
            return "选中已送出实时画面的相机，再测量曝光。"
        return exposure_unavailable(parameters[1])

    def _exposure_scope(self):
        if reason := self._exposure_block():
            raise CaptureError("EXPOSURE_UNAVAILABLE", reason)
        preview, identity = self.preview, self.active_camera
        binding = preview.camera_binding(identity)
        parameters = preview.parameters(identity)
        frame = preview.latest_image(identity)
        if binding is None or parameters is None or frame is None or preview.camera_binding(identity) is not binding:
            raise CaptureError("EXPOSURE_SCOPE_CHANGED", "相机预览已变更。")
        view = self.views[identity]
        rates = tuple(phase["fps"] for phase in self.config["phases"] if phase.get("fps") is not None)
        return ExposureScope(preview, identity, binding, view.roi, _image_layout(frame), parameters[1], parameters[2], rates)

    def show_exposure_advice(self):
        window = getattr(self, "_exposure_window", None)
        if window is not None and window.top.winfo_exists():
            window.top.lift()
            return
        try:
            scope = self._exposure_scope()
        except CaptureError as exc:
            self.activity.set(str(exc))
            return
        self._exposure_window = ExposureWindow(
            self.root, self._exposure_scope, scope, self._view_title, self._job, self._apply_exposure,
        )

    def _apply_exposure(self, scope, value, verify, completed, failed):
        try:
            if not scope.still_matches(self._exposure_scope()):
                raise CaptureError("EXPOSURE_SCOPE_CHANGED", "相机、检查框或计划帧率已经变更。")
            requested = {**scope.settings, "exposure_us": value}
            future = scope.preview.reconfigure(scope.camera_id, requested, expected_binding=scope.binding, verify=verify)
        except CaptureError as exc:
            failed(exc)
            return
        self._camera_settings_busy = True
        self._preview_adjustment = None
        self._sync_quality_scope(None)
        self.activity.set(f"正在复测相机 {scope.camera_id} 的曝光…")

        def finished(result):
            camera, _ = result
            self._set_camera_settings(camera.id, dict(camera.settings))
            self.activity.set(f"相机 {camera.id} 已复测并采用 {value:g} 微秒曝光。")
            completed()

        def failure(exc):
            self._camera_settings_busy = False
            self.activity.set(ui_error(exc, details=False))
            failed(exc)

        self._job(future.result, finished, failure)

    def _focus_help(self, _event=None):
        if self._focus_comparison is not None:
            return "结束本轮比较。再次开始时会从新画面重新记录。"
        return self._focus_block() or (
            "选中需要看清的部位，固定相机、试样、照明、曝光和增益，慢慢转动对焦环。"
            "比较当前和本轮最高响应，并在预览原图上核对细节。")

    def _toggle_focus_comparison(self):
        if self._focus_comparison is not None:
            self._focus_comparison = None
            self._show_focus_feedback("")
            return
        if reason := self._focus_block():
            self._show_focus_feedback(reason)
            return
        if current := self._quality_input():
            scope, image = current
            self._sync_quality_scope(scope)
            self._focus_comparison = FocusComparison(image.received_monotonic_ns)
            self._show_focus_feedback("等待三张新画面…")
            self._quality_due = 0.

    def _tick_quality(self, preview_states):
        current = self._quality_input()
        self._sync_quality_scope(current[0] if current else None)
        button = self.preview_quality_button
        multiple = len(self.devices.selection()) > 1
        if multiple and not button.winfo_manager():
            button.grid()
        elif not multiple and button.winfo_manager():
            button.grid_remove()
        if multiple:
            button.state(["disabled" if self._preview_quality_block(preview_states) else "!disabled"])
        self._refresh_preview_quality_window(preview_states)
        if self.tabs.select() != str(self.quality_panel) or self._board_visible:
            return
        self._refresh_quality_region_controls()
        self.exposure_button.state(["disabled" if self._exposure_block() else "!disabled"])
        reason = self._focus_block(preview_states)
        if reason and self._focus_comparison is not None:
            self._focus_comparison = None
            self._show_focus_feedback("对焦比较已结束，请悬停查看原因。")
        comparison = self._focus_comparison
        self.focus_button.state(["!disabled" if comparison is not None or not reason else "disabled"])
        label = "结束对焦比较" if comparison is not None else "开始对焦比较"
        if self.focus_button.cget("text") != label:
            self.focus_button.configure(text=label)
        now = time.monotonic()
        if current is None or self._quality_work or now < self._quality_due:
            return
        scope, image = current
        if comparison is not None and time.monotonic_ns() - image.received_monotonic_ns > 1_800_000_000:
            self._show_focus_feedback("等收到新画面后再调焦。")
            return
        self._quality_due = now + (.25 if comparison is not None else .65)
        self._quality_work = True
        roi = scope.roi
        serial = self._quality_serial
        adjustment = self._preview_adjustment

        def is_current():
            new = self._quality_input()
            return (new is not None and new[0] == scope and self._quality_serial == serial
                    and self._preview_adjustment is adjustment and not self._closing)

        def finish(result):
            self._quality_work = False
            if is_current():
                if self.preview is not None:
                    self._quality_checked_preview = self.preview
                if comparison is not None and self._focus_comparison is comparison:
                    comparison.update(image.received_monotonic_ns, result["focus_laplacian_variance"])
                    if not comparison.has_detail:
                        self._show_focus_feedback("细节不足，请重选检查框。")
                    elif comparison.current is not None:
                        self._show_focus_feedback(f"当前 {comparison.current:.3g} · 本轮最高 {comparison.highest:.3g}"
                                                  f"（{comparison.current / comparison.highest:.0%}）")
                    else:
                        self._show_focus_feedback("等待三张新画面…")
                self._quality_result(result, roi)

        def failure(exc):
            self._quality_work = False
            if is_current():
                if getattr(exc, "code", "") == "ROI_OUT_OF_BOUNDS":
                    self.quality_text.set("检查框超出当前图像。请重新绘制或清除检查框。")
                    self._focus_comparison = None
                    self._show_focus_feedback("")
                else:
                    self.quality_text.set(ui_error(exc, details=False))

        self._job(lambda: image_quality(image, roi), finish, failure)

    def _quality_result(self, result, roi):
        bins = result["histogram"]
        self.hist.delete("all")
        w, h = self.hist.winfo_width(), self.hist.winfo_height() - 19
        peak = max(1, max(bins))
        for index, count in enumerate(bins):
            self.hist.create_rectangle(index * w / len(bins), h, (index + 1) * w / len(bins),
                                       h - count / peak * (h - 5), fill="#67cdb9", outline="")
        self.hist.create_text(4, h + 9, text="0", anchor="w", fill=MUTED)
        self.hist.create_text(w - 4, h + 9, text=str(result["code_max"]), anchor="e", fill=MUTED)
        lo, med, hi = result["percentiles_p5_p50_p95"]
        response = result["focus_laplacian_variance"]
        focus_text = f"{response:.1f}" if response is not None else "选区过小或格式不支持"
        text = (f"近黑像素   {result['dark_fraction']:.1%}\n"
                f"近饱和像素   {result['bright_fraction']:.1%}\n"
                f"对焦响应   {focus_text}\n"
                f"亮度分位数 P5 / P50 / P95\n{lo:.0f} / {med:.0f} / {hi:.0f}")
        if result["dark_fraction"] >= QUALITY_ALERT_FRACTION and result["bright_fraction"] >= QUALITY_ALERT_FRACTION:
            text += "\n明暗两端像素都较多。可检查杂散反光和照明均匀性。"
        elif result["bright_fraction"] >= QUALITY_ALERT_FRACTION:
            text += "\n较多像素接近饱和。可降低曝光或减弱光照，再查看散斑。"
        elif result["dark_fraction"] >= QUALITY_ALERT_FRACTION:
            text += "\n较多像素接近全黑。可增大照明或检查曝光，再确认运动模糊。"
        if texture := result["texture"]:
            code = texture["failure_code"]
            guidance = TEXTURE_GUIDANCE.get(code, code) if code else "选区通过纹理检查"
            text += (f"\n\n局部纹理\n对比度 (P95−P5)   {texture['contrast_p95_p5']:.1f}"
                     f"\n结构张量特征值比   {texture['structure_tensor_ratio']:.3f}"
                     f"\n{guidance}")
        elif result["texture_error"]:
            text += "\n\n检查框较大。缩小至 100 万像素以内可查看局部纹理指标。"
        adjustment = self._preview_adjustment
        if (adjustment and adjustment["preview"] is self.preview and adjustment["camera"] == self.active_camera
                and adjustment.get("compared_roi", roi) == roi):
            text = f"{adjustment['note']}\n\n{text}"
        self.quality_text.set(text)

    def _update_controls(self, preview_states):
        def enable(widget, available):
            widget.state(["!disabled" if available else "disabled"])

        busy = self._busy()
        reviewing = not busy and self._saved_panel()
        self._render_workspace(reviewing=reviewing, busy=busy)
        unlocked = not self._setup_block()
        camera_idle = unlocked and self.preview is None
        selected = bool(self.devices.selection())
        for widget in (self.name_entry, self.output_entry, self.choose_output_button, self.external_box,
                       self.watch_quality_box, self.template_button, self.storage_mode):
            enable(widget, unlocked)
        enable(self.calibration_menu, unlocked)
        for widget in (*self.setup_buttons, self.refresh_button):
            enable(widget, camera_idle)
        discovered_cameras = bool(self._detected_experiment_cameras())
        if discovered_cameras != bool(self.camera_choice_button.winfo_manager()):
            if discovered_cameras:
                self.camera_choice_button.grid()
            else:
                self.camera_choice_button.grid_remove()
        enable(self.camera_choice_button, camera_idle and discovered_cameras)
        enable(self.cti_button, not self._cti_block())
        enable(self.settings_button, not self._camera_settings_block())
        enable(self.batch_settings_button, not self._batch_settings_block())
        enable(self.preview_button, unlocked and selected)
        save_available = self._preview_save_candidate()[0] is not None
        if save_available == self.save_preview_button.instate(["disabled"]):
            enable(self.save_preview_button, save_available)
        compare_available = not (busy or self._closing)
        if compare_available == self.compare_preview_button.instate(["disabled"]):
            enable(self.compare_preview_button, compare_available)
        compare = self._preview_compare_window
        if compare is not None and compare.top.winfo_exists():
            compare.sync_controls()
        can_retry = self._retry_target(preview_states) is not None
        if can_retry and not self.retry_preview_button.winfo_manager():
            self.retry_preview_button.grid()
        elif not can_retry and self.retry_preview_button.winfo_manager():
            self.retry_preview_button.grid_remove()
        enable(self.retry_preview_button, unlocked and can_retry)
        plan_block = self._plan_input_block(preview_states)
        enabled_plan = not plan_block
        enable(self.record_button, enabled_plan and not reviewing and not self._output_problem())
        enable(self.preflight_button, enabled_plan)
        enable(self.trial_progress_button, enabled_plan or self._trial_batch is not None)
        window = self._trial_progress_window
        if window is not None and window.top.winfo_exists():
            window.set_block(plan_block)
        enable(self.trial_button, enabled_plan and self._selected_phase() is not None)
        enable(self.restore_draft_button, camera_idle and self._stored_draft is not None)
        enable(self.board_button, not self._checkerboard_setup_block())
        for button in self.plan_buttons:
            enable(button, not self._phase_block(button.cget("text")))
        for button in self.review_navigation:
            enable(button, not self._review_help("", needs_camera=True, browsing=True))
        enable(self.frame_search_button, not self._frame_search_block())
        enable(self.timeline_button, not self._review_help(""))
        enable(self.archive_quality_button, not self._archive_quality_block())
        enable(self.history_button, not self._review_help(""))
        enable(self.review_part_menu, not self._analysis_help(""))
        enable(self.trigger_review_button, not self._review_help("", browsing=True))
        for button, camera in self.analysis_buttons:
            enable(button, not (self._analysis_help("") if camera else self._review_help("", needs_idle=True)))
        enable(self.corrected_analysis_button, not self._analysis_help("", needs_idle=True))
        can_review = not self._review_help("", needs_idle=True)
        enable(self.review_report_button, can_review and self._review_integrity is not None)
        enable(self.trial_comparison_button, can_review and self._has_trial_reference)
        enable(self.related_button, can_review)
        enable(self.repeat_button, not self._repeat_block())
        enable(self.pulse_audit_button, can_review)
        enable(self.copy_button, not self._copy_block())
        enable(self.sequence_export_button, not self._sequence_export_block())
        can_fit_board = self._board_scan is not None and not self._fit_block()
        enable(self.recover_button, can_review)
        enable(self.continue_button, can_fit_board if self._board_scan and not self._review_trial else can_review)
        enable(self.board_fit_button, can_fit_board)
        for action, button in self.region_buttons:
            enable(button, not self._region_help(action, ""))
        enable(self.stop_button, busy)
        enable(self.return_to_plan_button, not self._opening_experiment)
        enable(self.trigger_button, self._trigger_phase is not None)
        if self._pausable_phase:
            if not self.pause_button.winfo_manager():
                self.pause_button.grid()
            pause_state = self._phase_pause["state"] if self._phase_pause else None
            label = {"requested": "正在暂停", "paused": "继续采集", "resuming": "正在继续",
                     "stopping": "正在结束"}.get(pause_state, "暂停采集")
            if self.pause_button.cget("text") != label:
                self.pause_button.configure(text=label)
            enable(self.pause_button, self._pause_available or pause_state == "paused")
        elif self.pause_button.winfo_manager():
            self.pause_button.grid_remove()
        enable(self.marker_button, not self._marker_block())
        self.preview_button.configure(text="断开预览" if self.preview else "连接并预览")
        self.record_button.configure(text="开始模拟采集" if self._camera_kind() == "sim" else "开始采集")
        trial_images = getattr(self, "_trial_image_window", None)
        if trial_images is not None and trial_images.top.winfo_exists():
            trial_images.sync_plan_region_action()

    def _update_run_monitor(self, status, active):
        visible = active and status is not None
        if visible != self._monitor_visible:
            self._monitor_visible = visible
            old = self.plan_editor if visible else self._monitor_scroll
            new = self._monitor_scroll if visible else self.plan_editor
            old.grid_remove()
            new.grid()
            self._update_workspace_tabs()
            self._monitor_due = 0.
        if status is None and self.run_monitor.quality_window_open():
            self.run_monitor.update_quality_watch(None)
        if status is not None and (visible or self.run_monitor.write_window_open() or self.run_monitor.pulse_window_open()
                                   or self.run_monitor.quality_window_open()):
            now = time.monotonic()
            if now >= self._monitor_due:
                self._monitor_due = now + .25
                self.run_monitor.update_pulses(status)
                self.run_monitor.update_write_pressure(status)
                self.run_monitor.update_quality_watch(status)
                if visible and self.tabs.select() == str(self.plan_panel):
                    self.run_monitor.render(status, self.active_camera)

    def _tick(self):
        while not self._jobs.empty():
            send, value = self._jobs.get_nowait()
            send(value)
        status = self._displayed_session_status()
        pressure = status.get("write_pressure") if status else None
        notice = pulse_notice(status.get("hardware_groups")) if status else None
        quality = (status or {}).get("image_quality_watch", {})
        alert = tuple((camera, tuple(row["alerts"])) for camera, row in quality.get("cameras", {}).items()
                      if row["phase"] == status["phase"] and row["alerts"]) or None
        if ((pressure or {}).get("state") != (self._write_pressure or {}).get("state") or notice != self._pulse_notice
                or alert != self._image_quality_alert):
            self._monitor_due = 0.
        self._write_pressure = pressure
        self._pulse_notice = notice
        self._image_quality_alert = alert
        progress = status["phase_progress"] if status else None
        self._phase_confirmation = status.get("phase_confirmation") if status else None
        active = self._busy()
        self._phase_pause = status.get("phase_pause") if status else None
        self._pausable_phase = ((status["phase"], status["mode"])
                                if active and progress and not self._review_trial and status["mode"] in {"timed", "manual"}
                                else None)
        self._pause_available = bool(self._pausable_phase and progress.get("pause_available"))
        saved = None
        preview_states = self.preview.status() if self.preview else {}
        if self.preview and not self._releasing_preview:
            revision = self.preview.evidence_revision
            if self._preview_revision is not None and revision != self._preview_revision:
                self._invalidate_preflight(reason="相机连接已变化。请确认新画面并重新检查计划。")
            self._preview_revision = revision
        if self._start_stop is not None:
            self.state.set("OPENING")
        elif status:
            stage = status["state"].upper()
            if active and status["state"] in TERMINAL_STATES:
                stage = "DRAINING"
            if status["error"] and (active or not self._run_reviewed):
                error = status["error"]
                self.activity.set(CAPTURE_ERROR_LABELS.get(error["code"], "采集遇到问题。悬停可查看设备或写入信息。"))
            elif active and (gate := self._phase_confirmation):
                stage, detail = {
                    "saving_previous": ("PHASE SAVING", "等待上一阶段写入台账"),
                    "waiting": ("PHASE WAIT", "尚未拍摄本阶段的图像"),
                    "arming": ("PHASE ARMING", "已收到确认，正在启动相机"),
                    "stopping": ("DRAINING", "本次采集正在结束"),
                }[gate["state"]]
                self.activity.set(f"阶段 {gate['phase']} · {detail}")
            elif active and finishing_phase_writes(status):
                stage = "DRAINING"
                self.activity.set("本阶段收图已结束，正在完成图像写入和台账登记。")
            elif active and progress:
                phase_stage, details = phase_display(status["mode"], progress, pause=self._phase_pause)
                if stage == "RUNNING":
                    stage = phase_stage
                note = ""
                if status["buffered_raw_bytes"] >= .85 * status["raw_buffer_limit_bytes"]:
                    note = "  ·  字节缓冲接近上限"
                elif status["queue_size"] >= .85 * status["queue_capacity"]:
                    note = "  ·  待写队列接近上限"
                trial = self._review_trial
                phase = "标定采图" if self._board_scan else f"试采阶段 {trial['phase_number']}" if trial else f"阶段 {status['phase']}"
                if progress.get("host_sampling_cameras"):
                    phase += " · 主机选帧"
                if csv_watch := progress.get("csv_trigger"):
                    details += f" · {csv_trigger_caption(csv_watch)}"
                self.activity.set(f"{phase}  ·  {details}{note}")
            elif not self._run_reviewed and not active:
                self.activity.set("标定采图已结束。可查看棋盘格检查结果。" if self._board_scan else
                                  "采集已结束。请核对已保存图像。再次采集前请选择新目录。")
            self.state.set(stage)
            samples = status["cameras"]
            saved = sum(value["saved"] for value in samples.values())
            received = sum(value["received"] for value in samples.values())
            pending = sum(value["pending_write"] for value in samples.values())
            losses = reported_camera_losses(samples)
            csv_fault = any(row["state"] == "fault" for row in status.get("csv_triggers", ()))
            event_findings = any(row["partial_events"] or row["busy_signals"] or row["early_signals"]
                                 for row in status.get("event_phases", ()))
            need_review = losses or status["scheduled_misses"] or status["short_pre_windows"] or csv_fault or event_findings
            if not active and stage == "COMPLETED" and need_review:
                self.state.set("CHECK DATA")
                if not self._run_reviewed:
                    self.activity.set("试验机 CSV 监看曾中断。可在时间线查看原因。" if csv_fault else
                                      "请在采集结果中核对触发请求和收图数。" if event_findings else
                                      "请复查漏拍、设备传输或存储差异。")
            free = status["disk_free_bytes"]
            disk = "磁盘空间未知" if free is None else f"可用 {free / 2**30:.1f} GiB"
            used = status["buffered_raw_bytes"] / status["raw_buffer_limit_bytes"]
            review_notice = " · 需核对" if need_review else ""
            self.counter.set(f"已存 {saved}／收到 {received} · 待存 {pending} 帧 · "
                             f"字节缓冲 {used:.0%} · {status['disk_bytes_per_s'] / 2**20:.1f} MiB/s · {disk}{review_notice}")
            if not active and self.control:
                self.control.__exit__(None, None, None)
                self.control = None
            if not active and not self._run_reviewed and not self._closing:
                self._run_reviewed = True
                self._review_state = status["state"].upper()
                self.review_scroll.canvas.yview_moveto(0)
                self.tabs.select(self.review_panel)
                self._read_review_result()
                self.browse_frame(latest=True)
                if self._review_trial:
                    self.show_trial_report()
                elif status["error"]:
                    self.activity.set(f"{self.activity.get()} 在复查页打开“检查中断”可核对遗留图像。")
        elif self.preview:
            errors = [key for key, value in preview_states.items() if value["state"] == "failed"]
            retries = [key for key, value in preview_states.items()
                       if value["state"] in ("releasing", "retry_wait", "reconnecting")]
            unresolved = [key for key, value in preview_states.items() if value["state"] == PART_REQUIRED]
            if not self._checking_plan and not self._releasing_preview and not self._camera_settings_busy:
                if unresolved:
                    camera = unresolved[0]
                    steps = "请在预览下方选择需要保存的图像。" if camera == self.active_camera else "请选中该画面并使用下方列表。"
                    activity = f"{self._view_title(camera)} 等待选择图像部件。{steps}"
                elif errors:
                    activity = f"{self._view_title(errors[0])} 预览失败。请核对连接或悬停查看详情。"
                elif retries:
                    camera = retries[0]
                    activity = f"{self._view_title(camera)} · {PREVIEW_STATES[preview_states[camera]['state']]}"
                else:
                    activity = "　".join(f"{self._view_title(key)}  {PREVIEW_STATES.get(value['state'], '等待图像')}"
                                         for key, value in preview_states.items())
                if self.activity.get() != activity:
                    self.activity.set(activity)
            self.state.set("PREVIEW PART" if unresolved else "PREVIEW FAILED" if errors else
                           "PREVIEW RETRY" if retries else "PREVIEW")
        elif self.experiment and self.tabs.select() == str(self.review_panel):
            self.state.set(self._review_state or "READY")
        else:
            self.state.set("READY")
        if (not active and self.experiment and not self.preview and self.tabs.select() == str(self.review_panel)
                and self._review_state == "COMPLETED"
                and (self._review_frame_error or (self._review_outcome and self._review_outcome.has_incidents)
                     or self._review_issue_count)):
            self.state.set("CHECK DATA")
        stage = self.state.get()
        if status is None and self.counter.get():
            self.counter.set("")
        self.state_caption.set(STATE_LABELS.get(stage, "状态未知"))
        waiting = ("ARMED", "CHECK DATA", "BUFFERING", "TRIGGER QUEUED", "PREVIEW PART", "PREVIEW RETRY",
                   "PHASE SAVING", "PHASE WAIT", "PHASE PAUSING", "PHASE PAUSED", "PHASE RESUMING")
        self.state_label.configure(foreground=BAD if stage in ("FAILED", "PREVIEW FAILED") else (WARN if stage in waiting else GOOD))
        self._trigger_phase = status["phase"] if active and progress and progress["trigger_available"] else None
        window = (progress or {}).get("event_window")
        self._trigger_window = window.get("event_number") if self._trigger_phase is not None and window else None
        if self._board_scan and progress and (progress["queued_signals"] or progress["manual_signal_id"]):
            self._trigger_phase = None
        self._update_quality_panel()
        self._tick_checkerboard(saved, active)
        saved_view = not self.preview and self.experiment is not None and self._saved_panel()
        heading = "预览" if self.preview else "已保存" if saved_view else "监看" if active else "画面"
        if heading == "监看" and self._phase_confirmation:
            heading = "上阶段" if self._phase_confirmation["phase"] > 1 else "待启动"
        if self.preview_heading.cget("text") != heading:
            self.preview_heading.configure(text=heading)
        preview_note = self._preview_save_summary if not active and (self.preview or self._saving_preview_image) else ""
        footer = self.counter.get() or preview_note or "悬停查看说明 · F1 查看焦点 · Shift+F1 固定悬停说明"
        if active and (marker := self.marker_status.get()):
            footer = f"{marker.split('，', 1)[0].rstrip('。')} · {footer}"
        self.footer_detail.set(footer)
        self._update_run_monitor(status, active)
        self._update_part_menu(preview_states)
        self._update_controls(preview_states)
        self._update_next_step(preview_states, status)
        context = (("交接副本" if self._review_is_copy else "实验复查") if self._saved_panel() and not active else
                   {"sim": "模拟练习", "physical": "实验相机", "mixed": "模拟与实验相机"}.get(
                       self._camera_kind(), "正在查找相机" if self._discovering else "请选择相机"))
        if self.camera_context.get() != context:
            self.camera_context.set(context)
        board_cameras = self._board_result["cameras"] if self._board_visible and self._board_result else {}
        hardware_inputs = (progress or {}).get("hardware_inputs", {}) if active else {}
        wait_cameras = ((progress or {}).get("hardware_readiness") or {}).get("cameras", {}) if active else {}
        for camera, view in self.views.items():
            placeholder = "单击查看已保存图像" if saved_view else "等待相机送来图像" if self.preview or active else "连接后查看预览"
            empty_caption = "等待图像"
            if heading in ("上阶段", "待启动"):
                placeholder = "确认后启动本阶段"
            view_state = preview_states.get(camera, {}).get("state")
            if view_state == PART_REQUIRED:
                placeholder = ("在下方选择图像部件" if camera == self.active_camera else
                               "选中此画面后，在下方选择图像部件")
            elif view_state == "adjusting":
                placeholder = "正在更新相机参数…"
            elif view_state in ("failed", "releasing", "retry_wait", "reconnecting"):
                empty_caption = PREVIEW_STATES[view_state]
                placeholder = {
                    "failed": "预览未恢复\n悬停查看原因",
                    "releasing": "画面中断\n正在释放旧连接",
                    "retry_wait": "画面中断\n等待自动重连",
                    "reconnecting": "正在重连\n等待新画面",
                }[view_state]
            elif not saved_view and camera in hardware_inputs and not progress["received_by_camera"][camera]:
                wait = wait_cameras.get(camera)
                hint = WAIT_OUTPUT_LABELS[wait["state"]] if wait else f"计划输入 {hardware_inputs[camera]}"
                placeholder = f"{hint}\n主机未收到首帧"
            from_previous_phase = False
            if active and status and not saved_view and not self.preview:
                sample = self.session.latest_sample(camera)
                image = sample[1] if sample else None
                from_previous_phase = bool(sample and progress and sample[0] < status["phase"])
            else:
                image = self._image(camera)
            self._show_saved_quality_region(camera, view, image)
            self._display_image(view, image, placeholder=placeholder, preview_states=preview_states, empty_caption=empty_caption,
                                frame_caption="上阶段图像" if from_previous_phase else None)
            view.region_editable = not self._board_visible
            auxiliary = self._review_part and self.review_image and image is self.review_image[1]
            view.show_checkerboard(None if auxiliary else board_cameras.get(camera, {}).get("overlay"))
        self._tick_quality(preview_states)
        archive_quality = getattr(self, "_archive_quality_window", None)
        if (self._closing and not active and not self.preview and not self._camera_settings_busy
                and not self._checking_plan and not self._preparing_analysis and not self._draft_saving
                and self._trial_batch is None and not self._saving_preview_image
                and not (archive_quality and archive_quality.overview.exporting)):
            self.root.destroy()
            return
        self.root.after(90, self._tick)

    def close(self):
        if self._closing:
            return
        if self._write_probe_window is not None and self._write_probe_window.working:
            self._write_probe_window.cancel()
            self.root.after(150, self.close)
            return
        if self._busy():
            if not messagebox.askyesno("停止采集？", "停止采集，保存队列中的图像后退出？",
                                       parent=self.root):
                return
            self.stop()
        if self._trial_batch is not None and not messagebox.askyesno(
            "停止连续试采？", "停止本轮试采并保存已收到的图像后退出？", parent=self.root,
        ):
            return
        self._cancel_draft_timer()
        if not self._restoring_draft:
            self._save_draft()
        self._closing = True
        self._stop_trial_batch()
        archive_quality = getattr(self, "_archive_quality_window", None)
        if archive_quality is not None:
            archive_quality.overview.cancel_export()
        self._schedule_output_check()
        self._after_preview(lambda: None)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="straintrace-capture-gui")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--practice", action="store_true", help="open an independent simulated exercise")
    args = parser.parse_args(argv)
    if args.practice:
        if args.config or args.experiment:
            parser.error("--practice starts a new exercise and cannot be combined with --config or --experiment")
        os.environ["EZDIC_STATE_DIR"] = str(_operation_state_dir() / "capture-practice")
    root = create_tk_root()
    CaptureWorkbench(root, CaptureConfig.load(args.config) if args.config else None, args.experiment, practice=args.practice)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
