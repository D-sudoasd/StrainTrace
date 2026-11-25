"""Researcher-editable Tk view of the shared, restartable analysis API.

Only the Tk thread owns widgets. Workers receive plain snapshots and return
data through a queue; this module does not implement DIC or edit project files.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from ezdic_gui.controls import localized_combobox, notebook_help
from ezdic_gui.tooltips import ChoiceToolTip, ToolTip

from ._io import _read_json


OBJECTIVES = {"全场位移与应变": "full_field", "虚拟标距应变": "virtual_gauge", "应力与应变": "stress_strain"}
KINDS = ("统计 ROI", "虚拟标距", "排除区")
GEOMETRY_KEYS = dict(zip(KINDS, ("regions", "gauges", "exclusions_px")))
DRAW_TARGETS = ("计算区域", "统计 ROI", "排除区", "标距 A", "标距 B")
AXIS_LABELS = {"distance": "端点直线距离", "x": "水平距离（x）", "y": "竖直距离（y）"}
COORDINATE_LABELS = {"image": "图像坐标（像素）", "calibrated": "标定坐标"}
PROGRESS_LABELS = {
    "plan_complete": "试算完成，已生成计划",
    "plan_reused": "已读取现有计划",
    "run_started": "开始逐帧分析",
    "frame_reused": "已复用此前保存的结果",
    "frame_started": "正在分析",
    "dic_progress": "正在计算位移与应变",
    "frame_complete": "已完成计算",
    "frame_failed": "该帧计算失败",
    "run_cancelled": "已停止分析",
    "run_finished": "逐帧计算结束，正在读取结果",
}


def _project_file(project, relative):
    """Resolve persisted pointers only inside their owning project."""
    path = (project / relative).resolve()
    path.relative_to(project.resolve())
    return path


def _time_basis(api, config):
    """Resolve supplied timing in a worker using the API's time-basis helper."""
    if hasattr(api, "resolve_time_basis"):
        return api.resolve_time_basis(config)
    if config.get("frame_interval_s") is not None or config.get("loads") or config.get("loads_csv"):
        return "physical_time"
    if not config.get("frame_map_csv"):
        return "frame_index"
    from .discovery import source_files
    image_root = Path(config["image_dir"]).expanduser().resolve()
    mask = Path(config["mask_file"]).expanduser().resolve() if config.get("mask_file") else None
    paths = [path.resolve() for path in source_files(image_root)
             if path.suffix.lower() in api.IMAGE_SUFFIXES and path.resolve() != mask]
    parse_map = getattr(api, "parse_frame_map", None) or getattr(api, "_explicit_frame_map")
    mapping = parse_map(Path(config["frame_map_csv"]).expanduser().resolve(), image_root, paths)
    return "physical_time" if any(row["time_s"] is not None for row in mapping.values()) else "frame_index"


def _snapshot(api, project):
    config = api.read_project_config(project)
    status = api.status(project)
    inventory = _read_json(project / "inventory.json").get("frames", [])
    plan = _read_json(_project_file(project, status["current_plan"])) if status.get("current_plan") else None
    if hasattr(api, "load_reference_frame"):
        record, gray = api.load_reference_frame(config, inventory, plan=plan, allow_untextured_fallback=True)
    elif plan is None:
        # Fallback for minimal test doubles providing private reference hooks.
        auto_ref = getattr(api, "_automatic_reference")
        try:
            record, gray, _selection = auto_ref(config, inventory, allow_untextured_fallback=True)
        except TypeError:
            record, gray, _selection = auto_ref(config, inventory)
    else:
        native = plan.get("reference_native_frame")
        if native is None:
            raise ValueError("当前计划未指定参考原生帧；请重新生成计划。")
        record = getattr(api, "_reference_record")({"reference_native_frame": native}, inventory)
        gray = getattr(api, "_load_image")(record["path"])
    import ezdic_core as core
    import numpy as np
    gray8 = core.normalize_to_uint8(gray)
    image = Image.fromarray(np.asarray(gray8)).convert("RGB")
    run_dir = _project_file(project, status["current_run"]) if status.get("current_run") else None
    artifacts = {}
    if run_dir:
        for name, candidates in {"report": ("report.html", "report/index.html", "report.md"),
                                 "summary": ("summary.csv",), "gauges": ("gauges.csv",)}.items():
            artifacts[name] = next((run_dir / p for p in candidates if (run_dir / p).is_file()), None)
    return dict(project=project, config=config, status=status, inventory=inventory, plan=plan,
                image=image, record=record, run_dir=run_dir, artifacts=artifacts)


class AgentPage(ttk.Frame):
    """Automatic analysis with explicit editable geometry and visible QC."""

    def __init__(self, master, app=None, api=None):
        super().__init__(master, padding=6)
        self.app, self._api = app, api
        self._scheduler = self.winfo_toplevel()
        self._queue = queue.Queue()
        self._worker = None
        self._busy = self._closed = self._loading = self._dirty = self._editor_dirty = False
        self._poll_id = None
        self.project_dir = self.run_dir = None
        self.config, self.plan, self.artifacts = {}, None, {}
        self.items = []
        self._selected = None
        self._image = self._photo = self._record = None
        self._scale, self._offset, self._drag = 1.0, (0, 0), None
        self._discovered_options = {}
        self._geometry_edited = set()
        self._controls = []
        self._build()
        self._poll_id = self._scheduler.after(60, self._poll)

    def _var(self, value="", *, editor=False):
        variable = tk.StringVar(self, value=value)
        variable.trace_add("write", lambda *_: self._mark_dirty(editor))
        return variable

    def _help(self, widget, text, choices=None, *, title=None):
        def contextual(event):
            description = text(event) if callable(text) else text
            if isinstance(widget, ttk.Widget) and widget.instate(["disabled"]):
                reason = self._disabled_help(widget)
                if reason:
                    return f"{description}\n暂不可用：{reason}"
            return description

        if self.app is not None:
            self.app.add_tooltip(widget, contextual, choices, title=title)
        elif choices is None:
            ToolTip(widget, contextual, title=title)
        else:
            ChoiceToolTip(widget, contextual, choices, title=title)
        return widget

    def _button(self, parent, text, command, help_text, row, column, *, style="Compact.TButton", **kwargs):
        widget = ttk.Button(parent, text=text, command=command, style=style)
        widget.grid(row=row, column=column, sticky="ew", padx=3, pady=3, **kwargs)
        self._help(widget, help_text)
        self._controls.append((widget, "normal"))
        return widget

    def _entry(self, parent, label, variable, row, help_text):
        caption = ttk.Label(parent, text=label)
        caption.grid(row=row, column=0, sticky="w", padx=3, pady=3)
        widget = ttk.Entry(parent, textvariable=variable, width=18)
        widget.grid(row=row, column=1, sticky="ew", padx=3, pady=3)
        self._help(caption, help_text)
        self._help(widget, help_text, title=label)
        self._controls.append((widget, "normal"))
        return widget

    def _combo(self, parent, variable, choices, row, column, title, help_text, *, labels=None):
        if labels is None:
            widget = ttk.Combobox(parent, textvariable=variable, values=tuple(choices), state="readonly", width=15)
        else:
            widget = localized_combobox(parent, variable, labels, width=15)
            choices = {labels[value]: detail for value, detail in choices.items()}
        widget.grid(row=row, column=column, sticky="ew", padx=3, pady=3)
        self._help(widget, help_text, choices, title=title)
        self._controls.append((widget, "readonly"))
        return widget

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.project_var = self._var()
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)
        project_label = ttk.Label(bar, text="项目目录", style="Key.TLabel")
        project_label.grid(row=0, column=0, sticky="w", padx=3)
        self.project_entry = ttk.Entry(bar, textvariable=self.project_var, width=8)
        self.project_entry.grid(row=0, column=1, sticky="ew", padx=3)
        project_help = "项目设置、检查点和结果的保存位置。新建时选原始图像目录以外的空文件夹。"
        self._help(project_label, project_help)
        self._help(self.project_entry, project_help, title="项目目录")
        self._controls.append((self.project_entry, "normal"))
        self.open_button = self._button(bar, "打开项目", self.open_dialog, "打开已有自动分析项目，恢复配置、计划和结果。", 0, 2)
        self._button(bar, "选择新目录", self.choose_project, "为新项目选择一个空文件夹。原始图像仍从实验目录读取。", 0, 3)

        self.tabs = ttk.Notebook(self, style="Detail.TNotebook")
        self.tabs.grid(row=1, column=0, sticky="nsew", pady=5)
        self._help(self.tabs, lambda event: notebook_help(self.tabs, (
            "选择保存原始图像的实验目录，再点“发现输入”查看可用的图片、时间和载荷信息。",
            "检查或编辑用于统计应变的区域、排除区和标距端点。可在图像窗口拖出矩形。",
            "查看试算质量，运行后打开报告、逐帧表格或结果目录。",
        ), event))
        self.input_page = self._scroll_page("输入与参数")
        self.geometry_page = self._scroll_page("ROI 与标距")
        self.results_page = self._scroll_page("计划与结果")
        self._build_input()
        self._build_geometry()
        self._build_results()
        for page in (self.input_page, self.geometry_page, self.results_page):
            self._bind_scrolling(page, page._canvas)
        self.apply_theme()
        footer = ttk.Frame(self)
        footer.grid(row=2, column=0, sticky="ew")
        for col in range(4):
            footer.columnconfigure(col, weight=1)
        self.save_button = self._button(footer, "保存设置", self.save_project, "保存参数和已应用到列表的 ROI/标距。旧计算结果保留；需重新生成计划。", 0, 0)
        self.plan_button = self._button(footer, "生成计划", self.make_plan, "保存当前设置，试算有限参数候选；计算区域和候选标距仍可人工修改。", 0, 1)
        self.run_button = self._button(footer, "运行 / 恢复", self.run_project, "按当前计划计算。输入和设置相同时，可从已保存的进度继续。", 0, 2, style="Primary.TButton")
        self.cancel_button = ttk.Button(footer, text="停止", command=self.cancel, state="disabled", style="Compact.TButton")
        self.cancel_button.grid(row=0, column=3, sticky="ew", padx=3, pady=3)
        self._help(self.cancel_button, "请求停止逐帧计算；在安全检查点结束后停止，已保存检查点保留。")
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=4, sticky="ew", padx=3, pady=3)
        self._help(self.progress, "运行当前任务时显示进度。某些准备和核验操作没有可预估的完成比例。")
        self.status_var = tk.StringVar(self, value="处理新数据时，先选择实验目录并发现输入。已有项目可直接打开。")
        self.status_label = ttk.Label(footer, textvariable=self.status_var, style="NextStep.TLabel",
                                      wraplength=500, justify="left")
        self.status_label.grid(row=2, column=0, columnspan=4, sticky="ew", padx=3)
        self.bind("<Configure>", lambda event: self.status_label.configure(wraplength=max(220, event.width - 34)))
        self._help(self.status_label, "显示当前操作结果和下一步。详细输入信息和运行记录可在上方对应页面查看。")
        self._update_actions()

    def _scroll_page(self, title):
        outer = ttk.Frame(self.tabs)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        canvas = tk.Canvas(outer, width=360, height=180, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        content = ttk.Frame(canvas, padding=4)
        content.columnconfigure(1, weight=1)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        content.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        self._help(canvas, f"{title}可上下滚动；Tab / Shift+Tab 自动显示焦点控件，Alt+PageUp / PageDown 翻页。底部操作和状态保持可见。")
        self._help(scrollbar, "拖动滚动条查看此页其余设置。")
        self.tabs.add(outer, text=title)
        content._outer, content._canvas, content._canvas_window = outer, canvas, window
        return content

    def _bind_scrolling(self, widget, canvas):
        def reveal(_event):
            widget.after_idle(lambda: self._reveal_control(widget, canvas))
        for event in ("<FocusIn>", "<<TraverseIn>>"):
            widget.bind(event, reveal, add="+")
        for event, direction in (("<Alt-Prior>", -1), ("<Alt-Next>", 1)):
            widget.bind(event, lambda _event, d=direction: canvas.yview_scroll(d, "pages") or "break", add="+")
        widget.bind("<Alt-Home>", lambda _event: canvas.yview_moveto(0) or "break", add="+")
        widget.bind("<Alt-End>", lambda _event: canvas.yview_moveto(1) or "break", add="+")
        if not isinstance(widget, (tk.Text, ttk.Treeview)):
            def scroll(event):
                delta = getattr(event, "delta", 0)
                amount = (-1 if delta > 0 else 1) if delta else (-1 if getattr(event, "num", 0) == 4 else 1)
                canvas.yview_scroll(amount * 3, "units")
                return "break"
            for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(event, scroll)
        for child in widget.winfo_children():
            self._bind_scrolling(child, canvas)

    @staticmethod
    def _reveal_control(widget, canvas):
        if not widget.winfo_exists() or not canvas.winfo_exists():
            return
        top = canvas.canvasy(0)
        height = canvas.winfo_height()
        control_top, ancestor = 0, widget
        while ancestor.master is not canvas:
            control_top += ancestor.winfo_y()
            ancestor = ancestor.master
        control_top += canvas.coords(ancestor._canvas_window)[1]
        control_bottom = control_top + widget.winfo_height()
        target = None
        if control_top < top:
            target = control_top - 4
        elif control_bottom > top + height:
            target = control_top - 4 if widget.winfo_height() > height else control_bottom - height + 4
        bounds = canvas.bbox("all")
        if target is not None and bounds and bounds[3] > bounds[1]:
            canvas.yview_moveto(max(0, target - bounds[1]) / (bounds[3] - bounds[1]))

    def apply_theme(self, dark: bool | None = None):
        if self.app is not None:
            for page in (self.input_page, self.geometry_page, self.results_page):
                page._canvas.configure(background=self.app.card_bg)
            for text in (self.discovery_text, self.result_text):
                text.configure(background=self.app.panel_bg, foreground=self.app.text_color, insertbackground=self.app.text_color)
            canvas = getattr(self, "image_canvas", None)
            if canvas is not None:
                canvas_bg = getattr(self.app, "canvas_bg", None) or ("#0f172a" if dark else "#111827")
                canvas.configure(background=canvas_bg)

    def _build_input(self):
        p = self.input_page
        self.source_var = self._var()
        self.source_var.trace_add("write", self._source_changed)
        self.interval_var, self.map_var = self._var(), self._var()
        self.reference_var, self.step_var, self.mask_var = self._var(), self._var(), self._var()
        self.reference_var.trace_add("write", lambda *_: self._show_geometry_reference())
        self.analysis_roi_var = self._var()
        self.objective_var = self._var(next(iter(OBJECTIVES)))
        self._entry(p, "实验 / 图像目录", self.source_var, 0, "选择实验目录，可自动发现图像序列和 experiment.json；有多个序列时需选定一组。")
        self._button(p, "选择目录", self.choose_source, "选择原始实验或图像文件夹。", 1, 0)
        self.discover_button = self._button(p, "发现输入", self.discover, "只读检查图像、时间来源和实验元数据；缺少信息时明确显示需要填写的字段。", 1, 1)
        ttk.Label(p, text="分析目标").grid(row=2, column=0, sticky="w", padx=3)
        self._combo(p, self.objective_var, {
            "全场位移与应变": "得到整个计算区域的位移和应变图。只有图像时，可按帧号查看。",
            "虚拟标距应变": "追踪两端点的距离变化。请检查自动候选，或自行添加一条标距。",
            "应力与应变": "把应变与载荷换算得到的应力配对。需要可信时间、载荷和已确认的试样截面积。",
        }, 2, 1, "分析目标", "按需要的结果选择。点击“发现输入”后会列出需要补充的数据。")
        self.create_button = self._button(p, "新建项目", self.create_project, "建立项目并记录图像来源。随后可编辑测量区域和标距，再生成计划。", 3, 0)
        self.auto_button = self._button(p, "自动建项并分析", self.auto_analyze, "从新项目直接试算并运行。必要输入不足或试算未通过时会停下并提示原因。", 3, 1)
        ttk.Label(p, text="可选参数", style="StepTitle.TLabel").grid(
            row=4, column=0, columnspan=2, padx=3, pady=(8, 3), sticky="w")
        self._entry(p, "帧间隔（秒）", self.interval_var, 5, "填写相邻原生帧号之间的采集时间间隔，单位秒。已有带时间的帧映射或只按帧号分析时可留空。")
        self._entry(p, "帧映射 CSV", self.map_var, 6, "可选：包含图像文件、原生帧号和时间的显式映射表。")
        self._button(p, "选择映射", lambda: self._choose_file(self.map_var), "选择带原生帧身份和时间信息的 CSV。", 7, 0)
        self._button(p, "使用控制台路径", self.use_legacy_source, "复制手动分析已设置的图像路径。原来的测量区域和结果保留在手动分析页。", 7, 1)
        self._entry(p, "参考原生帧号", self.reference_var, 8, "填写文件中的原生帧号，或留空自动选择。改变参考帧后请重新检查所有测量区域与端点。")
        self._entry(p, "网格步长（像素）", self.step_var, 9, "填写相邻测量点的间距，如 5 像素。步长越小，测量点越密。留空由试算选择。")
        self._entry(p, "计算区域（像素）", self.analysis_roi_var, 10, "填写 x,y,宽,高，或在图像窗口拖出矩形。留空使用自动候选。")
        self._entry(p, "试样遮罩文件", self.mask_var, 11, "可选遮罩。选择与参考图同尺寸的图片，白色表示保留。也可在“ROI 与标距”逐个添加排除区。")
        self._button(p, "选择遮罩", lambda: self._choose_file(self.mask_var), "选择与参考图像同尺寸的遮罩。", 12, 0)
        self.preview_button = self._button(p, "图像中编辑 ↗", self.show_image, "在独立图像窗口查看参考图，绘制计算区域、统计 ROI、排除区或标距端点。", 12, 1)
        self.discovery_text = tk.Text(p, height=6, width=24, wrap="word", state="disabled")
        self.discovery_text.grid(row=13, column=0, columnspan=2, sticky="ew", padx=3, pady=4)
        self._help(self.discovery_text, "输入发现结果与所需补充信息。文字可选取复制，原始数据不会被修改。")

    def _build_geometry(self):
        p = self.geometry_page
        self.geometry_tree = ttk.Treeview(p, columns=("kind", "name"), show="headings", height=5, selectmode="browse")
        self.geometry_tree.heading("kind", text="类型")
        self.geometry_tree.heading("name", text="名称 / 方向")
        self.geometry_tree.column("kind", width=100, stretch=False)
        self.geometry_tree.column("name", width=180)
        self.geometry_tree.grid(row=0, column=0, columnspan=2, sticky="ew", padx=3)
        self.geometry_tree.bind("<<TreeviewSelect>>", self._select_item)
        self._help(self.geometry_tree, self._geometry_table_help, title="测量区域与标距")
        self.kind_var, self.name_var = self._var(KINDS[0], editor=True), self._var(editor=True)
        self.a_var, self.b_var, self.axis_var = self._var(editor=True), self._var(editor=True), self._var("distance", editor=True)
        self.coordinate_var = self._var("image", editor=True)
        self.fraction_var, self.count_var = self._var("0.5", editor=True), self._var("3", editor=True)
        self._combo(p, self.kind_var, {
            "统计 ROI": "在指定区域内汇总有效应变点。结果按区域名称导出。",
            "虚拟标距": "指定两个端点，计算它们之间的距离或投影距离变化。",
            "排除区": "圈出夹具、反光或缺少纹理的位置，计算时跳过该区域。",
        }, 1, 0, "测量类型", "选择要编辑的类型。切换后按下方格式填写坐标，再点“应用到列表”。")
        self._button(p, "新建一项", self.new_item, "清空当前编辑器，新建独立统计 ROI、标距或排除区。", 1, 1)
        self._entry(p, "名称", self.name_var, 2, "ROI 和标距名称应唯一；名称会保留在导出结果中。")
        self._entry(p, "区域 / 端点 A", self.a_var, 3, "ROI、排除区：x,y,宽,高。标距端点：x,y 点，或 x,y,宽,高矩形。图像坐标单位 px。")
        self._entry(p, "标距端点 B", self.b_var, 4, "仅标距使用；格式与端点 A 相同。两个端点在固定参考图中定义。")
        ttk.Label(p, text="标距方向").grid(row=5, column=0, sticky="w", padx=3)
        self.axis_box = self._combo(p, self.axis_var, {
            "distance": "使用两个端点之间的直线长度，适合标距与图像横竖轴不平行时。",
            "x": "只计算端点在图像水平方向的间距变化。",
            "y": "只计算端点在图像竖直方向的间距变化。",
        }, 5, 1, "标距方向", "为当前标距选择用什么距离计算应变。", labels=AXIS_LABELS)
        ttk.Label(p, text="统计 ROI 坐标").grid(row=6, column=0, sticky="w", padx=3)
        self._combo(p, self.coordinate_var, {
            "image": "以原图左上角为原点，用像素记录位置和大小。可在图像窗口直接画出。",
            "calibrated": "使用已保存标定定义的坐标。请在输入框填写对应标定坐标，图像窗口只支持像素绘制。",
        }, 6, 1, "统计区域的坐标", "选择用原图像素还是实验标定后的坐标填写统计区域。", labels=COORDINATE_LABELS)
        self._entry(p, "ROI 最低有效比例", self.fraction_var, 7, "统计 ROI 内有效应变点比例阈值，范围 0–1。")
        self._entry(p, "ROI 最低有效点数", self.count_var, 8, "统计 ROI 至少需多少有效测量点；不满足则保留质量失败状态。")
        self.apply_button = self._button(p, "应用到列表", self.apply_item, "把当前坐标和名称保存到编辑列表；此时尚未写入项目。", 9, 0)
        self.delete_button = self._button(p, "删除选中", self.delete_item, "删除编辑列表中选中的一项；保存项目后生效，已有运行文件保留。", 9, 1)
        self._button(p, "载入候选标距", self.use_candidate_gauges, "将当前计划的自动标距候选加入编辑列表；候选未确认，仍需检查位置与方向。", 10, 0)
        self._button(p, "图像中编辑 ↗", self.show_image, "在参考图像上直接拖动矩形后，点应用到列表。", 10, 1)
        self.geometry_hint = tk.StringVar(self, value="先选择类型，填写坐标或在图像中画框，再应用到列表。")
        ttk.Label(p, textvariable=self.geometry_hint, wraplength=390).grid(row=11, column=0, columnspan=2, sticky="ew", padx=3, pady=5)
        self.geometry_reference_var = tk.StringVar(self, value="打开项目后显示坐标对应的参考原生帧。")
        ttk.Label(p, textvariable=self.geometry_reference_var, wraplength=390).grid(row=12, column=0, columnspan=2, sticky="ew", padx=3, pady=3)

    def _build_results(self):
        p = self.results_page
        self.plan_summary_var = tk.StringVar(self, value="尚无计划。生成后检查参考帧、计算区域和试算质量。")
        ttk.Label(p, textvariable=self.plan_summary_var, wraplength=390, justify="left").grid(row=0, column=0, columnspan=2, sticky="ew", padx=3, pady=5)
        self._button(p, "采用候选计算区", self.use_candidate_roi, "把计划中的计算矩形填入输入设置，供人工修改；之后须保存并重新生成计划。", 1, 0)
        self.verify_button = self._button(p, "核验结果", self.verify_project, "后台核验来源、计划、检查点和产物完整性；完整性与数值质量分别显示。", 1, 1)
        self.report_button = self._button(p, "打开分析报告", lambda: self.open_artifact("report"), "打开当前运行的 HTML 报告；没有 HTML 时打开 Markdown 摘要。", 2, 0)
        self.folder_button = self._button(p, "打开结果目录", lambda: self.open_artifact("folder"), "打开当前运行目录，查看表格、场图、检查点与报告。", 2, 1)
        self.summary_button = self._button(p, "打开逐帧表格", lambda: self.open_artifact("summary"), "打开 summary.csv；无效测量保持空值，数值质量不等同实验验收。", 3, 0)
        self.gauges_button = self._button(p, "打开标距表格", lambda: self.open_artifact("gauges"), "打开每条标距的端点、应变与质量状态。", 3, 1)
        self.reprocess_button = self._button(p, "按新 ROI / 标距提取", self.reprocess, "只从已保存位移场提取当前列表中的 ROI / 标距，不重算 DIC。另存一份提取结果和报告；原项目设置与结果保留。", 4, 0, columnspan=2)
        self.result_text = tk.Text(p, height=15, width=24, wrap="word", state="disabled")
        self.result_text.grid(row=5, column=0, columnspan=2, sticky="ew", padx=3, pady=5)
        self._help(self.result_text, "查看候选试算、运行状态、失败原因和完整性核验。可选取复制。")

    def attach_visual(self, notebook):
        self.visual_page = ttk.Frame(notebook, padding=6)
        self.visual_page.columnconfigure(0, weight=1)
        self.visual_page.rowconfigure(1, weight=1)
        bar = ttk.Frame(self.visual_page)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="绘制目标").grid(row=0, column=0, padx=4)
        self.draw_var = tk.StringVar(self, value=DRAW_TARGETS[0])
        self._combo(bar, self.draw_var, dict(zip(DRAW_TARGETS, (
            "画出供位移和应变计算的矩形范围。画完后保存设置，再生成计划。",
            "画出要统计应变的区域。返回编辑器命名，再点“应用到列表”。",
            "画出不参与计算的位置，如夹具或反光。返回编辑器点“应用到列表”。",
            "画出第一端的测量窗口。再绘制标距 B，并在编辑器应用。",
            "画出第二端的测量窗口。检查两个端点和标距方向，再在编辑器应用。",
        ))), 0, 1, "绘制目标", "按住左键在参考图上拖出矩形，松开后把坐标填入对应字段。")
        self._button(bar, "返回编辑器", self.show_editor, "回到控制台 ROI 与标距编辑页，调整名称、方向与坐标。", 0, 2)
        self.image_canvas = tk.Canvas(self.visual_page, background="#111827", highlightthickness=0, cursor="crosshair")
        self.image_canvas.grid(row=1, column=0, sticky="nsew", pady=5)
        self.image_canvas.bind("<Configure>", lambda _: self._render_image())
        self.image_canvas.bind("<ButtonPress-1>", self._draw_start)
        self.image_canvas.bind("<B1-Motion>", self._draw_motion)
        self.image_canvas.bind("<ButtonRelease-1>", self._draw_end)
        self._help(self.image_canvas, lambda _event: (
            f"当前绘制：{self.draw_var.get()}。按住左键拖出矩形。绿色为计算区、青色为统计区、橙色为标距、红色为排除区。"
            if self._image is not None else "在控制台新建或打开项目后，这里会显示项目的参考图片。"
        ), title="参考图片")
        self.image_info_var = tk.StringVar(self, value="打开或建立项目后显示参考图像。")
        self.image_info_label = ttk.Label(self.visual_page, textvariable=self.image_info_var, wraplength=700)
        self.image_info_label.grid(row=2, column=0, sticky="ew")
        self.visual_page.bind("<Configure>", lambda e: self.image_info_label.configure(wraplength=max(200, e.width - 20)))
        notebook.add(self.visual_page, text="自动分析图像")
        self.apply_theme()

    def _mark_dirty(self, editor=False):
        if self._loading:
            return
        if editor:
            self._editor_dirty = True
        else:
            self._dirty = True
        if hasattr(self, "run_button"):
            self._update_actions()

    @staticmethod
    def _numbers(text, sizes, *, integer=False):
        values = [float(v.strip()) for v in text.replace("，", ",").split(",") if v.strip()]
        if len(values) not in sizes or not all(math.isfinite(v) for v in values):
            raise ValueError("坐标需以逗号分隔：点为 x,y；矩形为 x,y,宽,高。")
        if integer and any(v != int(v) for v in values):
            raise ValueError("图像矩形坐标必须为整数像素。")
        if len(values) == 4 and (values[2] <= 0 or values[3] <= 0):
            raise ValueError("矩形宽和高必须大于零。")
        return [int(v) for v in values] if integer else values

    def _init_options(self):
        options = copy.deepcopy(self._discovered_options)
        # Discovery's inferred axis is not experimental metadata. Re-evaluate it
        # when the user changes or removes the interval / mapping inputs.
        options.pop("time_basis", None)
        options["image_dir"] = self.source_var.get().strip()
        options["frame_interval_s"] = float(self.interval_var.get()) if self.interval_var.get().strip() else None
        options["frame_map_csv"] = self.map_var.get().strip() or None
        options.update(self._changes())
        if options["frame_interval_s"] is not None:
            options["time_basis"] = "physical_time"
        return options

    def _source_changed(self, *_args):
        if not self._loading:
            # Directly typing a path must have the same semantics as browsing
            # for it: hidden load channels / calibration belong to one source.
            self._discovered_options = {}

    def _discovery_options(self, options):
        # Initial blanks leave experiment.json available for discovery. Once a
        # field has been discovered or geometry explicitly edited, an empty
        # value is an intentional override, not permission to restore metadata.
        explicit = set(self._discovered_options) | self._geometry_edited
        return {key: value for key, value in options.items()
                if (value is not None and value != []) or key in explicit}

    def _changes(self):
        if self._editor_dirty:
            raise ValueError("ROI / 标距编辑尚未应用。请先点“应用到列表”。")
        regions = [copy.deepcopy(item["value"]) for item in self.items if item["kind"] == KINDS[0]]
        gauges = [copy.deepcopy(item["value"]) for item in self.items if item["kind"] == KINDS[1]]
        exclusions = [copy.deepcopy(item["value"]) for item in self.items if item["kind"] == KINDS[2]]
        return dict(reference_native_frame=int(self.reference_var.get()) if self.reference_var.get().strip() else None,
                    step_hint_px=int(self.step_var.get()) if self.step_var.get().strip() else None,
                    roi_override_px=self._numbers(self.analysis_roi_var.get(), (4,), integer=True) if self.analysis_roi_var.get().strip() else None,
                    mask_file=self.mask_var.get().strip() or None, regions=regions, gauges=gauges, exclusions_px=exclusions)

    def _project_changes(self):
        if self.config:
            source_text = self.source_var.get().strip()
            if not source_text or Path(source_text).expanduser().resolve() != Path(self.config["image_dir"]).expanduser().resolve():
                raise ValueError("当前项目的原始图像目录固定。分析其他序列请新建项目。")
        changes = self._changes()
        if self._reference_changed() and changes["roi_override_px"] != self.config.get("roi_override_px"):
            raise ValueError("参考帧已修改；请先还原计算区域并保存设置，显示新参考图后再编辑坐标。")
        changes["frame_interval_s"] = float(self.interval_var.get()) if self.interval_var.get().strip() else None
        changes["frame_map_csv"] = self.map_var.get().strip() or None
        return changes

    def _get_project(self):
        if not self.project_dir:
            raise ValueError("请先新建或打开自动分析项目。")
        if not self.project_var.get().strip() or Path(self.project_var.get().strip()).expanduser().resolve() != self.project_dir.resolve():
            raise ValueError("项目目录已修改；请先打开或新建该目录中的项目，再操作项目设置或结果。")
        return self.project_dir

    def _reference_changed(self):
        reference = self.reference_var.get().strip()
        try:
            value = int(reference) if reference else None
        except ValueError:
            return True
        return value != self.config.get("reference_native_frame")

    def _show_geometry_reference(self):
        if self._loading or not hasattr(self, "geometry_reference_var"):
            return
        self._drag = None
        if self._reference_changed():
            self.geometry_reference_var.set("参考帧设置已修改；先保存设置显示新参考图，再绘制或应用坐标。")
        elif self._record is not None:
            self.geometry_reference_var.set(f"当前坐标参考原生帧 {self._record['native_frame']}；请按此图像填写坐标。")

    def _require_geometry_reference(self):
        self._get_project()
        if self._reference_changed():
            raise ValueError("参考帧已修改；请先保存设置以显示新参考图，再编辑坐标。")
        if self._image is None or self._record is None:
            raise ValueError("请先打开项目并显示参考图，再编辑坐标。")

    def _guard(self, action):
        try:
            action()
        except Exception as exc:
            self._error(exc)

    def _error(self, exc):
        code = getattr(exc, "code", "")
        self.status_var.set(f"操作未完成：{code + ' · ' if code else ''}{exc}")
        self._text(self.result_text, self.status_var.get())
        self.tabs.select(self.results_page._outer)

    def choose_source(self):
        path = filedialog.askdirectory(parent=self, title="选择实验 / 图像目录")
        if path:
            self._discovered_options = {}
            self.source_var.set(path)

    def choose_project(self):
        path = filedialog.askdirectory(parent=self, title="选择空项目目录", mustexist=False)
        if path:
            self.project_var.set(path)

    def _choose_file(self, variable):
        path = filedialog.askopenfilename(parent=self)
        if path:
            variable.set(path)

    def use_legacy_source(self):
        if self.app is not None:
            self._discovered_options = {}
            self.source_var.set(self.app.image_folder.get())

    def open_dialog(self):
        path = filedialog.askdirectory(parent=self, title="打开自动分析项目")
        if path:
            self.open_project(path)

    def open_project(self, path):
        def start():
            project = Path(path).expanduser().resolve()
            self._launch("读取项目", lambda api, emit: _snapshot(api, project), self._accept_snapshot)
        self._guard(start)

    def discover(self):
        def start():
            folder = self.source_var.get().strip()
            if not folder:
                raise ValueError("请先选择实验或图像目录。")
            options = self._init_options()
            # Discovery should be free to locate a nested sequence.
            options.pop("image_dir", None)
            options = self._discovery_options(options)
            objective = OBJECTIVES[self.objective_var.get()]
            self._launch("发现实验输入", lambda api, emit: api.discover_experiment(folder, objective=objective, options=options), self._accept_discovery)
        self._guard(start)

    def _accept_discovery(self, result):
        self._discovered_options = copy.deepcopy(result.get("init_options") or {})
        self._loading = True
        try:
            self._fill_variables(self._discovered_options)
            self.items = [{"kind": kind, "value": copy.deepcopy(value)} for kind, key in zip(KINDS, ("regions", "gauges", "exclusions_px")) for value in self._discovered_options.get(key, [])]
            self._refresh_tree()
            self.new_item()
        finally:
            self._loading = False
        missing = result.get("required_inputs", [])
        self.status_var.set("输入可用于建立项目；可继续编辑设置。" if result.get("status") == "ready" else "需要补充输入：" + "；".join(f"{v.get('field')}: {v.get('reason')}" for v in missing))
        self._text(self.discovery_text, result)
        self._dirty = True
        self._update_actions()

    def create_project(self):
        def start():
            path = self.project_var.get().strip()
            if not path:
                raise ValueError("请填写新项目目录。")
            project, options = Path(path).expanduser().resolve(), self._init_options()
            image_dir = options.pop("image_dir")
            if not image_dir:
                raise ValueError("请先选择实验或图像目录。")
            def operation(api, emit):
                options["time_basis"] = _time_basis(api, {**options, "image_dir": image_dir})
                api.init_project(project, image_dir, **options)
                return _snapshot(api, project)
            self._launch("建立项目与图像索引", operation, self._accept_snapshot)
        self._guard(start)

    def auto_analyze(self):
        def start():
            path = self.project_var.get().strip()
            if not path:
                raise ValueError("请填写新项目目录。")
            project, options = Path(path).expanduser().resolve(), self._init_options()
            folder, objective = self.source_var.get().strip(), OBJECTIVES[self.objective_var.get()]
            if not folder:
                raise ValueError("请先选择实验或图像目录。")
            # Allow discovery to choose a nested sequence unless one was found.
            if not self._discovered_options.get("image_dir"):
                options.pop("image_dir", None)
            options = self._discovery_options(options)
            def operation(api, emit):
                try:
                    result = api.auto_analyze(folder, project, objective=objective, options=options, progress=emit)
                except Exception as exc:
                    if not (project / "project.json").is_file():
                        raise
                    result = {"status": "failed", "error": f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"}
                return result, _snapshot(api, project) if (project / "project.json").is_file() else None
            self._launch("自动建项并分析", operation, self._accept_run, cancellable=project)
        self._guard(start)

    def save_project(self):
        def start():
            project, changes = self._get_project(), self._project_changes()
            config = copy.deepcopy(self.config)
            def operation(api, emit):
                changes["time_basis"] = _time_basis(api, {**config, **changes})
                api.update_project(project, changes)
                return _snapshot(api, project)
            self._launch("保存项目设置", operation, self._accept_snapshot)
        self._guard(start)

    def make_plan(self):
        def start():
            project, changes = self._get_project(), self._project_changes()
            config = copy.deepcopy(self.config)
            def operation(api, emit):
                changes["time_basis"] = _time_basis(api, {**config, **changes})
                api.update_project(project, changes)
                api.plan_project(project, progress=emit)
                return _snapshot(api, project)
            self._launch("保存设置并生成计划", operation, self._accept_snapshot)
        self._guard(start)

    def run_project(self):
        def start():
            project = self._get_project()
            if self._dirty or self._editor_dirty:
                raise ValueError("设置有未保存修改。请先生成新计划。")
            if not self.plan or self.plan.get("status") != "ready":
                raise ValueError("当前计划不可运行。请检查候选试算和 ROI 后重新生成计划。")
            def operation(api, emit):
                result = api.run(project, progress=emit)
                return result, _snapshot(api, project)
            self._launch("逐帧分析 / 恢复", operation, self._accept_run, cancellable=project)
        self._guard(start)

    def verify_project(self):
        self._guard(lambda: self._launch("核验来源与结果", lambda api, emit, project=self._get_project(): api.verify(project), self._accept_verification))

    def reprocess(self):
        def start():
            project, changes = self._get_project(), self._project_changes()
            for key, value in changes.items():
                if key not in ("regions", "gauges") and value != self.config.get(key):
                    raise ValueError("计算区域、参考帧或参数有修改；请还原这些设置，或生成计划重算。只修改统计 ROI / 标距可提取已存场。")
            self._launch("从已存场提取 ROI / 标距", lambda api, emit: api.reprocess(project, regions=changes["regions"], gauges=changes["gauges"]), self._accept_reprocess)
        self._guard(start)

    def _accept_reprocess(self, result):
        self._text(self.result_text, result)
        self.tabs.select(self.results_page._outer)
        quality = "部分 ROI / 标距需检查质量；" if result.get("status") == "attention_required" else ""
        self.status_var.set(quality + "已生成独立提取结果；未重算 DIC，原项目设置和结果保留。")
        output = result.get("output_dir")
        if output:
            self.run_dir = Path(output)
        report = (result.get("report") or {}).get("report_html")
        if report:
            self.artifacts = {"report": Path(report), "summary": self.run_dir / "summary.csv", "gauges": self.run_dir / "gauges.csv"}
        elif output:
            self.artifacts = {"report": Path(output) / "report" / "index.html"}

    def _fill_variables(self, config):
        for variable, key in ((self.source_var, "image_dir"), (self.interval_var, "frame_interval_s"),
                              (self.map_var, "frame_map_csv"), (self.reference_var, "reference_native_frame"),
                              (self.step_var, "step_hint_px"), (self.mask_var, "mask_file")):
            if key in config:
                variable.set("" if config[key] is None else str(config[key]))
        self.analysis_roi_var.set(self._format(config.get("roi_override_px") or []))

    def _accept_snapshot(self, snapshot):
        self._discovered_options = {}
        self._geometry_edited.clear()
        self.project_dir, self.config, self.plan = snapshot["project"], snapshot["config"], snapshot["plan"]
        self.run_dir, self.artifacts = snapshot["run_dir"], snapshot["artifacts"]
        self._image, self._record = snapshot["image"], snapshot["record"]
        self._loading = True
        try:
            self.project_var.set(str(self.project_dir))
            self._fill_variables(self.config)
            self.items = [{"kind": kind, "value": copy.deepcopy(value)} for kind, key in zip(KINDS, ("regions", "gauges", "exclusions_px")) for value in self.config.get(key, [])]
            self._refresh_tree()
            self.new_item()
        finally:
            self._loading = False
        self._dirty = self._editor_dirty = False
        if self.plan:
            params = self.plan.get("selected_parameters", {})
            ready = self.plan.get("status") == "ready"
            self.plan_summary_var.set(f"计划：{'可运行' if ready else '需检查'} · 参考原生帧 {self.plan.get('reference_native_frame')}\n计算区 {self.plan.get('analysis_rectangle_candidate_px')}\n子集 {params.get('subset_size_px')} px · 步长 {params.get('step_px')} px · {params.get('solver')}\n候选区域与标距仍需按试样位置检查。")
            self._text(self.result_text, self.plan)
            self.status_var.set("计划可运行；可检查图像或继续修改 ROI / 标距。" if ready else "计划需要检查；请查看试算原因，调整后重新生成。")
        else:
            self.plan_summary_var.set("尚无当前计划。检查 ROI / 标距后生成计划；已有运行文件仍保留。")
            self.status_var.set("项目已读取；可编辑 ROI / 标距和参数，再生成计划。")
        if self.config.get("time_basis") == "frame_index":
            self.status_var.set(self.status_var.get() + " 无物理时间，结果按原生帧号显示。")
        self._show_geometry_reference()
        self._render_image()
        self._update_actions()

    def _accept_run(self, value):
        result, snapshot = value
        if snapshot is not None:
            self._accept_snapshot(snapshot)
        run = result.get("run") or result
        state = result.get("status", run.get("status", "unknown"))
        labels = {"complete": "计算完成", "partial": "部分帧未通过", "failed": "计算失败", "cancelled": "已停止", "needs_input": "需要补充输入", "review_required": "计划需要检查", "attention_required": "结果需要检查"}
        quality = run.get("numerical_quality_passed")
        quality_label = "通过" if quality is True else "未通过" if quality is False else "未评估"
        self.status_var.set(f"{labels.get(state, state)}；数值质量：{quality_label}；实验验收：未评估。")
        self._text(self.result_text, result)
        self.tabs.select(self.results_page._outer)

    def _accept_verification(self, report):
        self.status_var.set(f"完整性核验：{'通过' if report.get('valid') is True else '未通过'}；数值质量：{'通过' if report.get('numerical_quality_passed') is True else '未通过或未评估'}；实验验收：未评估。")
        self._text(self.result_text, report)
        self.tabs.select(self.results_page._outer)

    def _launch(self, label, operation, accept, cancellable=None):
        if self._busy:
            return
        if self.app is not None and (getattr(self.app, "is_processing", False) or getattr(self.app, "_completion_pending", False)):
            raise ValueError("手动分析仍在运行；完成后再启动自动分析任务。")
        self._busy, self._cancel_project = True, cancellable
        self.status_var.set(label + "…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(15)
        self._update_actions()
        for widget, _ in self._controls:
            widget.configure(state="disabled")
        self.cancel_button.configure(state="normal" if cancellable else "disabled")
        api_override, events = self._api, self._queue
        def worker():
            try:
                api = api_override or importlib.import_module("straintrace_agent.api")
                value = operation(api, lambda event: events.put(("progress", event)))
                events.put(("done", (accept, value, None)))
            except Exception as exc:
                events.put(("done", (accept, None, exc)))
        self._worker = threading.Thread(target=worker, name="straintrace-agent-ui", daemon=True)
        self._worker.start()

    def cancel(self):
        project = getattr(self, "_cancel_project", None)
        if not self._busy or project is None:
            return
        self.cancel_button.configure(state="disabled")
        self.status_var.set("正在发送停止请求；等待当前检查点结束…")
        api_override, events = self._api, self._queue
        def request():
            try:
                (api_override or importlib.import_module("straintrace_agent.api")).cancel(project)
                events.put(("cancel", None))
            except Exception as exc:
                events.put(("cancel", exc))
        threading.Thread(target=request, name="straintrace-agent-cancel", daemon=True).start()

    def _poll(self):
        if self._closed:
            return
        try:
            for _ in range(100):
                if self._closed:
                    return
                kind, payload = self._queue.get_nowait()
                if kind == "progress":
                    event = PROGRESS_LABELS.get(payload.get("event"), "正在处理")
                    native = payload.get("native_frame")
                    position, total = payload.get("position"), payload.get("total")
                    count = f"（{position}/{total}）" if position is not None and total else ""
                    self.status_var.set(event + (f" · 原生帧 {native}" if native is not None else "") + count)
                elif kind == "cancel":
                    if payload:
                        self._error(payload)
                        self.cancel_button.configure(state="normal" if self._busy else "disabled")
                    elif self._busy:
                        self.status_var.set("停止请求已发送；等待当前检查点结束…")
                elif kind == "done":
                    accept, value, error = payload
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    self._busy = False
                    self._cancel_project = None
                    for widget, state in self._controls:
                        widget.configure(state=state)
                    self.cancel_button.configure(state="disabled")
                    try:
                        self._error(error) if error else accept(value)
                    except Exception as exc:
                        self._error(exc)
                    if self._closed:
                        return
                    self._update_actions()
        except queue.Empty:
            pass
        if not self._closed:
            self._poll_id = self._scheduler.after(60, self._poll)

    def _update_actions(self):
        if not hasattr(self, "cancel_button"):
            return
        for button in (self.discover_button, self.create_button, self.auto_button,
                       self.save_button, self.plan_button, self.run_button, self.verify_button,
                       self.report_button, self.folder_button, self.summary_button,
                       self.gauges_button, self.reprocess_button):
            button.configure(state="disabled" if self._disabled_help(button) else "normal")

    def _disabled_help(self, widget):
        if widget is getattr(self, "cancel_button", None):
            return "只有正在逐帧分析时才可以请求停止。"
        if self._busy:
            return "请等待当前操作完成。"
        if widget in (self.discover_button, self.create_button, self.auto_button):
            if not self.source_var.get().strip():
                return "先选择存放实验图像的目录。"
            if widget is not self.discover_button and not self.project_var.get().strip():
                return "先用“选择新目录”设置项目保存位置。"
            return ""
        if self.project_dir is None:
            return "先新建或打开一个自动分析项目。"
        if widget is self.run_button:
            if self._editor_dirty:
                return "先点“应用到列表”保存当前测量项，再生成计划。"
            if self._dirty:
                return "设置已修改，请重新生成计划。"
            if not self.plan or self.plan.get("status") != "ready":
                return "点击“生成计划”，到“计划与结果”检查试算质量。计划可运行后再开始。"
        if widget in (self.folder_button, self.reprocess_button) and self.run_dir is None:
            return "先完成一次分析，生成结果目录。"
        for button, key, reason in (
            (self.report_button, "report", "先运行分析，生成可打开的报告。"),
            (self.summary_button, "summary", "先运行分析，生成逐帧数据表。"),
            (self.gauges_button, "gauges", "在“ROI 与标距”添加标距并运行分析，标距表生成后可打开。"),
        ):
            if widget is button and not self.artifacts.get(key):
                return reason
        return ""

    def _geometry_table_help(self, event=None):
        if event is not None and getattr(event, "keysym", "") != "F1":
            column = self.geometry_tree.identify_column(event.x)
            if column == "#1":
                return "类型决定如何使用坐标。统计区用来汇总应变，标距有两个端点，排除区不参与计算。"
            if column == "#2":
                return "名称会写入导出的数据表。标距名称旁显示所用的测量方向。"
        return "单击一行可编辑名称、坐标或方向。点“应用到列表”更新该行，再保存项目。"

    @staticmethod
    def _text(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str))
        widget.configure(state="disabled")

    @staticmethod
    def _format(values):
        return ", ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in values)

    def new_item(self):
        was_loading, self._loading = self._loading, True
        try:
            self._selected = None
            self.geometry_tree.selection_remove(self.geometry_tree.selection())
            self.name_var.set("")
            self.a_var.set("")
            self.b_var.set("")
            self.axis_var.set("distance")
            self.coordinate_var.set("image")
            self.fraction_var.set("0.5")
            self.count_var.set("3")
            self._editor_dirty = False
        finally:
            self._loading = was_loading
        self._update_actions()

    def _refresh_tree(self):
        self.geometry_tree.delete(*self.geometry_tree.get_children())
        for index, item in enumerate(self.items):
            value = item["value"]
            name = value.get("name", "") if isinstance(value, dict) else self._format(value)
            if item["kind"] == KINDS[1]:
                name += " · " + AXIS_LABELS.get(value.get("axis"), "")
            self.geometry_tree.insert("", "end", iid=str(index), values=(item["kind"], name))

    def _select_item(self, _event=None):
        selection = self.geometry_tree.selection()
        if self._busy:
            expected = (str(self._selected),) if self._selected is not None else ()
            if selection != expected:
                self.geometry_tree.selection_set(expected)
            return
        if not selection:
            return
        index = int(selection[0])
        item, self._selected = self.items[index], index
        kind, value = item["kind"], item["value"]
        self._loading = True
        try:
            self.kind_var.set(kind)
            self.name_var.set(value.get("name", "") if isinstance(value, dict) else "")
            self.b_var.set("")
            if kind == KINDS[0]:
                x0, x1, y0, y1 = value["bounds"]
                self.a_var.set(self._format([x0, y0, x1 - x0, y1 - y0]))
                self.coordinate_var.set(value.get("coordinate_system", "image"))
                self.fraction_var.set(str(value.get("min_valid_fraction", 0.5)))
                self.count_var.set(str(value.get("min_valid_pois", 3)))
            elif kind == KINDS[1]:
                self.a_var.set(self._format(value.get("a_point_px", value.get("a_roi_px", []))))
                self.b_var.set(self._format(value.get("b_point_px", value.get("b_roi_px", []))))
                self.axis_var.set(value["axis"])
            else:
                self.a_var.set(self._format(value))
            self._editor_dirty = False
        finally:
            self._loading = False
        self._update_actions()

    def apply_item(self):
        def apply():
            if self.project_dir is not None:
                self._require_geometry_reference()
            kind, name = self.kind_var.get(), self.name_var.get().strip()
            prior = copy.deepcopy(self.items[self._selected]["value"]) if self._selected is not None and self.items[self._selected]["kind"] == kind else {}
            if kind != KINDS[2] and not name:
                raise ValueError("请填写 ROI 或标距名称。")
            if kind != KINDS[2] and any(
                i != self._selected and item["kind"] == kind and isinstance(item["value"], dict)
                and str(item["value"].get("name", "")).strip().casefold() == name.casefold()
                for i, item in enumerate(self.items)
            ):
                raise ValueError("同类 ROI / 标距名称不能重复。")
            if kind == KINDS[0]:
                x, y, width, height = self._numbers(self.a_var.get(), (4,))
                fraction, count = float(self.fraction_var.get()), int(self.count_var.get())
                if not 0 <= fraction <= 1 or count < 1:
                    raise ValueError("有效比例应为 0–1，有效点数应为正整数。")
                value = {**prior, "name": name, "bounds": [x, x + width, y, y + height], "coordinate_system": self.coordinate_var.get(), "min_valid_fraction": fraction, "min_valid_pois": count}
            elif kind == KINDS[1]:
                value = {k: v for k, v in prior.items() if k not in ("a_point_px", "a_roi_px", "b_point_px", "b_roi_px")}
                value.update(name=name, axis=self.axis_var.get())
                for side, variable in (("a", self.a_var), ("b", self.b_var)):
                    coords = self._numbers(variable.get(), (2, 4))
                    if len(coords) == 4:
                        coords = self._numbers(variable.get(), (4,), integer=True)
                    value[f"{side}_{'point' if len(coords) == 2 else 'roi'}_px"] = coords
                value.setdefault("geometry_status", "user_defined")
                value.setdefault("max_distance_px", 0.0)
            else:
                value = self._numbers(self.a_var.get(), (4,), integer=True)
            item = {"kind": kind, "value": value}
            self._geometry_edited.add(GEOMETRY_KEYS[kind])
            if self._selected is None:
                self.items.append(item)
            else:
                previous_kind = self.items[self._selected]["kind"]
                self._geometry_edited.add(GEOMETRY_KEYS[previous_kind])
                self.items[self._selected] = item
            self._refresh_tree()
            self.new_item()
            self._dirty = True
            self.geometry_hint.set("已应用到列表；保存设置或生成计划后写入项目。")
            self._render_image()
            self._update_actions()
        self._guard(apply)

    def delete_item(self):
        if self._selected is not None:
            kind = self.items[self._selected]["kind"]
            self._geometry_edited.add(GEOMETRY_KEYS[kind])
            self.items.pop(self._selected)
            self._refresh_tree()
            self.new_item()
            self._dirty = True
            self._render_image()
            self._update_actions()

    def use_candidate_roi(self):
        if self.plan:
            try:
                self._require_geometry_reference()
            except ValueError as exc:
                self._error(exc)
                return
            self.analysis_roi_var.set(self._format(self.plan.get("analysis_rectangle_candidate_px", [])))
            self._render_image()

    def use_candidate_gauges(self):
        try:
            self._require_geometry_reference()
        except ValueError as exc:
            self._error(exc)
            return
        candidates = (self.plan or {}).get("automatic_gauge_candidates", [])
        if not candidates:
            self.status_var.set("请先生成计划，查看自动标距候选。")
            return
        existing = {str(item["value"].get("name", "")).strip().casefold() for item in self.items if item["kind"] == KINDS[1]}
        for candidate in candidates:
            candidate_name = str(candidate.get("name", "")).strip().casefold()
            if candidate_name not in existing:
                gauge = {k: copy.deepcopy(v) for k, v in candidate.items() if k not in ("estimated_initial_length_px", "loading_axis")}
                self.items.append({"kind": KINDS[1], "value": gauge})
                existing.add(candidate_name)
        self._dirty = True
        self._refresh_tree()
        self._render_image()
        self._update_actions()
        self.status_var.set("标距候选已加入列表；请检查端点、方向与试样对应关系后保存。")

    def show_image(self):
        if self.app is not None and hasattr(self, "visual_page"):
            self.app.show_visual_window()
            self.app.workspace_notebook.select(self.visual_page)
            self._render_image()

    def show_editor(self):
        self.tabs.select(self.geometry_page._outer)
        if self.app is not None:
            self.app.control_notebook.select(self)
            self.app.show_control_window()

    def _render_image(self):
        canvas = getattr(self, "image_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        if self._image is None:
            canvas.create_text(20, 25, anchor="nw", fill="#e2e8f0", text="新建或打开自动分析项目后显示参考图。")
            return
        width, height = max(canvas.winfo_width(), 20), max(canvas.winfo_height(), 20)
        self._scale = min((width - 8) / self._image.width, (height - 8) / self._image.height)
        size = (max(1, round(self._image.width * self._scale)), max(1, round(self._image.height * self._scale)))
        self._offset = ((width - size[0]) / 2, (height - size[1]) / 2)
        self._photo = ImageTk.PhotoImage(self._image.resize(size), master=canvas)
        canvas.create_image(*self._offset, image=self._photo, anchor="nw")
        rectangles = []
        try:
            roi = self._numbers(self.analysis_roi_var.get(), (4,)) if self.analysis_roi_var.get().strip() else (self.plan or {}).get("analysis_rectangle_candidate_px")
            if roi:
                rectangles.append((roi, "计算区域", "#4ade80"))
        except ValueError:
            pass
        for item in self.items:
            kind, value = item["kind"], item["value"]
            if kind == KINDS[0] and value.get("coordinate_system", "image") == "image":
                x0, x1, y0, y1 = value["bounds"]
                rectangles.append(([x0, y0, x1 - x0, y1 - y0], value["name"], "#22d3ee"))
            elif kind == KINDS[2]:
                rectangles.append((value, "排除", "#fb7185"))
            elif kind == KINDS[1]:
                for side in ("a", "b"):
                    coords = value.get(f"{side}_roi_px")
                    if coords is None:
                        point = value.get(f"{side}_point_px", [])
                        coords = [point[0] - 2, point[1] - 2, 4, 4] if len(point) == 2 else None
                    if coords:
                        rectangles.append((coords, value["name"] + " " + side.upper(), "#fbbf24"))
        ox, oy = self._offset
        for (x, y, w, h), label, color in rectangles:
            x0, y0 = ox + x * self._scale, oy + y * self._scale
            canvas.create_rectangle(x0, y0, x0 + w * self._scale, y0 + h * self._scale, outline=color, width=2)
            canvas.create_text(x0 + 3, y0 + 3, text=label, fill=color, anchor="nw")
        self.image_info_var.set(f"参考原生帧 {(self._record or {}).get('native_frame')} · {self._image.width}×{self._image.height} px · {(self._record or {}).get('path', '')}")

    def _image_point(self, event):
        if self._image is None:
            return None
        x, y = (event.x - self._offset[0]) / self._scale, (event.y - self._offset[1]) / self._scale
        if 0 <= x <= self._image.width and 0 <= y <= self._image.height:
            return round(x), round(y)
        return None

    def _draw_start(self, event):
        self._drag = None
        if hasattr(self, "image_canvas"):
            self.image_canvas.delete("draft")
        if not self._busy:
            try:
                self._require_geometry_reference()
            except ValueError as exc:
                self.status_var.set(str(exc))
                return
            self._drag = self._image_point(event)

    def _draw_motion(self, event):
        point = self._image_point(event)
        if self._drag is None or point is None:
            return
        self.image_canvas.delete("draft")
        ox, oy = self._offset
        self.image_canvas.create_rectangle(ox + self._drag[0] * self._scale, oy + self._drag[1] * self._scale,
                                           ox + point[0] * self._scale, oy + point[1] * self._scale, outline="#ffffff", dash=(4, 3), tags="draft")

    def _draw_end(self, event):
        if hasattr(self, "image_canvas"):
            self.image_canvas.delete("draft")
        end, start, self._drag = self._image_point(event), self._drag, None
        if start is None or end is None or self._busy:
            return
        try:
            self._require_geometry_reference()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        x0, x1 = sorted((start[0], end[0]))
        y0, y1 = sorted((start[1], end[1]))
        if x1 <= x0 or y1 <= y0:
            return
        rect = self._format([x0, y0, x1 - x0, y1 - y0])
        target = self.draw_var.get()
        if target == "计算区域":
            self.analysis_roi_var.set(rect)
            self._render_image()
        else:
            self.kind_var.set(KINDS[0] if target == "统计 ROI" else KINDS[2] if target == "排除区" else KINDS[1])
            self.coordinate_var.set("image")
            (self.b_var if target == "标距 B" else self.a_var).set(rect)
            self.geometry_hint.set("已从图像填入坐标；请检查名称和方向，点应用到列表。")
        self.status_var.set("图像坐标已填入；保存或生成计划前可继续人工修改。")

    def open_artifact(self, name):
        def open_path():
            self._get_project()
            path = self.run_dir if name == "folder" else self.artifacts.get(name)
            if path is None or not Path(path).exists():
                raise ValueError("当前运行没有此文件；请先完成分析或检查失败原因。")
            if os.name == "nt":
                os.startfile(str(path))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        self._guard(open_path)

    def destroy(self):
        self._closed = True
        if self._poll_id is not None:
            try:
                self._scheduler.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        project = getattr(self, "_cancel_project", None)
        worker = self._worker
        if self._busy and project is not None:
            try:
                (self._api or importlib.import_module("straintrace_agent.api")).cancel(project)
            except Exception:
                pass
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        super().destroy()
