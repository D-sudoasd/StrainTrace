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


OBJECTIVES = {"全场位移与应变": "full_field", "虚拟标距应变": "virtual_gauge", "应力与应变": "stress_strain"}
KINDS = ("统计 ROI", "虚拟标距", "排除区")
DRAW_TARGETS = ("计算区域", "统计 ROI", "排除区", "标距 A", "标距 B")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _project_file(project, relative):
    """Resolve persisted pointers only inside their owning project."""
    path = (project / relative).resolve()
    path.relative_to(project.resolve())
    return path


def _snapshot(api, project):
    config = api.read_project_config(project)
    status = api.status(project)
    inventory = _read_json(project / "inventory.json").get("frames", [])
    plan = _read_json(_project_file(project, status["current_plan"])) if status.get("current_plan") else None
    if plan is None:
        # Use the planner's reference selection and its already loaded image.
        record, gray, _selection = api._automatic_reference(config, inventory)
    else:
        native = plan.get("reference_native_frame")
        if native is None:
            raise ValueError("当前计划未指定参考原生帧；请重新生成计划。")
        record = api._reference_record({"reference_native_frame": native}, inventory)
        gray = api._load_image(record["path"])
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
        self._controls = []
        self._build()
        self._poll_id = self._scheduler.after(60, self._poll)

    def _var(self, value="", *, editor=False):
        variable = tk.StringVar(self, value=value)
        variable.trace_add("write", lambda *_: self._mark_dirty(editor))
        return variable

    def _help(self, widget, text, choices=None):
        if self.app is not None:
            self.app.add_tooltip(widget, text, choices) if choices is not None else self.app.add_tooltip(widget, text)
        return widget

    def _button(self, parent, text, command, help_text, row, column, **kwargs):
        widget = ttk.Button(parent, text=text, command=command)
        widget.grid(row=row, column=column, sticky="ew", padx=3, pady=3, **kwargs)
        self._help(widget, help_text)
        self._controls.append((widget, "normal"))
        return widget

    def _entry(self, parent, label, variable, row, help_text):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=3, pady=3)
        widget = ttk.Entry(parent, textvariable=variable, width=18)
        widget.grid(row=row, column=1, sticky="ew", padx=3, pady=3)
        self._help(widget, help_text)
        self._controls.append((widget, "normal"))
        return widget

    def _combo(self, parent, variable, values, row, column, help_text):
        widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=15)
        widget.grid(row=row, column=column, sticky="ew", padx=3, pady=3)
        self._help(widget, help_text, {v: f"选择“{v}”。{help_text}" for v in values})
        self._controls.append((widget, "readonly"))
        return widget

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.project_var = self._var()
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.project_entry = ttk.Entry(bar, textvariable=self.project_var, width=15)
        self.project_entry.grid(row=0, column=0, sticky="ew", padx=3)
        self._help(self.project_entry, "自动分析项目目录。新建时使用空目录，必须位于原始图像目录之外。")
        self._controls.append((self.project_entry, "normal"))
        self.open_button = self._button(bar, "打开项目", self.open_dialog, "打开已有自动分析项目，恢复配置、计划和结果。", 0, 1)
        self._button(bar, "选择新目录", self.choose_project, "选择保存新项目的目录；不会覆盖已有项目。", 0, 2)

        self.tabs = ttk.Notebook(self)
        self.tabs.grid(row=1, column=0, sticky="nsew", pady=5)
        self._help(self.tabs, "依次检查输入与参数、编辑 ROI 与标距、查看计划和运行结果。")
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
        self.run_button = self._button(footer, "运行 / 恢复", self.run_project, "按当前已保存且可运行的计划计算；相同输入和设置可复用逐帧检查点。", 0, 2)
        self.cancel_button = ttk.Button(footer, text="停止", command=self.cancel, state="disabled")
        self.cancel_button.grid(row=0, column=3, sticky="ew", padx=3, pady=3)
        self._help(self.cancel_button, "请求停止逐帧计算；在安全检查点结束后停止，已保存检查点保留。")
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=4, sticky="ew", padx=3, pady=3)
        self.status_var = tk.StringVar(self, value="选择实验文件夹；先发现输入或打开已有项目。")
        self.status_label = ttk.Label(footer, textvariable=self.status_var, wraplength=500, justify="left")
        self.status_label.grid(row=2, column=0, columnspan=4, sticky="ew", padx=3)
        self.bind("<Configure>", lambda event: self.status_label.configure(wraplength=max(220, event.width - 20)))
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

    def apply_theme(self):
        if self.app is not None:
            for page in (self.input_page, self.geometry_page, self.results_page):
                page._canvas.configure(background=self.app.card_bg)
            for text in (self.discovery_text, self.result_text):
                text.configure(background=self.app.panel_bg, foreground=self.app.text_color, insertbackground=self.app.text_color)

    def _build_input(self):
        p = self.input_page
        self.source_var = self._var()
        self.interval_var, self.map_var = self._var(), self._var()
        self.reference_var, self.step_var, self.mask_var = self._var(), self._var(), self._var()
        self.reference_var.trace_add("write", lambda *_: self._show_geometry_reference())
        self.analysis_roi_var = self._var()
        self.objective_var = self._var(next(iter(OBJECTIVES)))
        self._entry(p, "实验 / 图像目录", self.source_var, 0, "选择实验目录，可自动发现图像序列和 experiment.json；有多个序列时需选定一组。")
        self._button(p, "选择目录", self.choose_source, "选择原始实验或图像文件夹。", 1, 0)
        self.discover_button = self._button(p, "发现输入", self.discover, "只读检查图像、时间来源和实验元数据；缺少信息时明确显示需要填写的字段。", 1, 1)
        ttk.Label(p, text="分析目标").grid(row=2, column=0, sticky="w", padx=3)
        self._combo(p, self.objective_var, tuple(OBJECTIVES), 2, 1, "按目标检查所需的标距、时间或载荷信息；不会补造物理标定。")
        self._entry(p, "帧间隔 (s)", self.interval_var, 3, "相邻原生帧号之间的秒数；来自采集记录。帧映射有时间时可留空；均未知时按帧号分析，不能同步载荷或计算应变率。")
        self._entry(p, "帧映射 CSV", self.map_var, 4, "可选：包含图像文件、原生帧号和时间的显式映射表。")
        self._button(p, "选择映射", lambda: self._choose_file(self.map_var), "选择带原生帧身份和时间信息的 CSV。", 5, 0)
        self._button(p, "使用控制台路径", self.use_legacy_source, "复制上方图像路径；不会改变手动分析的 ROI 或结果。", 5, 1)
        self._entry(p, "参考原生帧号", self.reference_var, 6, "留空自动选择；填写文件中的原生帧号。改变参考帧后请重新检查所有 ROI 与端点。")
        self._entry(p, "网格步长 (px)", self.step_var, 7, "可选正整数。留空由候选比较选择；改变步长会改变测量点和端点支持。")
        self._entry(p, "计算区域 (px)", self.analysis_roi_var, 8, "填写 x,y,宽,高，或在图像窗口拖出矩形；留空使用自动候选。")
        self._entry(p, "试样遮罩文件", self.mask_var, 9, "可选遮罩；排除区也可在 ROI 页逐个添加。不会自动确认试样物理边界。")
        self._button(p, "选择遮罩", lambda: self._choose_file(self.mask_var), "选择与参考图像同尺寸的遮罩。", 10, 0)
        self.preview_button = self._button(p, "图像中编辑 ↗", self.show_image, "在独立图像窗口查看参考图，绘制计算区域、统计 ROI、排除区或标距端点。", 10, 1)
        self.create_button = self._button(p, "新建项目", self.create_project, "只建立图像索引和项目；之后可编辑 ROI、标距和参数，再生成计划。", 11, 0)
        self.auto_button = self._button(p, "自动建项并分析", self.auto_analyze, "对新项目发现输入、建立计划并计算。缺少必要输入或候选不通过时停止并显示原因。", 11, 1)
        self.discovery_text = tk.Text(p, height=6, width=24, wrap="word", state="disabled")
        self.discovery_text.grid(row=12, column=0, columnspan=2, sticky="ew", padx=3, pady=4)
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
        self._help(self.geometry_tree, "选择已有 ROI 或标距编辑。每项可改名、修改坐标或删除；修改后点“应用到列表”，再保存设置。")
        self.kind_var, self.name_var = self._var(KINDS[0], editor=True), self._var(editor=True)
        self.a_var, self.b_var, self.axis_var = self._var(editor=True), self._var(editor=True), self._var("distance", editor=True)
        self.coordinate_var = self._var("image", editor=True)
        self.fraction_var, self.count_var = self._var("0.5", editor=True), self._var("3", editor=True)
        self._combo(p, self.kind_var, KINDS, 1, 0, "统计 ROI 提取全场应变；虚拟标距使用固定参考端点；排除区禁止参与相关。")
        self._button(p, "新建一项", self.new_item, "清空当前编辑器，新建独立统计 ROI、标距或排除区。", 1, 1)
        self._entry(p, "名称", self.name_var, 2, "ROI 和标距名称应唯一；名称会保留在导出结果中。")
        self._entry(p, "区域 / 端点 A", self.a_var, 3, "ROI、排除区：x,y,宽,高。标距端点：x,y 点，或 x,y,宽,高矩形。图像坐标单位 px。")
        self._entry(p, "标距端点 B", self.b_var, 4, "仅标距使用；格式与端点 A 相同。两个端点在固定参考图中定义。")
        ttk.Label(p, text="标距方向").grid(row=5, column=0, sticky="w", padx=3)
        self._combo(p, self.axis_var, ("x", "y", "distance"), 5, 1, "x/y 使用该方向投影标距；distance 使用两端点的直线距离。")
        ttk.Label(p, text="统计 ROI 坐标").grid(row=6, column=0, sticky="w", padx=3)
        self._combo(p, self.coordinate_var, ("image", "calibrated"), 6, 1, "image 使用原图 px；calibrated 使用显式标定坐标，需有完整物理标定；后者仅用数值编辑。")
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
        self._button(p, "打开标距表格", lambda: self.open_artifact("gauges"), "打开每条标距的端点、应变与质量状态。", 3, 1)
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
        self._combo(bar, self.draw_var, DRAW_TARGETS, 0, 1, "按住左键拖出矩形，松开后填入对应坐标。统计 ROI/标距需再点应用到列表。")
        self._button(bar, "返回编辑器", self.show_editor, "回到控制台 ROI 与标距编辑页，调整名称、方向与坐标。", 0, 2)
        self.image_canvas = tk.Canvas(self.visual_page, background="#111827", highlightthickness=0, cursor="crosshair")
        self.image_canvas.grid(row=1, column=0, sticky="nsew", pady=5)
        self.image_canvas.bind("<Configure>", lambda _: self._render_image())
        self.image_canvas.bind("<ButtonPress-1>", self._draw_start)
        self.image_canvas.bind("<B1-Motion>", self._draw_motion)
        self.image_canvas.bind("<ButtonRelease-1>", self._draw_end)
        self._help(self.image_canvas, "参考图像坐标随窗口缩放保持一致。绿色为计算区、青色为统计 ROI、橙色为标距、红色为排除区。")
        self.image_info_var = tk.StringVar(self, value="打开或建立项目后显示参考图像。")
        self.image_info_label = ttk.Label(self.visual_page, textvariable=self.image_info_var, wraplength=700)
        self.image_info_label.grid(row=2, column=0, sticky="ew")
        self.visual_page.bind("<Configure>", lambda e: self.image_info_label.configure(wraplength=max(200, e.width - 20)))
        notebook.add(self.visual_page, text="自动分析图像")

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
        options["image_dir"] = self.source_var.get().strip()
        options["frame_interval_s"] = float(self.interval_var.get()) if self.interval_var.get().strip() else None
        options["frame_map_csv"] = self.map_var.get().strip() or None
        options.update(self._changes())
        if options["frame_interval_s"] is not None:
            options["time_basis"] = "physical_time"
        return options

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
        if self.config and Path(self.source_var.get()).resolve() != Path(self.config["image_dir"]).resolve():
            raise ValueError("当前项目的原始图像目录固定。分析其他序列请新建项目。")
        changes = self._changes()
        if self._reference_changed() and changes["roi_override_px"] != self.config.get("roi_override_px"):
            raise ValueError("参考帧已修改；请先还原计算区域并保存设置，显示新参考图后再编辑坐标。")
        changes["frame_interval_s"] = float(self.interval_var.get()) if self.interval_var.get().strip() else None
        changes["frame_map_csv"] = self.map_var.get().strip() or None
        if changes["frame_interval_s"] is not None or changes["frame_map_csv"]:
            changes["time_basis"] = "physical_time"
        elif not self.config.get("loads"):
            changes["time_basis"] = "frame_index"
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
        project = Path(path).expanduser().resolve()
        self._launch("读取项目", lambda api, emit: _snapshot(api, project), self._accept_snapshot)

    def discover(self):
        def start():
            folder = self.source_var.get().strip()
            options = self._init_options()
            # Discovery should be free to locate a nested sequence.
            options.pop("image_dir", None)
            options = {k: v for k, v in options.items() if v is not None and v != []}
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
            if not options.get("frame_interval_s") and not options.get("frame_map_csv") and not options.get("loads_csv"):
                options.setdefault("time_basis", "frame_index")
            def operation(api, emit):
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
            # Allow discovery to choose a nested sequence unless one was found.
            if not self._discovered_options.get("image_dir"):
                options.pop("image_dir", None)
            options = {k: v for k, v in options.items() if v is not None and v != []}
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
            def operation(api, emit):
                api.update_project(project, changes)
                return _snapshot(api, project)
            self._launch("保存项目设置", operation, self._accept_snapshot)
        self._guard(start)

    def make_plan(self):
        def start():
            project, changes = self._get_project(), self._project_changes()
            def operation(api, emit):
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
        if self.app is not None and getattr(self.app, "is_processing", False):
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
                kind, payload = self._queue.get_nowait()
                if kind == "progress":
                    event = payload.get("event", "处理中")
                    native = payload.get("native_frame")
                    self.status_var.set(f"{event}" + (f" · 原生帧 {native}" if native is not None else ""))
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
                    self._update_actions()
        except queue.Empty:
            pass
        self._poll_id = self._scheduler.after(60, self._poll)

    def _update_actions(self):
        if not hasattr(self, "run_button"):
            return
        ready = not self._busy and self.project_dir is not None
        for button in (self.save_button, self.plan_button, self.verify_button):
            button.configure(state="normal" if ready else "disabled")
        self.run_button.configure(state="normal" if ready and not self._dirty and not self._editor_dirty and self.plan and self.plan.get("status") == "ready" else "disabled")
        self.report_button.configure(state="normal" if ready and self.artifacts.get("report") else "disabled")
        self.folder_button.configure(state="normal" if ready and self.run_dir else "disabled")
        self.summary_button.configure(state="normal" if ready and self.artifacts.get("summary") else "disabled")
        self.reprocess_button.configure(state="normal" if ready and self.run_dir else "disabled")

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
                name += " · " + value.get("axis", "")
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
            if any(i != self._selected and item["kind"] == kind and isinstance(item["value"], dict) and item["value"].get("name") == name for i, item in enumerate(self.items)):
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
            if self._selected is None:
                self.items.append(item)
            else:
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
        existing = {item["value"].get("name") for item in self.items if item["kind"] == KINDS[1]}
        for candidate in candidates:
            if candidate.get("name") not in existing:
                gauge = {k: copy.deepcopy(v) for k, v in candidate.items() if k not in ("estimated_initial_length_px", "loading_axis")}
                self.items.append({"kind": KINDS[1], "value": gauge})
                existing.add(candidate.get("name"))
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
        super().destroy()
