"""Tk campaign workbench embedded in the legacy StrainTrace desktop app.

The page is intentionally a thin view over :mod:`straintrace_campaign.workflow`.
The workflow module is imported only when a user starts an operation, so the
established single-sequence GUI remains importable in headless environments.
"""

from __future__ import annotations

import importlib
import json
import math
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


_CONFIRMED_DIMENSION_STATUSES = frozenset({
    "confirmed",
    "verified_recorded_dimensions",
    "verified_recorded_area",
})


class CampaignPage(ttk.Frame):
    """Campaign project controls hosted inside the main control notebook."""

    def __init__(self, master, app=None, workflow=None):
        super().__init__(master, padding=8)
        self._scheduler = self.winfo_toplevel()
        self._closed = False
        self.app = app
        self._workflow = workflow
        self.project_dir: Path | None = None
        self.project: dict = {"acquisitions": [], "roots": {}}
        self.current_acquisition_id: str | None = None
        self._filtered_ids: list[str] = []
        self._proposal: dict | None = None
        self._preview_photo = None
        self._preview_image_size = None
        self._preview_scale = 1.0
        self._preview_image_cache: dict[tuple[str, int, int], tuple[object, tuple[int, int]]] = {}
        self._operation_queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        self._poll_after_id = None
        self._busy = False
        self._build()
        self._register_control_help()
        self._poll_after_id = self._scheduler.after(100, self._poll_queue)

    def destroy(self):
        """Cancel root-owned polling before tearing down this child frame.

        The legacy GUI test fixture cancels all callbacks through its Toplevel.
        Registering the callback on that same scheduler keeps Tkinter's Tcl
        command registry in sync when the fixture removes it first.
        """
        if not self._closed:
            self._closed = True
            if self._cancel_event is not None:
                self._cancel_event.set()
            if self._poll_after_id is not None:
                try:
                    self._scheduler.after_cancel(self._poll_after_id)
                except tk.TclError:
                    # The owning Toplevel may already have canceled it.
                    pass
                self._poll_after_id = None
            worker = self._worker
            if worker is not None and worker.is_alive() and worker is not threading.current_thread():
                worker.join(timeout=1.0)
            self._preview_image_cache.clear()
        super().destroy()

    def apply_theme(self):
        """Sync raw Tk canvases and text widgets with the host app's active theme."""
        app = self.app
        if app is None:
            return
        panel_bg = getattr(app, "panel_bg", None) or getattr(app, "card_bg", None)
        card_bg = getattr(app, "card_bg", None) or panel_bg
        text_fg = getattr(app, "text_color", None) or getattr(app, "fg_color", None)
        input_bg = getattr(app, "input_bg", None) or card_bg
        for widget, bg in (
            (getattr(self, "_viewport", None), panel_bg),
            (getattr(self, "roi_editor_canvas", None), panel_bg),
            (getattr(self, "result_canvas", None), card_bg),
        ):
            if widget is not None and bg:
                try:
                    widget.configure(background=bg)
                except tk.TclError:
                    pass
        for text_widget in (getattr(self, "alignment_preview", None), getattr(self, "result_text", None)):
            if text_widget is None:
                continue
            cfg = {}
            if input_bg:
                cfg["background"] = input_bg
            if text_fg:
                cfg["foreground"] = text_fg
                cfg["insertbackground"] = text_fg
            if cfg:
                try:
                    text_widget.configure(**cfg)
                except tk.TclError:
                    pass
        if hasattr(self, "result_canvas"):
            try:
                self._draw_result_curve()
            except tk.TclError:
                pass

    def _get_workflow(self):
        if self._workflow is None:
            self._workflow = importlib.import_module("straintrace_campaign.workflow")
        return self._workflow

    def _add_control_help(self, widget, text, choices=None):
        if self.app is None or not hasattr(self.app, "add_tooltip"):
            return None
        return self.app.add_tooltip(widget, text, choices) if choices is not None else self.app.add_tooltip(widget, text)

    def _register_control_help(self):
        """Keep every campaign widget in the desktop's shared F1/hover help system."""
        if self.app is None or not hasattr(self.app, "add_tooltip"):
            return

        def visit(widget):
            yield widget
            for child in widget.winfo_children():
                yield from visit(child)

        controls = (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Checkbutton, ttk.Radiobutton,
                    ttk.Treeview, ttk.Notebook, ttk.Scrollbar, tk.Button, tk.Checkbutton,
                    tk.Text, tk.Canvas)
        for widget in visit(self):
            if not isinstance(widget, controls) or hasattr(widget, "_tooltip"):
                continue
            text = self._control_help_text(widget)
            if isinstance(widget, ttk.Combobox):
                choices = self._combobox_help_choices(widget)
                self._add_control_help(widget, text, choices)
            else:
                self._add_control_help(widget, text)

    def _control_help_text(self, widget):
        if widget is self._viewport:
            return "批次项目工作区包含来源、批次列表、身份尺寸、ROI、时间载荷和结果页；页面可滚动。"
        if widget is self.roi_canvas:
            return "预览所选 DIC 批次的首帧、中帧或末帧；绿色框为自由段，其他框为待审核测量端点。"
        if widget is self.roi_editor_canvas:
            return "编辑自由段、排除区及每条虚拟引伸计的两个端点矩形；坐标为参考帧像素整数。"
        if widget is self.result_canvas:
            return "显示当前结果文件中可用的 Frame-strain 曲线；曲线只绘制有限应变值。"
        if widget is self.overlay_canvas:
            return "显示当前分析阶段附带的首帧、关键帧或结果叠图；没有附件时会明确提示。"
        if widget is self.alignment_preview:
            return "查看只读的帧时间与载荷支撑点预览；预览不会写入项目，确认对齐后才生成结果。"
        if widget is self.result_text:
            return "查看当前批次身份、运行和对齐阶段记录，以及质量异常帧的原因。"
        if widget is self.acquisition_tree:
            return "单击选择一个采集批次；列表状态区分待审核、待分析、已分析和已对齐。"
        if widget is self.detail_notebook:
            return "切换当前批次的身份尺寸、ROI 审核、时间载荷和结果异常信息。"
        if widget is getattr(self, "_viewport_scrollbar", None):
            return "滚动查看批次项目页面中被视口收起的内容。"
        if isinstance(widget, ttk.Notebook):
            return "切换当前项目的功能页；选择与当前工作相符的一页。"
        if isinstance(widget, ttk.Treeview):
            return "选择一条采集批次记录；筛选器可按名称和处理状态缩小列表。"
        if isinstance(widget, ttk.Scrollbar):
            return "拖动或点击滚动条查看被窗口遮住的内容。"
        if isinstance(widget, tk.Canvas):
            return "查看批次项目图像、曲线或叠图；取点坐标均相对于对应参考图像。"
        if isinstance(widget, tk.Text):
            return "查看本页生成的预览、结果或质量说明；文本可选择并复制。"
        if isinstance(widget, ttk.Combobox):
            label = self._associated_label(widget) or "选项"
            return f"选择{label}；请根据采集记录和时间来源选择匹配值。"
        if isinstance(widget, (ttk.Entry, tk.Entry)):
            label = self._associated_label(widget)
            if label:
                return f"填写{label}。数值和来源应来自可追溯记录；不确定时留空或标为候选。"
            return "填写当前字段。像素坐标使用整数；数值、单位和来源需与记录一致。"
        if isinstance(widget, (ttk.Checkbutton, tk.Checkbutton)):
            return f"{widget.cget('text')}；仅在已人工核实对应实验记录时勾选。"
        if isinstance(widget, (ttk.Button, tk.Button)):
            label = str(widget.cget("text"))
            help_by_button = {
                "新建项目": "创建一个 campaign 项目并登记至少一个数据根目录。",
                "打开项目": "打开已有 project.json，继续使用其来源、审核和恢复记录。",
                "导入 Keystone": "只读导入 Keystone 正式版本文件夹，或含 CURRENT.json 的版本根目录；保留 acquisition ID 和原始结果。",
                "导出": "将当前项目生成经过校验的版本化数据包。",
                "核验": "检查项目、阶段文件、校验和及逐帧数据的一致性。",
                "取消运行": "向当前后台运行发送取消请求；安全检查点完成后停止。",
                "扫描并登记": "按所选根目录别名和适配器发现批次；帧号使用文件名或显式映射。",
                "建议 ROI": "为当前批次生成自由段和端点候选；候选需要人工复核。",
                "图像窗口查看": "在独立图像窗口查看当前 DIC 帧和 ROI 叠加。",
                "保存身份与尺寸来源": "保存当前批次的身份、尺寸值和来源；只有确认尺寸可用于新应力计算。",
                "保存当前取点": "重新校验参考图和整数矩形并保存 ROI；已有审核将被清除。",
                "确认 ROI 与试样身份": "用审核人和试样身份确认绑定当前 proposal 与参考图哈希。",
                "选择…": "选择载荷文件；读取列名以配置帧时间和载荷映射。",
                "预览（不写入）": "只读检查时钟域、载荷支撑点、无效值和最大间隔；此操作不写项目。",
                "确认并对齐": "按已预览的时间来源、载荷列和最大间隔执行正式对齐。",
                "运行 / 恢复本批": "运行或从检查点恢复当前已审核批次；未审核 ROI 无法开始。",
                "运行待处理批次": "依次处理已有人工 ROI 审核且尚无结果的批次。",
            }
            return help_by_button.get(label, f"执行“{label}”操作；结果和错误会显示在本页。")
        return "当前 campaign 控件用于查看或编辑所选批次的来源、审核、对齐或结果信息。"

    @staticmethod
    def _associated_label(widget):
        try:
            info = widget.grid_info()
            row = info.get("row")
            column = int(info.get("column", 0))
            for sibling in widget.master.winfo_children():
                if sibling is widget or not isinstance(sibling, (ttk.Label, tk.Label)):
                    continue
                other = sibling.grid_info()
                if other.get("row") == row and int(other.get("column", 0)) < column:
                    return str(sibling.cget("text")).strip()
        except (tk.TclError, TypeError, ValueError):
            pass
        return ""

    def _combobox_help_choices(self, widget):
        values = [str(value) for value in widget.cget("values")]
        if widget is self.adapter_box:
            descriptions = {"generic": "按通用图像和显式表格规则发现批次。", "desy": "使用 DESY 采集目录适配规则。"}
        elif widget is self.state_filter_box:
            descriptions = {value: f"只显示状态为“{value}”的采集批次。" for value in values}
        elif widget is self.preview_position_box:
            descriptions = {"首帧": "预览序列的第一张图。", "中帧": "预览序列中间位置的图。", "末帧": "预览序列的最后一张图。"}
        elif widget is self.measurement_time_field_box:
            descriptions = {
                "timestamp_unix_s": "使用记录为绝对时间的 DIC/测量时间；推算字段不能冒充此列。",
                "estimated_timestamp_unix_s": "明确使用单独保存的推算绝对时间。",
                "time_s": "使用相对秒；必须与载荷文件具有相同 clock_id。",
            }
        elif widget is self.alignment_mode_box:
            descriptions = {"instantaneous": "在选定测量时刻作有界精确匹配或线性插值。",
                            "exposure_mean": "对曝光时间窗内的分段载荷信号求均值。"}
        elif widget is self.analysis_mode_box:
            descriptions = {"extensometer": "使用多条独立虚拟引伸计测量标距变化。",
                            "fullfield": "使用独立的二维全场 DIC 输出。"}
        elif widget is self.root_alias_box:
            descriptions = {value: f"项目中登记的数据根目录别名“{value}”。" for value in values}
        elif widget in (self.time_column_box, self.force_column_box):
            descriptions = {value: f"读取所选载荷文件中的“{value}”列；请核对列含义和单位。" for value in values}
        else:
            label = self._associated_label(widget) or "列表中的值"
            descriptions = {value: f"将{label}设为“{value}”；请根据来源记录确认。" for value in values}
        return {value: descriptions.get(value, f"使用“{value}”选项。") for value in values}

    def _refresh_combobox_help(self, widget, choices):
        tooltip = getattr(widget, "_tooltip", None)
        mapping = {str(key): str(value) for key, value in choices.items()}
        if tooltip is not None and hasattr(tooltip, "choices"):
            tooltip.choices = mapping
            widget._tooltip_choices = mapping

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._viewport = tk.Canvas(self, width=600, height=410, highlightthickness=0)
        self._viewport.grid(row=0, column=0, sticky="nsew")
        self._add_control_help(self._viewport, "批次项目工作区包含来源、批次列表、身份尺寸、ROI、时间载荷和结果页；页面可滚动。")
        self._viewport_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._viewport.yview)
        self._viewport_scrollbar.grid(row=0, column=1, sticky="ns")
        self._viewport.configure(yscrollcommand=self._viewport_scrollbar.set)
        self.content = ttk.Frame(self._viewport)
        self._viewport_window = self._viewport.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", self._update_content_scrollregion)
        self._viewport.bind("<Configure>", self._resize_content_window)
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(2, weight=1)

        bar = ttk.Frame(self.content)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for col in range(9):
            bar.columnconfigure(col, weight=0)
        bar.columnconfigure(8, weight=1)
        self.new_button = ttk.Button(bar, text="新建项目", command=self.new_project)
        self.open_button = ttk.Button(bar, text="打开项目", command=self.open_project_dialog)
        self.import_button = ttk.Button(bar, text="导入 Keystone", command=self.import_keystone_dialog)
        self.export_button = ttk.Button(bar, text="导出", command=self.export_dialog)
        self.verify_button = ttk.Button(bar, text="核验", command=self.verify_project)
        self.cancel_button = ttk.Button(bar, text="取消运行", command=self.cancel_operation, state=tk.DISABLED)
        for i, button in enumerate((self.new_button, self.open_button, self.import_button,
                                    self.export_button, self.verify_button, self.cancel_button)):
            button.grid(row=0, column=i, padx=(0, 5), sticky="w")
        self.project_label_var = tk.StringVar(value="尚未打开 campaign 项目")
        ttk.Label(bar, textvariable=self.project_label_var).grid(row=0, column=8, sticky="e")

        source = ttk.LabelFrame(self.content, text="数据来源与批次发现", padding=(8, 5))
        source.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        source.columnconfigure(3, weight=1)
        ttk.Label(source, text="根目录别名").grid(row=0, column=0, sticky="w")
        self.root_alias_var = tk.StringVar(value="data")
        self.root_alias_box = ttk.Combobox(source, textvariable=self.root_alias_var, width=14)
        self.root_alias_box.grid(row=0, column=1, padx=(4, 12), sticky="w")
        ttk.Label(source, text="适配器").grid(row=0, column=2, sticky="w")
        self.adapter_var = tk.StringVar(value="generic")
        self.adapter_box = ttk.Combobox(source, textvariable=self.adapter_var,
                                        values=("generic", "desy"), width=12, state="readonly")
        self.adapter_box.grid(row=0, column=3, padx=(4, 12), sticky="w")
        self.pattern_var = tk.StringVar()
        ttk.Label(source, text="名称筛选").grid(row=0, column=4, sticky="w")
        ttk.Entry(source, textvariable=self.pattern_var, width=22).grid(row=0, column=5, padx=(4, 8))
        self.scan_button = ttk.Button(source, text="扫描并登记", command=self.scan_root)
        self.scan_button.grid(row=0, column=6, sticky="w")

        split = ttk.Panedwindow(self.content, orient=tk.HORIZONTAL)
        split.grid(row=2, column=0, sticky="nsew", pady=(0, 6))
        left = ttk.Frame(split, padding=(0, 0, 5, 0))
        right = ttk.Frame(split, padding=(5, 0, 0, 0))
        split.add(left, weight=1)
        split.add(right, weight=3)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        filter_row = ttk.Frame(left)
        filter_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        filter_row.columnconfigure(1, weight=1)
        ttk.Label(filter_row, text="批次筛选").grid(row=0, column=0, sticky="w")
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_row, textvariable=self.filter_var)
        filter_entry.grid(row=0, column=1, sticky="ew", padx=4)
        filter_entry.bind("<KeyRelease>", lambda _event: self.refresh_table())
        self.state_filter_var = tk.StringVar(value="全部")
        self.state_filter_box = ttk.Combobox(filter_row, textvariable=self.state_filter_var,
                                             values=("全部", "待审核", "待分析", "已分析", "已对齐", "异常"),
                                             width=10, state="readonly")
        self.state_filter_box.grid(row=0, column=2, sticky="e")
        self.state_filter_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_table())
        self.acquisition_tree = ttk.Treeview(
            left, columns=("status", "frames", "dic", "area"), show="tree headings", selectmode="browse"
        )
        self.acquisition_tree.heading("#0", text="采集批次")
        self.acquisition_tree.heading("status", text="状态")
        self.acquisition_tree.heading("frames", text="帧")
        self.acquisition_tree.heading("dic", text="DIC")
        self.acquisition_tree.heading("area", text="面积")
        self.acquisition_tree.column("#0", width=245, minwidth=140, stretch=True)
        self.acquisition_tree.column("status", width=80, anchor="center")
        self.acquisition_tree.column("frames", width=58, anchor="e")
        self.acquisition_tree.column("dic", width=55, anchor="center")
        self.acquisition_tree.column("area", width=65, anchor="e")
        tree_scroll = ttk.Scrollbar(left, orient="vertical", command=self.acquisition_tree.yview)
        self.acquisition_tree.configure(yscrollcommand=tree_scroll.set)
        self.acquisition_tree.grid(row=2, column=0, sticky="nsew")
        tree_scroll.grid(row=2, column=1, sticky="ns")
        self.acquisition_tree.bind("<<TreeviewSelect>>", self._on_select_acquisition)
        self.batch_count_var = tk.StringVar(value="0 批次")
        ttk.Label(left, textvariable=self.batch_count_var).grid(row=3, column=0, sticky="w", pady=(4, 0))

        self.detail_notebook = ttk.Notebook(right)
        self.detail_notebook.grid(row=0, column=0, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.identity_tab = ttk.Frame(self.detail_notebook, padding=8)
        self.roi_tab = ttk.Frame(self.detail_notebook, padding=8)
        self.alignment_tab = ttk.Frame(self.detail_notebook, padding=8)
        self.results_tab = ttk.Frame(self.detail_notebook, padding=8)
        self.detail_notebook.add(self.identity_tab, text="身份与尺寸")
        self.detail_notebook.add(self.roi_tab, text="ROI 审核")
        self.detail_notebook.add(self.alignment_tab, text="时间与载荷")
        self.detail_notebook.add(self.results_tab, text="结果与异常")
        self._build_identity_tab()
        self._build_roi_tab()
        self._build_alignment_tab()
        self._build_results_tab()

        footer = ttk.Frame(self.content)
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100, length=150)
        self.progress.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=1, sticky="ew")

    def _update_content_scrollregion(self, _event=None):
        bounds = self._viewport.bbox("all")
        if bounds:
            self._viewport.configure(scrollregion=bounds)

    def _resize_content_window(self, event):
        self._viewport.itemconfigure(self._viewport_window, width=event.width)

    def _build_identity_tab(self):
        tab = self.identity_tab
        tab.columnconfigure(1, weight=1)
        self.identity_vars = {key: tk.StringVar() for key in
                              ("physical_specimen_id", "analysis_mode", "width_mm", "thickness_mm",
                               "area_mm2", "dimension_source", "dimension_status")}
        self.acquisition_name_var = tk.StringVar(value="未选择批次")
        ttk.Label(tab, text="采集批次").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        ttk.Label(tab, textvariable=self.acquisition_name_var).grid(row=0, column=1, sticky="w", pady=3)
        fields = (("物理试样编号", "physical_specimen_id"), ("分析模式", "analysis_mode"),
                  ("宽度 mm", "width_mm"),
                  ("厚度 mm", "thickness_mm"), ("确认面积 mm²", "area_mm2"),
                  ("尺寸来源", "dimension_source"))
        for field_row, (label, key) in enumerate(fields, start=1):
            ttk.Label(tab, text=label).grid(row=field_row, column=0, sticky="w", padx=(0, 10), pady=3)
            if key == "analysis_mode":
                widget = self.analysis_mode_box = ttk.Combobox(
                    tab, textvariable=self.identity_vars[key],
                    values=("extensometer", "fullfield"), state="readonly", width=24)
            else:
                widget = ttk.Entry(tab, textvariable=self.identity_vars[key], width=36)
            widget.grid(row=field_row, column=1, sticky="ew", pady=3)
        self.dimension_confirmed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="尺寸已由记录确认；允许新计算使用面积",
                        variable=self.dimension_confirmed_var).grid(row=len(fields) + 1, column=1, sticky="w", pady=5)
        self.save_identity_button = ttk.Button(tab, text="保存身份与尺寸来源", command=self.save_identity)
        self.save_identity_button.grid(row=len(fields) + 2, column=1, sticky="w", pady=(6, 2))
        self.identity_note = tk.StringVar(value="物理试样编号仅在有证据确认关联时填写。")
        ttk.Label(tab, textvariable=self.identity_note, wraplength=650).grid(
            row=len(fields) + 3, column=0, columnspan=2, sticky="w", pady=(7, 0))

    def _build_roi_tab(self):
        tab = self.roi_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=0)
        controls = ttk.Frame(tab)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.propose_roi_button = ttk.Button(controls, text="建议 ROI", command=self.propose_rois)
        self.propose_roi_button.grid(row=0, column=0, padx=(0, 5))
        self.visualize_roi_button = ttk.Button(controls, text="图像窗口查看", command=self.show_campaign_image)
        self.visualize_roi_button.grid(row=0, column=1, padx=(0, 5))
        ttk.Label(controls, text="预览帧").grid(row=0, column=2, padx=(6, 3))
        self.preview_position_var = tk.StringVar(value="首帧")
        self.preview_position_box = ttk.Combobox(controls, textvariable=self.preview_position_var,
                                                values=("首帧", "中帧", "末帧"), width=8, state="readonly")
        self.preview_position_box.grid(row=0, column=3)
        self.preview_position_box.bind("<<ComboboxSelected>>", lambda _event: self.render_roi_preview())
        self.roi_preview_var = tk.StringVar(value="先选择批次并生成候选区域。")
        ttk.Label(controls, textvariable=self.roi_preview_var).grid(row=0, column=4, sticky="w", padx=8)
        free_seed = ttk.Frame(tab)
        free_seed.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(free_seed, text="手工自由段候选 [x,y,w,h]").pack(side=tk.LEFT, padx=(0, 4))
        self.free_region_seed_vars = []
        for label in ("x", "y", "w", "h"):
            ttk.Label(free_seed, text=label).pack(side=tk.LEFT, padx=(3, 1))
            var = tk.StringVar()
            entry = ttk.Entry(free_seed, textvariable=var, width=6)
            entry.pack(side=tk.LEFT)
            entry.bind("<KeyRelease>", lambda _event: self.render_roi_preview())
            self.free_region_seed_vars.append(var)
        ttk.Label(free_seed, text="留空则使用自动候选；尺寸不足或候选不可靠时，在首帧上填写后重新建议。",
                  wraplength=420).pack(side=tk.LEFT, padx=8)
        body = ttk.Panedwindow(tab, orient=tk.HORIZONTAL)
        body.grid(row=2, column=0, sticky="nsew")
        tab.rowconfigure(2, weight=1)
        canvas_frame = ttk.Frame(body)
        edit_frame = ttk.Frame(body)
        body.add(canvas_frame, weight=2)
        body.add(edit_frame, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.roi_canvas = tk.Canvas(canvas_frame, background="#18212b", height=300, highlightthickness=0)
        self.roi_canvas.grid(row=0, column=0, sticky="nsew")
        self.roi_canvas.bind("<Configure>", lambda _event: self.render_roi_preview())
        edit_frame.columnconfigure(1, weight=1)
        self.roi_entries: dict[tuple[str, int | None, str], list[tk.StringVar]] = {}
        edit_frame.rowconfigure(0, weight=1)
        self.roi_editor_canvas = tk.Canvas(edit_frame, height=245, highlightthickness=0)
        self.roi_editor_canvas.grid(row=0, column=0, sticky="nsew")
        editor_scroll = ttk.Scrollbar(edit_frame, orient="vertical", command=self.roi_editor_canvas.yview)
        editor_scroll.grid(row=0, column=1, sticky="ns")
        self.roi_editor_canvas.configure(yscrollcommand=editor_scroll.set)
        self.roi_editor_frame = ttk.Frame(self.roi_editor_canvas)
        self.roi_editor_window = self.roi_editor_canvas.create_window(
            (0, 0), window=self.roi_editor_frame, anchor="nw")
        self.roi_editor_frame.bind("<Configure>", self._resize_roi_editor_scroll)
        self.roi_editor_canvas.bind("<Configure>", self._resize_roi_editor_window)
        self.roi_diagnostics_var = tk.StringVar(value="")
        ttk.Label(edit_frame, textvariable=self.roi_diagnostics_var, wraplength=280, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        self.save_roi_button = ttk.Button(edit_frame, text="保存当前取点", command=self.save_roi_edits, state=tk.DISABLED)
        self.save_roi_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(3, 4))
        ttk.Separator(edit_frame).grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.reviewer_var = tk.StringVar()
        ttk.Label(edit_frame, text="审核人").grid(row=4, column=0, sticky="w")
        ttk.Entry(edit_frame, textvariable=self.reviewer_var, width=24).grid(row=4, column=1, sticky="ew")
        self.specimen_confirm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(edit_frame, text="已核实自由段属于此物理试样", variable=self.specimen_confirm_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=4)
        self.approve_roi_button = ttk.Button(edit_frame, text="确认 ROI 与试样身份", command=self.approve_rois,
                                             state=tk.DISABLED)
        self.approve_roi_button.grid(row=6, column=0, columnspan=2, sticky="ew")

    def _build_alignment_tab(self):
        tab = self.alignment_tab
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=1)
        ttk.Label(tab, text="载荷文件").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.load_path_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.load_path_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(tab, text="选择…", command=self.choose_load_file).grid(row=0, column=2, padx=(6, 0))
        self.time_column_var = tk.StringVar(value="timestamp_unix_s")
        self.force_column_var = tk.StringVar(value="force_N")
        ttk.Label(tab, text="载荷时间列").grid(row=1, column=0, sticky="w", pady=4)
        self.time_column_box = ttk.Combobox(tab, textvariable=self.time_column_var, width=28)
        self.time_column_box.grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="DIC 时间来源").grid(row=1, column=2, sticky="w", padx=(8, 4), pady=4)
        self.measurement_time_field_var = tk.StringVar(value="timestamp_unix_s")
        self.measurement_time_field_box = ttk.Combobox(
            tab, textvariable=self.measurement_time_field_var,
            values=("timestamp_unix_s", "estimated_timestamp_unix_s", "time_s"),
            state="readonly", width=28)
        self.measurement_time_field_box.grid(row=1, column=3, sticky="w", pady=4)
        ttk.Label(tab, text="载荷列").grid(row=2, column=0, sticky="w", pady=4)
        self.force_column_box = ttk.Combobox(tab, textvariable=self.force_column_var, width=28)
        self.force_column_box.grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="对齐方式").grid(row=3, column=0, sticky="w", pady=4)
        self.alignment_mode_var = tk.StringVar(value="instantaneous")
        self.alignment_mode_box = ttk.Combobox(
            tab, textvariable=self.alignment_mode_var, values=("instantaneous", "exposure_mean"),
            state="readonly", width=26)
        self.alignment_mode_box.grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="最大插值间隔 s").grid(row=4, column=0, sticky="w", pady=4)
        self.max_gap_var = tk.StringVar(value="2.5")
        ttk.Entry(tab, textvariable=self.max_gap_var, width=12).grid(row=4, column=1, sticky="w", pady=4)
        action_row = ttk.Frame(tab)
        action_row.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 3))
        self.preview_alignment_button = ttk.Button(action_row, text="预览（不写入）", command=self.preview_alignment)
        self.preview_alignment_button.pack(side=tk.LEFT, padx=(0, 6))
        self.align_button = ttk.Button(action_row, text="确认并对齐", command=self.align, state=tk.DISABLED)
        self.align_button.pack(side=tk.LEFT)
        self.alignment_preview = tk.Text(tab, height=12, wrap="none", state=tk.DISABLED)
        self.alignment_preview.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=(5, 0))
        tab.rowconfigure(6, weight=1)
        self._alignment_preview_key = None
        for variable in (self.load_path_var, self.time_column_var, self.force_column_var,
                         self.measurement_time_field_var, self.alignment_mode_var, self.max_gap_var):
            variable.trace_add("write", self._invalidate_alignment_preview)

    def _invalidate_alignment_preview(self, *_args):
        self._alignment_preview_key = None
        if hasattr(self, "align_button"):
            self.align_button.configure(state=tk.DISABLED)

    def _build_results_tab(self):
        tab = self.results_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=2)
        tab.rowconfigure(2, weight=1)
        actions = ttk.Frame(tab)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.run_button = ttk.Button(actions, text="运行 / 恢复本批", command=self.run_selected, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=(0, 5))
        self.run_pending_button = ttk.Button(actions, text="运行待处理批次", command=self.run_pending, state=tk.DISABLED)
        self.run_pending_button.pack(side=tk.LEFT)
        self.view_campaign_results_button = ttk.Button(actions, text="图像窗口查看", command=self.show_campaign_results)
        self.view_campaign_results_button.pack(side=tk.LEFT, padx=(6, 0))
        visual = ttk.Panedwindow(tab, orient=tk.HORIZONTAL)
        visual.grid(row=1, column=0, sticky="nsew", pady=(0, 5))
        plot_frame = ttk.LabelFrame(visual, text="Frame 应变 / 应力曲线")
        overlay_frame = ttk.LabelFrame(visual, text="关键帧叠图")
        visual.add(plot_frame, weight=2)
        visual.add(overlay_frame, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.result_canvas = tk.Canvas(plot_frame, background="#ffffff", height=230, highlightthickness=0)
        self.result_canvas.grid(row=0, column=0, sticky="nsew")
        self.result_canvas.bind("<Configure>", lambda _event: self._draw_result_curve())
        overlay_frame.columnconfigure(0, weight=1)
        overlay_frame.rowconfigure(0, weight=1)
        self.overlay_canvas = tk.Canvas(overlay_frame, background="#18212b", height=230, highlightthickness=0)
        self.overlay_canvas.grid(row=0, column=0, sticky="nsew")
        self.overlay_canvas.bind("<Configure>", lambda _event: self._redraw_result_overlay())
        self._result_plot_points = []
        self._result_plot_labels = ("frame", "value")
        self._result_overlay_photo = None
        self._result_overlay_source = None
        self.result_text = tk.Text(tab, height=8, wrap="word", state=tk.DISABLED)
        self.result_text.grid(row=2, column=0, sticky="nsew")
        self.result_scroll = ttk.Scrollbar(tab, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=self.result_scroll.set)
        self.result_scroll.grid(row=2, column=1, sticky="ns")

    # ---- project lifecycle -------------------------------------------------
    def new_project(self):
        parent = filedialog.askdirectory(title="选择 campaign 项目保存位置")
        if not parent:
            return
        name = simpledialog.askstring("新建 campaign", "项目名称：", parent=self.winfo_toplevel())
        if not name:
            return
        root = filedialog.askdirectory(title="选择一个数据根目录")
        if not root:
            return
        path = Path(parent) / self._safe_project_name(name)
        self.create_project_at(path, name, {"data": str(Path(root).resolve())})

    @staticmethod
    def _safe_project_name(name):
        value = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip(" .")
        return value or "campaign"

    def create_project_at(self, path, name, roots):
        path = Path(path)
        self._start_operation("正在创建 campaign 项目…",
                              lambda wf, progress, cancel: wf.init_project(path, name, roots),
                              on_success=lambda _result: self.open_project_at(path))

    def open_project_dialog(self):
        chosen = filedialog.askdirectory(title="选择含 project.json 的 campaign 项目目录")
        if chosen:
            self.open_project_at(chosen)

    def open_project_at(self, path):
        path = Path(path)
        self._start_operation("正在读取 campaign 项目…",
                              lambda wf, _progress, _cancel: wf.load_project(path),
                              on_success=lambda project: self._accept_project(path, project))

    def _accept_project(self, path, project):
        self.project_dir = Path(path).resolve()
        self.project = project or {"acquisitions": [], "roots": {}}
        self.project.setdefault("acquisitions", [])
        self.project.setdefault("roots", {})
        self.project_label_var.set(f"{self.project.get('name', self.project_dir.name)}  ·  {self.project_dir}")
        aliases = list(self.project["roots"])
        self.root_alias_box.configure(values=aliases)
        self._refresh_combobox_help(
            self.root_alias_box, {value: f"项目中登记的数据根目录别名“{value}”。" for value in aliases})
        if aliases and self.root_alias_var.get() not in aliases:
            self.root_alias_var.set(aliases[0])
        self.current_acquisition_id = None
        self._proposal = None
        self.refresh_table()
        self._set_current_acquisition(None)
        self._set_status(f"已打开项目：{len(self.project['acquisitions'])} 批次")
        self._update_action_states()

    def import_keystone_dialog(self):
        if not self._require_project():
            return
        source = filedialog.askdirectory(
            parent=self.winfo_toplevel(),
            title="选择 Keystone 正式版本文件夹或含 CURRENT.json 的版本根目录")
        if source:
            self._start_operation("正在只读导入 Keystone…",
                                  lambda wf, _progress, _cancel: wf.import_keystone(self.project_dir, source),
                                  refresh=True)

    def scan_root(self):
        if not self._require_project():
            return
        alias = self.root_alias_var.get().strip()
        if not alias or alias not in self.project.get("roots", {}):
            messagebox.showwarning("数据根目录", "请先在新建项目时登记根目录，或选择项目中已有的根目录别名。")
            return
        pattern = self.pattern_var.get().strip() or None
        adapter = self.adapter_var.get()
        self._start_operation("正在扫描批次…",
                              lambda wf, _progress, _cancel: wf.scan(
                                  self.project_dir, alias, adapter=adapter, pattern=pattern),
                              refresh=True)

    def export_dialog(self):
        if not self._require_project():
            return
        output = filedialog.askdirectory(title="选择项目导出目录")
        if output:
            self._start_operation("正在导出项目…",
                                  lambda wf, _progress, _cancel: wf.export(self.project_dir, output),
                                  refresh=True)

    def verify_project(self):
        if not self._require_project():
            return
        self._start_operation("正在核验项目…",
                              lambda wf, _progress, _cancel: wf.verify(self.project_dir),
                              on_success=self._show_verification)

    # ---- acquisition list and identity ------------------------------------
    def refresh_table(self):
        if not hasattr(self, "acquisition_tree"):
            return
        query = self.filter_var.get().strip().casefold()
        wanted_state = self.state_filter_var.get()
        for item in self.acquisition_tree.get_children(""):
            self.acquisition_tree.delete(item)
        self._filtered_ids = []
        for acq in self.project.get("acquisitions", []):
            aid = str(acq.get("acquisition_id", ""))
            name = str(acq.get("name", aid))
            if query and query not in (name + " " + aid + " " + str(acq.get("relative_path", ""))).casefold():
                continue
            state = self._acquisition_state(acq)
            if wanted_state != "全部" and not self._matches_state(wanted_state, acq, state):
                continue
            frames = acq.get("measurement_frames") or acq.get("image_frames") or []
            dic = "有" if acq.get("image_frames") else "无"
            dims = acq.get("dimensions") or {}
            area = dims.get("area_mm2")
            area_text = self._fmt(area) if area is not None else "—"
            self.acquisition_tree.insert("", "end", iid=aid, text=name,
                                         values=(state, len(frames), dic, area_text))
            self._filtered_ids.append(aid)
        self.batch_count_var.set(f"{len(self._filtered_ids)} / {len(self.project.get('acquisitions', []))} 批次")
        if self.current_acquisition_id in self._filtered_ids:
            self.acquisition_tree.selection_set(self.current_acquisition_id)
            self.acquisition_tree.see(self.current_acquisition_id)
        self._update_action_states()

    @staticmethod
    def _needs_reanalysis(acq):
        meta = acq.get("metadata") if isinstance(acq.get("metadata"), dict) else {}
        return bool(meta.get("workflow_reanalysis_required") or meta.get("workflow_result_stale"))

    @staticmethod
    def _matches_state(wanted, acq, state):
        if wanted == "待审核":
            return acq.get("approval") is None
        if wanted == "待分析":
            return acq.get("approval") is not None and (acq.get("result") is None or CampaignPage._needs_reanalysis(acq))
        if wanted == "已分析":
            return acq.get("result") is not None and "异常" not in state and "失败" not in state
        if wanted == "已对齐":
            return acq.get("alignment") is not None and "异常" not in state and "失败" not in state
        if wanted == "异常":
            return "异常" in state or "失败" in state
        return True

    @staticmethod
    def _acquisition_state(acq):
        bad_statuses = {"failed", "error", "cancelled", "canceled", "blocked", "scientific_gate_failed"}
        if acq.get("alignment"):
            status = str(acq["alignment"].get("status", "已对齐"))
            status_cf = status.casefold()
            if status_cf in bad_statuses or "fail" in status_cf or "error" in status_cf:
                return "异常"
            return "已对齐" if status in ("COMPLETE", "COMPLETED", "SUCCESS", "ALIGNED", "ok", "OK") else status
        if acq.get("result"):
            status = str(acq["result"].get("status", "已分析"))
            status_cf = status.casefold()
            if status_cf in bad_statuses or "fail" in status_cf or "error" in status_cf:
                return "异常"
            return "已分析" if status in ("COMPLETE", "COMPLETED", "SUCCESS", "ANALYZED", "ok", "OK") else status
        if acq.get("approval"):
            return "待分析"
        return "待审核"

    def _on_select_acquisition(self, _event=None):
        selection = self.acquisition_tree.selection()
        aid = selection[0] if selection else None
        if aid == self.current_acquisition_id:
            return
        self.current_acquisition_id = aid
        self._set_current_acquisition(self._find_acquisition(aid))
        self._update_action_states()

    def _find_acquisition(self, aid):
        if aid is None:
            return None
        return next((a for a in self.project.get("acquisitions", [])
                     if str(a.get("acquisition_id")) == str(aid)), None)

    def _set_current_acquisition(self, acq):
        if acq is None:
            self.acquisition_name_var.set("未选择批次")
            for var in self.identity_vars.values():
                var.set("")
            self.dimension_confirmed_var.set(False)
            self._proposal = None
            for var in self.free_region_seed_vars:
                var.set("")
            self._clear_roi_editor()
            self.roi_canvas.delete("all")
            self._set_text(self.result_text, "")
            self._result_plot_points = []
            self._draw_result_curve()
            self._set_result_overlay(None)
            self._clear_visual_window_preview()
            return
        dims = acq.get("dimensions") or {}
        self.acquisition_name_var.set(acq.get("name", ""))
        self.identity_vars["physical_specimen_id"].set(acq.get("physical_specimen_id") or "")
        self.identity_vars["analysis_mode"].set(acq.get("analysis_mode", "extensometer"))
        self.identity_vars["width_mm"].set(self._fmt(dims.get("width_mm")))
        self.identity_vars["thickness_mm"].set(self._fmt(dims.get("thickness_mm")))
        self.identity_vars["area_mm2"].set(self._fmt(dims.get("area_mm2")))
        self.identity_vars["dimension_source"].set(dims.get("source") or "")
        raw_dim_status = str(dims.get("status") or "unknown")
        self.identity_vars["dimension_status"].set(raw_dim_status)
        self.dimension_confirmed_var.set(raw_dim_status.strip().casefold() in _CONFIRMED_DIMENSION_STATUSES)
        self._proposal = acq.get("roi")
        for var, value in zip(self.free_region_seed_vars, (self._proposal or {}).get("free_region", [])):
            var.set(str(value))
        if self._proposal is None:
            for var in self.free_region_seed_vars:
                var.set("")
        self._show_roi_proposal()
        if not self._proposal:
            self.render_roi_preview()
        self._render_acquisition_result(acq)

    def save_identity(self):
        acq = self._selected_acquisition()
        if acq is None or not self._require_project():
            return
        try:
            width = self._optional_float(self.identity_vars["width_mm"].get())
            thickness = self._optional_float(self.identity_vars["thickness_mm"].get())
            area = self._optional_float(self.identity_vars["area_mm2"].get())
            source = self.identity_vars["dimension_source"].get().strip() or None
            confirmed = bool(self.dimension_confirmed_var.get())
            has_area_basis = area is not None or (width is not None and thickness is not None)
            if confirmed and (not has_area_basis or not source):
                raise ValueError("确认尺寸必须有正面积（或宽×厚）和明确来源；请先填写来源记录中的尺寸。")
            if area is not None and area <= 0 or width is not None and width <= 0 or thickness is not None and thickness <= 0:
                raise ValueError("宽度、厚度和面积必须为正数。")
            previous_status = str((acq.get("dimensions") or {}).get("status") or "").strip()
            if confirmed:
                dim_status = previous_status if previous_status.casefold() in _CONFIRMED_DIMENSION_STATUSES else "confirmed"
            else:
                dim_status = "candidate" if has_area_basis else "unknown"
            dimensions = {"width_mm": width, "thickness_mm": thickness, "area_mm2": area,
                          "status": dim_status,
                          "source": source}
            patch = {"physical_specimen_id": self.identity_vars["physical_specimen_id"].get().strip() or None,
                     "analysis_mode": self.identity_vars["analysis_mode"].get() or "extensometer",
                     "dimensions": dimensions}
        except (ValueError, TypeError) as exc:
            messagebox.showwarning("身份或尺寸无效", str(exc))
            return
        self._start_operation("正在保存身份信息…",
                              lambda wf, _progress, _cancel: wf.update_acquisition(
                                  self.project_dir, acq["acquisition_id"], patch), refresh=True)

    # ---- ROI review --------------------------------------------------------
    def propose_rois(self):
        acq = self._selected_acquisition()
        if acq is None or not self._require_project():
            return
        try:
            free_region = self._parse_optional_integer_rect(self.free_region_seed_vars)
        except ValueError as exc:
            messagebox.showwarning("自由段候选无效", str(exc))
            return
        target_aid = acq["acquisition_id"]
        self._start_operation("正在生成 ROI 候选…",
                              lambda wf, _progress, _cancel: wf.propose_rois(
                                  self.project_dir, target_aid, free_region=free_region),
                              refresh=True,
                              on_success=lambda proposal, aid=target_aid: self._accept_proposal(proposal, acquisition_id=aid))

    def _accept_proposal(self, proposal, acquisition_id=None):
        if acquisition_id is not None and self.current_acquisition_id != acquisition_id:
            self._set_status(f"批次 {acquisition_id} 的 ROI 候选已生成（当前已切换至其他批次）。")
            return
        self._proposal = proposal
        for var, value in zip(self.free_region_seed_vars, proposal.get("free_region", [])):
            var.set(str(value))
        self._show_roi_proposal()
        self._set_status("ROI 候选已生成；检查自由段、排除夹具并审核各端点窗口后再保存。")

    def _show_roi_proposal(self):
        self._clear_roi_editor()
        proposal = self._proposal
        if not proposal:
            self.roi_diagnostics_var.set("暂无 ROI 候选。")
            self.approve_roi_button.configure(state=tk.DISABLED)
            self.save_roi_button.configure(state=tk.DISABLED)
            return
        row = 0
        self._add_rect_editor("自由段", "free_region", None, proposal.get("free_region"), row)
        row += 1
        for i, exclusion in enumerate(proposal.get("exclusions", [])):
            self._add_rect_editor(f"排除区 {i + 1}", "exclusion", i, exclusion, row)
            row += 1
        for i, group in enumerate(proposal.get("groups", [])):
            ttk.Label(self.roi_editor_frame, text=str(group.get("name", f"G{i + 1:02d}")),
                      font=("TkDefaultFont", 9, "bold")).grid(row=row, column=0, columnspan=5, sticky="w", pady=(6, 1))
            row += 1
            for key in ("roi1", "roi2"):
                self._add_rect_editor(key.upper(), "group", i, group.get(key), row, subkey=key)
                row += 1
        diagnostics = proposal.get("diagnostics", {})
        self.roi_diagnostics_var.set(json.dumps(diagnostics, ensure_ascii=False, indent=2)[:1800]
                                     if diagnostics else f"参考图：{proposal.get('reference_path', '未记录')}\n"
                                                          f"参考帧：{proposal.get('reference_frame', '未记录')}\n"
                                                          f"Proposal：{proposal.get('proposal_hash', '未记录')}")
        self.save_roi_button.configure(state=tk.NORMAL)
        self.approve_roi_button.configure(state=tk.NORMAL if self._proposal_ready_for_approval() else tk.DISABLED)
        self.render_roi_preview()

    def _add_rect_editor(self, label, kind, index, rect, row, subkey=None):
        ttk.Label(self.roi_editor_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 4))
        values = list(rect or ["", "", "", ""])
        vars_ = []
        for j in range(4):
            var = tk.StringVar(value=self._fmt(values[j]) if j < len(values) else "")
            entry = ttk.Entry(self.roi_editor_frame, textvariable=var, width=6)
            entry.grid(row=row, column=j + 1, sticky="ew", padx=(0, 2))
            self._add_control_help(entry, f"{label} ROI 的 {('x', 'y', 'width', 'height')[j]} 像素坐标；必须填写非负整数，宽高为正数。")
            entry.bind("<KeyRelease>", lambda _event, k=kind, i=index, s=subkey: self._on_roi_rect_edited(k, i, s))
            vars_.append(var)
        self.roi_entries[(kind, index, subkey)] = vars_

    def _on_roi_rect_edited(self, kind, index, subkey):
        if kind == "free_region":
            values = self.roi_entries.get((kind, index, subkey), [])
            if len(values) == 4:
                for target, source in zip(self.free_region_seed_vars, values):
                    target.set(source.get())
        self.render_roi_preview()

    def _clear_roi_editor(self):
        for child in self.roi_editor_frame.winfo_children():
            child.destroy()
        self.roi_entries.clear()
        self.roi_editor_canvas.configure(scrollregion=(0, 0, 0, 0))

    def _resize_roi_editor_scroll(self, _event=None):
        box = self.roi_editor_canvas.bbox("all")
        if box:
            self.roi_editor_canvas.configure(scrollregion=box)

    def _resize_roi_editor_window(self, event):
        self.roi_editor_canvas.itemconfigure(self.roi_editor_window, width=event.width)

    def _proposal_from_editors(self):
        if not self._proposal:
            raise ValueError("请先生成 ROI 候选。")
        proposal = json.loads(json.dumps(self._proposal))
        for (kind, index, subkey), vars_ in self.roi_entries.items():
            values = [float(v.get()) for v in vars_]
            if len(values) != 4 or not all(math.isfinite(v) for v in values):
                raise ValueError("ROI 坐标必须是有限数。")
            if any(not value.is_integer() for value in values):
                raise ValueError("像素 ROI 使用整数 [x, y, width, height]；请填写整数坐标。")
            rect = [int(value) for value in values]
            if rect[0] < 0 or rect[1] < 0 or rect[2] <= 0 or rect[3] <= 0:
                raise ValueError("ROI 左上角不能为负值，宽高必须为正整数。")
            if kind == "free_region":
                previous_free = proposal.get("free_region")
                proposal["free_region"] = rect
                acq = self._selected_acquisition() or {}
                if str(acq.get("analysis_mode", "extensometer")) == "fullfield" or (
                    proposal.get("field_roi") is not None and proposal.get("field_roi") == previous_free
                ):
                    proposal["field_roi"] = rect
            elif kind == "exclusion":
                proposal["exclusions"][index] = rect
            elif kind == "group":
                proposal["groups"][index][subkey] = rect
        return proposal

    @staticmethod
    def _parse_optional_integer_rect(variables):
        values = [str(variable.get()).strip() for variable in variables]
        if not values or all(value == "" for value in values):
            return None
        if len(values) != 4 or any(value == "" for value in values):
            raise ValueError("自由段需完整填写整数 x、y、width、height。")
        try:
            rect = [int(value) for value in values]
        except ValueError as exc:
            raise ValueError("自由段像素坐标必须为整数。") from exc
        if rect[0] < 0 or rect[1] < 0 or rect[2] <= 0 or rect[3] <= 0:
            raise ValueError("自由段左上角不能为负值，宽高必须为正整数。")
        return rect

    def save_roi_edits(self):
        acq = self._selected_acquisition()
        if acq is None or not self._require_project():
            return
        try:
            proposal = self._proposal_from_editors()
        except (ValueError, TypeError) as exc:
            messagebox.showwarning("ROI 坐标无效", str(exc))
            return
        update_roi = getattr(self._get_workflow(), "update_roi", None)
        if update_roi is None:
            messagebox.showerror("工作流接口缺失", "当前 workflow 未提供安全的 update_roi 接口，无法保存手工取点。")
            return
        self._start_operation("正在核对参考图并保存 ROI…",
                              lambda wf, _progress, _cancel: wf.update_roi(
                                  self.project_dir, acq["acquisition_id"], proposal), refresh=True)

    def _proposal_ready_for_approval(self):
        p = self._proposal or {}
        if not (p.get("proposal_hash") and p.get("reference_sha256") and p.get("reference_path")):
            return False
        acq = self._selected_acquisition() or {}
        if str(acq.get("analysis_mode", "extensometer")) == "fullfield":
            return bool(p.get("field_roi") or p.get("free_region"))
        return bool(p.get("groups"))

    def approve_rois(self):
        acq = self._selected_acquisition()
        reviewer = self.reviewer_var.get().strip()
        confirmed = bool(self.specimen_confirm_var.get())
        if not reviewer:
            messagebox.showwarning("缺少审核人", "请填写实际审核人姓名。")
            return
        if not confirmed:
            messagebox.showwarning("需要确认试样身份", "请先确认自由段确实属于该物理试样，不能批量或自动批准。")
            return
        if acq is None or not self._require_project() or not self._proposal_ready_for_approval():
            messagebox.showwarning("缺少 ROI 候选", "请先生成并保存当前批次的 ROI 候选。")
            return
        if any(self._editor_rect_is_dirty()):
            messagebox.showwarning("ROI 尚未保存", "取点已修改，请先点击“保存当前取点”，再确认审核。")
            return
        self._start_operation("正在登记 ROI 审核…",
                              lambda wf, _progress, _cancel: wf.approve_rois(
                                  self.project_dir, acq["acquisition_id"], reviewer=reviewer,
                                  specimen_confirmed=confirmed), refresh=True)

    def _editor_rect_is_dirty(self):
        if not self._proposal:
            return [False]
        try:
            return [self._proposal_from_editors() != self._proposal]
        except Exception:
            return [True]

    def _load_preview_rgb(self, path):
        from PIL import Image

        file_path = Path(path)
        stat_info = file_path.stat()
        cache_key = (str(file_path), stat_info.st_mtime_ns, stat_info.st_size)
        cached = self._preview_image_cache.get(cache_key)
        if cached is not None:
            cached_image, cached_size = cached
            return cached_image.copy(), cached_size

        def _load_via_core():
            import ezdic_core as core

            raw = core.read_gray_image(file_path)
            bounds = core.compute_reference_normalization(raw)["reference_bounds"]
            norm8 = core.normalize_with_bounds(raw, bounds)
            rgb = Image.fromarray(norm8, mode="L").convert("RGB")
            return rgb, (int(raw.shape[1]), int(raw.shape[0]))

        if file_path.suffix.casefold() == ".cbf":
            image, original_size = _load_via_core()
        else:
            try:
                with Image.open(file_path) as original:
                    original_size = original.size
                    if original.mode in ("I", "I;16", "I;16B", "I;16L", "I;16S", "F"):
                        image, original_size = _load_via_core()
                    else:
                        image = original.convert("RGB")
            except Exception:
                image, original_size = _load_via_core()

        if len(self._preview_image_cache) >= 8:
            oldest_key = next(iter(self._preview_image_cache))
            self._preview_image_cache.pop(oldest_key, None)
        self._preview_image_cache[cache_key] = (image, original_size)
        return image.copy(), original_size

    def render_roi_preview(self):
        if not hasattr(self, "roi_canvas"):
            return
        acq = self._selected_acquisition()
        frames = (acq or {}).get("image_frames", [])
        if not frames:
            self.roi_canvas.delete("all")
            self.roi_preview_var.set("该批次没有可用 DIC 图像。")
            self._clear_visual_window_preview()
            return
        pos = self.preview_position_var.get()
        index = 0 if pos == "首帧" else (len(frames) - 1 if pos == "末帧" else len(frames) // 2)
        frame = frames[index]
        path = frame.get("path") if isinstance(frame, dict) else str(frame)
        try:
            from PIL import ImageTk

            image, original_size = self._load_preview_rgb(path)
            image.thumbnail((max(self.roi_canvas.winfo_width() - 8, 100),
                             max(self.roi_canvas.winfo_height() - 8, 100)))
            self._preview_photo = ImageTk.PhotoImage(image)
            self._preview_image_size = original_size
            self._preview_scale = image.width / self._preview_image_size[0]
            self.roi_canvas.delete("all")
            self.roi_canvas.create_image(4, 4, image=self._preview_photo, anchor="nw")
            self._draw_proposal_rectangles()
            frame_num = frame.get("frame", index + 1) if isinstance(frame, dict) else index + 1
            native_num = frame.get("native_frame", frame_num) if isinstance(frame, dict) else frame_num
            frame_label = f"frame {frame_num}" if native_num == frame_num else f"frame {frame_num} (native {native_num})"
            self.roi_preview_var.set(f"{pos} · {frame_label} · {self._preview_image_size[0]}×{self._preview_image_size[1]} px")
            self._render_visual_window_preview(path, frame_label, pos)
        except Exception as exc:
            self.roi_canvas.delete("all")
            self.roi_canvas.create_text(8, 8, anchor="nw", fill="#fca5a5",
                                        text=f"无法预览图像：{exc}", width=max(200, self.roi_canvas.winfo_width() - 16))
            self.roi_preview_var.set("预览失败；图像来源保留在项目记录中。")
            self._clear_visual_window_preview(f"无法预览图像：{exc}")

    def _draw_proposal_rectangles(self, canvas=None, scale=None, offset=(4, 4)):
        if not self._preview_image_size:
            return
        proposal = self._proposal or {}
        canvas = canvas or self.roi_canvas
        scale = self._preview_scale if scale is None else scale
        offset_x, offset_y = offset
        def draw(rect, color, label):
            if not rect or len(rect) != 4:
                return
            x, y, w, h = (float(v) for v in rect)
            x1, y1, x2, y2 = offset_x + x * scale, offset_y + y * scale, offset_x + (x + w) * scale, offset_y + (y + h) * scale
            canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            canvas.create_text(x1 + 3, y1 + 2, anchor="nw", fill=color, text=label)
        free_fallback = proposal.get("free_region")
        if not proposal and self.free_region_seed_vars:
            try:
                free_fallback = self._parse_optional_integer_rect(self.free_region_seed_vars)
            except ValueError:
                free_fallback = None
        draw(self._read_rect("free_region", None, None, free_fallback), "#22c55e", "自由段")
        for i, rect in enumerate(proposal.get("exclusions", [])):
            draw(self._read_rect("exclusion", i, None, rect), "#ef4444", f"排除{i + 1}")
        colors = ("#38bdf8", "#f472b6", "#facc15", "#a78bfa", "#fb923c")
        for i, group in enumerate(proposal.get("groups", [])):
            name = group.get("name", f"G{i + 1:02d}")
            draw(self._read_rect("group", i, "roi1", group.get("roi1")), colors[i % len(colors)], f"{name} A")
            draw(self._read_rect("group", i, "roi2", group.get("roi2")), colors[i % len(colors)], f"{name} B")

    def _render_visual_window_preview(self, path, frame_label, position):
        canvas = getattr(self.app, "campaign_preview_canvas", None) if self.app is not None else None
        if canvas is None:
            return
        try:
            from PIL import ImageTk

            image, size = self._load_preview_rgb(path)
            canvas_w = max(canvas.winfo_width(), 240)
            canvas_h = max(canvas.winfo_height(), 180)
            image.thumbnail((canvas_w - 8, canvas_h - 8))
            self._campaign_window_photo = ImageTk.PhotoImage(image)
            canvas.delete("all")
            canvas.create_image(4, 4, image=self._campaign_window_photo, anchor="nw")
            scale = image.width / size[0]
            self._draw_proposal_rectangles(canvas, scale, (4, 4))
            info_var = getattr(self.app, "campaign_preview_info_var", None)
            if info_var is not None:
                info_var.set(f"{position} · {frame_label} · {size[0]}×{size[1]} px · {path}")
            self._campaign_window_path = str(path)
        except Exception as exc:
            self._clear_visual_window_preview(f"无法预览图像：{exc}")

    def _clear_visual_window_preview(self, message="该批次没有 DIC 图像。"):
        if self.app is None:
            return
        canvas = getattr(self.app, "campaign_preview_canvas", None)
        if canvas is not None:
            canvas.delete("all")
            canvas.create_text(max(canvas.winfo_width(), 240) / 2,
                               max(canvas.winfo_height(), 180) / 2,
                               text=message, fill="#d1d5db")
        info_var = getattr(self.app, "campaign_preview_info_var", None)
        if info_var is not None:
            info_var.set(message)

    def show_campaign_image(self):
        if self.app is None or not hasattr(self.app, "workspace_notebook"):
            return
        self.app.show_visual_window()
        self.app.workspace_notebook.select(self.app.campaign_visual_page)
        self.render_roi_preview()

    def show_campaign_results(self):
        if self.app is None or not hasattr(self.app, "workspace_notebook"):
            return
        self.app.show_visual_window()
        self.app.workspace_notebook.select(self.app.campaign_result_visual_page)
        self._draw_result_curve()
        self._redraw_result_overlay()

    def _read_rect(self, kind, index, subkey, fallback):
        vars_ = self.roi_entries.get((kind, index, subkey))
        if not vars_:
            return fallback
        try:
            return [float(var.get()) for var in vars_]
        except (ValueError, TypeError):
            return fallback

    # ---- alignment ---------------------------------------------------------
    def choose_load_file(self):
        path = filedialog.askopenfilename(title="选择载荷 CSV", filetypes=(("CSV", "*.csv *.txt"), ("所有文件", "*.*")))
        if not path:
            return
        self.load_path_var.set(path)
        self._load_column_choices(path)

    def _workflow_helper(self, name):
        wf = self._get_workflow()
        fn = getattr(wf, name, None)
        if callable(fn):
            return fn
        return getattr(importlib.import_module("straintrace_campaign.workflow"), name)

    def _load_column_choices(self, path):
        try:
            header = self._workflow_helper("read_load_columns")(path)
            self.time_column_box.configure(values=header)
            self.force_column_box.configure(values=header)
            descriptions = {name: f"读取载荷文件中的“{name}”列；请核对列含义、单位和时钟来源。"
                            for name in header}
            self._refresh_combobox_help(self.time_column_box, descriptions)
            self._refresh_combobox_help(self.force_column_box, descriptions)
            for name in header:
                key = name.strip().casefold()
                if key in ("timestamp_unix_s", "time_unix_s", "timestamp", "time", "time_s"):
                    self.time_column_var.set(name)
                if key in ("force_n", "force", "load_n", "load"):
                    self.force_column_var.set(name)
        except Exception as exc:
            messagebox.showwarning("载荷文件读取失败", str(exc))

    @staticmethod
    def _format_preview_status(row_info, *, mode="instantaneous"):
        if row_info.get("alignable"):
            return "可对齐"
        status = str(row_info.get("status") or "")
        is_window = mode in {"window_mean", "exposure_mean", "exposure_window_mean", "window"}
        src_row = row_info.get("invalid_source_row")
        if status in {"OUTSIDE_TIME_RANGE", "NO_TIME_SUPPORT"}:
            return "曝光窗超出载荷时间范围" if is_window else "超出载荷时间范围"
        if status == "LONG_TIME_GAP":
            return "曝光窗内载荷间隔超限" if is_window else "间隔超限"
        if status in {
            "INVALID_EXACT_SAMPLE", "INVALID_BRACKET", "INVALID_LOAD_BRACKET",
            "UNKNOWN_FORCE_UNIT", "INVALID_FORCE_VALUE", "SOURCE_MARKED_FORCE_INVALID",
        }:
            prefix = "曝光窗内载荷支撑点无效" if is_window else "载荷支撑点无效"
            return f"{prefix}（第 {src_row} 行）" if src_row is not None else prefix
        if status in {"RELATIVE_CLOCK_DOMAIN_MISMATCH", "RELATIVE_CLOCK_ID_REQUIRED"}:
            return "相对时钟 ID 缺失或不一致"
        if status in {"RECORDED_ESTIMATED_CLOCK_MISMATCH", "TIME_CLOCK_DOMAIN_MISMATCH", "MIXED_LOAD_CLOCK_DOMAINS"}:
            return "测量与载荷时钟域不一致"
        if status == "RECORDED_TIMESTAMP_MISSING_ESTIMATE_SEPARATE":
            return "该行只有推算时间，不能填入绝对时间列"
        if status == "TIME_FIELD_MARKED_DERIVED":
            return "该时间列标记为推算值"
        if status == "EXPOSURE_WINDOW_MISSING":
            return "缺少曝光窗"
        if status == "INVALID_EXPOSURE_WINDOW":
            return "曝光窗无效"
        if status == "TIME_VALUE_MISSING":
            return "时间值缺失"
        return status or "缺少可信时间"

    @classmethod
    def _preview_tuples(cls, preview):
        mode = preview.get("mode", "instantaneous")
        tuples = []
        for item in preview.get("rows", []):
            query = item.get("query_time")
            left = item.get("left_time")
            right = item.get("right_time")
            tuples.append((
                item.get("frame"),
                f"{query:.6g}" if query is not None else "",
                f"{left:.6g}" if left is not None else "",
                f"{right:.6g}" if right is not None else "",
                cls._format_preview_status(item, mode=mode),
            ))
        return tuples

    def _run_alignment_preview(self, acq, max_gap):
        preview_fn = self._workflow_helper("preview_alignment")
        return preview_fn(
            acq,
            load_path=self.load_path_var.get(),
            project_dir=self.project_dir,
            mode=self.alignment_mode_var.get(),
            max_gap_s=max_gap,
            measurement_time_field=self.measurement_time_field_var.get(),
            load_time_column=self.time_column_var.get(),
            force_column=self.force_column_var.get(),
        )

    def _load_force_rows(self):
        # Thin read-only compatibility adapter backed by workflow.preview_alignment.
        load_path = Path(self.load_path_var.get()).expanduser()
        if not load_path.is_file():
            raise ValueError("请先选择可读取的载荷文件。")
        preview = self._run_alignment_preview({"acquisition_id": "preview", "measurement_frames": [], "image_frames": []}, 2.5)
        return [{"source_row": i + 2} for i in range(int(preview["load_row_count"]))]

    def _preview_alignment_rows(self, acq, load_rows, max_gap):
        if not load_rows:
            raise ValueError("载荷文件没有可用的有限时间—载荷记录。")
        preview = self._run_alignment_preview(acq, max_gap)
        return self._preview_tuples(preview)

    def preview_alignment(self):
        acq = self._selected_acquisition()
        if acq is None:
            messagebox.showwarning("未选择批次", "请先选择一个采集批次。")
            return
        try:
            max_gap = float(self.max_gap_var.get())
            if not math.isfinite(max_gap) or max_gap <= 0:
                raise ValueError("最大插值间隔必须为正数。")
            preview = self._run_alignment_preview(acq, max_gap)
            rows = self._preview_tuples(preview)
        except Exception as exc:
            messagebox.showwarning("无法预览对齐", str(exc))
            return
        time_field = self.measurement_time_field_var.get()
        lines = ["预览只读载荷文件与项目时间轴；此时尚未调用 align，也未写入项目。",
                 f"DIC 时间来源={time_field}；方式={self.alignment_mode_var.get()}，最大间隔={max_gap:g} s，载荷行={preview['load_row_count']}",
                 "frame\t源时间(s)\t左支撑(s)\t右支撑(s)\t状态"]
        lines.extend("\t".join(str(value) for value in row) for row in rows[:60])
        if len(rows) > 60:
            lines.append(f"… 另有 {len(rows) - 60} 帧未显示")
        self._set_text(self.alignment_preview, "\n".join(lines))
        self._alignment_preview_key = self._alignment_key(acq)
        self.align_button.configure(state=tk.NORMAL if preview["alignable_count"] > 0 else tk.DISABLED)
        self._set_status(f"只读预览完成：{preview['alignable_count']} / {len(rows)} 帧处于有效时间支持内")

    def align(self):
        acq = self._selected_acquisition()
        if acq is None or not self._require_project():
            return
        if self._alignment_preview_key != self._alignment_key(acq):
            messagebox.showwarning("需要重新预览", "载荷映射或对齐参数已变化，请先重新生成只读预览。")
            return
        try:
            max_gap = float(self.max_gap_var.get())
            if not math.isfinite(max_gap) or max_gap <= 0:
                raise ValueError("最大插值间隔必须为正数。")
            if not Path(self.load_path_var.get()).is_file():
                raise ValueError("请先选择载荷 CSV。")
        except (TypeError, ValueError) as exc:
            messagebox.showwarning("对齐参数无效", str(exc))
            return
        kwargs = {"mode": self.alignment_mode_var.get(), "max_gap_s": max_gap,
                  "load_time_column": self.time_column_var.get(), "force_column": self.force_column_var.get(),
                  "measurement_time_field": self.measurement_time_field_var.get()}
        project_dir = self.project_dir
        load_path = self.load_path_var.get()
        self._start_operation("正在按项目时间轴对齐载荷…",
                              lambda wf, progress, cancel: wf.align(
                                  project_dir, acq["acquisition_id"], load_path, **kwargs),
                              refresh=True)

    def _alignment_key(self, acq):
        return (acq.get("acquisition_id"), str(Path(self.load_path_var.get()).expanduser()),
                self.time_column_var.get(), self.force_column_var.get(), self.measurement_time_field_var.get(),
                self.alignment_mode_var.get(), self.max_gap_var.get())

    # ---- run and background operations ------------------------------------
    def run_selected(self):
        acq = self._selected_acquisition()
        if acq is None or not self._require_project():
            return
        if not acq.get("approval"):
            messagebox.showwarning("ROI 尚未审核", "请先检查并明确审核当前批次 ROI 和试样身份。")
            return
        self._start_operation("正在运行 / 恢复 DIC…",
                              lambda wf, progress, cancel: wf.run(
                                  self.project_dir, acq["acquisition_id"], progress=progress, cancel=cancel),
                              refresh=True, cancellable=True)

    def run_pending(self):
        if not self._require_project():
            return
        pending = [a["acquisition_id"] for a in self.project.get("acquisitions", [])
                   if a.get("approval") and (not a.get("result") or self._needs_reanalysis(a))]
        if not pending:
            messagebox.showinfo("没有待处理批次", "没有同时具备审核 ROI 且尚无结果或需重算的批次。")
            return

        project_dir = self.project_dir

        def run_pending_batches(wf, progress, cancel):
            outcomes = []
            for index, acquisition_id in enumerate(pending, start=1):
                if cancel is not None and cancel.is_set():
                    outcomes.append({"acquisition_id": acquisition_id, "status": "CANCELLED"})
                    break
                progress({"message": f"正在处理待处理批次 {index}/{len(pending)}：{acquisition_id}"})
                try:
                    result = wf.run(project_dir, acquisition_id, progress=progress, cancel=cancel)
                    status = result.get("status", "COMPLETE") if isinstance(result, dict) else "COMPLETE"
                    outcomes.append({"acquisition_id": acquisition_id, "status": status, "result": result})
                except Exception as exc:
                    if getattr(exc, "code", None) == "CAMPAIGN_CANCELLED":
                        outcomes.append({"acquisition_id": acquisition_id, "status": "CANCELLED"})
                        break
                    outcomes.append({"acquisition_id": acquisition_id, "status": "FAILED", "error": str(exc)})
            return outcomes

        self._start_operation("正在运行已审核的待处理批次…", run_pending_batches,
                              refresh=True, cancellable=True)

    def _start_operation(self, label, operation, refresh=False, on_success=None, cancellable=False):
        if self._busy:
            self._set_status("已有后台操作运行中。")
            return
        try:
            wf = self._get_workflow()
        except Exception as exc:
            messagebox.showerror("Campaign 工作流不可用", str(exc))
            return
        self._busy = True
        self._cancel_event = threading.Event() if cancellable else None
        self._set_status(label)
        self.progress.configure(mode="indeterminate")
        self.progress.start(10)
        self.cancel_button.configure(state=tk.NORMAL if cancellable else tk.DISABLED)
        self._set_operation_buttons(False)

        def report(*args, **kwargs):
            self._operation_queue.put(("progress", (args, kwargs)))

        def worker():
            try:
                cancel = self._cancel_event
                value = operation(wf, report, cancel)
                self._operation_queue.put(("done", (value, refresh, on_success, None)))
            except BaseException as exc:
                self._operation_queue.put(("done", (None, refresh, on_success, exc)))

        self._worker = threading.Thread(target=worker, name="straintrace-campaign", daemon=True)
        self._worker.start()

    def cancel_operation(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._set_status("已发送取消请求；等待当前安全检查点结束…")

    def _poll_queue(self):
        if self._closed:
            return
        try:
            while True:
                kind, payload = self._operation_queue.get_nowait()
                if kind == "progress":
                    self._show_progress(payload)
                elif kind == "done":
                    self._finish_operation(*payload)
        except queue.Empty:
            pass
        try:
            if self.winfo_exists() and self._scheduler.winfo_exists():
                self._poll_after_id = self._scheduler.after(100, self._poll_queue)
        except tk.TclError:
            self._poll_after_id = None

    def _show_progress(self, payload):
        args, kwargs = payload
        detail = kwargs.get("message")
        if detail is None and args:
            detail = args[0]
        if isinstance(detail, dict):
            fraction = detail.get("progress_fraction", detail.get("fraction"))
            percent = detail.get("progress_percent", detail.get("percent"))
            if fraction is None and percent is None and detail.get("total"):
                try:
                    fraction = float(detail.get("completed", 0)) / float(detail["total"])
                except (TypeError, ValueError, ZeroDivisionError):
                    fraction = None
            if percent is None and fraction is not None:
                try:
                    percent = float(fraction) * 100.0
                except (TypeError, ValueError):
                    percent = None
            if percent is not None:
                try:
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=max(0.0, min(100.0, float(percent))))
                except (tk.TclError, TypeError, ValueError):
                    pass
            detail = detail.get("message") or detail.get("status") or json.dumps(detail, ensure_ascii=False)
        if detail is not None:
            self._set_status(str(detail))

    def _finish_operation(self, value, refresh, on_success, error):
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._busy = False
        self._cancel_event = None
        self._worker = None
        self.cancel_button.configure(state=tk.DISABLED)
        self._set_operation_buttons(True)
        self._update_action_states()
        if error is not None:
            self._set_status(f"操作失败：{error}")
            self._show_result("操作失败", {"error": str(error)})
            return
        if refresh and self.project_dir:
            try:
                self.project = self._get_workflow().load_project(self.project_dir)
                self.project.setdefault("acquisitions", [])
                self.project.setdefault("roots", {})
                self._refresh_after_operation()
            except Exception as exc:
                self._set_status(f"操作已返回，但重新读取项目失败：{exc}")
        handled_success = on_success is not None
        if on_success is not None:
            try:
                on_success(value)
            except Exception as exc:
                self._set_status(f"操作完成，界面更新失败：{exc}")
        elif self._is_reportable_result(value):
            self._show_result("操作结果", value)
        if not handled_success:
            self._set_status("操作完成")
        self._update_action_states()

    @staticmethod
    def _is_reportable_result(value):
        if isinstance(value, dict):
            return any(key in value for key in ("status", "error", "errors", "failures", "ok"))
        if isinstance(value, list):
            return any(isinstance(item, dict) and any(k in item for k in ("status", "error", "errors"))
                       for item in value)
        return False

    def _refresh_after_operation(self):
        previous = self.current_acquisition_id
        self.refresh_table()
        acq = self._find_acquisition(previous)
        self._set_current_acquisition(acq)

    def _set_operation_buttons(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for name in ("new_button", "open_button", "import_button", "export_button", "verify_button",
                     "scan_button", "save_identity_button", "propose_roi_button", "visualize_roi_button",
                     "save_roi_button", "approve_roi_button",
                     "preview_alignment_button", "align_button", "run_button", "run_pending_button"):
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass

    def _update_action_states(self):
        has_project = self.project_dir is not None
        selected = self._selected_acquisition()
        has_selection = selected is not None
        state = tk.NORMAL if has_project and has_selection and not self._busy else tk.DISABLED
        can_run = bool(selected and selected.get("approval"))
        self.run_button.configure(state=tk.NORMAL if has_project and can_run and not self._busy else tk.DISABLED)
        pending = any(
            a.get("approval") and (not a.get("result") or self._needs_reanalysis(a))
            for a in self.project.get("acquisitions", [])
        )
        self.run_pending_button.configure(state=tk.NORMAL if has_project and pending and not self._busy else tk.DISABLED)
        project_state = tk.NORMAL if has_project and not self._busy else tk.DISABLED
        for widget in (self.import_button, self.export_button, self.verify_button, self.scan_button):
            widget.configure(state=project_state)
        self.save_identity_button.configure(state=state)
        has_dic = bool(self._selected_acquisition() and self._selected_acquisition().get("image_frames"))
        self.propose_roi_button.configure(state=tk.NORMAL if has_project and has_selection and has_dic and not self._busy else tk.DISABLED)
        self.visualize_roi_button.configure(state=tk.NORMAL if has_project and has_selection and has_dic and not self._busy else tk.DISABLED)
        has_proposal = bool(self._proposal)
        self.save_roi_button.configure(state=tk.NORMAL if has_project and has_selection and has_proposal and not self._busy else tk.DISABLED)
        approve_ready = has_proposal and self._proposal_ready_for_approval()
        self.approve_roi_button.configure(state=tk.NORMAL if has_project and has_selection and approve_ready and not self._busy else tk.DISABLED)
        self.preview_alignment_button.configure(state=state)
        if self._alignment_preview_key != (self._alignment_key(self._selected_acquisition()) if has_selection else None):
            self.align_button.configure(state=tk.DISABLED)

    def _show_result(self, title, value):
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str) if not isinstance(value, str) else value
        self._set_text(self.result_text, f"{title}\n{text}")
        self.detail_notebook.select(self.results_tab)

    def _render_acquisition_result(self, acq):
        pieces = [f"采集批次：{acq.get('name')}\n采集 ID：{acq.get('acquisition_id')}\n"
                  f"相对路径：{acq.get('relative_path')}\n物理试样 ID：{acq.get('physical_specimen_id') or 'null（未确认）'}"]
        for key in ("result", "alignment"):
            record = acq.get(key)
            pieces.append(f"\n{key}:\n" + (json.dumps(record, ensure_ascii=False, indent=2, default=str)
                                             if record else "尚无记录"))
        self._set_text(self.result_text, "\n".join(pieces))
        self._load_result_visuals(acq)

    def _show_verification(self, report):
        self._show_result("项目核验", report)
        if isinstance(report, dict):
            if "valid" in report:
                status = "通过" if report["valid"] else "未通过"
            else:
                status = report.get("status", report.get("ok", "完成"))
        else:
            status = "完成"
        self._set_status(f"项目核验：{status}")

    def _safe_project_artifact(self, raw_path):
        if not raw_path or not self.project_dir:
            return None
        candidate = Path(str(raw_path))
        if candidate.is_absolute():
            return candidate
        project_root = self.project_dir.resolve()
        resolved = (project_root / candidate).resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError:
            return None
        return resolved

    def _load_result_visuals(self, acq):
        stage = acq.get("alignment") or acq.get("result") or {}
        rows_path = stage.get("rows_path")
        rows = []
        candidate = self._safe_project_artifact(rows_path)
        if candidate is not None:
            try:
                with candidate.open("r", encoding="utf-8-sig") as stream:
                    rows = [json.loads(line) for line in stream if line.strip()]
            except (OSError, json.JSONDecodeError):
                rows = []
        x_label = "frame"
        y_candidates = ("engineering_strain", "engineering_strain_mean", "true_strain", "stress_MPa", "force_N")
        chosen_y = "engineering_strain"
        for field in y_candidates:
            if any(
                self._finite_number(row.get("frame")) is not None and self._finite_number(row.get(field)) is not None
                for row in rows
            ):
                chosen_y = field
                break
        y_label = chosen_y
        points = []
        for row in rows:
            x = self._finite_number(row.get("frame"))
            y = self._finite_number(row.get(chosen_y))
            if x is not None and y is not None:
                points.append((x, y))
        self._result_plot_points = points
        self._result_plot_labels = (x_label, y_label)
        self._draw_result_curve()
        overlay_path = self._find_overlay_path(stage)
        self._set_result_overlay(overlay_path)
        info_var = getattr(self.app, "campaign_result_info_var", None) if self.app is not None else None
        if info_var is not None:
            status = stage.get("status", "无阶段结果")
            overlay_text = overlay_path.name if overlay_path else "无叠图附件"
            info_var.set(f"{acq.get('name', acq.get('acquisition_id'))} · 阶段={status} · "
                         f"Frame 行={len(rows)} · 有限绘图点={len(points)} · "
                         f"y={y_label} · {overlay_text}")
        exceptions = []
        for row in rows:
            quality = str(row.get("quality_status", ""))
            strain_status = str(row.get("strain_status", ""))
            reasons = row.get("reasons", row.get("failure_reasons", []))
            has_reason = bool(reasons)
            adverse_status = any(
                token in (quality + " " + strain_status).casefold()
                for token in ("fail", "reject", "invalid", "missing", "ambiguous", "out_of_view", "insufficient", "no_valid")
            )
            if has_reason or adverse_status:
                exceptions.append({"frame": row.get("frame"), "native_cbf_frame": row.get("native_cbf_frame"),
                                   "quality_status": quality or None,
                                   "strain_status": strain_status or None, "reasons": reasons or None})
        if exceptions:
            detail = f"\n\n异常帧明细（显示前 {min(len(exceptions), 30)} / {len(exceptions)} 条）:\n"
            detail += json.dumps(exceptions[:30], ensure_ascii=False, indent=2, default=str)
            self._set_text(self.result_text, self.result_text.get("1.0", tk.END).rstrip() + detail)

    @staticmethod
    def _finite_number(value):
        if value is None or value == "" or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _find_overlay_path(self, stage):
        candidates = []
        def visit(value):
            if isinstance(value, str):
                if Path(value).suffix.casefold() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
                    candidates.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
        visit(stage.get("artifacts", []))
        visit(stage.get("preview_paths", []))
        for value in candidates:
            path = self._safe_project_artifact(value)
            if path is not None and path.is_file():
                return path
        return None

    def _draw_result_curve(self):
        if not hasattr(self, "result_canvas"):
            return
        canvases = [self.result_canvas]
        other = getattr(self.app, "campaign_result_canvas", None) if self.app is not None else None
        if other is not None and other not in canvases:
            canvases.append(other)
        for canvas in canvases:
            self._draw_curve_on_canvas(canvas)

    def _draw_curve_on_canvas(self, canvas):
        canvas.delete("all")
        width = max(canvas.winfo_width(), 260)
        height = max(canvas.winfo_height(), 180)
        margin = (52, 18, 18, 38)
        left, top, right, bottom = margin[0], margin[1], width - margin[2], height - margin[3]
        axis_fg = getattr(self.app, "text_color", "#111827") if self.app is not None else "#111827"
        muted_fg = getattr(self.app, "muted_color", "#4b5563") if self.app is not None else "#4b5563"
        canvas.create_line(left, bottom, right, bottom, fill=muted_fg, width=1)
        canvas.create_line(left, bottom, left, top, fill=muted_fg, width=1)
        x_label, y_label = self._result_plot_labels
        canvas.create_text((left + right) / 2, height - 10, text=x_label, fill=axis_fg)
        canvas.create_text(12, (top + bottom) / 2, text=y_label, angle=90, fill=axis_fg)
        points = self._result_plot_points
        if not points:
            canvas.create_text(width / 2, height / 2, text="暂无可绘制的有限结果", fill=muted_fg)
            return
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        if xmin == xmax:
            xmin, xmax = xmin - 0.5, xmax + 0.5
        if ymin == ymax:
            pad = max(abs(ymin) * 0.05, 0.5)
            ymin, ymax = ymin - pad, ymax + pad
        coords = []
        for x, y in points:
            px = left + (x - xmin) / (xmax - xmin) * (right - left)
            py = bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
            coords.extend((px, py))
        if len(points) > 1:
            canvas.create_line(*coords, fill="#2563eb", width=2, smooth=False)
        for i in range(0, len(coords), 2):
            canvas.create_oval(coords[i] - 2, coords[i + 1] - 2,
                               coords[i] + 2, coords[i + 1] + 2, fill="#2563eb", outline="")
        canvas.create_text(left, bottom + 15, text=f"{xmin:.4g}", anchor="w", fill=muted_fg)
        canvas.create_text(right, bottom + 15, text=f"{xmax:.4g}", anchor="e", fill=muted_fg)
        canvas.create_text(left - 5, bottom, text=f"{ymin:.4g}", anchor="e", fill=muted_fg)
        canvas.create_text(left - 5, top, text=f"{ymax:.4g}", anchor="e", fill=muted_fg)

    def _set_result_overlay(self, path):
        self._result_overlay_source = Path(path) if path else None
        self._redraw_result_overlay()

    def _redraw_result_overlay(self):
        if not hasattr(self, "overlay_canvas"):
            return
        canvases = [(self.overlay_canvas, "_result_overlay_photo")]
        other = getattr(self.app, "campaign_result_overlay_canvas", None) if self.app is not None else None
        if other is not None and all(other is not canvas for canvas, _attribute in canvases):
            canvases.append((other, "_campaign_result_overlay_photo"))
        for canvas, photo_attribute in canvases:
            self._draw_overlay_on_canvas(canvas, photo_attribute)

    def _draw_overlay_on_canvas(self, canvas, photo_attribute):
        canvas.delete("all")
        path = self._result_overlay_source
        if not path or not path.is_file():
            canvas.create_text(max(canvas.winfo_width(), 240) / 2,
                               max(canvas.winfo_height(), 160) / 2,
                               text="当前结果没有叠图附件", fill="#d1d5db")
            return
        try:
            from PIL import ImageTk

            image, size = self._load_preview_rgb(path)
            image.thumbnail((max(canvas.winfo_width() - 8, 120), max(canvas.winfo_height() - 28, 100)))
            photo = ImageTk.PhotoImage(image)
            setattr(self, photo_attribute, photo)
            canvas.create_image(4, 4, image=photo, anchor="nw")
            canvas.create_text(4, image.height + 8, anchor="nw", fill="#e5e7eb",
                               text=f"{path.name} · {size[0]}×{size[1]} px")
        except Exception as exc:
            canvas.create_text(8, 8, anchor="nw", fill="#fca5a5", text=f"无法显示叠图：{exc}")

    # ---- small helpers -----------------------------------------------------
    def _selected_acquisition(self):
        return self._find_acquisition(self.current_acquisition_id)

    def _require_project(self):
        if self.project_dir is None:
            messagebox.showwarning("尚未打开项目", "请先新建或打开 campaign 项目。")
            return False
        return True

    def _set_status(self, text):
        self.status_var.set(str(text))

    @staticmethod
    def _fmt(value):
        if value is None or value == "":
            return ""
        try:
            return format(float(value), ".12g")
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _optional_float(value):
        value = str(value).strip()
        if not value:
            return None
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("数值必须为有限数。")
        return result

    @staticmethod
    def _set_text(widget, value):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, value)
        widget.configure(state=tk.DISABLED)


__all__ = ["CampaignPage"]
