"""Embedded Matplotlib viewers for 1D extensometer, Poisson ratio, and 2D DIC fields."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

import ezdic_core as _core
from .tooltips import HelpNavigationToolbar

DIC_COMPONENT_LABELS = {
    "u": "u (px)",
    "v": "v (px)",
    "zncc": "ZNCC",
    "Exx": "Exx (Green-Lagrange)",
    "Eyy": "Eyy (Green-Lagrange)",
    "Exy": "Exy (Green-Lagrange)",
    "exx": "exx (infinitesimal)",
    "eyy": "eyy (infinitesimal)",
    "exy": "exy (infinitesimal)",
}
DIC_COMPONENT_HELP = {
    "u": "左右移动了多少个像素；正值向右，负值向左。选择后立即更新图。",
    "v": "上下移动了多少个像素；正值向下，负值向上。选择后立即更新图。",
    "zncc": "两个图片块的纹理相似程度，越接近 1 越相似；用于核对追踪质量。选择后立即更新图。",
    "Exx": "左右方向相对参考图片的伸缩，包含较大变形时的修正；正值伸长、负值缩短。选择后立即更新图。",
    "Eyy": "上下方向相对参考图片的伸缩，包含较大变形时的修正；正值伸长、负值缩短。选择后立即更新图。",
    "Exy": "剪切变形，即原本垂直的两个方向变斜的程度，包含较大变形时的修正；工程剪切值是此值的两倍。选择后立即更新图。",
    "exx": "按小变形近似计算的左右伸缩；与大写 Exx 使用不同计算方式。选择后立即更新图。",
    "eyy": "按小变形近似计算的上下伸缩；与大写 Eyy 使用不同计算方式。选择后立即更新图。",
    "exy": "按小变形近似计算的剪切变化，即两个垂直方向变斜的程度；工程剪切值是此值的两倍。选择后立即更新图。",
}


class ViewersMixin:
    """In-app interactive results viewers (strain curves, Poisson ratio, full-field DIC)."""

    def show_field_viewer(
        self,
        field,
        component=None,
        *,
        image=None,
        frame_1based=None,
        filename=None,
        reference_frame_1based=None,
        reference_filename=None,
    ):
        """Embed a DIC field and bind it to the image/frame that produced it."""
        if field is None:
            return
        previous_image = self.dic_last_image
        previous_frame = self.dic_last_frame_1based
        previous_filename = self.dic_last_filename
        previous_reference_frame = self.dic_last_reference_frame_1based
        previous_reference_filename = self.dic_last_reference_filename
        self.clear_viewer(keep_placeholder=False)
        self._canvas_mask_overlay_rgb = None
        self.viewer_frame.grid()
        self.viewer_content_frame.grid()
        self.dic_last_field = field
        provenance = dict(field.get("provenance") or {})
        if image is not None:
            self.dic_last_image = np.asarray(image).copy()
        elif previous_image is not None:
            self.dic_last_image = np.asarray(previous_image).copy()
        elif self.current_fullres_img8 is not None:
            # Backwards-compatible direct calls use the current image once;
            # subsequent component changes stay bound to this snapshot.
            self.dic_last_image = np.asarray(self.current_fullres_img8).copy()
        self.dic_last_frame_1based = frame_1based if frame_1based is not None else provenance.get("frame_global_1based", field.get("frame_global_1based", previous_frame))
        self.dic_last_filename = filename if filename is not None else provenance.get("frame_filename", field.get("frame_filename", previous_filename))
        self.dic_last_reference_frame_1based = (
            reference_frame_1based
            if reference_frame_1based is not None
            else provenance.get("reference_frame_1based", field.get("reference_frame_1based", previous_reference_frame))
        )
        self.dic_last_reference_filename = (
            reference_filename
            if reference_filename is not None
            else provenance.get("reference_filename", field.get("reference_filename", previous_reference_filename))
        )
        options = dict(field.get("display_options") or {})
        self.dic_view_background.set({"none": "无底图", "reference": "参考图", "deformed": "变形图"}.get(options.get("background", "none"), "无底图"))
        self.dic_color_mode.set({"range": "数据范围", "symmetric": "零点对称", "manual": "手动范围"}.get(options.get("color_mode", "range"), "数据范围"))
        self.dic_display_style.set("测量点" if options.get("style") == "points" else "连续云图")
        self.dic_percent.set(bool(options.get("percent", False)))
        self.dic_colormap.set(options.get("cmap", "RdBu_r"))
        self.dic_color_min.set("" if options.get("vmin") is None else str(options.get("vmin")))
        self.dic_color_max.set("" if options.get("vmax") is None else str(options.get("vmax")))
        if "deformed_image" not in field and self.dic_last_image is not None:
            field["deformed_image"] = self.dic_last_image
        self._viewer_kind = "fullfield"
        if component:
            self.dic_field_component.set(component)
        if self.dic_field_component.get() not in _core.DIC_FIELD_COMPONENTS:
            self.dic_field_component.set("u")
        self.results_df = None
        self.results_groups = None
        self._has_poisson = False
        self._update_field_viewer_context()
        self._rebuild_field_viewer_plot()
        self._add_field_viewer_controls()
        self.viewer_placeholder.grid_remove()
        self.viewer_export_btn.config(state=tk.NORMAL)
        self.viewer_clear_btn.config(state=tk.NORMAL)
        self._update_field_viewer_overlay()
        self.show_visual_window()
        self.workspace_notebook.select(self.results_page)
        self.log("已更新全场 DIC 色图预览（可切换 u/v/Exx/Eyy/Exy）。")

    def _update_field_viewer_context(self):
        frame = self.dic_last_frame_1based
        filename = self.dic_last_filename or "未知文件"
        ref_frame = self.dic_last_reference_frame_1based
        ref_filename = self.dic_last_reference_filename or "未知文件"
        frame_text = f"第 {frame} 帧" if frame is not None else "帧未知"
        ref_text = f"参考帧 {ref_frame}" if ref_frame is not None else "参考帧未知"
        self.field_viewer_context_var.set(f"{frame_text}：{filename}；{ref_text}：{ref_filename}")

    def _resolved_field_display_options(self, component):
        """Return one consistent display-options dict shared by plot, colorbar, and overlay."""
        stored = dict((self.dic_last_field or {}).get("display_options") or {})
        default_cmap = "viridis" if component == "zncc" else "RdBu_r"
        raw_cmap = stored.get("cmap") or default_cmap
        if component == "zncc" and raw_cmap == "RdBu_r":
            raw_cmap = "viridis"
        if raw_cmap not in matplotlib.colormaps:
            raw_cmap = default_cmap
        options = {
            "style": stored.get("style", "contour"),
            "background": stored.get("background", "none"),
            "color_mode": stored.get("color_mode", "range"),
            "percent": bool(stored.get("percent", False)),
            "cmap": raw_cmap,
            "alpha": float(stored.get("alpha", 0.65)),
        }
        if "vmin" in stored:
            options["vmin"] = stored["vmin"]
        if "vmax" in stored:
            options["vmax"] = stored["vmax"]
        if "coordinate_frame" in stored:
            options["coordinate_frame"] = stored["coordinate_frame"]
        return options

    def _update_field_viewer_overlay(self):
        if self.dic_last_field is None or self.dic_last_image is None:
            return
        try:
            component = str(self.dic_field_component.get() or "u")
            if component not in _core.DIC_FIELD_COMPONENTS:
                component = "u"
                self.dic_field_component.set(component)
            display_options = self._resolved_field_display_options(component)
            overlay_options = {**display_options, "coordinate_frame": "deformed", "background": "none"}
            overlay = _core.overlay_dic_field_on_image(
                self.dic_last_image,
                self.dic_last_field,
                component=component,
                cmap=display_options["cmap"],
                options=overlay_options,
            )
            self.auto_fit_enabled = False
            self.display_img = overlay
            self._canvas_shows_field_overlay = True
            _height, width = overlay.shape[:2]
            _orig_h, orig_w = np.asarray(self.dic_last_image).shape[:2]
            self.display_scale = width / max(orig_w, 1)
            self.zoom_factor = self.display_scale
            self.show_image()
        except Exception as exc:
            self.log(f"全场 overlay 更新失败：{exc}")

    def _rebuild_field_viewer_plot(self):
        if self.dic_last_field is None:
            return
        self._dispose_viewer_plot()

        fig = Figure(figsize=(7.0, 4.6), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)
        component = str(self.dic_field_component.get() or "u")
        if component not in _core.DIC_FIELD_COMPONENTS:
            component = "u"
        display_options = self._resolved_field_display_options(component)
        cmap = matplotlib.colormaps[display_options["cmap"]].copy()
        cmap.set_bad(self.panel_bg)
        mesh = _core.render_dic_field_on_axes(
            ax,
            self.dic_last_field,
            component,
            cmap=cmap,
            options=display_options,
        )
        unit = "px" if component in ("u", "v") else ("%" if display_options["percent"] and component != "zncc" else "1")
        # Keep the equal-aspect specimen centered and its colorbar adjacent,
        # including very tall ROIs in a wide independently resized window.
        cax = make_axes_locatable(ax).append_axes("right", size=0.18, pad=0.24)
        cbar = fig.colorbar(mesh, cax=cax)
        _core.style_dic_colorbar(cbar, _core.get_plot_preset("raw_inspection"), f"{component} ({unit})")
        ax.set_title(f"全场 {DIC_COMPONENT_LABELS.get(component, component)}")
        self._style_viewer_plot_fonts(ax)
        self._style_viewer_axes_dark(ax)
        ax.grid(False)
        cbar.ax.yaxis.label.set_color(self.text_color)
        cbar.ax.tick_params(colors=self.muted_color)
        self._attach_viewer_canvas(fig)

    def _add_field_viewer_controls(self):
        if hasattr(self, "_viewer_controls") and self._viewer_controls:
            self._viewer_controls.destroy()
        ctrl = ttk.Frame(self.viewer_frame, style="Card.TFrame")
        ctrl.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ctrl.columnconfigure(1, weight=1)
        self._viewer_controls = ctrl
        ttk.Label(ctrl, text="场分量", style="Hint.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.dic_component_box = ttk.Combobox(
            ctrl,
            textvariable=self.dic_field_component,
            values=list(_core.DIC_FIELD_COMPONENTS),
            width=8,
            state="readonly",
        )
        self.dic_component_box.grid(row=0, column=1, sticky="w")
        toolbar = ttk.Frame(ctrl, style="Card.TFrame")
        toolbar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        display_controls = (
            (self.dic_view_background, 8, "选择结果颜色下方显示哪张图片，再点“应用”更新。", {
                "无底图": "只显示测量结果颜色，便于看清数值分布。",
                "参考图": "以开始分析时的参考图片作底图，便于核对最初的测量位置。",
                "变形图": "以当前分析图片作底图，便于对照样品变形后的外观。",
            }),
            (self.dic_color_mode, 9, "选择颜色刻度的上下限如何确定，再点“应用”更新。", {
                "数据范围": "按当前数据的最小值和最大值分配颜色，容易看出这一张图里的差别。",
                "零点对称": "正负数使用同样大的颜色范围，便于比较伸长与缩短、向前与向后移动。",
                "手动范围": "使用下方填写的最小值和最大值，便于让多张结果图采用相同刻度。",
            }),
            (self.dic_display_style, 8, "选择结果画成颜色区域还是单个测量点，再点“应用”更新。", {
                "连续云图": "把有测量支持的位置画成连续的颜色区域，便于查看整体分布；空白处表示没有有效结果。",
                "测量点": "逐个显示实际计算的点，便于查看测量点分布和未能计算的位置。",
            }),
            (self.dic_colormap, 8, "选择从小值到大值使用哪些颜色，再点“应用”更新。具体数值请对照图旁的颜色刻度。", {
                "RdBu_r": "从蓝到白再到红，便于区分负值、接近零和正值；需结合颜色刻度判断。",
                "viridis": "从深紫经绿色到黄色，通常用更亮的颜色表示更大的值。",
                "cividis": "从深蓝到黄色，颜色和亮度一起变化，便于区分不同数值。",
                "turbo": "从深蓝经青、绿、黄到红，色彩变化较多，需对照刻度读取数值。",
            }),
        )
        for column, (variable, width, tip, choices) in enumerate(display_controls):
            combo = ttk.Combobox(toolbar, textvariable=variable, values=list(choices), width=width, state="readonly")
            combo.grid(row=0, column=column, padx=(0, 5), sticky="w")
            self.add_tooltip(combo, tip, choices=choices)
        scales = ttk.Frame(ctrl, style="Card.TFrame")
        scales.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(scales, text="色标").grid(row=0, column=0, sticky="w", padx=(0, 5))
        for column, variable in enumerate((self.dic_color_min, self.dic_color_max), start=1):
            entry = ttk.Entry(scales, textvariable=variable, width=9)
            entry.grid(row=0, column=column, padx=(0, 5))
            limit = "最小" if column == 1 else "最大"
            self.add_tooltip(entry, f"填写颜色刻度的{limit}值，只在“手动范围”下生效，填好后点“应用”。最小值必须小于最大值；移动量用像素，伸缩值勾选“应变 %”后用百分数。")
        percent_check = ttk.Checkbutton(scales, text="应变 %", variable=self.dic_percent)
        percent_check.grid(row=0, column=3, padx=(0, 6))
        self.add_tooltip(percent_check, "勾选后，伸缩值按百分数显示，例如 0.01 显示为 1%。取消后显示原始比例；点击“应用”更新，移动量仍以像素计。")
        apply_button = ttk.Button(scales, text="应用", command=self.apply_dic_display_options, style="Compact.TButton")
        apply_button.grid(row=0, column=4)
        self.add_tooltip(apply_button, "点击用所选底图、颜色范围、绘图方式和单位重新显示当前结果，同时更新图像页面的叠加预览。这里只调整显示，不重新计算数据。")
        display_button = ttk.Button(ctrl, style="Compact.TButton")
        display_button.grid(row=0, column=2, sticky="e")

        def toggle_display_settings():
            self._dic_display_expanded = not getattr(self, "_dic_display_expanded", False)
            update_display_settings()

        def update_display_settings():
            expanded = getattr(self, "_dic_display_expanded", False)
            display_button.configure(text="收起显示设置" if expanded else "显示设置")
            for frame in (toolbar, scales):
                frame.grid() if expanded else frame.grid_remove()

        display_button.configure(command=toggle_display_settings)
        self.add_tooltip(display_button, "展开或收起底图、颜色刻度、百分数和绘图方式的设置，给结果图留出更多查看空间。")
        update_display_settings()
        ttk.Label(
            ctrl,
            textvariable=self.field_viewer_context_var,
            style="Hint.TLabel",
            wraplength=360,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        valid = np.asarray(self.dic_last_field.get("valid", []), dtype=bool)
        strain_valid = np.asarray(self.dic_last_field.get("strain_valid", []), dtype=bool)
        eligible_count = int(np.count_nonzero(self.dic_last_field.get("eligible", np.ones(valid.size))))
        gauge = self.dic_last_field.get("strain_gauge_span_px", "未知")
        summary = f"位移有效 {np.count_nonzero(valid)}/{eligible_count} · 应变有效 {np.count_nonzero(strain_valid)}/{eligible_count} · 应变窗口跨度 {gauge} px · 空白表示无有效结果"
        summary_label = ttk.Label(ctrl, text=summary, style="Hint.TLabel", wraplength=580)
        summary_label.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.add_tooltip(summary_label, "找到对应纹理后，还要利用周围足够多、分布合适的测量点计算伸缩，所以移动量与伸缩量的有效数量可能不同。空白或 NaN 表示没有有效数值，不能当作零。")
        ctrl.bind("<Configure>", lambda event: self._wrap_result_labels(ctrl, event.width))
        self.dic_component_box.bind("<<ComboboxSelected>>", self._on_field_component_change)
        self.add_tooltip(
            self.dic_component_box,
            "选择要查看哪一种移动、伸缩或纹理相似程度，选择后立即更新结果图和图像页面的叠加预览。",
            choices=DIC_COMPONENT_HELP,
        )

    def _on_field_component_change(self, _event=None):
        self._rebuild_field_viewer_plot()
        self._update_field_viewer_overlay()

    def apply_dic_display_options(self):
        if self.dic_last_field is None:
            return
        try:
            options = self.dic_display_options()
            self.dic_last_field["display_options"] = options
            self._rebuild_field_viewer_plot()
            self._update_field_viewer_overlay()
        except Exception as exc:
            messagebox.showerror("显示设置无效", str(exc))

    def show_results_viewer(self, df, groups):
        """Embed an interactive matplotlib figure showing engineering strain (and optionally Poisson)."""
        if df is None or df.empty:
            return

        # Destroy previous canvas if any
        self.clear_viewer(keep_placeholder=False)
        self.viewer_frame.grid()
        self.viewer_content_frame.grid()

        self.results_df = df.copy()
        self.results_groups = groups
        self._viewer_kind = "extensometer"
        self.dic_last_field = None

        # 检测是否具备泊松比数据（至少有一组 axial + 一组 transverse）
        self._has_poisson = self._detect_poisson_capable(groups, df)

        # 默认显示工程应变
        self._viewer_mode = "strain"  # "strain" or "poisson"

        self._rebuild_viewer_plot()

        # 控制区：模式切换 + 导出
        self._add_viewer_controls()

        # Hide placeholder, enable buttons
        self.viewer_placeholder.grid_remove()
        self.viewer_export_btn.config(state=tk.NORMAL)
        self.viewer_clear_btn.config(state=tk.NORMAL)
        self.show_visual_window()
        self.workspace_notebook.select(self.results_page)

        self.log("已更新应用内结果曲线预览（支持缩放、平移、泊松比切换）。")

    def _detect_poisson_capable(self, groups, df):
        if not groups or df is None or df.empty:
            return False
        try:
            if not _core.validate_poisson_role_groups(groups):
                return False
        except RuntimeError:
            return False
        if "AxialEngineeringStrain" in df.columns or "PoissonRatio" in df.columns:
            return True
        if "group" not in df.columns or "engineering_strain" not in df.columns:
            return False
        axial_groups, transverse_groups = _core.get_poisson_role_groups(groups)
        present = set(df["group"].dropna().astype(str))
        has_axial_rows = any(str(g.get("name", "")) in present for g in axial_groups)
        has_trans_rows = any(str(g.get("name", "")) in present for g in transverse_groups)
        return has_axial_rows and has_trans_rows

    def _rebuild_viewer_plot(self):
        if getattr(self, "_viewer_kind", "extensometer") == "fullfield":
            self._rebuild_field_viewer_plot()
            return
        if self.results_df is None or self.results_df.empty:
            return
        self._dispose_viewer_plot()
        df = self.results_df
        groups = self.results_groups or []
        fig = Figure(figsize=(7.0, 4.6), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)
        colors = ["#56b4e9", "#e69f00", "#63d2ab", "#df9bd0", "#cbd5e1"] if self.dark_mode.get() else _core.PLOT_COLOR_CYCLE
        if self._viewer_mode == "poisson" and self._has_poisson:
            try:
                if "PoissonRatio" in df.columns and "group" not in df.columns:
                    frame_col = "Frame" if "Frame" in df.columns else _core._frame_column(df)
                    table = pd.DataFrame(
                        {
                            "Frame": pd.to_numeric(df[frame_col], errors="coerce"),
                            "PoissonRatio": pd.to_numeric(df["PoissonRatio"], errors="coerce"),
                        }
                    )
                else:
                    _core.validate_poisson_role_groups(groups)
                    table = _core.build_poisson_ratio_table(df, groups)
            except RuntimeError as exc:
                ax.text(0.5, 0.5, f"无法计算泊松比：{exc}", ha="center", va="center", transform=ax.transAxes)
                table = None
            if table is not None and not table.empty:
                frame = table["Frame"].astype(float)
                ratio = table["PoissonRatio"].astype(float)
                ax.plot(frame, ratio, color=colors[2], linewidth=1.7, label="泊松比")
                invalid = ~np.isfinite(ratio)
                if invalid.any():
                    ax.scatter(frame[invalid], np.full(int(invalid.sum()), 0.04), transform=ax.get_xaxis_transform(),
                               marker="x", color=self.warning_color, s=24, label="无效帧 (NaN)")
            ax.set_ylabel("泊松比 (无量纲)")
            ax.set_title("泊松比")
        else:
            for idx, group in enumerate(groups):
                gdf = df[df["group"] == group["name"]]
                if gdf.empty:
                    continue
                frame = gdf["frame_global_1based"].astype(float)
                strain = gdf["engineering_strain"].astype(float)
                color = colors[idx % len(colors)]
                # Keep NaN gaps; the x marks use axes coordinates, never a
                # measured zero, so rejected frames remain visually distinct.
                ax.plot(frame, strain, color=color, linewidth=1.7, label=group["name"])
                invalid = ~np.isfinite(strain)
                if invalid.any():
                    ax.scatter(frame[invalid], np.full(int(invalid.sum()), 0.04), transform=ax.get_xaxis_transform(),
                               marker="x", color=color, s=24, label=f"{group['name']} 无效帧")
            ax.set_ylabel("工程应变 (无量纲)")
            ax.set_title("工程应变")
        ax.set_xlabel("帧号")
        if ax.lines:
            ax.legend(loc="best", fontsize=self.viewer_legend_font_size)
        else:
            ax.text(0.5, 0.5, "无可显示的有效应变数据", ha="center", va="center", transform=ax.transAxes)
        self._style_viewer_plot_fonts(ax)
        self._style_viewer_axes_dark(ax)
        self._attach_viewer_canvas(fig)

    def _attach_viewer_canvas(self, fig):
        self.viewer_figure = fig
        self.viewer_canvas = FigureCanvasTkAgg(fig, master=self.viewer_content_frame)
        # Establish DPI before grid allocates the widget.  A later Map event
        # can otherwise enlarge the backing figure without a Configure event
        # when grid keeps the widget at its existing size (cropping colorbars).
        self.viewer_canvas._update_device_pixel_ratio()
        self.viewer_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.viewer_toolbar = HelpNavigationToolbar(self.viewer_canvas, self.viewer_content_frame, pack_toolbar=False)
        self.viewer_toolbar.subplot_help = self._add_subplot_help
        self.viewer_toolbar.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        toolbar_help = {
            "Home": "点击恢复结果图最初的查看范围，放大或移动后可用它返回全图。",
            "Back": "点击返回上一次查看范围；只有放大或移动过结果图后才有可返回的记录。",
            "Forward": "点击重回刚才返回前的查看范围；需要先使用“返回上一次”。",
            "Pan": "点击开启移动模式，再在图内按住左键拖动，移动查看范围；右键拖动可缩放坐标轴。再点一次退出。",
            "Zoom": "点击开启框选放大，再在图内按住左键拖出想看的范围；右键拖框可缩小。再点一次退出，房子图标可恢复全图。",
            "Subplots": "点击打开图表边距和图间距调节窗口。当前结果图使用自动布局，手动间距设置可能不生效。",
            "Save": "点击选择位置和文件格式，保存当前结果图，包含现在的坐标范围和颜色刻度。",
        }
        for name, button in self.viewer_toolbar._buttons.items():
            button.configure(takefocus=True)
            self.add_tooltip(button, toolbar_help[name])
        self.add_tooltip(self.viewer_canvas.get_tk_widget(), self._result_plot_help)
        self.add_tooltip(self.viewer_toolbar._message_label, "鼠标移到图中时，这里显示当前位置的坐标或数值；可对照坐标轴和颜色刻度读取。")
        self._style_viewer_toolbar()
        self.viewer_canvas.draw()

    def _result_plot_help(self, _event=None):
        mode = self.viewer_toolbar.mode.name
        if mode == "PAN":
            return "移动模式：按住左键拖动查看范围，右键拖动缩放坐标轴。再次点击工具栏的移动图标退出，房子图标恢复全图。"
        if mode == "ZOOM":
            return "框选模式：按住左键拖出要放大的范围，右键拖框缩小。再次点击放大镜图标退出，房子图标恢复全图。"
        return (
            "这是移动量或伸缩量的彩色结果图，数值与单位在图旁的颜色刻度上。空白表示未算出有效结果；用下方图标移动、放大或保存。"
            if self._viewer_kind == "fullfield" else
            "横轴是图片序号，纵轴是伸缩比例或泊松比（横向收缩与纵向伸长的比值）。曲线断开表示没有有效结果；用下方图标移动、放大或保存。"
        )

    def _add_subplot_help(self, tool, window):
        canvas = tool.figure.canvas.get_tk_widget()
        if hasattr(canvas, "_tooltip"):
            return
        window.title("图表边距与间距")
        tool.figure._suptitle.set_text("点击或拖动滑块调整；悬停查看说明")
        controls = (
            (tool.sliderleft, "左边界", "向右拖动会增加图表左侧留白。数值是边界位置相对整个图宽的比例。"),
            (tool.sliderbottom, "下边界", "向右拖动会增加图表下方留白。数值是边界位置相对整个图高的比例。"),
            (tool.sliderright, "右边界", "向左拖动会增加图表右侧留白。数值是边界位置相对整个图宽的比例。"),
            (tool.slidertop, "上边界", "向左拖动会增加图表上方留白。数值是边界位置相对整个图高的比例。"),
            (tool.sliderwspace, "左右间距", "向右拖动会增加并排图表之间的距离；只有多列图表时有明显作用。"),
            (tool.sliderhspace, "上下间距", "向右拖动会增加上下图表之间的距离；只有多行图表时有明显作用。"),
        )
        regions = []
        for slider, name, tip in controls:
            slider.label.set_text(name)
            regions.append((slider.ax, tip))
        tool.buttonreset.label.set_text("恢复初始值")
        regions.append((tool.buttonreset.ax, "点击恢复打开这个窗口时的边距和间距设置。"))

        def help_at(event=None):
            if event is not None and getattr(event, "keysym", "") != "F1":
                for axes, tip in regions:
                    if axes.get_window_extent().contains(event.x, canvas.winfo_height() - event.y):
                        return tip + "当前结果图使用自动布局，手动设置可能不生效。"
            return "点击或拖动滑块调整图表留白，数值是位置或间距的比例。“恢复初始值”重置本窗口的修改；自动布局时，手动设置可能不生效。"

        self.add_tooltip(canvas, help_at)
        tool.figure.canvas.draw_idle()

    def _dispose_viewer_plot(self):
        """Release Tk images and variables on the UI thread, before new work."""
        canvas = self.viewer_canvas
        toolbar = self.viewer_toolbar
        figure = self.viewer_figure
        if canvas is not None:
            for attr in ("_idle_draw_id", "_event_loop_id"):
                job = getattr(canvas, attr, None)
                if job is not None:
                    self.root.after_cancel(job)
                    setattr(canvas, attr, None)
        if toolbar is not None:
            if hasattr(toolbar, "subplot_tool"):
                toolbar.close_subplots()
            toolbar.destroy()
            for button in toolbar._buttons.values():
                # Tooltips and callbacks can retain destroyed button objects.
                # Release their Tcl variables/images before a worker's GC.
                for attr in ("var", "_ntimage", "_ntimage_alt"):
                    if hasattr(button, attr):
                        setattr(button, attr, None)
            toolbar._buttons.clear()
            # The cached message label points back to its toolbar through
            # master; break that cycle and release native resources now.
            toolbar._message_label = None
            toolbar.message = None
            toolbar._label_font = None
            toolbar.canvas = None
        if canvas is not None:
            canvas.get_tk_widget().destroy()
            canvas.toolbar = None
            canvas._tkphoto = None
        if figure is not None:
            figure.set_canvas(None)
        self.viewer_toolbar = None
        self.viewer_canvas = None
        self.viewer_figure = None

    def _style_viewer_toolbar(self):
        def recolor(widget):
            if isinstance(widget, (tk.Frame, tk.Label, tk.Button, tk.Checkbutton)):
                widget.configure(background=self.card_bg)
                if isinstance(widget, (tk.Label, tk.Button, tk.Checkbutton)):
                    widget.configure(foreground=self.text_color)
                if isinstance(widget, (tk.Button, tk.Checkbutton)):
                    widget.configure(activebackground=self.hover_bg, activeforeground=self.text_color)
                if isinstance(widget, tk.Checkbutton):
                    widget.configure(selectcolor=self.panel_bg)
                if isinstance(widget, (tk.Button, tk.Checkbutton)) and getattr(widget, "_image_file", None):
                    self.viewer_toolbar._set_image_for_button(widget)
            for child in widget.winfo_children():
                recolor(child)

        recolor(self.viewer_toolbar)

    def _wrap_result_labels(self, frame, width):
        for child in frame.winfo_children():
            if isinstance(child, ttk.Label) and child.cget("wraplength"):
                child.configure(wraplength=max(200, width - 20))

    def _style_viewer_plot_fonts(self, ax):
        ax.tick_params(labelsize=self.viewer_tick_font_size)
        ax.xaxis.label.set_size(self.viewer_label_font_size)
        ax.yaxis.label.set_size(self.viewer_label_font_size)
        ax.title.set_size(self.viewer_title_font_size)
        for text in ax.texts:
            text.set_fontsize(max(text.get_fontsize(), self.viewer_label_font_size))
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(self.viewer_legend_font_size)

    def _style_viewer_axes_dark(self, ax):
        """Use the same surface and typography tokens in either theme."""
        ax.figure.patch.set_facecolor(self.card_bg)
        ax.set_facecolor(self.card_bg)
        ax.tick_params(colors=self.muted_color, labelsize=self.viewer_tick_font_size)
        for spine in ax.spines.values():
            spine.set_color(self.border_color)
            spine.set_linewidth(0.8)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)
        ax.grid(True, alpha=0.3, color=self.border_color, linewidth=0.7)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(self.card_bg)
            legend.get_frame().set_edgecolor(self.border_color)
            for text in legend.get_texts():
                text.set_color(self.text_color)

    def _add_viewer_controls(self):
        """在 viewer 内部添加模式切换控件（工程应变 / 泊松比）"""
        if hasattr(self, "_viewer_controls") and self._viewer_controls:
            self._viewer_controls.destroy()

        ctrl = ttk.Frame(self.viewer_frame, style="Card.TFrame")
        ctrl.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ctrl.columnconfigure(1, weight=1)
        self._viewer_controls = ctrl

        ttk.Label(ctrl, text="显示", style="Hint.TLabel").grid(row=0, column=0, padx=(0, 8))

        self.viewer_mode_var = tk.StringVar(value=self._viewer_mode)

        def switch_mode():
            new_mode = self.viewer_mode_var.get()
            if new_mode != self._viewer_mode:
                self._viewer_mode = new_mode
                self._rebuild_viewer_plot()

        strain_rb = ttk.Radiobutton(
            ctrl, text="工程应变", variable=self.viewer_mode_var, value="strain",
            command=switch_mode
        )
        strain_rb.grid(row=0, column=1, sticky="w", padx=4)
        self.add_tooltip(strain_rb, "点击显示各组的距离伸缩曲线。工程应变是距离变化除以初始距离，例如 0.01 表示伸长 1%；没有有效结果的图片处，曲线会断开。")

        poisson_rb = ttk.Radiobutton(
            ctrl, text="泊松比", variable=self.viewer_mode_var, value="poisson",
            command=switch_mode,
            state=tk.NORMAL if self._has_poisson else tk.DISABLED
        )
        self.viewer_poisson_rb = poisson_rb
        poisson_rb.grid(row=0, column=2, sticky="w", padx=4)
        self.add_tooltip(poisson_rb, lambda _event: (
            "点击显示横向收缩与纵向伸长的比值，称为泊松比。没有有效结果的位置会留空。"
            if self._has_poisson else
            "泊松比表示横向收缩与纵向伸长的比值。先同时添加“拉伸方向”和“横向方向”两类测量组，再重新分析，这里才能切换。"))

        if not self._has_poisson:
            ttk.Label(ctrl, text="泊松比需要同时定义轴向和横向 ROI 组。", style="Hint.TLabel", wraplength=560).grid(
                row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ctrl.bind("<Configure>", lambda event: self._wrap_result_labels(ctrl, event.width))

    def export_viewer_figure(self):
        if self.viewer_figure is None:
            messagebox.showinfo("无预览", "当前没有可导出的曲线预览。")
            return
        path = filedialog.asksaveasfilename(
            title="导出当前预览图",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("PDF 矢量图", "*.pdf"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            self.viewer_figure.savefig(path, dpi=200, bbox_inches="tight")
            self.log(f"已导出预览图：{path}")
            messagebox.showinfo("导出成功", f"已保存到：\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            self.log(f"预览图导出失败：{exc}")

    def clear_viewer(self, keep_placeholder=True):
        """Remove embedded matplotlib widgets and reset state."""
        self._dispose_viewer_plot()

        # 清理新增的模式切换控件
        if hasattr(self, "_viewer_controls") and self._viewer_controls:
            try:
                self._viewer_controls.destroy()
            except tk.TclError:
                pass
            self._viewer_controls = None

        self.results_df = None
        self.results_groups = None
        self._has_poisson = False
        self._viewer_mode = "strain"
        self.dic_last_field = None
        self.dic_last_image = None
        self.dic_last_frame_1based = None
        self.dic_last_filename = None
        self.dic_last_reference_frame_1based = None
        self.dic_last_reference_filename = None
        self._canvas_mask_overlay_rgb = None
        if hasattr(self, "field_viewer_context_var"):
            self.field_viewer_context_var.set("")
        self._viewer_kind = "extensometer"
        self._restore_sequence_preview()

        # 恢复占位提示
        if keep_placeholder:
            try:
                self.viewer_frame.grid()
                self.viewer_content_frame.grid_remove()
                self.viewer_placeholder.grid()
            except tk.TclError:
                pass
            self.viewer_export_btn.config(state=tk.DISABLED)
            self.viewer_clear_btn.config(state=tk.DISABLED)
