#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIC Virtual Extensometer GUI v7 Multi-ROI Range Preview
-----------------------------------------
用途：
    虚拟引伸计：多组 ROI pair，独立追踪，输出工程/真应变曲线。
    全场 2D DIC：子集网格 + IC-GN / IC-LM，输出位移场与 Green-Lagrange 应变图。

核心功能：
    1. 多组 ROI：可以添加 G01、G02、G03... 分别分析。
    2. 方向自动判断：auto 模式下，左右分开用 x，上下分开用 y，倾斜明显用 distance。
    3. 对齐辅助：
       - 水平对齐：强制 ROI1/ROI2 中心 y 相同，并建议用 x 应变。
       - 垂直对齐：强制 ROI1/ROI2 中心 x 相同，并建议用 y 应变。
       - 绘制 ROI2 时可自动对齐。
    4. 自适应追踪：
       - hard accept：相关系数高，应变增量连续。
       - adaptive accept：相关系数略低，但通过软阈值、应变连续、前后向检查。
       - rejected：不更新 ROI、不更新模板，应变写 NaN。
    5. 全场 2D DIC：矩形 ROI 内按 subset/step 布点，整数/模板匹配初值后 IC-GN 或 IC-LM
       一阶仿射细化（ZNSSD/ZNCC）。失败点保持 NaN，不插值填补。
    6. 由位移场窗口拟合 Green-Lagrange 与无穷小应变，可选高斯平滑；导出表与色图。
    7. Windows 中文路径兼容：使用 np.fromfile + cv2.imdecode 读取图片。
    8. 可预览任意帧，并设置分析起始帧/结束帧，方便避开无效前后段。

依赖：
    pip install opencv-python numpy pandas matplotlib pillow

运行：
    python dic_virtual_extensometer_gui_v7_multi_roi_range.py
"""

import os
import re
import math
import glob
import json
import queue
import shutil
import threading
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

# 科研人员中文 Windows 环境字体支持（解决 matplotlib 图中中文乱码/方框）
# 优先使用系统中常见的微软雅黑 / 黑体，失败时回退到 DejaVu Sans
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial",
    "Helvetica",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

PLOT_COLOR_CYCLE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
]

PLOT_EXPORT_FORMATS = ("png", "tiff", "pdf", "svg", "eps")
PLOT_EXPORT_PRESETS = {
    "single_column": {
        "figsize": (3.45, 2.55),
        "dpi": 600,
        "font_size": 8,
        "label_size": 8,
        "tick_size": 7,
        "legend_size": 7,
        "title_size": 8,
        "line_width": 1.15,
        "marker_size": 18,
        "axis_line_width": 0.8,
        "grid_alpha": 0.18,
        "colorbar_label_size": 8,
        "colorbar_tick_size": 7,
        "colorbar_fraction": 0.046,
        "colorbar_pad": 0.035,
        "constrained_layout": True,
    },
    "double_column": {
        "figsize": (7.1, 4.2),
        "dpi": 600,
        "font_size": 9,
        "label_size": 9,
        "tick_size": 8,
        "legend_size": 8,
        "title_size": 9,
        "line_width": 1.25,
        "marker_size": 20,
        "axis_line_width": 0.85,
        "grid_alpha": 0.18,
        "colorbar_label_size": 9,
        "colorbar_tick_size": 8,
        "colorbar_fraction": 0.042,
        "colorbar_pad": 0.03,
        "constrained_layout": True,
    },
    "presentation": {
        "figsize": (10.0, 5.8),
        "dpi": 300,
        "font_size": 13,
        "label_size": 13,
        "tick_size": 11,
        "legend_size": 11,
        "title_size": 14,
        "line_width": 2.0,
        "marker_size": 34,
        "axis_line_width": 1.1,
        "grid_alpha": 0.22,
        "colorbar_label_size": 13,
        "colorbar_tick_size": 11,
        "colorbar_fraction": 0.038,
        "colorbar_pad": 0.03,
        "constrained_layout": True,
    },
    "raw_inspection": {
        "figsize": (7.0, 4.6),
        "dpi": 300,
        "font_size": 10,
        "label_size": 10,
        "tick_size": 9,
        "legend_size": 9,
        "title_size": 11,
        "line_width": 1.2,
        "marker_size": 20,
        "axis_line_width": 0.9,
        "grid_alpha": 0.25,
        "colorbar_label_size": 10,
        "colorbar_tick_size": 9,
        "colorbar_fraction": 0.044,
        "colorbar_pad": 0.035,
        "constrained_layout": True,
    },
    "publication": {
        "figsize": (7.1, 4.2),
        "dpi": 600,
        "font_size": 9,
        "label_size": 9,
        "tick_size": 8,
        "legend_size": 8,
        "title_size": 9,
        "line_width": 1.25,
        "marker_size": 20,
        "axis_line_width": 0.85,
        "grid_alpha": 0.18,
        "colorbar_label_size": 9,
        "colorbar_tick_size": 8,
        "colorbar_fraction": 0.042,
        "colorbar_pad": 0.03,
        "constrained_layout": True,
    },
}

from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont

from PIL import Image, ImageTk

# The numerical/export implementation lives in the Tk-free core.  Importing
# this module still defines the legacy GUI adapter only; no Tk root is created
# until ``main`` is called.
#
# ARCHITECTURAL EXCEPTION (v0.2 stabilization): the pre-core GUI numerical and
# export bodies below remain only behind the explicit ``_legacy_direct_processing``
# compatibility switch for old integrations.  The normal GUI path delegates to
# ``ezdic_core`` and uses its bundle-aware provenance resolver.  Removing the
# legacy bodies is deferred until that compatibility switch is retired; the
# canonical module-level aliases at the end of this file are regression-tested.
import ezdic_core as _core


APP_NAME = "ezDIC"
APP_VERSION = "0.1.4"
APP_DEVELOPER = "Dr. Delun Gong"
APP_DOI = "10.5281/zenodo.20222465"
APP_DOI_URL = f"https://doi.org/{APP_DOI}"
APP_TITLE = f"{APP_NAME} v{APP_VERSION} - Developed by {APP_DEVELOPER} - DOI: {APP_DOI}"
ORIGIN_OPJU_FILENAME = "ezDIC_results.opju"

CITATION_TEXT = f"""Recommended citation:

Gong, D. (2026). ezDIC: A lightweight virtual extensometer for extracting linear strain from image sequences (Version {APP_VERSION}) [Computer software]. Zenodo. {APP_DOI_URL}
"""

USAGE_NOTICE = f"""ezDIC Attribution and Usage Notice

Developer:
{APP_DEVELOPER}

DOI:
{APP_DOI}

This software was developed by {APP_DEVELOPER} for lightweight extraction of linear strain from image sequences.

{CITATION_TEXT.strip()}

Users are not permitted to:
1. claim that they developed this software;
2. remove or alter the developer attribution;
3. redistribute, copy, forward, or share this software with unauthorized users;
4. use this software outside the authorized research or teaching context.

If you use ezDIC in a thesis, paper, presentation, or report, please cite the DOI above.

If you need to share or reuse this software, please obtain permission from {APP_DEVELOPER} first.
"""


IMAGE_EXTENSIONS = [
    "*.tif", "*.tiff", "*.TIF", "*.TIFF",
    "*.png", "*.jpg", "*.jpeg", "*.bmp"
]

TRACKING_PRESETS = {
    "标准": {
        "search_radius": 180,
        "hard_corr": 0.55,
        "soft_corr": 0.35,
        "max_frame_strain_jump": "0.01",
        "fb_tolerance_px": 12.0,
    },
    "低质量图像": {
        "search_radius": 220,
        "hard_corr": 0.45,
        "soft_corr": 0.30,
        "max_frame_strain_jump": "0.015",
        "fb_tolerance_px": 16.0,
    },
    "快速变形": {
        "search_radius": 300,
        "hard_corr": 0.50,
        "soft_corr": 0.30,
        "max_frame_strain_jump": "0.03",
        "fb_tolerance_px": 20.0,
    },
}

STRAIN_MODE_LABEL_TO_VALUE = {
    "自动判断": "auto",
    "横向应变": "x",
    "纵向应变": "y",
    "两点距离应变": "distance",
}
STRAIN_MODE_VALUE_TO_LABEL = {v: k for k, v in STRAIN_MODE_LABEL_TO_VALUE.items()}

ROI_ROLE_LABEL_TO_VALUE = {
    "普通": "none",
    "拉伸方向": "axial",
    "横向方向": "transverse",
}
ROI_ROLE_VALUE_TO_LABEL = {v: k for k, v in ROI_ROLE_LABEL_TO_VALUE.items()}
ROI_ROLE_VALUES = set(ROI_ROLE_VALUE_TO_LABEL)
POISSON_MIN_ABS_AXIAL_ENGINEERING_STRAIN = 1e-6

GROUP_TREE_HEADING_TEXTS = {
    "name": "组名",
    "role": "角色",
    "selected": "所选方向",
    "actual": "实际方向",
    "L0": "L0",
    "dx": "Δx",
    "dy": "Δy",
    "roi1": "ROI1",
    "roi2": "ROI2",
}
GROUP_TREE_COLUMN_WIDTHS = {
    "name": 56,
    "role": 52,
    "selected": 76,
    "actual": 76,
    "L0": 48,
    "dx": 44,
    "dy": 44,
    "roi1": 72,
    "roi2": 72,
}
TRACKING_ACCEPT_MODE_LABELS = {
    "initial": "初始",
    "hard": "硬接受",
    "adaptive": "自适应接受",
    "rejected": "已拒绝",
}

ANALYSIS_MODE_EXTENSOMETER = "extensometer"
ANALYSIS_MODE_FULLFIELD = "fullfield"
DIC_SOLVER_ICGN = "IC-GN"
DIC_SOLVER_ICLM = "IC-LM"
DIC_SOLVERS = (DIC_SOLVER_ICGN, DIC_SOLVER_ICLM)
DIC_FIELD_COMPONENTS = ("u", "v", "zncc", "Exx", "Eyy", "Exy", "exx", "eyy", "exy")
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


def format_tracking_status_line(frame_i, n, group_name, accept_mode, strain_text, score1, score2):
    accept_text = TRACKING_ACCEPT_MODE_LABELS.get(str(accept_mode), str(accept_mode))
    return (
        f"第 {frame_i}/{n} 帧，组 {group_name}：{accept_text}，"
        f"应变={strain_text}，相关=({score1:.3f}, {score2:.3f})"
    )


def open_output_folder(path):
    folder = Path(path)
    if not folder.exists():
        raise RuntimeError(f"结果目录不存在：{folder}")
    if not folder.is_dir():
        raise RuntimeError(f"结果路径不是文件夹：{folder}")
    if not hasattr(os, "startfile"):
        raise RuntimeError("自动打开结果目录仅支持 Windows Explorer。")
    os.startfile(str(folder))


def write_image_checked(path, image):
    """Write an image through an encoded buffer, preserving Unicode paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() or ".png"
    try:
        ok, encoded = cv2.imencode(extension, np.asarray(image))
        if not ok or encoded is None:
            raise RuntimeError("OpenCV 编码失败")
        encoded.tofile(str(path))
    except Exception as exc:
        raise RuntimeError(f"写入图像失败：{path}（{exc}）") from exc
    if not ok or not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"写入图像失败：{path}")
    return path


class ToolTip:
    def __init__(self, widget, text, wraplength=360, delay_ms=450):
        self.widget = widget
        self.owner = widget.winfo_toplevel()
        self.text = text
        self.wraplength = wraplength
        self.delay_ms = delay_ms
        self.after_id = None
        self.tip_window = None
        self.event = None
        self.pending_text = None
        self.shown_text = None

        self.widget._tooltip_text = text if isinstance(text, str) else ""
        self.widget._tooltip = self
        self.widget.bind("<Enter>", self.schedule, add="+")
        self.widget.bind("<Motion>", self.schedule, add="+")
        self.widget.bind("<Leave>", self.hide, add="+")
        self.widget.bind("<ButtonPress>", self.hide, add="+")
        self.widget.bind("<FocusOut>", self.hide, add="+")
        self.widget.bind("<Unmap>", self.hide, add="+")
        self.widget.bind("<Destroy>", self.dispose, add="+")
        self.widget.bind("<F1>", self.show_help, add="+")
        self.widget.bind("<Escape>", self.dismiss, add="+")

    def resolve_text(self, event=None):
        text = self.text(event) if callable(self.text) else self.text
        self.widget._tooltip_text = text
        return text

    def schedule(self, event=None):
        text = self.resolve_text(event)
        self.event = event
        if not text:
            self.hide()
            return
        if text == self.shown_text or (self.after_id is not None and text == self.pending_text):
            return
        self.hide()
        self.event = event
        self.pending_text = text
        self.after_id = self.owner.after(self.delay_ms, self.show)

    def show_help(self, event=None):
        self.hide()
        self.event = event
        self.show()
        return "break"

    def dismiss(self, event=None):
        visible = self.tip_window is not None
        self.hide()
        if visible:
            return "break"

    def unschedule(self):
        if self.after_id is not None:
            try:
                self.owner.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def screen_bounds(self, x, y):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class MonitorInfo(ctypes.Structure):
                _fields_ = [("size", wintypes.DWORD), ("monitor", wintypes.RECT),
                            ("work", wintypes.RECT), ("flags", wintypes.DWORD)]

            user32 = ctypes.windll.user32
            user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
            user32.MonitorFromPoint.restype = wintypes.HANDLE
            user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
            info = MonitorInfo(size=ctypes.sizeof(MonitorInfo))
            monitor = user32.MonitorFromPoint(wintypes.POINT(x, y), 2)
            if user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                rect = info.work
                return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
        return 0, 0, self.widget.winfo_screenwidth(), self.widget.winfo_screenheight()

    def show(self):
        self.after_id = None
        text = self.resolve_text(self.event)
        if self.tip_window is not None or not text:
            return

        try:
            if not self.widget.winfo_ismapped():
                return
            wx = self.widget.winfo_rootx()
            wy = self.widget.winfo_rooty()
            wh = self.widget.winfo_height()
            pointer = self.event is not None and getattr(self.event, "keysym", "??") in ("??", "")
            if pointer and getattr(self.event, "type", None) == tk.EventType.VirtualEvent:
                anchor_x, anchor_y = self.widget.winfo_pointerxy()
            else:
                anchor_x = self.event.x_root if pointer else wx
                anchor_y = self.event.y_root if pointer else wy + wh
            screen_x, screen_y, screen_w, screen_h = self.screen_bounds(anchor_x, anchor_y)

            self.tip_window = tk.Toplevel(self.widget)
            self.tip_window.withdraw()
            self.tip_window.wm_overrideredirect(True)
            self.tip_window.wm_attributes("-topmost", True)
            label = ttk.Label(
                self.tip_window, text=text, justify=tk.LEFT,
                wraplength=min(self.wraplength, screen_w - 40),
                background="#fff8dc", foreground="#1f2937",
                relief=tk.SOLID, borderwidth=1, padding=(8, 5),
            )
            label.pack()
            self.tip_window.update_idletasks()
            width = self.tip_window.winfo_reqwidth()
            height = self.tip_window.winfo_reqheight()
            region = getattr(self.event, "help_region", None)
            if region is not None:
                region_x, region_width = region
                x = region_x + region_width + 8
                if x + width > screen_x + screen_w - 8:
                    x = region_x - width - 8
                y = anchor_y
            else:
                x, y = anchor_x + 16, anchor_y + 20
            x = min(max(screen_x + 8, x), screen_x + screen_w - width - 8)
            if y + height > screen_y + screen_h - 8:
                y = anchor_y - height - 12
            y = min(max(screen_y + 8, y), screen_y + screen_h - height - 8)
            self.tip_window.wm_geometry(f"+{x}+{y}")
            self.tip_window.deiconify()
            self.shown_text = text
        except tk.TclError:
            self.hide()
            return

    def hide(self, event=None):
        self.unschedule()
        self.pending_text = None
        self.shown_text = None
        if self.tip_window is not None:
            try:
                self.tip_window.destroy()
            except tk.TclError:
                pass
            self.tip_window = None

    def dispose(self, event=None):
        self.hide()
        # Do not retain destroyed controls or their app through event objects
        # and dynamic text closures until a later worker-thread collection.
        self.event = None
        self.text = ""
        self.widget._tooltip_text = ""


class ChoiceToolTip(ToolTip):
    """Explain both the selected value and each native combobox list item."""

    def __init__(self, widget, text, choices):
        self.summary = text
        self.choices = {str(value): explanation for value, explanation in choices.items()}
        self.popup_choice = None
        self.listbox = None
        super().__init__(widget, self.choice_text)
        widget._tooltip_choices = self.choices
        self.previous_postcommand = widget.cget("postcommand")
        widget.configure(postcommand=self.prepare_choices)
        self.motion_command = widget.register(self.on_choice_motion)
        self.selection_command = widget.register(self.on_choice_selection)
        self.hide_command = widget.register(self.hide_choices)

    def choice_text(self, event=None):
        if self.popup_choice is not None:
            return f"{self.popup_choice}：{self.choices[self.popup_choice]}\n点击或按回车选择。"
        summary = self.summary(event) if callable(self.summary) else self.summary
        value = str(self.widget.get())
        return f"{summary}\n当前 {value}：{self.choices[value]}" if value in self.choices else summary

    def prepare_choices(self):
        if self.previous_postcommand:
            self.widget.tk.call(self.previous_postcommand)
        popup = self.widget.tk.call("ttk::combobox::PopdownWindow", str(self.widget))
        listbox = f"{popup}.f.l"
        if self.listbox == listbox:
            return
        self.listbox = listbox
        for sequence, script in (
            ("<Motion>", f"{self.motion_command} %y %X %Y"),
            ("<Map>", self.selection_command),
            ("<KeyRelease>", self.selection_command),
            ("<<ListboxSelect>>", self.selection_command),
            ("<Leave>", self.hide_command),
            ("<Unmap>", self.hide_command),
            ("<ButtonPress>", self.hide_command),
            ("<FocusOut>", self.hide_command),
        ):
            self.widget.tk.call("bind", listbox, sequence, "+" + script)

    def on_choice_motion(self, y, x_root, y_root):
        index = int(self.widget.tk.call(self.listbox, "nearest", int(y)))
        box = self.widget.tk.call(self.listbox, "bbox", index)
        if not box or not int(box[1]) <= int(y) < int(box[1]) + int(box[3]):
            self.hide_choices()
            return
        self.schedule_choice(index, x_root, y_root)

    def on_choice_selection(self):
        selected = self.widget.tk.call(self.listbox, "curselection")
        if selected:
            index = int(selected[0])
            box = self.widget.tk.call(self.listbox, "bbox", index)
            if box:
                x = int(self.widget.tk.call("winfo", "rootx", self.listbox))
                y = int(self.widget.tk.call("winfo", "rooty", self.listbox)) + int(box[1])
                self.schedule_choice(index, x, y)

    def schedule_choice(self, index, x_root, y_root):
        values = self.widget.cget("values")
        value = str(values[index])
        if value not in self.choices:
            self.hide_choices()
            return
        self.popup_choice = value
        event = tk.Event()
        event.x_root, event.y_root = int(x_root), int(y_root)
        event.help_region = (int(self.widget.tk.call("winfo", "rootx", self.listbox)),
                             int(self.widget.tk.call("winfo", "width", self.listbox)))
        self.schedule(event)

    def hide_choices(self):
        self.popup_choice = None
        self.hide()

    def dispose(self, event=None):
        super().dispose(event)
        self.summary = ""
        self.previous_postcommand = ""


class HelpNavigationToolbar(NavigationToolbar2Tk):
    # Use the same delayed help as the rest of the app, without duplicate
    # immediate English tooltips from Matplotlib's default toolbar.
    toolitems = tuple((name, None, icon, action)
                      for name, _tip, icon, action in NavigationToolbar2Tk.toolitems)

    def configure_subplots(self, *args):
        if hasattr(self, "subplot_tool"):
            self._subplot_window.deiconify()
            self._subplot_window.lift()
            return
        from matplotlib.widgets import SubplotTool

        # Keep the embedded app and this dialog in one Tcl interpreter.
        # Matplotlib's standalone manager would create another Tk root.
        window = tk.Toplevel(self)
        self._subplot_window = window
        figure = Figure(figsize=(6, 3), dpi=100)
        figure.subplots_adjust(top=0.9)
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.subplot_tool = SubplotTool(self.canvas.figure, figure)
        window.protocol("WM_DELETE_WINDOW", self.close_subplots)
        self.subplot_help(self.subplot_tool, window)
        canvas.draw()
        return self.subplot_tool

    def close_subplots(self):
        if not hasattr(self, "subplot_tool"):
            return
        canvas = self.subplot_tool.figure.canvas
        for attr in ("_idle_draw_id", "_event_loop_id"):
            job = getattr(canvas, attr, None)
            if job is not None:
                canvas.get_tk_widget().after_cancel(job)
                setattr(canvas, attr, None)
        self._subplot_window.destroy()
        canvas._tkphoto = None
        canvas.figure.set_canvas(None)
        del self.subplot_tool
        self._subplot_window = None


# ==========================
# 基础函数
# ==========================

def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def safe_name(s):
    s = str(s).strip()
    if not s:
        return "group"
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    return s[:80]


def collect_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(list(set(paths)), key=natural_sort_key)


def image_sequence_fingerprint(paths):
    """Stable sequence identity based on normalized paths and file metadata."""
    fingerprint = []
    for raw_path in paths or []:
        path = Path(raw_path)
        try:
            stat = path.stat()
            fingerprint.append(
                (
                    os.path.normcase(os.path.abspath(str(path))),
                    int(stat.st_size),
                    int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
                )
            )
        except OSError:
            fingerprint.append((os.path.normcase(os.path.abspath(str(path))), None, None))
    return tuple(fingerprint)


FULLFIELD_GENERATED_FILE_RE = re.compile(
    r"^(?:"
    r"frame_\d{4}\.(?:txt|csv)|"
    r"frame_\d{4}_(?:u|v|Exx|Eyy|Exy|overlay)\.png|"
    r"frame_\d{4}_parameters\.txt"
    r")$",
    flags=re.IGNORECASE,
)


def _create_unique_timestamped_dir(root, *, prefix=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{prefix}_{timestamp}" if prefix else timestamp
    candidate = root / name
    suffix = 1
    while candidate.exists():
        candidate = root / f"{name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _atomic_move_entries(entries, label, *, after_move=None):
    """Move exact file entries with best-effort inverse moves on failure."""
    normalized_entries = [(Path(source), Path(destination)) for source, destination in entries]
    moved = []
    try:
        for source, destination in normalized_entries:
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
        if after_move is not None:
            after_move()
    except Exception as exc:
        rollback_errors = []
        for source, destination in reversed(moved):
            if not destination.exists():
                continue
            try:
                shutil.move(str(destination), str(source))
            except Exception as rollback_exc:
                # Path.replace bypasses a transient shutil failure while
                # remaining confined to the exact generated destination.
                try:
                    destination.replace(source)
                except Exception as replace_exc:
                    rollback_errors.append(
                        f"{destination} -> {source}: {rollback_exc}; fallback: {replace_exc}"
                    )
        detail = f"{label}失败：{exc}"
        if rollback_errors:
            detail += "；回滚失败：" + " | ".join(rollback_errors)
        raise RuntimeError(detail) from exc


def archive_previous_fullfield_outputs(dic_dir):
    """Move exact legacy full-field outputs under the core transaction guards.

    The legacy GUI path predates the manifest ledger, so it retains the
    conservative exact-name allowlist above.  It must nevertheless obey the
    same filesystem boundary as the manifest transaction: all output and
    destination ancestors are checked before enumeration or directory
    creation, and each move uses the core's no-replace primitive (which also
    applies the Windows directory-handle guards).  These are deliberately
    private core calls because this compatibility migration has no public
    transaction API of its own.
    """
    dic_dir = Path(dic_dir)
    output_root = dic_dir.parent
    previous_root = dic_dir / "_previous_runs"

    def assert_safe(path, *, allow_missing_leaf):
        _core._assert_no_reparse_components(path, allow_missing_leaf=allow_missing_leaf)

    def assert_contained(path, *, code):
        _core._assert_contained_path(path, output_root, code=code)

    # Resolve no links before any exists/is_dir check.  In particular, a
    # junction at output_root/dic must fail before it can be enumerated.
    assert_safe(output_root, allow_missing_leaf=True)
    assert_safe(dic_dir, allow_missing_leaf=True)
    assert_contained(dic_dir, code="LEGACY_OUTPUT_OUTSIDE_ROOT")
    assert_safe(previous_root, allow_missing_leaf=True)
    assert_contained(previous_root, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")

    if not dic_dir.exists() or not dic_dir.is_dir():
        return None

    # Match the core's operation lock and in-process lock ordering.  The
    # second preflight closes the gap between the initial check and lock
    # acquisition before the directory is inspected or changed.
    with _core._operation_os_lock(output_root):
        with _core._transaction_lock(output_root):
            assert_safe(output_root, allow_missing_leaf=False)
            assert_safe(dic_dir, allow_missing_leaf=False)
            assert_contained(dic_dir, code="LEGACY_OUTPUT_OUTSIDE_ROOT")
            assert_safe(previous_root, allow_missing_leaf=True)
            assert_contained(previous_root, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")

            if not dic_dir.exists() or not dic_dir.is_dir():
                return None

            # A non-directory destination parent is unusable and, like a
            # reparse point, must be rejected before source enumeration.
            if previous_root.exists() and not previous_root.is_dir():
                raise RuntimeError(f"全场旧结果归档失败：归档目录不是文件夹：{previous_root}")

            # Select and validate the eventual destination chain before
            # enumerating legacy files.  The parent may be missing, so this
            # remains read-only until all source checks have completed.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive_name = timestamp
            suffix = 1
            while True:
                archive_dir = previous_root / archive_name
                assert_safe(archive_dir, allow_missing_leaf=True)
                assert_contained(archive_dir, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
                if not archive_dir.exists():
                    break
                archive_name = f"{timestamp}_{suffix:02d}"
                suffix += 1

            previous_files = []
            for path in dic_dir.iterdir():
                if not FULLFIELD_GENERATED_FILE_RE.fullmatch(path.name):
                    continue
                # A matching symlink/junction is not a user-owned exception:
                # reject it before creating an archive directory or moving a
                # different legacy file.
                assert_safe(path, allow_missing_leaf=False)
                if path.is_file():
                    assert_contained(path, code="LEGACY_OUTPUT_OUTSIDE_ROOT")
                    previous_files.append(path)
            if not previous_files:
                return None

            # Keep the destination parent containment check adjacent to the
            # source preflight; its reparse check above ran before enumeration.
            assert_contained(previous_root, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")

            entries = []
            for source in sorted(previous_files, key=lambda p: p.name):
                destination = archive_dir / source.name
                assert_safe(source, allow_missing_leaf=False)
                # The final archive directory may not exist yet, so this
                # check validates its complete existing ancestor chain while
                # still keeping directory creation below the preflight.
                assert_safe(destination.parent, allow_missing_leaf=True)
                assert_contained(source, code="LEGACY_OUTPUT_OUTSIDE_ROOT")
                assert_contained(destination.parent, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
                if not source.is_file():
                    raise RuntimeError(f"全场旧结果归档失败：源文件已变化：{source}")
                entries.append((source, destination))

            # Recheck and create the destination only after all legacy source
            # and destination paths have passed the read-only preflight.
            assert_safe(previous_root, allow_missing_leaf=True)
            previous_root.mkdir(parents=True, exist_ok=True)
            assert_safe(previous_root, allow_missing_leaf=False)
            assert_contained(previous_root, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
            archive_dir.mkdir()
            assert_safe(archive_dir, allow_missing_leaf=False)
            assert_contained(archive_dir, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
            for _source, destination in entries:
                assert_safe(destination.parent, allow_missing_leaf=False)
                if destination.exists() or destination.is_symlink():
                    raise RuntimeError(f"全场旧结果归档失败：目标已存在：{destination}")

            moved = []
            try:
                for source, destination in entries:
                    # _move_exact supplies no-replace publication, containment
                    # rechecks, and Windows directory-handle protection.
                    _core._move_exact(source, destination, root=output_root)
                    moved.append((source, destination))
            except Exception as exc:
                rollback_errors = []
                for source, destination in reversed(moved):
                    try:
                        _core._move_exact(destination, source, root=output_root)
                    except Exception as rollback_exc:
                        rollback_errors.append(f"{destination} -> {source}: {rollback_exc}")
                try:
                    assert_safe(archive_dir, allow_missing_leaf=False)
                    assert_contained(archive_dir, code="LEGACY_ARCHIVE_OUTSIDE_ROOT")
                    archive_dir.rmdir()
                except OSError:
                    pass
                detail = f"全场旧结果归档失败：{exc}"
                if rollback_errors:
                    detail += "；回滚失败：" + " | ".join(rollback_errors)
                raise RuntimeError(detail) from exc
            return archive_dir


def commit_fullfield_staging(staging_dir, dic_dir):
    """Commit verified staged files into the full-field output root."""
    staging_dir = Path(staging_dir)
    dic_dir = Path(dic_dir)
    dic_dir.mkdir(parents=True, exist_ok=True)
    entries = sorted(staging_dir.iterdir(), key=lambda p: p.name)
    move_entries = []
    for entry in entries:
        destination = dic_dir / entry.name
        if destination.exists():
            raise RuntimeError(f"全场输出提交目标已存在：{destination}")
        move_entries.append((entry, destination))
    _atomic_move_entries(move_entries, "全场输出提交", after_move=staging_dir.rmdir)
    return dic_dir


def archive_failed_fullfield_staging(staging_dir, dic_dir):
    """Retain a failed run's staging contents under a unique failure archive."""
    staging_dir = Path(staging_dir)
    if not staging_dir.exists():
        return None
    failed_dir = _create_unique_timestamped_dir(Path(dic_dir) / "_failed_runs")
    entries = sorted(staging_dir.iterdir(), key=lambda p: p.name)
    move_entries = [(entry, failed_dir / entry.name) for entry in entries]
    _atomic_move_entries(move_entries, "全场失败运行归档", after_move=staging_dir.rmdir)
    return failed_dir


def read_gray_image(path):
    """
    稳健读取图像。
    使用 np.fromfile + cv2.imdecode 解决 Windows 中文路径问题；
    如果失败则用 Pillow 兜底读取。
    """
    path = str(path)
    img = None
    errors = []

    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        errors.append(f"cv2.imdecode failed: {exc}")

    if img is None:
        try:
            with Image.open(path) as im:
                if getattr(im, "n_frames", 1) > 1:
                    im.seek(0)
                img = np.array(im)
        except Exception as exc:
            errors.append(f"Pillow failed: {exc}")

    if img is None:
        detail = " | ".join(errors) if errors else "unknown error"
        raise RuntimeError(f"无法读取图片：{path}\n原因：{detail}")

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    return img


def normalize_to_uint8(img, lo=None, hi=None):
    arr = img.astype(np.float32)

    if lo is None or hi is None:
        lo, hi = np.percentile(arr, [1, 99])

    if hi <= lo:
        hi = lo + 1.0

    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def get_display_image(img8, max_w=1120, max_h=720):
    h, w = img8.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    if scale < 1:
        disp = cv2.resize(img8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = img8.copy()

    rgb = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
    return rgb, scale


def rect_normalize(x1, y1, x2, y2):
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def clamp_rect(rect, img_shape):
    x, y, w, h = rect
    H, W = img_shape[:2]

    x = max(0, min(int(round(x)), W - 1))
    y = max(0, min(int(round(y)), H - 1))
    w = max(1, min(int(round(w)), W - x))
    h = max(1, min(int(round(h)), H - y))
    return x, y, w, h


def rect_center(rect):
    x, y, w, h = rect
    return float(x + w / 2.0), float(y + h / 2.0)


def move_rect_center(rect, new_cx=None, new_cy=None, img_shape=None):
    x, y, w, h = rect
    cx, cy = rect_center(rect)
    if new_cx is None:
        new_cx = cx
    if new_cy is None:
        new_cy = cy

    new_x = int(round(new_cx - w / 2.0))
    new_y = int(round(new_cy - h / 2.0))
    out = (new_x, new_y, w, h)
    if img_shape is not None:
        out = clamp_rect(out, img_shape)
    return out


def center_distance(rect_a, rect_b):
    ax, ay = rect_center(rect_a)
    bx, by = rect_center(rect_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def roi_separation(rect1, rect2):
    x1, y1 = rect_center(rect1)
    x2, y2 = rect_center(rect2)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return dx, dy, dist


def resolve_strain_mode(rect1, rect2, selected_mode):
    """
    自动判断应变方向：
    - 左右分开明显：x
    - 上下分开明显：y
    - 倾斜明显：distance
    """
    if selected_mode != "auto":
        return selected_mode

    dx, dy, dist = roi_separation(rect1, rect2)

    if dx >= 3.0 * max(dy, 1.0):
        return "x"
    if dy >= 3.0 * max(dx, 1.0):
        return "y"
    return "distance"


def length_between(rect1, rect2, mode="x"):
    x1, y1 = rect_center(rect1)
    x2, y2 = rect_center(rect2)

    if mode == "y":
        return abs(y2 - y1)
    if mode == "x":
        return abs(x2 - x1)
    if mode == "distance":
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    raise ValueError("应变方向只能是 x, y 或 distance。")


def extract_patch(img8, rect):
    x, y, w, h = clamp_rect(rect, img8.shape)
    return img8[y:y + h, x:x + w].astype(np.float32)


def roi_texture_metrics(img8, rect):
    patch = extract_patch(img8, rect)
    if patch.size == 0:
        return {
            "std_gray": 0.0,
            "contrast_p95_p5": 0.0,
            "low_frac": 1.0,
            "high_frac": 1.0,
        }

    p5, p95 = np.percentile(patch, [5, 95])
    return {
        "std_gray": float(np.std(patch)),
        "contrast_p95_p5": float(p95 - p5),
        "low_frac": float(np.mean(patch <= 5)),
        "high_frac": float(np.mean(patch >= 250)),
    }


def texture_is_ok(metrics, min_std, min_contrast, max_saturated_frac):
    if metrics["std_gray"] < min_std:
        return False
    if metrics["contrast_p95_p5"] < min_contrast:
        return False
    if metrics["low_frac"] > max_saturated_frac:
        return False
    if metrics["high_frac"] > max_saturated_frac:
        return False
    return True


def has_nonzero_variance(values, *, min_std=1e-8):
    """Return whether an image/template patch has finite, usable contrast.

    OpenCV's ``TM_CCOEFF_NORMED`` is undefined for a constant template or
    candidate patch and can return a misleading score of 1.0.  Keep this
    check independent of OpenCV so all normalized-correlation entry points
    share the same scientific rejection rule.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return bool(float(np.std(finite)) > float(min_std))


def subpixel_peak(corr, px, py):
    H, W = corr.shape
    dx = 0.0
    dy = 0.0

    if 1 <= px < W - 1:
        c1 = corr[py, px - 1]
        c2 = corr[py, px]
        c3 = corr[py, px + 1]
        denom = c1 - 2 * c2 + c3
        if abs(denom) > 1e-12:
            dx = 0.5 * (c1 - c3) / denom

    if 1 <= py < H - 1:
        c1 = corr[py - 1, px]
        c2 = corr[py, px]
        c3 = corr[py + 1, px]
        denom = c1 - 2 * c2 + c3
        if abs(denom) > 1e-12:
            dy = 0.5 * (c1 - c3) / denom

    return float(np.clip(dx, -0.5, 0.5)), float(np.clip(dy, -0.5, 0.5))


def match_template_candidate(img8, last_rect, template, search_radius):
    H, W = img8.shape[:2]
    x, y, w, h = last_rect
    x = float(x)
    y = float(y)
    w = int(round(w))
    h = int(round(h))

    sx1 = int(max(0, math.floor(x - search_radius)))
    sy1 = int(max(0, math.floor(y - search_radius)))
    sx2 = int(min(W, math.ceil(x + w + search_radius)))
    sy2 = int(min(H, math.ceil(y + h + search_radius)))

    search_img = img8[sy1:sy2, sx1:sx2].astype(np.float32)

    if search_img.shape[0] < h or search_img.shape[1] < w:
        return last_rect, -1.0

    if template.shape[0] != h or template.shape[1] != w:
        return last_rect, -1.0

    # TM_CCOEFF_NORMED is undefined for constant inputs.  OpenCV may report
    # a perfect score for these patches, so reject before calling it and also
    # validate the selected candidate window below.
    if not has_nonzero_variance(template) or not has_nonzero_variance(search_img):
        return last_rect, -1.0

    corr = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    finite_corr = np.isfinite(corr)
    if not finite_corr.any():
        return last_rect, -1.0
    safe_corr = np.where(finite_corr, corr, -1.0).astype(np.float32, copy=False)
    _, max_val, _, max_loc = cv2.minMaxLoc(safe_corr)
    if not np.isfinite(max_val):
        return last_rect, -1.0

    px, py = max_loc
    candidate_patch = search_img[py : py + h, px : px + w]
    if candidate_patch.shape != template.shape or not has_nonzero_variance(candidate_patch):
        return last_rect, -1.0
    dx, dy = subpixel_peak(safe_corr, px, py)

    new_x = sx1 + px + dx
    new_y = sy1 + py + dy

    return (float(new_x), float(new_y), w, h), float(max_val)


def update_template_from_rect(img8, rect, old_template, alpha):
    x, y, w, h = rect
    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))

    H, W = img8.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return old_template

    patch = img8[y:y + h, x:x + w].astype(np.float32)
    if patch.shape != old_template.shape:
        return old_template

    return (1.0 - alpha) * old_template + alpha * patch


def forward_backward_error(prev_img8, curr_img8, prev_rect, curr_candidate_rect, search_radius):
    """
    前后向一致性检查：
    curr 候选位置若真实，应能用当前 patch 反追踪回 prev_rect。
    """
    curr_patch = extract_patch(curr_img8, curr_candidate_rect)
    back_rect, back_score = match_template_candidate(
        prev_img8,
        prev_rect,
        curr_patch,
        search_radius=search_radius,
    )
    if back_score < 0 or not np.isfinite(back_score):
        return float("inf"), float(back_score)
    err = center_distance(back_rect, prev_rect)
    return float(err), float(back_score)


def draw_group_overlay(
    img8,
    group_name,
    actual_mode,
    used_rect1,
    used_rect2,
    candidate_rect1,
    candidate_rect2,
    frame_idx,
    strain,
    last_valid_strain,
    score1,
    score2,
    accepted,
    accept_mode,
    reason,
    fb_err1=None,
    fb_err2=None,
):
    rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    def draw_rect(rect, color, label, thickness=2):
        if rect is None:
            return
        x, y, w, h = rect
        x = int(round(x))
        y = int(round(y))
        w = int(round(w))
        h = int(round(h))

        cv2.rectangle(rgb, (x, y), (x + w, y + h), color, thickness)
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        cv2.circle(rgb, (cx, cy), 4, color, -1)
        cv2.putText(
            rgb,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # 候选框：橙/紫
    draw_rect(candidate_rect1, (0, 165, 255), "ROI1 candidate", 1)
    draw_rect(candidate_rect2, (255, 0, 255), "ROI2 candidate", 1)

    # 实际用于计算的框：红/蓝
    draw_rect(used_rect1, (0, 0, 255), "ROI1 used", 2)
    draw_rect(used_rect2, (255, 0, 0), "ROI2 used", 2)

    c1 = rect_center(used_rect1)
    c2 = rect_center(used_rect2)
    cv2.line(
        rgb,
        (int(round(c1[0])), int(round(c1[1]))),
        (int(round(c2[0])), int(round(c2[1]))),
        (0, 255, 255),
        2,
    )

    status = "ACCEPTED" if accepted else "REJECTED - ROI/TEMPLATE NOT UPDATED"
    if accepted and accept_mode:
        status += f" ({accept_mode})"

    strain_txt = f"Eng. strain: {strain:.6f}" if np.isfinite(strain) else "Eng. strain: NaN"
    last_txt = f"Last valid strain: {last_valid_strain:.6f}" if np.isfinite(last_valid_strain) else "Last valid strain: NaN"

    lines = [
        f"Group: {group_name} | mode: {actual_mode}",
        f"Frame: {frame_idx}",
        status,
        strain_txt,
        last_txt,
        f"Corr: ROI1={score1:.3f}, ROI2={score2:.3f}",
    ]

    if fb_err1 is not None and fb_err2 is not None:
        lines.append(f"FB error: ROI1={fb_err1:.2f}px, ROI2={fb_err2:.2f}px")

    lines.append(f"Reason: {reason}")

    y0 = 34
    for k, line in enumerate(lines):
        cv2.putText(
            rgb,
            line[:140],
            (20, y0 + 30 * k),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68 if k < 3 else 0.56,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return rgb


# ---------- Full-field 2D DIC (IC-GN / IC-LM) ----------


def generate_synthetic_speckle(height, width, *, seed=0, n_dots=None, sigma=1.8):
    """Smooth Gaussian-dot speckle used by tests and the GUI-independent DIC entry."""
    rng = np.random.default_rng(seed)
    height = int(height)
    width = int(width)
    if n_dots is None:
        n_dots = max(120, (height * width) // 28)
    img = np.full((height, width), 25.0, dtype=np.float32)
    ys = rng.uniform(2, max(3, height - 2), size=n_dots)
    xs = rng.uniform(2, max(3, width - 2), size=n_dots)
    amps = rng.uniform(90, 230, size=n_dots)
    rad = int(np.ceil(3.0 * float(sigma)))
    yy, xx = np.ogrid[-rad : rad + 1, -rad : rad + 1]
    blob = np.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) * float(sigma))).astype(np.float32)
    for x, y, amp in zip(xs, ys, amps):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = xi - rad, xi + rad + 1
        y0, y1 = yi - rad, yi + rad + 1
        gx0 = gy0 = 0
        gx1, gy1 = blob.shape[1], blob.shape[0]
        if x0 < 0:
            gx0 = -x0
            x0 = 0
        if y0 < 0:
            gy0 = -y0
            y0 = 0
        if x1 > width:
            gx1 -= x1 - width
            x1 = width
        if y1 > height:
            gy1 -= y1 - height
            y1 = height
        if x1 <= x0 or y1 <= y0:
            continue
        img[y0:y1, x0:x1] += amp * blob[gy0:gy1, gx0:gx1]
    return np.clip(img, 0, 255).astype(np.float32)


def _as_float_image(image):
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.shape[2] >= 3 else arr[:, :, 0]
    return arr.astype(np.float32, copy=False)


def warp_image_translation(image, tx, ty):
    """Move material points by (tx, ty) px using bicubic sampling: def(X) = ref(X - t)."""
    img = _as_float_image(image)
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        img,
        xs - np.float32(tx),
        ys - np.float32(ty),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def warp_image_deformation_gradient(image, F, center=None):
    """Apply a uniform 2x2 deformation gradient about `center` (default image center)."""
    img = _as_float_image(image)
    F = np.asarray(F, dtype=np.float64).reshape(2, 2)
    Finv = np.linalg.inv(F)
    h, w = img.shape[:2]
    if center is None:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx = xs - np.float32(cx)
    dy = ys - np.float32(cy)
    map_x = (np.float32(Finv[0, 0]) * dx + np.float32(Finv[0, 1]) * dy + np.float32(cx)).astype(np.float32)
    map_y = (np.float32(Finv[1, 0]) * dx + np.float32(Finv[1, 1]) * dy + np.float32(cy)).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def green_lagrange_from_F(F):
    """Oracle Green-Lagrange components from a 2x2 deformation gradient."""
    F = np.asarray(F, dtype=np.float64).reshape(2, 2)
    E = 0.5 * (F.T @ F - np.eye(2))
    return {"Exx": float(E[0, 0]), "Eyy": float(E[1, 1]), "Exy": float(E[0, 1])}


def _odd_subset_size(subset_size):
    size = int(subset_size)
    if size < 9:
        raise ValueError("subset_size must be >= 9.")
    if size % 2 == 0:
        size += 1
    return size


def build_poi_grid(roi, subset_size, step, image_shape):
    """POI centers inside `roi=(x,y,w,h)` that keep a full subset on the image."""
    x, y, w, h = [int(round(v)) for v in roi]
    subset_size = _odd_subset_size(subset_size)
    step = max(1, int(step))
    half = subset_size // 2
    H, W = image_shape[:2]
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, W)
    y1 = min(y + h, H)
    xs = np.arange(x0 + half, x1 - half, step, dtype=np.float64)
    ys = np.arange(y0 + half, y1 - half, step, dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def poi_grid_is_usable(X, Y, *, min_rows=3, min_cols=3):
    """Return whether a POI grid can support a 2-D affine strain fit."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
        return False
    rows, cols = X.shape
    return bool(rows >= int(min_rows) and cols >= int(min_cols) and X.size > 0 and Y.size > 0)


def _odd_window_size(window):
    size = max(3, int(window))
    if size % 2 == 0:
        size += 1
    return size


def rect_is_inside_image(rect, image_shape):
    """Return whether a positive rectangular ROI is fully inside an image."""
    if rect is None or image_shape is None:
        return False
    try:
        x, y, w, h = (int(round(value)) for value in rect)
        H, W = (int(value) for value in image_shape[:2])
    except (TypeError, ValueError, IndexError):
        return False
    return bool(x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= W and y + h <= H)


def validate_image_sequence_dimensions(paths):
    """Read a sequence and reject frames whose 2-D image dimensions differ."""
    paths = list(paths or [])
    if not paths:
        raise RuntimeError("全场 DIC 没有可读取的图像帧。")
    reference_shape = tuple(read_gray_image(paths[0]).shape[:2])
    for path in paths[1:]:
        shape = tuple(read_gray_image(path).shape[:2])
        if shape != reference_shape:
            raise RuntimeError(
                "全场 DIC 要求所有图像尺寸一致："
                f"参考帧 {reference_shape[1]}×{reference_shape[0]} px，"
                f"{Path(path).name} 为 {shape[1]}×{shape[0]} px。"
            )
    return reference_shape


def validate_fullfield_snapshot(settings):
    """Validate a frozen full-field settings snapshot without touching Tk.

    ``start_processing`` freezes all GUI values into ``settings`` before it
    starts the worker.  This helper is deliberately module-level and only
    consumes that mapping plus the selected image files, so it is safe to run
    from a worker thread.  It also keeps legacy-output migration behind the
    same read-only checks used by the GUI-side preflight.
    """
    if not isinstance(settings, Mapping):
        raise RuntimeError("全场 DIC 设置快照必须是映射。")

    raw_paths = settings.get("image_paths")
    if isinstance(raw_paths, (str, bytes)):
        raise RuntimeError("全场 DIC 图像路径快照无效。")
    try:
        image_paths = list(raw_paths or [])
    except TypeError as exc:
        raise RuntimeError("全场 DIC 图像路径快照无效。") from exc
    if not image_paths:
        raise RuntimeError("请先加载图像序列。")

    def snapshot_int(key, label, default=None):
        value = settings.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"{label}必须是整数。") from exc

    def snapshot_float(key, label, default=None):
        value = settings.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"{label}必须是数字。") from exc
        if not np.isfinite(value):
            raise RuntimeError(f"{label}必须是有限数字。")
        return value

    start_idx = snapshot_int("start_idx", "起始帧", 0)
    end_idx = snapshot_int("end_idx", "结束帧", len(image_paths) - 1)
    if start_idx < 0 or end_idx < start_idx or end_idx >= len(image_paths):
        raise RuntimeError("全场 DIC 的分析范围无效。")
    if end_idx <= start_idx:
        raise RuntimeError("分析范围至少应包含两帧（参考帧 + 变形帧）。")

    reference_frame = snapshot_int(
        "reference_frame_1based",
        "参考帧",
        start_idx + 1,
    )
    if reference_frame < start_idx + 1 or reference_frame > end_idx + 1:
        raise RuntimeError(
            "全场 DIC 的参考帧必须位于当前分析范围内："
            f"reference={reference_frame}, range={start_idx + 1}–{end_idx + 1}。"
        )
    field_ref = settings.get("field_roi_reference_frame_1based")
    if field_ref is None:
        raise RuntimeError("全场 ROI 缺少绘制参考帧记录，请在当前起始/参考帧重画。")
    try:
        field_ref = int(field_ref)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("全场 ROI 的参考帧记录必须是整数。") from exc
    if field_ref != reference_frame:
        raise RuntimeError(
            f"全场 ROI 在第 {field_ref} 帧定义，但冻结参考帧为第 {reference_frame} 帧，请重画全场 ROI。"
        )

    output_value = settings.get("output_dir")
    if output_value is None or not str(output_value).strip():
        raise RuntimeError("请设置输出文件夹。")
    try:
        output_path = Path(output_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("输出文件夹路径无效。") from exc
    if output_path.exists() and not output_path.is_dir():
        raise RuntimeError(f"输出路径已存在但不是文件夹：{output_path}")

    selected_paths = image_paths[start_idx : end_idx + 1]
    image_shape = validate_image_sequence_dimensions(selected_paths)
    try:
        reference_raw = read_gray_image(image_paths[reference_frame - 1])
    except Exception as exc:
        raise RuntimeError(f"全场 DIC 无法读取参考图像：{exc}") from exc
    if image_shape is None:
        image_shape = reference_raw.shape[:2]

    roi = settings.get("field_roi")
    if roi is None:
        raise RuntimeError("请先绘制全场分析 ROI。")
    if not rect_is_inside_image(roi, image_shape):
        raise RuntimeError("全场 ROI 必须完整位于参考图像内。")

    try:
        normalization = settings.get("normalization")
        if normalization:
            reference8 = normalize_with_bounds(reference_raw, normalization)
        else:
            reference8 = normalize_to_uint8(reference_raw)
    except Exception as exc:
        raise RuntimeError(f"全场 DIC 参考图像归一化失败：{exc}") from exc

    min_texture_std = snapshot_float("min_texture_std", "最小纹理标准差", 8.0)
    min_texture_contrast = snapshot_float("min_texture_contrast", "最小纹理对比度", 25.0)
    max_saturated_frac = snapshot_float("max_saturated_frac", "最大近黑/近白比例", 0.20)
    min_structure_ratio = snapshot_float(
        "min_structure_ratio",
        "最小二维结构张量比值",
        _core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
    )
    max_directional_coherence = snapshot_float(
        "max_directional_coherence",
        "最大方向梯度一致性",
        _core.DEFAULT_TEXTURE_MAX_DIRECTIONAL_COHERENCE,
    )
    min_periodicity_score = snapshot_float(
        "min_periodicity_score",
        "最小周期性分数",
        _core.DEFAULT_TEXTURE_MIN_PERIODICITY_SCORE,
    )
    metrics = roi_texture_metrics(reference8, roi)
    texture_code = texture_failure_code(
        metrics,
        min_texture_std,
        min_texture_contrast,
        max_saturated_frac,
        min_structure_ratio,
        max_directional_coherence,
        min_periodicity_score,
    )
    # The GUI adapter intentionally lets low/saturated texture reach the core
    # so it can retain a structured invalid-frame record.  Ambiguous texture
    # and malformed thresholds remain hard preflight failures.
    if texture_code == "AMBIGUOUS_TEXTURE":
        raise RuntimeError("AMBIGUOUS_TEXTURE：全场 ROI 的二维纹理不可辨识，请更换或增大 ROI。")
    if texture_code not in (None, "LOW_TEXTURE", "SATURATED_TEXTURE"):
        raise RuntimeError(f"全场 DIC 纹理预检失败：{texture_code}")

    subset = snapshot_int("dic_subset_size", "子集尺寸")
    step = snapshot_int("dic_step", "步长")
    window = snapshot_int("dic_strain_window", "应变窗口")
    search_radius = snapshot_int("dic_search_radius", "DIC 搜索半径")
    zncc_min = snapshot_float("dic_zncc_min", "ZNCC 下限")
    smooth_sigma = snapshot_float("dic_smooth_sigma", "高斯平滑 σ")
    pyramid_levels = snapshot_int("dic_pyramid_levels", "金字塔层数", 1)
    pyramid_scale = snapshot_float("dic_pyramid_scale", "金字塔缩放", 0.5)
    solver = str(settings.get("dic_solver", DIC_SOLVER_ICGN))
    if subset < 9:
        raise RuntimeError("子集尺寸必须 >= 9。")
    if subset % 2 == 0:
        raise RuntimeError("子集尺寸必须为奇数；请使用 21、23 等值。")
    if step < 1:
        raise RuntimeError("步长必须 >= 1。")
    if window < 3:
        raise RuntimeError("应变窗口必须 >= 3。")
    if window % 2 == 0:
        raise RuntimeError("应变窗口必须为奇数；请使用 5、7 等值。")
    if search_radius < 1:
        raise RuntimeError("DIC 搜索半径必须 > 0。")
    if not (0.0 <= zncc_min <= 1.0):
        raise RuntimeError("ZNCC 下限应在 0 到 1 之间。")
    if smooth_sigma < 0:
        raise RuntimeError("高斯平滑 σ 不能为负。")
    if not (1 <= pyramid_levels <= 8):
        raise RuntimeError("金字塔层数应在 1 到 8 之间。")
    if not (0.0 < pyramid_scale < 1.0):
        raise RuntimeError("金字塔缩放应在 0 到 1 之间。")
    if solver not in DIC_SOLVERS:
        raise RuntimeError("求解器必须是 IC-GN 或 IC-LM。")

    X, Y = build_poi_grid(roi, subset, step, image_shape)
    if not poi_grid_is_usable(X, Y, min_rows=3, min_cols=3):
        raise RuntimeError("当前 ROI / 子集 / 步长至少需要 3×3 个可分析的 2D POI，请增大 ROI 或减小步长。")
    if int(settings.get("strain_degree", 2)) not in (1, 2):
        raise RuntimeError("拟合最高阶次必须为 1 或 2。")
    outlier = float(settings.get("outlier_threshold_px", 1.0))
    if not np.isfinite(outlier) or outlier < 0:
        raise RuntimeError("异常位移阈值必须为非负有限数值。")
    mask, _, _ = _core._resolve_specimen_mask(reference8, roi, settings)
    if mask is not None:
        from scipy import ndimage
        supported = ndimage.minimum_filter(mask.astype(np.uint8), size=subset, mode="constant", cval=0)
        if np.count_nonzero(supported[Y.astype(int), X.astype(int)]) < 9:
            raise RuntimeError("遮罩内至少需要 9 个能容纳完整子集的测量点。请检查遮罩或减小子集。")


def fullfield_field_has_finite_strain(field):
    """Check that a DIC frame contains at least one valid finite strain point."""
    valid = np.asarray(field.get("valid", []), dtype=bool).ravel()
    if valid.size == 0 or not valid.any():
        return False
    try:
        finite_strain = (
            np.isfinite(np.asarray(field["Exx"], dtype=float).ravel())
            & np.isfinite(np.asarray(field["Eyy"], dtype=float).ravel())
            & np.isfinite(np.asarray(field["Exy"], dtype=float).ravel())
        )
    except (KeyError, TypeError, ValueError):
        return False
    if finite_strain.size != valid.size:
        return False
    return bool(np.any(valid & finite_strain))


def integer_cc_guess(reference, deformed, x, y, subset_size, search_radius):
    """Integer template-match (OpenCV ZNCC) plus quadratic sub-pixel peak."""
    reference = _as_float_image(reference)
    deformed = _as_float_image(deformed)
    if reference.shape != deformed.shape:
        raise ValueError(
            "reference and deformed image dimensions must match for full-field DIC."
        )
    subset_size = _odd_subset_size(subset_size)
    half = subset_size // 2
    H, W = reference.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    if ix - half < 0 or iy - half < 0 or ix + half >= W or iy + half >= H:
        return 0.0, 0.0, -1.0
    tpl = reference[iy - half : iy + half + 1, ix - half : ix + half + 1]
    radius = max(1, int(search_radius))
    sx1 = max(0, ix - half - radius)
    sy1 = max(0, iy - half - radius)
    sx2 = min(W, ix + half + 1 + radius)
    sy2 = min(H, iy + half + 1 + radius)
    search = deformed[sy1:sy2, sx1:sx2]
    if search.shape[0] < tpl.shape[0] or search.shape[1] < tpl.shape[1]:
        return 0.0, 0.0, -1.0
    if not has_nonzero_variance(tpl) or not has_nonzero_variance(search):
        return 0.0, 0.0, -1.0
    corr = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    finite_corr = np.isfinite(corr)
    if not finite_corr.any():
        return 0.0, 0.0, -1.0
    safe_corr = np.where(finite_corr, corr, -1.0).astype(np.float32, copy=False)
    _, max_val, _, max_loc = cv2.minMaxLoc(safe_corr)
    if not np.isfinite(max_val):
        return 0.0, 0.0, -1.0
    px, py = max_loc
    candidate_patch = search[py : py + tpl.shape[0], px : px + tpl.shape[1]]
    if candidate_patch.shape != tpl.shape or not has_nonzero_variance(candidate_patch):
        return 0.0, 0.0, -1.0
    dx, dy = subpixel_peak(safe_corr, px, py)
    u = (sx1 + px + dx) - (ix - half)
    v = (sy1 + py + dy) - (iy - half)
    return float(u), float(v), float(max_val)


def _compose_warp_inverse(p, dp):
    Mc = np.array(
        [[1.0 + p[1], p[2], p[0]], [p[4], 1.0 + p[5], p[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    Md = np.array(
        [[1.0 + dp[1], dp[2], dp[0]], [dp[4], 1.0 + dp[5], dp[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    try:
        Mn = Mc @ np.linalg.inv(Md)
    except np.linalg.LinAlgError:
        return None
    return np.array(
        [Mn[0, 2], Mn[0, 0] - 1.0, Mn[0, 1], Mn[1, 2], Mn[1, 0], Mn[1, 1] - 1.0],
        dtype=np.float64,
    )


def _refine_subset_ic(
    reference,
    deformed,
    x,
    y,
    subset_size,
    p0=None,
    method="GN",
    max_iter=25,
    tol=1e-3,
):
    """Inverse-compositional first-order affine subset match (ZNSSD / ZNCC)."""
    reference = _as_float_image(reference)
    deformed = _as_float_image(deformed)
    subset_size = _odd_subset_size(subset_size)
    half = subset_size // 2
    Himg, Wimg = reference.shape[:2]
    xi = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(xi, xi)
    xf = xx.ravel()
    yf = yy.ravel()
    x0 = float(x)
    y0 = float(y)
    ix, iy = int(round(x0)), int(round(y0))
    if ix - half < 0 or iy - half < 0 or ix + half >= Wimg or iy + half >= Himg:
        return None
    f = reference[iy - half : iy + half + 1, ix - half : ix + half + 1].astype(np.float64)
    if not np.isfinite(f).all() or not has_nonzero_variance(f):
        return None
    fy, fx = np.gradient(f)
    f_tilde = f - f.mean()
    f_norm = float(np.sqrt(np.sum(f_tilde * f_tilde)))
    if f_norm < 1e-8:
        return None
    fn = f_tilde / f_norm
    fxn = fx.ravel() / f_norm
    fyn = fy.ravel() / f_norm
    sd = np.column_stack([fxn, fxn * xf, fxn * yf, fyn, fyn * xf, fyn * yf])
    hess = sd.T @ sd
    try:
        hess_inv = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return None

    p = np.zeros(6, dtype=np.float64) if p0 is None else np.asarray(p0, dtype=np.float64).reshape(6).copy()
    mu = 0.01
    last_cost = np.inf
    best = None
    xx32 = xx.astype(np.float32)
    yy32 = yy.astype(np.float32)

    for it in range(int(max_iter)):
        map_x = (x0 + p[0] + (1.0 + p[1]) * xx32 + p[2] * yy32).astype(np.float32)
        map_y = (y0 + p[3] + p[4] * xx32 + (1.0 + p[5]) * yy32).astype(np.float32)
        if map_x.min() < 1 or map_y.min() < 1 or map_x.max() > Wimg - 2 or map_y.max() > Himg - 2:
            break
        g = cv2.remap(
            deformed,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)
        g_tilde = g - g.mean()
        g_norm = float(np.sqrt(np.sum(g_tilde * g_tilde)))
        if not np.isfinite(g).all() or g_norm < 1e-8:
            break
        gn = g_tilde / g_norm
        zncc = float(np.sum(fn * gn))
        residual = (fn - gn).ravel()
        cost = float(np.dot(residual, residual))
        if best is None or zncc > best["zncc"]:
            best = {
                "u": float(p[0]),
                "v": float(p[3]),
                "p": p.copy(),
                "zncc": zncc,
                "iters": it + 1,
            }

        b = sd.T @ residual
        if method == "LM":
            damped = hess + mu * np.diag(np.diag(hess))
            try:
                dp = -np.linalg.solve(damped, b)
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
        else:
            dp = -hess_inv @ b

        if max(abs(dp[0]), abs(dp[3])) < tol or zncc > 0.9995:
            break

        p_new = _compose_warp_inverse(p, dp)
        if p_new is None:
            break

        if method == "LM":
            if cost <= last_cost * 1.0000001:
                mu = max(mu / 10.0, 1e-8)
                p = p_new
                last_cost = cost
            else:
                mu *= 10.0
                if mu > 1e8:
                    break
                continue
        else:
            if cost > last_cost:
                break
            p = p_new
            last_cost = cost

    return best


def refine_subset_icgn(reference, deformed, x, y, subset_size, p0=None, max_iter=25, tol=1e-3):
    return _refine_subset_ic(
        reference, deformed, x, y, subset_size, p0=p0, method="GN", max_iter=max_iter, tol=tol
    )


def refine_subset_iclm(reference, deformed, x, y, subset_size, p0=None, max_iter=25, tol=1e-3):
    return _refine_subset_ic(
        reference, deformed, x, y, subset_size, p0=p0, method="LM", max_iter=max_iter, tol=tol
    )


def _nan_gaussian(arr, sigma):
    if sigma is None or float(sigma) <= 0:
        return arr
    orig_nan = ~np.isfinite(arr)
    mask = np.isfinite(arr).astype(np.float32)
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    ksize = int(max(3, 2 * int(3 * float(sigma)) + 1)) | 1
    sm = cv2.GaussianBlur(filled, (ksize, ksize), float(sigma))
    wt = cv2.GaussianBlur(mask, (ksize, ksize), float(sigma))
    out = sm / np.maximum(wt, 1e-6)
    out[wt < 0.15] = np.nan
    out[orig_nan] = np.nan
    return out.astype(np.float64)


def compute_strain_fields(X, Y, U, V, *, window=5, smooth_sigma=0.0):
    """Windowed plane fit of u,v → Green-Lagrange and infinitesimal strain grids."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64).copy()
    V = np.asarray(V, dtype=np.float64).copy()
    if smooth_sigma and float(smooth_sigma) > 0:
        U = _nan_gaussian(U, smooth_sigma)
        V = _nan_gaussian(V, smooth_sigma)

    ny, nx = X.shape
    win = _odd_window_size(window)
    half = win // 2
    Exx = np.full((ny, nx), np.nan)
    Eyy = np.full((ny, nx), np.nan)
    Exy = np.full((ny, nx), np.nan)
    exx = np.full((ny, nx), np.nan)
    eyy = np.full((ny, nx), np.nan)
    exy = np.full((ny, nx), np.nan)
    dudx = np.full((ny, nx), np.nan)
    dudy = np.full((ny, nx), np.nan)
    dvdx = np.full((ny, nx), np.nan)
    dvdy = np.full((ny, nx), np.nan)

    for i in range(ny):
        i0, i1 = max(0, i - half), min(ny, i + half + 1)
        for j in range(nx):
            j0, j1 = max(0, j - half), min(nx, j + half + 1)
            xs = X[i0:i1, j0:j1].ravel()
            ys = Y[i0:i1, j0:j1].ravel()
            us = U[i0:i1, j0:j1].ravel()
            vs = V[i0:i1, j0:j1].ravel()
            if not (np.isfinite(U[i, j]) and np.isfinite(V[i, j])):
                continue
            ok = np.isfinite(us) & np.isfinite(vs) & np.isfinite(xs) & np.isfinite(ys)
            if int(ok.sum()) < 6:
                continue
            A = np.column_stack([np.ones(int(ok.sum())), xs[ok] - X[i, j], ys[ok] - Y[i, j]])
            if np.linalg.matrix_rank(A) < 3:
                continue
            try:
                au, *_ = np.linalg.lstsq(A, us[ok], rcond=None)
                av, *_ = np.linalg.lstsq(A, vs[ok], rcond=None)
            except np.linalg.LinAlgError:
                continue
            du_dx, du_dy = float(au[1]), float(au[2])
            dv_dx, dv_dy = float(av[1]), float(av[2])
            Fxx, Fxy, Fyx, Fyy = 1.0 + du_dx, du_dy, dv_dx, 1.0 + dv_dy
            dudx[i, j] = du_dx
            dudy[i, j] = du_dy
            dvdx[i, j] = dv_dx
            dvdy[i, j] = dv_dy
            Exx[i, j] = 0.5 * (Fxx * Fxx + Fyx * Fyx - 1.0)
            Eyy[i, j] = 0.5 * (Fxy * Fxy + Fyy * Fyy - 1.0)
            Exy[i, j] = 0.5 * (Fxx * Fxy + Fyx * Fyy)
            exx[i, j] = du_dx
            eyy[i, j] = dv_dy
            exy[i, j] = 0.5 * (du_dy + dv_dx)

    return {
        "Exx": Exx,
        "Eyy": Eyy,
        "Exy": Exy,
        "exx": exx,
        "eyy": eyy,
        "exy": exy,
        "dudx": dudx,
        "dudy": dudy,
        "dvdx": dvdx,
        "dvdy": dvdy,
        "U": U,
        "V": V,
        "window": win,
    }


def run_2d_dic(
    reference,
    deformed,
    roi,
    *,
    subset_size=21,
    step=5,
    solver=DIC_SOLVER_ICGN,
    search_radius=None,
    max_iter=25,
    conv_tol=1e-3,
    zncc_min=0.75,
    strain_window=5,
    smooth_sigma=0.0,
    progress_callback=None,
):
    """
    Correlate a rectangular ROI with IC-GN or IC-LM.

    Failed or out-of-ROI points stay NaN (not interpolated). Strain is a windowed
    local fit of the displacement field, optionally Gaussian-smoothed first.
    """
    requested_subset_size = int(subset_size)
    requested_strain_window = int(strain_window)
    reference = _as_float_image(reference)
    deformed = _as_float_image(deformed)
    if reference.shape != deformed.shape:
        raise ValueError(
            "reference and deformed image dimensions must match for full-field DIC."
        )
    if not rect_is_inside_image(roi, reference.shape):
        raise ValueError("roi must be a positive rectangle fully inside the reference image.")
    subset_size = _odd_subset_size(subset_size)
    step = max(1, int(step))
    solver_name = str(solver).strip().upper().replace("_", "-")
    if solver_name not in (DIC_SOLVER_ICGN, DIC_SOLVER_ICLM):
        raise ValueError(f"solver must be {DIC_SOLVER_ICGN} or {DIC_SOLVER_ICLM}.")
    refine = refine_subset_icgn if solver_name == DIC_SOLVER_ICGN else refine_subset_iclm
    if search_radius is None:
        search_radius = max(8, subset_size // 2)

    X, Y = build_poi_grid(roi, subset_size, step, reference.shape)
    ny, nx = X.shape
    U = np.full((ny, nx), np.nan, dtype=np.float64)
    V = np.full((ny, nx), np.nan, dtype=np.float64)
    Z = np.full((ny, nx), np.nan, dtype=np.float64)
    valid = np.zeros((ny, nx), dtype=bool)
    P = np.full((ny, nx, 6), np.nan, dtype=np.float64)
    total = max(ny * nx, 1)
    u_prev = 0.0
    v_prev = 0.0

    for i in range(ny):
        for j in range(nx):
            px = float(X[i, j])
            py = float(Y[i, j])
            u0, v0, cc = integer_cc_guess(reference, deformed, px, py, subset_size, search_radius)
            if cc < 0.35:
                u0, v0 = u_prev, v_prev
            result = refine(
                reference,
                deformed,
                px,
                py,
                subset_size,
                p0=[u0, 0.0, 0.0, v0, 0.0, 0.0],
                max_iter=max_iter,
                tol=conv_tol,
            )
            try:
                result_p = np.asarray(result.get("p"), dtype=float).reshape(6)
                result_finite = (
                    np.isfinite(result["u"])
                    and np.isfinite(result["v"])
                    and np.isfinite(result["zncc"])
                    and np.isfinite(result_p).all()
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                result_p = None
                result_finite = False
            if result is not None and result_finite and result["zncc"] >= float(zncc_min):
                U[i, j] = result["u"]
                V[i, j] = result["v"]
                Z[i, j] = result["zncc"]
                valid[i, j] = True
                P[i, j, :] = result_p
                u_prev, v_prev = result["u"], result["v"]
            if progress_callback is not None:
                progress_callback(i * nx + j + 1, total)

    strains = compute_strain_fields(X, Y, U, V, window=strain_window, smooth_sigma=smooth_sigma)
    return {
        "x": X.ravel(),
        "y": Y.ravel(),
        "u": strains["U"].ravel(),
        "v": strains["V"].ravel(),
        "zncc": Z.ravel(),
        "valid": valid.ravel(),
        "Exx": strains["Exx"].ravel(),
        "Eyy": strains["Eyy"].ravel(),
        "Exy": strains["Exy"].ravel(),
        "exx": strains["exx"].ravel(),
        "eyy": strains["eyy"].ravel(),
        "exy": strains["exy"].ravel(),
        "X": X,
        "Y": Y,
        "U": strains["U"],
        "V": strains["V"],
        "P": P,
        "subset_size": subset_size,
        "step": step,
        "solver": solver_name,
        "roi": tuple(int(round(v)) for v in roi),
        "zncc_min": float(zncc_min),
        "smooth_sigma": float(smooth_sigma or 0.0),
        "requested_subset_size": requested_subset_size,
        "requested_strain_window": requested_strain_window,
        "strain_window": int(strains["window"]),
        "effective_subset_size": subset_size,
        "effective_strain_window": int(strains["window"]),
        "Exx_grid": strains["Exx"],
        "Eyy_grid": strains["Eyy"],
        "Exy_grid": strains["Exy"],
        "exx_grid": strains["exx"],
        "eyy_grid": strains["eyy"],
        "exy_grid": strains["exy"],
    }


def run_2d_dic_sequence(reference, deformed_frames, roi, **kwargs):
    """Correlate each deformed frame to the same reference (sequence batch)."""
    return [run_2d_dic(reference, frame, roi, **kwargs) for frame in deformed_frames]


def dic_field_to_dataframe(field):
    """Tabular export of a run_2d_dic result; invalid points remain NaN."""
    return pd.DataFrame(
        {
            "x": np.asarray(field["x"], dtype=float),
            "y": np.asarray(field["y"], dtype=float),
            "u": np.asarray(field["u"], dtype=float),
            "v": np.asarray(field["v"], dtype=float),
            "zncc": np.asarray(field["zncc"], dtype=float),
            "valid": np.asarray(field["valid"], dtype=bool),
            "Exx": np.asarray(field["Exx"], dtype=float).ravel(),
            "Eyy": np.asarray(field["Eyy"], dtype=float).ravel(),
            "Exy": np.asarray(field["Exy"], dtype=float).ravel(),
            "exx": np.asarray(field["exx"], dtype=float).ravel(),
            "eyy": np.asarray(field["eyy"], dtype=float).ravel(),
            "exy": np.asarray(field["exy"], dtype=float).ravel(),
        }
    )


def write_dic_field_txt(field, path):
    table = dic_field_to_dataframe(field)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(table.columns)
    lines = ["\t".join(columns)]
    for row in table.itertuples(index=False):
        cells = []
        for col, value in zip(columns, row):
            if col == "valid":
                cells.append("1" if bool(value) else "0")
            elif pd.isna(value) or not np.isfinite(float(value)):
                cells.append("NaN")
            else:
                cells.append(f"{float(value):.8f}")
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_dic_field_parameters(field, path):
    """Write human-readable provenance and solver parameters for one field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = dict(field.get("provenance") or {})
    defaults = {
        "analysis_mode": ANALYSIS_MODE_FULLFIELD,
        "roi": field.get("roi"),
        "subset_size_px": field.get("subset_size"),
        "effective_subset_size_px": field.get("effective_subset_size", field.get("subset_size")),
        "requested_subset_size": field.get("requested_subset_size", field.get("subset_size")),
        "step_px": field.get("step"),
        "solver": field.get("solver"),
        "zncc_min": field.get("zncc_min"),
        "strain_window": field.get("strain_window"),
        "effective_strain_window": field.get("effective_strain_window", field.get("strain_window")),
        "requested_strain_window": field.get("requested_strain_window", field.get("strain_window")),
        "smooth_sigma": field.get("smooth_sigma"),
    }
    for key, value in defaults.items():
        provenance.setdefault(key, value)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ezDIC full-field DIC provenance and parameters\n")
        handle.write("===============================================\n")
        for key, value in provenance.items():
            if isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            handle.write(f"{key} = {value}\n")
    return path


def style_dic_colorbar(cbar, preset, label):
    cbar.set_label(label, fontsize=preset["colorbar_label_size"])
    cbar.ax.tick_params(labelsize=preset["colorbar_tick_size"])
    return cbar


def add_dic_colorbar(fig, ax, mappable, label, preset_name="publication"):
    """Attach a publication-styled colorbar; used by exports and the in-app field viewer."""
    preset = get_plot_preset(preset_name)
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=preset["colorbar_fraction"],
        pad=preset["colorbar_pad"],
    )
    return style_dic_colorbar(cbar, preset, label)


def render_dic_field_on_axes(ax, field, component="u", *, cmap="turbo"):
    """Draw a POI-grid colormap of one DIC component on an existing axes."""
    if component not in DIC_FIELD_COMPONENTS:
        raise ValueError(f"unknown DIC component: {component}")
    X = np.asarray(field["X"], dtype=float)
    Y = np.asarray(field["Y"], dtype=float)
    raw = field[component]
    values = np.asarray(raw, dtype=float)
    if values.ndim == 1:
        values = values.reshape(X.shape)
    mesh = ax.pcolormesh(X, Y, values, cmap=cmap, shading="nearest")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return mesh


def plot_dic_field_map(field, path, component="u", title=None, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax, preset = create_plot_figure(preset_name)
    mesh = render_dic_field_on_axes(ax, field, component)
    label = DIC_COMPONENT_LABELS.get(component, component)
    add_dic_colorbar(fig, ax, mesh, label, preset_name=preset_name)
    if title is None:
        title = label
    style_publication_axes(ax, preset, "x (px)", "y (px)", title, show_legend=False)
    ax.set_aspect("equal")
    save_plot_figure(fig, path, preset_name)
    return path


def overlay_dic_field_on_image(image, field, component="u", *, alpha=0.55, cmap="turbo"):
    """Blend a DIC colormap onto the specimen image (uint8 RGB)."""
    gray = _as_float_image(image)
    h, w = gray.shape[:2]
    base = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    X = np.asarray(field["X"], dtype=float)
    Y = np.asarray(field["Y"], dtype=float)
    values = np.asarray(field[component], dtype=float)
    if values.ndim == 1:
        values = values.reshape(X.shape)
    finite = np.isfinite(values)
    if not finite.any():
        return base
    vmin = float(np.nanpercentile(values[finite], 2))
    vmax = float(np.nanpercentile(values[finite], 98))
    if not np.isfinite(vmin) or abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6
    norm = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    cmap_fn = plt.get_cmap(cmap)
    color = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    half = max(1, int(field.get("step", 5)) // 2)
    ny, nx = X.shape
    for i in range(ny):
        for j in range(nx):
            if not finite[i, j]:
                continue
            cx = int(round(X[i, j]))
            cy = int(round(Y[i, j]))
            x0, x1 = max(0, cx - half), min(w, cx + half + 1)
            y0, y1 = max(0, cy - half), min(h, cy + half + 1)
            rgb = cmap_fn(float(norm[i, j]))[:3]
            color[y0:y1, x0:x1, :] += np.float32(rgb)
            weight[y0:y1, x0:x1] += 1.0
    mask = weight > 0
    color[mask] /= weight[mask][:, None]
    overlay = (color * 255.0).astype(np.uint8)
    out = base.copy()
    out[mask] = (
        (1.0 - alpha) * base[mask].astype(np.float32) + alpha * overlay[mask].astype(np.float32)
    ).astype(np.uint8)
    return out


def export_dic_field_outputs(field, output_dir, *, stem="dic_field", preset_name="publication"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = write_dic_field_txt(field, output_dir / f"{stem}.txt")
    plot_paths = []
    for component in ("u", "v", "Exx", "Eyy", "Exy"):
        plot_paths.append(
            plot_dic_field_map(
                field,
                output_dir / f"{stem}_{component}.png",
                component=component,
                preset_name=preset_name,
            )
        )
    csv_path = output_dir / f"{stem}.csv"
    dic_field_to_dataframe(field).to_csv(csv_path, index=False)
    parameters_path = write_dic_field_parameters(field, output_dir / f"{stem}_parameters.txt")
    return {"txt": table_path, "csv": csv_path, "plots": plot_paths, "parameters": parameters_path}


def _frame_column(df):
    if "frame_global_1based" in df.columns:
        return "frame_global_1based"
    if "frame_local_1based" in df.columns:
        return "frame_local_1based"
    return "frame"


def _format_origin_float(value):
    if pd.isna(value) or not np.isfinite(float(value)):
        return "NaN"
    return f"{float(value):.8f}"


def _engineering_to_true(value):
    if pd.isna(value) or not np.isfinite(float(value)) or (1.0 + float(value)) <= 0:
        return np.nan
    return math.log1p(float(value))


def _format_origin_value(column, value):
    if column.startswith("ValidGroupCount_"):
        return "NaN" if pd.isna(value) else str(int(value))
    return _format_origin_float(value)


def normalize_roi_role(role):
    role = str(role or "none").strip()
    if role not in ROI_ROLE_VALUES:
        return "none"
    return role


def poisson_roles_are_configured(groups):
    return any(normalize_roi_role(g.get("role", "none")) != "none" for g in groups)


def get_poisson_role_groups(groups):
    axial = [g for g in groups if normalize_roi_role(g.get("role", "none")) == "axial"]
    transverse = [g for g in groups if normalize_roi_role(g.get("role", "none")) == "transverse"]
    return axial, transverse


def _validate_single_actual_mode(groups, role_label):
    modes = sorted({str(g.get("actual_mode", "unknown") or "unknown").strip() for g in groups})
    if len(modes) > 1:
        raise RuntimeError(f"{role_label} ROI 组的 actual_mode 必须一致；当前为 {', '.join(modes)}。")


def validate_poisson_role_groups(groups):
    if not poisson_roles_are_configured(groups):
        return False

    axial, transverse = get_poisson_role_groups(groups)
    errors = []
    if len(axial) < 1:
        errors.append(f"拉伸方向 ROI 组必须至少有 1 个；当前为 {len(axial)} 个。")
    if len(transverse) < 1:
        errors.append(f"横向方向 ROI 组必须至少有 1 个；当前为 {len(transverse)} 个。")
    if errors:
        raise RuntimeError("\n".join(errors))
    axial_names = {g.get("name") for g in axial}
    transverse_names = {g.get("name") for g in transverse}
    if axial_names & transverse_names:
        raise RuntimeError("同一个 ROI 组不能同时作为拉伸方向和横向方向。")
    _validate_single_actual_mode(axial, "拉伸方向")
    _validate_single_actual_mode(transverse, "横向方向")
    return True


def build_core_strain_table(gdf):
    """
    Build the minimal Origin-friendly strain table:
    Frame, EngineeringStrain, TrueStrain.
    True strain is recomputed from engineering strain to keep export logic explicit.
    """
    frame_col = _frame_column(gdf)
    frames = pd.to_numeric(gdf[frame_col], errors="coerce")
    eng = pd.to_numeric(gdf["engineering_strain"], errors="coerce")

    true_values = [_engineering_to_true(value) for value in eng]

    out = pd.DataFrame(
        {
            "Frame": frames.astype("Int64"),
            "EngineeringStrain": eng.astype(float),
            "TrueStrain": np.array(true_values, dtype=float),
        }
    )
    return out.reset_index(drop=True)


def write_origin_txt(gdf, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_core_strain_table(gdf)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("Frame\tEngineeringStrain\tTrueStrain\n")
        for _, row in table.iterrows():
            frame = "NaN" if pd.isna(row["Frame"]) else str(int(row["Frame"]))
            f.write(
                f"{frame}\t"
                f"{_format_origin_float(row['EngineeringStrain'])}\t"
                f"{_format_origin_float(row['TrueStrain'])}\n"
            )


def _group_engineering_strain_table(df, group_name, output_column):
    gdf = df[df["group"] == group_name].copy()
    if gdf.empty:
        return pd.DataFrame(columns=["Frame", output_column])

    table = build_core_strain_table(gdf)
    strain = table["EngineeringStrain"].astype(float)
    if "accepted" in gdf.columns:
        accepted = gdf["accepted"].astype(bool).reset_index(drop=True)
        strain = strain.where(accepted, np.nan)

    return pd.DataFrame(
        {
            "Frame": table["Frame"],
            output_column: strain,
        }
    ).reset_index(drop=True)


def _frame_table(df):
    frame_col = _frame_column(df)
    frames = pd.to_numeric(df[frame_col], errors="coerce").dropna().drop_duplicates().sort_values()
    return pd.DataFrame({"Frame": frames.astype("Int64")}).reset_index(drop=True)


def _mean_group_key(group):
    role = normalize_roi_role(group.get("role", "none"))
    actual_mode = str(group.get("actual_mode", "unknown") or "unknown").strip()
    return safe_name(f"{role}_{actual_mode}")


def _mean_group_specs(groups):
    specs = {}
    for group in groups or []:
        key = _mean_group_key(group)
        if key not in specs:
            specs[key] = {"key": key, "groups": []}
        specs[key]["groups"].append(group)
    return list(specs.values())


def _mean_engineering_strain_table_for_groups(df, groups, output_column):
    out = _frame_table(df)
    if out.empty:
        return pd.DataFrame(columns=["Frame", output_column])

    strain_cols = []
    merged = out.copy()
    for idx, group in enumerate(groups, start=1):
        col = f"__strain_{idx}"
        group_table = _group_engineering_strain_table(df, group.get("name"), col)
        merged = pd.merge(merged, group_table, on="Frame", how="left")
        strain_cols.append(col)

    if not strain_cols:
        merged[output_column] = np.nan
    else:
        strains = merged[strain_cols].apply(pd.to_numeric, errors="coerce")
        counts = strains.notna().sum(axis=1)
        merged[output_column] = strains.mean(axis=1, skipna=True).where(counts > 0, np.nan)

    return merged[["Frame", output_column]].reset_index(drop=True)


def build_mean_strain_table(df, groups):
    out = _frame_table(df)
    if out.empty or not groups:
        return out

    for spec in _mean_group_specs(groups):
        key = spec["key"]
        merged = out.copy()
        strain_cols = []
        for idx, group in enumerate(spec["groups"], start=1):
            col = f"__{key}_{idx}"
            group_table = _group_engineering_strain_table(df, group.get("name"), col)
            merged = pd.merge(merged, group_table, on="Frame", how="left")
            strain_cols.append(col)

        strains = merged[strain_cols].apply(pd.to_numeric, errors="coerce")
        counts = strains.notna().sum(axis=1)
        mean = strains.mean(axis=1, skipna=True).where(counts > 0, np.nan)
        std = strains.std(axis=1, skipna=True, ddof=1).where(counts >= 2, np.nan)
        sem = (std / np.sqrt(counts.astype(float))).where(counts >= 2, np.nan)

        out[f"MeanEngineeringStrain_{key}"] = mean
        out[f"MeanTrueStrain_{key}"] = mean.apply(_engineering_to_true)
        out[f"StdEngineeringStrain_{key}"] = std
        out[f"SemEngineeringStrain_{key}"] = sem
        out[f"ValidGroupCount_{key}"] = counts.astype(int)

    return out.reset_index(drop=True)


def build_poisson_ratio_table(df, groups, min_abs_axial=POISSON_MIN_ABS_AXIAL_ENGINEERING_STRAIN):
    if not validate_poisson_role_groups(groups):
        raise RuntimeError("请先设置 1 个拉伸方向 ROI 组和 1 个横向方向 ROI 组。")
    axial_groups, transverse_groups = get_poisson_role_groups(groups)

    axial = _mean_engineering_strain_table_for_groups(df, axial_groups, "AxialEngineeringStrain")
    transverse = _mean_engineering_strain_table_for_groups(df, transverse_groups, "TransverseEngineeringStrain")
    merged = pd.merge(axial, transverse, on="Frame", how="outer").sort_values("Frame").reset_index(drop=True)

    axial_strain = pd.to_numeric(merged["AxialEngineeringStrain"], errors="coerce").astype(float)
    transverse_strain = pd.to_numeric(merged["TransverseEngineeringStrain"], errors="coerce").astype(float)
    valid = (
        axial_strain.notna()
        & transverse_strain.notna()
        & np.isfinite(axial_strain)
        & np.isfinite(transverse_strain)
        & (axial_strain.abs() >= float(min_abs_axial))
    )
    poisson = pd.Series(np.nan, index=merged.index, dtype=float)
    poisson.loc[valid] = -transverse_strain.loc[valid] / axial_strain.loc[valid]
    merged["PoissonRatio"] = poisson

    return merged[
        ["Frame", "AxialEngineeringStrain", "TransverseEngineeringStrain", "PoissonRatio"]
    ].reset_index(drop=True)


def build_all_groups_strain_table(df, groups=None):
    tables = []
    for gname, gdf in df.groupby("group", sort=False):
        sg = safe_name(gname)
        table = build_core_strain_table(gdf).rename(
            columns={
                "EngineeringStrain": f"EngineeringStrain_{sg}",
                "TrueStrain": f"TrueStrain_{sg}",
            }
        )
        tables.append(table)

    if not tables:
        return pd.DataFrame(columns=["Frame"])

    merged = tables[0]
    for table in tables[1:]:
        merged = pd.merge(merged, table, on="Frame", how="outer")
    merged = merged.sort_values("Frame").reset_index(drop=True)

    if groups is not None:
        mean_table = build_mean_strain_table(df, groups)
        if len(mean_table.columns) > 1:
            merged = pd.merge(merged, mean_table, on="Frame", how="outer")
            merged = merged.sort_values("Frame").reset_index(drop=True)

    if groups is not None and poisson_roles_are_configured(groups):
        poisson = build_poisson_ratio_table(df, groups)
        merged = pd.merge(merged, poisson, on="Frame", how="outer")
        merged = merged.sort_values("Frame").reset_index(drop=True)

    return merged


def write_all_groups_origin_txt(df, path, groups=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_all_groups_strain_table(df, groups)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(table.columns) + "\n")
        for _, row in table.iterrows():
            values = []
            for col in table.columns:
                if col == "Frame":
                    values.append("NaN" if pd.isna(row[col]) else str(int(row[col])))
                else:
                    values.append(_format_origin_value(col, row[col]))
            f.write("\t".join(values) + "\n")


def write_mean_groups_origin_txt(df, groups, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_mean_strain_table(df, groups)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(table.columns) + "\n")
        for _, row in table.iterrows():
            values = []
            for col in table.columns:
                if col == "Frame":
                    values.append("NaN" if pd.isna(row[col]) else str(int(row[col])))
                else:
                    values.append(_format_origin_value(col, row[col]))
            f.write("\t".join(values) + "\n")


def write_poisson_ratio_txt(df, groups, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_poisson_ratio_table(df, groups)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("Frame\tAxialEngineeringStrain\tTransverseEngineeringStrain\tPoissonRatio\n")
        for _, row in table.iterrows():
            frame = "NaN" if pd.isna(row["Frame"]) else str(int(row["Frame"]))
            f.write(
                f"{frame}\t"
                f"{_format_origin_float(row['AxialEngineeringStrain'])}\t"
                f"{_format_origin_float(row['TransverseEngineeringStrain'])}\t"
                f"{_format_origin_float(row['PoissonRatio'])}\n"
            )


def build_origin_project_tables(df, groups):
    groups = list(groups or [])
    tables = []

    for group in groups:
        gname = group.get("name")
        sg = safe_name(gname)
        gdf = df[df["group"] == gname].copy()
        tables.append((f"strain_{sg}", build_core_strain_table(gdf)))

    tables.append(("strain_all_groups", build_all_groups_strain_table(df, groups)))

    mean_table = build_mean_strain_table(df, groups)
    if len(mean_table.columns) > 1:
        tables.append(("strain_mean_groups", mean_table))

    if poisson_roles_are_configured(groups):
        tables.append(("poisson_ratio", build_poisson_ratio_table(df, groups)))

    return tables


def _load_originpro_module():
    try:
        import originpro as op
    except ImportError as exc:
        raise RuntimeError(
            "无法导入 originpro。请在 Windows + OriginPro 2021+ 环境中安装 originpro Python 包后再导出 OPJU。"
        ) from exc
    return op


def write_origin_opju_project(df, groups, path, origin_module=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    op = origin_module if origin_module is not None else _load_originpro_module()

    try:
        op.new(asksave=True)
        for table_name, table in build_origin_project_tables(df, groups):
            worksheet = op.new_sheet("w", lname=table_name)
            if worksheet is None:
                raise RuntimeError(f"无法创建 Origin worksheet：{table_name}")
            worksheet.from_df(table)

        if not op.save(str(path)):
            raise RuntimeError(f"保存 Origin OPJU 项目失败：{path}")
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("保存 Origin OPJU 项目失败") or message.startswith("无法"):
            raise
        raise RuntimeError(f"生成 Origin OPJU 项目失败：{exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"生成 Origin OPJU 项目失败：{exc}") from exc

    return path


def get_plot_preset(name="publication"):
    preset = PLOT_EXPORT_PRESETS.get(name, PLOT_EXPORT_PRESETS["publication"])
    return dict(preset)


def create_plot_figure(preset_name="publication"):
    preset = get_plot_preset(preset_name)
    fig, ax = plt.subplots(
        figsize=preset["figsize"],
        constrained_layout=preset["constrained_layout"],
    )
    fig.patch.set_facecolor("white")
    ax.set_prop_cycle(color=PLOT_COLOR_CYCLE)
    return fig, ax, preset


def style_publication_axes(ax, preset, xlabel, ylabel, title=None, show_legend=True):
    ax.set_xlabel(xlabel, fontsize=preset["label_size"])
    ax.set_ylabel(ylabel, fontsize=preset["label_size"])
    if title:
        ax.set_title(title, fontsize=preset["title_size"], pad=8)
    ax.tick_params(axis="both", labelsize=preset["tick_size"], width=preset["axis_line_width"], length=3.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(preset["axis_line_width"])
    ax.grid(True, color="#b8c2cc", alpha=preset["grid_alpha"], linewidth=0.55)
    ax.margins(x=0.02, y=0.08)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncol = 2 if len(handles) > 6 else 1
            ax.legend(
                loc="best",
                frameon=False,
                fontsize=preset["legend_size"],
                handlelength=1.6,
                borderaxespad=0.3,
                ncol=ncol,
            )


def save_plot_figure(fig, path, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preset = get_plot_preset(preset_name)
    fig.savefig(path, dpi=preset["dpi"], bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def publication_figure_paths(folder, stem):
    folder = Path(folder)
    return [folder / f"{stem}.{ext}" for ext in PLOT_EXPORT_FORMATS]


def plot_engineering_strain(gdf, path, title, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = build_core_strain_table(gdf)
    frame = table["Frame"].astype(float)
    strain = table["EngineeringStrain"].astype(float)
    accepted = gdf["accepted"].astype(bool).reset_index(drop=True) if "accepted" in gdf.columns else strain.notna()
    accept_mode = gdf["accept_mode"].astype(str).reset_index(drop=True) if "accept_mode" in gdf.columns else pd.Series([""] * len(gdf))

    rejected_mask = (~accepted) | strain.isna()
    adaptive_mask = accept_mode.eq("adaptive") & (~rejected_mask) & strain.notna()
    normal_mask = (~rejected_mask) & (~adaptive_mask) & strain.notna()

    fig, ax, preset = create_plot_figure(preset_name)
    ax.plot(frame, strain, color=PLOT_COLOR_CYCLE[0], linewidth=preset["line_width"], alpha=0.72)
    ax.scatter(frame[normal_mask], strain[normal_mask], color=PLOT_COLOR_CYCLE[0], s=preset["marker_size"], label="Accepted")

    if adaptive_mask.any():
        ax.scatter(frame[adaptive_mask], strain[adaptive_mask], color="#E69F00", s=preset["marker_size"] * 1.35, label="Adaptive")

    if rejected_mask.any():
        finite_strain = strain[np.isfinite(strain)]
        if len(finite_strain) > 0:
            ymin = float(finite_strain.min())
            ymax = float(finite_strain.max())
            span = ymax - ymin if ymax > ymin else max(abs(ymax), 1e-6)
            rejected_y = np.full(int(rejected_mask.sum()), ymin - 0.08 * span)
        else:
            rejected_y = np.zeros(int(rejected_mask.sum()))
        ax.scatter(frame[rejected_mask], rejected_y, color="#CC3311", marker="x", s=preset["marker_size"] * 1.75, label="Rejected/NaN")

    style_publication_axes(ax, preset, "Frame", "Engineering strain (-)", title=title)
    save_plot_figure(fig, path, preset_name)


def plot_all_groups_engineering_strain(df, groups, path, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax, preset = create_plot_figure(preset_name)
    for group in groups:
        gname = group["name"]
        gdf = df[df["group"] == gname]
        if gdf.empty:
            continue
        ax.plot(
            gdf["frame_global_1based"],
            gdf["engineering_strain"],
            linewidth=max(preset["line_width"] * 0.8, 0.8),
            alpha=0.56,
            label=gname,
        )

    mean_table = build_mean_strain_table(df, groups)
    if not mean_table.empty:
        frame = mean_table["Frame"].astype(float)
        for col in [c for c in mean_table.columns if c.startswith("MeanEngineeringStrain_")]:
            key = col.replace("MeanEngineeringStrain_", "", 1)
            mean = pd.to_numeric(mean_table[col], errors="coerce").astype(float)
            std_col = f"StdEngineeringStrain_{key}"
            line = ax.plot(frame, mean, linewidth=preset["line_width"] * 1.8, label=f"Mean {key}")[0]
            if std_col in mean_table.columns:
                std = pd.to_numeric(mean_table[std_col], errors="coerce").astype(float)
                finite = mean.notna() & std.notna() & np.isfinite(mean) & np.isfinite(std)
                if finite.any():
                    ax.fill_between(
                        frame[finite],
                        mean[finite] - std[finite],
                        mean[finite] + std[finite],
                        color=line.get_color(),
                        alpha=0.13,
                        linewidth=0,
                    )

    style_publication_axes(ax, preset, "Frame", "Engineering strain (-)", title="Engineering strain - all ROI groups")
    save_plot_figure(fig, path, preset_name)


def plot_poisson_ratio(df, groups, path, preset_name="publication"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = build_poisson_ratio_table(df, groups)
    frame = table["Frame"].astype(float)
    ratio = table["PoissonRatio"].astype(float)
    valid = ratio.notna() & np.isfinite(ratio)

    fig, ax, preset = create_plot_figure(preset_name)
    ax.plot(frame, ratio, color="#009E73", linewidth=preset["line_width"], alpha=0.78)
    if valid.any():
        ax.scatter(frame[valid], ratio[valid], color="#009E73", s=preset["marker_size"], label="Valid")
    invalid = ~valid
    if invalid.any():
        finite_ratio = ratio[valid]
        if len(finite_ratio) > 0:
            ymin = float(finite_ratio.min())
            ymax = float(finite_ratio.max())
            span = max(ymax - ymin, 1e-6)
            invalid_y = np.full(int(invalid.sum()), ymin - 0.08 * span)
        else:
            invalid_y = np.zeros(int(invalid.sum()))
        ax.scatter(frame[invalid], invalid_y, color="#CC3311", marker="x", s=preset["marker_size"] * 1.75, label="NaN")

    style_publication_axes(ax, preset, "Frame", "Poisson ratio (-)", title="Poisson ratio")
    save_plot_figure(fig, path, preset_name)


def plot_correlation_scores(gdf, path, group_name, hard_corr, soft_corr, preset_name="publication"):
    fig, ax, preset = create_plot_figure(preset_name)
    frame = gdf["frame_global_1based"]
    ax.plot(frame, gdf["corr_score_roi1"], marker="o", markersize=4, linewidth=preset["line_width"], label="ROI 1")
    ax.plot(frame, gdf["corr_score_roi2"], marker="s", markersize=4, linewidth=preset["line_width"], label="ROI 2")
    ax.axhline(hard_corr, color="#CC3311", linestyle="--", linewidth=preset["line_width"] * 0.9, label="strict threshold")
    ax.axhline(soft_corr, color="#E69F00", linestyle=":", linewidth=preset["line_width"] * 0.9, label="weak threshold")
    ax.set_ylim(-0.05, 1.05)
    style_publication_axes(ax, preset, "Frame", "Normalized correlation score (-)", title=f"Correlation scores - {group_name}")
    save_plot_figure(fig, path, preset_name)


def build_qc_summary(df):
    levels = {"Good": 0, "Warning": 1, "Poor": 2}
    groups = {}

    for gname, sub in df.groupby("group", sort=False):
        frames = int(len(sub))
        accepted = sub["accepted"].astype(bool) if "accepted" in sub.columns else pd.Series([True] * frames, index=sub.index)
        eng = pd.to_numeric(sub["engineering_strain"], errors="coerce")
        rejected_mask = (~accepted) | eng.isna()
        adaptive_mask = sub["accept_mode"].astype(str).eq("adaptive") if "accept_mode" in sub.columns else pd.Series([False] * frames, index=sub.index)

        rejected_frames = int(rejected_mask.sum())
        valid_frames = int(frames - rejected_frames)
        adaptive_frames = int((adaptive_mask & (~rejected_mask)).sum())

        corr1 = pd.to_numeric(sub["corr_score_roi1"], errors="coerce") if "corr_score_roi1" in sub.columns else pd.Series(dtype=float)
        corr2 = pd.to_numeric(sub["corr_score_roi2"], errors="coerce") if "corr_score_roi2" in sub.columns else pd.Series(dtype=float)
        mean_corr1 = float(corr1.mean()) if len(corr1.dropna()) else np.nan
        mean_corr2 = float(corr2.mean()) if len(corr2.dropna()) else np.nan

        valid_eng = eng.dropna()
        max_abs_strain = float(valid_eng.abs().max()) if len(valid_eng) else np.nan
        jumps = valid_eng.diff().abs().dropna()
        max_jump = float(jumps.max()) if len(jumps) else 0.0

        frame_col = _frame_column(sub)
        rejected_frame_list = [
            int(x) for x in pd.to_numeric(sub.loc[rejected_mask, frame_col], errors="coerce").dropna().tolist()
        ]

        rejected_ratio = rejected_frames / frames if frames else 0.0
        finite_means = [value for value in [mean_corr1, mean_corr2] if np.isfinite(value)]
        min_mean_corr = min(finite_means) if finite_means else np.nan
        if rejected_ratio > 0.05:
            qc_level = "Poor"
        elif rejected_frames > 0 or adaptive_frames > 0 or (np.isfinite(min_mean_corr) and min_mean_corr < 0.80):
            qc_level = "Warning"
        else:
            qc_level = "Good"

        groups[str(gname)] = {
            "frames": frames,
            "valid_frames": valid_frames,
            "rejected_frames": rejected_frames,
            "adaptive_accepted_frames": adaptive_frames,
            "mean_corr_roi1": mean_corr1,
            "mean_corr_roi2": mean_corr2,
            "max_abs_engineering_strain": max_abs_strain,
            "max_frame_strain_jump": max_jump,
            "rejected_frame_list": rejected_frame_list,
            "qc_level": qc_level,
        }

    overall_level = "Good"
    for item in groups.values():
        if levels[item["qc_level"]] > levels[overall_level]:
            overall_level = item["qc_level"]

    return {
        "overall": {
            "qc_level": overall_level,
            "groups": len(groups),
            "rejected_frames": int(sum(item["rejected_frames"] for item in groups.values())),
            "adaptive_accepted_frames": int(sum(item["adaptive_accepted_frames"] for item in groups.values())),
        },
        "groups": groups,
    }


def _format_qc_number(value, digits=3):
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return "NaN"
    return f"{float(value):.{digits}f}"


def write_qc_summary(summary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("ezDIC QC Summary\n")
        f.write("================\n")
        f.write(f"Overall QC level: {summary['overall']['qc_level']}\n")
        f.write(f"Groups: {summary['overall']['groups']}\n")
        f.write(f"Rejected frames: {summary['overall']['rejected_frames']}\n")
        f.write(f"Adaptive accepted frames: {summary['overall']['adaptive_accepted_frames']}\n")

        for gname, item in summary["groups"].items():
            rejected_list = ", ".join(str(x) for x in item["rejected_frame_list"]) or "None"
            f.write(f"\n[{gname}]\n")
            f.write(f"Frames: {item['frames']}\n")
            f.write(f"Valid frames: {item['valid_frames']}\n")
            f.write(f"Rejected frames: {item['rejected_frames']}\n")
            f.write(f"Adaptive accepted frames: {item['adaptive_accepted_frames']}\n")
            f.write(f"Mean corr ROI1: {_format_qc_number(item['mean_corr_roi1'], 3)}\n")
            f.write(f"Mean corr ROI2: {_format_qc_number(item['mean_corr_roi2'], 3)}\n")
            f.write(f"Max abs engineering strain: {_format_qc_number(item['max_abs_engineering_strain'], 4)}\n")
            f.write(f"Max frame strain jump: {_format_qc_number(item['max_frame_strain_jump'], 4)}\n")
            f.write(f"Rejected frame list: {rejected_list}\n")
            f.write(f"QC level: {item['qc_level']}\n")


# ==========================
# GUI 主类
# ==========================

UI_FONT_FAMILY = "Microsoft YaHei UI"
UI_BASE_FONT_SIZE = 10
UI_TITLE_FONT_SIZE = 12
UI_PRIMARY_FONT_SIZE = 11
UI_STEP_FONT_SIZE = 11
UI_LOG_FONT_SIZE = 10
UI_TREE_ROWHEIGHT = 32
UI_VIEWER_TICK_FONT_SIZE = 9
UI_VIEWER_LEGEND_FONT_SIZE = 9
UI_VIEWER_LABEL_FONT_SIZE = 10
UI_VIEWER_TITLE_FONT_SIZE = 11
UI_MIN_TK_SCALING = 1.20
UI_MAX_TK_SCALING = 2.50
UI_SCALE_ENV_VAR = "EZDIC_UI_SCALE"
RECENT_CONFIG_ENV_VAR = "EZDIC_RECENT_CONFIG"
RECENT_CONFIG_FILENAME = "recent_paths.json"


def enable_windows_dpi_awareness():
    """Make the Windows process DPI aware before Tk creates the first window."""
    if os.name != "nt":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            return
        except Exception:
            pass
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


class MultiROIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("分析控制台 · " + APP_TITLE)
        self.configure_initial_window()

        self.image_folder = tk.StringVar()
        self.output_folder = tk.StringVar()

        self.search_radius = tk.IntVar(value=180)
        self.hard_corr = tk.DoubleVar(value=0.55)
        self.soft_corr = tk.DoubleVar(value=0.35)
        self.strain_mode = tk.StringVar(value="auto")
        self.strain_mode_display = tk.StringVar(value=STRAIN_MODE_VALUE_TO_LABEL["auto"])
        self.roi_role = tk.StringVar(value="none")
        self.roi_role_display = tk.StringVar(value=ROI_ROLE_VALUE_TO_LABEL["none"])
        self.tracking_preset = tk.StringVar(value="标准")
        self.preset_status_var = tk.StringVar(value="当前追踪模式：标准")
        self._applying_preset = False

        self.enable_adaptive = tk.BooleanVar(value=True)
        # Fixed-reference templates are the reproducible default.  Following
        # the current frame remains an explicit experimental option.
        self.use_prev_frame_template = tk.BooleanVar(value=False)
        self.template_alpha = tk.DoubleVar(value=0.70)
        self.max_frame_strain_jump = tk.StringVar(value="0.01")

        self.enable_fb_check = tk.BooleanVar(value=True)
        self.fb_tolerance_px = tk.DoubleVar(value=12.0)

        self.overlay_every = tk.IntVar(value=5)
        self.pixel_size_mm = tk.StringVar(value="")

        self.auto_align_roi2 = tk.BooleanVar(value=True)

        self.min_texture_std = tk.DoubleVar(value=8.0)
        self.min_texture_contrast = tk.DoubleVar(value=25.0)
        self.max_saturated_frac = tk.DoubleVar(value=0.20)
        self.advanced_visible = tk.BooleanVar(value=False)

        self.export_origin_txt = tk.BooleanVar(value=True)
        self.export_origin_opju = tk.BooleanVar(value=False)
        self.export_engineering_png = tk.BooleanVar(value=True)
        self.export_publication_figures = tk.BooleanVar(value=False)
        self.export_qc_summary = tk.BooleanVar(value=True)
        self.export_full_csv = tk.BooleanVar(value=False)
        self.export_corr_plot = tk.BooleanVar(value=False)
        self.export_overlays = tk.BooleanVar(value=False)
        self.export_parameters = tk.BooleanVar(value=False)

        self.image_paths = []
        self.loaded_image_folder = None
        self.loaded_image_sequence_fingerprint = None
        self.first_raw = None
        self.first_img8 = None  # 当前预览帧的 8-bit 图像，用于显示、画 ROI、纹理检查
        self.current_fullres_img8 = None  # 用于动态缩放的原始分辨率图
        self.display_img = None
        self.display_scale = 1.0
        self.photo = None
        self._resize_after_id = None
        self._has_shown_resize_hint = False

        # 暗色模式基础（可切换色板）
        self.dark_mode = tk.BooleanVar(value=False)

        # Dual workflow: virtual extensometer (default) or full-field 2D DIC
        self.analysis_mode = tk.StringVar(value=ANALYSIS_MODE_EXTENSOMETER)
        self.dic_subset_size = tk.IntVar(value=21)
        self.dic_step = tk.IntVar(value=5)
        self.dic_solver = tk.StringVar(value=DIC_SOLVER_ICGN)
        self.dic_strain_window = tk.IntVar(value=7)
        self.dic_smooth_sigma = tk.DoubleVar(value=0.0)
        self.dic_search_radius = tk.IntVar(value=20)
        self.dic_zncc_min = tk.DoubleVar(value=0.75)
        # Multiscale recovery controls.  Defaults preserve the historical
        # single-level solver and are persisted only as optional recent-state
        # fields, so older snapshots remain valid.
        self.dic_pyramid_levels = tk.IntVar(value=1)
        self.dic_pyramid_scale = tk.DoubleVar(value=0.5)
        self.dic_field_component = tk.StringVar(value="u")
        self.dic_strain_degree = tk.IntVar(value=2)
        self.dic_robust_strain = tk.BooleanVar(value=True)
        self.dic_reject_nonconverged = tk.BooleanVar(value=True)
        self.dic_outlier_threshold = tk.DoubleVar(value=1.0)
        self.dic_mask_mode = tk.StringVar(value="矩形 ROI")
        self.dic_mask_path = tk.StringVar(value="")
        self.dic_mask_exclusions = []
        self._drawing_mask_exclusion = False
        self.dic_view_background = tk.StringVar(value="无底图")
        self.dic_color_mode = tk.StringVar(value="数据范围")
        self.dic_color_min = tk.StringVar(value="")
        self.dic_color_max = tk.StringVar(value="")
        self.dic_percent = tk.BooleanVar(value=False)
        self.dic_display_style = tk.StringVar(value="连续云图")
        self.dic_colormap = tk.StringVar(value="RdBu_r")
        self.field_roi = None
        self.roi1_reference_frame_1based = None
        self.roi2_reference_frame_1based = None
        self.field_roi_reference_frame_1based = None
        self.dic_last_field = None
        self.dic_last_image = None
        self.dic_last_frame_1based = None
        self.dic_last_filename = None
        self.dic_last_reference_frame_1based = None
        self.dic_last_reference_filename = None
        self.field_viewer_context_var = tk.StringVar(value="")
        self._viewer_kind = "extensometer"
        self._committed_analysis_mode = ANALYSIS_MODE_EXTENSOMETER
        self._canvas_shows_field_overlay = False

        # 图像缩放状态
        self.zoom_factor = 1.0          # 相对于原始图像的缩放倍率
        self.auto_fit_enabled = True    # 是否跟随窗口自动适应

        if hasattr(self, "preview_scale_var"):
            try:
                self.preview_scale_var.set("")
            except Exception:
                pass

        self.preview_frame_1based = tk.IntVar(value=1)
        self.start_frame_1based = tk.IntVar(value=1)
        self.end_frame_1based = tk.IntVar(value=1)
        self.current_preview_index = 0  # 0-based

        self.roi1 = None
        self.roi2 = None
        self.current_roi_index = 1
        self.drag_start = None
        self.temp_rect_id = None

        self.roi_groups = []
        self.next_group_idx = 1
        self.group_name_var = tk.StringVar(value="")

        self.is_processing = False
        self._completion_pending = False
        self._run_generation = 0
        self._active_run_token = None
        self._worker_context = threading.local()
        self.tooltips = []
        self.ui_queue = queue.Queue()

        # In-app results viewer (Tier 0)
        self.results_df = None
        self.results_groups = None
        self.viewer_figure = None
        self.viewer_canvas = None
        self.viewer_toolbar = None
        self.last_qc_summary = None

        self.preflight_summary_var = tk.StringVar(value="")
        self.current_roi_summary_var = tk.StringVar(value="尚未绘制 ROI。")
        self.qc_overview_var = tk.StringVar(value="分析完成后显示 QC 总览。")
        self.recent_image_dir = ""
        self.recent_output_dir = ""
        self._recent_config_path = self.default_recent_config_path()
        self.load_recent_paths()

        for var in [
            self.search_radius,
            self.hard_corr,
            self.soft_corr,
            self.max_frame_strain_jump,
            self.fb_tolerance_px,
            self.template_alpha,
            self.min_texture_std,
            self.min_texture_contrast,
            self.max_saturated_frac,
        ]:
            var.trace_add("write", self.mark_tracking_custom)

        for var in [
            self.start_frame_1based,
            self.end_frame_1based,
            self.preview_frame_1based,
            self.strain_mode,
            self.roi_role,
            *self._export_option_vars(),
            self.image_folder,
            self.output_folder,
            self.dic_subset_size,
            self.dic_step,
            self.dic_solver,
            self.dic_strain_window,
            self.dic_smooth_sigma,
            self.dic_search_radius,
            self.dic_zncc_min,
            self.dic_pyramid_levels,
            self.dic_pyramid_scale,
            self.dic_strain_degree, self.dic_robust_strain, self.dic_outlier_threshold,
            self.dic_reject_nonconverged,
            self.dic_mask_mode, self.dic_mask_path,
        ]:
            var.trace_add("write", self._on_gui_state_change)

        self.configure_ui_style()
        self.build_ui()
        self.bind_common_shortcuts()
        self.configure_final_window_limits()
        self.start_ui_queue_polling()

    # ---------- UI ----------

    def configure_initial_window(self):
        self.ui_scaling = self.configure_tk_scaling()
        self.configure_ui_metrics()

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(650, screen_w - 80)
        height = min(940, screen_h - 100)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(620, 680)

    def configure_tk_scaling(self):
        try:
            platform_scaling = float(self.root.tk.call("tk", "scaling"))
        except Exception:
            platform_scaling = 1.0

        env_value = os.environ.get(UI_SCALE_ENV_VAR)
        if env_value:
            try:
                target_scaling = float(env_value)
            except ValueError:
                target_scaling = platform_scaling
        else:
            target_scaling = max(platform_scaling, UI_MIN_TK_SCALING)

        target_scaling = min(max(target_scaling, 1.0), UI_MAX_TK_SCALING)
        try:
            self.root.tk.call("tk", "scaling", target_scaling)
            return float(self.root.tk.call("tk", "scaling"))
        except Exception:
            return platform_scaling

    def configure_ui_metrics(self):
        self.ui_base_font = (UI_FONT_FAMILY, UI_BASE_FONT_SIZE)
        self.ui_title_font = (UI_FONT_FAMILY, UI_TITLE_FONT_SIZE, "bold")
        self.ui_primary_font = (UI_FONT_FAMILY, UI_PRIMARY_FONT_SIZE, "bold")
        self.ui_step_font = (UI_FONT_FAMILY, UI_STEP_FONT_SIZE, "bold")
        self.ui_log_font = (UI_FONT_FAMILY, UI_LOG_FONT_SIZE)
        self.canvas_group_font = (UI_FONT_FAMILY, UI_BASE_FONT_SIZE, "bold")
        self.canvas_current_roi_font = (UI_FONT_FAMILY, UI_TITLE_FONT_SIZE, "bold")
        self.viewer_tick_font_size = UI_VIEWER_TICK_FONT_SIZE
        self.viewer_legend_font_size = UI_VIEWER_LEGEND_FONT_SIZE
        self.viewer_label_font_size = UI_VIEWER_LABEL_FONT_SIZE
        self.viewer_title_font_size = UI_VIEWER_TITLE_FONT_SIZE

    def configure_final_window_limits(self):
        self.root.update_idletasks()
        # Requested widths of hidden tables must not dictate the window size.
        scale = max(1.0, self.ui_scaling / (120 / 72))
        control_scale = max(1.0, self.ui_scaling / (96 / 72))
        self.root.minsize(round(620 * control_scale), round(680 * scale))
        self.visual_window.minsize(round(740 * scale), round(580 * scale))
        self.arrange_windows()

    def add_tooltip(self, widget, text, choices=None):
        provider = lambda event: self._control_help(widget, text, event)
        tip = ChoiceToolTip(widget, provider, choices) if choices is not None else ToolTip(widget, provider)
        tip.owner = self.root
        widget._tooltip_text = tip.resolve_text()
        self.tooltips.append(tip)
        widget.bind("<Destroy>", lambda _event: self.tooltips.remove(tip) if tip in self.tooltips else None, add="+")
        return widget

    def _control_help(self, widget, text, event=None):
        explanation = text(event) if callable(text) else text
        if not explanation or not isinstance(widget, ttk.Widget) or not widget.instate(["disabled"]):
            return explanation
        if self.is_processing or self._completion_pending:
            reason = "正在计算或保存结果，请等本次分析完成后再操作。"
        elif widget == getattr(self, "start_button", None):
            reason = self._start_prerequisite_help()
        elif widget == getattr(self, "prev_frame_button", None) and self.image_paths:
            reason = "当前已是第一张图片。"
        elif widget == getattr(self, "next_frame_button", None) and self.image_paths:
            reason = "当前已是最后一张图片。"
        else:
            requirements = {
                "open_recent_output_button": "先选择输出文件夹，或完成一次分析。",
                "roi1_button": "先加载图像序列。", "roi2_button": "先加载图像序列。",
                "draw_field_roi_button": "先加载图像序列。",
                "show_preview_button": "先加载图像序列。",
                "prev_frame_button": "先加载图像序列。", "next_frame_button": "先加载图像序列。",
                "set_start_button": "先加载图像序列。", "set_end_button": "先加载图像序列。",
                "btn_zoom_in": "先加载图像序列。", "btn_zoom_out": "先加载图像序列。",
                "btn_fit": "先加载图像序列。", "btn_1to1": "先加载图像序列。",
                "align_x_button": "先画好两个测量框。", "align_y_button": "先画好两个测量框。",
                "add_group_button": "先在同一参考图片上画好两个测量框。",
                "load_group_button": "先在列表中单击选中一组。",
                "update_group_button": "先选中一组，并画好两个测量框。",
                "delete_group_button": "先在列表中单击选中一组。",
                "clear_rois_button": "先在图片上画出测量框。",
                "viewer_export_btn": "先完成分析，让这里显示结果图。",
                "viewer_clear_btn": "当前没有结果预览；完成分析后可使用。",
            }
            reason = next((message for name, message in requirements.items()
                           if widget == getattr(self, name, None)), "")
        return explanation + (f"\n暂不可用：{reason}" if reason else "")

    def _start_prerequisite_help(self):
        hint = self.workflow_hint_var.get()
        label, _, message = hint.partition("：")
        reasons = {
            "图像序列": "先选择存放图片的文件夹，再点“加载序列”。",
            "分析范围": "填写两张不同图片的起止序号：从 1 开始，终点要大于起点，并且不能超过已加载的图片总数。",
            "全场 ROI": "先在第一张要分析的图片上拖出矩形分析区域；整个矩形都要落在图片内。",
            "参考帧": (
                "先显示第一张要分析的图片，再重新画分析区域。"
                if self.is_fullfield_mode() else
                "测量框必须画在第一张要分析的图片上；请载入需要修改的分组，重画后点“更新选中”保存。"
            ),
            "纹理": "测量区域的明暗纹理无法同时确定左右和上下位置。请换一块纹理更清楚的区域，或画大一些再检查。",
            "子集尺寸": "每个测量点使用的正方形图片块，边长必须是至少 9 的奇数，如 21 个像素。",
            "步长": "相邻测量点的间距必须是大于 0 的整数，单位是像素。",
            "应变窗口": "计算局部伸缩所用的邻近测量点范围必须是至少 3 的奇数，如 7 行、7 列。",
            "高斯平滑": "平均附近移动量的范围不能为负。填 0 关闭平滑，或填写正数。",
            "求解器": "请从计算方法下拉列表选择一种方法。默认的 IC-GN 会逐步修正图片块的位置和形状。",
            "POI 网格": "区域内的测量点不足。请扩大分析区域、减小图片块边长或减小测量点间距，至少要有 3 行、3 列测量点。",
            "DIC 参数": "请给“子集尺寸”（图片块边长）、“步长”（测量点间距）、“应变窗口”（邻近点数）和“高斯平滑”（平均范围）填写有效数字。",
            "ROI 组": "先在第一张要分析的图片上画两个测量框，再点“添加 ROI 组”保存这一对。",
            "L0": "两个测量框的初始间距必须大于零且能读出数值。请检查测量方向，重新画框并更新该组。",
            "输出目录": (
                "当前输出路径指向已有文件。请换成文件夹路径，或点击“选择输出”。"
                if "不是文件夹" in message else
                "先填写结果文件夹路径，或点击“选择输出”设置保存位置。"
            ),
            "导出选项": "先勾选至少一种要保存的结果，例如数值表格或曲线图片。",
        }
        return reasons.get(label, hint.replace("ROI", "测量框").replace("参考帧", "参考图片"))

    def _workspace_help(self, event=None):
        if event is not None and getattr(event, "keysym", "") != "F1":
            try:
                index = self.workspace_notebook.index(f"@{event.x},{event.y}")
            except tk.TclError:
                return ""
        else:
            index = self.workspace_notebook.index("current")
        return (
            "点击查看图片并画测量框。ROI 就是要追踪或计算的图片区域；先选参考图片，再按住左键拖出矩形。",
            "点击查看分析得到的曲线或彩色结果图。完成分析后，可在这里放大、移动或保存图表。",
            "点击查看分析前缺少什么、哪些图片未能算出结果，以及运行中的详细记录。",
        )[index]

    def _group_table_help(self, event=None):
        action = "单击选中一组，双击载入编辑；右键可载入、更新或删除。"
        if event is None or getattr(event, "keysym", "") == "F1":
            return "每行是一对已保存的测量框，会单独计算距离变化。" + action
        region = self.group_tree.identify_region(event.x, event.y)
        if region == "separator":
            return "按住列标题之间的分隔线左右拖动，可调整这一列的宽度。"
        column = self.group_tree.identify_column(event.x)
        if not column or column == "#0":
            return action
        displayed = self.group_tree.cget("displaycolumns")
        key = displayed[int(column[1:]) - 1]
        explanation = {
            "name": "这对测量框的名称，也会用于结果文件名。",
            "role": "这组测量沿拉伸方向还是垂直于拉伸方向；两种角色一起用于计算泊松比，即横向收缩与纵向伸长的比值。",
            "selected": "添加这组时选择的距离计算方向；自动判断会根据两个框的位置确定方向。",
            "actual": "实际采用的距离方向：横向是左右，纵向是上下，两点距离是直线距离。",
            "L0": "L0 是两个测量框在参考图片上的初始间距，单位是像素。应变表示间距相对这个初始值变化了多少。",
            "dx": "Δx 是两个测量框中心在左右方向的初始间距，单位是像素。",
            "dy": "Δy 是两个测量框中心在上下方向的初始间距，单位是像素。",
            "roi1": "第一个测量框的位置和尺寸：左上角的左右坐标、上下坐标、宽、高，均为像素。",
            "roi2": "第二个测量框的位置和尺寸：左上角的左右坐标、上下坐标、宽、高，均为像素。",
        }[key]
        return explanation if region == "heading" else explanation + "\n" + action

    def _on_gui_state_change(self, *args):
        try:
            self.update_workflow_action_states()
        except Exception:
            pass

    def bind_common_shortcuts(self):
        for window in (self.root, self.visual_window):
            window.bind("<Control-l>", lambda _event: self.load_first_image() or "break")
            window.bind("<Control-f>", lambda _event: self.fit_image_to_view() or "break")
            window.bind("<Control-Return>", lambda _event: self.start_processing() or "break")
            window.bind("<Escape>", lambda _event: self.clear_current_rois() or "break")
            window.bind("<Control-plus>", lambda _event: self.zoom_image(1.25) or "break")
            window.bind("<Control-equal>", lambda _event: self.zoom_image(1.25) or "break")
            window.bind("<Control-minus>", lambda _event: self.zoom_image(1 / 1.25) or "break")
            window.bind("<Control-i>", lambda _event: self.show_visual_window() or "break")

    def default_recent_config_path(self):
        configured = os.environ.get(RECENT_CONFIG_ENV_VAR)
        if configured:
            return Path(configured)
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / ".ezdic"
        return base / "ezDIC" / RECENT_CONFIG_FILENAME

    def load_recent_paths(self):
        try:
            path = Path(self._recent_config_path)
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        image_dir = str(data.get("image_dir", "") or "")
        output_dir = str(data.get("output_dir", "") or "")
        self.recent_image_dir = image_dir
        self.recent_output_dir = output_dir
        if image_dir and not self.image_folder.get().strip():
            self.image_folder.set(image_dir)
        if output_dir and not self.output_folder.get().strip():
            self.output_folder.set(output_dir)
        # Optional fields were added after the original two-path snapshot.
        # Ignore absent/invalid values so old snapshots remain loadable.
        try:
            if "pyramid_levels" in data:
                value = int(data["pyramid_levels"])
                if 1 <= value <= 8:
                    self.dic_pyramid_levels.set(value)
            if "pyramid_scale" in data:
                value = float(data["pyramid_scale"])
                if 0.0 < value < 1.0:
                    self.dic_pyramid_scale.set(value)
        except (TypeError, ValueError, tk.TclError):
            pass

    def save_recent_paths(self):
        try:
            path = Path(self._recent_config_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "image_dir": self.recent_image_dir,
                "output_dir": self.recent_output_dir,
                "pyramid_levels": int(self.dic_pyramid_levels.get()),
                "pyramid_scale": float(self.dic_pyramid_scale.get()),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def remember_recent_paths(self, image_dir=None, output_dir=None):
        if image_dir:
            self.recent_image_dir = str(image_dir)
        if output_dir:
            self.recent_output_dir = str(output_dir)
        self.save_recent_paths()
        self.update_recent_output_button_state()

    def update_recent_output_button_state(self):
        if not hasattr(self, "open_recent_output_button"):
            return
        state = tk.NORMAL if self.recent_output_dir else tk.DISABLED
        self.open_recent_output_button.config(state=state)

    def open_recent_output_folder(self):
        if not self.recent_output_dir:
            messagebox.showinfo("无最近输出", "当前还没有可打开的最近输出目录。")
            return
        try:
            open_output_folder(self.recent_output_dir)
        except Exception as exc:
            messagebox.showerror("无法打开输出目录", str(exc))
            self.log(f"无法打开最近输出目录：{exc}")

    def _export_option_vars(self):
        return [
            self.export_origin_txt,
            self.export_origin_opju,
            self.export_engineering_png,
            self.export_publication_figures,
            self.export_qc_summary,
            self.export_full_csv,
            self.export_corr_plot,
            self.export_overlays,
            self.export_parameters,
        ]

    def has_any_export_option(self):
        return any(bool(var.get()) for var in self._export_option_vars())

    def _preflight_item(self, level, label, message):
        return {"level": level, "label": label, "message": message}

    def _safe_int_var(self, var):
        try:
            return int(var.get()), None
        except (tk.TclError, TypeError, ValueError) as exc:
            return None, exc

    def build_preflight_items(self, update_display=True):
        items = []
        has_sequence = bool(self.image_paths) and self.first_img8 is not None
        frame_count = len(self.image_paths)

        if has_sequence:
            items.append(self._preflight_item("ok", "图像序列", f"已加载 {frame_count} 帧。"))
        else:
            items.append(self._preflight_item("block", "图像序列", "请先加载图像序列。"))

        start, start_err = self._safe_int_var(self.start_frame_1based)
        end, end_err = self._safe_int_var(self.end_frame_1based)
        if start_err or end_err:
            items.append(self._preflight_item("block", "分析范围", "起始帧和结束帧必须是整数。"))
        elif has_sequence and (start < 1 or end < 1 or start > frame_count or end > frame_count):
            items.append(self._preflight_item("block", "分析范围", f"范围必须位于 1 到 {frame_count} 帧之间。"))
        elif start is not None and end is not None and end <= start:
            items.append(self._preflight_item("block", "分析范围", "至少需要两帧：结束帧必须大于起始帧。"))
        elif start is not None and end is not None:
            items.append(self._preflight_item("ok", "分析范围", f"第 {start} 到 {end} 帧。"))

        is_ff = self.is_fullfield_mode()
        if is_ff:
            field_roi = self.field_roi
            if field_roi is None:
                items.append(self._preflight_item("block", "全场 ROI", "请在参考帧拖出全场分析 ROI。"))
            elif has_sequence and not rect_is_inside_image(field_roi, self.first_img8.shape):
                items.append(self._preflight_item("block", "全场 ROI", "全场 ROI 必须完整位于参考图像内。"))
            else:
                x, y, w, h = field_roi
                items.append(self._preflight_item("ok", "全场 ROI", f"ROI {w}×{h} px @ ({x},{y})。"))

            if field_roi is not None and has_sequence and rect_is_inside_image(field_roi, self.first_img8.shape):
                metrics, texture_code = self._core_texture_metrics_and_code(field_roi)
                if texture_code is not None:
                    level = "block" if texture_code == "AMBIGUOUS_TEXTURE" else "warn"
                    items.append(
                        self._preflight_item(
                            level,
                            "纹理",
                            f"{texture_code}：二维结构张量比值={metrics['structure_tensor_ratio']:.4f}。",
                        )
                    )
                else:
                    items.append(
                        self._preflight_item(
                            "ok",
                            "纹理",
                            f"二维纹理可辨识，结构张量比值={metrics['structure_tensor_ratio']:.4f}。",
                        )
                    )

            if field_roi is not None and start is not None:
                field_ref = self.field_roi_reference_frame_1based
                if field_ref is None:
                    items.append(
                        self._preflight_item(
                            "block",
                            "参考帧",
                            "全场 ROI 缺少绘制参考帧记录，请在当前起始/参考帧重画。",
                        )
                    )
                elif int(field_ref) != start:
                    items.append(
                        self._preflight_item(
                            "block",
                            "参考帧",
                            f"全场 ROI 在第 {field_ref} 帧定义，但当前起始/参考帧为第 {start} 帧，请重画全场 ROI。",
                        )
                    )
                else:
                    items.append(self._preflight_item("ok", "参考帧", "全场 ROI 与当前起始/参考帧一致。"))

            try:
                subset = int(self.dic_subset_size.get())
                step = int(self.dic_step.get())
                window = int(self.dic_strain_window.get())
                solver = str(self.dic_solver.get())
                smooth_sigma = float(self.dic_smooth_sigma.get())
                if subset < 9:
                    items.append(self._preflight_item("block", "子集尺寸", "子集尺寸必须 >= 9。"))
                elif subset % 2 == 0:
                    items.append(self._preflight_item("block", "子集尺寸", "子集尺寸必须为奇数；当前偶数值会改变实际窗口。"))
                elif step < 1:
                    items.append(self._preflight_item("block", "步长", "步长必须 >= 1。"))
                elif window < 3:
                    items.append(self._preflight_item("block", "应变窗口", "应变窗口必须 >= 3。"))
                elif window % 2 == 0:
                    items.append(self._preflight_item("block", "应变窗口", "应变窗口必须为奇数；当前偶数值会改变实际窗口。"))
                elif smooth_sigma < 0:
                    items.append(self._preflight_item("block", "高斯平滑", "高斯平滑 σ 不能为负。"))
                elif solver not in DIC_SOLVERS:
                    items.append(self._preflight_item("block", "求解器", "请选择 IC-GN 或 IC-LM。"))
                elif field_roi is not None and has_sequence:
                    X, Y = build_poi_grid(field_roi, subset, step, self.first_img8.shape)
                    if not poi_grid_is_usable(X, Y, min_rows=3, min_cols=3):
                        items.append(self._preflight_item("block", "POI 网格", "当前 ROI / 子集 / 步长至少需要 3×3 个可分析的 2D POI。"))
                    else:
                        items.append(
                            self._preflight_item(
                                "ok",
                                "DIC 参数",
                                f"subset={subset} px, step={step} px, window={window}, {solver}；POI={X.size}。",
                            )
                        )
                else:
                    items.append(self._preflight_item("ok", "DIC 参数", f"subset={subset} px, step={step} px, window={window}, {solver}。"))
            except (tk.TclError, TypeError, ValueError):
                items.append(self._preflight_item("block", "DIC 参数", "子集、步长、应变窗口和高斯平滑必须是有效数字。"))
        elif self.roi_groups:
            items.append(self._preflight_item("ok", "ROI 组", f"已添加 {len(self.roi_groups)} 组。"))
            if has_sequence:
                for group in self.roi_groups:
                    for label, rect in (("ROI1", group.get("roi1")), ("ROI2", group.get("roi2"))):
                        metrics, texture_code = self._core_texture_metrics_and_code(rect)
                        if texture_code is None:
                            continue
                        level = "block" if texture_code == "AMBIGUOUS_TEXTURE" else "warn"
                        items.append(
                            self._preflight_item(
                                level,
                                "纹理",
                                f"{group['name']} {label}: {texture_code}，结构张量比值={metrics['structure_tensor_ratio']:.4f}。",
                            )
                        )
        else:
            items.append(self._preflight_item("block", "ROI 组", "请至少添加一组 ROI1/ROI2。"))

        if (not is_ff) and self.roi_groups and start is not None:
            mismatched = [
                g["name"]
                for g in self.roi_groups
                if g.get("reference_frame_1based") is not None and int(g.get("reference_frame_1based")) != start
            ]
            if mismatched:
                items.append(
                    self._preflight_item(
                        "block",
                        "参考帧",
                        f"ROI 组 {', '.join(mismatched)} 不是在当前起始/参考帧上定义的，请载入后重画或恢复参考帧。",
                    )
                )
            else:
                items.append(self._preflight_item("ok", "参考帧", "ROI 组与当前起始/参考帧一致。"))

        if (not is_ff) and self.roi_groups:
            invalid_l0 = [g["name"] for g in self.roi_groups if g.get("L0", 0) <= 0 or not np.isfinite(float(g.get("L0", np.nan)))]
            small_l0 = [g for g in self.roi_groups if g.get("L0", 0) > 0 and g.get("L0", 0) < 50]
            if invalid_l0:
                items.append(self._preflight_item("block", "L0", f"ROI 组 {', '.join(invalid_l0)} 的 L0 无效。"))
            elif small_l0:
                detail = ", ".join(f"{g['name']}={g['L0']:.1f}px" for g in small_l0)
                items.append(self._preflight_item("warn", "L0", f"L0 偏小：{detail}；应变噪声会被放大。"))
            else:
                min_l0 = min(float(g["L0"]) for g in self.roi_groups)
                items.append(self._preflight_item("ok", "L0", f"最小 L0={min_l0:.1f} px。"))

            actual_modes = sorted({str(g.get("actual_mode", "unknown")) for g in self.roi_groups})
            if "distance" in actual_modes:
                items.append(self._preflight_item("warn", "应变方向", "存在 distance 方向；请确认倾斜标距的物理意义。"))
            else:
                items.append(self._preflight_item("ok", "应变方向", "方向已解析为 " + ", ".join(actual_modes) + "。"))

        if not is_ff:
            try:
                poisson_enabled = validate_poisson_role_groups(self.roi_groups)
            except RuntimeError as exc:
                items.append(self._preflight_item("warn", "泊松比角色", str(exc).replace("\n", " ")))
            else:
                if poisson_enabled:
                    axial, transverse = get_poisson_role_groups(self.roi_groups)
                    if axial and transverse and axial[0].get("actual_mode") == transverse[0].get("actual_mode"):
                        items.append(self._preflight_item("warn", "泊松比角色", "轴向和横向组的 actual_mode 相同，请复核方向。"))
                    else:
                        items.append(self._preflight_item("ok", "泊松比角色", "轴向/横向角色已成对配置。"))
                else:
                    items.append(self._preflight_item("ok", "泊松比角色", "未启用泊松比导出角色。"))

        output_text = self.output_folder.get().strip()
        if not output_text:
            items.append(self._preflight_item("block", "输出目录", "请设置输出文件夹。"))
        else:
            output_path = Path(output_text)
            if output_path.exists() and not output_path.is_dir():
                items.append(self._preflight_item("block", "输出目录", "输出路径已存在但不是文件夹。"))
            else:
                items.append(self._preflight_item("ok", "输出目录", str(output_path)))

        if is_ff:
            items.append(self._preflight_item("ok", "全场输出", "固定保存数值表、完整数值文件与九个分量图；可选叠加图。"))
        elif self.has_any_export_option():
            items.append(self._preflight_item("ok", "导出选项", "至少已选择一种导出内容。"))
        else:
            items.append(self._preflight_item("block", "导出选项", "请至少选择一种导出内容。"))

        if update_display and hasattr(self, "preflight_summary_var"):
            self.preflight_summary_var.set(self.format_preflight_items(items))
        return items

    def format_preflight_items(self, items):
        prefix = {"ok": "通过", "warn": "警告", "block": "阻止"}
        return "\n".join(f"[{prefix.get(item['level'], item['level'])}] {item['label']}：{item['message']}" for item in items)

    def refresh_preflight_panel(self):
        return self.build_preflight_items(update_display=True)

    def _roi_texture_status_text(self, rect):
        if self.first_img8 is None or rect is None:
            return "未检查"
        metrics = roi_texture_metrics(self.first_img8, rect)
        ok = texture_is_ok(
            metrics,
            self.min_texture_std.get(),
            self.min_texture_contrast.get(),
            self.max_saturated_frac.get(),
        )
        status = "良好" if ok else "偏弱"
        return f"{status}(std={metrics['std_gray']:.1f}, P95-P5={metrics['contrast_p95_p5']:.1f})"

    def build_current_roi_summary(self):
        if self.is_fullfield_mode():
            if self.field_roi is None:
                return "尚未绘制全场 ROI。请在当前参考帧上拖出矩形分析区域。"
            x, y, w, h = self.field_roi
            ref = self.field_roi_reference_frame_1based
            ref_text = f"；参考帧={ref}" if ref is not None else "；参考帧未记录"
            return f"全场 ROI {w}×{h} px @ ({x},{y}){ref_text}。"
        if self.roi1 is None and self.roi2 is None:
            return "尚未绘制 ROI。请在参考帧上依次绘制 ROI1 和 ROI2。"
        if self.roi1 is None:
            return "ROI1 未绘制；请先绘制 ROI1。"
        if self.roi2 is None:
            _, _, w, h = self.roi1
            return f"ROI1 {w}×{h} px；ROI2 未绘制；ROI1 纹理：{self._roi_texture_status_text(self.roi1)}。"

        _, _, w1, h1 = self.roi1
        _, _, w2, h2 = self.roi2
        actual_mode = resolve_strain_mode(self.roi1, self.roi2, self.strain_mode.get())
        dx, dy, dist = roi_separation(self.roi1, self.roi2)
        l0 = length_between(self.roi1, self.roi2, actual_mode)
        return (
            f"ROI1 {w1}×{h1} px；ROI2 {w2}×{h2} px；"
            f"dx={dx:.1f} px, dy={dy:.1f} px, distance={dist:.1f} px, L0={l0:.1f} px；"
            f"方向={actual_mode}；纹理：ROI1 {self._roi_texture_status_text(self.roi1)}，"
            f"ROI2 {self._roi_texture_status_text(self.roi2)}。"
        )

    def refresh_current_roi_summary(self):
        if hasattr(self, "current_roi_summary_var"):
            self.current_roi_summary_var.set(self.build_current_roi_summary())

    def format_qc_overview(self, summary):
        if not summary or not summary.get("groups"):
            return "分析完成后显示 QC 总览。"

        groups = summary["groups"]
        total_frames = int(sum(item["frames"] for item in groups.values()))
        rejected = int(summary["overall"].get("rejected_frames", 0))
        adaptive = int(summary["overall"].get("adaptive_accepted_frames", 0))
        rejected_ratio = rejected / total_frames if total_frames else 0.0
        levels = {"Good": 0, "Warning": 1, "Poor": 2}

        def group_key(pair):
            _name, item = pair
            ratio = item["rejected_frames"] / item["frames"] if item["frames"] else 0.0
            return levels.get(item["qc_level"], 0), ratio, item["adaptive_accepted_frames"]

        worst_name, worst = max(groups.items(), key=group_key)
        review_frames = sorted({int(frame) for frame in worst.get("rejected_frame_list", [])})
        review_text = ", ".join(str(frame) for frame in review_frames[:12]) if review_frames else "无"
        if len(review_frames) > 12:
            review_text += " ..."

        return (
            f"QC 总览：{summary['overall']['qc_level']}；ROI 组={summary['overall']['groups']}；"
            f"拒绝帧比例={rejected}/{total_frames} ({rejected_ratio:.1%})；自适应接受={adaptive}。\n"
            f"最需复核：{worst_name}（{worst['qc_level']}，拒绝 {worst['rejected_frames']}/{worst['frames']}）。\n"
            f"建议复核帧：{review_text}。"
        )

    def update_qc_overview(self, summary):
        self.last_qc_summary = summary
        if hasattr(self, "qc_overview_var"):
            self.qc_overview_var.set(self.format_qc_overview(summary))

    def format_fullfield_qc_overview(self, frames):
        valid_count = sum(frame.get("status") == "scientific_valid" for frame in frames)
        failed = [str(frame.get("frame_global_1based", "未知")) for frame in frames
                  if frame.get("status") != "scientific_valid"]
        def minimum_fraction(key):
            values = [float(frame[key]) for frame in frames
                      if isinstance(frame.get(key), (int, float)) and np.isfinite(frame[key])]
            return f"{min(values):.1%}" if values else "未提供"
        correlation = minimum_fraction("correlation_valid_fraction")
        strain = minimum_fraction("strain_valid_fraction")
        failed_text = "、".join(failed[:12]) + ("…" if len(failed) > 12 else "") if failed else "无"
        return (f"全场 DIC：{valid_count}/{len(frames)} 个变形帧通过有效性检查。\n"
                f"已计算帧的最低有效点比例：相关 {correlation}；应变 {strain}。\n"
                f"未通过的帧：{failed_text}。")

    def get_int_setting(self, var, label):
        try:
            return int(var.get())
        except (tk.TclError, TypeError, ValueError):
            raise RuntimeError(f"{label}必须是整数。")

    def get_float_setting(self, var, label):
        try:
            return float(var.get())
        except (tk.TclError, TypeError, ValueError):
            raise RuntimeError(f"{label}必须是数字。")

    def configure_ui_style(self):
        self.style = ttk.Style(self.root)
        try:
            if "clam" in self.style.theme_names():
                self.style.theme_use("clam")
        except tk.TclError:
            pass

        self._apply_color_palette()

        base_font = self.ui_base_font
        title_font = self.ui_title_font
        primary_font = self.ui_primary_font
        step_font = self.ui_step_font

        for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(font_name).configure(family=UI_FONT_FAMILY, size=UI_BASE_FONT_SIZE)
            except tk.TclError:
                pass

        self.root.configure(background=self.ui_bg)
        self.style.configure(".", font=base_font, background=self.card_bg, foreground=self.text_color,
                             bordercolor=self.border_color, lightcolor=self.card_bg, darkcolor=self.border_color)
        self.style.configure("App.TFrame", background=self.ui_bg)
        self.style.configure("Panel.TFrame", background=self.panel_bg)
        self.style.configure("Card.TFrame", background=self.card_bg)
        self.style.configure("TLabel", background=self.card_bg, font=base_font, foreground=self.text_color)
        self.style.configure("Hint.TLabel", background=self.card_bg, font=base_font, foreground=self.muted_color)
        self.style.configure("Key.TLabel", background=self.card_bg, font=step_font, foreground=self.key_color)
        self.style.configure("Warning.TLabel", background=self.card_bg, font=step_font, foreground=self.warning_color)
        self.style.configure("StepTitle.TLabel", background=self.card_bg, font=step_font, foreground=self.text_color)
        self.style.configure("TLabelframe", background=self.card_bg, bordercolor=self.border_color, relief="solid")
        self.style.configure("TLabelframe.Label", background=self.card_bg, foreground=self.text_color, font=step_font)
        self.style.configure("Brand.TLabel", background=self.ui_bg, foreground=self.text_color,
                             font=(UI_FONT_FAMILY, 18, "bold"))
        self.style.configure("AppHint.TLabel", background=self.ui_bg, foreground=self.muted_color, font=base_font)
        self.style.configure("Badge.TLabel", background=self.panel_bg, foreground=self.key_color,
                             font=base_font, padding=(10, 4))
        self.style.configure("Ready.TLabel", background=self.success_bg, foreground=self.success_color,
                             font=step_font, padding=(10, 4))
        self.style.configure("Error.TLabel", background=self.card_bg, foreground=self.warning_color,
                             font=step_font, padding=(10, 4))
        self.style.configure("TButton", font=base_font, padding=(10, 6), background=self.card_bg, width=0,
                             foreground=self.text_color, borderwidth=1, focusthickness=2,
                             focuscolor=self.primary_color)
        self.style.map("TButton", background=[("disabled", self.panel_bg), ("pressed", self.border_color),
                                              ("active", self.hover_bg)],
                       foreground=[("disabled", self.muted_color)],
                       bordercolor=[("focus", self.primary_color)])
        self.style.configure("Compact.TButton", font=base_font, padding=(8, 4))
        self.style.configure("Primary.TButton", font=primary_font, padding=(14, 8), foreground="#ffffff", background=self.primary_color)
        self.style.configure("Secondary.TButton", font=base_font, padding=(9, 5), foreground=self.primary_text, background=self.card_bg)
        self.style.configure("Danger.TButton", font=base_font, padding=(8, 5), foreground=self.warning_color, background=self.card_bg)
        self.style.map(
            "Primary.TButton",
            foreground=[("disabled", self.muted_color), ("active", "#ffffff")],
            background=[("disabled", self.panel_bg), ("pressed", self.primary_active), ("active", self.primary_active)],
        )
        self.style.map(
            "Danger.TButton",
            foreground=[("disabled", self.muted_color), ("active", self.warning_color)],
            background=[("active", self.hover_bg), ("pressed", self.panel_bg)],
        )
        for style_name in ("TEntry", "TCombobox"):
            self.style.configure(style_name, font=base_font, padding=(6, 4), fieldbackground=self.card_bg,
                                 foreground=self.text_color, insertcolor=self.text_color,
                                 selectbackground=self.primary_color, selectforeground="#ffffff")
            self.style.map(style_name, fieldbackground=[("disabled", self.panel_bg), ("readonly", self.card_bg)],
                           foreground=[("disabled", self.muted_color), ("readonly", self.text_color)],
                           bordercolor=[("focus", self.primary_color)])
        self.root.option_add("*TCombobox*Listbox.background", self.card_bg)
        self.root.option_add("*TCombobox*Listbox.foreground", self.text_color)
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.primary_color)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        for style_name in ("TCheckbutton", "TRadiobutton"):
            self.style.configure(style_name, background=self.card_bg, foreground=self.text_color, padding=(2, 3))
            self.style.map(style_name, background=[("active", self.card_bg)],
                           foreground=[("disabled", self.muted_color)],
                           indicatorbackground=[("selected", self.primary_color), ("!selected", self.card_bg)])
        self.style.configure("Mode.TRadiobutton", padding=(10, 7), font=step_font, relief="flat")
        self.style.layout("Mode.TRadiobutton", [("Radiobutton.padding", {"sticky": "nswe", "children": [
            ("Radiobutton.focus", {"sticky": "nswe", "children": [("Radiobutton.label", {"sticky": "nswe"})]})
        ]})])
        self.style.map("Mode.TRadiobutton", background=[("selected", self.primary_color), ("active", self.hover_bg)],
                       foreground=[("disabled", self.muted_color), ("selected", "#ffffff")])
        self.style.configure("Treeview", font=base_font, rowheight=UI_TREE_ROWHEIGHT, background=self.card_bg,
                             fieldbackground=self.card_bg, foreground=self.text_color)
        self.style.map("Treeview", background=[("selected", self.primary_color)], foreground=[("selected", "#ffffff")])
        self.style.configure("Treeview.Heading", font=step_font, background=self.panel_bg, foreground=self.text_color,
                             padding=(6, 5), relief="flat")
        self.style.map("Treeview.Heading", background=[("active", self.hover_bg)])
        self.style.configure("TNotebook", background=self.ui_bg, borderwidth=0, tabmargins=(0, 0, 0, 6))
        self.style.configure("TNotebook.Tab", background=self.ui_bg, foreground=self.muted_color,
                             padding=(18, 8), font=step_font)
        self.style.map("TNotebook.Tab", background=[("selected", self.card_bg), ("active", self.hover_bg)],
                       foreground=[("selected", self.primary_text)])
        self.style.configure("Horizontal.TProgressbar", background=self.primary_color, troughcolor=self.panel_bg,
                             borderwidth=0, thickness=5)
        self.style.configure("TScrollbar", background=self.panel_bg, troughcolor=self.ui_bg,
                             arrowcolor=self.muted_color, borderwidth=0, arrowsize=12)
        self.style.map("TScrollbar", background=[("active", self.border_color)])

    def _apply_color_palette(self):
        """集中管理浅色/暗色配色，便于后续完整暗色模式切换。"""
        if self.dark_mode.get():
            # 暗色主题（实验室长时间使用更护眼）
            self.ui_bg = "#101827"
            self.panel_bg = "#243247"
            self.card_bg = "#192538"
            self.border_color = "#3b4c64"
            self.hover_bg = "#2c3d56"
            self.primary_color = "#2563eb"
            self.primary_active = "#1d4ed8"
            self.warning_color = "#f87171"
            self.key_color = "#e0f2fe"
            self.text_color = "#e2e8f0"
            self.muted_color = "#b0bfd2"
            self.success_color = "#86efac"
            self.success_bg = "#163b35"
        else:
            # 浅色主题（默认）
            self.ui_bg = "#f1f4f8"
            self.panel_bg = "#edf2f8"
            self.card_bg = "#ffffff"
            self.border_color = "#d5deea"
            self.hover_bg = "#e8eff9"
            self.primary_color = "#1d5dcc"
            self.primary_active = "#164ba8"
            self.warning_color = "#b91c1c"
            self.key_color = "#0f3f6e"
            self.text_color = "#0f172a"
            self.muted_color = "#475569"
            self.success_color = "#166534"
            self.success_bg = "#e5f4eb"
        self.primary_text = "#a8ccff" if self.dark_mode.get() else self.primary_color

    def toggle_dark_mode(self):
        """切换暗色/浅色模式，并尝试刷新结果预览图的配色。"""
        is_dark = self.dark_mode.get()
        self.dark_mode.set(not is_dark)
        self._apply_color_palette()

        self.configure_ui_style()
        self.root.configure(background=self.ui_bg)
        self.visual_window.configure(background=self.ui_bg)

        for attr in ("log_text", "preflight_summary_label"):
            widget = getattr(self, attr, None)
            if isinstance(widget, tk.Text):
                widget.configure(bg=self.panel_bg, fg=self.text_color, insertbackground=self.text_color)
        if hasattr(self, "preflight_summary_label"):
            self._update_preflight_text()
        if hasattr(self, "controls_canvas"):
            self.controls_canvas.configure(bg=self.card_bg, highlightbackground=self.border_color)
        self._draw_image_empty_state()

        # 刷新结果预览器（matplotlib）
        if (hasattr(self, "results_df") and self.results_df is not None) or getattr(self, "dic_last_field", None) is not None:
            try:
                self._refresh_viewer_for_dark_mode()
            except Exception:
                pass

        self.root.update_idletasks()
        if hasattr(self, "dark_mode_btn"):
            self.dark_mode_btn.configure(text="浅色模式" if self.dark_mode.get() else "暗色模式")
        self.log("已切换显示模式。" if self.dark_mode.get() else "已恢复默认显示模式。")

    def _refresh_viewer_for_dark_mode(self):
        """当暗色模式切换时，重新绘制当前结果预览图以匹配新主题。"""
        if getattr(self, "_viewer_kind", "extensometer") == "fullfield" and getattr(self, "dic_last_field", None) is not None:
            self._rebuild_field_viewer_plot()
            return
        if not hasattr(self, "results_df") or self.results_df is None:
            return
        if hasattr(self, "viewer_figure") and self.viewer_figure is not None:
            try:
                self._rebuild_viewer_plot()
            except Exception:
                pass

    def build_ui(self):
        self.root.configure(background=self.ui_bg)
        self.main_frame = ttk.Frame(self.root, style="App.TFrame", padding=(16, 12, 16, 12))
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)

        self._build_app_header(self.main_frame)
        self._build_project_section(self.main_frame)
        self._build_workspace(self.main_frame)
        self.update_workflow_action_states()

    def _build_app_header(self, parent):
        header = ttk.Frame(parent, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="StrainTrace", style="Brand.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="分析控制台 · 序列、参数与质量", style="AppHint.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.open_visual_button = ttk.Button(
            header, text="图像与结果 ↗", command=self.show_visual_window, style="Secondary.TButton")
        self.open_visual_button.grid(row=0, column=1, sticky="e", padx=(8, 6))
        self.add_tooltip(self.open_visual_button, "点击打开独立的图像与结果窗口（Ctrl + I），查看图片、绘制 ROI 或检查结果。关闭该窗口不会清空数据或停止分析。")
        self.open_recent_output_button = ttk.Button(
            header, text="打开输出", command=self.open_recent_output_folder, style="Compact.TButton")
        self.open_recent_output_button.grid(row=0, column=3, padx=(0, 6))
        self.dark_mode_btn = ttk.Button(header, text="暗色模式", command=self.toggle_dark_mode, style="Compact.TButton")
        self.dark_mode_btn.grid(row=0, column=2, padx=(0, 6))
        self.usage_notice_button = ttk.Button(header, text="关于 / 引用", command=self.show_usage_notice, style="Compact.TButton")
        self.usage_notice_button.grid(row=0, column=4)
        self.add_tooltip(self.usage_notice_button, "点击查看版本、作者和使用说明，以及论文中引用本软件时可用的信息。")
        self.add_tooltip(self.dark_mode_btn, "点击切换浅色或深色界面，结果图也会换色。")
        self.add_tooltip(self.open_recent_output_button, "点击在文件管理器中打开最近选择的结果文件夹，查看已保存的表格和图片。")
        self.update_recent_output_button_state()

    def _build_project_section(self, parent):
        self.project_frame = ttk.LabelFrame(parent, text="01  图像序列", padding=(12, 8))
        self.project_frame.grid(row=1, column=0, sticky="ew")
        self.project_frame.columnconfigure(1, weight=1)
        self.project_frame.columnconfigure(3, weight=0)

        image_folder_tip = (
            "填写连续拍摄同一样品、同一位置的图片文件夹路径。加载时会按文件名中的数字排序，如 1、2、10。"
        )
        self.image_folder_label = ttk.Label(self.project_frame, text="图像文件夹：", style="Key.TLabel")
        self.image_folder_label.grid(row=0, column=0, sticky="w", pady=1)
        self.add_tooltip(self.image_folder_label, image_folder_tip)
        self.image_folder_entry = ttk.Entry(self.project_frame, textvariable=self.image_folder, width=42)
        self.image_folder_entry.grid(
            row=0, column=1, sticky="ew", padx=(6, 8), pady=1
        )
        self.add_tooltip(self.image_folder_entry, image_folder_tip)
        self.select_image_button = ttk.Button(
            self.project_frame,
            text="选择图像",
            command=self.select_image_folder,
            style="Secondary.TButton",
        )
        self.select_image_button.grid(row=0, column=2, sticky="ew", pady=1)
        self.add_tooltip(self.select_image_button, "点击选择存放连续拍摄图片的文件夹，再点“加载序列”显示图片。")

        output_folder_tip = (
            "填写结果文件夹路径，分析后的表格和图片会保存在这里。开始分析前必须设置；也可点击“选择输出”设置保存位置。"
        )
        self.output_folder_label = ttk.Label(self.project_frame, text="输出文件夹：")
        self.output_folder_label.grid(row=1, column=0, sticky="w", pady=1)
        self.add_tooltip(self.output_folder_label, output_folder_tip)
        self.output_folder_entry = ttk.Entry(self.project_frame, textvariable=self.output_folder, width=42)
        self.output_folder_entry.grid(
            row=1, column=1, sticky="ew", padx=(6, 8), pady=1
        )
        self.add_tooltip(self.output_folder_entry, output_folder_tip)
        self.select_output_button = ttk.Button(
            self.project_frame,
            text="选择输出",
            command=self.select_output_folder,
            style="Secondary.TButton",
        )
        self.select_output_button.grid(row=1, column=2, sticky="ew", pady=1)
        self.add_tooltip(self.select_output_button, "点击选择分析结果的保存位置。选好后，顶部“打开输出”可打开这个文件夹。")

        self.load_images_button = ttk.Button(
            self.project_frame,
            text="加载序列",
            command=self.load_first_image,
            style="Secondary.TButton",
        )
        self.load_images_button.grid(row=0, column=3, rowspan=2, padx=(8, 0), pady=1, sticky="nsew")
        self.add_tooltip(
            self.load_images_button,
            "点击加载所选文件夹的图片并显示预览（Ctrl + L）。首次加载会选中全部图片；更换图片序列会清空原有测量框和分组。",
        )

    def _build_workspace(self, parent):
        workspace = ttk.Frame(parent, style="App.TFrame")
        workspace.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)
        self.control_notebook = ttk.Notebook(workspace)
        self.control_notebook.grid(row=0, column=0, sticky="nsew")
        self.settings_page = ttk.Frame(self.control_notebook, style="Card.TFrame")
        self.quality_page = ttk.Frame(self.control_notebook, style="Card.TFrame", padding=12)
        for page in (self.settings_page, self.quality_page):
            page.columnconfigure(0, weight=1)
            page.rowconfigure(0, weight=1)
        self.control_notebook.add(self.settings_page, text="分析设置")
        self.control_notebook.add(self.quality_page, text="质量与日志")
        self.add_tooltip(self.control_notebook, self._control_workspace_help)

        self.visual_window = tk.Toplevel(self.root)
        self.visual_window.title("StrainTrace · 图像与结果")
        self.visual_window.configure(background=self.ui_bg)
        self.visual_window.protocol("WM_DELETE_WINDOW", self.hide_visual_window)
        visual_frame = ttk.Frame(self.visual_window, style="App.TFrame", padding=(16, 12))
        visual_frame.pack(fill=tk.BOTH, expand=True)
        visual_frame.columnconfigure(0, weight=1)
        visual_frame.rowconfigure(1, weight=1)
        header = ttk.Frame(visual_frame, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="图像与结果", style="Brand.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="ROI · 遮罩 · 位移 · 应变", style="AppHint.TLabel").grid(
            row=0, column=1, sticky="w", padx=(16, 8))
        self.arrange_windows_button = ttk.Button(
            header, text="排列窗口", command=self.arrange_windows, style="Compact.TButton")
        self.arrange_windows_button.grid(row=0, column=2, padx=(0, 6))
        self.add_tooltip(self.arrange_windows_button, "点击把控制台和图像窗口排列到当前屏幕内；屏幕足够宽时并排显示。之后仍可独立移动、缩放或放到另一块屏幕。")
        self.return_control_button = ttk.Button(
            header, text="分析控制台 ↗", command=self.show_control_window, style="Secondary.TButton")
        self.return_control_button.grid(row=0, column=3)
        self.add_tooltip(self.return_control_button, "点击回到分析控制台，调整图片范围、计算参数或查看质量与日志。图像窗口的位置和内容保留。")
        self.workspace_notebook = ttk.Notebook(visual_frame)
        self.workspace_notebook.grid(row=1, column=0, sticky="nsew")
        self.image_page = ttk.Frame(self.workspace_notebook, style="Card.TFrame")
        self.results_page = ttk.Frame(self.workspace_notebook, style="Card.TFrame")
        for page in (self.image_page, self.results_page):
            page.columnconfigure(0, weight=1)
            page.rowconfigure(0, weight=1)
        self.workspace_notebook.add(self.image_page, text="图像与 ROI")
        self.workspace_notebook.add(self.results_page, text="分析结果")
        self.workspace_notebook.bind("<<NotebookTabChanged>>", self._on_workspace_tab_changed)
        self.add_tooltip(self.workspace_notebook, self._workspace_help)
        self._build_image_section(self.image_page)
        self._build_scrollable_controls(self.settings_page)
        ttk.Label(visual_frame, textvariable=self.status_var, style="AppHint.TLabel",
                  wraplength=700, justify=tk.LEFT).grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _control_workspace_help(self, event=None):
        try:
            index = (self.control_notebook.index(f"@{event.x},{event.y}")
                     if event is not None and getattr(event, "keysym", "") != "F1"
                     else self.control_notebook.index("current"))
        except tk.TclError:
            return ""
        return ("点击设置分析模式、参考图片、ROI 和计算参数；绘图操作在独立的图像窗口中进行。",
                "点击查看分析前缺少什么、哪些图片未能算出结果，以及运行中的详细记录。")[index]

    def show_control_window(self):
        self.root.deiconify()
        self.root.lift()

    def show_visual_window(self):
        self.visual_window.deiconify()
        self.visual_window.lift()
        self.root.after_idle(self._rescale_display_to_current_size)

    def hide_visual_window(self):
        self.visual_window.withdraw()

    def arrange_windows(self):
        """Place both independently resizable windows inside the primary screen."""
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        control_min_w, control_min_h = self.root.minsize()
        visual_min_w, visual_min_h = self.visual_window.minsize()
        control_w = max(control_min_w, min(650, screen_w - 80))
        height = max(control_min_h, visual_min_h, min(900, screen_h - 100))
        if screen_w >= control_w + visual_min_w + 60:
            visual_w = min(1360, screen_w - control_w - 60)
            visual_x = control_w + 40
        else:
            visual_w = max(visual_min_w, min(1120, screen_w - 80))
            visual_x = max(20, screen_w - visual_w - 20)
        self.root.geometry(f"{control_w}x{height}+20+30")
        self.visual_window.geometry(f"{visual_w}x{height}+{visual_x}+30")
        self.root.deiconify()
        self.visual_window.deiconify()

    def _build_scrollable_controls(self, parent):
        sidebar_width = round(370 * max(1.0, self.ui_scaling / (96 / 72)))
        self.controls_frame = ttk.Frame(parent, style="Card.TFrame", width=sidebar_width)
        self.controls_frame.grid(row=0, column=0, sticky="nsew")
        self.controls_frame.grid_propagate(False)
        self.controls_frame.columnconfigure(0, weight=1)
        self.controls_frame.rowconfigure(1, weight=1)

        self._build_workflow_guide(self.controls_frame)

        self.controls_canvas = tk.Canvas(
            self.controls_frame,
            width=sidebar_width,
            height=560,
            bg=self.card_bg,
            highlightthickness=0,
            highlightbackground=self.border_color,
            borderwidth=0,
        )
        controls_scrollbar = ttk.Scrollbar(self.controls_frame, orient=tk.VERTICAL, command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        self.controls_canvas.grid(row=1, column=0, sticky="nsew")
        controls_scrollbar.grid(row=1, column=1, sticky="ns")
        self.add_tooltip(controls_scrollbar, "上下拖动查看被隐藏的设置。在控制台的设置区滚动鼠标滚轮，也可上下移动。")
        self.add_tooltip(self.controls_canvas, "在这里滚动鼠标滚轮，查看下方的测量和导出设置。")

        self.controls_panel = ttk.Frame(self.controls_canvas, style="Card.TFrame", padding=(10, 6, 10, 12))
        self.controls_panel.columnconfigure(0, weight=1)
        self.controls_window = self.controls_canvas.create_window((0, 0), window=self.controls_panel, anchor="nw")
        self.controls_panel.bind(
            "<Configure>",
            lambda _event: self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all")),
        )
        self.controls_canvas.bind(
            "<Configure>",
            self._on_controls_configure,
        )
        self.workflow_canvas = self.controls_canvas
        self.workflow_panel = self.controls_panel

        self._build_measure_section(self.controls_panel)
        self._build_fullfield_section(self.controls_panel)
        self._build_roi_section(self.controls_panel)
        self._build_analysis_section(self.controls_panel)
        self._build_run_section(self.main_frame, row=3)
        self._build_quality_section(self.quality_page)
        self._build_results_section(self.results_page)
        self._bind_workflow_scroll_handler()
        self.set_analysis_mode()

    def _on_controls_configure(self, event):
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)
        wrap = max(180, event.width - 52)
        for label_name in ("preset_status_label", "current_roi_summary_label", "dic_field_summary_label",
                           "export_hint_label", "fullfield_export_info_label", "workflow_hint_label", "status_label"):
            label = getattr(self, label_name, None)
            if label is not None:
                label.configure(wraplength=wrap)

    def _on_workspace_tab_changed(self, _event=None):
        if self.workspace_notebook.select() == str(self.image_page):
            self.root.after_idle(self._rescale_display_to_current_size)

    def _show_image_workspace(self):
        if hasattr(self, "workspace_notebook"):
            self.show_visual_window()
            self.workspace_notebook.select(self.image_page)

    def _bind_workflow_scroll_handler(self):
        """Enable mouse-wheel scrolling inside the right workflow panel."""
        seen = set()

        def bind_tree(widget):
            widget_key = str(widget)
            if widget_key in seen:
                return
            seen.add(widget_key)
            widget.bind("<MouseWheel>", self._on_workflow_mouse_wheel, add="+")
            widget.bind("<Button-4>", self._on_workflow_mouse_wheel, add="+")
            widget.bind("<Button-5>", self._on_workflow_mouse_wheel, add="+")

            for child in widget.winfo_children():
                bind_tree(child)

        bind_tree(self.controls_canvas)
        bind_tree(self.controls_panel)
        if hasattr(self, "workflow_guide_frame"):
            bind_tree(self.workflow_guide_frame)

    def _on_workflow_mouse_wheel(self, event):
        """Scroll the workflow panel; image-canvas wheel events keep zoom behavior."""
        delta = getattr(event, "delta", 0)
        button_num = getattr(event, "num", 0)
        if button_num == 4:
            amount = -1
        elif button_num == 5:
            amount = 1
        elif delta > 0:
            amount = -max(1, abs(int(delta / 120)))
        elif delta < 0:
            amount = max(1, abs(int(delta / 120)))
        else:
            return "break"

        if event.widget == getattr(self, "group_tree", None) and self.group_tree.yview() != (0.0, 1.0):
            self.group_tree.yview_scroll(amount, "units")
        else:
            self.controls_canvas.yview_scroll(amount, "units")
        return "break"

    def _build_workflow_guide(self, parent):
        self.workflow_guide_frame = ttk.Frame(parent, style="Card.TFrame", padding=(12, 10))
        self.workflow_guide_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.workflow_guide_frame.columnconfigure(0, weight=1)

        self.workflow_step_texts = [
            "1. 选择图像文件夹和输出文件夹，加载序列",
            "2. 设置参考帧、分析范围和测量方向",
            "3. 画 ROI1/ROI2，添加 ROI 组",
            "4. 确认导出内容，点击开始分析",
        ]
        self.workflow_labels = []
        self.workflow_steps_label = ttk.Label(
            self.workflow_guide_frame,
            text=self._visible_workflow_steps_text(),
            style="Hint.TLabel",
            font=(UI_FONT_FAMILY, 9),
            justify=tk.LEFT,
            wraplength=430,
        )
        self.workflow_steps_label.grid(row=0, column=0, sticky="ew")
        self.workflow_labels.append(self.workflow_steps_label)
        self.add_tooltip(
            self.workflow_steps_label,
            lambda _event: ("先加载图片，选一张作为比较基准，再画要分析的矩形区域，最后开始计算整片区域的移动和伸缩。"
                            if self.is_fullfield_mode() else
                            "先加载图片，选一张作为比较基准，再画两个测量框并添加为一组，最后计算它们之间的距离变化。"),
        )

        self.workflow_hint_var = tk.StringVar(value="请选择并加载图像序列。")
        self.workflow_guide_frame.bind(
            "<Configure>",
            lambda event: self._sync_workflow_guide_wraplength(event.width),
            add="+",
        )

    def _sync_workflow_guide_wraplength(self, width):
        wrap = max(int(width) - 48, 220)
        self.workflow_steps_label.configure(wraplength=wrap)

    def _visible_workflow_steps_text(self):
        """Keep the current workflow visible without repeating the instructions."""
        if str(self.analysis_mode.get()) == ANALYSIS_MODE_FULLFIELD:
            return "选参考图片 → 在图像窗口画分析区域 → 开始分析 · 悬停或按 F1 查看说明"
        return "选参考图片 → 在图像窗口画两个测量框 → 开始分析 · 悬停或按 F1 查看说明"

    def _build_measure_section(self, parent):
        mode_bar = ttk.Frame(parent, style="Card.TFrame")
        mode_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        mode_bar.columnconfigure((0, 1), weight=1, uniform="mode")
        self.mode_extensometer_radio = ttk.Radiobutton(
            mode_bar,
            text="虚拟引伸计",
            variable=self.analysis_mode,
            value=ANALYSIS_MODE_EXTENSOMETER,
            command=self.set_analysis_mode,
            style="Mode.TRadiobutton",
        )
        self.mode_extensometer_radio.grid(row=0, column=0, sticky="ew")
        self.add_tooltip(
            self.mode_extensometer_radio,
            "选择后，在参考图片上画两个测量框（ROI），再添加为一组。程序追踪这两个框，画出它们之间的距离随时间伸长或缩短的曲线。",
        )
        self.mode_fullfield_radio = ttk.Radiobutton(
            mode_bar,
            text="全场 2D DIC",
            variable=self.analysis_mode,
            value=ANALYSIS_MODE_FULLFIELD,
            command=self.set_analysis_mode,
            style="Mode.TRadiobutton",
        )
        self.mode_fullfield_radio.grid(row=0, column=1, sticky="ew")
        self.add_tooltip(
            self.mode_fullfield_radio,
            "选择后，在参考图片上画一片矩形区域（ROI）。DIC 是比较图片纹理来测量变形的方法；程序会画出这片区域各处的移动和伸缩。",
        )

        self.measure_frame = ttk.LabelFrame(parent, text="02  测量设置", padding=(10, 8))
        self.measure_frame.grid(row=1, column=0, sticky="ew")
        self.measure_frame.columnconfigure(0, weight=1)

        frame_range = ttk.Frame(self.measure_frame, style="Card.TFrame")
        frame_range.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        frame_range.columnconfigure(0, weight=1)

        preview_row = ttk.Frame(frame_range, style="Card.TFrame")
        preview_row.grid(row=0, column=0, sticky="ew", pady=(0, 3))
        preview_row.columnconfigure(3, weight=1)

        preview_tip = (
            "输入要查看的图片序号，再点“显示”。帧就是序列中的一张图片，编号从 1 开始；这里只切换预览图片。"
        )
        self.preview_frame_label = ttk.Label(preview_row, text="预览帧：")
        self.preview_frame_label.grid(row=0, column=0, padx=(0, 4), sticky="w")
        self.add_tooltip(self.preview_frame_label, preview_tip)
        self.preview_frame_entry = ttk.Entry(preview_row, textvariable=self.preview_frame_1based, width=6)
        self.preview_frame_entry.grid(row=0, column=1, padx=(0, 5), sticky="w")
        self.add_tooltip(self.preview_frame_entry, preview_tip)
        self.show_preview_button = ttk.Button(
            preview_row,
            text="显示",
            command=self.go_to_preview_frame,
            style="Compact.TButton",
        )
        self.show_preview_button.grid(row=0, column=2, padx=(0, 5), sticky="w")
        self.add_tooltip(self.show_preview_button, "点击显示左侧序号对应的图片；编号从 1 开始。只切换预览，不会开始分析。")

        self.prev_frame_button = ttk.Button(
            preview_row,
            text="上一帧",
            command=lambda: self.step_preview_frame(-1),
            style="Compact.TButton",
        )
        self.prev_frame_button.grid(row=0, column=3, padx=(0, 5), sticky="e")
        self.add_tooltip(self.prev_frame_button, "点击查看上一张图片，检查画框的位置是否合适；分析范围保持原来的设置。")

        self.next_frame_button = ttk.Button(
            preview_row,
            text="下一帧",
            command=lambda: self.step_preview_frame(1),
            style="Compact.TButton",
        )
        self.next_frame_button.grid(row=0, column=4, sticky="e")
        self.add_tooltip(self.next_frame_button, "点击查看下一张图片，检查测量部位是否仍然清晰可见；分析范围保持原来的设置。")

        range_row = ttk.Frame(frame_range, style="Card.TFrame")
        range_row.grid(row=1, column=0, sticky="ew")
        range_row.columnconfigure(6, weight=1)

        analysis_range_tip = (
            "填写要分析的第一张和最后一张图片序号，包含两端，编号从 1 开始。起始图片也是比较变形的基准；测量框要在它上面绘制。"
        )
        self.analysis_range_label = ttk.Label(range_row, text="分析范围：", style="Key.TLabel")
        self.analysis_range_label.grid(row=0, column=0, padx=(0, 4), sticky="w")
        self.add_tooltip(self.analysis_range_label, analysis_range_tip)
        self.start_frame_entry = ttk.Entry(range_row, textvariable=self.start_frame_1based, width=6)
        self.start_frame_entry.grid(row=0, column=1, padx=(0, 4), sticky="w")
        self.add_tooltip(self.start_frame_entry, "填写第一张要分析的图片序号（从 1 开始）。它同时作为比较基准；改动后，请在新的参考图片上重新画测量框。")
        ttk.Label(range_row, text="到").grid(row=0, column=2, padx=(0, 4), sticky="w")
        self.end_frame_entry = ttk.Entry(range_row, textvariable=self.end_frame_1based, width=6)
        self.end_frame_entry.grid(row=0, column=3, padx=(0, 8), sticky="w")
        self.add_tooltip(self.end_frame_entry, "填写最后一张要分析的图片序号，计算会包含这张图片。序号不能小于起始值，也不能超过图片总数。")

        self.set_start_button = ttk.Button(
            range_row,
            text="设为起始/参考",
            command=self.set_start_to_current,
            style="Compact.TButton",
        )
        self.set_start_button.grid(row=1, column=1, columnspan=2, padx=(0, 5), pady=(2, 0), sticky="w")
        self.add_tooltip(
            self.set_start_button,
            "点击把正在预览的图片设为分析起点和比较基准。已有测量框时会询问是否清空；确认后，请在这张参考图片上重新画框。",
        )

        self.set_end_button = ttk.Button(
            range_row,
            text="设为结束帧",
            command=self.set_end_to_current,
            style="Compact.TButton",
        )
        self.set_end_button.grid(row=1, column=3, columnspan=2, pady=(2, 0), sticky="w")
        self.add_tooltip(
            self.set_end_button,
            "点击把正在预览的图片设为分析终点，计算会包含这张图片。可用来跳过后面模糊、断裂或样品离开画面的图片。",
        )

        measure_core = ttk.Frame(self.measure_frame, style="Card.TFrame", padding=(0, 10, 0, 0))
        self.measure_core_frame = measure_core
        measure_core.grid(row=1, column=0, sticky="ew")
        measure_core.columnconfigure(1, weight=1)

        strain_mode_tip = (
            "选择两个测量框之间要比较的距离方向，添加或更新分组后生效。应变就是这段距离相对初始值伸长或缩短的比例。"
        )
        self.strain_mode_label = ttk.Label(measure_core, text="应变方向", style="Key.TLabel")
        self.strain_mode_label.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=1)
        self.add_tooltip(self.strain_mode_label, strain_mode_tip)
        self.strain_mode_box = ttk.Combobox(
            measure_core,
            textvariable=self.strain_mode_display,
            values=list(STRAIN_MODE_LABEL_TO_VALUE.keys()),
            width=12,
            state="readonly",
        )
        self.strain_mode_box.grid(row=0, column=1, columnspan=3, sticky="ew", pady=1)
        self.strain_mode_box.bind("<<ComboboxSelected>>", self.sync_strain_mode_from_display)
        self.add_tooltip(self.strain_mode_box, strain_mode_tip, choices={
            "自动判断": "按两个框的位置自动选左右、上下或直线距离，首次使用可选此项。",
            "横向应变": "只比较两个框在图片左右方向的距离变化，适用于左右排列的测量部位。",
            "纵向应变": "只比较两个框在图片上下方向的距离变化，适用于上下排列的测量部位。",
            "两点距离应变": "比较两个框中心之间的直线距离，适用于倾斜排列的测量部位。",
        })

        tracking_preset_tip = (
            "选择后自动填写一组追踪参数，用于在后续图片中寻找两个测量框。首次使用选“标准”，再根据检查结果调整。"
        )
        self.tracking_preset_label = ttk.Label(measure_core, text="追踪模式", style="Key.TLabel")
        self.tracking_preset_label.grid(row=1, column=0, sticky="w", padx=(0, 6), pady=1)
        self.add_tooltip(self.tracking_preset_label, tracking_preset_tip)
        self.tracking_preset_box = ttk.Combobox(
            measure_core,
            textvariable=self.tracking_preset,
            values=list(TRACKING_PRESETS.keys()) + ["自定义"],
            width=12,
            state="readonly",
        )
        self.tracking_preset_box.grid(row=1, column=1, columnspan=3, sticky="ew", pady=1)
        self.tracking_preset_box.bind("<<ComboboxSelected>>", self.apply_tracking_preset)
        self.add_tooltip(self.tracking_preset_box, tracking_preset_tip, choices={
            "标准": "使用常规搜索范围和检查要求，适合纹理清晰、变化较缓慢的图片。",
            "低质量图像": "放宽纹理相似程度的要求，让较难识别的图片有机会算出结果；需核对追踪位置。",
            "快速变形": "扩大寻找范围，并允许相邻图片的伸缩变化更大；计算可能更慢。",
            "自定义": "保留当前参数。展开“高级设置”可逐项修改，修改后这里也会自动显示自定义。",
        })
        self.preset_status_label = ttk.Label(measure_core, textvariable=self.preset_status_var, style="Hint.TLabel", wraplength=430)
        self.preset_status_label.grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=(0, 2)
        )
        self.add_tooltip(self.preset_status_label, tracking_preset_tip)

        pixel_size_tip = (
            "填写一个像素对应多少毫米。例如 100 个像素等于 1 毫米，就填 0.01。用于记录实际初始间距；不清楚时可留空，仍能计算伸缩比例。"
        )
        self.pixel_size_label = ttk.Label(measure_core, text="标定 mm/px，可空")
        self.pixel_size_label.grid(row=3, column=0, sticky="w", padx=(0, 6), pady=1)
        self.add_tooltip(self.pixel_size_label, pixel_size_tip)
        self.pixel_size_entry = ttk.Entry(measure_core, textvariable=self.pixel_size_mm, width=9)
        self.pixel_size_entry.grid(row=3, column=1, sticky="w", pady=1)
        self.add_tooltip(self.pixel_size_entry, pixel_size_tip)
        auto_align_tip = (
            "勾选后，画完第二个测量框时会按测量方向把两个框对齐，便于测左右或上下间距。要保留倾斜位置时取消勾选。"
        )
        self.auto_align_roi2_check = ttk.Checkbutton(measure_core, text="绘制 ROI2 后自动对齐", variable=self.auto_align_roi2)
        self.auto_align_roi2_check.grid(
            row=4, column=0, columnspan=4, sticky="w", pady=1
        )
        self.add_tooltip(self.auto_align_roi2_check, auto_align_tip)

        self.advanced_toggle_btn = ttk.Button(
            self.measure_frame,
            text="显示高级设置",
            command=self.toggle_advanced_settings,
            style="Compact.TButton",
        )
        self.advanced_toggle_btn.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.add_tooltip(
            self.advanced_toggle_btn,
            "点击展开或收起搜索范围、图像相似程度和纹理检查等设置。首次使用可保留默认值；每项都有具体说明。",
        )

        self.advanced_frame = ttk.LabelFrame(self.measure_frame, text="高级设置", padding=(10, 8))
        self.advanced_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self._build_advanced_controls()
        self.advanced_frame.grid_remove()
        self.pixel_size_label.grid_remove()
        self.pixel_size_entry.grid_remove()
        self.preset_status_label.grid_remove()

    def _build_fullfield_section(self, parent):
        self.fullfield_frame = ttk.LabelFrame(parent, text="03  全场 ROI 与 DIC 参数", padding=(10, 8))
        self.fullfield_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.fullfield_frame.columnconfigure(1, weight=1)

        subset_tip = (
            "每个测量点用多大的正方形图片块来寻找移动位置，边长以像素计。填至少 9 的奇数，如 21；调大通常更容易识别，但细小变化会被平均。"
        )
        ttk.Label(self.fullfield_frame, text="子集 px", style="Key.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=1
        )
        self.dic_subset_size_entry = ttk.Entry(self.fullfield_frame, textvariable=self.dic_subset_size, width=8)
        self.dic_subset_size_entry.grid(row=0, column=1, sticky="w", pady=1)
        self.add_tooltip(self.dic_subset_size_entry, subset_tip)

        step_tip = (
            "相邻测量点之间隔多少个像素。填大于 0 的整数；调小会测更多位置、计算更慢，调大会让结果图中的测量点更稀。"
        )
        ttk.Label(self.fullfield_frame, text="步长 px", style="Key.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=1
        )
        self.dic_step_entry = ttk.Entry(self.fullfield_frame, textvariable=self.dic_step, width=8)
        self.dic_step_entry.grid(row=1, column=1, sticky="w", pady=1)
        self.add_tooltip(self.dic_step_entry, step_tip)

        solver_tip = (
            "选择逐步修正图片块位置和形状的计算方法，下一次分析时生效。通常先用 IC-GN；难以找到位置时可尝试 IC-LM。"
        )
        ttk.Label(self.fullfield_frame, text="求解器", style="Key.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=1
        )
        self.dic_solver_box = ttk.Combobox(
            self.fullfield_frame,
            textvariable=self.dic_solver,
            values=list(DIC_SOLVERS),
            width=10,
            state="readonly",
        )
        self.dic_solver_box.grid(row=2, column=1, sticky="w", pady=1)
        self.add_tooltip(self.dic_solver_box, solver_tip, choices={
            DIC_SOLVER_ICGN: "高斯–牛顿方法：逐步修正图片块的位置和形状，通常计算较快。",
            DIC_SOLVER_ICLM: "带阻尼的修正方法：限制每一步的改动，初始位置不够准确时可尝试，通常较慢。",
        })

        ttk.Label(self.fullfield_frame, text="应变窗口（点）").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_strain_window_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_strain_window, width=8
        )
        self.dic_strain_window_entry.grid(row=3, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_strain_window_entry,
            "计算一处伸缩时，使用周围多少行、多少列的测量点。填至少 3 的奇数，如 7；调大会减少杂乱起伏，也会减弱细小的局部变化。",
        )

        ttk.Label(self.fullfield_frame, text="平滑 σ").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_smooth_sigma_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_smooth_sigma, width=8
        )
        self.dic_smooth_sigma_entry.grid(row=4, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_smooth_sigma_entry,
            "计算伸缩前，先把邻近位置的移动量做加权平均。σ 表示平均范围，以测量点间距为单位；填 0 关闭，调大会减少起伏，也会压低尖峰。",
        )

        ttk.Label(self.fullfield_frame, text="金字塔层数").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_pyramid_levels_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_pyramid_levels, width=8
        )
        self.dic_pyramid_levels_entry.grid(row=5, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_pyramid_levels_entry,
            "先在缩小的图片上找位置，再回到大图细算的层数。填 1 只用原图；移动较大时可试 2 到 4 层，所画区域需能容纳缩小后的图片块。",
        )

        ttk.Label(self.fullfield_frame, text="金字塔缩放").grid(row=6, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_pyramid_scale_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_pyramid_scale, width=8
        )
        self.dic_pyramid_scale_entry.grid(row=6, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_pyramid_scale_entry,
            "每层图片边长缩为上一层的多少倍，填 0 到 1 之间的数，不能等于两端。0.5 表示宽和高各减半；只有层数大于 1 时生效。",
        )

        ttk.Label(self.fullfield_frame, text="搜索 px").grid(row=7, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_search_radius_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_search_radius, width=8
        )
        self.dic_search_radius_entry.grid(row=7, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_search_radius_entry,
            "在预计位置周围最多寻找多少个像素，填大于 0 的整数。图片中的移动较大时可调大，计算会更慢；太小可能找不到对应纹理。",
        )

        ttk.Label(self.fullfield_frame, text="ZNCC 下限").grid(row=8, column=0, sticky="w", padx=(0, 6), pady=1)
        self.dic_zncc_min_entry = ttk.Entry(
            self.fullfield_frame, textvariable=self.dic_zncc_min, width=8
        )
        self.dic_zncc_min_entry.grid(row=8, column=1, sticky="w", pady=1)
        self.add_tooltip(
            self.dic_zncc_min_entry,
            "ZNCC 表示两个图片块的纹理相似程度，越接近 1 越相似。填 0 到 1；提高下限会拒绝更多不相似的位置，这些位置留空，不当作零。",
        )
        self.fullfield_frame.columnconfigure(3, weight=1)
        for widget in self.fullfield_frame.winfo_children():
            info = widget.grid_info()
            row, col = int(info["row"]), int(info["column"])
            if isinstance(widget, ttk.Label):
                entry = self.fullfield_frame.grid_slaves(row=row, column=1)[0]
                self.add_tooltip(widget, entry._tooltip.resolve_text)
            widget.grid_configure(row=row // 2, column=(row % 2) * 2 + col,
                                  sticky="ew" if col else "w", padx=(0, 8), pady=3)
            if isinstance(widget, ttk.Entry):
                widget.configure(width=5)

        draw_row = ttk.Frame(self.fullfield_frame, style="Card.TFrame")
        draw_row.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        draw_row.columnconfigure(1, weight=1)
        self.draw_field_roi_button = ttk.Button(
            draw_row,
            text="画全场 ROI",
            command=lambda: self.set_roi_mode(1),
            style="Secondary.TButton",
        )
        self.draw_field_roi_button.grid(row=0, column=0, sticky="w")
        self.add_tooltip(
            self.draw_field_roi_button,
            "点击后，在参考图片上按住左键拖出要分析的矩形区域（ROI）。程序将在框内逐点测量移动和伸缩；太靠近边缘、放不下完整图片块的位置会跳过。",
        )
        self.dic_field_summary_var = tk.StringVar(value="尚未绘制全场 ROI。")
        self.dic_field_summary_label = ttk.Label(
            draw_row,
            textvariable=self.dic_field_summary_var,
            style="Hint.TLabel",
            wraplength=280,
        )
        self.dic_field_summary_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.add_tooltip(self.dic_field_summary_label, "显示所画分析区域的宽、高和左上角位置，单位都是像素。要更换区域，可再次点击“画全场 ROI”后重新拖框。")
        options = ttk.Frame(self.fullfield_frame, style="Card.TFrame")
        options.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="拟合最高阶次").grid(row=0, column=0, sticky="w")
        self.dic_degree_box = ttk.Combobox(options, textvariable=self.dic_strain_degree, values=(1,2), width=4, state="readonly")
        self.dic_degree_box.grid(row=0, column=1, sticky="w")
        self.add_tooltip(self.dic_degree_box, "选择如何用周围测量点的移动量计算局部伸缩，下一次分析时生效。", choices={
            1: "用直线变化近似周围的移动量，适合较均匀的变形。",
            2: "优先考虑弯曲变化，适合局部变形不均匀的区域；可用点不足时会尝试一阶计算。",
        })
        robust_check = ttk.Checkbutton(options, text="稳健拟合", variable=self.dic_robust_strain)
        robust_check.grid(row=0, column=2, sticky="w")
        self.add_tooltip(robust_check, "勾选后，计算局部伸缩时会减小少数偏离较大的测量点的影响。取消后，周围各点按普通方式参与计算。")
        ttk.Label(options, text="异常位移下限 px").grid(row=1, column=0, sticky="w", pady=4)
        self.dic_outlier_entry = ttk.Entry(options, textvariable=self.dic_outlier_threshold, width=6)
        self.dic_outlier_entry.grid(row=1, column=1, sticky="w")
        self.add_tooltip(self.dic_outlier_entry, "检查某点是否比周围位置预计的移动量偏离过多，数值以像素计，并结合周围点的起伏判断。填 0 关闭；被排除点的原始移动量仍会保存。")
        ttk.Label(options, text="试样遮罩").grid(row=2, column=0, sticky="w")
        self.dic_mask_box = ttk.Combobox(options, textvariable=self.dic_mask_mode, values=("矩形 ROI", "自动纹理", "导入遮罩"), width=10, state="readonly")
        self.dic_mask_box.grid(row=2, column=1, sticky="w")
        self.add_tooltip(self.dic_mask_box, "遮罩是一张指定哪些位置参与计算的黑白图。选择范围后，可点“预览遮罩”检查，下一次分析时生效。", choices={
            "矩形 ROI": "分析所画矩形内的位置，并跳过手动画出的排除区。ROI 就是在图上画出的分析区域。",
            "自动纹理": "根据明暗纹理自动排除平坦背景和孔洞；请先预览，检查样品边缘是否被误排除。",
            "导入遮罩": "使用与参考图片尺寸相同的黑白图片，白色保留、黑色排除；点旁边“导入”选择文件。",
        })
        self.dic_mask_load_button = ttk.Button(options, text="导入", command=self.select_dic_mask, style="Compact.TButton")
        self.dic_mask_load_button.grid(row=2, column=2, sticky="ew")
        self.add_tooltip(self.dic_mask_load_button, "点击选择与参考图片同样宽、高的黑白遮罩图，白色位置参与计算，黑色位置跳过。选好后会切换为“导入遮罩”并预览。")
        self.dic_mask_path_label = ttk.Label(options, textvariable=self.dic_mask_path, wraplength=280, style="Hint.TLabel")
        self.dic_mask_path_label.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
        self.add_tooltip(self.dic_mask_path_label, "这里显示所选黑白遮罩图片的完整路径。要换文件，请点击“导入”。")
        actions = ttk.Frame(options, style="Card.TFrame")
        actions.grid(row=4, column=0, columnspan=3, sticky="ew")
        for column, (label, action, tip) in enumerate((
            ("预览遮罩", self.preview_dic_mask, "先画好分析区域，再点击查看哪些位置被排除。参考图片上的红色区域将被跳过；这里只预览，不开始计算。"),
            ("画排除区", self.draw_dic_exclusion, "先画好分析区域，再点击并在参考图片上拖出矩形，排除孔洞、背景等不想计算的位置。可重复画多个框。"),
            ("清空排除区", self.clear_dic_exclusions, "点击移除所有手动画出的排除框，并更新预览。自动纹理或导入黑白图指定的排除位置仍然有效。"),
        )):
            button = ttk.Button(actions, text=label, command=action, style="Compact.TButton")
            button.grid(row=0, column=column, padx=(0,4), sticky="ew")
            self.add_tooltip(button, tip)
        for row in range(3):
            label = options.grid_slaves(row=row, column=0)[0]
            entry = options.grid_slaves(row=row, column=1)[0]
            self.add_tooltip(label, entry._tooltip.resolve_text)
        convergence_check = ttk.Checkbutton(options, text="仅接受收敛子集", variable=self.dic_reject_nonconverged)
        convergence_check.grid(row=5, column=0, columnspan=3, sticky="w", pady=3)
        self.add_tooltip(convergence_check, "收敛表示反复修正后，图片块位置和形状的改动已小到可以停止。勾选后，只使用达到停止条件的点；取消后可查看未达到条件的候选结果，计算状态仍会保存。")
        ttk.Label(options, text="遮罩中白色保留、黑色排除。\n计算范围越大，起伏越少，细小变化也越不明显。", style="Hint.TLabel", wraplength=300).grid(row=6, column=0, columnspan=3, sticky="w", pady=5)
        self.fullfield_frame.grid_remove()

    def dic_mask_settings(self):
        mode = {"矩形 ROI": "none", "自动纹理": "auto", "导入遮罩": "file"}[self.dic_mask_mode.get()]
        return {"mode": mode, "path": self.dic_mask_path.get().strip() if mode == "file" else None,
                "texture_threshold": 3.0, "exclusions": [list(rect) for rect in self.dic_mask_exclusions]}

    def dic_display_options(self):
        options = {"style": "contour" if self.dic_display_style.get() == "连续云图" else "points",
                   "background": {"无底图": "none", "参考图": "reference", "变形图": "deformed"}[self.dic_view_background.get()],
                   "color_mode": {"数据范围": "range", "零点对称": "symmetric", "手动范围": "manual"}[self.dic_color_mode.get()],
                   "percent": bool(self.dic_percent.get()), "cmap": self.dic_colormap.get(), "alpha": .65}
        if options["color_mode"] == "manual":
            try:
                options["vmin"], options["vmax"] = float(self.dic_color_min.get()), float(self.dic_color_max.get())
                if not np.isfinite([options["vmin"],options["vmax"]]).all() or options["vmin"] >= options["vmax"]:
                    raise ValueError()
            except ValueError as exc:
                raise RuntimeError("手动色标需要有限且递增的最小值和最大值。") from exc
        return options

    def select_dic_mask(self):
        if self.is_processing or self._completion_pending:
            return
        path = filedialog.askopenfilename(title="选择参考图像遮罩（白色保留）", filetypes=[("遮罩图像", "*.png *.tif *.tiff *.bmp")])
        if path:
            self.dic_mask_path.set(path)
            self.dic_mask_mode.set("导入遮罩")
            self.preview_dic_mask()

    def preview_dic_mask(self):
        if self.is_processing or self._completion_pending or self.field_roi is None:
            return
        try:
            index = int(self.field_roi_reference_frame_1based)-1
            raw = read_gray_image(self.image_paths[index])
            reference = normalize_to_uint8(raw)
            mask, record, _ = _core._resolve_specimen_mask(reference, self.field_roi, {"mask": self.dic_mask_settings()})
            self.current_preview_index = index
            self.current_fullres_img8 = reference
            self._canvas_shows_field_overlay = False
            rgb = cv2.cvtColor(reference, cv2.COLOR_GRAY2RGB)
            if mask is not None:
                rgb[~mask] = (.45*rgb[~mask]+.55*np.array([220,55,55])).astype(np.uint8)
            height, width = reference.shape
            self.display_img = cv2.resize(rgb, (max(1,round(width*self.display_scale)), max(1,round(height*self.display_scale))))
            self.show_image()
            self._show_image_workspace()
            self.log(f"遮罩预览：{record.get('included_pixels', reference.size)} 个保留像素；排除区 {len(self.dic_mask_exclusions)} 个。请核对孔洞、背景和试样边缘。")
        except Exception as exc:
            messagebox.showerror("遮罩无效", str(exc))

    def draw_dic_exclusion(self):
        if self.field_roi is None or self.is_processing or self._completion_pending:
            return
        self.set_roi_mode(1)
        if self.current_preview_index+1 != self.field_roi_reference_frame_1based:
            self.status_var.set("请先切换到绘制全场 ROI 的参考帧。")
            return
        self._drawing_mask_exclusion = True
        self.status_var.set("请拖出排除区域：孔洞、背景或无法跟踪的区域。")

    def clear_dic_exclusions(self):
        if self.is_processing or self._completion_pending:
            return
        self.dic_mask_exclusions = []
        self._drawing_mask_exclusion = False
        self.preview_dic_mask()

    def is_fullfield_mode(self):
        return str(self.analysis_mode.get()) == ANALYSIS_MODE_FULLFIELD

    def set_analysis_mode(self, *_args):
        requested = str(self.analysis_mode.get())
        committed = getattr(self, "_committed_analysis_mode", ANALYSIS_MODE_EXTENSOMETER)
        if getattr(self, "is_processing", False) or getattr(self, "_completion_pending", False):
            if requested != committed:
                self.analysis_mode.set(committed)
                if hasattr(self, "status_var"):
                    self.status_var.set("正在处理或收尾：不能切换分析模式。")
                if hasattr(self, "log"):
                    self.log("分析进行中，已保持当前模式。")
            return
        self._committed_analysis_mode = requested
        is_ff = self.is_fullfield_mode()
        previous_kind = getattr(self, "_viewer_kind", "extensometer")
        target_kind = "fullfield" if is_ff else "extensometer"
        if previous_kind != target_kind and (
            getattr(self, "viewer_figure", None) is not None
            or getattr(self, "results_df", None) is not None
            or getattr(self, "dic_last_field", None) is not None
        ):
            self.last_qc_summary = None
            if hasattr(self, "qc_overview_var"):
                self.qc_overview_var.set("分析完成后显示 QC 总览。")
            if hasattr(self, "clear_viewer"):
                self.clear_viewer(keep_placeholder=True)
            self._restore_sequence_preview()
        if hasattr(self, "fullfield_frame"):
            if is_ff:
                self.fullfield_frame.grid()
            else:
                self.fullfield_frame.grid_remove()
        # The frame range is shared.  All 1-D-only controls stay hidden in
        # full-field mode so stale ROI/export settings cannot look actionable.
        if hasattr(self, "measure_core_frame"):
            if is_ff:
                self.measure_core_frame.grid_remove()
                self.advanced_toggle_btn.grid_remove()
                self.advanced_frame.grid_remove()
            else:
                self.measure_core_frame.grid()
                self.advanced_toggle_btn.grid()
                if self.advanced_visible.get():
                    self.advanced_frame.grid()
        if hasattr(self, "roi_group_frame"):
            (self.roi_group_frame.grid_remove if is_ff else self.roi_group_frame.grid)()
        if hasattr(self, "export_frame"):
            (self.export_frame.grid_remove if is_ff else self.export_frame.grid)()
        if hasattr(self, "fullfield_export_info_frame"):
            (self.fullfield_export_info_frame.grid if is_ff else self.fullfield_export_info_frame.grid_remove)()
        if is_ff:
            self.current_roi_index = 1
            if hasattr(self, "status_var"):
                self.status_var.set("全场 2D DIC：请在参考帧拖出全场 ROI，再开始分析。")
            if hasattr(self, "export_hint_label"):
                self.export_hint_label.configure(
                    text="确认参考帧、全场 ROI 和 DIC 参数后开始；全场核心输出固定生成。"
                )
            if hasattr(self, "workflow_step_texts"):
                self.workflow_step_texts = [
                    "1. 选择图像文件夹和输出文件夹，加载序列",
                    "2. 设置参考帧与分析范围",
                    "3. 画全场 ROI，设置子集/步长/IC-GN 或 IC-LM",
                    "4. 开始分析，查看 u/v/应变图",
                ]
        else:
            if hasattr(self, "export_hint_label"):
                self.export_hint_label.configure(
                    text="确认参考帧、ROI 方向和导出内容后再开始。"
                )
            if hasattr(self, "workflow_step_texts"):
                self.workflow_step_texts = [
                    "1. 选择图像文件夹和输出文件夹，加载序列",
                    "2. 设置参考帧、分析范围和测量方向",
                    "3. 画 ROI1/ROI2，添加 ROI 组",
                    "4. 确认导出内容，点击开始分析",
                ]
        if hasattr(self, "workflow_steps_label"):
            self.workflow_steps_label.configure(text=self._visible_workflow_steps_text())
        if hasattr(self, "controls_canvas"):
            try:
                self.controls_panel.update_idletasks()
                self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))
            except Exception:
                pass
        self.update_workflow_action_states()

    def _build_advanced_controls(self):
        advanced_fields = [
            (
                "search_radius_entry",
                "搜索半径 px",
                self.search_radius,
                7,
                "在上一次找到的位置周围，最多再找多少个像素。填正整数；移动较大时可调大，计算会更慢，也更可能找到相似但错误的位置。",
            ),
            (
                "hard_corr_entry",
                "严格接受阈值",
                self.hard_corr,
                7,
                "两个测量框与原有纹理至少要多相似才通过检查，分数越接近 1 越相似。填 -1 到 1；调高会拒绝更多图片，调低也更容易接受错误位置。",
            ),
            (
                "soft_corr_entry",
                "弱接受阈值",
                self.soft_corr,
                7,
                "允许较难识别的图片再接受额外检查时，纹理相似分数的最低值。不得高于严格接受阈值；只有勾选“启用自适应弱接受”时生效。",
            ),
            (
                "max_frame_strain_jump_entry",
                "单帧应变突变上限",
                self.max_frame_strain_jump,
                8,
                "相邻两张有效图片之间，允许伸缩比例改变多少。0.01 表示相差 1 个百分点；调大允许变化更快，留空关闭这项检查。",
            ),
            (
                "fb_tolerance_entry",
                "FB 容差 px",
                self.fb_tolerance_px,
                7,
                "FB 是先向后追踪、再返回前一张图片检查位置。这里填返回位置允许偏离原位置多少个像素；调小更严格，仅在勾选前后向检查时生效。",
            ),
            (
                "template_alpha_entry",
                "模板跟随系数",
                self.template_alpha,
                7,
                "找到新位置后，用当前图片的纹理替换原有追踪模板的比例。填 0 到 1；0 保留原模板，1 完全替换。仅在勾选模板跟随时生效，过大可能逐渐跟错位置。",
            ),
            (
                "min_texture_std_entry",
                "最小灰度标准差",
                self.min_texture_std,
                8,
                "灰度标准差表示测量框里明暗变化有多明显。低于填写值时会提醒纹理太平坦、难以追踪；调高会更容易出现提醒，不会让图片本身变清晰。",
            ),
            (
                "min_texture_contrast_entry",
                "最小 P95-P5 对比度",
                self.min_texture_contrast,
                8,
                "P95-P5 是较亮与较暗像素的灰度差，忽略最亮和最暗各 5%。差值低于这里填写的数时，提醒测量框的明暗对比不足；调高会更容易提醒。",
            ),
            (
                "max_saturated_frac_entry",
                "最大近黑/近白比例",
                self.max_saturated_frac,
                8,
                "允许测量框里接近全黑或全白的像素占多少，填 0 到 1，例如 0.2 是 20%。超过时提醒可能曝光不足或过亮；调小会更早提醒。",
            ),
            (
                "overlay_every_entry",
                "overlay 间隔",
                self.overlay_every,
                7,
                "勾选“叠加图像”后，每隔多少张图片保存一次带测量框的检查图。填正整数；数值小会保存更多图片，便于查看是否跟错位置。",
            ),
        ]
        self.advanced_frame.columnconfigure(1, weight=1)
        for idx, (attr_name, label, variable, width, tooltip) in enumerate(advanced_fields):
            row = idx
            col = 0
            label_widget = ttk.Label(
                self.advanced_frame,
                text=label,
                style="Key.TLabel" if idx < 4 else "TLabel",
            )
            label_widget.grid(row=row, column=col, sticky="w", padx=(0, 4), pady=3)
            self.add_tooltip(label_widget, tooltip)
            entry = ttk.Entry(self.advanced_frame, textvariable=variable, width=width)
            entry.grid(
                row=row, column=col + 1, sticky="ew", padx=(0, 12), pady=3
            )
            setattr(self, attr_name, entry)
            self.add_tooltip(entry, tooltip)

        option_row = len(advanced_fields)
        adaptive_tip = (
            "勾选后，纹理不够相似的图片仍可按较低相似下限和其他已启用检查判断。取消后只接受严格检查通过的图片，未通过的位置会留空。"
        )
        self.enable_adaptive_check = ttk.Checkbutton(self.advanced_frame, text="启用自适应弱接受", variable=self.enable_adaptive)
        self.enable_adaptive_check.grid(
            row=option_row, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.add_tooltip(self.enable_adaptive_check, adaptive_tip)
        template_follow_tip = (
            "勾选后，用每次找到的图片块逐渐更新追踪纹理，更新比例由“模板跟随系数”决定。取消后保留参考图片的纹理；跟随可适应外观变化，也可能逐渐跟错位置。"
        )
        self.use_prev_frame_template_check = ttk.Checkbutton(
            self.advanced_frame,
            text="使用前一帧模板跟随",
            variable=self.use_prev_frame_template,
        )
        self.use_prev_frame_template_check.grid(
            row=option_row + 1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.add_tooltip(self.use_prev_frame_template_check, template_follow_tip)
        fb_tip = (
            "勾选后，找到新位置还会反向追踪回上一张有效图片；返回位置偏差超过“FB 容差”就不接受。取消可减少计算，但少了这项位置核对。"
        )
        self.enable_fb_check_check = ttk.Checkbutton(self.advanced_frame, text="前后向一致性检查", variable=self.enable_fb_check)
        self.enable_fb_check_check.grid(
            row=option_row + 2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        self.add_tooltip(self.enable_fb_check_check, fb_tip)

    def _build_roi_section(self, parent):
        group_frame = ttk.LabelFrame(parent, text="03  ROI 配对与分组", padding=(10, 8))
        self.roi_group_frame = group_frame
        group_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        group_frame.columnconfigure(0, weight=1)

        tool_row = ttk.Frame(group_frame, style="Card.TFrame")
        tool_row.grid(row=0, column=0, sticky="ew")
        tool_row.columnconfigure((0, 1), weight=1, uniform="roi")
        self.roi1_button = ttk.Button(tool_row, text="画 ROI 1", command=lambda: self.set_roi_mode(1), style="Compact.TButton")
        self.roi1_button.grid(row=0, column=0, padx=(0, 6), pady=2, sticky="ew")
        self.add_tooltip(
            self.roi1_button,
            "点击后，在参考图片上按住左键拖出第一个测量框（ROI 1）。框内要有清楚的斑点或纹理，程序将追踪这个位置；画完会切换到第二个框。",
        )
        self.roi2_button = ttk.Button(tool_row, text="画 ROI 2", command=lambda: self.set_roi_mode(2), style="Compact.TButton")
        self.roi2_button.grid(row=0, column=1, pady=2, sticky="ew")
        self.add_tooltip(
            self.roi2_button,
            "点击后，在参考图片上按住左键拖出第二个测量框（ROI 2）。程序比较两个框中心之间的距离变化；画好后点“添加 ROI 组”保存这一对。",
        )
        self.align_x_button = ttk.Button(tool_row, text="水平对齐 · x", command=lambda: self.align_current_pair("x", set_mode=True), style="Compact.TButton")
        self.align_x_button.grid(row=1, column=0, padx=(0, 6), pady=2, sticky="ew")
        self.add_tooltip(
            self.align_x_button,
            "点击把两个测量框移到同一水平线上，并切换为测量左右间距（x 方向）。位置会改变，适合测量部位左右排列的图片。",
        )
        self.align_y_button = ttk.Button(tool_row, text="垂直对齐 · y", command=lambda: self.align_current_pair("y", set_mode=True), style="Compact.TButton")
        self.align_y_button.grid(row=1, column=1, pady=2, sticky="ew")
        self.add_tooltip(
            self.align_y_button,
            "点击把两个测量框移到同一垂直线上，并切换为测量上下间距（y 方向）。位置会改变，适合测量部位上下排列的图片。",
        )

        roi_summary_frame = ttk.Frame(group_frame, style="Card.TFrame", padding=(0, 4))
        roi_summary_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        roi_summary_frame.columnconfigure(0, weight=1)
        self.current_roi_summary_label = ttk.Label(
            roi_summary_frame,
            textvariable=self.current_roi_summary_var,
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=430,
        )
        self.current_roi_summary_label.grid(row=0, column=0, sticky="ew")
        self.add_tooltip(
            self.current_roi_summary_label,
            "显示当前两个测量框的尺寸、测量方向和图像是否容易识别。L0 是参考图片上的初始间距；添加为一组前，可在这里核对。",
        )

        form_row = ttk.Frame(group_frame, style="Card.TFrame")
        form_row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        form_row.columnconfigure((1, 3), weight=1)
        group_name_tip = (
            "给这一对测量框起个名称，如“样品左侧”。添加或更新时会使用这个名称；留空会自动编号为 G01、G02 等，也会用于结果文件名。"
        )
        self.group_name_label = ttk.Label(form_row, text="组名：")
        self.group_name_label.grid(row=0, column=0, padx=(0, 4), pady=2, sticky="w")
        self.add_tooltip(self.group_name_label, group_name_tip)
        self.group_name_entry = ttk.Entry(form_row, textvariable=self.group_name_var, width=8)
        self.group_name_entry.grid(row=0, column=1, padx=(0, 8), pady=2, sticky="ew")
        self.add_tooltip(self.group_name_entry, group_name_tip)
        role_tip = (
            "说明这组测量沿拉伸方向还是垂直于拉伸方向，添加或更新后生效。泊松比表示横向收缩与纵向伸长的比值，需两种角色同时存在。"
        )
        self.roi_role_label = ttk.Label(form_row, text="角色：", style="Key.TLabel")
        self.roi_role_label.grid(row=0, column=2, padx=(0, 4), pady=2, sticky="w")
        self.add_tooltip(self.roi_role_label, role_tip)
        self.roi_role_box = ttk.Combobox(
            form_row,
            textvariable=self.roi_role_display,
            values=list(ROI_ROLE_LABEL_TO_VALUE.keys()),
            width=8,
            state="readonly",
        )
        self.roi_role_box.grid(row=0, column=3, pady=2, sticky="ew")
        self.roi_role_box.bind("<<ComboboxSelected>>", self.sync_roi_role_from_display)
        self.add_tooltip(self.roi_role_box, role_tip, choices={
            "普通": "只计算这组的距离变化，不参加泊松比计算，首次只测伸缩可选此项。",
            "拉伸方向": "这对框测量沿样品受拉方向的伸长，作为泊松比计算中的纵向测量。",
            "横向方向": "这对框测量垂直于样品受拉方向的收缩，需同时添加拉伸方向组才能计算泊松比。",
        })
        self.add_group_button = ttk.Button(form_row, text="添加 ROI 组", command=self.add_current_group, style="Secondary.TButton")
        self.add_group_button.grid(row=1, column=0, columnspan=4, pady=(6, 2), sticky="ew")
        self.add_tooltip(
            self.add_group_button,
            "点击把当前两个测量框、名称、方向和角色保存到下方列表。每组会单独计算一条伸缩曲线；先在同一参考图片上画好两个框。",
        )

        action_row = ttk.Frame(group_frame, style="Card.TFrame")
        action_row.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        action_row.columnconfigure((0, 1), weight=1, uniform="group")
        self.update_group_button = ttk.Button(action_row, text="更新选中", command=self.update_selected_group, style="Compact.TButton")
        self.update_group_button.grid(row=0, column=1, pady=2, sticky="ew")
        self.add_tooltip(
            self.update_group_button,
            "先选中列表中的一组，再点击，用当前两个测量框、名称、方向和角色替换该组设置。下一次分析将使用更新后的设置。",
        )
        self.load_group_button = ttk.Button(action_row, text="载入选中", command=self.load_selected_group, style="Compact.TButton")
        self.load_group_button.grid(row=0, column=0, padx=(0, 6), pady=2, sticky="ew")
        self.add_tooltip(
            self.load_group_button,
            "先选中列表中的一组，再点击，显示它的两个测量框并填回名称、方向和角色。修改后点击“更新选中”才会保存到这组。",
        )
        self.delete_group_button = ttk.Button(action_row, text="删除选中", command=self.delete_selected_group, style="Danger.TButton")
        self.delete_group_button.grid(row=1, column=1, pady=2, sticky="ew")
        self.add_tooltip(
            self.delete_group_button,
            "先选中一组，再点击并确认删除。这组将不再参加后续分析，已经保存的结果文件会保留。",
        )
        self.clear_rois_button = ttk.Button(action_row, text="清除当前 ROI", command=self.clear_current_rois, style="Danger.TButton")
        self.clear_rois_button.grid(row=1, column=0, padx=(0, 6), pady=2, sticky="ew")
        self.add_tooltip(
            self.clear_rois_button,
            "点击并确认后，清空当前正在画的两个测量框，便于重新绘制；列表中已保存的分组会保留。也可按 Esc。",
        )

        tree_frame = ttk.Frame(group_frame, style="Card.TFrame")
        tree_frame.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        tree_frame.columnconfigure(0, weight=1)
        columns = ("name", "role", "selected", "actual", "L0", "dx", "dy", "roi1", "roi2")
        # 保留水平滚动，首屏优先显示按钮和图像，不让列表请求宽度撑爆窗口。
        self.group_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=3,
                                      selectmode="browse", displaycolumns=("name", "role", "actual", "L0", "selected", "dx", "dy", "roi1", "roi2"))
        for col in columns:
            self.group_tree.heading(col, text=GROUP_TREE_HEADING_TEXTS[col])
            self.group_tree.column(
                col,
                width=80 if col in ("name", "role", "actual", "L0") else GROUP_TREE_COLUMN_WIDTHS[col],
                minwidth=40,
                anchor="center",
                stretch=False,
            )
        self.group_tree.grid(row=0, column=0, sticky="ew")
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.group_tree.xview)
        self.group_tree.configure(xscrollcommand=tree_scroll_x.set)
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        self.add_tooltip(tree_scroll_x, "左右拖动查看分组表格中被隐藏的列，如测量框的位置和尺寸。")
        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.group_tree.yview)
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        self.add_tooltip(tree_scroll_y, "上下拖动查看列表中更多已保存的测量组。")
        self.group_tree.configure(yscrollcommand=tree_scroll_y.set)
        self.group_tree.bind("<Double-1>", lambda event: self.load_selected_group())
        self.group_tree.bind("<<TreeviewSelect>>", lambda _event: self.update_workflow_action_states())
        self.group_tree.bind("<Button-3>", self._show_group_tree_context_menu)  # 右键菜单
        self.add_tooltip(
            self.group_tree,
            self._group_table_help,
        )

    def _build_image_section(self, parent):
        self.image_frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        self.image_frame.grid(row=0, column=0, sticky="nsew")
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(self.image_frame, style="Card.TFrame")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(5, weight=1)
        buttons = [
            ("btn_zoom_out", "−", lambda: self.zoom_image(1 / 1.25), 3, "点击缩小图片，看到更大的范围（Ctrl + 减号）。只改变显示大小。"),
            ("btn_zoom_in", "+", lambda: self.zoom_image(1.25), 3, "点击放大图片，便于看清纹理和画框（Ctrl + 加号）。超出窗口的部分可用滚动条查看。"),
            ("btn_fit", "适应窗口", self.fit_image_to_view, 8, "点击把整张原图缩放到窗口内（Ctrl + F）。窗口大小改变时会自动调整，测量框仍对应原图位置。"),
            ("btn_1to1", "1:1", self.show_image_1to1, 4, "点击按原始像素大小显示图片，一个图片像素对应一个显示像素。图片较大时，拖动滚动条查看边缘。"),
        ]
        for col, (attr, text, command, width, tip) in enumerate(buttons):
            button = ttk.Button(toolbar, text=text, command=command, style="Compact.TButton", width=width)
            button.grid(row=0, column=col, padx=(0, 4))
            setattr(self, attr, button)
            self.add_tooltip(button, tip)
        self.zoom_label_var = tk.StringVar(value="—")
        self.zoom_label = ttk.Label(toolbar, textvariable=self.zoom_label_var, style="Hint.TLabel", width=7, anchor="center")
        self.zoom_label.grid(row=0, column=4, padx=(2, 4))
        self.add_tooltip(self.zoom_label, "显示图片相对原始像素大小的比例。100% 是原始大小，200% 是放大到两倍；可用加减按钮或滚轮调整。")
        self.canvas = tk.Canvas(self.image_frame, bg="#111827", cursor="crosshair", highlightthickness=0,
                                width=400, height=260, xscrollincrement=1, yscrollincrement=1)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        scroll_x = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        scroll_y = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scroll_x.grid(row=2, column=0, sticky="ew")
        scroll_y.grid(row=1, column=1, sticky="ns")
        self.add_tooltip(scroll_x, "左右拖动查看放大图片的左侧或右侧，画出的测量框仍按原图位置记录。")
        self.add_tooltip(scroll_y, "上下拖动查看放大图片的顶部或底部，便于检查窗口之外的测量位置。")
        self.canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Configure>", self._on_image_canvas_configure, add="+")
        self.image_context_var = tk.StringVar(value="在参考帧上绘制 ROI；滚轮缩放，滚动条移动视图。")
        self.image_context_label = ttk.Label(self.image_frame, textvariable=self.image_context_var,
                                             style="Hint.TLabel", wraplength=600, justify=tk.LEFT)
        self.image_context_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.image_frame.bind("<Configure>", lambda event: self.image_context_label.configure(wraplength=max(240, event.width - 24)))
        self.add_tooltip(self.canvas, lambda _event: (
            "按住左键拖出不想计算的矩形区域，松开后会添加一个排除框。滚轮缩放图片；红色区域表示被排除的位置。"
            if self._drawing_mask_exclusion else
            "在参考图片上按住左键拖出要分析的矩形区域，松开后保存范围。滚轮缩放图片，滚动条移动视图。"
            if self.is_fullfield_mode() else
            f"在参考图片上按住左键拖出第 {self.current_roi_index} 个测量框，松开后保存。两个框画好后点“添加 ROI 组”；滚轮可缩放图片。"))
        self.add_tooltip(self.image_context_label, "显示当前图片名称、序号和尺寸，以及用来比较变形的参考图片。px 表示像素；绘制提示说明现在要画哪个测量框。")

    def _draw_image_empty_state(self):
        if not hasattr(self, "canvas") or self.display_img is not None:
            return
        self.canvas.delete("empty_state")
        cx, cy = self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2
        self.canvas.create_text(cx, cy - 24, text="加载图像序列", fill="#e2e8f0",
                                 font=(UI_FONT_FAMILY, 16, "bold"), tags="empty_state")
        self.canvas.create_text(cx, cy + 24, text="选择图像文件夹，然后点击“加载序列”\n支持 TIFF、PNG、JPEG 和 BMP",
                                 fill="#b0bfd2", justify=tk.CENTER, font=self.ui_base_font,
                                 width=max(200, self.canvas.winfo_width() - 50), tags="empty_state")

    def _on_image_canvas_configure(self, _event=None):
        if self.display_img is None:
            self._draw_image_empty_state()
        else:
            self._update_image_scrollregion()

    def _update_image_scrollregion(self):
        dh, dw = self.display_img.shape[:2]
        pad_x = max(0, (self.canvas.winfo_width() - dw) // 2)
        pad_y = max(0, (self.canvas.winfo_height() - dh) // 2)
        self.canvas.config(scrollregion=(-pad_x, -pad_y, dw + pad_x, dh + pad_y))
        if self.auto_fit_enabled or pad_x:
            self.canvas.xview_moveto(0)
        if self.auto_fit_enabled or pad_y:
            self.canvas.yview_moveto(0)

    def _update_image_context(self):
        if not hasattr(self, "image_context_var") or not self.image_paths or self.first_img8 is None:
            return
        if self._canvas_shows_field_overlay and self.dic_last_image is not None:
            h, w = self.dic_last_image.shape[:2]
            text = f"结果叠加 · {self.dic_field_component.get()} · 第 {self.dic_last_frame_1based} 帧 · {self.dic_last_filename} · {w} × {h} px · 参考帧 {self.dic_last_reference_frame_1based}"
        else:
            start, _ = self._safe_int_var(self.start_frame_1based)
            filename = os.path.basename(self.image_paths[self.current_preview_index])
            h, w = self.first_img8.shape[:2]
            label = "全场 ROI" if self.is_fullfield_mode() else f"ROI {self.current_roi_index}"
            text = f"{filename} · 第 {self.current_preview_index + 1}/{len(self.image_paths)} 帧 · {w} × {h} px · 参考帧 {start if start is not None else '待校验'} · 绘制 {label}"
        self.image_context_var.set(text)

    def _build_analysis_section(self, parent):
        self.analysis_frame = ttk.LabelFrame(parent, text="04  导出设置", padding=(10, 8))
        self.analysis_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.analysis_frame.columnconfigure(0, weight=1)
        self.export_frame = ttk.Frame(self.analysis_frame, style="Card.TFrame")
        self.export_frame.grid(row=0, column=0, sticky="ew")
        self.export_frame.columnconfigure((0, 1), weight=1)
        preset_bar = ttk.Frame(self.export_frame, style="Card.TFrame")
        preset_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        preset_bar.columnconfigure((0, 1, 2), weight=1, uniform="export")
        presets = [
            ("export_research_preset_button", "推荐", self._apply_research_preset, "点击自动勾选数值表格、伸缩曲线图片、质量摘要和参数记录，其余选项取消。开始分析后才会生成文件。"),
            ("export_quick_preset_button", "快速查看", self._apply_quick_view_preset, "点击只保留伸缩曲线图片和质量摘要，取消其他导出选项。适合先查看分析效果，开始分析后才会生成文件。"),
            ("export_all_preset_button", "完整导出", self._apply_all_export_preset, "点击勾选所有导出内容，分析时会保存更多文件。包括 Origin 绘图软件的工程文件，需本机装有 OriginPro 和相应连接组件。"),
        ]
        for col, (attr, text, command, tip) in enumerate(presets):
            button = ttk.Button(preset_bar, text=text, command=command, style="Compact.TButton")
            button.grid(row=0, column=col, sticky="ew", padx=(0, 4 if col < 2 else 0))
            setattr(self, attr, button)
            self.add_tooltip(button, tip)
        options = [
            ("Origin TXT", self.export_origin_txt, "勾选后保存可导入 Origin 或表格软件的文本数据：图片序号、相对初始距离的伸缩比例、按距离比取对数的真应变。无需安装 Origin。"),
            ("应变 PNG", self.export_engineering_png, "勾选后，给每组保存一张伸缩曲线图片。纵轴是距离变化除以初始距离，例如 0.01 表示伸长 1%。PNG 是常见图片格式。"),
            ("QC 摘要", self.export_qc_summary, "QC 表示质量检查。勾选后保存哪些图片通过或未通过检查、纹理相似程度等摘要，便于判断结果是否需要复查。"),
            ("参数记录", self.export_parameters, "勾选后把本次使用的分析设置和通过检查的数量保存为文本，方便以后核对或重复分析。"),
            ("完整 CSV", self.export_full_csv, "勾选后保存详细表格，包括每张图片上的测量框位置、伸缩值和检查结果。CSV 是可用 Excel 等表格软件打开的文本表格。"),
            ("相关系数 PNG", self.export_corr_plot, "勾选后保存纹理相似程度随图片序号变化的曲线图。分数越接近 1 越相似，突然下降的位置值得复查。"),
            ("叠加图像", self.export_overlays, "勾选后按“高级设置”中的间隔保存带追踪框的原图，用来检查框是否始终跟在同一部位。"),
            ("论文图表包", self.export_publication_figures, "勾选后额外保存每英寸 600 个点的高清图片，以及放大后仍清晰的矢量图，便于排版或编辑。会增加文件数量。"),
            ("Origin OPJU", self.export_origin_opju, "勾选后生成可在 Origin 绘图软件中继续编辑的工程文件。需要 OriginPro 2021 或更新版本及 originpro 连接组件；缺少时会报告生成失败。"),
        ]
        self.export_checkbuttons = []
        for idx, (text, variable, tip) in enumerate(options):
            check = ttk.Checkbutton(self.export_frame, text=text, variable=variable)
            check.grid(row=idx // 2 + 1, column=idx % 2, sticky="w", pady=2)
            self.export_checkbuttons.append(check)
            self.add_tooltip(check, tip)
        self.fullfield_export_info_frame = ttk.Frame(self.analysis_frame, style="Card.TFrame")
        self.fullfield_export_info_frame.grid(row=1, column=0, sticky="ew")
        self.fullfield_export_info_frame.columnconfigure(0, weight=1)
        self.fullfield_export_info_label = ttk.Label(
            self.fullfield_export_info_frame,
            text="每张算出有效结果的图片都会保存数值表和移动、伸缩图。\n移动量以像素计；算不出的位置留空，不当作零。",
            style="Hint.TLabel", justify=tk.LEFT, wraplength=360)
        self.fullfield_export_info_label.grid(row=0, column=0, sticky="ew")
        self.add_tooltip(self.fullfield_export_info_label, "全场分析会自动保存各测量点的数值表和结果图片。NaN 表示这里没有算出有效数值，不能按零移动或零伸缩理解。")
        self.fullfield_export_overlays_checkbutton = ttk.Checkbutton(
            self.fullfield_export_info_frame, text="额外导出 Exx 叠加图", variable=self.export_overlays)
        self.fullfield_export_overlays_checkbutton.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.add_tooltip(self.fullfield_export_overlays_checkbutton, "勾选后，额外把左右方向的伸缩值（Exx）用颜色画在对应的分析图片上，方便核对位置；分析结束时保存为图片。")
        self.export_hint_label = ttk.Label(self.analysis_frame, style="Hint.TLabel", justify=tk.LEFT, wraplength=360)
        self.export_hint_label.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.add_tooltip(self.export_hint_label, "导出选项决定下一次分析保存哪些文件。选好后点击“开始分析”，完成后可用顶部“打开输出”查看文件。")

    def _build_run_section(self, parent, row=2):
        self.run_frame = ttk.Frame(parent, style="Card.TFrame", padding=(12, 6))
        self.run_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.run_frame.columnconfigure(0, weight=1)
        self.run_state_var = tk.StringVar(value="待加载")
        self.run_state_label = ttk.Label(self.run_frame, textvariable=self.run_state_var, style="Badge.TLabel")
        self.run_state_label.grid(row=0, column=0, sticky="w")
        self.add_tooltip(self.run_state_label, "显示当前是否可以分析、是否正在计算或需要补充设置。具体下一步操作在下方；失败后可按提示修正并重试。")
        ttk.Label(self.run_frame, text="Ctrl + Enter", style="Hint.TLabel").grid(row=0, column=1, sticky="e")
        self.workflow_hint_label = ttk.Label(self.run_frame, textvariable=self.workflow_hint_var,
                                             style="Hint.TLabel", wraplength=360, justify=tk.LEFT)
        self.workflow_hint_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        self.add_tooltip(self.workflow_hint_label, "这里告诉你下一步需要做什么。出现质量提醒时，点击“质量与日志”查看具体图片或参数需要检查的原因。")
        self.start_button = ttk.Button(self.run_frame, text="开始分析", command=self.start_processing, style="Primary.TButton")
        self.start_button.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.add_tooltip(self.start_button, "点击按当前图片范围、测量框和参数开始计算（Ctrl + Enter）。完成后显示结果图，并把选定文件保存到输出文件夹。")
        self.progress_value = tk.DoubleVar(value=0)
        self.progress_percent_var = tk.StringVar(value="0%")
        self.progress_value.trace_add("write", lambda *_args: self.progress_percent_var.set(f"{self.progress_value.get():.0f}%"))
        self.progress = ttk.Progressbar(self.run_frame, orient=tk.HORIZONTAL, mode="determinate", variable=self.progress_value)
        self.progress.grid(row=3, column=0, sticky="ew", pady=(10, 6), padx=(0, 8))
        self.add_tooltip(self.progress, "显示本次计算已处理的比例。计算后还会保存文件并更新图表，请等状态显示“处理完成”后再进行下一次分析。")
        ttk.Label(self.run_frame, textvariable=self.progress_percent_var, style="Hint.TLabel").grid(row=3, column=1, sticky="e")
        self.status_var = tk.StringVar(value="未加载图像")
        self.preview_scale_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self.run_frame, textvariable=self.status_var, style="Hint.TLabel",
                                       wraplength=360, justify=tk.LEFT)
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.add_tooltip(self.status_label, "显示最近的操作、计算进度或错误原因。需要查看更完整的记录时，点击“质量与日志”。")

    def _build_quality_section(self, parent):
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=1)
        qc_frame = ttk.Frame(parent, style="Card.TFrame", padding=(0, 2))
        qc_frame.grid(row=0, column=0, sticky="ew")
        qc_frame.columnconfigure(1, weight=1)
        ttk.Label(qc_frame, text="质量概览", style="Key.TLabel").grid(row=0, column=0, sticky="nw", padx=(0, 10))
        self.qc_overview_label = ttk.Label(qc_frame, textvariable=self.qc_overview_var,
                                           style="Hint.TLabel", wraplength=500, justify=tk.LEFT)
        self.qc_overview_label.grid(row=0, column=1, sticky="ew")
        self.add_tooltip(self.qc_overview_label, "分析后显示多少张图片、多少个测量点通过了检查。找到纹理和算出伸缩是两项检查；未通过的位置留空，数值仍需结合原图核对。")
        self.quality_notebook = ttk.Notebook(parent)
        self.quality_notebook.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.preflight_page = ttk.Frame(self.quality_notebook, style="Card.TFrame", padding=8)
        self.log_page = ttk.Frame(self.quality_notebook, style="Card.TFrame", padding=8)
        for page in (self.preflight_page, self.log_page):
            page.columnconfigure(0, weight=1)
            page.rowconfigure(0, weight=1)
        self.quality_notebook.add(self.preflight_page, text="运行前检查")
        self.quality_notebook.add(self.log_page, text="运行日志")
        self.add_tooltip(self.quality_notebook, "点击“运行前检查”查看开始分析需要补充什么；点击“运行日志”查看加载、计算、保存和错误的详细记录。两项均可滚动查看和复制。")
        self.preflight_summary_label = tk.Text(self.preflight_page, height=5, width=30, wrap=tk.WORD,
                                               font=self.ui_base_font, relief=tk.FLAT, bg=self.panel_bg,
                                               fg=self.text_color, padx=10, pady=8, state=tk.DISABLED)
        self.preflight_summary_label.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(self.preflight_page, orient=tk.VERTICAL, command=self.preflight_summary_label.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.preflight_summary_label.configure(yscrollcommand=scrollbar.set)
        self.add_tooltip(self.preflight_summary_label, "这里列出开始分析前的检查。“阻止”项需要先修正；“警告”项可继续，但需检查原因。可拖动选中文字，再按 Ctrl + C 复制。")
        self.add_tooltip(scrollbar, "上下拖动查看其他分析前检查项目和具体原因。")
        self.preflight_summary_var.trace_add("write", self._update_preflight_text)
        self.log_text = tk.Text(self.log_page, width=30, height=4, wrap=tk.WORD, bg=self.panel_bg,
                                fg=self.text_color, font=self.ui_log_font, relief=tk.FLAT,
                                padx=10, pady=8, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_scroll = ttk.Scrollbar(self.log_page, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=self.log_scroll.set)
        self.add_tooltip(self.log_text, "这里按发生顺序记录加载、计算、保存和错误信息。内容不能修改；可拖动选中文字，再按 Ctrl + C 复制，便于核查问题。")
        self.add_tooltip(self.log_scroll, "上下拖动查看更早或更新的运行记录。")
        parent.bind("<Configure>", lambda event: self._resize_quality_labels(event.width))

    def _update_progress(self, value):
        self.progress.config(value=value)
        if hasattr(self, "progress_value"):
            self.progress_value.set(value)

    def _resize_quality_labels(self, width):
        self.qc_overview_label.configure(wraplength=max(220, width - 120))

    def _update_preflight_text(self, *_args):
        text = self.preflight_summary_label
        previous_view = text.yview()[0]
        text.configure(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        text.tag_configure("block", foreground=self.warning_color)
        text.tag_configure("warn", foreground=self.key_color)
        for line in self.preflight_summary_var.get().splitlines():
            tag = "block" if line.startswith("[阻止]") else "warn" if line.startswith("[警告]") else "ok"
            text.insert(tk.END, line + "\n", (tag,))
        text.configure(state=tk.DISABLED)
        text.yview_moveto(previous_view)

    def _build_results_section(self, parent):
        self.viewer_frame = ttk.Frame(parent, style="Card.TFrame", padding=12)
        self.viewer_frame.grid(row=0, column=0, sticky="nsew")
        self.viewer_frame.columnconfigure(0, weight=1)
        self.viewer_frame.rowconfigure(2, weight=1)
        self.viewer_placeholder = ttk.Label(
            self.viewer_frame, text="尚无分析结果\n\n完成分析后，在此查看应变曲线或全场位移与应变。\n结果图支持缩放、平移和导出。",
            style="Hint.TLabel", anchor="center", justify=tk.CENTER, wraplength=440)
        self.viewer_placeholder.grid(row=2, column=0, sticky="nsew")
        viewer_btns = ttk.Frame(self.viewer_frame, style="Card.TFrame")
        viewer_btns.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        viewer_btns.columnconfigure(0, weight=1)
        ttk.Label(viewer_btns, text="分析结果", style="StepTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.viewer_export_btn = ttk.Button(viewer_btns, text="导出当前图", command=self.export_viewer_figure,
                                             style="Secondary.TButton", state=tk.DISABLED)
        self.viewer_export_btn.grid(row=0, column=1, padx=(0, 6))
        self.add_tooltip(self.viewer_export_btn, "点击选择文件名和位置，把当前显示的结果图保存为图片或 PDF 文档。图中的坐标轴、颜色刻度和当前查看范围会一起保存。")
        self.viewer_clear_btn = ttk.Button(viewer_btns, text="清除预览", command=self.clear_viewer,
                                            style="Compact.TButton", state=tk.DISABLED)
        self.viewer_clear_btn.grid(row=0, column=2)
        self.add_tooltip(self.viewer_clear_btn, "点击清空这里的结果图，图像页面恢复原图。已保存的结果文件保留；要再次显示结果，可重新分析。")
        self.viewer_content_frame = ttk.Frame(self.viewer_frame, style="Card.TFrame")
        self.viewer_content_frame.grid(row=2, column=0, sticky="nsew")
        self.viewer_content_frame.columnconfigure(0, weight=1)
        self.viewer_content_frame.rowconfigure(0, weight=1)
        self.viewer_content_frame.grid_remove()

    def sync_strain_mode_from_display(self, event=None):
        label = self.strain_mode_display.get()
        self.strain_mode.set(STRAIN_MODE_LABEL_TO_VALUE.get(label, "auto"))

    def sync_strain_mode_display(self):
        value = self.strain_mode.get()
        self.strain_mode_display.set(STRAIN_MODE_VALUE_TO_LABEL.get(value, "自动判断"))

    def sync_roi_role_from_display(self, event=None):
        label = self.roi_role_display.get()
        self.roi_role.set(ROI_ROLE_LABEL_TO_VALUE.get(label, "none"))

    def sync_roi_role_display(self):
        value = normalize_roi_role(self.roi_role.get())
        self.roi_role.set(value)
        self.roi_role_display.set(ROI_ROLE_VALUE_TO_LABEL.get(value, "普通"))

    def apply_tracking_preset(self, event=None):
        name = self.tracking_preset.get()
        preset = TRACKING_PRESETS.get(name)
        if not preset:
            return

        self._applying_preset = True
        try:
            self.search_radius.set(preset["search_radius"])
            self.hard_corr.set(preset["hard_corr"])
            self.soft_corr.set(preset["soft_corr"])
            self.max_frame_strain_jump.set(preset["max_frame_strain_jump"])
            self.fb_tolerance_px.set(preset["fb_tolerance_px"])
        finally:
            self._applying_preset = False

        self.preset_status_var.set(f"当前追踪模式：{name}")
        if hasattr(self, "status_var"):
            self.status_var.set(f"当前追踪模式：{name}")

    def mark_tracking_custom(self, *args):
        if getattr(self, "_applying_preset", False):
            return
        if not hasattr(self, "preset_status_var"):
            return
        if self.tracking_preset.get() != "自定义":
            self.tracking_preset.set("自定义")
        self.preset_status_var.set("当前追踪模式：自定义")

    def toggle_advanced_settings(self):
        if self.advanced_visible.get():
            self.advanced_frame.grid_remove()
            self.advanced_visible.set(False)
            self.advanced_toggle_btn.config(text="显示高级设置")
            self.pixel_size_label.grid_remove()
            self.pixel_size_entry.grid_remove()
            self.preset_status_label.grid_remove()
        else:
            self.advanced_frame.grid()
            self.advanced_visible.set(True)
            self.advanced_toggle_btn.config(text="隐藏高级设置")
            self.pixel_size_label.grid()
            self.pixel_size_entry.grid()
            self.preset_status_label.grid()

    def show_usage_notice(self):
        messagebox.showinfo("About / Citation / Usage Notice", USAGE_NOTICE)

    # ---------- 日志和文件 ----------

    def log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.root.update_idletasks()

    def log_user_error(self, context, exc, details=None):
        message = str(exc).strip() or exc.__class__.__name__
        self.log(f"{context}失败：{message}")
        self.log("请检查输入路径、图像文件、ROI 和参数设置；详细调试信息已输出到控制台。")
        if details:
            print(details)
        else:
            traceback.print_exc()

    def show_completion_and_open_output_folder(self, done_msg, output_dir):
        messagebox.showinfo("完成", done_msg)
        self.remember_recent_paths(
            image_dir=self.image_folder.get().strip() or None,
            output_dir=output_dir,
        )
        try:
            open_output_folder(output_dir)
        except Exception as exc:
            self.log(f"无法自动打开结果目录：{exc}")

    def clear_sequence_dependent_state(self):
        self.roi1 = None
        self.roi2 = None
        self.field_roi = None
        self.dic_mask_exclusions = []
        self._drawing_mask_exclusion = False
        self.dic_mask_path.set("")
        self.dic_mask_mode.set("矩形 ROI")
        self.roi1_reference_frame_1based = None
        self.roi2_reference_frame_1based = None
        self.field_roi_reference_frame_1based = None
        self.dic_last_field = None
        self.dic_last_image = None
        self.dic_last_frame_1based = None
        self.dic_last_filename = None
        self.dic_last_reference_frame_1based = None
        self.dic_last_reference_filename = None
        self.current_roi_index = 1
        self.drag_start = None
        self.temp_rect_id = None
        self.roi_groups.clear()
        self.next_group_idx = 1
        self.group_name_var.set("")
        self.refresh_group_tree()
        self.refresh_current_roi_summary()
        self.last_qc_summary = None
        if hasattr(self, "qc_overview_var"):
            self.qc_overview_var.set("分析完成后显示 QC 总览。")
        if hasattr(self, "clear_viewer"):
            self.clear_viewer(keep_placeholder=True)

    def select_image_folder(self):
        folder = filedialog.askdirectory(title="选择图像序列文件夹")
        if folder:
            self.image_folder.set(folder)
            if not str(self.output_folder.get() or "").strip():
                default_name = (
                    "ezDIC_fullfield_output"
                    if self.is_fullfield_mode()
                    else "ezDIC_extensometer_output"
                )
                self.output_folder.set(os.path.join(folder, default_name))
            self.remember_recent_paths(image_dir=folder, output_dir=self.output_folder.get())
            self.log(f"图像文件夹：{folder}")

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_folder.set(folder)
            self.remember_recent_paths(output_dir=folder)
            self.log(f"输出文件夹：{folder}")

    def load_first_image(self):
        """
        加载图像序列，并显示当前 preview_frame_1based 指定的帧。
        保留函数名是为了兼容前面按钮调用。
        """
        if self.is_processing or self._completion_pending:
            messagebox.showinfo("正在处理", "当前分析仍在处理或收尾，请完成后再刷新图像序列。")
            return
        folder = self.image_folder.get().strip()
        if not folder:
            messagebox.showwarning("缺少文件夹", "请先选择图像文件夹。")
            return

        new_image_paths = collect_images(folder)
        if not new_image_paths:
            messagebox.showerror("未找到图像", "该文件夹中没有找到 tif/tiff/png/jpg/bmp 图像。")
            return

        preview, preview_err = self._safe_int_var(self.preview_frame_1based)
        start, start_err = self._safe_int_var(self.start_frame_1based)
        end, end_err = self._safe_int_var(self.end_frame_1based)
        if preview_err or start_err or end_err:
            bad = []
            if preview_err:
                bad.append("预览帧")
            if start_err:
                bad.append("起始帧")
            if end_err:
                bad.append("结束帧")
            message = f"{'、'.join(bad)}必须是整数，未加载新序列。"
            messagebox.showerror("加载失败", message)
            self.log(f"加载图像序列失败：{message}")
            return

        old_state = {
            "image_paths": list(self.image_paths),
            "loaded_image_folder": self.loaded_image_folder,
            "loaded_image_sequence_fingerprint": self.loaded_image_sequence_fingerprint,
            "first_raw": self.first_raw,
            "first_img8": self.first_img8,
            "current_fullres_img8": self.current_fullres_img8,
            "display_img": self.display_img,
            "display_scale": self.display_scale,
            "photo": self.photo,
            "preview": preview,
            "start": start,
            "end": end,
            "current_preview_index": self.current_preview_index,
        }

        n = len(new_image_paths)
        folder_key = os.path.normcase(os.path.abspath(folder))
        new_fingerprint = image_sequence_fingerprint(new_image_paths)
        sequence_changed = (
            self.loaded_image_folder is not None
            and (
                self.loaded_image_folder != folder_key
                or self.loaded_image_sequence_fingerprint != new_fingerprint
            )
        )

        # 如果用户还没设置范围，默认 1 到最后一帧。
        if end <= 1:
            self.end_frame_1based.set(n)

        self.image_paths = new_image_paths
        # Preview is display-only and may be brought into range for a valid
        # first render.  Start/end remain untouched so out-of-range input is
        # surfaced by preflight instead of silently clamped.
        preview = max(1, min(preview, n))
        self.preview_frame_1based.set(preview)

        try:
            self.load_preview_frame(preview - 1)
            if sequence_changed:
                self.clear_sequence_dependent_state()
            self.loaded_image_folder = folder_key
            self.loaded_image_sequence_fingerprint = new_fingerprint
            self.show_image()
            self.log(f"找到 {n} 张图像。")
            self.log(f"当前预览：第 {self.current_preview_index + 1} 帧 / 共 {n} 帧")
            self.remember_recent_paths(image_dir=folder, output_dir=self.output_folder.get().strip() or None)
            self.update_workflow_action_states()
        except Exception as exc:
            self.image_paths = old_state["image_paths"]
            self.loaded_image_folder = old_state["loaded_image_folder"]
            self.loaded_image_sequence_fingerprint = old_state["loaded_image_sequence_fingerprint"]
            self.first_raw = old_state["first_raw"]
            self.first_img8 = old_state["first_img8"]
            self.current_fullres_img8 = old_state["current_fullres_img8"]
            self.display_img = old_state["display_img"]
            self.display_scale = old_state["display_scale"]
            self.photo = old_state["photo"]
            self.preview_frame_1based.set(old_state["preview"])
            self.start_frame_1based.set(old_state["start"])
            self.end_frame_1based.set(old_state["end"])
            self.current_preview_index = old_state["current_preview_index"]
            self.show_image()
            messagebox.showerror("加载失败", str(exc))
            self.log_user_error("加载图像序列", exc)
            self.update_workflow_action_states()

    def load_preview_frame(self, index0):
        if not self.image_paths:
            raise RuntimeError("请先加载图像序列。")

        n = len(self.image_paths)
        index0 = int(max(0, min(index0, n - 1)))

        self.current_preview_index = index0
        self._show_image_workspace()
        self.preview_frame_1based.set(index0 + 1)

        path = self.image_paths[index0]
        self.first_raw = read_gray_image(path)
        self.first_img8 = normalize_to_uint8(self.first_raw)
        self.current_fullres_img8 = self.first_img8.copy()
        self._canvas_shows_field_overlay = False

        self.display_img = None
        self.display_scale = 1.0
        self.zoom_factor = 1.0
        self.auto_fit_enabled = True
        self.root.update_idletasks()
        self._rescale_display_to_current_size()
        if self.display_img is None:
            self.display_img, self.display_scale = get_display_image(self.first_img8, max_w=1280, max_h=820)
            self.zoom_factor = self.display_scale
            self.show_image()
        self._update_zoom_label()
        self._update_image_toolbar_state()

        # 绑定一次 resize 监听（只绑一次）
        self._bind_image_resize_handler()
        try:
            self.root.after_idle(self._rescale_display_to_current_size)
        except Exception:
            pass
        start, _ = self._safe_int_var(self.start_frame_1based)
        end, _ = self._safe_int_var(self.end_frame_1based)
        range_text = f"{start}–{end}" if start is not None and end is not None else "输入待校验"
        self.status_var.set(
            f"预览第 {index0 + 1}/{n} 帧：{os.path.basename(path)} | 分析范围 {range_text}"
        )
        self._update_preview_scale_label()
        self.log(f"已显示第 {index0 + 1} 帧：{path}")
        self.log(f"图像尺寸：{self.first_img8.shape[1]} × {self.first_img8.shape[0]} px")

    def go_to_preview_frame(self):
        if not self.image_paths:
            self.load_first_image()
            return

        try:
            idx = self.get_int_setting(self.preview_frame_1based, "预览帧") - 1
            self.load_preview_frame(idx)
        except Exception as exc:
            messagebox.showerror("预览失败", str(exc))
            self.log_user_error("预览帧", exc)

    def step_preview_frame(self, step):
        if not self.image_paths:
            self.load_first_image()
            return

        try:
            self.load_preview_frame(self.current_preview_index + int(step))
        except Exception as exc:
            messagebox.showerror("预览失败", str(exc))
            self.log_user_error("预览帧", exc)

    def clear_groups_if_start_changes(self, new_start_1based):
        old_start, old_start_err = self._safe_int_var(self.start_frame_1based)
        if old_start_err is None and new_start_1based == old_start:
            return True

        has_reference_state = bool(
            self.roi_groups
            or self.roi1 is not None
            or self.roi2 is not None
            or self.field_roi is not None
        )
        if has_reference_state:
            msg = (
                f"当前已有 {len(self.roi_groups)} 组 ROI 或正在编辑的 ROI。\\n\\n"
                f"这些 ROI 应该是在当前起始/参考帧第 {old_start if old_start is not None else '未知'} 帧上定义的。\\n"
                f"如果把起始/参考帧改为第 {new_start_1based} 帧，已有 ROI 模板可能不再对应。\\n\\n"
                f"建议清空已有 ROI、结果和 QC，并在新参考帧上重画。是否清空并修改起始帧？"
            )
            if not messagebox.askyesno("修改起始/参考帧", msg):
                return False
            self.clear_sequence_dependent_state()
            self.log("由于修改起始/参考帧，已清空不兼容的 ROI、结果和 QC。")
        return True

    def set_start_to_current(self):
        if not self.image_paths:
            messagebox.showwarning("未加载图像", "请先加载图像序列。")
            return

        new_start = self.current_preview_index + 1
        if not self.clear_groups_if_start_changes(new_start):
            return

        self.start_frame_1based.set(new_start)
        end, end_err = self._safe_int_var(self.end_frame_1based)
        if end_err is not None or end < new_start:
            self.end_frame_1based.set(len(self.image_paths))
        self.log(f"已将第 {new_start} 帧设为分析起始/参考帧。请在这张图上画 ROI。")
        end, _ = self._safe_int_var(self.end_frame_1based)
        self.status_var.set(f"起始/参考帧 = {new_start}；结束帧 = {end if end is not None else '待校验'}")

    def set_end_to_current(self):
        if not self.image_paths:
            messagebox.showwarning("未加载图像", "请先加载图像序列。")
            return

        end = self.current_preview_index + 1
        self.end_frame_1based.set(end)
        self.log(f"已将第 {end} 帧设为分析结束帧。")
        start, _ = self._safe_int_var(self.start_frame_1based)
        self.status_var.set(f"起始/参考帧 = {start if start is not None else '待校验'}；结束帧 = {end}")

    def get_analysis_indices(self):
        if not self.image_paths:
            raise RuntimeError("请先加载图像序列。")

        n = len(self.image_paths)
        s = self.get_int_setting(self.start_frame_1based, "起始帧") - 1
        e = self.get_int_setting(self.end_frame_1based, "结束帧") - 1

        if s < 0 or s >= n or e < 0 or e >= n:
            raise RuntimeError(f"分析范围必须位于 1 到 {n} 帧之间：start={s + 1}, end={e + 1}")

        if e < s:
            raise RuntimeError(
                f"分析结束帧不能早于起始帧：start={s + 1}, end={e + 1}"
            )

        return s, e

    # ---------- 图像显示与 ROI 绘制 ----------

    def show_image(self):
        if self.display_img is None:
            self._draw_image_empty_state()
            return

        pil_img = Image.fromarray(self.display_img)
        self.photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        # Extending the scroll region centers small images while canvasx/y
        # continue to return image coordinates for ROI drawing and zoom.
        self._update_image_scrollregion()
        self._update_image_context()

        self.redraw_rois_and_groups()
        if (
            getattr(self, "auto_fit_enabled", True)
            and not getattr(self, "_fitting_display", False)
            and self.current_fullres_img8 is not None
        ):
            try:
                cw = int(self.canvas.winfo_width())
                ch = int(self.canvas.winfo_height())
            except Exception:
                return
            dh, dw = self.display_img.shape[:2]
            if cw > 50 and ch > 50 and (dw > cw + 1 or dh > ch + 1):
                self._fitting_display = True
                try:
                    self._rescale_display_to_current_size()
                finally:
                    self._fitting_display = False

    def _bind_image_resize_handler(self):
        """只绑定一次，监听图像区尺寸变化，实现窗口拉大后自动提高预览分辨率。"""
        if getattr(self, "_image_resize_bound", False):
            return
        self._image_resize_bound = True

        def on_image_frame_configure(event):
            if self.current_fullres_img8 is None:
                return
            if getattr(self, "_fitting_display", False):
                return
            if getattr(self, "auto_fit_enabled", True) and self.display_img is not None:
                try:
                    cw = int(self.canvas.winfo_width())
                    ch = int(self.canvas.winfo_height())
                except Exception:
                    cw, ch = 0, 0
                dh, dw = self.display_img.shape[:2]
                if cw > 50 and ch > 50 and (dw > cw + 1 or dh > ch + 1):
                    self._fitting_display = True
                    try:
                        self._rescale_display_to_current_size()
                    finally:
                        self._fitting_display = False
                    return
            # 防抖：用户拖拽窗口时不要狂刷
            if self._resize_after_id:
                try:
                    self.root.after_cancel(self._resize_after_id)
                except Exception:
                    pass
            self._resize_after_id = self.root.after(180, self._rescale_display_to_current_size)

        # 绑在 image_frame 上更稳（它会随窗口变化）
        self.image_frame.bind("<Configure>", on_image_frame_configure, add="+")
        self.canvas.bind("<Configure>", on_image_frame_configure, add="+")

    def _rescale_display_to_current_size(self):
        """根据当前图像区可用空间重新计算显示图像（支持窗口拉大获得更高细节）。"""
        if self.current_fullres_img8 is None:
            return

        # 如果用户正在手动缩放，则不要自动覆盖
        if not getattr(self, "auto_fit_enabled", True):
            return

        try:
            self.root.update_idletasks()
            cw = max(200, self.canvas.winfo_width())
            ch = max(150, self.canvas.winfo_height())
        except Exception:
            cw, ch = 900, 620

        target_w = max(200, cw - 8)
        target_h = max(150, ch - 8)

        new_disp, new_scale = get_display_image(self.current_fullres_img8, max_w=target_w, max_h=target_h)

        old_h, old_w = (self.display_img.shape[:2] if self.display_img is not None else (0, 0))
        if self.display_img is None or abs(new_disp.shape[0] - old_h) > 2 or abs(new_disp.shape[1] - old_w) > 2:
            self.display_img = new_disp
            self.display_scale = new_scale
            self.zoom_factor = new_scale   # 同步 zoom_factor
            self.show_image()
            self._update_preview_scale_label()
            self._update_zoom_label()

            if not self._has_shown_resize_hint:
                self._has_shown_resize_hint = True
                self.log("提示：拉大窗口可自动提高预览分辨率，便于精细绘制 ROI。")

    def _update_preview_scale_label(self):
        """在状态区显示当前预览图像的缩放比例（对科研判断细节很有用）。"""
        if self.display_scale is None or self.display_img is None:
            self.preview_scale_var.set("")
            return
        try:
            scale_pct = self.display_scale * 100
            if scale_pct >= 99.5:
                text = "预览：原始分辨率"
            else:
                text = f"预览缩放：{scale_pct:.0f}%"
            self.preview_scale_var.set(text)
        except Exception:
            self.preview_scale_var.set("")

    # ========== 图像工具栏相关方法 ==========

    # ========== 图像缩放核心逻辑 ==========

    def zoom_image(self, factor):
        """按倍率手动缩放（支持工具栏 +/- 按钮）。"""
        if self.current_fullres_img8 is None:
            return

        self.auto_fit_enabled = False
        new_factor = max(0.1, min(8.0, self.zoom_factor * factor))
        self.zoom_factor = new_factor

        self._apply_manual_zoom()

    def _apply_manual_zoom(self):
        """根据当前 zoom_factor 重新生成显示图像。"""
        if self.current_fullres_img8 is None:
            return

        h, w = self.current_fullres_img8.shape[:2]
        target_w = int(w * self.zoom_factor)
        target_h = int(h * self.zoom_factor)

        if target_w < 50 or target_h < 50:
            target_w, target_h = 50, 50

        disp = cv2.resize(self.current_fullres_img8, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        self._canvas_shows_field_overlay = False

        self.display_img = rgb
        self.display_scale = self.zoom_factor
        self.show_image()
        self._update_preview_scale_label()
        self._update_zoom_label()

    def fit_image_to_view(self):
        """强制将当前图像适应当前图像区大小，并恢复自动适应模式。"""
        if self.current_fullres_img8 is None:
            return
        self.auto_fit_enabled = True
        self._canvas_shows_field_overlay = False
        self.display_img = None
        self._rescale_display_to_current_size()
        self._update_preview_scale_label()
        self._update_zoom_label()

    def show_image_1to1(self):
        """以 1:1 原始像素比例显示，并进入手动缩放模式。"""
        if self.current_fullres_img8 is None:
            return
        self.auto_fit_enabled = False
        self.zoom_factor = 1.0

        h, w = self.current_fullres_img8.shape[:2]
        rgb = cv2.cvtColor(self.current_fullres_img8, cv2.COLOR_GRAY2RGB)
        self._canvas_shows_field_overlay = False
        self.display_img = rgb
        self.display_scale = 1.0
        self.show_image()
        self._update_preview_scale_label()
        self._update_zoom_label()
        self.log("已切换为 1:1 原始比例显示。")

    def _update_zoom_label(self):
        """更新工具栏上的缩放百分比显示。"""
        if hasattr(self, "zoom_label_var"):
            pct = int(round(getattr(self, "display_scale", 1.0) * 100))
            self.zoom_label_var.set(f"{pct}%")
        self._update_image_toolbar_state()

    def _update_image_toolbar_state(self):
        """根据是否有图像 + 是否正在处理，控制工具栏按钮状态。"""
        has_image = self.current_fullres_img8 is not None
        can_use = has_image and not getattr(self, "is_processing", False)
        for attr in ("btn_zoom_in", "btn_zoom_out", "btn_fit", "btn_1to1"):
            if hasattr(self, attr):
                try:
                    getattr(self, attr).config(state=tk.NORMAL if can_use else tk.DISABLED)
                except Exception:
                    pass

    def update_workflow_action_states(self):
        """根据当前工作流进度控制关键操作按钮，避免用户在不可运行状态下误点。"""
        has_sequence = bool(self.image_paths) and self.first_img8 is not None
        has_groups = bool(self.roi_groups)
        has_field_roi = self.field_roi is not None
        has_current_roi = self.field_roi is not None if self.is_fullfield_mode() else (self.roi1 is not None or self.roi2 is not None)
        is_processing = getattr(self, "is_processing", False)
        completion_pending = getattr(self, "_completion_pending", False)
        is_ff = self.is_fullfield_mode()
        preflight_items = self.refresh_preflight_panel() if hasattr(self, "preflight_summary_var") else []
        blocking_items = [item for item in preflight_items if item["level"] == "block"]
        warning_items = [item for item in preflight_items if item["level"] == "warn"]
        busy = is_processing or completion_pending
        self._set_run_controls_locked(busy)
        self.refresh_current_roi_summary()
        if hasattr(self, "dic_field_summary_var"):
            roi = self.field_roi
            if roi is None:
                self.dic_field_summary_var.set("尚未绘制全场 ROI。")
            else:
                x, y, w, h = roi
                self.dic_field_summary_var.set(f"ROI {w}×{h} px @ ({x},{y})")

        can_switch_mode = not is_processing and not completion_pending
        for attr in ("mode_extensometer_radio", "mode_fullfield_radio"):
            if hasattr(self, attr):
                getattr(self, attr).config(state=tk.NORMAL if can_switch_mode else tk.DISABLED)

        if hasattr(self, "start_button"):
            ready = has_field_roi if is_ff else has_groups
            can_start = has_sequence and ready and not blocking_items and not is_processing and not completion_pending
            self.start_button.config(state=tk.NORMAL if can_start else tk.DISABLED)
            self.start_button.configure(text="正在分析…" if is_processing else "正在收尾…" if completion_pending else "开始分析")

        has_pair = self.roi1 is not None and self.roi2 is not None
        selected_group = bool(self.group_tree.selection()) if hasattr(self, "group_tree") else has_groups
        for attr in ("roi1_button", "roi2_button", "draw_field_roi_button", "show_preview_button", "set_start_button", "set_end_button"):
            if hasattr(self, attr):
                getattr(self, attr).configure(state=tk.NORMAL if has_sequence and not busy else tk.DISABLED)
        for attr in ("align_x_button", "align_y_button", "add_group_button"):
            if hasattr(self, attr):
                getattr(self, attr).configure(state=tk.NORMAL if has_sequence and has_pair and not busy else tk.DISABLED)
        for attr, idx in (("roi1_button", 1), ("roi2_button", 2)):
            if hasattr(self, attr):
                getattr(self, attr).configure(style="Secondary.TButton" if self.current_roi_index == idx else "Compact.TButton")
        if hasattr(self, "prev_frame_button"):
            self.prev_frame_button.configure(state=tk.NORMAL if has_sequence and self.current_preview_index > 0 and not busy else tk.DISABLED)
            self.next_frame_button.configure(state=tk.NORMAL if has_sequence and self.current_preview_index < len(self.image_paths) - 1 and not busy else tk.DISABLED)

        if hasattr(self, "clear_rois_button"):
            can_clear = has_current_roi and not busy
            self.clear_rois_button.config(state=tk.NORMAL if can_clear else tk.DISABLED)

        can_manage_groups = has_groups and selected_group and not busy
        for attr in ("load_group_button", "update_group_button", "delete_group_button"):
            if hasattr(self, attr):
                can_use = can_manage_groups and (has_pair if attr == "update_group_button" else True)
                getattr(self, attr).config(state=tk.NORMAL if can_use else tk.DISABLED)

        if hasattr(self, "workflow_hint_var"):
            if is_processing:
                hint = "正在计算；可在“质量与日志”查看详情。"
            elif completion_pending:
                hint = "正在保存结果并更新预览，请稍候。"
            elif blocking_items:
                first = blocking_items[0]
                hint = f"{first['label']}：{first['message']}"
            elif not has_sequence:
                hint = "请选择图像文件夹并加载序列。"
            elif is_ff and not has_field_roi:
                hint = "请在参考帧绘制全场 ROI。"
            elif not is_ff and not has_groups:
                hint = "请绘制 ROI 1 和 ROI 2，并添加 ROI 组。"
            elif warning_items:
                hint = f"可分析；请复核{warning_items[0]['label']}。详情见“质量与日志”。"
            else:
                hint = "准备就绪，分析后自动显示结果。"
            self.workflow_hint_var.set(hint)

        if hasattr(self, "run_state_var"):
            failed = self.status_var.get().startswith("分析失败")
            if is_processing:
                state = "分析中"
            elif completion_pending:
                state = "正在收尾"
            elif not has_sequence:
                state = "待加载"
            elif failed:
                state = "分析失败 · 可重试"
            elif blocking_items:
                state = "待设置"
            elif warning_items:
                state = "可分析 · 有提示"
            else:
                state = "准备就绪"
            self.run_state_var.set(state)
            style = "Error.TLabel" if failed and not busy else "Ready.TLabel" if not busy and has_sequence and not blocking_items else "Badge.TLabel"
            self.run_state_label.configure(style=style)
        self._update_image_context()

        self._update_image_toolbar_state()

    def _set_run_controls_locked(self, locked):
        """Freeze editable inputs while an immutable run snapshot is executing."""
        if not hasattr(self, "controls_panel"):
            return
        previous = getattr(self, "_run_controls_locked", False)
        if previous == locked:
            return
        self._run_controls_locked = locked
        if locked:
            self._run_control_states = []
            def freeze(widget):
                if isinstance(widget, (ttk.Entry, ttk.Combobox, ttk.Button, ttk.Checkbutton, ttk.Radiobutton, ttk.Treeview)):
                    self._run_control_states.append((widget, widget.instate(["disabled"])))
                    widget.state(["disabled"])
                for child in widget.winfo_children():
                    freeze(child)
            freeze(self.project_frame)
            freeze(self.controls_panel)
        else:
            for widget, was_disabled in self._run_control_states:
                if widget.winfo_exists():
                    widget.state(["disabled"] if was_disabled else ["!disabled"])
            self._run_control_states = []

    def _restore_sequence_preview(self):
        """Rebuild the image canvas from the loaded frame, dropping a stale overlay."""
        if self.current_fullres_img8 is None:
            self._canvas_shows_field_overlay = False
            return
        self.auto_fit_enabled = True
        self._canvas_shows_field_overlay = False
        self.display_img = None
        if hasattr(self, "_rescale_display_to_current_size"):
            self._rescale_display_to_current_size()
        if self.display_img is None:
            self.display_img, self.display_scale = get_display_image(
                self.current_fullres_img8, max_w=1280, max_h=820
            )
            self.zoom_factor = self.display_scale
            self.show_image()

    def _on_mouse_wheel(self, event):
        """支持鼠标滚轮缩放，围绕鼠标指针位置进行（专业图像工具标准行为）。"""
        if self.current_fullres_img8 is None:
            return

        # 确定缩放方向（跨平台）
        if event.num == 4 or event.delta > 0:
            factor = 1.2
        elif event.num == 5 or event.delta < 0:
            factor = 1 / 1.2
        else:
            return

        self.auto_fit_enabled = False

        # 计算鼠标在当前显示图上的位置。canvasx/canvasy 会考虑当前滚动偏移。
        view_x, view_y = event.x, event.y
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # 当前显示尺寸
        if self.display_img is None:
            return
        disp_h, disp_w = self.display_img.shape[:2]

        # 鼠标在原始图像坐标中的位置（使用当前 display_scale）
        inv_scale = 1.0 / self.display_scale
        img_x = cx * inv_scale
        img_y = cy * inv_scale

        # 应用新的缩放因子
        new_zoom = max(0.05, min(10.0, self.zoom_factor * factor))
        self.zoom_factor = new_zoom

        # 重新计算目标显示尺寸
        orig_h, orig_w = self.current_fullres_img8.shape[:2]
        new_disp_w = int(orig_w * self.zoom_factor)
        new_disp_h = int(orig_h * self.zoom_factor)

        if new_disp_w < 30 or new_disp_h < 30:
            return

        disp = cv2.resize(self.current_fullres_img8, (new_disp_w, new_disp_h), interpolation=cv2.INTER_AREA)
        self.display_img = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        self.display_scale = self.zoom_factor
        self._canvas_shows_field_overlay = False

        self.show_image()
        self._update_preview_scale_label()
        self._update_zoom_label()

        # 尝试保持鼠标指向的原始位置在缩放后仍大致在鼠标附近（简单版本）
        # 计算新位置
        new_cx = img_x * self.zoom_factor
        new_cy = img_y * self.zoom_factor

        # 将画布滚动到使 (new_cx, new_cy) 接近原鼠标位置
        try:
            self.canvas.xview_moveto(max(0, min(1, (new_cx - view_x) / new_disp_w)))
            self.canvas.yview_moveto(max(0, min(1, (new_cy - view_y) / new_disp_h)))
        except Exception:
            pass

    def set_roi_mode(self, idx):
        if self.is_processing or self._completion_pending:
            return
        self._drawing_mask_exclusion = False
        self._show_image_workspace()
        if self._canvas_shows_field_overlay:
            self._restore_sequence_preview()
        self.current_roi_index = idx
        label = "全场 ROI" if self.is_fullfield_mode() else f"ROI {idx}"
        self.status_var.set(f"当前绘制：{label}。请在参考帧上拖出矩形。")
        self.update_workflow_action_states()
        self.log(f"切换到绘制 ROI {idx}")

    def clear_current_rois(self):
        if getattr(self, "is_processing", False) or getattr(self, "_completion_pending", False):
            self.status_var.set("正在处理：请等待当前任务完成后再清除 ROI。")
            self.log("正在处理，已忽略清除当前 ROI 请求。")
            return

        has_current_roi = self.field_roi is not None if self.is_fullfield_mode() else (self.roi1 is not None or self.roi2 is not None)
        if not has_current_roi:
            self.status_var.set("当前没有正在编辑的 ROI。")
            self.log("当前没有正在编辑的 ROI，无需清除。")
            return

        if not messagebox.askyesno(
            "清除当前 ROI",
            (
                "确定清除当前正在编辑的全场 ROI 吗？\n\n"
                if self.is_fullfield_mode()
                else "确定清除当前正在编辑的 ROI1/ROI2 吗？\n\n"
            )
            + "已添加到 ROI 组列表中的结果不会被删除；如需删除已保存的组，请使用“删除选中组”。",
        ):
            self.log("已取消清除当前 ROI。")
            return

        if self.is_fullfield_mode():
            self.field_roi = None
            self.field_roi_reference_frame_1based = None
        else:
            self.roi1 = None
            self.roi2 = None
            self.roi1_reference_frame_1based = None
            self.roi2_reference_frame_1based = None
        self.show_image()
        self.log("已清除当前 ROI。")
        self.update_workflow_action_states()

    def on_mouse_down(self, event):
        if self.first_img8 is None:
            return
        if getattr(self, "is_processing", False) or getattr(self, "_completion_pending", False):
            return
        if getattr(self, "_canvas_shows_field_overlay", False):
            self.status_var.set("当前画布是全场结果叠加图。请先显示预览/参考帧再画 ROI。")
            self.log("已忽略在结果叠加图上的 ROI 绘制；请先恢复预览帧。")
            return

        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        dh, dw = self.display_img.shape[:2]
        if not (0 <= x < dw and 0 <= y < dh):
            return
        self.drag_start = (x, y)
        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None

    def on_mouse_drag(self, event):
        if self.drag_start is None:
            return

        x0, y0 = self.drag_start
        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)

        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)

        color = "red" if self.current_roi_index == 1 else "cyan"
        self.temp_rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)

    def on_mouse_up(self, event):
        if self.first_img8 is None or self.drag_start is None:
            return
        if getattr(self, "_canvas_shows_field_overlay", False):
            if self.temp_rect_id is not None:
                self.canvas.delete(self.temp_rect_id)
                self.temp_rect_id = None
            self.drag_start = None
            self.status_var.set("当前画布是全场结果叠加图。请先显示预览/参考帧再画 ROI。")
            return

        x0, y0 = self.drag_start
        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)
        self.drag_start = None

        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None

        inv = 1.0 / self.display_scale
        rx, ry, rw, rh = rect_normalize(x0 * inv, y0 * inv, x1 * inv, y1 * inv)
        if self.is_fullfield_mode() and self._drawing_mask_exclusion:
            self._drawing_mask_exclusion = False
            if rw >= 1 and rh >= 1:
                self.dic_mask_exclusions.append(clamp_rect((rx,ry,rw,rh), self.first_img8.shape))
                self.preview_dic_mask()
                self.update_workflow_action_states()
            return

        if rw < 15 or rh < 15:
            messagebox.showwarning("ROI 太小", "ROI 太小。建议至少 30×30 px，且包含清晰散斑纹理。")
            return

        rect = clamp_rect((rx, ry, rw, rh), self.first_img8.shape)
        reference_frame_1based = self.current_preview_index + 1

        if self.is_fullfield_mode():
            self.field_roi = rect
            self.dic_mask_exclusions = []
            self.field_roi_reference_frame_1based = reference_frame_1based
            self.log(f"全场 ROI = {rect}")
            self.log_roi_texture("全场 ROI", rect)
            self.status_var.set("全场 ROI 已设置。可开始 2D DIC 分析。")
            self.show_image()
            self.update_workflow_action_states()
            return

        if self.current_roi_index == 1:
            self.roi1 = rect
            self.roi1_reference_frame_1based = reference_frame_1based
            self.log(f"当前 ROI 1 = {rect}")
            self.log_roi_texture("当前 ROI 1", rect)
            self.current_roi_index = 2
            self.status_var.set("ROI 1 已设置。现在请绘制 ROI 2。")
        else:
            self.roi2 = rect
            self.roi2_reference_frame_1based = reference_frame_1based
            if self.auto_align_roi2.get() and self.roi1 is not None:
                axis = self.auto_choose_alignment_axis()
                self.align_current_pair(axis, set_mode=False)
                self.log(f"ROI2 已按 {axis} 方向自动对齐。")
            else:
                self.log(f"当前 ROI 2 = {rect}")
                self.log_roi_texture("当前 ROI 2", rect)
            self.status_var.set("ROI 2 已设置。可以添加为一组。")

        self.show_image()
        self.update_workflow_action_states()

    def redraw_rois_and_groups(self):
        if self.display_img is None:
            return

        s = self.display_scale

        field_roi = self.field_roi if self.is_fullfield_mode() else None
        if field_roi is not None and self.is_fullfield_mode():
            x, y, w, h = field_roi
            self.canvas.create_rectangle(
                x * s, y * s, (x + w) * s, (y + h) * s, outline="#38bdf8", width=2
            )
            self._draw_roi_label(x * s + 4, y * s - 24, "DIC ROI", "#38bdf8")

        # 1-D groups are intentionally invisible in full-field mode.
        if not self.is_fullfield_mode():
            for idx, group in enumerate(self.roi_groups, start=1):
                r1 = group["roi1"]
                r2 = group["roi2"]
                c1 = rect_center(r1)
                c2 = rect_center(r2)
                self.canvas.create_line(c1[0]*s, c1[1]*s, c2[0]*s, c2[1]*s, fill="lime", width=2)
                self._draw_roi_label(c1[0] * s + 4, c1[1] * s + 4, group["name"], "#4ade80")

        def draw(rect, color, label):
            if rect is None:
                return

            x, y, w, h = rect
            x1 = x * s
            y1 = y * s
            x2 = (x + w) * s
            y2 = (y + h) * s

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self._draw_roi_label(x1 + 4, y1 - 24, label, color)

        if not self.is_fullfield_mode():
            draw(self.roi1, "#ff6b6b", "ROI 1")
            draw(self.roi2, "#38bdf8", "ROI 2")

    def _draw_roi_label(self, x, y, text, color):
        label = self.canvas.create_text(x, max(0, y), text=text, anchor="nw", fill="#ffffff",
                                        font=self.canvas_group_font)
        x0, y0, x1, y1 = self.canvas.bbox(label)
        background = self.canvas.create_rectangle(x0 - 3, y0 - 2, x1 + 3, y1 + 2,
                                                   fill="#111827", outline=color, width=1)
        self.canvas.tag_lower(background, label)

    # ---------- 对齐与组管理 ----------

    def auto_choose_alignment_axis(self):
        """
        返回 'x' 或 'y'。
        x 表示水平对齐，使用 x 方向标距；
        y 表示垂直对齐，使用 y 方向标距。
        """
        selected = self.strain_mode.get()
        if selected in ("x", "y"):
            return selected

        if self.roi1 is None or self.roi2 is None:
            return "x"

        dx, dy, _ = roi_separation(self.roi1, self.roi2)
        return "x" if dx >= dy else "y"

    def align_current_pair(self, axis, set_mode=True):
        if self.first_img8 is None:
            messagebox.showwarning("未加载图像", "请先加载第一张图。")
            return
        if self.roi1 is None or self.roi2 is None:
            messagebox.showwarning("缺少 ROI", "请先画 ROI 1 和 ROI 2。")
            return

        c1x, c1y = rect_center(self.roi1)

        if axis == "x":
            self.roi2 = move_rect_center(self.roi2, new_cy=c1y, img_shape=self.first_img8.shape)
            if set_mode:
                self.strain_mode.set("x")
                self.sync_strain_mode_display()
            self.log("已水平对齐：ROI2 中心 y 已强制等于 ROI1 中心 y；建议使用横向应变。")
        elif axis == "y":
            self.roi2 = move_rect_center(self.roi2, new_cx=c1x, img_shape=self.first_img8.shape)
            if set_mode:
                self.strain_mode.set("y")
                self.sync_strain_mode_display()
            self.log("已垂直对齐：ROI2 中心 x 已强制等于 ROI1 中心 x；建议使用纵向应变。")
        else:
            raise ValueError("axis must be x or y")

        dx, dy, dist = roi_separation(self.roi1, self.roi2)
        self.log(f"对齐后距离：dx={dx:.3f}px, dy={dy:.3f}px, distance={dist:.3f}px")
        self.log_roi_texture("当前 ROI 1", self.roi1)
        self.log_roi_texture("当前 ROI 2", self.roi2)
        self.show_image()

    def make_group_from_current(self, name=None):
        if self.first_img8 is None:
            raise RuntimeError("请先加载第一张图。")
        if self.roi1 is None or self.roi2 is None:
            raise RuntimeError("请先画 ROI 1 和 ROI 2。")

        # 安全限制：ROI 必须在分析起始/参考帧上定义。
        start_1based = self.get_int_setting(self.start_frame_1based, "起始帧")
        current_1based = int(self.current_preview_index + 1)
        if current_1based != start_1based:
            msg = (
                f"当前 ROI 是在第 {current_1based} 帧上画的，"
                f"但分析起始/参考帧是第 {start_1based} 帧。\n\n"
                f"为了保证模板和初始 ROI 一致，请先显示第 {start_1based} 帧再画 ROI，"
                f"或者点击“设为起始/参考”。"
            )
            raise RuntimeError(msg)
        for label, reference_frame in (
            ("ROI1", self.roi1_reference_frame_1based),
            ("ROI2", self.roi2_reference_frame_1based),
        ):
            if reference_frame is not None and int(reference_frame) != start_1based:
                raise RuntimeError(
                    f"{label} 在第 {reference_frame} 帧定义，但当前起始/参考帧为第 {start_1based} 帧，请重画。"
                )

        selected = self.strain_mode.get()
        actual = resolve_strain_mode(self.roi1, self.roi2, selected)
        dx, dy, dist = roi_separation(self.roi1, self.roi2)
        L0 = length_between(self.roi1, self.roi2, actual)

        if name is None or not str(name).strip():
            name = f"G{self.next_group_idx:02d}"
            self.next_group_idx += 1

        name = safe_name(name)

        group = {
            "name": name,
            "roi1": tuple(self.roi1),
            "roi2": tuple(self.roi2),
            "role": normalize_roi_role(self.roi_role.get()),
            "selected_mode": selected,
            "actual_mode": actual,
            "dx0": dx,
            "dy0": dy,
            "dist0": dist,
            "L0": L0,
            "reference_frame_1based": current_1based,
            "roi1_reference_frame_1based": self.roi1_reference_frame_1based or current_1based,
            "roi2_reference_frame_1based": self.roi2_reference_frame_1based or current_1based,
        }

        return group

    def check_group_warnings(self, group):
        messages = []

        if group["L0"] <= 0 or not np.isfinite(group["L0"]):
            messages.append(
                "L0 无效（≤0），两个 ROI 中心重合或计算错误，无法计算应变。请重新绘制。"
            )
        elif group["L0"] < 50:
            messages.append(
                f"L0 偏小：{group['L0']:.1f}px，线性应变误差会被放大。"
            )

        m1 = roi_texture_metrics(self.first_img8, group["roi1"])
        m2 = roi_texture_metrics(self.first_img8, group["roi2"])

        ok1 = texture_is_ok(
            m1,
            self.min_texture_std.get(),
            self.min_texture_contrast.get(),
            self.max_saturated_frac.get(),
        )
        ok2 = texture_is_ok(
            m2,
            self.min_texture_std.get(),
            self.min_texture_contrast.get(),
            self.max_saturated_frac.get(),
        )

        if not ok1:
            messages.append(
                f"ROI1 质量：偏弱，建议增大 ROI 或选择更独特散斑。std={m1['std_gray']:.1f}, P95-P5={m1['contrast_p95_p5']:.1f}。"
            )
        if not ok2:
            messages.append(
                f"ROI2 质量：偏弱，建议增大 ROI 或选择更独特散斑。std={m2['std_gray']:.1f}, P95-P5={m2['contrast_p95_p5']:.1f}。"
            )

        return messages

    def add_current_group(self):
        try:
            group = self.make_group_from_current(self.group_name_var.get())
        except Exception as exc:
            messagebox.showerror("无法添加 ROI 组", str(exc))
            return

        if any(g["name"] == group["name"] for g in self.roi_groups):
            messagebox.showerror("组名重复", f"已经存在组名：{group['name']}")
            return

        warnings = self.check_group_warnings(group)
        if warnings:
            text = "\n".join(warnings) + "\n\n是否仍然添加该组？"
            if not messagebox.askyesno("ROI 组警告", text):
                return

        self.roi_groups.append(group)
        self.refresh_group_tree()
        self.log_group_info("已添加", group)
        self.group_name_var.set("")
        self.show_image()

    def update_selected_group(self):
        selected_iid = self.get_selected_group_iid()
        if selected_iid is None:
            messagebox.showwarning("未选择组", "请先在列表中选择一组。")
            return

        try:
            new_group = self.make_group_from_current(self.roi_groups[selected_iid]["name"])
        except Exception as exc:
            messagebox.showerror("无法更新 ROI 组", str(exc))
            return

        warnings = self.check_group_warnings(new_group)
        if warnings:
            text = "\n".join(warnings) + "\n\n是否仍然更新该组？"
            if not messagebox.askyesno("ROI 组警告", text):
                return

        self.roi_groups[selected_iid] = new_group
        self.refresh_group_tree()
        self.log_group_info("已更新", new_group)
        self.show_image()

    def get_selected_group_iid(self):
        sel = self.group_tree.selection()
        if not sel:
            return None
        try:
            return int(sel[0])
        except Exception:
            return None

    def load_selected_group(self):
        idx = self.get_selected_group_iid()
        if idx is None:
            messagebox.showwarning("未选择组", "请先在列表中选择一组。")
            return
        if idx < 0 or idx >= len(self.roi_groups):
            return

        group = self.roi_groups[idx]
        self.roi1 = tuple(group["roi1"])
        self.roi2 = tuple(group["roi2"])
        self.roi1_reference_frame_1based = group.get(
            "roi1_reference_frame_1based", group.get("reference_frame_1based")
        )
        self.roi2_reference_frame_1based = group.get(
            "roi2_reference_frame_1based", group.get("reference_frame_1based")
        )
        self.strain_mode.set(group["selected_mode"])
        self.sync_strain_mode_display()
        self.roi_role.set(normalize_roi_role(group.get("role", "none")))
        self.sync_roi_role_display()
        self.group_name_var.set(group["name"])
        self.log_group_info("已载入", group)
        self.show_image()
        self.update_workflow_action_states()

    def delete_selected_group(self):
        idx = self.get_selected_group_iid()
        if idx is None:
            messagebox.showwarning("未选择组", "请先在列表中选择一组。")
            return
        if idx < 0 or idx >= len(self.roi_groups):
            return

        group = self.roi_groups[idx]
        if not messagebox.askyesno("删除 ROI 组", f"确定删除 {group['name']} 吗？"):
            return

        del self.roi_groups[idx]
        self.refresh_group_tree()
        self.log(f"已删除组：{group['name']}")
        self.show_image()

    def _show_group_tree_context_menu(self, event):
        """为 ROI 组列表提供右键快捷菜单，提升多组操作效率。"""
        iid = self.group_tree.identify_row(event.y)
        if not iid:
            return
        try:
            idx = int(iid)
        except ValueError:
            return
        if idx < 0 or idx >= len(self.roi_groups):
            return

        # 临时选中该行
        self.group_tree.selection_set(iid)
        self.update_workflow_action_states()

        menu = tk.Menu(self.group_tree, tearoff=0)
        menu.add_command(label="载入选中组", command=self.load_selected_group, state=self.load_group_button.cget("state"))
        menu.add_command(label="更新选中组", command=self.update_selected_group, state=self.update_group_button.cget("state"))
        menu.add_separator()
        menu.add_command(label="删除选中组", command=self.delete_selected_group, state=self.delete_group_button.cget("state"))
        actions = {0: self.load_group_button, 1: self.update_group_button, 3: self.delete_group_button}
        self.add_tooltip(menu, lambda _event: actions[menu.index("active")]._tooltip.resolve_text()
                         if menu.index("active") in actions else "")
        menu.bind("<<MenuSelect>>", menu._tooltip.schedule, add="+")
        menu.bind("<Unmap>", lambda _event: self.root.after_idle(menu.destroy), add="+")
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def refresh_group_tree(self):
        self.group_tree.delete(*self.group_tree.get_children())
        for idx, g in enumerate(self.roi_groups):
            selected_label = STRAIN_MODE_VALUE_TO_LABEL.get(g.get("selected_mode"), g.get("selected_mode", ""))
            actual_label = STRAIN_MODE_VALUE_TO_LABEL.get(g.get("actual_mode"), g.get("actual_mode", ""))
            values = (
                g["name"],
                ROI_ROLE_VALUE_TO_LABEL.get(normalize_roi_role(g.get("role", "none")), "普通"),
                selected_label,
                actual_label,
                f"{g['L0']:.1f} px",
                f"{g['dx0']:.1f} px",
                f"{g['dy0']:.1f} px",
                self._format_roi_for_tree(g["roi1"]),
                self._format_roi_for_tree(g["roi2"]),
            )
            self.group_tree.insert("", "end", iid=str(idx), values=values)
        self.update_workflow_action_states()

    @staticmethod
    def _format_roi_for_tree(rect):
        x, y, w, h = (int(round(value)) for value in rect)
        return f"({x}, {y}, {w}, {h}) px"

    def log_group_info(self, prefix, group):
        self.log(
            f"{prefix}组 {group['name']}: role={normalize_roi_role(group.get('role', 'none'))}, "
            f"selected={group['selected_mode']}, "
            f"actual={group['actual_mode']}, L0={group['L0']:.3f}px, "
            f"dx={group['dx0']:.3f}px, dy={group['dy0']:.3f}px, "
            f"ROI1={group['roi1']}, ROI2={group['roi2']}"
        )

    def log_roi_texture(self, name, rect):
        if self.first_img8 is None or rect is None:
            return

        metrics = roi_texture_metrics(self.first_img8, rect)
        ok = texture_is_ok(
            metrics,
            self.min_texture_std.get(),
            self.min_texture_contrast.get(),
            self.max_saturated_frac.get(),
        )

        quality = "良好" if ok else "偏弱，建议增大 ROI 或选择更独特散斑"
        msg = (
            f"{name} 质量：{quality}。"
            f"std={metrics['std_gray']:.2f}, P95-P5={metrics['contrast_p95_p5']:.2f}, "
            f"low={metrics['low_frac']:.3f}, high={metrics['high_frac']:.3f}"
        )
        self.log(msg)

    def _core_texture_metrics_and_code(self, rect):
        """Run the shared texture contract for a GUI preflight ROI."""
        if self.first_img8 is None or rect is None:
            return None, "INVALID_TEXTURE"
        metrics = roi_texture_metrics(self.first_img8, rect)
        code = texture_failure_code(
            metrics,
            self.min_texture_std.get(),
            self.min_texture_contrast.get(),
            self.max_saturated_frac.get(),
            _core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
        )
        return metrics, code

    def _raise_on_ambiguous_texture(self, rois):
        """Block a scientific run for an ambiguous near-1D texture only."""
        for label, rect in rois:
            _metrics, code = self._core_texture_metrics_and_code(rect)
            if code == "AMBIGUOUS_TEXTURE":
                raise RuntimeError(f"AMBIGUOUS_TEXTURE：{label} 的二维纹理不可辨识，请更换或增大 ROI。")

    # ---------- 处理前检查 ----------

    def validate_before_processing(self):
        if not self.image_paths:
            raise RuntimeError("请先加载第一张图。")
        if self.is_fullfield_mode():
            return self.validate_fullfield_before_processing()
        if not self.roi_groups:
            raise RuntimeError("请先至少添加一组 ROI。")
        self._raise_on_ambiguous_texture(
            [
                (f"{group['name']} ROI1", group.get("roi1"))
                for group in self.roi_groups
            ]
            + [
                (f"{group['name']} ROI2", group.get("roi2"))
                for group in self.roi_groups
            ]
        )
        if not self.output_folder.get().strip():
            raise RuntimeError("请设置输出文件夹。")
        output_path = Path(self.output_folder.get().strip())
        if output_path.exists() and not output_path.is_dir():
            raise RuntimeError(f"输出路径已存在但不是文件夹：{output_path}")
        if not any(
            var.get()
            for var in [
                self.export_origin_txt,
                self.export_origin_opju,
                self.export_engineering_png,
                self.export_publication_figures,
                self.export_qc_summary,
                self.export_full_csv,
                self.export_corr_plot,
                self.export_overlays,
                self.export_parameters,
            ]
        ):
            raise RuntimeError("请至少选择一种导出内容。")

        start_idx, end_idx = self.get_analysis_indices()
        if end_idx <= start_idx:
            raise RuntimeError("分析范围至少应包含两帧。")

        search_radius = self.get_int_setting(self.search_radius, "搜索半径")
        overlay_every = self.get_int_setting(self.overlay_every, "overlay 保存间隔")
        hard_corr = self.get_float_setting(self.hard_corr, "硬相关阈值")
        soft_corr = self.get_float_setting(self.soft_corr, "软相关下限")
        template_alpha = self.get_float_setting(self.template_alpha, "模板跟随系数")
        fb_tolerance = self.get_float_setting(self.fb_tolerance_px, "FB 容差")
        max_saturated_frac = self.get_float_setting(self.max_saturated_frac, "最大近黑/近白比例")

        if search_radius <= 0:
            raise RuntimeError("搜索半径必须 > 0。")
        if self.export_overlays.get() and overlay_every <= 0:
            raise RuntimeError("overlay 保存间隔必须 > 0。")
        if not (-1 <= hard_corr <= 1):
            raise RuntimeError("硬相关阈值应在 -1 到 1 之间。")
        if not (-1 <= soft_corr <= 1):
            raise RuntimeError("软相关下限应在 -1 到 1 之间。")
        if soft_corr > hard_corr:
            raise RuntimeError("软相关下限不能高于硬相关阈值。")
        if not (0 <= template_alpha <= 1):
            raise RuntimeError("模板跟随系数应在 0 到 1 之间。")
        if fb_tolerance <= 0:
            raise RuntimeError("FB 容差必须 > 0。")
        if not (0 <= max_saturated_frac <= 1):
            raise RuntimeError("最大近黑/近白比例应在 0 到 1 之间。")

        if self.max_frame_strain_jump.get().strip():
            try:
                jump = float(self.max_frame_strain_jump.get().strip())
            except ValueError:
                raise RuntimeError("单帧应变突变上限必须是数字，或者留空禁用。")
            if jump <= 0:
                raise RuntimeError("单帧应变突变上限必须 > 0，或者留空禁用。")

        if self.pixel_size_mm.get().strip():
            try:
                pix = float(self.pixel_size_mm.get().strip())
            except ValueError:
                raise RuntimeError("像素尺寸必须是数字，或者留空。")
            if pix <= 0:
                raise RuntimeError("像素尺寸必须 > 0，或者留空。")

        poisson_enabled = validate_poisson_role_groups(self.roi_groups)
        if poisson_enabled:
            axial_groups, transverse_groups = get_poisson_role_groups(self.roi_groups)
            axial = axial_groups[0]
            transverse = transverse_groups[0]
            if axial.get("actual_mode") == transverse.get("actual_mode"):
                msg = (
                    "泊松比通常需要一组拉伸方向 ROI 和一组横向收缩 ROI。\n\n"
                    f"当前两组 actual_mode 都是 {axial.get('actual_mode')}：\n"
                    f"拉伸方向：{axial.get('name')}\n"
                    f"横向方向：{transverse.get('name')}\n\n"
                    "这可能说明方向选择或 ROI 对齐不合适。是否仍然继续？"
                )
                if not messagebox.askyesno("泊松比方向警告", msg):
                    raise RuntimeError("用户取消：泊松比 ROI 方向可能不合适。")

        very_small = [g for g in self.roi_groups if g["L0"] < 50]
        if very_small:
            names = ", ".join(g["name"] for g in very_small)
            msg = (
                f"以下 ROI 组的 L0 < 50 px：{names}\n\n"
                "这通常说明应变方向选错，或两个 ROI 太近。\n"
                "是否仍然继续？"
            )
            if not messagebox.askyesno("L0 过小警告", msg):
                raise RuntimeError("用户取消：某些 ROI 组 L0 太小。")

    def validate_fullfield_before_processing(self):
        start_idx, end_idx = self.get_analysis_indices()
        if self.first_img8 is None:
            raise RuntimeError("请先加载参考图像。")
        snapshot = {
            "output_dir": self.output_folder.get().strip(),
            "image_paths": list(self.image_paths),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "reference_frame_1based": start_idx + 1,
            "field_roi": self.field_roi,
            "field_roi_reference_frame_1based": self.field_roi_reference_frame_1based,
            "min_texture_std": self.get_float_setting(self.min_texture_std, "最小纹理标准差"),
            "min_texture_contrast": self.get_float_setting(self.min_texture_contrast, "最小纹理对比度"),
            "max_saturated_frac": self.get_float_setting(self.max_saturated_frac, "最大近黑/近白比例"),
            "dic_subset_size": self.get_int_setting(self.dic_subset_size, "子集尺寸"),
            "dic_step": self.get_int_setting(self.dic_step, "步长"),
            "dic_solver": str(self.dic_solver.get()),
            "dic_strain_window": self.get_int_setting(self.dic_strain_window, "应变窗口"),
            "dic_smooth_sigma": self.get_float_setting(self.dic_smooth_sigma, "高斯平滑 σ"),
            "dic_search_radius": self.get_int_setting(self.dic_search_radius, "DIC 搜索半径"),
            "dic_zncc_min": self.get_float_setting(self.dic_zncc_min, "ZNCC 下限"),
            "dic_pyramid_levels": self.get_int_setting(self.dic_pyramid_levels, "金字塔层数"),
            "dic_pyramid_scale": self.get_float_setting(self.dic_pyramid_scale, "金字塔缩放"),
            "strain_degree": self.get_int_setting(self.dic_strain_degree, "拟合阶次"),
            "robust_strain": bool(self.dic_robust_strain.get()),
            "reject_nonconverged": bool(self.dic_reject_nonconverged.get()),
            "outlier_threshold_px": self.get_float_setting(self.dic_outlier_threshold, "异常位移阈值"),
            "mask": self.dic_mask_settings(),
        }
        return validate_fullfield_snapshot(snapshot)

    def build_processing_settings(self):
        start_idx, end_idx = self.get_analysis_indices()
        is_ff = self.is_fullfield_mode()
        if is_ff:
            # Hidden 1-D controls are deliberately not read in full-field
            # mode.  Their stale/invalid values must not poison a valid 2-D
            # settings snapshot.
            search_radius_base = 180
            hard_corr = 0.55
            soft_corr = 0.35
            enable_adaptive = True
            use_prev_frame_template = False
            template_alpha = 0.70
            max_frame_jump = None
            enable_fb_check = True
            fb_tolerance = 12.0
            pixel_size_mm = None
            overlay_every = 5
        else:
            max_frame_jump = None
            if self.max_frame_strain_jump.get().strip():
                max_frame_jump = float(self.max_frame_strain_jump.get().strip())

            pixel_size_mm = None
            if self.pixel_size_mm.get().strip():
                pixel_size_mm = float(self.pixel_size_mm.get().strip())
            search_radius_base = self.get_int_setting(self.search_radius, "搜索半径")
            hard_corr = self.get_float_setting(self.hard_corr, "硬相关阈值")
            soft_corr = self.get_float_setting(self.soft_corr, "软相关下限")
            enable_adaptive = bool(self.enable_adaptive.get())
            use_prev_frame_template = bool(self.use_prev_frame_template.get())
            template_alpha = self.get_float_setting(self.template_alpha, "模板跟随系数")
            enable_fb_check = bool(self.enable_fb_check.get())
            fb_tolerance = self.get_float_setting(self.fb_tolerance_px, "FB 容差")
            overlay_every = self.get_int_setting(self.overlay_every, "overlay 保存间隔")

        if is_ff:
            export_origin_txt = False
            export_origin_opju = False
            export_engineering_png = False
            export_publication_figures = False
            export_qc_summary = False
            export_full_csv = False
            export_corr_plot = False
            export_parameters = False
            export_overlays = bool(self.export_overlays.get())
            tracking_preset = "fullfield"
        else:
            export_origin_txt = bool(self.export_origin_txt.get())
            export_origin_opju = bool(self.export_origin_opju.get())
            export_engineering_png = bool(self.export_engineering_png.get())
            export_publication_figures = bool(self.export_publication_figures.get())
            export_qc_summary = bool(self.export_qc_summary.get())
            export_full_csv = bool(self.export_full_csv.get())
            export_corr_plot = bool(self.export_corr_plot.get())
            export_overlays = bool(self.export_overlays.get())
            export_parameters = bool(self.export_parameters.get())
            tracking_preset = self.tracking_preset.get()

        field_roi = tuple(self.field_roi) if self.field_roi is not None else None
        reference_normalization = None
        if self.first_raw is not None:
            reference_raw = self.first_raw if start_idx == 0 else read_gray_image(self.image_paths[start_idx])
            reference_normalization = compute_reference_normalization(reference_raw)
        input_identities = ordered_input_manifest(self.image_paths)
        return {
            "output_dir": Path(self.output_folder.get().strip()),
            "image_paths": list(self.image_paths),
            "roi_groups": [dict(group) for group in self.roi_groups],
            "start_idx": start_idx,
            "end_idx": end_idx,
            "search_radius_base": search_radius_base,
            "hard_corr": hard_corr,
            "soft_corr": soft_corr,
            "enable_adaptive": enable_adaptive,
            "use_prev_frame_template": use_prev_frame_template,
            "template_policy": "experimental_follow" if use_prev_frame_template else "fixed_reference",
            "template_alpha": template_alpha,
            "max_frame_jump": max_frame_jump,
            "enable_fb_check": enable_fb_check,
            "fb_tolerance": fb_tolerance,
            "pixel_size_mm": pixel_size_mm,
            "overlay_every": overlay_every,
            "export_origin_txt": export_origin_txt,
            "export_origin_opju": export_origin_opju,
            "export_engineering_png": export_engineering_png,
            "export_publication_figures": export_publication_figures,
            "export_qc_summary": export_qc_summary,
            "export_full_csv": export_full_csv,
            "export_corr_plot": export_corr_plot,
            "export_overlays": export_overlays,
            "export_parameters": export_parameters,
            "image_folder": self.image_folder.get(),
            "tracking_preset": tracking_preset,
            "min_texture_std": float(self.min_texture_std.get()),
            "min_texture_contrast": float(self.min_texture_contrast.get()),
            "max_saturated_frac": float(self.max_saturated_frac.get()),
            "analysis_mode": str(self.analysis_mode.get()),
            "field_roi": field_roi,
            "field_roi_reference_frame_1based": self.field_roi_reference_frame_1based,
            "reference_frame_1based": start_idx + 1,
            "reference_filename": Path(self.image_paths[start_idx]).name,
            "image_sequence_fingerprint": image_sequence_fingerprint(self.image_paths),
            "input_identities": input_identities,
            "normalization": reference_normalization,
            "dic_subset_size": int(self.dic_subset_size.get()) if is_ff else 21,
            "dic_step": int(self.dic_step.get()) if is_ff else 5,
            "dic_solver": str(self.dic_solver.get()),
            "dic_strain_window": int(self.dic_strain_window.get()) if is_ff else 5,
            "dic_smooth_sigma": float(self.dic_smooth_sigma.get()) if is_ff else 0.0,
            "dic_search_radius": int(self.dic_search_radius.get()) if is_ff else 20,
            "dic_zncc_min": float(self.dic_zncc_min.get()) if is_ff else 0.75,
            "dic_pyramid_levels": int(self.dic_pyramid_levels.get()) if is_ff else 1,
            "dic_pyramid_scale": float(self.dic_pyramid_scale.get()) if is_ff else 0.5,
            "strain_degree": int(self.dic_strain_degree.get()) if is_ff else 2,
            "robust_strain": bool(self.dic_robust_strain.get()) if is_ff else True,
            "reject_nonconverged": bool(self.dic_reject_nonconverged.get()) if is_ff else False,
            "outlier_threshold_px": float(self.dic_outlier_threshold.get()) if is_ff else 1.0,
            "mask": self.dic_mask_settings() if is_ff else {"mode": "none"},
            "display": self.dic_display_options() if is_ff else {},
        }

    def post_to_ui(self, callback, run_token=None):
        if run_token is None:
            run_token = getattr(self._worker_context, "run_token", None)
        if run_token is not None:
            original_callback = callback

            def guarded_callback():
                if self._active_run_token != run_token:
                    return
                original_callback()

            callback = guarded_callback
        self.ui_queue.put(callback)
        return True

    def start_ui_queue_polling(self):
        try:
            self.root.after(50, self.drain_ui_queue)
        except (RuntimeError, tk.TclError):
            pass

    def drain_ui_queue(self):
        while True:
            try:
                callback = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except Exception:
                traceback.print_exc()
        self.start_ui_queue_polling()

    # ---------- 批量处理 ----------

    def _finalize_processing_state(self, run_token):
        """Release the completion gate only after all queued run callbacks drain."""
        if run_token is not None and self._active_run_token != run_token:
            return
        self._completion_pending = False
        self.update_workflow_action_states()

    def start_processing(self):
        if self.is_processing:
            messagebox.showinfo("正在处理", "程序正在处理，请等待当前任务完成。")
            return
        if self._completion_pending:
            messagebox.showinfo("正在收尾", "上一轮分析正在收尾，请等待完成后再开始。")
            return

        try:
            self.validate_before_processing()
            settings = self.build_processing_settings()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self._run_generation += 1
        run_token = self._run_generation
        settings["_run_token"] = run_token
        self._active_run_token = run_token
        self._completion_pending = False
        self.is_processing = True
        self._update_progress(0)
        if settings.get("analysis_mode") == ANALYSIS_MODE_FULLFIELD:
            self.status_var.set("正在分析：全场 2D DIC。")
            self.log("开始全场 2D DIC：IC-GN/IC-LM 匹配 subset 网格。")
            worker = self.process_fullfield_thread
        else:
            self.status_var.set("正在分析：开始追踪 ROI 组。")
            self.log("开始分析：正在追踪各 ROI 组，请稍候。")
            worker = self.process_images_thread
        self.update_workflow_action_states()

        thread = threading.Thread(target=worker, args=(settings,), daemon=True)
        thread.start()

    def process_fullfield_thread(self, settings):
        run_token = settings.get("_run_token")
        self._worker_context.run_token = run_token
        try:
            self.process_fullfield(settings)
        except Exception as exc:
            message = str(exc)
            details = traceback.format_exc()
            self.post_to_ui(lambda m=message: messagebox.showerror("处理失败", m))
            self.post_to_ui(lambda: self.status_var.set("分析失败，详情见“质量与日志”。"))
            self.post_to_ui(lambda e=exc, d=details: self.log_user_error("全场 DIC", e, d))
        finally:
            if run_token is not None:
                self._completion_pending = True
            self.is_processing = False
            if run_token is not None:
                self.post_to_ui(lambda token=run_token: self._finalize_processing_state(token))
            else:
                self.post_to_ui(lambda: self.update_workflow_action_states())
            self._worker_context.run_token = None

    def process_fullfield(self, settings):
        # The numerical/export engine is Tk-free; this adapter only translates
        # progress and final state back onto the UI thread.  Keep the legacy
        # body below as a compatibility fallback for old callers that
        # explicitly request it.
        if not settings.get("_legacy_direct_processing", False):
            # The worker must consume only the frozen settings snapshot and
            # image files.  In particular, do not call the GUI method here:
            # it reads Tk variables and is only safe on the main thread.
            snapshot_keys = (
                "image_paths",
                "start_idx",
                "end_idx",
                "reference_frame_1based",
                "field_roi",
                "field_roi_reference_frame_1based",
                "dic_subset_size",
                "dic_step",
                "dic_solver",
                "dic_strain_window",
                "dic_smooth_sigma",
                "dic_search_radius",
                "dic_zncc_min",
            )
            if (
                threading.current_thread() is threading.main_thread()
                and not all(key in settings for key in snapshot_keys)
            ):
                # Preserve the old direct-adapter extension point for callers
                # that supplied an intentionally minimal fake settings object.
                # Real GUI runs always provide a complete snapshot, and every
                # worker-thread invocation takes the pure path above.
                self.validate_fullfield_before_processing()
            else:
                validate_fullfield_snapshot(settings)
            engine_settings = dict(settings)
            # ``__file__`` may point into a PyInstaller archive.  Resolve the
            # real auditable copies under ``_MEIPASS/sources`` instead of
            # passing a path that cannot be hashed or verified.
            engine_settings["_code_paths"] = [
                str(path) for path in _core.resolve_code_paths(include_gui=True)
            ]
            engine_settings["_gui_adapter"] = True
            # A pre-v0.2 GUI run has no ownership ledger.  Preserve its
            # established exact-name migration behaviour once, before the new
            # transactional engine starts; every v0.2 run is governed solely
            # by the manifest ledger.  Similar user names (for example
            # ``frame_0002_u.txt``) are not selected by this allowlist.
            output_root = Path(settings["output_dir"])
            legacy_manifest = output_root / "run_manifest.json"
            if not legacy_manifest.exists():
                # The pure snapshot preflight above is complete before any
                # legacy output is migrated.  The new core then owns all
                # subsequent staging and publication.
                archive_previous_fullfield_outputs(output_root / "dic")
            # GUI preflight historically permits low/saturated texture so the
            # user can inspect a structured invalid-frame result.  Headless
            # callers remain fail-closed on this contract; this flag is scoped
            # to the adapter and is recorded in the run configuration.
            engine_settings["_allow_low_texture_preflight"] = True

            def progress(event):
                if isinstance(event, dict):
                    value = 100.0 * float(event.get("fraction", 0.0))
                    frame = event.get("frame_global_1based")
                    state = event.get("status", "")
                else:
                    value = 100.0 * float(event)
                    frame = None
                    state = ""
                self.post_to_ui(lambda v=value: self._update_progress(v))
                if frame is not None:
                    self.post_to_ui(lambda f=frame, s=state: self.status_var.set(f"正在处理全场 DIC：第 {f} 帧 {s}"))

            try:
                result = _core.run_fullfield_sequence(engine_settings, progress_callback=progress)
            except Exception:
                # Keep the pre-v0.2 GUI's discoverable failure location while
                # retaining the canonical root evidence written by the core.
                failed_root = output_root / "_failed_runs"
                legacy_failed_root = output_root / "dic" / "_failed_runs"
                if failed_root.is_dir():
                    legacy_failed_root.mkdir(parents=True, exist_ok=True)
                    for failed_dir in sorted((path for path in failed_root.iterdir() if path.is_dir()), key=lambda path: path.name):
                        target = legacy_failed_root / failed_dir.name
                        if not target.exists():
                            try:
                                shutil.copytree(failed_dir, target)
                            except OSError:
                                pass
                raise
            if not result.get("scientific_ok", False):
                raise RuntimeError("全场 DIC 科学有效性门未通过（无有效应变/有限应变）：" + ", ".join(result.get("errors", [])))
            last_field = result.get("last_field")
            if last_field is None:
                raise RuntimeError("全场 DIC 没有可展示的有效结果。")
            last_image = result.get("last_image")
            frames = result.get("frames", [])
            valid_count = sum(1 for item in frames if item.get("status") == "scientific_valid")
            total_count = len(frames)

            def finish():
                self.dic_last_field = last_field
                self.dic_last_image = np.asarray(last_image).copy() if last_image is not None else None
                self.dic_last_frame_1based = last_field.get("frame_global_1based")
                self.dic_last_filename = last_field.get("frame_filename")
                self.dic_last_reference_frame_1based = result.get("manifest", {}).get("reference_frame_1based", settings.get("reference_frame_1based", settings.get("start_idx", 0) + 1))
                self.dic_last_reference_filename = result.get("manifest", {}).get("reference_filename")
                self.show_field_viewer(
                    last_field,
                    image=self.dic_last_image,
                    frame_1based=self.dic_last_frame_1based,
                    filename=self.dic_last_filename,
                    reference_frame_1based=self.dic_last_reference_frame_1based,
                    reference_filename=self.dic_last_reference_filename,
                )
                self._update_progress(100)
                self.status_var.set(f"处理完成，全场 DIC：{valid_count}/{total_count} 帧科学有效。")
                if hasattr(self, "qc_overview_var"):
                    self.qc_overview_var.set(self.format_fullfield_qc_overview(frames))
                self.log(f"全场 DIC manifest 已验证：{result['manifest_path']}")
                self.show_completion_and_open_output_folder(
                    f"全场 2D DIC 完成：{valid_count}/{total_count} 个变形帧有效。结果已写入 {settings['output_dir']}",
                    settings["output_dir"],
                )

            self.post_to_ui(finish)
            return result

        paths = settings["image_paths"]
        start_idx = settings["start_idx"]
        end_idx = settings["end_idx"]
        roi = settings["field_roi"]
        output_dir = Path(settings["output_dir"]) / "dic"
        if roi is None:
            raise RuntimeError("全场 DIC 缺少独立的全场 ROI，不能回退到 1D ROI1。")
        if not paths or start_idx < 0 or end_idx >= len(paths) or end_idx <= start_idx:
            raise RuntimeError("全场 DIC 的分析范围无效。")
        field_ref = settings.get("field_roi_reference_frame_1based")
        if field_ref is None:
            raise RuntimeError("全场 ROI 缺少绘制参考帧记录，请在当前起始/参考帧重画。")
        if int(field_ref) != start_idx + 1:
            raise RuntimeError(
                f"全场 ROI 在第 {field_ref} 帧定义，但当前起始/参考帧为第 {start_idx + 1} 帧。"
            )
        validate_image_sequence_dimensions(paths[start_idx : end_idx + 1])
        ref_raw = read_gray_image(paths[start_idx])
        normalization = compute_reference_normalization(ref_raw)
        ref8 = normalize_with_bounds(ref_raw, normalization)
        if not rect_is_inside_image(roi, ref8.shape):
            raise RuntimeError("全场 ROI 必须完整位于参考图像内。")
        try:
            requested_subset_size = int(settings["dic_subset_size"])
            requested_step = int(settings["dic_step"])
            requested_strain_window = int(settings["dic_strain_window"])
            requested_search_radius = int(settings["dic_search_radius"])
            requested_zncc_min = float(settings["dic_zncc_min"])
            requested_smooth_sigma = float(settings["dic_smooth_sigma"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("全场 DIC 参数无效，无法开始计算。") from exc
        if requested_subset_size < 9:
            raise RuntimeError("子集尺寸必须 >= 9。")
        if requested_step < 1:
            raise RuntimeError("步长必须 >= 1。")
        if requested_strain_window < 3:
            raise RuntimeError("应变窗口必须 >= 3。")
        if requested_search_radius < 1:
            raise RuntimeError("DIC 搜索半径必须 > 0。")
        if not (0.0 <= requested_zncc_min <= 1.0):
            raise RuntimeError("ZNCC 下限应在 0 到 1 之间。")
        if requested_smooth_sigma < 0:
            raise RuntimeError("高斯平滑 σ 不能为负。")
        if str(settings.get("dic_solver")) not in DIC_SOLVERS:
            raise RuntimeError("求解器必须是 IC-GN 或 IC-LM。")
        effective_subset_size = _odd_subset_size(requested_subset_size)
        effective_strain_window = _odd_window_size(requested_strain_window)
        X, Y = build_poi_grid(roi, effective_subset_size, requested_step, ref8.shape)
        if not poi_grid_is_usable(X, Y, min_rows=3, min_cols=3):
            raise RuntimeError("当前全场 ROI / 子集 / 步长至少需要 3×3 个可分析的 2D POI。")
        # All read-only validation is complete.  Preserve old generated files
        # now, immediately before formal computation starts.
        archive_previous_fullfield_outputs(output_dir)
        staging_dir = _create_unique_timestamped_dir(output_dir, prefix=".staging")
        kwargs = {
            "subset_size": requested_subset_size,
            "step": requested_step,
            "solver": settings["dic_solver"],
            "search_radius": requested_search_radius,
            "zncc_min": requested_zncc_min,
            "strain_window": requested_strain_window,
            "smooth_sigma": requested_smooth_sigma,
        }
        n_def = end_idx - start_idx
        last_field = None
        last_img8 = ref8
        last_frame_idx = None
        last_filename = None
        valid_frame_count = 0
        reference_provenance = {
            "analysis_mode": ANALYSIS_MODE_FULLFIELD,
            "image_folder": settings.get("image_folder", ""),
            "reference_frame_1based": start_idx + 1,
            "reference_filename": Path(paths[start_idx]).name,
            "field_roi": tuple(int(round(value)) for value in roi),
            "field_roi_reference_frame_1based": settings.get(
                "field_roi_reference_frame_1based"
            ) or start_idx + 1,
            "image_shape_px": tuple(int(value) for value in ref8.shape[:2]),
            "subset_size_px": effective_subset_size,
            "step_px": requested_step,
            "solver": settings["dic_solver"],
            "zncc_min": requested_zncc_min,
            "strain_window": effective_strain_window,
            "smooth_sigma": requested_smooth_sigma,
            "image_sequence_fingerprint": settings.get("image_sequence_fingerprint"),
        }
        try:
            for k, frame_i in enumerate(range(start_idx + 1, end_idx + 1), start=1):
                img_raw = read_gray_image(paths[frame_i])
                if tuple(img_raw.shape[:2]) != tuple(ref8.shape[:2]):
                    raise RuntimeError(
                        f"全场 DIC 图像尺寸不一致：参考帧 {ref8.shape[1]}×{ref8.shape[0]} px，"
                        f"{Path(paths[frame_i]).name} 为 {img_raw.shape[1]}×{img_raw.shape[0]} px。"
                    )
                img8 = normalize_with_bounds(img_raw, normalization)
                field = run_2d_dic(ref8, img8, roi, **kwargs)
                if not fullfield_field_has_finite_strain(field):
                    msg = f"全场 DIC {k}/{n_def}：{Path(paths[frame_i]).name} 无有效/有限应变，已跳过导出。"
                    self.post_to_ui(lambda v=100.0 * k / max(n_def, 1): self._update_progress(v))
                    self.post_to_ui(lambda m=msg: self.status_var.set(m))
                    self.post_to_ui(lambda m=msg: self.log(m))
                    continue

                valid_frame_count += 1
                field["provenance"] = {
                    **reference_provenance,
                    "frame_global_1based": frame_i + 1,
                    "frame_filename": Path(paths[frame_i]).name,
                }
                field["reference_frame_1based"] = start_idx + 1
                field["reference_filename"] = Path(paths[start_idx]).name
                field["frame_global_1based"] = frame_i + 1
                field["frame_filename"] = Path(paths[frame_i]).name
                stem = f"frame_{frame_i + 1:04d}"
                export_dic_field_outputs(field, staging_dir, stem=stem)
                if settings.get("export_overlays"):
                    overlay = overlay_dic_field_on_image(img8, field, component="Exx")
                    write_image_checked(
                        staging_dir / f"{stem}_overlay.png",
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    )
                last_field = field
                last_img8 = img8
                last_frame_idx = frame_i
                last_filename = Path(paths[frame_i]).name
                progress_val = 100.0 * k / max(n_def, 1)
                mean_u = float(np.nanmean(field["u"])) if np.isfinite(field["u"]).any() else float("nan")
                mean_v = float(np.nanmean(field["v"])) if np.isfinite(field["v"]).any() else float("nan")
                n_valid = int(np.asarray(field["valid"]).sum())
                msg = (
                    f"全场 DIC {k}/{n_def}：{Path(paths[frame_i]).name}  "
                    f"有效点={n_valid}  mean u={mean_u:.4f} v={mean_v:.4f}  {settings['dic_solver']}"
                )
                self.post_to_ui(lambda v=progress_val: self._update_progress(v))
                self.post_to_ui(lambda m=msg: self.status_var.set(m))
                self.post_to_ui(lambda m=msg: self.log(m))

            if last_field is None or valid_frame_count == 0:
                raise RuntimeError("全场 DIC 所有变形帧均无有效/有限应变，未导出结果。")
            commit_fullfield_staging(staging_dir, output_dir)
        except Exception:
            try:
                archive_failed_fullfield_staging(staging_dir, output_dir)
            except Exception as archive_exc:
                self.post_to_ui(lambda m=f"全场 DIC 失败归档失败：{archive_exc}": self.log(m))
            raise

        def _finish():
            self.dic_last_field = last_field
            self.dic_last_image = last_img8.copy()
            self.dic_last_frame_1based = last_frame_idx + 1
            self.dic_last_filename = last_filename
            self.dic_last_reference_frame_1based = start_idx + 1
            self.dic_last_reference_filename = Path(paths[start_idx]).name
            self.show_field_viewer(
                last_field,
                image=last_img8,
                frame_1based=last_frame_idx + 1,
                filename=last_filename,
                reference_frame_1based=start_idx + 1,
                reference_filename=Path(paths[start_idx]).name,
            )
            self.show_completion_and_open_output_folder(
                f"全场 2D DIC 完成：{valid_frame_count}/{n_def} 个变形帧有效。结果已写入 {output_dir}",
                output_dir,
            )

        self.post_to_ui(_finish)

    def process_images_thread(self, settings):
        run_token = settings.get("_run_token")
        self._worker_context.run_token = run_token
        try:
            self.process_images(settings)
        except Exception as exc:
            message = str(exc)
            details = traceback.format_exc()
            self.post_to_ui(lambda m=message: messagebox.showerror("处理失败", m))
            self.post_to_ui(lambda: self.status_var.set("分析失败，详情见“质量与日志”。"))
            self.post_to_ui(lambda e=exc, d=details: self.log_user_error("批量处理", e, d))
        finally:
            if run_token is not None:
                self._completion_pending = True
            self.is_processing = False
            if run_token is not None:
                self.post_to_ui(lambda token=run_token: self._finalize_processing_state(token))
            else:
                self.post_to_ui(lambda: self.update_workflow_action_states())
            self._worker_context.run_token = None

    def init_group_states(self, first_img8, groups=None, texture_settings=None):
        if groups is None:
            groups = self.roi_groups
        if texture_settings is None:
            texture_settings = {
                "min_texture_std": self.min_texture_std.get(),
                "min_texture_contrast": self.min_texture_contrast.get(),
                "max_saturated_frac": self.max_saturated_frac.get(),
            }
        return [
            initialize_extensometer_group_state(
                first_img8,
                group,
                min_texture_std=texture_settings.get("min_texture_std", 8.0),
                min_texture_contrast=texture_settings.get("min_texture_contrast", 25.0),
                max_saturated_frac=texture_settings.get("max_saturated_frac", 0.20),
                min_structure_ratio=_core.DEFAULT_TEXTURE_MIN_STRUCTURE_RATIO,
            )
            for group in groups
        ]

    def process_one_group_one_frame(
        self,
        state,
        img8,
        frame_idx,
        filename,
        params,
    ):
        # Canonical numerical state machine lives in the Tk-free core.
        return track_extensometer_group_frame(
            state,
            img8,
            frame_idx,
            filename,
            params,
        )

    def process_images(self, settings=None):
        if settings is None:
            settings = self.build_processing_settings()

        if not settings.get("_legacy_direct_processing", False):
            image_paths = settings.get("image_paths", [])
            start_idx = int(settings.get("start_idx", 0))
            end_idx = int(settings.get("end_idx", len(image_paths) - 1))
            validate_image_sequence_dimensions(image_paths[start_idx:end_idx + 1])
            engine_settings = dict(settings)
            # Use the same source resolver as the full-field adapter; this is
            # required for frozen GUI runs where both module ``__file__``
            # values can be archive-style/non-filesystem paths.
            engine_settings["_code_paths"] = [
                str(path) for path in _core.resolve_code_paths(include_gui=True)
            ]
            engine_settings["_gui_adapter"] = True

            def progress(event):
                if isinstance(event, dict):
                    value = 100.0 * float(event.get("fraction", 0.0))
                    frame = event.get("frame_global_1based")
                else:
                    value = 100.0 * float(event)
                    frame = None
                self.post_to_ui(lambda v=value: self._update_progress(v))
                if frame is not None:
                    self.post_to_ui(lambda f=frame: self.status_var.set(f"正在批量追踪：第 {f} 帧"))

            result = _core.run_extensometer_sequence(engine_settings, progress_callback=progress)
            if not result.get("scientific_ok", False):
                raise RuntimeError("虚拟引伸计科学有效性门未通过：" + ", ".join(result.get("errors", [])))
            df = result.get("dataframe")
            summary = result.get("summary")
            output_dir = Path(settings["output_dir"])
            if df is None or summary is None:
                raise RuntimeError("引擎未返回可展示的结果表。")
            written_paths = [Path(path) for path in result.get("outputs", [])]
            qc_level = summary.get("overall", {}).get("qc_level", "Unknown")
            n_rejected = summary.get("overall", {}).get("rejected_frames", 0)
            n_adaptive = summary.get("overall", {}).get("adaptive_accepted_frames", 0)
            path_log = "输出文件：\n" + "\n".join(str(path) for path in written_paths)
            done_msg = (
                "处理完成。\n"
                f"核心结果已保存到: {output_dir / 'core'}\n"
                f"QC 状态：{qc_level}\n"
                f"拒绝帧数：{n_rejected}\n"
                f"自适应接受帧数：{n_adaptive}\n"
                f"manifest 已验证：{result['manifest_path']}"
            )
            if settings.get("export_origin_txt") and (output_dir / "core" / "strain_mean_groups.txt").is_file():
                done_msg += "\n平均应变文件: core\\strain_mean_groups.txt\n平均应变列也已写入: core\\strain_all_groups.txt"
            self.post_to_ui(lambda: self._update_progress(100))
            self.post_to_ui(lambda: self.status_var.set(f"处理完成，QC 状态：{qc_level}"))
            self.post_to_ui(lambda s=summary: self.update_qc_overview(s))
            self.post_to_ui(lambda m=done_msg + "\n" + path_log: self.log(m))
            self.post_to_ui(lambda m=done_msg: self.show_completion_and_open_output_folder(m, output_dir))
            try:
                if not df.empty and settings.get("roi_groups"):
                    self.post_to_ui(lambda: self.show_results_viewer(df, settings["roi_groups"]))
            except Exception:
                self.post_to_ui(lambda: self.log("结果预览窗口初始化失败（不影响已验证文件）"))
            return result

        start_idx = settings["start_idx"]
        end_idx = settings["end_idx"]
        image_paths = settings["image_paths"]
        image_paths_run = image_paths[start_idx:end_idx + 1]
        roi_groups = settings["roi_groups"]

        # Validate and freeze the reference normalization before creating any
        # output directory.  A non-finite input therefore cannot leave even a
        # staged/current result behind.
        validate_image_sequence_dimensions(image_paths_run)
        first_raw = read_gray_image(image_paths_run[0])
        normalization = compute_reference_normalization(first_raw)

        output_dir = settings["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        core_dir = output_dir / "core"
        qc_dir = output_dir / "qc"
        optional_dir = output_dir / "optional"

        search_radius_base = settings["search_radius_base"]
        hard_corr = settings["hard_corr"]
        soft_corr = settings["soft_corr"]

        enable_adaptive = settings["enable_adaptive"]
        use_prev_frame_template = settings["use_prev_frame_template"]
        template_alpha = settings["template_alpha"]

        max_frame_jump = settings["max_frame_jump"]
        enable_fb_check = settings["enable_fb_check"]
        fb_tolerance = settings["fb_tolerance"]
        pixel_size_mm = settings["pixel_size_mm"]
        overlay_every = settings["overlay_every"]

        export_origin_txt = settings["export_origin_txt"]
        export_engineering_png = settings["export_engineering_png"]
        export_publication_figures = settings["export_publication_figures"]
        export_qc_summary = settings["export_qc_summary"]
        export_full_csv = settings["export_full_csv"]
        export_corr_plot = settings["export_corr_plot"]
        export_overlays = settings["export_overlays"]
        export_parameters = settings["export_parameters"]
        export_origin_opju = settings["export_origin_opju"]

        params = {
            "search_radius_base": search_radius_base,
            "hard_corr": hard_corr,
            "soft_corr": soft_corr,
            "enable_adaptive": enable_adaptive,
            "use_prev_frame_template": use_prev_frame_template,
            "template_alpha": template_alpha,
            "max_frame_jump": max_frame_jump,
            "enable_fb_check": enable_fb_check,
            "fb_tolerance": fb_tolerance,
            "pixel_size_mm": pixel_size_mm,
        }

        first_img8 = normalize_with_bounds(first_raw, normalization)

        states = self.init_group_states(
            first_img8,
            roi_groups,
            texture_settings={
                "min_texture_std": settings.get("min_texture_std", 8.0),
                "min_texture_contrast": settings.get("min_texture_contrast", 25.0),
                "max_saturated_frac": settings.get("max_saturated_frac", 0.20),
            },
        )

        overlay_dirs = {}
        if export_overlays:
            overlay_root = optional_dir / "overlays"
            for state in states:
                gname = safe_name(state["group"]["name"])
                gdir = overlay_root / gname
                gdir.mkdir(parents=True, exist_ok=True)
                overlay_dirs[state["group"]["name"]] = gdir

        all_rows = []
        n = len(image_paths_run)
        total_work = n * len(states)
        done_work = 0

        self.post_to_ui(lambda: self.log(
            f"开始处理，分析范围：第 {start_idx + 1} 到第 {end_idx + 1} 帧，"
            f"共 {n} 张图，{len(states)} 组 ROI。"
        ))
        self.post_to_ui(lambda: self.status_var.set("正在批量追踪多组 ROI 并计算应变..."))

        for i, path in enumerate(image_paths_run):
            raw = read_gray_image(path)
            img8 = normalize_with_bounds(raw, normalization)
            fname = os.path.basename(path)

            for state in states:
                row, overlay_info = self.process_one_group_one_frame(
                    state=state,
                    img8=img8,
                    frame_idx=i,
                    filename=fname,
                    params=params,
                )
                row["frame_local_1based"] = i + 1
                row["frame_global_1based"] = start_idx + i + 1
                all_rows.append(row)

                group = state["group"]
                group_name = group["name"]
                actual_mode = group["actual_mode"]

                accepted = overlay_info["accepted"]
                accept_mode = overlay_info["accept_mode"]

                save_overlay = export_overlays and (
                    i % overlay_every == 0 or i == n - 1 or not accepted or accept_mode == "adaptive"
                )
                if save_overlay:
                    overlay = draw_group_overlay(
                        img8,
                        group_name,
                        actual_mode,
                        overlay_info["used_rect1"],
                        overlay_info["used_rect2"],
                        overlay_info["candidate_rect1"],
                        overlay_info["candidate_rect2"],
                        i,
                        overlay_info["strain"],
                        overlay_info["last_valid_strain"],
                        overlay_info["score1"],
                        overlay_info["score2"],
                        accepted,
                        accept_mode,
                        overlay_info["reason"],
                        fb_err1=overlay_info["fb_err1"] if np.isfinite(overlay_info["fb_err1"]) else None,
                        fb_err2=overlay_info["fb_err2"] if np.isfinite(overlay_info["fb_err2"]) else None,
                    )
                    out_name = overlay_dirs[group_name] / f"tracked_{i:05d}.png"
                    write_image_checked(out_name, overlay)

                done_work += 1
                if done_work % max(1, len(states) * 5) == 0 or not accepted or accept_mode == "adaptive" or i == n - 1:
                    progress_val = done_work / total_work * 100
                    strain_text = f"{overlay_info['strain']:.6f}" if np.isfinite(overlay_info["strain"]) else "NaN"
                    msg = format_tracking_status_line(
                        i + 1,
                        n,
                        group_name,
                        accept_mode,
                        strain_text,
                        overlay_info["score1"],
                        overlay_info["score2"],
                    )
                    self.post_to_ui(lambda v=progress_val: self._update_progress(v))
                    self.post_to_ui(lambda m=msg: self.status_var.set(m))
                    self.post_to_ui(lambda m=msg: self.log(m))

        df = pd.DataFrame(all_rows)
        summary = build_qc_summary(df)
        poisson_enabled = poisson_roles_are_configured(roi_groups)
        written_paths = []

        if export_origin_txt or export_engineering_png or export_origin_opju:
            core_dir.mkdir(parents=True, exist_ok=True)

        publication_dir = optional_dir / "publication_figures"
        if export_publication_figures:
            publication_dir.mkdir(parents=True, exist_ok=True)

        for group in roi_groups:
            gname = group["name"]
            sg = safe_name(gname)
            gdf = df[df["group"] == gname].copy()

            if export_origin_txt:
                txt_path = core_dir / f"strain_{sg}.txt"
                write_origin_txt(gdf, txt_path)
                written_paths.append(txt_path)

            if export_engineering_png:
                fig_path = core_dir / f"engineering_strain_{sg}.png"
                plot_engineering_strain(gdf, fig_path, f"Engineering strain - {gname}")
                written_paths.append(fig_path)

            if export_publication_figures:
                for fig_path in publication_figure_paths(publication_dir, f"engineering_strain_{sg}"):
                    plot_engineering_strain(gdf, fig_path, f"Engineering strain - {gname}", preset_name="publication")
                    written_paths.append(fig_path)

        if export_origin_txt:
            all_txt = core_dir / "strain_all_groups.txt"
            write_all_groups_origin_txt(df, all_txt, roi_groups)
            written_paths.append(all_txt)

            mean_txt = core_dir / "strain_mean_groups.txt"
            write_mean_groups_origin_txt(df, roi_groups, mean_txt)
            written_paths.append(mean_txt)

            if poisson_enabled:
                poisson_txt = core_dir / "poisson_ratio.txt"
                write_poisson_ratio_txt(df, roi_groups, poisson_txt)
                written_paths.append(poisson_txt)

        if export_engineering_png:
            combined_fig = core_dir / "engineering_strain_all_groups.png"
            plot_all_groups_engineering_strain(df, roi_groups, combined_fig)
            written_paths.append(combined_fig)

            if poisson_enabled:
                poisson_fig = core_dir / "poisson_ratio.png"
                plot_poisson_ratio(df, roi_groups, poisson_fig)
                written_paths.append(poisson_fig)

        if export_publication_figures:
            for fig_path in publication_figure_paths(publication_dir, "engineering_strain_all_groups"):
                plot_all_groups_engineering_strain(df, roi_groups, fig_path, preset_name="publication")
                written_paths.append(fig_path)

            if poisson_enabled:
                for fig_path in publication_figure_paths(publication_dir, "poisson_ratio"):
                    plot_poisson_ratio(df, roi_groups, fig_path, preset_name="publication")
                    written_paths.append(fig_path)

        if export_origin_opju:
            opju_path = core_dir / ORIGIN_OPJU_FILENAME
            try:
                write_origin_opju_project(df, roi_groups, opju_path)
            except Exception as exc:
                warning = f"Origin OPJU 项目生成失败：{exc}"
                self.post_to_ui(lambda m=warning: self.log(m))
                self.post_to_ui(lambda m=warning: messagebox.showwarning("Origin OPJU 生成失败", m))
            else:
                written_paths.append(opju_path)

        if export_qc_summary:
            qc_path = qc_dir / "qc_summary.txt"
            write_qc_summary(summary, qc_path)
            written_paths.append(qc_path)

        if export_full_csv:
            full_csv_dir = optional_dir / "full_csv"
            full_csv_dir.mkdir(parents=True, exist_ok=True)
            all_csv = full_csv_dir / "strain_results_all_groups.csv"
            df.to_csv(all_csv, index=False, encoding="utf-8-sig")
            written_paths.append(all_csv)

            group_dir = full_csv_dir / "per_group_results"
            group_dir.mkdir(parents=True, exist_ok=True)
            for group in roi_groups:
                gname = group["name"]
                sg = safe_name(gname)
                gdf = df[df["group"] == gname].copy()
                g_csv = group_dir / f"strain_results_{sg}.csv"
                gdf.to_csv(g_csv, index=False, encoding="utf-8-sig")
                written_paths.append(g_csv)

        if export_corr_plot:
            corr_dir = optional_dir / "correlation_plots"
            corr_dir.mkdir(parents=True, exist_ok=True)
            for group in roi_groups:
                gname = group["name"]
                sg = safe_name(gname)
                gdf = df[df["group"] == gname].copy()
                corr_path = corr_dir / f"correlation_scores_{sg}.png"
                plot_correlation_scores(gdf, corr_path, gname, hard_corr, soft_corr, preset_name="raw_inspection")
                written_paths.append(corr_path)

                if export_publication_figures:
                    for pub_corr_path in publication_figure_paths(publication_dir, f"correlation_scores_{sg}"):
                        plot_correlation_scores(gdf, pub_corr_path, gname, hard_corr, soft_corr, preset_name="publication")
                        written_paths.append(pub_corr_path)

        if export_parameters:
            param_dir = optional_dir / "parameters"
            param_dir.mkdir(parents=True, exist_ok=True)
            params_path = param_dir / "tracking_parameters.txt"
            with open(params_path, "w", encoding="utf-8") as f:
                f.write("DIC Virtual Extensometer GUI v7 Multi-ROI Range Preview Parameters\n")
                f.write("----------------------------------------------------\n")
                f.write(f"image_folder = {settings['image_folder']}\n")
                f.write(f"number_of_images_in_analysis_range = {n}\n")
                f.write(f"start_frame_1based = {start_idx + 1}\n")
                f.write(f"end_frame_1based = {end_idx + 1}\n")
                f.write(f"number_of_groups = {len(roi_groups)}\n")
                f.write(f"tracking_preset = {settings['tracking_preset']}\n")
                f.write(f"search_radius_base_px = {search_radius_base}\n")
                f.write(f"hard_corr = {hard_corr}\n")
                f.write(f"soft_corr = {soft_corr}\n")
                f.write(f"enable_adaptive = {enable_adaptive}\n")
                f.write(f"use_prev_frame_template = {use_prev_frame_template}\n")
                f.write(f"template_policy = {'experimental_follow' if use_prev_frame_template else 'fixed_reference'}\n")
                f.write(f"template_alpha = {template_alpha}\n")
                f.write(f"max_frame_strain_jump = {max_frame_jump}\n")
                f.write(f"enable_fb_check = {enable_fb_check}\n")
                f.write(f"fb_tolerance_px = {fb_tolerance}\n")
                f.write(f"pixel_size_mm = {pixel_size_mm}\n")
                f.write(f"overlay_every = {overlay_every}\n")
                f.write(f"export_publication_figures = {export_publication_figures}\n")
                f.write("\nGroups:\n")
                for g in roi_groups:
                    f.write(
                        f"{g['name']}: role={normalize_roi_role(g.get('role', 'none'))}, "
                        f"selected={g['selected_mode']}, actual={g['actual_mode']}, "
                        f"L0={g['L0']:.6f}px, dx={g['dx0']:.6f}px, dy={g['dy0']:.6f}px, "
                        f"roi1={g['roi1']}, roi2={g['roi2']}\n"
                    )
            written_paths.append(params_path)

            acceptance_path = param_dir / "acceptance_summary.txt"
            with open(acceptance_path, "w", encoding="utf-8") as f:
                f.write("Acceptance summary by group\n")
                f.write("---------------------------\n")
                for gname, sub in df.groupby("group"):
                    f.write(f"\n[{gname}]\n")
                    f.write(str(sub["accept_mode"].value_counts(dropna=False)))
                    f.write("\nRejected frames:\n")
                    rejected = sub[sub["accepted"] == False]
                    if len(rejected) == 0:
                        f.write("None\n")
                    else:
                        f.write(rejected[["frame_global_1based", "filename", "reason"]].to_string(index=False))
                        f.write("\n")
            written_paths.append(acceptance_path)

        n_rejected = summary["overall"]["rejected_frames"]
        n_adaptive = summary["overall"]["adaptive_accepted_frames"]
        qc_level = summary["overall"]["qc_level"]

        path_log = "输出文件：\n" + "\n".join(str(p) for p in written_paths)
        mean_export_note = ""
        if export_origin_txt:
            mean_export_note = (
                "\n平均应变文件: core\\strain_mean_groups.txt"
                "\n平均应变列也已写入: core\\strain_all_groups.txt"
            )
        done_msg = (
            f"处理完成。\n"
            f"核心结果已保存到: {core_dir if (export_origin_txt or export_engineering_png or export_origin_opju) else output_dir}\n"
            f"论文级图表包: {'已导出到 ' + str(publication_dir) if export_publication_figures else '未勾选'}\n"
            f"QC 状态：{qc_level}\n"
            f"拒绝帧数：{n_rejected}\n"
            f"自适应接受帧数：{n_adaptive}"
            f"{mean_export_note}"
        )

        self.post_to_ui(lambda: self._update_progress(100))
        self.post_to_ui(lambda: self.status_var.set(f"处理完成，QC 状态：{qc_level}"))
        self.post_to_ui(lambda s=summary: self.update_qc_overview(s))
        self.post_to_ui(lambda: self.log(done_msg + "\n" + path_log))
        self.post_to_ui(lambda: self.show_completion_and_open_output_folder(done_msg, output_dir))

        # === Tier 0: populate in-app viewer with final results ===
        try:
            if df is not None and not df.empty and roi_groups:
                self.post_to_ui(lambda: self.show_results_viewer(df, roi_groups))
        except Exception:
            # Viewer failure must never break the main success path
            self.post_to_ui(lambda: self.log("结果预览窗口初始化失败（不影响已导出文件）"))

    # ==========================
    # Tier 0: In-app interactive results viewer
    # ==========================

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
        options = field.get("display_options") or {}
        if options:
            self.dic_view_background.set({"none": "无底图", "reference": "参考图", "deformed": "变形图"}.get(options.get("background", "none"), "无底图"))
            self.dic_color_mode.set({"range": "数据范围", "symmetric": "零点对称", "manual": "手动范围"}.get(options.get("color_mode", "range"), "数据范围"))
            self.dic_display_style.set("测量点" if options.get("style") == "points" else "连续云图")
            self.dic_percent.set(bool(options.get("percent", False)))
            self.dic_colormap.set(options.get("cmap", "RdBu_r"))
            self.dic_color_min.set(str(options.get("vmin", "")))
            self.dic_color_max.set(str(options.get("vmax", "")))
        if "deformed_image" not in field and self.dic_last_image is not None:
            field["deformed_image"] = self.dic_last_image
        self._viewer_kind = "fullfield"
        if component:
            self.dic_field_component.set(component)
        if self.dic_field_component.get() not in DIC_FIELD_COMPONENTS:
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

    def _update_field_viewer_overlay(self):
        if self.dic_last_field is None or self.dic_last_image is None:
            return
        try:
            component = str(self.dic_field_component.get() or "u")
            if component not in DIC_FIELD_COMPONENTS:
                component = "u"
                self.dic_field_component.set(component)
            overlay = overlay_dic_field_on_image(
                self.dic_last_image,
                self.dic_last_field,
                component=component,
                options={"coordinate_frame": "deformed", "background": "none"},
            )
            self.auto_fit_enabled = False
            self.display_img = overlay
            self._canvas_shows_field_overlay = True
            _height, width = overlay.shape[:2]
            orig_h, orig_w = np.asarray(self.dic_last_image).shape[:2]
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
        if component not in DIC_FIELD_COMPONENTS:
            component = "u"
        cmap = matplotlib.colormaps["viridis" if component == "zncc" else self.dic_colormap.get()].copy()
        cmap.set_bad(self.panel_bg)
        mesh = render_dic_field_on_axes(ax, self.dic_last_field, component, cmap=cmap)
        values = np.asarray(self.dic_last_field[component], dtype=float)
        unit = "px" if component in ("u", "v") else ("%" if self.dic_percent.get() and component != "zncc" else "1")
        # Keep the equal-aspect specimen centered and its colorbar adjacent,
        # including very tall ROIs in a wide independently resized window.
        cax = make_axes_locatable(ax).append_axes("right", size=0.18, pad=0.24)
        cbar = fig.colorbar(mesh, cax=cax)
        _core.style_dic_colorbar(cbar, get_plot_preset("raw_inspection"), f"{component} ({unit})")
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
            values=list(DIC_FIELD_COMPONENTS),
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
            combo.grid(row=0,column=column, padx=(0,5), sticky="w")
            self.add_tooltip(combo, tip, choices=choices)
        scales = ttk.Frame(ctrl, style="Card.TFrame")
        scales.grid(row=2,column=0,columnspan=2,sticky="ew")
        ttk.Label(scales,text="色标").grid(row=0,column=0,sticky="w",padx=(0,5))
        for column, variable in enumerate((self.dic_color_min,self.dic_color_max), start=1):
            entry = ttk.Entry(scales,textvariable=variable,width=9)
            entry.grid(row=0,column=column,padx=(0,5))
            limit = "最小" if column == 1 else "最大"
            self.add_tooltip(entry, f"填写颜色刻度的{limit}值，只在“手动范围”下生效，填好后点“应用”。最小值必须小于最大值；移动量用像素，伸缩值勾选“应变 %”后用百分数。")
        percent_check = ttk.Checkbutton(scales,text="应变 %",variable=self.dic_percent)
        percent_check.grid(row=0,column=3,padx=(0,6))
        self.add_tooltip(percent_check, "勾选后，伸缩值按百分数显示，例如 0.01 显示为 1%。取消后显示原始比例；点击“应用”更新，移动量仍以像素计。")
        apply_button = ttk.Button(scales,text="应用",command=self.apply_dic_display_options,style="Compact.TButton")
        apply_button.grid(row=0,column=4)
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

    # ========== 导出预设 ==========

    def _apply_research_preset(self):
        """科研推荐组合：核心 TXT + PNG + QC + 参数（最常用）"""
        self.export_origin_txt.set(True)
        self.export_engineering_png.set(True)
        self.export_qc_summary.set(True)
        self.export_parameters.set(True)
        self.export_full_csv.set(False)
        self.export_corr_plot.set(False)
        self.export_overlays.set(False)
        self.export_publication_figures.set(False)
        self.export_origin_opju.set(False)
        self.log("已应用「科研推荐」导出预设")

    def _apply_quick_view_preset(self):
        """快速查看：只保留最常用的可视化结果"""
        self.export_origin_txt.set(False)
        self.export_engineering_png.set(True)
        self.export_qc_summary.set(True)
        self.export_parameters.set(False)
        self.export_full_csv.set(False)
        self.export_corr_plot.set(False)
        self.export_overlays.set(False)
        self.export_publication_figures.set(False)
        self.export_origin_opju.set(False)
        self.log("已应用「快速查看导出」预设")

    def _apply_all_export_preset(self):
        """全部勾选（用于最完整的存档）"""
        for var in [
            self.export_origin_txt, self.export_origin_opju,
            self.export_engineering_png, self.export_qc_summary,
            self.export_full_csv, self.export_corr_plot,
            self.export_overlays, self.export_parameters,
            self.export_publication_figures,
        ]:
            var.set(True)
        self.log("已应用「全量复核导出」预设（注意文件会较多）")

    def _detect_poisson_capable(self, groups, df):
        roles = {normalize_roi_role(g.get("role", "none")) for g in groups}
        has_axial = "axial" in roles
        has_trans = "transverse" in roles
        if not (has_axial and has_trans):
            return False
        # 再检查数据里是否有对应的列（兼容旧结果）
        return "AxialEngineeringStrain" in df.columns or "PoissonRatio" in df.columns

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
        colors = ["#56b4e9", "#e69f00", "#63d2ab", "#df9bd0", "#cbd5e1"] if self.dark_mode.get() else PLOT_COLOR_CYCLE
        if self._viewer_mode == "poisson" and self._has_poisson:
            table = build_poisson_ratio_table(df, groups)
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
        return ("这是移动量或伸缩量的彩色结果图，数值与单位在图旁的颜色刻度上。空白表示未算出有效结果；用下方图标移动、放大或保存。"
                if self._viewer_kind == "fullfield" else
                "横轴是图片序号，纵轴是伸缩比例或泊松比（横向收缩与纵向伸长的比值）。曲线断开表示没有有效结果；用下方图标移动、放大或保存。")

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
            except Exception:
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
            except Exception:
                pass
            self.viewer_export_btn.config(state=tk.DISABLED)
            self.viewer_clear_btn.config(state=tk.DISABLED)

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# GUI compatibility exports
# ---------------------------------------------------------------------------
# Keep these assignments at module scope so the established GUI import path
# remains valid while every numerical/field-export entry point resolves to the
# Tk-free implementation.  ``MultiROIGUI`` continues to own widgets, state,
# callbacks, and viewer lifecycle only.
CoreError = _core.CoreError
resolve_code_paths = _core.resolve_code_paths
resolve_source_paths = _core.resolve_source_paths
generate_synthetic_speckle = _core.generate_synthetic_speckle
warp_image_translation = _core.warp_image_translation
warp_image_deformation_gradient = _core.warp_image_deformation_gradient
green_lagrange_from_F = _core.green_lagrange_from_F
read_gray_image = _core.read_gray_image
collect_images = _core.collect_images
image_sequence_fingerprint = _core.image_sequence_fingerprint
normalize_to_uint8 = _core.normalize_to_uint8
get_display_image = _core.get_display_image
integer_cc_guess = _core.integer_cc_guess
match_template_candidate = _core.match_template_candidate
match_template_candidate_diagnostic = _core.match_template_candidate_diagnostic
extract_patch_subpixel = _core.extract_patch_subpixel
update_template_from_rect = _core.update_template_from_rect
forward_backward_error = _core.forward_backward_error
initialize_extensometer_group_state = _core.initialize_extensometer_group_state
track_extensometer_group_frame = _core.track_extensometer_group_frame
field_quality_summary = _core.field_quality_summary
DEFAULT_PEAK_MARGIN_MIN = _core.DEFAULT_PEAK_MARGIN_MIN
DEFAULT_PEAK_RATIO_MIN = _core.DEFAULT_PEAK_RATIO_MIN
DEFAULT_MAX_HESSIAN_CONDITION_NUMBER = _core.DEFAULT_MAX_HESSIAN_CONDITION_NUMBER
DEFAULT_MIN_CORRELATION_VALID_FRACTION = _core.DEFAULT_MIN_CORRELATION_VALID_FRACTION
DEFAULT_MIN_STRAIN_VALID_FRACTION = _core.DEFAULT_MIN_STRAIN_VALID_FRACTION
build_poi_grid = _core.build_poi_grid
poi_grid_is_usable = _core.poi_grid_is_usable
rect_is_inside_image = _core.rect_is_inside_image
validate_image_sequence_dimensions = _core.validate_image_sequence_dimensions
fullfield_field_has_finite_strain = _core.fullfield_field_has_finite_strain
roi_texture_metrics = _core.roi_texture_metrics
texture_is_ok = _core.texture_is_ok
texture_failure_code = _core.texture_failure_code
require_texture = _core.require_texture
TEXTURE_METRICS_VERSION = _core.TEXTURE_METRICS_VERSION
TEXTURE_DISCRIMINATOR_VERSION = _core.TEXTURE_DISCRIMINATOR_VERSION
TEXTURE_PREFLIGHT_VERSION = _core.TEXTURE_PREFLIGHT_VERSION
compute_reference_normalization = _core.compute_reference_normalization
compute_reference_normalization_bounds = _core.compute_reference_normalization_bounds
normalize_with_bounds = _core.normalize_with_bounds
normalize_sequence_frames = _core.normalize_sequence_frames
sha256_file = _core.sha256_file
file_identity = _core.file_identity
ordered_input_manifest = _core.ordered_input_manifest
canonicalize_json = _core.canonicalize_json
canonical_json_bytes = _core.canonical_json_bytes
canonical_json_hash = _core.canonical_json_hash
collect_environment = _core.collect_environment
code_fingerprint = _core.code_fingerprint
refine_subset_icgn = _core.refine_subset_icgn
refine_subset_iclm = _core.refine_subset_iclm
compute_strain_fields = _core.compute_strain_fields
run_2d_dic = _core.run_2d_dic
run_2d_dic_sequence = _core.run_2d_dic_sequence
dic_field_to_dataframe = _core.dic_field_to_dataframe
write_dic_field_txt = _core.write_dic_field_txt
write_dic_field_parameters = _core.write_dic_field_parameters
render_dic_field_on_axes = _core.render_dic_field_on_axes
plot_dic_field_map = _core.plot_dic_field_map
add_dic_colorbar = _core.add_dic_colorbar
dic_color_limits = _core.dic_color_limits
overlay_dic_field_on_image = _core.overlay_dic_field_on_image
export_dic_field_outputs = _core.export_dic_field_outputs
build_core_strain_table = _core.build_core_strain_table
write_origin_txt = _core.write_origin_txt
build_mean_strain_table = _core.build_mean_strain_table
build_poisson_ratio_table = _core.build_poisson_ratio_table
build_all_groups_strain_table = _core.build_all_groups_strain_table
write_all_groups_origin_txt = _core.write_all_groups_origin_txt
write_mean_groups_origin_txt = _core.write_mean_groups_origin_txt
write_poisson_ratio_txt = _core.write_poisson_ratio_txt
build_origin_project_tables = _core.build_origin_project_tables
build_qc_summary = _core.build_qc_summary
write_qc_summary = _core.write_qc_summary
plot_engineering_strain = _core.plot_engineering_strain
plot_all_groups_engineering_strain = _core.plot_all_groups_engineering_strain
plot_poisson_ratio = _core.plot_poisson_ratio
plot_correlation_scores = _core.plot_correlation_scores


def main():
    enable_windows_dpi_awareness()
    root = tk.Tk()
    app = MultiROIGUI(root)
    app.run()


if __name__ == "__main__":
    main()
