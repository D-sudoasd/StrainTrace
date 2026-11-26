"""Live curve inspection and verification of its linked source images."""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from collections import deque
from tkinter import ttk
from tkinter.font import nametofont

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .frame_search import format_receipt_utc
from .gui import ACCENT, BG, BORDER, ENTRY, FG, MUTED
from .live_curve import curve_segments, read_curve_frame
from .presentation import LOAD_SUPPORT_LABELS, LOAD_TIME_BASIS_LABELS, TRACKING_LABELS, ui_error
from .viewport import ImageViewport, repaint_after_resize


CURVE_HELP = {
    "应变–时间": "查看工程应变随收图时间的变化。底部短线标记有原图但应变无效的帧。",
    "载荷–时间": "查看已匹配的载荷随收图时间变化。底部短线可定位载荷缺失的帧。",
    "载荷–应变": "同一帧的工程应变和载荷组成一个点。线段按采集顺序连接，可查看加载与卸载路径。",
    "全场质量–时间": "查看通过应变计算的测点比例。橙色点对应需要复查全场质量的帧。",
}
_MISSING = "#efb477"
_SELECTED = "#f0dfa1"


def point_description(point):
    frame = point.frame
    lines = [f"帧 {frame.sequence} · {frame.camera_id}",
             f"收图 {format_receipt_utc(point.receipt_utc_ns)} UTC"]
    if point.strain_status is not None:
        quality = TRACKING_LABELS.get(point.strain_status, "请复查标距跟踪")
        if point.strain_pct is None:
            lines.append(f"工程应变无效 · {quality}")
        else:
            lines.append(f"工程应变 {point.strain_pct:.5g}% · {quality}")
    if point.force_status is not None:
        support = LOAD_SUPPORT_LABELS.get(point.force_status, "无可用载荷")
        if point.force_N is None:
            lines.append(f"载荷缺失 · {support}")
        else:
            basis = LOAD_TIME_BASIS_LABELS.get(point.time_basis)
            parts = [f"载荷 {point.force_N:.6g} N", support]
            if basis:
                parts.append(basis)
            if point.load_time_domain == "relative":
                parts.append("同步帧")
            lines.append(" · ".join(parts))
    if point.field_valid_pct is not None:
        status = "质量通过" if point.field_ok else "请复查全场质量"
        lines.append(f"全场有效应变点 {point.field_valid_pct:.3g}% · {status}")
    return "\n".join(lines)


def _bounds(values, minimum_padding):
    low, high = min(values), max(values)
    span = high - low
    pad = max(minimum_padding, .08 * span) if math.isfinite(span) else 0
    return (low - pad if math.isfinite(low - pad) else low,
            high + pad if math.isfinite(high + pad) else high)


def _fraction(value, bounds):
    low, high = bounds
    span = high - low
    if math.isfinite(span) and span > 0:
        return (value - low) / span
    return (value / 2 - low / 2) / (high / 2 - low / 2) if high > low else .5


def _tick(bounds, fraction):
    low, high = bounds
    value = low * (1 - fraction) + high * fraction
    span = high - low
    if not math.isfinite(span) or max(abs(low), abs(high)) >= 10**6:
        return f"{value:.3g}"
    return f"{value:.{max(0, min(6, math.ceil(-math.log10(span / 4)))) if span else 1}f}"


class LiveCurve(tk.Canvas):
    def __init__(self, parent, on_select, open_frame, *, history=False):
        super().__init__(parent, bg=ENTRY, highlightthickness=1, highlightbackground=BORDER, height=150, takefocus=True)
        self.points = deque(maxlen=500)
        self.mode = "strain-time"
        self._enabled = True
        self._empty_message = "全场结果显示在左侧图像上"
        self._font = nametofont("TkDefaultFont", root=self.winfo_toplevel())
        self._positions = {}
        self.selected = None
        self._hovered = None
        self._pointer = None
        self._on_select = on_select
        self._open_frame = open_frame
        self._history = history
        self.bind("<Configure>", lambda _event: self.draw())
        self.bind("<Motion>", self._restore_hover)
        self.bind("<Leave>", self._clear_hover)
        self.bind("<Button-1>", self._select_at)
        self.bind("<Double-Button-1>", self._open_at)
        self.bind("<Left>", lambda _event: self._step(-1))
        self.bind("<Right>", lambda _event: self._step(1))
        self.bind("<Home>", lambda _event: self._step(0))
        self.bind("<End>", lambda _event: self._step(0, last=True))
        self.bind("<Return>", self._open_selected)
        ToolTip(self, self._help, title="实时曲线")

    def configure_mode(self, mode, enabled=True, *, empty_message=None):
        self.mode, self._enabled = mode, enabled
        if empty_message is not None:
            self._empty_message = empty_message
        self._hovered = None
        self.draw()

    def clear(self):
        self.points.clear()
        self._pointer = None
        self.draw()

    def select_point(self, point):
        if point is None or point.frame.frame_id in self._positions:
            self._select(point)
            return True
        self._select(None)
        return False

    def draw(self):
        self.delete("all")
        self._positions = {}
        if self._enabled:
            self._draw_curve()
        else:
            self._empty(self._empty_message)
        if self.selected is not None and self.selected.frame.frame_id not in self._positions:
            self._select(None)
        self._hovered = self._near(*self._pointer) if self._pointer is not None else None
        self._draw_indicators()

    def _empty(self, text):
        self.create_text(self.winfo_width() / 2, self.winfo_height() / 2, text=text, fill=MUTED,
                         width=max(20, self.winfo_width() - 24))

    def _draw_curve(self):
        time_mode = self.mode != "force-strain"
        strain_axis = self.mode == "strain-time"
        field_axis = self.mode == "field-time"
        width, height = self.winfo_width(), self.winfo_height()
        lh = self._font.metrics("linespace")
        if width < 150 or height < 3 * lh + 20:
            if height >= lh + 10:
                self._empty("扩大窗口查看曲线")
            return
        groups = list(curve_segments(self.points, self.mode))
        y_values = [y for segment in groups for _, _, y in segment]
        x_values = ([p.elapsed_s for p in self.points if p.elapsed_s is not None] if time_mode else
                    [x for segment in groups for _, x, _ in segment])
        waiting = {"strain-time": "等待有效标距应变", "field-time": "等待全场质量结果",
                   "force-time": "等待可匹配的载荷", "force-strain": "等待同帧的有效载荷和应变"}[self.mode]
        if not x_values:
            self._empty(waiting)
            return
        x_bounds = _bounds(x_values, .02 if time_mode else .0005)
        y_bounds = _bounds(y_values, .0005 if strain_axis else .5) if y_values else None
        x_labels = [_tick(x_bounds, index / 3) for index in range(4)]
        y_count = max(1, min(3, (height - 2 * lh - 21) // (lh + 3)))
        y_labels = [_tick(y_bounds, 1 - index / y_count) for index in range(y_count + 1)] if y_bounds else []
        half_tick = max(self._font.measure(label) for label in x_labels) / 2
        left = max(half_tick + 5, max((self._font.measure(label) for label in y_labels), default=0) + 11)
        right, top, bottom = width - half_tick - 7, lh + 9, height - lh - 11
        if right - left < 20 or bottom <= top:
            return
        axis_label = "工程应变 (%)" if strain_axis else "有效应变点 (%)" if field_axis else "载荷 (N)"
        self.create_text(8, 3, anchor="nw", text=axis_label, fill=MUTED)
        for index, label in enumerate(y_labels):
            y = top + index / y_count * (bottom - top)
            self.create_line(left, y, right, y, fill=BORDER)
            self.create_text(left - 5, y, anchor="e", text=label, fill=FG)
        for index, label in enumerate(x_labels):
            x = left + index / 3 * (right - left)
            self.create_text(x, bottom + lh / 2 + 9, text=label, fill=FG)
        if not y_values:
            self._empty(waiting)
        for segment in groups:
            coords = []
            for point, x, y in segment:
                cx = left + _fraction(x, x_bounds) * (right - left)
                cy = bottom - _fraction(y, y_bounds) * (bottom - top)
                self._positions[point.frame.frame_id] = (point, cx, cy)
                coords.extend((cx, cy))
                review = field_axis and point.needs_review(self.mode)
                if review or len(segment) < 2 or len(y_values) <= 180:
                    self.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill=_MISSING if review else ACCENT,
                                     outline="", tags="quality" if review else "")
            if len(coords) >= 4:
                self.create_line(*coords, fill=ACCENT, width=2)
        self.tag_raise("quality")
        if time_mode:
            for point in self.points:
                if point.elapsed_s is not None and point.values(self.mode)[1] is None:
                    x = left + _fraction(point.elapsed_s, x_bounds) * (right - left)
                    self.create_line(x, bottom + 1, x, bottom + 7, width=2, fill=_MISSING)
                    self._positions[point.frame.frame_id] = (point, x, bottom + 4)

    def _near(self, px, py):
        radius = max(8, int(self.winfo_fpixels("1i") / 12))
        nearest, distance = None, radius * radius
        for point, x, y in self._positions.values():
            gap = (x - px) ** 2 + (y - py) ** 2
            if gap <= distance:
                nearest, distance = point, gap
        return nearest

    def _select(self, point):
        if self.selected != point:
            self.selected = point
            self._on_select(point)
            self._draw_indicators()

    def _draw_indicators(self):
        self.delete("cursor")
        for point, color in ((self._hovered, FG), (self.selected, _SELECTED)):
            if point is not None and point.frame.frame_id in self._positions:
                _, x, y = self._positions[point.frame.frame_id]
                self.create_oval(x - 6, y - 6, x + 6, y + 6, outline=color, width=2, tags="cursor")

    def _restore_hover(self, event):
        self._pointer = (event.x, event.y)
        point = self._near(*self._pointer)
        if point != self._hovered:
            self._hovered = point
            self._draw_indicators()

    def _clear_hover(self, _event):
        self._pointer = None
        self._hovered = None
        self._draw_indicators()

    def _select_at(self, event):
        ToolTip.dismiss_window(self.winfo_toplevel())
        self.focus_set()
        self._select(self._near(event.x, event.y))

    def _step(self, offset, *, last=False):
        options = [p for p in self.points if p.frame.frame_id in self._positions]
        if options:
            if not offset:
                point = options[-1] if last else options[0]
            elif self.selected not in options:
                point = options[-1] if offset < 0 else options[0]
            else:
                index = max(0, min(len(options) - 1, options.index(self.selected) + offset))
                point = options[index]
            self._select(point)
        return "break"

    def _open_selected(self, _event):
        if self.selected is not None:
            self._open_frame(self.selected)
        return "break"

    def _open_at(self, event):
        point = self._near(event.x, event.y)
        if point is not None:
            self._select(point)
            self._open_frame(point)

    def _help(self, event=None):
        keyboard = event is None or getattr(event, "keysym", "") not in ("", "??")
        point = self.selected if keyboard else self._near(event.x, event.y)
        actions = "点击选中。方向键可切换帧，回车或双击核对原图。"
        if point is not None:
            return f"{point_description(point)}\n\n{actions}"
        scope = "显示所选记录的当前段。" if self._history else "显示最近 500 张的分析结果。"
        return (scope + "失效或跳过的帧会断开曲线。"
                "时间视图的底部短线可选中缺测帧。" + actions)


class CurveFrameWindow:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent, bg=BG)
        self.top.title("核对曲线原图")
        self.top.geometry("760x600")
        self.top.minsize(490, 360)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.top.bind("<Destroy>", self._destroy, add="+")
        self._jobs = queue.SimpleQueue()
        self._request = self._loading = None
        self._timer = None
        self._error = None
        self.status = tk.StringVar(self.top)
        panel = ttk.Frame(self.top, padding=12)
        panel.pack(fill=tk.BOTH, expand=True)
        status = ttk.Label(panel, textvariable=self.status, style="CaptureMuted.TLabel", justify=tk.LEFT)
        status.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        wrap_label_on_resize(status)
        ToolTip(status, self._help, title="图像来源")
        self.view = ImageViewport(panel, "curve", "已保存图像", lambda _camera: None, lambda *_args: None)
        self.view.region_editable = False
        self.view.pack(fill=tk.BOTH, expand=True)
        ToolTip(self.view.canvas, "滚轮缩放，拖动平移，双击恢复适应窗口。", title="原图浏览")

    def _help(self, _event=None):
        if self._request is None:
            return "先在实时曲线上选中一帧。"
        frame = self._request.frame
        detail = f"{point_description(self._request)}\n\n{frame.frame_id}\n{frame.image_path}\nSHA-256 {frame.image_sha256}"
        return f"{ui_error(self._error)}\n\n{detail}" if self._error else detail

    def show(self, point):
        self._request = point
        self._error = None
        part = point.frame.image_part or "主图"
        self.top.title(f"核对曲线原图 · 帧 {point.frame.sequence}")
        self.view.set_title(f"帧 {point.frame.sequence} · {point.frame.camera_id} · {part}")
        self.view.image(None, placeholder="正在核对台账并读取原图")
        self.status.set("正在校验原图")
        self.top.lift()
        if self._loading is None:
            self._start()

    def _start(self):
        self._loading = self._request
        point = self._loading

        def read():
            try:
                pixels = read_curve_frame(point.frame)
            except Exception as exc:
                self._jobs.put((None, exc))
            else:
                self._jobs.put((pixels, None))

        threading.Thread(target=read, name="live-curve-frame", daemon=True).start()
        self._timer = self.top.after(40, self._poll)

    def _poll(self):
        self._timer = None
        try:
            pixels, error = self._jobs.get_nowait()
        except queue.Empty:
            self._timer = self.top.after(40, self._poll)
            return
        point, self._loading = self._loading, None
        if self._request != point:
            self._start()
            return
        self._error = error
        if error is not None:
            self.view.image(None, placeholder="无法核验曲线点的原图")
            self.status.set(ui_error(error, details=False))
        else:
            self.view.image(pixels)
            self.view.fit()
            self.status.set("图像已通过校验\n" + point_description(point))

    def _destroy(self, event):
        if event.widget is self.top and self._timer is not None:
            self.top.after_cancel(self._timer)
