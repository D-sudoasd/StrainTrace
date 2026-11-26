"""Tk preview viewport with display-only transforms and raw pixel inspection."""

from __future__ import annotations

import math
import sys
import tkinter as tk
from tkinter import ttk

from PIL import Image as PillowImage, ImageTk

from .config import CaptureError
from .preview import preview_pixels


def refresh_window_surface(window):
    """Paint native child controls after window resizing on Windows."""
    if sys.platform != "win32":
        return
    import ctypes

    window.update_idletasks()
    hwnd = ctypes.c_void_p(window.winfo_id())
    flags = 0x0001 | 0x0004 | 0x0080 | 0x0100  # invalidate, erase, children, update now
    ctypes.windll.user32.RedrawWindow(hwnd, None, None, flags)
    try:
        ctypes.windll.dwmapi.DwmFlush()
    except OSError:
        pass


def repaint_after_resize(window):
    if sys.platform != "win32":
        return
    pending = None
    size = None

    def deferred():
        nonlocal pending
        pending = None
        if window.winfo_ismapped():
            refresh_window_surface(window)

    def resized(event):
        nonlocal pending, size
        if event.widget is not window or (event.width, event.height) == size:
            return
        size = event.width, event.height
        if pending:
            window.after_cancel(pending)
        pending = window.after(110, deferred)

    def destroyed(event):
        nonlocal pending
        if event.widget is window and pending:
            window.after_cancel(pending)
            pending = None

    window.bind("<Configure>", resized, add="+")
    window.bind("<Destroy>", destroyed, add="+")


class ImageViewport(ttk.Frame):
    def __init__(self, master, camera_id: str, title: str, select, inspect, **kwargs):
        super().__init__(master, style="CaptureCard.TFrame", **kwargs)
        self.camera_id = camera_id
        self._select = select
        self._inspect = inspect
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._title = ttk.Label(self, text=title, style="CaptureCard.TLabel", wraplength=0)
        self._title.grid(row=0, column=0, sticky="ew", padx=8, pady=(7, 5))
        self._title.bind("<Configure>", self._wrap_title)
        self.canvas = tk.Canvas(self, bg="#0a1117", highlightthickness=1,
                                highlightbackground="#354b5a", takefocus=True, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=8)
        self._caption = ttk.Label(self, text="等待图像", style="CaptureMono.TLabel", anchor="w")
        self._caption.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 6))
        self.bind("<Configure>", self._fit_caption, add="+")
        self._image = self._prepared = self._photo = None
        self.displayed_image = None
        self._image_key = None
        self._placeholder = "连接预览后可查看图像"
        self._empty_caption = "等待图像"
        self._frame_caption = None
        self._zoom = None
        self._offset = (0.0, 0.0)
        self._project = (1.0, 0.0, 0.0)
        self._drag_origin = None
        self._roi_anchor = None
        self._probe_coordinate = None
        self.roi = None
        self.region_editable = True
        self.guide = "Off"
        self.palette = "native"
        self._checkerboard = None
        self._calibration = None
        self._calibration_error = None
        self._draw_after = None
        self.canvas.bind("<Configure>", self._schedule_draw)
        self.canvas.bind("<Motion>", self._probe)
        self.canvas.bind("<Leave>", self._clear_probe)
        self.canvas.bind("<ButtonPress-1>", self._begin)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._finish)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.fit())
        self.canvas.bind("<MouseWheel>", lambda event: self._magnify(event, 1.25 if event.delta > 0 else 0.8))
        self.canvas.bind("<Button-4>", lambda event: self._magnify(event, 1.25))
        self.canvas.bind("<Button-5>", lambda event: self._magnify(event, 0.8))
        self.canvas.bind("<KeyPress-plus>", lambda _event: self._magnify(None, 1.25))
        self.canvas.bind("<KeyPress-minus>", lambda _event: self._magnify(None, 0.8))
        self.canvas.bind("<KeyPress-0>", lambda _event: self.fit())

    def set_active(self, active: bool):
        self._title.configure(style="CaptureSelected.TLabel" if active else "CaptureCard.TLabel")
        self.canvas.configure(highlightbackground="#6dddc1" if active else "#354b5a")

    def set_title(self, title: str):
        self._title.configure(text=title)

    def _fit_caption(self, event):
        if event.height < 125:
            if self._caption.winfo_manager():
                self._caption.grid_remove()
        elif not self._caption.winfo_manager():
            self._caption.grid()

    def _wrap_title(self, event):
        if int(self._title.cget("wraplength")) != event.width:
            self._title.configure(wraplength=event.width)

    def image(self, image, *, placeholder=None, features=None, empty_caption="等待图像", frame_caption=None):
        key = (id(image), self.palette, id(self._calibration))
        if (self._image_key == key and (placeholder is None or placeholder == self._placeholder)
                and empty_caption == self._empty_caption and frame_caption == self._frame_caption):
            return
        if image is not None and self._calibration is not None:
            from .calibration import calibration_geometry, require_matching_calibration

            try:
                if features is None:
                    raise CaptureError("CALIBRATION_IMAGE_UNAVAILABLE", "no recorded geometry for this image")
                geometry = calibration_geometry(image.pixels.shape, features, image.metadata)
                require_matching_calibration(self._calibration, geometry)
            except CaptureError:
                self._calibration = None
                self._calibration_error = "校正已关闭 · 原图"
                key = (id(image), self.palette, id(None))
        self._image_key = key
        self._image = image
        self._prepared = None
        self._empty_caption = empty_caption
        self._frame_caption = frame_caption
        if placeholder is not None:
            self._placeholder = placeholder
        self._set_caption()
        self._schedule_draw()

    def show_calibration(self, profile, *, image=None, features=None):
        self._calibration = profile
        self._calibration_error = None
        self._image_key = None
        self.image(self._image if image is None else image, features=features)

    def show_checkerboard(self, overlay):
        if overlay != self._checkerboard:
            self._checkerboard = overlay
            self._schedule_draw()

    @property
    def corrected(self):
        return self._calibration is not None

    def fit(self):
        self._zoom = None
        self._offset = (0., 0.)
        self._schedule_draw()

    def one_to_one(self):
        self._zoom = 1.0
        self._offset = (0., 0.)
        self._schedule_draw()

    def _schedule_draw(self, _event=None):
        if self._draw_after is None:
            self._draw_after = self.after_idle(self._draw_when_idle)

    def _draw_when_idle(self):
        self._draw_after = None
        self.draw()

    def _cancel_draw(self):
        if self._draw_after is not None:
            self.after_cancel(self._draw_after)
            self._draw_after = None

    def destroy(self):
        self._cancel_draw()
        super().destroy()

    def _set_caption(self):
        image = self._image
        info = self._empty_caption
        if image is not None:
            height, width = image.pixels.shape[:2]
            info = self._frame_caption or f"{width}×{height} · {image.pixel_format}"
            if self._calibration_error:
                info = self._calibration_error
            elif self._calibration is not None:
                info = "已校正预览"
            elif self._probe_coordinate is not None:
                x, y = self._probe_coordinate
                if x < width and y < height:
                    info = f"({x},{y}) {image.pixels[y, x]}"
        self._caption.configure(text=info)

    def draw(self):
        self._cancel_draw()
        self.displayed_image = None
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self._image is None:
            self.canvas.create_text(w / 2, h / 2, text=self._placeholder, fill="#a9bdc9",
                                    font=("Segoe UI", 11), width=max(80, w - 32))
            return
        if self._prepared is None:
            pixels = preview_pixels(self._image, palette=self.palette)
            if self._calibration is not None:
                from .calibration import undistort

                try:
                    pixels = undistort(pixels, self._calibration)
                    self._calibration_error = None
                except CaptureError:
                    self._calibration = None
                    self._calibration_error = "校正已关闭 · 原图"
                self._set_caption()
            self._prepared = PillowImage.fromarray(pixels)
        iw, ih = self._prepared.size
        zoom = self._zoom if self._zoom is not None else min(w / iw, h / ih)
        zoom = max(0.01, zoom)
        left = (w - iw * zoom) / 2 + self._offset[0]
        top = (h - ih * zoom) / 2 + self._offset[1]
        self._project = (zoom, left, top)
        bounds = (
            max(0, math.floor(-left / zoom)), max(0, math.floor(-top / zoom)),
            min(iw, math.ceil((w - left) / zoom)), min(ih, math.ceil((h - top) / zoom)),
        )
        x0, y0, x1, y1 = bounds
        if x1 > x0 and y1 > y0:
            tile = self._prepared.crop(bounds)
            tile = tile.resize((max(1, round((x1 - x0) * zoom)), max(1, round((y1 - y0) * zoom))),
                               PillowImage.Resampling.NEAREST if zoom >= 1 else PillowImage.Resampling.BILINEAR)
            self._photo = ImageTk.PhotoImage(tile, master=self.canvas)
            self.canvas.create_image(left + x0 * zoom, top + y0 * zoom, image=self._photo, anchor="nw")
            self.displayed_image = self._image
        if self.guide == "Crosshair":
            self.canvas.create_line(left + iw * zoom / 2, top, left + iw * zoom / 2, top + ih * zoom, fill="#69b5ac")
            self.canvas.create_line(left, top + ih * zoom / 2, left + iw * zoom, top + ih * zoom / 2, fill="#69b5ac")
        if self.guide == "Grid":
            for i in range(1, 4):
                self.canvas.create_line(left + iw * zoom * i / 4, top, left + iw * zoom * i / 4, top + ih * zoom,
                                        fill="#69b5ac", dash=(4, 4))
                self.canvas.create_line(left, top + ih * zoom * i / 4, left + iw * zoom, top + ih * zoom * i / 4,
                                        fill="#69b5ac", dash=(4, 4))
        if self._checkerboard is not None and self._calibration is None:
            utc, monotonic, points = self._checkerboard
            if (utc, monotonic) == (self._image.received_utc_ns, self._image.received_monotonic_ns):
                for x, y in points:
                    sx, sy = left + x * zoom, top + y * zoom
                    if 0 <= sx < w and 0 <= sy < h:
                        self.canvas.create_oval(sx - 2.5, sy - 2.5, sx + 2.5, sy + 2.5,
                                                fill="#78f6c3", outline="#17342c", tags="checkerboard")
        if self.roi and self._calibration is None:
            x, y, rw, rh = self.roi
            self.canvas.create_rectangle(left + x * zoom, top + y * zoom, left + (x + rw) * zoom,
                                         top + (y + rh) * zoom, outline="#f2d47a", width=2)

    def _source(self, event, *, clamp=False):
        if self._image is None:
            return None
        zoom, left, top = self._project
        x, y = math.floor((event.x - left) / zoom), math.floor((event.y - top) / zoom)
        h, w = self._image.pixels.shape[:2]
        if clamp:
            return min(max(x, 0), w - 1), min(max(y, 0), h - 1)
        return (x, y) if 0 <= x < w and 0 <= y < h else None

    def _probe(self, event):
        self._probe_coordinate = self._source(event)
        self._set_caption()

    def _clear_probe(self, _event):
        self._probe_coordinate = None
        self._set_caption()

    def _begin(self, event):
        self.canvas.focus_set()
        self._select(self.camera_id)
        if event.state & 0x1 and self._calibration is None and self.region_editable:
            self._roi_anchor = self._source(event)
        else:
            self._drag_origin = event.x, event.y, *self._offset

    def _drag(self, event):
        if self._roi_anchor is not None:
            if (point := self._source(event, clamp=True)) is None:
                return
            x, y = point
            ox, oy = self._roi_anchor
            self.roi = (min(x, ox), min(y, oy), abs(x - ox) + 1, abs(y - oy) + 1)
        elif self._drag_origin is not None:
            start_x, start_y, offset_x, offset_y = self._drag_origin
            self._offset = offset_x + event.x - start_x, offset_y + event.y - start_y
        self._schedule_draw()

    def _finish(self, _event):
        if self._roi_anchor is not None and self.roi:
            self._inspect(self.camera_id, self.roi)
        self._roi_anchor = self._drag_origin = None

    def _magnify(self, event, factor):
        if self._image is None:
            return
        h, w = self._image.pixels.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        px = event.x if event else cw / 2
        py = event.y if event else ch / 2
        current, left, top = self._project
        zoom = min(16.0, max(0.05, current * factor))
        self._zoom = zoom
        self._offset = (px - (px - left) / current * zoom - (cw - w * zoom) / 2,
                        py - (py - top) / current * zoom - (ch - h * zoom) / 2)
        self._schedule_draw()
