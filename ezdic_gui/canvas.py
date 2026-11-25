"""Interactive image viewport, zoom/pan math, ROI drawing, and mask preview."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import ezdic_core as _core


class CanvasMixin:
    """Image canvas rendering, aspect-preserving zoom, ROI drawing, and mask preview."""

    def _draw_image_empty_state(self):
        if not hasattr(self, "canvas") or self.display_img is not None:
            return
        self.canvas.delete("empty_state")
        cx, cy = self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2
        self.canvas.create_text(cx, cy - 24, text="加载图像序列", fill="#e2e8f0",
                                font=(self.ui_font_family, 16, "bold"), tags="empty_state")
        self.canvas.create_text(cx, cy + 24, text="点击上方“选择并加载图像”，打开一组连续拍摄的图片。\n支持 TIFF、PNG、JPEG 和 BMP",
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
        if not hasattr(self, "image_context_var") or not self.image_paths or self.preview_img8 is None:
            return
        if self._canvas_shows_field_overlay and self.dic_last_image is not None:
            h, w = self.dic_last_image.shape[:2]
            text = (
                f"结果叠加 · {self.dic_field_component.get()} · 第 {self.dic_last_frame_1based} 帧 · "
                f"{self.dic_last_filename} · {w} × {h} px · 参考帧 {self.dic_last_reference_frame_1based}"
            )
        else:
            start, _ = self._safe_int_var(self.start_frame_1based)
            filename = os.path.basename(self.image_paths[self.current_preview_index])
            h, w = self.preview_img8.shape[:2]
            if self.current_preview_index + 1 != start:
                action = "画测量区域前请回到参考图"
            elif self._drawing_mask_exclusion:
                action = "正在绘制排除区"
            else:
                action = "绘制分析区域" if self.is_fullfield_mode() else f"绘制测量框 {self.current_roi_index}"
            text = (
                f"{filename} · 第 {self.current_preview_index + 1}/{len(self.image_paths)} 帧 · "
                f"{w} × {h} px · 参考帧 {start if start is not None else '待校验'} · {action}"
            )
        self.image_context_var.set(text)

    def select_dic_mask(self):
        if self.is_processing or self._completion_pending:
            return
        path = filedialog.askopenfilename(
            title="选择参考图像遮罩（白色保留）",
            filetypes=[("遮罩图像", "*.png *.tif *.tiff *.bmp")],
        )
        if path:
            self.dic_mask_path.set(path)
            self.dic_mask_mode.set("导入遮罩")
            self.preview_dic_mask()

    def preview_dic_mask(self):
        if self.is_processing or self._completion_pending or self.field_roi is None:
            return
        try:
            if self.field_roi_reference_frame_1based is None:
                raise RuntimeError("全场 ROI 缺少绘制参考帧记录，请先在参考帧重画全场 ROI。")
            index = int(self.field_roi_reference_frame_1based) - 1
            if not self.image_paths or index < 0 or index >= len(self.image_paths):
                raise RuntimeError("全场 ROI 的参考帧超出已加载图像序列范围。")
            start, start_err = self._safe_int_var(self.start_frame_1based)
            if start_err is None and start - 1 == index:
                raw, reference = self._get_reference_frame_images(index)
            else:
                raw = _core.read_gray_image(self.image_paths[index])
                reference = _core.normalize_to_uint8(raw)
            mask, record, _ = _core._resolve_specimen_mask(
                reference,
                self.field_roi,
                {"mask": self.dic_mask_settings()},
            )
            self.current_preview_index = index
            self.preview_frame_1based.set(index + 1)
            self.preview_raw = raw
            self.preview_img8 = reference
            self.current_fullres_img8 = reference.copy()
            self._canvas_shows_field_overlay = False
            rgb = cv2.cvtColor(reference, cv2.COLOR_GRAY2RGB)
            if mask is not None:
                rgb[~mask] = (0.45 * rgb[~mask] + 0.55 * np.array([220, 55, 55])).astype(np.uint8)
            self._canvas_mask_overlay_rgb = rgb
            if getattr(self, "auto_fit_enabled", True):
                self.display_img = None
                self._rescale_display_to_current_size()
                if self.display_img is None:
                    height, width = rgb.shape[:2]
                    scale = min(1.0, 1280.0 / max(width, 1), 820.0 / max(height, 1))
                    disp_w = max(1, int(round(width * scale)))
                    disp_h = max(1, int(round(height * scale)))
                    self.display_img = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                    self.display_scale = disp_w / max(width, 1)
                    self.zoom_factor = self.display_scale
                    self.show_image()
            else:
                self._apply_manual_zoom()
            self._show_image_workspace()
            self.log(
                f"遮罩预览：{record.get('included_pixels', reference.size)} 个保留像素；"
                f"排除区 {len(self.dic_mask_exclusions)} 个。请核对孔洞、背景和试样边缘。"
            )
        except Exception as exc:
            messagebox.showerror("遮罩无效", str(exc))

    def draw_dic_exclusion(self):
        if self.field_roi is None or self.is_processing or self._completion_pending:
            return
        self.set_roi_mode(1)
        if self.current_preview_index + 1 != self.field_roi_reference_frame_1based:
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
            except tk.TclError:
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

        def on_image_frame_configure(_event):
            if self.current_fullres_img8 is None:
                return
            if getattr(self, "_fitting_display", False):
                return
            if getattr(self, "auto_fit_enabled", True) and self.display_img is not None:
                try:
                    cw = int(self.canvas.winfo_width())
                    ch = int(self.canvas.winfo_height())
                except tk.TclError:
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
                except tk.TclError:
                    pass
            self._resize_after_id = self.root.after(180, self._rescale_display_to_current_size)

        # 绑在 image_frame 上更稳（它会随窗口变化）
        self.image_frame.bind("<Configure>", on_image_frame_configure, add="+")
        self.canvas.bind("<Configure>", on_image_frame_configure, add="+")

    def _canvas_source_rgb(self):
        if (
            getattr(self, "_canvas_mask_overlay_rgb", None) is not None
            and self.is_fullfield_mode()
            and not getattr(self, "_canvas_shows_field_overlay", False)
        ):
            return self._canvas_mask_overlay_rgb
        if self.current_fullres_img8 is None:
            return None
        return cv2.cvtColor(self.current_fullres_img8, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _zoom_bounds(width, height):
        max_dim = max(int(width), int(height), 1)
        min_zoom = min(1.0, max(0.05, 50.0 / max_dim))
        max_zoom = max(1.0, min(8.0, 6144.0 / max_dim))
        return min_zoom, max_zoom

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
        except tk.TclError:
            cw, ch = 900, 620

        target_w = max(200, cw - 8)
        target_h = max(150, ch - 8)

        mask_rgb = (
            self._canvas_mask_overlay_rgb
            if (
                getattr(self, "_canvas_mask_overlay_rgb", None) is not None
                and self.is_fullfield_mode()
                and not getattr(self, "_canvas_shows_field_overlay", False)
            )
            else None
        )
        if mask_rgb is not None:
            mh, mw = mask_rgb.shape[:2]
            new_scale = min(1.0, target_w / max(mw, 1), target_h / max(mh, 1))
            if new_scale < 1.0:
                disp_w = max(1, int(mw * new_scale))
                disp_h = max(1, int(mh * new_scale))
                new_disp = cv2.resize(mask_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            else:
                new_disp = mask_rgb.copy()
                new_scale = 1.0
        else:
            new_disp, new_scale = _core.get_display_image(
                self.current_fullres_img8,
                max_w=target_w,
                max_h=target_h,
            )

        old_h, old_w = (self.display_img.shape[:2] if self.display_img is not None else (0, 0))
        if self.display_img is None or abs(new_disp.shape[0] - old_h) > 2 or abs(new_disp.shape[1] - old_w) > 2:
            self.display_img = new_disp
            self.display_scale = new_scale
            self.zoom_factor = new_scale  # 同步 zoom_factor
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

    def zoom_image(self, factor):
        """按倍率手动缩放（支持工具栏 +/- 按钮）。"""
        if self.current_fullres_img8 is None or getattr(self, "is_processing", False) or getattr(self, "_completion_pending", False):
            return

        h, w = self.current_fullres_img8.shape[:2]
        min_zoom, max_zoom = self._zoom_bounds(w, h)
        self.auto_fit_enabled = False
        new_factor = max(min_zoom, min(max_zoom, self.zoom_factor * factor))
        self.zoom_factor = new_factor

        self._apply_manual_zoom()

    def _apply_manual_zoom(self):
        """根据当前 zoom_factor 重新生成显示图像。"""
        if self.current_fullres_img8 is None:
            return

        h, w = self.current_fullres_img8.shape[:2]
        min_zoom, max_zoom = self._zoom_bounds(w, h)
        self.zoom_factor = max(min_zoom, min(max_zoom, float(self.zoom_factor)))
        target_w = max(1, int(round(w * self.zoom_factor)))
        target_h = max(1, int(round(h * self.zoom_factor)))

        source_rgb = self._canvas_source_rgb()
        interp = cv2.INTER_AREA if self.zoom_factor <= 1.0 else cv2.INTER_LINEAR
        rgb = cv2.resize(source_rgb, (target_w, target_h), interpolation=interp)
        self._canvas_shows_field_overlay = False

        self.display_img = rgb
        self.display_scale = target_w / max(w, 1)
        self.zoom_factor = self.display_scale
        self.show_image()
        self._update_preview_scale_label()
        self._update_zoom_label()

    def fit_image_to_view(self):
        """强制将当前图像适应当前图像区大小，并恢复自动适应模式。"""
        if self.current_fullres_img8 is None or getattr(self, "is_processing", False) or getattr(self, "_completion_pending", False):
            return
        self.auto_fit_enabled = True
        self._canvas_shows_field_overlay = False
        self.display_img = None
        self._rescale_display_to_current_size()
        self._update_preview_scale_label()
        self._update_zoom_label()

    def show_image_1to1(self):
        """以 1:1 原始像素比例显示，并进入手动缩放模式。"""
        if self.current_fullres_img8 is None or getattr(self, "is_processing", False) or getattr(self, "_completion_pending", False):
            return
        self.auto_fit_enabled = False
        self.zoom_factor = 1.0

        source_rgb = self._canvas_source_rgb()
        self._canvas_shows_field_overlay = False
        self.display_img = source_rgb.copy()
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
        with self._state_lock_ctx():
            busy = bool(getattr(self, "is_processing", False) or getattr(self, "_completion_pending", False))
        can_use = has_image and not busy
        if hasattr(self, "image_action_button"):
            self.image_action_button.configure(
                text="回到参考图" if self.image_paths else "选择并加载图像",
                state=tk.DISABLED if busy else tk.NORMAL,
            )
        for attr in ("btn_zoom_in", "btn_zoom_out", "btn_fit", "btn_1to1"):
            if hasattr(self, attr):
                try:
                    getattr(self, attr).config(state=tk.NORMAL if can_use else tk.DISABLED)
                except tk.TclError:
                    pass

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
            self.display_img, self.display_scale = _core.get_display_image(
                self.current_fullres_img8, max_w=1280, max_h=820
            )
            self.zoom_factor = self.display_scale
            self.show_image()

    def _on_mouse_wheel(self, event):
        """支持鼠标滚轮缩放，围绕鼠标指针位置进行（专业图像工具标准行为）。"""
        if (
            self.current_fullres_img8 is None
            or self.display_img is None
            or getattr(self, "is_processing", False)
            or getattr(self, "_completion_pending", False)
        ):
            return

        # 确定缩放方向（跨平台）
        if event.num == 4 or event.delta > 0:
            factor = 1.2
        elif event.num == 5 or event.delta < 0:
            factor = 1 / 1.2
        else:
            return

        # 计算鼠标在当前显示图上的位置。canvasx/canvasy 会考虑当前滚动偏移。
        view_x, view_y = event.x, event.y
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # 鼠标在原始图像坐标中的位置（使用当前 display_scale）
        inv_scale = 1.0 / max(float(self.display_scale), 1e-9)
        img_x = cx * inv_scale
        img_y = cy * inv_scale

        # 先计算并校验目标缩放与尺寸，避免提前改写 zoom_factor 造成失步
        orig_h, orig_w = self.current_fullres_img8.shape[:2]
        min_zoom, max_zoom = self._zoom_bounds(orig_w, orig_h)
        new_zoom = max(min_zoom, min(max_zoom, self.zoom_factor * factor))
        new_disp_w = max(1, int(round(orig_w * new_zoom)))
        new_disp_h = max(1, int(round(orig_h * new_zoom)))
        if new_disp_w < 30 and new_disp_h < 30:
            return

        self.auto_fit_enabled = False
        source_rgb = self._canvas_source_rgb()
        interp = cv2.INTER_AREA if new_zoom <= 1.0 else cv2.INTER_LINEAR
        self.display_img = cv2.resize(source_rgb, (new_disp_w, new_disp_h), interpolation=interp)
        self.display_scale = new_disp_w / max(orig_w, 1)
        self.zoom_factor = self.display_scale
        self._canvas_shows_field_overlay = False

        self.show_image()
        self._update_preview_scale_label()
        self._update_zoom_label()

        # 尝试保持鼠标指向的原始位置在缩放后仍大致在鼠标附近，兼容居中 letterbox 偏移
        new_cx = img_x * self.display_scale
        new_cy = img_y * self.display_scale
        try:
            pad_x = max(0, (self.canvas.winfo_width() - new_disp_w) // 2)
            pad_y = max(0, (self.canvas.winfo_height() - new_disp_h) // 2)
            total_w = new_disp_w + 2 * pad_x
            total_h = new_disp_h + 2 * pad_y
            self.canvas.xview_moveto(max(0.0, min(1.0, (new_cx - view_x + pad_x) / total_w)))
            self.canvas.yview_moveto(max(0.0, min(1.0, (new_cy - view_y + pad_y) / total_h)))
        except (tk.TclError, ZeroDivisionError):
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
            self.dic_mask_exclusions = []
            self._drawing_mask_exclusion = False
            had_mask_preview = getattr(self, "_canvas_mask_overlay_rgb", None) is not None
            self._canvas_mask_overlay_rgb = None
            if had_mask_preview:
                self._restore_sequence_preview()
            else:
                self.show_image()
        else:
            self.roi1 = None
            self.roi2 = None
            self.roi1_reference_frame_1based = None
            self.roi2_reference_frame_1based = None
            self.show_image()
        self.log("已清除当前 ROI。")
        self.update_workflow_action_states()

    def on_mouse_down(self, event):
        if self.preview_img8 is None or self.display_img is None:
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
        if self.preview_img8 is None or self.drag_start is None:
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
        rx, ry, rw, rh = _core.rect_normalize(x0 * inv, y0 * inv, x1 * inv, y1 * inv)
        if self.is_fullfield_mode() and self._drawing_mask_exclusion:
            self._drawing_mask_exclusion = False
            if rw >= 1 and rh >= 1:
                self.dic_mask_exclusions.append(_core.clamp_rect((rx, ry, rw, rh), self.preview_img8.shape))
                self.preview_dic_mask()
                self.update_workflow_action_states()
            return

        if rw < 15 or rh < 15:
            messagebox.showwarning("ROI 太小", "ROI 太小。建议至少 30×30 px，且包含清晰散斑纹理。")
            return

        rect = _core.clamp_rect((rx, ry, rw, rh), self.preview_img8.shape)
        reference_frame_1based = self.current_preview_index + 1

        if self.is_fullfield_mode():
            self.field_roi = rect
            self.dic_mask_exclusions = []
            self._drawing_mask_exclusion = False
            had_mask_preview = getattr(self, "_canvas_mask_overlay_rgb", None) is not None
            self._canvas_mask_overlay_rgb = None
            self.field_roi_reference_frame_1based = reference_frame_1based
            self.log(f"全场 ROI = {rect}")
            self.log_roi_texture("全场 ROI", rect)
            self.status_var.set("全场 ROI 已设置。可开始 2D DIC 分析。")
            if had_mask_preview:
                self._restore_sequence_preview()
            else:
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
            for group in self.roi_groups:
                r1 = group["roi1"]
                r2 = group["roi2"]
                c1 = _core.rect_center(r1)
                c2 = _core.rect_center(r2)
                self.canvas.create_line(c1[0] * s, c1[1] * s, c2[0] * s, c2[1] * s, fill="lime", width=2)
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
        label = self.canvas.create_text(
            x, max(0, y), text=text, anchor="nw", fill="#ffffff",
            font=self.canvas_group_font,
        )
        x0, y0, x1, y1 = self.canvas.bbox(label)
        background = self.canvas.create_rectangle(
            x0 - 3, y0 - 2, x1 + 3, y1 + 2,
            fill="#111827", outline=color, width=1,
        )
        self.canvas.tag_lower(background, label)
