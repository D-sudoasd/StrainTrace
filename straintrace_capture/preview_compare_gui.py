"""Read-only side-by-side comparison of two preview originals."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .frame_search import format_receipt_utc
from .presentation import ui_error
from .preview import QUALITY_ALERT_FRACTION, image_quality
from .preview_image import preview_region_mismatch, read_preview_image
from .viewport import ImageViewport, repaint_after_resize


_MISMATCH = {
    "camera": "两张图记录的相机身份不同。",
    "identity_unknown": "现有记录无法核对两张图的相机身份。",
    "layout": "图像尺寸、像素格式、有效位深或主图部件发生变化。",
    "sensor_window": "传感器偏移或采样参数发生变化。",
    "sensor_unknown": "缺少可比的传感器范围记录。",
}
_SENSOR_LABELS = {
    "offset_x": "横向偏移", "offset_y": "纵向偏移", "binning_x": "横向合并", "binning_y": "纵向合并",
    "decimation_x": "横向抽取", "decimation_y": "纵向抽取",
}


def _camera_name(device):
    if device.backend == "sim":
        return f"模拟散斑 {device.id.rsplit(':', 1)[-1]}"
    return f"{device.model} · {device.serial}" if device.serial else device.model


class PreviewCompareWindow:
    def __init__(self, parent, submit, current_frame, locate_camera):
        self.submit = submit
        self.current_frame = current_frame
        self.locate_camera = locate_camera
        self.sources = [None, None]
        self._loading = self._measurement = self._region = self._archive = None
        self._can_share = False
        self._message = ""
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("原图对照")
        self.top.geometry("880x570")
        self.top.minsize(680, 465)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.status = tk.StringVar(self.top)
        self._build()
        self._refresh()

    def _build(self):
        body = ttk.Frame(self.top, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        tools = ttk.Frame(body)
        tools.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.open_left = ttk.Button(tools, text="打开左图…", style="Small.TButton", command=lambda: self.choose_file(0))
        self.open_right = ttk.Button(tools, text="打开右图…", style="Small.TButton", command=lambda: self.choose_file(1))
        self.use_current = ttk.Button(tools, text="取相机当前帧", style="Small.TButton", command=self._take_current)
        self.clear_region = ttk.Button(tools, text="恢复全图", style="Small.TButton", command=self._reset_region)
        self.camera_button = ttk.Button(tools, text="定位相机", style="Small.TButton", command=self._locate_camera)
        for button in (self.open_left, self.open_right, self.use_current, self.clear_region):
            button.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(self.open_left, "打开用 Capture 保存的预览 TIFF。左图用于确定要对照的相机。")
        ToolTip(self.open_right, lambda _event: self._file_block(1) or "打开另一张保存的预览 TIFF，可离线对照。")
        ToolTip(self.use_current, self._current_help)
        ToolTip(self.clear_region, lambda _event: "重新使用这张归档原图的检查框。"
                if self._archive and self._region is None else "让此窗口里的图像查看和统计使用完整画幅。")
        ToolTip(self.camera_button, "到主窗口定位归档时的相机。有预览时打开这台相机的画质页。"
                "画幅相符时可沿用这里的检查框。"
                "调整后从主窗口的“原图对照”返回，再取一张预览。")
        status = ttk.Label(body, textvariable=self.status, justify=tk.LEFT, style="CaptureMuted.TLabel", width=1, takefocus=True)
        status.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        wrap_label_on_resize(status)
        ToolTip(status, lambda _event: self._message or self.status.get())

        wall = ttk.Frame(body)
        wall.grid(row=2, column=0, sticky="nsew")
        wall.columnconfigure((0, 1), weight=1, uniform="preview_compare")
        wall.rowconfigure(0, weight=1)
        self.views, self.metadata, self.metrics = [], [], []
        for index, title in enumerate(("左图", "右图")):
            panel = ttk.Frame(wall)
            panel.grid(row=0, column=index, sticky="nsew", padx=(0, 7) if index == 0 else 0)
            panel.columnconfigure(0, weight=1)
            panel.rowconfigure(0, weight=1)
            view = ImageViewport(panel, title, title, lambda _camera: None, lambda _camera, roi: self._set_region(roi))
            view.grid(row=0, column=0, sticky="nsew")
            ToolTip(view.canvas, lambda _event, tile=view: (
                "滚轮缩放，拖动查看细节。按住 Shift 拖动可让两图使用相同像素坐标的检查框。"
                if tile.region_editable else "滚轮缩放，拖动查看完整原图或局部细节。"))
            metadata = ttk.Label(panel, justify=tk.LEFT, style="CaptureMuted.TLabel", width=1, takefocus=True)
            metadata.grid(row=1, column=0, sticky="ew", pady=(5, 1))
            wrap_label_on_resize(metadata)
            ToolTip(metadata, lambda _event, slot=index: self._source_help(slot), title="原图来源")
            metric = ttk.Label(panel, justify=tk.LEFT, width=1, takefocus=True)
            metric.grid(row=2, column=0, sticky="ew", pady=(3, 0))
            wrap_label_on_resize(metric)
            ToolTip(metric, "亮度中位数按有效位深的满量程归一。近黑和近饱和分别用最低与最高 1% 的码值判断，"
                    "彩色图像任一颜色通道接近饱和就计入。大图按均匀像素间隔取样。")
            self.views.append(view)
            self.metadata.append(metadata)
            self.metrics.append(metric)

    def _source_help(self, index):
        if (frame := self.sources[index]) is None:
            return "选择原图后可查看相机、收图时间和文件路径。"
        window = "，".join(f"{_SENSOR_LABELS[name]} {value}" for name, value in frame.sensor_window.items())
        archive = self._archive if index == 0 else None
        provenance = (f"{archive.experiment_name} · 阶段 {archive.phase} · 原帧 {archive.sequence}\n"
                      f"实验编号 {archive.experiment_id}\n已核对 TIFF · SHA-256 {archive.sha256}\n") if archive else ""
        return (f"{_camera_name(frame.device)} · {frame.device.id}\n"
                f"{provenance}"
                f"主机收图 UTC {format_receipt_utc(frame.image.received_utc_ns, precision=3)}\n"
                f"{frame.image.pixel_format} · 有效位深 {frame.image.significant_bits} 位\n"
                f"{window or '无传感器范围读数'}\n{frame.path if frame.path is not None else '取自当前连接的相机预览'}")

    def _file_block(self, index):
        if self._loading is not None:
            return "正在读取原图。"
        if index and self.sources[0] is None:
            return "请先打开左图。"
        return ""

    def _current_sample(self):
        if self._loading is not None:
            return None, "正在读取原图。"
        if self.sources[0] is None:
            return None, "请先打开左图。"
        left = self.sources[0]
        current, reason = self.current_frame(left.device.id, archive=self._archive is not None)
        if self._archive and current is not None:
            mismatch = preview_region_mismatch(left, current)
            if mismatch in ("camera", "identity_unknown"):
                return None, _MISMATCH[mismatch]
        return current, reason

    def _current_help(self, _event=None):
        _sample, reason = self._current_sample()
        return reason or "将左图相机当前显示的那一帧放到右侧。调整照明后再点击，可更新右图。"

    def sync_controls(self):
        current, reason = self._current_sample()
        if self._archive and self.sources[1] is None and self._loading is None and not self._message:
            hint = reason or "已收到同一相机的预览。点“取相机当前帧”开始对照。"
            if region := self.views[0].roi:
                hint += f" 左图使用原检查框 {region[2]} × {region[3]} px。"
            if hint != self.status.get():
                self.status.set(hint)
        for view in self.views:
            view.region_editable = self._can_share and self._loading is None
        can_restore = bool(self._archive and self._archive.roi and self._region is None
                           and (self._can_share or self.sources[1] is None))
        region_label = "恢复原检查框" if can_restore else "恢复全图"
        if self.clear_region.cget("text") != region_label:
            self.clear_region.configure(text=region_label)
        if self._archive and not self.camera_button.winfo_manager():
            self.camera_button.pack(side=tk.LEFT, padx=(0, 5))
        elif not self._archive and self.camera_button.winfo_manager():
            self.camera_button.pack_forget()
        for widget, ready in ((self.open_left, not self._file_block(0)), (self.open_right, not self._file_block(1)),
                              (self.use_current, current is not None),
                              (self.clear_region, self._loading is None and (can_restore or self.views[0].roi is not None)),
                              (self.camera_button, self._archive is not None and self._loading is None)):
            if ready == widget.instate(["disabled"]):
                widget.state(["!disabled" if ready else "disabled"])

    def choose_file(self, index):
        if self._file_block(index):
            return
        path = filedialog.askopenfilename(parent=self.top, title="选择左侧原图" if index == 0 else "选择右侧原图",
                                         filetypes=[("TIFF 原图", ("*.tif", "*.tiff"))])
        if path:
            self.open_file(index, Path(path))

    def open_file(self, index, path):
        if self._file_block(index):
            return

        def use(source):
            if not index and self._archive:
                self._archive, self._region, self.sources[1] = None, None, None
            self.sources[index] = source
            if not index and self.sources[1] is None:
                self.sources[1] = self._current_sample()[0]

        self._load(lambda: read_preview_image(path), use, "正在读取 TIFF 原图…")

    def open_archive(self, read):
        def use(archive):
            self._archive, self._region = archive, archive.roi
            self.sources[0] = archive.original
            self.sources[1] = self._current_sample()[0]

        self._load(read, use, "正在核对采集原图…")

    def _load(self, read, use, message):
        request = self._loading = object()
        self._measurement = None
        self._message = ""
        self.status.set(message)
        self.sync_controls()

        def loaded(source):
            if self.top.winfo_exists() and self._loading is request:
                self._loading = None
                use(source)
                self._refresh()

        def failed(exc):
            if self.top.winfo_exists() and self._loading is request:
                self._loading = None
                if any(self.sources):
                    self._refresh()
                self._message = ui_error(exc)
                self.status.set(self._message.split("\n", 1)[0])
                self.sync_controls()

        self.submit(read, loaded, failed)

    def _take_current(self):
        source, reason = self._current_sample()
        if source is None:
            self.status.set(reason)
            return
        self.sources[1] = source
        self._refresh()

    def _locate_camera(self):
        if self._archive is not None and self._loading is None:
            reason = self.locate_camera(self._archive.original, self.views[0].roi)
            if reason:
                self._message = reason
                self.status.set(reason)
            else:
                self.top.withdraw()

    def _reset_region(self):
        self._region = self._archive.roi if self._region is None and self._archive else None
        self._refresh()

    def _set_region(self, roi):
        if roi is not None and not self.views[0].region_editable:
            return
        self._region = roi
        self._refresh()

    def _refresh(self):
        self._message = ""
        left, right = self.sources
        reason = preview_region_mismatch(left, right) if left and right else None
        paired = self._can_share = bool(left and right and reason is None)
        if not paired and self._archive is None:
            self._region = None
        region = self._region if paired or (self._archive and not right) else None
        if not left:
            hint = "请打开一张 Capture 保存的预览原图。"
        elif not right:
            hint = "请选择右图，或取同一相机当前的预览帧。"
        elif paired:
            hint = "同一相机与画幅。按住 Shift 拖动可共用检查框。" if region is None else (
                f"共用检查框 {region[2]} × {region[3]} px · 起点 ({region[0]}, {region[1]})")
        else:
            hint = f"{_MISMATCH[reason]} 当前显示各自的全图统计。"
        self.status.set(hint)
        for index, (view, frame) in enumerate(zip(self.views, self.sources, strict=True)):
            view.roi = region
            title = "左图" if not index else "右图"
            if frame is not None:
                title += (f" · 阶段 {self._archive.phase} 原帧 {self._archive.sequence}" if not index and self._archive
                          else " · 已保存原图" if frame.path is not None else " · 当前预览帧")
                stamp = format_receipt_utc(frame.image.received_utc_ns, precision=3)
                line = f"{_camera_name(frame.device)} · {stamp} UTC"
            else:
                line = ""
            self.metadata[index].configure(text=line)
            self.metrics[index].configure(text="正在统计原图…" if frame is not None else "")
            view.set_title(title)
            view.image(frame.image if frame is not None else None, placeholder="请选择图像", empty_caption="等待图像")
            view._schedule_draw()
        self.sync_controls()
        self._measure(tuple(self.sources), region)

    def _measure(self, sources, region):
        request = self._measurement = object()
        if not any(sources):
            return

        def measure():
            return [image_quality(source.image, region, measure_texture=False) if source else None for source in sources]

        def measured(rows):
            if self.top.winfo_exists() and self._measurement is request:
                for label, row in zip(self.metrics, rows, strict=True):
                    if row is not None:
                        middle = row["percentiles_p5_p50_p95"][1] / row["code_max"]
                        label.configure(text=f"亮度中位 {middle:.1%} · 近黑 {row['dark_fraction']:.1%}"
                                             f" · 近饱和 {row['bright_fraction']:.1%}")
                if self._archive and self._can_share and sources[1].path is None and not self._message:
                    black = rows[1]["dark_fraction"] >= QUALITY_ALERT_FRACTION
                    bright = rows[1]["bright_fraction"] >= QUALITY_ALERT_FRACTION
                    if black and bright:
                        tip = "当前预览的亮暗两端均有提醒。可查看反光和照明均匀性。"
                    elif bright:
                        tip = "当前预览有近饱和提醒。可减小曝光或减弱光照，再取一帧复核。"
                    elif black:
                        tip = "当前预览有近黑提醒。可增加照明或检查曝光，再查看拖影。"
                    else:
                        tip = "当前预览的近黑和近饱和像素均低于提醒阈值。"
                    self.status.set(f"{self.status.get()}\n{tip}")

        def failed(exc):
            if self.top.winfo_exists() and self._measurement is request:
                self._message = ui_error(exc)
                for label, source in zip(self.metrics, sources, strict=True):
                    label.configure(text="该图像的统计无法计算。" if source else "")

        self.submit(measure, measured, failed)
