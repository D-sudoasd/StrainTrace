"""Choose and compare the recorded reference before opening continuation Analysis."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .config import CaptureError
from .continuation_analysis import (continuation_snapshot, prepare_continuation_binding, prepare_live_reference,
                                   preview_continuation, recorded_reference_choices, reference_experiments)
from .presentation import ui_error
from .recent import recent_experiments
from .viewport import ImageViewport, repaint_after_resize


class ContinuationReferenceWindow:
    def __init__(self, parent, root, identity, camera, title, part, submit, use_binding, use_current, *, live=False):
        self.root, self.identity, self.camera, self.part = root, identity, camera, part
        self.submit, self.use_binding, self.use_current = submit, use_binding, use_current
        self.live = live
        self._sources = {}
        self._snapshot = None
        self._busy = False
        self._detail = ""
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("选择续采的实时参考图" if live else "选择续采的参考图")
        self.top.geometry("960x630")
        self.top.minsize(720, 500)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.choice, self.sequence, self.status = (tk.StringVar(self.top) for _ in range(3))
        self._build(title)
        self.sequence.trace_add("write", self._sequence_changed)
        self._scan()

    def _build(self, title):
        body = ttk.Frame(self.top, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)
        ttk.Label(body, text=title, style="CaptureSection.TLabel").grid(row=0, column=0, sticky="w")
        location = ttk.Frame(body)
        location.grid(row=1, column=0, sticky="ew", pady=(8, 5))
        location.columnconfigure(1, weight=1)
        ttk.Label(location, text="原实验").grid(row=0, column=0, padx=(0, 8))
        self.source = ttk.Combobox(location, textvariable=self.choice, state="readonly", width=25)
        self.source.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.source.bind("<<ComboboxSelected>>", lambda _event: self._read_choices())
        self.find = ttk.Button(location, text="查找原实验…", command=self._locate, style="Small.TButton")
        self.find.grid(row=0, column=2, sticky="e")
        ToolTip(self.source, lambda _event: self._sources.get(self.choice.get(), {}).get(
            "directory", "选择台账关联的上一轮或更早的实验。"))
        ToolTip(self.find, "移动过原实验时可选择它的新目录。程序会核对台账中的实验标识。")

        reference = ttk.Frame(body)
        reference.grid(row=2, column=0, sticky="ew", pady=(0, 7))
        ttk.Label(reference, text="原参考图序号").pack(side=tk.LEFT, padx=(0, 8))
        self.number = ttk.Combobox(reference, textvariable=self.sequence, width=13)
        self.number.pack(side=tk.LEFT, padx=(0, 7))
        self.number.bind("<Return>", lambda _event: self._preview())
        self.number.bind("<<ComboboxSelected>>", lambda _event: self._preview())
        self.preview = ttk.Button(reference, text="核对并预览", command=self._preview, style="Small.TButton")
        self.preview.pack(side=tk.LEFT)
        ToolTip(self.number, "可从近期保存的参考图中选择，也可输入原实验中的参考图序号。")
        ToolTip(self.preview, "对照两轮的相机身份、图像尺寸、像素格式和已记录的成像区域，并显示两张原图。")
        info = ttk.Label(body, textvariable=self.status, style="CaptureMuted.TLabel", width=1, justify="left")
        info.grid(row=3, column=0, sticky="ew", pady=(0, 7))
        wrap_label_on_resize(info)
        ToolTip(info, lambda _event: self._detail or self.status.get())

        wall = ttk.Frame(body)
        wall.grid(row=4, column=0, sticky="nsew")
        wall.columnconfigure((0, 1), weight=1, uniform="reference")
        wall.rowconfigure(0, weight=1)
        self.views = []
        for index, label in enumerate(("原实验参考图", "本轮首张实验图")):
            view = ImageViewport(wall, self.camera, label, lambda _camera: None, lambda *_args: None)
            view.grid(row=0, column=index, sticky="nsew", padx=(0, 6) if index == 0 else 0)
            view.image(None, placeholder="等待核对参考图")
            ToolTip(view.canvas, "对照试样、成像位置和加载基准。滚轮缩放，拖动查看细节。", title=label)
            self.views.append(view)

        footer = ttk.Frame(body)
        footer.grid(row=5, column=0, sticky="ew", pady=(9, 0))
        self.current = ttk.Button(footer, text="改用本轮图像…", command=self._current, style="Small.TButton")
        self.current.pack(side=tk.LEFT)
        ToolTip(self.current, "返回本轮参考图选择。若本轮没有专门拍摄参考图，可按本轮图像建立新的分析基准。")
        self.confirm = ttk.Button(footer, text="使用此图并打开实时分析" if self.live else "使用此参考图并打开分析",
                                  command=self._confirm, style="Record.TButton")
        self.confirm.pack(side=tk.RIGHT)
        ToolTip(self.confirm, lambda _event: "请先核对并预览原参考图。" if self._snapshot is None else
                "确认试样和成像位置可对照后，以原参考图为基准计算本轮实验图的应变。")

    def _controls(self):
        found = bool(self._sources.get(self.choice.get()))
        for widget, ready in ((self.source, bool(self._sources)), (self.find, True), (self.number, found),
                              (self.preview, found and self.sequence.get().strip()), (self.current, True),
                              (self.confirm, self._snapshot is not None)):
            widget.state(["!disabled" if ready and not self._busy else "disabled"])

    def _job(self, action, finished, message):
        self._busy = True
        self.status.set(message)
        self._detail = ""
        self._controls()

        def done(value):
            if self.top.winfo_exists():
                self._busy = False
                try:
                    finished(value)
                except (CaptureError, OSError) as exc:
                    failed(exc)
                if self.top.winfo_exists():
                    self._controls()

        def failed(exc):
            if self.top.winfo_exists():
                self._busy = False
                self._detail = ui_error(exc)
                self.status.set(ui_error(exc, details=False))
                self._controls()

        self.submit(action, done, failed)

    def _locate(self):
        location = filedialog.askdirectory(parent=self.top, title="查找原实验目录", mustexist=True)
        if location:
            self._scan(Path(location).resolve())

    def _scan(self, location=None):
        def read():
            return reference_experiments(self.root, expected_id=self.identity,
                                         search_dirs=(location,) if location else (),
                                         recent_paths=tuple(row.directory for row in recent_experiments()))

        def loaded(rows):
            self._sources.clear()
            for number, row in enumerate(rows, 1):
                folder = Path(row["directory"]).name
                label = row["name"] if row["name"] == folder else f"{row['name']} · {folder}"
                self._sources[f"{number}. {label}"] = row
            self.source.configure(values=tuple(self._sources))
            choice = next((key for key, row in self._sources.items() if Path(row["directory"]) == location), "")
            if location and not choice:
                self.choice.set("")
                self._clear()
                self.status.set("该目录的实验标识与续采记录不匹配。请选择原实验的目录。")
            elif self._sources:
                self.choice.set(choice or next(iter(self._sources)))
                self._read_choices()
            else:
                self.status.set("尚未找到原实验目录。可以查找目录，或选择本轮图像作为分析基准。")

        self._clear()
        self._job(read, loaded, "正在定位与本轮关联的原实验…")

    def _read_choices(self):
        directory = self._sources[self.choice.get()]["directory"]
        self._clear()
        self.sequence.set("")
        self.number.configure(values=())

        def loaded(choices):
            self.number.configure(values=[str(number) for number, _phase in choices])
            if choices:
                self.sequence.set(str(choices[0][0]))
                self._preview()
            else:
                self.status.set("这个原实验没有登记该相机的参考图。请选择其他关联实验。")

        self._job(lambda: recorded_reference_choices(self.root, directory, self.camera, live=self.live),
                  loaded, "正在读取参考图序号…")

    def _clear(self):
        self._snapshot = None
        for view in self.views:
            view.image(None, placeholder="等待核对参考图")

    def _sequence_changed(self, *_args):
        if self._snapshot is not None:
            self._clear()
            self.status.set("参考序号已修改，请重新核对并预览。")
        self._controls()

    def _preview(self):
        if self._busy or not (source := self._sources.get(self.choice.get())):
            return
        try:
            sequence = int(self.sequence.get())
        except ValueError:
            self.status.set("请输入原实验中的参考图序号，例如 12。")
            return
        self._clear()

        def read():
            snapshot = continuation_snapshot(self.root, source["directory"], self.camera, sequence,
                                             image_part=self.part, live=self.live)
            if snapshot["experiment_id"] != self.identity or snapshot["external_reference"]["experiment_id"] != source["experiment_id"]:
                raise CaptureError("CONTINUATION_REFERENCE_CHANGED", "实验目录中的标识已变化。")
            return snapshot, preview_continuation(snapshot)

        def loaded(result):
            self._snapshot, images = result
            for view, image, row in zip(self.views, images, self._snapshot["frames"][:2]):
                view.image(image, frame_caption=f"原图序号 {row['sequence']} · 阶段 {row['phase']}")
            basis = "相机序列号一致" if self._snapshot["external_reference"]["identity_basis"] == "matched" else "设备路径一致"
            if len(images) == 1:
                self.views[1].image(None, placeholder="等待本轮首张已保存实验图")
                self.status.set(f"{basis}，已核验原参考图。用于跟踪的图像会核对成像区域和像素格式。")
            else:
                self.status.set(f"{basis}，已记录的成像区域与像素格式一致。请对照两图后确认参考图。")

        self._job(read, loaded, "正在核对成像设置并读取原图…")

    def _confirm(self):
        if self._busy or self._snapshot is None:
            return
        snapshot = self._snapshot

        def complete(binding):
            self.use_binding(binding)
            self.top.destroy()

        prepare = prepare_live_reference if self.live else prepare_continuation_binding
        self._job(lambda: prepare(snapshot), complete, "正在保存续采分析的图像来源…")

    def _current(self):
        if not self._busy:
            try:
                self.use_current()
            except (CaptureError, OSError) as exc:
                self._detail = ui_error(exc)
                self.status.set(ui_error(exc, details=False))
            else:
                self.top.destroy()
