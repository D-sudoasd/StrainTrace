"""Capture window for the latest quality evidence from each camera."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .image_quality_watch import CONFIRMATIONS, QUALITY_ALERT_FRACTION, SHARPNESS_ALERT
from .presentation import image_quality_alert_text, image_quality_sample_help
from .viewport import repaint_after_resize


_HELP = ("按每台相机的计划抽查范围统计已入队的主图。近黑、近饱和分别按最低和最高 1% 码值统计。\n"
         f"近黑或近饱和达 {QUALITY_ALERT_FRACTION:.0%}，或清晰度响应降至参照的 {SHARPNESS_ALERT:.0%} 时开始判断。\n"
         f"同一相机连续 {CONFIRMATIONS} 个异常样本会提示。采后可在时间线定位对应帧。")


class ImageQualityWatchWindow:
    def __init__(self, parent, title_camera, select_camera):
        self.camera_title = title_camera
        self.select_camera = select_camera
        self._samples = {}
        self.top = tk.Toplevel(parent)
        self.top.title("采集画质抽查")
        self.top.geometry("760x435")
        self.top.minsize(610, 340)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        main = ttk.Frame(self.top, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        self.heading = ttk.Label(main, width=1, justify=tk.LEFT)
        self.heading.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        wrap_label_on_resize(self.heading)
        ToolTip(self.heading, _HELP, title="抽查范围")

        frame = ttk.Frame(main)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(frame, columns=("state", "dark", "bright", "sharpness", "sequence"),
                                  show="tree headings", selectmode="browse", height=5, takefocus=True)
        for key, title, width in (
            ("#0", "相机", 150), ("state", "抽查状态", 135), ("dark", "近黑", 76),
            ("bright", "近饱和", 85), ("sharpness", "清晰度", 83), ("sequence", "帧号", 66),
        ):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=54, stretch=key in ("#0", "state"),
                              anchor="w" if key in ("#0", "state") else "center")
        self.table.tag_configure("watch", foreground="#ffda87")
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<<TreeviewSelect>>", self._show_detail)
        ToolTip(self.table, self._table_tip, title="选中相机的抽查数据")

        self.detail = ttk.Label(main, width=1, justify=tk.LEFT)
        self.detail.grid(row=2, column=0, sticky="ew", pady=(7, 8))
        wrap_label_on_resize(self.detail)
        ToolTip(self.detail, lambda _event: self.detail.cget("text"))
        actions = ttk.Frame(main)
        actions.grid(row=3, column=0, sticky="ew")
        ttk.Button(actions, text="关闭", style="Small.TButton", command=self.top.destroy).pack(side=tk.RIGHT)
        self.camera_button = ttk.Button(actions, text="查看相机画面", style="Small.TButton", command=self._open_camera)
        self.camera_button.pack(side=tk.RIGHT, padx=(0, 7))
        ToolTip(self.camera_button, "在采集窗口选中这一台相机。可在“画质”中框选局部复查。")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.table.focus_set()

    def render(self, status, *, preferred=None):
        report = status["image_quality_watch"]
        phase = status.get("phase") if status["phase_progress"] else None
        self._samples = {camera: row for camera, row in report["cameras"].items()
                         if phase is None or row["phase"] == phase}
        state = "图像已入队后抽查"
        if status.get("phase_pause"):
            state = "采集暂停 · 显示最近样本"
        elif report["deferred"]:
            state = "待写缓冲较多 · 暂缓抽查"
        elif status["state"] in {"completed", "stopped", "failed", "interrupted"}:
            state = "采集结束 · 显示最近样本"
        caption = f"{state} · 本次已抽查 {report['measurements']} 帧"
        if report["measurement_errors"] or report["ledger_errors"]:
            caption += f"\n抽查失败 {report['measurement_errors']} 次 · 提醒登记失败 {report['ledger_errors']} 条"
        if self.heading.cget("text") != caption:
            self.heading.configure(text=caption)
        cameras = status["cameras"]
        for camera in self.table.get_children():
            if camera not in cameras:
                self.table.delete(camera)
        for camera in cameras:
            sample = self._samples.get(camera)
            if sample:
                metrics = sample["metrics"]
                ratio = metrics["sharpness_ratio"]
                scope = sample.get("region")
                area = "检查框" if scope and not scope.get("fallback") else "全图"
                values = (image_quality_alert_text(sample["alerts"]) or f"{area}抽查",
                          f"{metrics['near_black_fraction']:.1%}", f"{metrics['near_saturation_fraction']:.1%}",
                          "—" if ratio is None else f"{ratio:.0%}", sample["sequence"])
                tags = ("watch",) if sample["alerts"] or (scope and scope.get("fallback")) else ()
            else:
                values = ("等待样本", "—", "—", "—", "—")
                tags = ()
            if not self.table.exists(camera):
                self.table.insert("", tk.END, iid=camera, text=self.camera_title(camera), values=values, tags=tags)
            elif self.table.item(camera, "values") != tuple(map(str, values)) or self.table.item(camera, "tags") != tags:
                self.table.item(camera, values=values, tags=tags)
        if preferred in cameras or not self.table.selection():
            choice = preferred if preferred in cameras else next(iter(cameras), None)
            if choice is not None:
                self.table.selection_set(choice)
                self.table.focus(choice)
                self.table.see(choice)
        self._show_detail()

    def _table_tip(self, event):
        camera = self.table.focus()
        if event is not None and getattr(event, "keysym", "??") in ("", "??"):
            camera = self.table.identify_row(event.y) or camera
        sample = self._samples.get(camera)
        return image_quality_sample_help(sample) if sample else _HELP

    def _show_detail(self, _event=None):
        camera = self.table.focus()
        sample = self._samples.get(camera)
        text = image_quality_sample_help(sample) if sample else "等待可抽查的主图。悬停表格可查看抽查范围。"
        if text != self.detail.cget("text"):
            self.detail.configure(text=text)
        self.camera_button.state(["!disabled" if camera else "disabled"])

    def _open_camera(self):
        if camera := self.table.focus():
            self.select_camera(camera)
            self.top.withdraw()
