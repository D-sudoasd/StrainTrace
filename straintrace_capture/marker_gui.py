"""Review the receipts around a recorded experiment marker."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import ScrollableForm, wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .frame_search import FrameLocation, format_receipt_utc
from .presentation import ui_error
from .storage import ExperimentReader
from .viewport import repaint_after_resize


def _offset(value):
    milliseconds = value / 1e6
    return f"{milliseconds:+.3f}" if abs(milliseconds) < 10_000 else f"{milliseconds:+.6g}"


class MarkerReviewWindow:
    def __init__(self, parent, experiment, identity, marker_id, cameras, submit, open_frame):
        self.experiment, self.experiment_id, self.marker_id = experiment, identity, marker_id
        self.cameras, self.submit, self.open_frame = cameras, submit, open_frame
        self._busy = False
        self._rows = {}
        self._marker = {}
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("实验标记 · 前后原图")
        self.top.transient(parent)
        self.top.geometry("665x445")
        self.top.minsize(495, 355)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        repaint_after_resize(self.top)

        body = ttk.Frame(self.top, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=3)
        body.rowconfigure(3, weight=1)
        self.title = ttk.Label(body, text="实验标记", style="CaptureSection.TLabel", width=1, takefocus=True)
        self.title.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ToolTip(self.title, lambda _event: self._marker.get("label") or "现场观察")
        self.status = ttk.Label(body, text="正在读取标记…", style="CaptureMuted.TLabel", width=1)
        self.status.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ToolTip(self.status, self._time_help, title="标记时刻")

        listing = ttk.Frame(body)
        listing.grid(row=2, column=0, sticky="nsew")
        columns = ("camera", "side", "frame", "offset")
        self.table = ttk.Treeview(listing, columns=columns, show="headings", selectmode="browse", height=4)
        for key, title, width, minimum in (
                ("camera", "相机", 168, 105), ("side", "前后", 78, 58),
                ("frame", "图像序号", 104, 95), ("offset", "时差（ms）", 145, 126)):
            self.table.column(key, width=width, minwidth=minimum, stretch=key in ("camera", "offset"))
            self.table.heading(key, text=title)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(listing, orient="vertical", command=self.table.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<<TreeviewSelect>>", self._select)
        self.table.bind("<Return>", lambda _event: self._open())
        self.table.bind("<Double-1>", self._double_click)
        ToolTip(self.table, self._row_help, title="前后原图")
        ToolTip(scroll, "滚动查看其他相机的图像。")

        detail_area = ScrollableForm(body, padding=(0, 5, 0, 0))
        detail_area.grid(row=3, column=0, sticky="nsew")
        self.detail = ttk.Label(detail_area.body, justify=tk.LEFT, width=1, takefocus=True)
        self.detail.pack(fill=tk.X)
        wrap_label_on_resize(self.detail)
        detail_area.bind_controls()
        ToolTip(self.detail, lambda _event: self.detail.cget("text"), title="所选原图")
        toolbar = ttk.Frame(body)
        toolbar.grid(row=4, column=0, sticky="ew", pady=(5, 0))
        self.open_button = ttk.Button(toolbar, text="打开原图", command=self._open, style="Record.TButton")
        self.open_button.pack(side=tk.RIGHT)
        ToolTip(self.open_button, lambda _event: self._open_help())
        self.refresh_button = ttk.Button(toolbar, text="刷新", command=self.refresh)
        self.refresh_button.pack(side=tk.RIGHT, padx=(0, 7))
        ToolTip(self.refresh_button, "重新搜索台账中已登记的图像。采集仍在进行时可纳入后到的帧。")
        self.refresh()

    def refresh(self):
        if self._busy:
            return
        self._busy = True
        self.status.configure(text="正在读取各相机已保存的图像…")
        self._select()

        def read():
            with ExperimentReader(self.experiment) as reader:
                return reader.marker_context(self.marker_id, expected_experiment_id=self.experiment_id)

        self.submit(read, self._loaded, self._read_failed)

    def _loaded(self, result):
        if not self.top.winfo_exists():
            return
        self._busy = False
        self._marker = marker = result["marker"]
        self.title.configure(text=marker.get("label") or "现场观察")
        self.status.configure(text=f"标记（UTC）{format_receipt_utc(marker['marked_utc_ns'], precision=3)}")
        self._rows.clear()
        self.table.delete(*self.table.get_children())
        for index, (camera, sides) in enumerate(result["cameras"].items()):
            for side, label in (("before", "之前"), ("after", "之后")):
                row = sides[side]
                key = f"{index}:{side}"
                self._rows[key] = row
                if row and row["offset_ns"] == 0:
                    label = "当时"
                self.table.insert("", "end", iid=key, values=(
                    self.cameras.get(camera, camera), label, row["sequence"] if row else "—",
                    _offset(row["offset_ns"]) if row else "—",
                ))
        if self._rows:
            selected = next((key for key, row in self._rows.items() if row), next(iter(self._rows)))
            self.table.selection_set(selected)
            self.table.focus(selected)
            self.table.see(selected)
        self._select()

    def _read_failed(self, error):
        if self.top.winfo_exists():
            self._busy = False
            self._rows.clear()
            self.table.delete(*self.table.get_children())
            self._select()
            self.status.configure(text="无法读取标记")
            self.detail.configure(text=ui_error(error))

    def _time_help(self, _event):
        if not self._marker:
            return self.status.cget("text")
        return (f"主机记录（UTC）{format_receipt_utc(self._marker['marked_utc_ns'])}\n"
                "时间差来自同一主机单调时钟，比较按键和收图时刻。曝光时刻需用独立时序记录核对。")

    def _description(self, row, *, precision=9):
        if row is None:
            return "这一侧没有同一主机时钟的已保存图像。"
        return (f"收图（UTC）{format_receipt_utc(row['received_utc_ns'], precision=precision)}\n"
                f"阶段 {row['phase']} · 图像 {row['sequence']}\n"
                f"相对标记 {row['offset_ns']:+,} 纳秒")

    def _selected(self):
        return next(iter(self.table.selection()), "")

    def _row_help(self, event):
        key = self._selected()
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            key = self.table.identify_row(event.y)
        return self._description(self._rows[key]) if key in self._rows else (
            "时差单位为毫秒（ms）。负值表示主机在标记前收到图像，正值表示标记后收到。")

    def _select(self, _event=None):
        key = self._selected()
        self.detail.configure(text=self._description(self._rows[key], precision=3)
                              if key in self._rows else "选中有序号的一行可查看原图。")
        self.open_button.state(["disabled" if self._busy or not self._rows.get(self._selected()) else "!disabled"])
        self.refresh_button.state(["disabled" if self._busy else "!disabled"])

    def _open_help(self):
        if self._busy:
            return "等候台账读取完成。"
        if not self._rows.get(self._selected()):
            return "请选择一行有图像序号的记录。"
        return "核对所选 TIFF 与台账的哈希并打开原图。"

    def _double_click(self, event):
        if key := self.table.identify_row(event.y):
            self.table.selection_set(key)
            self._open()

    def _open(self):
        if not self._busy and (row := self._rows.get(self._selected())):
            location = FrameLocation(self.experiment_id, row["camera_id"], row["sequence"],
                                     row["ordinal"], row["frame_id"], row["sha256"])
            self.open_frame(location, self.top.withdraw, self._open_failed)

    def _open_failed(self, error):
        if self.top.winfo_exists():
            self.detail.configure(text=ui_error(error))

    def reopen(self):
        self.top.deiconify()
        self.top.lift()
