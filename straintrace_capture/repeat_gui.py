"""Review the source and destination before loading a repeat acquisition plan."""

import tkinter as tk
from tkinter import filedialog, ttk

from ezdic_gui.controls import wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .presentation import DURABILITY_LABELS, MODE_HELP, MODE_LABELS, ROLE_LABELS, STATE_LABELS, phase_limit_text, ui_error
from .viewport import repaint_after_resize


class RepeatPlanWindow:
    def __init__(self, parent, draft, output, camera_title, new_path, load):
        self.draft, self.load, self.new_path = draft, load, new_path
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("图像采集 · 准备复拍")
        self.top.transient(parent)
        self.top.geometry("730x535")
        self.top.minsize(605, 445)
        repaint_after_resize(self.top)
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self.name = tk.StringVar(self.top, draft.config.name)
        self.output = tk.StringVar(self.top, output)
        self.notice = tk.StringVar(self.top)
        self._working = False
        phases = draft.config.phases
        layout = ttk.Frame(self.top, padding=14)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(1, weight=1)
        layout.rowconfigure(4, weight=1, minsize=99)

        state = STATE_LABELS.get(draft.source_state.upper(), "状态未知")
        heading = ttk.Label(layout, text=f"来源实验 · {draft.source_name}", width=1, style="CaptureSection.TLabel")
        heading.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 3))
        ToolTip(heading, f"{draft.source_name} · {state}\n{draft.source}", title="来源实验")
        description = f"{state} · {len(draft.config.cameras)} 台相机 · {len(phases)} 个阶段 · {DURABILITY_LABELS[draft.config.durability]}"
        scope = ttk.Label(layout, text=description, width=1, style="CaptureMuted.TLabel")
        scope.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        cameras = "\n".join(f"{camera_title(camera.id)}  {camera.id}" for camera in draft.config.cameras)
        ToolTip(scope, f"从台账载入相机、阶段和存储设置。\n{cameras}", title="采集设置")

        label = ttk.Label(layout, text="新实验名称")
        label.grid(row=2, column=0, sticky="w", padx=(0, 9), pady=3)
        self.name_entry = ttk.Entry(layout, textvariable=self.name)
        self.name_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=3)
        ToolTip(self.name_entry, "可填写本组试样编号。这个名称将写入新实验的台账。")
        ToolTip(label, self.name_entry._tooltip.resolve_text)
        label = ttk.Label(layout, text="新输出目录")
        label.grid(row=3, column=0, sticky="w", padx=(0, 9), pady=3)
        self.output_entry = ttk.Entry(layout, textvariable=self.output)
        self.output_entry.grid(row=3, column=1, sticky="ew", pady=3)
        ToolTip(self.output_entry, lambda _event: f"开始采集后在这里新建目录。\n{self.output.get()}",
                title="新输出目录")
        ToolTip(label, self.output_entry._tooltip.resolve_text)
        self.browse = ttk.Button(layout, text="选择位置…", style="Small.TButton", command=self._choose)
        self.browse.grid(row=3, column=2, padx=(7, 0), sticky="e")
        ToolTip(self.browse, "选择一个父文件夹，程序会建议一个新的实验目录。")

        frame = ttk.Frame(layout)
        frame.grid(row=4, column=0, columnspan=3, pady=(10, 6), sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(frame, columns=("phase", "mode", "role", "limits"), show="headings",
                                  height=4, selectmode="browse")
        for key, title, width, minimum in (("phase", "阶段", 63, 62), ("mode", "拍摄方式", 168, 145),
                                           ("role", "用途", 83, 70), ("limits", "计划参数", 247, 160)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=minimum, stretch=key != "phase")
        self.table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        ToolTip(scroll, "滚动查看来源计划的后续阶段。")
        for number, phase in enumerate(phases, 1):
            mode = MODE_LABELS[phase.mode] + (" · 待确认" if phase.start_confirmation else "")
            self.table.insert("", tk.END, iid=str(number),
                              values=(number, mode, ROLE_LABELS[phase.role], phase_limit_text(phase.to_dict())))
        self.table.focus("1")
        self.table.selection_set("1")
        ToolTip(self.table, self._phase_help, title="复拍阶段")
        if draft.cleared_anchors:
            numbers = "、".join(map(str, draft.cleared_anchors))
            self._instruction = f"第 {numbers} 阶段如需关联图像，请用本轮首个共享脉冲设置相机计数器锚点。"
        else:
            self._instruction = "载入后检查相机、触发条件和输出位置。正式采集由工作台的开始按钮启动。"
        self.notice.set(self._instruction)
        self.name.trace_add("write", self._edited)
        self.output.trace_add("write", self._edited)
        hint = ttk.Label(layout, width=1, textvariable=self.notice, justify=tk.LEFT, style="CaptureMuted.TLabel")
        hint.grid(row=5, column=0, columnspan=3, sticky="ew")
        wrap_label_on_resize(hint)
        ToolTip(hint, lambda _event: self.notice.get(), title="复拍准备")

        actions = ttk.Frame(layout)
        actions.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(9, 0))
        cancel = ttk.Button(actions, text="关闭", style="Small.TButton", command=self.top.destroy)
        cancel.pack(side=tk.RIGHT)
        ToolTip(cancel, "关闭这份复拍计划。")
        self.apply = ttk.Button(actions, text="载入并检查", command=self._apply, style="Record.TButton")
        self.apply.pack(side=tk.RIGHT, padx=(0, 7))
        ToolTip(self.apply, lambda _event: ("正在核对来源和输出位置。" if self._working else
                                           "将新实验名称、输出目录和来源计划载入工作台，打开采前检查。"))
        self.name_entry.focus_set()
        self.top.after_idle(self._show_output_tail)

    def _show_output_tail(self):
        if self.top.winfo_exists() and self.top.focus_get() is not self.output_entry:
            self.output_entry.xview_moveto(1.0)

    def _edited(self, *_args):
        if not self._working:
            self.notice.set(self._instruction)
            self.top.after_idle(self._show_output_tail)

    def _choose(self):
        folder = filedialog.askdirectory(parent=self.top, title="选择复拍的保存位置", mustexist=True)
        if folder:
            self.output.set(self.new_path(folder))

    def _phase_help(self, event):
        if event and event.type in (tk.EventType.Enter, tk.EventType.Motion):
            item = self.table.identify_row(event.y)
        else:
            item = self.table.focus()
        if not item:
            return "阶段按序执行，可在载入后继续编辑。"
        phase = self.draft.config.phases[int(item) - 1]
        notes = [MODE_HELP[phase.mode], phase_limit_text(phase.to_dict())]
        if int(item) in self.draft.cleared_anchors:
            notes.append("请用本轮实际起点设置相机计数器锚点。")
        if phase.start_confirmation:
            notes.append(f"开始前确认 · {phase.start_confirmation}")
        return "\n".join(notes)

    def _controls(self, working):
        self._working = working
        for widget in (self.name_entry, self.output_entry, self.browse, self.apply):
            widget.state(["disabled" if working else "!disabled"])

    def _apply(self):
        if self._working:
            return
        self._controls(True)
        self.notice.set("正在核对来源台账、标定文件和新输出目录…")

        def failed(exc):
            if self.top.winfo_exists():
                self._controls(False)
                code = getattr(exc, "code", None) or ""
                self.notice.set(str(exc) if code.startswith(("REPEAT_", "OUTPUT_")) else ui_error(exc, details=False))

        self.load(self.name.get(), self.output.get(), self.top.destroy, failed)
