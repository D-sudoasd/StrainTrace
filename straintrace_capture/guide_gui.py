"""Task-based help for the Capture desktop workflow."""

import tkinter as tk
from tkinter import ttk

from ezdic_gui.controls import ScrollableForm, notebook_help, wrap_label_on_resize
from ezdic_gui.tooltips import ToolTip

from .viewport import repaint_after_resize


GUIDE_STEPS = (
    ("连接相机", "定位到相机列表", (
        ("初次练习", "未发现实验相机时，可从顶部提示或点下方“打开模拟练习”，在独立窗口直接看到模拟散斑。"
         "可以依次检查画面和计划，再保存图像并练习复查。"),
        ("使用实验相机", "接上相机后点“刷新”，然后点“选择实验相机”。按型号和序列号勾选本次设备，再连接预览。"
         "相机未出现时，点“驱动状态”查看需要安装的驱动或厂商接口文件。"),
        ("普通 USB 相机取图", "这类相机可按主机请求取图。确认预览后点“检查计划”，再用试采核对实际帧率。"
         "悬停或按 F1 可查看取图和时间的含义。曝光同步需独立核实。"),
        ("更换使用外触发的相机", "先查看新相机的实时画面，再按顶部提示核对受影响阶段的物理输入和脉冲计数。"
         "可在设置中读取本台相机支持的端口，并按实际接线选择。"),
        ("恢复单台预览", "悬停在异常画面上可看原因。接回相机后，点“重连相机”打开这台设备，其他画面会继续更新。"
         "先核对恢复的画面，再检查采集计划。"),
    )),
    ("检查画面", "转到图像质量", (
        ("保存一张预览原图", "单击要保存的相机画面，再点预览下方的“保存预览原图”。"
         "选择文件名，程序会把当时显示的一帧按完整画幅保存为无损 TIFF。预览会继续更新。"),
        ("对照调整前后的画面", "点预览下方的“原图对照”，用已保存 TIFF 对照同一相机当前画面。"
         "断开相机后也可打开左右两张已保存原图。两张图的画幅条件匹配时，可按住 Shift 共用检查框。"),
        ("一次检查多台相机", "连接预览后点“检查各相机”。每台取 5 张新画面，检查近黑、近饱和和已有检查框内的纹理。"
         "选中有提醒的相机，点“查看画面”可打开它的画质页。"),
        ("检查局部光照", "在预览图上按住 Shift 拖出方框。右侧的亮度、近黑和近饱和比例，以及纹理指标会切换到框内。"
         "近饱和较多时检查试样反光或曝光过长。近黑较多时检查照明不足。"),
        ("搭配曝光和照明", "固定光源和试样，点“曝光建议”测量亮部。可按运动拖影要求填写最长曝光时间。"
         "点“试设并复测”后，改善的曝光会写入当前计划。可再试采核对帧率。"),
        ("比较对焦", "固定相机、试样、照明、曝光和增益，点“开始对焦比较”，再缓慢转动对焦环。"
         "用当前值与本轮最高值对照，并在原图里确认边缘或散斑细节。换相机、改选区或改参数后，可重新比较。"),
    )),
    ("检查计划", "定位到采集模板", (
        ("安排拍摄顺序", "打开“采集模板”，按任务选择并预览阶段。可以先改频率、张数和时长，再应用到计划。"
         "需要现场确认夹具或加载准备时，可用“参考图后确认”。"),
        ("统一多台相机的参数", "断开预览后，按住 Ctrl 选中两台或更多相机，点“多机参数”。"
         "只填要统一的参数。悬停字段可查各台的范围和步长。逐台试设并回读通过后，程序会更新采集计划。"),
        ("检查保存位置和设备", "选择新的实验目录，再点“检查计划”。处理报告中的阻止项。"
         "可在报告中点“写盘实测”测量目标盘。用“阶段试采”查看各阶段需要核对的项目，并从试采报告打开原图。"),
        ("一次试采多个阶段", "在“阶段试采”点“连续试采”，核对将要运行的阶段和每阶段时限。"
         "勾选“纳入需确认或触发的阶段”，可在同一窗口确认准备事项、手动触发或等待计划中的信号。"
         "停止后可从已有核验记录接着试采。"),
        ("拍摄镜头标定图像", "点“标定采图”，填写棋盘格内角点数和实测方格边长。"
         "每次摆稳并触发后检查角点，再改变棋盘格的位置和倾角，以便拟合镜头参数。"),
    )),
    ("开始采集", "定位到采集按钮", (
        ("开始与触发", "准备就绪后点右下角的开始按钮。手动或事件阶段可以点“触发一次”。"
         "等待操作员确认的阶段会在顶部显示提示。硬件触发阶段等相机就绪后，再由接线的设备发出脉冲。"),
        ("按载荷或位移触发", "编辑“事件前后”阶段，打开“试验机 CSV 阈值”，选量值列并按原单位填阈值。"
         "等事件前缓存满后，新增的相邻记录越线会自动触发。在“采集监看”悬停或按 F1 可查看 CSV 状态。"
         "结束后双击时间线中的 CSV 触发记录，可按相机查看这次保存的事件前后原图。"),
        ("反复记录事件前后", "在“事件前后”阶段把“事件次数”改为需要的次数。填 0 可一直等待，直到设定的时限或手动停止。"
         "每次收图结束后，等下一次前图缓存就绪，再手动触发或等下一次越线。复查时在时间线将“事件类型”设为“仅看事件窗口”。"),
        ("逐台查看进度", "在“采集监看”选中相机，可查看本阶段收到、已保存的帧数和本次实验的待存帧数。"
         "悬停或按 F1 查看当前状态的依据。有相机暂未送图时，可核对触发与接线。"),
        ("记录现场变化", "看到滑移或裂纹等现象时，按 F8 或点“标记事件”，可在“采集监看”预填说明。"
         "结束后在“采集时间线”将“事件类型”设为“仅看实验标记”，查看各相机在标记前后的原图。"),
        ("停止并保存", "底部显示电脑收到和已保存的张数。需要提前结束时点“停止并保存”，等队列中的图像写完。"
         "采集结束后会自动打开复查页。"),
    )),
    ("复查分析", "转到复查与分析", (
        ("核对采集结果", "在“复查与分析”查看登记帧数、TIFF 张数和文件校验状态。悬停可看各相机的计数。"
         "顶部会推荐查看异常记录、校验图像和台账，或检查中断。硬件触发实验可用独立脉冲记录核对相机和探测器图像。"),
        ("检查原图画质", "在“数据检查与恢复”点“检查原图画质”，按相机和阶段抽查。"
         "切换到“整场检查”可看各组覆盖与提醒，也可导出包含原图校验值的报告。选中提醒可在复查页打开对应原图。"),
        ("用归档原图复核调整", "选中异常原图后点“对照当前预览”。同一相机的画幅和传感器范围相符时，会沿用原检查框。"
         "点“定位相机”可用同一个框检查曝光或照明，再从预览下方的“原图对照”返回并取一张新画面。"),
        ("从异常找到原图", "点“查看异常记录”，也可在“采集时间线”将“事件类型”设为“仅看异常”。"
         "选中一行后，列表下方会提示能打开的图像。出现缺帧时，双击触发组记录查看各相机的原图。"
         "悬停或按 F1 可查看相机、时间来源和异常详情。"),
        ("找到特定图像", "在“查找图像”选择相机，再按阶段、用途或图像序号筛选。可打开经校验的原图，或在触发组中对照各相机。"
         "按主机收图时间也可以查找。"),
        ("选择分析方式", "“离线分析”逐张计算应变并可导出结果。“实时分析”可边采集边查看当前已保存图像的应变变化。"
         "有匹配的镜头标定时，可先校正畸变再进入分析。"),
        ("导出到其他 DIC 软件", "点“导出图像序列”，选择相机和实验阶段。检查范围后导出，参考图排在第 0 张。"
         "帧索引记录每张 TIFF 对应的原帧序号和收图时间。"),
        ("重开或续采", "通过“最近实验”打开上一次的图像。目录移动后可从列表定位新位置。"
         "如果采集中断，在复查页点“检查中断”核对遗留帧。点“准备续采”可按已保存帧和异常记录选起点，"
         "将所选阶段及后续阶段载入新计划。检查计划时可对照原实验的相机回读。"),
        ("准备下一组复拍", "在复查页点“准备复拍”，核对完整阶段并填写新试样名称和保存目录。"
         "载入后先检查计划。硬件触发实验需按本轮共享脉冲设置相机计数器锚点。"),
        ("交接实验", "用“制作交接副本”把原图、台账和归档标定一起复制到新位置。"
         "可以附上同步记录。对方打开副本后可用“校验图像和台账”核对全部文件。"),
    )),
)


class CaptureGuide:
    def __init__(self, parent, targets, navigate, *, open_practice=None):
        self.targets = targets
        self.navigate = navigate
        self.open_practice = open_practice
        self.top = tk.Toplevel(parent, bg=ttk.Style(parent).lookup("TFrame", "background"))
        self.top.title("图像采集 · 新手指南")
        self.top.geometry("690x470")
        self.top.minsize(550, 355)
        self.top.transient(parent)
        repaint_after_resize(self.top)
        layout = ttk.Frame(self.top, padding=14)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.columnconfigure(0, weight=1)
        layout.rowconfigure(0, weight=1)
        self.tabs = ttk.Notebook(layout)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        for index, (title, _caption, sections) in enumerate(GUIDE_STEPS):
            page = ScrollableForm(self.tabs, padding=(14, 8))
            if index == 0 and open_practice is None:
                sections = (
                    ("初次练习", "练习窗口自动连接模拟散斑。先到“图像质量”查看画面，再检查计划和输出目录。"
                     "使用默认计划时，会先保存一张参考图，再以每秒 6 张保存 12 张实验图。"),
                    ("使用实验相机", "回到图像采集主窗口。接上实验相机后点“刷新”，再选中设备并连接预览。"),
                    sections[2],
                )
            for heading, contents in sections:
                ttk.Label(page.body, text=heading, style="CaptureSection.TLabel").pack(anchor="w", pady=(8, 3))
                text = ttk.Label(page.body, text=contents, width=1, wraplength=450, justify=tk.LEFT)
                text.pack(fill=tk.X, pady=(0, 7))
                wrap_label_on_resize(text)
            for widget in (page.canvas, page.scrollbar):
                ToolTip(widget, "向下滚动查看本步骤的其余说明。", title=title)
            page.bind_controls()
            self.tabs.add(page, text=title)
        descriptions = tuple(sections[0][0] + "和" + sections[1][0] + "的操作说明。" for _, _, sections in GUIDE_STEPS)
        ToolTip(self.tabs, lambda event: notebook_help(self.tabs, descriptions, event), title="新手指南")
        self.tabs.enable_traversal()
        self.tabs.bind("<<NotebookTabChanged>>", self._select)

        shortcuts = ttk.Label(layout, text="悬停看说明 · Tab 后按 F1 查看焦点 · Shift+F1 固定悬停说明 · Esc 收起",
                              style="CaptureMuted.TLabel", width=1, wraplength=500, justify=tk.LEFT)
        shortcuts.grid(row=1, column=0, sticky="ew", pady=(9, 0))
        wrap_label_on_resize(shortcuts)
        ToolTip(shortcuts, "按钮变灰时仍可悬停查看原因。展开下拉框后，可悬停或按方向键比较候选项。",
                title="查看控件说明")
        actions = ttk.Frame(layout)
        actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.practice_button = None
        if open_practice is not None:
            self.practice_button = ttk.Button(actions, text="打开模拟练习", command=self._launch_practice,
                                              style="Record.TButton")
            ToolTip(self.practice_button, "在独立窗口自动连接内置模拟相机。可练习预览、采前检查、保存和复查。",
                    title="模拟练习")
        close = ttk.Button(actions, text="关闭", command=self.top.destroy)
        close.pack(side=tk.RIGHT)
        ToolTip(close, "回到图像采集。再次打开指南可查看各步骤。")
        self.go = ttk.Button(actions, command=self._go, style="Record.TButton")
        self.go.pack(side=tk.RIGHT, padx=(0, 8))
        ToolTip(self.go, "返回主窗口并定位到对应的控件。", title="定位操作")
        self.top.bind("<Escape>", lambda _event: ToolTip.dismiss_window(self.top) or self.top.destroy())
        self._select()
        self.tabs.focus_set()

    def _select(self, _event=None):
        index = self.tabs.index("current")
        offers_practice = self.practice_button is not None and index == 0
        self.go.configure(text=GUIDE_STEPS[index][1], style="TButton" if offers_practice else "Record.TButton")
        if self.practice_button is not None:
            if offers_practice:
                self.practice_button.pack(side=tk.LEFT)
            else:
                self.practice_button.pack_forget()

    def _launch_practice(self):
        if self.open_practice():
            self.top.destroy()

    def _go(self):
        target = self.targets[self.tabs.index("current")]
        self.top.destroy()
        self.navigate(*target)
