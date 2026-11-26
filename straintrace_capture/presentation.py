"""Chinese display labels for stable Capture configuration and status values."""

from datetime import datetime, timezone

from .preview import QUALITY_ALERT_FRACTION

MODE_LABELS = {
    "single": "单张",
    "timed": "定时采集",
    "continuous": "连续采集",
    "manual": "手动触发",
    "event": "事件前后",
    "hardware": "硬件触发",
}
MODE_HELP = {
    "single": "每台相机保存一张图像，可作为变形分析的参考图或标定图。",
    "timed": "程序按设定频率请求图像。例如填 10 表示每秒请求 10 张。到达张数或时长上限后停止，并记录错过的请求。",
    "continuous": "保存相机实际送达的画面。使用相机设置中的帧率。",
    "manual": "等待操作员或已启用的外部命令。每次触发请求一轮图像。",
    "event": "等待触发时滚动保留最近的原图。收到触发后保存事件前后的画面。可手动触发、由试验机 CSV 触发，也可连续保存多次事件。",
    "hardware": "用接线传给相机的物理脉冲触发拍摄。请为本阶段选择帧开始触发（FrameStart）的输入端口，再试采核对。",
}
ROLE_LABELS = {"test": "实验帧", "reference": "参考帧", "calibration": "标定帧"}
DRIVER_HOST_FRAME_HELP = "保存请求后主机从驱动收到的第一张。曝光同步需独立核实。"
DURABILITY_LABELS = {"buffered": "系统缓冲", "write_through": "逐张刷盘"}
DURABILITY_HELP = {
    "buffered": "默认，适合连续或多相机采集。\n"
                "原图写入并登记到台账后才计入“已存”。\n"
                "停采后，等“待存”归零再查看采集结果。\n"
                "断电可能丢失尚未写到磁盘的原图。",
    "write_through": "每张原图刷盘后登记台账。\n"
                     "高帧率时可先试采，核对写入速度和停采后的等待时间。",
}
ROLE_HELP = {
    "test": "参与后续应变分析的实验图像。",
    "reference": "变形分析使用的比较基准。通常在加载前拍摄。",
    "calibration": "标定板图像，用于求镜头参数或双相机几何。",
}
PHASE_FIELDS = {
    "frame_count": ("每台相机的张数", int), "duration_s": ("最长时长（秒）", float),
    "fps": ("定时频率（张/秒）", float), "frame_timeout_s": ("单帧超时（秒）", float),
    "pre_frames": ("事件前张数", int), "post_frames": ("事件后张数", int),
    "event_count": ("事件次数", int),
}
PHASE_FIELDS_BY_MODE = {
    "single": (), "timed": ("fps", "frame_count", "duration_s"), "continuous": ("frame_count", "duration_s"),
    "manual": ("frame_count", "duration_s"), "hardware": ("frame_count", "duration_s"),
    "event": ("fps", "pre_frames", "post_frames", "event_count", "duration_s"),
}
EVENT_OVERLAP_HELP = (
    "两次触发间隔很短时勾选。第一次缓存满后，即使还在保存前一次事件的后图，也可再次触发。\n"
    "同一张原图只保存一份。每次事件前后的原图可分别查看、分析和导出。"
)
RAW_BUFFER_HELP = (
    "字节缓冲统计仍留在内存中的原始像素，包括事件前缓存。"
    "图像写入文件后，台账登记完成前仍计入待存帧数。"
)
PHASE_FIELD_HELP = {
    "frame_count": "填写每台相机的正整数张数。与时长同时设置时，先达到的条件结束阶段。",
    "duration_s": "填写阶段最多持续的秒数。若设置了开始前确认，从确认后阶段启动时计时。"
                  "手动或硬件触发的时限也包含等待信号的时间。"
                  "定时和手动阶段暂停时，阶段计时同步暂停。",
    "fps": "填 10 表示每秒计划请求 10 张图像。相机与磁盘决定实际能保存的张数，可用试采核对。",
    "pre_frames": "填写每台相机在收到触发前要保留的图像张数。填 0 可只采触发后的图像。"
                  "保留的张数越多，占用的内存越大。",
    "post_frames": "填写每台相机在收到触发后要保存的图像张数，至少 1 张。",
    "event_count": "填 3 接收三次触发。填 0 持续等待触发，直到达到最长时长或点击“停止并保存”。"
                   "留空按 1 次执行。相邻事件可能同时收图时，可勾选“允许事件窗口重叠”。",
    "frame_timeout_s": "等待单次请求或数据到达的秒数。留空使用默认值 3 秒。"
                       "如果相机长曝光或响应慢，可增加超时值，再试采核对。",
}


def phase_limit_text(phase):
    limits = []
    if row_fps := phase.get("fps"):
        limits.append(f"{row_fps:g} 张/秒")
    if count := phase.get("frame_count"):
        limits.append(f"{count} 张")
    if duration := phase.get("duration_s"):
        limits.append(f"{duration:g} 秒")
    if phase.get("post_frames"):
        limits.append(f"前 {phase.get('pre_frames', 0)} / 后 {phase['post_frames']} 张")
    if (events := phase.get("event_count", 1)) != 1:
        limits.append(f"{events} 次事件" if events else "重复到停止")
    if phase.get("overlap_events"):
        limits.append("允许事件重叠")
    if phase.get("csv_trigger"):
        limits.append("CSV 阈值")
    return " / ".join(limits) or "手动停止"


CSV_DIRECTIONS = {"rising": "上穿", "falling": "下穿"}
CSV_STATES = {"opening": "正在打开文件", "watching": "等待越线", "triggered": "已触发",
              "fault": "监看已停用", "finished": "本阶段未自动触发"}


def csv_trigger_rule_text(rule):
    return f"{rule['quantity']} {CSV_DIRECTIONS[rule['direction']]} {rule['threshold']:g} {rule['unit']}"


def csv_trigger_caption(state):
    mode = state["state"]
    caption = f"CSV · {CSV_STATES[mode]}"
    if mode == "watching":
        if "last_value" in state:
            caption += f" · {state['last_value']:g} {state['unit']}"
        elif state.get("invalid_rows"):
            caption = "CSV · 尚未读到有效数值"
        else:
            caption = "CSV · 等待新增记录"
    if state["state"] == "watching" and state.get("early_crossings"):
        caption += " · 曾在缓存未满时越线"
    if mode == "finished" and (accepted := state.get("events_accepted")):
        caption = f"CSV · 已触发 {accepted} 次"
    return caption


def csv_trigger_help(state):
    rows = [csv_trigger_rule_text(state), state["path"], f"触发量列 {state['value_column']}"]
    if state.get("time_column"):
        rows.append(f"试验机时间列 {state['time_column']}")
    if "state" not in state:
        return "\n".join((*rows, "阶段启动后只读取新写完的行。事件前缓存达到计划张数后，再次越线才会触发。"))
    rows.append(csv_trigger_caption(state))
    rows.append(f"已读有效行 {state['samples']} · 无效行 {state['invalid_rows']} · 缓存未满时越线 {state['early_crossings']} 次")
    if "events_accepted" in state:
        rows.append(f"自动触发 {state['events_accepted']} 次 · 收事件后图像时越线 {state['busy_crossings']} 次")
    if state.get("early_crossings") and state["state"] == "watching":
        rows.append("缓存就绪后需要新的越线。也可按现场需要点“触发一次”。")
    if error := state.get("error"):
        rows.extend((error["message"], "需要继续保存事件窗口时，可点“触发一次”。"))
    if crossing := state.get("crossing"):
        rows.extend(csv_crossing_lines(crossing))
    return "\n".join(rows)


def csv_crossing_lines(crossing):
    earlier, later = crossing["previous_sample"], crossing["sample"]
    lines = [f"越线前后的量值 {crossing['previous']:g} → {crossing['value']:g} {crossing['unit']}"]
    if crossing.get("time_column"):
        lines.append(f"试验机时间原文 {earlier['source_time_text'][:60]} → {later['source_time_text'][:60]}")
    lines.extend((f"源文件字节位置 {earlier['offset']:,} → {later['offset']:,}",
                  "相机图像按主机收到触发请求的时刻归入事件前后。"))
    return lines


PHASE_PRESETS = {
    "参考帧和定时实验": (
        {"mode": "single", "role": "reference"}, {"mode": "timed", "role": "test", "fps": 12, "duration_s": 8},
    ),
    "参考图后确认": (
        {"mode": "single", "role": "reference"},
        {"mode": "timed", "role": "test", "fps": 12, "duration_s": 8,
         "start_confirmation": "参考图已登记。请核对实验准备，确认后开始记录。"},
    ),
    "连续记录": ({"mode": "continuous", "role": "test", "duration_s": 10},),
    "手动逐次记录": ({"mode": "manual", "role": "test", "frame_count": 10},),
    "事件前后记录": (
        {"mode": "single", "role": "reference"},
        {"mode": "event", "role": "test", "fps": 10, "pre_frames": 10, "post_frames": 30},
    ),
    "棋盘格标定": ({"mode": "manual", "role": "calibration", "frame_count": 12},),
}
STATE_LABELS = {
    "READY": "待采集", "PREVIEW": "预览中", "PREVIEW PART": "待选图像部件", "PREVIEW FAILED": "预览异常",
    "PREVIEW RETRY": "恢复预览", "OPENING": "连接设备",
    "RUNNING": "采集中", "ARMED": "等待触发", "DRAINING": "写入结束中",
    "PHASE SAVING": "阶段交接", "PHASE WAIT": "等待确认", "PHASE ARMING": "启动阶段",
    "PHASE PAUSING": "正在暂停", "PHASE PAUSED": "采集已暂停", "PHASE RESUMING": "正在继续",
    "BUFFERING": "预缓存中", "TRIGGER QUEUED": "信号已入队", "POST EVENT": "事件后采集",
    "COMPLETED": "采集完成", "STOPPED": "已停止", "FAILED": "采集失败",
    "INTERRUPTED": "已中断", "CHECK DATA": "请复查数据", "IDLE": "待启动",
}
LOAD_TIME_BASIS_LABELS = {"exposure-midpoint": "曝光中点", "host-receive": "主机收图"}
TRACKING_LABELS = {
    "INITIAL_ACCEPTED": "参考帧有效", "ACCEPTED_HARD": "跟踪通过", "ACCEPTED_ADAPTIVE": "自适应阈值通过",
    "AMBIGUOUS_PEAK": "出现重复匹配峰", "CORRELATION_BELOW_THRESHOLD": "相关系数低于阈值",
    "FB_FAILED": "正反向跟踪不一致", "STRAIN_JUMP": "帧间应变变化超限", "REJECTED": "跟踪失败",
}
LOAD_SUPPORT_LABELS = {
    "EXACT_TIME_SAMPLE": "有对应采样点", "LINEAR_TIME_INTERPOLATION": "相邻记录间插值",
    "EXPOSURE_WINDOW_MEAN": "曝光期间平均", "OUTSIDE_TIME_RANGE": "超出载荷记录范围",
    "LONG_TIME_GAP": "采样间隔超限", "INVALID_BRACKET": "相邻载荷有无效值",
    "INVALID_LOAD_BRACKET": "相邻载荷有无效值", "INVALID_EXACT_SAMPLE": "对应载荷无效",
    "NO_COMPATIBLE_LOAD_TIME": "图像时钟不可用", "LOAD_SYNC_CLOCK_MISMATCH": "与同步帧的时钟不一致",
    "EXPOSURE_WINDOW_MISSING": "缺少曝光起止时刻", "INVALID_EXPOSURE_WINDOW": "曝光起止时刻无效",
    "MECHANICS_SOURCE_UPDATING": "等待载荷表写入完成", "LOAD_TIME_AXIS_INVALID": "载荷表时间有无效值",
    "LOAD_TIME_AXIS_NOT_STRICTLY_INCREASING": "载荷时间重复或倒退", "LOAD_SOURCE_UNAVAILABLE": "无法读取载荷表",
    "LOAD_ABSOLUTE_TIME_REQUIRED": "载荷表缺少可靠的绝对时间", "LOAD_CLOCK_CONFLICT": "载荷表混用了多套时钟",
    "LOAD_SOURCE_EMPTY": "等待载荷记录", "LOAD_TIME_COLUMN_MISSING": "载荷表缺少所选时间列",
    "LOAD_FORCE_COLUMN_MISSING": "载荷表缺少所选载荷列", "LOAD_RELATIVE_SOURCE_CHANGED": "载荷表已重写，请重新核对同步帧",
}
PROVIDER_LABELS = {"sim": "模拟", "uvc": "视频设备", "pylon": "Basler", "gentl": "GenTL"}
PROVIDER_STATES = {"ready": "可用", "no_devices": "未发现设备", "unavailable": "缺少组件", "error": "发现失败"}
_PROVIDER_BINDINGS = {"PYLON_UNAVAILABLE": "capture-pylon", "GENTL_UNAVAILABLE": "capture-gentl"}
_PROVIDER_DIAGNOSTICS = {
    "PYLON_SDK_LOAD_FAILED": ("采集组件加载失败", "请重新解压工业相机便携版，或检查当前 Python 环境中的 Basler 组件。"),
    "GENTL_SDK_LOAD_FAILED": ("采集组件加载失败", "请重新解压工业相机便携版，或检查当前 Python 环境中的 GenTL 组件。"),
    "GENTL_NO_CTI": ("缺少厂商接口文件", "请安装相机厂商的 64 位驱动或 SDK，再选择原安装目录中的 .cti。"),
    "GENTL_CTI_INVALID": ("找不到接口文件", "请重新选择厂商安装目录中的 .cti。"),
    "GENTL_CTI_LOAD_FAILED": ("厂商接口加载失败", "请确认 .cti 为 64 位，并从厂商原安装目录选择，保留它旁边的依赖文件。"),
    "DUPLICATE_CAMERA_ID": ("发现重复设备", "请只选择对应接口的厂商 .cti，刷新后再连接相机。"),
}
PREVIEW_STATES = {"preview": "预览中", "connecting": "连接中", "adjusting": "更新参数中",
                  "disconnected": "已断开", "failed": "预览失败", "releasing": "释放旧连接",
                  "retry_wait": "等待自动重连", "reconnecting": "重连中"}
CAPTURE_ERROR_LABELS = {
    "CSV_TRIGGER_SOURCE_UNAVAILABLE": "试验机 CSV 读不到。请检查路径和文件是否可读。",
    "CSV_TRIGGER_SOURCE_CHANGED": "试验机 CSV 已被截断、改写或替换。",
    "CSV_TRIGGER_HEADER_INVALID": "试验机 CSV 首行无法用作表头。",
    "CSV_TRIGGER_HEADER_CHANGED": "试验机 CSV 表头已变化。请重新选择触发量列和时间列。",
    "CSV_TRIGGER_COLUMN_MISSING": "试验机 CSV 缺少计划指定的列。",
    "CSV_TRIGGER_ROW_INVALID": "试验机 CSV 有无法解析的数据行。",
    "MARKER_NOT_RECORDING": "采集已经结束或尚未开始。请在采集中记录标记。",
    "MARKER_LABEL_INVALID": "标记说明最多填写 160 个字。",
    "MARKER_QUEUE_FULL": "标记正在写入。请稍候再记录。",
    "MARKER_WRITE_FAILED": "这条标记未能写入台账。请查看保存状态。",
    "MARKER_NOT_FOUND": "台账中没有这条标记。可刷新采集时间线。",
    "MARKER_CLOCK_UNAVAILABLE": "标记和原图缺少可比较的主机时钟。",
    "PRACTICE_CAMERA_REQUIRED": "模拟练习只使用内置相机。请在图像采集主窗口打开实验相机的计划或存档。",
    "OUTPUT_INSIDE_REPEAT_SOURCE": "请选择来源实验目录之外的新输出位置。",
    "REPEAT_SCOPE_CHANGED": "来源实验或计划已变化。请重新打开复拍计划。",
    "REVIEW_SCOPE_CHANGED": "实验标识已变化。请重新打开要复查的实验。",
    "REPEAT_SOURCE_UNSUPPORTED": "请选择一次正式采集作为复拍的来源。",
    "REPEAT_ACTIVE": "请先结束采集或恢复中断台账，再准备复拍。",
    "REPEAT_CALIBRATION_INVALID": "已归档的标定文件缺失或已变化。请核对来源实验。",
    "REPEAT_NAME_REQUIRED": "请填写本次复拍的实验名称。",
    "WRITE_PROBE_GEOMETRY": "图像尺寸或像素格式待核对。请连接预览或补齐相机参数，再检查计划。",
    "WRITE_PROBE_LIMIT": "写盘测试的样本上限不合适。可用错误详情核对本次上限和图像大小。",
    "WRITE_PROBE_SCOPE_CHANGED": "目标卷或保存位置已变化。请重新检查计划。",
    "WRITE_PROBE_CANCELLED": "已取消写盘测试。",
    "WRITE_PROBE_VERIFY": "写盘测试的图像与台账校验失败。请查看错误详情。",
    "CAPTURE_DRAFT_INVALID": "本机草稿无法读取。可载入已导出的计划文件。",
    "TRIAL_IMAGE_CHECK_CANCELLED": "本次原图检查已停止。",
    "TRIAL_IMAGE_REGION_CHANGED": "这帧的图像尺寸、格式或部件发生变化。可单独打开原图确认检查框。",
    "TRIAL_REGION_BOUNDS": "检查框超出原图范围。请在试采原图上重新框选。",
    "ARCHIVE_QUALITY_ACTIVE": "请在采集完成或中断恢复后检查已保存的原图。",
    "ARCHIVE_QUALITY_LAYOUT_CHANGED": "这张原图的尺寸、格式或主图部件发生变化，按全图检查。",
    "ARCHIVE_QUALITY_SENSOR_CHANGED": "这张原图的传感器画幅与保存的检查框不匹配，按全图检查。",
    "QUALITY_REPORT_SUFFIX": "请选择 .zip 文件名导出画质报告。",
    "QUALITY_REPORT_INSIDE_ARCHIVE": "请选择原实验目录以外的位置保存画质报告。",
    "QUALITY_REPORT_EXISTS": "同名画质报告已存在。请换一个文件名。",
    "QUALITY_REPORT_PARENT": "请选择已经存在的保存文件夹。",
    "QUALITY_REPORT_CANCELLED": "已取消导出画质报告。",
    "QUALITY_REPORT_CLOSED": "整场检查结果已关闭。请重新检查后导出。",
    "TRIAL_REGION_IMAGE_LAYOUT": "当前相机计划或预览的画幅、像素格式或图像部件与试采原图不匹配。请在当前预览重新框选。",
    "TRIAL_REGION_SENSOR_WINDOW": "当前相机的传感器范围与试采不一致或待核对。请在当前预览重新框选。",
    "TRIAL_REGION_PLAN_CHANGED": "核对期间采集计划已修改。确认相机和设置后，再点一次“用于采集抽查”。",
    "TRIAL_REGION_SOURCE": "请在所选相机的试采报告中打开原图，再保存检查框。",
    "TRIAL_REFERENCE_UNAVAILABLE": "找不到采前关联的试采目录。可选择搬移后的试采目录。",
    "TRIAL_REFERENCE_MISMATCH": "所选目录的实验编号与采前试采记录不同。",
    "TRIAL_REFERENCE_CHANGED": "试采原帧在台账中的记录与开拍前不同。请定位当时使用的试采目录。",
    "INVALID_CONFIG": "采集配置有误。请检查修改过的字段和数值。",
    "HOST_FRAME_READ_FAILED": "主机选帧期间读图失败。请检查相机连接和连续预览。",
    "FEATURE_TYPE": "参数类型不符合相机要求。请按提示输入整数、小数或选择开关。",
    "FEATURE_CHOICE": "相机不支持此选项。请使用下拉列表中的值。",
    "FEATURE_READ_ONLY": "当前相机状态下该参数只读。",
    "FEATURE_RANGE": "数值超出相机允许的范围。",
    "FEATURE_INCREMENT": "数值不符合相机要求的步长。",
    "FEATURE_UNAVAILABLE": "这台相机不提供所选参数。请检查当前设备和驱动。",
    "PHASE_OWNS_TRIGGER": "请在采集阶段设置触发方式。",
    "PIXEL_FORMAT_UNSUPPORTED": "Capture 无法按原格式保存此像素格式。",
    "GENTL_BAD_PAYLOAD": "GenTL 图像数据与报告的尺寸或格式不一致。请核对相机参数并重新试采。",
    "GENTL_INCOMPLETE_BUFFER": "GenTL 相机返回的图像不完整。请检查连接和传输速率，再试采。",
    "GENTL_PACKING_AMBIGUOUS": "无法从 GenTL 缓冲区确认打包图像的行边界。请按详细信息调整宽度，或选用同位深的非打包格式。",
    "GENTL_PART_SELECTOR_INVALID": "GenTL 图像部件编号格式错误。",
    "GENTL_PART_AMBIGUOUS": "当前帧包含多个匹配的二维图像部件。请在预览下方选择。没有可选项时，请核对相机输出。",
    "GENTL_PART_MISSING": "当前帧缺少已选部件或图像来源已改变。请在预览中核对相机输出和所选部件。",
    "GENTL_PART_SET_AMBIGUOUS": "多个二维图像部件使用相同编号。请核对相机的部件配置后重新试采。",
    "GENTL_PART_SET_CHANGED": "采集中途收到的二维部件或尺寸发生变化。请确认相机输出，再开启续采。",
    "PYLON_BAD_PAYLOAD": "无法读取 Basler 图像的原始像素。请核对像素格式和相机连接，再试采。",
    "PYLON_PIXEL_FORMAT_MISMATCH": "Basler 收到的帧与设置的像素格式不一致。请核对相机参数并重新试采。",
    "FEATURE_WRITE_FAILED": "相机拒绝了参数。请核对范围、步长和自动设置。",
    "FEATURE_READBACK_MISMATCH": "回读值与设定值不同。请用相机接受的范围、步长和采集模式重试。",
    "PREVIEW_NOT_READY": "相机还没有送出预览图像。请等画面出现后再调参。",
    "PREVIEW_PART_REQUIRED": "请先在预览下方选择这台相机的图像部件。",
    "PREVIEW_PART_ADJUST_RESTORED": "没有收到符合新设置的图像，计划已保留原参数。请在预览下方选择图像部件。",
    "PREVIEW_SETTING_UNAVAILABLE": "当前参数无法在预览中回读和恢复。请断开预览后调参。",
    "PREVIEW_ADJUST_RESTORED": "新参数应用失败，已恢复原参数和预览。请核对输入值和驱动提示。",
    "EXPOSURE_UNAVAILABLE": "请悬停查看当前相机的曝光测量条件。",
    "EXPOSURE_SCOPE_CHANGED": "相机、画面或计划已变化。请重新测量。",
    "EXPOSURE_REGION_TOO_SMALL": "检查框太小。可扩大检查框，或清除检查框后测量全图。",
    "EXPOSURE_CODE_RANGE": "原始像素超出相机记录的位深范围。请检查像素格式。",
    "EXPOSURE_SAMPLE_TIMEOUT": "收到的新画面不足。请检查预览是否连续，再重新测量。",
    "EXPOSURE_LIMIT": "最长曝光时间不在相机允许的范围内。悬停可查看详情。",
    "EXPOSURE_CANCELLED": "已取消曝光测量。",
    "PREVIEW_RESTORE_FAILED": "预览调参失败，并且无法恢复相机参数。请断开预览，检查接线和设备状态。",
    "PREVIEW_TIMEOUT": "未收到符合当前参数的预览图像。请检查图像尺寸、格式和设备连接。",
    "PREVIEW_DEVICE_ERROR": "相机驱动未能完成预览。请查看详细信息并检查连接。",
    "PREVIEW_RETRY_UNAVAILABLE": "这台相机当前不能重连。请查看画面下方的连接状态。",
    "PREVIEW_IDENTITY_CHANGED": "设备身份与原预览不一致。请断开预览，刷新设备并核对相机身份。",
    "PREVIEW_GEOMETRY_CHANGED": "新画面的尺寸、像素格式或部件与原来不同。请重新核对相机参数和检查区域。",
    "CAMERA_ID_UNRESOLVED": "当前找不到唯一匹配的相机。请检查电源、接线和设备编号。",
    "UVC_FRAME_UNAVAILABLE": "视频设备停止送图。请检查 USB 连接和系统相机权限。",
    "UVC_DRIVER_TIMEOUT": "系统视频驱动仍占用旧连接。请恢复设备连接或关闭占用相机的软件。",
    "RAW_BUFFER_FULL": "图像缓冲达到上限。请核对已保存图像。可降低帧率、缩小图像或缩短事件前窗口。",
    "WRITE_QUEUE_FULL": "待写队列达到批次上限。请核对已保存图像，并在采前检查中复核队列设置。",
    "DISK_RESERVE": "磁盘剩余空间已达保留值。请检查已保存图像，并为续采选择空间充足的位置。",
    "FRAME_TIMEOUT": "等待相机图像超时。请检查设备连接与曝光，再用预览验证。",
    "WAIT_OUTPUT_SELECTOR_RESTORE_FAILED": "相机原选中的线路未能恢复。请检查设备连接，再核对厂商工具中的线路选择。",
    "EVENT_WINDOW_INCOMPLETE": "事件次数或事件后张数未达到计划。请在时间线核对实际收图。",
    "EVENT_LEDGER_QUEUE_FULL": "事件记录积压达到上限。请核对已保存的事件和原图，并检查输出磁盘。",
    "EVENT_LEDGER_WRITE_FAILED": "事件记录写入失败。请检查输出磁盘，再打开时间线核对已登记的事件和原图。",
    "CAMERA_DISCONNECTED": "采集期间相机断开。请检查连接，重新预览后准备续采。",
    "WRITE_FAILED": "写入失败。请检查输出目录、可用空间和磁盘状态，再校验已保存图像。",
    "FRAME_JOURNAL_WRITE_FAILED": "帧写入记录保存失败。请检查输出磁盘，再打开实验复查已保存图像。",
    "CAMERA_UNAVAILABLE": "无法打开相机。请检查设备连接、驱动和相机占用状态。",
    "GENTL_CTI_INVALID": _PROVIDER_DIAGNOSTICS["GENTL_CTI_INVALID"][1],
    "GENTL_CTI_LOAD_FAILED": _PROVIDER_DIAGNOSTICS["GENTL_CTI_LOAD_FAILED"][1],
    "CAMERA_ALREADY_OPEN": "此相机已有连接。请先断开其他预览或关闭占用相机的软件。",
    "INVALID_LEDGER": "无法读取采集台账。请选择含 capture.sqlite3 的实验总目录。",
    "EXPERIMENT_UNAVAILABLE": "无法读取此位置的采集台账。若实验已移动，请选择它的新目录。",
    "LEDGER_CHECKPOINT_FAILED": "无法保存实验台账的初始记录。请检查输出位置后重试。",
    "RECENT_EXPERIMENT_MISMATCH": "该目录包含另一个实验。请定位所选实验的实际目录。",
    "WRITER_ACTIVE": "该实验仍有采集进程写入。请等采集结束后再恢复台账。",
    "OUTPUT_EXISTS": "目标目录已存在。请选择尚未创建的目录。",
    "PREVIEW_SAVE_EXISTS": "同名原图已存在。请换一个文件名再保存。",
    "PREVIEW_SAVE_SUFFIX": "请选择 .tif 或 .tiff 文件名保存原图。",
    "PREVIEW_ORIGINAL_METADATA": "请打开由 Capture 的“保存预览原图”生成的 TIFF。",
    "PREVIEW_ORIGINAL_LAYOUT": "TIFF 像素无法按保存时的画幅读取。请核对原始文件。",
    "TRANSFER_ACTIVE": "请先结束采集并等待台账写完。如果原采集异常退出，请先检查中断记录。",
    "TRANSFER_CANCELLED": "已取消复制。使用相同的源实验、附件和目标目录，可继续核验并复用已有文件。",
    "TRANSFER_CHANGED": "文件内容与记录不符。请核对报告中的文件或重新从原实验复制。",
    "TRANSFER_MANIFEST": "交接清单无法读取或与实验范围不符。请核对副本来源。",
    "TRANSFER_MISSING": "找不到交接所需的文件。请查看详细信息并核对目录。",
    "TRANSFER_PATH": "复制涉及的文件路径、名称或文件类型不符合要求。请核对详细信息。",
    "TRANSFER_PUBLISH": "目标位置无法按所需的方式生成新目录。请选择其他磁盘或目录。",
    "TRANSFER_SCOPE": "源实验、附件或临时目录与本次复制不一致。请核对选择，也可使用新的目标名称。",
    "TRANSFER_SPACE": "目标盘剩余空间不足。悬停查看还需要多少空间，或选择空间更充足的位置。",
    "SEQUENCE_CANCELLED": "已取消图像序列导出。",
    "SEQUENCE_EMPTY": "选定范围内没有晚于参考图的实验帧。请检查阶段、帧号范围和参考图。",
    "SEQUENCE_PHASE": "请选取用于记录变形的实验阶段，或选择全部实验阶段。",
    "EVENT_IMAGES_SCOPE": "未找到对应的实验事件。请从采集时间线重新选择。",
    "EVENT_IMAGES_ACTIVE": "当前无法使用事件原图。可查看详情并核对采集与导出状态。",
    "EVENT_IMAGES_EMPTY": "这台相机在所选事件中没有可用的实验图像。",
    "EVENT_REFERENCE_REQUIRED": "请明确选择本次事件的参考图。",
    "EVENT_REFERENCE_OUTSIDE_SCOPE": "所选参考图与本次事件范围不符。",
    "SEQUENCE_SENSOR_CHANGED": "所选阶段的传感器窗口、合并采样或抽样设置不同。请缩小范围并选用相同设置拍摄的参考图。",
    "RECOVERY_SCOPE_CHANGED": "实验目录中的标识已变化。请重新打开实验。",
    "CONTINUATION_ACTIVE": "先结束当前采集或恢复中断台账，再准备续采。",
    "CONTINUATION_PHASE_INVALID": "请从原实验的阶段列表中选择续采起点。",
    "CONTINUATION_REFERENCE_ACTIVE": "请等原参考图所在实验结束。离线分析也需结束本轮采集。",
    "CONTINUATION_REFERENCE_SOURCE": "请选择台账关联的原实验目录。",
    "CONTINUATION_REFERENCE_CAMERA": "两轮的相机身份无法一致核对。请检查所选相机和原实验。",
    "CONTINUATION_REFERENCE_SEQUENCE": "请选择或输入原实验登记的参考图序号。",
    "CONTINUATION_REFERENCE_GEOMETRY": "两轮图像的成像区域或采样方式有差异。请核对相机设置。",
    "CONTINUATION_REFERENCE_CHANGED": "实验或原图记录已变化。请重新选择参考图。",
    "PHASE_CONFIRMATION_INVALID": "请核对开始确认所对应的阶段和标识。",
    "PHASE_CONFIRMATION_NOT_PENDING": "当前没有等待确认的阶段。",
    "PHASE_CONFIRMATION_CHANGED": "正在等待的阶段已变化，请核对当前提示。",
    "PHASE_CONFIRMATION_SAVING": "请等上一阶段的图像完成登记，再确认开始。",
    "PHASE_CONFIRMATION_REQUESTED": "已收到本阶段的确认，请等待相机启动。",
    "PHASE_CONFIRMATION_LEDGER_FAILED": "无法保存阶段确认记录。下一阶段未启动，请检查输出目录和磁盘状态。",
    "PHASE_PAUSE_INVALID": "请核对暂停所对应的阶段号。",
    "PHASE_PAUSE_CHANGED": "阶段或暂停记录已变化，请查看当前采集状态。",
    "PHASE_PAUSE_NOT_ACTIVE": "当前阶段无法执行此暂停或继续操作。",
    "PHASE_PAUSE_UNSUPPORTED": "暂停适用于正式采集中的定时和手动触发阶段。",
    "PHASE_PAUSE_TRANSITION": "采集正在暂停或已暂停，请确认状态后继续采集。",
    "PHASE_PAUSE_SIGNALS_QUEUED": "请等已接收的触发请求执行后再暂停。",
    "PHASE_PAUSE_LEDGER_FAILED": "无法保存暂停或继续记录。请检查输出目录，并复查已保存图像。",
    "CONTROL_PHASE_PAUSE_DISABLED": "请先启用外部命令，再使用命令行暂停或继续。",
    "RELATED_SCOPE_CHANGED": "实验标识或关联记录已变化。请重新打开实验。",
    "RELATED_INVALID_EXPERIMENT": "无法读取实验标识或关联记录。请核对所选目录的台账。",
    "CHOOSE_CAMERA": "请选择要分析的相机。",
    "INVALID_FRAME_QUERY": "图像查询条件无效。请检查阶段、序号范围或收图时间。",
    "FRAME_REVIEW_CAMERA": "请在这次实验的台账中选择一台相机。",
    "FRAME_REVIEW_CHANGED": "实验或帧记录已变化。请重新打开对应实验并查询。",
    "FRAME_REVIEW_UNAVAILABLE": "当前无法查找已保存图像。请查看操作状态。",
    "INVALID_TIMELINE_QUERY": "时间线查询条件无效。请刷新时间线后重试。",
    "CHOOSE_REFERENCE": "这台相机有多张参考帧。请指定用于计算应变的帧序号。",
    "REFERENCE_UNAVAILABLE": "找不到所选参考帧。请核对相机和序号，或先保存参考图。",
    "FRAME_UNAVAILABLE": "台账中的图像文件缺失。请先校验实验。",
    "TOO_FEW_FRAMES": "分析需要一张参考图和至少一张实验图。请检查已保存图像的用途。",
    "TRIGGER_GROUP_UNAVAILABLE": "台账中找不到此触发组。请刷新触发记录。",
    "HARDWARE_INPUT_INVALID": "请核对相机选择及阶段的物理输入。",
    "PULSE_AUDIT_ACTIVE": "请结束采集并等台账写完，再核对外部脉冲。",
    "PULSE_AUDIT_PHASE": "请选择登记过共享计数器锚点的硬件触发阶段。",
    "PULSE_AUDIT_RANGE": "请检查起止编号和所选核对范围。",
    "PULSE_CSV_INVALID": "无法读取脉冲 CSV。请检查列名、行和整数编号。",
    "PULSE_CSV_EVENT_COLUMN": "请选择记录脉冲整数编号的列。",
    "PULSE_CSV_EMPTY_SCOPE": "CSV 在所选阶段、信号源和编号范围内没有相关脉冲。",
    "PULSE_CSV_SOURCE_CHANGED": "请使用核对时的 DAQ 文件，或载入更新后的文件重新核对。",
    "PULSE_AUDIT_OUTPUT_FORMAT": "请选择 JSON 报告或 CSV 脉冲表。",
    "PULSE_BINDING_INVALID": "请检查两次确认脉冲及各台相机的原图序号。",
    "PULSE_BINDING_SCOPE": "请核对锚点所属实验、阶段和范围。",
    "PULSE_BINDING_FRAME": "无法核实所选原图。请核对相机、序号和文件。",
    "PULSE_BINDING_COUNTER": "锚点之间的计数变化无法用于补配。",
    "PULSE_BINDING_DAQ": "请检查两次锚点是否在 DAQ 文件中各记录一次。",
    "PULSE_TIMING_CSV_INVALID": "无法读取 DAQ 时序 CSV。请检查表头、相机编号和纳秒时间。",
    "PULSE_TIMING_CLOCK": "请核对 DAQ 时基和脉冲编号顺序。",
    "PULSE_TIMING_LIMITS": "请检查曝光时序验收限值。",
    "DETECTOR_AUDIT_INVALID": "请选择“核对 DAQ 脉冲”导出的完整 JSON 报告。",
    "DETECTOR_AUDIT_SCOPE": "请核对报告所属实验、阶段和锚点，或重新核对 DAQ 文件。",
    "DETECTOR_CSV_INVALID": "无法读取探测器 CSV。请检查列名、行和整数编号。",
    "DETECTOR_COLUMNS_MISSING": "请分别选择脉冲编号列和探测器帧标识列。",
    "DETECTOR_OUTPUT_FORMAT": "请选择 JSON 报告或逐相机关联 CSV。",
    "CAPTURE_IMAGE_CHANGED": "原图校验值与台账不符。请核对实验目录中的原图。",
    "ANALYSIS_PART_SELECTOR": "选择要分析的图像部件。命令行需要完整编号。",
    "ANALYSIS_PART_MISSING": "序列中有帧缺少所选图像部件。请核对帧序号和已保存的图像。",
    "ANALYSIS_PART_AMBIGUOUS": "一帧中有多个图像使用相同部件编号，无法唯一选出分析图像。",
    "FRAME_GEOMETRY_CHANGED": "图像部件、尺寸或像素格式与参考帧不同。请核对序列。",
    "FRAME_UNREADABLE": "这张原图无法读取。请在实验目录中核对图像文件。",
    "CAPTURE_BINDING_INVALID": "采集绑定文件无法与本实验的图像对应。请选择该实验生成的校正序列。",
    "CORRECTED_IMAGE_CHANGED": "校正图像校验失败。可用原图和标定重新生成校正序列。",
    "ANALYSIS_LAUNCH_FAILED": "校正序列已保存。可在分析窗口的“采集联动”中重新打开它。",
    "CALIBRATION_UNAVAILABLE": "请载入这台相机的镜头标定，或使用棋盘格图像拟合。",
    "ANALYSIS_ROI_REQUIRED": "请先在参考图上绘制两个标距端点区域，或绘制一个全场区域。",
    "ROI_INVALID": "请输入 x、y、宽、高四个整数。宽高须大于零。",
    "ROI_OUT_OF_BOUNDS": "区域超出了参考图。请在图像内重新绘制。",
    "INSPECTION_ROI_TOO_LARGE": "检查区域较大，请缩小方框后再计算局部纹理。",
    "BAYER_DERIVATION_REQUIRED": "Bayer 图像需要先转换为亮度图。实时分析可勾选“将 Bayer 图像转换为亮度”。",
    "LIVE_CSV_EXTENSION": "请使用 .csv 作为导出文件的扩展名。",
    "LIVE_CSV_CANCELLED": "已取消导出。",
    "LIVE_CSV_EMPTY_RANGE": "范围内没有已读取的实验帧。请核对帧号，或重新读取历史记录。",
    "LIVE_RESULT_EMPTY": "所选记录中还没有可以导出的完整结果。",
    "LIVE_RESULT_INVALID": "所选记录中有无法识别的结果。请重新选择实验的逐帧记录。",
    "LIVE_RESULT_SEQUENCE_GAP": "逐帧保存出现断点。可重置跟踪并重新保存。",
    "LIVE_RECORD_CHANGED": "逐帧记录在读取期间发生变化。请重试。",
    "LIVE_REPLAY_CANCELLED": "已取消重算。",
    "LIVE_REPLAY_NO_TEST": "参考图之后还没有已保存的实验图。采集到实验图后可重试。",
    "LIVE_REPLAY_SETUP": "无法沿用这份记录的测量设置。可在实时分析中重新设置测量区。",
    "LIVE_REPLAY_REFERENCE_CHANGED": "无法核对原记录的参考图。请检查参考图及其所属实验。",
    "LIVE_REPLAY_FRAME_CHANGED": "原记录的实验图或归属与台账不一致。请检查对应序号。",
    "LIVE_REPLAY_INCOMPLETE": "重算期间的实验图范围发生变化。请重新读取记录后重试。",
    "LIVE_REPLAY_LOAD_ERROR": "载荷对齐失败。请检查原载荷表或选择只重算图像。",
    "LIVE_HISTORY_OTHER_EXPERIMENT": "请选择属于当前采集实验的逐帧记录。",
    "LIVE_CSV_OVERWRITES_LOAD": "请选择新文件名，保留原载荷表。",
    "LIVE_CSV_OVERWRITES_RECORD": "请选择新的文件名，保留原逐帧记录。",
    "LOAD_ARCHIVE_SETTINGS": "保存的载荷设置无法用于补齐。可导出原实时结果。",
    "LOAD_ARCHIVE_CONFLICT": "早期采样的校验记录不一致。可导出原实时结果。",
    "LOAD_ARCHIVE_MISSING": "这份实时记录没有载荷对齐设置。可导出原实时结果。",
    "LOAD_ARCHIVE_FILE_MISSING": "找不到载荷表。请选择同一轮实验的更新表。",
    "LOAD_ARCHIVE_UNVERIFIED": "缺少可核对的载荷采样。请使用原先设置的绝对时间载荷表。",
    "LOAD_ARCHIVE_PREFIX_CHANGED": "目前的载荷表无法核对早期采样。请选择同一轮实验的载荷表。",
    "LOAD_ARCHIVE_TABLE_INVALID": "所选载荷表读取失败。请核对列名和文件编码。",
    **{code: LOAD_SUPPORT_LABELS[code] for code in (
        "MECHANICS_SOURCE_UPDATING", "LOAD_TIME_AXIS_INVALID", "LOAD_TIME_AXIS_NOT_STRICTLY_INCREASING",
        "LOAD_ABSOLUTE_TIME_REQUIRED", "LOAD_CLOCK_CONFLICT", "LOAD_RELATIVE_SOURCE_CHANGED",
        "LOAD_TIME_COLUMN_MISSING", "LOAD_FORCE_COLUMN_MISSING",
    )},
    "LOAD_HEADER_INVALID": "请核对首行的时间列、载荷列和分隔符。",
    "LOAD_SOURCE_EMPTY": "载荷表尚无记录。可保存后重试。",
    "LOAD_ENCODING_INVALID": "请核对文件编码，或把载荷表另存为 UTF-8。",
    "LOAD_COLUMNS_REQUIRED": "请选择不同的列作为时间、载荷和可选的位移。",
    "LOAD_TIME_FORMAT_INVALID": "请选择时间格式和数字时间的单位。",
    "CLOCK_BASIS_INVALID": "请选择曝光中点或主机收图时刻。",
    "ALIGNMENT_MODE_INVALID": "曝光窗口平均需要曝光起止时刻。请核对图像时刻和取值方式。",
    "LOAD_FORCE_UNIT_REQUIRED": "请核对并选择载荷单位。",
    "DISPLACEMENT_UNIT_UNSUPPORTED": "请选择位移列的长度单位。",
    "LOAD_MAX_GAP_INVALID": "最大插值间隔应为大于零的秒数。",
    "LOAD_SYNC_FRAME_REQUIRED": "请选择已保存的同步帧序号，并核对表内对应时间。",
    "LOAD_SYNC_FRAME_CLOCK": "所选同步帧缺少时钟读数。请核对图像时刻的选择。",
    "LOAD_SYNC_TIME_INVALID": "请输入有限的表内同步事件时间。",
    "LOAD_SYNC_TIME_OUTSIDE": "同步时刻超出载荷表范围。请核对数值和单位。",
    "ALPHA_DERIVATION_REQUIRED": "这组图像含透明度通道。请先根据相机颜色生成亮度图后再分析。",
    "CALIBRATION_GEOMETRY_MISMATCH": "当前图像尺寸或传感器区域与标定不符，请选择对应的标定文件。",
    "CALIBRATION_PART_MISMATCH": "请选择与当前图像部件对应的标定文件，或切回标定时使用的图像部件。",
    "CALIBRATION_PART_UNVERIFIED": "无法确认标定对应的 GenTL 部件。可以用该部件的原棋盘格图像重新拟合。",
    "CALIBRATION_IMAGE_UNAVAILABLE": "请先连接预览或选中一张已保存的原图。",
    "CALIBRATION_FILE_MISSING": "找不到与实验一同归档的标定文件。请检查实验目录。",
    "CALIBRATION_HASH_MISMATCH": "实验归档中的标定文件校验失败。请核对原标定文件。",
    "CALIBRATION_CAMERA_UNBOUND": "所选标定文件缺少这台相机的标定。",
    "CALIBRATION_PROFILE_INVALID": "标定文件中的相机参数或图像部件有误。请选择有效的标定文件。",
    "CALIBRATION_CAMERAS_INVALID": "请选择同一实验中的一台相机，或两台不同的相机。",
    "CALIBRATION_INPUT_CHANGED": "标定原图校验失败。请在复查页校验图像和台账。",
    "CALIBRATION_VIEWS_INSUFFICIENT": "可识别的棋盘格图像不足。请检查角点数和图像清晰度，补拍更多角度。",
    "CALIBRATION_VIEWS_REPEATED": "棋盘格位置和角度变化不足。请从不同位置和倾角补拍。",
    "CALIBRATION_GEOMETRY_CHANGED": "标定图像的部件、尺寸或传感器设置发生了变化。请使用同一路图像和相同的设置重采。",
    "STEREO_BOARD_PARITY": "双相机标定需要一边为奇数、另一边为偶数的内角点数。",
    "STEREO_IMAGE_SIZE_MISMATCH": "双相机标定需要两台相机使用相同的图像宽高。",
}


def provider_diagnostic(name, state, *, frozen=False):
    status = PROVIDER_STATES.get(state["state"], state["state"])
    code = state.get("code")
    if code in _PROVIDER_BINDINGS:
        if frozen:
            return status, "请使用 Windows 工业相机便携版。"
        return status, f"请在当前 Python 环境安装 straintrace[{_PROVIDER_BINDINGS[code]}]。"
    if code in _PROVIDER_DIAGNOSTICS:
        return _PROVIDER_DIAGNOSTICS[code]
    if state["state"] == "no_devices":
        if name == "pylon":
            return status, "请用 pylon Viewer 确认相机已连接，再返回 Capture 刷新。"
        if name == "gentl":
            return status, "请用厂商软件确认相机已连接，并选择对应传输接口的 .cti。"
    return status, ""


def format_provider_report(providers, *, frozen=False):
    if not providers:
        return "设备搜索中。列表更新后可再次查看。"
    blocks = []
    for name, state in providers.items():
        status, guidance = provider_diagnostic(name, state, frozen=frozen)
        lines = [f"{PROVIDER_LABELS.get(name, name)} · {status}"]
        if "count" in state:
            lines.append(f"发现 {state['count']} 台设备")
        if guidance:
            lines.append(guidance)
        if state.get("message"):
            lines.append(f"详细信息  {state['message']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


TRIGGER_ERROR_LABELS = {
    "SIGNAL_PHASE_CHANGED": "采集阶段已切换，本次触发未执行。请核对当前阶段。",
    "SIGNAL_PHASE_EXPIRED": "当前采集阶段已到时限，请查看阶段状态。",
    "EVENT_ALREADY_RECEIVED": "当前事件已有触发请求，请查看事件后采集进度。",
    "EVENT_TARGET_REACHED": "已接收计划中的全部触发，正在完成事件后图像。",
    "SIGNAL_WINDOW_CHANGED": "事件已切换。请确认当前编号，再触发下一次事件。",
    "PREBUFFER_NOT_READY": "下一次事件的前图仍在缓存。等界面显示就绪后再触发。",
    "NOT_ARMED": "当前阶段暂不接受操作员触发，请查看采集状态。",
    "SIGNAL_QUEUE_FULL": "触发请求队列已满，请等待正在执行的请求处理完毕。",
}
TEXTURE_GUIDANCE = {
    "LOW_TEXTURE": "局部可辨认的纹理变化较少。可检查对焦、散斑密度和局部照明。",
    "AMBIGUOUS_TEXTURE": "纹理主要沿单一方向变化，或存在重复图案。可另选同时含横向和竖向细节的区域。",
    "SATURATED_TEXTURE": "选区明暗端有较多像素被截断。请调整光照或曝光后再检查。",
}
FEATURE_HELP = {
    "width": ("采集宽度", "修改相机实际读出的列数。改变后需要重新核对标定和测量区。"),
    "height": ("采集高度", "修改相机实际读出的行数。改变后需要重新核对标定和测量区。"),
    "offset_x": ("水平偏移", "相机有效区域相对传感器左边界的位置，单位像素。"),
    "offset_y": ("竖直偏移", "相机有效区域相对传感器上边界的位置，单位像素。"),
    "pixel_format": ("像素格式", "决定保存的灰度、彩色或 Bayer 格式以及有效位数。"),
    "gentl_image_part": ("GenTL 主图像", "auto 自动选择唯一的二维图像。连接预览后可从画面下方选择。也可手动填写完整编号，例如 2/16/0/Mono8。"),
    "gentl_save_all_2d": ("保存全部二维部件", "为同一帧中的各个可解码二维部件分别写入原始 TIFF。复查时可选择任意一路图像做应变分析。"),
    "exposure_us": ("曝光时间", "相机每张图收光的时间。1000 微秒等于 1 毫秒。曝光越长画面通常越亮，运动边缘也更容易模糊。"),
    "exposure_auto": ("自动曝光", "启用时由相机调节曝光。加载过程中需要固定光照条件时，可核对是否关闭。"),
    "gain_db": ("增益", "放大相机信号，可让画面更亮，也可能放大噪声。调整后可在预览里比较散斑细节。"),
    "gain_auto": ("自动增益", "是否让相机在采集过程中调整增益。"),
    "fps": ("相机帧率", "相机目标每秒输出的图像张数。定时阶段另有软件请求频率。可用试采核对实际保存速度。"),
    "fps_enable": ("启用帧率设置", "相机支持时应用设定的输出帧率。"),
    "acquisition_mode": ("采集模式", "显示相机当前的采集模式。"),
    "trigger_mode": ("触发开关", "显示相机当前是否启用触发。"),
    "trigger_selector": ("触发对象", "显示相机当前触发所控制的采集操作。"),
    "trigger_source": ("触发源", "所用的触发事件或物理输入线。采集阶段会按计划配置触发。"),
    "trigger_activation": ("触发沿", "设备当前使用的信号边沿。按接线与 DAQ 的触发极性核对。"),
    "ptp_enable": ("启用 PTP", "使用设备支持的精密时间协议。需要从现场时钟记录核实同步状态。"),
    "motion_px": ("模拟平移", "练习时每张图移动的像素量，用于测试跟踪。"),
    "strain_per_frame": ("模拟应变增量", "练习时每张图增加的应变。例如 0.0005 对应每张增加 0.05%。"),
    "disconnect_after": ("模拟断开时机", "达到指定张数后让模拟相机断开，用于练习异常处理。填 0 表示持续连接。"),
}

ISSUE_LABELS = {
    "CSV_TRIGGER_SOURCE_UNAVAILABLE": CAPTURE_ERROR_LABELS["CSV_TRIGGER_SOURCE_UNAVAILABLE"],
    "CSV_TRIGGER_HEADER_INVALID": CAPTURE_ERROR_LABELS["CSV_TRIGGER_HEADER_INVALID"],
    "CSV_TRIGGER_HEADER_CHANGED": CAPTURE_ERROR_LABELS["CSV_TRIGGER_HEADER_CHANGED"],
    "CSV_TRIGGER_COLUMN_MISSING": CAPTURE_ERROR_LABELS["CSV_TRIGGER_COLUMN_MISSING"],
    "OUTPUT_EXISTS": "输出路径已存在，请选择一个尚未创建的实验文件夹。",
    "OUTPUT_INSIDE_REPEAT_SOURCE": "复拍目录需要位于来源实验之外。请选择新的保存位置。",
    "OUTPUT_PARENT_NOT_DIRECTORY": "输出路径的父级是文件，请更换保存位置。",
    "OUTPUT_INSPECTION_FAILED": "无法读取输出路径或可用空间，请检查路径和权限。",
    "DISK_RESERVE_UNAVAILABLE": "磁盘剩余空间低于保留空间，请换到空闲空间更多的卷。",
    "DEPENDENCY_FILE_MISSING": "找不到计划引用的 GenTL 生产者或标定文件，请更新路径。",
    "FRAME_GEOMETRY_UNAVAILABLE": "部分阶段的图像尺寸或格式未知。连接预览，或试采对应阶段后再估算。",
    "PREVIEW_IMAGE_QUALITY": "预览画质需要检查。可定位到该相机的画面。",
    "QUALITY_REGION_CHANGED": "采集抽查范围与当前画幅不匹配。",
    "TRIAL_NEEDS_REVIEW": "匹配当前设置的试采结果需要复查。请参考报告中的实测问题。",
    "TRIAL_IMAGES_NEED_REVIEW": "试采原图有亮度、纹理或文件读取提醒。请打开原图复核。",
    "TRIAL_CAMERA_READBACK_CHANGED": "当前相机身份或回读与试采有差异。",
    "CONTINUATION_SETUP_CHANGED": "本轮计划或相机回读与原实验有差异。请查看续采设置对照。",
    "CAMERAS_NOT_DISCOVERED": "未发现部分选定相机。请检查连接、驱动，并刷新设备。",
    "CAMERA_CONNECTION_CHECK_FAILED": "所选相机连接检查失败。处理后可刷新并重新检查计划。",
    "SIMULATED_CAMERAS": "计划包含模拟相机，适合熟悉流程和验证配置。",
    "NO_EXPLICIT_REFERENCE": "后续要做变形分析时，可在实验阶段前加一张参考帧。",
    "SOFTWARE_TRIGGER_UNSUPPORTED": "部分相机缺少这个阶段所需的相机端软件触发。",
    "HARDWARE_TRIGGER_UNSUPPORTED": "下列相机不支持物理触发。请更换设备，或按实验要求另选采集模式。",
    "HARDWARE_INPUT_INVALID": "请选择这台相机的物理触发输入。",
    "HARDWARE_INPUT_UNAVAILABLE": "这台相机当前的 FrameStart 目录未列出所选输入。",
    "TRIGGER_PROBE_FAILED": "读取相机触发能力失败。请检查连接或设备占用。",
    "CAMERA_SETTING_INVALID": "计划中的相机参数需要修改。",
    "CAMERA_SETTINGS_PROBE_FAILED": "无法读取相机参数。请检查连接或设备占用，再重新检查。",
    "CAMERA_SETTINGS_UNCONFIRMED": "部分参数需要复核。连接预览或试采时可检查设备是否接受这些设置。",
    "HOST_FRAME_SELECTION": "这些相机连续送图。每次请求保存请求后主机收到的第一张，曝光同步需独立核实。",
    "DRIVER_HOST_FRAME_SELECTION": "使用驱动送图。" + DRIVER_HOST_FRAME_HELP,
    "HARDWARE_PAIRING_UNVERIFIED": "多相机物理触发缺少已核实的计数器锚点。请核实脉冲身份后再用于配对。",
    "WAITING_FOR_EXTERNAL_INPUT": "手动或硬件触发阶段未设置时限。没有信号时会持续等待。",
    "EVENT_WINDOW_EXCEEDS_DURATION": "时限内排不下事件前后的全部图像。请延长时限或调整窗口张数。",
    "FRAME_LIMIT_UNAVAILABLE": "阶段的总张数无法估算。需要预算时可填写最长时长或适用的次数上限。",
    "RAW_BUFFER_TOO_SMALL": "一轮数据或事件前缓存预计超过缓冲限制。请缩小图像、缩短事件前窗口或调整缓冲上限。",
    "EXPOSURE_EXCEEDS_INTERVAL": "设定的曝光时间长于计划拍摄间隔。请缩短曝光或降低定时频率。",
    "PLAN_EXCEEDS_AVAILABLE_SPACE": "按计划保存的像素数据预计会超过可用空间。请更换输出卷或缩短采集。",
}

READINESS_CONFIRMATION_ISSUES = frozenset({
    "PLAN_EXCEEDS_AVAILABLE_SPACE", "RAW_BUFFER_TOO_SMALL", "EXPOSURE_EXCEEDS_INTERVAL",
    "TRIAL_NEEDS_REVIEW", "TRIAL_IMAGES_NEED_REVIEW", "TRIAL_CAMERA_READBACK_CHANGED", "CONTINUATION_SETUP_CHANGED",
})

INTEGRITY_LABELS = {
    "TRANSFER_MANIFEST_MISMATCH": "交接副本与交接清单不符",
    "MISSING_FILE": "缺少台账所列图像",
    "UNREADABLE_FILE": "图像无法读取",
    "FILE_SIZE_MISMATCH": "图像大小与台账不符",
    "FILE_HASH_MISMATCH": "图像校验值与台账不符",
    "CALIBRATION_FILE_MISSING": "缺少归档标定文件",
    "CALIBRATION_HASH_MISMATCH": "归档标定文件校验值不符",
    "UNCOMMITTED_FILE": "文件尚未登记到台账",
    "TRIGGER_EVENT_UNKNOWN": "找不到对应的触发事件",
    "EVENT_CONTEXT_UNKNOWN": "原图找不到本阶段的事件触发请求",
    "EVENT_FRAME_POSITION_MISMATCH": "原图的事件前后位置与主机收图时刻不符",
    "EVENT_WINDOW_FRAME_COUNT_MISMATCH": "事件原图张数与台账记录不符",
    "EVENT_FRAME_LINK_MISMATCH": "原图的事件归属与台账记录不符",
    "EVENT_FRAME_LINK_MISSING_IMAGE": "事件关联的原图未登记",
    "EVENT_FRAME_LINK_TABLE_MISSING": "台账缺少事件与原图的关联",
    "SHARED_HARDWARE_COUNTER_GAP": "共享触发计数存在缺口",
    "FRAME_JOURNAL_UNSUPPORTED": "无法识别帧写入记录格式",
    "FRAME_JOURNAL_MISSING": "找不到台账所列的帧写入记录",
    "FRAME_JOURNAL_UNREADABLE": "无法读取帧写入记录",
    "FRAME_JOURNAL_ENTRY_INVALID": "帧写入记录不完整或校验不符",
    "FRAME_LEDGER_CONFLICT": "帧身份或触发记录与现有台账冲突",
    "FRAME_PAYLOAD_PATH_CONFLICT": "图像路径指向其他位置",
    "FRAME_PAYLOAD_MISSING": "帧写入记录对应的图像未找到",
    "FRAME_PAYLOAD_MISMATCH": "遗留图像的大小或校验值不符",
    "FRAME_PAYLOAD_UNAVAILABLE": "无法读取或恢复遗留图像",
}


def ui_error(error, *, details=True):
    if isinstance(error, str):
        return error
    code = getattr(error, "code", None)
    if code:
        title = CAPTURE_ERROR_LABELS.get(code) or TEXTURE_GUIDANCE.get(code)
        if title is None:
            title = "采集操作未完成。请查看错误详情。"
        if details:
            return f"{title}\n\n详细信息\n{code}  {error}"
        return title
    if isinstance(error, OSError):
        title = "无法打开或写入文件。请检查路径、权限和磁盘空间。"
        return f"{title}\n\n详细信息\n{error}" if details else title
    return f"操作未完成。\n\n详细信息\n{error}" if details else "操作未完成。请查看错误详情。"


def format_bytes(num, *, small_units=False):
    if num is None:
        return "待估算"
    if num >= 1024**3:
        return f"{num / 1024**3:.2f} GiB"
    if small_units and num < 1024**2:
        return f"{num / 1024:.1f} KiB" if num >= 1024 else f"{num:,} B"
    return f"{num / 1024**2:.2f} MiB"


WRITE_PRESSURE_LABELS = {
    "idle": "等待入队", "sampling": "正在测量", "writing": "正在写入", "prebuffer": "正在预缓存",
    "growing": "短时积压", "recovering": "积压回落", "sustained": "写入持续落后",
    "queue_growth": "待写批次持续增加", "near_limit": "缓冲接近上限",
    "draining": "正在保存待写帧", "failed": "采集异常",
}
WRITE_PRESSURE_GUIDANCE = {
    "idle": "近期没有新的写入任务。",
    "sampling": "正在收集入队和登记速率。需有完整的观测区间才能比较。",
    "writing": "图像正在写入。可对照近期的入队和登记速率。",
    "prebuffer": "事件前图像占用字节缓冲，触发后按事件窗口入队。",
    "growing": "最近一个区间的待写像素增加。可继续观察它是否回落。",
    "recovering": "最近一个区间登记的像素多于入队的像素，待写量正在回落。",
    "sustained": "连续两个观测区间的登记速率低于入队速率，待写帧持续增加。需要提前结束时可点“停止并保存”。",
    "queue_growth": "连续两个观测区间的待写批次增加。需要提前结束时可点“停止并保存”。",
    "near_limit": "字节缓冲或待写批次已接近上限。需要提前结束时可点“停止并保存”。",
    "draining": "已入队的图像会继续写入并登记。",
    "failed": "请查看采集状态。可在采后核对已保存图像和台账。",
}


def write_pressure_estimate(estimate):
    if estimate is None:
        return ""
    target = {"bytes": "字节缓冲上限", "batches": "待写批次上限"}[estimate["resource"]]
    seconds = estimate["seconds"]
    if seconds <= 0:
        return f"当前已达{target}。"
    span = ("不足 1 秒" if seconds < 1 else f"约 {seconds:.0f} 秒" if seconds < 60
            else f"约 {seconds / 60:.1f} 分钟")
    return f"若近期增速持续，{span}触及{target}。"


def write_pressure_tip(pressure):
    if not pressure:
        return "开始采集后可查看写入趋势。"
    notes = [WRITE_PRESSURE_GUIDANCE[pressure["state"]]]
    if windows := pressure["windows"]:
        latest = windows[-1]
        notes.append(f"最近 {latest['duration_s']:.1f} 秒 · 入队 {format_bytes(latest['accepted_raw_bytes_per_s'])}/s"
                     f" · 已登记 {format_bytes(latest['committed_raw_bytes_per_s'])}/s")
    if estimate := write_pressure_estimate(pressure["estimate"]):
        notes.append(estimate)
    return "\n".join(notes)


WAIT_OUTPUT_LABELS = {
    "checking": "正在读取等待输出", "ready": "相机报告可接收触发", "not_ready": "相机尚未报告可接收触发",
    "received": "已收到首帧", "stale": "回读已过时", "unavailable": "未能核实等待输出",
}
WAIT_OUTPUT_REASONS = {
    "driver_unavailable": "当前相机接口未提供此功能",
    "line_catalog_unavailable": "相机未提供可读取的线路目录或状态节点",
    "wait_output_not_configured": "未找到配置为 FrameTriggerWait 的输出线",
    "output_polarity_unconfirmed": "输出反相或极性无法确认",
    "line_selection_changed": "相机的线路选择发生变化",
    "line_status_unreadable": "驱动无法返回最新的线路状态",
}


def wait_output_camera_text(row):
    state = row["state"]
    label = WAIT_OUTPUT_LABELS[state]
    if state != "received" and row.get("line"):
        label += f" · 等待输出 {row['line']}"
    if state == "unavailable" and (reason := row.get("reason")):
        label += f" · {WAIT_OUTPUT_REASONS.get(reason, reason)}"
    return label


def wait_output_help(readiness, names, *, inputs=None):
    inputs = inputs or {}
    lines = []
    for camera, row in readiness["cameras"].items():
        source = f" · 计划输入 {inputs[camera]}" if inputs.get(camera) else ""
        lines.append(f"{names.get(camera, camera)}{source} · {wait_output_camera_text(row)}")
    lines.append("界面轮询相机的等待输出。连续触发请使用实际接线的信号。")
    return "\n".join(lines)


PHASE_PAUSE_DISPLAY = {
    "requested": ("PHASE PAUSING", "当前拍摄轮次完成后暂停。"),
    "paused": ("PHASE PAUSED", "拍摄和阶段计时已暂停。可继续采集或停止并保存。"),
    "resuming": ("PHASE RESUMING", "正在确认继续操作，随后恢复本阶段的拍摄。"),
    "stopping": ("DRAINING", "正在保存剩余图像并结束采集。"),
}


def event_window_position(window):
    number, count = window["event_number"], window["event_count"]
    if window.get("event_target_reached"):
        return f"已接收 {window['accepted_events']}/{count} 次触发"
    return f"事件 {number}/{count}" if count else f"事件 {number}"


def finishing_phase_writes(status):
    return (status["state"] == "running" and status["phase_progress"] is None
            and any(row["pending_write"] for row in status["cameras"].values()))


def phase_display(mode, progress, *, pause=None):
    """Show the controller's state and observations in the current phase."""
    if pause is not None:
        return PHASE_PAUSE_DISPLAY[pause["state"]]
    if window := progress["event_window"]:
        state = window["state"]
        prefix = f"{event_window_position(window)} · " if window.get("overlap") or window.get("event_count", 1) != 1 else ""
        pre = f"{window['pre_buffered_rounds']}/{window['pre_required_rounds']}"
        if window.get("overlap"):
            active = window["active_events"]
            if window["event_target_reached"]:
                return "POST EVENT", prefix + (f"{active} 次事件正在收图" if active else "事件原图等待写入登记")
            if active:
                hint = "可继续触发" if progress["trigger_available"] else "等待正在收图的事件完成"
                return "POST EVENT", f"{active} 次事件正在收图 · {hint}"
        if state == "buffering":
            return "BUFFERING", f"{prefix}事件前缓存 {pre} 张/相机"
        if state == "waiting":
            message = f"事件前缓存 {pre} 张/相机，等待触发" if window["pre_required_rounds"] else "等待事件触发"
            return "ARMED", prefix + message
        if state == "trigger_queued":
            return "TRIGGER QUEUED", prefix + "已收到触发请求，等待执行"
        post = f"{window['post_received_rounds']}/{window['post_required_rounds']}"
        return "POST EVENT", f"{prefix}事件后已接收 {post} 张/相机"
    if mode == "hardware":
        count = sum(progress["received_by_camera"].values())
        if count:
            return "RUNNING", f"硬件触发 · 已接收 {count} 张"
        if readiness := progress.get("hardware_readiness"):
            if readiness["ready"]:
                return "ARMED", "所有相机报告可接收首个硬件触发"
            if readiness["state"] == "unverified":
                return "ARMED", "已启动接收 · 部分等待输出未能核实"
            available = sum(row["state"] == "ready" for row in readiness["cameras"].values())
            return "ARMED", f"{available}/{len(readiness['cameras'])} 台相机报告可接收触发"
        return "ARMED", "采集已启动，等待相机送来图像"
    if mode == "manual":
        if progress["manual_signal_id"]:
            return "RUNNING", "正在执行手动触发"
        if pending := progress["queued_signals"]:
            return "TRIGGER QUEUED", f"{pending} 个触发请求待执行"
        return "ARMED", "等待操作员或外部命令"
    return "RUNNING", MODE_LABELS[mode]


def hardware_trigger_guidance(progress):
    counts = progress["received_by_camera"]
    started = sum(count > 0 for count in counts.values())
    if counts and started == len(counts):
        return "硬件触发采集中", "各台相机均已送来首帧。可查看各相机的收图和保存进度。"
    if started:
        return "等待其余相机首帧", f"已收 {started}/{len(counts)} 台首帧。触发后无图可核对接线。"
    if readiness := progress.get("hardware_readiness"):
        _state, detail = phase_display("hardware", progress)
        title = "等待硬件触发" if readiness["ready"] else "核对相机就绪状态"
        return title, f"{detail}。可逐台查看计划输入和等待输出。"
    return "等待硬件触发", "采集已启动，主机还未收到本阶段首帧。可逐台核对计划输入和接线。"


def _trial_advice(finding, camera_labels):
    code = finding["code"]
    cameras = "、".join(camera_labels.get(key, key) for key in finding.get("cameras", ()))
    if code == "TRIAL_CAPTURE_ERROR":
        return CAPTURE_ERROR_LABELS.get(finding["capture_code"], "查看试采末尾的错误信息，处理后重新试采。")
    if code == "TRIAL_FILE_ISSUES":
        return f"校验发现 {finding['count']} 项差异。请在复查页点“校验图像和台账”，定位受影响的文件。"
    if code == "TRIAL_RECHECK_UNAVAILABLE":
        return "未能重新打开原试采。请核对实验目录及台账，再打开试采报告。"
    if code == "TRIAL_NO_FRAMES":
        action = {
            "manual": "开始后点“触发一次”，或用已启用的外部命令触发。",
            "event": "等待事件前缓存就绪，再点“触发一次”或发出外部命令。",
            "hardware": "检查相机输入线，并在试采时限内发送物理触发。",
        }.get(finding["mode"], "重新连接预览检查相机是否送回图像，再试采。")
        return f"{cameras} 未保存图像。{action}"
    if code == "TRIAL_VARIABLE_GEOMETRY":
        return f"{cameras} 的图像尺寸或格式有变化。固定相机参数后重新试采。"
    if code == "TRIAL_CAMERA_LOSSES":
        return f"{cameras} 有帧异常。核对相机连接、接收间隔和采集台账。"
    if code == "TRIAL_UNSAVED_FRAMES":
        return f"有 {finding['count']} 张已收到的图像未保存，请在复查页核对已保存图像。"
    if code == "TRIAL_SCHEDULE_MISSES":
        return f"错过 {finding['count']} 次定时请求。核对曝光与帧间隔，降低定时频率后重新试采。"
    if code == "TRIAL_SHORT_PRE_WINDOW":
        return "触发时事件前缓存未攒足。等界面显示等待触发后再发送信号，必要时延长试采时限。"
    if code == "TRIAL_HIGH_BUFFER":
        action = ("可减少事件前保留张数、降低频率或缩小图像。" if finding["mode"] == "event" else
                  "可检查输出盘写入能力、降低帧率或缩小图像，再试采。")
        return f"缓冲峰值达到上限的 {finding['fraction']:.0%}。{action}"
    if code == "TRIAL_HIGH_WRITE_QUEUE":
        action = ("可在“检查计划”核对是否适合改用自动队列。" if finding["queue_mode"] == "fixed" else
                  "可降低采集速率，或选用持续写入速度更高的输出卷。")
        return action
    if code == "TRIAL_SUSTAINED_WRITE_LAG":
        return "采集末段持续积压。建议降低帧率、缩小图像，或更换写入更快的输出卷，再用新设置试采。"
    if code == "TRIAL_LOW_RECEIPT_RATE":
        identity = finding["camera"]
        return (f"{camera_labels.get(identity, identity)} 实收 {finding['observed_fps']:.2f} Hz，设定 {finding['target_fps']:g} Hz。"
                "核对曝光时间、相机输出上限和传输设置。")
    return ""


def camera_setting_detail(issue):
    feature = issue["feature"]
    name = FEATURE_HELP.get(feature, (feature,))[0]
    unit = {"px": "像素", "us": "微秒"}.get(issue["unit"], issue["unit"] or "")
    unit = f" {unit}" if unit else ""
    reason = issue["reason"]
    detail = [f"{name}填了 {issue['value']}{unit}。{CAPTURE_ERROR_LABELS.get(reason, issue['detail'])}"]
    if reason in ("FEATURE_RANGE", "FEATURE_INCREMENT"):
        lower, upper = issue["minimum"], issue["maximum"]
        if lower is not None and upper is not None:
            detail.append(f"当前范围 {lower:g}–{upper:g}{unit}")
        elif lower is not None:
            detail.append(f"最小值 {lower:g}{unit}")
        elif upper is not None:
            detail.append(f"最大值 {upper:g}{unit}")
        if inc := issue["increment"]:
            if len(detail) > 1:
                detail[-1] += f" · 步长 {inc:g}{unit}"
            else:
                detail.append(f"步长 {inc:g}{unit}")
    if reason in ("FEATURE_CHOICE", "PIXEL_FORMAT_UNSUPPORTED") and issue["choices"]:
        detail.append("可选值 " + "、".join(map(str, issue["choices"])))
    return "\n".join(detail)


def readiness_issue_location(issue, phase_offset, camera_labels, *, include_path=True):
    details = []
    if issue.get("phase"):
        details.append(f"阶段 {issue['phase'] + phase_offset}")
    if include_path and issue.get("path"):
        details.append(issue["path"])
    if camera := issue.get("camera"):
        details.append(camera_labels.get(camera, camera))
    if cameras := issue.get("cameras"):
        details.append("、".join(camera_labels.get(camera, camera) for camera in cameras))
    return " · ".join(details)


def preview_quality_detail(summary):
    count = summary["frames"]
    if not count:
        return ""
    roi = summary["roi"]
    scope = (f"检查框 {roi[2]} × {roi[3]} 像素，左上角 {roi[0]}, {roi[1]}"
             if roi else "整幅画面")
    moment = datetime.fromtimestamp(summary["sampled_utc_ns"] / 1e9).strftime("%H:%M:%S")
    lines = [
        f"{count} 张新预览 · {scope} · {moment}",
        f"近黑最多 {summary['maximum_dark_fraction']:.1%} · 近饱和最多 {summary['maximum_saturated_fraction']:.1%}",
    ]
    dark, bright = summary["dark_frames"], summary["saturated_frames"]
    if dark or bright:
        frames = []
        if dark:
            frames.append(f"{dark} 张近黑比例达到 {QUALITY_ALERT_FRACTION:.0%}，可检查照明不足或遮挡。")
        if bright:
            frames.append(f"{bright} 张近饱和比例达到 {QUALITY_ALERT_FRACTION:.0%}，可检查反光或曝光过长。")
        lines.extend(frames)
    if roi:
        if alerts := summary["texture_alerts"]:
            lines.extend(TEXTURE_GUIDANCE.get(code, code) for code in alerts)
        elif summary["texture_frames"] < count:
            lines.append("部分画面无法评估框内纹理。可调整检查框后复测。")
    return "\n".join(lines)


def readiness_issue_detail(issue):
    code = issue["code"]
    if code.startswith("CSV_TRIGGER_"):
        return issue.get("message", "")
    if code == "CAMERA_SETTING_INVALID":
        return camera_setting_detail(issue)
    if code == "CAMERA_CONNECTION_CHECK_FAILED":
        hint = ("关闭这台相机的连接时出错。请核对占用设备的程序。"
                if issue["state"] == "close_failed" else CAPTURE_ERROR_LABELS.get(
                    issue["reason"], "无法打开这台相机。请检查供电、接线或设备占用。"))
        detail = issue.get("detail")
        lines = [f"设备编号 {issue['camera']}", hint]
        if detail and detail != hint:
            lines.append(detail)
        return "\n".join(lines)
    if code == "PREVIEW_IMAGE_QUALITY":
        return preview_quality_detail(issue)
    if code == "QUALITY_REGION_CHANGED":
        return f"{quality_region_summary(issue)}\n画幅不匹配时改为全图抽查。可在画质页重新框选或恢复全图。"
    if code == "CAMERA_SETTINGS_UNCONFIRMED":
        return "、".join(FEATURE_HELP.get(key, (key,))[0] for key in issue["fields"])
    if code == "HARDWARE_INPUT_INVALID":
        return f"计划输入 {issue['input']} · 可检查相机的 Line、CC 或 CoaXPress 触发输入名称。"
    if code == "HARDWARE_INPUT_UNAVAILABLE":
        available = "、".join(issue["available"]) or "未列出物理输入"
        return f"计划输入 {issue['input']} · 当前目录 {available}"
    if code in ("OUTPUT_INSPECTION_FAILED", "TRIGGER_PROBE_FAILED", "CAMERA_SETTINGS_PROBE_FAILED"):
        return issue.get("detail") or ""
    if code == "DISK_RESERVE_UNAVAILABLE":
        return (f"当前可用 {format_bytes(issue['free_bytes'], small_units=True)}"
                f" · 需保留 {format_bytes(issue['reserve_bytes'], small_units=True)}")
    if code == "PLAN_EXCEEDS_AVAILABLE_SPACE":
        return (f"预计 {format_bytes(issue['estimated_bytes'], small_units=True)}"
                f" · 扣除保留空间后可用 {format_bytes(issue['available_bytes'], small_units=True)}")
    if code == "RAW_BUFFER_TOO_SMALL":
        return (f"预计需要 {format_bytes(issue['estimated_bytes'], small_units=True)}"
                f" · 缓冲上限 {format_bytes(issue['buffer_bytes'], small_units=True)}")
    if code == "EXPOSURE_EXCEEDS_INTERVAL":
        return f"曝光 {issue['exposure_us']:g} 微秒 · 拍摄间隔 {issue['interval_us']:g} 微秒"
    if code == "EVENT_WINDOW_EXCEEDS_DURATION" and "planned_rounds" in issue:
        rows = (f"按时长和频率最多 {issue['planned_rounds']} 轮"
                f" · 每次事件前 {issue['pre_frames']} 轮、后 {issue['post_frames']} 轮")
        if count := issue.get("event_count"):
            rows += f" · 共 {count} 次"
        return rows
    if code == "TRIAL_IMAGES_NEED_REVIEW":
        return f"有 {issue['flagged_images']} 张试采原图需要复核。"
    if issue.get("stereo_calibration"):
        return "双相机标定需要同一触发下的两幅图像。请更换支持对应触发方式的相机。"
    if issue.get("suggested_frame_selection") == "independent_receipts":
        return "各台相机已在预览中送图。可分别保存各自收到的第一张，保留该阶段的图像用途。"
    if issue.get("suggested_frame_selection") == "host_receipt":
        return "各台相机已在预览中送图。可保留该阶段的请求方式和参数，由主机在请求后取帧。曝光同步需独立核实。"
    if issue.get("offer_preview"):
        return "各相机都能在预览中持续送图时，可将单张阶段改为连续采集，每台保存 1 张。"
    if issue.get("offer_host_preview"):
        return "预览确认各相机能送图后，可启用按主机到达选帧，按原计划发出采集请求。"
    return ""


def readiness_issue_text(issue, phase_offset, camera_labels):
    code = issue["code"]
    prefix = "阻塞" if issue["severity"] == "block" else "需复核" if code in READINESS_CONFIRMATION_ISSUES else "提示"
    parts = (f"【{prefix}】{ISSUE_LABELS[code]}", readiness_issue_location(issue, phase_offset, camera_labels),
             *readiness_issue_detail(issue).splitlines())
    return "\n  ".join(part for part in parts if part)


_PHASE_FIELD_LABELS = {
    "mode": "采集方式", "role": "帧用途", "fps": "定时频率", "frame_count": "张数", "duration_s": "时长",
    "hardware_source": "物理触发输入", "pre_frames": "事件前张数", "post_frames": "事件后张数",
    "software_trigger_fallback": "按收图选帧", "frame_timeout_s": "单张等待时限",
    "start_confirmation": "开始前确认", "monitor_trigger_wait": "检查触发就绪输出",
    "csv_trigger": "试验机 CSV 阈值",
    "event_count": "事件次数",
}
_READBACK_LABELS = {"driver_exposure": "驱动曝光值", "driver_gain": "驱动增益值"}


def _setup_value(value, unit=None):
    if value is None:
        return "未记录"
    if type(value) is bool:
        return "开" if value else "关"
    text = f"{value:.12g}" if isinstance(value, float) else str(value)
    unit = "µs" if unit == "us" else unit
    return f"{text} {unit}" if unit else text


def _setup_feature(name):
    return FEATURE_HELP.get(name, (_READBACK_LABELS.get(name, name),))[0]


def _phase_change_labels(fields):
    return "、".join(_PHASE_FIELD_LABELS.get(field, field) for field in fields)


def _phase_change_value(field, value):
    labels = {"mode": MODE_LABELS, "role": ROLE_LABELS}.get(field, {})
    unit = {"fps": "Hz", "duration_s": "秒", "frame_timeout_s": "秒"}.get(field)
    return _setup_value(labels.get(value, value) if isinstance(value, str) else value, unit)


IMAGE_QUALITY_ALERT_LABELS = {
    "near_black": "近黑偏多", "near_saturation": "近饱和偏多", "sharpness_down": "清晰度下降",
}


def image_quality_alert_text(codes):
    return "、".join(IMAGE_QUALITY_ALERT_LABELS.get(code, code) for code in codes)


def quality_region_summary(region):
    x, y, width, height = region["bounds"]
    text = f"检查框 ({x}, {y}) · {width} × {height} px"
    reason = region.get("fallback")
    if reason == "image_layout":
        return f"{text} · 图像尺寸、像素格式或主图部件不匹配"
    if reason == "sensor_window":
        return f"{text} · 传感器画幅偏移或采样参数不匹配"
    return text


def image_quality_sample_help(sample):
    metrics = sample["metrics"]
    lines = [
        f"阶段 {sample['phase']} · 序号 {sample['sequence']} · {sample['pixel_format']} 主图",
        f"近黑 {metrics['near_black_fraction']:.1%} · 近饱和 {metrics['near_saturation_fraction']:.1%}",
    ]
    region = sample.get("region")
    if region is None:
        lines.insert(1, "抽查范围 全图")
    else:
        lines.insert(1, f"抽查范围 {'全图 · ' if region.get('fallback') else ''}{quality_region_summary(region)}")
    ratio = metrics.get("sharpness_ratio")
    if ratio is not None:
        lines.append(f"清晰度响应为本阶段参照的 {ratio:.0%}。可核对拖影、焦点和试样纹理。")
    elif sample.get("compare_sharpness"):
        count = sample.get("reference_samples", 0)
        lines.append(f"清晰度参照已收集 {count}/3 张。亮度变化较大时暂缓比较。")
    return "\n".join(lines)


CAPTURE_EVENT_LABELS = {
    "operator_marker": "实验标记",
    "phase_started": "阶段开始", "phase_ended": "阶段结束", "phase_confirmation_waiting": "等待人工确认",
    "phase_confirmed": "已确认开始", "phase_confirmation_cancelled": "取消等待",
    "phase_paused": "暂停采集", "phase_resume_requested": "收到继续请求", "phase_pause_cancelled": "暂停结束",
    "image_quality_notice": "画质抽查",
    "csv_trigger": "试验机 CSV 阈值",
    "event_window": "事件前后采集",
    "event_phase_summary": "触发请求未接收 · 阶段汇总",
    "schedule_miss": "计划漏拍", "native_gap_unattributed": "相机计数跳号", "native_duplicate": "相机计数重复",
    "native_reset_or_wrap": "相机计数回退", "trigger_incomplete": "触发组图像不齐",
    "hardware_identity_unavailable": "硬件触发无法配对", "pre_window_short": "事件前图像不足",
    "signal_unused": "触发信号未使用", "error": "采集错误",
}


def _continuation_plan_lines(report, labels):
    lines = ["续采设置对照", f"原实验  {report['source_experiment_id']}"]
    start = report.get("source_start_phase", 1)
    if start > 1:
        lines.append(f"本轮阶段 1 对照原阶段 {start}，后续阶段依次对照。")
    for key, prompt in (("missing_cameras", "原实验中本次未选的相机"), ("added_cameras", "本次新增相机")):
        if report[key]:
            lines.append(prompt + "  " + "、".join(labels.get(camera, camera) for camera in report[key]))
    for row in report["phase_changes"]:
        previous, current = row["previous"], row["current"]
        source = row.get("source_phase")
        stage = f"阶段 {row['phase']}"
        if source is not None and source != row["phase"]:
            stage += f"（原阶段 {source}）"
        if not previous or not current:
            lines.append(f"{stage} · {'本次新增' if current else '本次移除'}")
        else:
            lines.append(f"{stage} · 计划有变化")
            for field in row["fields"]:
                before = _phase_change_value(field, previous.get(field))
                after = _phase_change_value(field, current.get(field))
                lines.append(f"  {_PHASE_FIELD_LABELS.get(field, field)}  {before} → {after}")
    bases = {"device_features": "设备空闲回读", "preview_readback": "按本次计划预览后回读", "not_probed": "待连接预览"}
    identities = {"matched": "序列号一致", "path_matched": "设备路径一致",
                  "unverified": "身份待核对", "changed": "本次读到不同相机"}
    for camera, data in report["readbacks"].items():
        fields = data["fields"]
        camera_name = labels.get(camera, camera)
        if not fields:
            lines.append(f"{camera_name} · 原实验没有可对照的相机回读")
            continue
        same = sum(row["status"] == "matched" for row in fields)
        unknown = sum(row["status"] == "unverified" for row in fields)
        lines.append(f"{camera_name} · {bases.get(data['basis'], data['basis'])} · {identities[data['identity']]}"
                     f" · 一致 {same} 项 · 待核验 {unknown} 项")
        for row in (item for item in fields if item["status"] in ("plan_changed", "readback_changed")):
            source = _setup_value(row["recorded"], row["unit"])
            current = _setup_value(row["current"], row["current_unit"])
            planned = _setup_value(row["planned"], row["unit"]) if row["planned"] is not None else "待设备回读"
            lines.append(f"  {_setup_feature(row['field'])}  原记录 {source} · 当前 {current} · 计划 {planned}")
        if proposed := report["suggested_settings"].get(camera):
            units = {row["field"]: row["unit"] for row in fields}
            lines.append("  可补入计划  " + "、".join(
                f"{_setup_feature(key)} {_setup_value(value, units.get(key))}" for key, value in proposed.items()))
        applying = [_setup_feature(item["field"]) for item in fields if item["status"] == "will_apply"]
        if applying:
            lines.append("  计划将尝试恢复原记录值  " + "、".join(applying))
    return lines


def _trial_image_lines(checks, camera_labels):
    for identity, check in checks.items():
        coverage = "全查" if check["mode"] == "all" else "均匀抽查"
        yield (f"  {camera_labels.get(identity, identity)} · {coverage} {check['selected']}/{check['available']} 张"
               f" · 已核验 {check['checked']} 张 · 提醒 {check['flagged']} 张")


def _trial_region_lines(regions, camera_labels):
    reasons = {"plan_changed": "当前计划的检查框已变更", "integrity": "原图或台账待复查",
               "camera_changed": "相机回读与试采存在差异",
               "incomplete": "检查框内有图像未完成核验", "no_images": "试采没有已登记图像"}
    for camera, check in regions.items():
        reason = check.get("fallback")
        bounds = {"bounds": check["region"]["bounds"], "fallback": reason}
        note = quality_region_summary(bounds)
        if reason:
            detail = f" · {reasons[reason]}" if reason in reasons else ""
            note += f"{detail} · 按全图复核"
        elif check.get("used_for_review"):
            note += " · 用于采前亮度复核"
        yield f"  {camera_labels.get(camera, camera)} · {note}"
        if data := check.get("check"):
            coverage = "全查" if data["mode"] == "all" else "均匀抽查"
            yield (f"    {coverage} {data['selected']}/{data['available']} 张"
                   f" · 已核验 {data['checked']} 张 · 提醒 {data['flagged']} 张")


def _trial_readback_delta(row):
    previous = _setup_value(row["recorded"], row["unit"])
    current = _setup_value(row["current"], row["current_unit"])
    return f"{_setup_feature(row['field'])}  试采 {previous} → 当前 {current}"


def _trial_readback_lines(checks, camera_labels):
    bases = {"device_features": "相机空闲回读", "preview_readback": "预览回读",
             "unavailable": "相机回读不可用", "not_probed": "尚未读取相机"}
    for identity, check in checks.items():
        pending = [row["field"] for row in check["fields"] if row["status"] == "pending"]
        changed = [row for row in check["fields"] if row["status"] == "changed"]
        unknown = sum(row["status"] == "unverified" for row in check["fields"])
        camera = camera_labels.get(identity, identity)
        basis = bases.get(check["basis"], "相机回读")
        lines = f"  {camera} · {basis} · 一致 {check['matched_fields']} 项 · 差异 {len(changed)} 项 · 待核验 {unknown} 项"
        yield lines
        if check["identity"] in ("changed", "unverified"):
            yield "    读到不同的设备标识" if check["identity"] == "changed" else "    相机身份待核对"
        for row in changed:
            yield "    " + _trial_readback_delta(row)
        if pending:
            yield "    计划应用后再核对  " + "、".join(_setup_feature(name) for name in pending)


def format_trial_comparison(report, *, camera_labels=None):
    labels = camera_labels or {}
    lines = ["试采与正式设置", "按阶段比较两次启动相机后读到的参数。"]
    for row in report["stages"]:
        stamp = datetime.fromtimestamp(row["created_utc_ns"] // 1_000_000_000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines.extend(("", f"阶段 {row['phase']} · 试采 {stamp}"))
        if check := row["setup_check"]:
            changed = [field for field in check["fields"] if field["status"] == "changed"]
            unknown = [field for field in check["fields"] if field["status"] == "unverified"]
            lines.append(f"实际回读 · 一致 {check['matched_fields']} 项 · 差异 {len(changed)} 项 · 待核验 {len(unknown)} 项")
            for field in changed:
                previous = _setup_value(field["recorded"], field["unit"])
                current = _setup_value(field["current"], field["current_unit"])
                camera = labels.get(field["camera"], field["camera"])
                lines.append(f"  {camera} · {_setup_feature(field['field'])}  试采 {previous} → 正式 {current}")
            for field in unknown:
                camera = labels.get(field["camera"], field["camera"])
                lines.append(f"  {camera} · {_setup_feature(field['field'])}  缺少可比回读")
            for key, title in (("missing_cameras", "正式阶段缺少相机"), ("added_cameras", "正式阶段新增相机"),
                               ("unverified_cameras", "试采阶段缺少回读")):
                if check[key]:
                    lines.append(f"  {title}  " + "、".join(labels.get(camera, camera) for camera in check[key]))
            for camera, state in check["identities"].items():
                if state in ("changed", "unverified"):
                    note = "检测到不同相机" if state == "changed" else "相机身份待核对"
                    lines.append(f"  {labels.get(camera, camera)} · {note}")
        else:
            lines.append("正式阶段尚未记录相机回读。")
        condition = "TIFF 校验通过" if row["observed"] else "试采结果需复查"
        lag = " · 写入积压需复查" if row["sustained_write_lag"] else ""
        lines.append(f"开拍前记录 · 试采保存 {row['committed_images']} 张 TIFF · {condition}{lag}")
        if row["image_warnings_confirmed"]:
            lines.append("试采原图有提醒 · 操作者已确认继续。")
        if row.get("readback_warnings_confirmed"):
            lines.append("采前相机回读有差异 · 操作者已确认继续。")
        lines.extend(_trial_readback_lines(row.get("preflight_camera_readbacks", {}), labels))
        lines.extend(_trial_region_lines(row.get("region_checks", {}), labels))
        lines.extend(_trial_image_lines(row["image_checks"], labels))
        lines.extend((f"  试采编号  {row['experiment_id']}", f"  试采原目录  {row['source']['directory']}"))
    return "\n".join(lines)


def format_readiness(report, *, phase_offset=0, camera_labels=None):
    camera_labels = camera_labels or {}
    image_alerts = [issue for issue in report["issues"] if issue["code"] == "TRIAL_IMAGES_NEED_REVIEW"]
    readback_alerts = [issue for issue in report["issues"] if issue["code"] == "TRIAL_CAMERA_READBACK_CHANGED"]
    if not report["ready"]:
        headline = "请先处理以下阻塞项"
    elif any(issue["code"] == "CONTINUATION_SETUP_CHANGED" for issue in report["issues"]):
        headline = "采前检查完成 · 续采设置需核对"
    elif readback_alerts:
        headline = "采前检查完成 · 相机回读与试采有差异"
    elif image_alerts:
        headline = "采前检查完成 · 试采原图需复核"
    elif any(issue["code"] == "TRIAL_NEEDS_REVIEW" for issue in report["issues"]):
        headline = "采前检查完成 · 试采结果需复查"
    else:
        headline = "采前条件检查通过"
    lines = [headline]
    blockers = [issue for issue in report["issues"] if issue["severity"] == "block"]
    if blockers:
        lines.extend(("", *(readiness_issue_text(issue, phase_offset, camera_labels) for issue in blockers), ""))
    if suggestion := report.get("output_suggestion"):
        lines.extend(("可使用同一父目录下的新路径", suggestion, ""))
    connected = [camera_labels.get(camera, camera) for camera, check in report.get("camera_connections", {}).items()
                 if check["state"] == "available"]
    if connected:
        lines.extend(("", "已打开相机并释放连接", *connected))
    if checks := report.get("preview_quality"):
        lines.extend(("", "预览画质抽查"))
        for camera, summary in checks.items():
            lines.append(camera_labels.get(camera, camera))
            lines.extend("  " + line for line in preview_quality_detail(summary).splitlines())
    if regions := report.get("quality_regions"):
        lines.extend(("", "采集画质抽查范围"))
        for camera, region in regions.items():
            if region.get("fallback"):
                state = "画幅不匹配时使用全图"
            else:
                state = "预览匹配" if region["verified_preview"] else "采集时核对画幅"
            lines.append(f"{camera_labels.get(camera, camera)} · {quality_region_summary(region)} · {state}")
    if image_alerts:
        lines.extend(("", *(f"阶段 {issue['phase'] + phase_offset} · {issue['flagged_images']} 张原图有提醒"
                             for issue in image_alerts), "可打开试采原图核对亮度、纹理和文件读取。"))
    if readback_alerts:
        assessments = {row["phase"]: row for row in report["trial_assessments"] if row.get("latest_match")}
        for issue in readback_alerts:
            lines.append("")
            checks = assessments[issue["phase"]]["camera_readbacks"]
            for camera in issue["cameras"]:
                check = checks[camera]
                lines.append(f"阶段 {issue['phase'] + phase_offset} · {camera_labels.get(camera, camera)}")
                if check["identity"] == "changed":
                    lines.append("  读到不同的设备标识")
                changed = [row for row in check["fields"] if row["status"] == "changed"]
                lines.extend("  " + _trial_readback_delta(row) for row in changed[:2])
                if len(changed) > 2:
                    lines.append(f"  另有 {len(changed) - 2} 项差异，可在下方试采对照查看。")
    if settings := report.get("trial_setting_suggestions"):
        units = {"exposure_us": "µs", "gain_db": "dB"}
        lines.extend(("", "可补入计划的试采参数"))
        for camera, fields in settings.items():
            values = " · ".join(f"{_setup_feature(key)} {_setup_value(value, units[key])}" for key, value in fields.items())
            lines.append(f"  {camera_labels.get(camera, camera)}  {values}")
    if previous := report.get("continuation"):
        lines.extend(("", *_continuation_plan_lines(previous, camera_labels), ""))
    if suggestions := report.get("phase_suggestions"):
        continuous = "、".join(str(item["phase"] + phase_offset) for item in suggestions if item["pairing"] == "independent_receipts")
        host = "、".join(str(item["phase"] + phase_offset) for item in suggestions if item["pairing"] == "host_receipt")
        if continuous:
            lines.extend((f"阶段 {continuous} 可改为连续采集。", "每台相机保存各自收到的第一张图像。"))
            if len(report["cameras"]) > 1:
                lines.append("各相机独立记录。需要同一触发下的配对时，请使用支持该触发方式的相机。")
        if host:
            lines.extend((f"阶段 {host} 可按主机到达选帧。", "按原计划发出请求。采用主机选帧的相机保存请求后收到的第一张。",
                          "台账按请求关联图像。曝光同步需独立核实。"))
        lines.append("")
    if queue := report.get("write_queue"):
        label = "自动" if queue["mode"] == "auto" else "固定"
        lines.extend(("", f"待写队列  {label} · {queue['capacity']} 批上限"))
        if queue["suggest_auto"]:
            lines.append("固定队列可能先达到上限。改用自动队列可让短时积压使用更多字节缓冲。")
        lines.append("")
    trials = report.get("trial_assessments", ())
    if trials:
        labels = {"phase": "阶段参数", "volume": "输出卷", "camera_settings": "相机设置",
                  "cti_files": "相机驱动路径", "storage": "存储设置", "preview_geometry": "当前预览尺寸或格式"}
        camera_units = {"exposure_us": "µs", "gain_db": "dB", "fps": "Hz", "width": "px", "height": "px"}
        lines.extend(("", "试采对照"))
        for trial in trials:
            title = f"阶段 {trial['phase'] + phase_offset}"
            if trial["status"] == "changed":
                status = "设置已变化 · " + "、".join(labels[key] for key in trial["changes"])
            elif trial["status"] == "unbound":
                status = "缺少可核对的试采设置或输出卷信息"
            elif not trial.get("latest_match"):
                status = "已有更新的试采"
            elif trial.get("sustained_write_lag"):
                status = "计划设置匹配 · TIFF 校验通过 · 写入积压需复查"
            else:
                status = "计划设置匹配 · TIFF 校验通过" if trial["observed"] else "计划设置匹配 · 试采结果需复查"
            duration = trial["duration_limit_s"]
            lines.append(f"{title}  {status}" + (f" · 试采时限 {duration:g} 秒" if duration is not None else ""))
            for camera, fields in trial.get("camera_setting_changes", {}).items():
                lines.append(f"  {camera_labels.get(camera, camera)} 试采计划 → 当前计划")
                for item in fields:
                    field = item["field"]
                    old, new = ("由设备决定" if item[key] is None else _setup_value(item[key], camera_units.get(field))
                                for key in ("trial", "plan"))
                    lines.append(f"    {_setup_feature(field)}  {old} → {new}")
            if trial.get("latest_match") and trial["status"] == "matched":
                lines.extend(f"  {text}" for item in trial["findings"] if (text := _trial_advice(item, camera_labels)))
            if checks := trial.get("image_checks"):
                scope = "全图" if trial.get("latest_match") and trial["status"] == "matched" else "历史试采"
                lines.extend((f"  主图像检查 · {scope}", *_trial_image_lines(checks, camera_labels)))
                lines.extend(_trial_region_lines(trial.get("region_checks", {}), camera_labels))
            if readbacks := trial.get("camera_readbacks"):
                lines.extend(_trial_readback_lines(readbacks, camera_labels))
        lines.append("")
    for suggestion in report.get("rate_suggestions", ()):
        lines.extend(format_rate_suggestion(suggestion, camera_labels))
        lines.append("")
    lines.extend((
        f"输出目录  {report['output']}",
        f"可用空间 {format_bytes(report['disk_free_bytes'])}　保留 {format_bytes(report['reserve_bytes'])}",
        f"计划原始像素量 {format_bytes(report['raw_payload_bytes'])}　采集缓冲 {format_bytes(report['buffer_limit_bytes'])}",
        "数据量按未压缩像素估算。TIFF 和台账另占空间，实际帧率和写入能力可通过试采确认。",
        "",
    ))
    sources = {"preview": "预览实测", "configured": "按设置估算", "simulator": "模拟参数", "unavailable": "待预览"}
    for camera, row in report["cameras"].items():
        geometry = f"{row['width_px']} × {row['height_px']} px  {row['pixel_format']}" if row["frame_bytes"] else "图像参数未知"
        lines.append(f"{camera_labels.get(camera, camera)}  {geometry}  {sources[row['basis']]}")
    lines.append("")
    for row in report["phases"]:
        rate = row["scheduled_raw_bytes_per_s"]
        stream_rate = row["configured_stream_raw_bytes_per_s"]
        if row["frame_limit_per_camera"] is not None:
            limit = f"最多 {row['frame_limit_per_camera']} 张/相机"
        elif row["frames_at_configured_rate"] is not None:
            limit = f"按设定帧率估算约 {row['frames_at_configured_rate']} 张（合计）"
        else:
            limit = "张数未知"
        cadence = f"  ·  计划 {format_bytes(rate)}/s" if rate is not None else ""
        if stream_rate is not None:
            cadence = f"  ·  约 {format_bytes(stream_rate)}/s"
        lines.append(f"阶段 {row['phase'] + phase_offset}  {MODE_LABELS[row['mode']]}  {limit}{cadence}")
        if instruction := row.get("start_confirmation"):
            lines.append(f"  确认后开始 · {instruction}")
        if inputs := row.get("hardware_inputs"):
            if len(set(inputs.values())) == 1:
                lines.append(f"  物理触发输入  全部相机 {next(iter(inputs.values()))}")
            else:
                lines.extend(f"  {camera_labels.get(camera, camera)}  ·  物理触发输入 {source}" for camera, source in inputs.items())
        if stall := row.get("write_stall_buffer"):
            limit = "批次上限" if stall["first_limit"] == "queue" else "字节上限"
            lines.append(f"  从空缓冲估算，写入暂停时可暂存约 {stall['seconds']:.2f} 秒的收图量 · 先达到{limit}")
        if row.get("trial_geometry_cameras"):
            lines.append("  使用已校验的试采尺寸  " + "、".join(camera_labels.get(key, key) for key in row["trial_geometry_cameras"])
                         + f"  ·  像素量 {format_bytes(row['raw_payload_bytes'])}")
    lines.append("")
    lines.extend(readiness_issue_text(issue, phase_offset, camera_labels) for issue in report["issues"]
                 if issue["severity"] != "block" and issue["code"] not in
                 {"PREVIEW_IMAGE_QUALITY", "TRIAL_IMAGES_NEED_REVIEW", "TRIAL_CAMERA_READBACK_CHANGED"})
    if trials:
        lines.extend(("", *(f"阶段 {trial['phase'] + phase_offset} 试采目录  {trial['output']}" for trial in trials)))
    return "\n".join(lines)


def format_rate_suggestion(suggestion, camera_labels):
    lines = [f"阶段 {suggestion['phase']} 的建议试采频率"]
    if rate := suggestion["request_rate"]:
        lines.append(f"定时请求  {rate['current']:g} → {rate['proposed']:g} Hz")
    else:
        for key, rate in suggestion["camera_rates"].items():
            lines.append(f"{camera_labels.get(key, key)}  {rate['current']:g} → {rate['proposed']:g} Hz")
        lines.append("采用后，相机帧率将在整个计划中生效。")
    traffic = format_bytes(suggestion["estimated_raw_bytes_per_s"])
    lines.append(f"预计每秒接收 {traffic} 原始像素。先试采至多 {suggestion['trial_seconds']:g} 秒，再决定是否采用。")
    return lines


RATE_TRIAL_LABELS = {
    "verified": "建议频率已通过复测，可重新核验并采用。",
    "current": "当前计划与本次复测的设置一致。",
    "original_plan": "请恢复生成建议时的采集计划，再打开本报告核对。",
    "source_unavailable": "找不到生成建议的原试采。请用顶部的“打开实验”打开原试采，再回到本报告。",
    "source_integrity": "原试采的图像或台账需要复查。请重新核对原试采。",
    "source_measurement": "原试采的写入测量无法支持当前建议。请重新检查计划。",
    "file_integrity": "本次试采的图像或台账待复查。请查看下方报告。",
    "volume": "当前输出卷与两次试采的保存卷不匹配。请在正式采集使用的输出卷上重新试采。",
    "geometry": "两次试采的图像尺寸或像素格式不同。请核对相机设置。",
    "cadence": "实收帧率与建议值相差超过 10%，或有效图像不足。请检查相机设置和试采时长。",
    "write_measurement": "采集末段的写入样本不足。请核对阶段结束条件。",
    "write_lag": "实测写入仍落后。可换用持续写入速度更高的输出卷后重新试采。",
    "buffer_pressure": "试采缓冲峰值接近上限。请核对报告中的队列和字节缓冲后重新试采。",
    "setup": "试采设置与建议不一致。请重新核对当前计划。",
}


def _confirmation_lines(events, *, trial_phase=None):
    actions = {
        "phase_confirmation_waiting": "进入等待",
        "phase_confirmed": "已确认",
        "phase_confirmation_cancelled": "启动已取消",
        "phase_started": "相机已启动",
    }
    timestamps = {
        "phase_confirmation_waiting": "waiting_since_utc_ns",
        "phase_confirmed": "requested_utc_ns",
        "phase_started": "camera_start_returned_utc_ns",
    }
    lines = ["阶段开始确认记录（主机 UTC）"]
    for event in events:
        kind = event["kind"]
        seconds, nanoseconds = divmod(event.get(timestamps.get(kind), event["utc_ns"]), 1_000_000_000)
        stamp = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        detail = event.get("source") or (event.get("instruction") if kind == "phase_confirmation_waiting" else "")
        number = event["phase"] if trial_phase is None else trial_phase
        lines.append(f"阶段 {number} · {actions[kind]} · {stamp}.{nanoseconds:09d}" + (f" · {detail}" if detail else ""))
    return lines


def format_trial(report, *, camera_labels=None, rate_check=None):
    camera_labels = camera_labels or {}
    trial, verification = report["trial"], report["verification"]
    mode = trial["mode"]
    duration = f" · 时限 {trial['duration_limit_s']:g} 秒" if trial["duration_limit_s"] is not None else ""
    fps = report["phase"].get("fps")
    schedule = f" · 定时请求 {fps:g} Hz" if fps is not None else ""
    durability = f"存储模式  {DURABILITY_LABELS[report['durability']]}"
    headline = "试采已完成，已保存图像通过校验" if report["observed"] else "试采结果需要复查"
    if report.get("sustained_write_lag"):
        headline = "图像通过校验，写入积压需复查"
    if rate_check is not None:
        headline = RATE_TRIAL_LABELS[rate_check] + "\n" + headline
    lines = [
        headline,
        f"阶段 {trial['phase_number']} · {MODE_LABELS[mode]}{duration}{schedule}",
        durability,
        *buffer_peak_lines(report["write_buffer"]),
        "",
    ]
    if confirmations := verification.get("phase_confirmation_events"):
        lines.extend((*_confirmation_lines(confirmations, trial_phase=trial["phase_number"]), ""))
    for source in report.get("csv_triggers", ()):
        lines.extend((csv_trigger_caption(source), csv_trigger_rule_text(source), source["path"]))
        if crossing := source.get("crossing"):
            lines.extend(csv_crossing_lines(crossing))
        if error := source.get("error"):
            lines.append(error["message"])
        lines.append("")
    findings = (item for item in report["findings"]
                if rate_check != "write_lag" or item["code"] != "TRIAL_SUSTAINED_WRITE_LAG")
    remarks = [text for item in findings if (text := _trial_advice(item, camera_labels))]
    if remarks:
        lines.extend(("试采建议", *remarks, ""))
    if regions := report.get("quality_regions"):
        lines.append("试采期间的画质抽查")
        watch = report.get("image_quality_watch")
        for camera, region in regions.items():
            sample = (watch or {}).get("cameras", {}).get(camera)
            title = camera_labels.get(camera, camera)
            if sample is not None:
                lines.append(title)
                lines.extend("  " + line for line in image_quality_sample_help(sample).splitlines()[:3])
            else:
                state = "本次未收到抽查样本" if watch is not None else "本次无画质抽查记录"
                lines.append(f"{title} · {quality_region_summary(region)} · {state}")
        lines.append("")
    checks = {identity: row["image_check"] for identity, row in report["cameras"].items() if "image_check" in row}
    if checks:
        regions = {camera: check["region_check"] for camera, check in checks.items() if "region_check" in check}
        if regions:
            if not report["observed"] or not verification.get("file_hashes_checked"):
                regions = {camera: {**check, "fallback": check["fallback"] or "integrity"} for camera, check in regions.items()}
            lines.extend(("从试采原图复核计划检查框", *_trial_region_lines(regions, camera_labels), ""))
        lines.append("试采主图像检查 · 全图")
        lines.extend(_trial_image_lines(checks, camera_labels))
        lines.append("")
    if timing := report.get("write_timing"):
        if windows := timing["acquisition_windows"]:
            lines.append("采集末段写入实测 · 按原始像素量计")
            for window in windows:
                incoming = format_bytes(window["accepted_raw_bytes_per_s"])
                written = format_bytes(window["committed_raw_bytes_per_s"])
                lines.append(f"  {window['duration_s']:.1f} 秒 · 进入队列 {incoming}/s · 写入并记账 {written}/s")
        elif mode in ("continuous", "timed"):
            lines.append("采集中写入 · 有效测量时间不足，无法判断是否持续积压")
        if saved := timing["post_capture_committed_frames"]:
            lines.append(f"停采后补写 {saved} 张 · 停采至写入结束 {timing['post_capture_elapsed_s']:.2f} 秒")
        lines.append("")
    for identity, row in report["cameras"].items():
        fps = row["host_receive_fps"]
        host_sampled = row.get("capture_method") == "host_next_receipt"
        rate_label = "选中帧接收速率" if host_sampled else "接收帧率"
        cadence = f" · {rate_label} {fps:.2f} Hz" if fps is not None else ""
        count = row["committed_frames"]
        source = {"simulated": "软件模拟", "vendor_emulator": "厂商模拟器"}.get(row["device_support"])
        label = f" · {source}" if source else ""
        lines.append(f"{camera_labels.get(identity, identity)}{label}  ·  保存 {count} 张{cadence}")
        if host_sampled:
            unselected = row.get("host_unselected_frames")
            lines.append("  按主机到达选帧" + (f" · 另有 {unselected} 张未被请求选中" if unselected else ""))
            if delay := row.get("host_selection_delay_ms"):
                lines.append(f"  请求到收图  中位 {delay['median']:.1f} ms · 最大 {delay['maximum']:.1f} ms")
        for fmt in row["formats"]:
            height, width = fmt["shape"][:2]
            lines.append(f"  {width} × {height} px · {fmt['pixel_format']} · 有效位数 {fmt['significant_bits']}")
        readbacks = [
            f"{FEATURE_HELP[key][0]} {feature['value']}{' ' + feature['unit'] if feature.get('unit') else ''}"
            for key, feature in row["phase_feature_readback"].items()
            if key in ("exposure_us", "gain_db", "fps", "trigger_source", "trigger_activation") and feature.get("value") is not None
        ]
        if readbacks:
            lines.append("  相机回读  " + " · ".join(readbacks))
        if count and row["frames_with_native_counter"] < count:
            lines.append("  部分或全部图像缺少相机计数器，接收端只能核对已观察到的帧。")
        if mode == "hardware" and count:
            if recorded := row.get("hardware_counters"):
                for feature, observed in recorded.items():
                    first, last = observed["first"], observed["last"]
                    lines.append(f"  {feature} · {observed['frames']}/{count} 张有读数 · "
                                 f"序号 {first['sequence']} = {first['value']} · 序号 {last['sequence']} = {last['value']}")
                    if any(observed[key] for key in ("duplicates", "decreases", "skipped_values")):
                        lines.append(f"    相邻读数 · 重复 {observed['duplicates']} 次 · 回退 {observed['decreases']} 次 · "
                                     f"跳过 {observed['skipped_values']} 个计数值")
            else:
                lines.append("  已保存帧中没有读到可用于脉冲编号的帧附带计数器。请检查相机的 Chunk 和计数器设置。")
    rate = report["disk_bytes_per_s"]
    if rate is not None:
        lines.extend(("", f"全程平均写入 {format_bytes(rate)}/s · 含相机连接与停采补写"))
    lines.append(f"台账状态 {STATE_LABELS.get(verification['state'].upper(), verification['state'])}")
    lines.append(runtime_integrity_text(verification))
    if event_phase := report.get("event_phase"):
        recorded = event_phase["completed_events"]
        partial = event_phase["partial_events"]
        planned = event_phase["event_count"]
        busy = "同时收图达到上限时收到" if event_phase.get("overlap") else "正在收图时收到"
        lines.extend(("", f"完成 {recorded}/{planned} 次事件" if planned else f"试采时限内完成 {recorded} 次事件",
                      f"部分收图 {partial} 次 · 缓存未满时收到 {event_phase['early_signals']} 次 · "
                      f"{busy} {event_phase['busy_signals']} 次", "在采集时间线中可按事件编号查看前后原图。"))
    elif window := report.get("event_window"):
        pre = window["pre_retained_rounds"]
        if pre is not None:
            lines.extend((
                "",
                f"事件前保留 {pre}/{window['pre_required_rounds']} 轮 · 边界 {window['boundary_rounds']} 轮",
                f"事件后收到 {window['post_received_rounds']}/{window['post_required_rounds']} 轮",
                "一轮包含每台相机的一张图。落盘张数见各相机记录。",
            ))
        elif window["signal_id"]:
            lines.append("触发请求已入队，阶段结束前未执行。")
        else:
            lines.append("阶段结束前没有触发记录。")
    if error := report["error"]:
        lines.extend(("", f"{error['code']} · {error['message']}"))
    if mode == "hardware":
        lines.extend(("", "报告的帧率按主机收到图像的时间计算。相邻计数的跳阶需对照 DAQ 脉冲核实。",
                      "请用正式实验的已知共享脉冲确认各台相机的计数器锚点。"))
    lines.extend((
        "", "写入速率和缓冲峰值对应这次试采的时长、图像设置与保存策略。",
        f"试采目录  {report['output']}",
        f"计划输出  {trial['formal_output']}",
    ))
    return "\n".join(lines)


def buffer_peak_lines(buffer):
    lines = []
    if buffer.get("queue_high_water") is not None and buffer.get("queue_capacity"):
        lines.append(f"待写队列峰值 {buffer['queue_high_water']} / {buffer['queue_capacity']} 批")
    if buffer.get("raw_buffer_high_water_bytes") is not None:
        lines.append(f"字节缓冲峰值 {format_bytes(buffer['raw_buffer_high_water_bytes'])}"
                     f" / {format_bytes(buffer.get('raw_buffer_limit_bytes'))}")
    return lines


_CAMERA_INCIDENT_LABELS = {
    "app_unsaved": ("未登记图像", "帧"),
    "native_counter_gaps": ("相机计数缺号", "个"),
    "native_counter_duplicates": ("相机计数重复", "次"),
    "hardware_identity_missing": ("缺少触发计数", "帧"),
    "reported_driver_skips": ("驱动报告跳过", "帧"),
    "transport_errors": ("传输错误", "次"),
}


def capture_incident_lines(camera_losses, scheduled_misses, short_pre_windows, camera_name=str):
    lines = []
    if scheduled_misses:
        lines.append(f"定时采集请求漏发 {scheduled_misses} 次")
    if short_pre_windows:
        lines.append(f"事件前缓存窗口不足 {short_pre_windows} 次")
    for camera, counts in camera_losses.items():
        details = [f"{label} {counts[key]} {unit}" for key, (label, unit) in _CAMERA_INCIDENT_LABELS.items()
                   if counts.get(key)]
        if details:
            lines.append(f"{camera_name(camera)} · " + " · ".join(details))
    return lines


def runtime_integrity_text(report):
    lines = capture_incident_lines(
        report["reported_camera_losses"], report["scheduled_misses"], report["short_pre_windows"],
    )
    if not report.get("runtime_summary_available"):
        lines.append("台账缺少完整的运行期统计")
    return "采集记录\n" + "\n".join(lines or ["未记录漏发、传输或写入异常。"])


def integrity_issues(report):
    journal_issues = report.get("recovery", {}).get("issues", [])
    explained_paths = {issue.get("source_path") or issue.get("image_path") or issue.get("path") for issue in journal_issues}
    return [issue for issue in report.get("issues", ())
            if issue["code"] != "UNCOMMITTED_FILE" or issue["path"] not in explained_paths] + journal_issues


def format_integrity(report):
    state = report["state"].upper()
    if state == "COMPLETED" and not report["complete"]:
        state = "CHECK DATA"
    transfer = report.get("transfer_copy")
    lines = [
        f"实验状态  {STATE_LABELS.get(state, state)}",
        f"台账中已保存 {report['committed_frames']} 帧 · {report.get('committed_images', report['committed_frames'])} 张 TIFF",
    ]
    if transfer and transfer["ok"]:
        lines.append(f"已按交接清单核验 {transfer['files']} 个文件的 SHA-256，其中包含 {transfer['attachment_files']} 个附件。")
    else:
        lines.append("已按台账校验 TIFF 和归档标定的 SHA-256。" if report["file_hashes_checked"] else "已检查文件大小。")
    if transfer and not transfer["ok"]:
        lines.extend(("", "交接清单校验未通过", transfer["detail"]))
    if trial := report.get("trial_comparison"):
        lines.append(f"已保存 {len(trial['stages'])} 个阶段的试采对照，可在复查页查看。")
    if checks := report.get("continuation_checks"):
        lines.extend(("", f"续采设置回读 · 原实验 {checks[0]['source_experiment_id']}"))
        for check in checks:
            changed = [row for row in check["fields"] if row["status"] == "changed"]
            unknown = [row for row in check["fields"] if row["status"] == "unverified"]
            source = check.get("source_phase", check["phase"])
            stage = f"阶段 {check['phase']}"
            if source is not None and source != check["phase"]:
                stage += f"（原阶段 {source}）"
            lines.append(f"{stage} · 参数一致 {check['matched_fields']} 项"
                         f" · 差异 {len(changed)} 项 · 待核验 {len(unknown)} 项")
            for key, prompt in (("missing_cameras", "未选原相机"), ("added_cameras", "新增相机"),
                                ("unverified_cameras", "原阶段缺少回读")):
                if check[key]:
                    lines.append(f"  {prompt}  " + "、".join(check[key]))
            for camera, identity in check.get("identities", {}).items():
                if identity in ("changed", "unverified"):
                    note = "本次读到不同相机" if identity == "changed" else "相机身份待核对"
                    lines.append(f"  {camera} · {note}")
            if check["phase_changes"]:
                lines.append("  计划有变化  " + _phase_change_labels(check["phase_changes"]))
            for row in changed:
                previous = _setup_value(row["recorded"], row["unit"])
                current = _setup_value(row["current"], row["current_unit"])
                lines.append(f"  {row['camera']} · {_setup_feature(row['field'])}  原记录 {previous} → 本次 {current}")
            for row in unknown:
                lines.append(f"  {row['camera']} · {_setup_feature(row['field'])}  待核验")
    if recovery := report.get("recovery"):
        lines.extend(("", f"本次恢复 {len(recovery['recovered'])} 帧 · 已在台账中 {recovery['already_committed']} 帧"))
        if recovery["journal"] in ("not_recorded", "missing"):
            lines.append("没有找到帧写入记录。遗留图像可按下列文件路径复查。")
        for row in recovery["recovered"]:
            lines.append(f"已恢复  {row['path']}")

    lines.extend(("", runtime_integrity_text(report)))
    lines.extend(buffer_peak_lines(report.get("write_buffer") or {}))
    if alerts := report.get("write_alerts"):
        lines.extend(("", "采集中写入提醒"))
        for alert in alerts:
            caption = WRITE_PRESSURE_LABELS[alert["kind"]]
            lines.append(f"阶段 {alert['phase']} · 开始后 {alert['phase_elapsed_s']:.1f} 秒 · {caption}"
                         f" · 字节缓冲 {format_bytes(alert['buffered_raw_bytes'])} · 待写 {alert['queued_batches']} 批")
            for window in alert["windows"]:
                lines.append(f"  {window['duration_s']:.1f} 秒 · 入队 {format_bytes(window['accepted_raw_bytes_per_s'])}/s"
                             f" · 已登记 {format_bytes(window['committed_raw_bytes_per_s'])}/s")
    if (error := report.get("capture_error")) and error["code"] in {"WRITE_QUEUE_FULL", "RAW_BUFFER_FULL"}:
        lines.append(CAPTURE_ERROR_LABELS[error["code"]])
    if confirmations := report.get("phase_confirmation_events"):
        lines.extend(("", *_confirmation_lines(confirmations)))
    if wait_checks := report.get("hardware_wait_observations"):
        lines.extend(("", "硬件触发首帧前的等待输出回读"))
        for check in wait_checks:
            for camera, row in check["cameras"].items():
                state = ("读到可接收触发" if row.get("first_asserted_poll_monotonic_ns") is not None else
                         WAIT_OUTPUT_REASONS.get(row["reason"], "未读到有效的等待输出"))
                received = check["received_by_camera"].get(camera, 0)
                line = f" · {row['line']}" if row["line"] else ""
                lines.append(f"阶段 {check['phase']} · {camera}{line} · {state} · 收图 {received} 张")
    if pulses := report.get("hardware_groups"):
        lines.extend(("", "采集结束时的脉冲监看快照",
                      f"阶段 {pulses['phase']} · {pulses['source_id']} · {pulses['counter_feature']}"))
        total = len(pulses["cameras"])
        for group in pulses["groups"]:
            if group["number"] != pulses["first_number"] and not (group["advanced"] or group["duplicates"]):
                continue
            lines.append(f"脉冲 {group['number']} · 收图 {group['received_cameras']}/{total} 台"
                         f" · 登记 {group['saved_cameras']}/{total} 台")
            if group["advanced"]:
                lines.append("  已收到更大编号 " + "、".join(group["advanced"]))
            if group["duplicates"]:
                lines.append("  编号重复 " + "、".join(group["duplicates"]))
        for camera, row in pulses["cameras"].items():
            if row["unidentified"] or row["backtracks"]:
                lines.append(f"{camera} · 未关联编号 {row['unidentified']} 帧 · 编号回退 {row['backtracks']} 次")
    issues = integrity_issues(report)
    lines.extend(("", f"文件与台账问题 {len(issues)} 项"))
    for issue in issues:
        code = issue["code"]
        label = INTEGRITY_LABELS.get(code, "一组触发图像关联不完整" if code.startswith("TRIGGER_PAIRING_") else code)
        detail = issue.get("source_path") or issue.get("image_path") or issue.get("path") or issue.get("event_id") or ""
        if issue.get("line"):
            detail = "  ·  ".join(filter(None, (detail, f"写入记录第 {issue['line']} 行")))
        if code == "SHARED_HARDWARE_COUNTER_GAP":
            detail = f"阶段 {issue['phase']} · 事件号 {issue['first_event_number']}–{issue['last_event_number']}"
        lines.append(f"{label}  {detail}".rstrip())
    return "\n".join(lines)
