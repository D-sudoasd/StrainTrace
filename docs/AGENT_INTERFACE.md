# Agent 接口与结果协议

## 适用范围

此说明定义无 GUI 操作合同，供 Agent、脚本和批处理服务调用。CLI 输出 JSON；进度可通过 JSON Lines 事件读取。字段计算使用已有 ezdic_core；项目 API 流式读取图像并逐帧落盘，调用方不必模拟 GUI 或把整个序列留在内存中。

当前命令的通用形式：

    python -m ezdic_cli agent <command> [options]
    python path/to/ezdic_cli.py agent <command> [options]
    straintrace agent <command> [options]

Python 源码 checkout 安装后可从 straintrace_agent 导入 api。CLI 是跨版本和多进程批处理更稳的边界；直接 API 适合本地 Python pipeline。

## 命令表

| 命令 | 功能 | 输出/注意 |
| --- | --- | --- |
| init | 建立单样品项目、索引原始图像、记录实验元数据 | 需 --project 和 --images；需 frame-interval-s 或有时间列的 frame map |
| inspect | 读取索引和项目状态 | 不运行 DIC |
| plan | 选择可审查参考帧、ROI 候选和有限求解参数 | 写入 plans/plan_v*.json；plan 状态可为 ready 或 review_required |
| run | 根据当前计划计算并续算 | --project 必需；可加 --progress-json、--field-csv、--no-plots |
| analyze | 依次执行 plan 与 run | 同 run 选项；最少操作的一次性分析入口 |
| status | 查询锁、当前 plan/run 和最后进度 | 不写入分析产物 |
| verify | 校验输入、指纹、计划、检查点与产物 | valid 是完整性字段；scientific_ok 是独立数值质量门 |
| plot | 从保存场重新生成稀疏热图 | 不重新运行 DIC |
| export | 根据保存结果重建表格/报告 | --field-csv 可选择逐帧场 CSV |
| extract | 从当前 run 提取 ROI 和/或路径数据 | 路径必须给 max-distance |
| batch | 多个独立项目并行分析 | 每个 --project 重复一次；max-workers 限制为 1 到 8 |
| cancel | 写入当前项目的取消请求 | DIC 在下一进度回调边界停止 |
| recover-lock | 恢复一个确认死亡的本机锁进程 | 必须提供锁记录 PID；活跃/远程/未知 PID 被拒绝 |

查看各子命令的精确 flags 可运行 agent <command> --help。当前 init 关键参数：

    --project PATH --images DIR
    --frame-interval-s SECONDS | --frame-map-csv FILE
    --reference-native-frame N --roi x,y,width,height --exclude x,y,width,height
    --mask-file FILE --gauges-json FILE --calibration-json FILE
    --step-hint-px N --plot-every N
    --loads-csv FILE --time-column COL --load-time-unit s|ms|min|h
    --time-offset-s SIGNED_SECONDS --max-gap-s SECONDS --clock-confirmed
    --force-column COL --force-unit N|kN --area-mm2 AREA
    --displacement-column COL --displacement-unit mm|um
    --displacement-source crosshead|machine_extensometer|other_machine_channel
    --machine-gauge-length-mm LENGTH

输出目录必须位于原始图像目录之外。init 不覆盖非空项目。初始输入索引含图像哈希，frame map/load/mask 等源文件也受输入指纹约束。

## Agent 调用顺序

建议的受控自主流程：

    init → inspect → plan → 检查 plan JSON → run
    status → verify → extract → plot/export

对低风险批次也可直接调用 analyze。Agent 应先读 inspect 返回的 native_frames、unreadable_frames、time_range_s、physical_calibration_available、load_clock_confirmed 和 analysis_rectangle_status。不可读帧不是可静默丢弃的输入；本机 QC 结果也不是物理标定。若计划为 review_required，读取其 candidate_trials 和 rejection 原因，判断是可执行的候选还是应停止并反馈用户。

run/checkpoint 复用依赖当前输入指纹、plan、代码和环境匹配。配置或输入变更后应建立新计划；不要把旧 run 的检查点并入新配置。cancel 是可协作停止，不会删除已经写入的检查点。异常结束后用 status 核对锁；只有本机且确认 PID 死亡时才能执行 recover-lock --expected-pid。无需常驻轮询高频 status，可在进度事件或定时低频读取。

Agent 可将 JSON Lines 标准输出转成进度面板、审查节点或任务队列事件。事件名包括 project_initialized、plan_complete、run_started、frame_started、dic_progress、frame_complete、frame_reused、frame_failed、run_finished、analysis_complete；命令最终记录为结构化 JSON。不要只看进程退出码：quality-gate 退出码表示现有数值 QC 未通过，须查看失败帧和各指标；完整性校验的退出状态另看 verify.valid。

## CLI 参数语义

init 的 roi 是像素矩形 x,y,width,height。extract 的 ROI 是所选坐标系下闭区间 xmin,xmax,ymin,ymax。frame_interval_s 让帧时间等于 (native_frame - first_native_frame) * interval，所以缺号自动形成时间空档。若帧映射显式含 time_s，必须为所有图提供严格递增时间。

加载机时间先乘显式单位换算为秒，再加有符号 time_offset_s。clock-confirmed 必须来自实验者的同步依据。force 与 displacement 的时间支持独立；summary.csv 中 force_alignment_kind/displacement_alignment_kind 区分 exact_sample、bounded_linear_interpolation、outside_time_support、gap_exceeds_max_gap 等状态。force_bracket_left/right_time_s 和 force_bracket_left/right_source_row、displacement_bracket_left/right_time_s 和 displacement_bracket_left/right_source_row 保留左右样本支持；source_row 是从 1 开始且不含表头的 CSV 数据行号。displacement_reference_alignment_kind 及其 reference bracket 字段说明横梁位移参考值如何对齐。插值值是根据两个样本估出的同步值，不是 DIC 曝光时刻的原生机器实测值。只在相邻样本包围图像时间且区间不超过 max_gap_s 时插值，不外推。force_N / area_mm2 才有应力 MPa；无面积则 stress_MPa 为空。machine_displacement_mm 可独立有效；机器应变需显式 machine-gauge-length-mm。机器通道应变与 DIC 工程/真应变分别命名、分别解读。

标定 JSON 是对象，必须包含 mm_per_px、origin_px、origin_mm、axis_sign；标量假设横纵向像素尺度相同。标尺、origin 或 sign 不可缺省。面内应变和主轴方向保持图像基底；校准位置/位移不会转换这些张量分量。mask file 和排除矩形限定计算支持，但不自动确认物理试样边界。

gauges-json 是数组。每个 gauge 对象有 name、axis (x/y/distance)，每个端点恰好给 a_point_px/b_point_px 或 a_roi_px/b_roi_px 一种定义。ROI 是像素 x,y,width,height；固定参考点身份与初始几何在计划时确定。遇到断裂、散斑脱落或端点无效时，端点必须标为无效，不得换取另一条容易匹配的测量路径。交付中保留参考身份、端点策略和状态。

CLI extract 的 path-json 是 N x 2 坐标数组。max-distance 为必须给出的距离容限。nearest-POI 方法只查询原始网格点，最近点无效时返回 invalid_poi；超过容限则 outside_tolerance。它不会插值或绕过孔洞。ROI 和路径 CSV 应保留每个帧号/时间/POI 状态，空值不编码为 0。

## Python API

API 与 CLI 共享数据与执行逻辑。可从已安装 package 或项目源码路径导入：

    from pathlib import Path
    from straintrace_agent import api

    project = Path("outputs/specimen_01")
    api.init_project(
        project,
        Path("experiment/specimen_01_images"),
        frame_interval_s=0.5,
        loads_csv=Path("experiment/tensile.csv"),
        load_time_column="Time_ms",
        load_time_unit="ms",
        load_time_offset_s=-0.125,
        clock_confirmed=True,
        max_gap_s=0.75,
        load_force_column="Force_kN",
        force_unit="kN",
        area_mm2=24.0,
        displacement_column="Crosshead_mm",
        displacement_unit="mm",
        displacement_source="crosshead",
        calibration={"mm_per_px": 0.01, "origin_px": [0, 0],
                     "origin_mm": [0, 0], "axis_sign": [1, -1]},
    )
    result = api.analyze(project, progress=lambda event: print(event))
    integrity = api.verify(project)

实际测量示例中的标尺与面积必须替换为实验的实测/记录值；示例数字不能作为默认假设。API 方法为 init_project、inspect_project、plan_project、run、analyze、status、verify、plot、export、extract、batch、cancel、recover_lock。progress callback 接收字典事件。batch 接收独立项目列表和 max_workers。

## 字段接口与数组支持

核心 run_2d_dic 结果按字段 mapping 暴露。支持的当前返回结构：

| Key | 含义 | 形状 |
| --- | --- | --- |
| x, y | 原始 POI 图像坐标，px | 展平 POI 数组 |
| u, v | 位移，px | 展平 POI 数组 |
| valid | 相关位移有效掩码 | 展平 POI 布尔数组 |
| strain_valid | 局部应变拟合有效掩码 | 展平 POI 布尔数组 |
| Exx, Eyy, Exy | Green-Lagrange 应变张量分量，无量纲 | 展平 POI 数组 |
| exx, eyy, exy | infinitesimal strain 分量，无量纲 | 展平 POI 数组 |
| X, Y, U, V | POI 网格/网格位移 | ny x nx |
| P | 子区仿射 warp 参数 [u, dudx, dudy, v, dvdx, dvdy] | ny x nx x 6 |

字段 helper 接受原数组或相同元素数后 reshape 的 POI 数组。principal_strains(field, measure) 返回 major/minor、theta_rad/theta_deg、tensor_shear、engineering_shear、measure 与 valid。Green-Lagrange principal strains 由完整有限变形应变张量求特征值；大刚体转动不会产生该测度下的虚假应变。principal angle 在两个主值相等时未定义，返回 NaN。小应变测度使用对称位移梯度；不能把它与 Green-Lagrange 混写。

calibrate_field(field, calibration) 只返回 x_mm、y_mm、u_mm、v_mm 和显式 calibration 记录。字段 mapping 本身不会被修改。sample_path(field, points, components, max_distance, coordinate_system) 返回逐查询点状态；summarize_roi(field, component, roi) 仅聚合有效原始 POI。空间样本 SD 描述 ROI 内散布，不是标准误、置信区间或仪器不确定度。iter_roi_rows/sample_path 的 interpolation 字段保持 none。

plot_field_map(field, component, ...) 支持 scatter 和原网格 quadmesh；无效点掩码保留，默认不平滑、不补点。common_color_limits(fields, component, symmetric=...) 汇总跨帧色阶，调用方应显式把同一 color_limits 传入每帧图。plot_position_time(fields, times_s, path_points, component, max_distance, ...) 逐个消耗字段迭代器，仅在每一帧保存 path 采样点，不要求全序列驻留内存。None field 或无支持时间会留下空档；绘图使用散点而不连线。科学定义、路径状态和 CSV 一行键参见 AGENT_FIELD_ANALYSIS.md。

## 输出文件和两类状态

| 文件 | 稳定用途 |
| --- | --- |
| project.json | 项目输入配置快照、显式来源通道和单位 |
| inventory.json | 原始图像 native_frame、time_s、相对/绝对路径、SHA-256、读取状态 |
| CURRENT.json | current_plan/current_run 相对路径指针 |
| plans/plan_v*.json | 参考帧、候选 ROI、参数比较、质量原因、场支持跨度、输入/代码/配置指纹 |
| checkpoints/frame_native_*.npz | 每帧数值场，安全 NumPy 数组，不 pickle |
| checkpoints/frame_native_*.json | 帧时间、源文件哈希、参数、质量和 checkpoint 关系 |
| summary.csv | 每个原生帧一行：status、质量指标、DIC virtual gauge、force/stress、机器位移和失败原因；force/displacement alignment_kind 与左右支持时间/源数据行索引说明精确采样或有界插值 |
| gauges.csv | 固定参考标距端点的支持计数、长度、方向、状态和测量值 |
| quality.json | 整体与逐帧数值质量状态 |
| manifest.json | 来源文件、代码、环境、checkpoint 和导出产物哈希 |
| report.md | 人可读分析摘要与实验边界 |
| plots/ | 默认稀疏关键帧的位移、应变、相关系数热图 |
| extractions/ | ROI/path 逐帧 CSV、ROI summary JSON、path position-time 图 |
| fields_csv/ | 可选的原始 POI 全场 CSV |

verify.valid 是数据来源与制品完整性；scientific_ok 是 core 已配置的数值质量门。两者互不替代。scientific_ok 不等于真实实验误差、测量不确定度、材料物理解释或科学结论通过。完整运行可以有 failed frame；此时逐帧状态和失败字段必须保留，不得转为零或相邻帧插值。

CLI JSON 退出码约定：0 表示命令执行成功且所请求状态未触发质量失败；2 表示参数/config 错误；3 表示运行错误；4 表示质量门或 verify 不通过。批处理可返回执行完整但 scientific_status=attention_required 的结果，同时退出 4。每次都解析 JSON 内容和机器状态字段，避免只依赖退出码。

## 典型 Agent 自动判定

Agent 读取 inventory 和 plan 后再开跑。若同步未知或物理标定缺失，可继续做像素域 DIC 与独立图像 QC，但禁止生成依赖这些信息的应力/物理位移。若帧失败，读取 failure_code 与 source hash，保留其空测量行并继续其他帧。若 QC 未过，选择检查参考图、纹理、ROI、匹配指标与时间连续性；不要提高阈值上限或为了全批成功把 invalid 改成有效。可以重跑单项目的 plan/run，或导出其他可用阶段；也可以停止并给科研人员可审核的异常清单。

完整性通过只说明当前目录的哈希关系一致。经过真实标定、独立位移/应变基准对照和实验设计复核后，科研人员才能对具体测量给出更强的结论。
