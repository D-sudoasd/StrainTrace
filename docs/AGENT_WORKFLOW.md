# Agent 工作流：从图像目录到可追溯 DIC 结果

本指南面向科研人员和负责执行任务的 AI Agent。分析入口不依赖 GUI；输入、参数、进度、检查点、质量状态与导出文件保存在每个样品项目中。一个项目对应一个样品；多个项目可并行批处理。

## 必要输入与不可推断的物理量

最少输入是一个图像目录和以下二者之一：

- 正的采集间隔，例如 0.5 秒/帧。文件名必须可解析为唯一原生帧号；原生帧号中间缺号会保留时间空隙。
- 显式 CSV 帧映射，列为 filename/native_frame；可选 time_s 用于不等间隔采集时间。多个数字文件名、非末尾帧号或设备时间戳都应显式映射。

物理位移标尺、坐标原点与坐标轴方向必须来自标定记录。缺少标定时保留像素单位；不从试样宽度、图像尺寸或默认值推测毫米。应力要求用户给出有效截面积，并有可同步的力通道；应力定义为 N/mm²（MPa）。缺少截面积时应力留空。

可选试验机 CSV 要明确时间列、时间单位、力和位移的列与单位、允许的最大采样间隙。只有实验记录确认相机与试验机时钟关系后才传入 clock-confirmed。同步关系为：

    aligned_time_s = raw_load_time * unit_factor + time_offset_s

time_offset_s 的符号由用户按设备零点关系决定。与图像时刻对齐的力/位移可能是原生试验机时刻的 exact_sample，也可能是两个相邻、有限样本间的 bounded_linear_interpolation；后一种是估值，并不是图像曝光时刻上的试验机原生实测值。只在样本包围图像时间且间隔不超过 max_gap_s 时插值；时间范围外不外推，过大缺口不补值。summary.csv 的 force_alignment_kind 与 displacement_alignment_kind 表示 exact_sample、bounded_linear_interpolation、outside_time_support、gap_exceeds_max_gap、channel_unavailable 等状态；*_bracket_left/right_time_s 和 *_bracket_left/right_source_row 保留支持样本时间与源数据记录序号（数据行从 1 开始，不含表头）。位移的参考时刻另有 displacement_reference_alignment_kind 和对应 bracket 字段。力与位移分别判断时间支持，不因某个通道缺值而丢弃另一个通道的有效观测。

机器横梁位移、机器标距应变和 DIC 局部应变是不同测量。机器位移不要求机器标距；只有用户显式提供标距，才计算机器工程应变。它不能被解释为 DIC 局部应变。应变为无量纲值，CSV 数字 0.01 表示 1%。

## 输入整理和最少 CLI 工作流

项目输出应与原始图像目录分开，原始文件保持只读。建议结构如下：

    experiment/
      sample_A_images/frame_000001.tif
      sample_A_images/frame_000002.tif
      tensile_machine.csv
      calibration.json
      frame_map_B.csv
    outputs/sample_A/
    outputs/sample_B/

文件名含唯一数字帧号时，可以直接使用采集间隔。帧号缺失不会被压缩成连续行号：如果帧 12 和 14 之间缺少 13，且间隔为 0.5 s，原生帧 14 的图像时间比帧 12 晚 1.0 s。多个数字的文件名不能安全解析，需为每张图提供唯一映射：

    filename,native_frame
    camera_01_run_02_frame_0011.tif,11
    camera_01_run_02_frame_0012.tif,12
    camera_01_run_02_frame_0014.tif,14

若帧映射包含任一 time_s，则所有图片都必须填写严格递增时间；否则还要传入 frame-interval-s。

使用样品的实测参数创建项目并一次性规划、计算：

    python -m ezdic_cli agent init --project outputs/sample_A --images experiment/sample_A_images --frame-interval-s 0.5 --loads-csv experiment/tensile_machine.csv --time-column Time_ms --load-time-unit ms --time-offset-s -0.125 --clock-confirmed --max-gap-s 0.75 --force-column Force_kN --force-unit kN --area-mm2 24.0 --displacement-column Crosshead_mm --displacement-unit mm --displacement-source crosshead --calibration-json experiment/calibration.json --roi 100,80,1200,700 --step-hint-px 12
    python -m ezdic_cli agent analyze --project outputs/sample_A --progress-json

也可分步 inspect → plan → 检查候选参数/ROI → run。run 使用已保存计划；analyze 是 plan 与 run 的便捷组合。roi 参数是像素坐标中的候选矩形 x,y,width,height，不等同于经过实验确认的试样边界。自动计划只在有限候选设置中比较核心求解参数，质量不足时保留 review_required，不放宽已有质量阈值。真实数据仍应核实参考帧、ROI、遮罩、标距和实验条件。

用户明确知道的标定与几何信息可通过 calibration-json、mask-file、roi、exclude 和 gauges-json 提供。其余参数不应伪装成实验事实。检查 plans/plan_v*.json 中的参考帧、候选矩形、试算结果和拒绝原因。

## 力学数据同步和物理解释

clock-confirmed 是由实验者确认的设置，不可按行号相同、文件长度相同或文件名推断。不同采样频率通过各自时间轴同步。缺失时钟偏移、通道超出时间范围、或加载数据大缺口时，依赖该数据的字段应留空并报告状态。

area-mm2 必须对应用户指定的横截面定义。若颈缩、厚度变化或截面随位置改变，单一面积所得应力仍按该面积定义，不代表局部真实应力。没有有效面积、物理像素标尺或真实长度时，不应填入默认值。

init 的 roi 使用 x,y,width,height；extract 的 roi 使用所选坐标系中的闭区间 xmin,xmax,ymin,ymax。两者不可混用。extract 可按 ROI 导出摘要与原始 POI 行，也可沿 path-json 指定路径提取最近原始 POI，并用 max-distance 限制距离。最近 POI 无效时不会跳到邻点；默认不插值。

DIC 热图在原始有效 POI 绘图并保留孔洞；场输出不会空间平滑或补点。位置-时间图是对实际采样时刻和路径位置的散点图，不连线；缺帧、无支持位置、NaN 和长时间间隙不会被伪装成连续变形过程。多个帧比较时用共同色阶。

Green-Lagrange Exy 和 infinitesimal exy 是张量剪切；相应工程剪切分别为 2*Exy 与 2*exy。校准只派生物理位置和位移，不转换应变基底。应变分量与主轴角仍在图像基底（+x 向右、+y 向下）；若物理基底翻转一根轴，张量剪切分量符号会改变，须显式做张量变换。

## ROI/路径提取与多样品批处理

提取一帧序列中闭区间 ROI 的 Exx 摘要和 POI 表：

    python -m ezdic_cli agent extract --project outputs/sample_A --component Exx --roi 200,900,150,650

沿路径输出位移时程和位置-时间图：

    python -m ezdic_cli agent extract --project outputs/sample_A --component u --path-json experiment/path_px.json --max-distance 8 --coordinate-system image

多个样品各自 init 后，以受限 worker 数运行：

    python -m ezdic_cli agent batch --project outputs/sample_A --project outputs/sample_B --max-workers 2 --field-csv

batch 的 execution_status 表示流程是否完成，scientific_status 表示现有 core 数值质量门状态。批次可以完整执行，但某个样品仍需要质量复核。不能把任务完成或 CSV/图表存在解释成测量可靠或科学结论已批准。

## 进度、续算、取消和锁恢复

传入 progress-json 后，CLI 按行向标准输出发 JSON 事件，run 目录另保存 progress.jsonl 和最新 progress.json。status 可返回项目指针、锁和最新阶段。逐帧 checkpoint 记录输入、计划、代码、时间、参数和状态；当前身份匹配时可复用。图像、配置、代码或运行环境变化后，过期计划/检查点不能当作当前结果。

运行期间可请求合作式停止：

    python -m ezdic_cli agent cancel --project outputs/sample_A

取消在 DIC 进度边界生效，已经写入的检查点保留。异常退出后先执行 status 检查 .agent.lock。只有记录主机为本机、PID 与锁记录一致且进程已确认不存在，才执行 recover-lock --expected-pid <锁内PID>。远程、仍运行或无法判断状态的锁不可删除。

典型故障的处置：

| 情况 | 可核查状态 | 处置 |
| --- | --- | --- |
| 帧文件名包含多个数字 | init 报帧号歧义 | 用完整唯一的 frame map |
| 损坏帧 | inventory/unreadable 或逐帧 failed，后续帧仍处理 | 保留失败记录和空测量值，核对源文件，不按行号填充 |
| 图像尺寸不同 | dimension_mismatch | 检查裁剪、相机模式和图像目录 |
| 弱散斑/局部脱落 | plan review、QC 失败或 NaN | 复核照明、ROI、参考构型与遮罩；保存旧 run，再用有依据的新方案 |
| 相机/试验机时钟未知 | 力学对齐不可用 | 根据实验记录确定 offset，不把行序当同步 |
| 载荷通道采样过疏 | 超出 max_gap_s 的样本留空 | 给出有实验依据的间隔；不要跨大缺口插值 |
| 哈希或检查点不匹配 | verify.valid=false 并列出 issues | 将完整性问题和数值 QC 分开处理，修复后重新生成受影响阶段 |

## 输出结构和科研数据交付

每个项目输出结构如下：

    outputs/sample_A/
      project.json
      inventory.json
      CURRENT.json
      plans/plan_v001.json
      runs/run-<fingerprint>/
        run_state.json
        progress.json
        progress.jsonl
        summary.csv
        gauges.csv
        quality.json
        report.md
        manifest.json
        checkpoints/frame_native_*.npz
        checkpoints/frame_native_*.json
        plots/*.png
        fields_csv/*.csv          # --field-csv 请求时生成
        extractions/*             # ROI/path CSV、JSON 摘要和演化图

project.json 保存参数快照；inventory 保存原始帧号、时间、路径、文件哈希和读取状态；plan 保存参考帧、候选 ROI、参数试算和几何跨度；检查点是一帧数值字段 NPZ 与来源 JSON sidecar；summary.csv 是逐帧力学/ DIC 状态和摘要；quality.json 记录数值质量；manifest 对源图、代码环境、检查点和产物计算哈希。CSV 带 UTF-8 BOM，空值为空单元格，可直接由 pandas/Origin 读取。NPZ 使用 NumPy 原生压缩数组，不依赖 pickle。计算逐帧加载场，不要求将数万帧全部驻留内存。

verify.valid 表示来源索引、输入/代码指纹、检查点和交付物的完整性，不表示科学正确性。兼容字段 scientific_ok 表示现有 core 数值门限（例如相关有效比例、应变支持率与有限值）的结果；不表示实验准确度已标定、测量不确定度已评估、方法已被独立验证或科学结论已批准。PASS 只代表满足当前代码质量门。报告数值时仍需科研人员审核物理标定、实验几何、相机同步和独立基准。

## 方法背景

成熟软件的公开资料用于方法和功能参考，本项目本次未复制下列项目的源代码。现有 ezdic_core 已实现 ICGN/ICLM 和五次样条相关插值；本工作流将其组织为有界参数计划、逐帧检查点、质量记录和标准交付，不声称这些算法是本次新增或与商业软件等效。局部应变拟合跨度为 (strain_window_poi - 1) * step_px；包含子区的支撑跨度为 subset_size_px + (strain_window_poi - 1) * step_px。该跨度定义借鉴公开软件术语，不主张算法相同。

- [Ncorr DIC 算法说明](https://ncorr.com/index.php/dic-algorithms)：子区相关、IC-GN、亚像素插值和应变后处理背景；源码为 BSD-3-Clause。
- [VIC subset/step/strain filter 说明](https://correlated.kayako.com/article/44-subset-step-size-and-strain-filter-selection)：商业软件，仅作功能术语参考。
- [DICe 官方文档](https://dicengine.github.io/dice/)：功能参考；复用源码时须审查并遵守其版权和许可证。
- [muDIC](https://github.com/PolymerGuy/muDIC)：MIT 许可的开源功能参考；无代码复用。
- [py2DIC](https://github.com/Geod-Geom/py2DIC)：当前 README 限定非商业/研究使用；未直接复用或设为依赖。
- [iDICs Good Practices Guide](https://idics.org/guide/)：实验布置、相关分析和后处理实践参考；不复制其文本。

## 可复现合成端到端示例

脚本使用 core 的确定性高斯散斑生成器、带限频率过滤和已知仿射/平移图像 warp，生成两个样品：sample_normal 有 4 张完整图；sample_gap_bad 有多数字文件名、原生帧 13 缺失和损坏的 14 帧。合成载荷表记录 125 ms 的时钟偏移、kN 力和 um 横梁位移，载荷采样间隔为 0.4 s，因此部分图像时刻会落在试验机样本之间并使用有界线性插值。示例内的物理标尺和面积只用于合成回归，不是实验默认值。

    python examples/agent/create_demo.py --output outputs/agent-demo --run --cli source

也可使用已安装入口：

    python examples/agent/create_demo.py --output outputs/agent-demo --run --cli straintrace

--output 必须是新目录或空目录。脚本调用实际 CLI，并核对仿射 Green-Lagrange Exx 绝对误差不大于 0.001、路径位移绝对误差不大于 0.1 px、坏帧状态、无数据空值、机器插值 bracket、输入 SHA-256 不变、默认图、续算复用、完整性 verify、多样品并发批处理，以及仅对本次新建合成锁的死 PID 恢复。这些阈值是可复现合成工程回归门槛，能区分本示例中相邻加载阶段的差别；它们不是实际实验精度或测量不确定度。每步 stdout/stderr 保存到 logs，最终机器可读 receipt 在 DEMO_VALIDATION.json。

该例证明接口、数据组织和合成输入上的数值回归能工作，不是对真实实验精度、标定质量或测量不确定度的验证。换成真实样品前，必须用用户实际测得的标定、横截面、同步信息和独立参考进行审核。
