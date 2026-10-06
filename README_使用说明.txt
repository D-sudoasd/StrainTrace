当前源码许可更新（2026-09-27）：StrainTrace 已获作者授权采用 MIT 许可。允许修改、商业使用和再分发，须保留版权与许可声明。旧 ezDIC 归档不代表当前候选版本；历史条款保留于 docs/history/pre-mit。

ezDIC v0.1.4 使用说明
======================
发布日期：2026-08-30
当前身份：main/source version 0.1.4（release-ready source）。本源代码快照尚未创建 v0.1.4 tag 或 GitHub release；本地可能存在未经本轮冻结 smoke 验证的 ZIP 工作产物，发布资产以 Releases 页面为准。

v0.2.0-dev 开发目标（尚未发布）
------------------------------
当前 `main` 源码树已经实现一个有明确边界的研究级升级，但没有修改 `VERSION.txt`、
历史归档 DOI；2026-09-27 的 JOSS 准备已更新源码许可及引用信息，源码版本仍为
v0.1.4 元数据。目标是固定参考、局部子集、面内 2D DIC 加现有 1D 虚拟引伸计，
不是“超过所有 DIC 软件”或“已证明适用于实验数据”的声明。

升级目标包括：无 Tk 的数值核心、基于参考帧的百分位归一化、有边界的粗到细多尺度
初始化、IC-GN/IC-LM 求解器质量诊断、明确区分点相关 `valid` 与应变拟合
`strain_valid`，以及带输入/输出哈希的可追溯运行 provenance。锁定的合成基准是
回归门槛，不是实验不确定度估计。

定位与开发者
------------
ezDIC 由 Dr. Delun Gong 开发，是一个面向图像序列的双模式应变工作站：

1. 虚拟引伸计：追踪两组用户定义的 ROI 标记，导出一维工程应变、真应变和 QC。
2. 全场 2D DIC：在矩形 ROI 内建立 POI（point of interest）网格，用 IC-GN 或 IC-LM 计算面内位移与应变场。

全场模式是面内、局部子集的 2D DIC。它不是逐像素 DIC、立体/3D DIC、DVC，也不是全局有限元 DIC。

DOI
---
10.5281/zenodo.20222465
https://doi.org/10.5281/zenodo.20222465

系统要求
--------
- Windows 10/11 x64。
- 便携版用户不需要在电脑上安装 Python。
- 正常运行不需要管理员权限。

如何运行便携版
--------------
1. 打开 [Releases 页面](https://github.com/D-sudoasd/StrainTrace/releases)。当前已发布的便携 ZIP 是 **v0.1.3**（仅虚拟引伸计）。本 `main` 源码是未发布的 **v0.2.0-dev**，引用元数据仍是 **v0.1.4**，不是 GitHub Release 资产。未经本轮验证的本地 ZIP 不应当当作发布资产。
2. 解压完整的已下载文件夹；便携包内的顶层目录名是 `ezDIC_Windows_x64`，版本号只出现在 ZIP 文件名中。
3. 双击 ezDIC.exe。源码运行也可双击仓库根目录的 `start_ezDIC.bat`。

重要：不要只复制 ezDIC.exe。`_internal` 文件夹以及 LICENSE.txt、CITATION.cff、VERSION.txt 和 NOTICE_Attribution_and_Usage.txt 必须与 ezDIC.exe 一起保留。 Do not copy ezDIC.exe alone.

源代码运行、测试与构建
----------------------
源代码运行需要基础依赖；测试和 Windows 发布构建使用包含基础依赖、测试工具和 PyInstaller 的构建集合。OriginPro 是独立的可选依赖，不会因基础构建而安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m pytest -q
```

可安装的 Python 包
--------------------
当前源码支持 Python 3.11–3.13；桌面 GUI 需要 Tk，Windows 是维护中的桌面平台。
在源码仓库根目录安装后，可使用新的 StrainTrace 命令；旧 ezDIC 模块和命令仍保留：

```powershell
python -m pip install ".[test]"
straintrace --help
straintrace-gui
```

在仓库外的工作目录运行合成检查：

```powershell
straintrace benchmark --output straintrace-synthetic-check
```

该检查自行生成合成图像并验证定义的数值门槛，不代表实验精度或质量阈值已标定。
API 与使用说明见 docs/API.md；这不会把当前源码变成已发布的便携版。

无界面 CLI（v0.2.0-dev 源代码与冻结合约）
----------------------------------------------
当前 CLI 不导入 GUI/Tk，使用 `schemas/run_config_v1.json` 的严格 UTF-8 JSON 配置。
未知字段、非有限数字、模式不允许的字段、输入形状问题和不支持的模式会被拒绝。
`image_paths` 与 `image_folder` 二选一；文件夹按 ezDIC 的自然排序和支持的图像后缀
收集。请把分析输出和基准报告放在仓库之外的调用方临时目录。

以下是最小但完整的配置片段；示例图像和输出路径需要替换成真实路径，字段名逐一
对应 `schemas/run_config_v1.json`。1D 的 `reference_frame_1based` 必须等于
`start_frame_1based`；2D 的 `field_roi_reference_frame_1based`（省略时默认同值）
必须等于 `reference_frame_1based`。2D 运行先把选定参考帧用于初始化，再把其他帧
全部与同一固定参考帧相关。当前 CLI/core 的归档目录是输出根下的
`_previous_runs/<run_id>/` 与 `_failed_runs/<run_id>/`；GUI 对尚无
`run_manifest.json` 的旧 `dic/frame_*` 会先一次性迁移到
`dic/_previous_runs/<timestamp>/`。

```json
{
  "schema_version": 1,
  "analysis_mode": "extensometer",
  "image_paths": ["C:/data/seq/frame_001.png", "C:/data/seq/frame_002.png"],
  "start_frame_1based": 1,
  "end_frame_1based": 2,
  "reference_frame_1based": 1,
  "output_dir": "C:/data/ezdic_1d_output",
  "roi_groups": [{
    "name": "axial",
    "roi1": [10, 20, 21, 21],
    "roi2": [100, 20, 21, 21],
    "strain_mode": "x",
    "role": "axial"
  }],
  "tracking": {
    "template_policy": "fixed_reference",
    "use_prev_frame_template": false
  },
  "quality": {
    "min_valid_frames": 1,
    "min_strain_valid_fraction": 0.8,
    "enable_fb_check": true
  },
  "normalization": {"policy": "reference_percentile", "clip": true},
  "export": {"write_manifest": true},
  "transaction": {"enabled": true},
  "metadata": {}
}
```

```json
{
  "schema_version": 1,
  "analysis_mode": "fullfield",
  "image_paths": ["C:/data/seq/frame_001.png", "C:/data/seq/frame_002.png"],
  "start_frame_1based": 1,
  "end_frame_1based": 2,
  "reference_frame_1based": 1,
  "output_dir": "C:/data/ezdic_2d_output",
  "field_roi": [20, 20, 120, 100],
  "solver": {
    "name": "IC-GN",
    "subset_size_px": 21,
    "step_px": 5,
    "strain_window_px": 5
  },
  "pyramid": {"levels": 1, "scale": 0.5},
  "quality": {
    "min_correlation_valid_fraction": 0.95,
    "min_strain_valid_fraction": 0.8
  },
  "normalization": {"policy": "reference_percentile", "clip": true},
  "export": {"write_manifest": true},
  "transaction": {"enabled": true},
  "metadata": {}
}
```

验证、运行、进度和 manifest 检查：

```powershell
py -3.11 -m ezdic_cli validate-config --config .\run_config.json
py -3.11 -m ezdic_cli run --config .\run_config.json
py -3.11 -m ezdic_cli run --config .\run_config.json --progress-json
py -3.11 -m ezdic_cli verify-manifest --manifest <output_dir>\run_manifest.json
py -3.11 -m ezdic_cli benchmark --output $env:TEMP\ezdic-benchmark
```

退出码固定为：`0` 表示配置、执行、scientific gate 和 manifest 验证都通过；`2`
表示配置/命令用法或 preflight 失败；`3` 表示 I/O、求解器、导出或其他运行时失败；
`4` 表示 scientific gate 或 manifest 验证失败。`run --progress-json` 输出逐行 JSON
的 `run_started`、`progress`、`run_finished` 事件。源代码入口是 `python ezdic_cli.py`；
冻结 onedir 包中的 `ezDIC-cli.exe` 提供相同合约。

无界面运行与 GUI 使用相同的固定参考和数值核心。事务生命周期是 output root 下的
当前 `core/`、`qc/`、`optional/` 或 `dic/`，以及同级的 `_previous_runs/<run_id>/`
（可恢复旧结果）和 `_failed_runs/<run_id>/`（致命失败 staging）；旧 current 结果只
在 commit 阶段归档，失败运行不会替换它们。1D 参考初始化行是注册基线，不计入科学
有效变形帧的 gate。失败 POI 可以保留 `x`/`y` 网格坐标，但测量字段（`u`/`v`、应变和
适用的质量字段）保持 `NaN`。

manifest 绑定规范化配置、有序输入身份与 SHA-256、图像尺寸/类型、归一化策略与边界、
代码/环境指纹、输出清单与哈希、状态和 scientific gate；`verify-manifest` 会重新计算
这些字段，发现缺失、改变、意外或篡改的 manifest-listed 文件，且不修改用户数据。

1D 默认使用固定参考模板（`template_policy=fixed_reference`、
`use_prev_frame_template=false`）；`follow_deformed_experimental` 只有在显式启用时
才会使用，不属于固定参考声明。全场正 coverage 门槛默认是相关 `valid` 比例至少
0.95、应变 `strain_valid` 比例至少 0.80；配置只能收紧，不能静默放宽。质量比
`best_to_second_peak_ratio_min` 的定义是 `best_peak / second_peak`，越大越好，
不能误写成反向的 `second_peak / best_peak`。

构建便携包：

```powershell
powershell -File .\build_release.ps1
```

开发版构建的 onedir 目录包含 `ezDIC.exe`、`ezDIC-cli.exe`、无 GUI 的 core/CLI/
benchmark 模块以及 `schemas/run_config_v1.json`。`ezdic_frozen_entrypoint.py
--smoke-test` 在创建任何 Tk 根窗口之前导入并检查 core、CLI、benchmark、schema 及
必要支持文件；只有当 `EZDIC_FROZEN_SMOKE_MARKER` 明确指向调用方临时路径时才写入
marker，默认不写仓库或用户输出目录。

锁定的 v5 合成工程 gate（`report_version=ezdic-benchmark-report-v5`、
`cases_version=ezdic-benchmark-cases-v3`、locked case hash
`3dbe0dae3fdf8f30ec32c9fd8f036f0a53b4a705380626e7860773f62f31cb20`）在固定几何、
种子、clean baseline 和图像 corruption panel 下的观测值为：

- 小平移 `[2.3, -1.2]`：RMSE **0.0199391 px**、P95 **0.0292620 px**、最大误差 **0.0325828 px**；
- 大平移 `[28, -18]`：RMSE **0.0115297 px**、P95 **0.0239809 px**、最大误差 **0.0269902 px**，三层 pyramid、search radius 8；
- 仿射场：位移 RMSE **0.00363440 px**、P95 **0.00651265 px**、最大误差 **0.0103737 px**；
- 仿射应变：最大分量误差 **0.000273879**、consistency 误差 **0.000270908**；
- 近一维周期纹理：求解器/导出前 typed `AMBIGUOUS_TEXTURE`，`solver_calls=0`，成功 artifact 数 **0**。

quality-score v1 的 ranking 覆盖 **567 个 numeric solver rows**：**563 个 good**、
**4 个 bad**（其中包含 **2 个 rejected bad**），ROC-AUC 为 **0.994227353463588**，
误差容限 **0.25 px**（gate ≥ 0.90）。说明性阈值明确为 `NOT_CALIBRATED`，
`quality_threshold_evaluated=false`、`quality_threshold_pass=null`，不是校准后的二元
接受规则。按 finite-error labels，false accept 为 **2/2 = 100%**；按 ranking outcomes，
false accept 为 **2/4 = 50%**。JSON 报告与非空逐点 CSV 通过 SHA-256 关联；canonical
evidence CSV 的 SHA-256 为
`39d4e52f35cd3161a1e877b6edcd5187568bf275c6c8d552422605b73b4c0bfb`。

这些是固定合成工程 gate 的观测值，不是实验不确定度、标定验证、自然纹理性能，也不是
ezDIC 相对所有 DIC 项目的精度或鲁棒性证明；AUC 不能解读为实验精度或普适错误检测能力。

Origin OPJU 是可选功能。若只需要基础运行/测试，不必安装 OriginPro；启用 OPJU 时，在 Windows + OriginPro 2021+、有效本地许可证条件下安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-origin.txt
```

基础依赖中不包含 `originpro`。如果 OriginPro 或该 Python 包不可用，TXT/PNG/CSV 等已有结果仍可保留，OPJU 导出会单独报告失败。

通用工作流
----------
1. 选择图像文件夹。
2. 选择或确认输出文件夹。
3. 加载图像序列。
4. 设置起始帧和结束帧。
5. 根据当前模式完成下列步骤。
6. 点击“开始分析并导出结果”。

虚拟引伸计模式
--------------
1. 选择应变方向和追踪模式。
2. 在参考帧绘制 ROI 1 和 ROI 2，然后添加 ROI 组。
3. 可选：为 ROI 组设置 axial（拉伸方向）/ transverse（横向方向）角色以导出泊松比。同一角色和实际方向的多组会逐帧求平均。
4. 检查预检信息后开始分析。

该模式的核心结果写入 `core/`，质量控制摘要写入 `qc/`；可选的完整 CSV、相关系数图、追踪 overlay、参数和论文图包写入 `optional/`。

全场 2D DIC 模式
----------------
1. 切换工作区模式到“全场 2D DIC”。
2. 在当前分析范围的第一帧（固定参考帧）绘制一个矩形 field ROI。
3. 设置 subset 尺寸、POI 步长以及 IC-GN 或 IC-LM 求解器。
4. 开始分析。每个后续变形帧都与同一个固定参考帧相关，不会静默改成逐帧参考。

所选全场分析范围内的图像尺寸必须一致；尺寸不一致会在预检/处理时失败。

全场输出采用事务式流程：本轮候选文件先写入 staging 临时区，只有整轮正常完成后才提交到输出根目录的 `dic/`，提交时才成为当前结果。提交阶段会把旧的成功 `frame_####*` 结果保留性移动到输出根目录（`dic/` 的同级）的 `_previous_runs/<run_id>/`。

全场结果固定写入输出根目录下的 `dic/`；对有有限应变场的变形帧，不依赖“任选导出项”即可得到核心数据。没有有限应变场的帧属于**正常跳过的失败帧**：它不生成本轮当前帧文件，但只要其他帧有效且没有致命异常，其他有效帧仍可提交：

```text
dic/
├─ frame_0002.txt       # x, y, u, v, zncc, valid, Exx, Eyy, Exy, exx, eyy, exy，随后是 strain_valid 与求解器质量列
├─ frame_0002.csv
├─ frame_0002_u.png
├─ frame_0002_v.png
├─ frame_0002_Exx.png
├─ frame_0002_Eyy.png
├─ frame_0002_Exy.png
└─ frame_0002_parameters.txt
```

每个有效变形帧的 `frame_####_parameters.txt` 都是固定的 provenance 记录，至少说明：固定 `reference_frame_1based` 与 `reference_filename`、当前 `frame_global_1based` 与 `frame_filename`、`field_roi`、图像尺寸，以及实际生效的 `subset_size_px`、`step_px`、`strain_window`、`solver`、`zncc_min` 和平滑参数；若可用，还记录图像序列 fingerprint。

如果重复运行时 `dic/` 中已经存在匹配的 `frame_####*` 结果，提交阶段会把旧结果保留性移动到输出根目录的 `_previous_runs/<run_id>/`，再提交本轮结果，避免把不同运行的同名帧混在一起。

字段和科学边界
--------------
- `u`/`v`/`x`/`y` 均以 **px** 记录；其中 `x`、`y` 是 POI 坐标，`u`、`v` 是位移。
- `Exx`、`Eyy`、`Exy` 是 Green-Lagrange 应变；`exx`、`eyy`、`exy` 是无穷小应变；应变均为**无量纲**。
- `Exy`/`exy` 是应变张量的非对角剪切分量，不是额外乘以 2 的工程剪切量。
- 输出是矩形 ROI 内的 POI 网格，不是每个图像像素的结果，也不提供 3D/DVC 信息。
- `valid` 标记 POI 是否通过相关质量阈值。失败点的测量字段（适用时的 `u`/`v`、应变和质量字段）保持 `NaN`；`x`/`y` 仍是网格坐标，不插值、不用邻点填补。
- 没有有限应变场的变形帧视为正常跳过的失败帧；如果整个全场分析没有有效应变点，运行失败，不报告为“完成”。
- 若中途发生致命的 I/O、求解或导出异常，本轮 staging 中已经产生的 partial 文件会保留性移动到输出根目录（`dic/` 的同级）的 `_failed_runs/<run_id>/`，不会作为当前成功结果留在 `dic/` 根目录；整轮报告失败且不提交 partial 结果。
- 预检要求至少形成 **3 × 3 POI 网格**；ROI 过窄、subset/step 组合无法形成三行三列 POI 时会在开始前阻止处理。
- 全场 UI 有一个可见的可选 **Exx overlay** 开关。开启后会在分析图像上叠加 Exx 色图，并保存为 `frame_NNNN_overlay.png`；它只是可视化 QC，不替代固定核心表格和应变图。
- 1D 导出项在全场模式下不参与 2D 核心输出，也不会成为 2D 分析的开始条件。

本开发目标明确不实现或不宣称：立体/3D DIC、DVC、GPU/MPI、全局有限元 DIC、
SIFT/AKAZE 特征引导、任意实验纹理鲁棒性、裂纹/遮挡拓扑 mask、实验标定和不确定度
量化。上述方向需要独立的基线、数据和科学验收，不能由本次合成基准替代。

默认输出
--------
- Origin-compatible TXT：`Frame`、`EngineeringStrain`、`TrueStrain`。
- 同一角色和实际方向的重复引伸计平均应变 TXT。
- 设置 axial/transverse 角色后生成泊松比 TXT/PNG。
- 工程应变 PNG 图。
- QC 摘要 TXT。

一维应变定义：

```text
engineering strain = (L - L0) / L0
true strain        = ln(L / L0)
PoissonRatio       = - ε_transverse / ε_axial   (engineering)
```

追踪失败帧、缺失应变值和过小轴向应变对应的结果保留为 `NaN`，不静默插值。

论文图包
--------
启用论文图包后，会在 `optional/publication_figures` 导出 PNG/TIFF/PDF/SVG/EPS。该图包只增加统一样式的图，不替代核心 TXT/CSV 数据或 QC 证据。

推荐引用
--------
Gong, D. (2026). ezDIC: A lightweight virtual extensometer for extracting linear strain from image sequences (Version 0.1.4) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20222465

当前源码许可
--------------
当前源码采用 MIT 许可。允许使用、复制、修改和再分发，须保留版权和许可声明。具体条款见 LICENSE.txt。上列 DOI 属于历史 ezDIC 归档，使用当前候选版本时请同时记录 Git 提交，勿将历史归档当作本次源码。

安全提示
--------
当前版本未进行代码签名。部分电脑上的 Windows Defender 或 SmartScreen 可能显示“未知发布者”警告。
