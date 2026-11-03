# 逐帧比较 StrainTrace 与 VIC-2D 数值导出

`benchmarks/compare_dic_exports.py` 用于比较同一图像序列的两套数值结果。它报告两者的差异及缺测位置；哪套结果更接近真实变形，还需要独立真值或实验依据。

## 先使比较对象一致

StrainTrace 的 `Exx/Eyy/Exy` 是 Green–Lagrange 应变，`exx/eyy/exy` 是无穷小应变。VIC 的列名通常仍为 `exx/eyy/exy`，但其应变定义由计算设置决定，默认是 Lagrange。因此，**列名必须映射，应变定义必须另行核对**。VIC 的 Engineering 选项也不能直接当成 StrainTrace 的无穷小应变。

官方说明：[Comparing 2D DIC Data to Theory/Strain Gauges/Other Techniques](https://correlated.kayako.com/article/75-comparing-2d-dic-data-to-theory-strain-gauges-other-techniques)；[Strain Tensors and Criteria in VIC](https://correlated.kayako.com/article/2-strain-tensors-and-criteria-in-vic)。前者区分局部应变、区域均值及引伸计测量，后者解释应变定义和剪切约定。本文只提取与本工具输入有关的条件，未把商业软件当作真值。

当前比较工具要求两侧均采用：

- 同一原始参考图 SHA-256、参考帧身份和目标图 SHA-256；逐帧对应关系由 manifest 明确列出。
- 参考图坐标，x 向右、y 向下，原点为同一参考图原点；x、y、u、v 均为 px。工具不自动旋转、移动原点、转换标定、变换应变或切换参考帧。
- 相对上述参考图的总位移。增量结果必须先正确组合回同一参考状态；工具不累加增量应变。
- 相同 Green–Lagrange 或无穷小应变，单位为 1，剪切为张量分量；百分数、工程剪切、Hencky 等输入需在原软件明确转换/重新导出后使用。
- 相同参考 ROI。子集、步长、形函数、应变窗口、参考策略、滤波和平滑、有效性规则分别记录；这些参数不同不会被隐藏，报告将列出差异。

manifest 内的图像 SHA-256 和计算设置是操作者从原始图像、项目及运行记录填写的声明。工具核对声明是否一致，并实际读取和记录 CSV、固定参考网格、manifest 和脚本的 SHA-256；它不能单凭 CSV 证明声明的参考图或应变设置真实正确。

## 准备文件

1. 复制 [模板](../examples/paired_dic_comparison.json)，填写实际参考与目标图 SHA-256、帧身份、ROI 和两套软件的设置。模板占位值会被拒绝，不是已有实验结果。
2. 导出逐点坐标、位移、应变和有效性。StrainTrace 可直接使用 `dic_field_to_dataframe`/现有逐帧 CSV；模板已映射其字段。VIC 侧根据真实列名填写 `columns`，区分大小写。不要根据列名猜测应变定义。
3. 两侧都必须有明确的 `eligible`、`correlation_valid`、`strain_valid` 布尔列，值为 `true/false` 或 `1/0`。这些列必须来自实际的试样遮罩/候选域及软件接受状态。若 VIC 导出只保留有效行，需根据其导出规则显式标记这些行，并将 `rows_representation` 设为 `partial_rows`。工具不会从 `sigma` 数值推断是否有效，也不会把缺失点标为有效。
4. 准备固定参考采样表 `comparison-grid.csv`，包含 `x,y,eligible`。可从参考图已确定的 StrainTrace POI 网格保留这三列；须保留预定候选位置，不能用某一变形帧的有效点集合替代。网格及其 `eligible` 不随帧变化。
5. 将均匀变形、颈缩起始、明显颈缩、后期等需要比较的实际帧依次列入 `frames`。某侧没有结果时填 `null`，保留该帧。`stage` 是实验人员给定的阶段标签，程序不依据帧号自动判定颈缩。

每份导出应如实声明 `rows_representation`：`complete_grid` 表示包括未通过点的完整原生候选网格，原生 ROI 覆盖率分母为其中 eligible 点；`partial_rows` 表示有省略，原生 ROI 覆盖率将为 `null`，避免将仅有效点导出误报为 100% 覆盖。固定参考采样表仍给出不随帧改变的共同覆盖分母；没有匹配到的点计入缺测。

StrainTrace 对子集翻折等情况会保留相关接受标记 `valid=true`，同时使发布的 `u/v` 和应变不可用。工具分别报告 `correlation_valid_n` 和 `displacement_available_n`；这类点标为 `displacement_unavailable`，不参加位移差计算，也不以 `u_raw/v_raw` 替补。

如果两侧 POI 网格不同，程序只比较两边共同存在的参考坐标。`coordinate_tolerance_px` 仅吸收坐标导出的舍入误差，最高 0.001 px；不会用相邻点、插值或扩大容差代替同一点。没有共同坐标时仍保留各侧原生 ROI 统计，但共同误差为 `null`。要扩大逐点比较范围，应在分析前设置相同参考采样位置，或另外设计并验证重采样实验，不能把重采样当成原生测量。

## 运行与输出

```powershell
python benchmarks/compare_dic_exports.py --manifest C:/data/paired/comparison.json --output C:/results/paired-comparison-01
```

路径相对 manifest 所在目录解析。输出目录必须是新目录；源文件只读，已有输出不覆盖。完整计算成功后才发布输出目录。

`report.json` 逐帧包含以下内容：

| 内容 | 含义 |
| --- | --- |
| `native_roi` | 各侧全 ROI 内自己的有效点、失效原因、相关质量，以及各应变分量的有符号最大值 `strain_peaks` 和最小值 `strain_minima`；`partial_rows` 的覆盖率不可用 |
| `native_roi_extrema_comparison` | 两侧各自在原生有效点集合上的有符号最大值/最小值、对应数值差及唯一极值位置距离；两侧有效集合可能不同，差异应结合缺测分布解释 |
| `fixed_domain` | 两侧分别在固定参考采样表上的匹配点、缺测、排除及相关/应变覆盖率 |
| `common_correlation_n` / `common_displacement_n` / `common_strain_n` | 同参考坐标上两边相关接受、发布位移可用及应变有效的点数，分别相对于固定参考域报告比例 |
| `component_differences` | StrainTrace 减另一侧的分量偏差、RMSE、95%绝对差和最大绝对差；这些是软件间差异，不是测量误差 |
| `displacement_vector_rmse_px` | 共同有效位移点上的二维位移向量差 RMSE |
| `common_domain_peaks` / `common_domain_minima` | 共同有效应变点上的各侧有符号最大值/最小值、对应数值差及唯一极值位置距离；与各自全 ROI 极值分开报告 |
| `parameter_differences` | 两侧子集、步长、应变窗口、参考策略、滤波及有效性设置的差别 |

每帧的 `frame_XXXX_points.csv` 保留固定参考坐标、两侧缺测/排除/失效状态、质量值、失效原因和逐分量差，供检查失效区域。缺测差值为空，不填零。极值仅由有效应变点产生；若有并列最大值或最小值，报告并列点数并将对应的唯一极值位置距离留空。

`strain_peaks` 和 `common_domain_peaks` 保留有符号最大值含义，输出中明确标为 `signed_maximum`，不是绝对幅值最大值。横向压缩或负剪切的局部化通常应查看 `strain_minima` / `common_domain_minima` 中标为 `signed_minimum` 的数值与位置。例如 −0.25 比 −0.01 压缩更强，但有符号最大值是 −0.01；工具同时给出这两个极值，不自动替用户选择拉伸或压缩方向。两侧最小值的 `amplitude_difference` 仍是 StrainTrace 数值减另一侧数值，保留符号。

ZNCC 与 VIC `sigma` 的含义和单位不同，因此各自报告分布，不相减、不转成统一“质量分数”。某侧整帧缺失、没有共同有效位移或没有共同有效应变时，状态分别表现为相应缺测统计和 `no_common_displacement` / `no_common_strain`，对应差值为 `null`。`compared` 只表示存在可比较的点；报告始终设置 `equivalence_claim=false`。

## 本次尚未得到的实验比较

2026-10-28 旧工作分支中的 FIO83 记录指向外部 H 盘原图，且历史参照为 VIC-3D 的一条局部引伸计，StrainTrace 则为五组不同位置端点的平均。该记录不构成本次 VIC-2D 同测量比较。2026-11-03 的本工具验证使用构造的 CSV 检查坐标匹配、应变映射、失败保留和覆盖分母；不是用户真实颈缩数据。

要完成实际差异归因，需要原始参考图与逐阶段目标图、两侧逐点数值导出和完整设置，以及原始失效/遮罩信息。仅有截图、区域平均曲线或不同位置标距的引伸计结果，不能定位颈缩峰值差异来自位移相关还是应变重建。
