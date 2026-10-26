# 从实验文件夹到可独立核查的应变结果

本工作流复用 StrainTrace 的固定参考、面内二维 DIC 引擎。输入是同一相机、同一试样的图像序列；输出是原始测量点的位移与应变、多区域统计、虚拟引伸计曲线和离线报告。一个项目对应一个图像序列，多个试样沿用 `agent batch`。

## 1. 明确输入及分析目标

自动发现先列出可用序列、帧号、图像状态及实验元数据，再建立项目。多个相机或试样目录不能仅因文件名相似而混成一个序列。计算结果、图表和检查点目录应与原始图像目录分开。

分析目标决定所需信息：

| 目标 | 可由图像得到 | 需要实验记录支持 |
| --- | --- | --- |
| 全场应变 | 图像坐标中的位移、Green–Lagrange 与无穷小应变 | 参考状态、实际试样区域；毫米位移需要像素标定 |
| 虚拟引伸计 | 固定参考端点或端点区域之间的相对长度变化 | 标距位置、方向和所代表的试样部位 |
| 应力–应变 | DIC 标距应变及与其对齐的力学记录 | 力单位、截面积、相机与试验机时钟关系 |

软件不会从图像大小推测毫米，不把纹理区域自动认定为试样，不把图像帧号直接当作秒。仅图像应变分析缺少采集时间时，自动入口使用 `time_basis: "frame_index"`，`time_s` 留空，按原生帧号分析和绘图。此模式不计算应变率或同步载荷。需要物理时间的目标会列出所需输入。缺少物理标定时仍可用像素坐标计算无量纲应变；物理长度留空。

### 命令行和 Agent 调用

源码环境先安装依赖，随后可直接运行；下例中 `E:/experiment/sample_A` 是输入，`E:/analysis/sample_A` 是独立输出目录。

```powershell
py -3.13 -m pip install -e .
py -3.13 -m ezdic_cli agent discover --folder E:/experiment/sample_A --objective full_field
py -3.13 -m ezdic_cli agent auto --folder E:/experiment/sample_A --project E:/analysis/sample_A --objective full_field --field-csv --progress-json
py -3.13 -m ezdic_cli agent verify --project E:/analysis/sample_A
```

项目支持 Python 3.11–3.13；命令中的 3.13 是本轮验证环境。已安装包时可把 `py -3.13 -m ezdic_cli` 换成 `straintrace`；便携版使用 `ezDIC-cli.exe`。自动入口支持 `full_field`、`virtual_gauge`、`stress_strain` 及帮助中列出的中英文别名。通用 AI Agent 可先理解用户的自然语言目标，再传递明确目标和结构化选项；软件本身不调用外部大语言模型。

`discover` 返回候选序列、可读取的表格列、已解析选项和 `required_inputs`。`auto` 返回 `complete`、`needs_input`、`review_required` 或 `attention_required`，并保留计算与校验明细；调用方必须读取状态，不能只看生成了文件。

实验目录可放置 `experiment.json`，记录已有实验信息。以下仅为图像坐标示例，ROI 和标距必须按实际图像修改；未提供像素物理标定、载荷或时间：

```json
{
  "schema": "straintrace.experiment/v1",
  "options": {
    "image_dir": "images",
    "regions": [
      {"name": "upper", "bounds": [32, 144, 32, 64], "coordinate_system": "image", "min_valid_fraction": 0.8, "min_valid_pois": 6},
      {"name": "lower", "bounds": [48, 128, 80, 112], "coordinate_system": "image", "min_valid_fraction": 0.8, "min_valid_pois": 6}
    ],
    "gauges": [
      {"name": "central", "axis": "distance", "a_point_px": [48, 64], "b_point_px": [128, 64], "max_distance_px": 8}
    ]
  }
}
```

相对源文件路径以实验目录解析。`--options-json FILE` 接收上述 `options` 对象本身，用于显式覆盖本次发现选项。采集间隔确知时添加 `frame_interval_s`；有不等间隔时使用含 `filename,native_frame,time_s` 的 `frame_map_csv`。应力分析还须配置前文列出的实测力学元数据。

## 2. 检查候选区域与计算支撑

计算区域规定 DIC 测量点所在范围。多个统计 ROI 是该位移场内各自独立的查询区域，因此可以比较不同位置和尺寸而不用重复相关求解。区域外或遮罩排除的像素不会因后处理查询而获得新的测量值；需要扩大计算范围时，应更改计算区域并重新分析。

区域推荐利用纹理、空间位置和可用测量点支撑。候选的分数表示适合相关计算的程度，不表示已识别试样材料或已确定加载方向。夹具、标签、孔洞和背景应通过计算区域、遮罩或排除区控制。

不同输入的坐标约定明确区分：

| 对象 | 字段/格式 | 坐标 |
| --- | --- | --- |
| DIC 计算区域 | `roi_override_px: [x, y, width, height]` | 图像像素 |
| 标距端点区域 | `a_roi_px` / `b_roi_px: [x, y, width, height]` | 图像像素 |
| 具名统计区域 | `bounds: [xmin, xmax, ymin, ymax]` | `image` 为像素，`calibrated` 为毫米 |
| 标距点端点 | `a_point_px` / `b_point_px: [x, y]` | 图像像素 |

统计 ROI 使用闭区间。每个区域记录最少有效测量点数、有效比例及可选 ZNCC 下限。位移有效与应变拟合有效是不同条件。质量不满足时保留区域、帧号和原因，结果留空；不会用相邻区域的数值代替。

一个 ROI 的均值是保留测量点的统计量，不是该区域内每个像素的实测应变。标准差描述空间离散，不是独立标定的测量不确定度；重叠子区的测量值也不能视为独立样本。

## 3. 设置多个虚拟引伸计

每根引伸计有独立名称、两个端点或端点区域、测量方向和质量状态。端点在参考帧上解析成固定测量点集合；后续帧不能悄悄换成附近其他点。如果任一所需端点失效，该标距的本帧结果留空。

`distance` 使用两端点的欧氏距离，适用于随试样刚体转动的标距；`x` 和 `y` 使用图像轴上的投影长度。两种定义不能混称。对于可发生明显转动的试样，图像轴投影的变化含有转动影响，应按研究目的选择。

对于所选长度定义，工程应变为 `(L-L0)/L0`，真应变为 `ln(L/L0)`。参考长度必须为正；未知物理标定时以 px 记录长度。有显式各向同性像素标定时，同时提供 mm 长度与伸长。

多根标距可以比较局部变形、不同标距长度或不同位置。不同方向或不同物理部位的应变不应不加区分地求总平均。自动标距始终保留其候选来源和几何状态。

## 4. 计算、检查和局部重新分析

计划记录参考帧、候选计算区域、子区尺寸、测量点间距、应变拟合窗口以及有限试算结果。只有计划通过现有数值检查才执行逐帧相关。输入或求解参数变化后，旧计划不能作为当前计划继续使用。

已保存的位移场可用于修改统计 ROI 和虚拟标距，再生成一个独立的后处理结果。此操作不重新运行 DIC，也不覆盖原始逐帧检查点。更改参考帧、计算区域、遮罩或测量点间距则需要新计划和新的相关计算。

例如，从保存场重算新的区域定义：

```python
import json
from straintrace_agent import api

regions = json.loads(open("regions.json", encoding="utf-8").read())
result = api.reprocess("E:/analysis/sample_A", regions=regions)
print(result)
```

`regions.json` 可参考 [multi_regions.json](../examples/agent/multi_regions.json)。同时传入 `gauges=[...]` 可重设标距。计算区域等项目参数的修改使用 `agent update --project ... --changes-json changes.json`；`changes.json` 为项目字段的替换值，例如 `{"step_hint_px": 12}`。随后执行 `agent analyze` 生成新计划和运行。

运行状态、数值质量和数据完整性分别检查：

- `complete` 表示对应操作完成。
- 数值质量门检查相关覆盖、应变支撑及有限值等条件。
- 完整性校验检查输入、参数、源代码、检查点和交付文件是否与记录一致。

任何一个状态都不能单独替代实验标定或独立测量验证。损坏图像、相关失败、缺帧和缺失载荷应保留在结果表与报告中。

## 5. 阅读和交付报告

报告同时展示计算区域、统计 ROI、标距位置、参考状态、参数和逐帧结果。曲线与表格来自保存的 CSV；失败值保留为空，缺帧不会被改成连续采样。HTML 可离线打开，无需外部服务。

完整报告包保存 CSV、参数和计划快照、几何信息、逐帧 NPZ/JSON、文件 SHA-256 清单及独立校验脚本。原始图像通过路径与 SHA-256 关联，参考图预览仅用于位置检查；需要独立重跑相关分析时，仍须保存对应的完整原始图像序列。

报告可以离开项目目录后核查包内数据；这与重新运行图像相关是两个操作。原始图像和配置修订应与报告一起按实验管理规则保存。

```powershell
py -3.13 -m ezdic_cli agent report --project E:/analysis/sample_A --include-checkpoints
```

返回 JSON 的 `report_html` 是浏览入口。将其整个 `report` 目录复制给同事后，可在该目录执行 `python verify_report.py .` 独立校验。`auto` 的最终交付已包含检查点；普通逐帧运行先生成轻量报告以减少复制，显式交付时再包含原始场。

### 桌面操作

启动主窗口后选择“自动分析”。在“输入与参数”选择实验文件夹和输出项目；在“ROI 与标距”添加、命名、数值调整或在独立图像窗口拖绘多个区域与端点。使用“保存设置”“生成计划”“运行 / 恢复”推进，并在“计划与结果”检查质量、打开报告或重做后处理。后台计算期间可请求停止，完成的逐帧检查点保留。

## 可复现验证

下列脚本创建独立的解析纹理和已知变形，运行真实 CLI，并检查多个 ROI/标距、缺帧、损坏图像、载荷单位、来源校验、重复执行和保存场后处理；输出必须使用新的仓库外目录：

```powershell
py -3.13 scripts/validate_agent_upgrade.py --output E:/analysis/straintrace-validation-new
```

`VALIDATION.json` 为本次运行收据，日志和详细数值保留在同一目录。`--core-only` 仅验证核心解析算例；它不能代替完整 CLI 流程验收。

本轮实测结果、便携版本验证和覆盖边界见[升级验证记录](AGENT_UPGRADE_VALIDATION.md)。

## 方法与适用范围

`x/y/u/v` 为像素坐标与位移；应变无量纲。`Exx/Eyy/Exy` 为 Green–Lagrange 应变，`exx/eyy/exy` 为无穷小应变。剪切字段为张量分量，工程剪切为相应字段的两倍。图像基底为 x 向右、y 向下；坐标单位换算不会自动旋转应变张量。

子区尺寸、测量点间距和应变拟合窗口共同决定空间支撑。本系统记录的满窗口支撑跨度为 `subset_size_px + (strain_window_poi-1)*step_px`；应按待识别变形尺度选择，并检查改变窗口对结果的影响。这个支撑定义参考 [Correlated Solutions 的公开说明](https://correlated.kayako.com/article/44-subset-step-size-and-strain-filter-selection)，不表示算法或实验精度与其产品等同。

算法背景参见 [Ncorr 算法说明](https://ncorr.com/index.php/dic-algorithms)，实验布置、相关与应变后处理的实践范围参见 [iDICs 指南](https://idics.org/guide/)。本系统仍为平面二维测量，不补偿出平面运动；工程测试的解析误差与实际实验不确定度分别报告。
