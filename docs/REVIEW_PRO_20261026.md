# Review-Pro 审查与修复记录（2026-10-26）

## Verdict: APPROVE

Spec：对照用户在本会话给出的 StrainTrace 原始功能升级要求。已确认的问题均已修复；未发现独立的 Missing、Wrong 或 Scope creep 项。

原始审查固定比较 `5f812ef9a0a0f7ce2391400a889c25ad15dae153...e38f35286bc92208082e20cb10b2870e35c00516`，共 31 个文件。代码修复作为后续差异单独核验。仓库没有安装 `.review-pro/` stack，使用各领域核心规则；commit message 没有需要分配的外部前提。

分批执行了 correctness、backend、api-contract、security、performance、frontend、tests、spec、a11y、craft 十个领域审查，每批至多三个审查代理。计算核心、规格和模块结构没有新增待处理问题。下面列出已关闭的 4 个 High、8 个 Medium。

## 已修复问题

| 原级别 | 问题 | 修复及证据 |
| --- | --- | --- |
| High | 自动流程可能将运行 A 的结果与运行 B 的报告组合为完成 | 报告在锁内检查预期运行；最终核验、目标检查和完成记录在同一锁内再次检查运行身份。两个阶段插入另一次运行均明确拒绝混合交付 |
| High | GUI 显示新项目目录却写入原项目 | 写入前核对显示目录与已打开项目身份，不一致时要求先打开或新建 |
| High | 清空参考帧后可在旧图或未经纹理检查的首帧上画坐标 | 参考配置变化暂停编辑；预览复用实际计划的自动参考选择与已载入图像。首帧全零、后续有纹理的保存、绘制和计划路径始终使用同一帧 |
| High | 短窗口键盘焦点进入屏外控件后不可见 | FocusIn / TraverseIn 按控件实际位置滚动，保留正反向遍历并添加键盘翻页 |
| Medium | 正常取消自动分析被转换成报告失败 | 返回 cancelled 和已保存的部分报告，恢复运行仍使用原检查点身份 |
| Medium | 报告替换与外层清单更新之间失败后无法重试 | 候选报告与旧包备份支持回滚；仅报告派生产物的哈希中断可重建恢复，源 CSV、输入或检查点损坏仍拒绝 |
| Medium | 包级 report 函数被同名子模块覆盖 | 实现模块改名 reporting，保留包级 report 函数；独立进程检查导入顺序 |
| Medium | 未登记的检查点外链被带入报告包 | 只处理源 receipt 登记且有预期摘要的文件，拒绝 symlink、junction 和 reparse 路径；独立校验核对源 receipt |
| Medium | 每次报告校验重复完整读取 NPZ | 仅在本次调用内复用实际计算的摘要，继续核对包清单、记录和源 receipt。64 MiB 例中完整读取量由 128 MiB 降为 64 MiB，不声称固定耗时倍数 |
| Medium | 核验期间选择另一 ROI 后可能修改错行 | 忙时还原到当前编辑对象，保持高亮与编辑器一致 |
| Medium | 放弃未应用的几何编辑后运行按钮不恢复 | 新建或切换项目条目后重新计算按钮状态 |
| Medium | stress_strain 验收测试被坏图帧的失败掩盖 | 增加全部 DIC 与完整性均通过、载荷时间完全无支持的独立算例，确认目标状态仍为 attention_required |

原问题的审查代理分别复核了 backend、API、security、performance、frontend 和 a11y 修复。参考帧问题在第一次补修后又覆盖了首帧无纹理的具体分支。对应回归位于 [工作流测试](../tests/agent/test_review_workflow.py)、[GUI 测试](../tests/agent/test_review_gui.py) 和 [报告测试](../tests/agent/test_upgrade_report.py)。

## 验证

- 整套回归：553 passed、1 skipped，339.55 s。跳过项由旧 jsonschema 缺少 Draft 2020-12 支持导致，已在独立依赖目录中精确补测通过。
- 最后一次参考图预览修正后，整个 Agent 域：127 passed，89.10 s；新增 GUI 回归单独为 12 passed。
- 最终真实 CLI：14 个命令符合预期，包含正常状态、部分失败、断点复用、检查点损坏恢复、配置更新、局部重处理、报告搬移与篡改拒绝。输入 SHA-256 保持不变。
- 锁定数值基准 v6/v4 通过，三项 clean 指标与现有 CI 基线一致，没有修改科学门限或数值容差。
- 安装后的 wheel 与 Windows 便携版均完成公开图像自动分析和完整性核验；与源码共比较 84 个数值/有效性数组，最大绝对差为 0。冻结 GUI smoke 与标准库独立报告校验通过。

结构化本地证据与最终源码摘要见 [review_pro_20261026.json](validation/review_pro_20261026.json)。远端 CI 按推送后的实际运行状态另行报告。

这些检查是软件回归，不构成实验不确定度标定。未执行屏幕阅读器验收；键盘修复由隐藏的真实 Tk 控件、正反向遍历及视口几何检查验证。

## 依据和交付

Python 子模块绑定行为核对了 [Python 3.13 导入系统说明](https://docs.python.org/3.13/reference/import.html#submodules)。键盘焦点检查参考 [WCAG2ICT 2025-12-11](https://www.w3.org/TR/2025/NOTE-wcag2ict-22-20251211/) 对非 Web 软件的指导，未将其误称为实验或产品认证。

原 `StrainTrace-agent-upgrade-20261026-Windows.zip` 保留为升级初稿。修复后的便携交付使用 `StrainTrace-reviewed-20261026-Windows.zip`，由其中 `BUILD_INFO.json` 的源码提交和 SHA-256 区分。原始数据及既有分析输出保留；Git 收尾将全部当前开发分支合入 main 后推送，具体远端提交与 CI 结果在本次最终报告中给出。
