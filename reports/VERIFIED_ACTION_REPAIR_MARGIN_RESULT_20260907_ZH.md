# Verified Action Repair：最小可行性实验结果

2026-09-07｜8B SFT｜ClawGym training pool｜按 v2 方案执行

**结论：当前修复数据生成方案 NO-GO，不进入训练。** 固定 64 个候选最终只得到 1 对合法、双 seed 验证通过的偏好数据，覆盖 1 个任务，未达到预定的 16 对／8 题门槛。没有执行 NLL、DPO 或 DPO+margin 更新，因此本轮不能判断这些 loss 的学习收益。

## 方法与结果

冻结初始 8B SFT 作为 Student／Teacher。Teacher 使用真实状态、原动作反馈和已有 Evidence Anchors，只提议一次完整替代动作。原／修复分支从同一真实快照执行，各最多续跑 3 turn、300 秒；同预算调用原 verifier。首次严格改善且无 criterion 退步后，换第二组固定 seed 复验。16 题按任务预先划分 12 train／4 held-out，未使用正式 200 题 eval。

| 检查项 | 结果 |
|---|---:|
| 固定候选／有效快照 | 64／64 |
| Teacher 请求／格式与工具范围检查通过 | 64／64 |
| 首次配对验证通过 | 4／64，6.25% |
| 第二组 seed 仍通过 | 2／4 |
| 信息边界审核后可训练 | **1／64，1.56%** |
| 实际分支恢复成功 | 136／136 |
| 原动作关键工具结果重放一致 | 68／68 |
| 完整动作 token 审计 | 128 个动作，80,237 tokens，全部对齐 |

最终拒绝原因：46 个无严格改善、16 个 criterion 退步、1 个修复文本虚构当前完成状态。基础设施／恢复失败及 300 秒超时均为 0；11／136 个分支触及续跑 turn 上限。以上 reward 是短窗口执行结果，不是完整任务成功率。

![数据准入与首次配对结果](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/verification_outcomes.png)

## 为什么没有形成足够修复数据

- **文字替代了操作。** 23／64 个修复没有工具调用，其中 15 对分数下降；16 个退步案例中有 15 个属于这类。Teacher 提示已明确要求从动作前状态重新执行，仍未稳定遵守。
- **真实局部改善存在，但覆盖低。** `task_1924-1` 把文件中泛化的“Mode 1/2”改为要求的模式名称，两组 seed 得分为 `0.625→0.875`、`0.625→0.750`，是唯一接受对。`task_0921-2` 首次 `0.8333→0.8889`，复验回到持平，未接受。
- **不能主要归因于上下文过长。** 动作前上下文中位数 13,291、最大 33,872 tokens；Teacher 输出中位数 409.5、最大 2,705 tokens。所有正式提议都通过格式检查，执行最长 152.3 秒。3-turn 窗口可能遗漏延迟收益，但本轮没有证据证明放宽窗口就能达到数据门槛。

## Margin 是否值得保留

64 个修复目标中，top-k/top-p 过滤集合之外的位置为 **1,924／35,410，5.43%**；其中 1,674 个（87.0%）在分析文字。唯一接受动作的 **64 个阻断位置全部属于分析文字，edit 调用内为 0**。

因此，当前证据不足以把 all-blocked margin 作为有效算法组件推进；保留实现和诊断，不启动正式训练。它是否优于 DPO 尚未测得，不能写成“margin 已证实无效”。分桶是语法诊断，不是因果贡献标注，也没有据此屏蔽 loss。

![修复目标的过滤集合外 token 分布](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/repair_token_blocking.png)

## 发现的采样实现问题

已安装 SGLang 的 seeded Gumbel 路径允许均匀数取到 `1.0`，导致 `Inf + (-Inf) = NaN`，进而选中 top-k/top-p 已屏蔽的 token。真实 seed／位置的 CPU 与 CUDA 最小复现均确认该问题；开区间映射修正通过同一测试，但**没有修改共享安装环境或重写本批结果**。

对全部 867 个请求、449,738 个生成位置扫描后，定位到 12 个端点：源 rollout 9 个、Teacher 1 个、续跑 2 个，涉及 12 个候选。前 10 个有实际 token／logprob 证据，logprob 约 −31.6 至 −47.4；两个终止续跑请求只有精确 seed／长度，缺少原始 token ID，不补造。排除这些候选的敏感性检查仍只有 1／52 对可用，不改变门槛结论；不代表该 bug 没有训练风险。上图“阻断”指过滤后概率为零，不能在存在该 bug 时声称实际实现绝不会输出它。后续新实验应先接入隔离的安全采样修正并复测。

## 成本与交付

正式采集墙钟 **55 分 54 秒**；两路配对进程分别 **61 分 46 秒、64 分 6 秒**，并行执行。正式采集启动至两路收尾约 **2 小时 1 分**。Teacher 共生成 35,410 tokens，累计请求时间 521 秒；136 次分支累计总耗时 5,314 秒；128 次原生完整动作 scoring 累计 1,343 秒。共 200 次验证调用（原 reward wrapper＋criterion 提取），累计 910 秒。上述子项有包含关系，不能相加当墙钟。

工程 smoke／失败尝试另计：6 个有完整起止记录的阶段共 2,838 秒单 GPU 槽位墙钟；端点 CUDA 诊断包装器另计 22 秒。一次未进入模型阶段的早期烤机交接失败缺少精确起止记录，保留日志，不虚报精确总成本。烤机不计实验模型计算时间。

5 状态真实 backward 检查通过，34 项 CPU 测试通过；**optimizer update 为 0**。数据门槛未通过，按方案不开展学习后的 held-out 自主复评及 GRPO 集成 smoke。所有原始结果保留，八卡烤机已恢复。

- [配置、复现命令与证据索引](../outputs/verified-repair-v2-20260907-pilot-a3/RESULTS_README_ZH.md)
- [完整最终统计与成本](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/final_summary.json)
- [64 候选逐例数据](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/candidate_metrics.csv)
- [动作行为对比](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/candidate_behavior.csv)
- [逐 token 原生评分与阻断位置](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/native_token_buckets.csv)
- [采样端点扫描](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/sampler_endpoint_scan.json)
