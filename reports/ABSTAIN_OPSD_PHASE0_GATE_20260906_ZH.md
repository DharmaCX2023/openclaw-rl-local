# Abstain-OPSD 离线门控报告

日期：2026-09-06

## 结论

**No-Go：当前版本不进入在线 smoke 或 100-step 训练。**

Abstain 规则确实减少了不稳定信用，但剩余信号仍不能可靠区分有价值动作与坏动作。直接训练更可能引入随机重加权，而不是改善 credit assignment。

## 实验设计

- 复用 Matched-Placebo Phase-0 的 399 个 turn、30 条负 advantage 逻辑轨迹；
- 每个 turn 已有 1 个 Real Anchor 与 3 个 matched placebo 的确定性 teacher-forcing logprob；
- 不重新 rollout、不更新模型、不占用 GPU；
- 主口径预先固定 `K=3`、`noise_floor=0.002`、`lambda_MAD=1`；
- 三个 `delta` 必须同号，否则 abstain；
- 信用定义：`c=sign(median(delta))*max(abs(median(delta))-MAD(delta)-noise_floor, 0)`；
- `c=0` 时完全回退标准 GRPO；推断区间按 logical trajectory 聚类 bootstrap。

动作类别只是现有日志的规则诊断标签，不视为完美 step-level ground truth。

## 主结果

| 指标 | 结果 |
|---|---:|
| 总 turn | 399 |
| 生效 turn | 114 |
| 覆盖率 | 28.57% |
| 正信用 / 负信用 | 45 / 69 |
| 可归类的生效 turn | 69 |
| 方向正确率 | **44.93%** |
| good-minus-bad credit | 0.000267 |
| 轨迹聚类 95% CI | **[-0.000653, 0.002775]** |
| 平均 `abs(weight-1)` | 0.01204 |

方向正确口径：good 类动作应获得正信用、减轻负 advantage；bad 类动作应获得负信用、加重惩罚。结果低于 50%，且 good-minus-bad 区间跨 0。

主要反例：

- `train-task-0-9:2` 是 `tool_error`，三个 delta 均为正，算法会错误保护它；
- `train-task-0-18:1` 是 `successful_read`，三个 delta 均为负，算法会错误加重惩罚；
- 82 个 `degenerate_repeat` 中仅 14 个通过门控，其中 9 个仍被保护、5 个被加重惩罚。

## 敏感性

| noise floor | 覆盖率 | 方向正确率 | good-minus-bad 95% CI |
|---:|---:|---:|---:|
| 0 | 52.63% | 51.18% | [-0.000882, 0.003542] |
| 0.001 | 38.85% | 47.25% | [-0.000764, 0.003093] |
| **0.002** | **28.57%** | **44.93%** | **[-0.000653, 0.002775]** |
| 0.005 | 16.04% | 54.05% | [-0.000355, 0.002076] |

提高阈值只会降低覆盖率，没有形成统计可靠的方向分离。

K=1 的平均覆盖率为 57.39%，不同 placebo 间生效符号一致率约 82.2%–83.4%；K=2 覆盖率为 33.42%；K=3 为 28.57%。因此多 placebo 能筛掉一部分不稳定 turn，但不能修复底层 Real-vs-Placebo 信号的语义失真。

## 决策

1. 不启动 Abstain-OPSD 在线训练；
2. 不根据事后表现挑选更高阈值；
3. 保留脚本和机器可读结果，作为该假设的负结果；
4. 若继续该方向，必须先获得独立的、可审计的 step-level ground truth，证明信用排序有效，再讨论权重函数。

产物：

- `scripts/analyze_abstain_opsd_phase0.py`
- `outputs/matched-placebo-phase0-20260906/abstain_opsd_phase0.json`
