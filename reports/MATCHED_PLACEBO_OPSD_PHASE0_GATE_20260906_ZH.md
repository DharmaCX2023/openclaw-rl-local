# Matched-Placebo OPSD Phase-0 门控报告

日期：2026-09-06

## 结论

**No-Go：不进入双组 one-step smoke 和 100-step RL 训练。**

Matched-Placebo 信号在不同 placebo 之间高度敏感，且没有表现出预注册要求的动作质量方向。继续训练会把不可辨识的 anchor 差值放大成信用权重，不符合实验门控条件。该结论不是由 scoring 基础设施失败、上下文溢出或权重裁剪饱和造成。

## 实验口径

- 模型：Qwen3-8B ClawGym SFT `final_model`；
- 数据：已有真实负 advantage 失败轨迹，不生成新 rollout、不更新模型；
- 覆盖：30 条逻辑轨迹、8 个任务、399 个 assistant turn；
- 每个 turn：1 个 Real Anchor + 3 个 matched placebo；
- 总评分：1,596/1,596 个条件成功；
- Real 与三个 Placebo 使用相同 response token、同一 session 路由和相同 SGLang 确定性 scoring 路径；
- 统计单位：turn 描述性统计，推断统计使用 logical-trajectory cluster bootstrap；
- Centered 配置：`beta_c=0.02`，权重裁剪 `[0.8, 1.2]`。

信号定义：

```text
delta = mean_logprob(Real Anchor) - mean_logprob(Placebo Anchor)
```

## 完整性

| 指标 | 结果 |
|---|---:|
| 可重建 turn | 399 |
| 完整分析 turn | 399 |
| 条件评分 | 1,596/1,596 |
| 条件评分失败 | 0 |
| 不完整 turn | 0 |
| conditioned context overflow | 0 |

最初并发诊断曾触发 SGLang request-slot 异常；失败请求未写入结果。加入每 engine 单请求信号量后，从断点重算全部缺失条件，最终报告中的评分失败为 0。

## Placebo 稳定性

| 指标 | 结果 |
|---|---:|
| K=3 符号翻转率 | **47.37%** |
| `max(abs(delta)) >= 0.005` 时翻转率 | **34.66%**（61/176） |
| `max(abs(delta)) >= 0.010` 时翻转率 | **16.90%**（12/71） |
| placebo 间 delta 标准差均值 | 0.002267 |
| placebo 间 delta 标准差中位数 | 0.001701 |
| 三个 placebo-index 的全局 logprob 均值范围 | 0.001008 |
| 三组 delta 的两两 Pearson | 0.7277–0.7629 |

因此翻转并非全部来自接近零的数值噪声。Placebo index 的全局均值没有单个异常离群，但具体 placebo 的语义仍显著改变 turn 级信号。

## 行为方向

| 动作类型 | turns | 平均 delta | Centered 减轻惩罚比例 |
|---|---:|---:|---:|
| successful_read | 38 | -0.001639 | 47.37% |
| modification | 56 | -0.000392 | 44.64% |
| validation | 2 | -0.002879 | 0.00% |
| recovery_after_error | 27 | -0.000742 | 44.44% |
| tool_error | 34 | -0.002637 | 38.24% |
| degenerate_repeat | 82 | -0.000984 | **48.78%** |
| response_truncated | 5 | +0.001969 | **80.00%** |
| final_answer | 21 | -0.011209 | 28.57% |

预注册的 good categories 与 bad categories 的平均 delta 差：

```text
estimate = 0.000432
trajectory-cluster bootstrap 95% CI = [-0.001454, 0.003532]
```

置信区间跨越 0，且有效读取、验证、修改和恢复没有获得稳定的相对高支持。退化重复和截断动作仍频繁被减轻惩罚，未通过信用方向门控。

## 权重与 Absolute beta

| 指标 | Centered | Absolute |
|---|---:|---:|
| 平均权重 | 1.006224 | 1.009320 |
| `mean(abs(w-1))` | 0.029127 | 0.029116 |
| 下边界比例 | 0% | 0% |
| 上边界比例 | 1.00% | 1.00% |

离线强度匹配得到 Absolute 候选 `beta_a=6.902398`，强度误差仅 `1.06e-5`。但因底层 delta 未通过稳定性和行为方向门控，此 beta 仅作为诊断产物保存，**不得据此启动正式训练**。

## 决策
