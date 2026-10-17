# Step99 logprob 抖动的概率分桶分析

日期：2026-08-20

## 结论

抖动不是均匀发生在所有 token 上，也不是主要发生在高概率 token 上。它高度集中于模型本来就不确定的中低概率 token。

在 84,432 个真实 `loss_mask=1` token 上：

- 约 87% token 的参考概率不低于 0.99；这些 token 几乎不抖。
- 概率最低的 10% token 承担约 97% 的绝对 logprob 误差和超过 99.8% 的平方误差。
- SGLang–Megatron 训推比较和 Megatron 重复 forward 都有相同的概率依赖结构。
- Megatron routing replay 将各个非平凡概率桶的平均绝对误差降低约 50%–58%，但不会消除低概率 token 的剩余长尾。
- 重复 forward 的带符号均值远小于绝对误差，主要表现为近似双向随机抖动；SGLang–Megatron 还叠加了小幅系统偏差，Megatron 对低概率 sampled token 整体略低估。

因此，更准确的表述是：

> 几乎每个非确定性概率区间都可能发生 logprob 抖动，但抖动强度随 token 概率降低显著增大；高置信 token 基本稳定，PPO 风险由少量中低概率 sampled token 主导。

### 概率桶、token 累计占比与抖动

| token 几何中心概率 | token 数 | 单桶占比 | 从高概率向低概率累计 | 训推 MAE | 重复 forward MAE，无 replay | 重复 forward MAE，有 replay |
|---|---:|---:|---:|---:|---:|---:|
| p ≥ 0.99 | 73,720 | 87.3129% | 87.3129% | 0.0000745 | 0.0000519 | 0.0000194 |
| 0.90 ≤ p < 0.99 | 3,626 | 4.2946% | 91.6074% | 0.01062 | 0.00790 | 0.00358 |
| 0.50 ≤ p < 0.90 | 4,310 | 5.1047% | 96.7121% | 0.05611 | 0.04134 | 0.02014 |
| 0.10 ≤ p < 0.50 | 2,603 | 3.0830% | 99.7951% | 0.14065 | 0.10194 | 0.05125 |
| 0.01 ≤ p < 0.10 | 173 | 0.2049% | 100.0000% | 0.24289 | 0.16311 | 0.06860 |

token 数和累计占比采用 SGLang–Megatron 两遍 logprob 的几何中心概率分桶。重复 forward 使用六遍平均 logprob，桶边界附近有少量 token 跨桶，因此其逐桶 token 数略有差异，但不影响误差随概率降低而增大的结论。

## 数据与方法

- checkpoint：正式 RL step99
- 固定数据：零更新一致性实验同一真实 rollout batch
- 有效策略 token：84,432
- sample：6
- SGLang–Megatron：比较 rollout 时保存的 logprob 与 Megatron 更新前 old logprob
- Megatron 重复 forward：相同权重、相同 token、相同 mask，初始 old forward 加 5 次重复 forward
- A/B：无 routing replay 与启用 routing replay
- 没有重新 tokenize、重新生成或执行 optimizer update

为避免“按第一次 forward 分桶”引入回归均值偏差，主结果采用对称方法：

1. 对每个 token，使用所有被比较 forward 的平均 logprob 作为分桶依据；
2. SGLang–Megatron 比较两遍的一个差值；
3. Megatron 六遍 forward 计算全部 15 个两两差值；
4. 概率定义为平均 logprob 的指数；
5. 所有统计只包含 `loss_mask=1` token。

## SGLang–Megatron 训推差异

| 平均概率桶 | token 数 | MAE(logprob) | P99 abs diff | ratio 误差 >1% | 绝对误差占比 | 平方误差占比 |
|---|---:|---:|---:|---:|---:|---:|
| p ≥ 0.99 | 73,720 | 0.0000745 | 0.00199 | 0.026% | 0.79% | 0.01% |
| 0.90 ≤ p < 0.99 | 3,626 | 0.01062 | 0.05766 | 35.47% | 5.55% | 0.59% |
| 0.50 ≤ p < 0.90 | 4,310 | 0.05611 | 0.29020 | 83.94% | 34.85% | 18.16% |
| 0.10 ≤ p < 0.50 | 2,603 | 0.14065 | 0.61551 | 94.05% | 52.76% | 59.88% |
| 0.01 ≤ p < 0.10 | 173 | 0.24289 | 1.26901 | 94.80% | 6.06% | 21.36% |

最低概率等频十分位（约 8,443 个 token，概率约 0.0155–0.9510）承担：

- 96.98% 的绝对误差；
- 99.85% 的平方误差；
- 83.04% 的比较满足 `|ratio-1|>1%`。

NLL 与每 token 平均绝对 logprob 抖动的 Pearson 相关系数为 0.671，说明 token 越低概率，抖动总体越大。

带符号差值定义为 `logp_megatron-logp_sglang`。全体均值为 -0.00173；在 0.01–0.10 桶中为 -0.0926，而该桶 MAE 为 0.2429。因此训推差异以双向变化为主，同时存在 Megatron 对低概率 sampled token 略低估的系统成分。

## Megatron 重复 forward：无 routing replay

| 平均概率桶 | token 数 | MAE(logprob) | P99 abs diff | ratio 误差 >1% | 绝对误差占比 | 平方误差占比 |
|---|---:|---:|---:|---:|---:|---:|
| p ≥ 0.99 | 73,632 | 0.0000519 | 0.00129 | 0.015% | 0.75% | 0.01% |
| 0.90 ≤ p < 0.99 | 3,694 | 0.00790 | 0.04804 | 25.87% | 5.72% | 0.78% |
| 0.50 ≤ p < 0.90 | 4,306 | 0.04134 | 0.21716 | 73.98% | 34.93% | 18.83% |
| 0.10 ≤ p < 0.50 | 2,613 | 0.10194 | 0.48724 | 84.28% | 52.27% | 61.19% |
| 0.01 ≤ p < 0.10 | 186 | 0.16311 | 1.01597 | 92.80% | 5.95% | 14.07% |
| 0.001 ≤ p < 0.01 | 1 | 1.90372 | 3.83606 | 100% | 0.37% | 5.11% |

最后一桶只有一个 token，不能单独据此估计总体频率，但它证明极低概率 sampled token 可以产生极端长尾。

最低概率等频十分位承担 96.89% 的绝对误差和 99.80% 的平方误差。NLL 与抖动的 Pearson 相关系数为 0.786。

全体两两差值的带符号均值为 -0.000059，而 MAE 为约 0.006，说明重复 forward 主要是双向随机抖动，不是单向漂移。

## Megatron 重复 forward：启用 routing replay

| 平均概率桶 | token 数 | MAE(logprob) | P99 abs diff | ratio 误差 >1% | 绝对误差占比 | 平方误差占比 |
|---|---:|---:|---:|---:|---:|---:|
| p ≥ 0.99 | 73,621 | 0.0000194 | 0.00056 | 0.001% | 0.57% | 0.01% |
| 0.90 ≤ p < 0.99 | 3,701 | 0.00358 | 0.02254 | 10.34% | 5.32% | 0.63% |
| 0.50 ≤ p < 0.90 | 4,301 | 0.02014 | 0.11067 | 51.06% | 34.84% | 19.94% |
| 0.10 ≤ p < 0.50 | 2,614 | 0.05125 | 0.21320 | 64.38% | 53.88% | 71.43% |
| 0.01 ≤ p < 0.10 | 195 | 0.06860 | 0.25930 | 76.07% | 5.38% | 8.01% |

相对无 replay，各概率桶 MAE 降幅为：

- p ≥ 0.99：62.7%
- 0.90–0.99：54.7%
- 0.50–0.90：51.3%
- 0.10–0.50：49.7%
- 0.01–0.10：57.9%

routing replay 的作用并不只发生在某个概率桶；它大约把全范围抖动减半。但最低概率十分位仍承担 97.32% 的绝对误差和 99.86% 的平方误差。

全体带符号差值均值为 -0.0000135，进一步支持“重复 forward 剩余误差主要是双向抖动”。

## 为什么低概率 token 更敏感

对一个 token，有近似关系：

```text
delta(log p_i) = delta(z_i) - sum_j p_j * delta(z_j)
```

小的 router、expert GEMM、collective 或 softmax 数值扰动会改变竞争 token 的相对 logits。高置信 token 通常具有很大的 logit margin，top-1 选择和条件概率都很稳定；中低概率 sampled token 常位于多个候选接近的决策点，同样大小的 logit 扰动更容易表现为明显的 logprob 差异。

PPO 使用：

```text
ratio = exp(delta logprob)
```

因此 `delta logprob=0.1` 大约对应 10.5% 的概率比变化，不会因为原始 token 概率低而自动消失。只要该 token 被实际采样并且 `loss_mask=1`，它就是需要训练的策略动作。

## 对正式 RL 的含义

1. 仅报告全 token 平均 `train/rollout_logprob_abs_diff` 会被 87% 近确定 token 稀释，不能代表真正风险。
2. 应增加按 rollout probability 分桶的 MAE、P95/P99、`|ratio-1|` 阈值比例和误差质量占比。
3. routing replay 明显有效，但剩余误差依然集中在中低概率 token；它不是完整的训推一致性解决方案。
4. ClawGym II 采用 token-level rollout importance correction，正好针对 SGLang rollout 与训练 old policy 的剩余差异。
5. 后续若比较训练稳定性，应进一步按 `|advantage|` 加权并统计这些低概率 token 对实际梯度的贡献，而不是只看 token 数量。

## 产物

- 原始分桶 JSON：`reports/logprob_jitter_probability_buckets_step99_20260820.json`
- 可复现脚本：`scripts/analyze_logprob_jitter_by_probability.py`
- 输入数据：`outputs/zero-update-consistency-step99-20260814/`

本次现有逐 token 数据已经覆盖所需比较，因此没有重新占用 GPU。分析期间 GPU 0 补启烤机，GPU 1–7 保持原有烤机；最终恢复为 8 卡持续烤机。
