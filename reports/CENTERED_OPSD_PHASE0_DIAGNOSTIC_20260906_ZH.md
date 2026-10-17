# Centered OPSD Phase-0 诊断（2026-09-06）

## 数据与口径

使用上一轮 8B OPSD 的 100 个 step summary：2,474 条已评分负 advantage logical rollout、6,739 个 segment、49,979 个 assistant step。权重始终在 logical rollout 内计算，并按有效 loss token 数保持 token-mass 均值为 1。

## 数值诊断

| beta | step 饱和率 | 含饱和 step 的轨迹率 | fallback | 最大 token-mass 误差 |
|---:|---:|---:|---:|---:|
| 0.02 | 0.190% | 3.072% | 0 | 4.17e-7 |
| 0.05 | 3.017% | 24.414% | 0 | 3.58e-7 |
| 0.10 | 13.842% | 61.156% | 0 | 2.98e-7 |
| 0.20 | 31.753% | 96.160% | 0 | 2.98e-7 |

数值上首选 `beta=0.02`；更大的 beta 明显过度触及 `[0.8,1.2]` 边界。所有轨迹均通过 MAD 与最小 step 数检查。

## 行为相关性诊断

旧 summary 未保存 session_id，因此只使用“同 rollout、同 task group、完整 step 集相同且唯一”的严格连接：1,199/2,474 条轨迹、30,431/49,979 个 step；1,275 条歧义轨迹全部排除。

风险信号：在负 advantage 下，下列权重小于 1 会减轻惩罚：

- tool parse failure：均值 0.9803；
- identical tool repeat：均值 0.9820；
- repetition degeneration：均值 0.9908；
- incomplete tool call at limit：均值 0.9909。

这与“退化动作应得到更强负向 credit”的预期相反。关联不等于因果，且旧日志匹配覆盖有限，但该结果已构成计划中的显著停止风险。新 summary 已增加 session_id，后续 smoke 可精确复核。

## 当前决策

暂不直接启动 100-step Centered OPSD。先以 `beta=0.02` 做短 smoke，额外检查：

1. 重复 teacher scoring 的 token logprob MAE、step Pearson/Spearman；
2. Centered OPSD 相对标准 GRPO 的冻结 batch 梯度 cosine 与长 step 梯度占比；
3. 精确 session-step 映射下的退化动作权重方向。

只有这些诊断排除反向 credit 风险，才满足正式训练 Go 条件。

原始结果：

- `reports/centered_opsd_phase0_weights_20260906.json`
- `reports/centered_opsd_phase0_behavior_beta002_20260906.json`
