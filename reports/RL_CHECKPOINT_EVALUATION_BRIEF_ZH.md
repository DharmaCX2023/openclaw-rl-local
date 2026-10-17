# OpenClaw-RL Checkpoint 评测简报

> 数据截至：2026-08-17  
> 模型：Qwen3-30B-A3B，ClawGym SFT 起训  
> 评测集：ClawGym Eval 200 题；每题 `n=1`；temperature `0.7`；单轮 8,192 tokens；上下文 65,536 tokens；最多 96 agent steps

## 一、结论

1. 本轮 RL 训练过程稳定，没有出现此前的持续 reward 崩塌、NaN/Inf、OOM 或无法恢复的训练故障。
2. 训练内在线 eval 从 iteration 19 的 **55.21%** 上升到 iteration 99 的 **57.63%**，correct rate 从 **59.0%** 上升到 **63.0%**，是模型可能变好的主要正向信号。
3. 但独立三次复评未复现这一上涨：五个 RL checkpoint 的平均 reward 均在 **54.05%–54.94%**，iteration 99 为 **54.64%**，与 SFT 完整基线 **55.00%** 基本相当。
4. 因此目前可以确认的是：**RL 没有发生宏观训练崩塌，策略相对 SFT 的整体漂移较小；但尚无充分证据证明最终模型稳定优于 SFT。**

## 二、Checkpoint 独立复评

主指标是 200 题 code verifier 的平均 reward；`correct rate` 仅指 `reward > 0.5` 的题目比例，不是论文主指标。

| checkpoint | 三次独立评测 reward | 均值 ± 标准差 | correct rate 均值 | 相对 SFT |
|---|---|---:|---:|---:|
| iteration 19 | 55.86 / 54.57 / 53.72 | **54.72 ± 1.08%** | 58.0% | -0.29 pp |
| iteration 39 | 55.88 / 54.11 / 54.81 | **54.93 ± 0.89%** | 61.2% | -0.07 pp |
| iteration 59 | 53.29 / 54.08 / 54.77 | **54.05 ± 0.74%** | 60.0% | -0.95 pp |
| iteration 79 | 56.23 / 53.30 / 55.30 | **54.94 ± 1.50%** | 60.0% | -0.06 pp |
| iteration 99 | 55.22 / 55.31 / 53.38 | **54.64 ± 1.09%** | 58.0% | -0.36 pp |
| SFT baseline | 55.00（一次完整评测） | **55.00%** | 57.5% | — |

独立复评中少数运行漏收 1–2 题，主分数会将漏收题按 0 计入。排除漏收题后的敏感性计算仍不改变结论：iteration 19/39/59/79/99 分别为 **54.90% / 55.30% / 54.41% / 55.04% / 54.82%**。

iteration 19 到 iteration 99 的独立均值变化仅为 **-0.08 个百分点**，明显小于单次评测约 **0.7–1.5 个百分点**的运行间波动，不能认为存在稳定提升。

## 三、训练内在线 Eval

训练过程中每 20 step 进行一次完整 200/200 eval：

| checkpoint | 在线 avg reward | 在线 correct rate |
|---|---:|---:|
| iteration 19 | 55.21% | 59.0% |
| iteration 39 | 54.94% | 57.5% |
| iteration 59 | 56.29% | 61.0% |
| iteration 79 | 57.23% | 61.5% |
| iteration 99 | **57.63%** | **63.0%** |

在线结果在后半程呈上升趋势，但 iteration 59/79/99 的在线分数分别比独立三次均值高 **2.24 / 2.29 / 2.99 个百分点**。由于独立复评没有复现，应将其视为“可能改善”的证据，而不是最终能力结论。差异可能来自 `n=1, temperature=0.7` 的采样方差、随机状态、SGLang 调度以及在线与离线运行时环境差异。

## 四、是否还有模型变好的证据

### 1. 有一定正向行为信号

对比训练前 50 个 rollout 与后 50 个 rollout：

| 指标（每轮 64 条轨迹） | 前 50 轮 | 后 50 轮 | 变化 |
|---|---:|---:|---:|
| 重复退化轨迹 | 6.18 | 4.48 | **下降 27.5%** |
| 命中单轮 response 上限 | 4.92 | 4.14 | **下降 15.9%** |
| 上限处未闭合 tool call | 3.86 | 3.22 | **下降 16.6%** |
| 自然结束轨迹 | 55.82 | 56.48 | 小幅增加 0.66 |
| tool call 解析保留率 | 99.55% | 99.63% | 小幅改善 |

这些指标说明后半程出现更少的重复输出和未闭合工具调用，是有价值的行为层正向信号。不过它们会受到任务批次难度和轨迹长度影响，不能单独证明通用能力提升。

### 2. 训练 reward 没有整体上涨

- 每条逻辑轨迹等权的 `avg_reward_submitted`：前 50 轮 **67.61%**，后 50 轮 **67.48%**，基本持平。
- 按切分训练 sample 加权的 `raw_reward`：前 50 轮 **58.88%**，后 50 轮 **56.73%**；该指标受长轨迹切分数量影响，不宜直接作为模型能力指标。
- 后半程平均 agent steps 从 17.63 增至 19.51，任务耗时也增加，因此“模型更愿意继续尝试”与“执行效率改善”不能画等号。

### 3. 模型确实发生了受控变化，但这不是性能证据

`train/kl_loss` 的阶段均值从约 **0.00102** 缓慢增加到 **0.00140**，说明 RL 策略逐步偏离初始 SFT reference；PPO clip fraction 长期约 **0.3%**，即约 99.7% 的有效 token 没有触发 PPO clipping。它们说明训练更新存在且整体策略漂移较小，但不能说明变化方向一定更优。

## 五、重要限制

- `n=1, temperature=0.7` 的 200 题评测方差较大，3 次运行仍不足以确认小于 1 个百分点的差异。
- 严格零更新一致性测试中，即使启用 MoE routing replay，同 checkpoint、同 batch 的两次完整 backward 梯度 cosine 仅为 **0.712**，梯度范数比为 **1.218**。这说明训练侧仍存在尚未完全定位的 MoE/重计算/并行数值非确定性，会降低对 RL 更新效果的归因置信度；它不等同于 checkpoint 已失效，但应在下一轮正式训练前继续解决。
- 现有独立复评聚合文件中的 `valid runs = 0/3` 是旧聚合脚本未解析新增字段造成的显示错误；本文数值已直接从每次最终 eval 日志重新计算。

## 六、汇报口径建议

建议对外表述为：

> 本轮完成了 OpenClaw-RL 训练链路的关键一致性修复，并稳定训练至 iteration 99。训练内在线评测和部分策略行为指标显示正向趋势，例如在线 reward 最高达到 57.63%，重复退化与未闭合工具调用分别下降约 28% 和 17%。但多 checkpoint 独立复评均值仍约为 54%–55%，与 SFT baseline 55.00% 基本持平。因此当前结果证明了训练稳定性和潜在行为改善，尚不足以宣称 RL 带来稳定、显著的总体能力提升。

后续若要得出能力提升结论，应使用相同最终 Harness，对 SFT、iteration 39/79/99 进行无漏收的多次评测，保存逐题 reward，并进行 paired comparison 与置信区间分析。

## 七、结果位置

- RL 三次复评聚合：`/root/openclaw-rl-local/outputs/clawgym-rl-eval-steps20-100-3runs-docker8-parallel8-20260813-012344/aggregate_results.md`
- SFT 完整基线：`/root/openclaw-rl-local/outputs/clawgym-sft-baseline-eval-same-budget-20260812-004700/logs/eval_20260812-004700.log`
- 统一评测指南：`/root/openclaw-rl-local/reports/RL_CHECKPOINT_EVALUATION_GUIDE_ZH.md`
- 零更新梯度一致性报告：`/root/openclaw-rl-local/outputs/zero-backward-consistency-step99-20260814/ZERO_BACKWARD_CONSISTENCY_REPORT_ZH.md`
