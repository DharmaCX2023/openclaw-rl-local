# Base 与 SFT 模型 ClawGym 确定性评测报告

## 1. 结论摘要

在完全相同的 200 题 ClawGym Agent 评测配置下，SFT 模型相对原始 Base 模型取得了明确提升：

- 平均 verifier reward 从 **0.41477** 提升到 **0.53455**，绝对提升 **0.11978（11.98 个百分点）**；
- Correct rate 从 **41.5%（83/200）** 提升到 **58.5%（117/200）**，增加 **34 题（17.0 个百分点）**；
- 零奖励任务从 **32 题**下降到 **9 题**；
- 逐题对比中，SFT 相对 Base 为 **108 胜、47 平、45 负**；
- 配对平均 reward 提升的 bootstrap 95% 置信区间为 **[0.0810, 0.1599]**；
- Correct 状态转移的 McNemar exact test 为 **p = 5.85e-5**。

因此，本轮结果支持：**ClawGym SFT 显著改善了原始 Qwen3-30B-A3B-Instruct-2507 在同分布 Agent 任务上的执行能力。**

## 2. 被评测模型

### Base 模型

```text
/mnt/OpenClaw-RL/models/Qwen3-30B-A3B-Instruct-2507
```

这是未经本项目 SFT 或 RL 的原始指令模型。

### SFT 模型

```text
/mnt/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127
```

这是完成 ClawGym 监督微调后的模型，尚未进行本轮 RL。

两个模型的 tokenizer、模型结构配置和 `generation_config.json` 一致；评测时仅替换模型权重路径。

## 3. 评测配置

两次评测严格复用同一套任务、采样、Agent 预算和推理拓扑。

| 配置项 | 数值 |
|---|---:|
| 数据集 | 本地 ClawGym eval 保留集 |
| 任务数 | 200 |
| 每题采样数 | n=1 |
| Agent 轨迹并发 | 32 |
| temperature | 0.7 |
| top-p | 0.8 |
| top-k | 20 |
| repetition penalty | 1.0 |
| 单轮最大 response | 8,192 tokens |
| live context window | 65,536 tokens |
| trajectory token budget | 65,536 tokens |
| max agent steps | 96 |
| max compaction events | 8 |
| 单任务外层超时 | 1,200 秒 |
| SGLang 拓扑 | TP2 × DP2，共 4 GPU |
| CUDA graph | 关闭 |
| SGLang deterministic inference | 开启 |
| sampling seed base | 424242 |
| engine seed | 12345 |
| rollout ID | 0 |
| deterministic tool-call ID | 开启 |

SGLang 使用两个 TP2 推理副本服务最多 32 条同时在飞的 Agent 轨迹。轨迹执行工具时会暂时离开模型队列，因此实际模型 forward batch 由 SGLang 动态形成，并不固定为 32。

评测期间 GPU 0–3 继续运行独立烤机，评测使用 GPU 4–7；两轮评测结束后均恢复为八卡烤机。

## 4. 计分口径

每个 ClawGym 任务使用其自带的 code-based verifier 返回区间为 `[0, 1]` 的 reward。

- **平均 reward**：200 个任务 reward 的算术平均；
- **Correct**：任务 reward 严格大于 0.5；
- 本轮没有使用外部 rubric judge 或 GPT judge；
- 环境错误导致的 skipped 任务按运行时单独记录，不与模型失败混淆。

## 5. 核心结果

| 指标 | Base | SFT | SFT 相对变化 |
|---|---:|---:|---:|
| 任务回收 | 200/200 | 200/200 | 相同 |
| skipped | 0 | 0 | 相同 |
| 平均 reward | 0.41477 | **0.53455** | **+0.11978** |
| 中位 reward | 0.44444 | **0.55903** | +0.11458 |
| Correct | 83/200 | **117/200** | **+34 题** |
| Correct rate | 41.5% | **58.5%** | **+17.0 pp** |
| 零奖励任务 | 32 | **9** | **-23 题** |
| 总模型请求 | 2,518 | 2,214 | -12.1% |
| 总墙钟时间 | 3,137.25 秒 | 3,839.99 秒 | +22.4% |
| 轨迹吞吐 | 3.825/min | 3.125/min | -18.3% |

Base 中有 6 条轨迹命中 `max_agent_steps=96`；SFT 日志中没有轨迹命中该限制。两组均没有因为任务超时或环境异常而跳过样本。

## 6. 逐题配对分析

由于两轮评测使用相同的 200 个 task，可以直接计算每个任务的差值：

```text
delta(task) = reward_SFT(task) - reward_Base(task)
```

配对结果为：

| 配对指标 | 结果 |
|---|---:|
| SFT reward 更高 | 108 题 |
| reward 相同 | 47 题 |
| SFT reward 更低 | 45 题 |
| 平均配对差值 | +0.11978 |
| bootstrap 95% CI | [0.08102, 0.15994] |

bootstrap 使用固定随机种子 `20260820`，对 200 个逐题差值进行 20,000 次有放回重采样。

按 Correct 状态统计：

| 状态转移 | 任务数 |
|---|---:|
| Base 不正确 → SFT 正确 | 52 |
| Base 正确 → SFT 不正确 | 18 |

两类不一致样本的 McNemar exact test：

```text
p = 5.85e-5
```

这表明 Correct rate 的提升不只是少量任务或简单采样波动造成的表面差异。

## 7. 效果与吞吐解释

SFT 的平均 reward 和 Correct rate 均明显提高，同时零奖励任务大幅减少，说明提升并非只来自已经较好任务上的小幅加分，而是把相当一部分原本失败的任务转化为部分成功或完整成功。

另一方面，SFT 总模型请求数比 Base 少 12.1%，但墙钟时间反而增加 22.4%。这说明本轮吞吐下降不能简单归因于更多 Agent turn。更可能的因素包括：

- 单次模型 response 更长；
- 单次 prefill/decode 计算量更高；
- 生成了更复杂、耗时更长的工具调用；
- 最后一批任务的工具等待或轨迹长尾更明显。

当前独立 eval 报告没有输出可靠的逐 token/逐 step trajectory audit，因此不能仅凭请求次数进一步区分上述因素。后续若要优化评测吞吐，应补充 response tokens、prefill tokens、decode time、tool time 和 task wall time 的逐轨迹统计。

## 8. 可复现性与局限

### 已控制的随机性

- SGLang deterministic inference 已开启；
- 每个请求的 sampling seed 根据固定 seed base、session ID、请求类型和 Agent step 派生；
- tool-call ID 使用确定性生成，不使用随机 UUID；
- 两轮使用相同任务、rollout ID、采样配置、并发和 GPU 拓扑。

### 仍需注意的局限

1. 每题只有 `n=1`。确定性机制能复现同一运行条件，但不能替代多样本统计。
2. 不同模型在第一轮输出后会进入不同的工具调用和上下文，因此这是同 task 的配对比较，不是逐 token 的同上下文比较。
3. live web、远端命令耗时、后台进程输出等外部工具结果仍可能随时间变化。
4. `eval_report.json` 中 `trajectory_audits=0` 表示该独立 eval 路径没有汇总训练用 trajectory audit，不能解释为“没有发生 compaction”。
5. SFT 清理阶段出现一次 `RemoteProtocolError`：它发生在 200 题全部计分和报告写入之后，是 Proxy/SGLang 关闭时未完成 HTTP 请求的竞态。评测退出码为 0、`passed=true`、`n_skipped=0`，不影响本轮分数。

## 9. 最终判断

在本轮严格对齐的确定性评测中：

> SFT 将平均 verifier reward 从 41.48% 提升至 53.46%，将 Correct rate 从 41.5% 提升至 58.5%。逐题配对提升为 11.98 个百分点，95% bootstrap 置信区间完全高于零，Correct 状态转移也具有显著差异。因此可以确认，当前 ClawGym SFT 对基模的 Agent 任务能力带来了实质性提升。

这份结果可以作为后续 RL checkpoint 的统一基线。后续应继续使用完全相同的确定性 launcher 评测 RL checkpoint，才能严谨量化 SFT → RL 的增益。

## 10. 结果与复现文件

### Base

- 结果：`/root/openclaw-rl-local/outputs/base-model-eval-clawgym200-deterministic-20260820/eval_report.json`
- 日志：`/root/openclaw-rl-local/outputs/base-model-eval-clawgym200-deterministic-20260820.launch.log`
- 启动器：`/root/openclaw-rl-local/.run_base_eval_full_20260820.sh`

### SFT

- 结果：`/root/openclaw-rl-local/outputs/sft-model-eval-clawgym200-deterministic-20260820/eval_report.json`
- 日志：`/root/openclaw-rl-local/outputs/sft-model-eval-clawgym200-deterministic-20260820.launch.log`
- 启动器：`/root/openclaw-rl-local/.run_sft_eval_full_deterministic_20260820.sh`
