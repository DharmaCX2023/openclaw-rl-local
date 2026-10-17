# OpenClaw-RL 训推一致性与训练稳定性实验复盘

> 模型：Qwen3-30B-A3B MoE  
> 训练框架：Slime + Megatron-LM  
> 推理与环境：SGLang + OpenClaw/ClawGym Harness  
> 主要诊断 checkpoint：RL step 99  
> 整理日期：2026-08-25

## 1. 一页结论

本轮工作先后解决了两类不同问题，并识别出一项仍需工程权衡的问题。

1. **Agent 训练语义已经修正。** 上下文 pruning/compaction 后不再把新 prompt 错接到旧 token 前缀；一次 assistant 输出的多个 tool call 会被完整执行；压缩总结请求不再被误记为主策略 action。模型实际看到的上下文、SGLang 生成 token 的上下文和 Megatron 计算 loss 的上下文现在语义一致。
2. **Megatron 训练端的重复计算不确定性已经找到有效修复。** 默认 MoE 配置下，同权重、同 batch、零参数更新的两次完整梯度方向严重不一致；启用 Slime/Megatron 官方 deterministic mode 后，完整真实 Agent batch 上两次 old/current logprob 和 305.3 亿个分布式 FP32 主梯度元素均逐元素一致。
3. **SGLang 与 Megatron 的跨后端概率差异仍然存在。** 确定性模式消除了训练端自身抖动，但不能让两个不同后端天然给出相同 token probability。完整 routing replay 可将 logprob 误差和 PPO ratio 长尾降低约一半，TIS 可校正剩余 behavior-policy mismatch；然而当前 per-turn full replay 使训练 context 增加约 9.9 倍、训练侧耗时增加约 6.8 倍，不能直接用于长训。
4. **训练运行稳定性已经显著改善，但 RL 能力增益尚未被独立评测确认。** 正式训练稳定完成至 step 99，没有持续 reward 崩塌、NaN/Inf 或 OOM。在线 eval 从 55.21% 上升到 57.63%，但各 checkpoint 独立复评约为 54%–55%，与确定性 SFT 基线 53.46%/历史近同口径 SFT 基线 55.00%处于相近区间。现有证据支持“训练闭环稳定、行为有局部改善”，还不足以宣称 RL 带来稳定显著的总体提升。

建议下一轮正式训练采用：**正确的 token ledger 与分段语义 + Megatron deterministic mode + TIS 小规模先验验证**。Full routing replay 需先完成 route stitching、紧凑编码和多 step 成本验证，再决定是否开启。

## 2. 先统一术语

| 术语 | 本项目中的准确含义 |
|---|---|
| Rollout/行为策略 | SGLang 实际生成 Agent token、调用工具并获得环境反馈的策略与执行过程。 |
| 训练策略 | Megatron 对已生成 token 做 teacher-forced forward，并据此计算 GRPO/PPO surrogate loss 的模型。 |
| Old policy | 当前 optimizer update 之前用于 PPO 分母的策略。本项目默认由 Megatron 重算 old logprob。 |
| Current policy | 当前 loss forward 中带梯度的策略；optimizer 更新前理论上与 old policy 权重相同。 |
| Reference policy | 冻结的 SFT 模型，用于监控或正则化当前策略相对 SFT 的偏离；它不是 PPO old policy。 |
| Logprob | 实际生成 token 的条件概率的自然对数，即 `log p(token | prefix)`。 |
| PPO ratio | `exp(logp_current - logp_old)`；零更新时理论值为 1。 |
| 训推差异 | 本文特指 SGLang rollout logprob 与 Megatron old logprob 对同一 token、同一前缀的差异。 |
| 重复计算差异 | 同权重、同 batch、无 optimizer update 时，Megatron 两次 forward/backward 的差异。它与跨后端训推差异不是同一问题。 |
| Routing replay | 固定 MoE 每层为 token 选择的 top-k expert ID。训练侧 replay 与 SGLang→Megatron full rollout replay 必须区分。 |
| TIS | Token-level Importance Sampling。用 rollout policy 与训练 old policy 的概率比修正每个 token 的 loss 权重。 |
| Segment | 当历史上下文被 pruning、compaction 或 system prompt 重写后，从真实新 prompt 开始建立的新训练片段；它不是新的逻辑轨迹。 |

## 3. Agent 训练语义：从“能跑”修到“训练的是实际发生的轨迹”

### 3.1 上下文重写与 token ledger

旧实现假设每一轮 prompt 都是上一轮 token 的严格追加，并采用“最长公共前缀后追加 delta token”的方式构造训练序列。但 OpenClaw 会做两类历史重写：

- tool-result pruning：把旧工具返回替换成较短占位内容；
- session compaction：把较长历史总结成新的摘要上下文。

重写后，新 prompt 已不再是旧 prompt 的前缀。如果继续强行追加 token，就会出现：SGLang 在真实新上下文上生成，Megatron 却在伪造的拼接上下文上计算 loss。

修复后的规则是：

- 严格 append-only 时，沿用当前 segment；
- prompt token 或 system prompt 发生历史重写时，关闭旧 segment；
- 从 SGLang 实际收到的新 prompt 重新 tokenize，建立新 segment；
- 只对真实 assistant response token 计算 loss，工具结果和环境文本只作为上下文。

这恢复了最重要的训练不变量：

```text
模型生成时的 prefix == rollout logprob 的 prefix == 训练 loss 的 prefix
```

长轨迹可能因此拆成多个 sample，但所有 segment 保留同一 `trajectory/group` 身份，reward 和 advantage 仍按逻辑轨迹计算；不会因为 segment 更多而重复计算 reward 或执行多次 optimizer update。

### 3.2 多工具调用

SFT 中 assistant 可在同一 turn 连续输出多个 `<tool_call>`。旧 Proxy 只执行第一个调用，使 RL 的交互语义与 SFT 不一致。修复后会：

1. 按原始输出顺序解析所有合法 tool call；
2. 为每个调用保留独立 tool-call ID；
3. 完整执行并返回全部工具结果；
4. OpenClaw 通过 ID 和顺序把结果关联回对应调用。

正式训练阶段的 tool-call 返回 retention 达到 100%；截至早期 rollout 54，multi-tool turn 平均占约 12.5%，因此它不是边缘格式问题。

### 3.3 压缩请求、后台辅助请求与轨迹边界

OpenClaw 的 compaction summary 也是一次模型请求，但它不是主 Agent 的策略 action。当前实现为主请求和 compact 请求使用独立模型 ID 与 ledger，summary 只改变下一轮上下文，不进入主策略 loss。

同时关闭 heartbeat 和后台 exec 完成通知，避免环境主动插入 SFT 中不存在的辅助 user turn；模型仍可显式调用 `process` 主动查询后台任务。该处理减少了无训练目标的上下文变化，也使轨迹来源更可解释。

### 3.4 长尾保护与监控

当前边界为：单轮 response 8,192 tokens、live context 65,536 tokens、累计 assistant token 65,536、最多 96 agent steps、最多 8 次 compaction、外层超时 1,200 秒。

达到限制的轨迹不默认过滤。若 verifier 可以评分，仍按真实失败轨迹进入训练；只对系统无法回收的轨迹做 skip/recovery。新增监控包括：

- response limit、未闭合 tool call、重复退化及明确 termination reason；
- pruning、compaction、context overflow、未分类历史重写；
- pre-clip grad norm、clip coefficient、post-clip grad norm；
- tool-call parse/return retention。

这样可以区分“模型策略失败”和“系统采集失败”，避免为了让曲线好看而删除真实负样本。

## 4. 训练端数值稳定性：严格零更新实验

### 4.1 为什么要做零更新实验

PPO/GRPO 在 optimizer 更新前，old policy 和 current policy 权重相同。因此在同一 token 上应满足：

```text
delta_logprob = logp_current - logp_old = 0
ratio         = exp(delta_logprob)      = 1
```

如果不更新参数，重复 forward/backward 仍产生明显不同结果，那么观测到的 PPO ratio、clipping 和梯度变化就混入了计算噪声，而不全是学习信号。

严格实验固定了 checkpoint、token IDs、loss mask、advantage、sample 顺序、dynamic packing 和全部 RNG state；不执行 `optimizer.step()`，比较的是正式 optimizer 真正使用的 pre-clip FP32 主梯度。

### 4.2 默认 Megatron MoE 路径确实不稳定

在 step 99 的真实 Agent batch 上，三次独立零更新复跑的完整梯度 cosine 分别为：

```text
0.560984 / 0.684099 / 0.460590
```

理想值应接近 1。Dense Qwen3-0.6B 控制组在相同诊断流程下得到 0.999914，说明 DataIterator、loss、梯度清理和比较流程本身基本正确，异常集中于 MoE 计算路径。

早期训练侧 routing replay 能把重复 forward 的 logprob MAE 从 0.00607 降至 0.00296，并把方向性假 PPO clipping 从 0.1767% 降至 0.00782%，但双 backward 梯度 cosine 仍只有约 0.71；所以“固定专家 ID”有帮助，但不是训练端确定性的完整修复。

### 4.3 消融定位

| 消融 | 结果 | 判断 |
|---|---:|---|
| Dense 0.6B 控制 | gradient cosine 0.999914 | 通用训练与诊断流程基本稳定 |
| 关闭 grouped GEMM | 0.723 / 0.538，两次不一致 | 不是单一根因 |
| 关闭 MoE permute fusion | 约 0.537 | 未解决 |
| DeepEP vs All-to-All | forward 误差几乎相同 | DeepEP 不是主要根因；约 0.7% 吞吐收益也不足以支持启用 |
| 关闭 activation recomputation | cosine 0.276→0.423，但 norm/L2 仍差 | 有影响但不是根因；显存接近 140GB 且 backward 慢约 5 倍，不可作为正式方案 |
| 仅设确定性环境变量 | cosine 0.578 | 部分改善，不充分 |
| 官方 deterministic mode | cosine 1.0，relative L2 0 | 完整修复训练端重复计算 |

### 4.4 最终修复：Megatron deterministic mode

启用以下组合：

```text
--deterministic-mode
NCCL_ALGO=Ring
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_DEVICE_MAX_CONNECTIONS=1
```

并由框架设置 PyTorch/cuDNN 的确定性执行。结果为：

- 105,994-token 控制子集：gradient cosine 1.0、norm ratio 1.0、relative L2 0；
- 239,629-token 完整真实 Agent batch：同样全部为理想值；
- 30,532,122,624 个分布式 FP32 主梯度元素最大绝对差为 0；
- Megatron old vs current、current pass 1 vs pass 2 的 84,432 个有效 token logprob 全部相同。

因此训练端不再在零更新时制造虚假 PPO ratio。代价是纯 backward 约慢 30%–38%；真实 RL 还包含工具执行和 rollout，端到端增幅预计更低，但必须在下一轮短程 smoke 中实测。

## 5. 跨后端训推一致性：剩余问题在哪里

### 5.1 确定性训练不等于 SGLang 与 Megatron 一致

在确定性 Megatron 下，同一批 84,432 个有效 token 的 SGLang→Megatron 结果为：

| 指标 | 数值 |
|---|---:|
| mean absolute logprob difference | 0.008204 |
| P50 / P90 / P95 / P99 | 3.58e-7 / 0.00642 / 0.04294 / 0.19961 |
| mean absolute probability-ratio error | 0.7995% |
| `abs(ratio-1)>1%` | 8.937% |
| `abs(ratio-1)>5%` | 4.544% |
| `abs(ratio-1)>10%` | 2.505% |

这里的 ratio 定义为：

```text
ratio = exp(logp_Megatron_old - logp_SGLang_rollout)
```

它不是“参数更新前后”的 PPO ratio，而是“训练 old policy 相对实际 rollout behavior policy”的概率比。当前 `use_rollout_logprobs=false` 可避免这项差异直接成为 PPO 分母，但轨迹仍由 SGLang 采样，loss 却按 Megatron 概率计算，因此系统仍是轻度 off-policy。

### 5.2 误差集中在低置信 token

约 87% token 的参考概率不低于 0.99，这些 token 几乎不抖。概率最低的 10% token 承担约 97% 的绝对 logprob 误差和超过 99.8% 的平方误差。

这解释了为什么全 token 平均 MAE 看起来不大，但 P95/P99 和 ratio 长尾仍不可忽略：大量标点、固定语法和复制文本稀释了平均值，真正敏感的是模型处在多个候选之间的决策 token。

因此监控不能只看 `train/rollout_logprob_abs_diff`，还应同时报告概率分桶、P95/P99、ratio 阈值比例和按 `abs(advantage)` 加权的误差。

## 6. Routing replay 与 TIS

### 6.1 两者解决的问题不同

- **Full rollout routing replay**：SGLang 保存每个 request、每个 token、48 个 MoE 层、top-8 的 expert ID；Megatron 对同一 token 强制使用相同专家。它直接减少 MoE 路由差异。
- **TIS**：不改变 forward，而是在 loss 中乘以行为策略校正权重：

```text
w_t = clip(exp(logp_Megatron_old - logp_SGLang_rollout), 0, 2)
```

TIS weight 不参与求导；它作为常数缩放当前 token 的 policy-gradient 项。Routing replay 降低 mismatch，TIS 校正 replay 后仍存在的 attention、GEMM、精度和并行归约差异。

### 6.2 受控小样本实验

在同一 step 99 checkpoint、同一真实 Agent 子集的实验中：

| 指标 | 无 replay | Full routing replay | 改善 |
|---|---:|---:|---:|
| logprob MAE | 0.013308 | 0.006784 | -49.0% |
| logprob RMS | 0.062910 | 0.025230 | -59.9% |
| P99 abs diff | 0.232825 | 0.119062 | -48.9% |
| mean `abs(ratio-1)` | 1.2864% | 0.6773% | -47.4% |
| `abs(ratio-1)>10%` | 4.0913% | 1.4970% | -63.4% |
| ratio 超出 `[0.8,1.28]` | 1.0228% | 0.1355% | -86.8% |

TIS `[0,2]` 在无 replay 的受控 batch 上：weight mean 0.999712、ESS 99.7454%、仅 1/14,763 token 触及上界，梯度范数只变化约 -0.5%。这表明校正总体温和，主要处理长尾，而不是重写整个 batch。

### 6.3 正式 8×8 端到端 A/B

配置为 8 个任务、每题 8 条，共 64 条真实多轮轨迹；SGLang 和 Megatron 均开启确定性模式。结果如下：

| 指标 | Baseline | Replay + TIS |
|---|---:|---:|
| logprob MAE | 0.015356 | 0.008030 |
| P99 abs diff | 0.27407 | 0.14719 |
| ratio 超出 `[0.8,1.28]` | 1.416% | 0.308% |
| TIS weight mean | — | 0.999417 |
| TIS ESS | — | 99.905% |
| skipped/lost/timeout | 0/0/0 | 0/0/0 |

数值改善与受控实验一致。但两组只有 56.7% 的同 session-step response 文本完全相同，环境反馈使 34/64 个 session 后续分叉，因此 reward 差不能归因于 replay/TIS；数值因果结论应以同 token 的受控实验为主。

### 6.4 当前 full replay 的成本

| 成本项 | Baseline | Per-turn full replay | 倍率 |
|---|---:|---:|---:|
| 训练 sample | 74 | 910 | 12.30× |
| 训练 context token | 1.37M | 13.53M | 9.87× |
| packed micro-batch | 18 | 170 | 9.44× |
| 训练侧总时间 | 283s | 1,918s | 6.77× |
| wait + train | 813s | 3,088s | 3.80× |
| rollout dump | 14.3MB | 20.88GB | 约 1464× |

根因不是有效 response token 变多，而是为精确重放每个 Agent request 的路由，当前实现重复保存和重算每一轮的完整长前缀。

当前已实现 route stitching：对于 token 与路由均严格一致的 append-only 前缀，只保留一次；遇到 pruning/compaction/history rewrite、engine switch 或实际路由冲突时再切 segment。最近一次初步 rollout 将 64 条轨迹拼为 132 个 segment，平均 2.06 segment/trajectory，明显低于 910 个 per-turn sample；但该 smoke 的训练阶段尚未完整收尾，不能把它作为最终吞吐结论。

## 7. 推理与评测可复现性

固定单个 8,196-token query，temperature 0.7、top-p 0.8、top-k 20，各重复 10 次：

| 配置 | 唯一输出序列数 | 结论 |
|---|---:|---|
| 无 request seed，无 deterministic inference | 6/10 | 不可复现 |
| 固定 request seed，未开 deterministic inference | 6/10 | request seed 单独不足 |
| 固定 request seed + SGLang deterministic inference | 1/10 | token 序列完全复现 |

因此确定性采样不是 temperature=0；它仍按 temperature/top-p/top-k 的分布采样，只是固定同一请求的随机流，并要求推理 kernel 和 batch 调度走可复现路径。

完整 Agent 轨迹还受工具返回、时间戳、后台 process 名、网络和环境状态影响，所以不能承诺跨时间 bitwise 一致。评测侧当前通过固定任务顺序、rollout ID、per-request sampling seed、确定性 tool-call ID、SGLang deterministic inference 和统一 Harness，尽可能把剩余差异限制在真实环境动态上。

确定性 200 题基线显示 SFT 相对 Base 明显提升：平均 verifier reward 0.41477→0.53455，Correct 83/200→117/200，逐题 bootstrap 95% CI 为 `[0.0810, 0.1599]`。这说明评测链路能够识别较大的真实能力变化。

对于 RL，训练内在线 eval 在 step 19/39/59/79/99 分别为 55.21%、54.94%、56.29%、57.23%、57.63%；但旧的独立复评均值约为 54%–55%。两类评测当时没有完全统一 request seed、确定性工具 ID和最终 Harness，因此在线曲线不能单独用于声明 RL 提升。下一轮必须从 step 0 就使用同一个确定性评测协议并保存逐题轨迹。

## 8. 吞吐与系统稳定性优化

### 8.1 Session-affinity 路由

同一 Agent session 优先路由到同一个 SGLang engine，使其复用 Radix/KV cache；若目标 engine 比最空闲 engine 多 2 个以上 active request，则回退 least-loaded。

在 8 sessions × 8 turns、12K 初始上下文的可控 A/B 中：

| 指标 | Least-active | Session affinity | 变化 |
|---|---:|---:|---:|
| 墙钟时间 | 6.602s | 3.695s | -44.0% |
| 吞吐 | 9.694 req/s | 17.323 req/s | 1.79× |
| engine switches | 6 | 0 | -100% |
| cache hit rate | 68.16% | 73.41% | +5.24 pp |
| prefill compute tokens | 291,456 | 243,456 | -16.47% |

这是 prefill 主导的微基准收益；真实 RL 还包含 Docker、工具、verifier 和训练，因此不能直接外推 1.79 倍端到端加速。

### 8.2 运行隔离与故障恢复

已落地的运行保护包括：

- 本地独立工作区，不原地修改共享 `/mnt` 代码；
- run-scoped gateway/SSH/模型端口，避免与其他任务冲突；
- gateway 启动最多三次恢复，每次使用新隔离端口；
- 容器启动并发限制为 4，降低瞬时 CPU/IO 压力；
- unhealthy worker 不进入 session-affinity 候选；
- 训练在 tmux server 下持久运行，checkpoint 每 20 step 保存；
- 完整 termination、skip/recovery、timeout、OOM/NaN/Inf 监控。

正式训练最终完成 100 个 rollout/约 99 次 optimizer update，没有此前 60–80 step 的持续 reward 崩塌，也没有 NaN/Inf、训练 timeout 或无法恢复的系统性故障。偶发大梯度来自真实重复退化/截断失败轨迹，`clip_grad=1.0` 正常限制更新幅度；不应删除这些有效负样本。

## 9. 当前推荐配置与待办

### 下一轮可以直接保留

- token ledger、history rewrite 正确分段、multi-tool 完整执行；
- main/compaction 请求隔离，heartbeat 和被动后台通知关闭；
- run-scoped 端口、启动恢复、容器并发 4、session affinity；
- `clip_grad=1.0`、activation recomputation 保持开启；
- Megatron `deterministic mode` 及其所需环境变量；
- 统一的确定性 eval：固定 task order、request seed、tool-call ID、Harness 与完整轨迹保存；
- MAE 之外继续监控 P95/P99、ratio 长尾、TIS ESS、pre/post clip grad norm。

### 需要短程 A/B 后决定

- **TIS `[0,2]`**：数值上温和、ESS 很高，优先做 TIS-only 5–10 step A/B，测实际吞吐与训练曲线；
- **Route stitching 后的 full routing replay**：先完成当前 smoke，并验证每个 stitched prefix 的 token hash、route hash、router agreement、训练时间和存储；
- **路由紧凑编码**：expert ID 范围为 0–127，可由 int32 改为 uint8，理论路由 payload 降约 4 倍。

### 暂不建议

- 不直接启用未优化的 per-turn full routing replay 长训；
- 不关闭 activation recomputation；
- 不因数值问题迁移到 veRL+Megatron，因为训练后端仍是 Megatron-Core；若做框架消融，应比较 FSDP2/VeOmni 等真正不同后端；
- 不用单次在线 eval 或单次 reward 波动宣称 RL 已提升。

## 10. 最终判断

本轮最明确的成果不是某个 checkpoint 多了几个百分点，而是把三种原本混在一起的问题拆开并分别处理：

1. **轨迹语义一致性**已经修复；
2. **Megatron 训练端重复计算稳定性**已经通过官方 deterministic mode 修复；
3. **SGLang–Megatron 跨后端行为策略差异**已被量化，routing replay 有效但昂贵，TIS 是更轻量的候选校正。

因此下一轮 RL 已具备比上一轮更可信的训练基础，但要证明算法收益，仍需在统一确定性 eval 下完成 SFT 与多个 RL checkpoint 的成对复评，并对 TIS/route stitching 做多 step A/B。

## 11. 三分钟组会汇报稿

大家好，我汇报一下 OpenClaw-RL 在训推一致性和训练稳定性上的排查与改造。

我们把问题分成三层。第一层是 Agent 轨迹语义。旧实现假设 prompt 只会追加，但 OpenClaw 会裁剪工具结果或压缩历史，导致 SGLang 在真实新上下文上生成，Megatron 却在错误拼接的上下文上计算 loss。旧 Proxy 还只执行一次输出中的第一个工具调用。现在只有严格追加时才续接 token；历史重写就从真实 prompt 新建 segment；多工具调用全部执行；压缩总结使用独立 ledger，不进入主策略 loss。这样生成、rollout logprob 和训练 loss 的上下文已经对齐。

第二层是训练端数值稳定性。我们用 step 99 checkpoint，对完全相同的数据做两次零更新 forward/backward。默认 MoE 配置的梯度 cosine 只有 0.46 到 0.68，而 dense 控制组为 0.9999。Routing replay、grouped GEMM、DeepEP 和关闭 activation recomputation 都不能彻底解决。启用 Megatron 官方 deterministic mode 后，在 23.96 万 token 的真实 Agent batch 上，两次 logprob 完全相同，305.3 亿个分布式 FP32 梯度元素逐元素一致，梯度 cosine 为 1。纯 backward 代价约增加 30%到38%，但训练端制造的虚假 PPO ratio 已经消除。

第三层是跨后端差异。训练端确定以后，SGLang 与 Megatron 的 logprob MAE 仍约为 0.0082，8.9%的 token 概率比偏差超过 1%，而且误差主要集中在低置信 token。Full routing replay 可把 MAE 和 P99 降低约一半，TIS 的 ESS 超过 99.7%，校正比较温和。但当前 per-turn replay 会把训练 context 放大 9.9 倍、训练时间放大 6.8 倍，暂时不适合长训。Route stitching 已初步把 64 条轨迹从 910 个 sample 降到 132 个 segment，但完整测试仍需收尾。

系统侧还加入 session affinity、隔离端口和启动恢复；可控 prefill 测试吞吐提升 1.79 倍。正式训练稳定跑到 step 99，没有持续 reward 崩塌、NaN 或 OOM。但在线 eval 的上涨没有被独立复评稳定复现。因此当前结论是：轨迹语义和训练端确定性已经解决；跨后端 mismatch 有有效方案，但还要降成本；RL 是否真正提升能力，需要下一轮统一确定性评测重新确认。

## 12. 主要证据与复现入口

- 数值稳定性总实验：`/root/openclaw-rl-local/reports/RL_FRAMEWORK_NUMERICAL_ABLATION_20260824_ZH.md`
- Full routing replay + TIS 端到端 A/B：`/root/openclaw-rl-local/reports/FORMAL_E2E_ROUTING_REPLAY_TIS_AB_STEP99_20260825_ZH.md`
- 早期逐 token/双 backward 诊断：`/root/openclaw-rl-local/reports/rl_numerical_stability_experiment_20260819_zh.md`
- Logprob 概率分桶：`/root/openclaw-rl-local/reports/logprob_jitter_probability_buckets_step99_20260820_zh.md`
- Session affinity A/B：`/root/openclaw-rl-local/outputs/session-affinity-ab-20260818/SESSION_AFFINITY_AB_REPORT_ZH.md`
- Base vs SFT 确定性评测：`/root/openclaw-rl-local/reports/BASE_VS_SFT_DETERMINISTIC_EVAL_20260820_ZH.md`
- RL checkpoint 独立复评：`/root/openclaw-rl-local/reports/RL_CHECKPOINT_EVALUATION_BRIEF_ZH.md`
- 统一正式启动脚本：`/root/openclaw-rl-local/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh`
