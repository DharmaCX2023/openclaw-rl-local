# Step-99 条件式 Routing Replay 拼接：正式 Smoke 报告

## 结论

本次改造与正式端到端 smoke **通过**。

条件式拼接把 64 条真实 ClawGym 轨迹的 870 个 agent request，从旧语义下的 870 个逐请求训练 sample，压缩为 132 个严格可回放 segment，sample 数减少 **84.83%**。完整流程成功执行了 SGLang rollout、expert route 回传、Megatron reference/old logprob、TIS、PPO backward 和 optimizer update，最终 exit code 为 0；无跳样、丢样、timeout、OOM、NaN、端口冲突或 route shape 错误。

相较此前逐请求 full routing replay，训练 context token 从 13.53M 降至 2.49M，训练侧总耗时从 1,917.6s 降至 442.7s，下降 **76.9%**。这已经把 routing replay 从不可接受的近 10 倍计算放大，降到了可继续做多 step 稳定性实验的范围。

## 改造语义

多轮请求只有同时满足以下条件才拼接：

1. 新 prompt 的 token IDs 严格以当前 segment 的完整 token 序列为前缀；
2. OpenClaw token ledger 将变化声明为 `append_only`；
3. SGLang 返回的新请求 expert IDs，与当前 segment 已保存的全部 prefix route rows 完全一致；
4. 新请求不超过 65,536-token 轨迹限制。

满足时，仅附加：

- 新增的 tool/environment token，`loss_mask=0`；
- 新生成的 assistant token，`loss_mask=1`；
- 从 `len(accumulated_tokens)-1` 开始的新增 expert-route rows。

任何 history rewrite、tool-result pruning、compaction、token-prefix 不一致或 route-prefix 冲突都会立即 flush 当前 segment，并从该请求的真实完整 prompt、response 和 route tensor 开始新 segment。所有 segment 继续共享同一 `traj_index/group_index`，不会因切段而把一条长轨迹重复计入 GRPO baseline。

同时新增了以下审计与 TensorBoard 指标：

- `routing_replay_segments_total`
- `routing_replay_avg_turns_per_segment`
- `routing_replay_stitched_turns_total`
- `routing_replay_route_conflicts_total`
- `routing_replay_rewrite_splits_total`

SGLang route 的 base64 解码结果现在显式复制为 writable NumPy array，消除了 `torch.from_numpy` 对只读 buffer 的潜在未定义行为警告。

## 正式 Smoke 配置

- checkpoint：RL step 99
- 任务：8 个固定 task，每题 8 条轨迹，共 64 条
- rollout：4 GPU，2 个 SGLang TP=2 engine
- train：4 GPU，Megatron TP=4、EP=4
- routing replay：开启
- TIS：开启，clip `[0, 2]`
- deterministic SGLang 与 deterministic Megatron：开启
- temperature：0.7
- request seed base：20260825
- context：65,536
- per-turn response：8,192
- max agent steps：96
- `max_tokens_per_gpu=81,920`
- GRPO、per-token loss、学习率 `1e-6`
- 单次真实 optimizer update
- 隔离端口：本地 Proxy `30323`，反向 SSH `65023`，远端 gateway 使用 run-scoped 动态端口

## 系统结果

| 指标 | 旧逐请求 replay | 条件式拼接 | 变化 |
|---|---:|---:|---:|
| agent request / turn | 910 | 870 | 两次工具环境轨迹略有分叉 |
| 训练 sample / segment | 910 | 132 | 当前相对自身 turn `-84.83%` |
| 训练 context token | 13,533,203 | 2,490,412 | `-81.60%` |
| packed microbatch | 170 | 32 | `-81.18%` |
| rollout dump | 20.883 GB | 3.847 GB | `-81.58%` |
| reference logprob | 290.6s | 63.2s | `-78.25%` |
| old/current logprob | 300.8s | 53.5s | `-82.21%` |
| actor train | 1,324.6s | 323.9s | `-75.55%` |
| 训练侧总时间 | 1,917.6s | 442.7s | `-76.91%` |
| wait + train | 3,087.9s | 1,535.6s | `-50.27%` |

当前 870 turn 的具体切段结果：

- segment：132
- 成功拼接的后续 turn：738
- 平均每 segment：6.59 turn
- route-prefix conflict 切段：49
- rewrite/pruning 切段：19
- segment 数中位数：每轨迹 1
- 最大 segment 数：单轨迹 28
- 最大 agent turn：96

动态装箱形成 32 个 microbatch，最大 81,915 token，不存在 oversized singleton。

## 数值结果

基于 353,402 个有效 response token：

| 指标 | 数值 |
|---|---:|
| SGLang–Megatron logprob MAE | 0.008662 |
| logprob RMS | 0.032757 |
| logprob abs P50 / P90 / P95 / P99 | 2.38e-7 / 0.02081 / 0.05549 / 0.15392 |
| 最大 logprob abs diff | 1.7717 |
| 平均 `abs(ratio-1)` | 0.8576% |
| `abs(ratio-1)>1%` | 13.557% |
| `abs(ratio-1)>5%` | 5.506% |
| `abs(ratio-1)>10%` | 2.402% |
| ratio 超出 PPO `[0.8,1.28]` | 0.3404% |
| TIS weight mean | 0.999295 |
| TIS ESS | 99.8981% |
| TIS clip fraction | 0.000283% |
| mismatch low-var KL | 5.619e-4 |

真实更新指标：

- `train/loss = train/pg_loss = 0.104485`
- `train/pg_clipfrac = 0`
- `train/ppo_kl = 0`
- `train/kl_loss = 0.001740`
- `pre_clip_grad_norm = post_clip_grad_norm = 0.545607`
- `grad_clip_coefficient = 1.0`

`pg_clipfrac=0` 和 `ppo_kl=0` 在本单步 smoke 中是正常的：Megatron old/current 在 optimizer 更新前权重相同；跨 SGLang–Megatron 的 mismatch 由 TIS 指标描述。TIS ESS 接近 100%，说明修正没有显著降低有效样本量。

## 轨迹健康度

- submitted / skipped / lost / timeout：`64 / 0 / 0 / 0`
- natural completion：62
- length-limited incomplete tool call：1
- max-agent-steps：1
- response-limit 命中：2
- compaction：0
- tool-result pruning event：20，共 prune 37 个 tool result
- unclassified history rewrite：0
- multi-tool turn 比例：11.03%
- tool-call parse retention：99.73%
- 平均 reward：0.72422

本轮没有触发 session compaction，因此 compaction 分支主要由单元测试覆盖；真实 pruning 已触发并按 rewrite 安全切段。

## 验证与限制

CPU 回归：43/43 通过；补充的 writable-route 定向回归：2/2 通过。

本实验是一次完整 optimizer update，足以验证数据语义、shape、TIS/PPO 和系统开销，但不能替代 5–10 step 的稳定性 A/B。旧、新两次正式运行使用相同任务规模与采样设置，但工具环境反馈会造成后续轨迹分叉，因此耗时与 token 对比是工程实测，不是逐 token 完全配对实验。

## 复现与产物

启动：

```bash
cd /root/openclaw-rl-local
bash scripts/run_formal_routing_stitch_smoke_step99.sh
```

关键产物：

- 实验目录：`/root/openclaw-rl-local/outputs/formal-routing-stitch-smoke-step99-20260825-091500`
- 结构化汇总：`/root/openclaw-rl-local/outputs/formal-routing-stitch-smoke-step99-20260825-091500/summary.json`
- runner 日志：`/root/openclaw-rl-local/outputs/formal-routing-stitch-smoke-step99-20260825-091500.runner.log`
- 启动脚本：`/root/openclaw-rl-local/scripts/run_formal_routing_stitch_smoke_step99.sh`

实验结束后，测试进程已退出，远端本次容器已清理，8 张 GPU 已恢复持续烤机；检查时 8 卡均为 100% 利用率、每卡约 95GB 显存。
