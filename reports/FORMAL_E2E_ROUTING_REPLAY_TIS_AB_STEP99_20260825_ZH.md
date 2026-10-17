# Step-99 Routing Replay + TIS 正式端到端 A/B 报告

## 结论

本次测试完成了两组真实的 ClawGym agent rollout 与 Megatron optimizer update，均成功退出，无丢样、timeout、NaN、OOM 或路由 shape 错误。

- **数值一致性明显改善**：routing replay + TIS 将 SGLang rollout 与 Megatron old-policy 的 token logprob MAE 从 `0.01536` 降至 `0.00803`（`-47.7%`），P99 从 `0.2741` 降至 `0.1472`（`-46.3%`），ratio 超出 PPO 窗口 `[0.8, 1.28]` 的比例从 `1.416%` 降至 `0.308%`（`-78.2%`）。
- **TIS 校正温和**：B 组 TIS weight mean 为 `0.999417`，ESS 为 `99.905%`，仅 `0.0011%` token 触及 `[0, 2]` 裁剪边界。它没有大范围重写梯度，只校正少量长尾 token。
- **当前完整 routing replay 的系统代价过高**：为精确回放每次 agent request 的 prompt 与 response 路由，当前实现将 64 条轨迹拆成 910 个 per-turn sample；训练 context token 从 `1.37M` 增至 `13.53M`（`9.87×`），训练侧耗时从 `283s` 增至 `1,918s`（`6.77×`），单步 wait+train 从 `813s` 增至 `3,088s`（`3.80×`）。
- **不建议未经优化就把当前 full replay 直接用于长周期正式 RL**。建议保留 TIS 为候选默认方案，先优化 routing replay 的路由编码与多轮前缀复用，再做多 step 稳定性实验。

## 实验设计

共同配置：

- checkpoint：step 99（加载目录 iteration `992000`，Slime 识别为 iteration 99）
- 8 个固定任务，每题 8 条轨迹，共 64 条轨迹
- rollout：SGLang 4 GPU，2 个 TP=2 engine
- train：Megatron 4 GPU，TP=4、EP=4
- temperature `0.7`，固定 request seed base `20260825`
- SGLang deterministic inference 开启
- Megatron deterministic mode 开启
- max response `8192`，context `65536`，max agent steps `96`
- per-token loss，GRPO，单次 optimizer update，学习率 `1e-6`

变量：

| 组别 | Routing replay | TIS |
|---|---:|---:|
| A：baseline | 关闭 | 关闭 |
| B：replay + TIS | 完整 agent-request 路由回放 | 开启，clip `[0,2]` |

这里的 B 组不是只替换一个标量：每次 SGLang 请求返回形状为 `[request_tokens-1, 48 layers, top-8]` 的 expert ID。由于多轮 agent 的每次请求都有独立完整前缀，当前实现按 agent turn 建立训练 sample；所有 turn 继续共享原轨迹的 `traj_index/group_index`，且仅 response token 进入 loss。

## 核心数值结果

下表基于各组 rank-0 train dump 中所有有效 response token。两组 rollout 后半程受到工具环境反馈影响而部分分叉，因此这是正式端到端的同配置对比，不是逐 token 配对检验；严格同数据结论应与此前 zero-update 配对实验联合解释。

| 指标 | A：baseline | B：replay + TIS | 相对变化 |
|---|---:|---:|---:|
| 有效 response token | 325,100 | 365,340 | 轨迹分叉，不直接比较 |
| logprob MAE | 0.015356 | 0.008030 | -47.7% |
| logprob RMS | 0.060457 | 0.031541 | -47.8% |
| logprob abs P50 | 7.15e-7 | 2.38e-7 | -66.7% |
| logprob abs P90 | 0.03662 | 0.01791 | -51.1% |
| logprob abs P95 | 0.09562 | 0.05069 | -47.0% |
| logprob abs P99 | 0.27407 | 0.14719 | -46.3% |
| 最大 logprob abs diff | 4.7170 | 1.8769 | -60.2% |
| 平均 `abs(ratio-1)` | 1.505% | 0.796% | -47.1% |
| `abs(ratio-1)>1%` | 16.331% | 12.756% | -21.9% |
| `abs(ratio-1)>5%` | 8.388% | 5.082% | -39.4% |
| `abs(ratio-1)>10%` | 4.731% | 2.190% | -53.7% |
| ratio 超出 `[0.8,1.28]` | 1.416% | 0.308% | -78.2% |
| 按 `abs(advantage)` 加权的 ratio 误差 | 1.412% | 0.707% | -49.9% |
| mismatch low-var KL（后处理） | 0.002434 | 0.000522 | -78.6% |

定义：

```text
delta = log p_Megatron_old - log p_SGLang_rollout
ratio = exp(delta)
TIS weight = clip(ratio, 0, 2)
ESS fraction = (sum w)^2 / (N * sum w^2)
```

B 组的 TIS 统计：

| 指标 | 数值 |
|---|---:|
| weight mean | 0.999417 |
| mean `abs(weight-1)` | 0.7962% |
| clip fraction | 0.001095% |
| ESS | 99.9050% |

`train/kl_loss` 不是上述训推 mismatch KL。它衡量当前 policy 与 SFT reference 的差异；A/B 分别为 `0.001807` 与 `0.001674`，由于两组 token 不完全相同，不能用该差值判断 routing replay 效果。

`train/ppo_kl=0`、`pg_clipfrac=0` 也是预期结果：本实验每组只做一次 update，loss 中的 Megatron old/current 在 optimizer step 前权重相同。TIS ratio 则比较 SGLang rollout 与 Megatron old policy，是另一条比值。

## 系统代价

| 指标 | A：baseline | B：replay + TIS | 倍率 |
|---|---:|---:|---:|
| 训练 sample | 74 | 910 | 12.30× |
| 训练 context token | 1,371,342 | 13,533,203 | 9.87× |
| packed micro-batch | 18 | 170 | 9.44× |
| rollout 时间 | 519.2s | 1,122.6s | 2.16× |
| ref logprob | 39.0s | 290.6s | 7.45× |
| old logprob | 28.5s | 300.8s | 10.55× |
| actor train | 214.1s | 1,324.6s | 6.19× |
| 训练侧总时间 | 283.1s | 1,917.6s | 6.77× |
| wait + train | 812.6s | 3,087.9s | 3.80× |
| rollout dump | 14.3MB | 20.88GB | 约 1464× |

20.88GB 基本等于：

```text
13.53M request tokens × 48 MoE layers × top-8 × int32
```

expert ID 范围仅为 0–127，使用 `uint8` 即可无损表达，理论上可把路由 payload、Ray spill 与 rollout dump 降约 4×。训练 dump 只有约 118.6MB/rank，是因为 routing tensor 在 replay forward 后已删除，未重复写入四份 train dump。

## 轨迹健康度与可比性

| 指标 | A | B |
|---|---:|---:|
| submitted / skipped / lost / timeout | 64 / 0 / 0 / 0 | 64 / 0 / 0 / 0 |
| 平均 reward | 0.71095 | 0.72083 |
| SGLang 请求 | 893 | 910 |
| max-agent-steps 终止 | 2 | 1 |
| response-limit 命中 | 1 | 1 |
| context overflow / compaction | 0 / 0 | 0 / 0 |
| tool pruning events | 10 | 12 |

固定 seed 保证相同的初始任务与采样随机源，但不能消除工具环境的动态返回。A/B 共有 766 个同 session-step，其中 434 个 response 文本逐字相同（56.66%）；34/64 个 session 在后续某一步分叉，多数首次分叉发生在第 3–7 步。因此：

- reward 的 `+0.00987` 不可归因于 replay/TIS；
- 系统吞吐是部署真实成本，但也包含两组轨迹长度差异；
- 数值改善与此前严格 zero-update、同 token 配对实验方向一致，后者仍是因果判断的主要证据。

## 建议

1. **TIS 单独保留为下一轮正式训练候选**：当前权重接近 1、ESS 很高，属于低侵入校正。正式启用前补一个 `TIS-only` 小规模端到端实验，隔离其独立效果和开销。
2. **暂不直接启用当前 full routing replay 长训**：单步 context 计算放大近 10×，长期训练成本不可接受。
3. **优先压缩路由表示**：SGLang→Proxy→Ray 使用 `uint8` 或等价紧凑编码，并记录编码/解码校验和。
4. **研究安全的 route stitching**：对 append-only 且路由一致的共享前缀只保留一次；仅在 history rewrite、engine switch/cache miss 后路由冲突处切 segment。必须逐 token 校验 stitched routes，不可假设前缀天然一致。
5. **优化后再做 5–10 step 稳定性 A/B**：比较 reward、entropy、grad cosine、ratio tail、ESS、墙钟时间与显存/Ray spill，而不是只做单步。

## 复现与产物

启动脚本：

```bash
cd /root/openclaw-rl-local
bash scripts/run_formal_e2e_routing_tis_ab_step99.sh
```

重新生成统计：

```bash
cd /root/openclaw-rl-local
.venv-openclaw/bin/python \
  scripts/analyze_formal_e2e_routing_tis_ab.py \
  outputs/formal-e2e-routing-tis-ab-step99-20260825-034500 \
  --output outputs/formal-e2e-routing-tis-ab-step99-20260825-034500/summary.json
```

关键路径：

- 实验目录：`/root/openclaw-rl-local/outputs/formal-e2e-routing-tis-ab-step99-20260825-034500`
- 结构化汇总：`/root/openclaw-rl-local/outputs/formal-e2e-routing-tis-ab-step99-20260825-034500/summary.json`
- 分析脚本：`/root/openclaw-rl-local/scripts/analyze_formal_e2e_routing_tis_ab.py`
- A/B 状态：`case_status.tsv`，两组 exit code 均为 0

实验结束后已验证：本地训练与 SGLang 进程退出、测试远端容器清理、未操作其他用户容器；8 张 GPU 已恢复持续烤机，检查时利用率为 99–100%。
