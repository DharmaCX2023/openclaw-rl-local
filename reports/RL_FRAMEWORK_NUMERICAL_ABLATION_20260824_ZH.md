# OpenClaw-RL 训练框架数值稳定性与运行隔离报告

日期：2026-08-24  
模型：Qwen3-30B-A3B ClawGym SFT / RL step99  
训练栈：Slime + Megatron-LM  
Rollout 栈：SGLang + OpenClaw harness

## 1. 结论摘要

当前训练后端在默认配置下存在显著的 MoE 数值不确定性；进一步实验已确认，Slime/Megatron 官方 deterministic mode 可以在真实 agent batch 上把两遍完整梯度恢复为逐元素完全一致。

- 同 checkpoint、同 batch、同 token、完整恢复 RNG、零 optimizer update 的三次 MoE 基线梯度 cosine 分别为 `0.560984`、`0.684099`、`0.460590`，远低于应有的接近 1，且复跑方差明显。
- 相同诊断流程在 dense Qwen3-0.6B 上得到 `0.999914`。通用 DataIterator、loss、梯度清理和双 backward 流程基本稳定；异常集中在 Megatron MoE 路由、专家 dispatch/compute、专家并行归约或相关低精度路径。
- 完整回放 SGLang rollout expert IDs 后，在另一批真实 multi-turn agent 数据上仍只有 `0.638781`。该数据与旧基线不是同一 batch，不能作为严格数值 A/B；但它证明 full rollout routing replay 并未让梯度恢复到接近确定。
- 单独关闭 grouped GEMM 的两次结果为 `0.723358` 和 `0.537949`，改善不可重复；不能认定 grouped GEMM 是唯一根因。
- 完整旧 batch 的 activation recomputation on/off 对照因 `53,389`-token 样本 OOM。随后从同一 saved batch 按完整轨迹筛出可执行控制子集，严格 A/B 得到：recompute on 的 cosine 为 `0.276396`，off 为 `0.423367`。关闭 recompute 有影响但没有恢复确定性，且单遍 backward 从 `37.1s` 增至 `207.2s`、第二遍增至 `322.0s`，显存接近 140GB 上限；因此 recompute 是影响因素，不是根因，也不是可行的正式修复。
- 同一 105,994-token agent 子集上，普通模式 cosine 为 `0.276396`；只设置 Ring NCCL、确定性 cuBLAS/NVTE 环境变量为 `0.578118`；再启用 `--deterministic-mode` 后得到 `cosine=1.0`、`norm ratio=1.0`、`relative L2=0`、`max abs diff=0`。
- 在原始完整 239,629-token agent batch（最大 packed micro-batch 81,000 tokens）上，deterministic mode 同样得到所有 30,532,122,624 个 FP32 optimizer-gradient 元素零差异；shared/dense 与 expert optimizer chain 都完全一致。
- veRL 的 Megatron 模式仍使用 Megatron-Core，不能天然规避该问题；但 veRL 当前主线也为 Qwen3-30B-A3B 提供 FSDP2 和 VeOmni 路径。由于本栈已通过官方确定性模式恢复训练侧 bitwise 一致，现阶段无需为此重写框架。

这是一项需要在下一次正式 RL 中显式开启的训练配置。它证明默认模式的梯度抖动可以消除，但仍不能反向证明此前 step 60 后 reward 下降完全由该问题造成。

## 2. 严格实验定义

有效的梯度一致性实验满足：

1. 从 RL step99 加载同一份 actor checkpoint；
2. 加载同一个已保存 rollout batch；
3. 两遍使用完全相同的 token、loss mask、advantage、old/reference logprob；
4. 两遍 backward 前重置同一 DataIterator；
5. 恢复 Python、NumPy、PyTorch CPU/CUDA 和 Megatron CUDA RNG tracker；
6. 重置 routing replay 的 forward/backward index；
7. 第一遍后清空 model gradient buffer 和 optimizer gradient；
8. 不执行 `optimizer.step()`，不推进 scheduler；
9. 比较两遍完整、pre-clip、FP32 optimizer main gradient。

指标定义：

- `gradient_cosine`：两次完整更新方向的一致性，越接近 1 越好；
- `norm_ratio = ||g2|| / ||g1||`：梯度尺度一致性，越接近 1 越好；
- `relative_l2_error = ||g2-g1|| / ||g1||`：相对差异，越接近 0 越好。

### 动态 batch 诊断修正

full-routing 数据包含多个 optimizer-step batch 时，旧诊断状态会在第一次 `train()` 的 step0/step1 内完成配对，错误地比较两个不同 batch。现已修复为：

- 选择一个完整 optimizer-step batch；
- 该 batch 内可包含多个 packed micro-batch；
- 两次外层 `train()` 都从 offset 0 重放该 batch；
- 只有第二次外层 backward 才与第一次比较。

同时，debug rollout 加载路径现在会重建 live rollout 的 `dynamic_global_batch_size`，避免保存数据回放时把一次 live update 错拆成多次 update。

中间产生的 `cosine≈0.0012` 来自不同 optimizer-step batch 的错误比较，已判定无效，不进入任何结论。

## 3. 数据口径

### 旧 saved batch

路径：

```text
/root/openclaw-rl-local/outputs/zero-update-consistency-step99-20260814/
  baseline/rollout_data/100.pt
```

原始 10 个 sample；GRPO 删除 constant-reward group 后保留 6 个训练 segment：

```text
总 token                         239,629
有效 response/loss token         84,432
81K packing micro-batch 数             3
55K packing micro-batch 数             6
```

六条有效训练 segment 中最长样本为 `53,389` tokens。它不能在不改变真实上下文的前提下继续切短，因此 no-recompute 的显存下界由该样本决定。

### Recompute 严格 A/B 控制子集

为避免把长轨迹从中间切断，控制子集按 `(group_index, trajectory index)` 整体保留或整体删除，只保留该轨迹所有 segment 均不超过 `45,000` tokens 的轨迹。原始 10 个 sample 保留 7 个；constant-reward group 依照正式 GRPO 逻辑过滤后，两侧都使用完全相同的 3 条轨迹：

```text
长度                         24,994 / 40,689 / 40,311
reward                       0.954545 / 0.909091 / 0.863636
累计 token                   105,994
packed micro-batch 数              3
optimizer update 数                0（诊断明确跳过）
```

这组实验能够回答“在当前硬件可执行的 40K 级真实轨迹上，recompute 是否是梯度不确定性的主因”，但不能替代含 53K 样本的完整 batch 结果。

### Full rollout routing replay batch

新采样得到 213 个 per-agent-request sample。过滤一个 constant-reward group 后：

```text
训练 sample                       180
累计输入 token               3,543,778
dynamic_global_batch_size         213
optimizer update 数                 1
packed micro-batch 数              45
```

每个真实 agent request 使用自己的精确 prompt、response 和 SGLang expert IDs；同一轨迹的 request sample 共享 `traj_index` 和 `group_index`，只在 assistant response token 上计算 loss。所有 request sample 在一次 global batch 内累积梯度，只执行一次 update。

## 4. 消融结果

| 编号 | 条件 | 梯度 cosine | norm ratio | relative L2 | 结论 |
|---:|---|---:|---:|---:|---|
| 1 | MoE 基线，完整 RNG 恢复 | 0.560984 | 1.101166 | 0.988480 | 明显不稳定 |
| 1R2 | 相同基线独立复跑 | 0.684099 | 1.055922 | 0.818695 | 仍不稳定 |
| 1R3 | 相同基线第三次复跑 | 0.460590 | 0.949834 | 1.013517 | 复跑方差大 |
| 3 | training routing replay | 0.526279 | 0.877387 | 0.919950 | 未改善 |
| 4 | full rollout routing replay；另一真实 agent batch | 0.638781 | 0.753189 | 0.777849 | 不同 batch，仅证明仍不确定 |
| 5 | 固定 expert IDs 与 router probabilities | 0.423981 | 0.710740 | 0.949985 | 更差；仅隔离诊断 |
| 7a | 仅关闭 grouped GEMM，第一次 | 0.723358 | 0.986200 | 0.738809 | 单次改善 |
| 7aR | 仅关闭 grouped GEMM，第二次 | 0.537949 | 1.243023 | 1.098973 | 改善不可重复 |
| 7b | 仅关闭 permute fusion | 0.536924 | 0.884142 | 0.912290 | 未改善 |
| 7c | 两种 fusion 都关闭 | 0.574047 | 0.864193 | 0.868709 | 未改善 |
| 7d | training replay + 两种 fusion 关闭 | 0.815988 | 0.966524 | 0.597348 | 单次结果，交互项未复跑 |
| 8 | dense Qwen3-0.6B 控制组 | 0.999914 | 1.000407 | 0.013133 | 通用框架路径稳定 |
| 15 | CP=1，55K，recompute on | 0.590189 | 0.975688 | 0.894588 | 仍不稳定 |
| 18 | CP=2，55K，recompute on | 0.516369 | 0.912919 | 0.943724 | 仍不稳定 |
| 21 | 同批完整轨迹子集，recompute on | 0.276396 | 0.754776 | 1.073524 | 严格 A/B 基准 |
| 22 | 同批完整轨迹子集，recompute off | 0.423367 | 1.816854 | 1.662097 | 方向略升但仍严重不稳定 |
| 23 | 同一子集，官方 deterministic mode | 1.000000 | 1.000000 | 0.000000 | 逐元素完全一致 |
| 24 | 完整 239,629-token batch，官方 deterministic mode | 1.000000 | 1.000000 | 0.000000 | 逐元素完全一致 |
| 25 | 同一子集，仅确定性环境变量、不开 mode | 0.578118 | 1.074263 | 0.954954 | 部分改善但仍不稳定 |
| 2/16/17/19/20 | 完整 batch，recompute off，多种 packing/CP/allocator | OOM | OOM | OOM | 53K 真实样本在当前硬件不可行 |

### shared/dense 与 expert 参数

Megatron optimizer chain 0 对应非 expert-parallel 参数；chain 1 对应 expert-parallel 参数。

三次基线：

```text
                 shared/dense cosine   expert cosine
baseline 1              0.574939          0.525148
baseline 2              0.695909          0.652973
baseline 3              0.473862          0.432105
```

expert 参数通常更差，但 shared/dense 参数也明显受影响。这符合 MoE 层误差沿残差流和反向传播扩散的预期，不是只有 expert 权重局部抖动。

## 5. Routing replay 结论

### Training routing replay

训练侧记录 old-actor forward 的 expert IDs；当前 actor 的训练 forward 和 activation recomputation 重放这些 IDs，但从当前 router scores 重新 gather 对应概率，因此 router 权重仍有梯度。

这是正式训练可使用的语义：

```text
固定 expert IDs
重新计算当前 router probabilities
保留 router gradient
```

本实验 cosine 为 `0.526279`，说明只固定 Megatron 内部训练路由不能消除抖动。

### Full rollout routing replay

SGLang 返回每个实际 agent request、每个 token、每层 MoE 的 top-k expert IDs；Megatron 在 ref/old/current/train forward 中重放。代理层同时验证：

```text
route shape = (prompt_tokens + response_tokens - 1, num_layers, topk)
```

路由在磁盘/CPU 上保持 int32，进入 `torch.gather` 前转换为 int64，避免 5.6GB 路由文件翻倍，同时满足 PyTorch index dtype 合约。

full-routing 真实 batch 的 cosine 为 `0.638781`。它与旧 batch 不同，不能量化“提升多少”；但可排除“只要重放 rollout expert IDs，梯度就会接近 1”的假设。

## 6. Activation recomputation

no-recompute 尝试结果：

- CP=1，81K：约 139.23/139.81 GiB 后 OOM；
- CP=1，55K：最长 53,389-token singleton 使显存达到约 139.38 GiB，额外申请 460 MiB 失败；
- CP=1，55K + expandable-segments：仍在额外申请 168 MiB 时失败；
- CP=2，55K：仍在 MoE combine/dispatch 阶段 OOM；
- CP=2，55K + expandable-segments：第一阶段可推进，但后续申请 3.41 GiB 时仅剩 2.67 GiB，仍 OOM。

因此，在当前 140GB/GPU、Qwen3-30B-A3B、保留真实 53K 单样本的条件下，无法完成完整 batch 的 no-recompute 对照。

为进一步隔离因果，使用上一节定义的完整轨迹控制子集做了严格同批 A/B：

| 指标 | recompute on | recompute off |
|---|---:|---:|
| gradient cosine | 0.276396 | 0.423367 |
| norm ratio | 0.754776 | 1.816854 |
| relative L2 | 1.073524 | 1.662097 |
| 第一遍 backward | 37.1s | 207.2s |
| 第二遍 backward | 66.7s | 322.0s |

关闭 recompute 后方向 cosine 有所提高，但仍远离 1；梯度尺度和相对 L2 反而更差。因此不能把异常归因于 activation recomputation。它最多是与 MoE 非确定路径相互作用的放大/扰动因素。与此同时，no-recompute 慢约 `5.6×–4.8×`，显存峰值约 `138.8–142.0GB/GPU`，对正式 40K–53K 轨迹也不具备工程可行性。

### 官方 deterministic mode：已找到可行修复

此前所有 MoE 消融日志均显示：

```text
deterministic_mode = False
```

Slime 自带的 reproducibility 配置要求：

```text
--deterministic-mode
NCCL_ALGO=Ring
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_DEVICE_MAX_CONNECTIONS=1
```

当前启动器已增加默认关闭的 `MEGATRON_DETERMINISTIC_MODE` 开关，并确保这些环境变量在 Ray 初始化前透传给 worker。严格同批结果为：

| 配置 | gradient cosine | norm ratio | relative L2 | max abs diff |
|---|---:|---:|---:|---:|
| 普通模式，105,994 tokens | 0.276396 | 0.754776 | 1.073524 | 0.0928233 |
| 仅环境变量，不开 deterministic mode | 0.578118 | 1.074263 | 0.954954 | 0.0365696 |
| 完整 deterministic mode，同一 105,994-token 子集 | 1.000000 | 1.000000 | 0 | 0 |
| 完整 deterministic mode，239,629-token 原始 batch | 1.000000 | 1.000000 | 0 | 0 |

完整 batch 中，shared/dense chain 和 expert chain 的 `gradient_cosine`、`norm_ratio` 均为 `1.0`，所有梯度元素 `max_abs_difference=0`。因此无需再用逐层 fingerprint 搜索首个分叉算子：官方模式已使最终完整梯度 bitwise identical。

性能代价以同一完整 batch 比较：普通模式 backward 约 `41–42s / 76–77s`，确定性模式为 `57.5s / 99.8s`，训练侧约慢 `30%–38%`。正式 RL 还包含远程工具 rollout，端到端增幅会低于纯 backward 增幅，需要在下一次 smoke 中实测。

### 确定性模式下的逐 token logprob 复测

Case 26 在同一个 239,629-token 完整 batch 上直接采集两遍训练 forward 的 selected-token logprob。共包含 125,066 个 response token，其中 84,432 个进入实际 RL loss；其余 token 被 segment/loss mask 排除，不应用于 PPO，因此训推统计以 84,432 个有效 token 为主。

训练侧内部结果严格一致：

| 比较 | mean abs logprob diff | P99 | max | \\(|ratio-1|>1%\\) |
|---|---:|---:|---:|---:|
| Megatron current pass 1 vs pass 2 | 0 | 0 | 0 | 0 |
| Megatron old vs current pass 1 | 0 | 0 | 0 | 0 |
| Megatron old vs current pass 2 | 0 | 0 | 0 | 0 |

说明在零更新条件下，Megatron old/current ratio 对全部有效 token 都严格为 1；训练侧重复 forward 不再制造虚假的 PPO ratio。

SGLang rollout logprob 与确定性 Megatron old/current 仍存在跨后端差异：

| 指标 | 84,432 个有效 loss token |
|---|---:|
| mean signed logprob diff（Megatron - SGLang） | -0.001831 |
| mean abs logprob diff | 0.008204 |
| P50 / P90 / P95 / P99 abs diff | 3.58e-7 / 0.00642 / 0.04294 / 0.19961 |
| mean abs ratio error | 0.7995% |
| 有效 token 中 \\(|ratio-1|>1%\\) | 8.937% |
| \\(|ratio-1|>5%\\) | 4.544% |
| \\(|ratio-1|>10%\\) | 2.505% |

按 SGLang reference logprob 分桶后，误差明显集中于低置信 token：

| SGLang logprob 桶 | token 数 | 占比 | mean abs ratio error | P95 | >1% |
|---|---:|---:|---:|---:|---:|
| [-6,-3) | 24 | 0.028% | 38.81% | 92.80% | 100% |
| [-3,-1) | 1,799 | 2.131% | 15.03% | 40.15% | 94.94% |
| [-1,-0.3) | 2,907 | 3.443% | 8.87% | 24.80% | 90.78% |
| [-0.3,inf) | 79,702 | 94.398% | 0.172% | 0.616% | 3.984% |


这项修复只保证“给定相同训练 batch，Megatron 的梯度计算确定”。OpenClaw 多轮工具调用受外部工具结果和完成时序影响，整条在线 rollout 不要求 bitwise 重现；评测仍应单独固定 request seed、工具环境和任务顺序。

### veRL 后端核查

核查 veRL 官方主线 commit `7292a74fedcdb70bb8d2c5171bc9371ce1274119`：

- Qwen3-30B-A3B 的 Megatron recipe 使用 `model_engine=megatron`，底层仍是 Megatron-Core；
- 同一模型另有官方 `actor.strategy=fsdp2` recipe，属于不同训练后端；
- 官方支持矩阵同时列出 Qwen3-30B-A3B 的 FSDP、Megatron，以及部分平台的 VeOmni；
- veRL 自身也默认不保证 bitwise reproducibility，并提供与本次相近的 full-determinism 环境与 PyTorch deterministic 设置；
- veRL 官方 full determinism 文档明确表示多轮 tool-agent rollout 不能端到端 bitwise 重现，但这不妨碍对给定 batch 的训练计算做确定化。

官方证据：

- `examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh`
- `examples/grpo_trainer/run_qwen3_30b_a3b_fsdp.sh`
- `examples/grpo_trainer/README.md`
- `docs/advance/determinism.md`

## 7. 远端 gateway 端口隔离与异常恢复

### 原因

旧实现按固定公式分配 host-network 端口：

```text
base + train_offset + task_index × stride
```

异常退出时远端 Docker 可能未进入 `finally` 清理；下次运行会复用相同端口。由于 `/health` 不带认证，新进程可能把旧 gateway 判为健康，但携带新 token 的真实请求随后返回 401。

### 当前实现

正式训练默认使用 `run_scoped` 端口模式：

1. 候选端口由 `run_id + rollout_id + train/eval + task_index` 派生；
2. 分配前用远端 `ss` 探测监听状态；
3. 进程内维护预约集合，避免并发线程选中同一端口；
4. 冲突后按新的候选序列自动换号，默认最多 512 个候选；
5. Docker bind 或健康检查失败时，清理容器、释放端口并重新启动，默认 3 次；
6. 401、连接错误和远端 curl 错误作为可恢复环境异常处理，不再终止整个 batch；
7. 错误日志不序列化 Authorization header，避免泄露 gateway token；
8. 远端 Docker 瞬时启动并发默认统一为 4；
9. `static` 模式仅保留作显式兼容/诊断，不再是正式默认。

旧遗留容器没有被删除；新 allocator 会探测并绕开其监听端口，避免误伤远端其他任务。

最终定向回归（同一次 pytest 调用）：

```text
端口/异常、proxy、token 对齐、routing replay、CP-aware packing
合计                                                     78/78 passed
```

实际 full-routing rollout 使用隔离端口后，8/8 轨迹网关成功启动，未出现 `Address already in use` 或 401。

## 8. 对正式训练的建议

1. 保留 run-scoped 端口、三次 gateway 启动恢复和容器启动并发 4；
2. 保留 token-ledger、compaction/pruning 正确分段、multi-tool 完整执行和一次 rollout 一次 optimizer update；
3. 下一次正式 RL 启用 `MEGATRON_DETERMINISTIC_MODE=1` 及四个必需环境变量；activation recomputation 保持开启；
4. routing replay 可以继续用于 SGLang→Megatron 的训推路由对齐，但不再承担训练侧重复 backward 稳定性的职责；
5. 继续监控 router agreement、old/current ratio 分桶、pre/post clip grad norm、NaN/Inf、reward/KL/entropy 联合异常；
6. 不因单次 grouped-GEMM 结果改变正式配置；deterministic mode 已在保留现有 fused/recompute 配置时达到 bitwise 一致；
7. 暂不迁移 veRL。若未来比较框架，必须选择 FSDP2/VeOmni 等不同后端；veRL+Megatron 不是有效的后端消融；
8. 下一次正式 RL 应固定可复现 eval request seed，并将训练中 eval 轨迹完整保存，避免训练后无法复现 eval 分数。

## 9. 复现命令与证据路径

通用 saved-batch 诊断：

```bash
cd /root/openclaw-rl-local
env USE_ROUTING_REPLAY=0 \
  bash scripts/run_framework_numerical_ablation_one.sh \
  outputs/framework-numerical-ablation-step99-20260824/configs/01_rng_restore.json
```

Full rollout routing replay：

```bash
cd /root/openclaw-rl-local
bash scripts/run_framework_full_rollout_routing_replay.sh
```

CP=2 recomputation 配对：

```bash
cd /root/openclaw-rl-local
bash scripts/run_framework_cp2_recompute_pair.sh
```

同一 saved batch、完整轨迹子集的 recompute 严格 A/B：

```bash
cd /root/openclaw-rl-local
bash scripts/run_framework_recompute_subset_pair.sh
```

控制子集构建脚本：

```text
/root/openclaw-rl-local/scripts/build_numerical_recompute_subset.py
```

官方 deterministic mode，子集与完整 batch：

```bash
cd /root/openclaw-rl-local
bash scripts/run_framework_deterministic_mode_subset.sh
bash scripts/run_framework_deterministic_mode_full_batch.sh
```

环境变量单独对照：

```bash
cd /root/openclaw-rl-local
bash scripts/run_framework_deterministic_env_only_subset.sh
```

主结果目录：

```text
/root/openclaw-rl-local/outputs/framework-numerical-ablation-step99-20260824/
```

关键实现：

```text
/root/openclaw-rl-local/clawgym-official/clawgym_rl_api_server.py
/root/openclaw-rl-local/clawgym-official/clawgym_rl_rollout.py
/root/openclaw-rl-local/slime/slime/backends/megatron_utils/numerical_consistency_diag.py
/root/openclaw-rl-local/slime/slime/backends/megatron_utils/actor.py
/root/openclaw-rl-local/slime/slime/backends/megatron_utils/data.py
/root/openclaw-rl-local/slime/slime/utils/routing_replay.py
```

所有数值诊断均为 opt-in；未设置 `SLIME_NUMERICAL_DIAG_CONFIG` 时不会进入双 backward 分支。

## 10. SGLang–Megatron 跨后端不一致：routing replay 与 TIS 对照

### 10.1 实验口径

三组实验均使用 step 99 同一 checkpoint 和同一份真实 agent rollout 子集：4 个轨迹、4 个 request sample，共 52,447 个 token，其中 14,763 个为有效 response/loss token。Megatron 开启 deterministic mode 并恢复 RNG，对同 batch 做两次完整 backward，不执行 optimizer/scheduler step。

`routing replay` 强制 Megatron 在每层使用 rollout 保存的 SGLang expert IDs。`TIS` 在真实训练 loss 路径中使用：

```text
w_t = clip(exp(logp_megatron_old - logp_sglang_rollout), 0, 2)
```

### 10.2 主要结果

| 配置 | logprob MAE | logprob RMS | P99 abs diff | mean \|ratio-1\| | \|ratio-1\| > 5% | ratio 超出 [0.8, 1.28] | IS ESS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 确定性，无 replay/无 TIS | 0.013308 | 0.062910 | 0.232825 | 1.2864% | 6.9972% | 1.0228% | 99.7358% |
| 确定性 + full routing replay | 0.006784 | 0.025230 | 0.119062 | 0.6773% | 4.3013% | 0.1355% | 99.9366% |
| 确定性 + TIS [0,2] | 0.013308 | 0.062910 | 0.232825 | 1.2864% | 6.9972% | 1.0228% | 99.7454%* |

`*` TIS 行为裁剪后权重 ESS。TIS 不改变前向 logprob 差异，因此前几列与无 replay 相同；它修正的是 loss 权重。

full routing replay 相对无 replay：logprob MAE 下降 49.0%，RMS 下降 59.9%，P99 下降 48.9%，最大绝对 logprob 差由 3.1765 降至 0.5783；`|ratio-1| > 10%` 的 token 由 4.0913% 降至 1.4970%，ratio 超出 `[0.8,1.28]` 的 token 由 1.0228% 降至 0.1355%。它的主要价值是压低长尾，但 attention/GEMM/kernel 等残余差异仍然存在。

TIS `[0,2]` 的权重均值为 0.999712，裁剪后 ESS 为 99.7454%，只有 1/14,763 token（0.00677%）触及上界 2。真实 loss 路径的梯度范数由无 TIS 的 0.695757 变为 0.692275，差异约 -0.50%，属于温和校正。

三组的 Megatron old/current logprob 均逐 token 完全相同，两次 backward 的 gradient cosine 均为 1.0，relative L2 error 均为 0。TIS 和 routing replay 没有破坏已修复的训练侧确定性。

### 10.3 结论与正式训练建议

1. 保留 Megatron deterministic mode，解决训练侧重复 forward/backward 本身的不稳定。
2. 启用 full rollout routing replay；它是本轮对 SGLang–Megatron MoE 跨后端差异最直接、收益最大的修复。
3. replay 之后保留 TIS `[0,2]` 作为残余 behavior-policy 校正。replay 对照中离线计算的 TIS clipfrac 为 0，对当前 batch 不会造成激进重加权。
4. 两者互补：routing replay 修正 MoE 路由，TIS 校正剩余概率差。
5. 本轮是 4 条真实 request sample 的受控实验，足以确认方向和代码路径，但不代表正式训练的吞吐、存储成本和全分布效果。正式重跑前应再做一次短程组合 smoke，同时开启 deterministic + routing replay + TIS，检查吞吐、router agreement、TIS ESS 和 reward/KL/entropy。

原始结果：

```text
/root/openclaw-rl-local/outputs/framework-numerical-ablation-step99-20260824/27_deterministic_routing_subset_no_replay/gradient_pair.json
/root/openclaw-rl-local/outputs/framework-numerical-ablation-step99-20260824/28_deterministic_routing_subset_full_replay/gradient_pair.json
/root/openclaw-rl-local/outputs/framework-numerical-ablation-step99-20260824/29_deterministic_routing_subset_tis/gradient_pair.json
```
