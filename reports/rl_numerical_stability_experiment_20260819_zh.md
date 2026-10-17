# OpenClaw-RL 数值稳定性实验报告

## 1. 实验目的

本实验评估 Qwen3-30B-A3B 在 OpenClaw-RL 训练链路中的数值一致性，重点回答三个问题：

1. SGLang rollout 与 Megatron 训练对相同 token 计算的 logprob 是否一致；
2. Megatron 在参数完全不更新时，重复 forward 的 PPO ratio 是否仍为 1；
3. 小幅 forward 差异是否会放大为显著的完整梯度差异。

## 2. 测试对象与数据口径

- Checkpoint：正式 RL step 99；
- Rollout：2 个不同任务，每题 4 条轨迹，共 8 条原始轨迹；
- 长轨迹拆分后：10 个训练 sample；
- GRPO 移除一个恒定 reward、advantage 为 0 的任务组后：6 个有效 sample；
- 有效 response loss token：84,432；
- 所有对照使用完全相同的 token IDs、loss masks、advantages、样本顺序及动态 batch 切分；
- attention dropout 和 hidden dropout 均为 0；
- 零更新测试不执行 `optimizer.step()`，参数保持不变。

本报告中的逐 token ratio 定义为：

```text
delta_logprob_t = logprob_2,t - logprob_1,t
ratio_t         = exp(delta_logprob_t)
                = probability_2,t / probability_1,t
```

参数完全未更新时，理论期望为：

```text
delta_logprob_t = 0
ratio_t         = 1
```

`abs(ratio-1)>1%` 表示同一 token 的两次相对概率估计相差超过 1%。它是数值敏感性指标，不等同于 PPO clipping。

## 3. 实验方法与实际执行流程

### 3.1 运行拓扑与关键配置

本次不是用一个脱离训练环境的小模型脚本近似模拟，而是加载正式 RL step 99 checkpoint，经过当时真实的 Slime + SGLang + Megatron 链路执行。日志记录的拓扑为：

| 组件 | 配置 |
|---|---|
| 模型 | Qwen3-30B-A3B MoE，BF16 |
| 训练侧 | 4 GPU，Megatron TP=4、DP=1、PP=1、CP=1 |
| Rollout 侧 | 4 GPU，2 个 SGLang engine，每个 TP=2 |
| 拓扑模式 | non-colocate；训练和 rollout 分别使用 4 张卡 |
| GRPO | 每个 prompt 采样 4 条轨迹 |
| Rollout sampling | temperature=0.7；模型 generation config 中 top_p=0.8、top_k=20 |
| PPO clipping | eps_clip=0.20，eps_clip_high=0.28，eps_clip_c=None |
| 训练分批 | dynamic batch 开启，micro_batch_size=1，max_tokens_per_gpu=16,384 |
| Dropout | attention dropout=0，hidden dropout=0 |
| Rollout logprob | `use_rollout_logprobs=false` |
| Rollout routing replay | `use_rollout_routing_replay=false` |

基线组的 `use_routing_replay=false`，对照组的 `use_routing_replay=true`。这里的 replay 仅是 **Megatron 训练侧重复 forward 之间的路由复用**，不是记录并重放 SGLang rollout 时的真实 expert 路由，因此不能等同于 R3 论文中的完整 rollout routing replay。

### 3.2 如何构造并固定测试 batch

基线作业首先在 SGLang/OpenClaw 中完整采样：

```text
2 个不同 RL 任务
  × 每题 4 条 rollout
= 8 条原始 agent 轨迹
  → 长轨迹按训练上下文规则切分
= 10 个训练 sample/segment
```

完整的 rollout 数据被保存为：

```text
outputs/zero-update-consistency-step99-20260814/
  baseline/rollout_data/100.pt
```

该文件保留了之后训练所需的 token IDs、loss masks、reward/group 信息、advantages、SGLang rollout logprobs 及 sample 元数据。对照作业不重新 rollout，而是通过：

```text
load_debug_rollout_data = baseline/rollout_data/{rollout_id}.pt
```

直接读取这个固定 batch。因此 A/B 两组不受采样随机性、工具返回变化、轨迹分叉或任务难度差异影响。

进入 GRPO 后，其中一个任务组的 4 条 rollout reward 完全相同，组内标准化后 advantage 全为 0，因而按正式训练逻辑不贡献 policy gradient。该组对应 segment 被移除后，最终为：

```text
保留 sample              = 6
存储的 response 候选位置 = 125,066
loss_mask=0 位置          = 40,634
真正参与 loss 的 token    = 84,432
其中 positive advantage   = 30,710
其中 negative advantage   = 53,722
```

所有 logprob、ratio、分位数和 clipping 统计都只在这 84,432 个 `loss_mask=1` 的模型 response token 上进行，不把 padding、环境或其他已 mask 位置混入分母。


#### Mask 语义与 veRL agent 对齐

这里需要严格区分“response 连续后缀范围”和“agent response mask”：

| 概念 | veRL agent | 本项目 |
|---|---|---|
| 初始 prompt 后的完整序列 | `response_ids`，包含 LLM token 和 tool response token | Sample 的 response span，由 `response_length=len(loss_mask)` 描述 |
| 真正由策略生成、参与训练的 token | `response_mask=1` | `loss_mask=1` |
| 工具结果、环境增量 | `response_mask=0` | `loss_mask=0`，rollout logprob 用 `0.0` 占位 |
| Padding/序列有效性 | response attention/padding mask | packing/padding 与长度信息 |

因此，在 agent 场景中，veRL 的 `response_mask` 与本项目的 `loss_mask` 语义一致；不能把 veRL agent `response_mask` 误解为“完整 response span 中所有非 padding 位置都为 1”。例如：

```text
response token:         assistant | tool result | assistant
veRL response_mask:         1     |      0      |     1
本项目 loss_mask:           1     |      0      |     1
response padding mask:      1     |      1      |     1
```

本报告以 `loss_mask` 选出的 84,432 个 token 计算 Pearson、CCC、ratio 和 clipping，正好对应 veRL agent 指标中的 `response_mask=1` 语义，而不是对官方口径做了额外放宽或收紧。40,634 个 `loss_mask=0` 位置属于有效上下文，但不是模型动作，不应进入策略一致性或 policy-loss 分母。
### 3.3 SGLang–Megatron 训推比较是怎样做的

对每个有效 token `t`，rollout 文件中已保存 SGLang 在真实生成该 token 时的 logprob：

```text
logp_rollout[t]
```

然后 Megatron 加载同一 step 99 权重，对已经固定的完整 token 序列做 teacher-forced forward，得到：

```text
logp_megatron_old[t]
```

比较量为：

```text
delta_train_rollout[t] = logp_megatron_old[t] - logp_rollout[t]
ratio_train_rollout[t] = exp(delta_train_rollout[t])
```

这个比较不需要两边重新生成同样的文本：被评分的 token IDs 已经固定，比较的是两个 backend 对同一 token、同一历史前缀给出的条件概率。

### 3.4 Megatron 严格零更新重复 forward

这一部分专门排除 optimizer update：

```text
1. 加载 step 99 checkpoint 和固定 rollout_data/100.pt
2. 用 Megatron 计算一次 old_logprob
3. 不执行 backward、gradient clipping 或 optimizer.step()
4. 参数保持不变，对同一 batch 连续 forward 5 次
5. 每次用 repeat_logprob[k] 与第 2 步的 old_logprob 比较
6. 分别统计 5 次结果，报告主表取 5 次均值
```

对第 `k` 次重复 forward：

```text
delta[k,t] = logp_repeat[k,t] - logp_old[t]
ratio[k,t] = exp(delta[k,t])
```

基线与 replay 两个作业使用同一个已保存 batch，并逐 tensor 验证了 tokens、loss masks、长度、group/sample index、advantages 和 SGLang logprobs 完全相等。两组的核心自变量只有：

```text
A: use_routing_replay = false
B: use_routing_replay = true
```

因此这个 A/B 回答的是“训练侧 routing replay 能否稳定 Megatron 的重复计算”，不是“完整 R3 能否消除 SGLang–Megatron 训推差异”。
日志中的实际 dynamic-batch 结果为 `num_samples=6`、`num_microbatches=[6]`、`sum_tokens=239,629`，其中最长一个 micro-batch/segment 为 53,389 tokens。重复 forward 不会重新 tokenize 或重新生成：数据迭代器只把读取 offset 重置到 0，再按完全相同的 `micro_batch_indices` 读取同一批对象；CP slicing、padding 和 batch tensor 构造只产生临时 tensor，不改写保存的原始 token tensor。因此两遍比较中的第 `t` 个位置确实对应同一个 token ID 和同一个 loss mask，而不是“文本相同但 tokenization 发生变化”。


### 3.5 假 PPO clipping 的精确判定

没有参数更新时，任何 old/current ratio 偏离 1 都来自重复计算差异。但“ratio 超出区间”不一定真正改变 PPO surrogate，还要看 advantage 方向。本次按当时的非对称边界精确计数：

```text
advantage > 0  且 ratio > 1.28
或
advantage < 0  且 ratio < 0.80
```

这被命名为“方向性假 PPO clipping”：如果 forward 完全可重复，且参数没有更新，该值理论上应为 0。

### 3.6 零更新双 backward 是怎样比较的

双 backward 测试再次加载同一个 `rollout_data/100.pt`，并开启训练侧 routing replay。运行流程为：

```text
Pass 1: 同一 tokens/masks/advantages/old_logprobs
        → 完整 forward
        → backward
        → optimizer.prepare_grads()
        → 单独捕获 FP32 主梯度 g1
        → 跳过 optimizer.step()

Pass 2: 重新从同一 batch 起点执行
        → 重置 routing replay forward/backward index 为 0
        → 完整 forward/backward/prepare_grads()
        → 单独捕获 FP32 主梯度 g2
        → 跳过 optimizer.step() 和 scheduler step
```

比较的不是 BF16 模型参数上临时的局部 gradient buffer，而是执行正式 `optimizer.prepare_grads()` 之后，precision-aware + CPU-offload optimizer 真正会用于更新的 FP32 主梯度。所有结果在 gradient clipping 之前读取。

4 个 TP rank 各自持有局部梯度分片，每个 rank 的局部元素约为 7.63B；报告中的 30,532,122,624 是全部 rank 上的分布式元素总数。不需要把 30.5B 个元素集中到一张 GPU，只需对各 rank 的局部点积、平方和以及最大值做分布式归约：

```text
dot        = sum_ranks(sum(g1_local * g2_local))
norm1^2    = sum_ranks(sum(g1_local^2))
norm2^2    = sum_ranks(sum(g2_local^2))
diff^2     = sum_ranks(sum((g2_local-g1_local)^2))
max_abs    = max_ranks(max(abs(g2_local-g1_local)))

cosine     = dot / (norm1 * norm2)
rel_l2     = sqrt(diff^2) / norm1
norm_ratio = norm2 / norm1
```

同时按 optimizer chain 分别计算：chain 0 是 dense/共享参数链，chain 1 是 MoE expert 主链。第一次完整 pass 耗时 71.6s，第二次为 92.1s。

### 3.7 安全性与对 checkpoint 的影响

两类严格测试都在 `optimizer.step()` 之前返回，不执行 scheduler step，不保存新 checkpoint，也不覆盖 step 99 checkpoint。测试所用的一次性 instrumentation 在实验后已移除，`actor.py` 和 `model.py` 已恢复到测试前版本。

因此，当前仓库不保留一条可直接重跑 instrumentation 的命令；可审计材料是已固化的 rollout、诊断 tensor/JSON 和完整运行日志。如需要在其他 checkpoint 上复制该实验，应将诊断逻辑重做为受版本控制的独立测试工具，而不是直接修改正式训练流程。

### 3.8 指标速查与正确读法

#### Token 概率指标

| 指标 | 定义或计算 | 理想值 | 如何理解 |
|---|---|---:|---|
| Logprob | 模型赋给实际 token 的自然对数概率 `log(p)` | 两次计算相同 | 越接近 0，模型认为该 token 越自然；数值越负，token 越意外。 |
| Logprob delta | `logp_2 - logp_1` | 0 | 同一个 token 在两次计算中的对数概率变化。它保留正负方向。 |
| Mean absolute logprob difference | `mean(abs(delta))` | 0 | 平均误差幅度。它容易被大量正常 token 稀释，不能单独反映长尾。 |
| RMS logprob delta | `sqrt(mean(delta²))` | 0 | 对大误差比 MAE 更敏感；RMS 明显高于 MAE 通常说明存在长尾。 |
| P50/P90/P95/P99 | 绝对误差的分位数 | 0 | P99=0.196 表示 99% token 的绝对误差不超过 0.196，最差约 1% 至少达到该量级。 |
| Probability ratio | `ratio=exp(delta)` | 1 | 两次计算对同一 token 概率的倍数关系。delta 较小时，`ratio-1≈delta`。 |
| `abs(ratio-1)>X%` | 概率比偏离 1 超过阈值的 token 比例 | 0% | 直接说明多少 token 的概率相对偏差超过 1%、10% 或 20%。 |
| Ratio 超出 `[0.8,1.28]` | 不考虑 advantage 符号，仅检查是否越过任一 PPO 边界 | 0% | 是数值风险上界，不等于这些 token 都真正改变 PPO surrogate。 |
| 方向性假 PPO clipping | 正 advantage 且 ratio>1.28，或负 advantage 且 ratio<0.8 | 0% | 零更新条件下本应为 0；它最直接衡量重复计算误差实际污染 PPO clipping 的比例。 |

例如 `delta=0.01` 时：

```text
ratio = exp(0.01) ≈ 1.01005
```

即该 token 的概率相对偏移约 1.005%。但均值为 0.01 不代表每个 token 都偏移 1%，也不代表 99% token 一致。

#### 相关性与一致性指标

| 指标 | 关注的问题 | 理想值 | 容易误读的地方 |
|---|---|---:|---|
| Logprob Pearson（本报告补算） | 两组 logprob 是否按相同线性趋势升降 | 1 | `Pearson=0.998` 不表示 99.8% token 相等。只要 `y=ax+b, a>0`，Pearson 都可以等于 1。 |
| Probability Pearson（veRL 官方） | 先对 logprob 取 `exp` 得到实际 token probability，再计算相关系数 | 1 | 高概率 token 权重更突出，低概率 token 的 logprob 长尾会被压缩，因此通常高于 logprob Pearson。 |
| Lin's CCC | 两组数值是否同时具有相近趋势、均值和尺度 | 1 | 比 Pearson 更严格，但仍是全局汇总，也可能掩盖少量 P99 长尾。 |
| 回归斜率 | 拟合 `y=slope*x+intercept` 时的尺度关系 | 1 | 大于 1 表示 y 的变化幅度整体更大；小于 1 表示整体被压缩。 |
| 回归截距 | 上述线性拟合中的整体偏移 | 0 | 非零说明存在系统性平移；它不能描述局部 token 长尾。 |

Pearson 的定义为：

```text
corr(x,y) = cov(x,y) / (std(x) * std(y))
```

CCC 的定义为：

```text
CCC = 2*cov(x,y) /
      (var(x) + var(y) + (mean(x)-mean(y))²)
```

veRL 官方实现实际计算 `corr(exp(actor_logprob), exp(rollout_logprob))`；本报告同时保留更敏感的 logprob Pearson。Logprob Pearson 主要检查“容易的 token 是否仍容易、困难的 token 是否仍困难”；CCC 还会惩罚整体均值偏移和尺度变化。本实验中 logprob Pearson 与 logprob CCC 很接近，说明主要问题不是简单的全局平移或缩放，而是局部扰动和长尾。

#### 梯度指标

| 指标 | 定义或计算 | 理想值 | 如何理解 |
|---|---|---:|---|
| Pre-clip grad norm | 梯度裁剪前的全局 L2 norm | 两遍相同 | 表示更新幅度，不表示更新方向。 |
| Grad norm ratio | `norm(g2)/norm(g1)` | 1 | 1.218 表示第二遍梯度范数比第一遍高约 21.8%。 |
| Gradient cosine | `dot(g1,g2)/(norm(g1)*norm(g2))` | 1 | 比较完整梯度方向；1 为同向，0 为正交，-1 为反向。 |
| Relative L2 error | `norm(g2-g1)/norm(g1)` | 0 | 0.865 表示差分向量范数达到第一遍梯度范数的 86.5%。 |
| Max absolute difference | 所有梯度元素中最大的绝对差 | 0 | 检查单元素极端误差，但容易受参数尺度影响。 |
| 分布式梯度元素数 | 参与全局比较的 FP32 主梯度元素总数 | — | 这是统计覆盖范围，不是模型新增了同等数量的独立参数。 |

Gradient cosine 和 norm ratio 必须一起看。例如 `g2=2*g1` 时 cosine 仍为 1，但更新幅度已经扩大一倍。当前保存的梯度产物没有逐元素梯度或元素和，因此可离线复算 cosine、norm、relative L2 等既有汇总，但无法从现有文件精确反推 gradient Pearson。

#### 推荐的联合判断方式

没有单一指标可以完整回答“训推是否一致”。本报告采用以下组合：

```text
Pearson / CCC       → 整体趋势与整体一致性
MAE / RMS           → 平均误差及大误差敏感度
P95 / P99           → 长尾
ratio 阈值比例       → token 概率偏移规模
方向性 clipping     → 对 PPO surrogate 的实际影响
cosine / relative L2→ 最终梯度方向与整体差异
```

其中任何一个指标很好，都不能自动抵消其他指标暴露的风险；例如 Pearson 很高但 P99、ratio 长尾或 gradient cosine 较差时，仍不能判定数值链路稳定。


## 4. SGLang–Megatron 训推一致性

在相同的 84,432 个有效 token 上，比较 SGLang rollout logprob 与 Megatron 重算的 old logprob：

| 指标 | 结果 |
|---|---:|
| Mean absolute logprob difference | 0.008219 |
| P50 / P90 / P95 / P99 | 3.58e-7 / 0.00623 / 0.04252 / 0.19610 |
| `abs(ratio-1)>1%` | 8.944% |
| Ratio 落在 `[0.8, 1.28]` 之外 | 0.694% |
| 按 advantage 方向实际触发 clipping | 0.355% |

结果表明误差不是均匀分布：中位 token 几乎完全一致，但 P95/P99 存在明显长尾。主要来源可能包括推理与训练后端的 kernel、精度、批处理方式、并行归约顺序和 MoE dispatch 差异。

使用保存的逐 token tensor 和真实 loss mask，在 CPU 上以 float64 离线计算相关性与一致性：


| 比较 | Logprob Pearson | Probability Pearson（veRL） | Logprob CCC | 回归斜率 | 截距 |
|---|---:|---:|---:|---:|---:|
| SGLang rollout → Megatron old（baseline） | 0.986743 | 0.993232 | 0.986435 | 1.010813 | -0.001065 |
| SGLang rollout → Megatron old（replay 作业重新加载） | 0.988157 | 0.993323 | 0.987866 | 1.011481 | -0.001079 |

第二行不是 rollout-routing replay 的效果评估：训练侧 replay 不会固定 SGLang 路由，而且该行的 Megatron old logprob 来自一次独立重新加载。两行仅说明训推相关性处于同一量级。Logprob Pearson 与 CCC 非常接近，表明主要问题不是整体均值或尺度偏移，而是局部 token 误差和长尾。veRL probability Pearson 更高，是因为 `exp` 将很负的 logprob 压缩到接近 0，对低概率 token 长尾不如 logprob 口径敏感。


当前正式配置未启用 `use_rollout_logprobs`，PPO denominator 由 Megatron 重算。这会防止 SGLang–Megatron logprob 差异**直接**把 PPO old/current ratio 从 1 推开，但并没有消除行为策略不一致：轨迹仍由 SGLang 策略采样，而 loss 用 Megatron 策略重算。因此该差异不能仅视为无影响的监控数值；直接改用 rollout logprob 也不是完整修复，需要 rollout routing replay 或严谨的 importance-sampling correction 对照。

## 5. Megatron 零更新重复 Forward

实验先计算一次 Megatron old logprob，再在同权重、同输入下连续 forward 5 次，并对 5 次结果取平均。

| 指标 | 无 routing replay | 开启 routing replay | 变化 |
|---|---:|---:|---:|
| Mean absolute logprob delta | 0.006067 | 0.002960 | -51.2% |
| RMS logprob delta | 0.03368 | 0.01659 | -50.7% |
| P90 absolute delta | 0.003370 | 0.000660 | -80.4% |
| P95 absolute delta | 0.03049 | 0.01152 | -62.2% |
| P99 absolute delta | 0.14991 | 0.08482 | -43.4% |
| `abs(ratio-1)>1%` | 7.727% | 5.254% | -32.0% |
| `abs(ratio-1)>10%` | 1.800% | 0.677% | -62.4% |
| `abs(ratio-1)>20%` | 0.520% | 0.0668% | -87.2% |
| 方向性假 PPO clipping | 0.1767% | 0.00782% | -95.6% |

理论上，参数不更新时 ratio 应严格接近 1。实际结果说明 Megatron MoE 训练路径本身存在稀疏长尾非确定性。

同一批已保存 tensor 的 CPU 离线结果如下；repeat 数值为 5 次 forward 的均值及范围：


| 指标 | 无 routing replay | 开启 routing replay |
|---|---:|---:|
| Logprob Pearson | 0.992224（0.991859–0.992695） | 0.998111（0.998020–0.998208） |
| Probability Pearson（veRL） | 0.996036（0.995976–0.996116） | 0.998910（0.998858–0.998952） |
| Logprob CCC | 0.992222（0.991858–0.992693） | 0.998110（0.998018–0.998208） |
| 回归斜率 | 0.992635（0.990460–0.995880） | 0.998987（0.997901–1.000061） |
| 截距 | -0.000487（-0.000631–-0.000350） | -0.000137（-0.000152–-0.000119） |

训练侧 routing replay 明显提高了重复 forward 的整体一致性，但 0.9981 仍不是逐 token 完全一致；前表的 P95/P99、ratio 阈值和方向性 clipping 继续用于描述相关性指标会掩盖的局部长尾。


Routing replay 固定了重复计算时的 top-k expert ID，显著降低了极端 ratio 偏移和方向性假 clipping；但仍有 5.254% 的有效 token 出现超过 1% 的相对概率偏移，说明剩余误差还来自 expert 内部计算、token dispatch、TP/EP collective、activation recomputation 或其他非确定性 kernel。

## 6. 零更新双 Backward 梯度一致性

在开启 routing replay、输入和参数完全一致的条件下，执行两次完整 forward/backward，并在 `optimizer.step()` 前比较实际进入 precision-aware CPU-offload optimizer 的 FP32 主梯度。

| 指标 | 结果 |
|---|---:|
| 比较的分布式 FP32 主梯度元素 | 30,532,122,624 |
| 第一次 pre-clip grad norm | 0.265066 |
| 第二次 pre-clip grad norm | 0.322926 |
| Grad norm ratio | 1.21829 |
| Gradient cosine similarity | 0.712445 |
| Relative L2 error | 0.865041 |
| 最大单元素绝对差 | 0.026847 |

分参数链结果：

| 参数链 | Gradient cosine | Relative L2 error |
|---|---:|---:|
| Dense/共享参数链 | 0.722725 | 0.849984 |
| MoE expert 主链 | 0.685930 | 0.902808 |

结果不能视为“基本一致”。虽然 routing replay 后方向性假 clipping 仅为 0.00782%，两次完整梯度的方向和范数仍存在显著差异。

原因是 GRPO 梯度由大量带正负 advantage 的 token 梯度相加而成。逐 token 权重和激活的稀疏扰动可能改变正负项抵消后的残差；同时，MoE forward 的差异还会改变激活及 backward Jacobian。因此，低 clipping 污染不等价于完整梯度稳定。

## 7. 综合判断

1. **SGLang–Megatron 差异真实存在，且只被部分隔离。** Megatron 重算 old logprob 避免它直接污染 PPO ratio，但不消除 rollout 行为策略与训练策略的 mismatch。
2. **Megatron 内部零更新非确定性真实存在。** 未开启 routing replay 时，平均 0.1767% 的 token 会发生方向性假 clipping。
3. **Routing replay 有明确收益，但不是完整修复。** 它将方向性假 clipping 降低 95.6%，但无法充分稳定完整 backward 梯度。
4. **梯度稳定性风险高于单纯 clipfrac 所显示的风险。** 单 batch 中 gradient cosine 仅为 0.712，说明数值扰动可能实质影响更新方向。
5. **现有证据尚不能单独证明此前 RL 效果不佳由该问题造成。** 当前结论来自一个 checkpoint、一个固定 batch，需要多 batch、多 checkpoint 复验后再量化对最终训练收益的贡献。

## 8. 建议

### 正式训练最低配置

- 保持 `use_rollout_logprobs=false`；
- 训练侧 `--use-routing-replay` 显著降低极端 ratio 和假 clipping，但在正式启用前应补充多 batch 梯度对照；
- 不应把 routing replay 视为完整的训推一致性方案；
- 持续记录 `train_rollout_logprob_abs_diff`、`pg_clipfrac`、pre/post-clip grad norm 及非有限值。

### 下一轮诊断优先级

1. 固定并恢复 CUDA、模型及数据迭代 RNG state，重复双 backward；
2. 对照关闭 activation recomputation；
3. 对照不同 MoE dispatcher、GEMM 与 collective 配置；
4. 在多个 checkpoint、多个真实 batch 上统计 gradient cosine 和 relative L2 error；
5. 优先测试完整 SGLang→Megatron rollout-routing replay，并与训练侧 replay 严格区分；
6. 如果无法完整回放 rollout 路由，再对照 TIS/MIS 等 correction，不应只切换 `use_rollout_logprobs` 就视为训推对齐。

## 9. 实验边界

- 本报告的严格 forward/backward 结果来自 step 99 的一个固定 batch；
- 未覆盖不同任务长度、不同 reward 分布和其他 checkpoint；
- 双 backward 测试未额外重置 RNG state；dropout 已关闭，但潜在随机算子和非确定性 kernel 仍属于当前正式执行路径的一部分；
- 测试 instrumentation 已移除，未修改或覆盖正式 checkpoint。

## 10. 原始产物

- `outputs/zero-update-consistency-step99-20260814/ZERO_UPDATE_CONSISTENCY_REPORT_ZH.md`
- `outputs/zero-backward-consistency-step99-20260814/ZERO_BACKWARD_CONSISTENCY_REPORT_ZH.md`
- `outputs/clawgym-step99-logprob-smoke-20260812/STEP99_LOGPROB_SMOKE_REPORT_ZH.md`

- `outputs/zero-update-consistency-step99-20260814/baseline/rollout_data/100.pt`
- `outputs/zero-update-consistency-step99-20260814/baseline/zero_update_baseline.pt`
- `outputs/zero-update-consistency-step99-20260814/replay/zero_update_replay.pt`
- `outputs/zero-backward-consistency-step99-20260814/zero_backward_routing_replay.json`
- `outputs/zero-backward-consistency-step99-20260814/zero_backward_routing_replay.pt`

关键完整日志：

- `outputs/zero-update-consistency-step99-20260814/baseline/logs/train_20260813-185420.log`
- `outputs/zero-update-consistency-step99-20260814/replay/logs/train_20260813-191152.log`
- `outputs/zero-backward-consistency-step99-20260814/logs/train_20260814-070803.log`
