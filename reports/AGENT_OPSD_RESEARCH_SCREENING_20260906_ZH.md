# Agent OPSD / Policy Optimization：前沿筛选与下一步研究建议

检索日期：2026-09-06。本文是研究建议，不是已验证的新算法，也不构成启动训练的记录。

## 结论

本轮只保留一个主候选：**将 privileged self-teacher 用作 reward policy gradient 的控制变量，而不是动作正确性裁判**。

工作标题可暂写为 **Teachers as Control Variates, Not Credit Judges**。标题只是定位，不代表已经确认首创。

核心问题是：**当 hindsight teacher 的局部判断不够可信时，能否仍利用其概率分布降低终局奖励梯度的采样方差，同时不额外改变基线更新的期望？**

它与我们之前的 centered / matched-placebo OPSD 有实质区别：不再把一个条件 logprob 差直接解释为动作信用，不再依靠“判断正确后再门控”，而是构造条件期望为零的梯度修正。

建议状态：**值得做有退出条件的预实验；尚不建议直接进行 100-step 正式训练。** 目前没有证据能保证它提高 ClawGym 分数。控制变量本身是经典技术，只有证明其在长程 agent self-distillation 中确有独立价值，才可能成为算法论文。

## 1. 最近的工作已经覆盖了什么

下表依据论文原文、摘要及作者仓库；“接近”是本次研究判断，不代表所有实现和假设相同。很多是近期预印本，其性能声明未由我们独立复现。

| 方向 / 论文 | 已有核心内容 | 对我们选题的影响 |
|---|---|---|
| [OpenClaw-RL](https://arxiv.org/abs/2603.10165) | 从 next-state feedback 提取 hindsight hint，形成 OPD 与 RL 的组合 | “工具反馈 + self-teacher + RL”已直接进入 Claw agent，不能作为新核心 |
| [SSPO](https://arxiv.org/abs/2608.12764) | Evidence Anchors、错误答案反馈、失败轨迹的 step-level 重加权 | 我们原先的方案主要属于适配，不能只凭换数据集主张算法创新 |
| [OCSD](https://arxiv.org/html/2608.04788v1) | 对比完整 observation 与保持结构的 observation-ablated teacher context，校准蒸馏信号 | 与 matched-placebo 的动机和机制很接近；换一种 placebo 不足以区分 |
| [TASPO](https://arxiv.org/html/2608.31077v1) | 从成功 sibling 提取适用的 PI，按 executable action 聚合；轨迹内相对化，使用正、有限、均值保持的权重 | action span、中心化、信用总量守恒和适用性门控都已被组合研究 |
| [ICSD](https://arxiv.org/html/2608.14945v1) | 用 RL surrogate 的局部影响校准 teacher supervision，并匹配 action-turn 的辅助权重总量 | “只保留和 RL 方向一致的 teacher 信号”也不是空白 |
| [SGCD](https://arxiv.org/html/2606.12634v1) | sibling-guided reference，蒸馏信号调节 policy-gradient credit，讨论长程工具使用中的副作用 | “让 PG 主导、避免 teacher 直接接管”已有很近的故事 |
| [CRAFT](https://arxiv.org/html/2606.29476v1) | 从已有 sibling rollouts 估计 teacher-following 的收益，校正 self-distillation | “不新增 rollout 的 sibling counterfactual”已有方法；还需区分真实同状态干预和带假设的估计 |
| [TRIAGE](https://arxiv.org/abs/2606.32017) | judge 将动作分为进展、探索、基础设施和回退，再给角色相关信用 | Role-Aware judge 本身不能作为我们的核心新意 |
| [VICT](https://arxiv.org/html/2608.28128v1) | 将 verifier 暴露为可执行 / 有证据的 atoms，经依赖有效的 proof edges 回链到动作；证据不足时弃权 | 与 verifier / artifact provenance 的信用分配路线直接接近 |
| [DRACO](https://arxiv.org/abs/2609.04094) | 动态 rubric 结合 step attribution，为 agent 构造过程信用 | “criterion + judge attribution”也很拥挤 |
| [OPPO](https://arxiv.org/abs/2605.21851) | 累积 oracle-conditioned likelihood evidence，递推成功概率并产生 token advantage | 不能把 teacher log-ratio 的贝叶斯累积本身当新意 |
| [AgentOPSD](https://arxiv.org/abs/2608.05987) | 将 outcome belief revision 用于多轮 agent 的信用分配 | 上述递推思想已进入 agent 场景 |
| [CVT-RL](https://arxiv.org/html/2606.05263v1) | 执行不同类型的干预，用冻结策略 continuation 和 selection-adjusted doubly robust estimator 估计贡献 | 真正反事实 credit 有先例，而且分支执行的预算不能忽略 |
| [ClawGym II](https://arxiv.org/abs/2608.16798) | 黑盒 harness、proxy 捕获、prefix tree、PPO/GRPO、mix-harness 训练 | 不能将我们修好的 segment / tree 训练契约单独包装为新算法 |
| [MemoryWalker](https://arxiv.org/html/2609.00865v1) | 压缩上下文的条件一致性；精确 tree 路径及 compressed / pre-eviction self-distillation | “压缩前 teacher 指导压缩后 student”已经很接近现有论文 |
| [vOPD](https://arxiv.org/html/2605.07865v1) | 给 OPD 的 sampled reverse-KL 梯度加入闭式、state-dependent baseline 来降方差 | 是主候选必须正面对比的近邻，但其保留的是蒸馏目标，不是 verifier reward 的基线梯度 |
| [Q-Prop](https://arxiv.org/abs/1611.02247) | action-dependent control variate：解析辅助梯度 + Monte Carlo residual correction | 主候选的基础思想来自经典 RL，必须引用，不能宣称发明控制变量 |
| [Unmasking OPD](https://arxiv.org/abs/2605.10889) | 用 targeted rollouts 估计理想局部梯度，诊断蒸馏梯度的有效性 | 仅做 gradient cosine 诊断不是新算法；可借鉴验证方式 |
| [Credit Without Ground Truth](https://arxiv.org/html/2608.19760v1) | 将隐式信用和 judge 信用与 executed replay、shuffled controls 对照；结果有明显统计与可测性限制 | 提醒我们不能用流畅度或工具类别标签冒充真实因果信用；不意味着所有信用方法都无效 |
| [POPE](https://arxiv.org/abs/2601.18779) | oracle solution prefixes 引导难题的 on-policy exploration | 利用 privileged hint 多采出成功轨迹并非新方向；它也不等于 placebo 校准 |
| [VRR-Stop](https://arxiv.org/abs/2607.17641) | 建模 noisy verify-repair 的修复 / 破坏风险并决定何时停止 | “避免越修越坏、学习停止”需与现有停止研究区分 |

另检索了 Skill-SD、SDAR、GRSD、StepOPSD、TRACE、动态 rollout 分配、语义动作等价和 hindsight relabeling 等邻近路线。本轮没有找到同时符合“明显区分 + 当前可低成本验证”的第二个同等强度主候选，因此不建议并行堆叠多套故事。

## 2. 我们现有数据真正支持什么

### 2.1 条件 logprob 信号并未通过现有门控

已有 matched-placebo 实验覆盖 **8 个任务、30 条负 advantage 轨迹、399 个 assistant turn**，完成 1,596 次评分。不同 placebo 的 delta 符号翻转率为 **47.37%**。

但必须修正统计解释：

- `successful_read`、`tool_error` 等是启发式类别，不是动作的因果质量真值。
- 读取成功不代表有贡献；工具报错不代表决策错误；重复有时是合法轮询。
- 先前的“44.9% 正确率”应表述为**与启发式类别标签的一致率**。
- 399 个 turn 也不等于 399 个独立实验对象；尤其只有 8 个任务，泛化推断非常有限。

这组结果支持“目前不应将这个差值直接放大为训练信用”，不支持“所有 OPSD 都无效”。来源：[matched-placebo 报告](/root/openclaw-rl-local/reports/MATCHED_PLACEBO_OPSD_PHASE0_GATE_20260906_ZH.md)、[abstain 报告](/root/openclaw-rl-local/reports/ABSTAIN_OPSD_PHASE0_GATE_20260906_ZH.md)。

### 2.2 Verifier 与 workspace 路线有资源，但不完全现成

历史静态审计中，2,000 个训练任务有 1,838 个包含静态 criterion dictionaries，739 个会直接输出 criterion vector。静态可提取不等于已经拥有可靠的逐动作因果标签。来源：[verifier 审计](/root/openclaw-rl-local/reports/clawgym_verifier_structure_20260821.md)。

已有 provenance smoke 保存的是路径、大小、mode、SHA-256 和 mutation manifest。**这不能单独恢复每个中间 workspace 的完整字节。** 终态归档也不能恢复被覆盖的旧文件。原 smoke 只有 2 题 4 轨迹，并对后台写入保守加 taint，不能据此声称所有轨迹可精确回放。来源：[provenance smoke](/root/openclaw-rl-local/reports/mosaic_opsd_p2_provenance_smoke_20260823_zh.md)。

因此，基于每一步 workspace 的停止 / 回退学习需要新增状态采集，不应写成“现有日志离线算一下就能做”。

## 3. 主候选：teacher 作为控制变量

### 3.1 一句话说明

**最终学什么仍由 verifier reward 决定；teacher 只帮助减少“这一批恰好采到了什么”带来的梯度波动。**

原方案是：

```text
teacher 更支持这个动作
→ 认为它更值得奖励 / 更不该惩罚
→ 改变该动作的 advantage
```

新候选是：

```text
计算 teacher 对本次采中 token 的辅助梯度
计算同一 prefix 下该辅助梯度的条件平均
两者相减，得到条件均值为零的修正
用这个修正降低原始 reward 梯度的采样方差
```

不需要 teacher 是正确裁判；但要降低方差，它必须与原始梯度的波动有可利用的相关性。没有相关性，就应回退到系数 0。

### 3.2 最小数学形式

下面先描述固定权重、固定 pre-token context 的情况，梯度采用最大化方向。

- `s`：生成当前 token 前的真实可见上下文，包含此前生成 token。
- `a ~ μ(.|s)`：当前真实采样分布生成的 token。
- `u(v) = ∇θ log πθ(v|s)`：将候选 token `v` 概率提高的参数梯度。
- `c(s,v)`：冻结 teacher 提供的有界辅助分数，例如静态 anchor 条件下的 logprob 差；不是 reward，也不声称是正确性。
- `g_base`：现有 GRPO / PPO 产生的随机梯度。

定义：

```text
B(a) = c(s,a) · u(a)
M(s) = sum_v μ(v|s) · c(s,v) · u(v)
Z    = B(a) - M(s)

g_new = g_base - α · Z
```

因为 `E[B(a)|s] = M(s)`，所以在条件成立时：

```text
E[Z | s] = 0
E[g_new] = E[g_base]
```

这与“权重平均等于 1”不同：权重的平均值相同，通常不能保证乘上各个不同 token 梯度后仍有相同均值。

对聚合控制变量 `Z`，固定标量 `α` 时：

```text
tr Var(g_base - α Z)
= tr Var(g_base)
  - 2α E[(g_base-E[g_base]) · Z]
  + α² E[||Z||²]
```

理想系数为 `α* = E[(g_base-E[g_base])·Z] / E[||Z||²]`；真实实验中要用独立数据或上一批估计，并做收缩。不能在同一批里挑出最有利的 α 后，把样本内下降宣称为泛化收益。

### 3.3 一个仅用于解释的数值例子

**以下是构造的三动作例子，不是 ClawGym 实测。**

假设同一状态有三个候选，采样概率各为 1/3：

```text
候选                     1       2       3
真实期望 reward          0.9     0.6     0.0
减去基线 0.5 后的 A      0.4     0.1    -0.5
teacher 辅助分数 c       0.3     0.0    -0.3
```

对三个动作 logits 的完整梯度：

```text
标准估计的平均梯度       [0.133333, 0.033333, -0.166667]
加入修正后的平均梯度     [0.133333, 0.033333, -0.166667]

标准估计的梯度方差迹     0.046667
α=1 时修正后的方差迹     0.006667
```

这个特意选取有用 teacher 的例子展示的是：平均方向不变，但单次抽样带来的波动可以下降。**85.7% 的例子内方差下降不是我们的效果预估。** teacher 换成无关分数，未经校准的修正反而可以增大方差。

### 3.4 为什么可能适合我们

- 每个任务只有 8 条采样，长轨迹却包含大量 token，值得测量 terminal reward 梯度的采样噪声；当前尚未测得该候选能减少多少。
- 已有 frozen teacher scoring、真实 prefix / segment、token span 与训练链路，不需要新增一个强 LLM judge。
- 可先复用已有静态 anchors，不要求得到一条满分轨迹。
- 不新增环境分支，不需要复制远端 workspace，不增加修复 rollout。
- 成立时主要是**样本效率**收益，不是让每条 rollout 本身更快，也不是修复两次相同 forward 的数值不确定性。

### 3.5 与近邻的实质差异和创新风险

| 方法 | Teacher 主要做什么 | 保留什么 |
|---|---|---|
| SSPO / TASPO | 重分配 outcome advantage | 符号 / 权重尺度等约束；一般不等于保留整个参数梯度期望 |
| ICSD | 按目标兼容性分配辅助蒸馏更新 | 基础 RL 项不变，但仍叠加辅助优化 |
| vOPD | 降低 sampled distillation gradient 方差 | teacher-matching / 蒸馏梯度的期望 |
| 本候选 | 提供条件均值可计算的辅助随机梯度 | 原来 reward-based 基线梯度的期望，限下述假设 |

与 [Q-Prop](https://arxiv.org/abs/1611.02247) 的相似性必须明确承认；与 [vOPD](https://arxiv.org/html/2605.07865v1) 的区别也不能只写“我们做 agent”。可能形成贡献的是：**在不可信 privileged teacher、长程黑盒交互和真实采样分布下，得到可计算的 outcome-preserving estimator，并证明实际降低样本成本。**

目前定性判断：机制区分有空间；创新强度中等、依赖实证；不能保证审稿人不会认为是经典控制变量的应用。

## 4. 实现不能省略的条件

### 4.1 不能把旧最终答案反馈直接搬过来

`c(s,v)` 必须在当前随机 token 之前已经确定。第一版只用训练任务的静态 anchors 和当前 prefix。

不能使用当前轨迹的最终 assistant answer、最终 reward、未来 tool result 来构造 `c`，然后假装其不依赖当前动作。也不能只在最终负 advantage 的轨迹上启用修正；这种选择依赖未来结果，破坏上述条件均值论证。

### 4.2 必须使用实际采样分布

真实 `μ` 是应用 temperature、top-p、top-k 等处理后的采样概率。Megatron 重算 old logprob、未经截断的 softmax，以及 SGLang 返回的某类 logprob 字段，都不能未经核对就当成 `μ`。

当前本地 `_score_opsd_tokens` 只请求 `max_new_tokens=0` 和 `input_token_logprobs`，返回实际 response token 的分数；它没有保存计算 `M(s)` 所需的候选分布。见 [clawgym_rl_rollout.py](/root/openclaw-rl-local/clawgym-official/clawgym_rl_rollout.py:2158)。

第一版应保存少量候选 token IDs、实际采样概率以及 teacher 对这些相同 IDs 的分数。若真实 top-k=20 且流程确认一致，其支持集上限较小；这仍需实际检查 sampler，不能仅凭启动参数断言。

若只使用候选集合 S，可明确设集合外 `c=0`，在 S 上按原始 `μ` 求和；不能把 S 内概率重归一化后宣称它还是同一个控制变量。候选集合和 token 位置的选择也必须由 pre-token 信息决定。

### 4.3 与现有 loss 的关系

- 原 GRPO / PPO、TIS 项先保持不变，CV 作为单独的零均值梯度项加入；不能把修正后的 A 随意再送入非线性 PPO clipping，仍宣称同样的恒等式。
- 现有 logical-rollout 的 `1 / 实际生成长度` 依赖未来动作。不能把 CV 也直接乘上这个随机长度因子；第一版 CV 使用固定预算尺度或其他可预测系数，基线保持现状。
- 对 `α`、teacher 分数和 `μ` 停梯度；解析和采样项必须采用一致系数，梯度仅通过目标 student logprob。
- pruning 后按真实可见 prefix 计算，不能把被删历史恢复到 student 条件里；同一个生成 token 不因重复出现在多个 segment 而重复计入 CV。
- 长度门控应依据当前 prefix 能否加入 anchor；不能根据整条轨迹最终是否超长选择性保留 CV 后，继续声称无偏。

这里的“保留期望”只指满足条件时的**原始梯度估计**。它不证明经过 global gradient clipping 和 Adam 后两种参数更新期望完全一样，也不证明 PPO / clipped TIS / 当前长度归一化本身是严格无偏的完整策略梯度。

### 4.4 工程量

概念原型约涉及三处：sampler 候选概率记录、teacher 指定候选评分、Megatron 的稀疏候选辅助 loss。它不是现成开关，但不需要改造 OpenClaw 执行语义或增加远端环境分支。

正式更新应争取合并到同一次 backward；诊断阶段为了分别计算原梯度与控制变量，需要额外 backward。不能用旧实验某次 `0.57s` 推断新方案的整批开销，长上下文和候选评分的成本要重测。

## 5. 最小预实验与退出条件

下列阈值是建议预注册的研究决策线，不是论文通用标准。

### A. 先验证估计器

1. 有限动作 / token 分布单元测试：逐项枚举，核对修正条件均值为零。
2. 固定真实 prefix 和模型、改变采样 seed，验证经验均值随重复次数收敛；这测试采样方差，不是相同 token 的重复 forward。
3. 覆盖 top-p / top-k、跨后端概率、mask、长短轨迹和 pruning 的处理。

任何一项不成立，不进入效果实验。

### B. 冻结模型的 ClawGym 梯度实验

建议先采集 **64 个不同训练任务 × 8 rollouts**，按任务划分 32 个校准任务和 32 个验证任务；按 reward、长度、工具使用分层。只作试点，不将 32 个验证任务视为充分的最终论文证据。

比较：

- 原始 GRPO 梯度；
- GRPO + 常数 / 简单非语义 feature 控制变量；
- GRPO + 无 anchor teacher 的控制变量；
- GRPO + 静态 anchor teacher 的控制变量；
- 打乱 anchor 的负对照。

关键统计：

- 验证任务上的梯度方差迹、signal-to-noise、α 稳定性；
- 相同数据划分下的梯度方向，不用启发式 action label 当真值；
- 控制项均值偏差和置信区间；
- 长度 / 正负 A / reward 桶的收益，而不只报全局平均；
- teacher 与候选评分墙钟时间、峰值显存、存储和额外 backward 成本。

完整梯度可流式累积范数和内积，避免保存每条轨迹的完整 8B 梯度。固定随机投影可辅助筛选，但不能替代最终关键对比的完整梯度检查。置信区间按任务或 rollout group 聚类，不把 token 当独立样本。

建议通过线：验证集方差下降至少约 10%–15%，置信区间支持正收益，且明显优于简单 / 打乱对照。若 α 在验证集回到 0、收益仅出现在校准集，或者没有 anchor 特异收益，应停止将其作为 OPSD 主论文。

### C. 小规模在线对照

通过后先做 20-step 配对实验：相同 8B SFT 起点、task order、预算、sampling 协议与 verifier，比较新鲜的 GRPO 基线和候选。历史曲线只能作背景，不能替代新控制组。

然后再决定 100-step、多 seed 实验。必须同时报告：

- 固定 rollout 数下的 eval reward / correct rate；
- 固定总 GPU 时间及环境执行预算下的表现；
- teacher 开销计入总成本后，达到同一 eval 水平需要多少采样；
- 是否仅压低梯度范数 / 实际更新幅度，而没有改善学习。

现有 200 题 eval 可以保留，但接近 50% 的单次 200 题二值正确率，粗略标准误就约 3.5 个百分点；一两个百分点不能靠单次曲线下结论。使用逐题配对分析、多个 sampling seeds，正式结论还需独立训练 seeds。反复查看的 eval 不应用于挑 anchor、挑门控阈值。

## 6. 为什么暂不选“免费 STOP 反事实”作为第二主线

对只由最终 workspace 决定分数的任务，理论上立即提交的 reward 是 `V(W_t)`，无需新增模型 continuation；这比删除任意动作后重跑更便宜。

但目前有三个缺口：

1. 旧日志的 hashes 和终态归档不能恢复每个 `W_t`，需要增量字节快照；后台写入、外部状态和 verifier side effects 也要处理。
2. 单纯把 `R_final - V(W_t)` 当 advantage，首先是一个 state baseline；只奖励相邻 verifier 差则接近 potential shaping。不能因此直接宣称新信用算法。
3. 若进一步学习 STOP / CONTINUE 决策，要处理自然 assistant 结束与动作空间的对应，并与已有停止方法比较。

它可以作为后续诊断方向，但当前不比主候选更省工程，也没有已经验证的独立创新点。

## 最终建议

本轮不建议继续在原来的绝对 delta、median/MAD、matched-placebo 或 role 常数上追加规则。建议首先验证一个更不同、可以被直接证伪的问题：

> **静态 privileged self-teacher 的概率分布，能否作为条件均值可计算的控制变量，减少 ClawGym 终局 reward 梯度的采样噪声，并带来扣除 teacher 成本后的样本效率收益？**

若真实梯度预实验不支持，就停止；若通过，再组织算法推导、近邻对照和在线训练证据。检索支持的是“这个具体切入点尚有区分空间”，不是“已证明首创”，更不是“已证明有效”。
