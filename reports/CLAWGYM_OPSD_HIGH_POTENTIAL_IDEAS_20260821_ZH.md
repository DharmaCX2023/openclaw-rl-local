# ClawGym × OPSD：高潜论文方向与严格撞车审查

日期：2026-08-21

## 结论摘要

经过对 2026 年最新 OPD/OPSD、agent RL、credit assignment、adaptive rollout、program repair 和 ClawGym II 文献的多轮检索与反向审稿，最终保留三个候选：

1. **Mosaic-OPSD：Verifier-Gated Artifact Mosaic Distillation**
   - 当前排序第一，**Conditional GO**。
   - 核心不是普通 patch crossover，而是：从同一 on-policy group 的已执行 artifact atoms 中构造严格优于所有父轨迹的、可重放的 `strict super-parent`；再把每个必要 atom 精确回链到原 on-policy tool-call token span，做 span-masked OPSD。
   - 最大风险是严格多父协同在真实数据中太少，或合成 artifact 只对训练 verifier 有效。

2. **SafeWitness-OPSD：Prefix-Safe Verifier Witness Compiler**
   - 当前排序第二，**Conditional GO**；若编译覆盖率和安全审计通过，可能升为第一。
   - 真正创新应是把隐藏 Python verifier 编译成 reward-preserving、可重放、按 action prefix 脱敏、带 artifact provenance 的 failure witness。OPSD/SSPO 只是可替换的消费后端。
   - 最大风险是 verifier secret 通过 packet 内容、长度、路由或梯度泄漏；动态 slice 也不等于动作因果。

3. **Criterion-Contrast Allocation：按 verifier 维度分配 rollout 预算**
   - **Weak Conditional GO**，实现风险最低，但论文撞车最重。
   - 创新边界只能是：以“每个 criterion 至少出现一次 pass/fail contrast”的期望覆盖作为分配目标，并用每任务质量守恒的 criterion-pair loss；不能宣称发明了 multi-reward GRPO 或 adaptive rollout。
   - 更适合作为独立 efficiency 工作，或 Mosaic-OPSD 的成本扩展，不适合与前两项并列塞进同一篇主论文。

不建议继续将“失败轨迹 + hindsight hint + OPSD”本身作为创新。它已经被 OpenClaw-RL、SDPO、HINT-SD、SSPO、OPID、MOPD、HSD、U-OPSD 和 OVCSD 从多个角度覆盖。

## 一、本地数据给出的研究信号

本地正式 RL 数据包含 6,395 条轨迹、800 个任务，主要为每任务 `G=8`：

| 观察 | 本地结果 | 对论文设计的含义 |
|---|---:|---|
| 完整 group 中存在标量 reward 差异 | 750/796，94.2% | 普通 GRPO 并不普遍缺少组内 advantage；不能把 zero-variance 当主要动机 |
| 低于同组最好结果的轨迹 | 约 64% | 同组存在大量可供比较的 partial failure |
| reward ≤ 0.5 且同组有更好 sibling | 1,191 条 | failure-aware 方法有足够候选，但不能无差别训练所有 failure |
| 零 reward 且同组有正 reward sibling | 364 条 | 有明确的 failure/success 对照，但这本身已被 HSD/MOPD 覆盖 |
| 全量任务有显式 criterion dict | 1,866/2,000，93.3% | 可获得约 15.8 维的 verifier profile，而非只有一个 scalar |
| 含 workspace 修改动作的轨迹 | 6,319/6,395，98.81% | artifact lineage、diff 和可执行消融在数据上高度相关 |
| task 有结构化或文本输入 | 1,985/1,987，99.90% | JSON/YAML/CSV/text 等 typed atom 有现实基础，但仍需审计输出可合并性 |
| multi-tool assistant step | 11,702/118,763，9.85% | provenance 必须细化到单个 tool-call block，不能整 turn 标注 |
| compaction summary | 现有审计约 2 条 | compaction-only 方向数据量不足 |
| retry/dead-leaf 严格候选 | ≤22/77,952 main records，≤0.028%；实际预计更低 | harness rejection distillation 不足以成为主方法 |

关键判断：ClawGym 最有价值的结构不是“失败很多”，而是**终局 workspace 可执行验证、同任务多 rollout 可能留下互补的持久状态增量、且 verifier 通常有多维 criterion**。

## 二、为什么朴素 failure-hindsight OPSD 已经不能投稿

主要邻近工作如下：

- [OPSD](https://arxiv.org/abs/2601.18734)：同模型在 privileged context 下作为 teacher，对原 on-policy response 提供密集信号。
- [SDPO](https://arxiv.org/abs/2601.20802)：使用 runtime error 或 judge feedback 自蒸馏。
- [OpenClaw-RL](https://arxiv.org/abs/2603.10165)：从 next state/tool output 抽取 hindsight hint，再做 RL 与 distillation。
- [HINT-SD](https://arxiv.org/abs/2605.17873)：用完整失败轨迹定位关键 action 并产生 corrective hint。
- [SSPO](https://arxiv.org/abs/2608.12764)：只对错误轨迹使用 evidence-conditioned self-teacher disagreement 调节 step-level GRPO 权重，报告约 5% 额外开销。
- [OPID](https://arxiv.org/abs/2606.26790)：从完整 on-policy trajectory 提取 episode/step skill，再以 logprob shift 提供密集监督。
- [MOPD](https://arxiv.org/abs/2605.12652)：teacher 同时利用同组成功与失败 peer。
- [HSD](https://arxiv.org/abs/2606.15576)：以同组成功 continuation 为 privileged context，把 credit 集中到 divergence。
- [U-OPSD](https://arxiv.org/abs/2608.06296)：从同 prompt 多 rollout 的一致性构造 pseudo-solution，再蒸馏 disagreeing completion。
- [OVCSD](https://arxiv.org/abs/2607.27937)：从学生到达的状态运行 teacher continuation，只蒸馏 outcome-verified 成功分支，额外 privileged interaction 低于约 3%。
- [Credit Without Ground Truth](https://arxiv.org/abs/2608.19760)：用 executed replay 审计 step credit，表明 teacher/judge/logprob 信号与真实动作贡献可能弱相关，且容易受训练 dose 混杂。

因此不能把以下内容写成创新点：

- 只对 failure 做 OPSD；
- teacher 查看未来 tool result、完整轨迹或 verifier feedback；
- 从成功 sibling 指导失败 sibling；
- teacher-student logprob gap 定位关键 token；
- 执行 teacher continuation 后只保留成功分支；
- 简单把 criterion reward 分解后广播给相关 step。

## 三、Idea 1：Mosaic-OPSD

### 3.1 唯一可站住的主张

宽泛的 “multiple partial repairs + verifier-guided crossover” 已被 [EvolRepair](https://arxiv.org/abs/2604.02134) 和程序修复/trajectory stitching 工作覆盖。因此论文必须收窄到：

> 同一任务的 on-policy population 中存在互补的、已执行的 artifact atoms。它们的 provenance-closed 组合可形成严格优于所有父轨迹的 verifier-valid super-parent；通过 artifact-side necessity 将这些贡献回链到原 on-policy producer spans，可以在不生成新 policy trajectory 的情况下内化多个失败轨迹的互补能力。

必须同时满足四个条件：

1. **严格超过所有父轨迹**

\[
R(W^\star)>\max_i R(W_i)+\delta.
\]

2. **至少两个父来源是必要贡献者**

\[
R(W^\star\setminus A_i)<R(W^\star),\qquad
R(W^\star\setminus A_j)<R(W^\star),\quad i\neq j.
\]

3. **合成中没有新内容**

\[
A^\star\subseteq\bigcup_i A_i.
\]

所有 byte、结构化叶子和 hunk 都必须来自真实 on-policy tool execution；不能让 LLM 生成 merge fix，否则会退化为 EvolRepair。

4. **只训练来源明确的 span**

只允许训练：

- 产生 mosaic 必要 atom 的 tool-call token span；
- 产生经真实 verifier 消融证明有害 conflict atom 的 tool-call span。

缺失但从未被原轨迹生成的 atom 没有 on-policy producer span，首版不应强行归因。

### 3.2 算法

对初始 workspace \(W_0\) 和同任务 \(G\) 个终局 workspace \(W_i\)：

1. 提取 typed edit atoms：
   - JSON/YAML/TOML leaf assignment；
   - CSV keyed row/cell update；
   - Markdown section、普通文本 hunk；
   - code AST-aware hunk；
   - add/delete/rename/permission metadata。
2. 每个 atom 记录：

\[
a=(\text{before hash},\text{after content},\text{entity},
\text{trajectory},\text{turn},\text{tool-call token span}).
\]

3. 建立 lineage-and-conflict DAG：
   - 同一 leaf 不同值、delete/write、重叠 hunk 为 conflict；
   - rename→edit、definition→reference 等为 dependency；
   - atom→原始 action token 为 lineage。
4. 在依赖闭包和冲突互斥约束下做小 beam search：

\[
W^\star=\arg\max_C\; w^\top V(W_0\oplus C)-\lambda|C|.
\]

5. 每个候选依次通过 patch apply、syntax/schema 检查和真实 verifier；禁止修改 verifier/test protected scope。
6. 对每个参与训练的父轨迹构造最小 verified bridge：

\[
z_i=\bigl(W^\star\ominus W_i,\;V(W^\star)-V(W_i),\;\text{provenance}\bigr).
\]

7. frozen/EMA self-teacher 查看 \(z_i\)，student 只查看原始 prefix；在原始 token IDs 上做 span-masked OPSD：

\[
L=L_{\text{tree-GRPO}}+
\lambda_{\text{mos}}
\frac{1}{\sum m_{it}}
\sum_{i,t}m_{it}
D_{\text{OPSD}}
\left(
\operatorname{sg}\pi_{\bar\theta}(\cdot\mid h_{it},z_i),
\pi_\theta(\cdot\mid h_{it})
\right).
\]

这里不应宣称 artifact necessity 等于完整 reasoning token 的因果贡献；mask 应尽量只覆盖实际 write/edit/exec producer block。

### 3.3 与邻近工作的边界

| 邻近方法 | 已有能力 | Mosaic-OPSD 必须证明的额外部分 |
|---|---|---|
| EvolRepair / SE-Agent | 多 candidate patch 重组、执行反馈、生成更强 repair | 训练期、跨通用 workspace artifact、无新 LLM patch、source-span OPSD |
| MOPD / HSD / U-OPSD | 利用 peer rollout 或共识作为 teacher context | 构造严格优于所有已有 rollout 的新 verified artifact |
| OVCSD / BPO | 从同状态额外生成/执行分支 | 不生成新 policy trajectory，仅重用已执行 atom |
| Patches-to-Trajectories | reference patch 作为 privileged process supervision | pseudo-reference 完全由当前 on-policy population 自己构造 |

### 3.4 成本优势应该怎样声称

只能声称 **zero additional policy-generated trajectories**，不能声称零额外 environment cost。

设原 group 有 \(G\) 次 verifier，beam 额外验证 \(K\) 个候选，necessity/ddmin 使用 \(L\) 次，有 \(C\) 条轨迹做 teacher forward，则必须报告：

- verifier executions、CPU-seconds/task；
- merge/search CPU-seconds；
- teacher-forward prefix/response tokens 与 GPU-seconds；
- agent rollout tokens、tool calls、wall time；
- peak sandbox/storage；
- 固定 rollout、固定生成 token、固定端到端成本三种口径。

不能只用 `G=4 vs GRPO G=8` 宣称 2 倍便宜，因为 teacher forward 和 verifier search 也有成本。

### 3.5 Phase-0 一票否决条件

先收集 500–1,000 个同任务 group 的完整终局 workspace，不训练模型。建议门槛：

- strict mosaic 占全部 group 至少约 5%，或 eligible group 至少约 15%；低于 2% 直接停止；
- all-parent-fail group 中至少约 3%–5% 能合成 verifier success；
- 至少 80% accepted mosaic 有两个以上 leave-source-out necessary parents；
- 从 \(W_0\) 确定性重放成功率至少 99%；
- 训练 verifier 的增益至少 70% 保留到 withheld/protected verifier；
- 额外端到端 wall time理想值不超过 30%，超过 50% 不再主打低成本；
- necessary span 与 random/excluded span 的 teacher-gap AUC 至少约 0.65；
- query-matched `Best-of-(G+m)` 若更强，则 Mosaic 主张失败。

最致命审稿意见会是：

> 你们只是把高分终态直接 teleport 到 sandbox；verifier 没有证明 agent 能通过合法动作到达它，反复查询 verifier 还可能只是 endpoint overfitting。

因此必须从 \(W_0\) 重放 source edit segments，并保留独立 hidden/protected verifier 只做最终评分。

## 四、Idea 2：SafeWitness-OPSD

### 4.1 论文问题与创新边界

最合适的问题不是“再发明一种 OPSD”，而是：

> Can hidden evaluators safely teach an agent without becoming answer keys?

候选题目：

> From Hidden Tests to Safe Teachers: Compiling Verifiers into Prefix-Safe Failure Witnesses for Agent RL

真正可声称的贡献是：

> 首个把隐藏可执行 evaluator 编译为 reward-preserving、可重放、按决策前缀脱敏、带 artifact-to-tool-call provenance 的 self-teacher packet 系统。

不能声称首个 rich-feedback RL、首个 verifier feedback、首个 failure action distillation 或首个 program slicing。

### 4.2 编译流程

1. **定义受支持的 Verifier-Py 子集**
   - Python 控制流、函数、常见容器；
   - JSON/CSV/text/filesystem；
   - filter/sort/aggregate/compare；
   - subprocess exit code/stdout/file effects；
   - criterion dict 和 scalar aggregation。

`eval/exec`、reflection、网络、非确定 native extension 作为 opaque node 或 fail closed。一般 Python 的精确最小语义切片不可判定，不能声称支持 arbitrary Python。

2. **双执行保证 reward preservation**

先运行原始 verifier 得到正式 reward；再在私有副本运行 instrumented shadow verifier。只有：

\[
(R,V)_{\text{original}}=(R,V)_{\text{instrumented}}
\]

才允许生成 packet，否则只保留原 scalar reward。

3. **Private Executable Witness**

从 failed criterion sink 向后做 dynamic slice，并用 delta debugging 缩减为：

\[
W^{\text{priv}}_{ik}=(P_k,I_k,A_k,B_k),
\]

其中包含执行过的 predicate slice、输入片段、agent artifact 片段和 branch fingerprint。独立 checker 必须验证：

\[
\operatorname{Replay}(W^{\text{priv}}_{ik})
= (\text{same criterion},\text{same failure class},\text{same branch}).
\]

只能声称 1-minimal，不能声称全局 cardinality-minimal。

4. **Artifact provenance**

为每次 tool call 保存 workspace version、file hash/diff、process tree 和 overwrite chain，把 witness 读取到的 bytes 反查到单个 `<tool_call>` token span。missing artifact 或 opaque subprocess 无法可靠定位时回退 vanilla GRPO。

5. **按 action prefix 脱敏**

对第 \(j\) 个 action，用其当时可观测信息 \(\mathcal O(h_j)\) 对 witness 分类：

- `OBSERVED`：已经出现在 prompt 或前序 tool result，可保留精确值；
- `ACCESSIBLE_UNOBSERVED`：agent 当时可读取但尚未读，只能提示公开 path/key/type，不给内容；
- `SECRET`：hidden fixture、checker literal、reference answer，只给固定 failure class，不给值、名称和精确数量。

teacher packet 控制在约 50–150 tokens，并附 replay-certificate hash。teacher-only 并不自动安全，因为 secret 仍可能通过梯度记忆。

6. **只选择有 on-policy 可达对照的 criterion**

首版只处理：

\[
c_{ik}=0,\qquad \max_{g\in G}c_{gk}=1.
\]

这样至少证明当前策略分布中有人能通过该 criterion；但不把成功 sibling 的答案直接给 teacher。

### 4.3 训练目标

最稳妥首版不是让 teacher 翻转 GRPO 方向，而是复用 SSPO 式 sign-preserving shaping：

\[
d_{ij}=\frac1{|a_{ij}|}\sum_t
\left[
\log q^W_{ij}(a_t)-\log q^0_{ij}(a_t)
\right],
\]

其中 \(q^0\) 使用同模板、同长度的 neutral packet，排除“多出一段 context”本身造成的 logprob 变化。

\[
w_{ij}=\operatorname{clip}
\left(
\exp(\operatorname{sign}(\hat A_i)d_{ij}),1/c,c
\right),
\]

\[
\widetilde A_{it}=hat A_i
\left[1+\lambda\mathbf 1(t\in a_{ij})(w_{ij}-1)\right].
\]

teacher 只调节原 GRPO advantage 的局部幅度，方向仍由真实 outcome 决定。若离线 paired decode/executed audit 证明 packet-conditioned teacher 确实更好，再增加 direct OPSD/forward CE。

### 4.4 安全和因果边界

必须明确：

- dynamic dependency 只说明 verifier 读取了某 action 产生的 bytes，不证明该 action 是失败原因；
- packet 内容、长度、是否出现、criterion routing 都可能泄漏 secret；
- 最小 witness 往往更容易暴露隐藏阈值，最小化不等于最少泄露；
- teacher-only secret 仍可能写入模型权重；
- train/test 必须按 checker AST template、author 和 hidden fixture 划分。

建议用少量 executed replay audit 检验 witness-relevant span 是否显著优于 shuffled span，但不要把这包装成个体因果估计器。

### 4.5 Phase-0 一票否决条件

- instrumented 与原 verifier 的 scalar/criterion 必须逐例完全一致；任何未 fail-closed mismatch 即停止；
- checker 静态支持率理想目标 ≥80%，最低可投稿覆盖率约 40%–50%；
- failed criterion 可重放 witness ≥60%，action-link coverage ≥50%–60%；
- replay deterministic ≥99%；
- packet 内容、长度、存在性和 routing 的 SECRET noninterference 必须通过；
- 隐藏 canary 恢复不得高于机会水平；
- real packet 必须显著优于同长度 null/shuffled packet；
- teacher semantic gap 至少是本地 repeated-forward/MoE 数值噪声的 3 倍，符号复现率 ≥95%；
- held-out ClawGym 至少 +2pp，或在 ±0.5pp 内减少至少 40% rollout；
- wall-clock/GPU-hour 增量超过约 15% 时，应重新评估低成本主张。

最致命审稿意见会是：

> replay certificate 只证明一个切片在一次执行中仍失败，1-minimal 不等于因果；脱敏标签也不等于 noninterference，secret 仍可能通过 packet 路由或梯度泄漏。

所以该方向的价值取决于可机器检查的安全保证，而不是提示词写得是否漂亮。

## 五、Idea 3：Criterion-Contrast Allocation

### 5.1 唯一可保留的主张

对任务 \(q\) 的 criterion \(k\)，设 pass 概率为 \(p_{qk}\)，采样 \(n\) 次后同时观察到至少一次 pass 和 fail 的概率为：

\[
1-p_{qk}^{n}-(1-p_{qk})^{n}.
\]

定义加权 contrast coverage：

\[
U_q(n)=\sum_k w_k
\left[1-p_{qk}^{n}-(1-p_{qk})^{n}\right],
\qquad \sum_k w_k=1.
\]

在总预算 \(\sum_q n_q=B\) 下按边际收益分配 rollout。可发表边界是：

> 对固定且可靠的 criterion pass probability，\(U_q(n)\) 精确表示 criterion contrast 的期望加权覆盖；利用其离散边际收益，可以在更少 rollout 下保留多维 verifier supervision。

不能声称它最优提高最终 return。

### 5.2 为什么撞车风险高

对单 criterion 且 \(n=2\)：

\[
U(2)=1-p^2-(1-p)^2=2p(1-p),
\]

即 Bernoulli reward variance 的两倍。因此它与 [VIP](https://arxiv.org/abs/2602.01601)、VIGOR、AERO 的 scalar variance/pass-rate allocation 高度相邻；loss 端又与 GDPO、SA-MRPO 的 per-objective normalization 和 PAIR 的 adaptive pair correction 相邻。

唯一差异是：分配目标为 **多 criterion 的 pass/fail contrast coverage**，可利用“scalar 相同但 criterion profile 不同”的 group。必须先证明这种额外监督在 ClawGym 中占比足够高。

### 5.3 训练时的质量守恒

变量 group size 不能让 rollout 更多的任务自动获得更大 loss mass。可使用 per-task criterion-pair objective：

\[
g_q=\sum_k w_k
\frac{1}{N_{qk}^+N_{qk}^-}
\sum_{i:c_{ik}=1}\sum_{j:c_{jk}=0}
\left[
\nabla\log\pi(y_i|q)-\nabla\log\pi(y_j|q)
\right].
\]

这定义的是 criterion-pair ranking objective，而不是标准 GRPO 的无偏复现；论文必须明确目标已经改变。

\(p_{qk}\) 只能来自历史 checkpoint 或计入成本的 pilot，不能先生成完整 G 再声称提前停止。若在线选择/停止，还要处理 pair inclusion bias，PAIR 是必须 baseline。

### 5.4 Phase-0 一票否决条件

先用 100–200 个任务、G=8 收集完整 criterion vector，然后只用前 1–2 条或历史数据估计 \(p\)，在剩余 rollout 上回放：

- scalar-tied 但 criterion-profile-discordant pair 比例；
- criterion-pair coverage；
- 相对完整 G 的估计梯度 cosine；
- uniform、VIP/VIGOR、criterion allocator 的 token savings；
- allocation 与 criterion loss 做 2×2 因子消融。

若新增 contrast coverage 不足约 10%，或同等估计偏差下不能节省至少约 15%–20% rollout，停止该方向。

最致命审稿意见会是：

> 这只是把已有 reward-variance allocation 对每个 criterion 求和，再接已有 multi-reward loss；没有证明 contrast 数量与 policy utility 或最终 return 对齐。

因此它目前是第三名的效率备选，不应作为主 OPSD 创新。

## 六、明确淘汰的方向

### 6.1 Sparse Causal Audit OPSD：当前 NO-GO

随机执行 2%–5% teacher branch，再用 cross-fitted AIPW/calibrator 推断其余 hint 可靠性，表面上很优雅，但当前因果 estimand 不成立：

- 单次 teacher continuation 与 factual suffix 的差，是两种 continuation policy 的 realized difference，不是 hint 的个体处理效应；
- AIPW 可以识别总体或预定义 strata 的平均效应，不能为未审计的 95%–98% node 恢复个体 counterfactual；
- OVCSD 已以低于约 3% privileged interaction 执行并筛选 teacher continuation；BPO 已做 snapshot branch comparison；
- 800 个任务的 2%–5% 只有 16–40 个独立 audited clusters，不足以支持高维 calibrator。

它最多作为小规模 audit 工具，而非当前 Top-3 主方法。

### 6.2 Compaction / subagent boundary distillation：本地数据 NO-GO

CompactionRL、ACON、FoldAct、Context-Folding、OVCSD 已覆盖大量相邻机制；本地真实 compaction summary 约 2 条，无法支撑独立论文。只有未来在 subagent final-return 边界发现千级样本，且能固定状态做真实 message intervention，才值得重开。

### 6.3 Harness retry/dead-leaf distillation：本地数据 NO-GO

ClawGym II 已明确识别 retry-induced dead leaves 并过滤。当前本地严格候选至多 0.028%，真实 exact-prefix rejected/accepted pair 很可能为个位数甚至零；结构约束解码还进一步消除了 malformed call。可保留为 appendix 的低成本 structural loss，不能做主线。

### 6.4 普通 criterion-factorized GRPO：创新不足

简单把 criterion vector 做 per-objective normalization，再按工具 provenance 路由，容易被判定为 GDPO/SA-MRPO + STAMP 的直接组合。若没有真实 artifact intervention、信息安全编译或新的预算理论，不应单独成文。

## 七、推荐的单篇论文方案

不建议把三个 Top-3 全塞进一篇。最清晰的第一篇应只有一条因果链：

> on-policy group 产生互补 workspace deltas → provenance-safe strict mosaic 形成父轨迹之外的 verified outcome → minimal improving diff 形成 privileged self-teacher context → 原 on-policy producer spans 上 OPSD → 新任务能力提升。

推荐题目：

> **Mosaic-OPSD: Learning a Verified Super-Parent from Partial Agent Failures**

SafeWitness 可以有两种处理：

- 若 Phase-0 编译覆盖率和 noninterference 很强，独立写成第二篇系统/ML 论文；
- 若与 Mosaic 合篇，只把它作为 “mosaic diff 的安全 packet compiler”，不能作为第二个同等主算法。

Criterion-Contrast 只作为后续效率扩展，不能在首篇同时改变 group size、loss 和 OPSD teacher，否则 baseline 与成本归因会失控。

## 八、实验设计与公平成本口径

### 8.1 三阶段路线

**Phase 0：不训练模型的可行性审计**

- 500–1,000 个 task-group；
- 保存 \(W_0\)、G 个终局 workspace、criterion vector、逐 tool-call file diff、原始 token IDs；
- 测 strict mosaic rate、multi-parent necessity、replay、withheld verifier 和成本；
- 同时测 SafeWitness compiler fidelity/coverage/leakage。

**Phase 1：4B/8B 小模型**

- 固定数据、rollout 和 verifier query budget；
- 至少 3 seeds；
- 先做同 G 的机制消融，再做等成本比较；
- 通过 +2pp 或同精度显著省成本的门槛后，才进入 30A3B。

**Phase 2：ClawGym 30A3B 正式实验**

- ClawGym train；ClawGym-Bench/PinchBench 以及至少一个外部或 verifier-template-held-out 集合；
- 与最新 ClawGym II tree-GRPO 对齐；
- 报告 deterministic seeded eval 和 live-agent eval 两套结果；
- 3 seeds 与置信区间，而不是单次 200 题 n=1。

### 8.2 Baselines

Mosaic 主线至少需要：

- tree-GRPO，G=4/G=8；
- Best-of-G 与 query-matched Best-of-(G+m)；
- basic OPSD / SDPO / HINT-SD / SSPO / OPID；
- MOPD / HSD / U-OPSD；
- OVCSD 或 BPO，匹配额外 privileged/environment cost；
- best-parent verified OPSD；
- unverified union；
- whole-trajectory mosaic distillation；
- 无 provenance mask、无 necessity gate、无 withheld verifier 的消融；
- 代码子集上的 EvolRepair-style recombination。

### 8.3 成本必须同时报告

- task rollout 数；
- assistant generated tokens；
- tool call 数和环境 wall time；
- verifier query 数及 CPU-seconds；
- teacher forward tokens、GPU-seconds；
- optimizer tokens/steps；
- 总 GPU-hours、CPU-hours、端到端 wall time；
- 推理阶段额外成本，目标应为零。

至少报告三种比较：

1. 固定 G/固定 rollout 数：看纯精度；
2. 固定生成 token/tool call：看 agent sampling efficiency；
3. 固定端到端资源：看真实性价比。

### 8.4 本地训练系统必须保持的约束

- 使用 Proxy 捕获的原始 on-policy token IDs，遵守 token-in-token-out；绝不从 OpenClaw message JSON 重新渲染训练 token；
- teacher、neutral teacher、student logprob 在同一训练 backend、同权重和可比 microbatch/router 条件下计算；
- teacher effect 必须显著大于当前 MoE repeated-forward 数值噪声；
- 每个 atom 先归一化、每轨迹再归一化、每任务最终质量守恒，避免多 hunk、多 segment 轨迹过度加权；
- compaction、tool-result prune、multi-tool 都必须保持真实 prefix，只通过 loss mask 选择 producer span；
- eval 固定 request seed，并同时记录工具输出、网络、进程时间戳等不可控分叉。

## 九、最终 Go / No-Go 决策

### 现在立即值得做

1. **Mosaic Phase-0**：这是最高优先级，也是最便宜的关键反证实验。
2. **SafeWitness 编译审计**：先做 fidelity、coverage 和 secret-flow，不急着训练。
3. 补存未来 rollout 的 terminal workspace snapshot、criterion vector、逐 tool-call state diff 和原始 prefix token IDs。

### 暂时不要做

- 直接上 30B 训练 failure-hindsight OPSD；
- 把完整 verifier traceback/expected value 给 teacher；
- 为了制造 novelty 同时改 group size、loss、teacher、reward 和 harness；
- 用单次 online eval 波动支持算法收益；
- 用 “zero extra rollout” 隐去 verifier/teacher 成本。

### 最终建议

先用 1–2 周完成 Mosaic 的无训练 Phase-0。若 strict super-parent 率、可重放率、withheld-verifier 泛化和 query-matched 成本全部通过，则它是目前最值得投入的主论文。若 Mosaic synergy 太稀疏，但 SafeWitness 的 checker 覆盖和 noninterference 审计通过，则转向 SafeWitness；若两者都不通过，再选择 Criterion-Contrast 做一篇更稳妥、但创新上限较低的 efficiency 工作。
