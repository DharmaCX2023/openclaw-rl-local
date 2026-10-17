# 可执行修复与采样边界 Margin：Agent Action-Span Repair 实验方案

日期：2026-09-07。版本：v2，所有阻断位置的 action 内平均 margin。状态：本轮确定实验方案，尚未改造训练代码或启动实验。

## 本版确定的方案

- 主体：Student 的真实轨迹保持 GRPO；Teacher 的完整修复动作经同状态执行验证后，单独用于 DPO 与 margin。
- 修复单位：一个完整 assistant turn，包括必要的短分析及完整工具调用，到需要新环境观察的边界停止；不要求与原动作等长。
- Margin：同时处理修复动作中所有被实际采样器排除的目标 token，不再只处理第一个；动作内对激活位置平均，再按 task→pair 等权。
- 候选范围：先做本地文件／代码产物、同步执行、可恢复且能取得可靠 criterion vector 的训练任务；每条原轨迹最多一个候选修复。
- 实验顺序：5 状态闭环 smoke → 64 候选状态的数据可行性检查 → 同数据 NLL / DPO / DPO+margin 对照 → 通过后再做短程 RL。当前不直接启动 100 步训练。
- Teacher 与 reference：阶段 A–C 均固定为初始 8B SFT；阶段 D 的 Teacher 固定到各自本轮 rollout 的 policy 版本，reference 仍固定初始 SFT。Teacher 可看受限 hindsight 信息，Student/reference 评分和自主复评不看 hint。
- 原有 Evidence Anchors 可复用，不再训练新的 judge，也不叠加上一轮 absolute OPSD 的负 advantage 重加权。

## 方法概览图

[历史 v1 示意图](figures/verified_action_repair_margin_20260907_final.png) 只处理首个采样阻断点，已不代表本版 margin 定义；暂不作为当前方法图展示。是否构成有效算法贡献，取决于相同修复数据和预算下的 NLL/DPO 强对照与独立 rollout 收益。

旧图由内置 imagegen 生成，保留供追溯：[初始提示词](VERIFIED_REPAIR_FIGURE_PROMPT_20260907.md)；[修订提示词](VERIFIED_REPAIR_FIGURE_REVISIONS_20260907.md)。

## 1. 核心问题与方法

问题来自用户的 [原始 PPT](../research/OPD_Agent_Action_Repair_PPT.pptx)：Teacher 的修复可能需要替换工具、增加前置动作、延长输出或避免过早结束。只在 Student 原错误前缀上评分，未必能有效传递完整替代路径。

本方案不声称 token-level reverse KL 在数学上无法学习这些行为；研究的是有限采样、错误前缀与实际截断采样下，修复信号是否难以转化为可执行行为。

方法主线：

```text
Student 真实访问的状态
→ Teacher/Judge 定位问题并生成当前可执行的完整替代动作
→ 从同一快照执行原动作与修复动作，分别真实续跑
→ 原 verifier 验证修复分支确实更好
→ 同状态 action-level DPO
→ 对仍被实际采样器排除的修复 token 加 entry margin
```

Teacher/Judge 提供候选，不作为“已验证改善”的最终裁判；验证结果决定是否进入修复训练。主 RL 的任务 reward 不改。

首轮不另设一个 judge 服务。冻结的 8B SFT Teacher 接收选定状态及受限反馈，直接输出一个原生 assistant/tool-call 格式的替代 turn。最多生成一个候选，非法格式或无法验证时弃用这条辅助样本，不循环生成直到成功。

Teacher 输入加修复目标超过 65,536-token 上下文预算、生成打满上限且动作不完整，或 span/token 对齐失败时，拒绝该辅助样本；不截断到半个工具调用，也不修改原 GRPO 轨迹来适配修复目标。

候选轨迹要求 reward < 1 且存在可检查的执行／产物问题，不把 A < 0 当作必须条件。即使原轨迹的组内 A 为正，也可能有可修复部分；原 GRPO 的 A 与 reward 保持不变。Teacher 输入可以含原任务、真实历史、原动作及其实际反馈、任务未完全完成的事实和已有 anchors；不直接暴露隐藏 verifier 源码、标准答案或密钥。

## 2. 监督单位：完整动作，到新的观察边界为止

记 h 为动作前真实模型上下文，c− 为原 assistant 动作，c+ 为 Teacher 替代动作。正负分支使用各自的自回归 token 前缀，不要求等长或逐 token 对齐。

- 替代动作包含必要的短分析和完整工具调用；若原分析错误，不能保留错误分析再接正确调用。
- 在需要新工具返回值时停止本次 Teacher 目标；后续工具结果由真实环境产生。
- 独立的多个工具调用可使用现有 multi-tool 格式。依赖前一调用返回、且当前 harness 不能在同轮解析变量依赖的调用，不得提前填入假参数。
- 若一个合法 exec 程序能在运行时取得并使用返回值，不必机械拆成多个模型 turn；关键是依赖是否由真实执行满足。
- 不虚构工具结果，不沿用原错误动作的返回，不给短动作人为补 EOS 或续写到原 trajectory 长度。

PPT 的例子：尚无 access_token 时，c+ 可以是 login；其后由环境返回真实 token，再让 Student 决定搜索。验证考察这一步干预对后续结果的影响，并非声称单独监督 login 就教会整个认证流程。

## 3. 从同一状态真实执行，而不是只检查 parser

### 3.1 首轮范围

优先选择文件/代码产物类、同步执行、无外部写入副作用的任务。不在第一轮处理不可恢复的后台进程、共享远端服务修改、网页实时状态和真实账号操作。

每个候选状态最多生成 1 个修复动作。不做树搜索，不反复尝试直至通过。

旧轨迹只有在能严格重建该时刻状态时才可使用；最终 workspace 和文件 hash 不能当作任意历史状态快照。新采集时先预选要保存的边界，事后 Judge 只能从实际保存的边界选择，不能假定任何历史 turn 都能恢复。

快照至少覆盖动作依赖的 workspace 内容、相关文件状态、工作目录和环境配置，以及真实 h、原 token IDs、工具 schema/版本和预算。若依赖未捕获的进程、容器外文件或外部状态，该候选不进入配对验证。先验证原动作能在恢复环境中复现关键结果。

第一轮只选已有结构化 criterion 结果的任务，不先开发通用 verifier 编译器。5 状态 smoke 可人工选定边界；64 候选阶段在新采集中保留近期写入类 turn 与终止响应前的实际快照，每条轨迹至多保留 3 个候选边界、事后只选一个。滚动保存的具体开销在 smoke 中测量，不承诺免费；选中位置无快照就跳过，不能用最终 workspace 顶替。

### 3.2 配对分支

```text
相同 pre-action 状态 h / 快照 W
├─ 原动作 c− → 真实工具返回 → 冻结 Student 续跑 → V−
└─ 修复动作 c+ → 真实工具返回 → 同一冻结 Student 续跑 → V+
```

两分支只在首次强制动作上不同，后续不再由 Teacher 接管。使用同一 checkpoint、生成配置、相同续跑预算和预先规定的配对 seed；不让任一分支读取 hindsight hint。相同 seed 不能保证不同历史下输出相同，也不能消除环境随机性。

首轮候选预算：每分支首次动作后最多续跑 3 个 assistant turn；上下文 65,536、每轮上限 8,192 tokens；两分支使用相同的分支累计 assistant-token 预算 32,768，并计入各自被强制执行的首动作长度。执行加续跑墙钟上限先设为 300 秒／分支；容器初始化与恢复另外计时。该 300 秒是 pilot 的成本上限，不是论文参数。发生超时则整对标记为预算删失，不比较截断一侧与正常结束一侧的 reward。提前正常结束不补齐长度。原分支和修复分支均在同一规则下调用原 verifier。此结果只能称为短窗口改善，不能称为最终任务成功。

两分支的初始化顺序交替，避免某一分支总在更空闲时执行。原动作重放时不沿用旧任务的最终 reward；每次都从恢复状态获取新的工具结果。verifier 只在分支 workspace 的隔离副本运行，避免检查脚本改变后续可恢复状态。

对于 login/read 等收益延迟的动作，短窗口没有改善记为“证据不足”，不是“动作错误”。若此类候选占比高，再单独做预注册的更长窗口实验，报告额外成本，不无限续跑。

### 3.3 接受规则

首轮保守接受条件：

1. 动作在当前可见信息下可执行，工具格式与参数合法。
2. 两分支均无恢复失败、基础设施错误或 verifier 执行异常。
3. `V+ > V− + ε`，其中 ε 仅覆盖经复测确认的评分数值容差，不按结果事后选择。
4. 对能可靠取得且可比较的 criterion vector，修复分支不破坏原分支已通过的检查项。

第 4 条是本实验的保守过滤策略，会降低覆盖率并排除部分有取舍的改善，不宣称为所有任务的最优规则。无法取得可靠 vector 的任务暂不进入这一严格版本；原 GRPO 不因此丢弃合法样本。

初始 pilot 的接受对要再用另一组预先固定的续跑 seed 做一次配对验证；两次都满足接受规则才进入高置信偏好集。所有尝试、两次结果及复测成本均保留，不反复挑 seed 直至成功。重复一致只支持当前状态和预算下的局部改善，不构成全局因果最优证明。

## 4. Loss：GRPO + verified repair DPO + entry margin

主方案只使用三项。对已验证修复序列做最大似然的 repair NLL（SFT 形式）作为强对照；它不是完整 Teacher 词表分布的 FKL，本实验不将二者混称。

### 4.1 完整动作偏好

定义固定参考策略 π_ref（第一轮拟用初始 8B SFT），正负分支均在无特权信息的 h 下评分：

$$
d_\theta(c,h)=\log P_\theta(c\mid h)-\log P_{\mathrm{ref}}(c\mid h).
$$

$$
L_{\mathrm{pref}}=-\log\sigma\left(\beta[d_\theta(c^+,h)-d_\theta(c^-,h)]\right).
$$

其中完整动作 log probability 是各自真实自回归前缀上的 token logprob 求和，包含实际输出边界；不是两个不同长度动作逐位置相减。β=0.1 仅为 pilot 起始候选。

DPO 只保证优化相对偏好，不保证正分支绝对概率单调增加。必须同时记录正/负分支的 logprob 变化及长度分桶，不能只看 preference accuracy。

### 4.2 采样入口 margin

对验证通过的 c+，在 Student 无 hint 的正分支前缀上重算采样分布。设 u_t(v) 为按 rollout temperature 等配置处理、尚未执行 top-k/top-p 截断的采样 logit。

- C_t：按实际 sampler 顺序得到的候选 token 集合。
- b_t：C_t 内最低的有限 logit，作为当前入口边界。
- B：c+ 中目标 token 不在 C_t 的所有位置组成的集合。只检查每个位置上 Teacher 实际生成的那个目标 token，不是优化该位置词表中所有低概率 token。

本版同时优化所有阻断位置，不再限定首个位置：

$$
L_{\mathrm{entry}}=
\begin{cases}
\frac{1}{|B|}\sum_{t\in B}\max\{0,m+\operatorname{sg}(b_t)-u_t(c^+_t)\},&B\ne\varnothing,\\
0,&B=\varnothing.
\end{cases}
$$

例如 B={4,5,9}，三个位置的 hinge 值为 0.6、0.2、0.1，则该 action 的 L_entry=(0.6+0.2+0.1)/3=0.3。一遍更新同时包含这三个位置，无需等待位置 4 被修好后才轮到位置 5。

每次更新前重新确定候选集合和 B；已进入集合的位置不再持续施加入口奖励。m=0.1（温度缩放后的 logit 单位）为 pilot 固定初值，不根据 eval 表现临时选择。

示意：修复 token 的 u=-3.4，边界 b=-2.9，m=0.1，则 loss=0.6；优化方向是抬高修复 token。固定边界下达到间隔对应未截断概率比至少 exp(0.1)≈1.105。边界会移动，所以这不是一次更新后必然进入候选集合的保证。

集合、排序、掩码、|B| 和边界停止梯度，目标 logit 保留梯度。若目标被语法或硬约束禁止，先拒绝候选，不能靠 margin 越过硬约束。没有 top-k/top-p 等截断时，此入口项自然为零。

默认覆盖整个修复目标的有效 assistant tokens，包括必要分析文字；不因为某类措辞看起来不重要就临时屏蔽 loss。工具名、参数、代码与普通文字的分桶只用于诊断。首个越界位置可继续记录为指标，但不再决定训练掩码。这样不能保证额外梯度一定落在关键决策上，是否提高真实行为由对照实验决定。

实现必须与 SGLang 实际 sampler 对照；训练端重建的候选集合若与实际推理不一致，不能把该损失叫做实际 decoder-entry 校正。Teacher-forced 前缀上可采样也不等于 free-running 时一定到达，需测真实生成。

首轮沿用 temperature=0.7、top_p=0.8、top_k=20，准确复现安装版本的采样处理顺序。只在目标 response 位置计算和保留所需边界统计，不保存整段 64K 上下文的全词表 logits。先检验候选集合／目标 membership 的跨后端一致性；不匹配时先修实现，或降级为训练侧近似指标，不把近似值称为真实 SGLang 排除证据。

### 4.3 总目标与归一化

$$
L=L_{\mathrm{GRPO}}+\lambda_p\,\mathbb E_{\mathrm{accepted\ pairs}}L_{\mathrm{pref}}
+\lambda_m\,\mathbb E_{\mathrm{accepted\ pairs}}L_{\mathrm{entry}}.
$$

原 GRPO 保留逻辑轨迹等权与原有数值保护。修复项按 task→pair 等权、每条原轨迹最多一对；entry=0 的接受对仍计入平均，避免越界样本少时权重被自动放大。

β=0.1、m=0.1 为 pilot 初值。学习率沿用已验证的 8B 训练配置作为起点；λ_p、λ_m 及 NLL 对照的系数只在训练侧 one-step pilot 检查 loss/梯度尺度后固定，保留校准记录，不用 held-out/eval 选系数。不得因 DPO 是动作 logprob 求和、NLL/margin 是均值，就把相同数值系数误称为同梯度强度。若 margin 无额外收益，则删除它，不以“组件更多”为目标。

## 5. 接到现有训练时必须保持的边界

- 正负修复分支是独立训练样本，重新构建各自 token/context；不能拼接进原 rollout 的 token ledger。
- h 来自真实 main 上下文，包括已发生的 pruning/compaction；样本按逻辑 turn ID 关联，不能因为 segment 拆分重复训练同一修复对。
- 阶段 A–C 的 Teacher 固定为初始 SFT；阶段 D 使用本轮 rollout-policy 的冻结副本加 hindsight 信息产生修复，记录 policy version，不能在修复生成中途静默换权重。reference 始终固定初始 SFT，不额外训练 judge。Student/ref 评分均去掉 hint。
- 辅助偏好样本不是 on-policy rollout，不冒充 GRPO sample，也不套用原 rollout 的 TIS 权重。状态来自 Student，动作来自 Teacher，属于混合 RL/监督偏好训练。
- 原轨迹按原来的 reward/A 进入 GRPO；修复分支不加入原 8 条采样组成的 reward 组，不覆盖原轨迹、不伪造 rollout-old logprob。修复 token 的 Student/ref logprob 在各自无 hint 前缀下重新计算。
- 不把原动作的 logprob、专家路由或工具返回复制给修复动作。8B dense 试验沿用其适用的数值配置；将来迁移 MoE 时需单独验证修复路径路由。
- 不改 benchmark verifier，不把隐藏答案、实际密钥或额外不可见事实写进 Student 目标；拿不到信息时应生成获取信息的动作。
- 只创建/清理本实验独立命名的容器与 workspace。禁止触碰其他用户的容器。敏感配置仅由运行时注入，不写普通报告和训练样本。

## 6. 执行顺序与对照组

### 阶段 A：5 个状态闭环，只验证工程正确性

验证恢复→原动作重放→生成一个修复→双分支执行→同预算 verifier→重复验证→token/span 对齐→loss 有限值与梯度。5 个状态均须能恢复原分支关键结果；无效 Teacher 候选不要求硬凑成功。恢复、边界或梯度不正确时，停止扩大实验，先修工程。

### 阶段 B：固定 64 个候选状态，检查数据能否产生

从 ClawGym training pool 预选至少 16 个任务，每题最多 4 条候选原轨迹，每条最多一个修复；优先取输出产物附近的可恢复状态。不使用正式 200 题 eval 生成 Teacher 修复或调参。

在生成修复前按 task_id 固定划分约 3:1 的训练／held-out 任务，同题状态不能跨两边。Teacher 不无限重试，分支初次通过后只增加一次固定 seed 复测。记录 64 个候选中恢复成功、可执行、初验改善、复验通过、无 criterion 退步的数量，分别报告分母。

预先约定的工程准入条件：64 个候选中至少 16 对通过两次验证、覆盖至少 8 个任务，且训练侧有可用于更新的接受对。该门槛只避免以极少个例开展训练，不是统计显著性标准。未达到时暂停扩大训练，分析是 Teacher、验证窗口还是恢复覆盖率不足，不通过不断追加候选掩盖失败率。

### 阶段 C：同一批修复数据的最小学习对照

| 组别 | 辅助学习目标 | 回答的问题 |
|---|---|---|
| S0 | 不更新的原 SFT | 新采集状态的行为基线 |
| S1 | repair NLL | 完整修复示范本身是否有用 |
| S2 | repair DPO | 完整动作偏好是否有用 |
| S3 | repair DPO + all-blocked margin | margin 是否带来额外行为收益 |

S1/S2/S3 从同一 8B SFT 权重分别开始，reference 与 Teacher 固定；使用相同接受对、task/pair 采样顺序。先各做一个 optimizer update 检查，再最多训练修复数据 1 个 epoch。起始单步计入该 epoch；若重新设置超参，则所有组重置到共同起点，不多给某组额外更新。保持 optimizer update 数一致，并单独报告实际 tokens、显存和 GPU 时间，不把等更新数称为等计算成本。

此阶段先只测试辅助学习，暂不叠加新的 GRPO 采样，以隔离修复数据与 margin 的效果。学习率与系数经训练侧尺度校准后冻结。复评使用预先留出的 task-disjoint 状态，恢复环境后由各 Student 自己生成首动作并续跑；不强制 Teacher 动作、不提供 hint，使用相同种子与预算。评价执行后的 verifier 结果与工具合法性，不以复现 Teacher 原文字串为成功标准。

当前 64 候选 pilot 的 held-out 数量有限，只能给出可行性证据，不能支撑论文效果结论。需要同时报告全部 held-out 状态与可验证修复子集，不只展示训练集或原本容易修复的状态。

### 阶段 D：通过后接入短程 RL，另行确定启动

先做一次 fresh 8 tasks × 8 samples 的 GRPO+辅助 loss 端到端更新，确认原逻辑轨迹等权、TIS、loss mask、segment 去重与恢复流程未受破坏。没有接受对时该步辅助项为零，GRPO 正常更新。

再以 GRPO+repair DPO 对 GRPO+repair DPO+margin 做 20-step 短程对照；每个 RL step 最多从 64 条原轨迹选 4 个候选、每个任务至多 1 个候选，以控制分支成本。两组使用相同任务顺序、选取规则和候选数量上限；各自轨迹不同后，不能宣称后续修复数据仍完全相同。

短程试验须记录共同起点的完整 200 题 eval，并在第 20 步同协议评测，使用逐题固定 seed；保存完整可续训 checkpoint、eval 轨迹和最终 workspace。未通过前序阶段不启动 20/100 步训练。如果最终要宣称优于普通 GRPO，需要同协议并按额外修复成本核算的 GRPO 对照；旧 100-step 结果仅作历史参照。

### 核心指标与成本

- 数据：恢复成功率、接受对覆盖率、重复验证一致率、criterion regression、超时／超长／格式失败及 task 覆盖。
- 学习：正负动作 logprob、目标候选集合 membership、所有阻断 token 比例、动作中存在任意阻断的比例、工具／参数／代码／普通文字分桶；首个阻断位置仅作为诊断。
- 行为：无 hint 自主生成的合法动作率、同预算 held-out verifier reward、完整 200 题 eval；报告配对差异及不确定性，不仅看 Teacher-forced 概率。
- 成本：Teacher 生成 tokens、快照／恢复／分支环境时间、verifier 次数、训练 GPU 时间、峰值显存及总墙钟。重复验证和失败尝试全部计费。

所有 GPU 实验仅在后续明确启动后执行，按已有隔离方式运行，不停止其他项目或他人任务。仅管理本实验 Ray/SGLang 和专属远端容器；不擅自删除权重或原始数据。实验执行时空闲 GPU 与结束后的 GPU 按用户已有要求运行烤机，前提是不与有效实验进程争抢设备。

## 7. 论文定位与停止条件

拟验证的故事是：完整修复监督解决“替代动作没有展开”，真实执行验证解决“Teacher 提议不一定有效”，entry margin 解决“有效动作仍被实际截断采样阻断”。三者分别对应可表达、可验证、可采样，但这只是待验证假设，不是已成立结论。

已有 [SOPD](https://arxiv.org/abs/2608.16333) 使用 Student 状态上的完整 Teacher step；偏好目标采用既有 [DPO](https://arxiv.org/abs/2305.18290)。另已检索到作者上传的 [REPAIR-DPO 方法稿](https://www.researchgate.net/publication/410595439_Repair_Don%27t_Just_Reject_Step-Level_Counterfactual_Learning_from_Agent_Failures)，同前缀修复与 step preference 存在直接相似，正式投稿前需进一步完整核对。本轮不是全面查新，不能保证组合没有撞车。

停止/降级规则：若恢复无法可信完成，先不做反事实偏好；若短窗口几乎取不到有效对，先报告覆盖率与成本问题；若强 NLL/DPO 对照已能同样恢复实际采样，而 margin 无改善，则不把 margin 当贡献；若主任务 held-out 表现下降，即使正分支 logprob 上升也不能宣称方案成功。
