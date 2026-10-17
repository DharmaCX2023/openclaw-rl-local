# Hindsight OPSD for Agentic Action-Span Repair

## 0. 核心定位

本文关注 agentic OPD / OPSD 在失败轨迹上的一个结构性问题：teacher 能在 hindsight feedback 下知道 student 哪里走偏、后面应该如何继续或修复，但传统 token-level OPD 只能在 student 已经采样出的失败 token 上做 log-prob / KL 监督，无法把 teacher hindsight 后发现的“更长 continuation、更合理 action、更晚终止或不该 EOS”的信息有效传回 student。

因此，本文的核心不是简单地在同一个 state 上随机采样 action，而是：

```text
在 student-visited state 上，
用 teacher hindsight feedback 生成更优 action / continuation segment，
再通过 action-span preference 或 segment-level embedding bridge，
把 hindsight 后的修复信息反馈给 student 的实际采样分布。
```

一句话概括：

```text
当 token-level OPD 无法表达 teacher hindsight feedback 时，
应在 student 访问到的状态上进行 action-span / segment-level 的 on-policy repair distillation。
```

## 1. 背景：为什么 token supervision 会失败

标准 OPD 通常沿 student rollout 轨迹做 token-level 监督。例如 student 生成了一条失败轨迹，teacher 对这条轨迹计算 log prob 或 token KL。如果 teacher 认为这条轨迹不好，标准做法只能压低 student 已经生成出的 token。

问题是，teacher 的真实反馈经常不是“这些 token 每一个都错了”，而是：

- student 不该在这里结束，不该输出 EOS；
- student 应该继续 rollout 更长的 segment；
- student 当前 action span 缺少前置动作；
- student 需要替换工具、补参数、拆分动作；
- student 需要先检查状态、读 schema、执行 recovery action；
- student 的短轨迹前半段信息其实够用，错在末尾过早停止或 action 选择。

这时 token-level OPD 会出现监督错位：它只能惩罚已有 token，却不能表达 teacher hindsight 之后“应该继续往哪里走”。

这也是 Hint-SD 类方法的问题：teacher feedback 可能描述了修复方向，但没有把明确的 action / continuation 信号传递给 student，导致 student 只能吸收模糊提示，仍然存在 action mismatch。

## 2. 本文要研究的 mismatch

### 2.1 Token-to-continuation mismatch

student 失败轨迹过短或过早终止时，teacher hindsight 认为后面仍然有可行 continuation。标准 OPD 只能在已有短轨迹上压低 EOS 或末尾 token，却无法直接示范后续 segment。

### 2.2 Feedback-to-action mismatch

teacher feedback 可能指出原因，例如“缺少 authentication”或“参数来自不可见状态”，但如果没有明确 action / decision target，student 很难把 feedback 转成可执行行为。

### 2.3 Action-span projection mismatch

agent 修复经常不是改几个 token，而是插入、替换、拆分或延长 action span。例如：

```text
Student action:
search_playlists(query="workout")

Teacher hindsight repair:
Reason: missing_authentication
Action segment:
login(...)
search_playlists(query="workout", access_token=login_result["access_token"])
```

这个修复无法压回 student 原 action span。

### 2.4 Thought-action mismatch

如果只替换 action，而保留 student 原错误 thought，就会形成“错误依据 + 正确动作”的训练样本。更合理的监督单位应是 action-span decision 或 short continuation segment，而不是孤立 action token。

## 3. 与 CHOP 的关系

这里借鉴 CHOP 的不是 mean pooling 或 SWD 本身，而是三个原则。

第一，不要在低可靠状态继续做 token-level OPD。应先判断当前 state / token supervision 是否可靠；可靠时做 token KL，不可靠时切到 segment-level repair。

第二，embedding alignment 应该约束 student 实际采样的 continuation，而不是只预测一个 teacher embedding。也就是说，student 需要自己采样 segment，再通过这个 segment 的 log-prob 接收训练信号。

第三，segment 不必是完整轨迹。可以是短 continuation segment，例如 action span 或固定长度 continuation，而不是一直 rollout 到 terminal。

因此本文更接近下面这个形式：

```text
student segments:
ĉ_i ~ π_θ(. | h_t)

teacher hindsight segments:
c_j^+ ~ π_T^H(. | h_t, failure feedback)

student embeddings:
ẑ_i = φ(h_t, ĉ_i, τ̂_i)

teacher hindsight embeddings:
z_j^+ = φ(h_t, c_j^+, τ_j^+)

bridge loss:
L_bridge-emb = D({ẑ_i}_{i=1}^N, {z_j^+}_{j=1}^M)
```

关键点是：

```text
不是预测 teacher embedding，
而是 student 采样 segment -> 编码 -> 与 teacher hindsight segment 分布比较 ->
用这个距离作为训练信号更新 student 采样分布。
```

## 4. 方法概述

定义 student rollout：

```text
τ^S = (h_1, a_1, o_1, ..., h_T, a_T, o_T)
```

其中：

```text
h_t: student 访问到的 pre-action state，包括任务、历史 action、observation、工具返回和 tool schema。
a_t: student action span，可以是工具调用、final answer、EOS 或 short continuation。
f_t: teacher hindsight feedback，解释当前失败或低可靠原因。
c_t: continuation segment，可以是 action span，也可以是短 rollout segment。
```

方法分为三步。

### 4.1 低可靠状态定位

先判断哪些 state 不适合继续做 token-level OPD。可靠性可以来自：

- teacher 对 student token / action 的 log-prob 或 margin；
- 是否过早 EOS；
- 工具调用是否 invalid；
- action 是否与 tool schema 不匹配；
- teacher feedback 是否指出当前 action span 需要结构性修复；
- downstream rollout 是否失败。

形式上可以定义：

```text
ρ_t = reliability(h_t, a_t^S, τ^S, π_T)
```

当 `ρ_t` 高时，保留标准 token-level OPD；当 `ρ_t` 低时，切换到 action-span / segment-level repair。

### 4.2 Teacher hindsight action / segment 生成

对低可靠 state，teacher 不只是评价 student 原 action，而是在同一个 `h_t` 上结合 failure feedback 生成明确的修复信号：

```text
a_t^+ or c_t^+ ~ π_T^H(. | h_t, f_t)
```

这里的 `a_t^+ / c_t^+` 是 teacher hindsight action / segment，而不是随机采样 action。它可以包括：

- 替换工具；
- 补全参数；
- 插入前置动作；
- 拆分多个工具调用；
- 继续 rollout 而不是 EOS；
- 执行 epistemic action，例如 inspect schema、read state、run test；
- recovery action。

### 4.3 两种训练信号

第一种是 action-span preference：

```text
(h_t, c_t^+) > (h_t, c_t^-)
```

其中 `c_t^-` 是 student 原 action span 或低质量 continuation，`c_t^+` 是 teacher hindsight continuation。

第二种是 CHOP-inspired embedding bridge：

```text
ĉ_i ~ π_θ(. | h_t)
c_j^+ ~ π_T^H(. | h_t, f_t)
ẑ_i = φ(h_t, ĉ_i, τ̂_i)
z_j^+ = φ(h_t, c_j^+, τ_j^+)
L_bridge-emb = D({ẑ_i}, {z_j^+})
```

训练时不只加 prediction head，而是用 student sampled segments 的 log-prob 更新 student 生成分布。直观上，就是让 student 更可能采样出 embedding 分布接近 teacher hindsight continuation 的 segments。

## 5. 可部署性约束

Teacher hindsight 不能泄露 student 部署时不可见的信息。一个 teacher repair segment 必须满足：

```text
Executable:
工具调用格式合法，能被 parser / executor 接受。

Grounded:
参数来自当前 h_t 可见信息、tool schema、历史 observation，
或来自 segment 内前序 action 的返回值。

Improving:
相对 student 原 action span，能减少 invalid、提升 task progress，或改善 recovery。
```

如果当前 state 信息不足，teacher 不应直接给最终动作，而应给 epistemic continuation：

```text
inspect_tool_schema
check_state
read_file
run_test
query_database
```

## 6. 训练目标草案

完整目标可以写成：

```text
L = L_reliable-token + λ L_action-pref + μ L_bridge-emb
```

其中：

```text
L_reliable-token:
只在高可靠 token / state 上做标准 OPD 或 token KL。

L_action-pref:
在低可靠 state 上比较 teacher hindsight segment 与 student 原 segment。

L_bridge-emb:
让 student sampled continuation segment 的 embedding 分布靠近 teacher hindsight segment 分布。
```

Action preference 可以先用 DPO 形式：

```text
L_action-pref = -log σ(β[(logπθ(c+|h)-logπref(c+|h)) - (logπθ(c-|h)-logπref(c-|h))])
```

Embedding bridge 可以先作为后续版本实现，第一阶段先做 teacher hindsight action-span pair。

## 7. 实验计划

### 7.1 诊断实验

目标是证明 token-level OPD 在失败轨迹上确实无法表达 teacher hindsight feedback。

统计类型：

```text
premature EOS
needs longer continuation
tool replacement
argument completion
precondition insertion
action split
epistemic action
recovery action
```

核心诊断指标：

```text
Hindsight Projection Mismatch Rate
Action-span Repair Rate
Premature Termination Rate
Feedback-to-action Ambiguity Rate
```

### 7.2 主实验环境

第一阶段：BFCL v3。

原因：BFCL 适合验证工具选择、参数合法性、可执行性和多步 function calling。这里重点不是随机 state-fork，而是：

```text
student failed action span
teacher hindsight feedback
teacher repaired action / continuation
student sampled candidate continuation
```

第二阶段：AppWorld 或 tau-Bench。

原因：它们更适合验证 recovery、状态变化、epistemic action 和长链 workflow。

### 7.3 Baselines

```text
Base agent
Standard OPD on failed rollout tokens
Hint-SD / feedback-only distillation
Action-only repair SFT
Teacher hindsight corrected trajectory SFT
State-anchored teacher action preference
CHOP-style segment bridge
RL-only / GRPO
RL + teacher hindsight OPSD
```

### 7.4 核心指标

```text
task success
invalid-call rate
wrong-tool rate
argument error rate
premature EOS rate
continuation recovery success
teacher feedback utilization
action-span repair success
state recovery success
average tool calls
over-repair rate
```

### 7.5 关键消融

```text
without teacher feedback
feedback-only, no explicit teacher action
teacher action SFT vs teacher action preference
student random candidates vs teacher hindsight candidates
action-span only vs short continuation segment
without reliability gate ρ_t
without deployability check
embedding prediction head vs sampled-segment bridge
```

## 8. 预期贡献

本文可以主张三点贡献：

1. 识别失败轨迹 OPD 中的 hindsight feedback mismatch：teacher 能发现更长 continuation 或结构性 action repair，但 token-level OPD 只能惩罚 student 已有 token。

2. 提出 state-anchored teacher hindsight action / segment supervision：在 student 访问到的 state 上，用 teacher feedback 生成明确 action-span 或 short continuation repair。

3. 提出 CHOP-inspired segment bridge：不只是预测 teacher embedding，而是约束 student 实际采样 continuation 的 embedding 分布，使 teacher hindsight 信息能反馈到 student 生成分布中。

## 9. 最终一句话

本文研究的是：当 token-level OPD 无法把 teacher hindsight feedback 传递给 student 时，如何在 student-visited state 上通过 teacher repaired action span、short continuation segment，以及 sampled-segment embedding bridge，把失败轨迹后的更多信息重新注入 student 的 on-policy learning。
