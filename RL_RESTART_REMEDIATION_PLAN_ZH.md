# OpenClaw RL 正式重跑改造计划

状态：实施前基线
工作目录：`/root/openclaw-rl-local`
训练目标：ClawGym 论文口径的 Qwen3-30B-A3B GRPO 正式重跑

## 1. 目标与边界

本轮不从已经异常的 RL checkpoint 续训，而是从固定的 SFT checkpoint
`qwen3-30b-a3b-instruct-clawgym-sft-iter11127` 重新启动。正式启动前，
必须消除已经确认的训练语义错误，并为可能再次出现的问题提供可审计证据。

代码和新输出只允许写入 `/root/openclaw-rl-local`。挂载盘上的模型与
Python 环境可以只读复用；不得修改 `/mnt/Openclaw-RL-hareness` 或
`/mnt/OpenClaw-RL` 中的代码。远端 rollout 镜像按 digest 固定。

明确不改：

- reward/verifier 规则；
- GRPO advantage 的组内归一化；
- 全失败组或低 reward 样本的既有训练策略；
- KL、entropy 配置；
- 任务数据划分和固定 seed 语义。

这些项可以观测，但不能借本轮修复顺便改变算法口径。

## 2. 当前已确认的问题

### 2.1 长度含义混用

当前正式脚本把 `OPENCLAW_MAX_TOKENS=65536` 同时用于 OpenClaw 单次生成
和 Slime 的 `rollout-max-response-len`。正确口径应为：

- 单次 assistant generation 上限：8192 tokens；
- 整条训练 trajectory/context 上限：65536 tokens；
- SGLang context length：65536 tokens。

单步响应预算与整条轨迹预算必须使用不同变量，且在启动日志和记录中同时
可见。

### 2.2 上下文压缩后的错误拼接

入口：`clawgym-official/clawgym_rl_api_server.py`。

现实现发现 prompt 改变后，会寻找旧累计序列与新 prompt 的最长公共前缀，
再把 `prompt_ids[shared:]` 接到旧序列。这会把压缩后的新上下文错误地
拼到旧上下文尾部，使 rollout logprob 对应真实新上下文，而训练 loss 却
对应一个不存在的混合上下文。

正确规则：

1. 仅当新 prompt 完整以前一累计 token 序列为前缀时，才允许 append；
2. 条件是 `len(prompt_ids) >= len(all_tokens)` 且
   `prompt_ids[:len(all_tokens)] == all_tokens`；
3. 否则立即 flush 当前训练 segment，并以精确的新 prompt 开始下一段；
4. 不得再使用最长公共前缀 suffix 拼接；
5. 同一原始 rollout 的分段保持相同 `traj_index` 和 `group_index`；
6. terminal reward 与 advantage 语义不变；
7. 每个训练 sample 的总长度不得超过 65536。

### 2.3 response token 与 rollout logprob 不严格对齐

入口：`clawgym-official/clawgym_rl_api_server.py`。

当前代码对重新编码后的 response token 和 chat-completions 返回的 logprob
做静默截断或补零。这会让训练 token 不一定是采样时真正生成的 token，
补零位置也不是真实 rollout policy logprob。

修复要求：

- 采样端返回原始生成 token ids 及其逐 token logprob；
- 训练 response ids 必须直接采用这组原始 token；
- 强制 `len(response_ids) == len(response_logprobs)`；
- raw generated token ids 必须等于送入训练的 response token ids；
- 任一条件不满足时显式失败或隔离样本，并记录原因；
- 禁止截断、补零和“重新 tokenize 后假定等价”。

实现前需用 mock 响应验证 SGLang 当前版本的 `/generate` 元信息格式及
tool-call parser 行为，避免依赖未提供 token ids 的 chat API 字段。

### 2.4 multi-tool 被截断

入口：`clawgym-official/clawgym_rl_api_server.py`。

当前 proxy 在解析后执行 `tool_calls = tool_calls[:1]`，导致模型同一
assistant turn 生成的第二个及后续调用没有交给 OpenClaw。SFT 中大量轨迹
训练过一个 turn 输出多个 `<tool_call>` 块，因此这是明确的 SFT/RL
交互语义不一致。

修复要求：

- 移除首调用截断；
- 按模型生成顺序完整返回所有解析成功的 tool calls；
- 每个调用保留唯一 `tool_call_id`、函数名和参数；
- OpenClaw 返回的每个 tool result 必须能按 id 关联到对应调用；
- 下一轮 prompt 中调用与结果顺序可验证；
- 某个调用失败不能让其他调用的记录消失；
- 原始调用数、解析成功数、返回数、下一轮观察到的结果数均写入审计记录。

不自行串行执行工具；proxy 只负责忠实转发，执行并发/顺序由当前 OpenClaw
协议和 agent loop 决定。

### 2.5 Serper 配置缺失

Serper 作为环境能力修复，不修改 reward。URL 与 key 必须通过环境变量
注入远端容器，不写入 prompt、trajectory、普通日志、W&B 或 checkpoint。

至少覆盖以下测试：

- URL/key 都存在时可完成一次受控请求；
- 缺 URL；
- 缺 key；
- 401/403；
- 超时；
- 响应格式错误；
- 所有错误记录均脱敏。

## 3. 实施顺序与 commit 边界

### 阶段 A：隔离运行基线

已完成的 commit：

- `85dfb6e2 Pin local RL runtime paths and output boundary`

验收：代码根、Slime、Megatron 和输出均被限制在本地副本；模型与 Python
环境只读引用挂载盘；镜像使用 digest；预检模式不会启动 Ray/SSH/训练。

### 阶段 B：长度变量拆分

修改启动脚本与 rollout 配置，将单步 response 设为 8192、trajectory/context
设为 65536。增加配置断言，禁止 response 上限大于 context 上限。

验收：单元/静态测试能同时观测两个不同限制，启动参数无歧义。

### 阶段 C：压缩分段正确性

把 prompt transition 判定提取为可测试纯函数；实现严格前缀 append 与
非前缀 flush。构造普通追加、工具结果追加、压缩替换、共同前缀但非追加、
正好 65536、超过 65536 六类测试。

验收：任何压缩后的 sample 都不含旧上下文尾部；分段共享原 trajectory/group
身份；reward 只按既有规则赋值。

### 阶段 D：token/logprob 一致性

接入原始生成 token ids，删除截断和补零。增加长度、token 一致性断言以及
结构化错误计数。

验收：正常样本逐 token 一一对应；故意制造少一个 logprob、多一个 logprob、
重新编码不一致时都拒绝进入训练 buffer。

### 阶段 E：multi-tool 全量转发

移除 `[:1]`，保留调用 id 和顺序，增加协议级单元测试与 mock OpenClaw
往返测试。

验收：一个 assistant turn 输出 1、2、3 个调用时分别返回 1、2、3 个；
多个结果在下一轮可按 id 和顺序核对；解析失败可审计。

### 阶段 F：Serper 与秘密管理

实现显式配置注入、远端容器传递、错误处理和脱敏。真实 URL/key 仅在正式
启动环境提供，不进入 Git。

验收：受控联网测试成功；负向测试齐全；仓库和输出扫描无 key。

### 阶段 G：集成预检与 smoke test

按固定 seed 先跑最小任务集合，检查：

- 每步 response、整轨迹和分段长度；
- context transition 类型与压缩次数；
- token/logprob 对齐拒绝数必须为零；
- raw/parsed/returned/observed tool-call 数；
- tool-call id/result 关联；
- Serper 成功率与脱敏；
- task id、seed、group、trajectory、segment 的关联；
- reward 分布只做对照，不改变计算。

通过后再从 SFT checkpoint 正式启动，不接续异常 RL checkpoint。

## 4. 审计记录最小字段

每个模型 step 至少记录：

- `task_id`、`seed`、`group_index`、`traj_index`、`segment_index`；
- `prompt_token_count`、`response_token_count`、`segment_token_count`；
- `context_transition`：initial、append、compressed/reset；
- `strict_prefix_match`、`segment_flush_reason`；
- `response_id_count`、`response_logprob_count`、`token_alignment_ok`；
- `raw_tool_call_count`、`parsed_tool_call_count`、
  `returned_tool_call_count`、`observed_tool_result_count`；
- tool-call ids、工具名和脱敏后的状态，不记录秘密参数；
- terminal reward 和 verifier 状态。

聚合指标至少包括：压缩分段率、对齐拒绝率、多工具出现率、调用解析/返回/
观察漏失率、每个 skill 的调用与成功率。

## 5. 测试与回滚原则

每个阶段一个独立 commit，先单元测试，再做不占 GPU 的集成预检。涉及真实
rollout 的 smoke test 在停止压力任务并确认 GPU 可用后执行。

任何阶段失败时只回退该阶段 commit；不得修改挂载盘恢复代码。正式训练前
保存最终 commit、完整启动参数、环境包清单、镜像 digest、任务清单 hash、
SFT checkpoint hash 与固定 seed。正式 run 的输出目录必须全新且位于
`/root/openclaw-rl-local/outputs`。
