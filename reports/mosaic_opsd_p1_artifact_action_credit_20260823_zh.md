# Mosaic OPSD P1：Artifact / Action / Token Credit 审计

日期：2026-08-23

## 结论

P1 已证明从“文件级反事实贡献”收紧到“原始 rollout token span”在工程上可行，但当前保守准入规则下信号过稀，尚不应接入正式 RL loss。

- 旧 Phase-0 的 318 条有效轨迹中，9 个可信 cross-parent 任务产生 21 个文件级正贡献，但只有 3 个文件可映射到唯一原子 `write`；旧记录未保存原始 response IDs，因此 0 个可用于 token loss。
- 新增 exact-ID 8×8 rollout 共 64 条、762 个 agent step，762/762 step 均保存原始 response IDs。
- 新一轮有 5 个文件通过 leave-one-file-out 因果门槛，1 个 action 同时通过结构门槛和 exact-token 门槛。
- 当前可训练密度约为 1/64 trajectory，且只覆盖 1 个任务，低于预设的最小 A/B 门槛（100 actions、30 tasks）。

因此当前决策是：保留 extractor 和 exact-ID instrumentation；下一阶段增加“工具执行→文件变化”的 provenance ledger，覆盖 `edit` 以及由 `exec/process` 生成的文件，再重新评估信号密度。不要在只有 1 条 exact action 时改训练目标。

## 三道独立门槛

1. Artifact gate：在 strict Mosaic workspace 中，单独把选中文件替换为其他 parent/initial 版本；要求 reward 严格下降、至少一个 criterion 下降，且没有 criterion 反向改善或受保护路径污染。
2. Action gate：文件 bytes、size、SHA-256 必须精确对应到一个已被 harness 执行并返回结果的原子 `write`；拒绝 multi-tool turn、同路径多次 mutation、后续 opaque mutation 等歧义。
3. Token gate：必须使用 SGLang 实际返回的 `response_ids`。仅重新 tokenize 文本，即使 token 数相同，也不视为原始 rollout IDs。

这三层不能合并：文件有因果贡献，不代表某段自然语言 reasoning 被证明正确；文件能映射到 action，也不代表旧日志拥有可用于 loss 的原始 token 序列。

## 实现

- `clawgym-official/clawgym_rl_api_server.py`：增加显式 opt-in 的 `OPENCLAW_RECORD_RESPONSE_IDS=1`，默认训练日志行为不变。
- `scripts/run_mosaic_phase0_rollouts.py`：Mosaic 审计 rollout 默认开启 exact response-ID 记录。
- `scripts/extract_mosaic_artifact_credit.py`：实现 whole-file leave-one-out、criterion no-regression、原子 action 映射及原始 token span 对齐。
- EOS 处理：SGLang 的原始 IDs 包含 tokenizer 声明的 `<|im_end|>`，而 Proxy `response_text` 不包含。对齐只允许移除尾部声明的 EOS；action span 不包含 EOS，其他文本差异仍严格拒绝。

## 实验结果

### 旧 Phase-0 数据

| 指标 | 结果 |
|---|---:|
| 有效终态轨迹 | 318/320 |
| 纳入的可信 cross-parent 任务 | 9 |
| 测试的 chosen artifacts | 136 |
| 文件级正贡献 artifacts | 21 |
| 唯一原子 write mappings | 3 |
| exact-token eligible | 0 |

旧数据的 0 不是算法否定，而是历史 Proxy 只保存了 `response_text` 和 token count，没有保存真实 `response_ids`。提取器明确拒绝用 tokenizer round-trip 冒充原始 IDs。

### 新 exact-ID 8×8 数据

运行目录：`outputs/mosaic-p1-exact-ids-8x8-20260822-153552`

| 指标 | 结果 |
|---|---:|
| 完成轨迹 | 64/64 |
| agent steps | 762 |
| 保存 exact response IDs 的 steps | 762/762 |
| cross-parent tasks | 3 |
| 测试的 chosen artifacts | 41 |
| 文件级正贡献 artifacts | 5 |
| atomic write mappings | 2 |
| structural-action eligible | 1 |
| exact-token-loss eligible | 1 |

全链路通过样本：

| 字段 | 值 |
|---|---|
| Task | `task_1748` |
| Artifact | `output/project_plan.json` |
| Leave-one-file-out reward contribution | 0.037037 |
| 改善 criterion | `project_plan_risks_ok`, `reward` |
| 原始 response IDs | 1,802 |
| Action token span | `[61, 1801)`，1,740 tokens |
| 尾部终止 token | 1 个 `<|im_end|>`，排除在 loss span 外 |
| Span decode audit | 与原始 `<tool_call>...</tool_call>` 逐字相同 |

另一个 atomic mapping 来自 multi-tool turn，因此被 action gate 正确拒绝。该拒绝不等于轨迹无效，只表示不能把文件级贡献无歧义地归给单个 action span。

## 验证

- 相关完整回归：34 passed。
- 新数据采集：rollout、Mosaic analysis、credit extraction 返回码均为 0。
- 新记录 exact-ID 完整性：762/762 agent steps。
- 最终 exact action 的 decoded span 与原始 raw tool-call 字符串逐字一致。
- 实验结束后 8 张 GPU 已自动恢复 synthetic burn，利用率 99%–100%。

## 下一阶段建议

优先实现只记录、不改变 harness 行为的 artifact provenance ledger：每次 `write/edit/exec/process` 前后记录 workspace 文件 manifest delta，并与 tool_call_id、session、step、原始 response IDs 绑定。

这样可以安全覆盖当前被 `no_matching_atomic_write` 排除的大量文件：脚本生成的 CSV/JSON、命令写出的日志，以及 edit 后的最终文件。对于一次 action 产生多个文件，只给整个 action/mutation set 归因；除非进一步反事实验证，否则不把贡献拆给 action 内的单个文件。

达到至少 100 个 exact actions、覆盖至少 30 个任务后，再做小规模 `GRPO + artifact-local auxiliary loss` A/B。正式训练前仍需分别报告任务正确率、训练 token 成本、梯度占比和是否出现 reward hacking。
