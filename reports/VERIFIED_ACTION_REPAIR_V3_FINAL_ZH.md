# Verified Action Repair v3：最终实验报告

## 结论

本轮结论为 **NO-GO**：不进入 NLL、DPO 或 DPO + 多位置 Margin 训练。固定数据门槛为至少 16 个可靠 train pair、覆盖至少 8 个 train task；实际为 **0 pair、0 task**。本轮没有 optimizer update，也没有用无效修复构造训练结果。

## 方法与关键改造

方案以 8B SFT 模型同时作为 Student 和冻结 Teacher。Teacher 查看原任务、静态 Evidence Anchors 与 Student 轨迹，自主选择待修复的 agent step/call，给出诊断和完整替代动作；允许明确放弃。短轨迹提供完整历史，长轨迹提供机械目录，由同一 Teacher 通过只读 `read_trace` 分页取回原文。

每个候选均在新建的专属容器中从初始 workspace 重放修复点以前的真实动作。重放阶段不请求 Student，工具仍由 OpenClaw 执行；到修复点执行 Teacher 动作，再让 Student 在原剩余预算内完整续跑。初筛改善后才会进行两次固定身份、交替顺序的 original/repair 配对复验。Teacher 动作始终标记为辅助数据，不冒充 on-policy rollout。

工程上还补齐了 seeded sampler 隔离补丁、长轨迹分页、selected-step 二次确认、语义 no-op 拒绝、action-stage 主动放弃、状态漂移审计以及原生 sampler support 的逐 token 记录。相关回归测试为 **72/72 通过**。

## 固定实验

- 数据：预先固定 16 tasks × 4 samples，共 64 条 source；12 train / 4 held-out task。
- 模型：`clawgym-8b-sft-formal-20260902-165551/final_model`。
- 采样：temperature 0.7、top-p 0.8、top-k 20、固定 seed、SGLang deterministic inference。
- 预算：65,536 context tokens；8,192 tokens/step；最多 96 agent steps；1,000 秒 agent 时间。
- 运行：GPU 0 推理、workers=4；GPU 1–7 烤机；隔离端口 32980/32981/40981。

64 条 source 全部正常完成，平均 reward 为 0.7002；平均 10.08 steps，中位数 6，P90 22，最大 96。唯一达到 96-step 上限的 `task_1045-0` 被正常记录为 `max_agent_steps`，不是基础设施失败。本轮 source 未触发 compaction。

## 结果

| 指标 | 结果 |
| --- | ---: |
| source 完成 | 64/64 |
| 合法 Teacher proposal | 9/64（14.1%） |
| 语义 no-op 拒绝 | 16/64（25.0%） |
| Teacher 明确 no-repair | 5/64（7.8%） |
| 无效 proposal | 34/64（53.1%） |
| 真实 repair 完整续跑 | 9/9 |
| 初筛净改善 | 0/9 |
| reward/criterion 不变 | 5/9 |
| criterion 回退 | 4/9 |
| 进入配对复验 | 0 |
| accepted train pair / task | 0 / 0 |

34 条无效 proposal 中，28 条是裸 `call_index: -`；其中至少 20 条文字语义为“任务已完成/无需修复”。这部分同时反映 Teacher 控制格式不稳和 decline 意图表达错误，不能全部归因于轨迹理解。但它不改变方法结论：真正可执行的 9 个 proposal 中，没有一个改善最终验证结果。

长轨迹分页本身可工作，但成本明显：13 条目录模式中 4 条产生合法 proposal，平均 Teacher 耗时 164.4 秒；51 条完整历史中 5 条产生合法 proposal，平均 18.8 秒。目录模式没有降低 proposal 率，却使平均审查耗时约增至 8.7 倍。全部 717 次 Teacher 请求的输入中位数为 20,226 tokens，P95 为 32,805，最大 39,342，未超过上下文上限。

对 9 个 proposal 的 original/repair 动作共完成 18 份原生 sampler audit，覆盖 32,993 个目标 token；270 个 token 位于实际采样 support 之外，占 0.818%。因此多位置 Margin 的触发集合并非空，但在没有可靠正偏好对的情况下训练它没有科学意义。

## 关键证据与归因

- `task_1906-2`：Teacher 认为原任务已完成，却仍给出三工具替代动作；完整续跑后 reward 0.7143→0.7143。
- `task_0921-1`：Teacher 修补一个输出文件，新增一项 criterion，但破坏三项已通过 criterion；reward 0.8333→0.7222。
- `task_0764-0`：修复后丢失一项 summary rule criterion；reward 0.5833→0.5000。
- `task_1043-3`：诊断文本自行推导出“全部要求满足”，却仍提交替代动作；reward 保持不变。

因此失败点位于 **verified repair 数据生成之前**：当前 8B Teacher 在只使用静态 anchors、原轨迹和可见工具结果时，无法稳定识别真正致错动作并生成带净收益的替代动作。DPO、NLL 与 Margin 只能学习已有偏好，不能修复错误标签；在此数据上比较三种 loss 会把 Teacher 噪声误当作算法差异。

## 可复核产物

- 完整 run：`/root/openclaw-rl-local/outputs/verified-repair-v3-20260907-fixed16x4-a13`
- 固定清单：`selection.json`，SHA256 `407c648efb744a7c151a5d4accf5d02883c30342c3b66d0e38e7b9cf039c3eae`
- 终态汇总：`data-summary.json`，SHA256 `47f7ca6542dffb297f42701e66acca61967d68d47e171da15396573c968019ca`
- 执行记录：`data-execution.json`，SHA256 `53431cd735c6a2d8fd62605e72d521b757d3c246ed9f4aa4cff21dd3acf07074`
- 每条真实 Teacher 输入：`teacher/<session>/request-*.json`
- Teacher 决策与替代动作：`teacher/<session>.json`
- 原始轨迹、完整续跑与 verifier：`sources/`、`branches/`、`verifiers/`

总墙钟时间为 6,391.8 秒（约 106.5 分钟）。本次专属远端容器残留为 0；本地实验进程已退出，八卡烤机已恢复。

## 下一步建议

不要扩大当前协议或启动 100-step 训练。若继续该方向，下一轮首先只做数据生成消融：将修复 Teacher 升级为更强的冻结模型，或提供经过防泄漏编译的 criterion-level failure witness；仍沿用本轮真实重放和配对复验门槛。只有可靠修复对达到预注册覆盖率后，再比较 NLL、DPO 与多位置 Margin。
