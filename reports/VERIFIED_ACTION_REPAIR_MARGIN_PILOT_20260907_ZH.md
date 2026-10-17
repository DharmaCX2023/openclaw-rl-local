# Verified Action Repair：预实验进展

本文件保留过程记录；实验现已结束，最终数字与结论见 [完整结果报告](VERIFIED_ACTION_REPAIR_MARGIN_RESULT_20260907_ZH.md)。下列中间截面不代表最终 64 候选结果。

更新：2026-09-07。依据 v2 方案。**工程闭环通过；64/64 源轨迹采集完成，正式配对验证进行中，尚无方法有效性结论。**

## 方法与当前范围

从真实 Student 轨迹保存动作前状态；冻结的同一起点 8B SFT Teacher 给出一次完整替代动作。原动作与修复动作从同一状态执行，各最多续跑 3 个 turn、300 秒，由原 verifier 比较。严格改善、无 criterion 退步且通过第二组固定 seed，才作为 DPO 偏好对。Margin 覆盖修复动作的所有实际采样阻断位置。当前没有执行模型参数更新。

## 五状态工程检查

| 任务 | 原分支 reward | 修复分支 reward | 结果 |
|---|---:|---:|---|
| task_0126 | 0 | — | Teacher 重复字符至 8192-token 上限，工具调用未闭合；拒绝 |
| task_0133 | — | — | 原任务最终已成功，仅检查恢复；首动作重放一致，后续安装依赖超出首轮范围 |
| task_0140 | 0.95 | 0 | Teacher 声称文件已写入，但未执行写入；criterion 退步 |
| task_0158 | 0 | 0 | 修复代码仍触发原来的日期类型错误，无严格改善 |
| task_0162 | 0.8667 | 0 | Teacher 直接给出完成总结，未生成要求的文件；criterion 退步 |

5/5 状态的恢复文件清单、哈希及原动作关键工具结果一致。4 次 Teacher 提议中，3 次格式合法，0 次形成接受对；另 1 个已成功任务不生成失败修复。上述是工程样本，不是 64 候选的覆盖率结果，不据此提前给出 NO-GO。

31 项 CPU 回归测试通过，覆盖实际 seed、失败尝试计时、固定分母、信息边界审核绑定、UTF-8 token 对齐及最终报告完整性检查。新增分支审计也已在全部 5 个真实 smoke 上通过。8 个完整动作、10,029 个 token 的原生 teacher-forcing 审计确认目标 ID 与前缀一致。5 个状态的真实 8B backward 均为有限值，0 次 optimizer update；新版 backward 合计约 37.4 秒，峰值显存 47.7 GiB，不含加载与服务切换时间。诊断中的 loss 系数不是后续训练系数校准。

## 采样边界的实现修正

训练端与 SGLang 自行重建的 membership 在 4,715 个目标位置中有 5 处不同。不能将训练端近似集合当成实际 sampler 的决定。

现使用相同权重、相同前缀下原生 SGLang 记录的候选 token 集合 C 决定阻断位置 B；训练侧在 C 内计算最低温度缩放 logit，停止该边界的梯度，仅保留目标 logit 的梯度。47 个实际阻断位置均被覆盖，未采用首位置筛选。两个后端的 logit 仍不完全一致：阻断位置上 native/train hinge 的平均绝对差约 0.139，保留为诊断，不能声称数值完全一致。

旧结果保留在 `gradient_smoke/`，原生集合版保存在 `gradient_smoke_native_support/`。目前权重未更新，原生集合可复用；真正训练时每次更新前必须刷新集合，相关推理成本单独记录。

## 下一阶段与停止门槛

正式采集于 UTC 05:25:55 成功结束，64/64 均取得选定状态，覆盖 16 题（12 train / 4 held-out）；64 个真实快照的哈希和文件清单全部通过核验。采集包装器墙钟 3,354 秒（55 分 54 秒），包含准备、rollout、快照、验证和烤机交接，不是纯推理耗时。

UTC 05:26:15 两分片开始运行。最新一轮独立文件审计已覆盖 47 对：实际恢复、原动作重放、续跑 seed、verifier 输出与原生 action-token 证据一致。实时结果与完整成本字段由 `scripts/summarize_verified_repair_data.py` 汇总到运行目录的 `analysis/`，未完成指标保留为空，不记为零成功或零成本；独立审计的时点和逐例结果见 `analysis/independent_artifact_audit_progress.json`。

固定 16 题 × 4 条轨迹，共 64 个候选；任务级 12/4 train/held-out 划分已在生成前固定。采集 workers=4，Gateway 启动并发=2，GPU 0 推理，其余 7 卡烤机。只操作本实验的远端容器。

正式配对阶段按预先固定的 source-session 字典序分成 32+32 两片，GPU 0/1 各一个独立 SGLang、Proxy 与远端命名空间；每片同时运行一个分支，原/修复比较始终使用同一引擎。其余 6 卡烤机。容器初始化次序按候选和复验序号交替，配对 seed 与预算不变。

衔接器在 `verified_repair_pairs64_20260907` tmux 中等待本次采集包装器的实际 PID 与启动时间标识；确认退出码 0、64 条固定候选齐全后才启动冻结的 `launcher-data-shards.sh`。启动前检查本地 30970–30973 和远端 39971/39973；若端口被占用则退出，不接管已有服务。此衔接不代表数据门槛已通过，也不自动启动训练。

只有至少 16 个双重复验通过的偏好对、覆盖至少 8 题且训练侧有接受对，才进行 NLL / DPO / DPO+margin 的同数据对照；否则报告覆盖率、失败原因和完整成本，不追加候选凑门槛。当前不启动 20/100 步训练。

## 复查入口

- [信息边界审核规则与首例](../outputs/verified-repair-v2-20260907-pilot-a3/INFORMATION_BOUNDARY_AUDIT.md)
- [最新完整 CPU 回归](../outputs/verified-repair-v2-20260907-pilot-a3/unit_tests-final-report-guards.xml)
- [v2 方案](VERIFIED_ACTION_REPAIR_MARGIN_PLAN_20260907_ZH.md)
- [工程准入证据](../outputs/verified-repair-v2-20260907-pilot-a3/smoke_passed.json)
- [逐状态审计](../outputs/verified-repair-v2-20260907-pilot-a3/smoke_audit.json)
- [真实分支结果](../outputs/verified-repair-v2-20260907-pilot-a3/smoke-branches.json)
- [真实 backward 汇总](../outputs/verified-repair-v2-20260907-pilot-a3/gradient_smoke_native_support.json)
- [工程尝试与修正溯源](../outputs/verified-repair-v2-20260907-pilot-a3/PROVENANCE.md)
- [正式任务与分片清单](../outputs/verified-repair-v2-20260907-pilot-a3/data_branch_shards.json)
- [新增回归结果](../outputs/verified-repair-v2-20260907-pilot-a3/unit_tests-prebranches-final.xml)
- [完整 64 状态快照审计](../outputs/verified-repair-v2-20260907-pilot-a3/source_snapshot_final_audit.json)
- [代码与依赖版本快照](../outputs/verified-repair-v2-20260907-pilot-a3/prebranches_code_manifest.json)
- [分母与成本汇总测试](../outputs/verified-repair-v2-20260907-pilot-a3/unit_tests-prebranches-summary.xml)
- [正式候选进度与成本](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/progress_summary.json)

所有旧尝试、被拒提议及失效实现的结果保留。最终报告还需补齐 64 候选的全部结果、成本和适用的学习对照；本文件不是最终结论。

## 首个分数改善与信息边界问题

`source-task_0626-2` 两组 seed 均得到原分支 0.878431、修复分支 0.941176，新增通过 1 项检查，恢复与 native token 审计通过。但 Teacher 先写出 “I've created all three output files as requested.”，随后读取三个输出文件。真实动作前快照中没有这些文件，此前只有读取输入的动作；三个 read 在两次复验中均返回 ENOENT，改善来自后续 Student 的恢复。

该结果保留为“双 seed 验证改善”，但不计作可训练偏好对：完整修复目标含有把原动作执行后的事实错误投射到当前状态的内容，违反 v2 的信息边界要求。没有删除前句后冒充原已验证动作，没有改 reward 或重生成。读取缺失文件本身不构成拒绝理由。

后续所有双 seed 改善均需同样的事实审核，决定绑定状态文件与 Teacher token IDs 哈希；待审核单列，不默认接受或判失败。各分片原始 `pairs/*.json` 的 `accepted` 仅是执行验证阶段标志，最终门槛以合并审计结果为准。

当前图表带有 PROVISIONAL 标识，不能作为最终效果结果：

![候选验证进展](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/verification_outcomes_progress.png)

![修复目标的原生采样阻断分布](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/repair_token_blocking_progress.png)

分桶仅用于诊断，不修改 loss。原始 token IDs 按 ByteLevel 字节重建并核对全文，未重新 tokenize；无法精确映射时单列 unmapped。阻断分布包含未接受的提议，不能据此直接推断 margin 的学习效果。

成本台账同时保留 6 个已完成工程阶段，共 2,838 秒单模型 GPU 槽位墙钟（含失败阶段，不是 kernel-active 时间）；两次接口探针发生在这些区间内，不另行重复相加。另有一次烤机交接失败未进入模型阶段，缺少精确阶段起止标记，保留日志并明确标注时长不可精确恢复。正式 64 候选成本与这些工程成本分开报告。

当前采集命令：

```bash
cd /root/openclaw-rl-local
bash outputs/verified-repair-v2-20260907-pilot-a3/launcher-data-collect.sh \
  data-collect /root/openclaw-rl-local/outputs/verified-repair-v2-20260907-pilot-a3 4
```

该采集命令已成功结束，请勿重复启动。当前运行的是 `verified_repair_pairs64_20260907` 内的两分片配对验证；启动命令为：

```bash
bash outputs/verified-repair-v2-20260907-pilot-a3/launcher-data-shards.sh \
  /root/openclaw-rl-local/outputs/verified-repair-v2-20260907-pilot-a3
```

仅用于复查命令；正在运行时不要重复执行。

## 中间行为诊断：不是所有拒绝都来自同一原因

在一个固定中间截面（46 个 Teacher 输出、45 个已完成首次配对）中：17 个修复没有工具调用，其中 11 对分数下降；9 个已验证的 write/edit 修复均与原分支持平。原动作 token 中位数 607.5，修复动作 404.5；动作前上下文中位数 13,393.5 tokens，范围 7,904–33,872。该截面尚不能代表最终 64 个候选，长度差异也不是效果下降的因果证明。

- `source-task_0190-1`：原动作写出四个账单相关产物，重放后分数 0.601504；Teacher 改为 exec 检查尚未写出的文件，冻结 Student 续跑到 3 turn 后仍为 0。它说明“先检查一下”不一定能在本窗口内完成替代；不能因此断言更长窗口也无效。
- `source-task_0921-2`：修复改为读取 citations.jsonl，再由 Student 生成后续动作。第一次分数由 0.833333 升到 0.888889，新增 `citations_quotes_match_inputs`；第二组 seed 仍为 0.833333，因没有重复获得严格改善而拒绝。这个例子直接支持保留复验规则。
- `source-task_0626-2`：双 seed 分数改善成立，但完整修复目标的信息边界不合格，详见上节。不能把“能提高分数”和“可直接用于完整动作监督”混为一谈。

这些是已经发生的短窗口执行结果，还没有训练 NLL/DPO/DPO+margin，因此不能据此判定 margin 无效。低接受率当前首先限制了可训练数据的形成。

可复查统计：[逐候选行为 CSV](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/candidate_behavior_progress.csv)、[行为分组 JSON](../outputs/verified-repair-v2-20260907-pilot-a3/analysis/behavior_statistics_progress.json)。工具分类使用保存的 parser 结果；`exec` 不自动视为写入，纯文字也不自动视为非法。统计代码不修改奖励或 loss mask。
