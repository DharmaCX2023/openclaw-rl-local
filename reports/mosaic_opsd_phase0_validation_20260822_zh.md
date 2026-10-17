# Mosaic-OPSD Phase-0 验证报告

日期：2026-08-22  
结论：**“同题失败轨迹之间存在可验证的产物互补性”得到小规模实证支持；“能够把失败轨迹组合为完整成功”尚未得到支持。**

## 1. 要验证的假设

对同一道 ClawGym 任务独立采样多条 on-policy 轨迹。即使每条轨迹都未满分，它们可能分别生成了正确的不同文件。若只从这些轨迹中选择模型真实生成过的文件版本，不生成新内容，并使用任务原始 verifier 重新验证，组合 workspace 可能严格优于所有父轨迹。

本阶段只验证该现象是否存在，不训练模型，也不把组合结果直接当作完整 SFT 轨迹。

## 2. 实验设置

- 模型：`/mnt/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127`
- 两个互不重叠的任务 seed：seed 0 为 8 个任务，seed 1 为 32 个任务，共 40 个独立任务。
- 每题计划采样 8 条轨迹，共计划 320 条；实际得到 318 个 terminal workspace。
- 采样为真实随机 on-policy rollout：SGLang deterministic inference 关闭，不设置 per-request sampling seed；只固定 tool-call ID，便于审计。
- 拓扑：8 GPU，TP=2、DP=4，并发 32；CUDA graph 关闭。
- 扩展轮耗时 3,310.8 秒，生成 1,395,243 个 response token；SGLang prefix-cache hit rate 为 83.61%。
- 扩展轮发生 2 条 pre-reward 基础设施失败，因没有 terminal workspace 而不进入组合分析，绝不伪装成 reward=0。

扩展采样入口：

```bash
bash /root/openclaw-rl-local/scripts/run_mosaic_phase0_extended_20260822.sh
```

扩展轮离线分析：

```bash
/root/openclaw-rl-local/.venv-openclaw/bin/python \
  /root/openclaw-rl-local/scripts/analyze_mosaic_phase0.py \
  --gate-output-dir /root/openclaw-rl-local/outputs/mosaic-phase0-32x8-seed1-20260822-023000/gate \
  --archive-dir /root/openclaw-rl-local/outputs/mosaic-phase0-32x8-seed1-20260822-023000/archives \
  --output-dir /root/openclaw-rl-local/outputs/mosaic-phase0-32x8-seed1-20260822-023000/analysis-v2 \
  --beam-width 32 \
  --max-queries-per-task 128
```

## 3. 保守验证协议

1. 在在线 reward 计算后、容器销毁前冻结 workspace，并按原始字节归档。
2. 归档明确排除 `reward/` 和临时 reward payload，防止 verifier 资产进入候选。
3. 当前 atom 仅为“完整文件版本”：内容字节和 mode 必须来自某条真实父轨迹；允许恢复初始版本或文件不存在状态。
4. 不调用 LLM 修补冲突，不生成新字节，不做 JSON/CSV/AST 内部拼接。
5. 每个父 workspace 必须由原始 `reward/check.py` 离线复现在线 reward；不能复现的整组不进入主统计。
6. 使用逐文件替换的 beam search 搜索候选；seed 0 每题最多 64 次 verifier 查询，seed 1 每题最多 128 次。
7. `strict_super_parent` 只表示候选 reward 高于所有父轨迹。
8. 真正的跨轨迹命中还必须满足 `minimum_source_cover >= 2`，从而排除“只删除单条轨迹的有害文件”这种非互补改善。

## 4. 数据质量与完整性审计

| 项目 | 结果 |
|---|---:|
| 计划轨迹 | 320 |
| 成功冻结并归档 | 318 |
| SHA-256 不匹配 | 0 |
| 未冻结归档 | 0 |
| reward 资产泄漏 | 0 |
| criterion 捕获错误 | 0 |
| 含运行时 scaffolding 的命中 | 0 |
| 本次遗留远端容器 | 0 |

两项额外审计处理：

- `task_0023` 有一条父轨迹的绝对 workspace 路径使在线/离线 verifier 调用语义不同；全组无法在单一离线调用模式下同时复现，因此保守剔除。
- `task_1341` 的 prompt 声称提供了两个日志文件，但任务数据实际缺失。模型在不同轨迹中自行构造“输入日志”，verifier 又按被构造后的输入重算答案。其组合虽从 0.7778 升到 0.8889，但属于数据/verifier 漏洞，不计为可信算法证据。

原 gate 中的 `serper_script_not_missing=false` 也被确认是误报：canonical Serper 脚本存在，真正缺失的是模型复制目标目录。检查逻辑已改为要求 canonical 路径和 `No such file` 出现在同一错误行；同一批日志重检结果为无 canonical 脚本缺失。

## 5. 主要结果

### 5.1 原始自动统计

- 39 个任务能够精确离线复现。
- 其中 10 个任务出现至少两个父来源的严格改善，即 10/39 = 25.64%。
- 这里包含上述有缺失输入漏洞的 `task_1341`，所以不能直接作为最终结论。

### 5.2 语义审计后的可信统计

- 可审计任务：38。
- 可信跨轨迹互补命中：9。
- 命中率：**9/38 = 23.68%**。
- 95% Wilson 区间：**13.00%–39.21%**。
- 命中任务绝对 reward gain：
  - 平均：**0.0601**；
  - 中位数：**0.0667**；
  - 范围：**0.0208–0.1111**。
- 在 33 个“所有父轨迹都未满分”的可审计任务中：
  - 9 个得到更高的部分 reward；
  - **0 个达到满分 1.0**。

因此，Phase-0 支持的是“失败产物存在互补、可以提高 verifier reward”，不是“多个失败一定可以合成成功答案”。

## 6. 九个可信案例

| 任务 | 最佳父 reward | 组合 reward | gain | 最小父来源数 | 互补内容概述 |
|---|---:|---:|---:|---:|---|
| `task_0114` | 0.2500 | 0.3333 | 0.0833 | 2 | 一条轨迹的网络抓取/搜索产物 + 另一条轨迹的合法 `studies.csv` |
| `task_1361` | 0.7333 | 0.8000 | 0.0667 | 2 | 不同轨迹的 long announcement 与 short announcement/diagnostics |
| `task_1748` | 0.7778 | 0.8519 | 0.0741 | 2 | 一条轨迹的 `project_plan.json` + 另一条轨迹的大部分复现与状态产物 |
| `task_0184` | 0.8667 | 0.9333 | 0.0667 | 2 | registry/list 文件互补，使 summary match count 正确 |
| `task_0366` | 0.6333 | 0.6542 | 0.0208 | 2 | 一条轨迹的 channel IDs + 另一条轨迹的视频数据、排名和 summary |
| `task_0446` | 0.9231 | 0.9615 | 0.0385 | 2 | 一条轨迹的分析脚本/架构/邮件 + 另一条轨迹的 cost-savings 文件 |
| `task_1348` | 0.8571 | 0.9048 | 0.0476 | 2 | VAT 汇总文件 + 另一条轨迹的 ID validation/notes |
| `task_1555` | 0.5556 | 0.6667 | 0.1111 | 2 | rewritten messages + 另一条轨迹的 content calendar/channel stats |
| `task_1862` | 0.9032 | 0.9355 | 0.0323 | 2 | 90-day plan + 另一条轨迹的 diagnosis/spawn plan/decision log |

较清楚的两个例子：

- `task_0446`：两个最佳父轨迹均为 0.9231。组合后相对父 5 补上 `cost_savings_numbers_correct`，相对父 7 补上 `email_includes_margin_rate`，没有 criterion 回退，最终为 0.9615。
- `task_1555`：父 2 的 rewritten messages 较好，父 4 的 content calendar 较好。组合后相对两者各补上一个 criterion，没有回退，从 0.5556 提升到 0.6667。

## 7. 结论与决策

### Phase-0 决策：GO，进入小型算法原型；尚不进入正式大规模训练

理由：

1. 两个互不重叠 seed 都出现命中，不是单一任务偶然现象。
2. 语义审计后仍有 9/38 的任务存在跨父轨迹严格改善。
3. 所有候选字节均来自真实 on-policy 父轨迹，且原 verifier 可重放，没有 teacher 生成成本。
4. 多个案例表现出 criterion 层面的清晰互补，而不只是整体 reward 数值偶然变高。

但目前只能主张“小规模存在性验证”。下一阶段建议：

1. 只对 verifier 可归因的 artifact/action span 分配局部正信号，不把整个拼接 workspace 当作正确轨迹做 SFT。
2. 把 dataset-integrity 检查加入准入门槛，特别是 prompt 声称存在但实际缺失的输入文件。
3. 使用 held-out verifier 或人工盲审检查组合是否语义真实改善，降低在同一个 verifier 上搜索 64–128 次带来的选择偏差。
4. 扩展到至少 500 个任务组，并按任务类型、父 reward、文件数量和 verifier criterion 数分层报告。
5. 单独测试是否能提高最终 Pass@1；本阶段没有证据证明训练后的 agent 会变好。

## 8. 产物位置

- 扩展轮机器报告：`/root/openclaw-rl-local/outputs/mosaic-phase0-32x8-seed1-20260822-023000/analysis-v2/mosaic_phase0_report.json`
- 先导轮机器报告：`/root/openclaw-rl-local/outputs/mosaic-phase0-8x8-20260822-020000/analysis-v3/mosaic_phase0_report.json`
- 扩展轮 gate 报告：`/root/openclaw-rl-local/outputs/mosaic-phase0-32x8-seed1-20260822-023000/gate/gate_report.json`
- 采样脚本：`/root/openclaw-rl-local/scripts/run_mosaic_phase0_rollouts.py`
- 分析脚本：`/root/openclaw-rl-local/scripts/analyze_mosaic_phase0.py`
- 单元测试：`/root/openclaw-rl-local/clawgym-official/tests/test_mosaic_phase0.py`
