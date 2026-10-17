# ClawGym Base vs SFT：本地 Base Judge Hybrid 评测

日期：2026-08-20

## 结论

在相同确定性 agent 预算下，以本地 Base 模型作为统一 rubric judge：

| 指标 | Base | SFT | SFT - Base |
|---|---:|---:|---:|
| 46 道 hybrid 题的 code 均值 | 37.739% | 52.868% | +15.129 pp |
| 46 道 hybrid 题的 rubric 均值 | 78.261% | 89.130% | +10.870 pp |
| 46 道 hybrid 题的 0.7 code + 0.3 rubric | 49.896% | 63.747% | +13.851 pp |
| 200 题最终 hybrid 均值 | 44.158% | 56.058% | +11.900 pp |
| 200 题分数大于 0.5 的任务数（诊断口径） | 97/200 | 133/200 | +36 |

该结果支持 SFT 相比 Base 有显著提升，但绝对分数不能与论文 GPT-5.4 judge 结果直接比较。

## 评测口径

- 当前官方 release 数据：200 题，其中 154 道纯 code、46 道带 rubric。仓库文档写 44 道，与实际数据不一致。
- 纯 code 题复用此前相同预算的 200 题确定性评测结果。
- 46 道 rubric 题重新运行 agent，并在容器销毁前保存规则指定的最终文件。
- Hybrid 权重：code 0.7、rubric 0.3。
- Judge：本地 `Qwen3-30B-A3B-Instruct-2507` Base 模型。
- Judge prompt、JSON 解析及 rubric 归一化：复用官方 ClawGym-Bench evaluator。
- Judge temperature 0.6，max tokens 8192。
- 每个“被测模型 × task”使用独立固定 seed；SGLang deterministic inference 开启。
- Judge 并发 8，TP2 × DP2；92/92 成功，HTTP/解析失败 0，retry 0。

## Agent 轨迹完整性

- Base：46/46。主批出现 1 次 gateway readiness 环境 skip，随后以原 session index 43 和相同 request seed 补跑成功。
- SFT：46/46，环境 skip 0。
- Base 缺少 rubric 目标文件：9 题、18 个文件。
- SFT 缺少 rubric 目标文件：5 题、8 个文件。

## 关键限制：Base Judge 区分力较弱

92 个 task-level rubric 分数只有 0 或 1：

- 0 分：15 个；
- 1 分：77 个；
- 0.25、0.5、0.75：均为 0 个。

SFT 的 5 个 rubric 0 分恰好对应 5 个缺失目标文件的任务；Base 的结果也基本呈现“目标文件存在即满分”。这表明本地 Base judge 主要判断文件存在性，对文本质量、语气和细粒度 rubric anchor 的区分能力有限。

此外，Base 同时作为被测模型和 judge，存在 self-judge 风格偏置。两组使用同一 judge，因此相对比较仍有意义，但该结果不应称为论文官方 hybrid 分数。

## 结果文件

- 完整逐题 judge 输出：`/root/openclaw-rl-local/outputs/local-base-rubric-judge-20260820/results/judge_results.json`
- 运行日志：`/root/openclaw-rl-local/outputs/local-base-rubric-judge-20260820.launch.log`
- Base rubric 轨迹产物：`/root/openclaw-rl-local/outputs/base-model-eval-clawgym46-rubric-deterministic-20260820/rubric_artifacts/`
- SFT rubric 轨迹产物：`/root/openclaw-rl-local/outputs/sft-model-eval-clawgym46-rubric-deterministic-20260820/rubric_artifacts/`

## 解读建议

本次结果可作为“本地、可复现、低成本”的 hybrid 对照。若需要评估写作质量、语气或细粒度 rubric 差异，仍建议使用能力更强且独立于被测模型的 judge，再直接复用已经保存的 92 份产物，无需重跑 agent。
