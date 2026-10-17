# Evidence Anchor 真实案例：task_1692

## 1. 数据来源

本例来自当前正式 OPSD 训练，而非人工构造：

- rollout：24
- task：`task_1692`
- trajectory index：1539
- verifier reward：0.6875
- 组内 GRPO advantage：-0.234667
- OPSD：已应用，无 fallback
- response tokens：5,814
- loss tokens：3,566

原始记录：

- 任务：[data_entry.json](/root/openclaw-rl-local/clawgym-official/data/clawgym_train/task_1692/data_entry.json)
- Anchor：[task_1692.json](/root/openclaw-rl-local/outputs/opsd-anchor-base-seed0-800-20260831/full800/records/task_1692.json)
- OPSD step：[24.json](/root/openclaw-rl-local/outputs/clawgym-official-30b-opsd-formal-seed0-20260901-v1/opsd_step_summaries/24.json)
- 轨迹：[instance-train-3.log](/root/openclaw-rl-local/outputs/clawgym-official-30b-opsd-formal-seed0-20260901-v1/checkpoints/openclaw_logs/rollout_24/instance-train-3.log)

## 2. 原始任务

用户要求根据本地课程资料生成一个可复现的教学流水线，包括：

- `output/week_summary.csv`：按周统计虚构/非虚构读物、页数和主题；
- `output/syllabus.md`：课程标题、逐字学习目标、逐字课程政策、每周读物和反思问题；
- `output/validation_report.json`：检查阅读类型缺失与120页上限；
- 两封指定字数、内容可核验的邮件；
- 三种不同风格和长度的公告改写。

所有数字必须来自生成的 CSV，且不得使用外部数据。

## 3. 实际生成的 Evidence Anchors

Anchor 生成器读取公开任务和 verifier，经安全审计后生成了8条静态要求：

| # | Condition | Artifact |
|---|---|---|
| 1 | CSV 必须严格包含 `week,fiction_count,nonfiction_count,total_pages,top_motifs` 五列 | `output/week_summary.csv` |
| 2 | syllabus 必须包含输入中的课程标题 | `output/syllabus.md` |
| 3 | syllabus 必须逐字包含全部 learning outcomes | `output/syllabus.md` |
| 4 | syllabus 必须逐字包含 course policies | `output/syllabus.md` |
| 5 | 每周必须列出全部读物的标题、作者、类型、页数和简介 | `output/syllabus.md` |
| 6 | 每周反思问题必须使用该周 `top_motifs` 的前两个主题 | `output/syllabus.md` |
| 7 | validation report 必须正确识别类型缺失和超过120页的周 | `output/validation_report.json` |
| 8 | 学生邮件必须用两个 bullet 准确列出 Week 1、Week 2 的读物标题 | `output/emails/students_announcement.txt` |

生成元数据：一次成功，耗时12.34秒；prompt 5,702 tokens，completion 843 tokens；source mode 为 `verifier_grounded`。

覆盖并不完整：受最多8条限制，这批 Anchor 没有覆盖 chair 邮件和三种公告改写的字数、语气与内容约束。因此 teacher 只能利用部分任务证据，不能据此判断所有失败原因。

## 4. 真实轨迹行为

模型实际执行了以下7个 assistant turn：

1. 并行读取四个输入文件；
2. 写入完整的 `pipeline.py`；
3. 执行流水线；
4. 读取并检查主要输出；
5. 发现 syllabus 标题重复并编辑脚本；
6. 重新运行流水线；
7. 检查其余输出并给出完成总结。

轨迹并非完全错误。例如读取输入、运行脚本和检查输出都是合理动作；但实现仍有问题：反思问题在每个 reading 后生成，而任务要求每周一个；主题取自单条 reading，而不是 `week_summary.csv` 的周级 `top_motifs`。另外，为删除重复标题而修改 course policies 文本，与“逐字包含”要求存在冲突。

## 5. 实际 OPSD 评分

由于0.6875低于同组其他采样，GRPO advantage 为负。每个turn使用：

```text
delta = mean(log P_teacher,anchor - log P_rollout-old)
weight = exp(-delta)          # 因 advantage < 0
weighted_advantage = weight × advantage
```

真实结果如下：

| Turn | 动作 | delta | weight | 加权后 advantage |
|---:|---|---:|---:|---:|
| 1 | 读取输入 | -0.004925 | 1.00494 | -0.23583 |
| 2 | 写 pipeline | -0.032385 | 1.03291 | -0.24239 |
| 3 | 运行 pipeline | -0.013413 | 1.01350 | -0.23784 |
| 4 | 检查输出 | -0.014213 | 1.01431 | -0.23803 |
| 5 | 修改脚本 | -0.001389 | 1.00139 | -0.23499 |
| 6 | 重新运行 | -0.005899 | 1.00592 | -0.23606 |
| 7 | 检查并总结 | -0.066882 | 1.06917 | -0.25090 |

## 6. 结论

这条真实轨迹的7个势差全部为负，因此所有turn的负向惩罚都被放大。错误实现所在的turn 2和最终总结turn 7确实获得了较强惩罚，但明显合理的读取、执行和检查动作也没有得到正向支持。

这个案例说明当前 Anchor 能提供任务约束，但现有 raw logprob difference 还没有可靠实现“保留合理步骤、重点惩罚错误步骤”的细粒度归因；它可能同时包含加入 privileged prompt 后产生的整体概率校准偏移。
