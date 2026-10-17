# Centered OPSD / SDAR 正式训练门控报告

## 结论

本次 `8 tasks × 8 samples` 双臂端到端 smoke 均完整成功，基础设施、分段、teacher scoring 和训练更新链路通过。但两种算法都没有通过正式 100-step 训练前的信用信号门控，因此当前不得直接启动正式训练。

- Centered OPSD：数值性质正确，但独立行为证据显示权重方向与明显退化动作相反。
- SDAR：原始 Anchor teacher gap 几乎全为负，sigmoid gate 接近常数 0.5，并显著偏向重复和工具错误；当前辅助项实质是被错误 gate 加权的行为克隆。

这符合既定计划中的停止条件：信号若仅由公式产生、且与独立行为证据无正向关系，则停止该方向，不能用 100-step 预算强行验证。

## 运行信息

- 代码提交：`d4e55754`
- Centered：`outputs/qwen3-8b-centered-formal-seed0-smoke-20260906-0618`
- SDAR：`outputs/qwen3-8b-sdar-formal-seed0-smoke-20260906-0618`
- 两组均从相同 8B SFT checkpoint 启动。
- 每组 2 卡 Megatron + 2 卡 SGLang；每组 64 条逻辑轨迹。
- sampling seed、任务、温度、top-p、top-k、上下文预算一致。

## 基础设施与可复现性

两组合计 128 条逻辑轨迹：

- 最终丢失：0；
- rollout timeout：0；
- Gateway 启动失败/重试：0/0；
- session engine switch：0；
- Anchor fallback/overflow：0/0；
- 重复 teacher scoring：token logprob MAE = 0，step Pearson = 1，Spearman = 1。

因此本次异常来自信用信号本身，而不是端口、容器、推理随机性或评分不稳定。

## Centered OPSD

共有 30 条负 advantage 逻辑轨迹、399 个已评分 assistant step。数值检查：

- step 权重分位数：min 0.8169，P50 0.9977，P90 1.0978，max 1.2；
- 饱和比例：0.752%；
- token 加权均值：0.99999995；
- 单轨迹最大 mass 误差：1.78e-7；
- 1 条轨迹被 pruning 切成20个 segment，仍保持 logical-rollout 总质量不变。

但对负 advantage 而言，权重越小代表惩罚越轻。精确 session/step 关联结果为：

- identical tool repeat：平均权重0.9763，相对其他 step 低0.0443，置换检验 p=0.001；
- tool-result error：平均权重0.9874，相对低0.0339，p=0.002；
- workspace mutation：平均权重0.9878，相对低0.0281，p=0.015；
- final/text：平均权重1.0593，相对高0.0512，p=0.001。

即算法显著减轻对重复调用和工具错误的惩罚，同时加重 final/text step。该方向不满足 Phase-0 行为门控。

完整诊断：`reports/centered_opsd_smoke_behavior_beta002_20260906.json`。

## SDAR

64/64 条逻辑轨迹应用 Anchor，共393,878个 loss token、963个 assistant step：

- token teacher gap 均值：-0.02580；
- token gap > 0：2.23%；
- step gap > 0：1.77%；
- gate 均值：0.4732；step gate P10/P50/P90 = 0.4279/0.4723/0.4959；
- gate active fraction：约2.23%（token）/1.77%（step）。

行为关联同样反向：

- identical tool repeat 的 gate 相对其他 step 高0.0250，p=0.001；
- repetition degeneration 高0.0263，p=0.027；
- tool-result error 高0.0152，p=0.001；
- final/text 低0.0173，p=0.001；
- validation exec 无显著优势，p=0.731。

`lambda=0.01` 时，按 logical-rollout/token 口径计算的辅助梯度 L1 系数约为 GRPO 的0.617%。日志中的0.224%是动态 micro-batch 指标归并偏差，不应作为最终比例。即使提高 lambda，错误 gate 也会被同步放大。

此外，当前 token loss `gate × (teacher_logprob - student_logprob)` 对 student 的梯度与 `-gate × student_logprob` 完全相同；teacher logprob 只参与 gate，不构成全词表 KL。因此它是 gated behavior cloning，而不是完整分布蒸馏。

## 后续建议

1. 暂停两组100-step正式训练，保持八卡烤机。
2. 不通过简单增大 beta/lambda 修复；这只会放大错误排序。
3. 若继续 Anchor 信用分配，先建立同长度的对照条件（真实 Anchor 对 placebo/shuffled Anchor），消除向 user prompt 追加文本造成的共同分布偏移。
4. 在新信号上重新执行同一套行为置换检验；至少要求重复退化和工具错误不再获得显著保护，validation/progress step 获得更高支持。
5. 只有新信号通过门控后，再做20–40 step与标准 GRPO 的固定-seed配对实验；短程胜出后才进入100步。

## Placebo 对照可行性

对现有800个任务、5,904条 Anchor 使用正式8B tokenizer 统计：

- 每个任务完整 SDAR feedback 长度：453–1,214 tokens，中位743；
- 为每个任务选择另一任务的最近长度 Anchor 后，85.13% 可做到 token 数完全相同；
- 99.625%的配对长度差不超过8 tokens；
- 极少数长度差超过8的任务可以直接 fallback 到标准 GRPO。

因此 matched-placebo 不需要生成新 Anchor，也不需要额外 judge。它只把 teacher scoring 从一个条件扩展为两个长度近似相同的条件。正式训练关闭重复评分后，其额外成本预计与本次重复评分 smoke 同阶，远小于约1,300秒的 rollout 长尾。
