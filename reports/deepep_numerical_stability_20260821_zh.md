# DeepEP 数值稳定性严格对照（2026-08-21）

## 结论

DeepEP 不是当前数值不稳定的主要来源。相同权重、相同输入下，DeepEP 与原始 All-to-All 的重复前向误差基本相同；开启 routing replay 后，两种 dispatcher 的前向误差均下降约一半。

剩余问题主要来自 MoE 计算链本身：一部分是临界 router 决策变化，另一部分是在固定专家后仍存在的 BF16 expert GEMM、并行归约和浮点累加顺序差异。微小的 token-level logprob 扰动在长序列、正负 advantage 梯度相互抵消时会被完整 backward 明显放大。

DeepEP 在单次完整梯度对照中略差，但 All-to-All 本身也已经明显不稳定，因此当前证据不支持“安装 DeepEP 导致不稳定”。正式训练暂不建议仅为约 0.7% 的吞吐收益启用 DeepEP。

## 严格实验口径

- 模型：step 99 RL checkpoint。
- 数据：同一份已保存 rollout；6 个训练 sample，239,629 个总输入 token，84,432 个有效 response token。
- Packing：3 个确定性 packed micro-batch，最大 81,000 tokens；`max_tokens_per_gpu=81,920`。
- 前向：每组相同数据连续执行 5 次，共统计 10 个两两 repeat pair。
- 反向：相同 checkpoint、token、loss mask、advantage 和 micro-batch 顺序执行两次完整 backward。
- 零更新：不执行 optimizer step，不执行 scheduler step；两次 backward 前均重置 DataIterator 并清空全部梯度 buffer。
- Dropout：attention/hidden dropout 均为 0。
- 对照变量仅为 dispatcher（All-to-All / DeepEP）和 routing replay（关闭 / 开启）。

## 重复前向结果

| Dispatcher | Routing replay | logprob MAE | RMS | `P99 abs diff` | `abs(ratio-1)>1%` | 方向性误 clip |
|---|---:|---:|---:|---:|---:|---:|
| All-to-All | 关 | 0.005963 | 0.033189 | 0.14649 | 7.658% | 0.1790% |
| All-to-All | 开 | 0.002953 | 0.016668 | 0.08415 | 5.271% | 0.0128% |
| DeepEP | 关 | 0.005874 | 0.032176 | 0.14474 | 7.601% | 0.1774% |
| DeepEP | 开 | 0.002983 | 0.016592 | 0.08559 | 5.218% | 0.0102% |

观察：

1. 不开 replay 时，DeepEP 与 All-to-All 的 MAE、RMS、P99 和 ratio 长尾几乎相同。
2. routing replay 将 MAE/RMS 降低约 50%，把方向性误 clip 降低约 93%–94%。router 临界翻转是重要来源，但不是全部来源。
3. 固定路由后仍有约 5.2% token 的概率比偏移超过 1%，说明 expert 计算和并行归约仍有不可忽略的非确定性。

## 重复完整 backward 结果

两组均开启 routing replay，避免把明显的专家选择变化混入 dispatcher 对比。

| Dispatcher | 第一次梯度范数 | 第二次梯度范数 | 梯度 cosine | 范数比 | Relative L2 |
|---|---:|---:|---:|---:|---:|
| All-to-All | 0.25330 | 0.30280 | 0.66878 | 1.1954 | 0.9111 |
| DeepEP | 0.24806 | 0.30993 | 0.62280 | 1.2495 | 1.0024 |

两次 backward 的数据、模型参数和代理 loss 完全相同。All-to-All 的梯度方向已只有 0.669 cosine，DeepEP 单次结果为 0.623。DeepEP 在该 pair 上更差，但差异远小于两种 backend 共同存在的重复 backward 不一致，且目前每个 backend 只有一个完整梯度 pair，不能据此断言 DeepEP 会系统性恶化梯度。

## 原因归因

1. **MoE router 临界决策变化**：约解释一半的重复前向 MAE/RMS；routing replay 明显有效。
2. **固定路由后的计算非确定性**：BF16 grouped expert GEMM、token permutation、通信归约及浮点累加顺序仍会改变 logits；DeepEP 与 All-to-All 前向结果接近，说明这不是 DeepEP 独有问题。
3. **梯度放大效应**：GRPO 的最终梯度是大量正、负 advantage token 梯度的向量和。净梯度较小时，前向/Jacobian 的小扰动会改变抵消结果，因此 token-level 平均误差不大，完整梯度 cosine 仍可能很低。
4. **DeepEP 影响**：目前没有证据显示它是根因；单次 backward 略差，需多次重复 pair 才能估计其统计影响。

## 建议

1. 下一轮正式训练的稳定性改造应优先启用并验证 routing replay，同时监控 `logprob_abs_diff`、ratio 长尾、方向性误 clip 和梯度 cosine；本次临时启用入口已随诊断代码移除，尚未写入生产启动配置。
2. DeepEP 的实测端到端 step 加速仅约 0.72%，在获得至少 3–5 个完整 backward pair 的稳定统计前，正式训练维持 All-to-All 更稳妥。
3. 下一阶段若继续定位残余误差，应分别消融 grouped GEMM、通信归约和 activation recomputation；这比继续比较 dispatcher 更接近剩余根因。
4. 本次所有临时零更新/双 backward 诊断代码和 `/tmp` 开关均已移除，不会改变后续正式训练语义。

## 结果文件

- 前向：`forward_alltoall_no_replay.pt`、`forward_alltoall_replay.pt`、`forward_deepep_no_replay.pt`、`forward_deepep_replay.pt`
- 反向：`backward_alltoall_replay.json`、`backward_deepep_replay.json`
- 启动日志：`run_*.launch.log`

结果目录：`/root/openclaw-rl-local/outputs/deepep-numerical-stability-20260821`
