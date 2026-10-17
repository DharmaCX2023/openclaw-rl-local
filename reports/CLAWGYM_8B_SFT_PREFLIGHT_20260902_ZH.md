# ClawGym 8B SFT 预检与 Smoke 结果

## 结论

8B 全参数 SFT 已具备正式启动条件。64K RingAttention 两步 smoke 成功完成，无 OOM、NaN 或通信错误；正式训练尚未启动，等待配置确认。

## 数据与契约

- 原始数据：22,631 条 OpenClaw 2026.4.1 多轮轨迹。
- system prompt SHA256：`c6bd8aecbd320671714a7e290da0c39546b07af0f8929659dad51808f7b47472`。
- 工具：`read/edit/write/exec/process/web_fetch`。
- 修复：OpenRLHF 渲染时显式传入每行 `tools`；否则六工具 schema 会被遗漏。
- 8B tokenizer 重渲染后，3 条轨迹超过 65,536 tokens，已严格过滤而非截断。
- 最终数据：22,628 条、237,363 个 assistant span；P50/P90/P95/P99/Max 长度为 15,128/26,286/33,034/47,481/65,430 tokens。
- assistant loss mask 只覆盖 assistant 内容（包括 tool call），不训练 system/user/tool result。

## 模型与框架

- 基模：`/root/humf/models/Qwen3-8B`。
- 64K staging：`/root/openclaw-rl-local/models/Qwen3-8B-yarn2-64k`。
- YaRN：factor 2.0，原生长度 32,768，训练上限 65,536。
- ClawGym commit：`0e8526d8f5e0999896c74cee4d6b998fb8a0429e`。
- OpenRLHF commit：`3c3be6234e0cb353e76bb8019947db9dfe99fca7`，加本地 tools/preprocessed-data 补丁。
- DeepSpeed 0.19.5，Transformers 5.15.0，Ring FlashAttention 0.1.8，BF16。

## Smoke

- 数据：16 条长度分层轨迹（8,826–65,430 tokens）。
- 拓扑：8 GPU，RingAttention 4，data parallel 2，ZeRO-2。
- batch：global 8，micro 1，4 次梯度累积，共 2 次 optimizer update。
- 训练计算：约 27 秒；峰值显存约 61 GB/GPU。
- update 1：loss_mean 0.2609，grad_norm 9.70。
- update 2：loss_mean 0.8953，grad_norm 18.38。
- `max_norm=1.0` 执行梯度裁剪；短 smoke 的 loss 跨 batch 不用于判断收敛。
- 最终权重成功保存，抽查 4,096 个参数有 370 个发生 BF16 可见变化。

## 拟正式训练配置

- 8 GPU；RingAttention 4 × data parallel 2。
- global batch 128；micro batch 1；梯度累积 64。
- 3 epochs；约 176 optimizer steps/epoch、528 steps 总计。
- AdamW：lr `1e-5`，betas `(0.9, 0.95)`，eps `1e-8`，weight decay `0`。
- cosine scheduler，warmup ratio 0.03，min lr ratio 0.1。
- BF16、FlashAttention 2、packing、gradient checkpointing、ZeRO-2。
- 每 80 steps 保存 DeepSpeed 续训 checkpoint 和 HF 权重；最多保留 3 份 DeepSpeed checkpoint；结束另存 final HF model。
- 预计训练墙钟约 28–34 小时，checkpoint IO 另计。

启动脚本：`/root/openclaw-rl-local/scripts/run_clawgym_8b_sft.sh formal`
