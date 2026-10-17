# OPSD 实验简报

**目标：**用Evidence Anchor改善失败轨迹信用分配。

**方法：**保留GRPO，仅对负advantage turn用“Anchor＋冻结rollout-old”重算logprob并加权；异常回退GRPO。

**配置：**Qwen3-30B-A3B，4卡推理＋4卡训练；每步8题×8轨迹，100步，每20步评测200题并存checkpoint；启用routing replay和TIS。

**结果：**已更新20次。2,238个segment中986个应用OPSD；10,466个turn权重均值1.030，8个回退。raw reward均值0.547，未见提升；KL均值9.92e-4，grad norm 0.176–0.910，数值稳定。

**风险：**训练102/1,329个instance出现Serper 401；eval 3/196，均失败。OPSD逐步advantage因设计均为负；teacher势差98.59%为负、1.41%为正，主要放大失败惩罚，逐步归因需验证。

**结论：**工程链路稳定，收益待100步固定采样评测确认。
