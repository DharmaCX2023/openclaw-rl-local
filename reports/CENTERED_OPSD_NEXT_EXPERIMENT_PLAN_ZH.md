# 轨迹内中心化 OPSD：下一阶段实验方案

## 1. 背景与动机

当前 Qwen3-8B 实验中，标准 GRPO 最终评测为 `reward=0.5355 / correct=55.5%`，现有 OPSD 为 `reward=0.5331 / correct=54.5%`，尚无性能增益证据。

现有 OPSD 对失败轨迹的每个 assistant step 计算：

$$
\Delta_\tau=\operatorname{mean}_{t\in\tau}
\left[\log \pi_{\mathrm{teacher+anchor}}(a_t)-\log \pi_{\mathrm{rollout-old}}(a_t)\right],
$$

并使用

$$
w_\tau=\operatorname{clip}\!\left(e^{-\beta\Delta_\tau},0,1.2\right),\qquad
\widetilde A_\tau=w_\tau A.
$$

正式实验采用 $\beta=1.0$。在 49,979 个已评分 step 中，98.745% 的权重大于 1，token 加权平均权重为 1.0285。因此，该信号主要表现为对失败轨迹整体加重惩罚，而非轨迹内部的有效 credit assignment。

## 2. 方法：轨迹内中心化 OPSD

对每条 logical rollout，使用其中所有有效 step 的中位数和 MAD 标准化 teacher gap：

$$
z_\tau=
\frac{\Delta_\tau-\operatorname{median}(\Delta)}
{\operatorname{MAD}(\Delta)+\varepsilon}.
$$

仅对负 advantage 轨迹调整惩罚幅度：

$$
w_\tau=\operatorname{clip}\!\left(e^{-\beta z_\tau},0.8,1.2\right),
\qquad \widetilde A_\tau=w_\tau A.
$$

随后按各 step 的有效 loss token 数 $N_\tau$ 归一化：

$$
w'_\tau=
\frac{w_\tau}
{\sum_\tau N_\tau w_\tau/\sum_\tau N_\tau},
$$

使整条 logical rollout 的 token 加权平均权重严格为 1。该方法只重新分配原有负 advantage，不改变符号，也不增加失败轨迹的总惩罚预算。正 advantage 轨迹继续使用标准 GRPO。

建议首先测试 $\beta\in\{0.02,0.05,0.10,0.20\}$，不沿用现有的 $\beta=1.0$。

## 3. 与现有 OPSD 的差异

| 项目 | 现有 OPSD | 中心化 OPSD |
|---|---|---|
| 信号 | 绝对 teacher gap | logical rollout 内相对 gap |
| 公共 prompt 偏移 | 保留 | 通过减中位数消除加性偏移 |
| 权重分布 | 几乎全部大于 1 | 同时允许大于和小于 1 |
| 总惩罚强度 | 实际增加约 2.85% | token 加权平均严格保持 1 |
| 作用 | 整体放大失败轨迹 | 在失败轨迹内部重新分配惩罚 |
| segment 处理 | 分 segment 评分 | 分 segment 评分、logical rollout 级归一化 |

## 4. Phase 0：离线止损实验

首先利用已有 49,979 个 step summary 重算候选权重，不直接启动完整训练。

需要检查：

1. **数值退化**：MAD 接近零、权重触及 0.8/1.2、单 step 轨迹及极短轨迹的比例；
2. **稳定性**：重复 teacher scoring 后，step 排序和权重是否稳定；
3. **行为相关性**：高惩罚 step 是否更常对应无效 tool call、重复调用、明确报错、无效 edit 或截断；低惩罚 step 是否更常对应有效修改、错误恢复和提交前验证；
4. **优化影响**：在少量冻结 batch 上比较与标准 GRPO 梯度的 cosine、主要梯度贡献 step 及长轨迹占比；
5. **分组鲁棒性**：分别统计短/长轨迹、单/多 segment、不同 reward 区间的结果。

现有 summary 足以检查权重分布；动作类型需要关联完整轨迹日志；精确梯度比较需要冻结 batch 的 GPU forward/backward。

## 5. 继续与停止条件

满足以下条件后，才进入训练实验：

- MAD 非退化，权重不被数值噪声主导；
- 权重饱和比例处于可控范围；
- 重复评分的 step 排序具有较高稳定性；
- 相比随机重排权重，与独立行为证据存在显著相关性；
- 冻结 batch 中未出现少数长 step 主导更新或梯度方向严重偏转。

若权重变化只是公式强制产生，且与动作证据、重复评分及梯度诊断无关，则停止该方向。

## 6. 后续训练计划

Phase 0 通过后，先运行同起点、同任务顺序、同采样 seed 的 20–40 step 配对实验：

1. 标准 GRPO；
2. 现有绝对-gap OPSD；
3. 轨迹内中心化 OPSD。

重点比较固定-seed eval AUC、最终 verifier reward/correct rate、轨迹长度、工具错误率、权重分布和训练稳定性。只有中心化 OPSD 在短实验中稳定优于 GRPO，才进入 100-step、多 seed 正式实验。

## 7. 结论边界

中心化解决的是现有 OPSD 的“几乎全局同向加权”问题，但相对 teacher gap 仍不等于真实因果贡献。Phase 0 只能验证信号是否稳定、非退化并具有基本行为相关性；最终有效性必须由受控训练和固定协议评测确认。
