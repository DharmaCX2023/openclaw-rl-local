# GRPO-in-GRPO

普通 GRPO 只比较整条轨迹的最终 outcome。GRPO-in-GRPO 在同一 group 内再比较每个 artifact 的 verifier 分数，并把局部优势只分给该 artifact 对应的写入链 span。整条轨迹的 outcome 信号仍然保留；没有可确定 span 的 artifact 退回整条轨迹。

评测对比的是同步数检查点：OR40 为 iter 39，GRPO-in-GRPO 为 iter 39（temperature=0）或 iter 40（temperature=0.7），都从同一 30B ClawGym-SFT 出发，约 40 个 rollout step。

## 1. 写入链如何追责 span

对每个最终 artifact，沿已经发生的文件写入向前回溯：

- 直接 `write` / `edit` 目标文件，记为一次写入；
- 脚本执行、复制、转换导致文件落盘，也记为一次写入；
- 若执行的脚本此前由另一个文件写出，则继承该文件的最早写入起点；
- 子进程继承父进程已确认的文件依赖。

span 起点是相关写入链中最早的那一次写入，终点是目标文件最后一次实际修改。这个区间表示：从相关文件开始出现，到该 artifact 被改完，模型应对这段负责。artifact 的组内优势只加到这段上的 token。

写入链只能看见文件系统上的边。若模型先读文件或 observation，在自然语言里计算、整理后再写入另一个文件，中间没有可追踪的写入依赖，span 起点会偏晚，前面的推理 turn 拿不到该 artifact 的局部 credit。系统层也只能证明“访问过某文件”，不能证明“内容被用到了”。因此当前 span 是保守的责任范围，不是完整因果图。

## 2. 两次同步数评测

固定 200 题、每题 1 条、code-only verifier、seed 1101–1105。temperature=0 的三次评测都是 200/200；temperature=0.7 只评了 OR40 与 GRPO-in-GRPO。

### temperature=0（贪心，10/2–10/4）

| 系统 | 平均 reward | 样本标准差 | Correct |
|---|---:|---:|---:|
| SFT | 0.5066 | 0.0049 | 52.3% |
| OR40 | 0.5185 | 0.0054 | 52.9% |
| GRPO-in-GRPO | 0.5234 | 0.0029 | 57.8% |

相对 SFT：OR40 **+1.19 pp**，GRPO-in-GRPO **+1.67 pp**。  
相对 OR40：GRPO-in-GRPO **+0.48 pp**（bootstrap 95% CI 含 0）。Correct 仍高约 5 pp。

### temperature=0.7（接近训练采样，10/7–10/8）

| 系统 | 1101 | 1102 | 1103 | 1104 | 1105 | 平均 reward | 样本标准差 | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OR40 | 0.540 | 0.512 | 0.537 | 0.516 | 0.535 | 0.528 | 0.0131 | 56.3% |
| GRPO-in-GRPO | 0.552 | 0.556 | 0.559 | 0.554 | 0.535 | 0.551 | 0.0092 | 59.0% |

相对 OR40：GRPO-in-GRPO **+2.36 pp**（bootstrap 95% CI `[0.95, 3.77]`，不含 0）。5 个 seed 全部非负。

### 同步数增益怎么读

约 40 step 时：两条 RL 在 temperature=0 上都超过 SFT；GRPO-in-GRPO 相对普通 OR 的额外增益，贪心评测约 **0.5 pp**（不确定），采样评测约 **2.4 pp**（确定）。更长训练（到 100 step）尚未评测。
