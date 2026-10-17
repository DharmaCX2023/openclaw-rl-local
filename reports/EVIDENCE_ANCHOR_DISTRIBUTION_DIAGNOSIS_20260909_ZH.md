# Evidence Anchor 分布差异诊断

## 结论

当前分布与 SSPO 不同，主要不是 Anchor 数量或长度，而是监督语义不同。SSPO 的 Anchor 是支持正确答案的外部事实，任务 step 也较短；ClawGym 的 Anchor 多为验收清单，而一个逻辑轨迹包含读取、写入、执行、报错、恢复和验证。把静态清单或事实放进上下文后，再用 Teacher 对 Student 原动作的 likelihood gap 充当信用，会出现 action-span projection mismatch：Teacher 是否更容易复现动作文本，不等价于该动作是否有价值。

此外，`Δ = log P_teacher(a|h,H) - log P_old(a|h)` 对 `a ~ P_old` 的期望本来就是 `-KL(P_old || P_teacher) <= 0`。因此“绝大多数 Δ 为负”本身不是 Anchor 错误证据；应检查相对行为排序，而不是追求正负各半。

## 实测

固定同一批真实动作 token，只改变 teacher-only evidence：

| 方案 | step | 错误→下一步配对 | 恢复动作 Δ 更高 | good−bad Δ（95% bootstrap CI） |
|---|---:|---:|---:|---:|
| 旧静态 Anchor | 399 | 26 | 61.5% | 0.00043（−0.00145, 0.00353） |
| 动作化清单 Anchor | 388 | 26 | 50.0% | 0.00015（−0.00174, 0.00527） |
| 公开输入事实 Anchor | 351 | 26 | 61.5% | −0.00505（−0.00673, −0.00079） |
| 动态“该动作报错/恢复”提示 + likelihood gap | 61 | 26 | 46.2% | — |

三轮改写均未通过行为区分门槛。公开事实 Anchor 甚至使重复退化动作的相对分数高于成功读取，说明继续调静态文案不能解决投影错位。

## 显式 judge 复核

同一 8B 模型直接读取候选动作和真实 tool result，输出 `progress/error/neutral`：

- 明确 tool error 判为 error：81.4%；
- “紧随错误的下一步”判为 progress：56.9%；
- 71 个相邻配对中，两者同时满足：43.7%。

这也揭示原诊断标签 `recovery_after_error` 过宽：它只表示“上一 step 失败、当前 step 未被正则识别为失败”，其中仍包含空读、无效检查、后台未完成和语义失败，不能当作真实恢复真值。

## 处理决定

1. 保留已完成的 step 长度归一化修复，避免长动作因 token 数被重复放大。
2. 不把任何新 Anchor 直接接入正式训练；三版都未通过真实行为门槛。
3. 安全的 8 题 Anchor 仅作为数据构造试点，不宣称算法有效。
4. 下一步应将信用信号改为显式、可审计的状态转移判断，并以确定性工具错误规则兜底；不要再用 hinted action likelihood 的绝对正负解释动作质量。

## 产物

- 旧/新配对评分：`outputs/action-grounded-anchors-pilot8-20260909/phase0_scores.report.json`
- 公开事实评分：`outputs/public-fact-anchors-v4-pilot8-20260909/phase0_scores.report.json`
- 动态提示评分：`outputs/outcome-grounded-step-evidence-pilot-20260909.json`
- 显式 judge：`outputs/llm-step-evidence-pilot3-20260909.json`
- 人工校正、零审计错误的 Anchor 试点：`outputs/public-fact-anchors-v4-pilot8-20260909/anchors.audited_pilot8.jsonl`
