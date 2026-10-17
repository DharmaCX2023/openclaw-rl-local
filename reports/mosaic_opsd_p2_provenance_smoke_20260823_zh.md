# Mosaic OPSD P2：Workspace Provenance Smoke

日期：2026-08-23

## 决策

结论分成两层：

- **继续采集：GO。** Tool-result 到下一次 policy generation 之间存在无竞态 workspace snapshot 边界；远端实测无错误，且明显提高 verifier-positive artifact 到 exact action 的覆盖率。
- **立即修改 RL loss：NO-GO。** 当前只有 2 个任务、4 条轨迹，不能据此证明训练收益或安全性。仍需达到预设的 100 exact actions、30 tasks 后再做小规模 A/B。

## 方法

仅在 Mosaic audit wrapper 中启用，不改变正式训练路径：

1. 远端 workspace 创建后记录 baseline manifest；
2. OpenClaw 执行工具并把 tool result 送回 Proxy；
3. Proxy 在下一次 SGLang generation **之前**阻塞一次只读 manifest snapshot；
4. 以 session、source step、tool_call_id、文件 path/size/mode/SHA-256 记录 mutation delta；
5. 最终用 verifier-positive artifact 的终态 SHA 精确连接 provenance event 和原始 response token span。

受保护的 `input/`、`reward/`、运行时 prompt 文件和缓存文件不进入 manifest。Ledger 只保存 tool-result 长度与 SHA-256，不保存完整结果正文。

## 后台进程处理

现有 64 条轨迹中，12 条（18.75%）出现真实的 `Command still running` 或 `Process still running`。

后台命令返回后仍可能继续写文件，因此：

- 启动后台进程的 event 不可用于 provenance credit；
- session 后续 event 继承 background taint；
- 当前实现不猜测后台进程何时结束，无法证明时宁可丢弃；
- 本次 smoke 未触发后台运行，因而没有 taint rejection。

## 2×2 端到端结果

运行目录：`outputs/mosaic-p2-provenance-smoke-20260823-025918`

| 指标 | 结果 |
|---|---:|
| 完成轨迹 | 4/4 |
| Baseline snapshots | 4 |
| Tool-result boundary events | 44 |
| 有 workspace mutation 的 events | 26 |
| Provenance-eligible events | 26 |
| Snapshot errors | 0 |
| 无 mutation 的只读 events | 18 |
| Snapshot 平均延迟 | 0.2381 s |
| Snapshot 中位延迟 | 0.2419 s |
| Snapshot P95 | 0.2467 s |
| 44 次 snapshot 累计时间 | 10.48 s |

远端暖 ControlMaster 的独立往返约 0.217 秒；真实大型终态 workspace 的本地全量 SHA-256 约 0.005 秒。因此当前成本主要是 SSH RTT，不是文件哈希。

## 终态因果连接

本批 1 个 cross-parent 任务产生 4 个 verifier-positive artifacts：

| Artifact | Action 类型 | 说明 |
|---|---|---|
| `output/events_normalized.json` | `exec` | 与成功 normalize 命令精确连接 |
| `output/normalize.log` | 同一 `exec` | 与上项聚合，不重复累加 credit |
| `output/diagnostics.md` | `write` | 单 action 精确连接 |
| `output/announcements/long.md` | multi-tool turn 中的 `write` | 通过 path 与 content SHA 唯一定位该 call |

Join gate 结果：

| 指标 | 旧 direct-write gate | Provenance gate |
|---|---:|---:|
| Verifier-positive artifacts | 4 | 4 |
| 有 exact action 的 artifacts | 1 | 4 |
| 聚合后的 exact action candidates | 1 | 4 |

4 个候选 action 的原始 tool-call span 分别为 48、461、49、342 tokens；工具分布为 2 个 `exec`、2 个 `write`。同一 action 对应多个正贡献 artifact 时，权重取最大 artifact credit，绝不求和，避免重复计奖。

## 当前安全闸门

候选 action 必须同时满足：

- verifier leave-one-file-out 为正，至少一个 criterion 改善且无回退；
- provenance event 的 tool-result IDs 与 tool-call IDs 顺序完全一致；
- mutation 后的 path 与 SHA-256 等于终态选中 artifact；
- 此后同一路径没有再次 mutation；
- 无 background taint、snapshot gap 或大文件 hash 省略；
- 能在保存的原始 response IDs 上得到 exact tool-call token span；
- multi-tool turn 中只能唯一定位到对应 action，否则拒绝。

仍然不奖励 action 前的分析文字，因为当前反事实只证明 artifact/action 的因果贡献，没有证明自然语言 reasoning token 的正确性。

## 下一步

建议扩大 provenance-enabled Mosaic 采集到更多不同任务，先报告：

- exact actions / trajectory 与 distinct task 覆盖；
- `write/edit/exec` 分布；
- background taint、snapshot error、later-mutation rejection；
- 额外墙钟时间和存储成本；
- verifier-positive artifact 覆盖相对 direct gate 的提升。

只有达到至少 100 exact actions、覆盖至少 30 tasks，且人工抽检 precision ≥95%，才进入 `GRPO + artifact-local auxiliary loss` 的小规模 A/B。若扩大后 action 密度显著下降、后台 taint 过高或 wall time 增幅不可接受，应主动终止该方向。
