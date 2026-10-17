# ClawGym RL Checkpoint 统一评测指南

## 1. 目标与标准口径

本指南用于比较 SFT baseline 与不同 RL checkpoint（如 step19、39、59、79、99）。所有模型必须使用完全相同的任务、采样参数、OpenClaw 环境和执行预算。

标准配置：

| 项目 | 配置 |
|---|---|
| 数据集 | `clawgym-official/data/clawgym_eval`，200 题 |
| 每题采样 | `n=1` |
| temperature | `0.7` |
| top-p | `1.0`（SGLang/model generation config 仍可能提供默认 top-k/top-p，必须核对启动日志） |
| 单轮 response 上限 | 8,192 tokens |
| live context / trajectory 上限 | 65,536 tokens |
| agent step 上限 | 96 |
| compaction 上限 | 8 |
| worker 数 | 32 |
| reward | ClawGym code verifier reward，范围通常为 `[0,1]` |
| 主指标 | 200 题 `avg_reward` |

`correct_rate` 是本地辅助指标，定义为 `reward > 0.5` 的任务比例，不等同于论文的平均 reward。比较 checkpoint 时优先使用 `avg_reward`。

## 2. 运行前检查

评测独占本机 8 张 GPU（Megatron TP4 + 两个 SGLang TP2 engine），同一时间只能运行一个本地训练或评测 Ray 集群。

```bash
nvidia-smi
tmux ls
/root/openclaw-rl-local/.venv-openclaw/bin/python \
  /root/openclaw-rl-local/.venv-openclaw/bin/ray status \
  --address=127.0.0.1:6379
ss -ltnp | rg '30001|39081|8265|6379'
```

必须确认：

- GPU 上没有训练、评测或烤机进程；
- 本机没有现存 Ray 集群；
- 本次使用的本地 Proxy 端口、SSH reverse port、远端 gateway 端口段没有被其他任务占用；
- 远端共享 Docker 节点上没有使用同一端口段的任务；
- `/root/openclaw-rl-local/.env` 存在，但不要打印其内容；launcher 只把它作为容器环境来源；
- checkpoint 目录包含 `.metadata`、`common.pt` 和所有 `*.distcp` shard。

不要为了评测停止或删除不属于本次任务的远端容器。

## 3. 选择 checkpoint（不复制权重）

Megatron 的 `--load` 接收训练根目录，并根据 `latest_checkpointed_iteration.txt` 选择 iteration。为了评测任意 checkpoint，又不修改正式训练目录，建立一个只包含软链接的只读视图。

以下以 step99 为例：

```bash
REPO=/root/openclaw-rl-local
TRAIN_RUN=$REPO/outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700
ITER=99
ITER_PADDED=$(printf '%07d' "$ITER")
STAMP=$(date -u +%Y%m%d-%H%M%S)
EXP=clawgym-rl-step${ITER}-eval-same-budget-$STAMP
OUT=$REPO/outputs/$EXP
LOAD_VIEW=$OUT/checkpoint_load

test -d "$TRAIN_RUN/iter_$ITER_PADDED"
mkdir -p "$LOAD_VIEW"
ln -s "$TRAIN_RUN/iter_$ITER_PADDED" "$LOAD_VIEW/iter_$ITER_PADDED"
printf '%s\n' "$ITER" > "$LOAD_VIEW/latest_checkpointed_iteration.txt"

readlink -f "$LOAD_VIEW/iter_$ITER_PADDED"
cat "$LOAD_VIEW/latest_checkpointed_iteration.txt"
```

step59/79/99 可能是指向 `/mnt/humf/openclaw-rl-checkpoints/...` 的链接，这是正常的；只需确认最终目标存在，不要复制数百 GB checkpoint，也不要修改正式训练根目录中的 iteration 指针。

## 4. 启动 RL checkpoint 评测

当前已验证的 eval-only 入口：

```text
launcher: clawgym-official/.run_step39_eval_sync_20260812-091700.sh
driver:   slime/train_step39_eval_only_20260812_091700.py
```

文件名保留了首次 step39 复评的历史名称，但实现本身不绑定 step39；实际 checkpoint 由 `LOAD_CKPT` 中的 iteration 指针决定。driver 只执行：加载 Megatron checkpoint → 同步权重到 SGLang → 跑一次完整 eval → 退出，不执行 optimizer update，也不保存新 checkpoint。

为每个评测分配独立端口。下面的端口只是示例；并发任务必须换成另一组无冲突端口。

```bash
SESSION=eval_step${ITER}_$STAMP
LAUNCHER=$REPO/clawgym-official/.run_step39_eval_sync_20260812-091700.sh

tmux new-session -d -s "$SESSION" \
  env \
    RUN_STAMP="$STAMP" \
    EXP_NAME="$EXP" \
    SAVE_CKPT="$OUT" \
    LOAD_CKPT="$LOAD_VIEW" \
    OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
    OPENCLAW_RECORD_FILE="$OUT/official_proxy_records.jsonl" \
    TENSORBOARD_DIR="$OUT/tensorboard" \
    OPENCLAW_REMOTE_PROXY_PORT=39101 \
    PORT=30101 \
    OPENCLAW_GATEWAY_PORT_BASE=32000 \
    OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=3000 \
    OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
    OPENCLAW_EVAL_NUM_WORKERS=32 \
    OPENCLAW_EVAL_N_SAMPLES=1 \
    OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
    OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 \
    OPENCLAW_CONTEXT_WINDOW=65536 \
    OPENCLAW_MAX_AGENT_STEPS=96 \
    OPENCLAW_MAX_COMPACTION_EVENTS=8 \
    ROLLOUT_TEMPERATURE=0.7 \
    ROLLOUT_TOP_P=1.0 \
    bash "$LAUNCHER"
```

端口含义：

- `PORT=30101`：本机 RL Proxy；
- `OPENCLAW_REMOTE_PROXY_PORT=39101`：远端容器访问本机 Proxy 的 SSH reverse port；
- `OPENCLAW_GATEWAY_PORT_BASE=32000`：远端 OpenClaw gateway 起始端口；
- `OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256`：覆盖 200 个 eval task，并限制预留端口范围；
- 本例完整 gateway 校验范围截至 `37550`，不要与其他训练/评测使用的端口段重叠。

## 5. 监控与完成判定

```bash
tmux attach -t "$SESSION"

# 另一个终端
LOG=$(find "$OUT/logs" -type f -name '*.log' | sort | tail -1)
tail -f "$LOG"
```

确认 checkpoint 真的加载正确：

```bash
rg 'loading distributed checkpoint|successfully loaded checkpoint' "$LOG"
```

应出现类似：

```text
loading distributed checkpoint ... at iteration 99
successfully loaded checkpoint ... at iteration 99
```

提取最终结果：

```bash
rg 'eval 0: \{' "$LOG" | tail -1
```

一次有效的完整评测必须同时满足：

```text
task_count = 200
n_collected = 200
n_skipped = 0
```

主结果字段：

- `eval/clawgym_eval-avg_reward`：200 题平均 reward，论文图表比较优先使用该值；
- `correct_rate`：`reward > 0.5` 的任务比例，仅作辅助；
- `avg_reward_collected_only`：只对成功收集的轨迹求平均；完整评测时应与 `avg_reward` 相等；
- `n_skipped`：未成功收集的任务数。

若 `n_skipped > 0`，默认 `avg_reward` 会把缺失任务按 0 计入，因此不能当成 checkpoint 的最终分数。应修复基础设施问题后完整重跑；仅补跑缺失任务可用于诊断，但正式比较最好使用一次完整的 200/200 运行。

## 6. SFT baseline 的同预算评测

SFT 使用专用脚本，不加载 Megatron RL checkpoint：

```bash
cd /root/openclaw-rl-local

SFT_STAMP=$(date -u +%Y%m%d-%H%M%S)
SFT_EXP=clawgym-sft-baseline-eval-same-budget-$SFT_STAMP

tmux new-session -d -s "sft_eval_$SFT_STAMP" \
  env \
    RUN_STAMP="$SFT_STAMP" \
    EXP_NAME="$SFT_EXP" \
    OPENCLAW_REMOTE_PROXY_PORT=39111 \
    PORT=30111 \
    OPENCLAW_GATEWAY_PORT_BASE=40000 \
    OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=2000 \
    OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
    bash /root/openclaw-rl-local/scripts/run_sft_baseline_eval_same_budget.sh
```

启动后必须从日志开头核对实际端口和预算。

## 7. 多 checkpoint 比较规范

建议按 `SFT → step19 → step39 → step59 → step79 → step99` 顺序串行评测，并记录：

| checkpoint | avg_reward | correct_rate | collected/skipped | 温度 | 运行时间 | 输出目录 |
|---|---:|---:|---:|---:|---:|---|
| SFT |  |  | 200/0 | 0.7 |  |  |
| step19 |  |  | 200/0 | 0.7 |  |  |
| step39 |  |  | 200/0 | 0.7 |  |  |
| step59 |  |  | 200/0 | 0.7 |  |  |
| step79 |  |  | 200/0 | 0.7 |  |  |
| step99 |  |  | 200/0 | 0.7 |  |  |

### 7.1 当前初步结果

正式训练内每 20 step 的在线 eval 均为 200/200、`n=1`：

| checkpoint | avg_reward | correct rate (`reward>0.5`) | collected/skipped |
|---|---:|---:|---:|
| step19 | 0.552119 | 0.590 | 200/0 |
| step39 | 0.549433 | 0.575 | 200/0 |
| step59 | 0.562933 | 0.610 | 200/0 |
| step79 | 0.572333 | 0.615 | 200/0 |
| step99 | 0.576312 | 0.630 | 200/0 |

独立同预算复评：

| 模型/运行 | avg_reward | correct rate | collected/skipped | 结论 |
|---|---:|---:|---:|---|
| SFT baseline | 0.550022 | 0.575 | 200/0 | 有效 baseline |
| RL step99 独立复评 | 0.548956 | 0.580 | 200/0 | 有效，但与在线 eval 差异明显 |
| RL step39 首次独立复评 | 0.506216 | 0.575 | 178/22 | 无效最终分数；缺失题被按 0 计入 |
| step39 缺失 22 题补跑 | 0.568471 | 0.591 | 22/0 | 仅用于诊断缺失任务，不是完整 200 题复评 |

step39 在线 eval 的 `0.549433`（54.94%）是一次完整 200/200 运行。独立复评的 `0.506216` 不能用来否定它，因为该次有 22 题未收集；这 22 题单独补跑的平均 reward 为 `0.568471`。

step99 在线 eval 为 `0.576312`，独立复评为 `0.548956`，相差 `0.027356`（2.74 个百分点）。两次都是 200/200，因此该差异主要反映 `n=1, temperature=0.7` 下的采样波动和可能的运行时随机性，而不是 skipped。当前结果只能支持“在线 eval 从 step39 到 step99 呈上升趋势”，尚不足以证明 step99 的真实性能稳定高于 SFT。

结果来源：

- 在线 eval：`outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700/tensorboard/`
- SFT baseline：`outputs/clawgym-sft-baseline-eval-same-budget-20260812-004700/logs/eval_20260812-004700.log`
- step99 独立复评：`outputs/clawgym-rl-step99-reeval-same-budget-20260812-105300/logs/train_20260812-105300.log`
- step39 独立复评：`outputs/clawgym-rl-step39-reeval-same-budget-20260812-091700/logs/train_20260812-091700.log`
- step39 缺失题补跑：`outputs/clawgym-rl-step39-missing22-retry-20260812-104600/logs/train_20260812-104600.log`

### 7.2 结果解释与后续要求

- `n=1` 且 `temperature=0.7` 方差较大；不要仅凭单次 200 题运行选择最佳 checkpoint。
- 对 step39、step79、step99 至少各独立复评 2–3 次，报告均值、标准差和 95% bootstrap CI。
- 保留每题 reward，并在不同 checkpoint 间做 paired comparison；总体均值之外，应报告每题提升/下降/持平数量。
- 同一 checkpoint 的复评必须保持 200 题、预算、环境镜像、skills、网络配置和采样参数一致。
- 复评时不要为了降低方差而临时降低温度；若要增加 deterministic eval，应作为另一套口径单独报告。

## 8. 常见错误

1. **加载错 checkpoint**：只看目录名不够，必须在日志中确认 `iteration N`。
2. **端口冲突**：会表现为 gateway 启动失败、请求进入其他 Proxy、SSH tunnel 失败或远端容器互相干扰。
3. **复用 Ray**：launcher 会拒绝已有 Ray；不要让评测接入训练中的集群。
4. **把 skipped 当模型失败**：先区分 agent 正常失败与基础设施未收集。正式分数要求 200/200。
5. **比较不同预算**：response、context、agent steps、compaction、workers、temperature 任一变化都会破坏可比性。
6. **误读 correct rate**：它只是 `reward > 0.5`，主指标仍是平均 verifier reward。
7. **泄露密钥**：不要 `cat .env`，不要把 `.env` 复制到报告或日志；仅通过 `OPENCLAW_ENV_FILE` 注入远端容器。
8. **会话不持久**：必须使用 tmux；确认主进程属于 tmux server，而不是 Codex/SSH 前台会话。
9. **清理越界**：评测结束只清理本次 session、Ray、SSH tunnel 和本次远端容器，不操作其他人的容器。

## 9. 建议归档内容

每次评测至少保留：

```text
checkpoint_load/
logs/
openclaw_logs/
tensorboard/
official_proxy_records.jsonl
```

同时在汇总表记录 checkpoint 的真实路径、iteration、launcher commit/工作树状态、gateway image digest、采样参数和端口分配。这样才能区分模型差异、采样波动与运行环境变化。
