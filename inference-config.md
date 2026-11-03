# ClawGym RL 权重评测 — 完整推理配置（迁移用）

生成时间：2026-10-28。本配置与官方 4B 训练脚本（`RL/run_qwen3_4b.sh`）的 rollout 参数一致，
用于评测作者发布的 Base / RL step 40 / RL step 100 权重（Qwen3-30B-A3B MoE，BF16）。

## 1. 采样参数（核心）

| 参数 | 值 | 说明 |
|---|---|---|
| temperature | **0.7** | 训练 rollout 同值 |
| top_p | **1.0** | 官方训练脚本未显式设置 top-p，slime 默认 1.0（注意：**不是** 本地训练实验里用过的 0.8） |
| top_k | **-1**（关闭） | 同上，slime 默认 |
| fixed sampling seed | **无** | 不设置任何固定采样种子，与自然 RL rollout 一致 |
| deterministic inference | **关闭** | `SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0` |
| deterministic tool call ids | **关闭** | `OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=0` |
| Megatron deterministic mode | **关闭** | `MEGATRON_DETERMINISTIC_MODE=0` |

评测与 rollout 用同一组采样参数：`eval_temperature=0.7, eval_top_p=1.0, eval_top_k=-1`；
`rollout_temperature=0.7, rollout_top_p=1.0, rollout_top_k=-1`。

评测时显式清空/移除的确定性变量（迁移时同样不要设置）：
`OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE`、`PYTHONHASHSEED`、`CUBLAS_WORKSPACE_CONFIG`、
`NVIDIA_TF32_OVERRIDE`、`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`、`NCCL_ALGO`、`NCCL_PROTO`、
`NVTE_ALLOW_NONDETERMINISTIC_ALGO`。

## 2. 长度与上下文（8k / 64k，与训练一致）

| 参数 | 值 |
|---|---|
| 单次响应最大 token（`OPENCLAW_MAX_RESPONSE_TOKENS`） | **8192** |
| 轨迹总长（`OPENCLAW_TRAJECTORY_MAX_TOKENS`） | **65536** |
| 上下文窗口（`OPENCLAW_CONTEXT_WINDOW`） | **65536** |
| SGLang server `--context-length` | **65536** |
| 单个 agent 最大步数（`OPENCLAW_MAX_AGENT_STEPS`） | **96** |
| 压缩事件上限（`OPENCLAW_MAX_COMPACTION_EVENTS`） | **8** |
| token ledger 模式（`OPENCLAW_TOKEN_LEDGER_MODE`） | **enforce** |

## 3. SGLang 服务端参数（本次实测值）

| 参数 | 值 |
|---|---|
| tp_size | 2（8 卡 = 4 个 tp=2 的引擎，base_gpu_id 0/2/4/6） |
| `mem_fraction_static` | 0.65 |
| `chunked_prefill_size` | 8192 |
| `max_prefill_tokens` | 16384 |
| attention backend | fa3 |
| sampling backend | flashinfer |
| CUDA graph | **开启**（`SGLANG_ENABLE_CUDA_GRAPH=1`，`cuda_graph_max_bs=256`） |
| `SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION` | 1 |
| tool_call_parser | `qwen25`（slime 传入；新版 SGLang 警告 deprecated，等价 `qwen`） |
| dtype | auto（BF16 权重） |
| 页大小 / radix cache | page_size=1，radix cache 开 |
| overlap schedule / mixed chunk | 开 / 关（默认） |

## 4. Agent / 评测运行参数

| 参数 | 值 |
|---|---|
| 评测任务 | `clawgym_eval` 200 题，每轮全量 |
| 每任务采样次数（`OPENCLAW_EVAL_N_SAMPLES`） | 1 |
| 评测并发 worker（`OPENCLAW_EVAL_NUM_WORKERS`） | 8 |
| 评测轮数（`OPENCLAW_EVAL_MAX_TURNS`） | 1 |
| 单轮对话超时（`OPENCLAW_CHAT_TURN_TIMEOUT`） | 1200 秒 |
| format penalty（`OPENCLAW_FORMAT_PENALTY`） | 0 |
| 远程 Docker 启动并发（`OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY`） | 8 |
| SSH 连接池 | 8 |
| gateway 镜像 | `clawgym-rl@sha256:33a30ab107cb7fef6a84f17b3fb70a0e849654e9201c61dc9f20baeac21385a3` |

## 5. 端口布局（多轮并行时避免冲突）

serial = 模型序号×5 + 轮次：
- proxy 端口 = 60000 + serial
- api server 端口 = 37000 + serial
- gateway 端口基数 = 20000 + serial×1000，步长 10（每 run 独立网关槽位）
- gateway max slots = 256

## 6. 与官方训练 rollout 的对齐情况

- 官方 4B 脚本：temperature 0.7；未设 top-p/top-k（默认 1.0/-1）；`OPENCLAW_MAX_TOKENS` 8192；context 65536 → 本评测完全对齐。
- 与**本地** GRPO 训练的已知差异（此前审计结论，仅记录）：本地训练期 rollout 用过 top-p 0.8 / top-k 20；评测按官方口径用 1.0/-1。
- SGLang `ServerArgs.random_seed`（1234/1235 等）是引擎内部初始化值；不设置 per-request seed 且关掉确定性推理时，采样即为自然随机，不需要在迁移时处理。

## 7. 评测口径

- 指标：200 题 hybrid verifier 平均 reward（`avg_reward`，满分 1.0），correct = reward > 0.5。
- 每个权重跑 5 轮，报告均值与样本标准差。
- 此口径与论文 Figure 3 的 code-only 曲线（纵轴为 code-only 通过率，端点 35.7%/56.7%）**不是同一指标**，不能直接对数。
