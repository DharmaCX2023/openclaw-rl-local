#!/usr/bin/env bash
set -euo pipefail

cd /root/openclaw-rl-local

export EXP_NAME="${EXP_NAME:-clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129}"
export RUN_STAMP="${RUN_STAMP:-20260826-0129}"

# Isolated local proxy, reverse-tunnel endpoint, and run-scoped gateway ports.
export PORT=30321
export OPENCLAW_REMOTE_PROXY_PORT=39321
export OPENCLAW_GATEWAY_PORT_BASE=17000
export OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=6000
export OPENCLAW_GATEWAY_PORT_MAX_SLOTS=4000
export OPENCLAW_GATEWAY_PORT_STRIDE=10
export OPENCLAW_REMOTE_GATEWAY_PORT_MODE=run_scoped
export OPENCLAW_REMOTE_GATEWAY_PORT_MIN=20000
export OPENCLAW_REMOTE_GATEWAY_PORT_MAX=62000
export OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS=512
export OPENCLAW_REMOTE_START_ATTEMPTS=3
export OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4
export OPENCLAW_REMOTE_SSH_POOL_SIZE=8

# Formal 100-step GRPO run: 8 tasks x 8 trajectories per rollout step.
export NUM_ROLLOUT=100
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=8
export OPENCLAW_TRAIN_NUM_WORKERS=32
export OPENCLAW_TRAIN_SCHEDULING=sequential
export OPENCLAW_TRAIN_SHUFFLE_SEED=0
export ROLLOUT_TEMPERATURE=0.7
export ROLLOUT_TOP_P=0.8

# OpenClaw execution budgets and history handling.
export OPENCLAW_MAX_RESPONSE_TOKENS=8192
export OPENCLAW_TRAJECTORY_MAX_TOKENS=65536
export OPENCLAW_CONTEXT_WINDOW=65536
export OPENCLAW_MAX_AGENT_STEPS=96
export OPENCLAW_MAX_COMPACTION_EVENTS=8
export OPENCLAW_TOKEN_LEDGER_MODE=enforce
export OPENCLAW_CHAT_TURN_TIMEOUT=1200
export SGLANG_WATCHDOG_TIMEOUT=1800

# Train/inference consistency and PPO stabilization.
export USE_ROUTING_REPLAY=1
export USE_ROLLOUT_ROUTING_REPLAY=1
export USE_TIS=1
export TIS_CLIP=2.0
export TIS_CLIP_LOW=0.0
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
export MEGATRON_DETERMINISTIC_MODE=1
export SGLANG_ENABLE_CUDA_GRAPH=1
export MAX_TOKENS_PER_GPU=81920
export KL_LOSS_MONITOR_INTERVAL=5
export ENTROPY_COEF=0.0

# Strict session-level balancing: each task's eight rollout samples are assigned
# 4/4 across the two engines, and no in-flight agent session may migrate.
export SGLANG_SESSION_AFFINITY=1
export SGLANG_AFFINITY_MODE=task_sample_fixed
export SGLANG_AFFINITY_SAMPLES_PER_TASK=8
export SGLANG_AFFINITY_MAX_LOAD_DELTA=2
export SGLANG_AFFINITY_MIGRATION_LOAD_DELTA=4
export SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS=10

# Reproducible full-benchmark eval at semantic steps 0/20/40/60/80/100.
export EVAL_INTERVAL=20
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"
export OPENCLAW_EVAL_NUM_WORKERS=32
export OPENCLAW_EVAL_N_SAMPLES=1
export OPENCLAW_EVAL_SAVE_WORKSPACES=1
# A transient remote freeze/archive failure must not terminate training. The
# manifest remains explicitly incomplete and the affected checkpoint eval is
# rerun out of band with the same deterministic configuration.
export OPENCLAW_EVAL_ARTIFACT_STRICT=0
export OPENCLAW_EVAL_WORKSPACE_CAPTURE_ATTEMPTS=3
export OPENCLAW_EVAL_WORKSPACE_RETRY_DELAY_SECONDS=2
export OPENCLAW_EVAL_WORKSPACE_MAX_BYTES=134217728
export OPENCLAW_EVAL_WORKSPACE_TIMEOUT=600
export OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE=20260826
export OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=1

# Five full resumable checkpoints, retained locally at steps 20/40/60/80/100.
export OPENCLAW_DISABLE_CHECKPOINT_SAVE=0
export SAVE_INTERVAL=20
export SAVE_RETAIN_INTERVAL=0
export LOCAL_CHECKPOINT_MIN_FREE_GB=500

export USE_TENSORBOARD=1
export USE_WANDB=0

exec bash clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh
