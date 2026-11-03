#!/usr/bin/env bash
# Continue GRPO training from the authors' released RL step9 checkpoint.
# Mainline pipeline (per-rollout loss, multi-tool rollouts) with natural
# sampling: temperature 0.7, top-p 1.0, top-k -1, no fixed sampling seed,
# deterministic inference off. 100 steps, batch 8 tasks x 8 samples.
set -euo pipefail
cd /root/openclaw-rl-local

NAME=step9cont-grpo100-20261029
export RUN_STAMP="20261029-${NAME}"
export EXP_NAME="clawgym-${NAME}"
export SAVE_CKPT="/mnt/humf/openclaw-rl-checkpoints/${NAME}"
export OPENCLAW_SPAN_TRACE_ROOT="${SAVE_CKPT}/span_audit"
export HF_CKPT=/mnt/humf/openclaw-rl-checkpoints/official-clawgym-paper-checkpoints-20261027/step9
export REF_LOAD="$HF_CKPT"
if [ "${WATCHDOG_RESUME:-0}" = "1" ]; then
  # Resume from the last saved Megatron iteration in SAVE_CKPT.
  export LOAD_CKPT="$SAVE_CKPT"
  export OVERRIDE_OPT_PARAM_SCHEDULER=1
else
  unset LOAD_CKPT || true
fi

export NUM_GPUS=8 ACTOR_GPUS=4 EXPECTED_ACTOR_GPUS=4 ROLLOUT_GPUS=4 SGLANG_TP=2
export NUM_ROLLOUT=100 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8
export OPENCLAW_TRAIN_NUM_WORKERS=8 OPENCLAW_TRAIN_SCHEDULING=sequential
export OPENCLAW_TRAIN_SHUFFLE_SEED=0
export OPENCLAW_GATEWAY_IMAGE=clawgym-span-trace:20260921
export OPENCLAW_REMOTE_BASE="/root/clawgym_${NAME}"
export OPENCLAW_REMOTE_GATEWAY_PORT_MODE=run_scoped
export OPENCLAW_REMOTE_GATEWAY_PORT_MIN=42000 OPENCLAW_REMOTE_GATEWAY_PORT_MAX=62000
export OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=2 OPENCLAW_REMOTE_SSH_POOL_SIZE=8
export OPENCLAW_CONTEXT_WINDOW=65536 OPENCLAW_TRAJECTORY_MAX_TOKENS=65536
export OPENCLAW_MAX_RESPONSE_TOKENS=8192 OPENCLAW_MAX_AGENT_STEPS=96
export OPENCLAW_MAX_COMPACTION_EVENTS=8 OPENCLAW_CHAT_TURN_TIMEOUT=1200
export OPENCLAW_TOKEN_LEDGER_MODE=enforce

# Natural (official-style) rollout sampling.
export ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=1.0 ROLLOUT_TOP_K=-1
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0 MEGATRON_DETERMINISTIC_MODE=0
export OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=0

# Mainline MoE rollout/train consistency mechanisms.
export USE_ROUTING_REPLAY=1 USE_ROLLOUT_ROUTING_REPLAY=1 USE_TIS=1
export TIS_CLIP=2.0 TIS_CLIP_LOW=0.0
export SGLANG_AFFINITY_MODE=task_sample_fixed SGLANG_AFFINITY_SAMPLES_PER_TASK=8
export MAX_TOKENS_PER_GPU=81920 KL_LOSS_MONITOR_INTERVAL=5 ENTROPY_COEF=0.0

export SKIP_EVAL_BEFORE_TRAIN=1 EVAL_INTERVAL=20
export SAVE_INTERVAL=5 SAVE_RETAIN_INTERVAL=0 OPENCLAW_DISABLE_CHECKPOINT_SAVE=0
export USE_WANDB=0 USE_TENSORBOARD=1
export OPENCLAW_RECORD_ENABLED=1
export OPENCLAW_RECORD_FILE="${SAVE_CKPT}/proxy_records.jsonl"
export OPENCLAW_SAVE_DEBUG_ROLLOUT_DATA="${SAVE_CKPT}/rollout_data/{rollout_id}.pt"

# Distinct ports from earlier runs.
export PORT=30711 OPENCLAW_REMOTE_PROXY_PORT=39711
export RAY_GCS_PORT=6413 RAY_DASHBOARD_PORT=8313 RAY_CLIENT_PORT=10113
export RAY_MIN_WORKER_PORT=26000 RAY_MAX_WORKER_PORT=28999
export SLIME_SGLANG_ENGINE_PORT_BASE=19630 RAY_TEMP_DIR=/tmp/ray-step9cont

exec bash clawgym-official/run_span_trace_remote_grpo.sh
