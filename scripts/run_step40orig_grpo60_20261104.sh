#!/usr/bin/env bash
# 60 GRPO updates from the authors' released step40 weights.
# Same mainline as the step9 continuation that stayed near 0.54:
# per-rollout loss, our gateway (Serper skill copied in), span-trace image,
# exact token ledger, affinity on, natural sampling. Multi-tool, TIS, and
# routing replay stay on. Task cursor starts at rollout 40.
set -euo pipefail
cd /root/openclaw-rl-local

NAME=step40orig-grpo60-20261104
export RUN_STAMP="20261104-${NAME}"
export EXP_NAME="clawgym-${NAME}"
export SAVE_CKPT="/mnt/humf/openclaw-rl-checkpoints/${NAME}"
export OPENCLAW_SPAN_TRACE_ROOT="${SAVE_CKPT}/span_audit"
export HF_CKPT=/mnt/humf/openclaw-rl-checkpoints/official-clawgym-paper-checkpoints-20261027/step40
export REF_LOAD="$HF_CKPT"
export NUM_ROLLOUT=100
if [ -f "${SAVE_CKPT}/latest_checkpointed_iteration.txt" ]; then
  ITER="$(tr -d '[:space:]' < "${SAVE_CKPT}/latest_checkpointed_iteration.txt")"
  export LOAD_CKPT="${SAVE_CKPT}"
  export START_ROLLOUT_ID="$((ITER + 1))"
  export USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1
  echo "Resuming from iteration ${ITER}, next rollout ${START_ROLLOUT_ID}"
else
  unset LOAD_CKPT || true
  export START_ROLLOUT_ID=40
fi
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=8
export NUM_GPUS=8 ACTOR_GPUS=4 EXPECTED_ACTOR_GPUS=4 ROLLOUT_GPUS=4 SGLANG_TP=2
export OPENCLAW_TRAIN_NUM_WORKERS=8 OPENCLAW_TRAIN_SCHEDULING=sequential
export OPENCLAW_TRAIN_SHUFFLE_SEED=0
export OPENCLAW_GATEWAY_IMAGE=clawgym-span-trace:20260921
unset OPENCLAW_GATEWAY_CONFIG || true
unset OPENCLAW_ASSEMBLE_OFFICIAL_SPLICE || true
unset OPENCLAW_MAX_TOOL_CALLS_PER_TURN || true
unset CALCULATE_PER_TOKEN_LOSS || true
export OPENCLAW_REMOTE_BASE="/root/clawgym_${NAME}"
export OPENCLAW_REMOTE_GATEWAY_PORT_MODE=run_scoped
export OPENCLAW_REMOTE_GATEWAY_PORT_MIN=43000 OPENCLAW_REMOTE_GATEWAY_PORT_MAX=63000
export OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=2 OPENCLAW_REMOTE_SSH_POOL_SIZE=8
export OPENCLAW_CONTEXT_WINDOW=65536 OPENCLAW_TRAJECTORY_MAX_TOKENS=65536
export OPENCLAW_MAX_RESPONSE_TOKENS=8192 OPENCLAW_MAX_AGENT_STEPS=96
export OPENCLAW_MAX_COMPACTION_EVENTS=8 OPENCLAW_CHAT_TURN_TIMEOUT=1200
export OPENCLAW_TOKEN_LEDGER_MODE=enforce

export ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=1.0 ROLLOUT_TOP_K=-1
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0 MEGATRON_DETERMINISTIC_MODE=0
export OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=0
export SGLANG_SESSION_AFFINITY=1
export SGLANG_AFFINITY_MODE=task_sample_fixed SGLANG_AFFINITY_SAMPLES_PER_TASK=8
export NO_GRADIENT_ACCUMULATION_FUSION=1

export USE_ROUTING_REPLAY=1 USE_ROLLOUT_ROUTING_REPLAY=1 USE_TIS=1
export TIS_CLIP=2.0 TIS_CLIP_LOW=0.0
export MAX_TOKENS_PER_GPU=81920 KL_LOSS_MONITOR_INTERVAL=5 ENTROPY_COEF=0.0

export SKIP_EVAL_BEFORE_TRAIN=1 EVAL_INTERVAL=20
export SAVE_INTERVAL=5 SAVE_RETAIN_INTERVAL=0 OPENCLAW_DISABLE_CHECKPOINT_SAVE=0
export USE_WANDB=0 USE_TENSORBOARD=1
export OPENCLAW_RECORD_ENABLED=1
export OPENCLAW_RECORD_FILE="${SAVE_CKPT}/proxy_records.jsonl"

export PORT=30841 OPENCLAW_REMOTE_PROXY_PORT=39841
export RAY_GCS_PORT=6543 RAY_DASHBOARD_PORT=8443 RAY_CLIENT_PORT=10243
export RAY_MIN_WORKER_PORT=36000 RAY_MAX_WORKER_PORT=38999
export SLIME_SGLANG_ENGINE_PORT_BASE=19850 RAY_TEMP_DIR=/tmp/ray-step40orig

bash clawgym-official/run_span_trace_remote_grpo.sh
