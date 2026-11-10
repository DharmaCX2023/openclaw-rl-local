#!/usr/bin/env bash
# 100 GRPO updates from the authors' HF step9.
# Kept: per-rollout loss, multi-tool, 6-tool gateway prompt.
# Off: TITO, TIS, routing replay, rollout routing replay, span-trace,
# affinity, deterministic sampling, fixed tool-call ids.
# Context assembly is the public splice. 30B still needs RoPE 1e7, EP4,
# SGLang mem 0.65, 81920 tokens/GPU, and fusion off.
set -euo pipefail
cd /root/openclaw-rl-local

NAME=step9plain-grpo100-20261104
export RUN_STAMP="20261104-${NAME}"
export EXP_NAME="clawgym-${NAME}"
export SAVE_CKPT="/mnt/humf/openclaw-rl-checkpoints/${NAME}"
export HF_CKPT=/mnt/humf/openclaw-rl-checkpoints/official-clawgym-paper-checkpoints-20261027/step9
export REF_LOAD="$HF_CKPT"
# range(start, num_rollout): rollouts 9..108 inclusive = 100 updates.
export NUM_ROLLOUT=109
if [ -f "${SAVE_CKPT}/latest_checkpointed_iteration.txt" ]; then
  ITER="$(tr -d '[:space:]' < "${SAVE_CKPT}/latest_checkpointed_iteration.txt")"
  export LOAD_CKPT="${SAVE_CKPT}"
  export START_ROLLOUT_ID="$((ITER + 1))"
  export USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1
  echo "Resuming from iteration ${ITER}, next rollout ${START_ROLLOUT_ID}"
else
  unset LOAD_CKPT || true
  export START_ROLLOUT_ID=9
fi
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=8
export NUM_GPUS=8 ACTOR_GPUS=4 EXPECTED_ACTOR_GPUS=4 ROLLOUT_GPUS=4 SGLANG_TP=2
export OPENCLAW_TRAIN_NUM_WORKERS=64 OPENCLAW_TRAIN_SCHEDULING=sequential
export OPENCLAW_EVAL_NUM_WORKERS=32
export OPENCLAW_TRAIN_SHUFFLE_SEED=0
export OPENCLAW_GATEWAY_IMAGE=clawgym-rl:v0.1
unset OPENCLAW_GATEWAY_CONFIG || true
export OPENCLAW_ASSEMBLE_OFFICIAL_SPLICE=1
export OPENCLAW_TOKEN_LEDGER_MODE=legacy
unset OPENCLAW_MAX_TOOL_CALLS_PER_TURN || true
export OPENCLAW_ROLLOUT_FN=clawgym_rl_rollout.generate_rollout_clawgym_rl
export OPENCLAW_GENERATE_FN=clawgym_rl_api_server.generate
export OPENCLAW_RM_FN=clawgym_rl_api_server.reward_func
unset CALCULATE_PER_TOKEN_LOSS || true
export OPENCLAW_MAX_AGENT_STEPS=1000000
export OPENCLAW_MAX_COMPACTION_EVENTS=1000000
export OPENCLAW_REMOTE_BASE="/root/clawgym_${NAME}"
export OPENCLAW_REMOTE_GATEWAY_PORT_MODE=run_scoped
export OPENCLAW_REMOTE_GATEWAY_PORT_MIN=16000 OPENCLAW_REMOTE_GATEWAY_PORT_MAX=29000
export OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4 OPENCLAW_REMOTE_SSH_POOL_SIZE=8
export OPENCLAW_CONTEXT_WINDOW=65536 OPENCLAW_TRAJECTORY_MAX_TOKENS=65536
export OPENCLAW_MAX_RESPONSE_TOKENS=8192
export OPENCLAW_CHAT_TURN_TIMEOUT=1200

export ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=1.0 ROLLOUT_TOP_K=-1
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0 MEGATRON_DETERMINISTIC_MODE=0
export OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=0
export SGLANG_SESSION_AFFINITY=0
export NO_GRADIENT_ACCUMULATION_FUSION=1

export USE_ROUTING_REPLAY=0 USE_ROLLOUT_ROUTING_REPLAY=0 USE_TIS=0
export MAX_TOKENS_PER_GPU=81920 KL_LOSS_MONITOR_INTERVAL=5 ENTROPY_COEF=0.0

# Eval interval must be a positive integer. 1000000 never fires before step 108.
export SKIP_EVAL_BEFORE_TRAIN=1 EVAL_INTERVAL=1000000
export SAVE_INTERVAL=5 SAVE_RETAIN_INTERVAL=0 OPENCLAW_DISABLE_CHECKPOINT_SAVE=0
export USE_WANDB=0 USE_TENSORBOARD=1
export OPENCLAW_RECORD_ENABLED=1
export OPENCLAW_RECORD_FILE="${SAVE_CKPT}/proxy_records.jsonl"

export PORT=30871 OPENCLAW_REMOTE_PROXY_PORT=39871
export RAY_GCS_PORT=6743 RAY_DASHBOARD_PORT=8643 RAY_CLIENT_PORT=10443
export RAY_MIN_WORKER_PORT=46000 RAY_MAX_WORKER_PORT=48999
export SLIME_SGLANG_ENGINE_PORT_BASE=20250 RAY_TEMP_DIR=/tmp/ray-step9plain

bash clawgym-official/run_span_trace_remote_grpo.sh
