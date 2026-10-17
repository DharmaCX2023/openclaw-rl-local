#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
TRAIN_ROOT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129"
EVAL_STEP="${EVAL_STEP:-${1:-80}}"
EVAL_RUN_TAG="${EVAL_RUN_TAG:-20260827}"
if [[ ! "${EVAL_STEP}" =~ ^[0-9]+$ ]] || (( EVAL_STEP <= 0 )); then
  echo "EVAL_STEP must be a positive integer: ${EVAL_STEP}" >&2
  exit 1
fi
FRAMEWORK_ITERATION=$((EVAL_STEP - 1))
printf -v ITER_DIR 'iter_%07d' "${FRAMEWORK_ITERATION}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/outputs/clawgym-step${EVAL_STEP}-supplemental-eval-${EVAL_RUN_TAG}}"
LOAD_VIEW="${EVAL_ROOT}/checkpoint_view"

if [[ ! -s "${TRAIN_ROOT}/${ITER_DIR}/.metadata" ]] || \
   [[ ! -s "${TRAIN_ROOT}/${ITER_DIR}/common.pt" ]]; then
  echo "Incomplete step-${EVAL_STEP} checkpoint: ${TRAIN_ROOT}/${ITER_DIR}" >&2
  exit 1
fi

mkdir -p "${LOAD_VIEW}" "${EVAL_ROOT}/logs"
ln -sfn "${TRAIN_ROOT}/${ITER_DIR}" "${LOAD_VIEW}/${ITER_DIR}"
printf '%s\n' "${FRAMEWORK_ITERATION}" >"${LOAD_VIEW}/latest_checkpointed_iteration.txt"

export EXP_NAME="clawgym-step${EVAL_STEP}-supplemental-eval-${EVAL_RUN_TAG}"
export RUN_STAMP="${RUN_STAMP:-step${EVAL_STEP}-supplement-$(date -u +%Y%m%d-%H%M%S)}"
export TRAIN_ENTRYPOINT="${REPO_ROOT}/slime/eval_loaded_checkpoint_once.py"
export OPENCLAW_EVAL_SEMANTIC_STEP="${EVAL_STEP}"
export LOAD_CKPT="${LOAD_VIEW}"
# Keep --save pointed at the source run solely so the evidence manifest resolves
# the selected iteration to the actual checkpoint. The eval-only Python entrypoint never saves.
export SAVE_CKPT="${TRAIN_ROOT}"
export USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1
export OPENCLAW_DISABLE_CHECKPOINT_SAVE=0
export SAVE_INTERVAL=20
export SAVE_RETAIN_INTERVAL=0

# Isolate the supplemental eval from the formal run's local proxy and tunnel.
export PORT="${PORT:-$((30500 + EVAL_STEP))}"
export OPENCLAW_REMOTE_PROXY_PORT="${OPENCLAW_REMOTE_PROXY_PORT:-$((39500 + EVAL_STEP))}"
export OPENCLAW_GATEWAY_PORT_BASE="${OPENCLAW_GATEWAY_PORT_BASE:-18000}"
export OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=7000
export OPENCLAW_BUNDLED_PLUGINS_DIR="${OPENCLAW_BUNDLED_PLUGINS_DIR:-/app/dist/extensions}"

export NUM_ROLLOUT=100
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=8
export ROLLOUT_TEMPERATURE=0.7
export ROLLOUT_TOP_P=0.8
export OPENCLAW_TRAIN_NUM_WORKERS=32
export OPENCLAW_TRAIN_SCHEDULING=sequential
export OPENCLAW_TRAIN_SHUFFLE_SEED=0

export OPENCLAW_MAX_RESPONSE_TOKENS=8192
export OPENCLAW_TRAJECTORY_MAX_TOKENS=65536
export OPENCLAW_CONTEXT_WINDOW=65536
export OPENCLAW_MAX_AGENT_STEPS=96
export OPENCLAW_MAX_COMPACTION_EVENTS=8
export OPENCLAW_TOKEN_LEDGER_MODE=enforce
export OPENCLAW_CHAT_TURN_TIMEOUT=1200
export SGLANG_WATCHDOG_TIMEOUT=1800

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

export SGLANG_SESSION_AFFINITY=1
export SGLANG_AFFINITY_MODE=task_sample_fixed
export SGLANG_AFFINITY_SAMPLES_PER_TASK=8
export SGLANG_AFFINITY_MAX_LOAD_DELTA=2
export SGLANG_AFFINITY_MIGRATION_LOAD_DELTA=4
export SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS=10

export EVAL_INTERVAL=20
export SKIP_EVAL_BEFORE_TRAIN=1
export OPENCLAW_EVAL_NUM_WORKERS=32
export OPENCLAW_EVAL_N_SAMPLES=1
export OPENCLAW_EVAL_SAVE_WORKSPACES=1
export OPENCLAW_EVAL_ARTIFACT_STRICT=0
export OPENCLAW_EVAL_WORKSPACE_CAPTURE_ATTEMPTS=3
export OPENCLAW_EVAL_WORKSPACE_RETRY_DELAY_SECONDS=2
export OPENCLAW_EVAL_WORKSPACE_MAX_BYTES=134217728
export OPENCLAW_EVAL_WORKSPACE_TIMEOUT=600
export OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE=20260826
export OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=1
export OPENCLAW_EVAL_ARTIFACT_DIR="${EVAL_ROOT}/eval_artifacts"
export OPENCLAW_RECORD_FILE="${EVAL_ROOT}/official_proxy_records.jsonl"
export TENSORBOARD_DIR="${EVAL_ROOT}/tensorboard"
export USE_TENSORBOARD=1
export USE_WANDB=0

exec bash "${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
