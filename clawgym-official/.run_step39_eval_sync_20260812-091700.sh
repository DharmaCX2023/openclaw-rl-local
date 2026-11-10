#!/bin/bash
# Official ClawGym/OpenClaw 2026.4.1 remote-Docker GRPO for Qwen3-30B-A3B-Instruct-2507.
# GPU node: Slime actor + SGLang + ClawGym RL proxy.
# CPU node: one digest-pinned clawgym-rl Docker container per official rollout.

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd -P)"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${REPO_ROOT}/Megatron-LM}"
PYTHON_ENV="${PYTHON_ENV:-${REPO_ROOT}/.venv-openclaw}"
HF_ASSET_ROOT="${HF_ASSET_ROOT:-/mnt/OpenClaw-RL/hf_ckpt}"
export PYTHON_ENV
export PATH="${PYTHON_ENV}/bin:${PATH}"

NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
SGLANG_TP="${SGLANG_TP:-2}"

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS" >&2
  exit 1
fi
if (( ACTOR_GPUS != 4 )); then
  echo "This 30B production entry expects ACTOR_GPUS=4 (TP4/EP4)." >&2
  exit 1
fi

HF_CKPT="${HF_CKPT:-${HF_ASSET_ROOT}/qwen3-30b-a3b-instruct-clawgym-sft-iter11127}"
REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
EXP_NAME="${EXP_NAME:-clawgym-official-30b-sft11127-${RUN_STAMP}}"
export SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION="${SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION:-1}"

LOCAL_CHECKPOINT_MIN_FREE_GB="${LOCAL_CHECKPOINT_MIN_FREE_GB:-500}"
CHECKPOINT_FALLBACK_ROOT="${CHECKPOINT_FALLBACK_ROOT:-/mnt/humf/openclaw-rl-checkpoints}"
if ! [[ "${LOCAL_CHECKPOINT_MIN_FREE_GB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LOCAL_CHECKPOINT_MIN_FREE_GB must be a positive integer." >&2
  exit 1
fi
LOCAL_CHECKPOINT_AVAILABLE_KIB="$(df -Pk "${REPO_ROOT}" | awk 'NR == 2 {print $4}')"
LOCAL_CHECKPOINT_AVAILABLE_GB=$((LOCAL_CHECKPOINT_AVAILABLE_KIB / 1024 / 1024))
if [ -n "${SAVE_CKPT:-}" ]; then
  CHECKPOINT_PATH_SOURCE="explicit"
elif (( LOCAL_CHECKPOINT_AVAILABLE_GB < LOCAL_CHECKPOINT_MIN_FREE_GB )); then
  SAVE_CKPT="${CHECKPOINT_FALLBACK_ROOT}/${EXP_NAME}"
  CHECKPOINT_PATH_SOURCE="mnt_fallback"
else
  SAVE_CKPT="${REPO_ROOT}/outputs/${EXP_NAME}"
  CHECKPOINT_PATH_SOURCE="local"
fi

EXPECTED_REPO_ROOT="/root/openclaw-rl-local"
if [ "${REPO_ROOT}" != "${EXPECTED_REPO_ROOT}" ]; then
  echo "Refusing to run outside the isolated workspace: ${REPO_ROOT}" >&2
  exit 1
fi
for LOCAL_RUNTIME_ROOT in "${SLIME_ROOT}" "${MEGATRON_ROOT}"; do
  if [ ! -d "${LOCAL_RUNTIME_ROOT}" ] || [ -L "${LOCAL_RUNTIME_ROOT}" ]; then
    echo "Local runtime root must be a real directory: ${LOCAL_RUNTIME_ROOT}" >&2
    exit 1
  fi
  LOCAL_RUNTIME_RESOLVED="$(readlink -f "${LOCAL_RUNTIME_ROOT}")"
  case "${LOCAL_RUNTIME_RESOLVED}" in
    "${REPO_ROOT}"/*) ;;
    *)
      echo "Runtime root escapes the isolated workspace: ${LOCAL_RUNTIME_RESOLVED}" >&2
      exit 1
      ;;
  esac
done
if [ ! -x "${PYTHON_ENV}/bin/python" ] || [ ! -f "${PYTHON_ENV}/bin/ray" ]; then
  echo "Python environment is incomplete: ${PYTHON_ENV}" >&2
  exit 1
fi
# Console scripts inside a copied/mounted venv can retain an absolute shebang
# from the source machine. Execute Ray's script with the verified interpreter
# so the isolated workspace remains relocatable.
RAY_CLI=("${PYTHON_ENV}/bin/python" "${PYTHON_ENV}/bin/ray")
"${RAY_CLI[@]}" --version >/dev/null
if [ ! -d "${HF_CKPT}" ]; then
  echo "HF checkpoint not found: ${HF_CKPT}" >&2
  exit 1
fi
if [ ! -d "${REF_LOAD}" ]; then
  echo "Reference checkpoint not found: ${REF_LOAD}" >&2
  exit 1
fi
CHECKPOINT_FALLBACK_ROOT_RESOLVED="$(readlink -m "${CHECKPOINT_FALLBACK_ROOT}")"
case "${CHECKPOINT_FALLBACK_ROOT_RESOLVED}" in
  /mnt/*) ;;
  *)
    echo "CHECKPOINT_FALLBACK_ROOT must stay under /mnt: ${CHECKPOINT_FALLBACK_ROOT_RESOLVED}" >&2
    exit 1
    ;;
esac
SAVE_CKPT_RESOLVED="$(readlink -m "${SAVE_CKPT}")"
case "${SAVE_CKPT_RESOLVED}" in
  "${REPO_ROOT}/outputs"/*|"${CHECKPOINT_FALLBACK_ROOT_RESOLVED}"/*) ;;
  *)
    echo "SAVE_CKPT must stay under ${REPO_ROOT}/outputs or ${CHECKPOINT_FALLBACK_ROOT_RESOLVED}: ${SAVE_CKPT_RESOLVED}" >&2
    exit 1
    ;;
esac
if [ -z "${LOAD_CKPT:-}" ] && [ -e "${SAVE_CKPT}" ] && [ -n "$(find "${SAVE_CKPT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "SAVE_CKPT must be a new or empty directory: ${SAVE_CKPT}" >&2
  exit 1
fi

export OPENCLAW_REMOTE_HOST="${OPENCLAW_REMOTE_HOST:-159.138.50.147}"
export OPENCLAW_REMOTE_USER="${OPENCLAW_REMOTE_USER:-root}"
export OPENCLAW_REMOTE_SSH_KEY="${OPENCLAW_REMOTE_SSH_KEY:-/root/openclaw_cpu_key}"
export OPENCLAW_REMOTE_BASE="${OPENCLAW_REMOTE_BASE:-/root/clawgym_rl_runs}"
export OPENCLAW_REMOTE_PROXY_PORT="${OPENCLAW_REMOTE_PROXY_PORT:-39081}"
export OPENCLAW_GATEWAY_IMAGE="${OPENCLAW_GATEWAY_IMAGE:-clawgym-rl@sha256:33a30ab107cb7fef6a84f17b3fb70a0e849654e9201c61dc9f20baeac21385a3}"
export OPENCLAW_BUNDLED_PLUGINS_DIR="${OPENCLAW_BUNDLED_PLUGINS_DIR:-/app/dist/extensions}"
export OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY="${OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY:-8}"
export OPENCLAW_GATEWAY_PORT_BASE="${OPENCLAW_GATEWAY_PORT_BASE:-16000}"
export OPENCLAW_GATEWAY_PORT_STRIDE="${OPENCLAW_GATEWAY_PORT_STRIDE:-10}"
export OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET="${OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET:-5000}"
export OPENCLAW_GATEWAY_PORT_MAX_SLOTS="${OPENCLAW_GATEWAY_PORT_MAX_SLOTS:-4000}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30001}"
export SGLANG_API_KEY="${SGLANG_API_KEY:-}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-30b-a3b-instruct-clawgym-official}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export OPENCLAW_CONTEXT_WINDOW="${OPENCLAW_CONTEXT_WINDOW:-65536}"
export OPENCLAW_TRAJECTORY_MAX_TOKENS="${OPENCLAW_TRAJECTORY_MAX_TOKENS:-65536}"
export OPENCLAW_MAX_RESPONSE_TOKENS="${OPENCLAW_MAX_RESPONSE_TOKENS:-8192}"
export OPENCLAW_MAX_AGENT_STEPS="${OPENCLAW_MAX_AGENT_STEPS:-96}"
export OPENCLAW_MAX_COMPACTION_EVENTS="${OPENCLAW_MAX_COMPACTION_EVENTS:-8}"
export OPENCLAW_TOKEN_LEDGER_MODE="${OPENCLAW_TOKEN_LEDGER_MODE:-enforce}"
export OPENCLAW_TRAIN_MAX_TURNS="${OPENCLAW_TRAIN_MAX_TURNS:-1}"
export OPENCLAW_EVAL_MAX_TURNS="${OPENCLAW_EVAL_MAX_TURNS:-1}"
export OPENCLAW_TRAIN_NUM_WORKERS="${OPENCLAW_TRAIN_NUM_WORKERS:-32}"
export OPENCLAW_EVAL_NUM_WORKERS="${OPENCLAW_EVAL_NUM_WORKERS:-32}"
export OPENCLAW_EVAL_N_SAMPLES="${OPENCLAW_EVAL_N_SAMPLES:-1}"
export OPENCLAW_FORMAT_PENALTY="${OPENCLAW_FORMAT_PENALTY:-0}"
export OPENCLAW_CHAT_TURN_TIMEOUT="${OPENCLAW_CHAT_TURN_TIMEOUT:-1200}"
export OPENCLAW_REMOTE_SSH_POOL_SIZE="${OPENCLAW_REMOTE_SSH_POOL_SIZE:-8}"
export OPENCLAW_TRAIN_TASKS="${OPENCLAW_TRAIN_TASKS:-${SCRIPT_DIR}/data/clawgym_train}"
export OPENCLAW_EVAL_TASKS="${OPENCLAW_EVAL_TASKS:-${SCRIPT_DIR}/data/clawgym_eval}"
export OPENCLAW_TRAIN_SHUFFLE_SEED="${OPENCLAW_TRAIN_SHUFFLE_SEED:-0}"
export CLAWGYM_SERPER_SKILL_DIR="${CLAWGYM_SERPER_SKILL_DIR:-${REPO_ROOT}/sft_contract/recovered_skills/serper}"
export OPENCLAW_ENV_FILE="${OPENCLAW_ENV_FILE:-${REPO_ROOT}/.env}"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${OPENCLAW_RECORD_FILE:-${SAVE_CKPT}/official_proxy_records.jsonl}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SAVE_CKPT}/tensorboard}"
USE_TENSORBOARD="${USE_TENSORBOARD:-1}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
OPENCLAW_DISABLE_CHECKPOINT_SAVE="${OPENCLAW_DISABLE_CHECKPOINT_SAVE:-0}"
SGLANG_ENABLE_CUDA_GRAPH="${SGLANG_ENABLE_CUDA_GRAPH:-1}"
SAVE_INTERVAL_VALUE="${SAVE_INTERVAL:-20}"
SAVE_RETAIN_INTERVAL_VALUE="${SAVE_RETAIN_INTERVAL:-0}"
EVAL_INTERVAL_VALUE="${EVAL_INTERVAL:-20}"
if [ -n "${OPENCLAW_MAX_TOKENS:-}" ]; then
  echo "OPENCLAW_MAX_TOKENS is ambiguous and no longer supported; use OPENCLAW_MAX_RESPONSE_TOKENS." >&2
  exit 1
fi
for TOKEN_LIMIT in OPENCLAW_MAX_RESPONSE_TOKENS OPENCLAW_TRAJECTORY_MAX_TOKENS OPENCLAW_CONTEXT_WINDOW OPENCLAW_MAX_AGENT_STEPS OPENCLAW_MAX_COMPACTION_EVENTS; do
  if ! [[ "${!TOKEN_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${TOKEN_LIMIT} must be a positive integer: ${!TOKEN_LIMIT}" >&2
    exit 1
  fi
done
for EVAL_VALUE in EVAL_INTERVAL_VALUE OPENCLAW_EVAL_NUM_WORKERS OPENCLAW_EVAL_N_SAMPLES; do
  if ! [[ "${!EVAL_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${EVAL_VALUE} must be a positive integer: ${!EVAL_VALUE}" >&2
    exit 1
  fi
done
if ! [[ "${USE_TENSORBOARD}" =~ ^[01]$ ]]; then
  echo "USE_TENSORBOARD must be 0 or 1." >&2
  exit 1
fi
if ! [[ "${OPENCLAW_DISABLE_CHECKPOINT_SAVE}" =~ ^[01]$ ]]; then
  echo "OPENCLAW_DISABLE_CHECKPOINT_SAVE must be 0 or 1." >&2
  exit 1
fi
if ! [[ "${SGLANG_ENABLE_CUDA_GRAPH}" =~ ^[01]$ ]]; then
  echo "SGLANG_ENABLE_CUDA_GRAPH must be 0 or 1." >&2
  exit 1
fi
if [ "${OPENCLAW_DISABLE_CHECKPOINT_SAVE}" = "0" ]; then
  if ! [[ "${SAVE_INTERVAL_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAVE_INTERVAL_VALUE must be a positive integer: ${SAVE_INTERVAL_VALUE}" >&2
    exit 1
  fi
  if ! [[ "${SAVE_RETAIN_INTERVAL_VALUE}" =~ ^[0-9]+$ ]]; then
    echo "SAVE_RETAIN_INTERVAL_VALUE must be a non-negative integer: ${SAVE_RETAIN_INTERVAL_VALUE}" >&2
    exit 1
  fi
  if (( SAVE_RETAIN_INTERVAL_VALUE > 0 && SAVE_RETAIN_INTERVAL_VALUE % SAVE_INTERVAL_VALUE != 0 )); then
    echo "SAVE_RETAIN_INTERVAL must be 0 (retain all) or divisible by SAVE_INTERVAL." >&2
    exit 1
  fi
fi
if (( OPENCLAW_MAX_RESPONSE_TOKENS > OPENCLAW_TRAJECTORY_MAX_TOKENS )); then
  echo "Response limit must not exceed trajectory limit." >&2
  exit 1
fi
if (( OPENCLAW_TRAJECTORY_MAX_TOKENS > OPENCLAW_CONTEXT_WINDOW )); then
  echo "Trajectory limit must not exceed the SGLang/OpenClaw context window." >&2
  exit 1
fi
if ! [[ "${OPENCLAW_TRAIN_SHUFFLE_SEED}" =~ ^[0-9]+$ ]]; then
  echo "OPENCLAW_TRAIN_SHUFFLE_SEED must be a non-negative integer." >&2
  exit 1
fi
for PORT_VALUE in OPENCLAW_GATEWAY_PORT_BASE OPENCLAW_GATEWAY_PORT_STRIDE OPENCLAW_GATEWAY_PORT_MAX_SLOTS; do
  if ! [[ "${!PORT_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${PORT_VALUE} must be a positive integer." >&2
    exit 1
  fi
done
if ! [[ "${OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET must be a non-negative integer." >&2
  exit 1
fi
GATEWAY_LAST_PORT=$((OPENCLAW_GATEWAY_PORT_BASE + OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET + (OPENCLAW_GATEWAY_PORT_MAX_SLOTS - 1) * OPENCLAW_GATEWAY_PORT_STRIDE))
if (( GATEWAY_LAST_PORT > 65535 )); then
  echo "Gateway port range exceeds 65535: ${GATEWAY_LAST_PORT}" >&2
  exit 1
fi
if [ "${OPENCLAW_PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "Local RL preflight passed."
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "SLIME_ROOT=${SLIME_ROOT}"
  echo "MEGATRON_ROOT=${MEGATRON_ROOT}"
  echo "PYTHON_ENV=${PYTHON_ENV}"
  echo "HF_CKPT=${HF_CKPT}"
  echo "SAVE_CKPT=${SAVE_CKPT_RESOLVED}"
  echo "CHECKPOINT_PATH_SOURCE=${CHECKPOINT_PATH_SOURCE}"
  echo "LOCAL_CHECKPOINT_AVAILABLE_GB=${LOCAL_CHECKPOINT_AVAILABLE_GB}"
  echo "LOCAL_CHECKPOINT_MIN_FREE_GB=${LOCAL_CHECKPOINT_MIN_FREE_GB}"
  echo "OPENCLAW_GATEWAY_IMAGE=${OPENCLAW_GATEWAY_IMAGE}"
  echo "OPENCLAW_BUNDLED_PLUGINS_DIR=${OPENCLAW_BUNDLED_PLUGINS_DIR}"
  echo "OPENCLAW_MAX_RESPONSE_TOKENS=${OPENCLAW_MAX_RESPONSE_TOKENS}"
  echo "OPENCLAW_TRAJECTORY_MAX_TOKENS=${OPENCLAW_TRAJECTORY_MAX_TOKENS}"
  echo "OPENCLAW_CONTEXT_WINDOW=${OPENCLAW_CONTEXT_WINDOW}"
  echo "OPENCLAW_MAX_AGENT_STEPS=${OPENCLAW_MAX_AGENT_STEPS}"
  echo "OPENCLAW_MAX_COMPACTION_EVENTS=${OPENCLAW_MAX_COMPACTION_EVENTS}"
  echo "ENTROPY_COEF=${ENTROPY_COEF}"
  echo "USE_TENSORBOARD=${USE_TENSORBOARD}"
  echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
  echo "OPENCLAW_DISABLE_CHECKPOINT_SAVE=${OPENCLAW_DISABLE_CHECKPOINT_SAVE}"
  echo "SGLANG_ENABLE_CUDA_GRAPH=${SGLANG_ENABLE_CUDA_GRAPH}"
  echo "EVAL_INTERVAL=${EVAL_INTERVAL_VALUE}"
  echo "OPENCLAW_EVAL_NUM_WORKERS=${OPENCLAW_EVAL_NUM_WORKERS}"
  echo "OPENCLAW_EVAL_N_SAMPLES=${OPENCLAW_EVAL_N_SAMPLES}"
  echo "OPENCLAW_EVAL_TASKS=${OPENCLAW_EVAL_TASKS}"
  if [ "${OPENCLAW_DISABLE_CHECKPOINT_SAVE}" = "0" ]; then
    echo "SAVE_INTERVAL=${SAVE_INTERVAL_VALUE}"
    if (( SAVE_RETAIN_INTERVAL_VALUE == 0 )); then
      echo "CHECKPOINT_RETENTION=all"
    else
      echo "SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL_VALUE}"
    fi
  fi
  echo "OPENCLAW_GATEWAY_PORT_BASE=${OPENCLAW_GATEWAY_PORT_BASE}"
  echo "OPENCLAW_GATEWAY_LAST_PORT=${GATEWAY_LAST_PORT}"
  exit 0
fi
mkdir -p "${SAVE_CKPT}/logs"
exec > >(tee -a "${SAVE_CKPT}/logs/train_${RUN_STAMP}.log") 2>&1
echo "REPO_ROOT=${REPO_ROOT}"
echo "SLIME_ROOT=${SLIME_ROOT}"
echo "MEGATRON_ROOT=${MEGATRON_ROOT}"
echo "PYTHON_ENV=${PYTHON_ENV}"
echo "HF_CKPT=${HF_CKPT}"
echo "SAVE_CKPT=${SAVE_CKPT_RESOLVED}"
echo "CHECKPOINT_PATH_SOURCE=${CHECKPOINT_PATH_SOURCE}"
echo "LOCAL_CHECKPOINT_AVAILABLE_GB=${LOCAL_CHECKPOINT_AVAILABLE_GB}"
echo "LOCAL_CHECKPOINT_MIN_FREE_GB=${LOCAL_CHECKPOINT_MIN_FREE_GB}"
echo "OPENCLAW_GATEWAY_IMAGE=${OPENCLAW_GATEWAY_IMAGE}"
echo "OPENCLAW_BUNDLED_PLUGINS_DIR=${OPENCLAW_BUNDLED_PLUGINS_DIR}"
echo "OPENCLAW_MAX_RESPONSE_TOKENS=${OPENCLAW_MAX_RESPONSE_TOKENS}"
echo "OPENCLAW_TRAJECTORY_MAX_TOKENS=${OPENCLAW_TRAJECTORY_MAX_TOKENS}"
echo "OPENCLAW_CONTEXT_WINDOW=${OPENCLAW_CONTEXT_WINDOW}"
echo "OPENCLAW_MAX_AGENT_STEPS=${OPENCLAW_MAX_AGENT_STEPS}"
echo "OPENCLAW_MAX_COMPACTION_EVENTS=${OPENCLAW_MAX_COMPACTION_EVENTS}"
echo "ENTROPY_COEF=${ENTROPY_COEF}"
echo "USE_TENSORBOARD=${USE_TENSORBOARD}"
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "OPENCLAW_DISABLE_CHECKPOINT_SAVE=${OPENCLAW_DISABLE_CHECKPOINT_SAVE}"
echo "SGLANG_ENABLE_CUDA_GRAPH=${SGLANG_ENABLE_CUDA_GRAPH}"
echo "EVAL_INTERVAL=${EVAL_INTERVAL_VALUE}"
echo "OPENCLAW_EVAL_NUM_WORKERS=${OPENCLAW_EVAL_NUM_WORKERS}"
echo "OPENCLAW_EVAL_N_SAMPLES=${OPENCLAW_EVAL_N_SAMPLES}"
echo "OPENCLAW_EVAL_TASKS=${OPENCLAW_EVAL_TASKS}"
if [ "${OPENCLAW_DISABLE_CHECKPOINT_SAVE}" = "0" ]; then
  echo "SAVE_INTERVAL=${SAVE_INTERVAL_VALUE}"
  if (( SAVE_RETAIN_INTERVAL_VALUE == 0 )); then
    echo "CHECKPOINT_RETENTION=all"
  else
    echo "SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL_VALUE}"
  fi
fi
echo "OPENCLAW_GATEWAY_PORT_BASE=${OPENCLAW_GATEWAY_PORT_BASE}"
echo "OPENCLAW_GATEWAY_LAST_PORT=${GATEWAY_LAST_PORT}"

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60
export MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-10000000}"
source "${SLIME_ROOT}/scripts/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --rotary-base "${ROTARY_BASE:-10000000}"
)
if [ "${OPENCLAW_DISABLE_CHECKPOINT_SAVE}" = "0" ]; then
  CKPT_ARGS+=(
    --save "${SAVE_CKPT}"
    --save-interval "${SAVE_INTERVAL_VALUE}"
  )
  if (( SAVE_RETAIN_INTERVAL_VALUE > 0 )); then
    CKPT_ARGS+=(--save-retain-interval "${SAVE_RETAIN_INTERVAL_VALUE}")
  fi
fi
if [ -n "${LOAD_CKPT:-}" ]; then
  CKPT_ARGS+=(--load "${LOAD_CKPT}" --use-checkpoint-opt-param-scheduler)
fi

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --rollout-function-path clawgym_rl_rollout.generate_rollout_clawgym_rl
  --num-rollout "${NUM_ROLLOUT:-100}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
  --rollout-max-response-len "${OPENCLAW_MAX_RESPONSE_TOKENS}"
  --rollout-max-context-len "${OPENCLAW_TRAJECTORY_MAX_TOKENS}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.7}"
  --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
  --rollout-top-k "${ROLLOUT_TOP_K:--1}"
  --reward-key score
  --num-steps-per-rollout 1
)
if [ -n "${OPENCLAW_LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
  ROLLOUT_ARGS+=(--load-debug-rollout-data "${OPENCLAW_LOAD_DEBUG_ROLLOUT_DATA}")
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --dynamic-history
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-dynamic-global-batch-size
  --calculate-per-token-loss
)

PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 4
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --use-slime-router
  --rollout-num-gpus-per-engine "${SGLANG_TP}"
  --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.65}"
  --sglang-context-length "${OPENCLAW_CONTEXT_WINDOW}"
  --sglang-ep-size "${SGLANG_EP_SIZE:-1}"
  --sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND:-triton}"
  --sglang-watchdog-timeout "${SGLANG_WATCHDOG_TIMEOUT:-1800}"
)
if [ "${SGLANG_ENABLE_CUDA_GRAPH}" = "0" ]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

CUSTOM_ARGS=(
  --custom-generate-function-path clawgym_rl_api_server.generate
  --custom-rm-path clawgym_rl_api_server.reward_func
)

EVAL_ARGS=(
  --eval-interval "${EVAL_INTERVAL_VALUE}"
  --eval-function-path clawgym_rl_rollout.generate_rollout_clawgym_rl
)
if [ -n "${EVAL_TEMPERATURE:-}" ]; then
  EVAL_ARGS+=(--eval-temperature "${EVAL_TEMPERATURE}")
fi
if [ -n "${EVAL_TOP_P:-}" ]; then
  EVAL_ARGS+=(--eval-top-p "${EVAL_TOP_P}")
fi
if [ -n "${EVAL_TOP_K:-}" ]; then
  EVAL_ARGS+=(--eval-top-k "${EVAL_TOP_K}")
fi
if [ -n "${EVAL_MAX_RESPONSE_LEN:-}" ]; then
  EVAL_ARGS+=(--eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}")
fi
if [ -n "${EVAL_MAX_CONTEXT_LEN:-}" ]; then
  EVAL_ARGS+=(--eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}")
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --no-gradient-accumulation-fusion
  --attention-softmax-in-fp32
  --attention-backend flash
  --moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"
)

if [ "${USE_TENSORBOARD}" = "1" ]; then
  TENSORBOARD_ARGS=(
    --use-tensorboard
    --tb-project-name "${TENSORBOARD_PROJECT:-clawgym_rl}"
    --tb-experiment-name "${EXP_NAME}"
  )
else
  TENSORBOARD_ARGS=()
fi

USE_WANDB="${USE_WANDB:-0}"
WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-clawgym_rl}"
    --wandb-group "${EXP_NAME}"
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

if "${RAY_CLI[@]}" status --address="${MASTER_ADDR}:6379" >/dev/null 2>&1; then
  echo "An existing Ray cluster is active; refusing to stop or reuse it." >&2
  exit 1
fi

TUNNEL_PID=""
STARTED_RAY=0
cleanup() {
  if [ -n "${TUNNEL_PID}" ] && kill -0 "${TUNNEL_PID}" 2>/dev/null; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
    wait "${TUNNEL_PID}" 2>/dev/null || true
  fi
  if [ "${STARTED_RAY}" = "1" ]; then
    "${RAY_CLI[@]}" stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

ssh -i "${OPENCLAW_REMOTE_SSH_KEY}" \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=4 \
  -N -R "127.0.0.1:${OPENCLAW_REMOTE_PROXY_PORT}:127.0.0.1:${PORT}" \
  "${OPENCLAW_REMOTE_USER}@${OPENCLAW_REMOTE_HOST}" &
TUNNEL_PID=$!
sleep 2
if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
  wait "${TUNNEL_PID}" || true
  echo "SSH reverse tunnel failed." >&2
  exit 1
fi

"${RAY_CLI[@]}" start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
STARTED_RAY=1

export PYTHONPATH="${MEGATRON_ROOT}:${SCRIPT_DIR}:${SLIME_ROOT}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
RUNTIME_ENV_JSON="$(python3 -c 'import json, os; keys = [
"PYTHONPATH", "PYTHON_ENV", "PATH", "CUDA_DEVICE_MAX_CONNECTIONS", "FLASHINFER_WORKSPACE_BASE",
"OPENCLAW_REMOTE_HOST", "OPENCLAW_REMOTE_USER", "OPENCLAW_REMOTE_SSH_KEY",
"OPENCLAW_REMOTE_BASE", "OPENCLAW_REMOTE_PROXY_PORT", "OPENCLAW_GATEWAY_IMAGE",
"OPENCLAW_BUNDLED_PLUGINS_DIR",
"OPENCLAW_GATEWAY_PORT_BASE", "OPENCLAW_GATEWAY_PORT_STRIDE",
"OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET", "OPENCLAW_GATEWAY_PORT_MAX_SLOTS",
"OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY", "OPENCLAW_CONTEXT_WINDOW",
"OPENCLAW_TRAJECTORY_MAX_TOKENS", "OPENCLAW_MAX_RESPONSE_TOKENS",
"OPENCLAW_MAX_AGENT_STEPS", "OPENCLAW_MAX_COMPACTION_EVENTS",
"OPENCLAW_TOKEN_LEDGER_MODE", "OPENCLAW_ASSEMBLE_OFFICIAL_SPLICE",
"OPENCLAW_MAX_TOOL_CALLS_PER_TURN",
"OPENCLAW_TRAIN_MAX_TURNS",
"OPENCLAW_EVAL_MAX_TURNS", "OPENCLAW_TRAIN_NUM_WORKERS", "OPENCLAW_EVAL_NUM_WORKERS",
"OPENCLAW_EVAL_N_SAMPLES",
"OPENCLAW_FORMAT_PENALTY", "OPENCLAW_CHAT_TURN_TIMEOUT", "OPENCLAW_TRAIN_TASKS",
"OPENCLAW_EVAL_TASKS", "OPENCLAW_EVAL_TASK_IDS", "OPENCLAW_EVAL_TASK_INDEX_MAP",
"OPENCLAW_EVAL_TASK_INDEX_OFFSET",
"CLAWGYM_SERPER_SKILL_DIR", "OPENCLAW_ENV_FILE", "OPENCLAW_RECORD_ENABLED",
"OPENCLAW_RECORD_FILE", "TENSORBOARD_DIR", "HOST", "PORT", "SGLANG_API_KEY", "SERVED_MODEL_NAME",
"OPENCLAW_TRAIN_SHUFFLE_SEED", "OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE",
"OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS", "SGLANG_ENABLE_DETERMINISTIC_INFERENCE",
"MEGATRON_DETERMINISTIC_MODE",
"TOOL_CALL_PARSER", "SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION"
]; print(json.dumps({"env_vars": {k: os.environ.get(k, "") for k in keys}}))')"

"${RAY_CLI[@]}" job submit --address="http://127.0.0.1:8265" \
  --working-dir "${SLIME_ROOT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_ROOT}/train_step39_eval_only_20260812_091700.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${TENSORBOARD_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${EVAL_ARGS[@]}"
