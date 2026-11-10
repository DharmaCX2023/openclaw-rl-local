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
SGLANG_SESSION_AFFINITY="${SGLANG_SESSION_AFFINITY:-1}"
SGLANG_AFFINITY_MAX_LOAD_DELTA="${SGLANG_AFFINITY_MAX_LOAD_DELTA:-2}"
# Balance the samples of each task across rollout engines once, then keep every
# multi-turn session pinned. Never spill an in-flight session for load balance.
SGLANG_AFFINITY_MODE="${SGLANG_AFFINITY_MODE:-task_sample_fixed}"
SGLANG_AFFINITY_SAMPLES_PER_TASK="${SGLANG_AFFINITY_SAMPLES_PER_TASK:-8}"
SGLANG_AFFINITY_MIGRATION_LOAD_DELTA="${SGLANG_AFFINITY_MIGRATION_LOAD_DELTA:-4}"
SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS="${SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS:-10}"

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS" >&2
  exit 1
fi
EXPECTED_ACTOR_GPUS="${EXPECTED_ACTOR_GPUS:-4}"
if (( ACTOR_GPUS != EXPECTED_ACTOR_GPUS )); then
  echo "This entry expects ACTOR_GPUS=${EXPECTED_ACTOR_GPUS}." >&2
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
export OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY="${OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY:-4}"
export OPENCLAW_GATEWAY_PORT_BASE="${OPENCLAW_GATEWAY_PORT_BASE:-16000}"
export OPENCLAW_GATEWAY_PORT_STRIDE="${OPENCLAW_GATEWAY_PORT_STRIDE:-10}"
export OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET="${OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET:-5000}"
export OPENCLAW_GATEWAY_PORT_MAX_SLOTS="${OPENCLAW_GATEWAY_PORT_MAX_SLOTS:-4000}"
export OPENCLAW_REMOTE_GATEWAY_PORT_MODE="${OPENCLAW_REMOTE_GATEWAY_PORT_MODE:-run_scoped}"
export OPENCLAW_REMOTE_GATEWAY_PORT_MIN="${OPENCLAW_REMOTE_GATEWAY_PORT_MIN:-20000}"
export OPENCLAW_REMOTE_GATEWAY_PORT_MAX="${OPENCLAW_REMOTE_GATEWAY_PORT_MAX:-62000}"
export OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS="${OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS:-512}"
export OPENCLAW_REMOTE_START_ATTEMPTS="${OPENCLAW_REMOTE_START_ATTEMPTS:-3}"

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
export OPENCLAW_TRAIN_SCHEDULING="${OPENCLAW_TRAIN_SCHEDULING:-sequential}"
export OPENCLAW_EVAL_NUM_WORKERS="${OPENCLAW_EVAL_NUM_WORKERS:-32}"
export OPENCLAW_EVAL_N_SAMPLES="${OPENCLAW_EVAL_N_SAMPLES:-1}"
export OPENCLAW_EVAL_SAVE_WORKSPACES="${OPENCLAW_EVAL_SAVE_WORKSPACES:-1}"
export OPENCLAW_EVAL_ARTIFACT_STRICT="${OPENCLAW_EVAL_ARTIFACT_STRICT:-1}"
export OPENCLAW_EVAL_WORKSPACE_MAX_BYTES="${OPENCLAW_EVAL_WORKSPACE_MAX_BYTES:-134217728}"
export OPENCLAW_EVAL_WORKSPACE_TIMEOUT="${OPENCLAW_EVAL_WORKSPACE_TIMEOUT:-600}"
if [ "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}" = "1" ]; then
  export OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE="${OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE:-20260826}"
else
  # Natural sampling: no fixed per-request seed base (empty value is
  # filtered out of the Ray runtime env, so the api server sees it unset).
  export OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE="${OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE:-}"
fi
export OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS="${OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS:-1}"
export OPENCLAW_FORMAT_PENALTY="${OPENCLAW_FORMAT_PENALTY:-0}"
export OPENCLAW_CHAT_TURN_TIMEOUT="${OPENCLAW_CHAT_TURN_TIMEOUT:-1200}"
export OPENCLAW_REWARD_SUBMIT_TIMEOUT_SECONDS="${OPENCLAW_REWARD_SUBMIT_TIMEOUT_SECONDS:-120}"
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
KL_LOSS_MONITOR_INTERVAL="${KL_LOSS_MONITOR_INTERVAL:-5}"
USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-1}"
USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-1}"
USE_TIS="${USE_TIS:-1}"
TIS_CLIP="${TIS_CLIP:-2.0}"
export OPENCLAW_OPSD_ENABLE="${OPENCLAW_OPSD_ENABLE:-0}"
export OPENCLAW_OPSD_MATCHED_PLACEBO="${OPENCLAW_OPSD_MATCHED_PLACEBO:-0}"
export OPENCLAW_SDAR_ENABLE="${OPENCLAW_SDAR_ENABLE:-0}"
OPENCLAW_OPSD_MODE="${OPENCLAW_OPSD_MODE:-absolute}"
OPENCLAW_OPSD_STEP_BETA="${OPENCLAW_OPSD_STEP_BETA:-1.0}"
OPENCLAW_OPSD_STEP_CLIP_LOWER="${OPENCLAW_OPSD_STEP_CLIP_LOWER:-0.0}"
OPENCLAW_OPSD_STEP_CLIP_UPPER="${OPENCLAW_OPSD_STEP_CLIP_UPPER:-1.2}"
OPENCLAW_OPSD_CENTERED_EPSILON="${OPENCLAW_OPSD_CENTERED_EPSILON:-1e-6}"
OPENCLAW_OPSD_CENTERED_MIN_STEPS="${OPENCLAW_OPSD_CENTERED_MIN_STEPS:-2}"
OPENCLAW_AGENTOPSD_DECAY_GAMMA="${OPENCLAW_AGENTOPSD_DECAY_GAMMA:-0.95}"
OPENCLAW_AGENTOPSD_RESHAPE_LAMBDA="${OPENCLAW_AGENTOPSD_RESHAPE_LAMBDA:-0.5}"
OPENCLAW_AGENTOPSD_MULTIPLIER_BAND="${OPENCLAW_AGENTOPSD_MULTIPLIER_BAND:-0.2}"
OPENCLAW_AGENTOPSD_EPSILON="${OPENCLAW_AGENTOPSD_EPSILON:-1e-6}"
OPENCLAW_OPSD_TEACHER_CONCURRENCY="${OPENCLAW_OPSD_TEACHER_CONCURRENCY:-8}"
OPENCLAW_SDAR_LOSS_COEF="${OPENCLAW_SDAR_LOSS_COEF:-0.01}"
OPENCLAW_SDAR_GATE_BETA="${OPENCLAW_SDAR_GATE_BETA:-5.0}"
export OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE="${OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE:-}"
export OPENCLAW_OPSD_PLACEBO_MAPPING_FILE="${OPENCLAW_OPSD_PLACEBO_MAPPING_FILE:-}"
export OPENCLAW_OPSD_PLACEBO_INDEX="${OPENCLAW_OPSD_PLACEBO_INDEX:-0}"
export OPENCLAW_OPSD_TEACHER_TIMEOUT="${OPENCLAW_OPSD_TEACHER_TIMEOUT:-600}"
TIS_CLIP_LOW="${TIS_CLIP_LOW:-0.0}"
OPENCLAW_DISABLE_CHECKPOINT_SAVE="${OPENCLAW_DISABLE_CHECKPOINT_SAVE:-0}"
SGLANG_ENABLE_CUDA_GRAPH="${SGLANG_ENABLE_CUDA_GRAPH:-1}"
SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}"
MEGATRON_DETERMINISTIC_MODE="${MEGATRON_DETERMINISTIC_MODE:-1}"
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE MEGATRON_DETERMINISTIC_MODE
SGLANG_ENABLE_NGRAM_SPECULATIVE="${SGLANG_ENABLE_NGRAM_SPECULATIVE:-0}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-81920}"
SAVE_INTERVAL_VALUE="${SAVE_INTERVAL:-20}"
SAVE_RETAIN_INTERVAL_VALUE="${SAVE_RETAIN_INTERVAL:-0}"
EVAL_INTERVAL_VALUE="${EVAL_INTERVAL:-20}"
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"
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
case "${OPENCLAW_TRAIN_SCHEDULING}" in
  sequential|group_interleaved) ;;
  *)
    echo "OPENCLAW_TRAIN_SCHEDULING must be sequential or group_interleaved." >&2
    exit 1
    ;;
esac
if ! [[ "${KL_LOSS_MONITOR_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KL_LOSS_MONITOR_INTERVAL must be a positive integer." >&2
  exit 1
fi
for BOOLEAN_FLAG in USE_ROUTING_REPLAY USE_ROLLOUT_ROUTING_REPLAY USE_TIS SKIP_EVAL_BEFORE_TRAIN OPENCLAW_OPSD_ENABLE OPENCLAW_OPSD_MATCHED_PLACEBO OPENCLAW_SDAR_ENABLE; do
  if ! [[ "${!BOOLEAN_FLAG}" =~ ^[01]$ ]]; then
    echo "${BOOLEAN_FLAG} must be 0 or 1." >&2
    exit 1
  fi
done
if [ "${OPENCLAW_OPSD_MATCHED_PLACEBO}" = "1" ] && [ "${OPENCLAW_OPSD_ENABLE}" != "1" ]; then
  echo "OPENCLAW_OPSD_MATCHED_PLACEBO requires OPENCLAW_OPSD_ENABLE=1." >&2
  exit 1
fi
if [ "${OPENCLAW_OPSD_MATCHED_PLACEBO}" = "1" ]; then
  if [ ! -f "${OPENCLAW_OPSD_PLACEBO_MAPPING_FILE}" ]; then
    echo "Matched-placebo mapping not found: ${OPENCLAW_OPSD_PLACEBO_MAPPING_FILE}" >&2
    exit 1
  fi
  if ! [[ "${OPENCLAW_OPSD_PLACEBO_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "OPENCLAW_OPSD_PLACEBO_INDEX must be a non-negative integer." >&2
    exit 1
  fi
fi
if [ "${OPENCLAW_OPSD_ENABLE}" = "1" ] && [ "${OPENCLAW_SDAR_ENABLE}" = "1" ]; then
  echo "OPENCLAW_OPSD_ENABLE and OPENCLAW_SDAR_ENABLE are mutually exclusive." >&2
  exit 1
fi
case "${OPENCLAW_OPSD_MODE}" in
  absolute|centered|agentopsd|bb_agentopsd) ;;
  *) echo "OPENCLAW_OPSD_MODE must be absolute, centered, agentopsd, or bb_agentopsd." >&2; exit 1 ;;
esac
if [ "${OPENCLAW_OPSD_ENABLE}" = "1" ] || [ "${OPENCLAW_SDAR_ENABLE}" = "1" ]; then
  if [ ! -f "${OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE}" ]; then
    echo "Evidence anchor JSON not found: ${OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE}" >&2
    exit 1
  fi
fi
TIS_FLOAT_RE='^([0-9]+([.][0-9]*)?|[.][0-9]+)$'
if ! [[ "${TIS_CLIP}" =~ ${TIS_FLOAT_RE} ]] || ! [[ "${TIS_CLIP_LOW}" =~ ${TIS_FLOAT_RE} ]]; then
  echo "TIS_CLIP and TIS_CLIP_LOW must be non-negative decimal numbers." >&2
  exit 1
fi
if ! awk -v low="${TIS_CLIP_LOW}" -v high="${TIS_CLIP}" 'BEGIN { exit !(high > 0 && low <= high) }'; then
  echo "TIS bounds require TIS_CLIP > 0 and TIS_CLIP_LOW <= TIS_CLIP." >&2
  exit 1
fi
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
if ! [[ "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE}" =~ ^[01]$ ]]; then
  echo "SGLANG_ENABLE_DETERMINISTIC_INFERENCE must be 0 or 1." >&2
  exit 1
fi
if ! [[ "${MEGATRON_DETERMINISTIC_MODE}" =~ ^[01]$ ]]; then
  echo "MEGATRON_DETERMINISTIC_MODE must be 0 or 1." >&2
  exit 1
fi
if [ "${MEGATRON_DETERMINISTIC_MODE}" = "1" ]; then
  export NCCL_ALGO="${NCCL_ALGO:-Ring}"
  export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi
if ! [[ "${SGLANG_ENABLE_NGRAM_SPECULATIVE}" =~ ^[01]$ ]]; then
  echo "SGLANG_ENABLE_NGRAM_SPECULATIVE must be 0 or 1." >&2
  exit 1
fi
if ! [[ "${MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || (( MAX_TOKENS_PER_GPU > 100000 )); then
  echo "MAX_TOKENS_PER_GPU must be an integer in [1, 100000]." >&2
  exit 1
fi
if ! [[ "${SGLANG_SESSION_AFFINITY}" =~ ^[01]$ ]]; then
  echo "SGLANG_SESSION_AFFINITY must be 0 or 1." >&2
  exit 1
fi
if ! [[ "${SGLANG_AFFINITY_MAX_LOAD_DELTA}" =~ ^[0-9]+$ ]]; then
  echo "SGLANG_AFFINITY_MAX_LOAD_DELTA must be a non-negative integer." >&2
  exit 1
fi
case "${SGLANG_AFFINITY_MODE}" in
  hash_bounded|task_sample_fixed|task_sample_sustained) ;;
  *) echo "Invalid SGLANG_AFFINITY_MODE: ${SGLANG_AFFINITY_MODE}" >&2; exit 1 ;;
esac
if ! [[ "${SGLANG_AFFINITY_SAMPLES_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SGLANG_AFFINITY_SAMPLES_PER_TASK must be a positive integer." >&2
  exit 1
fi
if ! [[ "${SGLANG_AFFINITY_MIGRATION_LOAD_DELTA}" =~ ^[0-9]+$ ]]; then
  echo "SGLANG_AFFINITY_MIGRATION_LOAD_DELTA must be a non-negative integer." >&2
  exit 1
fi
if ! [[ "${SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS must be non-negative." >&2
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
  echo "OPENCLAW_TRAIN_SCHEDULING=${OPENCLAW_TRAIN_SCHEDULING}"
  echo "ENTROPY_COEF=${ENTROPY_COEF}"
  echo "KL_LOSS_MONITOR_INTERVAL=${KL_LOSS_MONITOR_INTERVAL}"
  echo "USE_ROUTING_REPLAY=${USE_ROUTING_REPLAY}"
  echo "USE_ROLLOUT_ROUTING_REPLAY=${USE_ROLLOUT_ROUTING_REPLAY}"
  echo "USE_TIS=${USE_TIS}"
  echo "TIS_CLIP=${TIS_CLIP}"
  echo "TIS_CLIP_LOW=${TIS_CLIP_LOW}"
  echo "USE_TENSORBOARD=${USE_TENSORBOARD}"
  echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
  echo "OPENCLAW_DISABLE_CHECKPOINT_SAVE=${OPENCLAW_DISABLE_CHECKPOINT_SAVE}"
  echo "SGLANG_ENABLE_CUDA_GRAPH=${SGLANG_ENABLE_CUDA_GRAPH}"
  echo "SGLANG_ENABLE_DETERMINISTIC_INFERENCE=${SGLANG_ENABLE_DETERMINISTIC_INFERENCE}"
  echo "MEGATRON_DETERMINISTIC_MODE=${MEGATRON_DETERMINISTIC_MODE}"
  if [ "${MEGATRON_DETERMINISTIC_MODE}" = "1" ]; then
    echo "NCCL_ALGO=${NCCL_ALGO}"
    echo "NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO}"
    echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
  fi
  echo "SGLANG_ENABLE_NGRAM_SPECULATIVE=${SGLANG_ENABLE_NGRAM_SPECULATIVE}"
  echo "MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU}"
  echo "SGLANG_SESSION_AFFINITY=${SGLANG_SESSION_AFFINITY}"
  echo "SGLANG_AFFINITY_MAX_LOAD_DELTA=${SGLANG_AFFINITY_MAX_LOAD_DELTA}"
  echo "SGLANG_AFFINITY_MODE=${SGLANG_AFFINITY_MODE}"
  echo "SGLANG_AFFINITY_SAMPLES_PER_TASK=${SGLANG_AFFINITY_SAMPLES_PER_TASK}"
  echo "SGLANG_AFFINITY_MIGRATION_LOAD_DELTA=${SGLANG_AFFINITY_MIGRATION_LOAD_DELTA}"
  echo "SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS=${SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS}"
  echo "EVAL_INTERVAL=${EVAL_INTERVAL_VALUE}"
  echo "SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN}"
  echo "OPENCLAW_EVAL_NUM_WORKERS=${OPENCLAW_EVAL_NUM_WORKERS}"
  echo "OPENCLAW_EVAL_N_SAMPLES=${OPENCLAW_EVAL_N_SAMPLES}"
  echo "OPENCLAW_EVAL_SAVE_WORKSPACES=${OPENCLAW_EVAL_SAVE_WORKSPACES}"
  echo "OPENCLAW_EVAL_ARTIFACT_STRICT=${OPENCLAW_EVAL_ARTIFACT_STRICT}"
  echo "OPENCLAW_EVAL_WORKSPACE_MAX_BYTES=${OPENCLAW_EVAL_WORKSPACE_MAX_BYTES}"
  echo "OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE=${OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE}"
  echo "OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=${OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS}"
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
echo "OPENCLAW_TRAIN_SCHEDULING=${OPENCLAW_TRAIN_SCHEDULING}"
echo "ENTROPY_COEF=${ENTROPY_COEF}"
echo "KL_LOSS_MONITOR_INTERVAL=${KL_LOSS_MONITOR_INTERVAL}"
echo "USE_ROUTING_REPLAY=${USE_ROUTING_REPLAY}"
echo "USE_ROLLOUT_ROUTING_REPLAY=${USE_ROLLOUT_ROUTING_REPLAY}"
echo "USE_TIS=${USE_TIS}"
echo "TIS_CLIP=${TIS_CLIP}"
echo "TIS_CLIP_LOW=${TIS_CLIP_LOW}"
echo "USE_TENSORBOARD=${USE_TENSORBOARD}"
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "OPENCLAW_DISABLE_CHECKPOINT_SAVE=${OPENCLAW_DISABLE_CHECKPOINT_SAVE}"
echo "SGLANG_ENABLE_CUDA_GRAPH=${SGLANG_ENABLE_CUDA_GRAPH}"
echo "SGLANG_ENABLE_DETERMINISTIC_INFERENCE=${SGLANG_ENABLE_DETERMINISTIC_INFERENCE}"
echo "MEGATRON_DETERMINISTIC_MODE=${MEGATRON_DETERMINISTIC_MODE}"
if [ "${MEGATRON_DETERMINISTIC_MODE}" = "1" ]; then
  echo "NCCL_ALGO=${NCCL_ALGO}"
  echo "NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO}"
  echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
fi
echo "SGLANG_ENABLE_NGRAM_SPECULATIVE=${SGLANG_ENABLE_NGRAM_SPECULATIVE}"
echo "MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU}"
echo "SGLANG_SESSION_AFFINITY=${SGLANG_SESSION_AFFINITY}"
echo "SGLANG_AFFINITY_MAX_LOAD_DELTA=${SGLANG_AFFINITY_MAX_LOAD_DELTA}"
echo "SGLANG_AFFINITY_MODE=${SGLANG_AFFINITY_MODE}"
echo "SGLANG_AFFINITY_SAMPLES_PER_TASK=${SGLANG_AFFINITY_SAMPLES_PER_TASK}"
echo "SGLANG_AFFINITY_MIGRATION_LOAD_DELTA=${SGLANG_AFFINITY_MIGRATION_LOAD_DELTA}"
echo "SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS=${SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS}"
echo "EVAL_INTERVAL=${EVAL_INTERVAL_VALUE}"
echo "SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN}"
echo "OPENCLAW_EVAL_NUM_WORKERS=${OPENCLAW_EVAL_NUM_WORKERS}"
echo "OPENCLAW_EVAL_N_SAMPLES=${OPENCLAW_EVAL_N_SAMPLES}"
echo "OPENCLAW_EVAL_SAVE_WORKSPACES=${OPENCLAW_EVAL_SAVE_WORKSPACES}"
echo "OPENCLAW_EVAL_ARTIFACT_STRICT=${OPENCLAW_EVAL_ARTIFACT_STRICT}"
echo "OPENCLAW_EVAL_WORKSPACE_MAX_BYTES=${OPENCLAW_EVAL_WORKSPACE_MAX_BYTES}"
echo "OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE=${OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE}"
echo "OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=${OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS}"
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
MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-${SLIME_ROOT}/scripts/models/qwen3-30B-A3B.sh}"
source "${MODEL_CONFIG_SCRIPT}"
if [ "${DISABLE_MOE_FUSED_KERNELS:-0}" = "1" ] || \
   [ "${DISABLE_MOE_GROUPED_GEMM:-0}" = "1" ] || \
   [ "${DISABLE_MOE_PERMUTE_FUSION:-0}" = "1" ]; then
  FILTERED_MODEL_ARGS=()
  for MODEL_ARG in "${MODEL_ARGS[@]}"; do
    case "${MODEL_ARG}" in
      --moe-grouped-gemm)
        if [ "${DISABLE_MOE_FUSED_KERNELS:-0}" != "1" ] && \
           [ "${DISABLE_MOE_GROUPED_GEMM:-0}" != "1" ]; then
          FILTERED_MODEL_ARGS+=("${MODEL_ARG}")
        fi
        ;;
      --moe-permute-fusion)
        if [ "${DISABLE_MOE_FUSED_KERNELS:-0}" != "1" ] && \
           [ "${DISABLE_MOE_PERMUTE_FUSION:-0}" != "1" ]; then
          FILTERED_MODEL_ARGS+=("${MODEL_ARG}")
        fi
        ;;
      *) FILTERED_MODEL_ARGS+=("${MODEL_ARG}") ;;
    esac
  done
  MODEL_ARGS=("${FILTERED_MODEL_ARGS[@]}")
fi

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
  CKPT_ARGS+=(--load "${LOAD_CKPT}")
  if [ "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}" = "1" ] && [ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
    echo "USE_CHECKPOINT_OPT_PARAM_SCHEDULER and OVERRIDE_OPT_PARAM_SCHEDULER cannot both be set." >&2
    exit 2
  fi
  if [ "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
    CKPT_ARGS+=(--use-checkpoint-opt-param-scheduler)
  fi
  if [ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
    CKPT_ARGS+=(--override-opt-param-scheduler)
  fi
fi
if [ "${NO_LOAD_OPTIM:-0}" = "1" ]; then
  CKPT_ARGS+=(--no-load-optim)
fi
if [ "${NO_LOAD_RNG:-0}" = "1" ]; then
  CKPT_ARGS+=(--no-load-rng)
fi

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --rollout-function-path "${OPENCLAW_ROLLOUT_FN:-span_trace_rollout.generate_rollout_clawgym_rl}"
  --num-rollout "${NUM_ROLLOUT:-100}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
  --rollout-max-response-len "${OPENCLAW_MAX_RESPONSE_TOKENS}"
  --rollout-max-context-len "${OPENCLAW_TRAJECTORY_MAX_TOKENS}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.7}"
  --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
  --rollout-top-k "${ROLLOUT_TOP_K:-20}"
  --reward-key score
  --num-steps-per-rollout 1
)
if [ -n "${START_ROLLOUT_ID:-}" ]; then
  ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi
if [ -n "${OPENCLAW_LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
  ROLLOUT_ARGS+=(--load-debug-rollout-data "${OPENCLAW_LOAD_DEBUG_ROLLOUT_DATA}")
fi
if [ -n "${OPENCLAW_SAVE_DEBUG_ROLLOUT_DATA:-}" ]; then
  ROLLOUT_ARGS+=(--save-debug-rollout-data "${OPENCLAW_SAVE_DEBUG_ROLLOUT_DATA}")
fi
if [ -n "${OPENCLAW_SAVE_DEBUG_TRAIN_DATA:-}" ]; then
  ROLLOUT_ARGS+=(--save-debug-train-data "${OPENCLAW_SAVE_DEBUG_TRAIN_DATA}")
fi
if [ -n "${OPENCLAW_SAVE_OPSD_STEP_SUMMARY:-}" ]; then
  ROLLOUT_ARGS+=(--save-opsd-step-summary "${OPENCLAW_SAVE_OPSD_STEP_SUMMARY}")
fi
if [ "${DEBUG_ROLLOUT_ONLY:-0}" = "1" ]; then
  ROLLOUT_ARGS+=(--debug-rollout-only)
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --dynamic-history
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-monitor-interval "${KL_LOSS_MONITOR_INTERVAL}"
  --kl-loss-type low_var_kl
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-dynamic-global-batch-size
)
if [ "${CALCULATE_PER_TOKEN_LOSS:-0}" = "1" ]; then
  GRPO_ARGS+=(--calculate-per-token-loss)
else
  GRPO_ARGS+=(--calculate-per-rollout-loss)
fi
if [ "${USE_ROUTING_REPLAY}" = "1" ]; then
  GRPO_ARGS+=(--use-routing-replay)
fi
if [ "${USE_ROLLOUT_ROUTING_REPLAY}" = "1" ]; then
  GRPO_ARGS+=(--use-rollout-routing-replay)
fi
if [ "${OPENCLAW_OPSD_ENABLE}" = "1" ]; then
  GRPO_ARGS+=(
    --opsd-enable
    --opsd-mode "${OPENCLAW_OPSD_MODE}"
    --opsd-step-beta "${OPENCLAW_OPSD_STEP_BETA}"
    --opsd-step-clip-lower "${OPENCLAW_OPSD_STEP_CLIP_LOWER}"
    --opsd-step-clip-upper "${OPENCLAW_OPSD_STEP_CLIP_UPPER}"
    --opsd-centered-epsilon "${OPENCLAW_OPSD_CENTERED_EPSILON}"
    --opsd-centered-min-steps "${OPENCLAW_OPSD_CENTERED_MIN_STEPS}"
    --agentopsd-decay-gamma "${OPENCLAW_AGENTOPSD_DECAY_GAMMA}"
    --agentopsd-reshape-lambda "${OPENCLAW_AGENTOPSD_RESHAPE_LAMBDA}"
    --agentopsd-multiplier-band "${OPENCLAW_AGENTOPSD_MULTIPLIER_BAND}"
    --agentopsd-epsilon "${OPENCLAW_AGENTOPSD_EPSILON}"
    --opsd-teacher-concurrency "${OPENCLAW_OPSD_TEACHER_CONCURRENCY}"
  )
  if [ "${OPENCLAW_OPSD_MATCHED_PLACEBO}" = "1" ]; then
    GRPO_ARGS+=(--opsd-matched-placebo)
  fi
fi
if [ "${OPENCLAW_SDAR_ENABLE}" = "1" ]; then
  GRPO_ARGS+=(
    --sdar-enable
    --sdar-loss-coef "${OPENCLAW_SDAR_LOSS_COEF}"
    --sdar-gate-beta "${OPENCLAW_SDAR_GATE_BETA}"
    --opsd-teacher-concurrency "${OPENCLAW_OPSD_TEACHER_CONCURRENCY}"
  )
fi

if [ "${USE_TIS}" = "1" ]; then
  GRPO_ARGS+=(--use-tis --tis-clip "${TIS_CLIP}" --tis-clip-low "${TIS_CLIP_LOW}")
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-4}"
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-4}"
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}"
)
if [ "${ENABLE_SEQUENCE_PARALLEL:-1}" = "1" ]; then
  PERF_ARGS+=(--sequence-parallel)
fi
if [ "${DISABLE_ACTIVATION_RECOMPUTE:-0}" != "1" ]; then
  PERF_ARGS+=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
  )
fi

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
if [ -n "${SGLANG_ROUTER_PORT:-}" ]; then
  SGLANG_ARGS+=(--sglang-router-port "${SGLANG_ROUTER_PORT}")
fi
if [ "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE}" = "1" ]; then
  SGLANG_ARGS+=(--sglang-enable-deterministic-inference)
fi
if [ "${SGLANG_ENABLE_NGRAM_SPECULATIVE}" = "1" ]; then
  SGLANG_ARGS+=(--sglang-speculative-algorithm NGRAM)
fi
if [ "${SGLANG_SESSION_AFFINITY}" = "1" ]; then
  SGLANG_ARGS+=(
    --slime-router-session-affinity
    --slime-router-affinity-max-load-delta "${SGLANG_AFFINITY_MAX_LOAD_DELTA}"
    --slime-router-affinity-mode "${SGLANG_AFFINITY_MODE}"
    --slime-router-affinity-samples-per-task "${SGLANG_AFFINITY_SAMPLES_PER_TASK}"
    --slime-router-affinity-migration-load-delta "${SGLANG_AFFINITY_MIGRATION_LOAD_DELTA}"
    --slime-router-affinity-migration-sustained-seconds "${SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS}"
  )
fi
if [ "${SGLANG_ENABLE_CUDA_GRAPH}" = "0" ]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

CUSTOM_ARGS=(
  --custom-generate-function-path "${OPENCLAW_GENERATE_FN:-span_trace_api.generate}"
  --custom-rm-path "${OPENCLAW_RM_FN:-span_trace_api.reward_func}"
)

EVAL_ARGS=(
  --eval-interval "${EVAL_INTERVAL_VALUE}"
  --eval-function-path "${OPENCLAW_ROLLOUT_FN:-span_trace_rollout.generate_rollout_clawgym_rl}"
  --eval-temperature "${EVAL_TEMPERATURE:-0.7}"
  --eval-top-p "${EVAL_TOP_P:-1.0}"
  --eval-top-k "${EVAL_TOP_K:--1}"
)
if [ "${SKIP_EVAL_BEFORE_TRAIN}" = "1" ]; then
  EVAL_ARGS+=(--skip-eval-before-train)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"
)
if [ "${NO_GRADIENT_ACCUMULATION_FUSION:-1}" = "1" ]; then
  MISC_ARGS+=(--no-gradient-accumulation-fusion)
fi
if [ "${MEGATRON_DETERMINISTIC_MODE}" = "1" ]; then
  MISC_ARGS+=(--deterministic-mode)
fi

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

# Each run owns a scoped Ray head. Never call global `ray stop` here: it would
# terminate a second experiment sharing this host.
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-10001}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-10002}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-19999}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-${EXP_NAME}}"
RAY_HEAD_LOG="${RAY_HEAD_LOG:-${SAVE_CKPT}/logs/ray-head.log}"
RAY_RESOURCE_ARGS=()
if [[ -n "${RAY_NUM_CPUS:-}" ]]; then
  if ! [[ "$RAY_NUM_CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "RAY_NUM_CPUS must be a positive integer: $RAY_NUM_CPUS" >&2
    exit 1
  fi
  RAY_RESOURCE_ARGS+=(--num-cpus "$RAY_NUM_CPUS")
fi
RAY_FIXED_PORT_ARGS=()
RAY_FIXED_PORT_VALUES=()
add_ray_fixed_port() {
  local flag="$1" variable="$2" value="${!2:-}"
  [[ -n "$value" ]] || return 0
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]] || (( value > 65535 )); then
    echo "$variable must be an integer in [1, 65535]: $value" >&2
    exit 1
  fi
  RAY_FIXED_PORT_ARGS+=("$flag" "$value")
  RAY_FIXED_PORT_VALUES+=("$value")
}
add_ray_fixed_port --object-manager-port RAY_OBJECT_MANAGER_PORT
add_ray_fixed_port --node-manager-port RAY_NODE_MANAGER_PORT
add_ray_fixed_port --dashboard-agent-listen-port RAY_DASHBOARD_AGENT_LISTEN_PORT
add_ray_fixed_port --dashboard-agent-grpc-port RAY_DASHBOARD_AGENT_GRPC_PORT
add_ray_fixed_port --runtime-env-agent-port RAY_RUNTIME_ENV_AGENT_PORT
add_ray_fixed_port --metrics-export-port RAY_METRICS_EXPORT_PORT
for RAY_PORT_VALUE in RAY_GCS_PORT RAY_DASHBOARD_PORT RAY_CLIENT_PORT RAY_MIN_WORKER_PORT RAY_MAX_WORKER_PORT; do
  if ! [[ "${!RAY_PORT_VALUE}" =~ ^[1-9][0-9]*$ ]] || (( ${!RAY_PORT_VALUE} > 65535 )); then
    echo "${RAY_PORT_VALUE} must be an integer in [1, 65535]: ${!RAY_PORT_VALUE}" >&2
    exit 1
  fi
done
if (( RAY_MIN_WORKER_PORT > RAY_MAX_WORKER_PORT )); then
  echo "RAY_MIN_WORKER_PORT must not exceed RAY_MAX_WORKER_PORT" >&2
  exit 1
fi
if timeout 5s "${RAY_CLI[@]}" status --address="${MASTER_ADDR}:${RAY_GCS_PORT}" >/dev/null 2>&1; then
  echo "Ray address already active: ${MASTER_ADDR}:${RAY_GCS_PORT}" >&2
  exit 1
fi
if "${PYTHON_ENV}/bin/python" -c 'import socket,sys; host=sys.argv[1]; ports=map(int,sys.argv[2:]); sys.exit(0 if any(socket.socket().connect_ex((host,p)) == 0 for p in ports) else 1)' \
  "${MASTER_ADDR}" "${RAY_GCS_PORT}" "${RAY_DASHBOARD_PORT}" "${RAY_CLIENT_PORT}" "${RAY_FIXED_PORT_VALUES[@]}"; then
  echo "One or more requested Ray ports are already occupied." >&2
  exit 1
fi
mkdir -p "${RAY_TEMP_DIR}" "$(dirname -- "${RAY_HEAD_LOG}")"

TUNNEL_PID=""
STARTED_RAY=0
RAY_HEAD_PID=""
cleanup() {
  if [ -n "${TUNNEL_PID}" ] && kill -0 "${TUNNEL_PID}" 2>/dev/null; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
    wait "${TUNNEL_PID}" 2>/dev/null || true
  fi
  if [ "${STARTED_RAY}" = "1" ]; then
    if [ -n "${RAY_HEAD_PID}" ] && kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
      kill -TERM -- "-${RAY_HEAD_PID}" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "${RAY_HEAD_PID}" 2>/dev/null || break
        sleep 1
      done
      kill -KILL -- "-${RAY_HEAD_PID}" 2>/dev/null || true
      wait "${RAY_HEAD_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

if [ -z "${OPENCLAW_LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
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
else
  echo "OPENCLAW_LOAD_DEBUG_ROLLOUT_DATA is set; skipping unused remote SSH tunnel."
fi

setsid "${RAY_CLI[@]}" start --head --block \
  --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  "${RAY_RESOURCE_ARGS[@]}" \
  --port "${RAY_GCS_PORT}" --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}" \
  --ray-client-server-port "${RAY_CLIENT_PORT}" \
  --min-worker-port "${RAY_MIN_WORKER_PORT}" --max-worker-port "${RAY_MAX_WORKER_PORT}" \
  "${RAY_FIXED_PORT_ARGS[@]}" \
  --temp-dir "${RAY_TEMP_DIR}" --disable-usage-stats >"${RAY_HEAD_LOG}" 2>&1 &
RAY_HEAD_PID=$!
STARTED_RAY=1
for _ in $(seq 1 120); do
  if timeout 10s "${RAY_CLI[@]}" status --address="${MASTER_ADDR}:${RAY_GCS_PORT}" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
    echo "Scoped Ray head exited during startup; see ${RAY_HEAD_LOG}" >&2
    exit 1
  fi
  sleep 1
done
if ! timeout 30s "${RAY_CLI[@]}" status --address="${MASTER_ADDR}:${RAY_GCS_PORT}" >/dev/null 2>&1; then
  echo "Scoped Ray head did not become ready; see ${RAY_HEAD_LOG}" >&2
  exit 1
fi

# `ray status` only proves that GCS/raylet are reachable. The dashboard's
# JobAgent starts independently and can still return HTTP 500 for several
# seconds. Wait for the actual job endpoint before submitting either arm.
RAY_JOB_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
RAY_JOB_AGENT_READY=0
for _ in $(seq 1 120); do
  if timeout 5s "${PYTHON_ENV}/bin/python" "${REPO_ROOT}/scripts/span_trace/ray_job_agent_ready.py" "${MASTER_ADDR}:${RAY_GCS_PORT}" >/dev/null 2>&1; then
    RAY_JOB_AGENT_READY=1
    break
  fi
  if ! kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
    echo "Scoped Ray head exited while waiting for JobAgent; see ${RAY_HEAD_LOG}" >&2
    exit 1
  fi
  sleep 1
done
if (( RAY_JOB_AGENT_READY == 0 )); then
  echo "Scoped Ray JobAgent did not become ready at ${RAY_JOB_ADDRESS}; see ${RAY_HEAD_LOG}" >&2
  exit 1
fi

export PYTHONPATH="${MEGATRON_ROOT}:${SCRIPT_DIR}:${SLIME_ROOT}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
RUNTIME_ENV_JSON="$(python3 -c 'import json, os; keys = [
"PYTHONPATH", "PYTHON_ENV", "PATH", "CUDA_DEVICE_MAX_CONNECTIONS", "FLASHINFER_WORKSPACE_BASE",
"SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "MEGATRON_DETERMINISTIC_MODE",
"NCCL_ALGO", "NCCL_PROTO", "NVTE_ALLOW_NONDETERMINISTIC_ALGO", "CUBLAS_WORKSPACE_CONFIG",
"PYTORCH_CUDA_ALLOC_CONF",
"OPENCLAW_ARTIFACT_GRPO_STRICT", "OPENCLAW_CONTAINER_PREFIX", "OPENCLAW_TRAIN_EXCLUDE_TASK_IDS", "OPENCLAW_TRAIN_ATTEMPT_BASE",
"OPENCLAW_ARTIFACT_GRPO", "OPENCLAW_ARTIFACT_SCORE_MANIFEST", "OPENCLAW_ARTIFACT_GRPO_STRENGTH",
"OPENCLAW_ARTIFACT_GRPO_DYNAMIC_MANIFEST", "OPENCLAW_ARTIFACT_GRPO_FAIL_CLOSED",
"OPENCLAW_SPAN_TRACE_ROOT", "OPENCLAW_REMOTE_HOST", "OPENCLAW_REMOTE_USER", "OPENCLAW_REMOTE_SSH_KEY",
"OPENCLAW_REMOTE_BASE", "OPENCLAW_REMOTE_PROXY_PORT", "OPENCLAW_GATEWAY_IMAGE",
"OPENCLAW_BUNDLED_PLUGINS_DIR",
"OPENCLAW_GATEWAY_PORT_BASE", "OPENCLAW_GATEWAY_PORT_STRIDE",
"OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET", "OPENCLAW_GATEWAY_PORT_MAX_SLOTS",
"OPENCLAW_REMOTE_GATEWAY_PORT_MODE", "OPENCLAW_REMOTE_GATEWAY_PORT_MIN",
"OPENCLAW_REMOTE_GATEWAY_PORT_MAX", "OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS",
"OPENCLAW_REMOTE_START_ATTEMPTS",
"OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY", "OPENCLAW_GATEWAY_PREWARM_COUNT",
"OPENCLAW_GATEWAY_PREWARM_TIMEOUT_SECONDS", "OPENCLAW_CONTEXT_WINDOW",
"OPENCLAW_TRAJECTORY_MAX_TOKENS", "OPENCLAW_MAX_RESPONSE_TOKENS",
"OPENCLAW_MAX_AGENT_STEPS", "OPENCLAW_MAX_COMPACTION_EVENTS",
"OPENCLAW_TOKEN_LEDGER_MODE",
"OPENCLAW_ASSEMBLE_OFFICIAL_SPLICE", "OPENCLAW_GATEWAY_CONFIG",
"OPENCLAW_MAX_TOOL_CALLS_PER_TURN",
"OPENCLAW_TRAIN_MAX_TURNS", "OPENCLAW_TRAIN_SCHEDULING",
"OPENCLAW_EVAL_MAX_TURNS", "OPENCLAW_TRAIN_NUM_WORKERS", "OPENCLAW_EVAL_NUM_WORKERS",
"OPENCLAW_EVAL_N_SAMPLES", "OPENCLAW_EVAL_SAVE_WORKSPACES",
"OPENCLAW_EVAL_ARTIFACT_STRICT", "OPENCLAW_EVAL_WORKSPACE_MAX_BYTES",
"OPENCLAW_EVAL_WORKSPACE_TIMEOUT", "OPENCLAW_EVAL_WORKSPACE_CAPTURE_ATTEMPTS",
"OPENCLAW_EVAL_WORKSPACE_RETRY_DELAY_SECONDS", "OPENCLAW_EVAL_ARTIFACT_DIR",
"OPENCLAW_EVAL_SEMANTIC_STEP",
"OPENCLAW_FORMAT_PENALTY", "OPENCLAW_CHAT_TURN_TIMEOUT",
"OPENCLAW_REWARD_SUBMIT_TIMEOUT_SECONDS", "OPENCLAW_TRAIN_TASKS",
"OPENCLAW_EVAL_TASKS", "CLAWGYM_SERPER_SKILL_DIR", "OPENCLAW_ENV_FILE", "OPENCLAW_RECORD_ENABLED",
"OPENCLAW_EVAL_TASK_IDS", "OPENCLAW_EVAL_TASK_INDEX_OFFSET",
"OPENCLAW_EVAL_TASK_INDEX_MAP",
"OPENCLAW_RECORD_FILE", "TENSORBOARD_DIR", "HOST", "PORT", "SGLANG_API_KEY", "SERVED_MODEL_NAME",
"OPENCLAW_TRAIN_SHUFFLE_SEED", "OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE",
"OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS",
"OPENCLAW_SERPER_FAIL_CLOSED", "OPENCLAW_SERPER_FAIL_CLOSED_ATTEMPTS",
"OPENCLAW_OPSD_ENABLE", "OPENCLAW_OPSD_MATCHED_PLACEBO", "OPENCLAW_SDAR_ENABLE",
"OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE", "OPENCLAW_OPSD_PLACEBO_MAPPING_FILE",
"OPENCLAW_OPSD_PLACEBO_INDEX", "OPENCLAW_OPSD_TEACHER_TIMEOUT",
"OPENCLAW_GUIDANCE_REPEAT_SCORE",
"OPENCLAW_ROUTING_CONFLICT_DIAGNOSTICS",
"TOOL_CALL_PARSER", "SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION",
"SLIME_NUMERICAL_DIAG_CONFIG", "SLIME_ROUTING_REPLAY_FIXED_PROBS",
"SLIME_SGLANG_ENGINE_PORT_BASE", "SLIME_PRM_ENGINE_PORT_BASE"
]; print(json.dumps({"env_vars": {k: v for k in keys if (v := os.environ.get(k)) not in (None, "")}}))')"

TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${SLIME_ROOT}/train_async.py}"

"${RAY_CLI[@]}" job submit --address="${RAY_JOB_ADDRESS}" \
  --working-dir "${SLIME_ROOT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${TRAIN_ENTRYPOINT}" \
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
