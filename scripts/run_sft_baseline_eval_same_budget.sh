#!/bin/bash

set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
SLIME_ROOT="${REPO_ROOT}/slime"
MEGATRON_ROOT="${REPO_ROOT}/Megatron-LM"
OFFICIAL_ROOT="${REPO_ROOT}/clawgym-official"
PYTHON_ENV="${REPO_ROOT}/.venv-openclaw"
HF_CKPT="${HF_CKPT:-/mnt/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
EXP_NAME="${EXP_NAME:-clawgym-sft-baseline-eval-same-budget-${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${EXP_NAME}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
NUM_GPUS=8
ROLLOUT_GPUS=4
SGLANG_TP=2

if [ ! -x "${PYTHON_ENV}/bin/python" ] || [ ! -f "${PYTHON_ENV}/bin/ray" ]; then
  echo "Python/Ray environment is incomplete: ${PYTHON_ENV}" >&2
  exit 1
fi
if [ ! -d "${HF_CKPT}" ]; then
  echo "SFT checkpoint not found: ${HF_CKPT}" >&2
  exit 1
fi
if [ -e "${OUTPUT_DIR}" ] && [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "OUTPUT_DIR must be new or empty: ${OUTPUT_DIR}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHON_ENV
export PATH="${PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${MEGATRON_ROOT}:${OFFICIAL_ROOT}:${SLIME_ROOT}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

# Keep this contract identical to the formal RL intermediate evaluations.
export OPENCLAW_REMOTE_HOST="${OPENCLAW_REMOTE_HOST:-159.138.50.147}"
export OPENCLAW_REMOTE_USER="${OPENCLAW_REMOTE_USER:-root}"
export OPENCLAW_REMOTE_SSH_KEY="${OPENCLAW_REMOTE_SSH_KEY:-/root/openclaw_cpu_key}"
export OPENCLAW_REMOTE_BASE="${OPENCLAW_REMOTE_BASE:-/root/clawgym_rl_runs}"
export OPENCLAW_REMOTE_PROXY_PORT="${OPENCLAW_REMOTE_PROXY_PORT:-39081}"
export OPENCLAW_GATEWAY_IMAGE="${OPENCLAW_GATEWAY_IMAGE:-clawgym-rl@sha256:33a30ab107cb7fef6a84f17b3fb70a0e849654e9201c61dc9f20baeac21385a3}"
export OPENCLAW_BUNDLED_PLUGINS_DIR="${OPENCLAW_BUNDLED_PLUGINS_DIR:-/app/dist/extensions}"
export OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=8
export OPENCLAW_GATEWAY_PORT_BASE=16000
export OPENCLAW_GATEWAY_PORT_STRIDE=10
export OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=5000
export OPENCLAW_GATEWAY_PORT_MAX_SLOTS=4000
export OPENCLAW_CONTEXT_WINDOW=65536
export OPENCLAW_TRAJECTORY_MAX_TOKENS=65536
export OPENCLAW_MAX_RESPONSE_TOKENS=8192
export OPENCLAW_MAX_AGENT_STEPS=96
export OPENCLAW_MAX_COMPACTION_EVENTS=8
export OPENCLAW_TOKEN_LEDGER_MODE=enforce
export OPENCLAW_TRAIN_MAX_TURNS=1
export OPENCLAW_EVAL_MAX_TURNS=1
export OPENCLAW_TRAIN_NUM_WORKERS=32
export OPENCLAW_EVAL_NUM_WORKERS=32
export OPENCLAW_EVAL_N_SAMPLES=1
export OPENCLAW_FORMAT_PENALTY=0
export OPENCLAW_CHAT_TURN_TIMEOUT=1200
export OPENCLAW_REMOTE_SSH_POOL_SIZE=8
export OPENCLAW_TRAIN_TASKS="${OFFICIAL_ROOT}/data/clawgym_train"
export OPENCLAW_EVAL_TASKS="${OFFICIAL_ROOT}/data/clawgym_eval"
export OPENCLAW_TRAIN_SHUFFLE_SEED=0
export CLAWGYM_SERPER_SKILL_DIR="${REPO_ROOT}/sft_contract/recovered_skills/serper"
export OPENCLAW_ENV_FILE="${REPO_ROOT}/.env"
export OPENCLAW_RECORD_ENABLED=1
export OPENCLAW_RECORD_FILE="${OUTPUT_DIR}/official_proxy_records.jsonl"
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
export HOST=0.0.0.0
export PORT=30001
export SGLANG_API_KEY=""
export SERVED_MODEL_NAME=qwen3-30b-a3b-instruct-clawgym-official
export TOOL_CALL_PARSER=qwen25
export SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION=1

RAY_CLI=("${PYTHON_ENV}/bin/python" "${PYTHON_ENV}/bin/ray")
source "${SLIME_ROOT}/scripts/models/qwen3-30B-A3B.sh"

EVAL_ARGS=(
  --debug-rollout-only
  --actor-num-nodes 1
  --actor-num-gpus-per-node 4
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  --num-gpus-per-node "${NUM_GPUS}"
  "${MODEL_ARGS[@]}"
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${HF_CKPT}"
  --rotary-base "${ROTARY_BASE:-10000000}"
  --save "${OUTPUT_DIR}"
  # eval_only does not write an actor checkpoint, but the shared Megatron
  # argument validator still requires a save interval to be present.
  --save-interval 20
  --disable-rollout-global-dataset
  --rollout-function-path clawgym_rl_rollout.generate_rollout_clawgym_rl
  --num-rollout 0
  --rollout-batch-size 8
  --n-samples-per-prompt 8
  --rollout-max-response-len 8192
  --rollout-max-context-len 65536
  --rollout-temperature 0.7
  --rollout-top-p 1.0
  --reward-key score
  --num-steps-per-rollout 1
  --use-slime-router
  --rollout-num-gpus-per-engine "${SGLANG_TP}"
  --sglang-tool-call-parser qwen25
  --sglang-mem-fraction-static 0.65
  --sglang-context-length 65536
  --sglang-ep-size 1
  --sglang-moe-runner-backend triton
  --sglang-watchdog-timeout 1800
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend flash
  --moe-token-dispatcher-type alltoall
  --use-tensorboard
  --tb-project-name clawgym_rl
  --tb-experiment-name "${EXP_NAME}"
  --custom-generate-function-path clawgym_rl_api_server.generate
  --custom-rm-path clawgym_rl_api_server.reward_func
  --eval-interval 20
  --eval-function-path clawgym_rl_rollout.generate_rollout_clawgym_rl
)

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  "${PYTHON_ENV}/bin/python" -c '
import json
from slime.utils.arguments import parse_args
a = parse_args()
print(json.dumps({
    "debug_rollout_only": a.debug_rollout_only,
    "hf_checkpoint": a.hf_checkpoint,
    "rollout_num_gpus": a.rollout_num_gpus,
    "rollout_num_gpus_per_engine": a.rollout_num_gpus_per_engine,
    "rollout_temperature": a.rollout_temperature,
    "rollout_top_p": a.rollout_top_p,
    "rollout_max_response_len": a.rollout_max_response_len,
    "rollout_max_context_len": a.rollout_max_context_len,
    "eval_function_path": a.eval_function_path,
}, indent=2))
' "${EVAL_ARGS[@]}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}/logs"
exec > >(tee -a "${OUTPUT_DIR}/logs/eval_${RUN_STAMP}.log") 2>&1

if timeout 5 "${RAY_CLI[@]}" status --address="${MASTER_ADDR}:6379" >/dev/null 2>&1; then
  echo "An existing Ray cluster is active; refusing to reuse it." >&2
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

RUNTIME_ENV_JSON="$(python3 -c 'import json, os; keys = [
"PYTHONPATH", "PYTHON_ENV", "PATH", "CUDA_DEVICE_MAX_CONNECTIONS", "FLASHINFER_WORKSPACE_BASE",
"OPENCLAW_REMOTE_HOST", "OPENCLAW_REMOTE_USER", "OPENCLAW_REMOTE_SSH_KEY",
"OPENCLAW_REMOTE_BASE", "OPENCLAW_REMOTE_PROXY_PORT", "OPENCLAW_GATEWAY_IMAGE",
"OPENCLAW_BUNDLED_PLUGINS_DIR",
"OPENCLAW_GATEWAY_PORT_BASE", "OPENCLAW_GATEWAY_PORT_STRIDE", "OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET",
"OPENCLAW_GATEWAY_PORT_MAX_SLOTS", "OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY",
"OPENCLAW_CONTEXT_WINDOW", "OPENCLAW_TRAJECTORY_MAX_TOKENS", "OPENCLAW_MAX_RESPONSE_TOKENS",
"OPENCLAW_MAX_AGENT_STEPS", "OPENCLAW_MAX_COMPACTION_EVENTS", "OPENCLAW_TOKEN_LEDGER_MODE",
"OPENCLAW_TRAIN_MAX_TURNS", "OPENCLAW_EVAL_MAX_TURNS", "OPENCLAW_TRAIN_NUM_WORKERS",
"OPENCLAW_EVAL_NUM_WORKERS", "OPENCLAW_EVAL_N_SAMPLES", "OPENCLAW_FORMAT_PENALTY",
"OPENCLAW_CHAT_TURN_TIMEOUT", "OPENCLAW_TRAIN_TASKS", "OPENCLAW_EVAL_TASKS",
"CLAWGYM_SERPER_SKILL_DIR", "OPENCLAW_ENV_FILE", "OPENCLAW_RECORD_ENABLED",
"OPENCLAW_RECORD_FILE", "TENSORBOARD_DIR", "HOST", "PORT", "SGLANG_API_KEY",
"SERVED_MODEL_NAME", "OPENCLAW_TRAIN_SHUFFLE_SEED", "TOOL_CALL_PARSER",
"SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION"
]; print(json.dumps({"env_vars": {k: os.environ.get(k, "") for k in keys}}))')"

"${RAY_CLI[@]}" job submit --address="http://127.0.0.1:8265" \
  --working-dir "${SLIME_ROOT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_ROOT}/eval_only.py" "${EVAL_ARGS[@]}"
