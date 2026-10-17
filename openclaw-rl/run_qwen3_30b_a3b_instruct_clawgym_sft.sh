#!/bin/bash
# Qwen3-30B-A3B-Instruct-2507 SFT on the OpenClaw 2026.4.1-aligned
# six-tool, fixed-skills, 64K ClawGym trajectories.
# This consumes RUC-AIBOX/ClawGym-Trajectory directly:
#   {"messages": [...], "tools": [...]}
#
# It is an offline SFT/behavior-cloning path. It does not start the
# OpenClaw API rollout server or PRM workers.

SKIP_CLUSTER_CLEANUP=${SKIP_CLUSTER_CLEANUP:-0}
if [ "${SKIP_CLUSTER_CLEANUP}" != "1" ]; then
  pkill -9 sglang
  sleep 3
  ray stop --force
  pkill -9 ray
  pkill -9 -f "slime/train_async.py"
  sleep 3
  pkill -9 ray
  pkill -9 -f "slime/train_async.py"
fi

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-8}

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPO_ROOT}/slime"
export MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-10000000}"
source "${SLIME_ROOT}/scripts/models/qwen3-30B-A3B.sh"

HF_CKPT=${HF_CKPT:-${REPO_ROOT}/models/Qwen3-30B-A3B-Instruct-2507}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-${REPO_ROOT}/ckpt/qwen3-30b-a3b-instruct-clawgym-sft}
CLAWGYM_TRAJECTORY_DATA=${CLAWGYM_TRAJECTORY_DATA:-${REPO_ROOT}/data/clawgym-trajectory/sft_trajectory_openclaw_v2026_4_1_6tools_skills_len64k.jsonl}

CKPT_ARGS=(
   --megatron-to-hf-mode bridge
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval "${SAVE_INTERVAL:-100}"
   --rotary-base "${ROTARY_BASE:-10000000}"
)
if [ -n "${LOAD_CKPT:-}" ]; then
   CKPT_ARGS+=(--load "${LOAD_CKPT}")
fi

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${CLAWGYM_TRAJECTORY_DATA}"
   --input-key messages
   --tool-key tools
   --metadata-key metadata
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH:-1}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-4}"

   --loss-type sft_loss
   # Full-history ClawGym rendering with assistant-only loss ranges.
   --loss-mask-type "${LOSS_MASK_TYPE:-clawgym}"
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --disable-rewards-normalization
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-4}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
   --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-4}"
   --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}"

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-5}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR:-1e-6}"
   --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.03}"
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --no-gradient-accumulation-fusion
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"
)

if [ "${MOE_ENABLE_DEEPEP:-0}" = "1" ]; then
  MISC_ARGS+=(--moe-enable-deepep)
fi

USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-openclaw_rl}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group qwen3-30b-a3b-instruct-clawgym-sft
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

for i in $(seq 1 60); do
  if python3 - <<'PY'
import urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:8265/api/version", timeout=1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 1
done

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"FLASHINFER_WORKSPACE_BASE\": \"${FLASHINFER_WORKSPACE_BASE}\"
  }
}"

RAY_JOB_WAIT=${RAY_JOB_WAIT:-1}
RAY_JOB_SUBMIT_WAIT_ARGS=()
if [ "${RAY_JOB_WAIT}" = "0" ]; then
  RAY_JOB_SUBMIT_WAIT_ARGS=(--no-wait)
fi

ray job submit --address="http://127.0.0.1:8265" \
   "${RAY_JOB_SUBMIT_WAIT_ARGS[@]}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_ROOT}/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]}
