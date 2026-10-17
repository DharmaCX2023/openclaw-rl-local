#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 CONFIG_JSON" >&2
  exit 2
fi

CONFIG_JSON="$(readlink -f -- "$1")"
CONFIG_NAME="${CONFIG_JSON##*/}"
RUN_LABEL="${CONFIG_NAME%.json}"
REPO_ROOT="/root/openclaw-rl-local"
FULL_BURN_SESSION="gpu_burn_all8_20260823"
IDLE_BURN_SESSION="numerical_diag_idle_burn_4_7"
DIAG_NUM_GPUS="${DIAG_NUM_GPUS:-4}"
if [ "${DIAG_NUM_GPUS}" = "8" ]; then
  DIAG_CUDA_VISIBLE_DEVICES="${DIAG_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
else
  DIAG_CUDA_VISIBLE_DEVICES="${DIAG_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi

stop_burn() {
  local session="$1"
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux send-keys -t "${session}" C-c
    sleep 2
    tmux kill-session -t "${session}" 2>/dev/null || true
  fi
}

start_full_burn() {
  stop_burn "${IDLE_BURN_SESSION}"
  if ! tmux has-session -t "${FULL_BURN_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${FULL_BURN_SESSION}" \
      "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29570 --nproc-per-node=8 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5"
  fi
}

start_idle_burn() {
  stop_burn "${FULL_BURN_SESSION}"
  stop_burn "${IDLE_BURN_SESSION}"
  tmux new-session -d -s "${IDLE_BURN_SESSION}" \
    "cd /root && exec env CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29572 --nproc-per-node=4 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5"
}

trap start_full_burn EXIT INT TERM
if [ "${DIAG_NUM_GPUS}" = "8" ]; then
  stop_burn "${FULL_BURN_SESSION}"
  stop_burn "${IDLE_BURN_SESSION}"
else
  start_idle_burn
fi

cd "${REPO_ROOT}"
DEFAULT_DEBUG_ROLLOUT_DATA="${REPO_ROOT}/outputs/zero-update-consistency-step99-20260814/baseline/rollout_data/{rollout_id}.pt"
DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA_OVERRIDE:-${DEFAULT_DEBUG_ROLLOUT_DATA}}"
env \
  CUDA_VISIBLE_DEVICES="${DIAG_CUDA_VISIBLE_DEVICES}" \
  NUM_GPUS="${DIAG_NUM_GPUS}" \
  ACTOR_GPUS="${DIAG_NUM_GPUS}" \
  EXPECTED_ACTOR_GPUS="${DIAG_NUM_GPUS}" \
  ROLLOUT_GPUS=0 \
  EXP_NAME="framework-numerical-ablation-step99-${RUN_LABEL}" \
  SAVE_CKPT="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/runtime_${RUN_LABEL}" \
  LOAD_CKPT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700" \
  USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1 \
  NO_LOAD_OPTIM=1 \
  NO_LOAD_RNG=1 \
  OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
  USE_TENSORBOARD=0 \
  OPENCLAW_RECORD_ENABLED=0 \
  OPENCLAW_LOAD_DEBUG_ROLLOUT_DATA="${DEBUG_ROLLOUT_DATA}" \
  SLIME_NUMERICAL_DIAG_CONFIG="${CONFIG_JSON}" \
  USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-0}" \
  USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-0}" \
  DISABLE_ACTIVATION_RECOMPUTE="${DISABLE_ACTIVATION_RECOMPUTE:-0}" \
  DISABLE_MOE_FUSED_KERNELS="${DISABLE_MOE_FUSED_KERNELS:-0}" \
  DISABLE_MOE_GROUPED_GEMM="${DISABLE_MOE_GROUPED_GEMM:-0}" \
  DISABLE_MOE_PERMUTE_FUSION="${DISABLE_MOE_PERMUTE_FUSION:-0}" \
  NUM_ROLLOUT=101 \
  ROLLOUT_BATCH_SIZE=2 \
  N_SAMPLES_PER_PROMPT=4 \
  bash "${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
