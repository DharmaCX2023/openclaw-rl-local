#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
DENSE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
CONFIG_JSON="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/configs/08_dense_qwen3_0_6b.json"
FULL_BURN_SESSION="gpu_burn_all8_20260823"
IDLE_BURN_SESSION="numerical_diag_idle_burn_1_7"

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

trap start_full_burn EXIT INT TERM
stop_burn "${FULL_BURN_SESSION}"
stop_burn "${IDLE_BURN_SESSION}"
tmux new-session -d -s "${IDLE_BURN_SESSION}" \
  "cd /root && exec env CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29573 --nproc-per-node=7 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5"

cd "${REPO_ROOT}"
env \
  CUDA_VISIBLE_DEVICES=0 \
  NUM_GPUS=1 \
  ACTOR_GPUS=1 \
  ROLLOUT_GPUS=0 \
  EXPECTED_ACTOR_GPUS=1 \
  TENSOR_MODEL_PARALLEL_SIZE=1 \
  CONTEXT_PARALLEL_SIZE=1 \
  EXPERT_MODEL_PARALLEL_SIZE=1 \
  ENABLE_SEQUENCE_PARALLEL=0 \
  MODEL_CONFIG_SCRIPT="${REPO_ROOT}/slime/scripts/models/qwen3-0.6B.sh" \
  MODEL_ARGS_ROTARY_BASE=1000000 \
  ROTARY_BASE=1000000 \
  HF_CKPT="${DENSE_MODEL}" \
  REF_LOAD="${DENSE_MODEL}" \
  EXP_NAME="framework-numerical-dense-qwen3-0.6b" \
  SAVE_CKPT="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/runtime_08_dense_qwen3_0_6b_retry2" \
  OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
  USE_TENSORBOARD=0 \
  OPENCLAW_RECORD_ENABLED=0 \
  OPENCLAW_LOAD_DEBUG_ROLLOUT_DATA="${REPO_ROOT}/outputs/zero-update-consistency-step99-20260814/baseline/rollout_data/100.pt" \
  SLIME_NUMERICAL_DIAG_CONFIG="${CONFIG_JSON}" \
  NUM_ROLLOUT=1 \
  ROLLOUT_BATCH_SIZE=2 \
  N_SAMPLES_PER_PROMPT=4 \
  bash "${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
