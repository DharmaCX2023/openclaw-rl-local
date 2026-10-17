#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
CONFIG_JSON="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/configs/04_full_rollout_routing_replay.json"
FULL_BURN_SESSION="gpu_burn_all8_20260823"

stop_burn() {
  local session="$1"
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux send-keys -t "${session}" C-c
    sleep 2
    tmux kill-session -t "${session}" 2>/dev/null || true
  fi
}

start_full_burn() {
  if ! tmux has-session -t "${FULL_BURN_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${FULL_BURN_SESSION}" \
      "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29570 --nproc-per-node=8 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5"
  fi
}

trap start_full_burn EXIT INT TERM
stop_burn "${FULL_BURN_SESSION}"

cd "${REPO_ROOT}"
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  ACTOR_GPUS=4 \
  ROLLOUT_GPUS=4 \
  EXP_NAME="framework-numerical-ablation-step99-full-rollout-replay" \
  SAVE_CKPT="${SAVE_CKPT_OVERRIDE:-${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/runtime_04_full_rollout_routing_replay}" \
  LOAD_CKPT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700" \
  USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1 \
  NO_LOAD_OPTIM=1 \
  NO_LOAD_RNG=1 \
  OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
  USE_TENSORBOARD=0 \
  OPENCLAW_RECORD_ENABLED=1 \
  OPENCLAW_RECORD_FILE="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/04_full_rollout_routing_replay/trajectories.jsonl" \
  OPENCLAW_SAVE_DEBUG_ROLLOUT_DATA="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/04_full_rollout_routing_replay/rollout_data/{rollout_id}.pt" \
  SLIME_NUMERICAL_DIAG_CONFIG="${CONFIG_JSON}" \
  USE_ROUTING_REPLAY=1 \
  USE_ROLLOUT_ROUTING_REPLAY=1 \
  OPENCLAW_REMOTE_PROXY_PORT=39241 \
  PORT=30241 \
  OPENCLAW_GATEWAY_PORT_BASE="${OPENCLAW_GATEWAY_PORT_BASE_OVERRIDE:-36000}" \
  OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=20000 \
  OPENCLAW_GATEWAY_PORT_MAX_SLOTS=8 \
  OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4 \
  OPENCLAW_TRAIN_NUM_WORKERS=8 \
  NUM_ROLLOUT=101 \
  ROLLOUT_BATCH_SIZE=2 \
  N_SAMPLES_PER_PROMPT=4 \
  bash "${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
