#!/bin/bash
# One real GRPO update through the production OpenClaw/ClawGym path.
#
# This is deliberately not an inference-only check. It initializes the
# Megatron actor and two TP2 SGLang rollout engines, synchronizes weights,
# collects two groups of four trajectories, computes rewards/advantages,
# performs one optimizer update and synchronizes the trained weights back to
# SGLang. Checkpoint output is disabled by default because a single 30B
# optimizer checkpoint is roughly 399 GiB; set
# OPENCLAW_DISABLE_CHECKPOINT_SAVE=0 explicitly when persistence is required.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd -P)"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
export RUN_STAMP
export EXP_NAME="${EXP_NAME:-clawgym-official-30b-real-rl-smoke-${RUN_STAMP}}"
export SAVE_CKPT="${SAVE_CKPT:-${REPO_ROOT}/outputs/${EXP_NAME}}"

# Eight-GPU disaggregated production topology.
export NUM_GPUS=8
export ACTOR_GPUS=4
export ROLLOUT_GPUS=4
export SGLANG_TP=2

# Minimum batch that remains a real GRPO update. The local rollout adapter
# requires at least four samples per group; the fixed-seed first two tasks
# have demonstrated non-zero within-group reward variance.
export NUM_ROLLOUT=1
export ROLLOUT_BATCH_SIZE=2
export N_SAMPLES_PER_PROMPT=4
export OPENCLAW_TRAIN_NUM_WORKERS=8

# Keep the smoke self-contained without writing a production-sized checkpoint.
export OPENCLAW_DISABLE_CHECKPOINT_SAVE="${OPENCLAW_DISABLE_CHECKPOINT_SAVE:-1}"
export USE_WANDB=0
export OPENCLAW_RECORD_ENABLED=1
export OPENCLAW_TRAIN_SHUFFLE_SEED="${OPENCLAW_TRAIN_SHUFFLE_SEED:-0}"

echo "Starting real RL smoke:"
echo "  topology: actor TP4/EP4 on 4 GPUs; rollout TP2 x 2 engines on 4 GPUs"
echo "  batch: 2 prompts x 4 samples; 1 optimizer rollout"
echo "  output: ${SAVE_CKPT}"

exec bash "${SCRIPT_DIR}/run_qwen3_30b_a3b_official_remote_grpo.sh"
