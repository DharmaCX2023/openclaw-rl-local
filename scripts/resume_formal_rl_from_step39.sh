#!/bin/bash

set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
RUN_ROOT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700"

# Keep the experiment/save paths unchanged so Slime loads the complete
# checkpoint tracked by RUN_ROOT/latest_checkpointed_iteration.txt and appends
# TensorBoard/proxy records to the original run.
export EXP_NAME="clawgym-official-30b-formal-rl-local-retain-all-20260809-085700"
export SAVE_CKPT="${RUN_ROOT}"
export LOAD_CKPT="${RUN_ROOT}"
export RUN_STAMP="${RUN_STAMP:-resume39-$(date -u +%Y%m%d-%H%M%S)}"

export NUM_ROLLOUT=100
export SAVE_INTERVAL=20
export SAVE_RETAIN_INTERVAL=0
export EVAL_INTERVAL=20
export USE_TENSORBOARD=1

export OPENCLAW_RECORD_ENABLED=1
export OPENCLAW_RECORD_FILE="${RUN_ROOT}/official_proxy_records.jsonl"
export TENSORBOARD_DIR="${RUN_ROOT}/tensorboard"
export OPENCLAW_ENV_FILE="${REPO_ROOT}/.env"

export OPENCLAW_REMOTE_HOST="159.138.50.147"
export OPENCLAW_REMOTE_USER="root"
export OPENCLAW_REMOTE_SSH_KEY="/root/openclaw_cpu_key"

exec "${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
