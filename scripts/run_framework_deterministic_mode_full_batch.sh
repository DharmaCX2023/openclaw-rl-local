#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
CONFIG="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/configs/24_deterministic_mode_full_batch.json"

env \
  MAX_TOKENS_PER_GPU=81920 \
  MEGATRON_DETERMINISTIC_MODE=1 \
  NCCL_ALGO=Ring \
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash "${REPO_ROOT}/scripts/run_framework_numerical_ablation_one.sh" "${CONFIG}"
