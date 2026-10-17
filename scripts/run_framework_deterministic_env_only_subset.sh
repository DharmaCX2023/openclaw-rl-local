#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
CONFIG="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/configs/25_deterministic_env_only_subset.json"
SUBSET_PATTERN="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/recompute_subset_max45k/rollout_data/{rollout_id}.pt"

env \
  LOAD_DEBUG_ROLLOUT_DATA_OVERRIDE="${SUBSET_PATTERN}" \
  MAX_TOKENS_PER_GPU=45000 \
  MEGATRON_DETERMINISTIC_MODE=0 \
  NCCL_ALGO=Ring \
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash "${REPO_ROOT}/scripts/run_framework_numerical_ablation_one.sh" "${CONFIG}"
