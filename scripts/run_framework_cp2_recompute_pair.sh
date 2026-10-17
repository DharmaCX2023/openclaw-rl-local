#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
CONFIG_ROOT="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/configs"

run_case() {
  local label="$1"
  shift
  echo "[cp2-recompute-pair] starting ${label}"
  env \
    DIAG_NUM_GPUS=8 \
    CONTEXT_PARALLEL_SIZE=2 \
    MAX_TOKENS_PER_GPU=55000 \
    "$@" \
    bash "${REPO_ROOT}/scripts/run_framework_numerical_ablation_one.sh" \
      "${CONFIG_ROOT}/${label}.json"
}

run_case 18_cp2_recompute_on_55k
# The launcher tears down and recreates the local Ray head. Give the dashboard
# agent time to register before immediately submitting the paired job.
sleep 30
run_case 19_cp2_recompute_off_55k DISABLE_ACTIVATION_RECOMPUTE=1

echo "[cp2-recompute-pair] complete"
