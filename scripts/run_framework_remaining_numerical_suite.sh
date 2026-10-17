#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
CONFIG_ROOT="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/configs"

run_case() {
  local label="$1"
  shift
  echo "[numerical-suite] starting ${label}"
  env "$@" bash "${REPO_ROOT}/scripts/run_framework_numerical_ablation_one.sh" \
    "${CONFIG_ROOT}/${label}.json"
}

run_case 13_baseline_repeat_3
run_case 14_grouped_gemm_off_repeat_2 DISABLE_MOE_GROUPED_GEMM=1
run_case 15_recompute_on_55k MAX_TOKENS_PER_GPU=55000
run_case 16_recompute_off_55k MAX_TOKENS_PER_GPU=55000 DISABLE_ACTIVATION_RECOMPUTE=1

echo "[numerical-suite] complete"
