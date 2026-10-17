#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
CONFIG_ROOT="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/configs"
LOG_ROOT="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824"

cd "${REPO_ROOT}"

env \
  USE_ROUTING_REPLAY=0 \
  DISABLE_MOE_GROUPED_GEMM=0 \
  DISABLE_MOE_PERMUTE_FUSION=0 \
  bash scripts/run_framework_numerical_ablation_one.sh \
    "${CONFIG_ROOT}/12_baseline_repeat_2.json" \
    > "${LOG_ROOT}/12_baseline_repeat_2_runner.log" 2>&1

env \
  USE_ROUTING_REPLAY=0 \
  DISABLE_MOE_GROUPED_GEMM=0 \
  DISABLE_MOE_PERMUTE_FUSION=0 \
  bash scripts/run_framework_numerical_ablation_one.sh \
    "${CONFIG_ROOT}/13_baseline_repeat_3.json" \
    > "${LOG_ROOT}/13_baseline_repeat_3_runner.log" 2>&1

env \
  USE_ROUTING_REPLAY=0 \
  DISABLE_MOE_GROUPED_GEMM=1 \
  DISABLE_MOE_PERMUTE_FUSION=0 \
  bash scripts/run_framework_numerical_ablation_one.sh \
    "${CONFIG_ROOT}/14_grouped_gemm_off_repeat_2.json" \
    > "${LOG_ROOT}/14_grouped_gemm_off_repeat_2_runner.log" 2>&1
