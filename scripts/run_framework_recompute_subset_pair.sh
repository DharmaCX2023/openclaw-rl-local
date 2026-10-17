#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
CONFIG_ROOT="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/configs"
SUBSET="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/recompute_subset_max45k/rollout_data/100.pt"
SOURCE="${REPO_ROOT}/outputs/zero-update-consistency-step99-20260814/baseline/rollout_data/100.pt"

"${REPO_ROOT}/.venv-openclaw/bin/python" \
  "${REPO_ROOT}/scripts/build_numerical_recompute_subset.py" \
  --input "${SOURCE}" \
  --output "${SUBSET}" \
  --max-sample-tokens 45000

run_case() {
  local label="$1"
  shift
  echo "[recompute-subset-pair] starting ${label}"
  env \
    LOAD_DEBUG_ROLLOUT_DATA_OVERRIDE="${REPO_ROOT}/outputs/framework-numerical-ablation-step99-20260824/recompute_subset_max45k/rollout_data/{rollout_id}.pt" \
    MAX_TOKENS_PER_GPU=45000 \
    "$@" \
    bash "${REPO_ROOT}/scripts/run_framework_numerical_ablation_one.sh" \
      "${CONFIG_ROOT}/${label}.json"
}

run_case 21_recompute_subset_on
sleep 30
run_case 22_recompute_subset_off DISABLE_ACTIVATION_RECOMPUTE=1

echo "[recompute-subset-pair] complete"
