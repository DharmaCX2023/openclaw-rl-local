#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
RUN_ROOT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129"
TRACKER="${RUN_ROOT}/latest_checkpointed_iteration.txt"

if [[ ! -r "${TRACKER}" ]] || [[ "$(tr -d '\n' <"${TRACKER}")" != "79" ]]; then
  echo "Expected complete step-80 checkpoint tracker value 79: ${TRACKER}" >&2
  exit 1
fi
for required in .metadata common.pt; do
  if [[ ! -s "${RUN_ROOT}/iter_0000079/${required}" ]]; then
    echo "Incomplete step-80 checkpoint file: ${RUN_ROOT}/iter_0000079/${required}" >&2
    exit 1
  fi
done

# Reuse the original save/TensorBoard roots. Megatron loads iteration 79 and
# returns start_rollout_id=80, so the first new policy update is semantic step 81.
export EXP_NAME="clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129"
export SAVE_CKPT="${RUN_ROOT}"
export LOAD_CKPT="${RUN_ROOT}"
export RUN_STAMP="${RUN_STAMP:-resume80-$(date -u +%Y%m%d-%H%M%S)}"
export USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1
export SKIP_EVAL_BEFORE_TRAIN=1

exec bash "${REPO_ROOT}/scripts/run_formal_retrain_20260826.sh"
