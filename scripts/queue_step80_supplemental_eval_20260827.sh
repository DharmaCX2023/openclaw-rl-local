#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
TRAIN_ROOT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129"
TRAIN_SESSION="formal_rl_retrain_resume80_20260827"
QUEUE_LOG="${TRAIN_ROOT}/logs/step80_supplemental_eval_queue.log"

exec >>"${QUEUE_LOG}" 2>&1
echo "[$(date -u +%FT%TZ)] waiting for ${TRAIN_SESSION} to finish"
while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; do
  sleep 60
done

if [[ "$(tr -d '\n' <"${TRAIN_ROOT}/latest_checkpointed_iteration.txt" 2>/dev/null || true)" != "99" ]] || \
   [[ ! -s "${TRAIN_ROOT}/iter_0000099/.metadata" ]] || \
   [[ ! -s "${TRAIN_ROOT}/iter_0000099/common.pt" ]]; then
  echo "[$(date -u +%FT%TZ)] formal training did not finish at iter_0000099; supplemental eval not started"
  exit 1
fi

echo "[$(date -u +%FT%TZ)] formal training complete; starting step-80 supplemental eval"
set +e
bash "${REPO_ROOT}/scripts/run_step80_supplemental_eval_20260827.sh"
eval_status=$?
set -e
echo "[$(date -u +%FT%TZ)] supplemental eval exited with status ${eval_status}"

# Whether eval succeeds or needs diagnosis, do not leave eight idle GPUs after
# its Ray processes have drained. The monitor waits for true GPU idleness first.
exec env \
  TRAIN_SESSION="step80_supplemental_eval_finished" \
  BURN_SESSION="gpu_burn_all8_after_step80_supplement_20260827" \
  CHECK_INTERVAL=60 \
  TRAIN_LOG="${QUEUE_LOG}" \
  MONITOR_LOG="${TRAIN_ROOT}/logs/post_supplement_burn_monitor.log" \
  BURN_LOG_ROOT="/root/logs/openclaw-burn/post-step80-supplement-20260827" \
  bash "${REPO_ROOT}/scripts/monitor_formal_retrain_20260826.sh"
