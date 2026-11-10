#!/usr/bin/env bash
# Wait until the step9plain run has saved step 108 and released the GPUs,
# then run five natural-sampling evals. Do not start while training is alive.
set -uo pipefail
BASE=/mnt/humf/openclaw-rl-checkpoints
NAME=step9plain-grpo100-20261104
SAVE="$BASE/${NAME}"
TRAIN_LOG="$SAVE/logs/train_20261104-${NAME}.log"
WD_LOG="$BASE/${NAME}-watchdog.log"
OUT="$BASE/step9plain-step108-natural-eval-20261105"
PY=/root/openclaw-rl-local/.venv-openclaw/bin/python
mkdir -p "$OUT"
exec >> "$OUT/queue.log" 2>&1
echo "WAIT_TRAIN started at=$(date -u +%FT%TZ)"

gpus_free() {
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')"
  [ "${used:-999999}" -lt 2048 ]
}

step_logged() {
  python3 - "$TRAIN_LOG" << 'PY'
import sys
data = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
raise SystemExit(0 if "step 108: {'train/loss'" in data else 1)
PY
}

while true; do
  if [ -f "$WD_LOG" ] && grep -q 'WATCHDOG_GIVE_UP' "$WD_LOG" \
    && ! step_logged; then
    echo "EVAL_QUEUE_FAILED training watchdog gave up before step 108 at=$(date -u +%FT%TZ)"
    exit 1
  fi
  if step_logged \
    && [ -f "$WD_LOG" ] && grep -q 'WATCHDOG_DONE' "$WD_LOG" \
    && ! pgrep -f 'bash /root/openclaw-rl-local/scripts/run_step9plain_grpo100_20261104.sh' >/dev/null \
    && ! pgrep -f 'bash /root/openclaw-rl-local/scripts/watchdog_step9plain_20261104.sh' >/dev/null \
    && ! pgrep -f '/tmp/ray-step9plain' >/dev/null \
    && [ "$(tr -d '[:space:]' < "$SAVE/latest_checkpointed_iteration.txt" 2>/dev/null || true)" = "108" ] \
    && gpus_free; then
    break
  fi
  sleep 60
done

echo "TRAIN_DONE step=108 at=$(date -u +%FT%TZ); starting 5 natural evals"
"$PY" /root/openclaw-rl-local/scripts/run_step9plain_step108_eval5_20261105.py
status=$?
if [ "$status" -ne 0 ]; then
  echo "EVAL_QUEUE_FAILED code=${status} at=$(date -u +%FT%TZ)"
  exit "$status"
fi
echo "EVAL_QUEUE_DONE at=$(date -u +%FT%TZ)"
