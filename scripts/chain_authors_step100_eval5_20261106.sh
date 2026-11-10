#!/usr/bin/env bash
# After the step108 five-run eval exits successfully, evaluate the authors'
# released step100 with the same natural-sampling protocol.
set -uo pipefail
PREV=/mnt/humf/openclaw-rl-checkpoints/step9plain-step108-natural-eval-20261105/queue.log
OUT=/mnt/humf/openclaw-rl-checkpoints/authors-step100-natural-eval-20261106
PY=/root/openclaw-rl-local/.venv-openclaw/bin/python
mkdir -p "$OUT"
exec >> "$OUT/queue.log" 2>&1
echo "WAIT_STEP108_EVAL started at=$(date -u +%FT%TZ)"

gpus_free() {
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')"
  [ "${used:-999999}" -lt 2048 ]
}

while true; do
  if [ -f "$PREV" ] && grep -q 'EVAL_QUEUE_FAILED' "$PREV"; then
    echo "EVAL_QUEUE_FAILED step108 eval failed; authors step100 not started at=$(date -u +%FT%TZ)"
    exit 1
  fi
  if [ -f "$PREV" ] && grep -q 'EVAL_QUEUE_DONE' "$PREV" \
    && ! pgrep -f 'run_step9plain_step108_eval5_20261105.py' >/dev/null \
    && ! pgrep -f 'chain_step9plain_step108_eval5_20261105.sh' >/dev/null \
    && ! pgrep -f '.run_step39_eval_sync_20260812-091700.sh' >/dev/null \
    && gpus_free; then
    break
  fi
  sleep 60
done

echo "STEP108_EVAL_DONE at=$(date -u +%FT%TZ); starting authors step100, 5 runs"
"$PY" /root/openclaw-rl-local/scripts/run_authors_step100_eval5_20261106.py
status=$?
if [ "$status" -ne 0 ]; then
  echo "EVAL_QUEUE_FAILED code=${status} at=$(date -u +%FT%TZ)"
  exit "$status"
fi
echo "EVAL_QUEUE_DONE at=$(date -u +%FT%TZ)"
