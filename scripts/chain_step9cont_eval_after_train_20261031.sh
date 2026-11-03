#!/usr/bin/env bash
# After the step9-continuation GRPO run finishes step 99 and exits,
# evaluate semantic steps 40 and 100, five times each, with the
# multi-tool natural-sampling protocol.
set -uo pipefail
TRAIN_LOG=/mnt/humf/openclaw-rl-checkpoints/step9cont-grpo100-20261029/logs/train_20261029-step9cont-grpo100-20261029.log
OUT=/mnt/humf/openclaw-rl-checkpoints/step9cont-step40-100-natural-eval-20261031
mkdir -p "$OUT"
exec >> "$OUT/queue.log" 2>&1
echo "WAIT_TRAIN started at=$(date -u +%FT%TZ)"
while true; do
  if grep -q 'step 99: {' "$TRAIN_LOG" && ! pgrep -f 'run_span_trace_remote_grpo.sh' >/dev/null; then
    break
  fi
  sleep 60
done
echo "TRAIN_DONE at=$(date -u +%FT%TZ); starting step40 then step100, 5 runs each"
exec /root/openclaw-rl-local/.venv-openclaw/bin/python \
  /root/openclaw-rl-local/scripts/run_step9cont_step40_100_eval_20261031.py
