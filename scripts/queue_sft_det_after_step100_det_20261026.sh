#!/usr/bin/env bash
# Wait for the live step-100 deterministic eval, then start the matching SFT eval.
set -uo pipefail

STEP100=/mnt/humf/openclaw-rl-checkpoints/scientific-eval-step100-deterministic-t07-20261025
SFT_OUT=/mnt/humf/openclaw-rl-checkpoints/scientific-eval-sft-deterministic-t07-20261026
SCRIPT=/root/openclaw-rl-local/scripts/run_scientific_sft_deterministic_t07_20261026.sh
mkdir -p "$SFT_OUT"
exec > >(tee -a "$SFT_OUT/waiter.log") 2>&1

pid_file="$STEP100/queue.pid"
if [[ ! -f "$pid_file" ]]; then
  echo "WAIT missing step100 pid file at=$(date -u +%FT%TZ)"
  exit 1
fi
pid=$(cat "$pid_file")
echo "WAIT start step100_pid=$pid at=$(date -u +%FT%TZ)"
while kill -0 "$pid" 2>/dev/null; do
  sleep 20
done
echo "WAIT step100 queue exited at=$(date -u +%FT%TZ)"
if grep -q '^QUEUE DONE' "$STEP100/queue.log"; then
  echo "WAIT saw QUEUE DONE $(grep '^QUEUE DONE' "$STEP100/queue.log" | tail -n1)"
else
  echo "WAIT no QUEUE DONE line; starting SFT anyway"
fi
sleep 90
if [[ -f "$SFT_OUT/queue.pid" ]] && kill -0 "$(cat "$SFT_OUT/queue.pid")" 2>/dev/null; then
  echo "WAIT SFT already running pid=$(cat "$SFT_OUT/queue.pid")"
  echo 'AGENT_LOOP_WAKE_sft-det {"prompt":"SFT deterministic eval is already running. Confirm the queue is healthy and report status."}'
  exit 0
fi
setsid bash "$SCRIPT" </dev/null >> "$SFT_OUT/launcher.out" 2>&1 &
echo $! > "$SFT_OUT/queue.pid"
echo "WAIT started SFT pid=$(cat "$SFT_OUT/queue.pid") at=$(date -u +%FT%TZ)"
echo 'AGENT_LOOP_WAKE_sft-det {"prompt":"Step-100 deterministic eval finished and the matching SFT eval has started. Confirm SFT is running, then watch until QUEUE DONE and report whether OR100 / GRPO-in-GRPO100 beat SFT under temperature 0.7 deterministic inference."}'
