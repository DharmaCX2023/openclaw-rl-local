#!/usr/bin/env bash
set -euo pipefail
cd /root/openclaw-rl-local

OR_PID_FILE=/mnt/humf/openclaw-rl-checkpoints/clean_or40_queue.pid
OR_ROOT=/mnt/humf/openclaw-rl-checkpoints/clawgym-or100-clean-serper-20261012

wait_for_or40() {
  local pid
  while true; do
    pid="$(cat "$OR_PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      sleep 60
      continue
    fi
    if [ -f "$OR_ROOT/latest_checkpointed_iteration.txt" ] \
       && [ "$(cat "$OR_ROOT/latest_checkpointed_iteration.txt")" = "39" ]; then
      echo "OR40 committed iteration 39; starting GRPO-in-GRPO resume."
      return 0
    fi
    echo "OR queue is down without iter39; restarting OR40." >&2
    stamp=$(date +%Y%m%d_%H%M%S)
    log=/mnt/humf/openclaw-rl-checkpoints/clean_or40_queue_${stamp}.log
    setsid bash scripts/queue_clean_or40_20261016.sh >"$log" 2>&1 </dev/null &
    echo $! > "$OR_PID_FILE"
    echo "$log" > /mnt/humf/openclaw-rl-checkpoints/clean_or40_queue.current_log
    sleep 60
  done
}

wait_for_or40
stamp=$(date +%Y%m%d_%H%M%S)
log=/mnt/humf/openclaw-rl-checkpoints/clean_grpo100_queue_v2_${stamp}.log
setsid bash scripts/queue_clean_grpo100_20261012.sh >"$log" 2>&1 </dev/null &
echo $! > /mnt/humf/openclaw-rl-checkpoints/clean_grpo100_queue_20261012.pid
echo "$log" > /mnt/humf/openclaw-rl-checkpoints/clean_grpo100_queue_20261012.current_log
wait $!
