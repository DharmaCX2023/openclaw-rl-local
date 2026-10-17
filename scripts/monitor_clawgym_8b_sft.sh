#!/usr/bin/env bash
set -uo pipefail

RUN="${1:?usage: $0 RUN_DIR TMUX_SESSION}"
TRAIN_SESSION="${2:?usage: $0 RUN_DIR TMUX_SESSION}"
INTERVAL="${SFT_MONITOR_INTERVAL_SECONDS:-1200}"
OUT="$RUN/logs/resource_monitor.log"

while true; do
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  state="$(head -n 1 "$RUN/status" 2>/dev/null || echo UNKNOWN)"
  {
    echo "===== $now state=$state ====="
    if [[ -f "$RUN/logs/train.log" ]]; then
      tr '\r' '\n' < "$RUN/logs/train.log" | grep 'Train step of epoch' | tail -n 1 || true
      grep -E 'CUDA out of memory|OutOfMemoryError|RuntimeError|Traceback|Saving checkpoint|Loaded the checkpoint' "$RUN/logs/train.log" | tail -n 8 || true
    fi
    if [[ -f "$RUN/ds_checkpoints/latest" ]]; then
      echo "latest_checkpoint=$(cat "$RUN/ds_checkpoints/latest")"
      du -sh "$RUN/ds_checkpoints" 2>/dev/null || true
    else
      echo "latest_checkpoint=none"
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
    df -h /root
  } >> "$OUT" 2>&1

  if [[ "$state" != RUNNING* ]]; then
    exit 0
  fi
  if ! tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    echo "===== $now WARNING training tmux session missing while status is RUNNING =====" >> "$OUT"
    exit 1
  fi
  sleep "$INTERVAL"
done
