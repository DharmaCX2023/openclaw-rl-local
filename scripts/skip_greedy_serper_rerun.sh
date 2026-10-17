#!/usr/bin/env bash
# Stop the already-running queue if it reaches greedy/temp-0 reruns.
set -u
QUEUE_PID="${1:-3028096}"
LOG=/mnt/humf/openclaw-rl-checkpoints/serper-subset-rerun-20261011/queue.log
WATCH=/mnt/humf/openclaw-rl-checkpoints/serper-subset-rerun-20261011/skip_greedy.watch.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$WATCH"; }
log "watchdog started queue_pid=$QUEUE_PID"
while kill -0 "$QUEUE_PID" 2>/dev/null; do
  if rg -q 'START det-' "$LOG" 2>/dev/null; then
    log "greedy start detected; stopping queue $QUEUE_PID"
    pkill -TERM -P "$QUEUE_PID" 2>/dev/null || true
    kill -TERM "$QUEUE_PID" 2>/dev/null || true
    sleep 5
    pkill -KILL -P "$QUEUE_PID" 2>/dev/null || true
    kill -KILL "$QUEUE_PID" 2>/dev/null || true
    pkill -TERM -f 'tmp_eval_deterministic_20261002' 2>/dev/null || true
    log "queue stopped"
    exit 0
  fi
  if rg -q 'END sci-grpoingrpo-seed1105' "$LOG" 2>/dev/null; then
    log "scientific queue finished; stopping parent so greedy cannot start"
    sleep 2
    if rg -q 'START det-' "$LOG" 2>/dev/null; then
      continue
    fi
    kill -TERM "$QUEUE_PID" 2>/dev/null || true
    sleep 3
    kill -KILL "$QUEUE_PID" 2>/dev/null || true
    log "parent stopped after scientific"
    exit 0
  fi
  sleep 10
done
log "queue already exited"
