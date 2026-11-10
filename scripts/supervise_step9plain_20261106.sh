#!/usr/bin/env bash
# Keep the step9plain run and its five evals moving.
# Training is restarted from the saved iteration if it dies before step 108.
# A live run is left alone unless the train log has been silent for 30 minutes.
set -uo pipefail
BASE=/mnt/humf/openclaw-rl-checkpoints
NAME=step9plain-grpo100-20261104
SAVE="$BASE/${NAME}"
TRAIN_LOG="$SAVE/logs/train_20261104-step9plain-grpo100-20261104.log"
OUTER="$BASE/${NAME}-train.log"
WD_LOG="$BASE/${NAME}-watchdog.log"
QUEUE="$BASE/step9plain-step108-natural-eval-20261105/queue.log"
TRAIN_SCRIPT=/root/openclaw-rl-local/scripts/run_step9plain_grpo100_20261104.sh
WD_SCRIPT=/root/openclaw-rl-local/scripts/watchdog_step9plain_20261104.sh
CHAIN=/root/openclaw-rl-local/scripts/chain_step9plain_step108_eval5_20261105.sh
EVAL_PY=/root/openclaw-rl-local/scripts/run_step9plain_step108_eval5_20261105.py
SUP_LOG="$BASE/${NAME}-supervisor.log"
PYBIN=/root/openclaw-rl-local/.venv-openclaw/bin/python
STALLS=0
STUCK_LOGGED=0
DEAD_LOGGED=0
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$SUP_LOG"; }

stop_running_job() {
  local id
  id="$("$PYBIN" -m ray.scripts.scripts job list --address http://127.0.0.1:8643 2>/dev/null | awk '/raysubmit_/ {id=$1} /RUNNING/ {if (id) print id; id=""}' | tail -1)"
  if [ -z "$id" ]; then
    id="$("$PYBIN" -m ray.scripts.scripts job list --address http://127.0.0.1:8643 2>/dev/null | awk '/RUNNING/ {print $1; exit}')"
  fi
  if [ -n "$id" ]; then
    "$PYBIN" -m ray.scripts.scripts job stop --address http://127.0.0.1:8643 --no-wait "$id" || true
    log "SUPERVISOR_ACTION stopped stalled job ${id}"
  else
    log "SUPERVISOR_STUCK no running job on dashboard 8643"
    STUCK_LOGGED=1
  fi
}

latest_step() {
  python3 - "$TRAIN_LOG" << 'PY'
import re, sys
try:
    data = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
except FileNotFoundError:
    print(-1)
    raise SystemExit
steps = re.findall(r"step (\d+): \{'train/loss'", data)
print(steps[-1] if steps else -1)
PY
}

driver_alive() { pgrep -f "bash ${TRAIN_SCRIPT}" >/dev/null 2>&1; }
wd_alive() { pgrep -f "bash ${WD_SCRIPT}" >/dev/null 2>&1; }
chain_alive() { pgrep -f "bash ${CHAIN}" >/dev/null 2>&1; }
eval_alive() { pgrep -f "run_step9plain_step108_eval5_20261105.py" >/dev/null 2>&1; }

log_age() {
  python3 - "$TRAIN_LOG" << 'PY'
import os, sys, time
try:
    print(int(time.time() - os.path.getmtime(sys.argv[1])))
except OSError:
    print(999999)
PY
}

start_train() {
  log "SUPERVISOR_ACTION restarting training from the saved checkpoint"
  nohup setsid bash "$TRAIN_SCRIPT" >> "$OUTER" 2>&1 < /dev/null &
  log "train pid=$!"
}

start_watchdog() {
  log "SUPERVISOR_ACTION starting watchdog"
  nohup setsid bash "$WD_SCRIPT" >> "$BASE/${NAME}-watchdog.stdout" 2>&1 < /dev/null &
  log "watchdog pid=$!"
}

start_chain() {
  log "SUPERVISOR_ACTION starting eval chain"
  nohup setsid bash "$CHAIN" >/dev/null 2>&1 < /dev/null &
  log "chain pid=$!"
}

log "supervisor started step=$(latest_step)"
while true; do
  sleep 120
  STEP="$(latest_step)"
  AGE="$(log_age)"
  if [ -f "$QUEUE" ] && grep -q 'EVAL_QUEUE_DONE' "$QUEUE"; then
    log "SUPERVISOR_DONE step=${STEP}"
    exit 0
  fi
  if [ "$STEP" -lt 108 ]; then
    if driver_alive; then
      DEAD_LOGGED=0
      if [ "$AGE" -gt 1800 ]; then
        STALLS=$((STALLS + 1))
        log "SUPERVISOR_STALL step=${STEP} log_age=${AGE}s count=${STALLS}"
        if [ "$STALLS" -le 2 ]; then
          stop_running_job
          log "SUPERVISOR_ACTION watchdog will resume the stalled run"
        elif [ "$STUCK_LOGGED" -eq 0 ]; then
          log "SUPERVISOR_STUCK stall repeats step=${STEP}"
          STUCK_LOGGED=1
        fi
      else
        STALLS=0
      fi
      continue
    fi
    if [ "$DEAD_LOGGED" -eq 0 ]; then
      log "SUPERVISOR_DEAD step=${STEP} log_age=${AGE}s"
      DEAD_LOGGED=1
    fi
    if wd_alive; then
      continue
    fi
    start_train
    start_watchdog
    continue
  fi
  # Step 108 is logged. Leave the handoff to the chain once the driver exits.
  if ! chain_alive && ! eval_alive; then
    if [ -f "$QUEUE" ] && grep -q 'EVAL_QUEUE_FAILED' "$QUEUE"; then
      if [ "$STUCK_LOGGED" -eq 0 ]; then
        log "SUPERVISOR_STUCK eval queue failed"
        STUCK_LOGGED=1
      fi
      continue
    fi
    start_chain
  fi
done
