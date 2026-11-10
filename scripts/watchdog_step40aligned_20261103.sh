#!/usr/bin/env bash
# Watch the step40-aligned 60-step run. On death, record the cause and restart
# once the save directory can resume. Stop after two deaths with no new optimizer step.
set -uo pipefail
SCRIPT=/root/openclaw-rl-local/scripts/run_step40aligned_grpo60_20261103.sh
BASE=/mnt/humf/openclaw-rl-checkpoints
NAME=step40aligned-grpo60-20261103
DRIVER_LOG="$BASE/${NAME}-train.log"
SAVE="$BASE/${NAME}"
WD_LOG="$BASE/${NAME}-watchdog.log"
PY=/root/openclaw-rl-local/.venv-openclaw/bin/python
LAST_MARK=""
RESTARTS=0
LAST_GOOD_STEP=-1

log() { echo "[$(date -u +%FT%TZ)] $*"; echo "[$(date -u +%FT%TZ)] $*" >> "$WD_LOG"; }

driver_alive() {
  pgrep -f "bash clawgym-official/run_span_trace_remote_grpo.sh" >/dev/null 2>&1
}

progress() {
  local train_log perf step
  train_log="$(ls -1 "$SAVE"/logs/train_*.log 2>/dev/null | tail -1 || true)"
  perf=-1
  step=-1
  if [ -n "$train_log" ]; then
    perf="$(grep -oE "perf [0-9]+:" "$train_log" | tail -1 | grep -oE "[0-9]+" || true)"
    step="$(grep -oE "step [0-9]+: \{'train/loss'" "$train_log" | tail -1 | grep -oE "[0-9]+" || true)"
  fi
  echo "${perf:--1} ${step:--1}"
}

clean_orphans() {
  "$PY" - <<'PYEOF'
import json, subprocess
from pathlib import Path
r = subprocess.run(
    ["ssh","-i","/root/openclaw_cpu_key","-o","BatchMode=yes","-o","StrictHostKeyChecking=no","-o","ConnectTimeout=12","root@159.138.50.147","docker ps --format '{{.Names}}'"],
    text=True, capture_output=True,
)
orphans=[]
for name in [n for n in r.stdout.splitlines() if n.startswith("clawgym-rl-")]:
    parts=name.split("-")
    if len(parts)<4 or not parts[3].isdigit():
        continue
    if not Path(f"/proc/{parts[3]}").exists():
        orphans.append(name)
if not orphans:
    print("removed 0 orphans")
else:
    quoted=" ".join(json.dumps(n) for n in orphans)
    cmd=f"docker rm -f {quoted} >/dev/null 2>&1 || true"
    subprocess.run(
        ["ssh","-i","/root/openclaw_cpu_key","-o","BatchMode=yes","-o","StrictHostKeyChecking=no","-o","ConnectTimeout=20","root@159.138.50.147",cmd],
        capture_output=True,
    )
    print(f"removed {len(orphans)} orphans")
PYEOF
}

stop_scoped_ray() {
  if pgrep -f "/tmp/ray-step40aligned" >/dev/null 2>&1; then
    pkill -TERM -f "/tmp/ray-step40aligned" >/dev/null 2>&1 || true
    sleep 5
    pkill -KILL -f "/tmp/ray-step40aligned" >/dev/null 2>&1 || true
  fi
}

log "watchdog started"
while true; do
  sleep 180
  read -r PERF STEP < <(progress)
  MARK="perf=${PERF} step=${STEP}"
  if [ "$MARK" != "$LAST_MARK" ]; then
    log "WATCHDOG_PROGRESS ${MARK}"
    LAST_MARK="$MARK"
  fi
  if driver_alive; then
    continue
  fi
  if [ "${STEP:--1}" -ge 99 ] 2>/dev/null; then
    log "WATCHDOG_DONE step=${STEP}"
    exit 0
  fi
  CAUSE="$(tail -n 80 "$DRIVER_LOG" 2>/dev/null | grep -E "Traceback|Error|Exception|Scoped Ray|CUDA out of memory|ActorDied|SAVE_CKPT must" | tail -n 5 | tr '\n' ' ' )"
  log "WATCHDOG_DEAD step=${STEP} perf=${PERF} cause=${CAUSE:-unknown}"
  if [ "${STEP:--1}" -ge 40 ] 2>/dev/null && [ "${STEP}" != "${LAST_GOOD_STEP}" ]; then
    RESTARTS=0
    LAST_GOOD_STEP="${STEP}"
  fi
  if [ "$RESTARTS" -ge 2 ]; then
    log "WATCHDOG_GIVE_UP restarts=${RESTARTS} cause=${CAUSE:-unknown}"
    exit 1
  fi
  clean_orphans >> "$WD_LOG" 2>&1 || true
  stop_scoped_ray
  if [ ! -f "$SAVE/latest_checkpointed_iteration.txt" ] && [ -d "$SAVE" ]; then
    mv "$SAVE" "${SAVE}-stale-$(date -u +%H%M%S)"
    log "no checkpoint yet; moved partial save aside and restarting from rollout 40"
  else
    log "restarting; script will resume from the saved iteration"
  fi
  nohup setsid bash "$SCRIPT" >> "$DRIVER_LOG" 2>&1 < /dev/null &
  RESTARTS=$((RESTARTS + 1))
  log "restart pid=$! count=${RESTARTS}"
  sleep 120
done
