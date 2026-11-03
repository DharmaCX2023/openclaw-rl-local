#!/usr/bin/env bash
# Watchdog for the step9 continuation GRPO run (20261029).
# - Checks every 5 minutes that the training driver is alive.
# - If it died before step 99 and no natural completion marker exists,
#   cleans orphan remote containers and restarts:
#     * resuming from the last saved iteration when one exists,
#     * otherwise fresh (moving the stale save dir aside first).
# All actions are logged.
set -uo pipefail
SCRIPT=/root/openclaw-rl-local/scripts/run_step9cont_grpo100_20261029.sh
RUN_STAMP_NAME=step9cont-grpo100-20261029
BASE=/mnt/humf/openclaw-rl-checkpoints
DRIVER_LOG="$BASE/${RUN_STAMP_NAME}-train.log"
TRAIN_LOG="$BASE/${RUN_STAMP_NAME}/logs/train_20261029-${RUN_STAMP_NAME}.log"
SAVE="$BASE/${RUN_STAMP_NAME}"
WD_LOG="$BASE/${RUN_STAMP_NAME}-watchdog.log"
KEY=/root/openclaw_cpu_key
HOST=root@159.138.50.147
PY=/root/openclaw-rl-local/.venv-openclaw/bin/python

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$WD_LOG"; }

clean_orphans() {
  "$PY" - <<'PYEOF'
import json, subprocess
from pathlib import Path
r = subprocess.run(["ssh","-i","/root/openclaw_cpu_key","-o","BatchMode=yes","-o","StrictHostKeyChecking=no","-o","ConnectTimeout=8","root@159.138.50.147","docker ps --format '{{.Names}}'"],text=True,capture_output=True)
orphans=[]
for name in [n for n in r.stdout.splitlines() if n.startswith("clawgym-rl-")]:
    parts=name.split("-")
    if len(parts)<4 or not parts[3].isdigit(): continue
    if not Path(f"/proc/{parts[3]}").exists(): orphans.append(name)
if orphans:
    quoted=" ".join(json.dumps(n) for n in orphans)
    cmd=("ids=$(docker ps -aq -f status=exited -f status=created -f status=dead); "
         'if [ -n "$ids" ]; then docker rm -f $ids >/dev/null || true; fi; '
         f"if [ -n {json.dumps(quoted)} ]; then docker rm -f {quoted} >/dev/null 2>&1 || true; fi;")
    subprocess.run(["ssh","-i","/root/openclaw_cpu_key","-o","BatchMode=yes","-o","StrictHostKeyChecking=no","-o","ConnectTimeout=8","root@159.138.50.147",cmd],capture_output=True)
    print(f"removed {len(orphans)} orphans")
else:
    print("removed 0 orphans")
PYEOF
}

driver_alive() {
  pgrep -f "run_step9cont_grpo100_20261029.sh" >/dev/null 2>&1 || pgrep -f "run_span_trace_remote_grpo.sh" >/dev/null 2>&1
}

last_step() {
  [ -f "$TRAIN_LOG" ] || { echo -1; return; }
  grep -oE "step [0-9]+: \{" "$TRAIN_LOG" | tail -1 | grep -oE "[0-9]+" || echo -1
}

mkdir -p "$(dirname "$WD_LOG")"
log "watchdog started"

while true; do
  sleep 300
  if driver_alive; then
    continue
  fi
  st="$(last_step)"
  if [ "$st" -ge 99 ] 2>/dev/null; then
    log "training reached step $st; treating as complete; watchdog exiting"
    exit 0
  fi
  log "driver dead at step $st; restarting"
  clean_orphans >> "$WD_LOG" 2>&1
  if [ -f "$SAVE/latest_checkpointed_iteration.txt" ]; then
    log "resuming from saved checkpoint: $(cat "$SAVE/latest_checkpointed_iteration.txt")"
    WATCHDOG_RESUME=1 nohup setsid "$SCRIPT" >> "$DRIVER_LOG" 2>&1 < /dev/null &
  else
    log "no saved checkpoint; restarting fresh"
    if [ -d "$SAVE" ]; then
      mv "$SAVE" "$SAVE-stale-$(date -u +%H%M%S)"
      log "moved stale save dir aside"
    fi
    WATCHDOG_RESUME=0 nohup setsid "$SCRIPT" >> "$DRIVER_LOG" 2>&1 < /dev/null &
  fi
  sleep 120
done
