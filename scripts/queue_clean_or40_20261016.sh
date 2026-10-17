#!/usr/bin/env bash
set -euo pipefail
cd /root/openclaw-rl-local

set -a
# shellcheck disable=SC1091
source /root/openclaw-rl-local/.env
set +a

OR_ROOT=/mnt/humf/openclaw-rl-checkpoints/clawgym-or100-clean-serper-20261012

serper_preflight() {
  local output
  output="$(mktemp)"
  if ! python3 - >"$output" 2>&1 <<'PY'
import json
import os
import urllib.request

request = urllib.request.Request(
    os.environ.get("SERPER_URL", "https://google.serper.dev").rstrip("/") + "/search",
    data=json.dumps({"q": "OpenAI", "num": 1}).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['SERPER_API_KEY']}",
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.load(response)
results = payload.get("organic", [])
if not isinstance(results, list):
    raise SystemExit("Serper preflight returned an invalid response")
print(json.dumps({"results": results, "count": len(results)}))
PY
  then
    rm -f "$output"
    echo "Serper preflight failed; no training was started." >&2
    return 1
  fi
  python3 - "$output" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if not isinstance(payload.get("results"), list):
    raise SystemExit("Serper preflight returned an invalid response")
print(f"Serper preflight passed (results={payload.get('count', 0)}).")
PY
  rm -f "$output"
}

run_once() {
  local train_pid retain_pid
  serper_preflight
  if [ -f "$OR_ROOT/latest_checkpointed_iteration.txt" ]; then
    echo "Resuming OR from committed iteration $(cat "$OR_ROOT/latest_checkpointed_iteration.txt")."
    CLEAN_RESUME=1 CLEAN_SAVE_CKPT="$OR_ROOT" CLEAN_NUM_ROLLOUT=40 \
      bash scripts/run_clean_grpo100_20261012.sh or &
  else
    CLEAN_SAVE_CKPT="$OR_ROOT" CLEAN_NUM_ROLLOUT=40 \
      bash scripts/run_clean_grpo100_20261012.sh or &
  fi
  train_pid=$!

  for _ in $(seq 1 120); do
    if [ -d "$OR_ROOT/logs" ]; then
      break
    fi
    if ! kill -0 "$train_pid" 2>/dev/null; then
      wait "$train_pid"
    fi
    sleep 1
  done
  if [ ! -d "$OR_ROOT/logs" ]; then
    kill "$train_pid" 2>/dev/null || true
    wait "$train_pid" 2>/dev/null || true
    echo "Training launcher did not initialize $OR_ROOT" >&2
    return 1
  fi

  python3 scripts/retain_semantic_checkpoints.py \
    --root "$OR_ROOT" --retain-steps 20 --final-iteration 39 &
  retain_pid=$!
  if ! wait "$train_pid"; then
    kill "$retain_pid" 2>/dev/null || true
    wait "$retain_pid" 2>/dev/null || true
    return 1
  fi
  wait "$retain_pid"
}

attempt=0
while true; do
  attempt=$((attempt+1))
  if run_once; then
    echo "OR40 finished."
    exit 0
  fi
  echo "OR attempt $attempt failed; restarting from the latest committed checkpoint." >&2
  sleep 30
done
