#!/usr/bin/env bash
set -euo pipefail
cd /root/openclaw-rl-local

set -a
# shellcheck disable=SC1091
source /root/openclaw-rl-local/.env
set +a

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

run_arm_once() {
  local arm="$1" root="$2" train_pid retain_pid
  serper_preflight
  if [ -f "$root/latest_checkpointed_iteration.txt" ]; then
    echo "Resuming $arm from committed iteration $(cat "$root/latest_checkpointed_iteration.txt")."
    CLEAN_RESUME=1 CLEAN_SAVE_CKPT="$root" \
      bash scripts/run_clean_grpo100_20261012.sh "$arm" &
  else
    CLEAN_SAVE_CKPT="$root" bash scripts/run_clean_grpo100_20261012.sh "$arm" &
  fi
  train_pid=$!

  # Let the launcher complete its new/empty-directory guard before the
  # retention monitor writes status into the checkpoint root.
  for _ in $(seq 1 120); do
    if [ -d "$root/logs" ]; then
      break
    fi
    if ! kill -0 "$train_pid" 2>/dev/null; then
      wait "$train_pid"
    fi
    sleep 1
  done
  if [ ! -d "$root/logs" ]; then
    kill "$train_pid" 2>/dev/null || true
    wait "$train_pid" 2>/dev/null || true
    echo "Training launcher did not initialize $root" >&2
    return 1
  fi

  python3 scripts/retain_semantic_checkpoints.py \
    --root "$root" --retain-steps 20 --final-iteration 99 &
  retain_pid=$!
  if ! wait "$train_pid"; then
    kill "$retain_pid" 2>/dev/null || true
    wait "$retain_pid" 2>/dev/null || true
    return 1
  fi
  wait "$retain_pid"
}

run_arm() {
  local arm="$1" root="$2" attempt=0
  while true; do
    attempt=$((attempt+1))
    if run_arm_once "$arm" "$root"; then
      return 0
    fi
    echo "$arm attempt $attempt failed; restarting from the latest committed checkpoint." >&2
    sleep 30
  done
}

ARTIFACT_ROOT=/mnt/humf/openclaw-rl-checkpoints/clawgym-artifact-grpo100-clean-serper-v2-20261015
OR_ROOT=/mnt/humf/openclaw-rl-checkpoints/clawgym-or100-clean-serper-20261012

run_arm artifact "$ARTIFACT_ROOT"
run_arm or "$OR_ROOT"
