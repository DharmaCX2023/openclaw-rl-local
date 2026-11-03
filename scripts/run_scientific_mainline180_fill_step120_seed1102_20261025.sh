#!/usr/bin/env bash
# Rerun the missing official score after the mainline queue exits.
set -uo pipefail
REPO=/root/openclaw-rl-local
OUT=/mnt/humf/openclaw-rl-checkpoints/scientific-eval-mainline180-20261025
PY="$REPO/.venv-openclaw/bin/python"
RAY="$REPO/.venv-openclaw/bin/ray"
MAIN="$REPO/scripts/run_scientific_mainline180_eval_20261025.sh"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/fill-step120-seed1102.log") 2>&1
echo "WAIT mainline queue at=$(date -u +%FT%TZ)"
while pgrep -f "$MAIN" >/dev/null 2>&1; do
  sleep 30
done
echo "MAINLINE EXITED at=$(date -u +%FT%TZ)"
for _ in $(seq 1 30); do
  if ! "$PY" "$RAY" status --address="127.0.0.1:6379" >/dev/null 2>&1; then
    break
  fi
  echo "WAIT default Ray still up at=$(date -u +%FT%TZ)"
  sleep 20
done
if "$PY" "$RAY" status --address="127.0.0.1:6379" >/dev/null 2>&1; then
  echo "REFUSING to start: default Ray is still active" >&2
  exit 1
fi
ONLY_STEP=120 ONLY_SEED=1102 bash "$MAIN"
