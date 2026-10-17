#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
TRAIN_ROOT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129"
RUN_TAG="20260828"
LOG="${TRAIN_ROOT}/logs/rerun_infra_affected_evals_${RUN_TAG}.log"
BURN_SESSION="gpu_burn_all8_after_infra_eval_reruns_${RUN_TAG}"

exec >>"${LOG}" 2>&1
echo "[$(date -u +%FT%TZ)] rerunning infrastructure-affected evals: 20 40 60"

run_and_validate() {
  local step="$1"
  local eval_root="${REPO_ROOT}/outputs/clawgym-step${step}-supplemental-eval-${RUN_TAG}"
  echo "[$(date -u +%FT%TZ)] starting step ${step}"
  EVAL_STEP="${step}" EVAL_RUN_TAG="${RUN_TAG}" \
    bash "${REPO_ROOT}/scripts/run_step80_supplemental_eval_20260827.sh"
  python3 - "${eval_root}/eval_artifacts/train_step_$(printf '%06d' "${step}")/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
summary = manifest["summaries"]["clawgym_eval"]
results = [json.loads(line) for line in Path(manifest["results_jsonl"]).read_text().splitlines() if line]
workspace_errors = [row for row in results if row.get("workspace", {}).get("error")]
missing_archives = [
    row for row in results
    if not Path(row.get("workspace", {}).get("archive", "")).is_file()
]
if summary["n_collected"] != 200 or summary["n_skipped"] != 0:
    raise SystemExit(f"incomplete eval summary: {summary}")
if len(results) != 200 or workspace_errors or missing_archives:
    raise SystemExit(
        f"incomplete evidence: rows={len(results)}, "
        f"workspace_errors={len(workspace_errors)}, missing_archives={len(missing_archives)}"
    )
print(
    f"validated step={manifest['semantic_train_step']} "
    f"reward={summary['average_reward']:.6f} "
    f"correct={summary['correct_count']}/200"
)
PY
  echo "[$(date -u +%FT%TZ)] step ${step} validated"
}

status=0
for step in 20 40 60; do
  if run_and_validate "${step}"; then
    :
  else
    status=$?
    echo "[$(date -u +%FT%TZ)] step ${step} failed validation; stopping rerun sequence"
    break
  fi
done
echo "[$(date -u +%FT%TZ)] rerun sequence finished with status ${status}"

# Restore burn even if a rerun fails; the monitor waits for all Ray/GPU
# processes to drain before allocating the cards.
exec env \
  TRAIN_SESSION="infra_eval_reruns_finished" \
  BURN_SESSION="${BURN_SESSION}" \
  CHECK_INTERVAL=60 \
  TRAIN_LOG="${LOG}" \
  MONITOR_LOG="${TRAIN_ROOT}/logs/post_infra_eval_reruns_burn_monitor.log" \
  BURN_LOG_ROOT="/root/logs/openclaw-burn/post-infra-eval-reruns-${RUN_TAG}" \
  bash "${REPO_ROOT}/scripts/monitor_formal_retrain_20260826.sh"
