#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
TRAIN_ROOT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129"
RUN_TAG="20260828"
LOG="${TRAIN_ROOT}/logs/rerun_infra_affected_eval_subsets_${RUN_TAG}.log"

exec >>"${LOG}" 2>&1
echo "[$(date -u +%FT%TZ)] subset reruns: step20=1, step40=2, step60=4"

run_subset() {
  local step="$1"
  local task_ids="$2"
  local index_map="$3"
  local expected="$4"
  local eval_root="${REPO_ROOT}/outputs/clawgym-step${step}-infra-subset-eval-${RUN_TAG}"
  echo "[$(date -u +%FT%TZ)] starting step ${step}: ${task_ids}"
  if ! EVAL_STEP="${step}" \
    EVAL_RUN_TAG="${RUN_TAG}" \
    EVAL_ROOT="${eval_root}" \
    OPENCLAW_EVAL_TASK_IDS="${task_ids}" \
    OPENCLAW_EVAL_TASK_INDEX_MAP="${index_map}" \
      bash "${REPO_ROOT}/scripts/run_step80_supplemental_eval_20260827.sh"; then
    echo "[$(date -u +%FT%TZ)] step ${step} subset runner failed"
    return 1
  fi

  if ! python3 - "${eval_root}/eval_artifacts/train_step_$(printf '%06d' "${step}")/manifest.json" "${expected}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
summary = manifest["summaries"]["clawgym_eval"]
rows = [json.loads(line) for line in Path(manifest["results_jsonl"]).read_text().splitlines() if line]
bad = [row for row in rows if row.get("status") != "completed"]
workspace_errors = [row for row in rows if row.get("workspace", {}).get("error")]
missing_archives = [row for row in rows if not Path(row.get("workspace", {}).get("archive", "")).is_file()]
if summary["n_collected"] != expected or summary["n_skipped"] != 0:
    raise SystemExit(f"incomplete subset summary: {summary}")
if len(rows) != expected or bad or workspace_errors or missing_archives:
    raise SystemExit(
        f"incomplete subset evidence: rows={len(rows)}, bad={len(bad)}, "
        f"workspace_errors={len(workspace_errors)}, missing_archives={len(missing_archives)}"
    )
print(f"validated subset step={manifest['semantic_train_step']} rows={len(rows)}")
PY
  then
    echo "[$(date -u +%FT%TZ)] step ${step} subset validation failed"
    return 1
  fi
}

status=0
run_subset 20 'task_0190' '{"task_0190":190}' 1 || status=$?
if (( status == 0 )); then
  run_subset 40 'task_0104,task_0165' '{"task_0104":104,"task_0165":165}' 2 || status=$?
fi
if (( status == 0 )); then
  run_subset 60 'task_0045,task_0119,task_0138,task_0140' \
    '{"task_0045":45,"task_0119":119,"task_0138":138,"task_0140":140}' 4 || status=$?
fi

if (( status == 0 )); then
  python3 "${REPO_ROOT}/scripts/merge_clean_eval_results_20260828.py"
fi
echo "[$(date -u +%FT%TZ)] subset reruns finished with status ${status}"

exec env \
  TRAIN_SESSION="infra_eval_subset_reruns_finished" \
  BURN_SESSION="gpu_burn_all8_after_infra_eval_subset_reruns_${RUN_TAG}" \
  CHECK_INTERVAL=60 \
  TRAIN_LOG="${LOG}" \
  MONITOR_LOG="${TRAIN_ROOT}/logs/post_infra_eval_subset_reruns_burn_monitor.log" \
  BURN_LOG_ROOT="/root/logs/openclaw-burn/post-infra-eval-subset-reruns-${RUN_TAG}" \
  bash "${REPO_ROOT}/scripts/monitor_formal_retrain_20260826.sh"
