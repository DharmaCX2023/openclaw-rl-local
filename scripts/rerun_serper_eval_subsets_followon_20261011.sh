#!/usr/bin/env bash
# After the main 25-run queue, patch remaining official evals that hit HTTP 401.
set -uo pipefail

WAIT_PID="${SERPER_QUEUE_PID:-3028096}"
REPO=/root/openclaw-rl-local
LAUNCHER="$REPO/clawgym-official/.run_step39_eval_sync_20260812-091700.sh"
TASKS="$REPO/clawgym-official/data/clawgym_eval"
BASE=/mnt/humf/openclaw-rl-checkpoints
OUT="$BASE/serper-subset-rerun-20261011"
OR_ROOT="$BASE/clawgym-or40-20260929-or40-nonstrict-r6"
GRPO_ROOT="$BASE/clawgym-artifact-grpo40-20260923-strict2"
PY="$REPO/.venv-openclaw/bin/python"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/followon.log") 2>&1

echo "FOLLOWON waiting for pid=$WAIT_PID at=$(date -u +%FT%TZ)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "FOLLOWON parent queue finished at=$(date -u +%FT%TZ)"

fail_ids() {
  local log_root="$1"
  "$PY" - "$log_root" <<'PY'
import re, sys
from pathlib import Path
pat=re.compile(r'Serper request failed|credentials are not configured|invalid endpoint URL', re.I)
idpat=re.compile(r'instance-eval-(\d+)\.log$')
root=Path(sys.argv[1])
ids=[]
for p in sorted((root/'openclaw_logs'/'rollout_0').glob('instance-eval-*.log')):
    m=idpat.search(p.name)
    if not m: continue
    try: t=p.read_text(errors='replace')
    except Exception: continue
    if pat.search(t):
        ids.append(f'task_{int(m.group(1)):04d}')
print(','.join(ids))
PY
}

index_map() {
  local ids="$1"
  "$PY" - "$ids" <<'PY'
import json,sys
ids=[x for x in sys.argv[1].split(',') if x]
print(json.dumps({tid:int(tid.split('_')[1]) for tid in ids}))
PY
}

merge_results() {
  local orig="$1" new="$2" dest="$3"
  "$PY" - "$orig" "$new" "$dest" <<'PY'
import json,sys
from pathlib import Path
orig=Path(sys.argv[1]); new=Path(sys.argv[2]); dest=Path(sys.argv[3])
old=[json.loads(l) for l in orig.read_text().splitlines() if l]
rep={json.loads(l)['task_id']:json.loads(l) for l in new.read_text().splitlines() if l}
merged=[]
for row in old:
    tid=row['task_id']
    if tid in rep:
        r=dict(rep[tid])
        for k in ('flat_index','task_index','runtime_task_index'):
            if k in row: r[k]=row[k]
        r['replaced_serper_failure']=True
        merged.append(r)
    else:
        merged.append(row)
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(''.join(json.dumps(x, sort_keys=True)+'\n' for x in merged))
n=len(merged)
rew=sum(float(x['reward']) for x in merged)/n
cor=sum(bool(x.get('correct')) for x in merged)
print(json.dumps({'n':n,'avg_reward':rew,'correct_count':cor,'replaced':sorted(rep)}))
PY
}

run_subset() {
  local tag="$1" orig_jsonl="$2" log_root="$3" ckpt_root="$4" iter="$5" port_off="$6"
  local ids; ids="$(fail_ids "$log_root")"
  if [[ -z "$ids" ]]; then
    echo "SKIP no serper failures $tag"
    return 0
  fi
  local out="$OUT/runs/$tag"
  local merged="$OUT/merged/$tag/results.jsonl"
  if [[ -f "$out/result.json" && -f "$merged" ]]; then
    echo "SKIP already done $tag"
    return 0
  fi
  mkdir -p "$out/checkpoint_view" "$out/save"
  local map; map="$(index_map "$ids")"
  echo "START $tag ids=$ids at=$(date -u +%FT%TZ)"
  ln -sfn "$ckpt_root/iter_$(printf '%07d' "$iter")" "$out/checkpoint_view/iter_$(printf '%07d' "$iter")"
  printf '%s\n' "$iter" > "$out/checkpoint_view/latest_checkpointed_iteration.txt"
  env \
    RUN_STAMP="serper-rerun-20261011-${tag}" \
    EXP_NAME="serper-rerun-${tag}" \
    SAVE_CKPT="$out/save" \
    LOAD_CKPT="$out/checkpoint_view" \
    OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
    NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 \
    EVAL_INTERVAL=1000 SKIP_EVAL_BEFORE_TRAIN=0 \
    OPENCLAW_EVAL_NUM_WORKERS=8 OPENCLAW_EVAL_N_SAMPLES=1 \
    OPENCLAW_EVAL_TASKS="$TASKS" \
    OPENCLAW_EVAL_TASK_IDS="$ids" \
    OPENCLAW_EVAL_TASK_INDEX_MAP="$map" \
    OPENCLAW_TRAIN_SHUFFLE_SEED=0 \
    OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE=0 \
    OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=0 \
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0 \
    MEGATRON_DETERMINISTIC_MODE=0 \
    SGLANG_ENABLE_CUDA_GRAPH=0 \
    EVAL_TEMPERATURE=0.7 EVAL_TOP_P=1.0 EVAL_TOP_K=-1 \
    ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=1.0 \
    OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
    OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 OPENCLAW_CONTEXT_WINDOW=65536 \
    OPENCLAW_MAX_AGENT_STEPS=96 OPENCLAW_MAX_COMPACTION_EVENTS=8 \
    OPENCLAW_REMOTE_PROXY_PORT=$((54000 + port_off)) \
    PORT=$((31800 + port_off)) \
    OPENCLAW_GATEWAY_PORT_BASE=$((25000 + port_off * 20)) \
    OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
    OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=8 OPENCLAW_REMOTE_SSH_POOL_SIZE=8 \
    OPENCLAW_TRAIN_NUM_WORKERS=8 \
    OPENCLAW_RECORD_FILE="$out/official_proxy_records.jsonl" \
    TENSORBOARD_DIR="$out/tensorboard" \
    bash "$LAUNCHER" > "$out/run.log" 2>&1
  local code=$?
  local new_jsonl; new_jsonl="$(find "$out/save" -name results.jsonl | head -n1)"
  local merge_json='{}'
  if [[ -n "$new_jsonl" && -f "$orig_jsonl" ]]; then
    merge_json="$(merge_results "$orig_jsonl" "$new_jsonl" "$merged")"
  fi
  echo "{\"tag\":\"$tag\",\"ids\":\"$ids\",\"exit_code\":$code,\"merge\":$merge_json}" > "$out/result.json"
  echo "END $tag code=$code merge=$merge_json at=$(date -u +%FT%TZ)"
}

EVAL8="$GRPO_ROOT/eval8"
# Official 200-task attempts used in aggregate_results.json
run_subset eval8-r1a2 "$EVAL8/run1/attempt2/eval_artifacts/train_step_000000/results.jsonl" "$EVAL8/run1/attempt2" "$GRPO_ROOT" 39 1
run_subset eval8-r2a1 "$EVAL8/run2/attempt1/eval_artifacts/train_step_000000/results.jsonl" "$EVAL8/run2/attempt1" "$GRPO_ROOT" 39 2
run_subset eval8-r3a1 "$EVAL8/run3/attempt1/eval_artifacts/train_step_000000/results.jsonl" "$EVAL8/run3/attempt1" "$GRPO_ROOT" 39 3
run_subset eval8-r4a1 "$EVAL8/run4/attempt1/eval_artifacts/train_step_000000/results.jsonl" "$EVAL8/run4/attempt1" "$GRPO_ROOT" 39 4
run_subset eval8-r5a1 "$EVAL8/run5/attempt1/eval_artifacts/train_step_000000/results.jsonl" "$EVAL8/run5/attempt1" "$GRPO_ROOT" 39 5
run_subset eval8-r6a1 "$EVAL8/run6/attempt1/eval_artifacts/train_step_000000/results.jsonl" "$EVAL8/run6/attempt1" "$GRPO_ROOT" 39 6
run_subset eval8-r7a1 "$EVAL8/run7/attempt1/eval_artifacts/train_step_000000/results.jsonl" "$EVAL8/run7/attempt1" "$GRPO_ROOT" 39 7
run_subset eval8-r8a2 "$EVAL8/run8/attempt2/eval_artifacts/train_step_000000/results.jsonl" "$EVAL8/run8/attempt2" "$GRPO_ROOT" 39 8

EVAL3="$OR_ROOT/eval3"
run_subset or-eval3-r1a1 "$EVAL3/run1/attempt1/eval_artifacts/train_step_000000/results.jsonl" "$EVAL3/run1/attempt1" "$OR_ROOT" 39 11
run_subset or-eval3-r2a1 "$EVAL3/run2/attempt1/eval_artifacts/train_step_000000/results.jsonl" "$EVAL3/run2/attempt1" "$OR_ROOT" 39 12
run_subset or-eval3-r3a1 "$EVAL3/run3/attempt1/eval_artifacts/train_step_000000/results.jsonl" "$EVAL3/run3/attempt1" "$OR_ROOT" 39 13

echo "FOLLOWON DONE at=$(date -u +%FT%TZ)"
