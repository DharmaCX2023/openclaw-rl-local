#!/usr/bin/env bash
# Re-run Serper-failed eval tasks with the working key, then merge into 200-task scores.
set -uo pipefail

REPO=/root/openclaw-rl-local
LAUNCHER="$REPO/clawgym-official/.run_step39_eval_sync_20260812-091700.sh"
DET_LAUNCHER="$REPO/clawgym-official/.tmp_eval_deterministic_20261002.sh"
TASKS="$REPO/clawgym-official/data/clawgym_eval"
BASE=/mnt/humf/openclaw-rl-checkpoints
OUT="$BASE/serper-subset-rerun-20261011"
OR_ROOT="$BASE/clawgym-or40-20260929-or40-nonstrict-r6"
GRPO_ROOT="$BASE/clawgym-artifact-grpo40-20260923-strict2"
SFT_HF=/mnt/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127
PY="$REPO/.venv-openclaw/bin/python"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/queue.log") 2>&1

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
  local protocol="$1" model="$2" seed="$3" orig_jsonl="$4" log_root="$5" ckpt_root="$6" iter="$7" launcher="$8"
  local ids; ids="$(fail_ids "$log_root")"
  if [[ -z "$ids" ]]; then
    echo "SKIP no serper failures protocol=$protocol model=$model seed=$seed"
    return 0
  fi
  local tag="${protocol}-${model}-seed${seed}"
  local out="$OUT/runs/$tag"
  local merged="$OUT/merged/$tag/results.jsonl"
  if [[ -f "$out/result.json" && -f "$merged" ]]; then
    echo "SKIP already done $tag"
    return 0
  fi
  mkdir -p "$out/checkpoint_view" "$out/save"
  local map; map="$(index_map "$ids")"
  echo "START $tag ids=$ids at=$(date -u +%FT%TZ)"
  local load_args=()
  if [[ "$model" == "sft" ]]; then
    load_args=( HF_CKPT="$SFT_HF" REF_LOAD="$SFT_HF" )
  else
    ln -sfn "$ckpt_root/iter_$(printf '%07d' "$iter")" "$out/checkpoint_view/iter_$(printf '%07d' "$iter")"
    printf '%s\n' "$iter" > "$out/checkpoint_view/latest_checkpointed_iteration.txt"
    load_args=( LOAD_CKPT="$out/checkpoint_view" )
  fi
  local extra=()
  if [[ "$protocol" == "det" ]]; then
    extra=(
      OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE="$seed"
      OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=1
      SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
      MEGATRON_DETERMINISTIC_MODE=1
      PYTHONHASHSEED=0
      CUBLAS_WORKSPACE_CONFIG=:4096:8
      CUDA_LAUNCH_BLOCKING=1
      NVIDIA_TF32_OVERRIDE=0
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
      NCCL_ALGO=Ring NCCL_PROTO=Simple
      SGLANG_ENABLE_CUDA_GRAPH=0
      ROLLOUT_TEMPERATURE=0 ROLLOUT_TOP_P=1.0 ROLLOUT_TOP_K=1
      EVAL_TEMPERATURE=0 EVAL_TOP_P=1.0 EVAL_TOP_K=1
    )
  else
    extra=(
      OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE="$seed"
      OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=0
      SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0
      MEGATRON_DETERMINISTIC_MODE=0
      SGLANG_ENABLE_CUDA_GRAPH=0
      EVAL_TEMPERATURE=0.7 EVAL_TOP_P=1.0 EVAL_TOP_K=-1
      ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=1.0
    )
  fi
  env \
    RUN_STAMP="serper-rerun-20261011-${tag}" \
    EXP_NAME="serper-rerun-${tag}" \
    SAVE_CKPT="$out/save" \
    "${load_args[@]}" \
    OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
    NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 \
    EVAL_INTERVAL=1000 SKIP_EVAL_BEFORE_TRAIN=0 \
    OPENCLAW_EVAL_NUM_WORKERS=8 OPENCLAW_EVAL_N_SAMPLES=1 \
    OPENCLAW_EVAL_TASKS="$TASKS" \
    OPENCLAW_EVAL_TASK_IDS="$ids" \
    OPENCLAW_EVAL_TASK_INDEX_MAP="$map" \
    OPENCLAW_TRAIN_SHUFFLE_SEED=0 \
    OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
    OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 OPENCLAW_CONTEXT_WINDOW=65536 \
    OPENCLAW_MAX_AGENT_STEPS=96 OPENCLAW_MAX_COMPACTION_EVENTS=8 \
    OPENCLAW_REMOTE_PROXY_PORT=$((53000 + seed % 100)) \
    PORT=$((31600 + seed % 100)) \
    OPENCLAW_GATEWAY_PORT_BASE=$((24000 + seed % 1000)) \
    OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
    OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=8 OPENCLAW_REMOTE_SSH_POOL_SIZE=8 \
    OPENCLAW_TRAIN_NUM_WORKERS=8 \
    OPENCLAW_RECORD_FILE="$out/official_proxy_records.jsonl" \
    TENSORBOARD_DIR="$out/tensorboard" \
    "${extra[@]}" \
    bash "$launcher" > "$out/run.log" 2>&1
  local code=$?
  local new_jsonl
  new_jsonl="$(find "$out/save" -name results.jsonl | head -n1)"
  local merge_json='{}'
  if [[ -n "$new_jsonl" && -f "$orig_jsonl" ]]; then
    merge_json="$(merge_results "$orig_jsonl" "$new_jsonl" "$merged")"
  fi
  echo "{\"protocol\":\"$protocol\",\"model\":\"$model\",\"seed\":$seed,\"ids\":\"$ids\",\"exit_code\":$code,\"merge\":$merge_json}" > "$out/result.json"
  echo "END $tag code=$code merge=$merge_json at=$(date -u +%FT%TZ)"
}

# Scientific eval (temp 0.7): OR iter 39, GRPO-in-GRPO iter 40
SCI="$BASE/scientific-eval-or40-grpoingrpo40-20261007"
for seed in 1101 1102 1103 1104 1105; do
  run_subset sci or "$seed" \
    "$SCI/or-seed${seed}/save/eval_artifacts/train_step_000000/results.jsonl" \
    "$SCI/or-seed${seed}/save" "$OR_ROOT" 39 "$LAUNCHER"
done
for seed in 1101 1102 1103 1104 1105; do
  run_subset sci grpoingrpo "$seed" \
    "$SCI/grpoingrpo-seed${seed}/save/eval_artifacts/train_step_000000/results.jsonl" \
    "$SCI/grpoingrpo-seed${seed}/save" "$GRPO_ROOT" 40 "$LAUNCHER"
done

# Deterministic / greedy (temp 0) subset reruns are skipped on request.

"$PY" - "$OUT" <<'PY'
import json, statistics
from pathlib import Path
root=Path(__file__).resolve().parent if False else Path(__import__('sys').argv[1])
rows=[]
for p in sorted((root/'runs').glob('*/result.json')):
    try: rows.append(json.loads(p.read_text()))
    except Exception: pass
groups={}
for r in rows:
    if not isinstance(r.get('merge'), dict): continue
    if r.get('exit_code')!=0: continue
    key=(r['protocol'], r['model'])
    groups.setdefault(key, {})[r['seed']]=r['merge'].get('avg_reward')
lines=['# Serper subset rerun merged 200-task scores','']
for key in sorted(groups):
    d=groups[key]; vals=[d[s] for s in sorted(d) if d[s] is not None]
    if not vals: continue
    mean=statistics.mean(vals); sd=statistics.stdev(vals) if len(vals)>1 else 0
    lines.append(f"- {key[0]} {key[1]} n={len(vals)} mean={mean:.6f} sd={sd:.6f} by_seed={ {s:d[s] for s in sorted(d)} }")
(root/'aggregate.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
echo "QUEUE DONE at=$(date -u +%FT%TZ)"
