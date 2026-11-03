#!/usr/bin/env bash
# Fill only the two skipped tasks from mainline step 120, seed 1102.
# Merges them into the 198 completed rows from attempt 2. Does not rerun the other tasks.
set -uo pipefail

REPO=/root/openclaw-rl-local
LAUNCHER="$REPO/clawgym-official/.run_step39_eval_sync_20260812-091700.sh"
BASE=/mnt/humf/openclaw-rl-checkpoints
OUT="$BASE/scientific-eval-mainline180-20261025"
PARENT="$OUT/step120-seed1102"
RUN="$PARENT/subset-task0117-task0139"
CKPT_ROOT=/mnt/Openclaw-RL-hareness/outputs/clawgym-segment-r3-bkup-aligned-30b-sft11127-0-80-20260812-045034
ORIG="$PARENT/attempt2/save/eval_artifacts/train_step_000000/results.jsonl"
MAIN="$REPO/scripts/run_scientific_mainline180_eval_20261025.sh"
SFT_ROOT="$BASE/scientific-eval-sft-20261012"
PY="$REPO/.venv-openclaw/bin/python"
RAY="$REPO/.venv-openclaw/bin/ray"
IDS="task_0117,task_0139"
ITER=119
SEED=1102
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

mkdir -p "$RUN/checkpoint_view" "$RUN/save"
ln -sfn "$CKPT_ROOT/iter_$(printf '%07d' "$ITER")" "$RUN/checkpoint_view/iter_$(printf '%07d' "$ITER")"
printf '%s\n' "$ITER" > "$RUN/checkpoint_view/latest_checkpointed_iteration.txt"
MAP="$("$PY" - "$IDS" <<'PY'
import json,sys
ids=[x for x in sys.argv[1].split(",") if x]
print(json.dumps({tid:int(tid.split("_")[1]) for tid in ids}))
PY
)"
echo "START subset ids=$IDS map=$MAP at=$(date -u +%FT%TZ)"
env \
  RUN_STAMP="scientific-mainline180-20261025-step120-seed1102-subset" \
  EXP_NAME="scientific-mainline180-step120-seed1102-subset" \
  SAVE_CKPT="$RUN/save" \
  LOAD_CKPT="$RUN/checkpoint_view" \
  OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
  NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 \
  EVAL_INTERVAL=1000 SKIP_EVAL_BEFORE_TRAIN=0 \
  OPENCLAW_EVAL_NUM_WORKERS=8 OPENCLAW_EVAL_N_SAMPLES=1 \
  OPENCLAW_EVAL_TASKS="$REPO/clawgym-official/data/clawgym_eval" \
  OPENCLAW_EVAL_TASK_IDS="$IDS" \
  OPENCLAW_EVAL_TASK_INDEX_MAP="$MAP" \
  OPENCLAW_TRAIN_SHUFFLE_SEED=0 \
  OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE="$SEED" \
  OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=0 \
  SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0 \
  MEGATRON_DETERMINISTIC_MODE=0 \
  EVAL_TEMPERATURE=0.7 EVAL_TOP_P=1.0 EVAL_TOP_K=-1 \
  ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=1.0 \
  EVAL_MAX_RESPONSE_LEN=8192 EVAL_MAX_CONTEXT_LEN=65536 \
  OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
  OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 OPENCLAW_CONTEXT_WINDOW=65536 \
  OPENCLAW_MAX_AGENT_STEPS=96 OPENCLAW_MAX_COMPACTION_EVENTS=8 \
  OPENCLAW_REMOTE_BASE="/root/clawgym_eval180_s120_seed1102_subset" \
  OPENCLAW_REMOTE_PROXY_PORT=59302 \
  PORT=35902 \
  OPENCLAW_GATEWAY_PORT_BASE=56102 \
  OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
  OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=8 \
  OPENCLAW_RECORD_FILE="$RUN/official_proxy_records.jsonl" \
  TENSORBOARD_DIR="$RUN/tensorboard" \
  bash "$LAUNCHER" > "$RUN/run.log" 2>&1
code=$?
new_jsonl="$(find "$RUN/save" -name results.jsonl | head -n 1)"
"$PY" - "$ORIG" "$new_jsonl" "$PARENT/result.json" "$code" "$OUT" "$SFT_ROOT" <<'PY'
import json, statistics, sys
from pathlib import Path
orig=Path(sys.argv[1])
new=Path(sys.argv[2]) if sys.argv[2] else None
result_path=Path(sys.argv[3])
code=int(sys.argv[4])
out=Path(sys.argv[5])
sft=Path(sys.argv[6])
old=[json.loads(l) for l in orig.read_text().splitlines() if l.strip()]
rep={}
if new and new.exists():
    rep={json.loads(l)["task_id"]: json.loads(l) for l in new.read_text().splitlines() if l.strip()}
wanted={"task_0117","task_0139"}
merged=[]
replaced=[]
for row in old:
    tid=row["task_id"]
    if tid in wanted and tid in rep and rep[tid].get("status")=="completed" and rep[tid].get("reward") is not None:
        item=dict(rep[tid])
        for key in ("flat_index","task_index","runtime_task_index"):
            if key in row:
                item[key]=row[key]
        merged.append(item)
        replaced.append(tid)
    else:
        merged.append(row)
done=[r for r in merged if r.get("status")=="completed" and r.get("reward") is not None]
ok=len(merged)==200 and len(done)==200 and set(replaced)==wanted
rewards=[float(r["reward"]) for r in done] if ok else []
correct=sum(1 for r in done if r.get("correct")) if ok else 0
avg=sum(rewards)/len(rewards) if ok else None
if ok:
    dest=orig.parent/"results.merged.jsonl"
    dest.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n" for r in merged))
metrics=None
if ok:
    metrics={
        "eval/clawgym_eval-avg_reward": avg,
        "eval/clawgym_eval-correct_count": correct,
        "eval/clawgym_eval-correct_rate": correct/200,
        "eval/clawgym_eval-task_count": 200,
        "eval/clawgym_eval-n_collected": 200,
        "eval/clawgym_eval-n_skipped": 0,
        "eval/clawgym_eval-avg_reward_collected_only": avg,
        "eval/clawgym_eval": avg,
    }
result_path.write_text(json.dumps({
    "model":"mainline","semantic_step":120,"seed":1102,"iteration":119,
    "exit_code":0 if ok else code,"output_dir":str(result_path.parent),
    "recovered_from":"attempt2-plus-subset","replaced_task_ids":replaced,
    "metrics":metrics,
}, ensure_ascii=False))
print(json.dumps({"ok":ok,"replaced":replaced,"avg":avg,"subset_exit":code}, ensure_ascii=False))
if not ok:
    raise SystemExit(1)
rows=[]
for p in sorted(sft.glob("sft-seed*/result.json")):
    o=json.loads(p.read_text()); m=o.get("metrics") or {}
    if o.get("exit_code")==0 and m.get("eval/clawgym_eval-n_collected")==200 and m.get("eval/clawgym_eval-n_skipped")==0:
        rows.append((0,int(o["seed"]),m["eval/clawgym_eval-avg_reward"]))
for p in sorted(out.glob("step*-seed*/result.json")):
    o=json.loads(p.read_text()); m=o.get("metrics") or {}
    if o.get("exit_code")==0 and m.get("eval/clawgym_eval-n_collected")==200 and m.get("eval/clawgym_eval-n_skipped")==0:
        rows.append((int(o["semantic_step"]),int(o["seed"]),m["eval/clawgym_eval-avg_reward"]))
seeds=(1101,1102,1103,1104,1105); steps=(0,40,80,120,180)
table={s:{} for s in steps}
for step,seed,score in rows:
    table.setdefault(step,{})[seed]=score
lines=["| step | 1101 | 1102 | 1103 | 1104 | 1105 | mean |","|---|---:|---:|---:|---:|---:|---:|"]
for step in steps:
    cells=[]; vals=[]
    for seed in seeds:
        v=table.get(step,{}).get(seed)
        if v is None: cells.append("")
        else:
            cells.append(f"{v:.6f}"); vals.append(v)
    mean=f"{statistics.mean(vals):.6f}" if vals else ""
    lines.append(f"| {step} | {' | '.join(cells)} | {mean} |")
text="# Mainline 180 scientific eval\n\nTemperature 0.7, top-p 1.0, top-k -1, 200 tasks, n=1. Step 0 reuses the 2026-10-12 SFT eval.\n\n"+"\n".join(lines)+"\n"
(out/"aggregate_results.md").write_text(text)
print(text)
PY
echo "SUBSET DONE code=$code at=$(date -u +%FT%TZ)"
