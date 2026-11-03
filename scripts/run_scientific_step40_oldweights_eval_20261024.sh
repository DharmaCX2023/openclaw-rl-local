#!/usr/bin/env bash
# Same temperature-0.7 scientific eval as run_scientific_step40_clean_eval_20261024.sh.
# Only the checkpoints change: the weights behind the October 7 scores.
# OR loads iteration 39 of or40-nonstrict-r6. GRPO-in-GRPO loads iteration 40 of
# artifact-grpo40-strict2, which is the checkpoint that eval used, not iteration 39.
# Starts only after the clean step-40 queue has exited and default Ray is down.
# Checkpoint views are symlinks; training roots are not modified.
set -uo pipefail

REPO=/root/openclaw-rl-local
LAUNCHER="$REPO/clawgym-official/.run_step39_eval_sync_20260812-091700.sh"
BASE=/mnt/humf/openclaw-rl-checkpoints
OUT="$BASE/scientific-eval-step40-oldweights-20261024"
OR_ROOT="$BASE/clawgym-or40-20260929-or40-nonstrict-r6"
GRPO_ROOT="$BASE/clawgym-artifact-grpo40-20260923-strict2"
SFT_ROOT="$BASE/scientific-eval-sft-20261012"
CLEAN_SCRIPT="$REPO/scripts/run_scientific_step40_clean_eval_20261024.sh"
PY="$REPO/.venv-openclaw/bin/python"
RAY="$REPO/.venv-openclaw/bin/ray"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/queue.log") 2>&1

echo "WAIT clean step-40 eval at=$(date -u +%FT%TZ)"
while pgrep -f "$CLEAN_SCRIPT" >/dev/null 2>&1; do
  sleep 30
done
echo "CLEAN EVAL EXITED at=$(date -u +%FT%TZ)"
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

run_one() {
  local model="$1" seed="$2" root="$3" iter="$4"
  local tag="${model}-seed${seed}"
  local out="$OUT/$tag"
  local result="$out/result.json"
  if [[ -f "$result" ]] && "$PY" - "$result" <<'PY'
import json,sys
o=json.load(open(sys.argv[1]))
m=o.get("metrics") or {}
raise SystemExit(0 if o.get("exit_code")==0 and m.get("eval/clawgym_eval-task_count")==200 and m.get("eval/clawgym_eval-n_collected")==200 and m.get("eval/clawgym_eval-n_skipped")==0 else 1)
PY
  then
    echo "SKIP complete $tag"
    return 0
  fi
  local attempt
  for attempt in 1 2; do
    local run="$out/attempt${attempt}"
    mkdir -p "$run/checkpoint_view" "$run/save"
    ln -sfn "$root/iter_$(printf '%07d' "$iter")" "$run/checkpoint_view/iter_$(printf '%07d' "$iter")"
    printf '%s\n' "$iter" > "$run/checkpoint_view/latest_checkpointed_iteration.txt"
    echo "START $tag attempt=$attempt iter=$iter weight=$root at=$(date -u +%FT%TZ)"
    env \
      RUN_STAMP="scientific-step40-oldweights-20261024-${tag}-a${attempt}" \
      EXP_NAME="scientific-step40-oldweights-${tag}" \
      SAVE_CKPT="$run/save" \
      LOAD_CKPT="$run/checkpoint_view" \
      OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
      NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 \
      EVAL_INTERVAL=1000 SKIP_EVAL_BEFORE_TRAIN=0 \
      OPENCLAW_EVAL_NUM_WORKERS=8 OPENCLAW_EVAL_N_SAMPLES=1 \
      OPENCLAW_EVAL_TASKS="$REPO/clawgym-official/data/clawgym_eval" \
      OPENCLAW_TRAIN_SHUFFLE_SEED=0 \
      OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE="$seed" \
      OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=0 \
      SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0 \
      MEGATRON_DETERMINISTIC_MODE=0 \
      EVAL_TEMPERATURE=0.7 EVAL_TOP_P=1.0 EVAL_TOP_K=-1 \
      ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=1.0 \
      EVAL_MAX_RESPONSE_LEN=8192 EVAL_MAX_CONTEXT_LEN=65536 \
      OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
      OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 OPENCLAW_CONTEXT_WINDOW=65536 \
      OPENCLAW_MAX_AGENT_STEPS=96 OPENCLAW_MAX_COMPACTION_EVENTS=8 \
      OPENCLAW_REMOTE_BASE="/root/clawgym_eval40old_${model}_seed${seed}" \
      OPENCLAW_REMOTE_PROXY_PORT=$((56100 + seed % 100)) \
      PORT=$((33700 + seed % 100)) \
      OPENCLAW_GATEWAY_PORT_BASE=$((48000 + seed % 1000)) \
      OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
      OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=8 \
      OPENCLAW_RECORD_FILE="$out/official_proxy_records.jsonl" \
      TENSORBOARD_DIR="$out/tensorboard" \
      bash "$LAUNCHER" > "$run/run.log" 2>&1
    local code=$?
    "$PY" - "$run/run.log" "$model" "$seed" "$iter" "$code" "$out" > "$result" <<'PY'
import ast,json,re,sys
from pathlib import Path
text=Path(sys.argv[1]).read_text(errors="replace")
result=None
for raw in text.splitlines():
    line=re.sub(r"\x1b\[[0-9;]*m","",raw)
    if "eval 0: " in line:
        try: result=ast.literal_eval(line.split("eval 0: ",1)[1])
        except Exception: pass
print(json.dumps({"model":sys.argv[2],"seed":int(sys.argv[3]),"iteration":int(sys.argv[4]),"exit_code":int(sys.argv[5]),"output_dir":sys.argv[6],"metrics":result},ensure_ascii=False))
PY
    echo "END $tag attempt=$attempt code=$code result=$(cat "$result") at=$(date -u +%FT%TZ)"
    if "$PY" - "$result" <<'PY'
import json,sys
o=json.load(open(sys.argv[1]))
m=o.get("metrics") or {}
raise SystemExit(0 if o.get("exit_code")==0 and m.get("eval/clawgym_eval-n_collected")==200 and m.get("eval/clawgym_eval-n_skipped")==0 else 1)
PY
    then
      return 0
    fi
  done
  echo "FAILED $tag after 2 attempts" >&2
  return 1
}

failed=0
run_one or 1101 "$OR_ROOT" 39 || failed=$((failed+1))
run_one or 1102 "$OR_ROOT" 39 || failed=$((failed+1))
run_one or 1103 "$OR_ROOT" 39 || failed=$((failed+1))
run_one or 1104 "$OR_ROOT" 39 || failed=$((failed+1))
run_one or 1105 "$OR_ROOT" 39 || failed=$((failed+1))
run_one grpoingrpo 1101 "$GRPO_ROOT" 40 || failed=$((failed+1))
run_one grpoingrpo 1102 "$GRPO_ROOT" 40 || failed=$((failed+1))
run_one grpoingrpo 1103 "$GRPO_ROOT" 40 || failed=$((failed+1))
run_one grpoingrpo 1104 "$GRPO_ROOT" 40 || failed=$((failed+1))
run_one grpoingrpo 1105 "$GRPO_ROOT" 40 || failed=$((failed+1))

"$PY" - "$OUT" "$SFT_ROOT" <<'PY'
import json,statistics,sys
from pathlib import Path
out=Path(sys.argv[1]); sft=Path(sys.argv[2])
rows=[]
for p in sorted(out.glob("*/result.json")):
    try: rows.append(json.loads(p.read_text()))
    except Exception: pass
for p in sorted(sft.glob("sft-seed*/result.json")):
    try:
        row=json.loads(p.read_text())
        row["model"]="sft"
        row["iteration"]=None
        rows.append(row)
    except Exception: pass
def metric(row,name):
    return (row.get("metrics") or {}).get(name)
groups={}
for row in rows:
    if row.get("exit_code")!=0: continue
    if metric(row,"eval/clawgym_eval-n_collected")!=200: continue
    groups.setdefault(row["model"], {})[int(row["seed"])]=metric(row,"eval/clawgym_eval-avg_reward")
lines=["# Step 40 old-weight scientific eval","", "temperature 0.7, top-p 1.0, top-k -1, 200 tasks, n=1, seeds 1101-1105.", "OR iteration 39 of or40-nonstrict-r6. GRPO-in-GRPO iteration 40 of artifact-grpo40-strict2.", ""]
for model in ("sft","or","grpoingrpo"):
    scores=groups.get(model, {})
    vals=[scores[s] for s in (1101,1102,1103,1104,1105) if scores.get(s) is not None]
    cells=" | ".join(f"{scores[s]:.6f}" if scores.get(s) is not None else "" for s in (1101,1102,1103,1104,1105))
    mean=f"{statistics.mean(vals):.6f}" if vals else ""
    sd=f"{statistics.stdev(vals):.6f}" if len(vals)>1 else ""
    lines.append(f"| {model} | {cells} | {mean} | {sd} |")
text="| model | 1101 | 1102 | 1103 | 1104 | 1105 | mean | sample stdev |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"+"\n".join(lines[5:])+"\n"
(out/"aggregate_results.md").write_text("# Step 40 old-weight scientific eval\n\n"+text)
print("\n".join(lines))
PY
echo "QUEUE DONE failed=$failed at=$(date -u +%FT%TZ)"
exit "$failed"
