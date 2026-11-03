#!/usr/bin/env bash
# Current scientific protocol on the 180-step mainline.
# Step 0 is the existing SFT eval (same weights, same protocol) and is not rerun.
# Semantic steps 40, 80, 120, 180 load iterations 39, 79, 119, 179.
# Views are symlinks. The training latest_checkpointed_iteration.txt stays at 179.
set -uo pipefail

REPO=/root/openclaw-rl-local
LAUNCHER="$REPO/clawgym-official/.run_step39_eval_sync_20260812-091700.sh"
BASE=/mnt/humf/openclaw-rl-checkpoints
OUT="$BASE/scientific-eval-mainline180-20261025"
CKPT_ROOT=/mnt/Openclaw-RL-hareness/outputs/clawgym-segment-r3-bkup-aligned-30b-sft11127-0-80-20260812-045034
SFT_ROOT="$BASE/scientific-eval-sft-20261012"
PY="$REPO/.venv-openclaw/bin/python"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/queue.log") 2>&1

valid_result() {
  "$PY" - "$1" <<'PY'
import json,sys
o=json.load(open(sys.argv[1]))
m=o.get("metrics") or {}
ok=o.get("exit_code")==0 and m.get("eval/clawgym_eval-task_count")==200 and m.get("eval/clawgym_eval-n_collected")==200 and m.get("eval/clawgym_eval-n_skipped")==0
raise SystemExit(0 if ok else 1)
PY
}

run_one() {
  local step="$1" seed="$2" iter="$3"
  local tag="step${step}-seed${seed}"
  local out="$OUT/$tag"
  local result="$out/result.json"
  if [[ -f "$result" ]] && valid_result "$result"; then
    echo "SKIP complete $tag"
    return 0
  fi
  local attempt
  for attempt in 1 2; do
    local run="$out/attempt${attempt}"
    mkdir -p "$run/checkpoint_view" "$run/save"
    ln -sfn "$CKPT_ROOT/iter_$(printf '%07d' "$iter")" "$run/checkpoint_view/iter_$(printf '%07d' "$iter")"
    printf '%s\n' "$iter" > "$run/checkpoint_view/latest_checkpointed_iteration.txt"
    local port_base=$((34600 + attempt * 100))
    local proxy_base=$((58000 + attempt * 200))
    local gateway_base=$((50000 + attempt * 2000))
    echo "START $tag attempt=$attempt iter=$iter at=$(date -u +%FT%TZ)"
    env \
      RUN_STAMP="scientific-mainline180-20261025-${tag}-a${attempt}" \
      EXP_NAME="scientific-mainline180-${tag}" \
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
      OPENCLAW_REMOTE_BASE="/root/clawgym_eval180_s${step}_seed${seed}_a${attempt}" \
      OPENCLAW_REMOTE_PROXY_PORT=$((proxy_base + seed % 100)) \
      PORT=$((port_base + seed % 100)) \
      OPENCLAW_GATEWAY_PORT_BASE=$((gateway_base + seed % 1000)) \
      OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
      OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=8 \
      OPENCLAW_RECORD_FILE="$out/official_proxy_records.jsonl" \
      TENSORBOARD_DIR="$out/tensorboard" \
      bash "$LAUNCHER" > "$run/run.log" 2>&1
    local code=$?
    "$PY" - "$run/run.log" "$run/save/eval_artifacts/train_step_000000/results.jsonl" "$step" "$seed" "$iter" "$code" "$out" "$attempt" > "$result" <<'PY'
import ast,json,re,sys
from pathlib import Path
log=Path(sys.argv[1])
jsonl=Path(sys.argv[2])
text=log.read_text(errors="replace") if log.exists() else ""
metrics=None
for raw in text.splitlines():
    line=re.sub(r"\x1b\[[0-9;]*m","",raw)
    if "eval 0: " in line:
        try: metrics=ast.literal_eval(line.split("eval 0: ",1)[1])
        except Exception: pass
recovered=False
def usable(m):
    return isinstance(m, dict) and m.get("eval/clawgym_eval-n_collected")==200 and m.get("eval/clawgym_eval-n_skipped")==0
if not usable(metrics) and jsonl.exists():
    rows=[json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
    done=[r for r in rows if r.get("status")=="completed" and r.get("reward") is not None]
    if len(rows)==200 and len(done)==200:
        rewards=[float(r["reward"]) for r in done]
        correct=sum(1 for r in done if r.get("correct"))
        avg=sum(rewards)/len(rewards)
        metrics={
            "eval/clawgym_eval-avg_reward": avg,
            "eval/clawgym_eval-correct_count": correct,
            "eval/clawgym_eval-correct_rate": correct/len(done),
            "eval/clawgym_eval-task_count": 200,
            "eval/clawgym_eval-n_collected": 200,
            "eval/clawgym_eval-n_skipped": 0,
            "eval/clawgym_eval-avg_reward_collected_only": avg,
            "eval/clawgym_eval": avg,
        }
        recovered=True
exit_code=0 if usable(metrics) else int(sys.argv[6])
print(json.dumps({
    "model":"mainline","semantic_step":int(sys.argv[3]),"seed":int(sys.argv[4]),
    "iteration":int(sys.argv[5]),"exit_code":exit_code,"output_dir":sys.argv[7],
    "attempt":int(sys.argv[8]),"recovered_from_jsonl":recovered,"metrics":metrics,
},ensure_ascii=False))
PY
    echo "END $tag attempt=$attempt code=$code result=$(cat "$result") at=$(date -u +%FT%TZ)"
    if valid_result "$result"; then
      return 0
    fi
    sleep 20
  done
  echo "FAILED $tag after 2 attempts" >&2
  return 1
}

failed=0
for step in 40 80 120 180; do
  if [[ -n "${ONLY_STEP:-}" && "$step" != "$ONLY_STEP" ]]; then
    continue
  fi
  iter=$((step - 1))
  for seed in 1101 1102 1103 1104 1105; do
    if [[ -n "${ONLY_SEED:-}" && "$seed" != "$ONLY_SEED" ]]; then
      continue
    fi
    run_one "$step" "$seed" "$iter" || failed=$((failed+1))
  done
done

"$PY" - "$OUT" "$SFT_ROOT" <<'PY'
import json,statistics,sys
from pathlib import Path
out=Path(sys.argv[1]); sft=Path(sys.argv[2])
rows=[]
for p in sorted(sft.glob("sft-seed*/result.json")):
    o=json.loads(p.read_text())
    m=o.get("metrics") or {}
    if o.get("exit_code")==0 and m.get("eval/clawgym_eval-n_collected")==200 and m.get("eval/clawgym_eval-n_skipped")==0:
        rows.append((0,int(o["seed"]),m["eval/clawgym_eval-avg_reward"]))
for p in sorted(out.glob("step*-seed*/result.json")):
    o=json.loads(p.read_text())
    m=o.get("metrics") or {}
    if o.get("exit_code")==0 and m.get("eval/clawgym_eval-n_collected")==200 and m.get("eval/clawgym_eval-n_skipped")==0:
        rows.append((int(o["semantic_step"]),int(o["seed"]),m["eval/clawgym_eval-avg_reward"]))
seeds=(1101,1102,1103,1104,1105)
steps=(0,40,80,120,180)
table={s:{} for s in steps}
for step,seed,score in rows:
    table[step][seed]=score
lines=["| step | 1101 | 1102 | 1103 | 1104 | 1105 | mean |","|---|---:|---:|---:|---:|---:|---:|"]
for step in steps:
    cells=[]
    vals=[]
    for seed in seeds:
        v=table[step].get(seed)
        if v is None:
            cells.append("")
        else:
            cells.append(f"{v:.6f}")
            vals.append(v)
    mean=f"{statistics.mean(vals):.6f}" if vals else ""
    lines.append(f"| {step} | {' | '.join(cells)} | {mean} |")
text="# Mainline 180 scientific eval\n\nTemperature 0.7, top-p 1.0, top-k -1, 200 tasks, n=1. Step 0 reuses the 2026-10-12 SFT eval.\n\n"+"\n".join(lines)+"\n"
(out/"aggregate_results.md").write_text(text)
print(text)
PY
echo "QUEUE DONE failed=$failed at=$(date -u +%FT%TZ)"
exit "$failed"
