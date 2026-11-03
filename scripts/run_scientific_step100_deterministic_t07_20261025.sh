#!/usr/bin/env bash
# Reproducible temperature-0.7 evaluation of the clean step-100 checkpoints.
# External tools remain live; only model/runtime sampling is deterministic.
set -uo pipefail

REPO=/root/openclaw-rl-local
LAUNCHER="$REPO/clawgym-official/.run_step39_eval_sync_20260812-091700.sh"
BASE=/mnt/humf/openclaw-rl-checkpoints
OUT="$BASE/scientific-eval-step100-deterministic-t07-20261025"
OLD="$BASE/scientific-eval-step100-20261024"
OR_ROOT="$BASE/clawgym-or100-clean-serper-20261012"
GRPO_ROOT="$BASE/clawgym-artifact-grpo100-clean-serper-v2-20261015"
PY="$REPO/.venv-openclaw/bin/python"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/queue.log") 2>&1

valid_result() {
  "$PY" - "$1" <<'PY'
import json,sys
o=json.load(open(sys.argv[1]))
m=o.get("metrics") or {}
ok=(o.get("exit_code")==0
    and m.get("eval/clawgym_eval-task_count")==200
    and m.get("eval/clawgym_eval-n_collected")==200
    and m.get("eval/clawgym_eval-n_skipped")==0)
raise SystemExit(0 if ok else 1)
PY
}

run_one() {
  local model="$1" seed="$2" root="$3"
  local iter=99 tag="${model}-seed${seed}" out="$OUT/${model}-seed${seed}"
  local result="$out/result.json"
  if [[ -f "$result" ]] && valid_result "$result"; then
    echo "SKIP complete $tag"
    return 0
  fi
  local model_offset=0
  [[ "$model" == "grpoingrpo" ]] && model_offset=200
  local attempt
  for attempt in 1 2; do
    local run="$out/attempt${attempt}"
    mkdir -p "$run/checkpoint_view" "$run/save"
    ln -sfn "$root/iter_0000099" "$run/checkpoint_view/iter_0000099"
    printf '99\n' > "$run/checkpoint_view/latest_checkpointed_iteration.txt"
    echo "START $tag attempt=$attempt deterministic=1 temperature=0.7 at=$(date -u +%FT%TZ)"
    env \
      RUN_STAMP="scientific-step100-det-t07-20261025-${tag}-a${attempt}" \
      EXP_NAME="scientific-step100-det-t07-${tag}" \
      SAVE_CKPT="$run/save" \
      LOAD_CKPT="$run/checkpoint_view" \
      OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
      NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 \
      EVAL_INTERVAL=1000 SKIP_EVAL_BEFORE_TRAIN=0 \
      OPENCLAW_EVAL_NUM_WORKERS=8 OPENCLAW_EVAL_N_SAMPLES=1 \
      OPENCLAW_EVAL_TASKS="$REPO/clawgym-official/data/clawgym_eval" \
      OPENCLAW_TRAIN_NUM_WORKERS=8 OPENCLAW_TRAIN_SHUFFLE_SEED=0 \
      OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE="$seed" \
      OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=1 \
      SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 \
      MEGATRON_DETERMINISTIC_MODE=1 \
      PYTHONHASHSEED=0 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      CUDA_LAUNCH_BLOCKING=1 \
      NVIDIA_TF32_OVERRIDE=0 \
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
      NCCL_ALGO=Ring NCCL_PROTO=Simple \
      NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
      SGLANG_ENABLE_CUDA_GRAPH=0 \
      EVAL_TEMPERATURE=0.7 EVAL_TOP_P=1.0 EVAL_TOP_K=-1 \
      ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=1.0 ROLLOUT_TOP_K=-1 \
      EVAL_MAX_RESPONSE_LEN=8192 EVAL_MAX_CONTEXT_LEN=65536 \
      OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
      OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 OPENCLAW_CONTEXT_WINDOW=65536 \
      OPENCLAW_MAX_AGENT_STEPS=96 OPENCLAW_MAX_COMPACTION_EVENTS=8 \
      OPENCLAW_REMOTE_BASE="/root/clawgym_eval100_det_t07_${model}_seed${seed}_a${attempt}" \
      OPENCLAW_REMOTE_PROXY_PORT=$((59000 + model_offset + attempt * 100 + seed % 100)) \
      PORT=$((36700 + model_offset + attempt * 100 + seed % 100)) \
      OPENCLAW_GATEWAY_PORT_BASE=$((26000 + model_offset + attempt * 1000 + seed % 100)) \
      OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
      OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=8 \
      OPENCLAW_RECORD_FILE="$run/official_proxy_records.jsonl" \
      TENSORBOARD_DIR="$run/tensorboard" \
      bash "$LAUNCHER" > "$run/run.log" 2>&1
    local code=$?
    "$PY" - "$run/run.log" "$run/save/eval_artifacts/train_step_000000/results.jsonl" \
      "$model" "$seed" "$code" "$out" "$attempt" > "$result" <<'PY'
import ast,json,re,sys
from pathlib import Path
log=Path(sys.argv[1]); jsonl=Path(sys.argv[2])
text=log.read_text(errors="replace") if log.exists() else ""
metrics=None
for raw in text.splitlines():
    line=re.sub(r"\x1b\[[0-9;]*m","",raw)
    if "eval 0: " in line:
        try: metrics=ast.literal_eval(line.split("eval 0: ",1)[1])
        except Exception: pass
def usable(m):
    return (isinstance(m,dict)
            and m.get("eval/clawgym_eval-task_count")==200
            and m.get("eval/clawgym_eval-n_collected")==200
            and m.get("eval/clawgym_eval-n_skipped")==0)
recovered=False
if not usable(metrics) and jsonl.exists():
    rows=[json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
    done=[r for r in rows if r.get("status")=="completed" and r.get("reward") is not None]
    if len(rows)==200 and len(done)==200:
        rewards=[float(r["reward"]) for r in done]
        correct=sum(1 for r in done if r.get("correct"))
        avg=sum(rewards)/200
        metrics={
            "eval/clawgym_eval-avg_reward":avg,
            "eval/clawgym_eval-correct_count":correct,
            "eval/clawgym_eval-correct_rate":correct/200,
            "eval/clawgym_eval-task_count":200,
            "eval/clawgym_eval-n_collected":200,
            "eval/clawgym_eval-n_skipped":0,
            "eval/clawgym_eval-avg_reward_collected_only":avg,
            "eval/clawgym_eval":avg,
        }
        recovered=True
exit_code=0 if usable(metrics) else int(sys.argv[5])
print(json.dumps({
    "model":sys.argv[3],"seed":int(sys.argv[4]),"iteration":99,
    "exit_code":exit_code,"output_dir":sys.argv[6],"attempt":int(sys.argv[7]),
    "temperature":0.7,"deterministic_inference":True,
    "external_tools_frozen":False,"recovered_from_jsonl":recovered,"metrics":metrics,
},ensure_ascii=False))
PY
    echo "END $tag attempt=$attempt launcher_code=$code result=$(cat "$result") at=$(date -u +%FT%TZ)"
    if valid_result "$result"; then
      return 0
    fi
  done
  echo "FAILED $tag after 2 attempts" >&2
  return 1
}

failed=0
for seed in 1101 1102 1103 1104 1105; do
  run_one or "$seed" "$OR_ROOT" || failed=$((failed+1))
  run_one grpoingrpo "$seed" "$GRPO_ROOT" || failed=$((failed+1))
done

"$PY" - "$OUT" "$OLD" <<'PY'
import json,statistics,sys
from pathlib import Path
out=Path(sys.argv[1]); old=Path(sys.argv[2])
def collect(root):
    groups={}
    for p in sorted(root.glob("*/result.json")):
        try: o=json.loads(p.read_text())
        except Exception: continue
        m=o.get("metrics") or {}
        if (o.get("exit_code")==0 and m.get("eval/clawgym_eval-n_collected")==200
                and m.get("eval/clawgym_eval-n_skipped")==0):
            groups.setdefault(o["model"],{})[int(o["seed"])]=m["eval/clawgym_eval-avg_reward"]
    return groups
new=collect(out); previous=collect(old)
seeds=(1101,1102,1103,1104,1105)
lines=["| protocol | model | 1101 | 1102 | 1103 | 1104 | 1105 | mean |",
       "|---|---|---:|---:|---:|---:|---:|---:|"]
for protocol,groups in (("nondeterministic",previous),("deterministic",new)):
    for model in ("or","grpoingrpo"):
        scores=groups.get(model,{})
        vals=[scores[s] for s in seeds if s in scores]
        cells=" | ".join(f"{scores[s]:.6f}" if s in scores else "" for s in seeds)
        mean=f"{statistics.mean(vals):.6f}" if vals else ""
        lines.append(f"| {protocol} | {model} | {cells} | {mean} |")
text=("# Clean step-100 temperature-0.7 deterministic eval\n\n"
      "Model sampling is deterministic. External tools remain live and are not frozen.\n\n"
      +"\n".join(lines)+"\n")
(out/"aggregate_results.md").write_text(text)
print(text)
PY
echo "QUEUE DONE failed=$failed at=$(date -u +%FT%TZ)"
exit "$failed"
