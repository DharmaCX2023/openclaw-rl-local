#!/usr/bin/env bash
set -uo pipefail

REPO=/root/openclaw-rl-local
LAUNCHER="$REPO/clawgym-official/.run_step39_eval_sync_20260812-091700.sh"
BASE=/mnt/humf/openclaw-rl-checkpoints
OUT="$BASE/scientific-eval-sft-20261012"
SEEDS=(1101 1102 1103 1104 1105)
TASKS="$REPO/clawgym-official/data/clawgym_eval"
HF_CKPT=/mnt/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127
mkdir -p "$OUT"
exec > >(tee -a "$OUT/queue.log") 2>&1

run_one() {
  local seed="$1"
  local tag="sft-seed${seed}"
  local out="$OUT/$tag"
  mkdir -p "$out"
  echo "START model=sft seed=$seed out=$out at=$(date -u +%FT%TZ)"
  env \
    RUN_STAMP="scientific-sft-20261012-${tag}" \
    EXP_NAME="scientific-${tag}" \
    SAVE_CKPT="$out/save" \
    HF_CKPT="$HF_CKPT" \
    REF_LOAD="$HF_CKPT" \
    OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
    NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 \
    EVAL_INTERVAL=1000 SKIP_EVAL_BEFORE_TRAIN=0 \
    OPENCLAW_EVAL_NUM_WORKERS=8 OPENCLAW_EVAL_N_SAMPLES=1 \
    OPENCLAW_EVAL_TASKS="$TASKS" \
    OPENCLAW_TRAIN_SHUFFLE_SEED=0 \
    OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE="$seed" \
    OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=0 \
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0 \
    MEGATRON_DETERMINISTIC_MODE=0 \
    EVAL_TEMPERATURE=0.7 EVAL_TOP_P=1.0 EVAL_TOP_K=-1 \
    EVAL_MAX_RESPONSE_LEN=8192 EVAL_MAX_CONTEXT_LEN=65536 \
    OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
    OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 OPENCLAW_CONTEXT_WINDOW=65536 \
    OPENCLAW_MAX_AGENT_STEPS=96 OPENCLAW_MAX_COMPACTION_EVENTS=8 \
    OPENCLAW_REMOTE_PROXY_PORT=$((52100 + seed % 100)) \
    PORT=$((30700 + seed % 100)) \
    OPENCLAW_GATEWAY_PORT_BASE=$((23000 + seed % 1000)) \
    OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
    OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=8 OPENCLAW_REMOTE_SSH_POOL_SIZE=8 \
    OPENCLAW_TRAIN_NUM_WORKERS=8 \
    OPENCLAW_RECORD_FILE="$out/official_proxy_records.jsonl" \
    TENSORBOARD_DIR="$out/tensorboard" \
    bash "$LAUNCHER" > "$out/run.log" 2>&1
  local code=$?
  "$REPO/.venv-openclaw/bin/python" - "$out/run.log" "$seed" "$code" "$out" > "$out/result.json" <<'PY'
import ast,json,re,sys
from pathlib import Path
p=Path(sys.argv[1]); result=None; text=p.read_text(errors='replace')
for raw in text.splitlines():
    line=re.sub(r'\x1b\[[0-9;]*m','',raw)
    if 'eval 0: ' in line:
        try: result=ast.literal_eval(line.split('eval 0: ',1)[1])
        except Exception: pass
obj={'model':'sft','seed':int(sys.argv[2]),'exit_code':int(sys.argv[3]),'output_dir':sys.argv[4],'metrics':result}
print(json.dumps(obj,ensure_ascii=False))
PY
  echo "END model=sft seed=$seed code=$code result=$(cat "$out/result.json") at=$(date -u +%FT%TZ)"
}

for seed in "${SEEDS[@]}"; do run_one "$seed"; done

"$REPO/.venv-openclaw/bin/python" - "$OUT" <<'PY'
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for p in sorted(root.glob('*/result.json')):
    try: rows.append(json.loads(p.read_text()))
    except Exception: pass
def metric(r,name): return (r.get('metrics') or {}).get(name)
scores={str(r['seed']):metric(r,'eval/clawgym_eval-avg_reward') for r in rows}
vals=[float(scores[s]) for s in ['1101','1102','1103','1104','1105'] if scores.get(s) is not None]
report={'protocol':{'seeds':[1101,1102,1103,1104,1105],'temperature':.7,'top_p':1.0,'top_k':-1},'rows':rows,
        'mean':statistics.mean(vals) if vals else None,
        'sample_stdev':statistics.stdev(vals) if len(vals)>1 else None,
        'n':len(vals),'scores_by_seed':scores}
(root/'aggregate_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
f=lambda v:'' if v is None else f'{float(v):.6f}'
line='| sft | '+' | '.join(f(scores.get(str(s))) for s in [1101,1102,1103,1104,1105])+' | '+f(report['mean'])+' | '+f(report['sample_stdev'])+' |'
(root/'aggregate_results.md').write_text('# Scientific SFT evaluation\n\n| model | 1101 | 1102 | 1103 | 1104 | 1105 | mean | sample stdev |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+line+'\n')
print(root/'aggregate_results.md')
PY
echo "QUEUE DONE at=$(date -u +%FT%TZ)"
