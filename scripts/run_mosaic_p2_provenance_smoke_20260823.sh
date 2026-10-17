#!/usr/bin/env bash
set -uo pipefail

REPO=/root/openclaw-rl-local
OUT="$REPO/outputs/mosaic-p2-provenance-smoke-20260823-025918"
STATUS="$OUT/orchestrator_status.txt"

mkdir -p "$OUT"
cd "$REPO"
set -a
source "$REPO/.env" 2>/dev/null || true
set +a

export SGLANG_SESSION_AFFINITY=1
export OPENCLAW_TRAIN_SCHEDULING=group_interleaved
export OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4

rollout_rc=0
"$REPO/.venv-openclaw/bin/python" scripts/run_mosaic_phase0_rollouts.py \
  --mosaic-archive-dir "$OUT/archives" \
  --output-dir "$OUT/gate" \
  --task-count 2 \
  --samples-per-task 2 \
  --parallel 4 \
  --allow-flexible-batch \
  --task-seed 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --backend-port 30780 \
  --proxy-port 30781 \
  --reverse-port 39781 \
  --gateway-port-base 39000 \
  --gateway-train-offset 1000 \
  --remote-base /root/clawgym_rl_mosaic_p2_provenance_smoke_20260823 \
  --ssh-pool-size 4 \
  --rollout-timeout 1200 \
  --serve-timeout 900 \
  --deterministic-tool-call-ids \
  >> "$OUT/rollout.log" 2>&1 || rollout_rc=$?

{
  printf 'rollout_rc=%s\n' "$rollout_rc"
  printf 'finished_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$STATUS"

exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /usr/local/bin/torchrun --standalone --nproc-per-node=8 \
  /root/synthetic_train_stress.py \
  --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 \
  --micro-batch-size 1 --seq-len 1024 --grad-accum 1 \
  --precision bf16 --lr 1e-4 --log-every 5 \
  >> "$OUT/post_experiment_burn.log" 2>&1
