#!/usr/bin/env bash
set -uo pipefail

REPO=/root/openclaw-rl-local
OUT="$REPO/outputs/mosaic-phase0-32x8-seed1-20260822-023000"
STATUS="$OUT/orchestrator_status.txt"

mkdir -p "$OUT"
cd "$REPO"
set -a
source "$REPO/.env" 2>/dev/null || true
set +a

export SGLANG_SESSION_AFFINITY=1
export OPENCLAW_TRAIN_SCHEDULING=group_interleaved
export OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=32

rollout_rc=0
analysis_rc=not_run
"$REPO/.venv-openclaw/bin/python" scripts/run_mosaic_phase0_rollouts.py \
  --mosaic-archive-dir "$OUT/archives" \
  --output-dir "$OUT/gate" \
  --task-count 32 \
  --samples-per-task 8 \
  --parallel 32 \
  --allow-flexible-batch \
  --task-seed 1 \
  --gpus 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --backend-port 30580 \
  --proxy-port 30581 \
  --reverse-port 39581 \
  --gateway-port-base 37000 \
  --gateway-train-offset 1000 \
  --remote-base /root/clawgym_rl_mosaic_phase0_32x8_seed1 \
  --ssh-pool-size 8 \
  --rollout-timeout 1200 \
  --serve-timeout 900 \
  --deterministic-tool-call-ids \
  >> "$OUT/rollout.log" 2>&1 || rollout_rc=$?

if [[ "$rollout_rc" -eq 0 ]]; then
  analysis_rc=0
  "$REPO/.venv-openclaw/bin/python" scripts/analyze_mosaic_phase0.py \
    --gate-output-dir "$OUT/gate" \
    --archive-dir "$OUT/archives" \
    --output-dir "$OUT/analysis" \
    --beam-width 32 \
    --max-queries-per-task 128 \
    >> "$OUT/analysis.log" 2>&1 || analysis_rc=$?
fi

{
  printf 'rollout_rc=%s\n' "$rollout_rc"
  printf 'analysis_rc=%s\n' "$analysis_rc"
  printf 'finished_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$STATUS"

# This process intentionally becomes the all-GPU burner so the tmux session
# remains useful after either success or failure, with no idle-GPU gap.
exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /usr/local/bin/torchrun --standalone --nproc-per-node=8 \
  /root/synthetic_train_stress.py \
  --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 \
  --micro-batch-size 1 --seq-len 1024 --grad-accum 1 \
  --precision bf16 --lr 1e-4 --log-every 5 \
  >> "$OUT/post_experiment_burn.log" 2>&1
