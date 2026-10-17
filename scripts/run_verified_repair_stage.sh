#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/openclaw-rl-local
OUT="${2:-$ROOT/outputs/verified-repair-v2-20260907-pilot}"
[[ "$OUT" == "$ROOT/outputs/verified-repair-"* ]] || exit 1
STAGE="${1:-smoke-collect}"
WORKERS="${3:-2}"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || exit 1
STAMP="$(date -u +%Y%m%d-%H%M%S)"
FULL=pypenguin_phase57_full_burn
IDLE=verified_repair_idle_burn
LOG="$OUT/$STAGE-$STAMP.log"
mkdir -p "$OUT"
exec >>"$LOG" 2>&1

stop_known_burn() {
    local session="$1" start
    tmux has-session -t "$session" 2>/dev/null || return 0
    start="$(tmux list-panes -t "$session" -F '#{pane_start_command}')"
    [[ "$start" == *synthetic_train_stress.py* ]] || {
        echo "Refusing to stop unrecognized tmux command"; return 1;
    }
    tmux send-keys -t "$session" C-c
    for ((i=0;i<60;i++)); do
        tmux has-session -t "$session" 2>/dev/null || return 0
        sleep 1
    done
    echo "Burn did not stop cleanly; leaving it intact"; return 1
}

restore_burn() {
    local rc=$?
    trap - EXIT INT TERM
    stop_known_burn "$IDLE" || true
    for ((i=0;i<45;i++)); do
        [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
        sleep 1
    done
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
        tmux has-session -t "$FULL" 2>/dev/null || tmux new-session -d -s "$FULL" \
          "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --nproc_per_node=8 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>$OUT/restored_full_burn.log 2>&1"
        echo "restored_full_burn=1"
    else
        echo "GPU processes remain; not starting overlapping burn. Inspect process ownership."
    fi
    echo "stage=$STAGE exit_code=$rc finished_utc=$(date -u +%FT%TZ)"
    exit "$rc"
}

trap restore_burn EXIT INT TERM
stop_known_burn "$FULL"
for ((i=0;i<45;i++)); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
    sleep 1
done
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
    echo "Cannot acquire GPU lease without overlapping processes"; exit 1;
}
tmux new-session -d -s "$IDLE" \
  "cd /root && exec env CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --nproc_per_node=7 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>$OUT/idle_burn.log 2>&1"
cd "$ROOT"
echo "stage=$STAGE started_utc=$(date -u +%FT%TZ)"
if [[ "$STAGE" == gradient-smoke ]]; then
    CUDA_VISIBLE_DEVICES=0 "$ROOT/.venv-openclaw/bin/python" -u scripts/run_verified_repair_gradient_smoke.py --output "$OUT"
elif [[ "$STAGE" == sampler-endpoint ]]; then
    CUDA_VISIBLE_DEVICES=0 "$ROOT/.venv-openclaw/bin/python" -u scripts/diagnose_verified_repair_sampler_endpoint.py --root "$OUT" --device cuda
else
    "$ROOT/.venv-openclaw/bin/python" -u scripts/run_verified_repair_pilot.py \
        --stage "$STAGE" --output "$OUT" --gpus 0 --workers "$WORKERS"
fi
