#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/openclaw-rl-local
OUT="${1:?absolute run-owned output directory required}"
STAGE="${2:-smoke}"
WORKERS="${3:-2}"
SOURCE_OUT="${4:-}"
BACKEND_PORT="${VERIFIED_REPAIR_BACKEND_PORT:-31780}"
PROXY_PORT="${VERIFIED_REPAIR_PROXY_PORT:-31781}"
REVERSE_PORT="${VERIFIED_REPAIR_REVERSE_PORT:-39781}"
[[ "$OUT" == "$ROOT/outputs/verified-repair-"* && "$OUT" != *".."* ]] || exit 2
[[ "$STAGE" == smoke || "$STAGE" == data ]] || exit 2
[[ "$WORKERS" =~ ^[1-4]$ ]] || exit 2
EXTRA=()
if [[ -n "$SOURCE_OUT" ]]; then
    [[ "$SOURCE_OUT" == "$ROOT/outputs/verified-repair-"* && "$SOURCE_OUT" != *".."* ]] || exit 2
    EXTRA+=(--source-output "$SOURCE_OUT")
fi
FULL=pypenguin_phase57_full_burn
IDLE=verified_repair_v3_idle_burn
RUN_PID=""
mkdir -p "$OUT"
exec >>"$OUT/$STAGE-supervisor.log" 2>&1

stop_known_burn() {
    local session="$1" start
    tmux has-session -t "$session" 2>/dev/null || return 0
    start="$(tmux list-panes -t "$session" -F '#{pane_start_command}')"
    [[ "$start" == *"/root/synthetic_train_stress.py"* ]] || {
        echo "Unrecognized process; refusing to stop $session"; return 1;
    }
    tmux send-keys -t "$session" C-c
    for ((i=0;i<60;i++)); do
        tmux has-session -t "$session" 2>/dev/null || return 0
        sleep 1
    done
    echo "Known burn did not exit; refusing overlapping workload"
    return 1
}

restore() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "$RUN_PID" ]] && kill -0 "$RUN_PID" 2>/dev/null; then
        kill -TERM "$RUN_PID"
        for ((i=0;i<60;i++)); do
            kill -0 "$RUN_PID" 2>/dev/null || break
            sleep 1
        done
    fi
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
        echo "Remaining GPU processes: no overlapping burn launched; inspect ownership"
    fi
    echo "stage=$STAGE exit_code=$rc finished_utc=$(date -u +%FT%TZ)"
    exit "$rc"
}

trap restore EXIT INT TERM
stop_known_burn "$FULL"
for ((i=0;i<45;i++)); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
    sleep 1
done
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 3
tmux new-session -d -s "$IDLE" \
  "cd /root && exec env CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --nproc_per_node=7 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>$OUT/idle_burn.log 2>&1"
cd "$ROOT"
echo "stage=$STAGE started_utc=$(date -u +%FT%TZ)"
"$ROOT/.venv-openclaw/bin/python" -u scripts/run_verified_repair_replay_pilot.py \
    --output "$OUT" --stage "$STAGE" --gpus 0 --workers "$WORKERS" \
    --backend-port "$BACKEND_PORT" --proxy-port "$PROXY_PORT" --reverse-port "$REVERSE_PORT" "${EXTRA[@]}" &
RUN_PID=$!
wait "$RUN_PID"
RUN_PID=""
