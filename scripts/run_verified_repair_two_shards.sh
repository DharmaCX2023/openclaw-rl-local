#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/openclaw-rl-local
OUT="${1:?run output required}"
[[ "$OUT" == "$ROOT/outputs/verified-repair-"* ]] || exit 1
PY="$ROOT/.venv-openclaw/bin/python"
FULL=pypenguin_phase57_full_burn
IDLE=verified_repair_data_idle_burn
FINISHED0=verified_repair_shard0_finished_burn
FINISHED1=verified_repair_shard1_finished_burn
P0=""; P1=""
mkdir -p "$OUT"
exec >>"$OUT/data-shards-supervisor-$(date -u +%Y%m%d-%H%M%S).log" 2>&1
cd "$ROOT"
[[ -f "$OUT/data-collect.json" && -f "$OUT/smoke_passed.json" ]] || exit 1
"$PY" scripts/merge_verified_repair_shards.py --root "$OUT" --prepare
# Read-only preflight before touching even our own burn processes. A stale
# collector or foreign listener must cause an abort, not a port takeover.
"$PY" - <<'PY'
import socket, subprocess
checks=[]
try:
    for port in (30970,30971,30972,30973):
        handle=socket.socket();checks.append(handle)
        handle.bind(('127.0.0.1',port))
    result=subprocess.run(['ssh','-i','/root/openclaw_cpu_key','-o','BatchMode=yes',
        '-o','ConnectTimeout=10','-o','ControlMaster=no','-o','ControlPath=none',
        'root@159.138.50.147',
        "ss -H -ltn '( sport = :39971 or sport = :39973 )'"],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    if result.returncode or result.stdout.strip():
        raise RuntimeError('remote preflight failed or experiment tunnel port occupied')
finally:
    for handle in checks:handle.close()
print('isolated_port_preflight=passed',flush=True)
PY

stop_burn() {
    local session="$1" start
    tmux has-session -t "$session" 2>/dev/null || return 0
    start="$(tmux list-panes -t "$session" -F '#{pane_start_command}')"
    [[ "$start" == *synthetic_train_stress.py* ]] || return 1
    tmux send-keys -t "$session" C-c
    for ((i=0;i<60;i++)); do
        tmux has-session -t "$session" 2>/dev/null || return 0
        sleep 1
    done
    return 1
}

cleanup() {
    local rc=$? pid args pending
    trap - EXIT INT TERM
    for pid in "$P0" "$P1"; do
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            args="$(ps -p "$pid" -o args=)"
            if [[ "$args" == *run_verified_repair_pilot.py* && "$args" == *"$OUT-shard-"* ]]; then
                kill -TERM "$pid"
            fi
        fi
    done
    for ((i=0;i<60;i++)); do
        pending=0
        for pid in "$P0" "$P1"; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then pending=1; fi
        done
        [[ "$pending" == 0 ]] && break
        sleep 1
    done
    stop_burn "$IDLE" || true
    stop_burn "$FINISHED0" || true
    stop_burn "$FINISHED1" || true
    for ((i=0;i<45;i++)); do
        [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
        sleep 1
    done
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
        tmux has-session -t "$FULL" 2>/dev/null || tmux new-session -d -s "$FULL" \
          "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --nproc_per_node=8 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>$OUT/data-shards-restored-full-burn.log 2>&1"
        echo 'restored_full_burn=1'
    else
        echo 'GPU processes remain: no overlapping burn started; ownership inspection required.'
    fi
    echo "supervisor_exit=$rc finished_utc=$(date -u +%FT%TZ)"
    exit "$rc"
}

trap cleanup EXIT INT TERM
stop_burn "$FULL"
for ((i=0;i<45;i++)); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
    sleep 1
done
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || {
    echo 'Refusing to overlap existing GPU processes'; exit 1;
}
tmux new-session -d -s "$IDLE" \
  "cd /root && exec env CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --nproc_per_node=6 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>$OUT/data-shards-idle-burn.log 2>&1"
mkdir -p "$OUT-shard-0" "$OUT-shard-1"
"$PY" -u scripts/run_verified_repair_pilot.py --stage data-branches \
  --source-output "$OUT" --output "$OUT-shard-0" --gpus 0 --workers 1 \
  --shard-index 0 --num-shards 2 --backend-port 30970 --proxy-port 30971 --reverse-port 39971 \
  >>"$OUT-shard-0/runner.log" 2>&1 &
P0=$!
"$PY" -u scripts/run_verified_repair_pilot.py --stage data-branches \
  --source-output "$OUT" --output "$OUT-shard-1" --gpus 1 --workers 1 \
  --shard-index 1 --num-shards 2 --backend-port 30972 --proxy-port 30973 --reverse-port 39973 \
  >>"$OUT-shard-1/runner.log" 2>&1 &
P1=$!
echo "shard_0_pid=$P0 shard_1_pid=$P1 started_utc=$(date -u +%FT%TZ)"
while kill -0 "$P0" 2>/dev/null || kill -0 "$P1" 2>/dev/null; do
    for slot in 0 1; do
        if [[ "$slot" == 0 ]]; then pid="$P0"; finished="$FINISHED0"; else pid="$P1"; finished="$FINISHED1"; fi
        if ! kill -0 "$pid" 2>/dev/null && ! tmux has-session -t "$finished" 2>/dev/null \
            && [[ -z "$(nvidia-smi -i "$slot" --query-compute-apps=pid --format=csv,noheader)" ]]; then
            tmux new-session -d -s "$finished" \
              "cd /root && exec env CUDA_VISIBLE_DEVICES=$slot /usr/local/bin/torchrun --standalone --nproc_per_node=1 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>$OUT/shard-$slot-finished-burn.log 2>&1"
        fi
    done
    sleep 20
done
set +e
wait "$P0"; R0=$?
wait "$P1"; R1=$?
set -e
echo "shard_0_exit=$R0 shard_1_exit=$R1"
[[ "$R0" == 0 && "$R1" == 0 ]] || exit 1
"$PY" scripts/merge_verified_repair_shards.py --root "$OUT"
