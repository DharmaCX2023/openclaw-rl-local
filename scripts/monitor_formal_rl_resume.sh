#!/bin/bash

set -euo pipefail

TRAIN_SESSION="${TRAIN_SESSION:-openclaw_rl_resume39}"
BURN_SESSION="${BURN_SESSION:-openclaw_gpu_burn}"
CHECK_INTERVAL="${CHECK_INTERVAL:-900}"
RUN_ROOT="/root/openclaw-rl-local/outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700"
TRAIN_LOG="${RUN_ROOT}/logs/train_resume39-20260810-210000.log"
MONITOR_LOG="${RUN_ROOT}/logs/monitor_resume39.log"
BURN_LOG="/root/logs/openclaw-burn/openclaw-burn-gpu0-7-after-rl.log"
REMOTE_HOST="159.138.50.147"
REMOTE_SSH_KEY="/root/openclaw_cpu_key"
REMOTE_RUN_PREFIX="clawgym-rl-1786366907-4033979"

mkdir -p "$(dirname -- "${BURN_LOG}")"

log_snapshot() {
  local tracker="missing"
  if [ -r "${RUN_ROOT}/latest_checkpointed_iteration.txt" ]; then
    tracker="$(<"${RUN_ROOT}/latest_checkpointed_iteration.txt")"
  fi

  {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "train_session_alive=$1 tracker=${tracker}"
    stat -c 'train_log_size=%s train_log_mtime=%y' "${TRAIN_LOG}" 2>/dev/null || true
    df -h /root /mnt/humf | tail -n +2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader
    if [ -r /sys/fs/cgroup/memory/memory.oom_control ]; then
      tr '\n' ' ' </sys/fs/cgroup/memory/memory.oom_control
      echo
    fi
    if [ -r /sys/fs/cgroup/memory/memory.failcnt ]; then
      echo "memory_failcnt=$(</sys/fs/cgroup/memory/memory.failcnt)"
    fi
    timeout 20 ssh -i "${REMOTE_SSH_KEY}" -o BatchMode=yes -o ConnectTimeout=8 \
      "root@${REMOTE_HOST}" \
      "docker ps -a --format '{{.Names}} {{.Status}}' | awk '/^${REMOTE_RUN_PREFIX}-/{if (/unhealthy/) u++; else if (/healthy/) h++; else if (/Exited/) e++; else s++} END{print \"remote_ours healthy=\"h+0,\"unhealthy=\"u+0,\"exited=\"e+0,\"starting_or_running=\"s+0}'; df -h /root/clawgym_rl_runs | tail -n 1" \
      2>&1 || true
    rg -n 'perf [0-9]+:|rollout [0-9]+:|eval [0-9]+:|Traceback|CUDA out of memory|OutOfMemory|NCCL.*(error|failed)|GCS may have been killed' \
      "${TRAIN_LOG}" 2>/dev/null | tail -n 8 || true
  } >>"${MONITOR_LOG}" 2>&1
}

final_checkpoint_is_complete() {
  local checkpoint="${RUN_ROOT}/iter_0000099"
  local tracker_file="${RUN_ROOT}/latest_checkpointed_iteration.txt"
  [ -r "${tracker_file}" ] || return 1
  [ "$(<"${tracker_file}")" = "99" ] || return 1
  [ -s "${checkpoint}/.metadata" ] || return 1
  [ -s "${checkpoint}/common.pt" ] || return 1
  [ "$(find -L "${checkpoint}" -maxdepth 1 -type f -name '*.distcp' -size +49G | wc -l)" -eq 8 ]
}

start_burn_if_needed() {
  if tmux has-session -t "${BURN_SESSION}" 2>/dev/null; then
    return 0
  fi

  tmux new-session -d -s "${BURN_SESSION}" -c /root \
    "exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --nproc-per-node=8 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --log-every 5 >>${BURN_LOG} 2>&1"
}

while true; do
  if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    log_snapshot 1
  else
    log_snapshot 0
    if final_checkpoint_is_complete; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) final checkpoint verified; starting 8-GPU burn" \
        >>"${MONITOR_LOG}"
      start_burn_if_needed
      exit 0
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) training stopped before a complete iter_0000099" \
      >>"${MONITOR_LOG}"
  fi
  sleep "${CHECK_INTERVAL}"
done
