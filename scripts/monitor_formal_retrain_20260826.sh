#!/usr/bin/env bash

set -euo pipefail

TRAIN_SESSION="${TRAIN_SESSION:-formal_rl_retrain_retry1_20260826}"
BURN_SESSION="${BURN_SESSION:-gpu_burn_all8_after_formal_retrain_20260826}"
CHECK_INTERVAL="${CHECK_INTERVAL:-600}"
RUN_ROOT="${RUN_ROOT:-/root/openclaw-rl-local/outputs/clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129}"
TRAIN_LOG="${TRAIN_LOG:-${RUN_ROOT}/logs/train_20260826-0129.log}"
MONITOR_LOG="${MONITOR_LOG:-${RUN_ROOT}/logs/health_monitor.log}"
BURN_LOG_ROOT="${BURN_LOG_ROOT:-/root/logs/openclaw-burn/formal-retrain-20260826}"
REMOTE_HOST="${REMOTE_HOST:-159.138.50.147}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-/root/openclaw_cpu_key}"
REMOTE_RUN_PREFIX="${REMOTE_RUN_PREFIX:-clawgym-rl-1787679120-1871232}"

mkdir -p "$(dirname -- "${MONITOR_LOG}")" "${BURN_LOG_ROOT}"

count_matches() {
  local pattern="$1"
  local file="$2"
  rg -c -- "${pattern}" "${file}" 2>/dev/null || echo 0
}

snapshot() {
  local alive="$1"
  local tracker="missing"
  if [[ -r "${RUN_ROOT}/latest_checkpointed_iteration.txt" ]]; then
    tracker="$(tr -d '\n' <"${RUN_ROOT}/latest_checkpointed_iteration.txt")"
  fi

  {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    echo "train_session_alive=${alive} tracker=${tracker}"
    echo "eval_rewards=$(count_matches '\[ClawGym-RL\] eval task .*reward=' "${TRAIN_LOG}")"
    echo "verifier_failures=$(count_matches 'reward\.sh failed for task_' "${TRAIN_LOG}")"
    echo "python_tracebacks=$(count_matches 'Traceback \(most recent call last\)' "${TRAIN_LOG}")"
    echo "critical_runtime_errors=$(count_matches 'RayTaskError|CUDA out of memory|OutOfMemory|NCCL.*(error|failed)|GCS may have been killed|SGLang(Engine| server).*(crashed|died|exited unexpectedly)' "${TRAIN_LOG}")"
    stat -c 'train_log_size=%s train_log_mtime=%y' "${TRAIN_LOG}" 2>/dev/null || true
    df -h /root | tail -n 1
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader
    timeout 20 ssh -i "${REMOTE_SSH_KEY}" -o BatchMode=yes -o ConnectTimeout=8 \
      "root@${REMOTE_HOST}" \
      "docker ps -a --format '{{.Names}} {{.Status}}' | awk '/^${REMOTE_RUN_PREFIX}-/{if (/unhealthy/) u++; else if (/healthy/) h++; else if (/Exited/) e++; else s++} END{print \"remote_ours healthy=\"h+0,\"unhealthy=\"u+0,\"exited=\"e+0,\"starting_or_running=\"s+0}'; free -h | sed -n '2p'; df -h /root/clawgym_rl_runs | tail -n 1" \
      2>&1 || true
    rg -n 'perf [0-9]+:|rollout [0-9]+:|eval [0-9]+:|Traceback|CUDA out of memory|RayTaskError|NCCL.*(error|failed)' \
      "${TRAIN_LOG}" 2>/dev/null | tail -n 12 || true
  } >>"${MONITOR_LOG}" 2>&1
}

final_checkpoint_is_complete() {
  local checkpoint="${RUN_ROOT}/iter_0000099"
  local tracker_file="${RUN_ROOT}/latest_checkpointed_iteration.txt"
  [[ -r "${tracker_file}" ]] || return 1
  [[ "$(tr -d '\n' <"${tracker_file}")" == "99" ]] || return 1
  [[ -s "${checkpoint}/.metadata" ]] || return 1
  [[ -s "${checkpoint}/common.pt" ]] || return 1
  [[ "$(find -L "${checkpoint}" -maxdepth 1 -type f -name '*.distcp' -size +45G | wc -l)" -eq 8 ]]
}

all_gpus_idle() {
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]
}

start_burn_if_safe() {
  tmux has-session -t "${BURN_SESSION}" 2>/dev/null && return 0
  all_gpus_idle || return 1

  tmux new-session -d -s "${BURN_SESSION}" -n gpu0 -c /root \
    "exec env CUDA_VISIBLE_DEVICES=0 /usr/local/bin/torchrun --standalone --master_port=29600 --nproc-per-node=1 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 32 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>${BURN_LOG_ROOT}/gpu0.log 2>&1"
  local gpu
  for gpu in 1 2 3 4 5 6 7; do
    tmux new-window -t "${BURN_SESSION}" -n "gpu${gpu}" -c /root \
      "exec env CUDA_VISIBLE_DEVICES=${gpu} /usr/local/bin/torchrun --standalone --master_port=$((29600 + gpu)) --nproc-per-node=1 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 32 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>${BURN_LOG_ROOT}/gpu${gpu}.log 2>&1"
  done
}

while true; do
  if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    snapshot 1
  else
    snapshot 0
    if final_checkpoint_is_complete; then
      if start_burn_if_safe; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) final checkpoint verified; 8-GPU burn started" >>"${MONITOR_LOG}"
        exit 0
      fi
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) final checkpoint verified; waiting for all training GPU processes to exit" >>"${MONITOR_LOG}"
    else
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) training stopped without a complete iter_0000099; leaving GPUs untouched for diagnosis/resume" >>"${MONITOR_LOG}"
    fi
  fi
  sleep "${CHECK_INTERVAL}"
done
