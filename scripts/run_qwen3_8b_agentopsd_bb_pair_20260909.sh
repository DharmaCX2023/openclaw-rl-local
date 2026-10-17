#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/openclaw-rl-local
ARM_SCRIPT="$ROOT/scripts/run_qwen3_8b_agentopsd_bb_arm_20260909.sh"
PAIR_STAMP="${PAIR_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
PAIR_ROOT="$ROOT/outputs/qwen3-8b-agentopsd-bb-pair-$PAIR_STAMP"
STATUS_FILE="$PAIR_ROOT/status.env"
BURN_SESSION=agentopsd_bb_postrun_all8_burn
BURN_LOG=/tmp/agentopsd_bb_postrun_all8_burn.log
GRPO_PID=""
OPSD_PID=""

[[ "$PAIR_STAMP" =~ ^[A-Za-z0-9._-]+$ ]]
mkdir -p "$PAIR_ROOT"
exec > >(tee -a "$PAIR_ROOT/supervisor.log") 2>&1

gpu_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d' | wc -l
}

stop_full_burn() {
  if ! tmux has-session -t "$BURN_SESSION" 2>/dev/null; then
    [[ "$(gpu_process_count)" -eq 0 ]]
    return
  fi
  local command
  command="$(tmux list-panes -t "$BURN_SESSION" -a -F '#{pane_start_command}')"
  [[ "$command" == *synthetic_train_stress.py* ]] || {
    echo "Refusing to stop non-burn tmux session: $BURN_SESSION" >&2
    return 1
  }
  tmux send-keys -t "$BURN_SESSION" C-c
  for _ in $(seq 1 30); do
    tmux has-session -t "$BURN_SESSION" 2>/dev/null || break
    sleep 1
  done
  tmux kill-session -t "$BURN_SESSION" 2>/dev/null || true
  for _ in $(seq 1 120); do
    [[ "$(gpu_process_count)" -eq 0 ]] && return
    sleep 1
  done
  echo "GPU burn did not release all devices" >&2
  return 1
}

start_full_burn() {
  tmux has-session -t "$BURN_SESSION" 2>/dev/null && return
  [[ "$(gpu_process_count)" -eq 0 ]] || {
    echo "GPU processes remain; refusing to overlap the burn workload" >&2
    return 1
  }
  tmux new-session -d -s "$BURN_SESSION" \
    "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29641 --nproc_per_node=8 synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>$BURN_LOG 2>&1"
  echo "full_burn_started=$BURN_SESSION"
}

terminate_arm() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return
    fi
    # A terminated direct child may remain as a zombie until the supervisor
    # reaps it; it has already released GPU/Ray resources.
    if [[ "$(ps -o stat= -p "$pid" 2>/dev/null)" == Z* ]]; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  terminate_arm "$GRPO_PID"
  terminate_arm "$OPSD_PID"
  if [[ "$(gpu_process_count)" -eq 0 ]]; then
    start_full_burn || true
  fi
  printf 'state=%s\nexit_code=%s\nfinished_utc=%s\npair_stamp=%s\n' \
    "$([[ $rc -eq 0 ]] && echo COMPLETED || echo FAILED)" "$rc" \
    "$(date -u +%FT%TZ)" "$PAIR_STAMP" >"$STATUS_FILE"
  exit "$rc"
}
trap cleanup EXIT INT TERM

if pgrep -af 'raylet|gcs_server|sglang.*launch_server|train_async.py' \
  | grep -v -E "pgrep -af|run_qwen3_8b_dual_pair" >/dev/null; then
  echo "Existing local Ray/SGLang training processes detected; refusing to overlap." >&2
  pgrep -af 'raylet|gcs_server|sglang.*launch_server|train_async.py' || true
  exit 1
fi

printf 'state=STARTING\nstarted_utc=%s\npair_stamp=%s\n' \
  "$(date -u +%FT%TZ)" "$PAIR_STAMP" >"$STATUS_FILE"
stop_full_burn

echo "Launching AgentOPSD on physical GPUs 0-3"
setsid env PAIR_STAMP="$PAIR_STAMP" SKIP_EVAL_BEFORE_TRAIN=1 OPENCLAW_TRAIN_NUM_WORKERS=32 OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4 bash "$ARM_SCRIPT" agentopsd &
GRPO_PID=$!
echo "Launching BB-OPSD on physical GPUs 4-7"
setsid env PAIR_STAMP="$PAIR_STAMP" SKIP_EVAL_BEFORE_TRAIN=1 OPENCLAW_TRAIN_NUM_WORKERS=32 OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4 bash "$ARM_SCRIPT" bb_agentopsd &
OPSD_PID=$!
printf 'state=RUNNING\nstarted_utc=%s\npair_stamp=%s\ngrpo_pid=%s\nopsd_pid=%s\n' \
  "$(date -u +%FT%TZ)" "$PAIR_STAMP" "$GRPO_PID" "$OPSD_PID" >"$STATUS_FILE"

GRPO_RC=""
OPSD_RC=""
while [[ -n "$GRPO_PID" || -n "$OPSD_PID" ]]; do
  active_pids=()
  [[ -n "$GRPO_PID" ]] && active_pids+=("$GRPO_PID")
  [[ -n "$OPSD_PID" ]] && active_pids+=("$OPSD_PID")
  set +e
  wait -n -p finished_pid "${active_pids[@]}"
  finished_rc=$?
  set -e
  if [[ "$finished_pid" = "$GRPO_PID" ]]; then
    GRPO_RC=$finished_rc
    GRPO_PID=""
  elif [[ "$finished_pid" = "$OPSD_PID" ]]; then
    OPSD_RC=$finished_rc
    OPSD_PID=""
  else
    echo "Supervisor received unknown child pid: $finished_pid" >&2
    exit 1
  fi
  if (( finished_rc != 0 )); then
    echo "Arm pid=$finished_pid failed with rc=$finished_rc; terminating its paired arm." >&2
    [[ -n "$GRPO_PID" ]] && terminate_arm "$GRPO_PID"
    [[ -n "$OPSD_PID" ]] && terminate_arm "$OPSD_PID"
    exit 1
  fi
done
echo "agentopsd_exit_code=$GRPO_RC bb_agentopsd_exit_code=$OPSD_RC"
