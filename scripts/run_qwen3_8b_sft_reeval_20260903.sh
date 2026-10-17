#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/openclaw-rl-local
PYTHON="$ROOT/.venv-openclaw/bin/python"
DRIVER="$ROOT/scripts/run_fixed_seed_8x8_rollout_gate.py"
TASKS="$ROOT/clawgym-official/data/clawgym_eval"
BASE_MODEL="$ROOT/models/Qwen3-8B-yarn2-64k"
SFT_MODEL="$ROOT/outputs/clawgym-8b-sft-formal-20260902-165551/final_model"
STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="$ROOT/outputs/qwen3-8b-sft-reeval-$STAMP"
STATUS="$RUN_ROOT/status.env"
LOG="$RUN_ROOT/supervisor.log"
REMOTE_PREFIX="/data/clawgym_rl_runs/qwen3-8b-sft-reeval-$STAMP"
SSH=(ssh -i /root/openclaw_cpu_key -o BatchMode=yes -o ConnectTimeout=10 root@159.138.50.147)

BURN_SESSION=pypenguin_phase57_full_burn
BURN_LOG=/tmp/pypenguin_phase57_full_burn.log
SELECTED_PARALLEL=0
ACTIVE_REMOTE_BASE=""
FORCE_PARALLEL="${EVAL_FORCE_PARALLEL:-64}"

mkdir -p "$RUN_ROOT"
exec >>"$LOG" 2>&1

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

gpu_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d' | wc -l
}

stop_burn() {
  if ! tmux has-session -t "$BURN_SESSION" 2>/dev/null; then
    return 0
  fi
  local command
  command="$(tmux list-panes -t "$BURN_SESSION" -F '#{pane_start_command}')"
  [[ "$command" == *synthetic_train_stress.py* ]] || {
    log "refusing_to_stop_non_burn_session=$BURN_SESSION"
    return 1
  }
  tmux send-keys -t "$BURN_SESSION" C-c
  for _ in $(seq 1 60); do
    tmux has-session -t "$BURN_SESSION" 2>/dev/null || return 0
    sleep 1
  done
  tmux kill-session -t "$BURN_SESSION"
}

start_burn() {
  tmux has-session -t "$BURN_SESSION" 2>/dev/null && return 0
  for _ in $(seq 1 120); do
    [[ "$(gpu_process_count)" -eq 0 ]] && break
    sleep 1
  done
  if [[ "$(gpu_process_count)" -ne 0 ]]; then
    log "burn_restore_deferred_gpu_processes_remain=$(gpu_process_count)"
    return 1
  fi
  tmux new-session -d -s "$BURN_SESSION" \
    "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29632 --nproc_per_node=8 synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>'$BURN_LOG' 2>&1"
  log "full_burn_started session=$BURN_SESSION"
}

cleanup_remote() {
  local remote_base="${1:-}"
  [[ "$remote_base" == "$REMOTE_PREFIX"* ]] || {
    log "refusing_remote_cleanup_unexpected_path=$remote_base"
    return 1
  }
  "${SSH[@]}" "remote_base='$remote_base'; ids=''; for id in \$(docker ps -aq --filter name=clawgym-rl-); do if docker inspect --format '{{range .Mounts}}{{println .Source}}{{end}}' \"\$id\" 2>/dev/null | grep -Fq -- \"\$remote_base/\"; then ids=\"\$ids \$id\"; fi; done; if [ -n \"\$ids\" ]; then docker rm -f \$ids >/dev/null; fi; rm -rf -- \"\$remote_base\"; printf 'cleaned_own_containers=%s\\n' \"\$(printf '%s\\n' \$ids | sed '/^\$/d' | wc -l)\"" || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "$ACTIVE_REMOTE_BASE" ]]; then
    cleanup_remote "$ACTIVE_REMOTE_BASE" || true
  fi
  start_burn || true
  {
    printf 'state=%s\n' "$([[ "$rc" -eq 0 ]] && echo COMPLETED || echo FAILED)"
    printf 'exit_code=%s\n' "$rc"
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'run_root=%s\n' "$RUN_ROOT"
    printf 'selected_parallel=%s\n' "$SELECTED_PARALLEL"
  } >"$STATUS"
  exit "$rc"
}
trap cleanup EXIT INT TERM

run_driver() {
  local label="$1" model="$2" output="$3" remote_base="$4" parallel="$5"
  local backend_port="$6" proxy_port="$7" reverse_port="$8" gateway_base="$9"
  shift 9
  ACTIVE_REMOTE_BASE="$remote_base"
  cleanup_remote "$remote_base"
  log "phase_start=$label model=$model parallel=$parallel remote_base=$remote_base"
  set +e
  env \
    OPENCLAW_ENV_FILE="$ROOT/.env" \
    OPENCLAW_MAX_AGENT_STEPS=96 \
    OPENCLAW_MAX_COMPACTION_EVENTS=8 \
    OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4 \
    OPENCLAW_EVAL_SAVE_WORKSPACES=1 \
    "$PYTHON" "$DRIVER" \
      --model "$model" \
      --tasks "$TASKS" \
      --output-dir "$output" \
      --parallel "$parallel" \
      --rollout-id 0 \
      --deterministic \
      --sampling-seed-base 20260901 \
      --deterministic-tool-call-ids \
      --gpus 0,1,2,3,4,5,6,7 \
      --tensor-parallel-size 1 \
      --data-parallel-size 8 \
      --enable-cuda-graph \
      --mem-fraction-static 0.70 \
      --backend-port "$backend_port" \
      --proxy-port "$proxy_port" \
      --reverse-port "$reverse_port" \
      --served-model "$label" \
      --remote-base "$remote_base" \
      --gateway-port-base "$gateway_base" \
      --gateway-train-offset 1000 \
      --gateway-port-stride 2 \
      --ssh-pool-size 8 \
      --rollout-timeout 1200 \
      "$@"
  local rc=$?
  set -e
  cleanup_remote "$remote_base"
  ACTIVE_REMOTE_BASE=""
  log "phase_end=$label rc=$rc"
  return "$rc"
}

printf 'state=RUNNING\nstarted_utc=%s\nrun_root=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ROOT" >"$STATUS"
{
  printf 'base_model=%s\n' "$BASE_MODEL"
  printf 'sft_model=%s\n' "$SFT_MODEL"
  printf 'tasks=%s\n' "$TASKS"
  printf 'task_count=200\nsamples_per_task=1\nseed_base=20260901\n'
  printf 'topology=TP1xDP8\nconcurrency_candidates=64,48\n'
  printf 'force_parallel=%s\n' "$FORCE_PARALLEL"
  printf 'gateway_start_concurrency=4\nssh_pool=8\n'
  printf 'context_window=65536\nmax_response_tokens=8192\nmax_agent_steps=96\nmax_compactions=8\n'
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  "${SSH[@]}" 'free -h; printf "running_containers="; docker ps -q | wc -l'
} >"$RUN_ROOT/manifest.txt"

[[ -x "$PYTHON" && -f "$DRIVER" && -d "$TASKS" && -d "$BASE_MODEL" && -d "$SFT_MODEL" ]]
"$PYTHON" -m pytest -q \
  "$ROOT/clawgym-official/tests/test_token_logprob_alignment.py::test_eval_sampling_seed_is_stable_across_runtime_sessions"

stop_burn
for _ in $(seq 1 120); do
  [[ "$(gpu_process_count)" -eq 0 ]] && break
  sleep 1
done
[[ "$(gpu_process_count)" -eq 0 ]]

SELECTED_PARALLEL=64
log "rope_fix_config_sha256=$(sha256sum "$SFT_MODEL/config.json" | awk '{print $1}')"
log "smoke_source=$ROOT/outputs/qwen3-8b-sft-ropefix-smoke-20260903 avg_reward=0.5349789915966386 correct=3/4"
run_driver \
  sft "$SFT_MODEL" "$RUN_ROOT/sft" "$REMOTE_PREFIX-sft" "$SELECTED_PARALLEL" \
  30640 30641 39641 56000 \
  --eval-mode --task-count 200 --samples-per-task 1

"$PYTHON" - "$RUN_ROOT" <<'PY_SUMMARY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
report=json.loads((root/'sft/eval_report.json').read_text())
(root/'summary.json').write_text(json.dumps({'sft':report['summary']},indent=2)+'\n')
print(json.dumps({'sft':report['summary']},indent=2))
PY_SUMMARY
log "sft_reeval_completed"
