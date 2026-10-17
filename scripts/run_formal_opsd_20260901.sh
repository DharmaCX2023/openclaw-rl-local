#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
LAUNCHER="${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
ANCHORS="${OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE:-${REPO_ROOT}/outputs/opsd-anchor-base-seed0-800-20260831/full800/anchors.jsonl}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
EXP_NAME="${EXP_NAME:-clawgym-official-30b-opsd-formal-seed0-${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/${EXP_NAME}}"
SAVE_CKPT="${SAVE_CKPT:-${OUTPUT_ROOT}/checkpoints}"
REMOTE_BASE="${OPENCLAW_REMOTE_BASE:-/root/clawgym_rl_runs/${EXP_NAME}}"
RUN_LOG="${OUTPUT_ROOT}/orchestrator.log"
STATUS_FILE="${OUTPUT_ROOT}/status.env"
RUN_ENV="${OUTPUT_ROOT}/run.env"
BURN_SESSION="pypenguin_phase56_full_burn"
BURN_LOG="/tmp/pypenguin_phase56_full_burn.log"
SSH=(ssh -i /root/openclaw_cpu_key -o BatchMode=yes -o ConnectTimeout=10 root@159.138.50.147)

mkdir -p "${OUTPUT_ROOT}/opsd_step_summaries" "${OUTPUT_ROOT}/tensorboard"
exec >>"${RUN_LOG}" 2>&1

gpu_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d' | wc -l
}

stop_burn() {
  if ! tmux has-session -t "${BURN_SESSION}" 2>/dev/null; then
    [[ "$(gpu_process_count)" -eq 0 ]]
    return
  fi
  local command
  command="$(tmux list-panes -t "${BURN_SESSION}" -a -F '#{pane_start_command}')"
  [[ "${command}" == *synthetic_train_stress.py* ]] || {
    echo "Refusing to stop non-burn session ${BURN_SESSION}" >&2
    return 1
  }
  tmux send-keys -t "${BURN_SESSION}" C-c
  for _ in $(seq 1 30); do
    tmux has-session -t "${BURN_SESSION}" 2>/dev/null || break
    sleep 1
  done
  tmux kill-session -t "${BURN_SESSION}" 2>/dev/null || true
  for _ in $(seq 1 90); do
    [[ "$(gpu_process_count)" -eq 0 ]] && return
    sleep 1
  done
  echo "GPU burn processes did not exit" >&2
  return 1
}

cleanup_remote() {
  "${SSH[@]}" "remote_base='${REMOTE_BASE}'; ids=''; for id in \$(docker ps -aq --filter name=clawgym-rl-); do if docker inspect --format '{{range .Mounts}}{{println .Source}}{{end}}' \"\$id\" 2>/dev/null | grep -Fq -- \"\$remote_base/\"; then ids=\"\$ids \$id\"; fi; done; if [ -n \"\$ids\" ]; then docker rm -f \$ids >/dev/null; fi; rm -rf -- \"\$remote_base\"; printf 'cleaned_own_containers=%s\\n' \"\$(printf '%s\\n' \$ids | sed '/^\$/d' | wc -l)\"" || true
}

start_burn() {
  tmux has-session -t "${BURN_SESSION}" 2>/dev/null && return
  for _ in $(seq 1 120); do
    [[ "$(gpu_process_count)" -eq 0 ]] && break
    sleep 1
  done
  [[ "$(gpu_process_count)" -eq 0 ]] || {
    echo "GPU processes remain; burn restore deferred" >&2
    return 1
  }
  tmux new-session -d -s "${BURN_SESSION}" \
    "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29631 --nproc_per_node=8 synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>${BURN_LOG} 2>&1"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  cleanup_remote
  start_burn || true
  printf 'exit_code=%s\nfinished_utc=%s\n' "${status}" "$(date -u +%FT%TZ)" >"${STATUS_FILE}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

[[ -s "${ANCHORS}" ]] || {
  echo "Missing OPSD evidence anchors: ${ANCHORS}" >&2
  exit 1
}
if [[ -e "${SAVE_CKPT}" && -z "${LOAD_CKPT:-}" ]] && \
   [[ -n "$(find "${SAVE_CKPT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty checkpoint root without LOAD_CKPT: ${SAVE_CKPT}" >&2
  exit 1
fi

# Full tensor/logprob/routing dumps are intentionally disabled. Only the
# compact, per-rollout OPSD scalar summaries below are persisted.
unset OPENCLAW_SAVE_DEBUG_ROLLOUT_DATA
unset OPENCLAW_SAVE_DEBUG_TRAIN_DATA
unset OPENCLAW_LOAD_DEBUG_ROLLOUT_DATA

cat >"${RUN_ENV}" <<EOF
run_stamp=${RUN_STAMP}
exp_name=${EXP_NAME}
output_root=${OUTPUT_ROOT}
save_ckpt=${SAVE_CKPT}
load_ckpt=${LOAD_CKPT:-}
remote_base=${REMOTE_BASE}
anchors=${ANCHORS}
rollout_batch_size=8
n_samples_per_prompt=8
train_workers=48
gateway_start_concurrency=4
save_interval=20
eval_interval=20
eval_before_train=0
opsd_step_summary=${OUTPUT_ROOT}/opsd_step_summaries/{rollout_id}.json
debug_tensor_dumps=disabled
EOF

stop_burn

echo "Starting formal OPSD training"
cat "${RUN_ENV}"

cd "${REPO_ROOT}"
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 ACTOR_GPUS=4 EXPECTED_ACTOR_GPUS=4 ROLLOUT_GPUS=4 SGLANG_TP=2 \
  EXP_NAME="${EXP_NAME}" RUN_STAMP="${RUN_STAMP}" SAVE_CKPT="${SAVE_CKPT}" \
  LOAD_CKPT="${LOAD_CKPT:-}" \
  OPENCLAW_DISABLE_CHECKPOINT_SAVE=0 SAVE_INTERVAL=20 SAVE_RETAIN_INTERVAL=0 \
  LOCAL_CHECKPOINT_MIN_FREE_GB=500 USE_WANDB=0 USE_TENSORBOARD=1 \
  TENSORBOARD_DIR="${OUTPUT_ROOT}/tensorboard" TENSORBOARD_PROJECT=opsd_formal \
  OPENCLAW_RECORD_ENABLED=1 OPENCLAW_RECORD_FILE="${OUTPUT_ROOT}/proxy_records.jsonl" \
  OPENCLAW_SAVE_OPSD_STEP_SUMMARY="${OUTPUT_ROOT}/opsd_step_summaries/{rollout_id}.json" \
  PORT=30901 OPENCLAW_REMOTE_PROXY_PORT=39901 \
  OPENCLAW_REMOTE_BASE="${REMOTE_BASE}" OPENCLAW_BUNDLED_PLUGINS_DIR=/app/dist/extensions \
  OPENCLAW_REMOTE_GATEWAY_PORT_MODE=run_scoped \
  OPENCLAW_REMOTE_GATEWAY_PORT_MIN=20000 OPENCLAW_REMOTE_GATEWAY_PORT_MAX=62000 \
  OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS=512 OPENCLAW_REMOTE_START_ATTEMPTS=3 \
  OPENCLAW_GATEWAY_PORT_BASE=25000 OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=3000 \
  OPENCLAW_GATEWAY_PORT_MAX_SLOTS=512 OPENCLAW_GATEWAY_PORT_STRIDE=10 \
  OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4 OPENCLAW_REMOTE_SSH_POOL_SIZE=8 \
  NUM_ROLLOUT=100 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 \
  OPENCLAW_TRAIN_NUM_WORKERS=48 OPENCLAW_TRAIN_SCHEDULING=sequential \
  OPENCLAW_TRAIN_SHUFFLE_SEED=0 OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE=20260901 \
  OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=1 ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=0.8 \
  OPENCLAW_MAX_RESPONSE_TOKENS=8192 OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 \
  OPENCLAW_CONTEXT_WINDOW=65536 OPENCLAW_MAX_AGENT_STEPS=96 \
  OPENCLAW_MAX_COMPACTION_EVENTS=8 OPENCLAW_TOKEN_LEDGER_MODE=enforce \
  OPENCLAW_CHAT_TURN_TIMEOUT=1200 SGLANG_WATCHDOG_TIMEOUT=1800 \
  SKIP_EVAL_BEFORE_TRAIN=1 EVAL_INTERVAL=20 OPENCLAW_EVAL_NUM_WORKERS=32 \
  OPENCLAW_EVAL_N_SAMPLES=1 OPENCLAW_EVAL_SAVE_WORKSPACES=1 \
  OPENCLAW_EVAL_ARTIFACT_STRICT=0 OPENCLAW_EVAL_WORKSPACE_CAPTURE_ATTEMPTS=3 \
  OPENCLAW_EVAL_WORKSPACE_RETRY_DELAY_SECONDS=2 OPENCLAW_EVAL_WORKSPACE_MAX_BYTES=134217728 \
  OPENCLAW_EVAL_WORKSPACE_TIMEOUT=600 \
  USE_ROUTING_REPLAY=1 USE_ROLLOUT_ROUTING_REPLAY=1 USE_TIS=1 \
  TIS_CLIP=2.0 TIS_CLIP_LOW=0.0 SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 \
  MEGATRON_DETERMINISTIC_MODE=1 SGLANG_ENABLE_CUDA_GRAPH=1 MAX_TOKENS_PER_GPU=81920 \
  KL_LOSS_MONITOR_INTERVAL=5 ENTROPY_COEF=0.0 \
  SGLANG_SESSION_AFFINITY=1 SGLANG_AFFINITY_MODE=task_sample_fixed \
  SGLANG_AFFINITY_SAMPLES_PER_TASK=8 SGLANG_AFFINITY_MAX_LOAD_DELTA=2 \
  SGLANG_AFFINITY_MIGRATION_LOAD_DELTA=4 SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS=10 \
  OPENCLAW_OPSD_ENABLE=1 OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE="${ANCHORS}" \
  OPENCLAW_OPSD_STEP_BETA=1.0 OPENCLAW_OPSD_STEP_CLIP_LOWER=0.0 \
  OPENCLAW_OPSD_STEP_CLIP_UPPER=1.2 OPENCLAW_OPSD_TEACHER_CONCURRENCY=8 \
  OPENCLAW_OPSD_TEACHER_TIMEOUT=600 \
  bash "${LAUNCHER}"
