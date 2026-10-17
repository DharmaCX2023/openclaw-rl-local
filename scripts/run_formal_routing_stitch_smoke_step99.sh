#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
LAUNCHER="${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
LOAD_CKPT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700"
FULL_BURN_SESSION="gpu_burn_all8_20260823"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/formal-routing-stitch-smoke-step99-${RUN_STAMP}}"
STATUS_FILE="${OUTPUT_ROOT}/case_status.tsv"
SMOKE_WORKERS="${SMOKE_WORKERS:-32}"
SMOKE_LOCAL_PORT="${SMOKE_LOCAL_PORT:-30323}"
SMOKE_REVERSE_PORT="${SMOKE_REVERSE_PORT:-65023}"
SGLANG_METRICS_INTERVAL="${SGLANG_METRICS_INTERVAL:-2}"
SMOKE_AFFINITY_MODE="${SMOKE_AFFINITY_MODE:-task_sample_fixed}"
SMOKE_AFFINITY_MIGRATION_LOAD_DELTA="${SMOKE_AFFINITY_MIGRATION_LOAD_DELTA:-4}"
SMOKE_AFFINITY_MIGRATION_SUSTAINED_SECONDS="${SMOKE_AFFINITY_MIGRATION_SUSTAINED_SECONDS:-10}"
SMOKE_AFFINITY_SAMPLES_PER_TASK="${SMOKE_AFFINITY_SAMPLES_PER_TASK:-8}"
SGLANG_METRICS_PID=""

stop_session() {
  local session="$1"
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux send-keys -t "${session}" C-c
    sleep 3
    tmux kill-session -t "${session}" 2>/dev/null || true
  fi
}

start_full_burn() {
  if ! tmux has-session -t "${FULL_BURN_SESSION}" 2>/dev/null; then
    local gpu
    local port
    local command
    command="cd /root && exec env CUDA_VISIBLE_DEVICES=0 /usr/local/bin/torchrun --standalone --master_port=29600 --nproc-per-node=1 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 32 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5"
    tmux new-session -d -s "${FULL_BURN_SESSION}" -n gpu0 "${command}"
    for gpu in 1 2 3 4 5 6 7; do
      port=$((29600 + gpu))
      command="cd /root && exec env CUDA_VISIBLE_DEVICES=${gpu} /usr/local/bin/torchrun --standalone --master_port=${port} --nproc-per-node=1 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 32 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5"
      tmux new-window -d -t "${FULL_BURN_SESSION}" -n "gpu${gpu}" "${command}"
    done
  fi
}

restore_burn() {
  if [ -n "${SGLANG_METRICS_PID}" ]; then
    kill "${SGLANG_METRICS_PID}" 2>/dev/null || true
    wait "${SGLANG_METRICS_PID}" 2>/dev/null || true
    SGLANG_METRICS_PID=""
  fi
  start_full_burn
}


start_sglang_metrics_sampler() {
  local output_file="$1"
  (
    printf 'timestamp\tengine\tmetric_and_labels\tvalue\n'
    while true; do
      timestamp="$(date -u +%FT%T.%3NZ)"
      engine=0
      for endpoint in "http://172.16.47.29:15000/v1/loads?include=core,memory,queues&format=prometheus" "http://172.16.47.29:15002/v1/loads?include=core,memory,queues&format=prometheus"; do
        if payload="$(curl -fsS --max-time 1.5 "${endpoint}" 2>/dev/null)"; then
          printf '%s\n' "${payload}" | rg '^sglang_' | sed "s/^/${timestamp}\\t${engine}\\t/"
        else
          printf '%s\t%s\tscrape_error\t1\n' "${timestamp}" "${engine}"
        fi
        engine=$((engine + 1))
      done
      sleep "${SGLANG_METRICS_INTERVAL}"
    done
  ) >> "${output_file}" &
  SGLANG_METRICS_PID="$!"
}
run_case() {
  local case_name="$1"
  local use_replay="$2"
  local use_tis="$3"
  local local_port="$4"
  local reverse_port="$5"
  local case_dir="${OUTPUT_ROOT}/${case_name}"

  mkdir -p "${case_dir}/rollout_data" "${case_dir}/train_data" "${case_dir}/tensorboard"
  printf "[%s] starting %s replay=%s tis=%s\n" "$(date -u +%FT%TZ)" "${case_name}" "${use_replay}" "${use_tis}"

  env -u SLIME_NUMERICAL_DIAG_CONFIG \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NUM_GPUS=8 \
    ACTOR_GPUS=4 \
    EXPECTED_ACTOR_GPUS=4 \
    ROLLOUT_GPUS=4 \
    SGLANG_TP=2 \
    EXP_NAME="formal-routing-stitch-step99-${case_name}-${RUN_STAMP}" \
    RUN_STAMP="${RUN_STAMP}-${case_name}" \
    SAVE_CKPT="${case_dir}/runtime" \
    LOAD_CKPT="${LOAD_CKPT}" \
    USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1 \
    NO_LOAD_OPTIM=1 \
    NO_LOAD_RNG=1 \
    OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
    USE_TENSORBOARD=1 \
    TENSORBOARD_DIR="${case_dir}/tensorboard" \
    TENSORBOARD_PROJECT="formal_routing_stitch_smoke" \
    USE_WANDB=0 \
    OPENCLAW_RECORD_ENABLED=1 \
    OPENCLAW_RECORD_FILE="${case_dir}/proxy_records.jsonl" \
    OPENCLAW_SAVE_DEBUG_ROLLOUT_DATA="${case_dir}/rollout_data/{rollout_id}.pt" \
    OPENCLAW_SAVE_DEBUG_TRAIN_DATA="${case_dir}/train_data/{rollout_id}_{rank}.pt" \
    OPENCLAW_REMOTE_PROXY_PORT="${reverse_port}" \
    PORT="${local_port}" \
    OPENCLAW_REMOTE_GATEWAY_PORT_MODE=run_scoped \
    OPENCLAW_REMOTE_GATEWAY_PORT_MIN=20000 \
    OPENCLAW_REMOTE_GATEWAY_PORT_MAX=62000 \
    OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS=512 \
    OPENCLAW_REMOTE_START_ATTEMPTS=3 \
    OPENCLAW_GATEWAY_PORT_BASE=16000 \
    OPENCLAW_GATEWAY_PORT_STRIDE=10 \
    OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=5000 \
    OPENCLAW_GATEWAY_PORT_MAX_SLOTS=128 \
    OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4 \
    OPENCLAW_REMOTE_SSH_POOL_SIZE=8 \
    OPENCLAW_TRAIN_NUM_WORKERS="${SMOKE_WORKERS}" \
    OPENCLAW_TRAIN_SCHEDULING=sequential \
    OPENCLAW_TRAIN_SHUFFLE_SEED=0 \
    OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE=20260825 \
    OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=1 \
    NUM_ROLLOUT=101 \
    ROLLOUT_BATCH_SIZE=8 \
    N_SAMPLES_PER_PROMPT=8 \
    ROLLOUT_TEMPERATURE=0.7 \
    ROLLOUT_TOP_P=1.0 \
    OPENCLAW_CONTEXT_WINDOW=65536 \
    OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 \
    OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
    OPENCLAW_MAX_AGENT_STEPS=96 \
    OPENCLAW_MAX_COMPACTION_EVENTS=8 \
    OPENCLAW_CHAT_TURN_TIMEOUT=1200 \
    EVAL_INTERVAL=1000 \
    MAX_TOKENS_PER_GPU=81920 \
    MEGATRON_DETERMINISTIC_MODE=1 \
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 \
    SGLANG_AFFINITY_MODE="${SMOKE_AFFINITY_MODE}" \
    SGLANG_AFFINITY_SAMPLES_PER_TASK="${SMOKE_AFFINITY_SAMPLES_PER_TASK}" \
    SGLANG_AFFINITY_MIGRATION_LOAD_DELTA="${SMOKE_AFFINITY_MIGRATION_LOAD_DELTA}" \
    SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS="${SMOKE_AFFINITY_MIGRATION_SUSTAINED_SECONDS}" \
    USE_ROUTING_REPLAY="${use_replay}" \
    USE_ROLLOUT_ROUTING_REPLAY="${use_replay}" \
    USE_TIS="${use_tis}" \
    TIS_CLIP_LOW=0.0 \
    TIS_CLIP=2.0 \
    bash "${LAUNCHER}"
}

if [ ! -f "${LOAD_CKPT}/latest_checkpointed_iteration.txt" ]; then
  echo "Missing checkpoint metadata: ${LOAD_CKPT}" >&2
  exit 1
fi
if [ -e "${OUTPUT_ROOT}" ] && [ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "OUTPUT_ROOT is not empty: ${OUTPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
printf "case\texit_code\tfinished_utc\n" > "${STATUS_FILE}"
printf "output_root=%s\ncheckpoint_iteration=%s\n" \
  "${OUTPUT_ROOT}" "$(tr -d "[:space:]" < "${LOAD_CKPT}/latest_checkpointed_iteration.txt")" \
  > "${OUTPUT_ROOT}/experiment_meta.txt"
printf "workers=%s\nlocal_port=%s\nreverse_port=%s\naffinity_mode=%s\naffinity_samples_per_task=%s\nmigration_load_delta=%s\nmigration_sustained_seconds=%s\n" \
  "${SMOKE_WORKERS}" "${SMOKE_LOCAL_PORT}" "${SMOKE_REVERSE_PORT}" \
  "${SMOKE_AFFINITY_MODE}" "${SMOKE_AFFINITY_SAMPLES_PER_TASK}" \
  "${SMOKE_AFFINITY_MIGRATION_LOAD_DELTA}" "${SMOKE_AFFINITY_MIGRATION_SUSTAINED_SECONDS}" \
  >> "${OUTPUT_ROOT}/experiment_meta.txt"

trap restore_burn EXIT INT TERM
stop_session "${FULL_BURN_SESSION}"

status=0
start_sglang_metrics_sampler "${OUTPUT_ROOT}/sglang_engine_loads.tsv"
run_case stitched_replay_plus_tis 1 1 "${SMOKE_LOCAL_PORT}" "${SMOKE_REVERSE_PORT}" || status=$?
printf "stitched_replay_plus_tis\t%s\t%s\n" "${status}" "$(date -u +%FT%TZ)" >> "${STATUS_FILE}"

if [ -n "${SGLANG_METRICS_PID}" ]; then
  kill "${SGLANG_METRICS_PID}" 2>/dev/null || true
  wait "${SGLANG_METRICS_PID}" 2>/dev/null || true
  SGLANG_METRICS_PID=""
fi
printf "[%s] completed status=%s output=%s\n" \
  "$(date -u +%FT%TZ)" "${status}" "${OUTPUT_ROOT}"

if [ "${status}" -ne 0 ]; then
  exit 1
fi
exit 0
