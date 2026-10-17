#!/bin/bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
LAUNCHER="${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
LOAD_CKPT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700"
FULL_BURN_SESSION="gpu_burn_all8_20260823"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/formal-e2e-routing-tis-ab-step99-${RUN_STAMP}}"
STATUS_FILE="${OUTPUT_ROOT}/case_status.tsv"

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
    tmux new-session -d -s "${FULL_BURN_SESSION}" \
      "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29570 --nproc-per-node=8 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5"
  fi
}

restore_burn() {
  start_full_burn
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
    EXP_NAME="formal-e2e-step99-${case_name}-${RUN_STAMP}" \
    RUN_STAMP="${RUN_STAMP}-${case_name}" \
    SAVE_CKPT="${case_dir}/runtime" \
    LOAD_CKPT="${LOAD_CKPT}" \
    USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1 \
    NO_LOAD_OPTIM=1 \
    NO_LOAD_RNG=1 \
    OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
    USE_TENSORBOARD=1 \
    TENSORBOARD_DIR="${case_dir}/tensorboard" \
    TENSORBOARD_PROJECT="formal_e2e_routing_tis_ab" \
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
    OPENCLAW_TRAIN_NUM_WORKERS=32 \
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

trap restore_burn EXIT INT TERM
stop_session "${FULL_BURN_SESSION}"

status_a=0
run_case baseline_no_replay_no_tis 0 0 30321 65021 || status_a=$?
printf "baseline_no_replay_no_tis\t%s\t%s\n" "${status_a}" "$(date -u +%FT%TZ)" >> "${STATUS_FILE}"

status_b=0
run_case replay_plus_tis 1 1 30322 65022 || status_b=$?
printf "replay_plus_tis\t%s\t%s\n" "${status_b}" "$(date -u +%FT%TZ)" >> "${STATUS_FILE}"

printf "[%s] completed baseline=%s combo=%s output=%s\n" \
  "$(date -u +%FT%TZ)" "${status_a}" "${status_b}" "${OUTPUT_ROOT}"

if [ "${status_a}" -ne 0 ] || [ "${status_b}" -ne 0 ]; then
  exit 1
fi
exit 0
