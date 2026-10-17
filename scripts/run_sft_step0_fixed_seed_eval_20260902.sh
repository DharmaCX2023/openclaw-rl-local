#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/root/openclaw-rl-local"
DRIVER="${REPO_ROOT}/scripts/run_fixed_seed_8x8_rollout_gate.py"
PYTHON="${REPO_ROOT}/.venv-openclaw/bin/python"
MODEL="/mnt/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127"
TASKS="${REPO_ROOT}/clawgym-official/data/clawgym_eval"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
EXP_NAME="sft-step0-fixed-seed20260901-${RUN_STAMP}"
OUTPUT_DIR="${REPO_ROOT}/outputs/${EXP_NAME}"
REMOTE_BASE="/root/clawgym_rl_runs/${EXP_NAME}"
LAUNCH_LOG="${OUTPUT_DIR}.launch.log"
STATUS_FILE="${OUTPUT_DIR}.status.env"

FULL_BURN_SESSION="pypenguin_phase56_full_burn"
PARTIAL_BURN_SESSION="sft_step0_fixed_eval_burn_0_3"
FULL_BURN_LOG="/tmp/pypenguin_phase56_full_burn.log"
PARTIAL_BURN_LOG="/tmp/sft_step0_fixed_eval_burn_0_3.log"
SSH=(ssh -i /root/openclaw_cpu_key -o BatchMode=yes -o ConnectTimeout=10 root@159.138.50.147)

mkdir -p "$(dirname "${LAUNCH_LOG}")"
exec >>"${LAUNCH_LOG}" 2>&1

gpu_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d' | wc -l
}

stop_tmux_burn() {
  local session="$1"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    return 0
  fi
  local command
  command="$(tmux list-panes -t "${session}" -a -F '#{pane_start_command}')"
  [[ "${command}" == *synthetic_train_stress.py* ]] || {
    echo "Refusing to stop non-burn tmux session: ${session}" >&2
    return 1
  }
  tmux send-keys -t "${session}" C-c
  for _ in $(seq 1 30); do
    tmux has-session -t "${session}" 2>/dev/null || return 0
    sleep 1
  done
  tmux kill-session -t "${session}" 2>/dev/null || true
}

start_partial_burn() {
  tmux has-session -t "${PARTIAL_BURN_SESSION}" 2>/dev/null && return 0
  tmux new-session -d -s "${PARTIAL_BURN_SESSION}" \
    "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/bin/torchrun --standalone --master_port=29632 --nproc_per_node=4 synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>${PARTIAL_BURN_LOG} 2>&1"
}

start_full_burn() {
  tmux has-session -t "${FULL_BURN_SESSION}" 2>/dev/null && return 0
  for _ in $(seq 1 120); do
    [[ "$(gpu_process_count)" -eq 0 ]] && break
    sleep 1
  done
  [[ "$(gpu_process_count)" -eq 0 ]] || {
    echo "GPU processes remain; full burn restore deferred" >&2
    return 1
  }
  tmux new-session -d -s "${FULL_BURN_SESSION}" \
    "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29631 --nproc_per_node=8 synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>${FULL_BURN_LOG} 2>&1"
}

cleanup_remote() {
  "${SSH[@]}" "remote_base='${REMOTE_BASE}'; ids=''; for id in \$(docker ps -aq --filter name=clawgym-rl-); do if docker inspect --format '{{range .Mounts}}{{println .Source}}{{end}}' \"\$id\" 2>/dev/null | grep -Fq -- \"\$remote_base/\"; then ids=\"\$ids \$id\"; fi; done; if [ -n \"\$ids\" ]; then docker rm -f \$ids >/dev/null; fi; rm -rf -- \"\$remote_base\"; printf 'cleaned_own_containers=%s\\n' \"\$(printf '%s\\n' \$ids | sed '/^\$/d' | wc -l)\"" || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  cleanup_remote
  stop_tmux_burn "${PARTIAL_BURN_SESSION}" || true
  start_full_burn || true
  printf 'exit_code=%s\nfinished_utc=%s\noutput_dir=%s\nsampling_seed_base=20260901\n' \
    "${status}" "$(date -u +%FT%TZ)" "${OUTPUT_DIR}" >"${STATUS_FILE}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

[[ -x "${PYTHON}" ]]
[[ -f "${DRIVER}" ]]
[[ -d "${MODEL}" ]]
[[ -d "${TASKS}" ]]
[[ ! -e "${OUTPUT_DIR}" ]]

"${PYTHON}" -m pytest -q \
  "${REPO_ROOT}/clawgym-official/tests/test_token_logprob_alignment.py::test_eval_sampling_seed_is_stable_across_runtime_sessions"

env \
  OPENCLAW_ENV_FILE="${REPO_ROOT}/.env" \
  OPENCLAW_MAX_AGENT_STEPS=96 \
  OPENCLAW_MAX_COMPACTION_EVENTS=8 \
  OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4 \
  "${PYTHON}" "${DRIVER}" \
    --eval-mode \
    --preflight-only \
    --model "${MODEL}" \
    --tasks "${TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --task-count 200 \
    --samples-per-task 1 \
    --parallel 32 \
    --rollout-id 0 \
    --deterministic \
    --sampling-seed-base 20260901 \
    --deterministic-tool-call-ids \
    --gpus 4,5,6,7 \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --enable-cuda-graph \
    --backend-port 30480 \
    --proxy-port 30481 \
    --reverse-port 39481 \
    --served-model sft-step0-fixed-seed20260901 \
    --remote-base "${REMOTE_BASE}" \
    --gateway-port-base 45000 \
    --gateway-train-offset 1000 \
    --gateway-port-stride 2 \
    --ssh-pool-size 8 \
    --rollout-timeout 1200

stop_tmux_burn "${FULL_BURN_SESSION}"
for _ in $(seq 1 90); do
  [[ "$(gpu_process_count)" -eq 0 ]] && break
  sleep 1
done
[[ "$(gpu_process_count)" -eq 0 ]]
start_partial_burn

echo "Starting fixed-seed SFT step0 eval"
echo "output_dir=${OUTPUT_DIR}"
echo "sampling_identity=eval:v1:<dataset>:<task_id>:sample:<sample_index>"
echo "sampling_seed_base=20260901"

cd "${REPO_ROOT}"
env \
  OPENCLAW_ENV_FILE="${REPO_ROOT}/.env" \
  OPENCLAW_MAX_AGENT_STEPS=96 \
  OPENCLAW_MAX_COMPACTION_EVENTS=8 \
  OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=4 \
  OPENCLAW_EVAL_SAVE_WORKSPACES=1 \
  "${PYTHON}" "${DRIVER}" \
    --eval-mode \
    --model "${MODEL}" \
    --tasks "${TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --task-count 200 \
    --samples-per-task 1 \
    --parallel 32 \
    --rollout-id 0 \
    --deterministic \
    --sampling-seed-base 20260901 \
    --deterministic-tool-call-ids \
    --gpus 4,5,6,7 \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --enable-cuda-graph \
    --backend-port 30480 \
    --proxy-port 30481 \
    --reverse-port 39481 \
    --served-model sft-step0-fixed-seed20260901 \
    --remote-base "${REMOTE_BASE}" \
    --gateway-port-base 45000 \
    --gateway-train-offset 1000 \
    --gateway-port-stride 2 \
    --ssh-pool-size 8 \
    --rollout-timeout 1200
