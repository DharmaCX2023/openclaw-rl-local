#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/openclaw-rl-local"
LAUNCHER="${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
LOAD_CKPT="${REPO_ROOT}/outputs/clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/route-conflict-diagnostic-8x8-${RUN_STAMP}}"
REMOTE_BASE="/root/clawgym_rl_runs/route-conflict-diagnostic-8x8-${RUN_STAMP}"
RUN_LOG="${OUTPUT_ROOT}/orchestrator.log"
STATUS_FILE="${OUTPUT_ROOT}/status.env"
LOCAL_GPU_LOG="${OUTPUT_ROOT}/local_gpu.tsv"
SGLANG_LOAD_LOG="${OUTPUT_ROOT}/sglang_loads.tsv"
REMOTE_LOG="${OUTPUT_ROOT}/remote_resources.tsv"
SAFETY_LOG="${OUTPUT_ROOT}/safety_events.log"

REMOTE_HOST="159.138.50.147"
REMOTE_KEY="/root/openclaw_cpu_key"
LOCAL_PORT="30764"
REVERSE_PORT="39764"
TRAIN_WORKERS="${TRAIN_WORKERS_OVERRIDE:-32}"
START_CONCURRENCY="4"
REMOTE_MEM_FLOOR_KIB=$((48 * 1024 * 1024))
REMOTE_ROOT_FLOOR_KIB=$((8 * 1024 * 1024))
REMOTE_LOAD_ABORT="96"
REMOTE_BASELINE_CONTAINERS=""
RUN_PID=""
MONITOR_PID=""
STOP_MONITOR_FILE="${OUTPUT_ROOT}/stop_monitor"
BURN_SESSION="pypenguin_phase52_post_burn"
BURN_LOG="/root/logs/pypenguin-phase52-burn/post-campaign-all8.log"

SSH=(ssh -i "${REMOTE_KEY}" -o BatchMode=yes -o ConnectTimeout=10 root@"${REMOTE_HOST}")

mkdir -p "${OUTPUT_ROOT}"
exec >>"${RUN_LOG}" 2>&1

gpu_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d' | wc -l
}

stop_existing_burn() {
  if ! tmux has-session -t "${BURN_SESSION}" 2>/dev/null; then
    if [[ "$(gpu_process_count)" -ne 0 ]]; then
      echo "GPU processes exist but expected burn session is absent; refusing to interrupt them"
      return 1
    fi
    return 0
  fi
  local commands
  commands="$(tmux list-panes -t "${BURN_SESSION}" -a -F '#{pane_start_command}')"
  if [[ "${commands}" != *synthetic_train_stress.py* ]]; then
    echo "tmux ${BURN_SESSION} is not a synthetic burn; refusing to stop it"
    return 1
  fi
  echo "[$(date -u +%FT%TZ)] stopping verified synthetic burn ${BURN_SESSION}"
  tmux send-keys -t "${BURN_SESSION}" C-c
  for _ in $(seq 1 20); do
    tmux has-session -t "${BURN_SESSION}" 2>/dev/null || break
    sleep 1
  done
  tmux kill-session -t "${BURN_SESSION}" 2>/dev/null || true
  for _ in $(seq 1 90); do
    [[ "$(gpu_process_count)" -eq 0 ]] && return 0
    sleep 1
  done
  echo "GPU processes did not exit after stopping burn"
  return 1
}

start_full_burn() {
  tmux has-session -t "${BURN_SESSION}" 2>/dev/null && return 0
  for _ in $(seq 1 180); do
    [[ "$(gpu_process_count)" -eq 0 ]] && break
    sleep 1
  done
  if [[ "$(gpu_process_count)" -ne 0 ]]; then
    echo "[$(date -u +%FT%TZ)] GPUs still occupied; burn restore deferred"
    return 1
  fi
  mkdir -p "$(dirname "${BURN_LOG}")"
  tmux new-session -d -s "${BURN_SESSION}" \
    "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29570 --nproc-per-node=8 /root/synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>${BURN_LOG} 2>&1"
  echo "[$(date -u +%FT%TZ)] restored 8-GPU burn ${BURN_SESSION}"
}

cleanup_our_remote_resources() {
  "${SSH[@]}" "remote_base='${REMOTE_BASE}'; names=''; for id in \$(docker ps -aq --filter name=clawgym-rl-); do if docker inspect --format '{{range .Mounts}}{{println .Source}}{{end}}' \"\$id\" 2>/dev/null | grep -Fq -- \"\$remote_base/\"; then names=\"\$names \$id\"; fi; done; if [ -n \"\$names\" ]; then docker rm -f \$names >/dev/null; fi; rm -rf -- \"\$remote_base\"; printf 'cleaned_ids=%s\\n' \"\$(printf '%s\\n' \$names | sed '/^\$/d' | wc -l)\"" \
    >>"${SAFETY_LOG}" 2>&1 || true
}

stop_monitor() {
  touch "${STOP_MONITOR_FILE}"
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

restore_and_cleanup() {
  local original_status=$?
  stop_monitor
  if [[ -n "${RUN_PID}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM "${RUN_PID}" 2>/dev/null || true
    wait "${RUN_PID}" 2>/dev/null || true
  fi
  cleanup_our_remote_resources
  start_full_burn || true
  return "${original_status}"
}
trap restore_and_cleanup EXIT INT TERM

monitor_loop() {
  printf 'timestamp\tgpu\tutil_pct\tmem_used_mib\tmem_total_mib\tpower_w\n' >"${LOCAL_GPU_LOG}"
  printf 'timestamp\tengine\tpayload\n' >"${SGLANG_LOAD_LOG}"
  printf 'timestamp\tload1\tmem_available_kib\troot_available_kib\tclawgym_running\tclawgym_healthy\tclawgym_starting\tclawgym_unhealthy\n' >"${REMOTE_LOG}"
  local tick=0 ssh_failures=0 high_load_samples=0 low_mem_samples=0 low_disk_samples=0
  while [[ ! -e "${STOP_MONITOR_FILE}" ]]; do
    local ts
    ts="$(date -u +%FT%T.%3NZ)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', *' -v ts="${ts}" '{print ts "\t" $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5}' \
      >>"${LOCAL_GPU_LOG}" || true
    local engine=0 endpoint payload
    for endpoint in \
      'http://172.16.47.29:15000/v1/loads?include=core,memory,queues&format=prometheus' \
      'http://172.16.47.29:15002/v1/loads?include=core,memory,queues&format=prometheus'; do
      if payload="$(curl -fsS --max-time 1.5 "${endpoint}" 2>/dev/null)"; then
        printf '%s\t%s\t%s\n' "${ts}" "${engine}" "$(printf '%s' "${payload}" | tr '\n' ';')" >>"${SGLANG_LOAD_LOG}"
      else
        printf '%s\t%s\tscrape_error\n' "${ts}" "${engine}" >>"${SGLANG_LOAD_LOG}"
      fi
      engine=$((engine + 1))
    done

    if (( tick % 3 == 0 )); then
      local remote_line
      if remote_line="$("${SSH[@]}" 'load=$(cut -d" " -f1 /proc/loadavg); mem=$(awk "/MemAvailable:/ {print \$2}" /proc/meminfo); root_avail=$(df -Pk /root | awk "NR==2 {print \$4}"); running=$(docker ps --filter name=clawgym-rl- -q | wc -l); states=$(docker ps --filter name=clawgym-rl- --format "{{.Status}}" | awk "BEGIN{h=0;s=0;u=0} /\(healthy\)/{h++} /health: starting/{s++} /unhealthy/{u++} END{printf \"%d %d %d\",h,s,u}"); printf "%s %s %s %s %s\\n" "$load" "$mem" "$root_avail" "$running" "$states"' 2>/dev/null)"; then
        ssh_failures=0
        local load1 mem_kib root_kib running healthy starting unhealthy
        read -r load1 mem_kib root_kib running healthy starting unhealthy <<<"${remote_line}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${ts}" "${load1}" "${mem_kib}" "${root_kib}" "${running}" "${healthy}" "${starting}" "${unhealthy}" >>"${REMOTE_LOG}"
        awk -v x="${load1}" -v y="${REMOTE_LOAD_ABORT}" 'BEGIN{exit !(x>y)}' && high_load_samples=$((high_load_samples + 1)) || high_load_samples=0
        (( mem_kib < REMOTE_MEM_FLOOR_KIB )) && low_mem_samples=$((low_mem_samples + 1)) || low_mem_samples=0
        (( root_kib < REMOTE_ROOT_FLOOR_KIB )) && low_disk_samples=$((low_disk_samples + 1)) || low_disk_samples=0
        if (( running > REMOTE_BASELINE_CONTAINERS + TRAIN_WORKERS + START_CONCURRENCY )); then
          echo "${ts} abort: unexpected clawgym container count ${running}" >>"${SAFETY_LOG}"
          kill -TERM "${RUN_PID}" 2>/dev/null || true
          break
        fi
        if (( high_load_samples >= 3 || low_mem_samples >= 2 || low_disk_samples >= 2 )); then
          echo "${ts} abort: load=${load1} mem_kib=${mem_kib} root_kib=${root_kib}" >>"${SAFETY_LOG}"
          kill -TERM "${RUN_PID}" 2>/dev/null || true
          break
        fi
      else
        ssh_failures=$((ssh_failures + 1))
        printf '%s\tssh_error\n' "${ts}" >>"${REMOTE_LOG}"
        if (( ssh_failures >= 3 )); then
          echo "${ts} abort: three consecutive remote SSH failures" >>"${SAFETY_LOG}"
          kill -TERM "${RUN_PID}" 2>/dev/null || true
          break
        fi
      fi
    fi
    tick=$((tick + 1))
    sleep 5
  done
}

if [[ "$(tr -d '[:space:]' <"${LOAD_CKPT}/latest_checkpointed_iteration.txt")" != "99" ]]; then
  echo "Expected complete step99 checkpoint under ${LOAD_CKPT}"
  exit 1
fi
if [[ "$(find -L "${LOAD_CKPT}/iter_0000099" -maxdepth 1 -type f -name '*.distcp' -size +45G | wc -l)" -ne 8 ]]; then
  echo "step99 checkpoint is incomplete"
  exit 1
fi
if timeout 15 .venv-openclaw/bin/python .venv-openclaw/bin/ray status --address=127.0.0.1:6379 >/dev/null 2>&1; then
  echo "An existing Ray cluster is active; refusing to reuse it"
  exit 1
fi

REMOTE_BASELINE_CONTAINERS="$("${SSH[@]}" "docker ps --filter name=clawgym-rl- -q | wc -l")"
printf 'run_stamp=%s\noutput_root=%s\nremote_base=%s\nrollout_batch_size=8\nn_samples_per_prompt=8\ntarget_trajectories=64\ntrain_workers=%s\nstart_concurrency=%s\naffinity_mode=%s\nremote_baseline_clawgym=%s\n' \
  "${RUN_STAMP}" "${OUTPUT_ROOT}" "${REMOTE_BASE}" "${TRAIN_WORKERS}" "${START_CONCURRENCY}" \
  "${SGLANG_AFFINITY_MODE_OVERRIDE:-hash_bounded}" "${REMOTE_BASELINE_CONTAINERS}" >"${OUTPUT_ROOT}/experiment_meta.env"

stop_existing_burn

echo "[$(date -u +%FT%TZ)] launching route-conflict diagnostic: 8 tasks x 8 trajectories"
env -u NO_LOAD_OPTIM -u NO_LOAD_RNG -u SLIME_NUMERICAL_DIAG_CONFIG \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 ACTOR_GPUS=4 EXPECTED_ACTOR_GPUS=4 ROLLOUT_GPUS=4 SGLANG_TP=2 \
  EXP_NAME="route-conflict-diagnostic-8x8-${RUN_STAMP}" RUN_STAMP="${RUN_STAMP}" \
  SAVE_CKPT="${OUTPUT_ROOT}/runtime" LOAD_CKPT="${LOAD_CKPT}" \
  USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1 OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
  USE_TENSORBOARD=1 TENSORBOARD_DIR="${OUTPUT_ROOT}/tensorboard" TENSORBOARD_PROJECT="rl_32x8_stress" USE_WANDB=0 \
  OPENCLAW_RECORD_ENABLED=1 OPENCLAW_RECORD_FILE="${OUTPUT_ROOT}/proxy_records.jsonl" \
  OPENCLAW_REMOTE_BASE="${REMOTE_BASE}" OPENCLAW_REMOTE_PROXY_PORT="${REVERSE_PORT}" PORT="${LOCAL_PORT}" \
  OPENCLAW_REMOTE_GATEWAY_PORT_MODE=run_scoped OPENCLAW_REMOTE_GATEWAY_PORT_MIN=20000 OPENCLAW_REMOTE_GATEWAY_PORT_MAX=62000 \
  OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS=512 OPENCLAW_REMOTE_START_ATTEMPTS=3 \
  OPENCLAW_GATEWAY_PORT_BASE=16000 OPENCLAW_GATEWAY_PORT_STRIDE=10 OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=5000 OPENCLAW_GATEWAY_PORT_MAX_SLOTS=512 \
  OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY="${START_CONCURRENCY}" OPENCLAW_REMOTE_SSH_POOL_SIZE=8 \
  OPENCLAW_TRAIN_NUM_WORKERS="${TRAIN_WORKERS}" OPENCLAW_TRAIN_SCHEDULING=group_interleaved OPENCLAW_TRAIN_SHUFFLE_SEED=0 \
  OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE=20260829 OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS=1 \
  OPENCLAW_ROUTING_CONFLICT_DIAGNOSTICS=1 \
  NUM_ROLLOUT=101 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=0.8 \
  OPENCLAW_CONTEXT_WINDOW=65536 OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
  OPENCLAW_MAX_AGENT_STEPS=96 OPENCLAW_MAX_COMPACTION_EVENTS=8 OPENCLAW_CHAT_TURN_TIMEOUT=1200 \
  EVAL_INTERVAL=1000 SKIP_EVAL_BEFORE_TRAIN=1 MAX_TOKENS_PER_GPU=81920 \
  MEGATRON_DETERMINISTIC_MODE=1 SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 SGLANG_ENABLE_CUDA_GRAPH=1 \
  SGLANG_SESSION_AFFINITY=1 SGLANG_AFFINITY_MODE="${SGLANG_AFFINITY_MODE_OVERRIDE:-hash_bounded}" SGLANG_AFFINITY_SAMPLES_PER_TASK=8 \
  SGLANG_AFFINITY_MAX_LOAD_DELTA=2 SGLANG_AFFINITY_MIGRATION_LOAD_DELTA=4 SGLANG_AFFINITY_MIGRATION_SUSTAINED_SECONDS=10 \
  USE_ROUTING_REPLAY=1 USE_ROLLOUT_ROUTING_REPLAY=1 USE_TIS=1 TIS_CLIP_LOW=0.0 TIS_CLIP=2.0 \
  KL_LOSS_MONITOR_INTERVAL=5 ENTROPY_COEF=0.0 \
  bash "${LAUNCHER}" &
RUN_PID=$!
rm -f "${STOP_MONITOR_FILE}"
monitor_loop &
MONITOR_PID=$!

set +e
wait "${RUN_PID}"
run_status=$?
set -e
RUN_PID=""
stop_monitor

completed_groups="$(rg -o 'collected [0-9]+ groups' "${OUTPUT_ROOT}"/runtime/logs/train_*.log 2>/dev/null | tail -1 | rg -o '[0-9]+' || true)"
train_perf_lines="$(rg -c 'perf 100:|train 100:' "${OUTPUT_ROOT}"/runtime/logs/train_*.log 2>/dev/null || true)"
printf 'exit_code=%s\ncompleted_groups=%s\ntrain_perf_lines=%s\nfinished_utc=%s\n' \
  "${run_status}" "${completed_groups:-0}" "${train_perf_lines:-0}" "$(date -u +%FT%TZ)" >"${STATUS_FILE}"
echo "[$(date -u +%FT%TZ)] run finished exit=${run_status} groups=${completed_groups:-0}"
exit "${run_status}"
