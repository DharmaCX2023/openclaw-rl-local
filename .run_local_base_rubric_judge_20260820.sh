#!/usr/bin/env bash
set -uo pipefail
ROOT=/root/openclaw-rl-local
OUTPUT_DIR=${ROOT}/outputs/local-base-rubric-judge-20260820
LAUNCH_LOG=${OUTPUT_DIR}.launch.log
SERVER_LOG=${OUTPUT_DIR}.sglang.log
SERVER_PID=
restore_burn() {
  if ! tmux has-session -t rl_e2e_post_burn_4_7 2>/dev/null; then
    tmux new-session -d -s rl_e2e_post_burn_4_7 \
      "cd /root && exec env CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29572 --nproc_per_node=4 synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>/tmp/rl_e2e_post_burn_4_7.log 2>&1"
  fi
}
cleanup() {
  if test -n "${SERVER_PID}" && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  restore_burn
}
trap cleanup EXIT
cd "${ROOT}" || exit 1
mkdir -p "${OUTPUT_DIR}"
setsid env CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION=1 .venv-openclaw/bin/python -m sglang.launch_server \
  --model-path /mnt/OpenClaw-RL/models/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name clawgym-base-local-rubric-judge \
  --host 127.0.0.1 \
  --port 30580 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --context-length 65536 \
  --mem-fraction-static 0.7 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --random-seed 12345 \
  --trust-remote-code \
  --tool-call-parser qwen25 \
  --disable-cuda-graph \
  --enable-deterministic-inference \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:30580/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    break
  fi
  sleep 5
done
if (( ready != 1 )); then
  echo "judge server failed readiness" >>"${LAUNCH_LOG}"
  exit 2
fi
.venv-openclaw/bin/python scripts/judge_rubric_artifacts_local.py \
  --official-evaluator-root /tmp/clawgym-bench-official-20260820/evaluation/localclawbench \
  --dataset /mnt/OpenClaw-RL/data/clawgym-bench/benchmark_data.jsonl \
  --base-artifacts "${ROOT}/outputs/base-model-eval-clawgym46-rubric-deterministic-20260820/rubric_artifacts" \
  --sft-artifacts "${ROOT}/outputs/sft-model-eval-clawgym46-rubric-deterministic-20260820/rubric_artifacts" \
  --base-code-report "${ROOT}/outputs/base-model-eval-clawgym200-deterministic-20260820/eval_report.json" \
  --sft-code-report "${ROOT}/outputs/sft-model-eval-clawgym200-deterministic-20260820/eval_report.json" \
  --base-url http://127.0.0.1:30580/v1 \
  --model clawgym-base-local-rubric-judge \
  --output-dir "${OUTPUT_DIR}/results" \
  --workers 8 \
  --seed-base 540000 \
  --timeout 300 \
  >>"${LAUNCH_LOG}" 2>&1
status=$?
echo "LOCAL_BASE_RUBRIC_JUDGE_STATUS=${status}" >>"${LAUNCH_LOG}"
exit "${status}"
