#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/openclaw-rl-local
OUTPUT_DIR=${ROOT}/outputs/base-model-eval-clawgym200-deterministic-20260820
LAUNCH_LOG=${ROOT}/outputs/base-model-eval-clawgym200-deterministic-20260820.launch.log

restore_burn() {
  if ! tmux has-session -t rl_e2e_post_burn_4_7 2>/dev/null; then
    tmux new-session -d -s rl_e2e_post_burn_4_7 \
      "cd /root && exec env CUDA_VISIBLE_DEVICES=4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29572 --nproc_per_node=4 synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>/tmp/rl_e2e_post_burn_4_7.log 2>&1"
  fi
}

trap restore_burn EXIT

cd "${ROOT}" || exit 1
.venv-openclaw/bin/python scripts/run_fixed_seed_8x8_rollout_gate.py \
  --eval-mode \
  --model /mnt/OpenClaw-RL/models/Qwen3-30B-A3B-Instruct-2507 \
  --tasks /root/openclaw-rl-local/clawgym-official/data/clawgym_eval \
  --output-dir "${OUTPUT_DIR}" \
  --task-count 200 \
  --samples-per-task 1 \
  --parallel 32 \
  --rollout-id 0 \
  --deterministic \
  --sampling-seed-base 424242 \
  --deterministic-tool-call-ids \
  --gpus 4,5,6,7 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --backend-port 30380 \
  --proxy-port 30381 \
  --reverse-port 39381 \
  --served-model clawgym-base-eval-deterministic \
  --remote-base /root/clawgym_base_eval_200_20260820 \
  --gateway-port-base 32000 \
  --gateway-train-offset 1000 \
  --gateway-port-stride 2 \
  --ssh-pool-size 8 \
  --rollout-timeout 1200 \
  >>"${LAUNCH_LOG}" 2>&1
status=$?
printf 'BASE_EVAL_EXIT_STATUS=%s\n' "${status}" >>"${LAUNCH_LOG}"
exit "${status}"
