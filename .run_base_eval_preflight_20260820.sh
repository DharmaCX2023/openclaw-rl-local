#!/usr/bin/env bash
set -euo pipefail

cd /root/openclaw-rl-local
exec .venv-openclaw/bin/python scripts/run_fixed_seed_8x8_rollout_gate.py \
  --eval-mode \
  --model /mnt/OpenClaw-RL/models/Qwen3-30B-A3B-Instruct-2507 \
  --tasks /root/openclaw-rl-local/.base_eval_preflight_tasks_20260820 \
  --output-dir /root/openclaw-rl-local/outputs/base-model-eval-preflight-deterministic-20260820 \
  --task-count 1 \
  --samples-per-task 1 \
  --parallel 1 \
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
  --remote-base /root/clawgym_base_eval_preflight_20260820 \
  --gateway-port-base 32000 \
  --gateway-train-offset 1000 \
  --gateway-port-stride 2 \
  --ssh-pool-size 8 \
  --rollout-timeout 1200
