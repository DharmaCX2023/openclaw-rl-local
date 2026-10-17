#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd -P)"
LAUNCHER="${REPO_ROOT}/clawgym-official/run_qwen3_30b_a3b_official_remote_grpo.sh"
TEST_TMP="$(mktemp -d)"
TEST_OUTPUT="${REPO_ROOT}/outputs/.length-preflight-${BASHPID}"
trap 'rm -rf -- "${TEST_TMP}"' EXIT

test ! -e "${TEST_OUTPUT}"
env -u OPENCLAW_MAX_TOKENS -u ENTROPY_COEF -u TENSORBOARD_DIR \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/valid.log"
test ! -e "${TEST_OUTPUT}"

grep -qx 'OPENCLAW_MAX_RESPONSE_TOKENS=8192' "${TEST_TMP}/valid.log"
grep -qx 'OPENCLAW_TRAJECTORY_MAX_TOKENS=65536' "${TEST_TMP}/valid.log"
grep -qx 'OPENCLAW_CONTEXT_WINDOW=65536' "${TEST_TMP}/valid.log"
grep -qx 'MAX_TOKENS_PER_GPU=81920' "${TEST_TMP}/valid.log"
grep -qx 'SGLANG_ENABLE_NGRAM_SPECULATIVE=0' "${TEST_TMP}/valid.log"
grep -qx 'MEGATRON_DETERMINISTIC_MODE=0' "${TEST_TMP}/valid.log"
grep -qx 'OPENCLAW_MAX_AGENT_STEPS=96' "${TEST_TMP}/valid.log"
grep -qx 'OPENCLAW_MAX_COMPACTION_EVENTS=8' "${TEST_TMP}/valid.log"
grep -qx 'OPENCLAW_TRAIN_SCHEDULING=sequential' "${TEST_TMP}/valid.log"
grep -qx 'ENTROPY_COEF=0.0' "${TEST_TMP}/valid.log"
grep -qx 'KL_LOSS_MONITOR_INTERVAL=5' "${TEST_TMP}/valid.log"
grep -qx 'EVAL_INTERVAL=20' "${TEST_TMP}/valid.log"
grep -qx 'SAVE_INTERVAL=20' "${TEST_TMP}/valid.log"
grep -qx 'CHECKPOINT_RETENTION=all' "${TEST_TMP}/valid.log"
grep -qx 'USE_TENSORBOARD=1' "${TEST_TMP}/valid.log"
grep -qx "TENSORBOARD_DIR=${TEST_OUTPUT}/tensorboard" "${TEST_TMP}/valid.log"
grep -qx 'CHECKPOINT_PATH_SOURCE=explicit' "${TEST_TMP}/valid.log"
grep -qx 'OPENCLAW_EVAL_NUM_WORKERS=32' "${TEST_TMP}/valid.log"
grep -qx 'OPENCLAW_EVAL_N_SAMPLES=1' "${TEST_TMP}/valid.log"
grep -qx 'OPENCLAW_GATEWAY_PORT_BASE=16000' "${TEST_TMP}/valid.log"
grep -qx 'OPENCLAW_GATEWAY_LAST_PORT=60990' "${TEST_TMP}/valid.log"

env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  MEGATRON_DETERMINISTIC_MODE=1 \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/deterministic.log"
grep -qx 'MEGATRON_DETERMINISTIC_MODE=1' "${TEST_TMP}/deterministic.log"
grep -qx 'NCCL_ALGO=Ring' "${TEST_TMP}/deterministic.log"
grep -qx 'NVTE_ALLOW_NONDETERMINISTIC_ALGO=0' "${TEST_TMP}/deterministic.log"
grep -qx 'CUBLAS_WORKSPACE_CONFIG=:4096:8' "${TEST_TMP}/deterministic.log"

env -u OPENCLAW_MAX_TOKENS -u SAVE_CKPT \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  LOCAL_CHECKPOINT_MIN_FREE_GB=999999999 \
  EXP_NAME=checkpoint-fallback-test \
  bash "${LAUNCHER}" >"${TEST_TMP}/fallback.log"
grep -qx \
  'SAVE_CKPT=/mnt/humf/openclaw-rl-checkpoints/checkpoint-fallback-test' \
  "${TEST_TMP}/fallback.log"
grep -qx 'CHECKPOINT_PATH_SOURCE=mnt_fallback' "${TEST_TMP}/fallback.log"
test ! -e /mnt/humf/openclaw-rl-checkpoints/checkpoint-fallback-test

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  OPENCLAW_MAX_RESPONSE_TOKENS=65537 \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-order.log" 2>&1; then
  echo "response > trajectory was accepted" >&2
  exit 1
fi
grep -q 'Response limit must not exceed trajectory limit' "${TEST_TMP}/bad-order.log"

if OPENCLAW_PREFLIGHT_ONLY=1 \
  OPENCLAW_MAX_TOKENS=65536 \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/legacy.log" 2>&1; then
  echo "ambiguous legacy variable was accepted" >&2
  exit 1
fi
grep -q 'OPENCLAW_MAX_TOKENS is ambiguous' "${TEST_TMP}/legacy.log"

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  OPENCLAW_TRAIN_SHUFFLE_SEED=invalid \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-shuffle-seed.log" 2>&1; then
  echo "invalid task shuffle seed was accepted" >&2
  exit 1
fi
grep -q 'OPENCLAW_TRAIN_SHUFFLE_SEED must be a non-negative integer' "${TEST_TMP}/bad-shuffle-seed.log"

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  SGLANG_ENABLE_NGRAM_SPECULATIVE=invalid \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-ngram.log" 2>&1; then
  echo "invalid NGRAM toggle was accepted" >&2
  exit 1
fi
grep -q 'SGLANG_ENABLE_NGRAM_SPECULATIVE must be 0 or 1' "${TEST_TMP}/bad-ngram.log"

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  MEGATRON_DETERMINISTIC_MODE=invalid \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-deterministic-mode.log" 2>&1; then
  echo "invalid Megatron deterministic toggle was accepted" >&2
  exit 1
fi
grep -q 'MEGATRON_DETERMINISTIC_MODE must be 0 or 1' "${TEST_TMP}/bad-deterministic-mode.log"

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  KL_LOSS_MONITOR_INTERVAL=0 \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-kl-monitor-interval.log" 2>&1; then
  echo "zero KL monitor interval was accepted" >&2
  exit 1
fi
grep -q 'KL_LOSS_MONITOR_INTERVAL must be a positive integer' "${TEST_TMP}/bad-kl-monitor-interval.log"

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  OPENCLAW_TRAIN_SCHEDULING=unknown \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-train-scheduling.log" 2>&1; then
  echo "unknown train scheduling was accepted" >&2
  exit 1
fi
grep -q 'OPENCLAW_TRAIN_SCHEDULING must be sequential or group_interleaved' "${TEST_TMP}/bad-train-scheduling.log"

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  EVAL_INTERVAL=0 \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-eval-interval.log" 2>&1; then
  echo "zero eval interval was accepted" >&2
  exit 1
fi
grep -q 'EVAL_INTERVAL_VALUE must be a positive integer' "${TEST_TMP}/bad-eval-interval.log"

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  OPENCLAW_GATEWAY_PORT_BASE=65000 \
  OPENCLAW_GATEWAY_PORT_MAX_SLOTS=100 \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-gateway-range.log" 2>&1; then
  echo "overflowing gateway port range was accepted" >&2
  exit 1
fi
grep -q 'Gateway port range exceeds 65535' "${TEST_TMP}/bad-gateway-range.log"

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  USE_TIS=invalid \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-tis-toggle.log" 2>&1; then
  echo "invalid TIS toggle was accepted" >&2
  exit 1
fi
grep -q 'USE_TIS must be 0 or 1' "${TEST_TMP}/bad-tis-toggle.log"

if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  USE_TIS=1 \
  TIS_CLIP=1.0 \
  TIS_CLIP_LOW=2.0 \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-tis-bounds.log" 2>&1; then
  echo "invalid TIS bounds were accepted" >&2
  exit 1
fi
grep -q 'TIS bounds require' "${TEST_TMP}/bad-tis-bounds.log"


if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  OPENCLAW_OPSD_ENABLE=1 \
  OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE="${TEST_TMP}/missing-anchors.json" \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-opsd-anchors.log" 2>&1; then
  echo "OPSD accepted a missing evidence anchor file" >&2
  exit 1
fi
grep -q 'OPSD evidence anchor JSON not found' "${TEST_TMP}/bad-opsd-anchors.log"

printf '%s\n' '{}' >"${TEST_TMP}/opsd-anchors.json"
if env -u OPENCLAW_MAX_TOKENS \
  OPENCLAW_PREFLIGHT_ONLY=1 \
  OPENCLAW_OPSD_ENABLE=1 \
  USE_ROLLOUT_ROUTING_REPLAY=0 \
  OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE="${TEST_TMP}/opsd-anchors.json" \
  SAVE_CKPT="${TEST_OUTPUT}" \
  bash "${LAUNCHER}" >"${TEST_TMP}/bad-opsd-routing.log" 2>&1; then
  echo "OPSD accepted disabled rollout routing replay" >&2
  exit 1
fi
grep -q 'OPENCLAW_OPSD_ENABLE requires USE_ROLLOUT_ROUTING_REPLAY=1' "${TEST_TMP}/bad-opsd-routing.log"

echo "length and seed configuration preflight tests passed"
