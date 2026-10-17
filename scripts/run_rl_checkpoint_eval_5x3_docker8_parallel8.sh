#!/usr/bin/env bash
set -uo pipefail

REPO=/root/openclaw-rl-local
TRAIN_RUN="$REPO/outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700"
LAUNCHER="$REPO/clawgym-official/.run_step39_eval_sync_20260812-091700.sh"
BATCH_STAMP="${BATCH_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
BATCH_ROOT="${BATCH_ROOT:-$REPO/outputs/clawgym-rl-eval-steps20-100-3runs-docker8-parallel8-$BATCH_STAMP}"
SUMMARY="$BATCH_ROOT/run_summary.tsv"
QUEUE_LOG="$BATCH_ROOT/queue.log"

mkdir -p "$BATCH_ROOT"
if [ ! -f "$SUMMARY" ]; then
  printf 'requested_step\titeration\trepetition\texit_code\toutput_dir\tresult_line\n' > "$SUMMARY"
fi

exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "batch_root=$BATCH_ROOT"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "docker_start_concurrency=8"
echo "eval_workers=8"

for requested_step in 20 40 60 80 100; do
  iteration=$((requested_step - 1))
  padded=$(printf '%07d' "$iteration")
  checkpoint="$TRAIN_RUN/iter_$padded"

  if [ ! -f "$checkpoint/.metadata" ] || [ ! -f "$checkpoint/common.pt" ]; then
    echo "checkpoint incomplete: $checkpoint"
    exit 1
  fi

  for repetition in 1 2 3; do
    run_stamp=$(date -u +%Y%m%d-%H%M%S)
    exp_name="clawgym-rl-step${requested_step}-iter${iteration}-eval-r${repetition}-docker8-parallel8-$run_stamp"
    out="$BATCH_ROOT/step${requested_step}/run${repetition}-$run_stamp"
    load_view="$out/checkpoint_load"
    mkdir -p "$load_view"
    ln -s "$checkpoint" "$load_view/iter_$padded"
    printf '%s\n' "$iteration" > "$load_view/latest_checkpointed_iteration.txt"

    echo "run_start step=$requested_step iteration=$iteration repetition=$repetition at=$(date -u +%Y-%m-%dT%H:%M:%SZ) out=$out"
    env \
      RUN_STAMP="$run_stamp" \
      EXP_NAME="$exp_name" \
      SAVE_CKPT="$out" \
      LOAD_CKPT="$load_view" \
      OPENCLAW_DISABLE_CHECKPOINT_SAVE=1 \
      OPENCLAW_RECORD_FILE="$out/official_proxy_records.jsonl" \
      TENSORBOARD_DIR="$out/tensorboard" \
      OPENCLAW_REMOTE_PROXY_PORT=39101 \
      PORT=30101 \
      OPENCLAW_GATEWAY_PORT_BASE=32000 \
      OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET=3000 \
      OPENCLAW_GATEWAY_PORT_MAX_SLOTS=256 \
      OPENCLAW_BUNDLED_PLUGINS_DIR="${OPENCLAW_BUNDLED_PLUGINS_DIR:-/app/dist/extensions}" \
      OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY=8 \
      OPENCLAW_REMOTE_SSH_POOL_SIZE=8 \
      OPENCLAW_TRAIN_NUM_WORKERS=8 \
      OPENCLAW_EVAL_NUM_WORKERS=8 \
      OPENCLAW_EVAL_N_SAMPLES=1 \
      OPENCLAW_MAX_RESPONSE_TOKENS=8192 \
      OPENCLAW_TRAJECTORY_MAX_TOKENS=65536 \
      OPENCLAW_CONTEXT_WINDOW=65536 \
      OPENCLAW_MAX_AGENT_STEPS=96 \
      OPENCLAW_MAX_COMPACTION_EVENTS=8 \
      ROLLOUT_TEMPERATURE=0.7 \
      ROLLOUT_TOP_P=1.0 \
      bash "$LAUNCHER"
    exit_code=$?

    log=$(find "$out/logs" -type f -name '*.log' 2>/dev/null | sort | tail -1)
    result_line=""
    if [ -n "$log" ]; then
      result_line=$(rg 'eval 0: \{' "$log" | tail -1 || true)
    fi
    result_line=${result_line//$'\t'/ }
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$requested_step" "$iteration" "$repetition" "$exit_code" "$out" "$result_line" >> "$SUMMARY"
    echo "run_end step=$requested_step iteration=$iteration repetition=$repetition exit_code=$exit_code at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "result=$result_line"

    if [ "$exit_code" -ne 0 ]; then
      echo "queue stopped after failed run; inspect $log"
      exit "$exit_code"
    fi
  done
done

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "summary=$SUMMARY"
