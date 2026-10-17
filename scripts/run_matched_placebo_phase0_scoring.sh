#!/usr/bin/env bash
set -euo pipefail

REPO=/root/openclaw-rl-local
MODEL="$REPO/outputs/clawgym-8b-sft-formal-20260902-165551/final_model"
RUN_DIR="$REPO/outputs/matched-placebo-phase0-20260906"
SESSION=matched_placebo_phase0_servers

mkdir -p "$RUN_DIR/server_logs"
tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "tmux session already exists: $SESSION" >&2
  exit 1
}

for gpu in 0 1 2 3 4 5 6 7; do
  port=$((31010 + gpu))
  window="engine_${gpu}"
  command="cd '$REPO' && exec env CUDA_VISIBLE_DEVICES='$gpu' '$REPO/.venv-openclaw/bin/python' -m sglang.launch_server --model-path '$MODEL' --served-model-name 'matched-placebo-phase0-$gpu' --host 127.0.0.1 --port '$port' --tensor-parallel-size 1 --context-length 65536 --mem-fraction-static 0.80 --chunked-prefill-size 8192 --max-prefill-tokens 16384 --max-running-requests 1 --random-seed 20260906 --trust-remote-code --tool-call-parser qwen25 --disable-cuda-graph --enable-deterministic-inference >>'$RUN_DIR/server_logs/engine_${gpu}.log' 2>&1"
  if [[ "$gpu" == 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "$window" "$command"
  else
    tmux new-window -t "$SESSION" -n "$window" "$command"
  fi
done

echo "Started $SESSION on ports 31010-31017"
