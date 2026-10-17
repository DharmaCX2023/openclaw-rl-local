#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/openclaw-rl-local
RUN="${1:?usage: $0 RUN_DIR}"
STATUS="$RUN/status"
LIFECYCLE="$RUN/logs/lifecycle.log"
BURN_SESSION=pypenguin_phase57_full_burn
BURN_LOG=/tmp/pypenguin_phase57_full_burn.log

start_burn() {
  if tmux has-session -t "$BURN_SESSION" 2>/dev/null; then
    return
  fi
  if pgrep -f 'torchrun --standalone --master_port=29632 .*synthetic_train_stress.py' >/dev/null; then
    return
  fi
  tmux new-session -d -s "$BURN_SESSION" "cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/local/bin/torchrun --standalone --master_port=29632 --nproc_per_node=8 synthetic_train_stress.py --steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 --seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 >>'$BURN_LOG' 2>&1"
}

mkdir -p "$RUN/logs"
printf 'RUNNING\n' > "$STATUS"
printf '%s supervisor_started pid=%s ppid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$PPID" >> "$LIFECYCLE"
git -C "$ROOT/third_party/OpenRLHF" rev-parse HEAD > "$RUN/logs/openrlhf_base_commit.txt"
git -C "$ROOT/third_party/OpenRLHF" diff > "$RUN/logs/openrlhf_worktree.patch"
git -C "$ROOT/third_party/OpenRLHF" status --short > "$RUN/logs/openrlhf_status.txt"

set +e
SFT_RUN_DIR="$RUN" SFT_RESUME="${SFT_RESUME:-0}" SFT_PRECREATED_RUN=1 \
  SFT_MASTER_PORT="${SFT_MASTER_PORT:-29784}" \
  bash "$ROOT/scripts/run_clawgym_8b_sft.sh" formal
rc=$?
set -e

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "$rc" -eq 0 ]] && find "$RUN/final_model" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .; then
  printf 'COMPLETED rc=0 finished_at=%s\n' "$finished_at" > "$STATUS"
  printf '%s supervisor_completed rc=0\n' "$finished_at" >> "$LIFECYCLE"
else
  printf 'FAILED rc=%s finished_at=%s\n' "$rc" "$finished_at" > "$STATUS"
  printf '%s supervisor_failed rc=%s\n' "$finished_at" "$rc" >> "$LIFECYCLE"
fi
sync
start_burn
exit "$rc"
