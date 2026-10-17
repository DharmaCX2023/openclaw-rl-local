#!/usr/bin/env bash
set -euo pipefail
cd /root/openclaw-rl-local
export SAVE_CKPT=/mnt/humf/openclaw-rl-checkpoints/clawgym-artifact-grpo40-20260923-strict2
export LOAD_CKPT="$SAVE_CKPT"
export USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1 NO_LOAD_OPTIM=0 NO_LOAD_RNG=0
export RUN_STAMP=20260923-artifact-grpo40-strict2
export OPENCLAW_ARTIFACT_GRPO_STRICT=0 OPENCLAW_TRAIN_EXCLUDE_TASK_IDS=task_0674
export OPENCLAW_TRAIN_ATTEMPT_BASE="${OPENCLAW_TRAIN_ATTEMPT_BASE:-300}"
export SAVE_INTERVAL=1
# Restore the newest committed checkpoint; iteration30 remains protected.
.venv-openclaw/bin/python - "$LOAD_CKPT" <<'VERIFY'
import sys
from pathlib import Path
sys.path.insert(0,'scripts/span_trace')
from retain_latest_artifact_checkpoint import committed_checkpoint
root=Path(sys.argv[1]);latest=int((root/'latest_checkpointed_iteration.txt').read_text())
assert 30 <= latest < 39 and committed_checkpoint(root,latest), 'No valid unfinished recovery checkpoint'
VERIFY
exec bash scripts/run_artifact_grpo40_20260922.sh
