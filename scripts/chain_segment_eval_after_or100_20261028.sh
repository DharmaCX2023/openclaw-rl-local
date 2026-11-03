#!/usr/bin/env bash
# Wait for the local or100 natural eval queue to finish, then run the
# segment-r3 iter99 natural eval with the same protocol.
set -uo pipefail
ORLOG=/mnt/humf/openclaw-rl-checkpoints/local-or100-natural-eval-20261028/queue.log
while ! grep -q '^QUEUE_DONE' "$ORLOG"; do sleep 60; done
exec /root/openclaw-rl-local/.venv-openclaw/bin/python \
  /root/openclaw-rl-local/scripts/run_segment_r3_iter99_natural_eval_20261028.py
