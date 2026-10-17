#!/usr/bin/env bash
set -euo pipefail

SOURCE=/mnt/humf/openclaw-rl-local
TARGET=/root/openclaw-rl-local
SSH_KEY_SOURCE=/mnt/humf/openclaw_cpu_key
SSH_KEY_TARGET=/root/openclaw_cpu_key

if [ -f "$SSH_KEY_SOURCE" ]; then
  chmod 600 "$SSH_KEY_SOURCE"
  ln -sfn "$SSH_KEY_SOURCE" "$SSH_KEY_TARGET"
fi

if mountpoint -q "$TARGET"; then
  echo "$TARGET is already mounted."
  exit 0
fi

if [ -L "$TARGET" ]; then
  unlink "$TARGET"
elif [ -d "$TARGET" ] && [ -n "$(ls -A "$TARGET")" ]; then
  mv "$TARGET" "${TARGET}.ephemeral-backup-$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$TARGET"
mount --bind "$SOURCE" "$TARGET"
findmnt -T "$TARGET" -o TARGET,SOURCE,FSTYPE,OPTIONS
