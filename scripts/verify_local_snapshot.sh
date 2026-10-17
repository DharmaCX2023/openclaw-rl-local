#!/bin/bash
set -euo pipefail

LOCAL_ROOT=/root/openclaw-rl-local
ASSET_ROOT=/mnt/OpenClaw-RL
FULL_CHECK=0

if [ "${1:-}" = "--full" ]; then
  FULL_CHECK=1
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--full]" >&2
  exit 2
fi

if [ -L "${LOCAL_ROOT}" ] || [ "$(readlink -f "${LOCAL_ROOT}")" != "${LOCAL_ROOT}" ]; then
  echo "local root must be a real directory: ${LOCAL_ROOT}" >&2
  exit 1
fi

for path in \
  "${LOCAL_ROOT}/clawgym-official" \
  "${LOCAL_ROOT}/slime" \
  "${LOCAL_ROOT}/Megatron-LM" \
  "${LOCAL_ROOT}/outputs"; do
  if [ ! -d "${path}" ] || [ -L "${path}" ]; then
    echo "required local directory is missing or symlinked: ${path}" >&2
    exit 1
  fi
done

if find "${LOCAL_ROOT}" -path "${LOCAL_ROOT}/.git" -prune -o -type l -lname '/mnt/*' -print -quit | grep -q .; then
  echo "local code tree contains a symlink into /mnt" >&2
  exit 1
fi

if [ -n "${SAVE_CKPT:-}" ]; then
  resolved_parent=$(readlink -f "$(dirname "${SAVE_CKPT}")")
  case "${resolved_parent}/" in
    "${LOCAL_ROOT}/outputs/"*) ;;
    *) echo "SAVE_CKPT must be below ${LOCAL_ROOT}/outputs: ${SAVE_CKPT}" >&2; exit 1 ;;
  esac
fi

test -d "${ASSET_ROOT}/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127"
test -x "${ASSET_ROOT}/.venv-openclaw/bin/python"

sha256sum -c "${LOCAL_ROOT}/manifests/task_data_sha256.txt" >/dev/null

if [ "${FULL_CHECK}" = "1" ]; then
  sha256sum -c "${LOCAL_ROOT}/manifests/sft_checkpoint_sha256.txt" >/dev/null
fi

if [ -n "$(git -C "${LOCAL_ROOT}" remote)" ]; then
  echo "local snapshot unexpectedly has a configured Git remote" >&2
  exit 1
fi

"${ASSET_ROOT}/.venv-openclaw/bin/python" -c \
  'import ray, sglang, torch; print(f"ray={ray.__version__} sglang={sglang.__version__} torch={torch.__version__}")'
echo "snapshot_ok root=${LOCAL_ROOT} full_check=${FULL_CHECK}"
