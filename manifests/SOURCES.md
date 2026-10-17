# Local RL workspace provenance

This directory is a private working snapshot. Training code and new outputs must
remain under `/root/openclaw-rl-local`; mounted paths are external read-only
inputs unless explicitly stated otherwise.

## Copied sources

- Harness source: `/mnt/Openclaw-RL-hareness`
- Runtime slime source used by the previous launcher: `/mnt/OpenClaw-RL/slime`
- Runtime Megatron source used by the previous launcher: `/mnt/OpenClaw-RL/Megatron-LM`
- Harness slime comparison copy: `runtime_sources/harness_slime`
- Harness Megatron comparison copy: `runtime_sources/harness_megatron`

The local `slime/` and `Megatron-LM/` directories come from the runtime stack,
not from the similarly named directories in the harness source.

## External read-only assets

- SFT checkpoint: `/mnt/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127`
- Historical RL outputs: `/mnt/Openclaw-RL-hareness/outputs`
- Python environment: `/mnt/OpenClaw-RL/.venv-openclaw`

No symlink to historical outputs is created. New checkpoints and logs belong in
`/root/openclaw-rl-local/outputs`.

## Safety boundary

- Never set `SAVE_CKPT` below `/mnt`.
- Resolve and print all code/model/output paths before a launch.
- Verify the recorded hashes and package versions before formal training.
- Pin the remote gateway image by digest before formal training.
