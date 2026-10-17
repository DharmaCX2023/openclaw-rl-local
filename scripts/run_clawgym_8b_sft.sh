#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT=/root/openclaw-rl-local
VENV="$ROOT/.venv-openrlhf-8b"
MODEL="$ROOT/models/Qwen3-8B-yarn2-64k"
STAMP="$(date -u +%Y%m%d-%H%M%S)"

case "$MODE" in
  smoke)
    DATA="$ROOT/data/clawgym-trajectory/sft_qwen3_8b_smoke16"
    RUN="$ROOT/outputs/clawgym-8b-sft-smoke-$STAMP"
    MASTER_PORT=29783
    TRAIN_BATCH=8
    MAX_EPOCHS=1
    MAX_SAMPLES=16
    SAVE_STEPS=100000
    CKPT_ARGS=(--ckpt.disable_ds)
    ;;
  formal)
    DATA="$ROOT/data/clawgym-trajectory/sft_qwen3_8b_len64k_openrlhf"
    RUN="$ROOT/outputs/clawgym-8b-sft-formal-$STAMP"
    MASTER_PORT=29784
    TRAIN_BATCH=128
    MAX_EPOCHS=3
    MAX_SAMPLES=1000000
    SAVE_STEPS=80
    CKPT_ARGS=(--ckpt.max_num 10 --ckpt.max_mem 1800)
    ;;
  *)
    echo "usage: $0 {smoke|formal}" >&2
    exit 2
    ;;
esac

RUN="${SFT_RUN_DIR:-$RUN}"
MASTER_PORT="${SFT_MASTER_PORT:-$MASTER_PORT}"
RESUME="${SFT_RESUME:-0}"
PRECREATED_RUN="${SFT_PRECREATED_RUN:-0}"
CKPT_LOAD_ARGS=()
test -x "$VENV/bin/deepspeed"
test -d "$MODEL"
test -d "$DATA"
if ! "$VENV/bin/python" -c 'import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()' "$MASTER_PORT"; then
  echo "master port ${MASTER_PORT} is already occupied" >&2
  exit 1
fi
if [[ "$RESUME" == 1 ]]; then
  if [[ ! -f "$RUN/ds_checkpoints/latest" ]]; then
    echo "resume requested but no DeepSpeed latest checkpoint exists: $RUN/ds_checkpoints/latest" >&2
    exit 1
  fi
  CKPT_LOAD_ARGS=(--ckpt.load_enable)
elif [[ -e "$RUN" ]]; then
  # A detached supervisor pre-creates only status/log metadata. Accept that
  # scaffold when explicitly declared, but never overwrite training artifacts.
  if [[ "$PRECREATED_RUN" != 1 ]] \
    || [[ ! -d "$RUN/logs" ]] \
    || [[ -e "$RUN/launch_snapshot.sh" ]] \
    || [[ -e "$RUN/logs/train.log" ]] \
    || [[ -e "$RUN/ds_checkpoints/latest" ]] \
    || find "$RUN/final_model" -maxdepth 1 -type f -name '*.safetensors' -print -quit 2>/dev/null | grep -q .; then
    echo "refusing to reuse output directory: $RUN" >&2
    exit 1
  fi
fi

mkdir -p "$RUN/logs" "$RUN/tensorboard" "$RUN/ds_checkpoints" "$RUN/final_model"
if [[ "$RESUME" != 1 ]]; then
  cp "$0" "$RUN/launch_snapshot.sh"
  {
    echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mode=$MODE"
    echo "run_dir=$RUN"
    echo "model=$MODEL"
    echo "dataset=$DATA"
    echo "train_batch=$TRAIN_BATCH"
    echo "micro_batch=1"
    echo "epochs=$MAX_EPOCHS"
    echo "save_steps=$SAVE_STEPS"
    echo "intermediate_hf=false"
    echo "final_hf=$RUN/final_model"
    "$VENV/bin/python" -c 'import torch, transformers, deepspeed; print(f"torch={torch.__version__}\ntransformers={transformers.__version__}\ndeepspeed={deepspeed.__version__}")'
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    df -h /root
  } > "$RUN/logs/run_manifest.txt"
fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=$MODE resume=$RESUME" >> "$RUN/logs/lifecycle.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

exec "$VENV/bin/deepspeed" \
  --master_port "$MASTER_PORT" \
  --num_gpus 8 \
  --module openrlhf.cli.train_sft \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$DATA" \
  --data.input_key messages \
  --data.max_len 65536 \
  --data.max_samples "$MAX_SAMPLES" \
  --data.apply_chat_template \
  --data.multiturn \
  --data.dataloader_num_workers 4 \
  --train.batch_size "$TRAIN_BATCH" \
  --train.micro_batch_size 1 \
  --train.max_epochs "$MAX_EPOCHS" \
  --train.seed 42 \
  --ds.zero_stage 2 \
  --ds.param_dtype bf16 \
  --ds.attn_implementation flash_attention_2 \
  --ds.packing_samples \
  --ds.ring_attn_size 4 \
  --ds.ring_attn_head_stride 4 \
  --ds.overlap_comm \
  --data.length_bucket_size "$TRAIN_BATCH" \
  --model.gradient_checkpointing_enable \
  --optim adam \
  --adam.lr 1e-5 \
  --max_norm 1.0 \
  --logger.logging_steps 1 \
  --logger.tensorboard_dir "$RUN/tensorboard" \
  --eval.steps 100000 \
  --ckpt.output_dir "$RUN/final_model" \
  --ckpt.path "$RUN/ds_checkpoints" \
  --ckpt.save_steps "$SAVE_STEPS" \
  "${CKPT_LOAD_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  2>&1 | tee -a "$RUN/logs/train.log"
