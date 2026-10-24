#!/usr/bin/env bash
# Train / evaluate LayoutLMv3 on FUNSD or CORD (base or large), optionally uploading results to Google Drive.
#
#   bash scripts/run_baseline.sh eval-released                      # FUNSD base released ckpt (README numbers)
#   SIZE=large bash scripts/run_baseline.sh eval-released           # FUNSD large released ckpt
#   GPUS=1 SEED=42 bash scripts/run_baseline.sh train               # FUNSD base, README recipe
#   DATASET=cord LR=5e-5 EFF_BATCH=64 STEPS=1000 bash scripts/run_baseline.sh train   # CORD (values UNVERIFIED)
#   SIZE=large LR=... bash scripts/run_baseline.sh train            # large: README gives no recipe, LR must be set
#
# Env vars: DATASET=funsd|cord  SIZE=base|large  GPUS  SEED  LR  STEPS  EFF_BATCH  PER_DEVICE  OUT  PORT
#           USE_BOUNDARY_LOSS=0|1  BOUNDARY_LOSS_WEIGHT (default 0.2)  BOUNDARY_POS_WEIGHT (default 5.0)
#           DRIVE_MOUNT (e.g. /content/drive/MyDrive) or DRIVE_REMOTE (e.g. gdrive)
#           DRIVE_DIR (default thesis/layoutlmv3) -> upload run dir when finished
#
# Defaults:
#   * FUNSD (base / large): lr 1e-5, 1000 steps, effective batch 16 (= 8 GPUs x 2 in the README).
#   * CORD  (base / large): lr 5e-5, 1000 steps, effective batch 64.
# Effective batch is kept constant for any GPU count via gradient accumulation.
# Run from layoutlmv3/ with the conda env active.
set -euo pipefail

MODE="${1:?usage: $0 eval-released|train}"
DATASET="${DATASET:-funsd}"
SIZE="${SIZE:-base}"
GPUS="${GPUS:-1}"
SEED="${SEED:-42}"
PORT="${PORT:-4398}"
PER_DEVICE="${PER_DEVICE:-2}"
OUT="${OUT:-$HOME/runs}"
USE_BOUNDARY_LOSS="${USE_BOUNDARY_LOSS:-0}"
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.2}"
BOUNDARY_POS_WEIGHT="${BOUNDARY_POS_WEIGHT:-5.0}"
SAVE_MODEL="${SAVE_MODEL:-0}"

if [ "$DATASET" = "funsd" ]; then
  LR="${LR:-1e-5}"; STEPS="${STEPS:-1000}"; EFF_BATCH="${EFF_BATCH:-16}"
elif [ "$DATASET" = "cord" ]; then
  LR="${LR:-5e-5}"; STEPS="${STEPS:-1000}"; EFF_BATCH="${EFF_BATCH:-64}"
else
  : "${LR:?set LR}"
  : "${STEPS:?set STEPS}"
  : "${EFF_BATCH:?set EFF_BATCH}"
fi

DENOM=$((PER_DEVICE * GPUS))
if [ $((EFF_BATCH % DENOM)) -ne 0 ]; then echo "EFF_BATCH ($EFF_BATCH) must be divisible by PER_DEVICE*GPUS ($DENOM)"; exit 1; fi
ACCUM=$((EFF_BATCH / DENOM))

LAUNCH="python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT"
COMMON="--dataset_name $DATASET --segment_level_layout 1 --visual_embed 1 --input_size 224 --dataloader_num_workers 8 --seed $SEED --overwrite_output_dir"

BOUNDARY_ARGS=""
RUN_TAG="baseline"
if [ "$USE_BOUNDARY_LOSS" = "1" ] || [ "$USE_BOUNDARY_LOSS" = "true" ]; then
  BOUNDARY_ARGS="--use_boundary_loss True --boundary_loss_weight $BOUNDARY_LOSS_WEIGHT --boundary_pos_weight $BOUNDARY_POS_WEIGHT"
  RUN_TAG="boundary-w${BOUNDARY_LOSS_WEIGHT}-pw${BOUNDARY_POS_WEIGHT}"
fi

RUN_DIR="$OUT/$DATASET-$SIZE-$RUN_TAG-seed$SEED"

case "$MODE" in
  eval-released)
    [ "$DATASET" = "funsd" ] || { echo "README only lists released FUNSD checkpoints"; exit 1; }
    RUN_DIR="$OUT/released-$DATASET-$SIZE"
    $LAUNCH examples/run_funsd_cord.py $COMMON --do_eval \
      --model_name_or_path "HYPJUDY/layoutlmv3-$SIZE-finetuned-funsd" \
      --output_dir "$RUN_DIR"
    ;;
  train)
    $LAUNCH examples/run_funsd_cord.py $COMMON --do_train --do_eval \
      --model_name_or_path "microsoft/layoutlmv3-$SIZE" \
      --output_dir "$RUN_DIR" \
      --max_steps "$STEPS" --save_steps -1 --evaluation_strategy steps --eval_steps 100 \
      --learning_rate "$LR" --per_device_train_batch_size "$PER_DEVICE" --gradient_accumulation_steps "$ACCUM" \
      $BOUNDARY_ARGS
    if [ "$SAVE_MODEL" = "0" ]; then
      rm -f "$RUN_DIR/pytorch_model.bin" "$RUN_DIR/optimizer.pt" "$RUN_DIR/scheduler.pt"
    fi
    ;;
  *) echo "unknown mode $MODE"; exit 1;;
esac

cat "$RUN_DIR/eval_results.json"

# Save the exact settings next to the results (for the thesis' reproducibility appendix)
{
  echo "mode=$MODE dataset=$DATASET size=$SIZE gpus=$GPUS seed=$SEED lr=${LR:-n/a} steps=${STEPS:-n/a} eff_batch=${EFF_BATCH:-n/a}"
  echo "use_boundary_loss=$USE_BOUNDARY_LOSS boundary_loss_weight=$BOUNDARY_LOSS_WEIGHT boundary_pos_weight=$BOUNDARY_POS_WEIGHT"
  echo "git=$(git rev-parse HEAD 2>/dev/null) dirty=$(git status --porcelain 2>/dev/null | wc -l)"
  nvidia-smi --query-gpu=name --format=csv,noheader | head -1
} > "$RUN_DIR/run_info.txt"

if [ -n "${DRIVE_REMOTE:-}" ] || [ -n "${DRIVE_MOUNT:-}" ]; then
  bash "$(dirname "$0")/sync_to_drive.sh" "$RUN_DIR"
fi

