#!/bin/bash
# Training script for SAR Ship Detection on multi-GPU server
# Optimized for 6x RTX 5880 GPUs

set -e

# Default configurations
DATA_ROOT="${DATA_ROOT:-./dinov3/data/SSDD}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/sar_detection}"
IMG_SIZE="${IMG_SIZE:-896}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Multi-GPU configuration
NGPU="${NGPU:-6}"

# Print configuration
echo "========================================"
echo "SAR Ship Detection Training"
echo "========================================"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Image size: $IMG_SIZE"
echo "Epochs: $EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((NGPU * BATCH_SIZE))"
echo "Learning rate: $LR"
echo "GPUs: $NGPU"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch distributed training
cd "$(dirname "$0")/.."

python -m torch.distributed.launch \
    --nproc_per_node=$NGPU \
    --use_env \
    dinov3/sar_detection/train.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --img-size $IMG_SIZE \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --save-freq 10

echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
