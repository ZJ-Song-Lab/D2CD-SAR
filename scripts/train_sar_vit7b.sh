#!/bin/bash
# Training script for SAR Ship Detection with DINOv3 ViT-7B/16
# Optimized for 6x RTX 5880 GPUs
# ViT-7B/16: 6.7B parameters, pretrained on SAT-493M satellite dataset

set -e

# Default configurations
DATA_ROOT="${DATA_ROOT:-./dinov3/data/SSDD}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/sar_detection_vit7b}"
IMG_SIZE="${IMG_SIZE:-896}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"  # Reduced for 7B model
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Multi-GPU configuration
NGPU="${NGPU:-6}"

# Print configuration
echo "========================================"
echo "SAR Ship Detection Training"
echo "Model: DINOv3 ViT-7B/16"
echo "Backbone: 6.7B parameters, SAT-493M"
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

# Check GPU memory
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo "========================================"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch distributed training
cd "$(dirname "$0")/.."

# Use torchrun for better distributed training support
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NGPU \
    dinov3/sar_detection/train_vit7b.py \
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
echo ""
echo "To run inference:"
echo "  bash scripts/inference_sar_vit7b.sh /path/to/image.jpg"
