#!/bin/bash
# Inference script for SAR Ship Detection

set -e

# Default configurations
CHECKPOINT="${CHECKPOINT:-./outputs/sar_detection/best_checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inference}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.5}"
IMG_SIZE="${IMG_SIZE:-896}"
DEVICE="${DEVICE:-cuda}"

# Check if image path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_sar_image>"
    echo "Environment variables:"
    echo "  CHECKPOINT: Path to model checkpoint (default: $CHECKPOINT)"
    echo "  OUTPUT_DIR: Output directory (default: $OUTPUT_DIR)"
    echo "  CONF_THRESHOLD: Confidence threshold (default: $CONF_THRESHOLD)"
    echo "  IMG_SIZE: Input image size (default: $IMG_SIZE)"
    echo "  DEVICE: Device to use, cuda or cpu (default: $DEVICE)"
    exit 1
fi

IMAGE_PATH="$1"

# Print configuration
echo "========================================"
echo "SAR Ship Detection Inference"
echo "========================================"
echo "Image: $IMAGE_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Confidence threshold: $CONF_THRESHOLD"
echo "Image size: $IMG_SIZE"
echo "Device: $DEVICE"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
cd "$(dirname "$0")/.."

python dinov3/sar_detection/inference.py \
    --image "$IMAGE_PATH" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --conf-threshold $CONF_THRESHOLD \
    --img-size $IMG_SIZE

echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR"
