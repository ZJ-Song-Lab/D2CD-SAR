#!/bin/bash
# Inference script for SAR Ship Detection with DINOv3 ViT-7B/16

set -e

# Default configurations
CHECKPOINT="${CHECKPOINT:-./outputs/sar_detection_vit7b/best_checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inference_vit7b}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.5}"
IMG_SIZE="${IMG_SIZE:-896}"
DEVICE="${DEVICE:-cuda}"

# Check if image path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_sar_image>"
    echo ""
    echo "Environment variables:"
    echo "  CHECKPOINT: Path to model checkpoint (default: $CHECKPOINT)"
    echo "  OUTPUT_DIR: Output directory (default: $OUTPUT_DIR)"
    echo "  CONF_THRESHOLD: Confidence threshold (default: $CONF_THRESHOLD)"
    echo "  IMG_SIZE: Input image size (default: $IMG_SIZE)"
    echo "  DEVICE: Device to use, cuda or cpu (default: $DEVICE)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/sar_image.jpg"
    echo "  CHECKPOINT=./my_model.pth $0 /path/to/sar_image.jpg"
    exit 1
fi

IMAGE_PATH="$1"

# Print configuration
echo "========================================"
echo "SAR Ship Detection Inference"
echo "Model: DINOv3 ViT-7B/16"
echo "========================================"
echo "Image: $IMAGE_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Confidence threshold: $CONF_THRESHOLD"
echo "Image size: $IMG_SIZE"
echo "Device: $DEVICE"
echo "========================================"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please train the model first or specify a valid checkpoint path."
    exit 1
fi

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image not found: $IMAGE_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
cd "$(dirname "$0")/.."

python dinov3/sar_detection/inference_vit7b.py \
    --image "$IMAGE_PATH" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --conf-threshold $CONF_THRESHOLD \
    --img-size $IMG_SIZE

echo "========================================"
echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
