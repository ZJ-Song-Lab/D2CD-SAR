#!/bin/bash
# Evaluation script for D2CD-SAR deployment model.
#
# Produces the reproducibility artifacts referenced in the paper's Experiment
# section:
#   predictions_<seed>.json  - COCO-style detection results
#   latency_<seed>.json      - per-image inference latency + FPS
#   metrics_<seed>.json      - mAP@[.5:.95] and AP50 for one seed
#   seed_summary.json        - mean +/- std across seeds (>=2 seeds only)

set -e

# Default configurations
CHECKPOINT="${CHECKPOINT:-./outputs/d2cd_sar/deploy_student.pth}"
DATASET="${DATASET:-ssdd}"
DATA_ROOT="${DATA_ROOT:-./dinov3/data/SSDD}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/d2cd_sar/eval}"
IMG_SIZE="${IMG_SIZE:-640}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEEDS="${SEEDS:-42,123,2024}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.0}"

# Print configuration
echo "========================================"
echo "D2CD-SAR Evaluation"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET"
echo "Data root: $DATA_ROOT"
echo "Split: $SPLIT"
echo "Output dir: $OUTPUT_DIR"
echo "Image size: $IMG_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Seeds: $SEEDS"
echo "Confidence threshold: $CONF_THRESHOLD"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
cd "$(dirname "$0")/.."

python -m dinov3.sar_detection.evaluate \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --seeds "$SEEDS" \
    --conf-threshold "$CONF_THRESHOLD"

echo "Evaluation completed!"
echo "Artifacts saved to: $OUTPUT_DIR"
