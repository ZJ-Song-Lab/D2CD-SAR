# SAR Ship Detection with DINOv3

This project implements SAR (Synthetic Aperture Radar) ship detection using Meta's DINOv3 vision foundation model, fine-tuned on the SSDD (SAR Ship Detection Dataset).

## Features

- **DINOv3 Backbone**: Uses ViT-L/16 pretrained on satellite imagery (SAT-493M)
- **DETR Detection Head**: End-to-end object detection with transformer architecture
- **Multi-GPU Training**: Optimized for distributed training on 6x RTX 5880 GPUs
- **Visualization**: Outputs detection bounding boxes and attention heatmaps

## Project Structure

```
dinov3/
├── sar_detection/
│   ├── __init__.py
│   ├── train.py          # Training script
│   └── inference.py      # Inference script with visualization
├── data/SSDD/
│   ├── ssdd_dataset.py   # SSDD dataset loader
│   └── transforms.py     # Data augmentation and preprocessing
└── scripts/
    ├── train_sar.sh      # Training launch script
    └── inference_sar.sh  # Inference launch script
```

## Dataset Format

The SSDD dataset uses YOLO OBB (Oriented Bounding Box) format:
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Where coordinates are normalized [0, 1]. The dataset is automatically converted to axis-aligned bounding boxes for training.

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.7.1
- CUDA >= 11.8
- 6x NVIDIA RTX 5880 GPUs (or compatible)

### Setup Environment

```bash
# Clone and navigate to repository
cd dinov3-main

# Create conda environment
micromamba env create -f conda.yaml
micromamba activate dinov3

# Install additional dependencies
pip install scipy matplotlib
```

## Training

### Quick Start

```bash
# Set environment variables (optional)
export DATA_ROOT="./dinov3/data/SSDD"
export OUTPUT_DIR="./outputs/sar_detection"
export EPOCHS=50
export BATCH_SIZE=4

# Run training
bash scripts/train_sar.sh
```

### Advanced Configuration

```bash
# Custom training configuration
python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --use_env \
    dinov3/sar_detection/train.py \
    --data-root ./dinov3/data/SSDD \
    --output-dir ./outputs/sar_detection \
    --img-size 896 \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --num-workers 8 \
    --save-freq 10
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--img-size` | 896 | Input image size |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 4 | Batch size per GPU |
| `--lr` | 1e-4 | Learning rate |
| `--lr-drop` | 40 | Epoch to drop learning rate |
| `--weight-decay` | 1e-4 | Weight decay |
| `--num-workers` | 4 | Data loading workers |
| `--save-freq` | 10 | Checkpoint save frequency |

## Inference

### Single Image Detection

```bash
# Basic usage
bash scripts/inference_sar.sh /path/to/sar_image.jpg

# With custom checkpoint
CHECKPOINT=./outputs/sar_detection/best_checkpoint.pth \
CONF_THRESHOLD=0.6 \
bash scripts/inference_sar.sh /path/to/sar_image.jpg
```

### Python API

```python
from dinov3.sar_detection.inference import detect

# Run detection
result = detect(
    image_path="/path/to/sar_image.jpg",
    checkpoint_path="./outputs/sar_detection/best_checkpoint.pth",
    output_dir="./outputs/inference",
    device="cuda",
    conf_threshold=0.5,
    img_size=896,
)

# Access detection results
boxes = result["boxes"]      # Bounding boxes [N, 4] (x1, y1, x2, y2)
scores = result["scores"]    # Confidence scores [N]
labels = result["labels"]    # Class labels [N]
```

### Output Files

For each input image, the following outputs are generated:

1. **Detection Visualization** (`{image_name}_detection.jpg`):
   - Original SAR image with bounding boxes
   - Confidence scores for each detection
   - Red boxes with yellow labels

2. **Heatmap Visualization** (`{image_name}_heatmap.jpg`):
   - Original image
   - Attention heatmap from DINOv3 features
   - Overlay of heatmap with detection boxes

## Model Architecture

### Backbone
- **Model**: DINOv3 ViT-L/16
- **Pretraining**: SAT-493M (satellite imagery)
- **Parameters**: 300M
- **Patch Size**: 16x16

### Detection Head
- **Architecture**: DETR (DEtection TRansformer)
- **Decoder**: Global RPE Decomposition Decoder
- **Queries**: 300 one-to-one + 300 one-to-many
- **Feature Levels**: 4
- **Decoder Layers**: 6

### Training Strategy
- **Backbone**: Frozen (feature extractor)
- **Detection Head**: Trainable
- **Optimizer**: AdamW
- **Loss**: DETR loss with auxiliary losses

## Performance Tips

### Multi-GPU Training
- The script automatically uses all available GPUs
- Batch size is per GPU, so total batch size = `batch_size * num_gpus`
- For 6x RTX 5880, recommended batch size is 4 per GPU (total 24)

### Memory Optimization
- Reduce `--img-size` if running out of memory
- Reduce `--batch-size` if needed
- Use gradient accumulation for effective larger batches

### Inference Speed
- Use `--device cuda` for GPU acceleration
- Batch inference can be implemented for multiple images
- First inference may be slower due to model loading

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size or image size
   --batch-size 2 --img-size 768
   ```

2. **DINOv3 Weights Not Found**:
   - Download weights from [Meta AI](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
   - Place in appropriate directory or use direct URL

3. **Dataset Not Found**:
   - Ensure SSDD dataset is in `./dinov3/data/SSDD`
   - Check folder structure: `images/train`, `images/val`, `labels/train`, `labels/val`

### Getting Help

For issues related to:
- **DINOv3**: See [official repository](https://github.com/facebookresearch/dinov3)
- **SSDD Dataset**: Check dataset documentation
- **This implementation**: Open an issue in this repository

## Citation

If you use this code for your research, please cite:

```bibtex
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and others},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
}
```

## License

This project follows the DINOv3 License Agreement. See LICENSE.md for details.
