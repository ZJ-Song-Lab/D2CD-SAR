# SAR-RTDETR: Training-Time Cross-Modal Distillation for Deployment-Efficient SAR Ship Detection

<p align="center">
  <img src="figures/overall_architecture.jpg" alt="Overall Architecture" width="100%"/>
</p>
<p align="center"><i>Overall architecture of the proposed framework. RT-DETR is the student detector and a frozen DINOv3 vision foundation model is the teacher. The framework jointly decouples representation learning in feature space, parameter space, and optimization space through the proposed DRCP and A²TD-LoRA modules.</i></p>

---

## Overview

Lightweight synthetic aperture radar (SAR) ship detectors are attractive for edge deployment but remain vulnerable to small targets and heterogeneous coastal clutter. **SAR-RTDETR** is a training-time cross-modal distillation framework that transfers dense features from a **frozen DINOv3 teacher** to an **RT-DETR-R18 student**.

- A **Depth-Routed Cross-Modal Purifier (DRCP)** combines multi-level student routing with channel and target-region weighting.
- **Adaptive Task-Decoupled Orthogonal LoRA (A²TD-LoRA)** separates the detection and distillation updates and regulates the latter with a gradient-conditioned gate.
- The teacher and auxiliary alignment modules are **discarded after training**, and the low-rank updates are **merged into the student** — preserving the original deployment graph.

### Highlights

- 🔒 Frozen DINOv3 supervision is used **only during detector training**.
- 🧭 DRCP routes student depth and suppresses modality-specific background transfer.
- ⚖️ A²TD-LoRA separates detection and distillation adaptation subspaces.
- 🧩 Low-rank updates are **merged before deployment**, preserving the student graph.

## Key Results

Across five repeated runs (mean ± standard deviation), SAR-RTDETR increases `AP50:95` from **66.8±0.2 → 71.4±0.2** on SSDD and from **64.0±0.2 → 68.9±0.2** on HRSID.

### Aggregate Comparison on SSDD and HRSID

| Method | Params (M) | GFLOPs | FPS | SSDD AP50 | SSDD AP75 | SSDD AP50:95 | HRSID AP50 | HRSID AP75 | HRSID AP50:95 |
|---|---|---|---|---|---|---|---|---|---|
| Rep-SAR | 5.8 | 15.2 | 155 | 95.8 | 71.0 | 65.5 | 88.5 | 68.9 | 62.1 |
| YOLO11-S | 9.4 | 22.5 | 178 | 96.8 | 72.5 | 67.0 | 90.1 | 70.8 | 64.3 |
| YOLOv12-S | 9.1 | 21.8 | 172 | 97.2 | 73.0 | 68.1 | 91.5 | 71.2 | 65.8 |
| YOLOv13-S | 9.2 | 21.9 | 170 | 97.4 | 73.2 | 68.1 | 91.8 | 71.5 | 66.0 |
| YOLO26-S | 9.0 | 20.5 | 195 | 97.5 | 73.8 | 68.4 | 92.0 | 72.1 | 66.3 |
| RT-DETR-R18 (baseline) | 20.0 | 60.0 | 115 | 96.5 | 72.8 | 66.8 | 89.9 | 69.9 | 64.0 |
| RT-DETRv3-R18 | 20.0 | 60.0 | 115 | 97.0 | 74.3 | 68.2 | 91.2 | 72.0 | 65.9 |
| D-FINE-S | 10.0 | 25.0 | 142 | 97.5 | 74.9 | 68.5 | 91.6 | 72.5 | 66.4 |
| RF-DETR-S | 15.0 | 45.0 | 130 | 97.7 | 75.1 | 68.8 | 91.9 | 72.8 | 66.9 |
| SARES-DEIM-S | 10.5 | 26.5 | 135 | 98.1 | 76.5 | 69.8 | 92.8 | 74.2 | 67.8 |
| **Ours (Baseline + DRCP + A²TD-LoRA)** | **20.0** | **60.0** | **115** | **98.5** | **78.6** | **71.4** | **93.6** | **75.8** | **68.9** |

> The deployed model retains the baseline's parameter count, FLOPs, and measured GPU throughput.

### Controlled Distillation Comparison (RT-DETR-R18 student)

| Distillation Method | SSDD AP50:95 | HRSID AP50:95 | HRSID AP50 |
|---|---|---|---|
| Baseline (RT-DETR-R18) | 66.8 ±0.2 | 64.0 ±0.2 | 89.9 ±0.2 |
| + Direct DINOv3 feature matching | 67.3 ±0.2 | 64.6 ±0.2 | 90.1 ±0.2 |
| + FGD | 67.8 ±0.2 | 65.1 ±0.2 | 90.5 ±0.2 |
| + CWD | 67.5 ±0.2 | 64.8 ±0.2 | 90.3 ±0.2 |
| + CrossKD | 68.6 ±0.2 | 65.9 ±0.2 | 91.2 ±0.2 |
| + FreeKD | 68.9 ±0.2 | 66.2 ±0.2 | 91.6 ±0.2 |
| + DCSF-KD | 69.4 ±0.2 | 66.7 ±0.2 | 92.1 ±0.2 |
| + DIPKD | 70.2 ±0.2 | 67.3 ±0.2 | 92.7 ±0.2 |
| + RT-DETRv4-style injection | 69.6 ±0.2 | 66.9 ±0.2 | 92.3 ±0.2 |
| **+ Ours (DRCP + A²TD-LoRA)** | **71.4 ±0.2** | **68.9 ±0.2** | **93.6 ±0.2** |

SAR-RTDETR exceeds the strongest generic baseline (DCSF-KD) by **+2.0 / +2.2 AP** and the SAR-specific DIPKD baseline by **+1.2 / +1.6 AP** on SSDD / HRSID respectively. On an Ascend 310P, the merged INT8 detector runs at **115 FPS** and loses only **0.4 AP** from FP16 (vs. a 3.5 AP loss for the baseline).

## Method

The framework follows a teacher–student distillation paradigm that is explicitly split into a **training-only distillation branch** and an **inference-only detection pipeline**. It resolves cross-modal conflicts through a unified decoupling strategy across feature, parameter, and optimization spaces.

### DRCP — Depth-Routed Cross-Modal Purifier

<p align="center">
  <img src="figures/drcp.jpg" alt="DRCP" width="100%"/>
</p>
<p align="center"><i>DRCP sequentially performs depth-wise semantic routing to recover diluted scattering responses, followed by joint spatio-channel purification to suppress modality redundancy and spatial interference.</i></p>

Aligning optical-based DINOv3 tokens with CNN-based SAR representations requires bridging conflicts across three dimensions:

1. **Phase 1 — Depth-wise Semantic Routing.** The student backbone feature hierarchy `{S3, S4, S5}` is treated as a historical state sequence. A learnable pseudo-query performs softmax matching against RMSNorm-GAP keys to route a weighted student representation, allowing distillation gradients to bypass the deep-layer bottleneck and directly optimize shallow layers that preserve scattering signatures.
2. **Phase 2 — Joint Spatio-Channel Purification.** A frequency-guided channel gate (1D FFT + bottleneck MLP) suppresses modality redundancy, while a differentiable spatial soft mask derived from local scattering energy and ground-truth geometry (OBB for HRSID, HBB for SSDD) attenuates background responses before the weighted cosine alignment loss.

### A²TD-LoRA — Adaptive Task-Decoupled Orthogonal LoRA

<p align="center">
  <img src="figures/atd_lora.jpg" alt="A²TD-LoRA" width="80%"/>
</p>
<p align="center"><i>A²TD-LoRA introduces two parallel low-rank branches for detection and distillation, enforces orthogonality between their subspaces, and internally regulates the distillation branch through direction-aware variance gating.</i></p>

The AIFI module (the deepest global-reasoning block of the RT-DETR hybrid encoder) is the primary conflict bottleneck and is selected as the sole adaptation site. Two task-specific low-rank branches are injected into the frozen AIFI weight matrix:

- **Orthogonal regularization** (`L_ortho`) on the low-dimensional basis matrices preemptively decouples the detection and distillation parameter subspaces.
- **Direction-aware variance gating** uses a Gradient Consistency Coefficient and a smoothed, clipped gradient-ratio statistic to self-modulate the distillation branch when gradient statistics become unstable.
- **Reparameterization:** both LoRA branches are statically merged into the frozen backbone before deployment, so the final detector shares the original RT-DETR operator graph and tensor shapes.

## Repository Structure

```
dinov3-main/
├── dinov3/
│   ├── sar_detection/                # SAR-RTDETR implementation
│   │   ├── __init__.py
│   │   ├── train.py                  # Distillation training entry (DRCP + A²TD-LoRA)
│   │   ├── train_vit7b.py            # Variant for ViT-7B teacher
│   │   ├── inference.py              # Inference & visualization
│   │   ├── inference_vit7b.py        # Inference variant for ViT-7B
│   │   ├── rtdetr.py                 # RT-DETR-R18 student detector
│   │   ├── drcp.py                   # Depth-Routed Cross-Modal Purifier
│   │   ├── atd_lora.py               # Adaptive Task-Decoupled Orthogonal LoRA
│   │   └── distillation.py           # Cross-modal distillation framework
│   ├── data/
│   │   ├── SSDD/                     # SAR Ship Detection Dataset
│   │   └── HRSID/                    # High-Resolution SAR Images Dataset
│   └── ...
├── figures/                          # Paper figures used in this README
│   ├── overall_architecture.jpg
│   ├── drcp.jpg
│   └── atd_lora.jpg
├── conda.yaml
├── DINOv3_README.md                  # Original DINOv3 documentation
└── README.md                         # This file
```

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.7.1
- CUDA >= 11.8
- GPUs: 2× RTX 4090 (paper setting) or compatible

### Setup

```bash
git clone <this-repo-url>
cd dinov3-main

# Create environment
micromamba env create -f conda.yaml
micromamba activate dinov3

# Additional dependencies
pip install scipy matplotlib
```

### Datasets

Download the SSDD and HRSID datasets from their official sources and place them under `dinov3/data/`:

```
dinov3/data/SSDD/
├── images/
│   ├── train/   (*.jpg)
│   ├── val/     (*.jpg)
│   └── test/    (*.jpg)
├── labels/
│   ├── train/   (*.txt, YOLO HBB format)
│   ├── val/     (*.txt)
│   └── test/    (*.txt)
└── dataset.yaml

dinov3/data/HRSID/
└── hrsid_dataset.py   # HRSID loader (OBB annotations)
```

> Datasets are **not redistributed** in this repository (see `.gitignore`). SSDD uses HBB-derived masks; HRSID supplies native OBBs used to construct orientation-aligned decay fields.

### Teacher Weights

Download the frozen DINOv3 teacher checkpoint from the [official DINOv3 downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and place it where `train.py` expects, or pass the path via the relevant argument. The frozen checkpoint identity and SHA-256 hash are fixed for all primary experiments.

## Quick Start

### Training (Distillation)

Single-GPU:

```bash
python dinov3/sar_detection/train.py \
    --dataset ssdd \
    --data-root ./dinov3/data/SSDD \
    --output-dir ./outputs/sar_rtdetr \
    --img-size 896 \
    --epochs 120 \
    --batch-size 16 \
    --lr 1e-4 \
    --r-lora 16 \
    --lambda-distill 1.0 \
    --lambda-ortho 0.1
```

Multi-GPU (distributed, paper setting uses 2× RTX 4090):

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    dinov3/sar_detection/train.py \
    --dataset hrsid \
    --data-root ./dinov3/data/HRSID \
    --output-dir ./outputs/sar_rtdetr_hrsid \
    --img-size 896 \
    --epochs 120 \
    --batch-size 16
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `ssdd` | Dataset: `ssdd` or `hrsid` |
| `--data-root` | auto | Root dir of dataset |
| `--output-dir` | `./outputs/sar_rtdetr` | Output directory |
| `--img-size` | `896` | Input image size |
| `--num-classes` | `1` | Number of classes (ship) |
| `--num-queries` | `300` | DETR queries |
| `--r-lora` | `16` | LoRA rank |
| `--lambda-distill` | `1.0` | Distillation loss weight |
| `--lambda-ortho` | `0.1` | Orthogonality loss weight |
| `--lambda-sparsity` | `0.01` | Sparsity loss weight |
| `--epochs` | `120` | Training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--weight-decay` | `1e-4` | Weight decay |
| `--save-freq` | `10` | Checkpoint save frequency |

### Inference

```bash
python dinov3/sar_detection/inference.py \
    --image-path /path/to/sar_image.jpg \
    --checkpoint ./outputs/sar_rtdetr/best_checkpoint.pth \
    --output-dir ./outputs/inference \
    --conf-threshold 0.5 \
    --img-size 896
```

### Deployment (LoRA Merging)

After training, the two task-specific LoRA branches are statically merged into the frozen backbone via reparameterization, producing a deployment-ready detector with the **same operator graph and tensor shapes as the RT-DETR student**. The merged model is compatible with FP16/INT8 post-training quantization on edge devices (Ascend 310P, 115 FPS).

## Citation

If you find this work useful, please cite:

```bibtex
@article{sar_rtdetr,
  title   = {SAR-RTDETR: Training-Time Cross-Modal Distillation for Deployment-Efficient SAR Ship Detection},
  author  = {Anonymous Author(s)},
  year    = {2026},
  note    = {Under review}
}
```

This repository builds on the DINOv3 codebase:

```bibtex
@misc{simeoni2025dinov3,
  title         = {{DINOv3}},
  author        = {Sim{\'e}oni, Oriane and Vo, Huy V. and others},
  year          = {2025},
  eprint        = {2508.10104},
  archivePrefix = {arXiv},
}
```

## License

This project follows the DINOv3 License Agreement. See [LICENSE.md](LICENSE.md) for details. The original DINOv3 documentation is preserved in [DINOv3_README.md](DINOv3_README.md).
