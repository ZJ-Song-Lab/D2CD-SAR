# SAR-RTDETR: Training-Time Cross-Modal Distillation for Deployment-Compatible SAR Ship Detection

<p align="center">
  <img src="figures/overall_architecture.jpg" alt="Overall Architecture" width="100%"/>
</p>
<p align="center"><i>Overall architecture of the proposed framework. RT-DETR is the student detector and a frozen DINOv3 vision foundation model is the teacher. The framework jointly decouples representation learning in feature space, parameter space, and optimization space through the proposed DRCP and A²TD-LoRA modules.</i></p>

---

## Overview

Lightweight synthetic aperture radar (SAR) ship detectors are attractive for edge deployment but remain vulnerable to small targets and heterogeneous coastal clutter. **SAR-RTDETR** is a training-time cross-modal distillation framework that transfers dense features from a **frozen DINOv3 teacher** to an **RT-DETR-R18 student**.

- A **Depth-Routed Cross-Modal Purifier (DRCP)** combines multi-level student routing `{S3, S4, F5}` with channel and target-region weighting.
- **Adaptive Task-Decoupled Orthogonal LoRA (A²TD-LoRA)** separates the detection and distillation updates via two low-rank branches on the AIFI attention output projection and regulates the distillation branch with an activation-cosine direction gate.
- The teacher and auxiliary alignment modules are **discarded after training**, and the low-rank updates are **merged into the student** — preserving the original deployment graph.

## Method

The framework follows a teacher–student distillation paradigm that is explicitly split into a **training-only distillation branch** and an **inference-only detection pipeline**. It resolves cross-modal conflicts through a unified decoupling strategy across feature, parameter, and optimization spaces.

### DRCP — Depth-Routed Cross-Modal Purifier

<p align="center">
  <img src="figures/drcp.jpg" alt="DRCP" width="100%"/>
</p>
<p align="center"><i>DRCP sequentially performs depth-wise semantic routing to recover diluted scattering responses, followed by channel and spatial weighting before the token-wise cosine alignment loss.</i></p>

1. **Phase 1 — Depth-wise Routing.** The student hierarchy `{S3, S4, F5}` (where `F5 = AIFI(S5)`) is interpolated to the teacher resolution and projected to `C_t=768`. A learnable routing query `w_d` performs softmax matching against `RMSNorm(GAP(U_i))` keys. The same routing weights (with stop-gradient) are applied to the teacher blocks `T3=T^{(3)}`, `T4=T^{(6)}`, `T5=(T^{(9)}+T^{(12)})/2`.
2. **Phase 2 — Channel and Spatial Weighting.** A channel gate is computed as `GAP → 1D rFFT → magnitude[:k_cut] → MLP(192→512→768) → sigmoid`, producing `g`. The purified teacher is `F_tea_hat = g ⊙ F_tea_sp` (channel gate only; `W_soft` is used solely as a spatial loss weight). The spatial weight `W_soft` uses horizontal bounding boxes with a token-center-in-box rule, normalized local intensity inside boxes, and Gaussian decay outside.

### A²TD-LoRA — Adaptive Task-Decoupled Orthogonal LoRA

<p align="center">
  <img src="figures/atd_lora.jpg" alt="A²TD-LoRA" width="80%"/>
</p>
<p align="center"><i>A²TD-LoRA introduces two parallel low-rank branches (rank 100+100) on the AIFI attention output projection, enforces scale-normalized orthogonality between realized updates, and regulates the distillation branch through activation-cosine direction gating.</i></p>

- **Loss-specific backward routing:** `sg_θ` detaches branch *parameters* (not the input `z`), so `∇_{θ_distill} L_task = 0` and `∇_{θ_det} L_DRCP = 0` while both losses remain differentiable w.r.t. the shared AIFI input activation `z`.
- **Activation-cosine direction gating:** probes `p_task = ∇_z L_task` and `p_distill = ∇_z L_DRCP` are computed at the AIFI input. Direction agreement `c_dir = sqrt((1+clip(ρ,-1,1))/2)` and magnitude ratio `r_mag` combine into `r = clip(r_mag·c_dir, 0, 1)`, driving an EMA gate `w`.
- **Scale-normalized orthogonality:** `L_ortho = (<ΔW_det, ΔW_distill>_F / (||ΔW_det||_F ||ΔW_distill||_F + ε))²` on the realized low-rank updates.
- **Reparameterization:** both LoRA branches are statically merged into the frozen AIFI weight before deployment.

## Reproducibility

### Frozen Configuration

All reported runs use one versioned implementation and one run registry. The frozen hyperparameters that directly control the proposed modules:

| Group | Hyperparameter | Registered value |
|---|---|---|
| Input | Resolution; channel construction | 640×640; 8-bit grayscale replicated to three channels |
| Teacher | Model; taps | Frozen DINOv3-ViT-B/16; one-based {3,6,9,12} |
| LoRA | Target; ranks; initialization | `self_attn.out_proj.weight`; 100+100; Kaiming/0/1 for A/B/Σ |
| DRCP | Student projections; spectral cutoff; local window | 3×(256→768); k_cut=192; 5×5 |
| Spatial weight | μ, σ, ε, w_flat | 0.5, 4 token cells, 1e-6, 0.5 |
| Dynamic gate | r_max, K_g, α_base, γ, α_max, w⁰ | 8, 1000, 0.9, 0.5, 0.999, 1.0 |
| Loss | λ_distill, λ_ortho, λ_sparsity | 1.0, 0.1, 0.01 |
| Optimization | AdamW; LR; weight decay; batch | (0.9,0.999); 1e-4; 1e-4; 16 |
| Schedule | Scheduled steps; warm-up; decay; min LR | 18,002; 750 steps; cosine; 1e-6 |
| Numerical | Mixed precision; grad clip; invalid step | bfloat16; max norm 1.0; skip without extending schedule |
| Augmentation | Mosaic; MixUp; affine; HSV; hflip | 1.0; 0.15; ±10°/±5%; disabled; 0.5 |
| Selection | Checkpoint criterion | Validation AP50:95 |
| Seeds | Seed list | (42,123,456,789,1024,2048,3072,4096,5120,6144) |

### Datasets

| Dataset | Images | Ships | Train/Val/Test | Source scenes | Detection boxes |
|---|---|---|---|---|---|
| BBox-SSDD | 1160 | 2456 | 928/116/116 | 85 | Released HBB |
| HRSID | 5604 | 16951 | 4525/546/533 | 136 | COCO HBB |
| LS-SSDD-v1.0 | 9000 | 6015 | 6600/1200/1200 | 15 | Released HBB |

Chips from the same source SAR scene are assigned to exactly one of train, validation, and test. The artifact publishes the scene-to-chip map and immutable split manifests.

### Training

```bash
python -m dinov3.sar_detection.train \
    --dataset ssdd \
    --data-root ./dinov3/data/SSDD \
    --output-dir ./outputs/sar_rtdetr \
    --img-size 640 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --r-lora 100 \
    --lambda-distill 1.0 \
    --lambda-ortho 0.1 \
    --lambda-sparsity 0.01 \
    --seed 42
```

Training uses 18,002 scheduled steps with 750-step linear warm-up followed by cosine decay to 1e-6. Checkpoint selection uses validation AP50:95. The gate state and running statistics are part of the checkpoint.

### Evaluation

```bash
python -m dinov3.sar_detection.evaluate \
    --checkpoint outputs/sar_rtdetr/deploy_student.pth \
    --dataset ssdd --data-root ./dinov3/data/SSDD \
    --split val --output-dir outputs/sar_rtdetr/eval
```

Evaluation uses the standard pycocotools COCO evaluator (AP@.5:.95, AP50). Multiple seeds aggregate mean ± std.

### Deployment (Reparameterization)

After training, both LoRA branches are merged into the frozen AIFI weight via:

```
W_deploy = W0 + B_det·Σ_det·A_det + B_distill·(w*·Σ_distill)·A_distill
```

The teacher, DRCP, activation probes, and gate update are removed. The merged detector preserves the registered RT-DETR student architecture and tensor interface. Actual latency, memory, and quantized accuracy are empirical properties measured on the target device rather than inferred from graph structure.

## Repository Structure

```
dinov3-main/
├── dinov3/
│   ├── sar_detection/                # SAR-RTDETR implementation
│   │   ├── __init__.py
│   │   ├── train.py                  # Distillation training entry
│   │   ├── evaluate.py               # Standalone evaluation (COCO evaluator)
│   │   ├── rtdetr.py                 # RT-DETR-R18 student (query selection + AIFI)
│   │   ├── drcp.py                   # Depth-Routed Cross-Modal Purifier
│   │   ├── atd_lora.py               # Adaptive Task-Decoupled Orthogonal LoRA
│   │   ├── distillation.py           # Cross-modal distillation framework
│   │   └── test_smoke.py             # Smoke tests
│   ├── data/
│   │   ├── SSDD/                     # SAR Ship Detection Dataset (HBB)
│   │   └── HRSID/                    # High-Resolution SAR Images Dataset (COCO HBB)
│   └── ...
├── figures/
│   ├── overall_architecture.jpg
│   ├── drcp.jpg
│   └── atd_lora.jpg
├── conda.yaml
└── README.md
```

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.2
- CUDA >= 11.8
- GPU: 1× RTX 4090 (paper setting)

### Setup

```bash
git clone <this-repo-url>
cd dinov3-main

# Create environment
micromamba env create -f conda.yaml
micromamba activate dinov3

# Additional dependencies
pip install pycocotools scipy matplotlib
```

### Datasets

Download SSDD and HRSID from their official sources and place them under `dinov3/data/`. Both datasets provide horizontal bounding boxes (HBB). Datasets are not redistributed in this repository.

### Teacher Weights

Download the frozen DINOv3-ViT-B/16 checkpoint from the [official DINOv3 downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). The frozen checkpoint identity and hash are fixed for all primary experiments.

## Smoke Tests

```bash
python -m dinov3.sar_detection.test_smoke
```

## Citation

```bibtex
@article{sar_rtdetr,
  title   = {SAR-RTDETR: Training-Time Cross-Modal Distillation for Deployment-Compatible SAR Ship Detection},
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

This project follows the DINOv3 License Agreement. See [LICENSE.md](LICENSE.md) for details.
