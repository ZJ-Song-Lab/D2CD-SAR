# D²CD-SAR: Depth-Routed and Decoupled Cross-Domain Distillation for SAR

<p align="center">
  <img src="figures/overall_architecture.jpg" alt="Overall Architecture" width="100%"/>
</p>
<p align="center"><i>Overall architecture. RT-DETR is the student detector; a frozen DINOv3 vision foundation model is the teacher. The framework decouples representation learning across feature space (DRCP), parameter space (A²TD-LoRA), and optimization space (direction gating). All auxiliary modules are discarded after training; low-rank updates are merged for deployment.</i></p>

---

## Overview

**D²CD-SAR** transfers dense features from a **frozen DINOv3 teacher** to an **RT-DETR-R18 student** via a unified decoupling strategy:

- **DRCP** (Depth-Routed Cross-Modal Purifier) — multi-level routing `{S3, S4, F5}` + channel/spatial weighting
- **A²TD-LoRA** (Adaptive Task-Decoupled Orthogonal LoRA) — two low-rank branches on the AIFI attention output projection with activation-cosine direction gating
- **Deployment** — teacher, DRCP, and gate are discarded; LoRA branches merged into the student graph

## Method

### DRCP — Depth-Routed Cross-Modal Purifier

<p align="center">
  <img src="figures/drcp.jpg" alt="DRCP" width="98%"/>
</p>
<p align="center"><i>Student features are routed across depth; the teacher feature is channel-reweighted by g. The spatial map W_soft weights the token-wise cosine loss before normalized reduction.</i></p>

| Phase | Operation |
|---|---|
| Routing | `{S3, S4, F5}` interpolated to teacher resolution, projected to C_t=768, softmax-routed by learnable query w_d. Same weights (sg) route teacher blocks T3=T⁽³⁾, T4=T⁽⁶⁾, T5=(T⁽⁹⁾+T⁽¹²⁾)/2 |
| Channel gate | GAP → 1D rFFT → magnitude[:192] → MLP(192→512→768) → sigmoid → g |
| Purified teacher | F_tea_hat = g ⊙ F_tea_sp (channel gate only) |
| Spatial weight | HBB token-center-in-box rule, normalized local intensity inside, Gaussian decay outside (μ=0.5, σ=4 token cells) |
| Loss | L_DRCP = Σ W_soft · (1−cos) / (Σ W_soft + ε) |

### A²TD-LoRA — Adaptive Task-Decoupled Orthogonal LoRA

<p align="center">
  <img src="figures/atd_lora.jpg" alt="A²TD-LoRA" width="75%"/>
</p>
<p align="center"><i>Two parallel low-rank branches (rank 100+100) on the AIFI attention output projection. Branch parameters are detached per-loss (sg_θ) while the input z remains differentiable. The direction gate uses activation-cosine probes at z.</i></p>

| Component | Formula |
|---|---|
| Backward routing | sg_θ detaches branch params (not z): ∇_{θ_distill}L_task=0, ∇_{θ_det}L_DRCP=0 |
| Direction gate | ρ = ⟨p_task, p_distill⟩/(‖p_task‖·‖p_distill‖+ε), c_dir=√((1+clip(ρ,−1,1))/2) |
| Magnitude ratio | r_mag=clip(‖p_distill‖₁/(‖p_task‖₁+ε), 0, r_max), r=clip(r_mag·c_dir, 0, 1) |
| EMA gate | w = α·w_prev + (1−α)·r, α=clip(α_base+γ·tanh(σ_r), 0, α_max) |
| Orthogonality | L_ortho = (⟨ΔW_det, ΔW_distill⟩_F / (‖ΔW_det‖_F·‖ΔW_distill‖_F+ε))² |
| Reparameterization | W_deploy = W₀ + B_det·Σ_det·A_det + B_distill·(w·Σ_distill)·A_distill |

## Qualitative Results

<p align="center">
  <img src="figures/qual_detection.jpg" alt="Qualitative Detection Comparison" width="100%"/>
</p>
<p align="center"><i>Qualitative detections in a cluttered inshore scene. GT (green) vs. predictions (blue) from YOLOv13-S, RT-DETRv3-R18, D-FINE-S, SARES-DEIM-S, and D²CD-SAR. Yellow ellipses: missed ships; red ellipses: over-detection errors.</i></p>

## Feature-Response Visualization

<p align="center">
  <img src="figures/heatmap_inshore.png" alt="Inshore Feature Heatmap" width="100%"/>
</p>
<p align="center"><i>Inshore scene. Columns: SAR image, heatmap, overlay. Rows (top→bottom): RT-DETR-R18, +DRCP, +DRCP+A²TD-LoRA. Warmer = stronger response; same within-scene color scale.</i></p>

<p align="center">
  <img src="figures/heatmap_offshore.png" alt="Offshore Feature Heatmap" width="100%"/>
</p>
<p align="center"><i>Offshore scene. Same layout as above. The baseline response spreads across wave clutter; the full model concentrates on target ships.</i></p>

## Reproducibility

### Frozen Configuration

| Group | Registered value |
|---|---|
| Input | 640×640; 8-bit grayscale × 3 channels |
| Teacher | Frozen DINOv3-ViT-B/16; taps {3,6,9,12} |
| LoRA | `self_attn.out_proj.weight`; rank 100+100; Kaiming/0/1 |
| DRCP | 3×(256→768) projections; k_cut=192; 5×5 window |
| Spatial weight | μ=0.5, σ=4 token cells, ε=1e-6, w_flat=0.5 |
| Dynamic gate | r_max=8, K_g=1000, α_base=0.9, γ=0.5, α_max=0.999, w⁰=1.0 |
| Loss weights | λ_distill=1.0, λ_ortho=0.1, λ_sparsity=0.01 |
| Optimizer | AdamW(0.9,0.999); lr=1e-4; wd=1e-4; batch=16 |
| Schedule | 18,002 steps; 750 warmup; cosine→1e-6 |
| Precision | bfloat16; grad clip 1.0; skip invalid steps |
| Augmentation | Mosaic 1.0; MixUp 0.15; affine ±10°/±5%; HSV off; hflip 0.5 |
| Selection | Validation AP50:95 |
| Seeds | (42,123,456,789,1024,2048,3072,4096,5120,6144) |

### Datasets

| Dataset | Images | Ships | Train/Val/Test | Boxes |
|---|---|---|---|---|
| BBox-SSDD | 1160 | 2456 | 928/116/116 | Released HBB |
| HRSID | 5604 | 16951 | 4525/546/533 | COCO HBB |
| LS-SSDD-v1.0 | 9000 | 6015 | 6600/1200/1200 | Released HBB |

### Training

```bash
python -m dinov3.sar_detection.train \
    --dataset ssdd --data-root ./dinov3/data/SSDD \
    --output-dir ./outputs/d2cd_sar \
    --img-size 640 --batch-size 16 --lr 1e-4 --weight-decay 1e-4 \
    --r-lora 100 --lambda-distill 1.0 --lambda-ortho 0.1 --lambda-sparsity 0.01 \
    --seed 42
```

### Evaluation

```bash
python -m dinov3.sar_detection.evaluate \
    --checkpoint outputs/d2cd_sar/deploy_student.pth \
    --dataset ssdd --data-root ./dinov3/data/SSDD \
    --split val --output-dir outputs/d2cd_sar/eval
```

Uses standard pycocotools COCO evaluator (AP@.5:.95, AP50).

### Deployment

Both LoRA branches merge into the frozen AIFI weight: `W_deploy = W₀ + ΔW_det + ΔW_distill`. Teacher, DRCP, probes, and gate are removed. The merged detector preserves the RT-DETR student architecture. Latency, memory, and quantized accuracy are measured on the target device.

## Repository Structure

```
dinov3-main/
├── dinov3/
│   ├── sar_detection/
│   │   ├── train.py              # Distillation training entry
│   │   ├── evaluate.py           # COCO evaluator evaluation
│   │   ├── rtdetr.py             # RT-DETR-R18 student (query selection + AIFI)
│   │   ├── drcp.py               # Depth-Routed Cross-Modal Purifier
│   │   ├── atd_lora.py           # Adaptive Task-Decoupled Orthogonal LoRA
│   │   ├── distillation.py       # Cross-modal distillation framework
│   │   └── test_smoke.py         # Smoke tests
│   └── data/
│       ├── SSDD/                 # SAR Ship Detection Dataset (HBB)
│       └── HRSID/                # High-Resolution SAR Images Dataset (COCO HBB)
├── figures/
│   ├── overall_architecture.jpg
│   ├── drcp.jpg
│   ├── atd_lora.jpg
│   ├── qual_detection.jpg
│   ├── heatmap_inshore.png
│   └── heatmap_offshore.png
├── conda.yaml
└── README.md
```

## Installation

```bash
git clone <this-repo-url>
cd dinov3-main
micromamba env create -f conda.yaml
micromamba activate dinov3
pip install pycocotools scipy matplotlib
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.2, CUDA ≥ 11.8, 1× RTX 4090.

Datasets and teacher weights are not redistributed. Download from official sources.

## Smoke Tests

```bash
python -m dinov3.sar_detection.test_smoke
```

## Citation

```bibtex
@article{d2cd_sar,
  title   = {D²CD-SAR: Depth-Routed and Decoupled Cross-Domain Distillation for SAR},
  author  = {Anonymous Author(s)},
  year    = {2026},
  note    = {Under review}
}
```

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
