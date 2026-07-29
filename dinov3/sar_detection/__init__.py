# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
SAR Ship Detection with DINOv3

Two model options available:
1. ViT-L/16 (300M parameters) - Faster, less memory
2. ViT-7B/16 (6.7B parameters) - Higher accuracy, more memory

Both models are pretrained on satellite imagery (SAT-493M).
"""

# ViT-L/16 imports
from .train import main as train_main
from .evaluate import main as evaluate_main
from .inference import detect, visualize_detection, visualize_heatmap

# ViT-7B/16 imports
from .train_vit7b import main as train_main_vit7b
from .inference_vit7b import detect as detect_vit7b
from .inference_vit7b import visualize_detection as visualize_detection_vit7b
from .inference_vit7b import visualize_heatmap as visualize_heatmap_vit7b

__all__ = [
    # ViT-L/16
    "train_main",
    "evaluate_main",
    "detect",
    "visualize_detection",
    "visualize_heatmap",
    # ViT-7B/16
    "train_main_vit7b",
    "detect_vit7b",
    "visualize_detection_vit7b",
    "visualize_heatmap_vit7b",
]
