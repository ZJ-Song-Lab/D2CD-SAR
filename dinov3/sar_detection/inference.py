# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Inference script for SAR Ship Detection using DINOv3
Outputs: Detection bounding boxes + Heatmap visualization
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dinov3.hub.backbones import dinov3_vitl16
from dinov3.eval.detection.models.detr import build_model, PostProcess
from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.eval.detection.models.position_encoding import PositionEncoding
from dinov3.eval.detection.util.misc import nested_tensor_from_tensor_list
from dinov3.data.SSDD.transforms import make_val_transforms


class SARDetectorInference:
    """SAR Ship Detector for inference"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        img_size: int = 896,
        conf_threshold: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.conf_threshold = conf_threshold

        # Build model
        self.model, self.config, self.postprocessor = self._build_model()

        # Load checkpoint
        self._load_checkpoint(checkpoint_path)

        # Transforms
        self.transform = make_val_transforms(img_size=img_size)

        self.model.eval()

    def _build_model(self):
        """Build detection model"""
        detection_kwargs = dict(
            with_box_refine=True,
            two_stage=True,
            mixed_selection=True,
            look_forward_twice=True,
            k_one2many=6,
            lambda_one2many=1.0,
            num_queries_one2one=300,
            num_queries_one2many=300,
            reparam=True,
            position_embedding=PositionEncoding.SINE,
            num_feature_levels=1,
            dec_layers=6,
            dim_feedforward=2048,
            dropout=0.0,
            norm_type="pre_norm",
            proposal_feature_levels=4,
            proposal_min_size=20,
            decoder_type="global_rpe_decomp",
            decoder_use_checkpoint=False,
            decoder_rpe_hidden_dim=512,
            decoder_rpe_type="linear",
            layers_to_use=None,
            blocks_to_train=None,
            add_transformer_encoder=True,
            num_encoder_layers=6,
            backbone_use_layernorm=False,
            num_classes=2,  # ship + background
            aux_loss=True,
            topk=300,
            hidden_dim=768,
            nheads=8,
        )

        config = DetectionHeadConfig(**detection_kwargs)

        # Load backbone
        backbone = dinov3_vitl16(pretrained=False, weights=None, check_hash=False)

        # Configure
        config.n_windows_sqrt = 2
        config.proposal_in_stride = backbone.patch_size
        config.proposal_tgt_strides = [int(m * backbone.patch_size) for m in (0.5, 1, 2, 4)]

        if config.layers_to_use is None:
            config.layers_to_use = [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]

        detector = build_model(backbone, config)
        detector.num_queries = detector.num_queries_one2one
        detector.transformer.two_stage_num_proposals = detector.num_queries

        postprocessor = PostProcess(config.topk, config.reparam)

        return detector.to(self.device), config, postprocessor

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Load state dict
        self.model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully")

    @torch.no_grad()
    def detect(self, image_path: str) -> Tuple[List[dict], torch.Tensor]:
        """
        Detect ships in SAR image

        Returns:
            results: List of detection dicts with 'boxes', 'scores', 'labels'
            features: Feature maps for heatmap visualization
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # Apply transforms
        image_tensor, _ = self.transform(image, {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,))})
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Create nested tensor
        samples = nested_tensor_from_tensor_list([image_tensor[0]])

        # Forward pass
        outputs = self.model(samples)

        # Get target size
        target_sizes = torch.tensor([[self.img_size, self.img_size]], device=self.device)
        orig_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)

        # Post-process
        results = self.postprocessor(outputs, target_sizes, orig_sizes)

        # Filter by confidence
        filtered_results = []
        for result in results:
            mask = result["scores"] > self.conf_threshold
            filtered_result = {
                "boxes": result["boxes"][mask],
                "scores": result["scores"][mask],
                "labels": result["labels"][mask],
            }
            filtered_results.append(filtered_result)

        # Extract features for heatmap
        features = self._extract_features(samples)

        return filtered_results[0], features

    @torch.no_grad()
    def _extract_features(self, samples) -> torch.Tensor:
        """Extract feature maps from backbone"""
        # Get features from backbone
        features, _ = self.model.backbone(samples)

        # Use the first feature level
        feat = features[0].tensors

        return feat


def visualize_detection(
    image_path: str,
    detection_result: dict,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    score_threshold: float = 0.5,
) -> np.ndarray:
    """
    Visualize detection results with bounding boxes

    Args:
        image_path: Path to input image
        detection_result: Detection result dict with 'boxes', 'scores', 'labels'
        output_path: Path to save visualization (optional)
        figsize: Figure size
        score_threshold: Minimum score to display

    Returns:
        visualization: Numpy array of visualization
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Create figure
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image_np)

    # Get detections
    boxes = detection_result["boxes"].cpu().numpy()
    scores = detection_result["scores"].cpu().numpy()
    labels = detection_result["labels"].cpu().numpy()

    # Draw boxes
    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue

        # Box coordinates (xyxy format)
        x1, y1, x2, y2 = box

        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add label
        label_text = f"Ship: {score:.2f}"
        ax.text(
            x1,
            y1 - 5,
            label_text,
            color="red",
            fontsize=10,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    ax.set_title("SAR Ship Detection Results")
    ax.axis("off")

    plt.tight_layout()

    # Save or return
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Detection visualization saved to {output_path}")

    # Convert to numpy array
    fig.canvas.draw()
    visualization = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    visualization = visualization.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return visualization


def visualize_heatmap(
    image_path: str,
    features: torch.Tensor,
    detection_result: dict,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Visualize attention heatmap overlay

    Args:
        image_path: Path to input image
        features: Feature maps from backbone
        detection_result: Detection result for reference
        output_path: Path to save visualization (optional)
        figsize: Figure size
        alpha: Heatmap transparency

    Returns:
        visualization: Numpy array of visualization
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    # Process features for heatmap
    # Average across channels
    feat_map = features.mean(dim=1)[0].cpu().numpy()

    # Normalize
    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)

    # Resize to original image size
    from scipy.ndimage import zoom

    zoom_h = orig_h / feat_map.shape[0]
    zoom_w = orig_w / feat_map.shape[1]
    feat_map_resized = zoom(feat_map, (zoom_h, zoom_w), order=1)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original SAR Image")
    axes[0].axis("off")

    # Heatmap only
    im = axes[1].imshow(feat_map_resized, cmap="jet")
    axes[1].set_title("Attention Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(image_np)
    axes[2].imshow(feat_map_resized, cmap="jet", alpha=alpha)

    # Draw detection boxes on overlay
    boxes = detection_result["boxes"].cpu().numpy()
    scores = detection_result["scores"].cpu().numpy()

    for box, score in zip(boxes, scores):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="white",
            facecolor="none",
        )
        axes[2].add_patch(rect)

    axes[2].set_title("Overlay with Detections")
    axes[2].axis("off")

    plt.tight_layout()

    # Save or return
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap visualization saved to {output_path}")

    # Convert to numpy array
    fig.canvas.draw()
    visualization = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    visualization = visualization.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return visualization


def detect(
    image_path: str,
    checkpoint_path: str,
    output_dir: str = "./outputs/inference",
    device: str = "cuda",
    conf_threshold: float = 0.5,
    img_size: int = 896,
):
    """
    Run detection on a single image and save visualizations

    Args:
        image_path: Path to input SAR image
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save outputs
        device: Device to run inference on
        conf_threshold: Confidence threshold
        img_size: Input image size
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    print("Initializing detector...")
    detector = SARDetectorInference(
        checkpoint_path=checkpoint_path,
        device=device,
        img_size=img_size,
        conf_threshold=conf_threshold,
    )

    # Run detection
    print(f"Processing image: {image_path}")
    result, features = detector.detect(image_path)

    # Print detection summary
    num_detections = len(result["boxes"])
    print(f"Detected {num_detections} ships")
    for i, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
        print(f"  Ship {i+1}: score={score:.3f}, box={box.tolist()}")

    # Generate visualizations
    image_name = Path(image_path).stem

    # Detection visualization
    det_output = output_dir / f"{image_name}_detection.jpg"
    visualize_detection(
        image_path=image_path,
        detection_result=result,
        output_path=str(det_output),
        score_threshold=conf_threshold,
    )

    # Heatmap visualization
    heatmap_output = output_dir / f"{image_name}_heatmap.jpg"
    visualize_heatmap(
        image_path=image_path,
        features=features,
        detection_result=result,
        output_path=str(heatmap_output),
    )

    print(f"\nResults saved to:")
    print(f"  Detection: {det_output}")
    print(f"  Heatmap: {heatmap_output}")

    return result


def main():
    parser = argparse.ArgumentParser("SAR Ship Detection Inference")

    parser.add_argument("--image", required=True, type=str, help="Path to input SAR image")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="./outputs/inference", type=str, help="Output directory")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="Confidence threshold")
    parser.add_argument("--img-size", default=896, type=int, help="Input image size")

    args = parser.parse_args()

    detect(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        conf_threshold=args.conf_threshold,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
