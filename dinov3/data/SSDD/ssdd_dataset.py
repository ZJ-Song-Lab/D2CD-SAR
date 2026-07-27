# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
SSDD Dataset for SAR Ship Detection
Dataset format: YOLO OBB (Oriented Bounding Box)

Implements hash-based deduplication and disjoint splitting to prevent
data leakage between train/val/test.  OBB corner coordinates are preserved
in target["obb"] (pixel space) for the DRCP direction-aware soft mask.
"""

import json
import hashlib
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class SSDDDataset(Dataset):
    """
    SSDD Dataset for oriented bounding box detection.
    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates)

    The existing on-disk train/val/test sub-directories leak images (val
    and test are literal copies of train files).  To guarantee disjoint
    splits we gather *all* labelled images across the three sub-folders,
    hash each file, deduplicate by hash, sort by hash, and cut into
    80 / 10 / 10.  The resulting manifest is cached as
    ``ssdd_splits.json`` next to the dataset root so that every run uses
    the same split.

    Images without a corresponding label file are silently dropped so
    that the loader never emits an empty-target sample (the original
    1160-vs-928 mismatch).
    """

    SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        max_size: int = 1024,
    ):
        """
        Args:
            root_dir: Root directory of SSDD dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied
            max_size: Maximum image size for resizing
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.max_size = max_size

        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"

        # Class names (only ship in SSDD)
        self.class_names = ["ship"]
        self.num_classes = len(self.class_names)

        self.image_files = self._build_disjoint_split()

    # ------------------------------------------------------------------
    # Hash-based deduplication + disjoint split
    # ------------------------------------------------------------------
    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _build_disjoint_split(self):
        """Return the list of image paths for *this* split.

        All labelled images across train/val/test are gathered, hashed,
        and deduplicated.  Sorting by hash then splitting 80/10/10 yields
        a deterministic, leak-free partition.  The result is cached in
        ``ssdd_splits.json``.
        """
        manifest_path = self.root_dir / "ssdd_splits.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            return [Path(p) for p in manifest[self.split]]

        all_images = []
        seen_hashes = {}
        for sub in ("train", "val", "test"):
            img_subdir = self.image_dir / sub
            if not img_subdir.exists():
                continue
            for img_path in sorted(img_subdir.glob("*.jpg")):
                label_path = self.label_dir / sub / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue
                h = self._md5(img_path)
                if h in seen_hashes:
                    continue
                seen_hashes[h] = img_path
                all_images.append((h, str(img_path)))

        all_images.sort(key=lambda x: x[0])
        n = len(all_images)
        n_train = int(n * self.SPLIT_RATIOS["train"])
        n_val = int(n * self.SPLIT_RATIOS["val"])
        splits = {
            "train": [p for _, p in all_images[:n_train]],
            "val": [p for _, p in all_images[n_train:n_train + n_val]],
            "test": [p for _, p in all_images[n_train + n_val:]],
        }
        with open(manifest_path, "w") as f:
            json.dump(splits, f, indent=2)
        return [Path(p) for p in splits[self.split]]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # The label file mirrors the image sub-directory:
        # images/<sub>/<stem>.jpg -> labels/<sub>/<stem>.txt
        sub = img_path.parent.name
        label_path = self.label_dir / sub / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        obb_corners = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 9:  # class_id + 8 coordinates
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        x_coords = [c * orig_w for c in coords[0::2]]
                        y_coords = [c * orig_h for c in coords[1::2]]

                        # Preserve OBB corners (pixel) for DRCP W_soft.
                        corners = []
                        for px, py in zip(x_coords, y_coords):
                            corners.extend([px, py])
                        obb_corners.append(corners)

                        # Axis-aligned bounding box (cxcywh, pixel).
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        cx = (x_min + x_max) / 2
                        cy = (y_min + y_max) / 2
                        w = x_max - x_min
                        h = y_max - y_min

                        boxes.append([cx, cy, w, h])
                        labels.append(class_id)

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            obb = torch.tensor(obb_corners, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            obb = torch.zeros((0, 8), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "obb": obb,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([orig_h, orig_w]),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def get_image_path(self, idx):
        """Get image path by index"""
        return str(self.image_files[idx])


def collate_fn(batch):
    """Custom collate function for detection"""
    images = []
    targets = []

    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    return images, targets
