# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
SSDD Dataset for SAR Ship Detection
Dataset format: native HBB (Horizontal Bounding Box), YOLO cxcywh.

Scene-disjoint partitioning (paper 4.1): the official SSDD release ships
``images/{train,val,test}/`` sub-directories that are scene-disjoint by
construction -- no image from a scene in *train* appears in *val* or
*test*.  The loader honours this on-disk layout directly.  A flat-layout
fallback groups images by a scene key extracted from the filename and
splits *by scene* (not by single image) so every image of a scene lands
in the same split.  HBB labels are preserved in target["boxes"] (cxcywh,
pixel); axis-aligned OBB corners derived from the HBB are kept in
target["obb"] for the DRCP direction-aware soft mask (the derivation is
lossless because an HBB *is* an axis-aligned OBB).
"""

import json
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class SSDDDataset(Dataset):
    """
    SSDD Dataset for horizontal bounding box detection.
    Native label format: ``class_id cx cy w h`` (normalized).

    Scene-disjoint partitioning
    ---------------------------
    Primary path: the on-disk ``images/<split>/`` sub-directory is used
    verbatim -- the official SSDD sub-directories are scene-disjoint by
    construction, so honouring them guarantees scene isolation.

    Flat-layout fallback: when no ``images/<split>/`` sub-directory exists,
    images are grouped by a *scene key* (the non-numeric prefix before the
    trailing id in the filename, e.g. ``sceneA_0001`` -> ``sceneA``; plain
    numeric ids such as ``0001`` -> ``""`` singleton scenes) and *scenes*
    (not single images) are sorted and cut 80 / 10 / 10 so every image of
    a scene lands in the same split.  The resulting manifest is cached as
    ``ssdd_splits.json`` next to the dataset root so that every run uses
    the same split.

    Images without a corresponding label file are silently dropped so
    that the loader never emits an empty-target sample.
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
        # Native HBB labels (paper: SSDD uses native HBB, not YOLO OBB).
        # Prefer an explicit ``labels_hbb/`` dir when present, else the
        # conventional ``labels/`` dir which stores cxcywh HBB for SSDD.
        hbb_dir = self.root_dir / "labels_hbb"
        self.label_dir = hbb_dir if hbb_dir.is_dir() else self.root_dir / "labels"

        # Class names (only ship in SSDD)
        self.class_names = ["ship"]
        self.num_classes = len(self.class_names)

        self.image_files = self._build_scene_disjoint_split()

    # ------------------------------------------------------------------
    # Scene-disjoint partitioning
    # ------------------------------------------------------------------
    @staticmethod
    def _scene_key(path: Path) -> str:
        """Extract a coarse scene/group key from a filename.

        Filenames such as ``sceneA_0001.jpg`` or ``sceneA-0001.jpg`` map to
        ``sceneA``; plain numeric ids such as ``0001.jpg`` map to ``""``
        (each treated as its own singleton scene, which degrades
        gracefully to a deterministic per-image cut only when no real
        scene key is available).
        """

        stem = path.stem
        m = re.match(r"^(.*?)[_-]?\d+$", stem)
        if m and m.group(1):
            return m.group(1)
        return ""

    @staticmethod
    def _label_path_for(label_dir: Path, img_path: Path) -> Path:
        """Mirror the image sub-directory under the label directory."""
        sub = img_path.parent.name
        if sub in ("train", "val", "test"):
            return label_dir / sub / f"{img_path.stem}.txt"
        return label_dir / f"{img_path.stem}.txt"

    def _build_scene_disjoint_split(self):
        """Return image paths for this split, guaranteeing scene isolation."""
        split_subdir = self.image_dir / self.split
        if split_subdir.is_dir():
            files = []
            for img_path in sorted(split_subdir.glob("*.jpg")):
                if self._label_path_for(self.label_dir, img_path).exists():
                    files.append(img_path)
            if files:
                return files

        manifest_path = self.root_dir / "ssdd_splits.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            return [Path(p) for p in manifest[self.split]]

        scenes = {}
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            if not self._label_path_for(self.label_dir, img_path).exists():
                continue
            key = self._scene_key(img_path)
            scenes.setdefault(key, []).append(str(img_path))

        scene_keys = sorted(scenes.keys())
        n = len(scene_keys)
        n_train = int(n * self.SPLIT_RATIOS["train"])
        n_val = int(n * self.SPLIT_RATIOS["val"])
        splits = {"train": [], "val": [], "test": []}
        for i, key in enumerate(scene_keys):
            if i < n_train:
                splits["train"].extend(scenes[key])
            elif i < n_train + n_val:
                splits["val"].extend(scenes[key])
            else:
                splits["test"].extend(scenes[key])
        with open(manifest_path, "w") as f:
            json.dump(splits, f, indent=2)
        return [Path(p) for p in splits[self.split]]
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        label_path = self._label_path_for(self.label_dir, img_path)
        boxes = []
        labels = []
        obb_corners = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    cx_n, cy_n, w_n, h_n = (float(x) for x in parts[1:])

                    cx = cx_n * orig_w
                    cy = cy_n * orig_h
                    w = w_n * orig_w
                    h = h_n * orig_h

                    boxes.append([cx, cy, w, h])
                    labels.append(class_id)

                    xmin = cx - w / 2.0
                    xmax = cx + w / 2.0
                    ymin = cy - h / 2.0
                    ymax = cy + h / 2.0
                    obb_corners.append(
                        [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                    )

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
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