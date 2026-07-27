# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
HRSID Dataset for SAR Ship Detection
Dataset format: PASCAL VOC XML (horizontal bounding boxes)

Like SSDDDataset, this loader uses hash-based deduplication and
disjoint splitting to prevent data leakage between train/val/test.
Horizontal bounding boxes from XML are returned in cxcywh pixel space;
four OBB corners are derived from the AABB for DRCP's direction-aware
soft mask (so the same DRCP / soft-mask code path works for both datasets).

Supported directory layouts (auto-detected):
  1. PASCAL VOC:  JPEGImages/  Annotations/  ImageSets/Main/{train,val,test}.txt
  2. Flat:        images/  annotations/   (optionally with train/ val/ test/ subdirs)

If no ImageSets split file is found, a hash-based 80/10/10 split is used
and cached as ``hrsid_splits.json`` next to the dataset root.
"""

import json
import hashlib
import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


_SHIP_NAMES = {"ship", "ship*", "boat", "vessel"}

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class HRSIDDataset(Dataset):
    """HRSID dataset for SAR ship detection.

    Returns the same target dict as ``SSDDDataset`` so that the same
    transforms, collate_fn, distiller, and evaluation code work unchanged::

        target = {
            "boxes":     FloatTensor[N, 4]   # cxcywh, pixel space
            "labels":    LongTensor [N]      # 0 = ship
            "obb":       FloatTensor[N, 8]   # 4 corner points, pixel space
            "image_id":  tensor([idx])
            "orig_size": tensor([H, W])
        }
    """

    SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        max_size: int = 1024,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.max_size = max_size

        self.class_names = ["ship"]
        self.num_classes = len(self.class_names)

        self.image_dir, self.anno_dir = self._find_dirs()
        self.image_files = self._build_disjoint_split()

    # ------------------------------------------------------------------
    # Directory layout auto-detection
    # ------------------------------------------------------------------
    def _find_dirs(self):
        candidates = [
            ("JPEGImages", "Annotations"),
            ("images", "annotations"),
            ("images", "Annotations"),
            ("JPEGImages", "annotations"),
        ]
        for img_name, anno_name in candidates:
            img_dir = self.root_dir / img_name
            anno_dir = self.root_dir / anno_name
            if img_dir.exists() and anno_dir.exists():
                return img_dir, anno_dir
        return self.root_dir / "images", self.root_dir / "annotations"

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

    def _find_annotation(self, img_path: Path) -> Path:
        """Return the XML annotation path for *img_path*, or ``None``."""
        stem = img_path.stem
        sub = img_path.parent.name
        for ext in (".xml", ".XML"):
            p = self.anno_dir / f"{stem}{ext}"
            if p.exists():
                return p
            p_sub = self.anno_dir / sub / f"{stem}{ext}"
            if p_sub.exists():
                return p_sub
        return None

    def _build_disjoint_split(self):
        manifest_path = self.root_dir / "hrsid_splits.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            return [Path(p) for p in manifest[self.split]]

        # 1. Try PASCAL VOC ImageSets split files.
        split_file = self.root_dir / "ImageSets" / "Main" / f"{self.split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                names = [ln.strip() for ln in f if ln.strip()]
            result = []
            for name in names:
                img = self._find_image_by_name(name)
                anno = self._find_annotation(img) if img else None
                if img and anno and anno.exists():
                    result.append(img)
            if result:
                splits = {s: [] for s in ("train", "val", "test")}
                splits[self.split] = [str(p) for p in result]
                with open(manifest_path, "w") as f:
                    json.dump(splits, f, indent=2)
                return result

        # 2. Hash-based dedup + 80/10/10 split.
        all_images = []
        seen_hashes = {}
        for img_path in sorted(self.image_dir.rglob("*")):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            anno_path = self._find_annotation(img_path)
            if anno_path is None or not anno_path.exists():
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

    def _find_image_by_name(self, name: str):
        for ext in _IMG_EXTS:
            p = self.image_dir / f"{name}{ext}"
            if p.exists():
                return p
        return None

    # ------------------------------------------------------------------
    # PASCAL VOC XML parsing
    # ------------------------------------------------------------------
    def _parse_xml(self, xml_path: Path, orig_w: float, orig_h: float):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        obb_corners = []

        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None:
                continue
            name = name_el.text.strip().lower()
            if name not in _SHIP_NAMES:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            def _get(tag):
                el = bndbox.find(tag)
                return float(el.text) if el is not None else 0.0

            xmin = _get("xmin")
            ymin = _get("ymin")
            xmax = _get("xmax")
            ymax = _get("ymax")

            xmin = max(0.0, min(xmin, orig_w))
            ymin = max(0.0, min(ymin, orig_h))
            xmax = max(0.0, min(xmax, orig_w))
            ymax = max(0.0, min(ymax, orig_h))

            if xmax <= xmin or ymax <= ymin:
                continue

            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            boxes.append([cx, cy, w, h])
            labels.append(0)

            # Derive 4 OBB corners from the AABB for DRCP W_soft.
            obb_corners.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])

        return boxes, labels, obb_corners

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        anno_path = self._find_annotation(img_path)
        if anno_path is not None and anno_path.exists():
            boxes, labels, obb_corners = self._parse_xml(anno_path, orig_w, orig_h)
        else:
            boxes, labels, obb_corners = [], [], []

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
        return str(self.image_files[idx])


def collate_fn(batch):
    """Custom collate function for detection (same format as SSDD)."""
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return images, targets
