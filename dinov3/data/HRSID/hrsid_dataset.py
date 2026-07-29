# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

r"""
HRSID Dataset for SAR Ship Detection
====================================

Annotation source
-----------------
The official HRSID release (`official repo
<https://github.com/chaozhong2010/HRSID>`_) ships COCO-style detection +
instance-segmentation annotations (``*.json``) — it does **not** natively
provide oriented bounding boxes (OBB).  Each ship instance carries a
polygon ``segmentation``.  This loader therefore derives the four OBB
corners required by DRCP's direction-aware soft mask (Eq. 9-11) from the
**minimum-area bounding rectangle** of the instance polygon.  This is an
explicit, documented *derived* annotation, not a native one::

    polygon  ->  convex hull  ->  rotating-calipers min-area rect  ->  4 OBB corners

The conversion is implemented in pure NumPy (``_min_area_rect``) so the
loader has no OpenCV dependency.  The horizontal bounding box (``boxes``,
cxcywh pixel space) is taken from the COCO ``bbox`` field directly.

Scene-disjoint split
--------------------
HRSID's official release provides separate ``train`` / ``val`` / ``test``
JSON files, which are scene-disjoint by construction.  This loader reads
the COCO JSON whose name matches the requested split (e.g.
``HRSID_train.json`` / ``train2017.json``) and uses it verbatim — no
hash-based re-cutting.

A PASCAL-VOC XML fallback is retained for layouts that only ship HBB XML;
in that case OBB corners are derived from the axis-aligned HBB (an HBB is
an axis-aligned OBB).

Target dict (same interface as ``SSDDDataset``)::

    boxes:     FloatTensor[N, 4]   # cxcywh, pixel space
    labels:    LongTensor [N]      # 0 = ship
    obb:       FloatTensor[N, 8]   # 4 corner xy-pairs, pixel space,
                                  # DERIVED from polygon min-area rect
    image_id:  tensor([idx])
    orig_size: tensor([H, W])
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


# ---------------------------------------------------------------------------
# Minimum-area bounding rectangle of a polygon (rotating calipers, pure NumPy)
# ---------------------------------------------------------------------------
def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of an Nx2 point set.  Returns an Nx2 array
    of hull vertices in counter-clockwise order (no duplicate end point)."""
    pts = np.unique(points, axis=0)
    if len(pts) <= 1:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull)


def _min_area_rect(points: np.ndarray) -> np.ndarray:
    """Minimum-area bounding rectangle of an Nx2 point set.

    Implements the rotating-calipers search over hull edges in pure NumPy.
    Returns the 4 corner points as a 4x2 array, ordered counter-clockwise.
    """
    hull = _convex_hull(points)
    n = len(hull)
    if n < 3:
        # Degenerate (line / point): fall back to the axis-aligned bbox.
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    best_area = float("inf")
    best_corners = None
    for i in range(n):
        p1 = hull[i]
        p2 = hull[(i + 1) % n]
        edge = p2 - p1
        angle = float(np.arctan2(edge[1], edge[0]))
        c, s = np.cos(angle), np.sin(angle)
        # Rotate hull so the edge is horizontal (rotate by -angle).
        R_inv = np.array([[c, -s], [s, c]])  # rotates points by -angle
        rotated = (hull - p1) @ R_inv
        xmin, ymin = rotated.min(axis=0)
        xmax, ymax = rotated.max(axis=0)
        area = (xmax - xmin) * (ymax - ymin)
        if area < best_area:
            best_area = area
            # Corners of the AABB in the rotated frame.
            corners_rot = np.array([
                [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax],
            ])
            # Rotate back (by +angle) and translate to p1.
            R_back = np.array([[c, s], [-s, c]])  # rotates points by +angle
            best_corners = corners_rot @ R_back + p1
    if best_corners is None:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        best_corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return best_corners


def _poly_to_obb_corners(segmentation, orig_w, orig_h):
    """Convert a COCO polygon segmentation to 8 OBB corner coords (pixel).

    The polygon is flattened to (x, y) pixel points and the minimum-area
    bounding rectangle is computed via rotating calipers.  Returns a flat
    list ``[x1, y1, x2, y2, x3, y3, x4, y4]`` or ``None`` if the polygon
    is degenerate.
    """
    polys = segmentation if isinstance(segmentation[0], (list, np.ndarray)) else [segmentation]
    all_pts = []
    for poly in polys:
        xs = np.asarray(poly[0::2], dtype=np.float64)
        ys = np.asarray(poly[1::2], dtype=np.float64)
        if len(xs) < 3:
            continue
        for x, y in zip(xs, ys):
            all_pts.append([float(x), float(y)])
    if len(all_pts) < 3:
        return None
    pts = np.array(all_pts, dtype=np.float64)
    rect = _min_area_rect(pts)  # 4x2
    corners = []
    for x, y in rect:
        corners.append(float(x))
        corners.append(float(y))
    return corners


class HRSIDDataset(Dataset):
    """HRSID dataset for SAR ship detection.

    See module docstring for the annotation derivation and scene-disjoint
    split policy.
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
        # COCO JSON takes priority; VOC XML is the fallback.
        self.coco_json = self._find_coco_json(split)
        self._coco_index = None
        if self.coco_json is not None:
            self._coco_index = self._load_coco(self.coco_json)

        self.image_files = self._build_split()

    # ------------------------------------------------------------------
    # Directory layout auto-detection
    # ------------------------------------------------------------------
    def _find_dirs(self):
        candidates = [
            ("JPEGImages", "annotations"),
            ("images", "annotations"),
            ("images", "Annotations"),
            ("JPEGImages", "Annotations"),
        ]
        for img_name, anno_name in candidates:
            img_dir = self.root_dir / img_name
            anno_dir = self.root_dir / anno_name
            if img_dir.exists():
                return img_dir, anno_dir
        return self.root_dir / "images", self.root_dir / "annotations"

    def _find_coco_json(self, split: str):
        """Locate the COCO JSON for *split*.

        Tries common naming conventions used by the official HRSID release
        and COCO-style datasets generally::
            HRSID_<split>.json, <split>.json, <split>2017.json,
            instances_<split>2017.json
        """
        if self.anno_dir is None or not self.anno_dir.exists():
            return None
        names = [
            f"HRSID_{split}.json",
            f"{split}.json",
            f"{split}2017.json",
            f"instances_{split}2017.json",
            f"HRSID_{split}2017.json",
        ]
        for name in names:
            p = self.anno_dir / name
            if p.exists():
                return p
        # Any single JSON that mentions the split in its name.
        for p in sorted(self.anno_dir.glob("*.json")):
            if split in p.stem.lower():
                return p
        return None

    # ------------------------------------------------------------------
    # COCO JSON loading + scene-disjoint split
    # ------------------------------------------------------------------
    def _load_coco(self, json_path: Path):
        with open(json_path, "r") as f:
            data = json.load(f)
        images = {}
        for im in data.get("images", []):
            images[im["id"]] = im
        anns_by_img = {}
        for ann in data.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)
        return {"images": images, "anns_by_img": anns_by_img, "raw": data}

    def _build_split(self):
        # COCO JSON: use the official scene-disjoint split file verbatim.
        if self._coco_index is not None:
            result = []
            for im_id, im in self._coco_index["images"].items():
                img_path = self.image_dir / im["file_name"]
                if img_path.exists():
                    result.append(img_path)
            return result

        manifest_path = self.root_dir / "hrsid_splits.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if self.split in manifest and manifest[self.split]:
                return [Path(p) for p in manifest[self.split]]

        # VOC ImageSets split files.
        split_file = self.root_dir / "ImageSets" / "Main" / f"{self.split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                names = [ln.strip() for ln in f if ln.strip()]
            result = []
            for name in names:
                img = self._find_image_by_name(name)
                if img and img.exists():
                    result.append(img)
            if result:
                return result

        # Hash-based 80/10/10 fallback (only when no official split exists).
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

    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _find_annotation(self, img_path: Path) -> Path:
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

    def _find_image_by_name(self, name: str):
        for ext in _IMG_EXTS:
            p = self.image_dir / f"{name}{ext}"
            if p.exists():
                return p
        return None

    # ------------------------------------------------------------------
    # Annotation parsing
    # ------------------------------------------------------------------
    def _parse_coco_image(self, img_path: Path, orig_w: float, orig_h: float):
        """Parse annotations for one image from the COCO index.

        Returns (boxes_cxcywh, labels, obb_corners).  OBB corners are
        DERIVED from the polygon min-area rectangle (see module docstring).
        """
        boxes, labels, obb_corners = [], [], []
        im_id = None
        for iid, im in self._coco_index["images"].items():
            if Path(im["file_name"]).stem == img_path.stem:
                im_id = iid
                break
        if im_id is None:
            return boxes, labels, obb_corners
        for ann in self._coco_index["anns_by_img"].get(im_id, []):
            cat = ann.get("category_id", 1)
            bbox = ann.get("bbox")  # COCO: [x, y, w, h] top-left, pixel
            if bbox is None or len(bbox) != 4:
                continue
            x, y, w, h = (float(v) for v in bbox)
            if w <= 0 or h <= 0:
                continue
            cx = x + w / 2.0
            cy = y + h / 2.0
            boxes.append([cx, cy, w, h])
            labels.append(0)  # single ship class

            seg = ann.get("segmentation")
            corners = None
            if seg is not None:
                corners = _poly_to_obb_corners(seg, orig_w, orig_h)
            if corners is None:
                # Fallback: axis-aligned corners from the HBB.
                xmin, ymin = x, y
                xmax, ymax = x + w, y + h
                corners = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            obb_corners.append(corners)
        return boxes, labels, obb_corners

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

            # VOC XML has no polygon; derive axis-aligned OBB from the HBB.
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

        if self._coco_index is not None:
            boxes, labels, obb_corners = self._parse_coco_image(img_path, orig_w, orig_h)
        else:
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
