# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Transforms for SSDD Dataset
"""

import math

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image


SSDD_MEAN = (0.430, 0.411, 0.296)
SSDD_STD = (0.213, 0.156, 0.143)

HRSID_MEAN = (0.379, 0.379, 0.379)
HRSID_STD = (0.191, 0.191, 0.191)

DATASET_STATS = {
    "ssdd": (SSDD_MEAN, SSDD_STD),
    "hrsid": (HRSID_MEAN, HRSID_STD),
}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    """Resize image and bounding boxes"""

    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        image = F.resize(image, (self.size, self.size))

        sx, sy = self.size / orig_w, self.size / orig_h

        # Scale boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= sx  # cx, w
            boxes[:, [1, 3]] *= sy  # cy, h
            target["boxes"] = boxes

        # Scale OBB corners
        if "obb" in target and len(target["obb"]) > 0:
            obb = target["obb"].clone()
            obb[:, 0::2] *= sx  # x corners
            obb[:, 1::2] *= sy  # y corners
            target["obb"] = obb

        target["size"] = torch.tensor([self.size, self.size])
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            w = target.get("size", torch.tensor(image.size[::-1]))[1]
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, 0] = w - boxes[:, 0]  # flip cx
                target["boxes"] = boxes
            if "obb" in target and len(target["obb"]) > 0:
                obb = target["obb"].clone()
                obb[:, 0::2] = w - obb[:, 0::2]  # flip x corners
                target["obb"] = obb
        return image, target


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.vflip(image)
            h = target.get("size", torch.tensor(image.size[::-1]))[0]
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, 1] = h - boxes[:, 1]  # flip cy
                target["boxes"] = boxes
            if "obb" in target and len(target["obb"]) > 0:
                obb = target["obb"].clone()
                obb[:, 1::2] = h - obb[:, 1::2]  # flip y corners
                target["obb"] = obb
        return image, target


class RandomAffine:
    """Random affine: rotation ±10°, scale ±5%, no shear/translate by default.

    Matches the paper Table 3 frozen augmentation config: affine ±10°, ±5%.
    Bounding boxes (cxcywh pixel) and OBB polygon corners are transformed in
    lockstep with the image.
    """

    def __init__(self, degrees=10.0, scale=(0.95, 1.05),
                 translate=None, shear=None, fill=0):
        self.degrees = degrees
        self.scale = scale
        self.translate = translate
        self.shear = shear
        self.fill = fill

    def __call__(self, image, target):
        w, h = image.size
        angle = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(self.scale[0], self.scale[1])
        t = [0.0, 0.0] if self.translate is None else [
            random.uniform(-self.translate[0], self.translate[0]),
            random.uniform(-self.translate[1], self.translate[1]),
        ]
        shear = [0.0, 0.0] if self.shear is None else [
            random.uniform(-self.shear[0], self.shear[0]),
            random.uniform(-self.shear[1], self.shear[1]),
        ]
        # Forward-transform the image.
        image = F.affine(image, angle=angle, translate=t, scale=s, shear=shear,
                         fill=self.fill, interpolation=T.InterpolationMode.BILINEAR)
        # Inverse matrix to apply the same transform to boxes (img->affined).
        theta = math.radians(angle)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        # 2x3 affine matrix for point transformation: [x', y'] = M @ [x, y, 1]
        # PyTorch affine uses center as origin.
        cx, cy = w / 2.0, h / 2.0
        sx_t, sy_t = t[0] * w, t[1] * h

        def _tf_pt(x, y):
            # Center, rotate+scale, un-center, translate.
            xp = x - cx
            yp = y - cy
            xr = cos_a * xp - sin_a * yp
            yr = sin_a * xp + cos_a * yp
            return s * xr + cx + sx_t, s * yr + cy + sy_t

        if "boxes" in target and len(target["boxes"]) > 0:
            # cxcywh pixel -> four corners -> transform -> rebuild cxcywh
            boxes = target["boxes"].clone()
            cx_b, cy_b, bw, bh = boxes.unbind(-1)
            x1, x2 = cx_b - 0.5 * bw, cx_b + 0.5 * bw
            y1, y2 = cy_b - 0.5 * bh, cy_b + 0.5 * bh
            # Four corners in image pixel space.
            pts = [
                _tf_pt(x1, y1), _tf_pt(x2, y1),
                _tf_pt(x2, y2), _tf_pt(x1, y2),
            ]
            xs = torch.stack([torch.as_tensor(p[0]).float() for p in pts], dim=0)
            ys = torch.stack([torch.as_tensor(p[1]).float() for p in pts], dim=0)
            nx1, nx2 = xs.min(dim=0).values, xs.max(dim=0).values
            ny1, ny2 = ys.min(dim=0).values, ys.max(dim=0).values
            ncx, ncy = 0.5 * (nx1 + nx2), 0.5 * (ny1 + ny2)
            nw, nh = nx2 - nx1, ny2 - ny1
            target["boxes"] = torch.stack([ncx, ncy, nw, nh], dim=-1)

        if "obb" in target and len(target["obb"]) > 0:
            obb = target["obb"].clone()
            N = obb.shape[0]
            xy = obb.reshape(N, -1, 2)
            new_xy = torch.zeros_like(xy)
            for i in range(N):
                for j in range(xy.shape[1]):
                    nx, ny = _tf_pt(float(xy[i, j, 0].item()), float(xy[i, j, 1].item()))
                    new_xy[i, j, 0] = nx
                    new_xy[i, j, 1] = ny
            target["obb"] = new_xy.reshape(N, -1)

        return image, target


class Mosaic:
    """Mosaic 4-image concatenation (probability 1.0 in paper Table 3).

    Combines four random samples from ``dataset`` into a 2x2 grid of size
    ``img_size``, with each quadrant randomly cropped/scaled. Box coordinates
    are updated accordingly. Requires a reference ``dataset`` that supports
    ``__getitem__`` to fetch companion samples. The transform instance should
    be bound to the dataset after construction (see ``bind_dataset``).
    """

    def __init__(self, img_size=640, p=1.0, dataset=None):
        self.img_size = img_size
        self.p = p
        self._dataset = dataset

    def bind_dataset(self, dataset):
        self._dataset = dataset

    def __call__(self, image, target):
        if random.random() >= self.p or self._dataset is None:
            return image, target
        ds = self._dataset
        n = len(ds)
        indices = [random.randrange(n) for _ in range(3)]
        s = self.img_size
        # Big canvas is 2s x 2s; after the 4 images are placed it will be
        # centrally cropped to s x s, matching the downstream Resize.
        canvas = Image.new(image.mode, (2 * s, 2 * s), 0)
        # Collect boxes for the four quadrants.
        all_boxes = []
        all_labels = []
        all_obbs = []
        xc, yc = [random.randint(s // 2, 3 * s // 2) for _ in range(2)]
        sources = [(image, target)] + [ds[i] for i in indices]
        positions = [
            (0, 0, xc, yc),             # top-left
            (xc, 0, 2 * s, yc),         # top-right
            (0, yc, xc, 2 * s),         # bottom-left
            (xc, yc, 2 * s, 2 * s),     # bottom-right
        ]
        for idx, (img, tgt) in enumerate(sources):
            x1, y1, x2, y2 = positions[idx]
            dst_w, dst_h = x2 - x1, y2 - y1
            try:
                resized = img.resize((dst_w, dst_h), Image.BILINEAR)
            except Exception:
                resized = F.resize(img, (dst_h, dst_w), interpolation=T.InterpolationMode.BILINEAR)
            canvas.paste(resized, (x1, y1))
            scale_x = dst_w / max(img.size[0], 1)
            scale_y = dst_h / max(img.size[1], 1)
            if "boxes" in tgt and len(tgt["boxes"]) > 0:
                b = tgt["boxes"].clone()
                b[:, 0] = (b[:, 0] * scale_x) + x1
                b[:, 1] = (b[:, 1] * scale_y) + y1
                b[:, 2] = b[:, 2] * scale_x
                b[:, 3] = b[:, 3] * scale_y
                all_boxes.append(b)
                if "labels" in tgt:
                    all_labels.append(tgt["labels"].clone())
            if "obb" in tgt and len(tgt["obb"]) > 0:
                o = tgt["obb"].clone()
                o[:, 0::2] = o[:, 0::2] * scale_x + x1
                o[:, 1::2] = o[:, 1::2] * scale_y + y1
                all_obbs.append(o)
        # Center-crop to s x s and re-offset boxes.
        canvas = canvas.crop((s // 2, s // 2, s // 2 + s, s // 2 + s))
        merged_target = dict(target) if isinstance(target, dict) else {}
        if all_boxes:
            bs = torch.cat(all_boxes, dim=0)
            bs[:, 0] -= s // 2
            bs[:, 1] -= s // 2
            # Clip boxes that are partially/fully outside.
            bw_ok = (bs[:, 2] > 1) & (bs[:, 3] > 1)
            bs = bs[bw_ok]
            merged_target["boxes"] = bs
            if all_labels:
                ls = torch.cat(all_labels, dim=0)[bw_ok]
                merged_target["labels"] = ls
        if all_obbs:
            os_ = torch.cat(all_obbs, dim=0)
            os_[:, 0::2] -= s // 2
            os_[:, 1::2] -= s // 2
            merged_target["obb"] = os_
        return canvas, merged_target


class MixUp:
    """MixUp augmentation (paper Table 3: p=0.15).

    Blends two samples with a random Beta(1.5, 1.5) coefficient alpha ∈ [0, 1].
    Image = alpha * img_a + (1-alpha) * img_b. Boxes/labels are concatenated.
    Requires a bound dataset to fetch the partner sample.
    """

    def __init__(self, img_size=640, p=0.15, alpha=1.5, dataset=None):
        self.img_size = img_size
        self.p = p
        self.alpha = alpha
        self._dataset = dataset

    def bind_dataset(self, dataset):
        self._dataset = dataset

    def __call__(self, image, target):
        if random.random() >= self.p or self._dataset is None:
            return image, target
        ds = self._dataset
        partner_idx = random.randrange(len(ds))
        img_b, tgt_b = ds[partner_idx]
        try:
            img_a = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_b = img_b.resize((self.img_size, self.img_size), Image.BILINEAR)
        except Exception:
            img_a = F.resize(image, (self.img_size, self.img_size))
            img_b = F.resize(img_b, (self.img_size, self.img_size))
        lam = random.betavariate(self.alpha, self.alpha)
        arr_a = np.asarray(img_a, dtype=np.float32)
        arr_b = np.asarray(img_b, dtype=np.float32)
        mixed = np.clip(lam * arr_a + (1.0 - lam) * arr_b, 0, 255).astype(arr_a.dtype)
        img_mix = Image.fromarray(mixed.astype(np.uint8))
        merged = dict(target) if isinstance(target, dict) else {}
        boxes_a = target.get("boxes")
        boxes_b = tgt_b.get("boxes")
        if boxes_a is not None or boxes_b is not None:
            parts = [b for b in [boxes_a, boxes_b] if b is not None and len(b) > 0]
            merged["boxes"] = torch.cat(parts, dim=0) if parts else torch.zeros(0, 4)
        labels_a = target.get("labels")
        labels_b = tgt_b.get("labels")
        if labels_a is not None or labels_b is not None:
            parts = [l for l in [labels_a, labels_b] if l is not None and len(l) > 0]
            merged["labels"] = torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.long)
        obb_a = target.get("obb")
        obb_b = tgt_b.get("obb")
        if obb_a is not None or obb_b is not None:
            parts = [o for o in [obb_a, obb_b] if o is not None and len(o) > 0]
            merged["obb"] = torch.cat(parts, dim=0) if parts else torch.zeros(0, 8)
        return img_mix, merged


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize for D²CD-SAR SAR ship detection"""

    def __init__(self, mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def make_train_transforms(img_size=640, mean=SSDD_MEAN, std=SSDD_STD):
    """Training transforms matching D^2CD-SAR Table 3 frozen augmentation config.

    Config (see paper Sec. 4.3 / Table 3):
      * Mosaic probability 1.0
      * MixUp probability 0.15
      * Random affine: ±10° rotation, ±5% scale
      * HSV / ColorJitter: disabled (SAR input is a single intensity channel
        replicated to three channels; color jitter is not meaningful).
      * Horizontal flip: p = 0.5

    Usage note: Because Mosaic and MixUp require a dataset reference to fetch
    companion samples, the caller should call ``bind_dataset(train_set)`` on the
    returned Compose after construction, or pass the dataset through the
    dataset's constructor.
    """
    return Compose([
        Mosaic(img_size=img_size, p=1.0),
        MixUp(img_size=img_size, p=0.15),
        RandomHorizontalFlip(0.5),
        RandomAffine(degrees=10.0, scale=(0.95, 1.05)),
        Resize(img_size),
        ToTensor(),
        Normalize(mean, std),
    ])


def make_val_transforms(img_size=640, mean=SSDD_MEAN, std=SSDD_STD):
    """Validation transforms for SAR detection"""
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(mean, std),
    ])
