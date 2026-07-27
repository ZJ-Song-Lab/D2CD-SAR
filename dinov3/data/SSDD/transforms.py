# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Transforms for SSDD Dataset
"""

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


class ColorJitter:
    """Color jittering for SAR images (adapted for grayscale-like images)"""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize for DINOv3 satellite model"""

    def __init__(self, mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def make_train_transforms(img_size=896, mean=SSDD_MEAN, std=SSDD_STD):
    """Training transforms for SAR detection"""
    return Compose([
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        ColorJitter(brightness=0.2, contrast=0.2),
        Resize(img_size),
        ToTensor(),
        Normalize(mean, std),
    ])


def make_val_transforms(img_size=896, mean=SSDD_MEAN, std=SSDD_STD):
    """Validation transforms for SAR detection"""
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(mean, std),
    ])
