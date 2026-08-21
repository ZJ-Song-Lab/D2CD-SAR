# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

"""Smoke tests for the SAR-RTDETR components.

Run with::

    python -m pytest dinov3/sar_detection/test_smoke.py -v
    # or
    python -m dinov3.sar_detection.test_smoke

These tests are intentionally lightweight — no GPU, no pretrained weights,
no real dataset required — so they can run in CI in under a minute.
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dinov3.sar_detection.rtdetr import RTDETRStudent, PostProcess, box_cxcywh_to_xyxy
from dinov3.sar_detection.drcp import DRCP
from dinov3.sar_detection.atd_lora import VarianceGate
from dinov3.data.HRSID.hrsid_dataset import _min_area_rect, _convex_hull


def test_student_forward():
    """RTDETRStudent produces correctly shaped outputs including {s3,s4,s5,f5,z}."""
    student = RTDETRStudent(
        num_classes=1, num_queries=10, r_lora=4,
        backbone_pretrained=False, freeze_backbone=False,
    )
    student.eval()
    img = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = student(img)
    assert "pred_logits" in out
    assert "pred_boxes" in out
    assert out["pred_logits"].shape[0] == 2
    assert out["pred_logits"].shape[1] == 10   # num_queries
    assert out["pred_logits"].shape[2] == 2    # num_classes + 1 (foreground + bg)
    assert out["pred_boxes"].shape[-1] == 4
    # Feature routing keys: {S3, S4, S5} from backbone, F5 from AIFI.
    assert "s3" in out and "s4" in out and "s5" in out
    assert "f5" in out  # AIFI-refined, used by DRCP routing interface R.
    assert "z" in out   # AIFI input activation for gradient probes.
    print("[OK] test_student_forward: output keys + shapes correct")


def test_drcp_forward():
    """DRCP routes {S3,S4,F5} and returns purified teacher feature + loss."""
    C_t = 64
    drcp = DRCP(C_t=C_t, student_dims=(16, 32, 64), K_window=5)
    B, Ht, Wt = 2, 8, 8
    s3 = torch.randn(B, 16, 32, 32)
    s4 = torch.randn(B, 32, 16, 16)
    f5 = torch.randn(B, 64, 8, 8)  # AIFI-refined S5
    # Teacher features: 3 levels [T3, T4, T5_avg]
    teacher_features = [torch.randn(B, C_t, Ht, Wt) for _ in range(3)]
    sar_gray = torch.rand(B, 1, 64, 64)
    hbb_norm = [
        torch.tensor([[0.2, 0.2, 0.8, 0.8]]),  # 1 HBB in [0,1] xyxy
        None,  # empty GT image
    ]
    f_tea_hat, loss = drcp([s3, s4, f5], teacher_features, hbb_norm, sar_gray)
    assert f_tea_hat.shape == teacher_features[0].shape
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    print(f"[OK] test_drcp_forward: loss={loss.item():.4f}, shape={f_tea_hat.shape}")


def test_postprocess():
    """PostProcess converts model outputs to COCO-style detection results."""
    post = PostProcess()
    outputs = {
        "pred_logits": torch.randn(1, 10, 2),
        "pred_boxes": torch.rand(1, 10, 4),
    }
    target_sizes = torch.tensor([[100, 100]])
    results = post(outputs, target_sizes)
    assert len(results) == 1
    r = results[0]
    assert "scores" in r and "labels" in r and "boxes" in r
    assert r["boxes"].shape[-1] == 4
    assert torch.isfinite(r["boxes"]).all()
    print("[OK] test_postprocess: output format correct")


def test_convex_hull():
    """Convex hull returns CCW vertices for a square point set."""
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=np.float64)
    hull = _convex_hull(pts)
    assert len(hull) == 4
    for p in hull:
        assert p[0] in (0.0, 1.0) and p[1] in (0.0, 1.0)
    print("[OK] test_convex_hull: 4 hull vertices for unit square")


def test_min_area_rect():
    """Rotating calipers returns the correct min-area rectangle for a rotated box."""
    pts = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
    rect = _min_area_rect(pts)
    assert rect.shape == (4, 2)
    dists = np.linalg.norm(rect, axis=1)
    assert np.allclose(dists, 1.0, atol=1e-6)
    print("[OK] test_min_area_rect: correct for diamond (rotated square)")


def test_min_area_rect_axis_aligned():
    """Min-area rect of an axis-aligned box is the box itself."""
    pts = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=np.float64)
    rect = _min_area_rect(pts)
    xs, ys = rect[:, 0], rect[:, 1]
    assert np.isclose(xs.min(), 0) and np.isclose(xs.max(), 2)
    assert np.isclose(ys.min(), 0) and np.isclose(ys.max(), 1)
    print("[OK] test_min_area_rect_axis_aligned: correct for axis-aligned box")


def test_box_cxcywh_to_xyxy():
    """cxcywh -> xyxy conversion."""
    boxes = torch.tensor([[5.0, 5.0, 4.0, 2.0]])
    xyxy = box_cxcywh_to_xyxy(boxes)
    expected = torch.tensor([[3.0, 4.0, 7.0, 6.0]])
    assert torch.allclose(xyxy, expected)
    print("[OK] test_box_cxcywh_to_xyxy: conversion correct")


def test_variance_gate():
    """VarianceGate initialises to 1.0 and updates on gradient signals."""
    gate = VarianceGate()
    assert abs(gate.value() - 1.0) < 1e-6
    p_distill = torch.randn(10)
    p_task = torch.randn(10)
    w = gate.update(p_distill, p_task)
    assert 0.0 <= w <= 1.0
    print(f"[OK] test_variance_gate: w={w:.4f}")


def test_student_two_pass_isolation():
    """Task / distill forward passes share the backbone features (Eq. 6)."""
    student = RTDETRStudent(
        num_classes=1, num_queries=10, r_lora=4,
        backbone_pretrained=False, freeze_backbone=False,
    )
    student.eval()
    img = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        s3, s4, s5 = student.backbone(img)
        out_task = student(img, detach_distill=True, backbone_features=(s3, s4, s5))
        out_dist = student(img, detach_det=True, backbone_features=(s3, s4, s5))
    for key in ("s3", "s4", "s5", "f5", "z"):
        assert key in out_task and key in out_dist
    print("[OK] test_student_two_pass_isolation: both passes return features")


def run_all():
    test_student_forward()
    test_drcp_forward()
    test_postprocess()
    test_convex_hull()
    test_min_area_rect()
    test_min_area_rect_axis_aligned()
    test_box_cxcywh_to_xyxy()
    test_variance_gate()
    test_student_two_pass_isolation()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    run_all()
