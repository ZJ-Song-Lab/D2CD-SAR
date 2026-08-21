# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

"""D²CD-SAR cross-modal knowledge distiller.

Wires together the three components of the D²CD-SAR paper:
  * a frozen DINOv3-ViT-Base semantic teacher (provides dense patch-level
    features F_tea^sp),
  * a lightweight RT-DETR-R18 student whose AIFI attention output
    projection is replaced by A^2TD-LoRA (parameter-level decoupling),
  * the DRCP module (feature-level alignment, produces L_DRCP).

The forward pass implements the overall objective (Eq. 14):
    L_total = L_cls + L_box
            + lambda_distill * L_DRCP
            + lambda_ortho  * L_ortho
            + lambda_sparsity * L_sparsity

`update_gate` measures the directional consensus between L_DRCP and L_task on
the shared AIFI input activation z and refreshes the built-in variance gate
(Eq. 8-11). The gate value takes effect on the *next* forward pass, matching
the EMA-gating interpretation of Algorithm 1.

`reparameterize` merges both LoRA branches back into the frozen AIFI weights
(Eq. 15); afterwards the student is a plain RT-DETR-R18 with zero deployment
overhead, and the teacher / DRCP / gate can be discarded.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from dinov3.hub.backbones import dinov3_vitb16, Weights
from dinov3.sar_detection.atd_lora import (
    collect_atd_params,
    flattened_grad,
    total_orthogonal_loss,
    total_sparsity_loss,
)
from dinov3.sar_detection.drcp import DRCP
from dinov3.sar_detection.rtdetr import build_student, box_cxcywh_to_xyxy

# Must match dinov3/data/SSDD/transforms.Normalize so we can recover the raw
# SAR intensity magnitude |I_SAR| from the normalized input tensor.
SSDD_MEAN = (0.430, 0.411, 0.296)
SSDD_STD = (0.213, 0.156, 0.143)


class SARRTDETRDistiller(nn.Module):
    """Cross-modal distillation container (teacher + student + DRCP + gate)."""

    def __init__(
        self,
        num_classes: int = 1,
        num_queries: int = 300,
        r_lora: int = 100,
        lambda_distill: float = 1.0,
        lambda_ortho: float = 0.1,
        lambda_sparsity: float = 0.01,
        teacher_pretrained: bool = True,
        teacher_weights=Weights.SAT493M,
        backbone_pretrained: bool = True,
        freeze_backbone: bool = True,
        drcp_kwargs: dict = None,
        norm_mean=SSDD_MEAN,
        norm_std=SSDD_STD,
    ):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.lambda_ortho = lambda_ortho
        self.lambda_sparsity = lambda_sparsity
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # --- Student (RT-DETR-R18 + A^2TD-LoRA in AIFI) + detection criterion ---
        self.student, self.criterion, self.postprocessors = build_student(
            num_classes=num_classes,
            num_queries=num_queries,
            r_lora=r_lora,
            backbone_pretrained=backbone_pretrained,
            freeze_backbone=freeze_backbone,
        )
        # The shared variance gate lives on the student and is referenced by every
        # AIFI LoRA module; the distiller updates it from the outside.
        self.gate = self.student.gate

        # --- Frozen DINOv3-ViT-Base teacher ---
        self.teacher = self._build_teacher(teacher_pretrained, teacher_weights)
        self.embed_dim = self.teacher.embed_dim

        self.teacher_blocks = [2, 5, 8, 11]
        # Paper: "The teacher is frozen and has no trainable projector."
        # Teacher features are used directly at C_t=768.

        # --- DRCP (feature-level alignment) ---
        drcp_kwargs = dict(drcp_kwargs or {})
        self.drcp = DRCP(
            C_t=self.embed_dim,
            student_dims=(self.student.backbone.dims[0], self.student.backbone.dims[1], self.student.d_model),
            **drcp_kwargs,
        )

    # ------------------------------------------------------------------ teacher
    @staticmethod
    def _build_teacher(pretrained: bool, weights):
        w = weights if isinstance(weights, Weights) else Weights(weights)
        teacher = dinov3_vitb16(pretrained=pretrained, weights=w, check_hash=False)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def _teacher_features(self, images: torch.Tensor) -> list:
        """Three teacher block levels [T3, T4, T5_avg] for DRCP routing.

        Patch tokens are extracted from DINOv3-ViT-B transformer blocks
        {3,6,9,12} (1-indexed; ``self.teacher_blocks`` stores the 0-based
        indices). The teacher is frozen with no trainable projector (paper
        Sec. 3.1). T3=T^{(3)}, T4=T^{(6)}, T5_avg=(T^{(9)}+T^{(12)})/2.
        """
        with torch.no_grad():
            feats = self.teacher.get_intermediate_layers(
                images, n=self.teacher_blocks, reshape=True, norm=True
            )
        t3 = feats[0].float()
        t4 = feats[1].float()
        t5 = 0.5 * (feats[2].float() + feats[3].float())
        return [t3, t4, t5]

    # ----------------------------------------------------------- box utilities
    @staticmethod
    def _scale_factor(H: int, W: int, device, dtype) -> torch.Tensor:
        # cxcywh pixel -> [0, 1] normalization factor [W, H, W, H].
        return torch.tensor([W, H, W, H], dtype=dtype, device=device)

    def _normalize_targets(self, targets, H, W):
        """Normalize cxcywh pixel boxes to [0, 1] for the detection criterion."""
        out = []
        for t in targets:
            t = dict(t)
            if len(t["boxes"]) > 0:
                scale = self._scale_factor(H, W, t["boxes"].device, t["boxes"].dtype)
                t["boxes"] = t["boxes"] / scale
            out.append(t)
        return out

    def _normalize_hbb_for_drcp(self, targets, H, W):
        """Per-image list of normalized HBB tensors [N, 4] (xyxy, [0, 1]).

        Horizontal bounding boxes (cxcywh pixel from the dataset loaders) are
        converted to xyxy and scaled to [0, 1] for the DRCP spatial loss
        weight W_soft (Eq. soft_mask). ``None`` is returned for images with no
        annotation, in which case DRCP uses uniform feature matching.
        """
        out = []
        for t in targets:
            boxes = t.get("boxes")
            if boxes is not None and len(boxes) > 0:
                xyxy = box_cxcywh_to_xyxy(boxes)
                scale = torch.tensor(
                    [W, H, W, H],
                    device=xyxy.device, dtype=xyxy.dtype,
                )
                out.append((xyxy / scale).detach())
            else:
                out.append(None)
        return out

    def _sar_intensity(self, images: torch.Tensor) -> torch.Tensor:
        """Recover the raw SAR intensity magnitude |I_SAR| (grayscale) [B,1,H,W].

        The dataloader returns a mean/std-normalized RGB tensor; we invert the
        normalization then average channels, because SSDD SAR scenes are
        single-channel intensity replicated to 3 channels.
        """
        device, dtype = images.device, images.dtype
        mean = torch.tensor(self.norm_mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.norm_std, device=device, dtype=dtype).view(1, 3, 1, 1)
        raw = images * std + mean
        return raw.mean(dim=1, keepdim=True).clamp(min=0.0)

    # ------------------------------------------------------------------ forward
    def forward(self, images: torch.Tensor, targets: list):
        """Args:
            images:  [B, 3, H, W] batched and normalized SAR images.
            targets: list of dicts {boxes (cxcywh, pixel), labels}.
        Returns: dict of losses + the purified teacher feature.

        The forward is split into two passes through the AIFI so that the
        detection LoRA branch only receives gradients from L_task and the
        distillation LoRA branch only receives gradients from L_DRCP
        (true task-gradient isolation, Eq. 6).  The frozen backbone features
        are computed once and reused for both passes.
        """
        B, _, H, W = images.shape

        # 1. Frozen teacher patch features.
        teacher_features = self._teacher_features(images)  # [T3, T4, T5_avg]

        # 2. Backbone features (frozen — computed once, reused for both passes).
        with torch.no_grad():
            s3, s4, s5 = self.student.backbone(images)

        # 3. Task forward: detection branch active, distillation branch
        #    detached so L_task gradients do not contaminate the distill LoRA.
        outputs_task = self.student(
            images, detach_distill=True, backbone_features=(s3, s4, s5)
        )
        targets_norm = self._normalize_targets(targets, H, W)
        det_losses = self.criterion(outputs_task, targets_norm)
        loss_cls = det_losses["loss_ce"]
        loss_bbox = det_losses["loss_bbox"]
        loss_giou = det_losses["loss_giou"]
        loss_box = loss_bbox + loss_giou
        loss_task = loss_cls + loss_box

        # 4. Distill forward: distillation branch active, detection branch
        #    detached so L_DRCP gradients do not contaminate the detection LoRA.
        outputs_distill = self.student(
            images, detach_det=True, backbone_features=(s3, s4, s5)
        )
        sar_gray = self._sar_intensity(images)
        hbb_norm = self._normalize_hbb_for_drcp(targets, H, W)
        f_tea_hat, loss_drcp = self.drcp(
            [outputs_distill["s3"], outputs_distill["s4"], outputs_distill["f5"]],
            teacher_features,
            hbb_norm,
            sar_gray,
        )

        # 5. A^2TD-LoRA regularization (Eq. 7, Eq. 13).
        atd = self.student.atd_modules()
        loss_ortho = total_orthogonal_loss(atd)
        loss_sparsity = total_sparsity_loss(atd)

        # 6. Overall objective (Eq. 14).
        loss_total = (
            loss_task
            + self.lambda_distill * loss_drcp
            + self.lambda_ortho * loss_ortho
            + self.lambda_sparsity * loss_sparsity
        )

        return {
            "loss_total": loss_total,
            "loss_drcp": loss_drcp,        # kept differentiable for the gate update
            "loss_task": loss_task,        # kept differentiable for the gate update
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "loss_ortho": loss_ortho.detach(),
            "loss_sparsity": loss_sparsity.detach(),
            "gate_value": self.gate.value(),
            "f_tea_hat": f_tea_hat.detach(),
            "z_distill": outputs_distill["z"],
            "z_task": outputs_task["z"],
        }

    # --------------------------------------------------- variance gate update
    def update_gate(self, loss_drcp: torch.Tensor, loss_task: torch.Tensor,
                    z_distill: torch.Tensor, z_task: torch.Tensor) -> float:
        """Refresh the direction-aware variance gate on the shared AIFI input z.

        Implements Eq. (8)-(11): activation-cosine direction agreement ``c_dir``,
        direction-aware ratio ``r``, adaptive momentum ``alpha`` and EMA ``w``.
        Probes are gradients of L_task and L_DRCP w.r.t. the common AIFI input
        activation z (Eq. activation_probes), computed with ``retain_graph=True``
        so the graph is still available for ``loss_total.backward()``.
        """
        p_distill = torch.autograd.grad(
            loss_drcp, z_distill, retain_graph=True, allow_unused=True)[0]
        p_task = torch.autograd.grad(
            loss_task, z_task, retain_graph=True, allow_unused=True)[0]
        if p_distill is None:
            p_distill = torch.zeros_like(z_distill)
        if p_task is None:
            p_task = torch.zeros_like(z_task)
        p_distill = p_distill.detach().flatten()
        p_task = p_task.detach().flatten()
        # Keep the gating signal consistent across data-parallel ranks.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(p_distill, op=dist.ReduceOp.SUM)
                dist.all_reduce(p_task, op=dist.ReduceOp.SUM)
                p_distill = p_distill / world_size
                p_task = p_task / world_size
        return self.gate.update(p_distill, p_task)

    # --------------------------------------------------------- deployment
    @torch.no_grad()
    def reparameterize(self):
        """Merge both LoRA branches into the frozen AIFI weights (Eq. 15).

        After this call the student's AIFI holds plain ``nn.Linear`` layers (zero
        extra inference cost / memory). The teacher, DRCP and gate can then be
        discarded, leaving a pristine RT-DETR-R18.
        """
        for layer in self.student.aifi.layers:
            layer.linear1 = layer.linear1.merge()
            layer.linear2 = layer.linear2.merge()
        return self.student


def build_distiller(
    num_classes: int = 1,
    num_queries: int = 300,
    r_lora: int = 16,
    lambda_distill: float = 1.0,
    lambda_ortho: float = 0.1,
    lambda_sparsity: float = 0.01,
    teacher_pretrained: bool = True,
    teacher_weights=Weights.SAT493M,
    backbone_pretrained: bool = True,
    freeze_backbone: bool = True,
    drcp_kwargs: dict = None,
    norm_mean=SSDD_MEAN,
    norm_std=SSDD_STD,
):
    """Factory with the paper's default hyperparameters (Implementation Details)."""
    distiller = SARRTDETRDistiller(
        num_classes=num_classes,
        num_queries=num_queries,
        r_lora=r_lora,
        lambda_distill=lambda_distill,
        lambda_ortho=lambda_ortho,
        lambda_sparsity=lambda_sparsity,
        teacher_pretrained=teacher_pretrained,
        teacher_weights=teacher_weights,
        backbone_pretrained=backbone_pretrained,
        freeze_backbone=freeze_backbone,
        drcp_kwargs=drcp_kwargs,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    return distiller
