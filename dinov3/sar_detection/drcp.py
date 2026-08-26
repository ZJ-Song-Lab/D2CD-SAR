# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

r"""DRCP: Depth-Routed Cross-Modal Purifier.

Implements the feature-level alignment module of the D²CD-SAR paper.

Phase 1 - Depth-wise Semantic Routing (Eq. 3-5):
  Student levels U = {S3, S4, F5} interpolated to the teacher spatial
  resolution and projected to C_t; routing weights alpha_i =
  softmax_i(w_d^T RMSNorm(GAP(U_i))); F_stu_routed = sum_i alpha_i U_i.
  The same alpha_i (with stop-gradient) route the teacher blocks:
  F_tea_sp = sum_i sg(alpha_i) T_i, where T_3=T^{(3)}, T_4=T^{(6)},
  T_5 = (T^{(9)}+T^{(12)})/2.

Phase 2 - Channel and Spatial Weighting (Eq. 6-12):
  Channel gate: GAP -> 1D rFFT -> magnitude[:k_cut] -> MLP -> sigmoid -> g
  Local intensity: E(x,y) = mean_{K_I x K_I} |I|^2  (empirical heuristic)
  W_soft: token-center-in-box rule on horizontal bounding boxes, with
  normalized energy inside and Gaussian decay outside.
  Purified teacher: F_tea_hat = g * F_tea_sp  (W_soft is loss weight only)

L_DRCP (Eq. 13): spatially weighted cosine distance between the purified
teacher and the routed student feature, with W_soft as the explicit spatial
loss weight (not multiplied into the teacher feature before cosine).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.layers.rms_norm import RMSNorm


class DRCP(nn.Module):
    """Depth-Routed Cross-Domain Purifier.

    Args:
        C_t: teacher channel dimension (DINOv3-ViT-B -> 768).
        student_dims: channel dims of the routed student levels
            (e.g. (C3, C4, C5=F5) = (128, 256, 256) for RT-DETR-R18). The
            routed levels are {S3, S4, F5}, where F5 is the AIFI-refined
            deepest representation; routing F5 (not raw S5) is what makes the
            L_DRCP feature-consistency gradient flow through the AIFI's
            distillation LoRA branch (paper Sec. 3.2, routed interface R).
        K_window: local-intensity averaging window K_I (pixels).
        mu: background soft-mask magnitude outside boxes.
        sigma: background soft-mask Gaussian bandwidth (token cells).
        k_cut: number of low-index rFFT magnitude coefficients retained.
    """

    def __init__(
        self,
        C_t: int = 768,
        student_dims=(256, 256, 256),
        K_window: int = 5,
        mu: float = 0.5,
        sigma: float = 4.0,
        k_cut: int = 192,
    ):
        super().__init__()
        self.C_t = C_t
        self.K_window = K_window
        self.mu = mu
        self.sigma = sigma
        self.k_cut = min(k_cut, C_t // 2 + 1)  # clamp to available rFFT bins

        # Student-level projections to the teacher channel dim.
        self.proj = nn.ModuleList(
            [nn.Conv2d(c, C_t, kernel_size=1, bias=True) for c in student_dims]
        )
        # Learnable pseudo-query for depth-wise routing.
        self.w_d = nn.Parameter(torch.randn(C_t) * 0.02)
        self.key_norm = RMSNorm(C_t)

        # Frequency-guided channel gate: k_cut -> 512 -> C_t.
        hidden = 512
        self.channel_gate = nn.Sequential(
            nn.Linear(self.k_cut, hidden),
            nn.GELU(),
            nn.Linear(hidden, C_t),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Phase 1: depth-wise semantic routing
    # ------------------------------------------------------------------
    def _route(self, student_features: list, Ht: int, Wt: int):
        """Route student features and return (routed, alphas)."""
        routed_levels = []
        keys = []
        for feat, proj in zip(student_features, self.proj):
            s = F.interpolate(feat, size=(Ht, Wt), mode="bilinear", align_corners=False)
            s = proj(s)  # [B, C_t, Ht, Wt]
            routed_levels.append(s)
            key = F.adaptive_avg_pool2d(s, 1).flatten(2).squeeze(2)  # [B, C_t]
            keys.append(self.key_norm(key))  # [B, C_t]
        # alpha_i = softmax_i(w_d^T k_i) over the depth index i.
        logits = torch.stack([torch.einsum("b c, c -> b", k, self.w_d) for k in keys], dim=1)  # [B, L]
        alphas = torch.softmax(logits, dim=1)  # [B, L]
        routed = sum(a[..., None, None, None] * s for a, s in zip(alphas.unbind(1), routed_levels))
        return routed, alphas

    @staticmethod
    def _route_teacher(teacher_features: list, alphas: torch.Tensor) -> torch.Tensor:
        """Route teacher blocks with stop-gradient on routing weights.

        F_tea_sp = sum_i sg(alpha_i) * T_i, where sg stops gradient through
        the routing weights so the routing query learns only from the
        student-side alignment gradient.
        """
        sg_alphas = alphas.detach()
        routed = sum(a[..., None, None, None] * t for a, t in zip(sg_alphas.unbind(1), teacher_features))
        return routed  # [B, C_t, Ht, Wt]

    # ------------------------------------------------------------------
    # Phase 2: channel and spatial weighting
    # ------------------------------------------------------------------
    def _channel_gate(self, f_tea: torch.Tensor) -> torch.Tensor:
        # GAP -> 1D rFFT along channel -> magnitude of first k_cut -> MLP -> sigmoid.
        desc = f_tea.flatten(2).mean(dim=2)  # [B, C_t]
        zeta = torch.fft.rfft(desc, dim=1)  # [B, C_t//2+1] complex
        a = zeta[:, : self.k_cut].abs()  # [B, k_cut]
        g = self.channel_gate(a)  # [B, C_t]
        return g.unsqueeze(-1).unsqueeze(-1)  # [B, C_t, 1, 1]

    def _local_intensity(self, sar_gray: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
        # E(x,y) = mean_{K_I x K_I} |I|^2, then area-average to token grid.
        e = sar_gray.pow(2)
        k = self.K_window
        pad = k // 2
        e = F.avg_pool2d(F.pad(e, [pad, pad, pad, pad], mode="replicate"), k, stride=1)
        e = F.adaptive_avg_pool2d(e, (Ht, Wt))  # patch-cell area average
        return e  # [B, 1, Ht, Wt]

    def _soft_mask(self, energy: torch.Tensor, hbb_norm: list, Ht: int, Wt: int) -> torch.Tensor:
        """Build W_soft per image from horizontal bounding boxes (Eq. soft_mask).

        ``hbb_norm`` is a per-image list of HBB tensors [N, 4] in normalized
        [0, 1] xyxy coordinates.  The token-center-in-box rule assigns each
        token to the largest containing box; inside-box tokens use normalized
        energy, outside-box tokens use Gaussian decay.
        """
        device = energy.device
        yy, xx = torch.meshgrid(
            torch.arange(Ht, device=device, dtype=energy.dtype),
            torch.arange(Wt, device=device, dtype=energy.dtype),
            indexing="ij",
        )
        # Token centers in token-cell units.
        cx = xx.float() + 0.5  # [Ht, Wt]
        cy = yy.float() + 0.5  # [Ht, Wt]

        out = torch.empty((energy.shape[0], 1, Ht, Wt), device=device, dtype=energy.dtype)

        for b in range(energy.shape[0]):
            e = energy[b, 0]  # [Ht, Wt]
            boxes = hbb_norm[b]
            if boxes is None or len(boxes) == 0:
                out[b, 0] = 1.0  # Empty GT: uniform weight
                continue

            # Convert normalized [0,1] boxes to token-cell coordinates.
            boxes_tc = boxes.detach().clone()
            boxes_tc[:, 0] *= Wt  # x1
            boxes_tc[:, 1] *= Ht  # y1
            boxes_tc[:, 2] *= Wt  # x2
            boxes_tc[:, 3] *= Ht  # y2

            N = boxes_tc.shape[0]
            assigned = torch.full((Ht, Wt), -1, device=device, dtype=torch.long)
            best_area = torch.full((Ht, Wt), -1.0, device=device, dtype=energy.dtype)
            min_d2 = torch.full((Ht, Wt), float("inf"), device=device, dtype=energy.dtype)

            for k in range(N):
                x1, y1, x2, y2 = boxes_tc[k].tolist()
                inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
                box_area = (x2 - x1) * (y2 - y1)

                # Euclidean distance to box boundary (0 inside).
                dx = torch.clamp(torch.min(cx - x2, x1 - cx), min=0.0)
                dy = torch.clamp(torch.min(cy - y2, y1 - cy), min=0.0)
                d2 = dx * dx + dy * dy
                min_d2 = torch.minimum(min_d2, d2)

                # Assign to largest box (smallest k breaks ties via >).
                mask = inside & (box_area > best_area)
                assigned = torch.where(mask, torch.full_like(assigned, k), assigned)
                best_area = torch.where(mask, torch.full_like(best_area, box_area), best_area)

            # Compute per-box energy extrema and assign W_soft.
            w = torch.full((Ht, Wt), 0.0, device=device, dtype=energy.dtype)
            for k in range(N):
                mask_k = assigned == k
                if not mask_k.any():
                    continue
                e_vals = e[mask_k]
                e_min_k = e_vals.min()
                e_max_k = e_vals.max()
                if (e_max_k - e_min_k).abs() < 1e-6:
                    w[mask_k] = 0.5
                else:
                    w[mask_k] = (e[mask_k] - e_min_k) / (e_max_k - e_min_k + 1e-6)

            # Outside-box tokens: Gaussian decay.
            outside = assigned < 0
            if outside.any():
                w[outside] = self.mu * torch.exp(-min_d2[outside] / (2.0 * self.sigma ** 2))

            out[b, 0] = w.clamp(0.0, 1.0)

        return out

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------
    def forward(
        self,
        student_features: list,
        teacher_features: list,
        hbb_norm: list,
        sar_gray: torch.Tensor,
    ):
        """Args:
            student_features: list [S3, S4, F5] where F5 is the AIFI-refined S5
                produced by the student's AIFI encoder (paper routed interface
                R = {S3, S4, F5}).
            teacher_features: list [T3, T4, T5_avg] — projected teacher block
                features, each [B, C_t, Ht, Wt]. T3=T^{(3)}, T4=T^{(6)},
                T5_avg=(T^{(9)}+T^{(12)})/2.
            hbb_norm: per-image list of HBB tensors [N, 4] (xyxy, normalized
                [0,1]) for the spatial loss weight W_soft.
            sar_gray: SAR intensity magnitude [B, 1, H, W].
        Returns:
            f_tea_hat: purified teacher feature [B, C_t, Ht, Wt].
            loss: L_DRCP scalar.
        """
        B, C_t, Ht, Wt = teacher_features[0].shape

        # Phase 1: Route student features and compute routing weights.
        f_stu_routed, alphas = self._route(student_features, Ht, Wt)

        # Route teacher features with stop-gradient on alpha.
        f_tea = self._route_teacher(teacher_features, alphas)

        # Phase 2: Channel gate g (GAP -> rFFT -> magnitude -> MLP -> sigmoid).
        g = self._channel_gate(f_tea)  # [B, C_t, 1, 1]
        f_tea_hat = g * f_tea  # purified teacher: only g, NOT w_soft

        # Spatial loss weight W_soft (horizontal bounding boxes + local intensity).
        energy = self._local_intensity(sar_gray, Ht, Wt)  # [B, 1, Ht, Wt]
        w_soft = self._soft_mask(energy, hbb_norm, Ht, Wt)  # [B, 1, Ht, Wt]

        # Eq.13: spatially weighted cosine distance.  W_soft is the explicit
        # spatial loss weight; it is NOT multiplied into the teacher feature
        # before cosine normalization (paper Sec. 3.2, Eq. drcp_loss).
        a = f_tea_hat.flatten(2)  # [B, C_t, HW]
        b = f_stu_routed.flatten(2)  # [B, C_t, HW]
        a_norm = a.norm(dim=1)  # [B, HW]
        b_norm = b.norm(dim=1)  # [B, HW]
        cos = (a * b).sum(dim=1) / (a_norm * b_norm + 1e-8)  # Eq. stable_cosine_distance
        weight = w_soft.flatten(2).squeeze(1)  # [B, HW]
        loss = (weight * (1.0 - cos)).sum(dim=1) / (weight.sum(dim=1) + 1e-6)
        loss = loss.mean()

        return f_tea_hat, loss
