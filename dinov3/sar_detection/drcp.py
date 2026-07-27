# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

r"""DRCP: Depth-Routed Cross-Modal Purifier.

Implements the feature-level alignment module of the SAR-RTDETR paper.

Phase 1 - Depth-wise Semantic Routing (Eq. 3-5):
  S_i interpolated to the teacher spatial resolution and projected to C_t;
  routing weights alpha_i = softmax_i(w_d^T RMSNorm(GAP(S_i)));
  F_stu_routed = sum_i alpha_i S_i.

Phase 2 - Joint Spatio-Channel Purification (Eq. 6-12):
  g         = MLP(low_pass_FFT(F_tea))           # frequency-guided channel gate
  E(x,y)    = mean_{N_K} |I_SAR|^2               # local scattering energy
  W_soft    = inside-OBB normalized energy | mu * exp(-d^2 / 2 sigma^2)
  M_joint   = g \otimes W_soft
  F_tea_hat = M_joint \odot F_tea

L_DRCP (Eq. 13): spatially weighted cosine distance between the teacher and
the routed student feature, with W_soft re-applied as the region weight to
avoid zero-vector numerical instability.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.layers.rms_norm import RMSNorm


class DRCP(nn.Module):
    """Depth-Routed Cross-Modal Purifier.

    Args:
        C_t: teacher channel dimension (DINOv3-ViT-B -> 768).
        student_dims: channel dims of the routed student levels
            (e.g. (C3, C4, C5) = (128, 256, 512) for RT-DETR-R18). The deepest
            level is the AIFI output F5 so the distillation branch receives
            gradient from L_DRCP.
        K_window: scattering-energy averaging window (pixels).
        mu: background soft-mask magnitude outside OBBs.
        sigma: background soft-mask Gaussian bandwidth.
        lowpass_ratio: fraction of channel frequencies kept by the low-pass.
    """

    def __init__(
        self,
        C_t: int = 768,
        student_dims=(128, 256, 512),
        K_window: int = 3,
        mu: float = 0.3,
        sigma: float = 2.0,
        lowpass_ratio: float = 0.125,
    ):
        super().__init__()
        self.C_t = C_t
        self.K_window = K_window
        self.mu = mu
        self.sigma = sigma
        self.lowpass_L = max(2, int(C_t * lowpass_ratio))

        # Student-level projections to the teacher channel dim.
        self.proj = nn.ModuleList(
            [nn.Conv2d(c, C_t, kernel_size=1, bias=True) for c in student_dims]
        )
        # Learnable pseudo-query for depth-wise routing.
        self.w_d = nn.Parameter(torch.randn(C_t) * 0.02)
        self.key_norm = RMSNorm(C_t)

        # Frequency-guided channel gate (bottleneck MLP).
        hidden = max(C_t // 4, 8)
        self.channel_gate = nn.Sequential(
            nn.Linear(C_t, hidden),
            nn.GELU(),
            nn.Linear(hidden, C_t),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Phase 1: depth-wise semantic routing
    # ------------------------------------------------------------------
    def _route(self, student_features: list, Ht: int, Wt: int) -> torch.Tensor:
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
        return routed  # [B, C_t, Ht, Wt]

    # ------------------------------------------------------------------
    # Phase 2: joint spatio-channel purification
    # ------------------------------------------------------------------
    def _channel_gate(self, f_tea: torch.Tensor) -> torch.Tensor:
        # 1D FFT along the channel dim, low-pass, inverse FFT, GAP -> MLP.
        freq = torch.fft.rfft(f_tea, dim=1)
        freq[..., self.lowpass_L:, :, :] = 0
        recon = torch.fft.irfft(freq, n=self.C_t, dim=1)  # [B, C_t, Ht, Wt]
        desc = recon.flatten(2).mean(dim=2)  # [B, C_t]
        g = self.channel_gate(desc).unsqueeze(-1).unsqueeze(-1)  # [B, C_t, 1, 1]
        return g

    def _scattering_energy(self, sar_gray: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
        # sar_gray: [B, 1, H, W] (magnitude of the SAR intensity).
        e = sar_gray.pow(2)
        e = F.interpolate(e, size=(Ht, Wt), mode="bilinear", align_corners=False)
        k = self.K_window
        pad = k // 2
        e = F.avg_pool2d(F.pad(e, [pad, pad, pad, pad], mode="replicate"), k, stride=1)
        return e  # [B, 1, Ht, Wt]

    def _soft_mask(self, energy: torch.Tensor, obb_norm: list, Ht: int, Wt: int) -> torch.Tensor:
        """Build W_soft per image from oriented bounding boxes (Eq. 9-11).

        ``obb_norm`` is a per-image list of OBB corner tensors [N, 8] in
        normalized [0, 1] coordinates (4 corner xy-pairs).  The point-to-OBB
        distance is computed in each OBB's local frame, so the Gaussian decay
        is *anisotropic* and aligned with the ship orientation, as required
        by Eq. (9).  When OBBs are unavailable the routine falls back to the
        axis-aligned rectangle derived from the same corners.
        """
        device = energy.device
        yy, xx = torch.meshgrid(
            torch.arange(Ht, device=device, dtype=energy.dtype),
            torch.arange(Wt, device=device, dtype=energy.dtype),
            indexing="ij",
        )
        pts = torch.stack([xx, yy], dim=-1)  # [Ht, Wt, 2]  (x, y)
        out = torch.empty((energy.shape[0], 1, Ht, Wt), device=device, dtype=energy.dtype)
        for b in range(energy.shape[0]):
            e = energy[b, 0]  # [Ht, Wt]
            corners = obb_norm[b]
            if corners is None or len(corners) == 0:
                out[b, 0] = self.mu
                continue
            inside_val = torch.full((Ht, Wt), -1.0, device=device, dtype=energy.dtype)
            min_d2 = torch.full((Ht, Wt), float("inf"), device=device, dtype=energy.dtype)
            for obb in corners:
                c = obb.reshape(4, 2)  # [4, 2] normalized (x, y)
                cx = c[:, 0] * Wt
                cy = c[:, 1] * Ht
                c_pix = torch.stack([cx, cy], dim=1)  # [4, 2]
                center = c_pix.mean(dim=0)  # [2]
                e1 = c_pix[1] - c_pix[0]
                e2 = c_pix[3] - c_pix[0]
                l1 = e1.norm()
                l2 = e2.norm()
                if l1 < 1e-6 or l2 < 1e-6:
                    continue
                u1 = e1 / l1
                u2 = e2 / l2
                h1 = l1 / 2.0
                h2 = l2 / 2.0
                d = pts - center  # [Ht, Wt, 2]
                local_x = (d * u1).sum(dim=-1)  # [Ht, Wt]
                local_y = (d * u2).sum(dim=-1)  # [Ht, Wt]
                dist_x = torch.clamp(local_x.abs() - h1, min=0.0)
                dist_y = torch.clamp(local_y.abs() - h2, min=0.0)
                d2 = dist_x * dist_x + dist_y * dist_y
                min_d2 = torch.minimum(min_d2, d2)
                inside = (local_x.abs() <= h1) & (local_y.abs() <= h2)
                if inside.any():
                    e_in = e[inside]
                    e_min = e_in.min()
                    e_max = e_in.max()
                    norm_e = (e - e_min) / (e_max - e_min + 1e-6)
                    inside_val = torch.maximum(inside_val, torch.where(inside, norm_e, torch.full_like(norm_e, -1.0)))
            outside_val = self.mu * torch.exp(-min_d2 / (2.0 * self.sigma ** 2))
            mask = torch.where(inside_val >= 0, inside_val, outside_val)
            out[b, 0] = mask
        return out.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------
    def forward(
        self,
        student_features: list,
        f_tea: torch.Tensor,
        obb_norm: list,
        sar_gray: torch.Tensor,
    ):
        """Args:
            student_features: list [S3, S4, F5] (deepest is the AIFI output).
            f_tea: teacher patch features [B, C_t, Ht, Wt].
            obb_norm: per-image list of OBB corner tensors [N, 8] (normalized
                xy-pairs in [0, 1]) for the anisotropic W_soft (Eq. 9-11).
            sar_gray: SAR intensity magnitude [B, 1, H, W].
        Returns:
            f_tea_hat: purified teacher feature [B, C_t, Ht, Wt].
            loss: L_DRCP scalar.
        """
        B, C_t, Ht, Wt = f_tea.shape

        f_stu_routed = self._route(student_features, Ht, Wt)  # [B, C_t, Ht, Wt]
        g = self._channel_gate(f_tea)  # [B, C_t, 1, 1]
        energy = self._scattering_energy(sar_gray, Ht, Wt)  # [B, 1, Ht, Wt]
        w_soft = self._soft_mask(energy, obb_norm, Ht, Wt)  # [B, 1, Ht, Wt]

        m_joint = g * w_soft  # [B, C_t, Ht, Wt]
        f_tea_hat = m_joint * f_tea

        # Eq.13: spatially weighted cosine distance between the *purified*
        # teacher feature and the routed student feature.  Using f_tea_hat
        # (rather than raw f_tea) ensures the FFT channel gate g and the OBB
        # soft mask W_soft both enter the distillation gradient.  Norms are
        # clamped to avoid division-by-near-zero in masked regions.
        a = f_tea_hat.flatten(2)  # [B, C_t, HW]
        b = f_stu_routed.flatten(2)  # [B, C_t, HW]
        a_norm = a.norm(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1, HW]
        b_norm = b.norm(dim=1, keepdim=True).clamp(min=1e-6)
        cos = (a * b).sum(dim=1) / (a_norm * b_norm)  # [B, HW]
        weight = w_soft.flatten(2).squeeze(1)  # [B, HW]
        loss = (weight * (1.0 - cos)).sum(dim=1) / (weight.sum(dim=1) + 1e-6)
        loss = loss.mean()

        return f_tea_hat, loss
