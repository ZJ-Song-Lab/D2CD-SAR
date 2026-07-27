# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

"""A^2TD-LoRA: Adaptive Task-Decoupled Orthogonal LoRA.

Implements the parameter-level decoupling module of the SAR-RTDETR paper.
Two parallel low-rank branches (detection / distillation) are injected into
the frozen AIFI weight matrix W0. An internal direction-aware variance gate
self-modulates the distillation branch capacity. Orthogonal regularization and
an L1 sparsity penalty keep the branches independent and compact.

Reference equations (SAR-RTDETR.tex):
  Eq.(6)  forward:  h = W0 x + B_det Sigma_det A_det x + B_distill Sigma_hat A_distill x
  Eq.(7)  L_ortho = ||A_det A_distill^T||_F^2 + ||B_det^T B_distill||_F^2
  Eq.(8)  c = ||g_drcp + g_task||_2 / (||g_drcp||_2 + ||g_task||_2 + eps)
  Eq.(9)  r = clip(||g_drcp||_1 / (||g_task||_1 + eps), 0, r_max) * c
  Eq.(10) alpha = clip(alpha_base + gamma * tanh(sigma_r), 0, alpha_max)
  Eq.(11) w = alpha * w_prev + (1 - alpha) * r
  Eq.(12) Sigma_hat_distill = w * Sigma_distill
  Eq.(13) L_sparsity = ||Sigma_det||_1 + ||Sigma_distill||_1
  Eq.(15) W_deploy = W0 + B_det Sigma_det A_det + B_distill (w * Sigma_distill) A_distill
"""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceGate(nn.Module):
    """Built-in direction-aware variance gating (internal self-modulating valve).

    Maintains an EMA gating scalar ``w`` that scales the distillation singular
    values :math:`\\Sigma_{distill}`. Updated every step from the gradient
    consistency / direction-aware ratio measured on the AIFI LoRA parameter
    space. All state is kept in buffers so it is device-aware and not trained
    by autograd.
    """

    def __init__(
        self,
        K: int = 10,
        alpha_base: float = 0.9,
        alpha_max: float = 0.99,
        gamma: float = 0.05,
        r_max: float = 5.0,
    ):
        super().__init__()
        self.K = K
        self.alpha_base = alpha_base
        self.alpha_max = alpha_max
        self.gamma = gamma
        self.r_max = r_max
        # w^(0): start with an active distillation branch.
        self.register_buffer("gate", torch.tensor(1.0))
        # sliding historical buffer of r^(t)
        self.register_buffer("r_buffer", torch.zeros(K))
        self.register_buffer("num_seen", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, grad_drcp: torch.Tensor, grad_task: torch.Tensor, eps: float = 1e-8) -> float:
        """Push one observation and return the updated gate value.

        Args:
            grad_drcp: flattened gradient of L_DRCP on the AIFI LoRA space.
            grad_task: flattened gradient of L_task  on the AIFI LoRA space.
        """
        g_drcp = grad_drcp.detach().flatten()
        g_task = grad_task.detach().flatten()

        # Eq.(8): gradient consistency coefficient c in [0, 1].
        denom = g_drcp.norm(2) + g_task.norm(2) + eps
        c = ((g_drcp + g_task).norm(2)) / denom
        c = c.clamp(0.0, 1.0)

        # Eq.(9): direction-aware gradient ratio r.
        ratio = (g_drcp.abs().sum()) / (g_task.abs().sum() + eps)
        ratio = ratio.clamp(0.0, self.r_max)
        r = ratio * c

        # Update sliding buffer and compute local sample variance sigma_r.
        idx = int(self.num_seen.item()) % self.K
        self.r_buffer[idx] = r
        self.num_seen += 1
        n = min(int(self.num_seen.item()), self.K)
        if n > 1:
            sigma_r = self.r_buffer[:n].var(unbiased=False)
        else:
            sigma_r = torch.zeros((), device=r.device)

        # Eq.(10): adaptive smoothing factor alpha.
        alpha = (self.alpha_base + self.gamma * torch.tanh(sigma_r)).clamp(0.0, self.alpha_max)

        # Eq.(11): dynamic EMA gating scalar w.
        w_new = alpha * self.gate + (1.0 - alpha) * r
        self.gate.copy_(w_new.detach())
        return float(self.gate.item())

    def value(self) -> float:
        return float(self.gate.item())


class ATDLoRALinear(nn.Module):
    """A frozen ``Linear`` augmented with two orthogonal LoRA branches.

    The base weight ``W0`` is frozen. The detection branch (``A_det, B_det,
    Sigma_det``) is a static trainable low-rank update; the distillation branch
    (``A_distill, B_distill, Sigma_distill``) is gated by the shared
    :class:`VarianceGate`. ``B`` matrices are zero-initialised so the module
    starts exactly from ``W0``.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r_det: int = 16,
        r_distill: int = 16,
        gate: VarianceGate = None,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r_det = r_det
        self.r_distill = r_distill

        # Frozen base parameters (W0).
        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        # Detection branch: B zero-init => zero contribution at start.
        self.A_det = nn.Parameter(torch.randn(r_det, self.in_features) * init_scale)
        self.B_det = nn.Parameter(torch.zeros(self.out_features, r_det))
        self.sigma_det = nn.Parameter(torch.ones(r_det))

        # Distillation branch: B zero-init => zero contribution at start.
        self.A_distill = nn.Parameter(torch.randn(r_distill, self.in_features) * init_scale)
        self.B_distill = nn.Parameter(torch.zeros(self.out_features, r_distill))
        self.sigma_distill = nn.Parameter(torch.ones(r_distill))

        self.gate = gate if gate is not None else VarianceGate()

    def forward(
        self,
        x: torch.Tensor,
        detach_distill: bool = False,
        detach_det: bool = False,
    ) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)

        # Detection branch: B_det (Sigma_det (A_det x))
        det = F.linear(x, self.A_det)
        det = det * self.sigma_det
        det = F.linear(det, self.B_det)

        # Distillation branch: B_distill (w * Sigma_distill (A_distill x))
        w = self.gate.value()
        dist = F.linear(x, self.A_distill)
        dist = dist * (self.sigma_distill * w)
        dist = F.linear(dist, self.B_distill)

        # Gradient isolation: when computing the task loss the distillation
        # branch must not receive task gradients (detach_distill=True), and
        # when computing the distillation loss the detection branch must not
        # receive distillation gradients (detach_det=True).
        if detach_distill:
            dist = dist.detach()
        if detach_det:
            det = det.detach()

        return out + det + dist

    # ---- losses -----------------------------------------------------------
    def orthogonal_loss(self) -> torch.Tensor:
        """Eq.(7): L_ortho on the low-dimensional basis matrices."""
        a_cross = torch.matmul(self.A_det, self.A_distill.t())  # [r_det, r_distill]
        b_cross = torch.matmul(self.B_det.t(), self.B_distill)  # [r_det, r_distill]
        return a_cross.pow(2).sum() + b_cross.pow(2).sum()

    def sparsity_loss(self) -> torch.Tensor:
        """Eq.(13): L_sparsity L1 penalty on the singular-value diagonals."""
        return self.sigma_det.abs().sum() + self.sigma_distill.abs().sum()

    # ---- deployment -------------------------------------------------------
    def delta_weight(self) -> torch.Tensor:
        """The full low-rank delta to merge back into W0 (Eq. 15)."""
        w = self.gate.value()
        delta_det = self.B_det @ torch.diag(self.sigma_det) @ self.A_det
        delta_dist = self.B_distill @ torch.diag(self.sigma_distill * w) @ self.A_distill
        return delta_det + delta_dist

    def merge(self) -> nn.Linear:
        """Reparameterize into a plain ``nn.Linear`` for zero-overhead inference."""
        merged_weight = self.weight + self.delta_weight()
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        with torch.no_grad():
            linear.weight.copy_(merged_weight.detach())
            if self.bias is not None:
                linear.bias.copy_(self.bias.detach())
        linear.eval()
        return linear


def collect_atd_params(modules) -> list:
    """Collect the AIFI LoRA parameters (theta_AIFI) for variance gating."""
    params = []
    for m in modules:
        params.extend([
            m.A_det, m.B_det, m.sigma_det,
            m.A_distill, m.B_distill, m.sigma_distill,
        ])
    return params


def flattened_grad(loss, params, retain_graph=True) -> torch.Tensor:
    """Flattened gradient of ``loss`` w.r.t. ``params`` (no accumulation)."""
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            g = torch.zeros_like(p)
        flat.append(g.detach().flatten())
    return torch.cat(flat) if flat else torch.zeros(1)


def total_orthogonal_loss(modules) -> torch.Tensor:
    loss = modules[0].orthogonal_loss()
    for m in modules[1:]:
        loss = loss + m.orthogonal_loss()
    return loss


def total_sparsity_loss(modules) -> torch.Tensor:
    loss = modules[0].sparsity_loss()
    for m in modules[1:]:
        loss = loss + m.sparsity_loss()
    return loss
