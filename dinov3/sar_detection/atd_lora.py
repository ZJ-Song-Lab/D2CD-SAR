# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

"""A^2TD-LoRA: Adaptive Task-Decoupled Orthogonal LoRA.

Implements the parameter-level decoupling module of the D²CD-SAR paper.
Two parallel low-rank branches (detection / distillation) are injected into
the frozen AIFI attention output projection. An internal direction-aware
variance gate self-modulates the distillation branch capacity. Orthogonal
regularization and an L1 sparsity penalty keep the branches independent and
compact.

Reference equations (D²CD-SAR.tex):
  Eq.(6)  forward:  v = [W0 + B_det Sigma_det A_det + B_distill (w*Sigma_distill) A_distill] u
  Eq.(7)  L_ortho = (<dW_det, dW_distill>_F / (||dW_det||_F ||dW_distill||_F + eps))^2
  Eq.(8)  rho = <p_task, p_distill> / (||p_task||_2 ||p_distill||_2 + eps);
          c_dir = sqrt((1 + Clip(rho, -1, 1)) / 2)
  Eq.(9)  r_mag = clip(||p_distill||_1 / (||p_task||_1 + eps), 0, r_max);
          r = Clip(r_mag * c_dir, 0, 1)
  Eq.(10) alpha = Clip(alpha_base + gamma * tanh(sigma_r), 0, alpha_max)
  Eq.(11) w = alpha * w_prev + (1 - alpha) * r
  Eq.(12) Sigma_hat_distill = w * Sigma_distill
  Eq.(13) L_sparsity = ||Sigma_det||_1 + ||Sigma_distill||_1
  Eq.(15) W_deploy = W0 + B_det Sigma_det A_det + B_distill (w * Sigma_distill) A_distill

Probes p_task and p_distill are gradients of L_task and L_DRCP with respect
to the shared AIFI input activation x^(t) (Eq. activation_probes), not the
LoRA parameter union.
"""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceGate(nn.Module):
    """Built-in direction-aware variance gating (internal self-modulating valve).

    Maintains an EMA gating scalar ``w`` that scales the distillation singular
    values :math:`\\Sigma_{distill}`. Updated every step from the direction
    consistency / direction-aware ratio measured on the shared AIFI input
    activation x^(t). All state is kept in buffers so it is device-aware and
    not trained by autograd.
    """

    def __init__(
        self,
        K: int = 1000,
        alpha_base: float = 0.9,
        alpha_max: float = 0.999,
        gamma: float = 0.5,
        r_max: float = 8.0,
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
    def update(self, p_distill: torch.Tensor, p_task: torch.Tensor, eps: float = 1e-8) -> float:
        """Push one observation and return the updated gate value.

        Args:
            p_distill: gradient of L_DRCP w.r.t. the shared AIFI input z.
            p_task: gradient of L_task w.r.t. the shared AIFI input z.
        """
        p_d = p_distill.detach().flatten()
        p_t = p_task.detach().flatten()

        # Eq.(8): activation-cosine direction agreement c_dir in [0, 1].
        rho = (p_d * p_t).sum() / (p_d.norm(2) * p_t.norm(2) + eps)
        rho = rho.clamp(-1.0, 1.0)
        c = torch.sqrt((1.0 + rho) / 2.0)

        # Eq.(9): direction-aware gradient ratio r, clipped to [0, 1].
        ratio = (p_d.abs().sum()) / (p_t.abs().sum() + eps)
        ratio = ratio.clamp(0.0, self.r_max)
        r = (ratio * c).clamp(0.0, 1.0)

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
        r"""Forward with loss-specific backward routing (Eq. task/distill_backward_view).

        ``sg_theta`` acts on the branch *parameters* rather than on the input
        ``x``: when ``detach_distill=True`` the distillation branch parameters
        are detached (so L_task cannot update theta_distill), but the graph
        from ``x`` through the branch is preserved, so the gradient
        :math:`\nabla_x L_{task}` still receives a contribution from this
        branch.  Likewise for ``detach_det`` and L_DRCP.
        """
        out = F.linear(x, self.weight, self.bias)
        w = self.gate.value()

        # Detection branch: B_det (Sigma_det (A_det x))
        if detach_det:
            A_e, s_e, B_e = self.A_det.detach(), self.sigma_det.detach(), self.B_det.detach()
        else:
            A_e, s_e, B_e = self.A_det, self.sigma_det, self.B_det
        det = F.linear(x, A_e)
        det = det * s_e
        det = F.linear(det, B_e)

        # Distillation branch: B_distill (w * Sigma_distill (A_distill x))
        if detach_distill:
            A_d, s_d, B_d = self.A_distill.detach(), self.sigma_distill.detach(), self.B_distill.detach()
        else:
            A_d, s_d, B_d = self.A_distill, self.sigma_distill, self.B_distill
        dist = F.linear(x, A_d)
        dist = dist * (s_d * w)
        dist = F.linear(dist, B_d)

        return out + det + dist

    # ---- losses -----------------------------------------------------------
    def orthogonal_loss(self) -> torch.Tensor:
        """Eq.(7): scale-normalized overlap of realized low-rank updates."""
        dw_det = self.B_det @ torch.diag(self.sigma_det) @ self.A_det
        dw_distill = self.B_distill @ torch.diag(self.sigma_distill) @ self.A_distill
        inner = (dw_det * dw_distill).sum()
        norm_det = dw_det.norm(p="fro")
        norm_distill = dw_distill.norm(p="fro")
        return (inner / (norm_det * norm_distill + 1e-8)) ** 2

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
