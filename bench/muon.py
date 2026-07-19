"""Minimal Muon optimizer (Keller Jordan's MomentUm Orthogonalized by
Newton-Schulz), for the 7B-on-64GB memory study.

Muon applies to 2D matrix parameters only and keeps ONE momentum buffer
(vs AdamW's two moments). Embeddings / lm_head / 1D params should be given
to a separate AdamW group, per standard practice.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Approximate orthogonalization of G (quintic Newton-Schulz, bf16)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                d = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                d = newton_schulz5(d.reshape(d.size(0), -1),
                                   steps=group["ns_steps"]).reshape_as(p)
                scale = max(1.0, p.size(0) / p[0].numel()) ** 0.5
                p.add_(d, alpha=-group["lr"] * scale)
        return loss


def build_muon_adamw(model, muon_lr=0.02, adamw_lr=1e-4):
    """Split params: 2D non-embedding matrices -> Muon; rest -> AdamW."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return (Muon(muon_params, lr=muon_lr),
            torch.optim.AdamW(adamw_params, lr=adamw_lr, foreach=True))
