"""Quantization layers built purely on torch>=2.4 native CUDA ops.

No TransformerEngine / torchao / bitsandbytes required — everything below uses
kernels that ship inside stock PyTorch (cu128 build):

Training-side (QAT = quantization-aware training, fake-quant + straight-through
estimator; this SIMULATES low-bit arithmetic in high precision, so it is
*slower* than the plain-precision baseline — the payoff is at deploy time):
  - QATLinear      : intN per-channel weight + int8 per-token activation fake quant
  - Fp8QATLinear   : fake quant via real float8_e4m3fn round-trip casts
  - Fp8TrainLinear : REAL fp8 GEMMs in forward and backward via torch._scaled_mm
                     (per-tensor dynamic scaling, e4m3 fwd / e5m2 grads — the
                     same recipe as TransformerEngine "delayed scaling" minus
                     the history window)

Inference-side (real low-bit kernels):
  - Fp8InferLinear : dynamic per-tensor act fp8 x per-tensor weight fp8, _scaled_mm
  - Int8InferLinear: dynamic per-token act int8 x per-channel weight int8, _int_mm
  - Int4InferLinear: weight-only groupwise int4 (tinygemm _weight_int4pack_mm)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

E4M3_MAX = 448.0
E5M2_MAX = 57344.0


# --------------------------------------------------------------------------
# fake-quant primitives (QAT)
# --------------------------------------------------------------------------

def qdq_weight_per_channel(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel quantize->dequantize. w: (N, K)."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return (w / scale).round().clamp(-qmax - 1, qmax) * scale


def qdq_act_per_token(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Symmetric per-token (last-dim group) quantize->dequantize."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    return (x / scale).round().clamp(-qmax - 1, qmax) * scale


def qdq_fp8(t: torch.Tensor) -> torch.Tensor:
    """Fake quant through a real e4m3 round-trip with per-tensor scale."""
    scale = t.abs().amax().clamp(min=1e-12) / E4M3_MAX
    return (t / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(t.dtype) * scale


class QATLinear(nn.Module):
    """Fake-quant wrapper around an existing nn.Linear (weights stay fp32 master).

    Straight-through estimator: forward sees quantized values, backward flows
    through unmodified.
    """

    def __init__(self, lin: nn.Linear, w_bits: int = 4, a_bits: int = 8):
        super().__init__()
        self.lin = lin
        self.w_bits = w_bits
        self.a_bits = a_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.lin.weight
        w = w + (qdq_weight_per_channel(w, self.w_bits) - w).detach()
        x = x + (qdq_act_per_token(x, self.a_bits) - x).detach()
        return F.linear(x, w, self.lin.bias)


class Fp8QATLinear(nn.Module):
    """FP8 fake-quant (weights + activations through real e4m3 casts) + STE."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.lin = lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.lin.weight
        w = w + (qdq_fp8(w) - w).detach()
        x = x + (qdq_fp8(x) - x).detach()
        return F.linear(x, w, self.lin.bias)


# --------------------------------------------------------------------------
# real FP8 training GEMM (forward + backward through torch._scaled_mm)
# --------------------------------------------------------------------------

def _to_fp8(t: torch.Tensor, dtype: torch.dtype, fmax: float):
    """Per-tensor dynamic scaling: t ~= q * scale, scale is a 0-dim fp32 tensor."""
    scale = t.abs().amax().float().clamp(min=1e-12) / fmax
    q = (t.float() / scale).clamp(-fmax, fmax).to(dtype)
    return q, scale


class _Fp8MatmulFn(torch.autograd.Function):
    """y = x @ w.T with fp8 GEMMs in fwd (e4m3 x e4m3) and bwd (e5m2 x e4m3)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor):
        # x: (M, K) bf16, w: (N, K) bf16
        x8, sx = _to_fp8(x, torch.float8_e4m3fn, E4M3_MAX)
        w8, sw = _to_fp8(w, torch.float8_e4m3fn, E4M3_MAX)
        # second operand must be column-major: w8.t() is a (K, N) col-major view
        y = torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw,
                             out_dtype=torch.bfloat16)
        ctx.save_for_backward(x8, sx, w8, sw)
        return y

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        x8, sx, w8, sw = ctx.saved_tensors
        g8, sg = _to_fp8(g, torch.float8_e5m2, E5M2_MAX)
        # grad_x (M,K) = g (M,N) @ w (N,K); need w8 as (N,K) column-major
        w8_cm = w8.t().contiguous().t()
        gx = torch._scaled_mm(g8, w8_cm, scale_a=sg, scale_b=sw,
                              out_dtype=torch.bfloat16)
        # grad_w (N,K) = g.T (N,M) @ x (M,K); row-major g.T, col-major x
        g8_t = g8.t().contiguous()
        x8_cm = x8.t().contiguous().t()
        gw = torch._scaled_mm(g8_t, x8_cm, scale_a=sg, scale_b=sx,
                              out_dtype=torch.bfloat16)
        return gx, gw


class Fp8TrainLinear(nn.Module):
    """Linear with real fp8 GEMMs for fwd + bwd. fp32 master weights."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.lin = lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        m = x2.shape[0]
        pad = (-m) % 16
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        y = _Fp8MatmulFn.apply(x2.to(torch.bfloat16),
                               self.lin.weight.to(torch.bfloat16))
        if pad:
            y = y[:m]
        y = y.reshape(*shp[:-1], -1)
        if self.lin.bias is not None:
            y = y + self.lin.bias.to(y.dtype)
        return y


# --------------------------------------------------------------------------
# real quantized inference linears
# --------------------------------------------------------------------------

class Fp8InferLinear(nn.Module):
    """Weight pre-quantized e4m3 (per-tensor); dynamic per-tensor act quant."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach()
        w8, sw = _to_fp8(w, torch.float8_e4m3fn, E4M3_MAX)
        self.register_buffer("w8", w8)          # (N, K) row-major
        self.register_buffer("sw", sw)
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.detach().to(torch.bfloat16),
                                 persistent=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        m = x2.shape[0]
        pad = (-m) % 16
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        x8, sx = _to_fp8(x2, torch.float8_e4m3fn, E4M3_MAX)
        y = torch._scaled_mm(x8, self.w8.t(), scale_a=sx, scale_b=self.sw,
                             out_dtype=torch.bfloat16)
        if pad:
            y = y[:m]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], -1).to(x.dtype)


class Int8InferLinear(nn.Module):
    """W8A8 dynamic: per-channel int8 weight, per-token int8 act, _int_mm."""

    MIN_M = 32  # cuBLAS int8 GEMM wants M > 16; pad tiny batches up

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach()                              # (N, K)
        sw = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        w8 = (w / sw).round().clamp(-128, 127).to(torch.int8)
        self.register_buffer("w8t", w8.t().contiguous())      # (K, N) for _int_mm
        self.register_buffer("sw", sw.t().float())            # (1, N)
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.detach().to(torch.bfloat16),
                                 persistent=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        m = x2.shape[0]
        pad = max(0, self.MIN_M - m)
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        sx = x2.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-8) / 127.0
        x8 = (x2.float() / sx).round().clamp(-128, 127).to(torch.int8)
        y32 = torch._int_mm(x8, self.w8t)                     # (M, N) int32
        y = (y32.float() * sx * self.sw).to(torch.bfloat16)
        if pad:
            y = y[:m]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], -1).to(x.dtype)


def _group_quantize_int4(w: torch.Tensor, groupsize: int):
    """Asymmetric groupwise 4-bit quant (gpt-fast layout).

    Returns packed uint8 (N, K/2) and scales_and_zeros (K/groupsize, N, 2) bf16.
    """
    n, k = w.shape
    wg = w.float().reshape(-1, groupsize)
    wmax = wg.amax(dim=1, keepdim=True)
    wmin = wg.amin(dim=1, keepdim=True)
    scales = (wmax - wmin).clamp(min=1e-6) / 15.0
    zeros = wmin + scales * 8.0
    wq = wg.sub(wmin).div(scales).round().clamp(0, 15).to(torch.int32).reshape(n, k)
    scales_and_zeros = torch.cat(
        [scales.reshape(n, k // groupsize, 1), zeros.reshape(n, k // groupsize, 1)],
        dim=2,
    ).transpose(0, 1).contiguous().to(torch.bfloat16)
    packed = (wq[:, ::2] << 4 | wq[:, 1::2]).to(torch.uint8)   # (N, K/2)
    return packed, scales_and_zeros


class Int4InferLinear(nn.Module):
    """Weight-only groupwise int4 via tinygemm (_weight_int4pack_mm), bf16 act."""

    GROUPSIZE = 128
    INNER_K_TILES = 8

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach()
        packed, saz = _group_quantize_int4(w, self.GROUPSIZE)
        w4 = torch.ops.aten._convert_weight_to_int4pack(packed, self.INNER_K_TILES)
        self.register_buffer("w4", w4)
        self.register_buffer("saz", saz)
        self.out_features = w.shape[0]
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.detach().to(torch.bfloat16),
                                 persistent=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).to(torch.bfloat16)
        y = torch.ops.aten._weight_int4pack_mm(
            x2, self.w4, self.GROUPSIZE, self.saz)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], self.out_features).to(x.dtype)


# --------------------------------------------------------------------------
# module swapping
# --------------------------------------------------------------------------

def _dims_ok(lin: nn.Linear, mode: str) -> bool:
    n, k = lin.out_features, lin.in_features
    if min(n, k) < 256:          # skip tiny projections (heads, gates)
        return False
    if mode == "int4":
        return k % (Int4InferLinear.INNER_K_TILES * 16) == 0 and n % 16 == 0
    if mode in ("fp8", "int8", "fp8_train"):
        return k % 16 == 0 and n % 16 == 0
    return True                   # fake-quant QAT has no shape constraints


_FACTORY = {
    "int4_qat": lambda lin: QATLinear(lin, w_bits=4, a_bits=8),
    "int8_qat": lambda lin: QATLinear(lin, w_bits=8, a_bits=8),
    "fp8_qat": Fp8QATLinear,
    "fp8_train": Fp8TrainLinear,
    "fp8": Fp8InferLinear,
    "int8": Int8InferLinear,
    "int4": Int4InferLinear,
}

_DIMCHECK = {
    "int4_qat": "qat", "int8_qat": "qat", "fp8_qat": "qat",
    "fp8_train": "fp8_train", "fp8": "fp8", "int8": "int8", "int4": "int4",
}


def swap_linears(model: nn.Module, mode: str):
    """Replace eligible nn.Linear modules in-place. Returns (replaced, skipped)."""
    factory = _FACTORY[mode]
    dimmode = _DIMCHECK[mode]
    replaced = skipped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                if _dims_ok(child, dimmode):
                    setattr(parent, name, factory(child))
                    replaced += 1
                else:
                    skipped += 1
    return replaced, skipped
