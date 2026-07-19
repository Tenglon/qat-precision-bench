"""GEMM-level microbenchmark: the hardware ceiling for each precision.

For each (M, K, N) computes y = x @ w.T with the fastest stock-torch kernel of
each precision and reports TFLOP/s (2*M*K*N / t). This is the upper bound that
the end-to-end model benchmarks should be read against.

  fp32 : torch.mm, TF32 disabled
  tf32 : torch.mm, TF32 enabled
  bf16 / fp16 : torch.mm
  fp8  : torch._scaled_mm (e4m3 x e4m3, per-tensor scales)
  int8 : torch._int_mm (reported as TOPS on the same axis)
  int4 : tinygemm _weight_int4pack_mm (weight-only, bf16 activations —
         memory-bound kernel, expected to win only at small M)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant import _group_quantize_int4, _to_fp8, E4M3_MAX  # noqa: E402

SHAPES = [
    (4096, 4096, 4096),      # square, compute-bound
    (8192, 8192, 8192),      # large square, compute-bound
    (16384, 1536, 8960),     # Qwen2.5-1.5B MLP up-proj, bs8 x seq1024 tokens
    (16, 8192, 8192),        # decode-like tall-skinny (memory-bound)
]

PRECISIONS = ["fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4"]


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_shape(m, k, n):
    out = {}
    x32 = torch.randn(m, k, device="cuda")
    w32 = torch.randn(n, k, device="cuda")
    flops = 2 * m * k * n

    for prec in PRECISIONS:
        try:
            if prec in ("fp32", "tf32"):
                on = prec == "tf32"
                torch.backends.cuda.matmul.allow_tf32 = on
                a, b = x32, w32.t()
                fn = lambda: torch.mm(a, b)  # noqa: E731
            elif prec in ("bf16", "fp16"):
                dt = torch.bfloat16 if prec == "bf16" else torch.float16
                a, b = x32.to(dt), w32.to(dt).t()
                fn = lambda: torch.mm(a, b)  # noqa: E731
            elif prec == "fp8":
                a, sa = _to_fp8(x32, torch.float8_e4m3fn, E4M3_MAX)
                w8, sb = _to_fp8(w32, torch.float8_e4m3fn, E4M3_MAX)
                b = w8.t()
                fn = lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb,  # noqa: E731
                                              out_dtype=torch.bfloat16)
            elif prec == "int8":
                a = (x32 * 10).round().clamp(-128, 127).to(torch.int8)
                b = (w32 * 10).round().clamp(-128, 127).to(torch.int8).t().contiguous()  # (K, N)
                mm = max(m, 32)
                if mm != m:
                    a = torch.nn.functional.pad(a, (0, 0, 0, mm - m))
                fn = lambda: torch._int_mm(a, b)  # noqa: E731
            elif prec == "int4":
                packed, saz = _group_quantize_int4(w32, 128)
                w4 = torch.ops.aten._convert_weight_to_int4pack(packed, 8)
                a = x32.to(torch.bfloat16)
                fn = lambda: torch.ops.aten._weight_int4pack_mm(a, w4, 128, saz)  # noqa: E731
            t = timeit(fn)
            out[prec] = {"ok": True, "us": t * 1e6, "tflops": flops / t / 1e12}
        except Exception as e:  # noqa: BLE001
            out[prec] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    torch.backends.cuda.matmul.allow_tf32 = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.cuda.get_device_properties(0)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": f"{dev.major}.{dev.minor}",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "shapes": {},
    }
    for m, k, n in SHAPES:
        key = f"{m}x{k}x{n}"
        print(f"== {key}", flush=True)
        result["shapes"][key] = bench_shape(m, k, n)
        print(json.dumps(result["shapes"][key]), flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print("GEMM_DONE", flush=True)


if __name__ == "__main__":
    main()
