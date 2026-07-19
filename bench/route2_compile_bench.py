"""Route 2: what does torch.compile (inductor) recover, with ZERO new deps?

Two levels, same hardware, same stock-torch venv as the eager benchmark:

1. GEMM: eager vs compiled(mode=max-autotune) for bf16 / fp8 / int8.
   For `_int_mm`, max-autotune lets inductor race a Triton int8 template
   against the slow cuBLASLt path — this isolates "interface problem vs
   hardware problem" for Hopper INT8.

2. Language model (Qwen2.5-1.5B, random init, same spec as run_bench):
   - batch forward (bs=16, seq=1024): bf16 / fp8 / int8 swapped models,
     eager vs torch.compile(default mode)
   - decode (bs=1 / bs=32, 128+128 tokens): bf16 / int8 / int4,
     eager vs torch.compile(mode=reduce-overhead) + StaticCache — the
     CUDA-graph path that removes the launch-bound bs=1 ceiling.

  python route2_compile_bench.py --out out/route2.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import GpuSampler  # noqa: E402
from models import get_spec  # noqa: E402
from quant import E4M3_MAX, _group_quantize_int4, _to_fp8, swap_linears  # noqa: E402


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    with GpuSampler() as g:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return statistics.median(times), g.stats()


# ------------------------------------------------------------------ GEMM

GEMM_SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192), (16, 8192, 8192)]


def gemm_section():
    out = {}
    for m, k, n in GEMM_SHAPES:
        key = f"{m}x{k}x{n}"
        out[key] = {}
        flops = 2 * m * k * n
        x32 = torch.randn(m, k, device="cuda")
        w32 = torch.randn(n, k, device="cuda")

        def variants():
            a, b = x32.to(torch.bfloat16), w32.to(torch.bfloat16).t()
            yield "bf16", (lambda: torch.mm(a, b)), (a, b)
            x8, sx = _to_fp8(x32, torch.float8_e4m3fn, E4M3_MAX)
            w8, sw = _to_fp8(w32, torch.float8_e4m3fn, E4M3_MAX)
            bt = w8.t()
            yield "fp8", (lambda: torch._scaled_mm(
                x8, bt, scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)), (x8, bt, sx, sw)
            mm_ = max(m, 32)
            ai = (x32 * 10).round().clamp(-128, 127).to(torch.int8)
            if mm_ != m:
                ai = torch.nn.functional.pad(ai, (0, 0, 0, mm_ - m))
            bi = (w32 * 10).round().clamp(-128, 127).to(torch.int8).t().contiguous()
            yield "int8", (lambda: torch._int_mm(ai, bi)), (ai, bi)

        for name, fn, _keep in variants():
            try:
                t_e, g_e = timeit(fn, 10, 50)
                cfn = torch.compile(fn, mode="max-autotune", dynamic=False)
                t_c, g_c = timeit(cfn, 15, 50)   # warmup covers autotuning
                out[key][name] = {
                    "ok": True,
                    "eager_tflops": flops / t_e / 1e12,
                    "compiled_tflops": flops / t_c / 1e12,
                    "compile_speedup": t_e / t_c,
                    "eager_gpu": g_e, "compiled_gpu": g_c,
                }
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                out[key][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            torch._dynamo.reset()
        print(key, json.dumps(out[key]), flush=True)
    return out


# ------------------------------------------------------------------ model

def setup_lang(precision):
    spec = get_spec("lang")
    model = spec.build()
    model = model.to(torch.bfloat16)
    replaced = 0
    if precision in ("fp8", "int8", "int4"):
        replaced, _ = swap_linears(model, precision)
    model.eval()
    return spec, model, replaced


def bench_model_infer(precision, compiled, bs=16, steps=15, warmup=8):
    spec, model, replaced = setup_lang(precision)
    if compiled:
        model = torch.compile(model)
        warmup = max(warmup, 3)
    batch = spec.make_batch(bs, train=False)

    def step():
        with torch.inference_mode():
            model(**batch)

    med, gpu = timeit(step, warmup, steps)
    rec = {"mode": "infer", "precision": precision, "compiled": compiled,
           "batch_size": bs, "ok": True, **gpu,
           "ms_per_iter_median": med * 1e3,
           "tokens_per_s": bs * 1024 / med, "replaced_linears": replaced}
    del model, batch
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return rec


def bench_model_decode(precision, compiled, bs, new_tokens=128):
    spec, model, replaced = setup_lang(precision)
    model.config.use_cache = True
    model.generation_config.use_cache = True
    gen_kwargs = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                      do_sample=False)
    if compiled:
        # static KV cache + cudagraph-backed forward: the launch-bound fix
        gen_kwargs["cache_implementation"] = "static"
        model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                      fullgraph=False)
    prompt = torch.randint(0, 32000, (bs, 128), device="cuda")

    def gen():
        # no_grad (not inference_mode): cudagraph pools dislike inference tensors.
        # mark_step_begin: tell the cudagraph pool a new iteration starts, else
        # the HF static cache tensors from the previous replay get flagged as
        # overwritten outputs (RuntimeError).
        if compiled:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            model.generate(prompt, **gen_kwargs)

    med, gpu = timeit(gen, 3 if compiled else 2, 5)
    rec = {"mode": f"decode_bs{bs}", "precision": precision, "compiled": compiled,
           "batch_size": bs, "ok": True, **gpu,
           "ms_per_iter_median": med * 1e3,
           "tokens_per_s": bs * new_tokens / med, "replaced_linears": replaced}
    del model, prompt
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list: gemm,infer,decode")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    def want(section):
        return only is None or section in only
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gemm": None, "model": [],
    }

    def save():
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

    if want("gemm"):
        print("== GEMM eager vs max-autotune ==", flush=True)
        result["gemm"] = gemm_section()
        save()

    print("== model-level ==", flush=True)
    for compiled in ((False, True) if want("infer") else ()):
        for prec in ("bf16", "fp8", "int8"):
            try:
                rec = bench_model_infer(prec, compiled)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rec = {"mode": "infer", "precision": prec, "compiled": compiled,
                       "ok": False, "error": f"{type(e).__name__}: {e}"}
                torch._dynamo.reset()
                torch.cuda.empty_cache()
            result["model"].append(rec)
            print(json.dumps(rec), flush=True)
            save()
    for bs in ((1, 32) if want("decode") else ()):
        for compiled in (False, True):
            for prec in ("bf16", "int8", "int4"):
                try:
                    rec = bench_model_decode(prec, compiled, bs)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    rec = {"mode": f"decode_bs{bs}", "precision": prec,
                           "compiled": compiled, "ok": False,
                           "error": f"{type(e).__name__}: {e}"}
                    torch._dynamo.reset()
                    torch.cuda.empty_cache()
                result["model"].append(rec)
                print(json.dumps(rec), flush=True)
                save()
    print("ROUTE2_DONE", flush=True)


if __name__ == "__main__":
    main()
