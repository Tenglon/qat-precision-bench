"""End-to-end precision benchmark for one modality.

Measures, on a single GPU, relative to FP32:
  train : optimizer-step throughput for fp32 / tf32 / bf16 / fp16 mixed
          precision, real-FP8 GEMM training, and int8/int4/fp8 QAT
          (fake-quant) training overhead
  infer : batch forward throughput for fp32 / tf32 / bf16 / fp16 and real
          quantized fp8 (_scaled_mm), int8 (_int_mm), int4 (tinygemm) linears
  decode: (lang only) autoregressive generation tokens/s at bs=1 and bs=32

Usage:
  python run_bench.py --modality lang --out out/lang.json
"""

from __future__ import annotations

import argparse
import contextlib
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
from quant import swap_linears  # noqa: E402

TRAIN_PRECISIONS = ["fp32", "tf32", "bf16", "fp16",
                    "fp8_train", "fp8_qat", "int8_qat", "int4_qat"]
INFER_PRECISIONS = ["fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4"]
DECODE_PRECISIONS = ["fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4"]


def set_tf32(on: bool):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def fused_ctx(fused, precision):
    """Best-effort fused attention: cuDNN SDPA (FA3-class on Hopper) with
    graceful fallback for dtypes it rejects (fp32/tf32 attention)."""
    if not fused or precision in ("fp32",):
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    try:
        return sdpa_kernel(backends, set_priority=True)
    except TypeError:  # older signature
        return sdpa_kernel(backends)


def maybe_compile(model, fused, precision):
    if fused and precision != "fp32":
        return torch.compile(model)
    return model


def autocast_for(precision):
    if precision in ("bf16", "fp8_train", "fp8_qat", "int8_qat", "int4_qat"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def cast_batch(batch, dtype):
    return {k: (v.to(dtype) if torch.is_floating_point(v) else v)
            for k, v in batch.items()}


def timed(fn, warmup, iters):
    """Times ONLY the steady-state loop; samples GPU util inside the window."""
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
    return times, g.stats()


def free_cuda(*objs):
    import gc
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ------------------------------------------------------------------ train

def bench_train(spec, precision, bs, steps, warmup, fused=False):
    set_tf32(precision != "fp32")
    model = spec.build()
    replaced = skipped = 0
    if precision in ("fp8_train", "fp8_qat", "int8_qat", "int4_qat"):
        replaced, skipped = swap_linears(model, precision)
    model.train()
    model = maybe_compile(model, fused, precision)
    model.gradient_checkpointing_disable() if hasattr(model, "gradient_checkpointing_disable") else None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    batch = spec.make_batch(bs, train=True)
    ctx = autocast_for(precision)
    losses = []

    def step():
        opt.zero_grad(set_to_none=True)
        # fused_ctx must be re-created per step: sdpa_kernel returns a
        # single-use generator context manager
        with fused_ctx(fused, precision), ctx:
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))

    times, gpu = timed(step, warmup, steps)
    med = statistics.median(times)
    rec = {
        "mode": "train", "precision": precision, "batch_size": bs, "ok": True,
        "fused": fused, **gpu,
        "ms_per_iter_median": med * 1e3,
        "ms_per_iter_mean": statistics.mean(times) * 1e3,
        "samples_per_s": bs / med,
        "tokens_per_s": bs * spec.tokens_per_sample / med if spec.tokens_per_sample else None,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "replaced_linears": replaced, "skipped_linears": skipped,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    free_cuda(model, opt, batch)
    return rec


# ------------------------------------------------------------------ infer

def setup_infer_model(spec, precision):
    set_tf32(precision != "fp32")
    model = spec.build()
    replaced = skipped = 0
    if precision in ("bf16",):
        model = model.to(torch.bfloat16)
    elif precision == "fp16":
        model = model.to(torch.float16)
    elif precision in ("fp8", "int8", "int4"):
        model = model.to(torch.bfloat16)
        replaced, skipped = swap_linears(model, precision)
    model.eval()
    return model, replaced, skipped


def infer_dtype(precision):
    if precision in ("bf16", "fp8", "int8", "int4"):
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def bench_infer(spec, precision, bs, steps, warmup, ref_logits, fused=False):
    model, replaced, skipped = setup_infer_model(spec, precision)
    model = maybe_compile(model, fused, precision)
    batch = cast_batch(spec.make_batch(bs, train=False), infer_dtype(precision))

    def step():
        with fused_ctx(fused, precision), torch.inference_mode():
            model(**batch)

    times, gpu = timed(step, warmup, steps)
    med = statistics.median(times)

    # quality probe vs fp32 logits on a tiny fixed batch
    cos = None
    if ref_logits is not None:
        torch.manual_seed(41)
        probe = cast_batch(spec.make_batch(1, train=False), infer_dtype(precision))
        with torch.inference_mode():
            lg = model(**probe).logits.float().flatten()
        cos = float(torch.nn.functional.cosine_similarity(
            lg, ref_logits.flatten(), dim=0))
    rec = {
        "mode": "infer", "precision": precision, "batch_size": bs, "ok": True,
        "fused": fused, **gpu,
        "ms_per_iter_median": med * 1e3,
        "ms_per_iter_mean": statistics.mean(times) * 1e3,
        "samples_per_s": bs / med,
        "tokens_per_s": bs * spec.tokens_per_sample / med if spec.tokens_per_sample else None,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "replaced_linears": replaced, "skipped_linears": skipped,
        "logit_cos_vs_fp32": cos,
    }
    free_cuda(model, batch)
    return rec


def fp32_ref_logits(spec):
    set_tf32(False)
    model = spec.build()
    model.eval()
    torch.manual_seed(41)
    probe = spec.make_batch(1, train=False)
    with torch.inference_mode():
        lg = model(**probe).logits.float().clone()
    free_cuda(model, probe)
    return lg


# ------------------------------------------------------------------ decode

def bench_decode(spec, precision, bs, new_tokens=128):
    model, replaced, skipped = setup_infer_model(spec, precision)
    model.config.use_cache = True
    model.generation_config.use_cache = True
    prompt = torch.randint(0, 32000, (bs, 128), device="cuda")

    def gen():
        with torch.inference_mode():
            model.generate(prompt, max_new_tokens=new_tokens,
                           min_new_tokens=new_tokens, do_sample=False)

    times, gpu = timed(gen, 2, 5)
    med = statistics.median(times)
    rec = {
        "mode": f"decode_bs{bs}", "precision": precision, "batch_size": bs,
        "ok": True,
        **gpu,
        "ms_per_iter_median": med * 1e3,
        "tokens_per_s": bs * new_tokens / med,
        "samples_per_s": bs / med,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "replaced_linears": replaced, "skipped_linears": skipped,
    }
    free_cuda(model, prompt)
    return rec


# ------------------------------------------------------------------ driver

def find_train_bs(spec, start_bs, steps=3):
    """fp32 + AdamW is the memory high-water mark; shrink bs until it fits."""
    bs = start_bs
    while bs >= 1:
        try:
            bench_train(spec, "fp32", bs, steps=steps, warmup=1)
            return bs
        except torch.OutOfMemoryError:
            free_cuda()
            bs //= 2
    raise RuntimeError("even bs=1 OOMs in fp32")


def selftest():
    """Numerical sanity of each quantized linear vs an fp32 reference."""
    from quant import (Fp8InferLinear, Fp8TrainLinear, Int4InferLinear,
                       Int8InferLinear, QATLinear)
    torch.manual_seed(3)
    lin = torch.nn.Linear(1024, 2048, bias=True).cuda()
    x = torch.randn(64, 1024, device="cuda")
    ref = lin(x)
    out = {}
    for name, mod in [
        ("fp8", Fp8InferLinear(lin)), ("int8", Int8InferLinear(lin)),
        ("int4", Int4InferLinear(lin)), ("fp8_train", Fp8TrainLinear(lin)),
        ("int4_qat", QATLinear(lin, 4, 8)),
    ]:
        try:
            y = mod(x.clone()).float()
            out[name] = float(torch.nn.functional.cosine_similarity(
                y.flatten(), ref.flatten(), dim=0))
        except Exception as e:  # noqa: BLE001
            out[name] = f"FAILED: {type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True,
                    choices=["lang", "image", "video", "audio", "mm",
                             "lang05", "lang3", "lang7"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--train-bs", type=int, default=0)
    ap.add_argument("--infer-bs", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-infer", action="store_true")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--fused", action="store_true",
                    help="all precisions except fp32: torch.compile + cuDNN "
                         "fused attention (FA3-class on Hopper)")
    args = ap.parse_args()

    spec = get_spec(args.modality)
    print(f"== {spec.name}: {spec.desc}", flush=True)
    dev = torch.cuda.get_device_properties(0)
    header = {
        "modality": spec.name, "model": spec.desc,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_gib": dev.total_memory / 2**30,
        "capability": f"{dev.major}.{dev.minor}",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "selftest_cos_vs_fp32": selftest(),
    }
    print(json.dumps(header, indent=2), flush=True)

    records = []

    def run(tag, fn, *a, **kw):
        t0 = time.time()
        try:
            rec = fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rec = {"mode": tag[0], "precision": tag[1], "ok": False,
                   "error": f"{type(e).__name__}: {e}"}
            free_cuda()
        rec["wall_s"] = time.time() - t0
        # each variant gets a fresh dynamo cache: the per-code-object cache
        # limit (8) otherwise silently exhausts across variants and later
        # torch.compile calls fall back to eager
        try:
            torch._dynamo.reset()
        except Exception:  # noqa: BLE001
            pass
        records.append(rec)
        print(json.dumps(rec), flush=True)
        # incremental save so a walltime kill loses nothing
        with open(args.out, "w") as f:
            json.dump({"header": header, "records": records}, f, indent=2)

    if not args.skip_train:
        if args.train_bs:
            train_bs = args.train_bs
        else:
            try:
                train_bs = find_train_bs(spec, spec.train_bs)
            except (RuntimeError, torch.OutOfMemoryError):
                # optimizer state alone may exceed the GPU (e.g. 7B + AdamW);
                # keep going — each precision will record its own OOM
                free_cuda()
                train_bs = spec.train_bs
        print(f"== train batch size: {train_bs}", flush=True)
        for p in TRAIN_PRECISIONS:
            run(("train", p), bench_train, spec, p, train_bs,
                args.steps, args.warmup, fused=args.fused)

    if not args.skip_infer:
        infer_bs = args.infer_bs or spec.infer_bs
        ref = fp32_ref_logits(spec)
        for p in INFER_PRECISIONS:
            run(("infer", p), bench_infer, spec, p, infer_bs,
                args.steps, args.warmup, ref, fused=args.fused)
        free_cuda(ref)

    if spec.supports_decode and not args.skip_decode:
        for bs in (1, 32):
            for p in DECODE_PRECISIONS:
                run((f"decode_bs{bs}", p), bench_decode, spec, p, bs)

    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
