"""Can a 7B model be full-parameter-trained on one 64 GB H100?

Memory ledger experiment for the OOM wall found in the scale sweep. Variants
(each in a fresh CUDA state; OOMs are recorded with the allocator's numbers):

  adamw_fp32      : fp32 weights/grads + AdamW      (~126 GB predicted, OOM)
  muon_fp32       : fp32 weights/grads + Muon+AdamW (~99 GB predicted, OOM)
  adamw_pure_bf16 : bf16 weights/grads/moments      (~65 GB predicted,边缘)
  muon_pure_bf16  : bf16 weights/grads/momentum     (~52 GB predicted, fits)

Records peak memory, throughput, and 20-step loss trajectory (sanity that the
optimizer actually optimizes).

  python mem7b_bench.py --out out/mem7b.json [--modality lang7]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import GpuSampler  # noqa: E402
from models import get_spec  # noqa: E402
from muon import build_muon_adamw  # noqa: E402

# *_fp32  : fp32 storage + bf16-autocast matmuls (standard mixed precision)
# *_tf32  : fp32 storage, NO autocast, TF32 tensor-core matmuls
#           (memory-identical to fp32 — a compute knob, not a memory knob)
# *_pure_bf16 : bf16 storage/grads/optimizer state
VARIANTS = ["adamw_fp32", "muon_fp32", "adamw_tf32", "muon_tf32",
            "adamw_pure_bf16", "muon_pure_bf16"]

# torchao low-bit optimizer states (quantized m/v), paired with pure-bf16
# weights/grads — run via --variants on the venv with torchao installed
LOWBIT_VARIANTS = ["adamw8bit_pure_bf16", "adamw4bit_pure_bf16",
                   "adamwfp8_pure_bf16", "adamw_pure_bf16", "muon_pure_bf16"]


def setup(spec, variant):
    tf32 = variant.endswith("_tf32") or variant.endswith("pure_bf16")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    model = spec.build()  # fp32 on cuda
    if variant.endswith("pure_bf16"):
        model = model.to(torch.bfloat16)
    model.train()
    if variant.startswith("adamw8bit") or variant.startswith("adamw4bit") \
            or variant.startswith("adamwfp8"):
        try:
            from torchao.optim import AdamW4bit, AdamW8bit, AdamWFp8
        except ImportError:
            from torchao.prototype.low_bit_optim import (AdamW4bit, AdamW8bit,
                                                         AdamWFp8)
        cls = {"adamw8bit": AdamW8bit, "adamw4bit": AdamW4bit,
               "adamwfp8": AdamWFp8}[variant.split("_")[0]]
        opts = [cls(model.parameters(), lr=1e-5)]
    elif variant.startswith("adamw"):
        opts = [torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)]
    else:
        opts = list(build_muon_adamw(model, muon_lr=5e-4, adamw_lr=1e-5))
    return model, opts


def run_variant(spec, variant, bs, steps=20, warmup=3):
    model, opts = setup(spec, variant)
    batch = spec.make_batch(bs, train=True)
    losses = []

    def step():
        for o in opts:
            o.zero_grad(set_to_none=True)
        if variant.endswith("_fp32"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
        else:  # _tf32 and _pure_bf16 run without autocast
            loss = model(**batch).loss
        loss.backward()
        for o in opts:
            o.step()
        losses.append(float(loss.detach()))

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    times = []
    with GpuSampler() as g:
        for _ in range(steps):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    rec = {
        "variant": variant, "ok": True, "batch_size": bs,
        "ms_per_step_median": med * 1e3,
        "tokens_per_s": bs * spec.tokens_per_sample / med,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss_first": losses[0], "loss_last": losses[-1],
        **g.stats(),
    }
    del model, opts, batch
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modality", default="lang7")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--variants", default="",
                    help="'lowbit' for torchao low-bit optimizer set")
    args = ap.parse_args()
    global VARIANTS
    if args.variants == "lowbit":
        VARIANTS = LOWBIT_VARIANTS
    spec = get_spec(args.modality)
    dev = torch.cuda.get_device_properties(0)
    result = {"model": spec.desc, "gpu": torch.cuda.get_device_name(0),
              "gpu_mem_gib": dev.total_memory / 2**30, "records": []}
    for v in VARIANTS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            rec = run_variant(spec, v, args.bs)
        except torch.OutOfMemoryError as e:
            msg = str(e)
            tried = re.search(r"Tried to allocate ([0-9.]+ [MG]iB)", msg)
            rec = {"variant": v, "ok": False, "oom": True,
                   "alloc_at_oom_gib": torch.cuda.memory_allocated() / 2**30,
                   "reserved_at_oom_gib": torch.cuda.memory_reserved() / 2**30,
                   "tried_to_allocate": tried.group(1) if tried else None,
                   "error": msg.split(".")[0]}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rec = {"variant": v, "ok": False, "oom": False,
                   "error": f"{type(e).__name__}: {e}"}
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        result["records"].append(rec)
        print(json.dumps(rec), flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print("MEM7B_DONE", flush=True)


if __name__ == "__main__":
    main()
