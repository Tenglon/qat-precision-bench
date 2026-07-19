"""Table 5: FUSION LEVEL as the single variable.

Pinned: Qwen2.5-1.5B (random init, seed 17), 1 GPU, bf16, AdamW fp32-states,
bs=4 train / bs=16 infer, seq=1024, SDPA math semantics identical.
Variable: eager / torch.compile / torch.compile + cuDNN fused attention.

Reported for BOTH one training step and one batch forward. dynamo cache is
reset between variants (silent-eager-fallback gotcha, v1 finding).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import GpuSampler  # noqa: E402
from models import get_spec  # noqa: E402

LEVELS = ["eager", "compile", "compile_cudnn_attn"]


def attn_ctx(level):
    if level != "compile_cudnn_attn":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])


def bench(level, mode, spec):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    model = spec.build()
    if mode == "infer":
        model = model.to(torch.bfloat16).eval()
        bs = 16
    else:
        model.train()
        bs = 4
    if level.startswith("compile"):
        model = torch.compile(model)
    opt = (torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
           if mode == "train" else None)
    torch.manual_seed(23)
    batch = spec.make_batch(bs, train=(mode == "train"))
    if mode == "infer":
        batch = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
                 for k, v in batch.items()}

    def step():
        with attn_ctx(level):
            if mode == "train":
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                loss.backward()
                opt.step()
            else:
                with torch.inference_mode():
                    model(**batch)

    for _ in range(8):
        step()
    torch.cuda.synchronize()
    times = []
    with GpuSampler() as g:
        for _ in range(20):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    rec = {"level": level, "mode": mode, "ok": True, "batch_size": bs,
           "ms_median": med * 1e3, "tokens_per_s": bs * 1024 / med,
           "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
           **g.stats()}
    del model, opt, batch
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = get_spec("lang")
    result = {"table": "5: fusion level",
              "pinned": "Qwen2.5-1.5B, 1xH100, bf16, train bs=4 / infer bs=16",
              "records": []}
    for mode in ("train", "infer"):
        for level in LEVELS:
            try:
                rec = bench(level, mode, spec)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                rec = {"level": level, "mode": mode, "ok": False,
                       "error": f"{type(e).__name__}: {e}"}
                torch._dynamo.reset()
                torch.cuda.empty_cache()
            result["records"].append(rec)
            print(json.dumps(rec), flush=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
    print("TABLE5_DONE", flush=True)


if __name__ == "__main__":
    main()
