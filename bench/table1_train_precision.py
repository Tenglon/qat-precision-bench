"""Table 1: training COMPUTE PRECISION as the single variable.

Pinned: Qwen2.5-1.5B (random init, seed 17), 1 GPU, bs=4, seq=1024,
AdamW (fp32 states, lr=1e-5), eager, SDPA, identical data batches.
Variable: fp32 / tf32 / bf16-autocast / fp16-autocast(+GradScaler).

Metrics per row: median ms/step (5+20), tokens/s, speedup vs fp32,
peak GiB, GPU util/power, and NUMERICS:
  - 30-step loss trajectory (first/last)
  - update-direction fidelity vs the fp32 run over the same 30 steps:
      cos(dW_p, dW_fp32) and ||dW_p - dW_fp32|| / ||dW_fp32||
    where dW = W_after_30_steps - W_init, accumulated parameter-wise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import GpuSampler  # noqa: E402
from models import get_spec  # noqa: E402

PRECISIONS = ["fp32", "tf32", "bf16", "fp16"]


def make_ctx(precision):
    import contextlib
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def run(precision, spec, steps_timed=20, steps_numerics=30, bs=4):
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
    torch.backends.cudnn.allow_tf32 = precision != "fp32"
    torch.cuda.reset_peak_memory_stats()
    model = spec.build()                       # identical init (seed inside)
    # snapshot on CPU: keep GPU headroom for fp32 activations
    w_init = {n: p.detach().float().cpu().clone()
              for n, p in model.named_parameters()}
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    torch.manual_seed(23)
    batch = spec.make_batch(bs, train=True)    # identical batch
    ctx = make_ctx(precision)
    losses = []

    def step():
        opt.zero_grad(set_to_none=True)
        with ctx:
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(round(float(loss.detach()), 4))

    # numerics phase: 30 deterministic steps
    for _ in range(steps_numerics):
        step()
    dW = {n: (p.detach().float().cpu() - w_init[n])
          for n, p in model.named_parameters()}

    # timing phase (5 warmup naturally done; take 20 timed)
    times = []
    with GpuSampler() as g:
        for _ in range(steps_timed):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    rec = {
        "precision": precision, "ok": True,
        "ms_per_step_median": med * 1e3,
        "tokens_per_s": bs * 1024 / med,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss_first": losses[0], "loss_step30": losses[29],
        **g.stats(),
    }
    del model, opt, w_init, batch
    torch.cuda.empty_cache()
    return rec, dW


def update_fidelity(dW, dW_ref):
    dot = nrm = nrm_ref = diff = 0.0
    for n, a in dW.items():
        b = dW_ref[n]
        dot += float((a * b).sum())
        nrm += float((a * a).sum())
        nrm_ref += float((b * b).sum())
        diff += float(((a - b) ** 2).sum())
    import math
    return {
        "update_cos_vs_fp32": dot / math.sqrt(nrm * nrm_ref + 1e-30),
        "update_rel_err_vs_fp32": math.sqrt(diff / (nrm_ref + 1e-30)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = get_spec("lang")
    result = {"table": "1: training compute precision",
              "pinned": "Qwen2.5-1.5B, 1xH100, bs=4, seq=1024, AdamW fp32-states, eager",
              "gpu": torch.cuda.get_device_name(0), "records": []}
    ref_dW = None
    for p in PRECISIONS:
        rec, dW = run(p, spec)
        if p == "fp32":
            ref_dW = dW
            rec.update({"update_cos_vs_fp32": 1.0, "update_rel_err_vs_fp32": 0.0})
        else:
            rec.update(update_fidelity(dW, ref_dW))
        del dW
        torch.cuda.empty_cache()
        result["records"].append(rec)
        print(json.dumps(rec), flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print("TABLE1_DONE", flush=True)


if __name__ == "__main__":
    main()
