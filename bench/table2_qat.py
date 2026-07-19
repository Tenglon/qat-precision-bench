"""Table 2: QAT FAKE-QUANT SCHEME as the single variable.

Pinned: Qwen2.5-1.5B (random init, seed 17), 1 GPU, bs=4, seq=1024,
AdamW (fp32 states, lr=1e-5), eager, SDPA, bf16-autocast base,
identical init & data.
Variable: none (bf16 baseline) / fp8_qat / int8_qat / int4_qat
          (straight-through fake quant on every eligible nn.Linear).

Numerics: cos / rel-err of the 30-step weight update ΔW vs the *bf16
baseline* run — isolating what QAT itself does to the training update,
independent of bf16's own error (Table 1 covers bf16 vs fp32).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import GpuSampler  # noqa: E402
from models import get_spec  # noqa: E402
from quant import swap_linears  # noqa: E402

SCHEMES = ["none", "fp8_qat", "int8_qat", "int4_qat"]


def run(scheme, spec, steps_timed=20, steps_numerics=30, bs=4):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    model = spec.build()
    replaced = 0
    if scheme != "none":
        replaced, _ = swap_linears(model, scheme)
    w_init = {n: p.detach().float().cpu().clone()
              for n, p in model.named_parameters()}
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
    torch.manual_seed(23)
    batch = spec.make_batch(bs, train=True)
    losses = []

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        opt.step()
        losses.append(round(float(loss.detach()), 4))

    for _ in range(steps_numerics):
        step()
    dW = {n: (p.detach().float().cpu() - w_init[n])
          for n, p in model.named_parameters()}

    times = []
    with GpuSampler() as g:
        for _ in range(steps_timed):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    rec = {
        "scheme": scheme, "ok": True, "replaced_linears": replaced,
        "ms_per_step_median": med * 1e3,
        "tokens_per_s": bs * 1024 / med,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss_first": losses[0], "loss_step30": losses[29],
        **g.stats(),
    }
    del model, opt, w_init, batch
    torch.cuda.empty_cache()
    return rec, dW


def fidelity(dW, ref):
    dot = nrm = nrm_ref = diff = 0.0
    for n, a in dW.items():
        b = ref[n]
        dot += float((a * b).sum())
        nrm += float((a * a).sum())
        nrm_ref += float((b * b).sum())
        diff += float(((a - b) ** 2).sum())
    return {
        "update_cos_vs_bf16": dot / math.sqrt(nrm * nrm_ref + 1e-30),
        "update_rel_err_vs_bf16": math.sqrt(diff / (nrm_ref + 1e-30)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = get_spec("lang")
    result = {"table": "2: QAT fake-quant scheme",
              "pinned": "Qwen2.5-1.5B, 1xH100, bs=4, seq=1024, AdamW fp32-states, "
                        "eager, bf16-autocast base",
              "gpu": torch.cuda.get_device_name(0), "records": []}
    ref = None
    for s in SCHEMES:
        rec, dW = run(s, spec)
        if s == "none":
            ref = dW
            rec.update({"update_cos_vs_bf16": 1.0, "update_rel_err_vs_bf16": 0.0})
        else:
            rec.update(fidelity(dW, ref))
        del dW
        torch.cuda.empty_cache()
        result["records"].append(rec)
        print(json.dumps(rec), flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print("TABLE2_DONE", flush=True)


if __name__ == "__main__":
    main()
