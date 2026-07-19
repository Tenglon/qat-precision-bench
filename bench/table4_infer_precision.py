"""Table 4: INFERENCE PRECISION as the single variable (+ logit fidelity).

Pinned: Qwen2.5-1.5B (random init, seed 17), 1 GPU, batch forward bs=16,
seq=1024, eager, SDPA.
Variable: fp32 / tf32 / bf16 / fp16 / fp8(_scaled_mm) / int8(_int_mm) /
int4(tinygemm, vocab head excluded).
Numerics: logit cosine + max-abs-rel error vs the fp32 forward on an
identical probe batch.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import GpuSampler  # noqa: E402
from models import get_spec  # noqa: E402
from quant import swap_linears  # noqa: E402

PRECISIONS = ["fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4"]


def setup(spec, precision):
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
    torch.backends.cudnn.allow_tf32 = precision != "fp32"
    model = spec.build()
    replaced = 0
    if precision in ("bf16",):
        model = model.to(torch.bfloat16)
    elif precision == "fp16":
        model = model.to(torch.float16)
    elif precision in ("fp8", "int8", "int4"):
        model = model.to(torch.bfloat16)
        replaced, _ = swap_linears(model, precision)
    model.eval()
    return model, replaced


def dtype_of(precision):
    return {"bf16": torch.bfloat16, "fp8": torch.bfloat16,
            "int8": torch.bfloat16, "int4": torch.bfloat16,
            "fp16": torch.float16}.get(precision, torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    spec = get_spec("lang")
    result = {"table": "4: inference precision",
              "pinned": "Qwen2.5-1.5B, 1xH100, batch fwd bs=16, seq=1024, eager",
              "gpu": torch.cuda.get_device_name(0), "records": []}
    ref_logits = None
    for p in PRECISIONS:
        torch.cuda.reset_peak_memory_stats()
        model, replaced = setup(spec, p)
        torch.manual_seed(23)
        batch = spec.make_batch(args.bs, train=False)
        batch = {k: (v.to(dtype_of(p)) if torch.is_floating_point(v) else v)
                 for k, v in batch.items()}

        def fwd():
            with torch.inference_mode():
                return model(**batch).logits

        for _ in range(5):
            fwd()
        torch.cuda.synchronize()
        times = []
        with GpuSampler() as g:
            for _ in range(20):
                t0 = time.perf_counter()
                fwd()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        # fidelity on one probe forward (same tokens each precision)
        torch.manual_seed(41)
        probe = spec.make_batch(1, train=False)
        with torch.inference_mode():
            lg = model(**probe).logits.float()
        if p == "fp32":
            ref_logits = lg.clone()
            cos, mre = 1.0, 0.0
        else:
            a, b = lg.flatten(), ref_logits.flatten()
            cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            mre = float(((a - b).abs() / (b.abs() + 1e-3)).mean())
        rec = {"precision": p, "ok": True, "replaced_linears": replaced,
               "ms_per_iter_median": med * 1e3,
               "tokens_per_s": args.bs * 1024 / med,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
               "logit_cos_vs_fp32": cos, "logit_mean_rel_err": mre,
               **g.stats()}
        del model, batch, lg
        torch.cuda.empty_cache()
        result["records"].append(rec)
        print(json.dumps(rec), flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print("TABLE4_DONE", flush=True)


if __name__ == "__main__":
    main()
