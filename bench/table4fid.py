"""Fidelity-only recompute for the T4 family, in float64.

fp32 reductions over ~1.5e8-element logit tensors inflate cosine by up to
~4% (measured on this cluster) — so every fidelity number is recomputed here
with float64 dot/norms. One job per scale; loops all precisions; no timing.
"""
import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_spec  # noqa: E402
from quant import swap_linears  # noqa: E402
from table4big import build  # noqa: E402  (reuses dispatch logic)

PRECISIONS = ["fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = get_spec(args.modality)
    result = {"table": f"4-family fidelity (float64) {args.modality}",
              "records": []}
    ref = None
    for p in PRECISIONS:
        model, replaced = build(spec, p, args.ngpu, args.modality)
        torch.manual_seed(41)
        probe = spec.make_batch(1, train=False)
        with torch.inference_mode():
            lg = model(**probe).logits.double().cpu().flatten()
        if p == "fp32":
            ref = lg
            cos, mre = 1.0, 0.0
        else:
            cos = float(lg @ ref / (lg.norm() * ref.norm()))
            mre = float(((lg - ref).abs() / (ref.abs() + 1e-3)).mean())
        result["records"].append({"precision": p, "logit_cos_fp64": cos,
                                  "logit_mean_rel_err": mre})
        print(json.dumps(result["records"][-1]), flush=True)
        del model, lg
        torch.cuda.empty_cache()
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print("TABLE4FID_DONE", flush=True)


if __name__ == "__main__":
    main()
