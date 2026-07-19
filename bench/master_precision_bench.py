"""Why trainable MASTER weights resist FP8/INT8/INT4 storage.

After every optimizer step, re-quantize the linear weights in place —
simulating "the weights themselves are stored in the target precision".
If the optimizer update (~lr) is smaller than the quantization step,
round-to-nearest (RTN) erases it and the loss stops moving; stochastic
rounding (SR) preserves updates in expectation at the cost of noise.

This is a CONVERGENCE demonstration (memory is not actually saved here);
it explains why every practical recipe keeps a bf16/fp32 master copy.

  python master_precision_bench.py --out out/master_prec.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_spec  # noqa: E402

VARIANTS = ["bf16_master", "fp8_master_rtn", "int8_master_rtn",
            "int8_master_sr", "int4_master_rtn", "int4_master_sr"]


def qdq_weights(model, kind):
    """In-place quantize->dequantize of every big linear weight."""
    if kind == "bf16":
        return
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and min(m.weight.shape) >= 256:
                w = m.weight
                if kind.startswith("fp8"):
                    scale = w.abs().amax().clamp(min=1e-12) / 448.0
                    w.copy_((w / scale).clamp(-448, 448)
                            .to(torch.float8_e4m3fn).to(w.dtype) * scale)
                    continue
                bits = 8 if kind.startswith("int8") else 4
                qmax = 2 ** (bits - 1) - 1
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
                x = w / scale
                if kind.endswith("sr"):  # stochastic rounding
                    q = torch.floor(x + torch.rand_like(x))
                else:                    # round-to-nearest
                    q = torch.round(x)
                w.copy_(q.clamp(-qmax - 1, qmax) * scale)


def run_variant(spec, variant, bs=4, steps=30):
    kind = variant.split("_master")[0] + (
        "_sr" if variant.endswith("_sr") else ("_rtn" if variant.endswith("_rtn") else ""))
    torch.manual_seed(5)
    model = spec.build().to(torch.bfloat16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
    qdq_weights(model, kind)  # start from stored-precision weights
    batch = spec.make_batch(bs, train=True)
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        qdq_weights(model, kind)  # weights live in the target precision
        losses.append(round(float(loss.detach()), 4))
    torch.cuda.synchronize()
    rec = {"variant": variant, "ok": True, "steps": steps,
           "wall_s": time.perf_counter() - t0,
           "losses": losses,
           "loss_drop": losses[0] - losses[-1]}
    del model, opt, batch
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modality", default="lang")
    args = ap.parse_args()
    spec = get_spec(args.modality)
    result = {"model": spec.desc, "records": []}
    for v in VARIANTS:
        try:
            rec = run_variant(spec, v)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rec = {"variant": v, "ok": False, "error": f"{type(e).__name__}: {e}"}
        result["records"].append(rec)
        print(json.dumps(rec), flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print("MASTER_PREC_DONE", flush=True)


if __name__ == "__main__":
    main()
