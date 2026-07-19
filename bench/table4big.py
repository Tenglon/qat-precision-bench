"""Table 4 big-model series: ONE precision per job, fused, uniform GPU count
per scale (the no-OOM rule doubles GPUs for the WHOLE table, keyed by the
fp32 row which is always the memory high-water mark).

Multi-GPU memory scaling uses accelerate's dispatch_model (layer-wise
pipeline placement) — same kernels, weights simply spread across GPUs.
The fp32 job also writes the reference probe logits to disk; other-precision
jobs load them for the fidelity column (submit with --dependency=afterok).

  python table4big.py --modality lang7 --precision fp8 --ngpu 1 --fused \
      --ref $Q/out/ref_lang7.pt --out $Q/out/table4_lang7_fp8.json
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

DTYPE = {"bf16": torch.bfloat16, "fp8": torch.bfloat16, "int8": torch.bfloat16,
         "int4": torch.bfloat16, "fp16": torch.float16}


def build(spec, precision, ngpu, modality):
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
    torch.backends.cudnn.allow_tf32 = precision != "fp32"
    import models as _models
    on_cpu = ngpu > 1
    if on_cpu:
        _models.DEVICE = "cpu"
    model = spec.build()
    if on_cpu:
        _models.DEVICE = "cuda"
    if precision in ("bf16", "fp8", "int8", "int4"):
        model = model.to(torch.bfloat16)
    elif precision == "fp16":
        model = model.to(torch.float16)
    model.eval()
    replaced = 0
    if ngpu == 1:
        model = model.cuda()
        if precision in ("fp8", "int8", "int4"):
            replaced, _ = swap_linears(model, precision)
    else:
        from accelerate import dispatch_model, infer_auto_device_map
        mem = {i: "52GiB" for i in range(ngpu)}
        dmap = infer_auto_device_map(
            model, max_memory=mem,
            no_split_module_classes=["Qwen2DecoderLayer"])
        model = dispatch_model(model, dmap)
        if precision in ("fp8", "int8", "int4"):
            replaced, _ = swap_linears(model, precision)  # per-device weights
    return model, replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True)
    ap.add_argument("--precision", required=True)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--ref", required=True, help="path of fp32 probe logits .pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = args.precision
    spec = get_spec(args.modality)
    torch.cuda.reset_peak_memory_stats()
    model, replaced = build(spec, p, args.ngpu, args.modality)
    if args.fused and p != "fp32":
        model = torch.compile(model)

    torch.manual_seed(23)
    batch = spec.make_batch(args.bs, train=False)
    batch = {k: (v.to(DTYPE.get(p, torch.float32)) if torch.is_floating_point(v) else v)
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

    torch.manual_seed(41)
    probe = spec.make_batch(1, train=False)
    with torch.inference_mode():
        lg = model(**probe).logits.float().cpu()
    if p == "fp32":
        torch.save(lg, args.ref)
        cos, mre = 1.0, 0.0
    else:
        ref = torch.load(args.ref).flatten()
        a = lg.flatten()
        cos = float(torch.nn.functional.cosine_similarity(a, ref, dim=0))
        mre = float(((a - ref).abs() / (ref.abs() + 1e-3)).mean())
    peaks = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
             for i in range(args.ngpu)]
    rec = {"precision": p, "ok": True, "ngpu": args.ngpu, "fused": args.fused,
           "replaced_linears": replaced,
           "ms_per_iter_median": med * 1e3,
           "tokens_per_s": args.bs * 1024 / med,
           "peak_mem_gib_per_gpu": peaks,
           "logit_cos_vs_fp32": cos, "logit_mean_rel_err": mre,
           **g.stats()}
    with open(args.out, "w") as f:
        json.dump({"table": f"4-series {args.modality} fused ngpu={args.ngpu}",
                   "records": [rec]}, f, indent=2)
    print(json.dumps(rec), flush=True)
    print("TABLE4BIG_DONE", flush=True)


if __name__ == "__main__":
    main()
