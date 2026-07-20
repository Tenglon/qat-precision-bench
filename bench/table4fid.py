"""Fidelity-only recompute for the T4 family, in float64.

Each precision runs in its own SUBPROCESS: dispatched multi-GPU models leave
allocator residue across in-process iterations (measured: GPU full by the
6th rebuild), so a clean CUDA context per precision is the only robust way.
The fp32 child writes the reference logits file; later children load it.
"""
import argparse
import json
import os
import subprocess
import sys

PRECISIONS = ["fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4"]


def run_one(precision, modality, ngpu, ref_path):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import get_spec
    from table4big import build
    spec = get_spec(modality)
    model, _ = build(spec, precision, ngpu, modality)
    torch.manual_seed(41)
    probe = spec.make_batch(1, train=False)
    with torch.inference_mode():
        lg = model(**probe).logits.double().cpu().flatten()
    if precision == "fp32":
        torch.save(lg, ref_path)
        cos, mre = 1.0, 0.0
    else:
        ref = torch.load(ref_path)
        cos = float(lg @ ref / (lg.norm() * ref.norm()))
        mre = float(((lg - ref).abs() / (ref.abs() + 1e-3)).mean())
    print("REC " + json.dumps({"precision": precision,
                               "logit_cos_fp64": cos,
                               "logit_mean_rel_err": mre}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--one", default=None)
    args = ap.parse_args()
    ref_path = args.out + ".ref64.pt"
    if args.one:
        run_one(args.one, args.modality, args.ngpu, ref_path)
        return
    result = {"table": f"4-family fidelity (float64) {args.modality}",
              "records": []}
    for p in PRECISIONS:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--modality",
             args.modality, "--ngpu", str(args.ngpu), "--out", args.out,
             "--one", p], capture_output=True, text=True, timeout=3000)
        got = False
        for line in proc.stdout.splitlines():
            if line.startswith("REC "):
                result["records"].append(json.loads(line[4:]))
                got = True
        if not got:
            result["records"].append({"precision": p, "ok": False,
                                      "error": proc.stderr[-800:]})
        print(f"== {p} done ok={got}", flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    if os.path.exists(ref_path):
        os.remove(ref_path)
    print("TABLE4FID_DONE", flush=True)


if __name__ == "__main__":
    main()
