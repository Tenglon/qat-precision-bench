"""Route 1: torchao quantize_ + torch.compile inside plain PyTorch.

Variants (each in its own subprocess, fresh model load):
  bf16      : compiled baseline
  int8da    : Int8 dynamic activation + Int8 weight (W8A8, Triton fused)
  int4wo    : Int4 weight-only, group=128 (tinygemm / marlin-style path)
  fp8dyn    : FP8 dynamic activation + FP8 weight (needs SM89+)

Protocol mirrors the eager benchmark on the same real checkpoint:
  infer  : batch forward bs=16, seq=1024
  decode : bs=1 / bs=32, 128-token prompt + 128 new, static cache +
           mode=reduce-overhead (CUDA graphs)

  python route1_torchao_bench.py --model $Q/models/Qwen2.5-1.5B-Instruct --out out/route1.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

VARIANTS = ["bf16", "int8da", "int4wo", "fp8dyn"]


def load_model(path):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda().eval()
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model


def apply_torchao(model, variant):
    if variant == "bf16":
        return
    from torchao.quantization import quantize_
    try:  # new-style config objects (torchao >= 0.7)
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Int4WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig)
        try:  # version=1: legacy tinygemm packing, no mslk kernel dep
            int4 = Int4WeightOnlyConfig(group_size=128, version=1)
        except TypeError:
            int4 = Int4WeightOnlyConfig(group_size=128)
        cfgs = {"int8da": Int8DynamicActivationInt8WeightConfig(),
                "int4wo": int4,
                "fp8dyn": Float8DynamicActivationFloat8WeightConfig()}
    except ImportError:  # older functional API
        from torchao.quantization import (
            float8_dynamic_activation_float8_weight, int4_weight_only,
            int8_dynamic_activation_int8_weight)
        cfgs = {"int8da": int8_dynamic_activation_int8_weight(),
                "int4wo": int4_weight_only(group_size=128),
                "fp8dyn": float8_dynamic_activation_float8_weight()}
    quantize_(model, cfgs[variant])


def timeit(fn, warmup, iters):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gpuutil import GpuSampler
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
    return statistics.median(times), g.stats()


def run_variant(variant, model_path, scale, emit):
    import torch

    # ---- batch forward
    model = load_model(model_path)
    apply_torchao(model, variant)
    fwd = torch.compile(model)
    ids = torch.randint(0, 32000, (16, 1024), device="cuda")
    mask = torch.ones_like(ids)

    def step():
        with torch.inference_mode():
            fwd(input_ids=ids, attention_mask=mask)

    med, gpu = timeit(step, 8, 15)
    emit({"mode": "infer_bs16", "variant": variant, "scale_b": scale, "ok": True,
          "ms": med * 1e3, "tokens_per_s": 16 * 1024 / med, **gpu})
    del fwd, ids, mask
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ---- decode with static cache + compiled forward. NOTE: default mode,
    # not reduce-overhead — transformers 5.14 StaticCache mutates
    # cumulative_length across iterations, which cudagraph pools reject.
    model.forward = torch.compile(model.forward)
    for bs in (1, 32):
        prompt = torch.randint(0, 32000, (bs, 128), device="cuda")

        def gen():
            with torch.no_grad():
                model.generate(prompt, max_new_tokens=128, min_new_tokens=128,
                               do_sample=False, cache_implementation="static")

        med, gpu = timeit(gen, 3, 5)
        emit({"mode": f"decode_bs{bs}", "variant": variant, "scale_b": scale, "ok": True,
              "ms": med * 1e3, "tokens_per_s": bs * 128 / med, **gpu})
        del prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--one", default=None)
    ap.add_argument("--scale", default="1.5")
    ap.add_argument("--scales", default="0.5,1.5,3,7")
    args = ap.parse_args()

    if args.one:
        try:
            path = os.path.join(args.models_dir,
                                f"Qwen2.5-{args.scale}B-Instruct")
            run_variant(args.one, path, args.scale,
                        lambda r: print("REC " + json.dumps(r), flush=True))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            sys.exit(1)
        return

    result = {"stack": "torchao+compile", "records": []}
    try:
        import torchao
        import torch
        result["torchao_version"] = torchao.__version__
        result["torch_version"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    for scale in args.scales.split(","):
        for v in VARIANTS:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--models-dir",
                 args.models_dir, "--out", args.out, "--one", v,
                 "--scale", scale],
                capture_output=True, text=True, timeout=2400)
            got = False
            for line in p.stdout.splitlines():
                if line.startswith("REC "):
                    result["records"].append(json.loads(line[4:]))
                    got = True
            if not got:
                tail = (p.stdout + "\n" + p.stderr)[-2000:]
                result["records"].append({"variant": v, "scale_b": scale,
                                          "ok": False, "error": tail})
            print(f"== {scale}B {v} done (ok={got})", flush=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
    print("ROUTE1_DONE", flush=True)


if __name__ == "__main__":
    main()
