"""Route 3: vLLM serving stack — what do production kernels + CUDA graphs give?

Variants (each in its own subprocess; vLLM engines don't like sharing a GPU):
  bf16 : Qwen2.5-1.5B-Instruct, dtype=bfloat16
  fp8  : same checkpoint, quantization="fp8" (data-free, cutlass scaled-mm)
  int8 : official GPTQ-Int8 checkpoint (W8A16, gptq_marlin kernel)
  int4 : official AWQ checkpoint (W4A16, awq_marlin kernel)

Protocol mirrors the eager benchmark:
  prefill : 16 prompts x 1024 random tokens, max_tokens=1  -> prompt tok/s
  decode  : bs=1 and bs=32, 128-token prompt + 128 new     -> gen tok/s

  python route3_vllm_bench.py --models-dir $Q/models --out out/route3.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time

VARIANTS = ["bf16", "fp8", "int8", "int4"]


def build_llm(variant, models_dir, scale, enforce_eager=False):
    from vllm import LLM
    common = dict(gpu_memory_utilization=0.85, max_model_len=2304,
                  disable_log_stats=True, trust_remote_code=False,
                  enforce_eager=enforce_eager)
    base = os.path.join(models_dir, f"Qwen2.5-{scale}B-Instruct")
    if variant == "bf16":
        return LLM(model=base, dtype="bfloat16", **common)
    if variant == "fp8":
        return LLM(model=base, dtype="bfloat16", quantization="fp8", **common)
    if variant == "int8":
        return LLM(model=base + "-GPTQ-Int8", **common)
    if variant == "int4":
        return LLM(model=base + "-AWQ", **common)
    raise ValueError(variant)


def token_prompts(n, length, vocab_hi=32000, seed=13):
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ids = [rng.randrange(1, vocab_hi) for _ in range(length)]
        try:
            from vllm.inputs import TokensPrompt
            out.append(TokensPrompt(prompt_token_ids=ids))
        except Exception:  # noqa: BLE001
            out.append({"prompt_token_ids": ids})
    return out


def timed_generate(llm, prompts, sp, repeats=3):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gpuutil import GpuSampler
    times = []
    with GpuSampler() as g:
        for _ in range(repeats):
            t0 = time.perf_counter()
            llm.generate(prompts, sp, use_tqdm=False)
            times.append(time.perf_counter() - t0)
    return statistics.median(times), g.stats()


def run_variant(variant, models_dir, scale):
    from vllm import SamplingParams
    try:
        llm = build_llm(variant, models_dir, scale)
        eager = False
    except Exception:  # noqa: BLE001 — retry without vLLM-side torch.compile
        import traceback
        traceback.print_exc()
        print("RETRYING with enforce_eager=True", flush=True)
        llm = build_llm(variant, models_dir, scale, enforce_eager=True)
        eager = True
    recs = []

    # prefill throughput
    sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    pre = token_prompts(16, 1024)
    timed_generate(llm, pre, sp1, repeats=1)          # warmup
    t, gpu = timed_generate(llm, pre, sp1)
    recs.append({"mode": "prefill_bs16", "variant": variant, "scale_b": scale, "ok": True,
                 "enforce_eager": eager, **gpu,
                 "ms": t * 1e3, "tokens_per_s": 16 * 1024 / t})

    # decode throughput (prompt 128 + 128 new, matches the eager protocol)
    sp = SamplingParams(max_tokens=128, min_tokens=128, temperature=0,
                        ignore_eos=True)
    for bs in (1, 32):
        ps = token_prompts(bs, 128)
        timed_generate(llm, ps, sp, repeats=1)        # warmup
        t, gpu = timed_generate(llm, ps, sp)
        recs.append({"mode": f"decode_bs{bs}", "variant": variant, "scale_b": scale, "ok": True,
                     "enforce_eager": eager, **gpu,
                     "ms": t * 1e3, "tokens_per_s": bs * 128 / t})
    del llm
    gc.collect()
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--one", default=None)
    ap.add_argument("--scale", default="1.5")
    ap.add_argument("--scales", default="0.5,1.5,3,7")
    args = ap.parse_args()

    if args.one:  # child mode: run one variant, print records as JSON lines
        for rec in run_variant(args.one, args.models_dir, args.scale):
            print("REC " + json.dumps(rec), flush=True)
        return

    result = {"stack": "vllm", "records": []}
    try:
        import vllm
        result["vllm_version"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass
    for scale in args.scales.split(","):
        for v in VARIANTS:
            childlog = args.out + f".child_{scale}B_{v}.log"
            with open(childlog, "w") as lf:
                p = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--models-dir",
                     args.models_dir, "--out", args.out, "--one", v,
                     "--scale", scale],
                    stdout=subprocess.PIPE, stderr=lf, text=True, timeout=1800)
            got = False
            for line in p.stdout.splitlines():
                if line.startswith("REC "):
                    result["records"].append(json.loads(line[4:]))
                    got = True
            if not got:
                tail = p.stdout[-1500:] + "\nSTDERR_TAIL:\n" + \
                    open(childlog).read()[-4500:]
                result["records"].append({"variant": v, "scale_b": scale,
                                          "ok": False, "error": tail})
            print(f"== {scale}B {v} done (ok={got})", flush=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
    print("ROUTE3_DONE", flush=True)


if __name__ == "__main__":
    main()
