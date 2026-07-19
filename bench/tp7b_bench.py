"""Megatron-style tensor parallelism for the 7B wall, via transformers'
native `tp_plan="auto"` (torch DTensor colwise/rowwise sharding — the same
intra-layer partitioning Megatron-LM uses, minus the framework).

Run under torchrun; the whole world is one TP group (DP=1):

  torchrun --standalone --nproc_per_node=4 tp7b_bench.py \
      --model $Q/models/Qwen2.5-1.5B-Instruct --out out/tp7b_4gpu.json

Unlike FSDP (data parallel: aggregate tokens/s scales with world size),
TP shards each layer's math: one batch is processed cooperatively, so
tokens/s is per-model, and communication (all-reduce per layer) is the tax —
which is why TP belongs inside a node (NVLink) and cross-node TP is expected
to fall off a cliff.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import GpuSampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dtype", default="fp32_autocast",
                    choices=["fp32_autocast", "bf16"])
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)

    from transformers import AutoModelForCausalLM
    dt = torch.float32 if args.dtype == "fp32_autocast" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, tp_plan="auto")
    model.train()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)

    vocab = model.config.vocab_size
    torch.manual_seed(7)  # same batch on every TP rank
    ids = torch.randint(0, vocab, (args.bs, args.seq), device="cuda")
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
             "labels": ids.clone()}
    losses = []

    def step():
        opt.zero_grad(set_to_none=True)
        if args.dtype == "fp32_autocast":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
        else:
            out = model(**batch)
        loss = out.loss
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    with GpuSampler() as g:
        for _ in range(args.steps):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    dist.barrier()
    med = statistics.median(times)
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30],
                        device="cuda")
    peaks = [torch.zeros_like(peak) for _ in range(world)]
    dist.all_gather(peaks, peak)
    if rank == 0:
        nodes = int(os.environ.get("GROUP_WORLD_SIZE",
                                   os.environ.get("SLURM_NNODES", 1)))
        rec = {
            "variant": f"tp{world}_{args.dtype}", "ok": True,
            "tp_size": world, "nodes": nodes, "batch_size": args.bs,
            "ms_per_step_median": med * 1e3,
            "tokens_per_s": args.bs * args.seq / med,
            "tokens_per_s_per_gpu": args.bs * args.seq / med / world,
            "peak_mem_gib_per_rank": [round(float(p), 2) for p in peaks],
            "loss_first": losses[0], "loss_last": losses[-1],
            **g.stats(),
        }
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "records": [rec]}, f, indent=2)
        print(json.dumps(rec), flush=True)
        print("TP_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
