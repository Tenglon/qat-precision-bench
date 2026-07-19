"""Distributed answer to the 7B-on-64GB wall: FSDP full-shard AdamW.

Run under torchrun (--standalone --nproc_per_node=N). Shards params, grads
and AdamW state across ranks (ZeRO-3 equivalent), bf16 compute with fp32
sharded master params — the honest "standard AdamW recipe, just distributed".

Records per-rank peak memory, aggregate tokens/s, and the 20-step loss path.

  torchrun --standalone --nproc_per_node=4 fsdp7b_bench.py \
      --modality lang7 --out out/fsdp7b_4gpu.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import GpuSampler  # noqa: E402
from models import get_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modality", default="lang7")
    ap.add_argument("--bs", type=int, default=1, help="per-rank batch size")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)

    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        MixedPrecision, ShardingStrategy)
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    spec = get_spec(args.modality)
    model = spec.build()  # fp32 on this rank's GPU
    wrap = functools.partial(transformer_auto_wrap_policy,
                             transformer_layer_cls={Qwen2DecoderLayer})
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                       reduce_dtype=torch.bfloat16,
                                       buffer_dtype=torch.bfloat16),
        device_id=local,
        use_orig_params=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
    torch.manual_seed(100 + rank)
    batch = spec.make_batch(args.bs, train=True)
    losses = []

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(**batch).loss
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
        rec = {
            "variant": f"fsdp_adamw_{world}gpu", "ok": True,
            "world_size": world, "per_rank_bs": args.bs,
            "ms_per_step_median": med * 1e3,
            "tokens_per_s_aggregate": world * args.bs * spec.tokens_per_sample / med,
            "peak_mem_gib_per_rank": [round(float(p), 2) for p in peaks],
            "loss_first": losses[0], "loss_last": losses[-1],
            **g.stats(),
        }
        out = {"model": spec.desc, "gpu": torch.cuda.get_device_name(0),
               "records": [rec]}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(rec), flush=True)
        print("FSDP_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
