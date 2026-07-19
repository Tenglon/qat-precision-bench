"""Table 3: MODEL SCALE as the single variable (no-OOM rule: double GPUs).

Pinned: bf16-autocast training, AdamW fp32-states (lr=1e-5), eager, SDPA,
global batch = 4x1024 tokens where world allows (per-rank bs >= 1), seed 17.
Variable: Qwen2.5 0.5B / 1.5B / 3B / 7B / 14B. GPU count is escalated
1 -> 4 -> 8 per the no-OOM rule and reported per row (7B: 2 GPUs already
proven OOM in v1; 14B needs 8 ranks so global batch grows to 8x1024 - noted).

Run under torchrun (any nproc); world==1 trains plainly, world>1 uses FSDP
full-shard with the SAME recipe (fp32 master + bf16 compute).

  torchrun --standalone --nproc_per_node=N table3_scale.py \
      --modality lang7 --global-bs 4 --out out/table3_7b.json
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import GpuSampler  # noqa: E402
from models import get_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True)
    ap.add_argument("--global-bs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    bs = max(1, args.global_bs // world)
    spec = get_spec(args.modality)

    import models as _models
    big = args.modality in ("lang14", "lang32")
    if big:
        _models.DEVICE = "cpu"     # cannot even build pre-shard on one GPU
    model = spec.build()
    if big:
        _models.DEVICE = "cuda"

    if world > 1:
        from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                            MixedPrecision, ShardingStrategy)
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Qwen2DecoderLayer}),
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                           reduce_dtype=torch.bfloat16,
                                           buffer_dtype=torch.bfloat16),
            device_id=local, use_orig_params=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
    torch.manual_seed(23 + rank)
    batch = spec.make_batch(bs, train=True)
    losses = []

    def step():
        opt.zero_grad(set_to_none=True)
        if world > 1:
            loss = model(**batch).loss     # FSDP MixedPrecision handles dtype
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    times = []
    with GpuSampler() as g:
        for _ in range(args.steps):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    if world > 1:
        dist.barrier()
    med = statistics.median(times)
    peak = torch.cuda.max_memory_allocated() / 2**30
    if rank == 0:
        rec = {"modality": args.modality, "ok": True, "world_size": world,
               "per_rank_bs": bs, "global_tokens_per_step": world * bs * 1024,
               "ms_per_step_median": med * 1e3,
               "tokens_per_s_aggregate": world * bs * 1024 / med,
               "peak_mem_gib_rank0": round(peak, 2),
               "loss_first": losses[0], "loss_last": losses[-1],
               **g.stats()}
        with open(args.out, "w") as f:
            json.dump({"table": "3: model scale", "records": [rec]}, f, indent=2)
        print(json.dumps(rec), flush=True)
        print("TABLE3_DONE", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
