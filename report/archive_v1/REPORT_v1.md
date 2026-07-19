# Precision speedup report: FP32 → TF32 / BF16 / FP16 / FP8 / INT8 / INT4

Measured on **1× NVIDIA H100 64GB (MareNostrum 5 `acc` partition, BSC)**,
torch 2.11.0+cu128, single GPU. Two rounds:

- **Round 1 (eager)**: five ~1B models, one per modality, stock-PyTorch eager
  kernels only — the floor.
- **Round 2 (routes + scale)**: Qwen2.5 0.5B→7B across four acceleration
  stacks — eager, `torch.compile` (inductor), torchao+compile, and vLLM — the
  path from floor to production ceiling.

Raw data: [`../results/*.json`](../results). Generated tables:
[`tables.md`](tables.md) (eager, 5 modalities), 
[`tables_scale_routes.md`](tables_scale_routes.md) (scale × route matrix).

> 中文结论速览见 [§10](#10-中文结论速览)。

## 0. Measurement validity (GPU saturation proof)

Every throughput record embeds a 200 ms `nvidia-smi` sampler covering **only
the timed window** (setup/compile/load excluded by construction):
`gpu_util_avg`, `power_w_avg`, `sm_clock_mhz_avg`. Reading rules:

- `utilization.gpu` = "a kernel was resident" — **memory-bound kernels also
  show ~100%**, so utilization alone does not prove compute saturation.
  Power draw and achieved-TFLOPS/MFU are the compute indicators
  (e.g. eager INT8 inference: 100% util but only 370 W — busy, not fed).
- Compute-bound comparisons in this report (train / batch-forward) all ran at
  **99–100% util, 360–690 W**; decode rows are *intentionally* memory- or
  launch-bound — that is the workload being measured, and they are labeled so.
- Sub-100 ms measurement windows (GEMM microbench, vLLM small-model prefill)
  are too short for the 200 ms sampler; those cells rely on achieved TFLOPS
  and are marked accordingly.

Causes of low GPU utilization we hit during this work, and their fixes:

| Cause | Symptom | Fix |
|---|---|---|
| Engine/compile startup (CPU-bound) | minutes of 0% at job start | exclude from timed window; persist `TORCHINDUCTOR_CACHE_DIR`; reuse engines |
| Crash loop (missing `ninja` for flashinfer JIT) | whole job ~0% | fail-fast + full child logs; ship `ninja` in venv and put `venv/bin` on `PATH` |
| Checkpoint IO from GPFS | GPU idle during load | /dev/shm staging, mmap, warm cache |
| Kernel-launch overhead (eager bs=1 decode) | ~90 tok/s regardless of precision | CUDA graphs / compile / vLLM |
| Memory-bound decode / small batch | high util, low power/MFU | intrinsic to the workload — label, don't "fix" |
| Unfused dynamic-quant ops | fragmented small kernels | fusion (routes 1–3 below) |

## 1. Question framing — what "QAT speedup" actually means

The question was "以 FP32 为基准，TF32 / BF16 / TF16 / FP8 / INT4 的 QAT 分别能加速多少".
Two clarifications define the report:

1. **"TF16" is not a real format** — interpreted as **FP16**. Measured set:
   TF32, BF16, FP16, FP8 (E4M3/E5M2), INT8, INT4.
2. **QAT (quantization-aware training) does not speed up training.** It
   inserts fake-quant (quantize→dequantize + straight-through estimator) into
   a full-precision graph — QAT training is *slower* than its mixed-precision
   baseline (§5 measures exactly how much; low-bit hardware sits idle during
   QAT by construction). QAT's payoff is **deployment**: the checkpoint runs
   real INT8/INT4/FP8 kernels at inference with recovered accuracy (§6.2
   shows why recovery is needed). So the question decomposes into
   (a) mixed-precision training speedup, (b) QAT overhead, (c) quantized
   inference speedup — and (c) depends heavily on the software stack (§8).

## 2. Hardware ceiling (H100, dense)

| Precision | Engine | Peak | Theoretical × vs FP32 |
|---|---|---:|---:|
| FP32 | CUDA cores | 67 TF | 1.0× |
| TF32 | Tensor Core | 495 TF | 7.4× |
| BF16 / FP16 | Tensor Core | 989 TF | 14.8× |
| FP8 (E4M3) | Tensor Core | 1979 TF | 29.5× |
| INT8 | Tensor Core | 1979 TOPS | 29.5× |
| INT4 | — | *no INT4 tensor core on Hopper* | n/a |

INT4 on H100 is a **memory-bandwidth play** (weights are 4× smaller; kernels
dequantize into BF16 tensor cores) — it can only win where weight loading
dominates.

## 3. Setup

Round-1 models (random-init from vendored configs — throughput is
architecture-determined, so identical to pretrained):

| Modality | Model | Params | train bs | infer bs |
|---|---|---:|---:|---:|
| language | Qwen2.5-1.5B, seq 1024 | 1.5B | 4 | 16 |
| image | DINOv2-giant + cls head, 224px | 1.1B | 16 | 64 |
| video | VideoMAE-huge, 16×224px | 0.6B | 4 | 16 |
| audio | Whisper-large-v3, 30 s mel | 1.5B | 4 | 8 |
| multimodal | Qwen2-VL-2B, 224px + 64 tok | 2.2B | 4 | 8 |

Round-2 scale sweep: Qwen2.5 **0.5B / 1.5B / 3B / 7B** (eager sweep uses
random-init configs; torchao/vLLM routes use the real Instruct checkpoints +
Qwen's official GPTQ-Int8 and AWQ releases). Protocols: train = AdamW, 5+20
steps, median; infer = batch fwd bs16×1024; decode = 128-token prompt + 128
forced new tokens at bs=1/32. Quality proxy = logit cosine vs FP32.

## 4. GEMM microbenchmark — the measured kernel ceiling

TFLOP/s (× vs FP32), stock kernels, eager:

| precision | 4096³ | 8192³ | LLM-MLP 16384×1536×8960 | decode-like 16×8192×8192 |
|---|---|---|---|---|
| `fp32` | 50.2 (1.00×) | 50.4 (1.00×) | 52.3 (1.00×) | 7.5 (1.00×) |
| `tf32` | 378.3 (**7.54×**) | 308.1 (6.12×) | 286.6 (5.48×) | 11.3 (1.52×) |
| `bf16` | 772.7 (**15.39×**) | 718.8 (14.27×) | 695.5 (13.30×) | 23.0 (3.09×) |
| `fp16` | 750.4 (14.95×) | 684.9 (13.60×) | 646.8 (12.37×) | 23.2 (3.12×) |
| `fp8` | 1181.0 (**23.53×**) | 1193.3 (23.69×) | 998.2 (19.09×) | 44.1 (5.92×) |
| `int8` | 126.8 (2.53×) | 128.1 (2.54×) | 124.1 (2.37×) | 8.8 (1.17×) |
| `int4` | 29.4 (0.59×) | 29.7 (0.59×) | 28.3 (0.54×) | 26.9 (**3.61×**) |

- TF32/BF16/FP8 track the datasheet hierarchy (FP8 = 1.19 PFLOP/s ≈ 60% of
  dense peak with per-tensor scaling).
- **Stock `torch._int_mm` is broken-slow on Hopper** (2.5×; cuBLASLt's
  row-major int8 path misses the fast IMMA pipeline). With
  `torch.compile(mode="max-autotune")`, inductor's Triton int8 template
  reaches **267 TFLOP/s (2.1× over cuBLASLt)** — better, still ~13% of peak;
  full recovery needs CUTLASS-class kernels (vLLM, §8).
- **INT4 tinygemm inverts with batch**: 0.59× at large M, 3.61× (and 1.17×
  vs BF16) in the decode-like shape — the memory-bound regime it was built for.

## 5. Training speedup vs FP32 (eager, all 99%+ util)

| precision | lang 1.5B | image | video | audio | mm | ← modality |
|---|---|---|---|---|---|---|
| `tf32` | 2.39× | 2.40× | 1.90× | 1.96× | 1.70× | |
| `bf16` | **3.90×** | **3.71×** | **5.36×** | **3.75×** | 1.91× | |
| `fp16` | 3.76× | 3.57× | 5.17× | 3.60× | 1.80× | |
| `fp8_train` (real FP8 GEMM) | 2.26× | 2.01× | 2.56× | 1.98× | 1.23× | |
| `fp8_qat` / `int8_qat` / `int4_qat` (fake quant) | 2.52–2.56× | 2.31–2.35× | 3.04–3.11× | 2.22–2.27× | 1.22–1.24× | |

Scale trend (Qwen2.5): BF16 3.97× (0.5B) → 3.90× (1.5B) → 3.19× (3B, bs
shrinks to fit fp32 states) → **7B: every precision OOMs** — AdamW's ~16
bytes/param of fp32 state exceeds 64 GB regardless of compute precision.
That is the real "7B wall" on a single GPU: beyond ~3–4B you shard (ZeRO/FSDP)
or change the optimizer/recipe (LoRA, 8-bit optim); mixed precision alone
cannot help because master weights, gradients and moments stay fp32.

Key readings:

- **BF16 is the training workhorse: typ. ~3.8× vs FP32** (memory: 51.5 →
  39.9 GiB on lang). FP16 ≈ BF16 − 3%; no reason to prefer it on H100.
  TF32 gives ~2× for a flag flip.
- **QAT costs 1.5–1.7× over BF16** (bit-width irrelevant — int4/int8/fp8
  fake-quant cost the same round+clamp ops), landing at 2.2–3.1× vs FP32.
- Naive per-tensor FP8 training loses to BF16 at ≤3B; the fused/fp8 payoff
  belongs to TE/compile at larger scale — consistent with the "FP8 pays at
  ~7B+" folklore, though on one 64 GB GPU 7B full training is memory-blocked
  before compute even matters.

## 6. Inference speedup vs FP32 (eager) and fidelity

Batch forward (all 100% util):

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `tf32` | 2.75× | 3.18× | 2.37× | 2.41× | 2.76× |
| `bf16` | **7.99×** | **8.10×** | **9.24×** | **6.75×** | **4.68×** |
| `fp16` | 7.92× | 7.94× | 8.98× | 6.66× | 4.67× |
| `fp8` | 3.83× | 2.83× | 2.76× | 2.56× | 2.78× |
| `int8` | 1.28× | 1.04× | 1.14× | 1.10× | 1.09× |
| `int4` | 0.78× | 0.63× | 0.76× | 0.80× | 0.75× |

### 6.2 Fidelity probe (logit cosine vs FP32) — why QAT exists

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `bf16` | 0.9901 | 0.9983 | 1.0000 | 0.9999 | 0.9996 |
| `fp8` | 0.9264 | 0.8927 | 0.9973 | 0.9935 | 0.9665 |
| `int8` | 0.9778 | 0.9705 | 0.9997 | 0.9991 | 0.9942 |
| `int4` | **0.6520** | **0.5172** | 0.9695 | 0.9188 | **0.7615** |

Naive round-to-nearest INT4 collapses deep-network logits — recovering this
is QAT's entire job, paid for with §5's 1.5–1.7× training overhead.

## 7. Scale sweep (Qwen2.5 0.5B → 7B, eager)

Speedup vs FP32 at each scale:

| | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| infer `bf16` | 7.76× | 7.95× | 8.77× | **10.16×** |
| infer `fp8` | 3.85× | 3.83× | 4.46× | **6.32×** |
| decode32 `bf16` | 1.29× | 1.94× | 2.54× | **3.19×** |
| decode32 `int4` | 1.14× | 1.42× | 1.59× | 1.41× |
| decode1 `int4` vs `bf16` | 0.98× | 1.09× | 1.01× | **1.24×** |

**Low-precision benefit grows with model size** — larger GEMMs are more
compute-bound (less launch/overhead dilution), and bigger weights make
memory-bound decode more weight-dominated. The user-quoted claim "要到 7B
效果才明显" is directionally confirmed on every axis we measured.


### 7.5 The 7B single-GPU wall: deficit ledger, Muon, TF32, and FSDP

The scale sweep ended at a hard wall: 7B full-parameter AdamW OOMs on 64 GB
at every compute precision. We quantified the deficit and tested the two
escape routes users actually ask about (all cells empirically measured,
`results/mem_lang3.json`, `mem_lang7.json`, `fsdp7b_4gpu.json`):

**Deficit.** AdamW's standard recipe costs ~16 B/param of fp32 state
(weights 4 + grads 4 + two moments 8): 7.62B params → **~122 GB state +
activations ≈ 126 GB vs 67.7 GB available — short by ~58 GB (1.9×)**.
The allocator dies at 61.6 GiB with 260 MiB requested, during optimizer-state
materialization.

**Single-GPU variants (7B, bs=1):**

| recipe | storage | peak mem | outcome |
|---|---|---:|---|
| AdamW, bf16-autocast | fp32 | died @61.6 GiB | OOM |
| Muon(+AdamW for embeds), bf16-autocast | fp32 | died @61.6 GiB | **OOM — Muon alone does NOT fix 7B** |
| AdamW, TF32 matmuls | fp32 | died @61.6 GiB | OOM — **TF32 is a compute knob, memory-identical to FP32** |
| Muon, TF32 matmuls | fp32 | died @61.6 GiB | OOM (same reason) |
| AdamW, pure bf16 states | bf16 | died @56.8 GiB | OOM (borderline, as ledger predicts ~65 GB) |
| **Muon, pure bf16** | bf16 | **46.7 GiB** | **fits with ~16 GB headroom; 729 tok/s @99.8% util/618 W; loss 12.67→3.05 over 20 steps** |

Muon's saving is the optimizer state (one momentum vs Adam's two moments:
−27 GB at 7B), but the fp32 weights+grads floor (61 GB) survives it — hence
Muon only rescues 7B when combined with bf16 weights/grads/momentum (the
Moonlight-style recipe). Newton–Schulz orthogonalization tolerates bf16 well.

**TF32 + Muon at 3B** (largest scale where fp32 recipes fit — memory AND
throughput measured):

| recipe (3B, bs=1) | peak mem | tok/s | note |
|---|---:|---:|---|
| AdamW bf16-autocast | 57.5 GiB | 3487 | reference |
| AdamW TF32 | 57.5 GiB | 2914 | identical memory, 16% slower than bf16-autocast |
| Muon bf16-autocast | 36.9 GiB | 1679 | **−20.6 GiB vs AdamW** |
| Muon TF32 | 38.0 GiB | 1536 | memory ≈ Muon-fp32; no reason to prefer TF32 over autocast |
| AdamW pure bf16 | 28.8 GiB | 6000 | fastest (no fp32 master) |
| Muon pure bf16 | 19.0 GiB | 2041 | memory champion (3× less than AdamW-fp32) |

Muon's throughput cost here (~2× vs AdamW) is our *naive* per-parameter
Newton–Schulz at bs=1, where the fixed per-step optimizer cost dominates a
tiny fwd/bwd; production Muon (fused, comm-overlapped, large global batch)
amortizes this to single-digit percent.

**Distributed (FSDP full-shard = ZeRO-3, AdamW standard recipe, 7B):**

| setup | per-rank peak | aggregate tok/s | outcome |
|---|---:|---:|---|
| 2× H100 | (state shard ~61 GB + gathered layers) | — | **OOM on both ranks** — as ledger predicts |
| **4× H100** | **35.5 GiB** | **10 147 @99.5% util** | fits comfortably; loss 12.68→0.00 |

So the escalation ladder for "7B won't fit" is: **Muon+pure-bf16 (1 GPU,
46.7 GiB) → FSDP ≥4 GPUs for the standard AdamW recipe (35.5 GiB/rank) — and
they compose** (distributed Muon is exactly how Moonlight trains). TF32
belongs in the compute-speed conversation (§5), never in the memory one.


### 7.6 Distributed scaling for the 7B wall (FSDP vs Megatron-style TP)

All runs: Qwen2.5-7B, per-rank bs=1, seq 1024; NVLink within node, InfiniBand
across nodes. TP = transformers `tp_plan="auto"` (torch DTensor colwise/rowwise
sharding — Megatron-style intra-layer partitioning). "std recipe" = fp32
master + bf16 autocast + AdamW.

| setup | GPUs | nodes | per-rank peak | aggregate tok/s | util | note |
|---|---:|---:|---:|---:|---:|---|
| single GPU, std recipe | 1 | 1 | OOM @61.6 GiB | — | — | needs ~126 GB |
| FSDP full-shard | 2 | 1 | OOM | — | — | shard ~61 GB still too big |
| FSDP full-shard | 3 | 1 | 47.4 GiB | 5 538 | 99.6% | minimal fix for the wall |
| FSDP full-shard | 4 | 1 | 35.5 GiB | 10 147 | 99.5% | |
| FSDP full-shard | 8 | **2** | 29.4 GiB | **14 675** | 97.9% | 72% scaling eff. vs 2×4GPU (step 404→558 ms over IB) |
| TP=2, std recipe | 2 | 1 | OOM | — | — | ~61 GB/rank, same wall |
| TP=2, bf16 storage | 2 | 1 | 32.8 GiB | 4 625 | 99.0% | |
| TP=4, std recipe | 4 | 1 | 38.9 GiB | 4 379 | 98.5% | 1 095 tok/s/GPU |
| TP=4, std recipe | 4 | **2** | 38.9 GiB | 3 255 | 99.3% | **−26% for crossing nodes** (step 234→315 ms) |
| TP=8 | 8 | 2 | — | — | — | **infeasible**: 28 Q-heads / 4 KV-heads not divisible by 8 |

Readings:

1. **FSDP is the default answer to "it doesn't fit"**: at 4 GPUs it delivers
   2.3× the aggregate throughput of TP=4 (10 147 vs 4 379) at similar
   per-rank memory, because data parallelism processes W independent streams
   while TP taxes every layer with an all-reduce to co-process one stream.
2. **TP's constraints are structural**: the TP degree must divide the head
   counts (Qwen2.5-7B: 28 Q / 4 KV → TP ∈ {2, 4} only). TP earns its place
   when the model (or KV cache) cannot fit even sharded, or for latency —
   which is why Megatron deployments put TP inside the node and scale out
   with DP/PP.
3. **Cross-node penalties are real but survivable on this fabric**: −28%
   step-time for FSDP (all-gathers over IB), −26% throughput for TP.
   Both keep 97–99% util — another reminder that util% ≠ efficiency
   (NCCL wait time counts as "kernel resident").
4. Composition (not run here) follows directly: Muon/bf16 recipes from §7.5
   cut the sharded state further, and TP×DP meshes combine both axes —
   the standard Megatron/Moonlight production layout.

### 7.7 Infeasible / failed items and root causes (完整失败清单)

| item | root cause | status |
|---|---|---|
| 7B single-GPU full-param AdamW (any compute precision) | 16 B/param fp32 state ≈ 126 GB > 64 GB | physics; use §7.5/§7.6 routes |
| TF32 as a memory fix | TF32 is a matmul mode; storage identical to FP32 (measured byte-identical peaks) | misconception, documented |
| TP=8 for Qwen2.5-7B | head counts (28 Q / 4 KV) not divisible by 8 | architectural constraint |
| torchao INT4 (`int4wo`) | torchao 0.17 requires `mslk` kernel package; PyPI only has a 0.0.0 placeholder | covered via vLLM AWQ-Marlin + eager tinygemm instead |
| HF `generate` + `torch.compile(mode="reduce-overhead")` | CUDA-graph pool conflicts with cache-tensor mutation (both transformers 5.10 and 5.14, different symptoms incl. segfault) | dropped; vLLM provides the CUDA-graph decode data |
| vLLM from PyPI on MN5 | default wheel links `libcudart.so.13` (CUDA 13) — node driver is CUDA-12.x | fixed via GitHub release `+cu129` wheel |
| flashinfer JIT at vLLM startup | `ninja` not on compute-node PATH | fixed: venv/bin on PATH |
| model/venv downloads via transfer1.bsc.es | transfer node had no DNS at benchmark time (contrary to older docs) | worked around: build/download locally + rsync |
| vLLM 0.5–1.5B prefill util cells | measurement window shorter than 200 ms sampler period; engine CPU-scheduling-bound at small scale | cells rely on achieved throughput; flagged |


### 7.8 Low-bit AdamW states, and the master-weight precision floor

Two follow-up questions from the 7B wall: (a) can AdamW itself go
BF16/FP8/INT8/INT4? (b) why not store the *weights* in the target precision
too? Both answered empirically (`results/mem_lowbit7b.json`,
`results/master_prec.json`).

**(a) Quantized optimizer states work — and beat Muon on speed** (7B, bs=1,
bf16 weights/grads, torchao low-bit optimizers):

| optimizer | states | peak mem | tok/s | 20-step loss |
|---|---|---:|---:|---|
| AdamW (bf16 states) | 30.5 GB | OOM @56.8 GiB | — | — |
| **AdamW-FP8 states** | ~15 GB | **43.9 GiB** | **4 704** | 12.67→0.01 ✓ |
| AdamW-INT8 states | ~8 GB | 46.9 GiB | 3 588 | 12.67→0.01 ✓ |
| AdamW-INT4 states | ~4 GB | 48.2 GiB | 2 379 | 12.67→0.10 (noisier) |
| Muon (bf16, §7.5) | 13 GB | 46.7 GiB | 721 | 12.67→3.05 |

torchao's fused low-bit optimizers make **AdamW-FP8 the best single-GPU 7B
recipe measured in this study**: less memory than Muon-bf16 *and* 6.5× its
throughput at bs=1 (Muon's per-step Newton–Schulz dominates at tiny batch).
Optimizer-state precision is nearly free to lower (states only feed the
update; INT4 states show mild noise); INT8-state Adam is production-proven
(bitsandbytes lineage).

**Head-to-head at 3B** (largest scale where every variant fits; bs=1, same
process/venv, 20 steps):

| optimizer (weights) | peak mem | vs fp32 recipe | tok/s | 20-step loss |
|---|---:|---:|---:|---|
| AdamW fp32-states (fp32 master, mixed prec) | 57.5 GiB | 1.00× | 3 486 | →0.01 (reference convergence) |
| AdamW bf16-states (bf16 master) | 28.8 GiB | −50% | **5 993** | →1.21 |
| AdamW-INT8 states (bf16 master) | 19.7 GiB | −66% | 4 994 | →1.20 |
| AdamW-INT4 states (bf16 master) | 21.6 GiB | −62% | 4 935 | →3.62 (noisy) |
| **AdamW-FP8 states (bf16 master)** | **18.6 GiB** | **−68%** | 5 100 | →1.20 |
| Muon (bf16 master) | 19.0 GiB | −67% | 2 042 | →3.47 |

Readings: (1) **AdamW-FP8 is the memory champion** (3× less than the fp32
recipe) at only ~15% throughput cost vs plain-bf16 states — the cost is the
per-step quantize/dequantize of the moments (visible as util dropping to
60–72% during optimizer segments). (2) INT4 states get noticeably noisy;
INT8/FP8 match bf16-state convergence. (3) All bf16-master variants stop at
loss ≈1.2 where the fp32-master reference reaches 0.01 — the master-weight
precision effect of §7.8(b) again, now at the bf16-vs-fp32 boundary.
(4) Muon's memory is competitive but its naive NS step is 2.4× slower at
bs=1; its niche vs low-bit Adam is optimizer-quality, not economics.


**Why "8-bit training is faster" for big models — the memory→throughput
exchange, quantified** (7B, batch-size sweep, `mem_lowbit7b_bs*.json`):

| optimizer states | bs=1 | bs=2 | bs=4 | bs=8 |
|---|---:|---:|---:|---:|
| bf16 (30.5 GB states) | OOM | OOM | OOM | OOM |
| **FP8 states** | 4 704 | 6 330 | **7 650** | OOM (activations) |
| INT8 states | 3 588 | 5 233 | 6 792 | OOM |

The FP8-state codec is *slower per element* than bf16 (§ above), but the
~12 GB it frees buys batch size, and batching amortizes every fixed per-step
cost: **+63% throughput for FP8-states from bs=1→4, while the bf16-state
recipe cannot run at any batch size**. This — plus FP8 GEMM/communication at
scale (DeepSeek-V3-style compute-FP8) and avoided
checkpointing/offload/extra-parallelism — is where the "low-bit training is
faster on large models" folklore actually comes from; it is a systems-level
exchange, never an elementwise-kernel speedup.

**(b) Trainable master weights have a hard precision floor.** If the stored
weights themselves are quantized after every update ("权重也转换为相应精度"),
the update (~lr ≈ 1e-5) must survive the quantization step
(INT8/per-channel ≈ 5e-4, INT4 ≈ 8e-3, FP8 ≈ 6% relative). Measured
(Qwen2.5-1.5B, fixed batch, 30 steps, same lr — loss drop is the signal):

| master storage | loss 12.2 → (30 steps) | verdict |
|---|---|---|
| bf16 | **7.38** (−4.84) | trains normally |
| fp8, round-to-nearest | 12.00 (−0.22) | 22× slower progress |
| int8, round-to-nearest | 12.21 (**−0.006**) | **frozen — updates rounded away** |
| int4, round-to-nearest | 12.22 (−0.006) | frozen |
| int8, stochastic rounding | 12.18 (−0.06) | survives in expectation, 80× slower here |
| int4, stochastic rounding | 12.14 (−0.13) | noise-dominated |

This is why every real recipe — TE FP8, our `fp8_train`, QAT, even BitNet —
keeps a bf16/fp32 master (or hides it in distributed optimizer shards, as
FP8-LM does), and why the only mainstream "weights truly stored in INT4
during training" is QLoRA, where the INT4 base is **frozen** and only bf16
adapters train. Low precision belongs to *compute* and *deployment*;
trainable *storage* bottoms out around bf16-with-care.

## 8. Realizing the potential: eager vs compile vs torchao vs vLLM

The eager numbers above are the floor. Three escalation routes
(full matrix with per-cell GPU util: `tables_scale_routes.md`):

**Batch forward / prefill (tokens/s, Qwen2.5-1.5B and 7B):**

| stack | bf16 | fp8 | int8 | int4 |
|---|---|---|---|---|
| eager 1.5B | 101k | 48k | 16k | 10k |
| +compile (route 2) | 142k | **155k** | 35k | — |
| torchao+compile (route 1) | 142k | **161k** | 105k | (needs `mslk`) |
| vLLM (route 3) 1.5B | 898k* | 924k* | 829k* | 872k* |
| eager 7B | 32k | 20k | 5k | 2k |
| torchao 7B | 39k | **56k** | 33k | — |
| vLLM 7B | 480k | **651k** | 581k | 695k |

\* small-model vLLM prefill is engine-scheduling-bound (util 19–43%); treat
as engine throughput, not kernel speed. 7B cells ran at 86–100% util.

**Decode tokens/s (bs=1 / bs=32):**

| stack, 7B | bf16 | fp8 | int8 | int4 |
|---|---|---|---|---|
| eager | 69 / 1912 | 37 / 1283 | 18 / 526 | 86 / 846 |
| torchao+compile | 77 / 1718 | 92 / 1908 | — | — |
| vLLM | 92 / 2764 | 150 / 4579 | 134 / 3876 | **175 / 5040** |

Findings:

1. **Fusion flips the FP8 verdict.** Eager FP8 loses to BF16 (unfused
   quant/dequant launches); one `torch.compile` makes FP8 the fastest
   PyTorch-native option (155–161k > 142k at 1.5B; 1.41× over BF16 at 7B).
   Exactly the mechanism predicted in §4.
2. **INT8's datasheet 29.5× stays unreachable in pure PyTorch** — compile
   triples `_int_mm` throughput but torchao W8A8 still trails BF16; vLLM's
   CUTLASS/Marlin kernels are where INT8 finally pays.
3. **vLLM turns quantized decode from a loss into the win**: at 7B,
   FP8 1.66×, INT8 1.40×, **AWQ-INT4 1.82×** over vLLM-BF16 — and the INT4
   advantage grows with scale (0.83× at 0.5B → 1.12× → 1.82×), while in
   eager it never beat BF16 at all. bs=1 latency: 588–637 tok/s (0.5B) vs
   eager's launch-bound ~100.
4. **Stack choice dwarfs precision choice.** From eager-BF16 to vLLM-INT4 at
   7B decode: 846 → 5040 tok/s (6×); no precision selection inside a single
   stack comes close.


### 8.5 Max-fusion sweep: compile + FA3-class attention for everything except FP32

Per user request: FP32 stays the eager baseline; every other precision gets
`torch.compile` + fused attention. True FA3 is source-build-only and FA4 beta
is a JIT/CuTe-DSL distribution — both impractical on an air-gapped cluster —
so the fused-attention arm uses **torch 2.11's cuDNN 9 SDPA backend**
(FA3-class fused attention on Hopper, zero install; the vLLM rows in §8
already embed real FA3 via vllm-flash-attn). Qwen2.5, bs as in §3, all rows
99–100% util at 380–693 W (`results/lang_fused.json`, `lang7_fused.json`).

Speedup vs FP32-eager ( → change from the eager-only value):

| precision | train 1.5B | infer 1.5B | infer 7B |
|---|---|---|---|
| `tf32` | 2.93× (←2.39×) | 4.20× (←2.75×) | 5.07× (←3.77×) |
| `bf16` | **5.35×** (←3.90×) | **11.11×** (←7.99×) | **12.25×** (←10.16×) |
| `fp16` | 5.10× (←3.76×) | 10.99× (←7.92×) | 12.08× (←10.09×) |
| `fp8` | **5.21×** (←2.26×, real-FP8 train) | **12.25×** (←3.83×) | **16.49×** (←6.32×) |
| `int8` (QAT / W8A8) | 5.31× (←2.56×) | 2.75× (←1.28×) | 2.63× (←1.67×) |
| `int4` (QAT / W4 tinygemm) | 5.32× (←2.56×) | 0.80× (←0.78×) | 0.65× (←0.64×) |

Headline findings:

1. **Fusion makes QAT training overhead vanish**: int8/int4 fake-quant
   training lands at 5.31–5.32× vs FP32 — within **1% of BF16's 5.35×**
   (eager: 35–40% slower than BF16). Inductor fuses the quantize→dequantize
   elementwise chains into the surrounding kernels, so with `torch.compile`
   the answer to "QAT 训练要付多少代价" drops from ×1.5–1.7 to ≈×1.01.
2. **FP8 becomes the fastest option end-to-end once fused**: real-FP8
   training reaches BF16 parity at 1.5B (5.21× vs 5.35× — the ≥7B folklore
   about FP8 training is largely a statement about unfused overhead), and
   FP8 inference clearly beats BF16 (12.25× vs 11.11× at 1.5B; 16.49× vs
   12.25× at 7B — kernel ratio finally visible end-to-end).
3. INT8 W8A8 inference triples with fusion (2.75×) but still can't touch
   BF16 in pure PyTorch (§4's cuBLASLt wall); INT4 batch inference stays a
   loss — its domain remains memory-bound decode (vLLM, §8).
4. A measurement gotcha worth recording: `torch._dynamo`'s per-code-object
   cache limit (8) silently exhausts across many compiled variants in one
   process, making later `torch.compile` calls run eager. Symptom: exactly
   1.00× "speedup". Fix: `torch._dynamo.reset()` between variants — at the
   cost of a full recompile each time (the 0%-util compile gaps the user
   observed; timed windows are unaffected).

## 9. Synthesis — the answer in one table

Speedup vs FP32, ~1B-class models, one H100:

| Precision | GEMM ceiling | Training | Inference floor (eager) | Inference realized (best stack) |
|---|---:|---:|---:|---|
| TF32 | 5.5–7.5× | **1.7–2.4×** | 2.4–3.2× | (superseded by BF16) |
| BF16 | 13–15× | **1.9–5.4×** (typ ~3.8×) | 4.7–10.2× | reference for stacks |
| FP16 | 12–15× | ≈ BF16 | ≈ BF16 | — |
| FP8 | 19–24× | 2.0–2.6× naive; needs TE/compile | 2.6–3.8× | **>BF16 with compile; 1.66× over BF16 in vLLM decode @7B** |
| INT8 (QAT→deploy) | 2.5× stock / ~30× theory | QAT: 2.2–3.1× (=BF16÷1.5–1.7) | 1.0–1.3× | 1.40× over BF16 (vLLM @7B decode) |
| INT4 (QAT→deploy) | 3.6× small-M only | QAT: same as INT8 | 0.6–0.8× batch; ~1× decode | **1.82× over BF16 (vLLM AWQ @7B decode)** |

## 10. 中文结论速览

以 FP32 为基准，H100 单卡，两轮实测（每条吞吐记录都带计时窗口内的
GPU util/功耗采样；吞吐类对比全部在 99–100% util、360–690W 下取得）：

1. **训练**：TF32 约 1.7–2.4×（零成本）；**BF16 约 3.8× 是默认选择**；FP16
   与 BF16 持平无优势；朴素 FP8 训练在 ≤3B 反而不如 BF16（要 TE/compile 融合
   才划算）。**QAT 不加速训练**——比 BF16 慢 1.5–1.7×（int4/int8/fp8 假量化
   开销相同），它是在为部署精度付训练税（§6.2 显示直接 INT4 量化会把 logit
   余弦打到 0.52–0.76，QAT 就是为了补回这个）。
2. **单卡 7B 训练墙（§7.5 实测）**：AdamW 全套需 ~126GB,缺 ~58GB。
   换 Muon（标准 fp32 配方）省 27GB 优化器状态但仍 OOM——fp32 权重+梯度这
   61GB 地板动不了；**TF32 完全不省显存**（存储仍 fp32,实测 3B 时 57.5GiB
   与 fp32 分毫不差,7B 时死在同一位置），它只是计算模式。真正的出路：
   **Muon+纯 bf16 单卡 46.7GiB 放下**（729 tok/s,loss 正常下降）；或
   **FSDP 全分片 ≥4 卡**跑标准 AdamW（35.5GiB/卡,聚合 10147 tok/s;
   2 卡仍 OOM）。两者可叠加（分布式 Muon 即 Moonlight 路线）。
3. **规模效应确认**："越大越明显"在所有轴上成立：推理 BF16 加速
   7.8×→10.2×（0.5B→7B），FP8 3.9×→6.3×，vLLM 里 INT4 相对 BF16 的 decode
   优势 0.83×→1.82×，torchao 里 FP8 相对 BF16 0.99×→1.42×。
4. **最大融合（§8.5,除 FP32 全开 compile + cuDNN/FA3 级注意力）**：BF16 训练
   3.9→5.35×,推理 8.0→11.1×（7B 12.25×）；**FP8 全面登顶**——训练 5.21×
   追平 BF16（"FP8 要 7B 才划算"多半是未融合开销的错觉），推理 1.5B 12.25×、
   7B 16.49× 明确超过 BF16；**QAT 训练税从 1.5–1.7× 降到 ≈1.01×**（fake-quant
   被融进邻近 kernel）——即"QAT 几乎白送"成立的前提是 torch.compile。
5. **量化收益能否兑现取决于软件栈而非精度本身**：
   - eager（地板）：BF16 最优，FP8/INT8/INT4 全部跑输 BF16；
   - `torch.compile`（一行代码）：FP8 反超 BF16（155–161k vs 142k tok/s）；
     Triton 把 int8 GEMM 提速 2.1× 但仍只有峰值 13%；
   - vLLM（生产答案）：7B decode 上 FP8 1.66×、INT8 1.40×、**AWQ-INT4
     1.82×** 全面超过 BF16；bs=1 时延比 eager 好 5–6×。
   - 从 eager-BF16 到 vLLM-INT4（7B decode）总提升 **6×**——**选对栈比选
     精度更重要**。
6. **分布式（§7.6）**：FSDP 是"放不下"的默认解——3 卡即破 7B 墙（47.4GiB/卡），
   4 卡 10147 tok/s,2 节点 8 卡 14675 tok/s（跨节点扩展效率 72%）。Megatron 式
   TP 受结构约束（TP 度必须整除头数,本模型只能 2/4），同显存下吞吐只有 FSDP
   的 43%（每层 all-reduce 税），跨节点再 −26%——所以生产实践是节点内 TP、
   节点间 DP/PP,且可与 Muon/bf16 配方叠加。
7. **测量学**：`nvidia-smi` util=100% 不代表算力跑满（eager INT8 推理
   100% util 只有 370W）；判据是功耗 + 达成 TFLOPS/MFU。decode 类数字天然
   低算力利用（memory/launch-bound），那是被测对象而非测量缺陷。

## 11. Caveats

- Single GPU; multi-GPU adds communication that dilutes compute speedups.
- torchao 0.17 INT4 requires the `mslk` kernel package (no usable PyPI wheel
  at benchmark time) — INT4 deployment numbers come from vLLM AWQ-Marlin and
  eager tinygemm instead. torchao numbers use transformers 5.14; eager/route2
  used 5.10 (same kernels, minor version drift).
- HF `generate` + `mode="reduce-overhead"` CUDA graphs was unstable in this
  stack (cache-tensor overwrite / segfault) — compiled-decode cells for route
  2 are absent; vLLM covers the CUDA-graph decode story.
- vLLM small-model prefill cells are engine-throughput (CPU-scheduling-bound
  at 0.5–1.5B), not kernel measurements.
- vLLM INT8 = Qwen official GPTQ-Int8 (W8A16 Marlin), not W8A8; eager INT8 is
  W8A8 `_int_mm`. Schemes noted because they answer different questions.
- Fidelity probe is numerical (random weights for eager round), not task
  accuracy; real QAT accuracy requires task training.
- GEMM/vLLM-prefill windows shorter than the 200 ms sampler period have
  unreliable util cells; those comparisons rest on achieved TFLOPS.
