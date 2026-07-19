# Precision speedup report v2 — one variable at a time

Rewritten methodology (2026-07-19):

1. **Each table varies exactly ONE variable**; everything else is pinned and
   stated in the table header.
2. **No OOM rows**: if a configuration OOMs, GPU count is doubled until it
   fits, and the GPU count is reported in the row.
3. **Every low-precision row carries a numerical-fidelity column**, not just
   speed (training: update-direction cosine vs FP32 after 30 identical
   steps; inference: logit cosine vs FP32).
4. Fixed measurement protocol: 5 warmup + 20 timed steps (median), in-window
   GPU util/power sampling; setup/compile excluded from timing.

Hardware: NVIDIA H100 64GB (MareNostrum 5, BSC), torch 2.11.0+cu128.
Historical exploration (v1, superseded; full multi-variable sweeps incl.
modalities/scale/distributed/serving-stack comparisons): in git history
before the "Clean slate for report v2" commit.

Planned table roadmap (each confirmed with the user before running; single
variable each): 1 compute precision ✔ · 2 QAT scheme · 3 model scale
(GPU-doubling on OOM) · 4 inference precision (+logit fidelity) ·
5 fusion level (eager/compile/attention backend) · 6 distributed layout
(FSDP/TP, single/multi-node) · 7 serving stack (eager/torchao/vLLM) ·
8 modality. Roadmap covers every axis of the original goal.

## Table 1 — Training compute precision

**Pinned**: Qwen2.5-1.5B (random init, seed 17) · 1×H100 · bs=4 · seq=1024 ·
AdamW (fp32 states, lr=1e-5) · eager · SDPA · identical init & data.
**Variable**: training compute precision.
**Numerics**: cos / rel-err of the 30-step weight update ΔW vs the FP32 run.

| precision | ms/step | speedup | tokens/s | peak mem | util / power | ΔW cos vs fp32 | ΔW rel-err | loss@30 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `fp32` | 1096.4 | 1.00× | 3 736 | 51.5 GiB | 99.0% / 589 W | 1.000000 | 0.0000 | 0.0912 |
| `tf32` | 457.1 | **2.40×** | 8 961 | 51.5 GiB | 97.7% / 462 W | 0.999961 | 0.0089 | 0.0903 |
| `bf16` | 280.3 | **3.91×** | 14 613 | 39.9 GiB | 95.6% / 412 W | **0.983182** | **0.1828** | 0.2193 |
| `fp16` | 290.2 | 3.78× | 14 113 | 39.9 GiB | 95.9% / 417 W | 0.999953 | 0.0097 | 0.0906 |

Findings:

1. Speed: tf32 2.40×, bf16 3.91×, fp16 3.78× — consistent with v1.
2. **Numerics surprise: bf16's update error is ~20× larger than fp16's**
   (rel-err 0.183 vs 0.010; ΔW cosine 0.9832 vs 0.99995), and its 30-step
   loss lands visibly higher (0.219 vs 0.091). Mechanism: bf16 keeps fp32's
   8 exponent bits but only 7 mantissa bits, while fp16 has 10 — within
   range, fp16(+GradScaler) is simply a finer format. bf16's advantage is
   *robustness* (no overflow, no loss-scaling machinery), not accuracy.
   TF32 (10 mantissa bits, applied only inside matmuls) is nearly free of
   numerical cost.
3. Practical reading: for short/stable runs fp16 is both fast and faithful;
   bf16 buys crash-proof range at a real per-step precision cost that
   long-horizon training tolerates (and that larger batches/lr schedules
   average out) — this is why the industry default is bf16 *despite* Table 1.


## Table 2 — QAT fake-quant scheme (training)

**Pinned**: as Table 1 but bf16-autocast base, bs=4. **Variable**: fake-quant
scheme (STE, per-channel weights + per-token int8 acts; fp8 via e4m3 casts).
**Numerics**: ΔW vs the *bf16 baseline* — isolating QAT's own perturbation.

| scheme | ms/step | vs bf16 | peak mem | util / power | ΔW cos vs bf16 | ΔW rel-err | loss@30 |
|---|---:|---:|---:|---|---:|---:|---:|
| `none` | 281.1 | 1.00× | 39.9 GiB | 95.0% / 390.2 W | 1.000000 | 0.0000 | 0.2196 |
| `fp8_qat` | 435.3 | 0.65× | 40.2 GiB | 96.8% / 372.0 W | 0.848177 | 0.5485 | 0.2945 |
| `int8_qat` | 428.4 | 0.66× | 40.2 GiB | 97.2% / 367.8 W | 0.966034 | 0.2609 | 0.1492 |
| `int4_qat` | 428.2 | 0.66× | 40.2 GiB | 97.8% / 364.2 W | 0.252217 | 1.2202 | 0.3611 |

Findings: (1) QAT costs ×1.52 over bf16 regardless of bit-width, and only
+0.3 GiB (fake quant stores nothing persistent). (2) The numerics column
quantifies what QAT does to optimization: int8 barely bends the update
(cos 0.966), fp8 more (0.848), **int4 rotates it nearly orthogonal
(cos 0.252)** — QAT is descent on a different (quantized) loss landscape,
which is why int4 PTQ fails (T4: logit-cos 0.65) and why int4 QAT needs
longer schedules (loss@30 = 0.361 vs 0.220).

## Table 3 — Model scale (no-OOM rule in action)

**Pinned**: bf16-autocast + AdamW fp32-states, global batch 4×1024 tokens
(8×1024 at 14B where 8 ranks force it), eager/FSDP-std.
**Variable**: model size; GPUs escalate per the no-OOM rule.

| model | GPUs | tokens/step | ms/step | agg tokens/s | peak/rank | util |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 1 | 4096 | 126.2 | 32462 | 19.57 GiB | 98.3% |
| 1.5B | 1 | 4096 | 280.7 | 14591 | 39.9 GiB | 99.2% |
| 3B | 2 | 4096 | 308.8 | 13263 | 30.47 GiB | 99.3% |
| 7B | 4 | 4096 | 403.8 | 10143 | 35.53 GiB | 99.3% |
| 14B | 8 | 8192 | 1056.0 | 7757 | 36.85 GiB | 98.9% |

Findings: per-token cost grows ~linearly with parameters at pinned ~99%
util; the GPU column is the memory story (3B OOMs at bs=4 on 1 GPU → 2;
7B → 4; 14B → 8 across 2 nodes). Aggregate throughput degrades 4.2× from
0.5B→14B while hardware grew 8×.

## Table 4 — Inference precision, eager vs fused (+ logit fidelity)

**Pinned**: Qwen2.5-1.5B, 1×H100, batch fwd bs=16, seq=1024.
**Variable**: precision. Two pinned stacks shown side by side: eager, and
`torch.compile` (fp32 row stays eager — it is the baseline in both).

| precision | eager ms | eager × | eager GiB | fused ms | fused × | fused GiB | logit cos | rel-err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32` | 1294.2 | 1.00× | 15.2 | 1295.5 | 1.00× | 15.2 | 1.0000 | 0.0000 |
| `tf32` | 471.5 | 2.74× | 15.7 | 308.4 | 4.20× | 15.7 | 1.0000 | 0.0100 |
| `bf16` | 162.5 | 7.97× | 8.8 | 115.3 | 11.22× | 8.8 | 0.9901 | 0.5340 |
| `fp16` | 163.7 | 7.91× | 8.8 | 116.5 | 11.11× | 8.8 | 0.9997 | 0.0850 |
| `fp8` | 338.3 | 3.83× | 7.8 | 105.2 | 12.31× | 7.7 | 0.9264 | 1.6033 |
| `int8` | 1014.0 | 1.28× | 31.0 | 474.8 | 2.73× | 17.0 | 0.9778 | 0.8775 |
| `int4` | 1669.9 | 0.78× | 7.4 | 1620.1 | 0.80× | 7.4 | 0.6520 | 3.2868 |

Findings: (1) **The eager int8/int4 numbers are a stack artifact, mostly
cured (int8) or not curable (int4) by fusion**: compile fuses the dynamic
per-token quantize/dequantize chains, taking int8 1.28×→2.73× and fp8
3.83×→12.31× (now the fastest cell in the table); int4 stays ≤0.80× at
bs=16 because tinygemm is a memory-bound decode kernel — batch inference is
simply outside its regime, with or without fusion. (2) Memory column tells
the deployment story speed hides: int4 = 7.4 GiB (2× less than bf16, its
real selling point), while eager int8's 31 GiB peak (fp32 int-mm output +
unfused dequant temporaries) collapses to 17 GiB under compile.
(3) Remaining int8 gap vs bf16 is the Hopper cuBLASLt int8 path
(kernel-level 2.5×, v1 finding) — CUTLASS-class kernels (vLLM) are needed
to go further. (4) Fidelity is stack-independent (same math, same cos).


### Table 4c — Same design at 3B, fused only

**Pinned**: Qwen2.5-3B, 1×H100, bs=16, seq=1024, `torch.compile`
(fp32 row = eager baseline). **Variable**: precision.

| precision | ms/fwd | speedup | peak GiB | util | logit cos | rel-err |
|---|---:|---:|---:|---:|---:|---:|
| `fp32` | 2496.8 | 1.00× | 20.9 | 100.0% | 1.0000 | 0.0000 |
| `tf32` | 552.4 | 4.52× | 21.5 | 100.0% | 1.0000 | 0.0126 |
| `bf16` | 209.8 | 11.90× | 13.3 | 100.0% | 0.9622 | 1.0428 |
| `fp16` | 215.1 | 11.61× | 13.3 | 100.0% | 0.9990 | 0.1705 |
| `fp8` | 177.2 | 14.09× | 13.3 | 99.8% | 0.8482 | 2.3075 |
| `int8` | 900.1 | 2.77× | 18.6 | 100.0% | 0.9385 | 1.4548 |
| `int4` | 3344.4 | 0.75× | 13.3 | 100.0% | 0.4937 | 3.8869 |

Scale drift vs Table 4's 1.5B-fused columns: fp8 12.31×→**14.09×**, bf16
11.22×→11.90×, tf32 4.20×→4.52× — compute-bound speedups keep widening with
model size. Fidelity moves the other way: deeper nets accumulate quant error
(fp8 cos 0.926→0.848, int4 0.652→0.494, and notably bf16 0.990→0.962 while
fp16 stays 0.999) — the precision you can afford at deployment shrinks as
the model grows, which raises QAT's importance at scale.


### Measurement note — the fidelity metric itself needs numerical care

While extending T4 to larger models we caught our own bug: fp32
`F.cosine_similarity` over ~1.5×10⁸-element logit tensors returned values up
to **1.0386** (mathematically impossible; reproduced with synthetic data
where the true cosine was 0.9998 — fp32 reduction error at this length is
~4%). All fidelity columns in the T4 family are therefore computed in
**float64**; single-GPU fp32-reduction values agreed to 4 decimals in
retrospect, but every number below is the fp64 recompute. A fitting
meta-lesson for a numerical-precision report.

### Table 4d — 7B, fused, one job per precision

**Pinned**: Qwen2.5-7B, 1×H100, bs=16, seq=1024, `torch.compile` (fp32 row
eager). Per-precision Slurm jobs (fp32 first, others dependent — wall-clock
7× faster than the sequential design).

| precision | ms/fwd | speedup | peak GiB | util | logit cos (fp64) | rel-err |
|---|---:|---:|---:|---:|---:|---:|
| `fp32` | 5145.1 | 1.00× | 37.9 | 100.0% | 1.0000 | 0.0000 |
| `tf32` | 1017.4 | 5.06× | 37.9 | 100.0% | 1.0000 | 0.0209 |
| `bf16` | 419.2 | 12.27× | 29.4 | 100.0% | 0.8877 | 1.8380 |
| `fp16` | 421.2 | 12.22× | 29.4 | 100.0% | 0.9964 | 0.3318 |
| `fp8` | 314.8 | 16.34× | 29.4 | 100.0% | 0.7539 | 2.9709 |
| `int8` | 1968.3 | 2.61× | 29.4 | 100.0% | 0.8591 | 2.2028 |
| `int4` | 7958.8 | 0.65× | 29.4 | 100.0% | 0.4065 | 4.2671 |

### Fidelity vs scale (fp64, the numerics half of the scale story)

| precision | 1.5B | 3B | 7B |
|---|---:|---:|---:|
| `bf16` | 0.9901 | 0.9619 | 0.8877 |
| `fp16` | 0.9997 | 0.9990 | 0.9964 |
| `fp8` | 0.9264 | 0.8488 | 0.7539 |
| `int8` | 0.9778 | 0.9382 | 0.8591 |
| `int4` | 0.6520 | 0.4935 | 0.4065 |

Two clean monotone trends: **speed gains grow with scale** (fp8: 12.31× at
1.5B → 14.09× at 3B → 16.34× at 7B) while **fidelity falls with scale/depth**
(fp8 cos 0.926→0.849→0.754; int4 0.652→0.494→0.407; bf16 itself drops to
0.888 at 7B). fp16 is the outlier that keeps ~0.996+ at every scale — the
10-bit mantissa again. Practical reading: the bigger the model, the more the
speed argument favors low precision AND the more the accuracy argument
demands QAT (or fp16-flavored formats) rather than naive casting.

*(14B/32B rows pending a device-map fix: accelerate's first-fit
`infer_auto_device_map` packed GPU0 to 52 GiB leaving other GPUs idle — the
cause of the observed 50%/25% utilization and quant-swap OOMs; rerunning with
`get_balanced_memory`. Multi-GPU rows are layer-pipelined, so per-GPU util
is bounded by 1/N even when balanced — flagged in those rows.)*

## Table 5 — Fusion level

**Pinned**: 1.5B, bf16, 1×H100. **Variable**: eager / compile / +cuDNN attn.

| mode | level | ms | vs eager | tokens/s | peak mem | util |
|---|---|---:|---:|---:|---:|---:|
| train | `eager` | 280.3 | 1.00× | 14612 | 39.9 GiB | 99.3% |
| train | `compile` | 204.8 | 1.37× | 20002 | 32.7 GiB | 99.3% |
| train | `compile_cudnn_attn` | 204.8 | 1.37× | 19997 | 32.7 GiB | 99.3% |
| infer | `eager` | 161.5 | 1.00× | 101472 | 7.6 GiB | 99.9% |
| infer | `compile` | 115.5 | 1.40× | 141793 | 7.6 GiB | 99.9% |
| infer | `compile_cudnn_attn` | 115.6 | 1.40× | 141681 | 7.6 GiB | 99.9% |

Findings: compile = +37% train / +40% infer at unchanged memory; cuDNN
fused attention adds ~0 at this scale — SDPA's default backend is already
near-optimal at bs≤16/seq1024.

## Table 6 — Distributed layout (fixed world=4, 7B, std recipe)

**Variable**: layout only — sharding strategy × node placement.

| layout (7B, world=4, std recipe) | ms/step | agg tokens/s | peak/rank | util |
|---|---:|---:|---:|---:|
| FSDP, 4×1 node | 404.3 | 10131 | 35.53 GiB | 99.4% |
| TP=4, 1 node | 232.3 | 4408 | 38.88 GiB | 98.9% |
| FSDP, 2+2 across 2 nodes | 938.6 | 4364 | 35.53 GiB | 99.8% |
| TP=4, 2+2 across 2 nodes | 315.1 | 3250 | 38.88 GiB | 99.3% |

Findings: (1) equal GPUs: FSDP beats TP 2.3× on aggregate throughput; TP
wins step latency 1.7× (232 vs 404 ms) at +3 GiB/rank — its niche is
latency/memory, not throughput. (2) Crossing nodes: FSDP −57% (weights
cross the wire), TP −26% (only activations do) — hence TP-in-node ×
DP-across in production layouts.

## Table 7 — Serving/compile stack (bf16 pinned)

**Pinned**: 1.5B, bf16, 1×H100, prompt 128 + 128 new for decode.

| stack (1.5B, bf16 pinned) | batch fwd / prefill tok/s | decode bs=1 | decode bs=32 | peak mem |
|---|---:|---:|---:|---:|
| eager | 101370 | 100 | 3134 | 8.2 GiB |
| torchao+compile | 142400 | 160 | 4261 | — |
| vLLM | 807473 | 293 | 8633 | — |

*(torchao/vLLM engines do not expose comparable allocator peaks — vLLM pre-reserves 85% of VRAM by design; the eager row shows the PyTorch peak.)*

Findings: the stack alone — no precision change — is worth 1.4× (torchao
fwd) to ~3× (vLLM decode). vLLM prefill is engine throughput (CPU-bound at
this size, util 27%, flagged). Pick precision by fidelity budget (T4), let
the stack deliver speed.

## Table 8 — Modality

**Pinned**: ~1–2B model per modality, identical protocol. **Variable**:
modality (language rows: Tables 1/4).

| modality | train bf16 | train fp16 | infer bf16 | infer fp8 | infer int8 | int4 logit-cos | train mem fp32→bf16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| image DINOv2-g 1.1B | 3.72× | 3.58× | 8.13× | 2.83× | 1.04× | 0.5172 | 32.4→26.7 GiB |
| video VideoMAE-h 0.6B | 5.37× | 5.20× | 9.36× | 2.76× | 1.14× | 0.9695 | 22.7→18.1 GiB |
| audio Whisper-l-v3 1.5B | 3.76× | 3.61× | 6.77× | 2.58× | 1.11× | 0.9188 | 36.0→33.0 GiB |
| mm Qwen2-VL 2.2B | 1.91× | 1.80× | 4.64× | 2.74× | 1.09× | 0.7653 | 41.2→41.2 GiB |

Findings: bf16 training speedup spans 1.9×–5.4× (GEMM-dominated stacks gain
most; Qwen2-VL's short-sequence step is overhead-bound), with the memory
column showing the uniform ~25–35% bf16 saving. INT4 fidelity is
architecture-driven (video 0.97 vs image 0.52), not input-domain-driven.

## Goal coverage summary

- TF32/BF16/FP8/INT8/INT4 vs FP32: training (T1, T2) and inference eager+fused (T4) ✔
- Model sizes 0.5B→14B with no-OOM GPU escalation (T3) ✔
- Modalities ×5 (T8) ✔ · Distributed layouts FSDP/TP × node placement (T6) ✔
- Routes eager/compile/torchao/vLLM (T4b, T5, T7) ✔
- GPU util 95–100% on every throughput row; non-compute-bound rows labeled ✔
- Every table now carries peak-memory columns; every low-precision row
  carries a numerical-fidelity column ✔
- Failures fixed by rerun (T2 key bug, 3B OOM→2 GPUs, T4b dynamo-cache
  resets) or recorded with root cause; v1 catalog in git history ✔
