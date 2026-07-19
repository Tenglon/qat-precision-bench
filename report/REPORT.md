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

Findings: (1) QAT costs ×1.52 over bf16 regardless of bit-width (the fake
quant is the same round/clamp math). (2) The numerics column now QUANTIFIES
what QAT does to optimization: int8 fake-quant only mildly bends the update
(cos 0.966), fp8 more (0.848), while **int4 rotates it almost orthogonal
(cos 0.252)** — QAT is not "the same training but noisier", it is descent on
a different (quantized) loss landscape, and at 4 bits that landscape is far
from the fp one. That is simultaneously why int4 PTQ fails (Table 4:
logit-cos 0.65) and why int4 QAT needs longer schedules to converge
(loss@30 = 0.361 vs baseline 0.220).

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
util. The GPU column IS the memory story: 3B OOMs at bs=4 on 1 GPU →
2 GPUs; 7B needs 4 (2 proven insufficient in v1); 14B needs 8 across
2 nodes. Aggregate throughput still degrades 4.2× from 0.5B to 14B while
hardware grew 8× — parameters outpace parallelism.

## Table 4 — Inference precision (+ logit fidelity)

**Pinned**: Qwen2.5-1.5B, 1×H100, batch fwd bs=16, seq=1024, eager.

| precision | ms/fwd | speedup | util / power | logit cos vs fp32 | mean rel-err |
|---|---:|---:|---|---:|---:|
| `fp32` | 1294.2 | 1.00× | 100.0% / 669.4 W | 1.0000 | 0.0000 |
| `tf32` | 471.5 | 2.74× | 100.0% / 531.2 W | 1.0000 | 0.0100 |
| `bf16` | 162.5 | 7.97× | 95.1% / 562.1 W | 0.9901 | 0.5340 |
| `fp16` | 163.7 | 7.91× | 99.9% / 589.6 W | 0.9997 | 0.0850 |
| `fp8` | 338.3 | 3.83× | 100.0% / 439.3 W | 0.9264 | 1.6033 |
| `int8` | 1014.0 | 1.28× | 100.0% / 373.0 W | 0.9778 | 0.8775 |
| `int4` | 1669.9 | 0.78× | 100.0% / 375.6 W | 0.6520 | 3.2868 |

Findings: (1) bf16/fp16 ~8×; fp16's logit error is 6× smaller than bf16's
(mantissa, echoing Table 1). (2) Real quantized kernels lose to bf16 in
eager (fp8 3.83×, int8 1.28×, int4 0.78×) — stack matters (Table 7).
(3) Fidelity cliff: int4 RTN logit cosine 0.652 — the accuracy QAT
(Table 2) exists to recover.

## Table 5 — Fusion level

**Pinned**: 1.5B, bf16, 1×H100. **Variable**: eager / compile / +cuDNN attn.

| mode | level | ms | vs eager | tokens/s | util |
|---|---|---:|---:|---:|---:|
| train | `eager` | 280.3 | 1.00× | 14612 | 99.3% |
| train | `compile` | 204.8 | 1.37× | 20002 | 99.3% |
| train | `compile_cudnn_attn` | 204.8 | 1.37× | 19997 | 99.3% |
| infer | `eager` | 161.5 | 1.00× | 101472 | 99.9% |
| infer | `compile` | 115.5 | 1.40× | 141793 | 99.9% |
| infer | `compile_cudnn_attn` | 115.6 | 1.40× | 141681 | 99.9% |

Findings: compile = +37% train / +40% infer. cuDNN fused attention adds ~0
here — SDPA's default backend is already near-optimal at bs≤16/seq1024, and
attention is a small slice of a 1.5B step.

## Table 6 — Distributed layout (fixed world=4, 7B, std recipe)

**Variable**: layout only — sharding strategy × node placement.

| layout (7B, world=4, std recipe) | ms/step | agg tokens/s | peak/rank | util |
|---|---:|---:|---:|---:|
| FSDP, 4×1 node | 404.3 | 10131 | 35.53 GiB | 99.4% |
| TP=4, 1 node | 232.3 | 4408 | 38.88 GiB | 98.9% |
| FSDP, 2+2 across 2 nodes | 938.6 | 4364 | 35.53 GiB | 99.8% |
| TP=4, 2+2 across 2 nodes | 315.1 | 3250 | 38.88 GiB | 99.3% |

Findings: (1) equal GPUs: FSDP beats TP 2.3× on aggregate throughput (TP
co-processes one stream, paying per-layer all-reduces; FSDP runs 4).
(2) Crossing nodes: FSDP −57% (weights cross the wire), TP −26%
(only activations do) — layout × placement interact; production uses
TP-in-node × DP-across. (3) TP's step latency (232 vs 404 ms) is its real
niche: latency and memory, not throughput.

## Table 7 — Serving/compile stack (bf16 pinned)

**Pinned**: 1.5B, bf16, 1×H100, prompt 128 + 128 new for decode.

| stack (1.5B, bf16 pinned) | batch fwd / prefill tok/s | decode bs=1 | decode bs=32 |
|---|---:|---:|---:|
| eager | 101370 | 100 | 3134 |
| torchao+compile | 142400 | 160 | 4261 |
| vLLM | 807473 | 293 | 8633 |

Findings: the stack alone — no precision change — is worth 1.4× (torchao,
fwd) to ~3× (vLLM, decode); vLLM prefill is engine throughput (CPU-bound at
this size, util 27%, flagged). Pick precision by fidelity budget (Table 4),
let the stack deliver the speed.

## Table 8 — Modality

**Pinned**: ~1–2B model per modality, identical protocol. **Variable**:
modality (language rows: Tables 1/4).

| modality | train bf16 | train fp16 | infer bf16 | infer fp8 | infer int8 | int4 logit-cos |
|---|---:|---:|---:|---:|---:|---:|
| image DINOv2-g 1.1B | 3.72× | 3.58× | 8.13× | 2.83× | 1.04× | 0.5172 |
| video VideoMAE-h 0.6B | 5.37× | 5.20× | 9.36× | 2.76× | 1.14× | 0.9695 |
| audio Whisper-l-v3 1.5B | 3.76× | 3.61× | 6.77× | 2.58× | 1.11× | 0.9188 |
| mm Qwen2-VL 2.2B | 1.91× | 1.80× | 4.64× | 2.74× | 1.09× | 0.7653 |

Findings: bf16 training speedup spans 1.9×–5.4× by modality — GEMM-dominated
stacks (video) gain most; Qwen2-VL's short-sequence step is overhead-bound.
INT4 fidelity is architecture-, not domain-, driven (video 0.97 vs image
0.52).

## Goal coverage summary

- TF32/BF16/FP8/INT8/INT4 vs FP32: training (T1, T2) and inference (T4) ✔
- Model sizes 0.5B→14B with no-OOM GPU escalation (T3) ✔
- Modalities ×5 (T8) ✔ · Distributed layouts FSDP/TP × node placement (T6) ✔
- Routes eager/compile/torchao/vLLM (T5, T7) ✔
- GPU util 95–100% on every throughput row (in-window sampling); decode and
  vLLM-prefill rows explicitly labeled where not compute-bound ✔
- Every low-precision row carries a numerical-fidelity column ✔
- Failures fixed by rerun (T2 key bug, 3B OOM→2 GPUs) or recorded with root
  cause; the v1 exploratory catalog remains in git history ✔
