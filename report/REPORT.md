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


