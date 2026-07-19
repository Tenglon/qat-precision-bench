# Precision speedup report: FP32 → TF32 / BF16 / FP16 / FP8 / INT8 / INT4

Measured on **1× NVIDIA H100 (MareNostrum 5 `acc` partition, BSC)**, torch
2.11.0+cu128, transformers 5.10.2, single GPU, eager mode (no `torch.compile`).
Five ~1B-parameter models, one per modality, random-initialized from their real
HF configs (throughput is architecture-determined; weight values don't matter).

> 结论速览（中文）见文末 [§8](#8-中文结论速览).

## 1. Question framing — what "QAT speedup" actually means

The request was "以 FP32 为基准，TF32 / BF16 / TF16 / FP8 / INT4 的 QAT 分别能加速多少".
Two clarifications define the whole report:

1. **"TF16" is not a real format** — we interpret it as **FP16** (IEEE half).
   The measured set is TF32, BF16, FP16, FP8 (E4M3/E5M2), INT8, INT4.
2. **QAT (quantization-aware training) does not speed up training.** QAT
   inserts *fake-quant* ops (quantize→dequantize with a straight-through
   estimator) into a full-precision compute graph, so QAT training is
   *slower* than its mixed-precision baseline — we measure exactly how much.
   The speedup QAT buys you is at **deployment**: the trained checkpoint can
   run with real INT8/INT4/FP8 kernels at inference with minimal accuracy
   loss. So "how much does each precision accelerate" decomposes into:
   - **(a) training compute precision** (TF32/BF16/FP16/FP8 mixed-precision
     training speedup vs FP32),
   - **(b) QAT overhead** (int8/int4/fp8 fake-quant training vs FP32 and vs
     BF16), and
   - **(c) quantized inference speedup** (real FP8/INT8/INT4 kernels vs FP32).

## 2. Hardware ceiling (H100 datasheet, dense, no sparsity)

| Precision | Engine | Peak TFLOP/s | Theoretical × vs FP32 |
|---|---|---:|---:|
| FP32 | CUDA cores | 67 | 1.0× |
| TF32 | Tensor Core | 495 | 7.4× |
| BF16 / FP16 | Tensor Core | 989 | 14.8× |
| FP8 (E4M3) | Tensor Core | 1979 | 29.5× |
| INT8 | Tensor Core | 1979 TOPS | 29.5× |
| INT4 | — | *no INT4 tensor core on Hopper* | n/a |

Two consequences to keep in mind when reading the measurements:

- End-to-end models never reach datasheet ratios: attention, normalization,
  optimizer steps, memory-bound elementwise ops, and Python/kernel launch
  overhead all dilute the GEMM speedup (Amdahl's law).
- **INT4 on Hopper is a memory-bandwidth play, not a compute play.** The
  tinygemm kernel (`_weight_int4pack_mm`) dequantizes INT4 weights into BF16
  tensor-core GEMMs; it wins only where weight loading dominates (small-batch
  autoregressive decode), and can *lose* at large batch.

## 3. Setup

| Item | Value |
|---|---|
| GPU | H100 (MN5 `acc`, 1 GPU/job via Slurm array) |
| Stack | Python 3.12.13, torch 2.11.0+cu128, transformers 5.10.2, eager |
| Kernels | stock torch only: `_scaled_mm` (FP8), `_int_mm` (INT8), `_weight_int4pack_mm` (INT4), autocast (BF16/FP16), TF32 flags |
| Models | Qwen2.5-1.5B / DINOv2-giant 1.1B / VideoMAE-huge 0.6B / Whisper-large-v3 1.5B / Qwen2-VL-2B |
| Train protocol | AdamW, fixed synthetic batch, 5 warmup + 20 timed optimizer steps, median step time; batch size chosen so FP32 (the memory high-water mark) fits |
| Infer protocol | batch forward, 5 warmup + 20 timed, median; quantized modes swap every eligible `nn.Linear` (≥256 dims, alignment-gated) |
| Decode protocol | lang only: greedy 128 new tokens, bs=1 and bs=32 |
| Quality proxy | logit cosine vs FP32 on a fixed probe batch |

Precision modes:

| Mode | Training | Inference |
|---|---|---|
| `fp32` | pure FP32, TF32 off | FP32 weights+compute |
| `tf32` | FP32 storage, TF32 tensor-core matmuls | same |
| `bf16` | autocast BF16 (fp32 master weights) | BF16 weights |
| `fp16` | autocast FP16 + GradScaler | FP16 weights |
| `fp8_train` | real FP8 GEMMs fwd (E4M3) + bwd (E5M2) via `_scaled_mm`, per-tensor dynamic scaling | — |
| `fp8` | — | dynamic per-tensor act ×  per-tensor weight E4M3 `_scaled_mm` |
| `int8_qat` / `int4_qat` / `fp8_qat` | fake-quant STE (per-channel weight, per-token act), autocast BF16 base | — |
| `int8` | — | W8A8 dynamic per-token, `_int_mm` |
| `int4` | — | weight-only groupwise (g=128) tinygemm |

## 4. GEMM microbenchmark (measured ceiling)

<!-- TABLE:GEMM -->

## 5. End-to-end training speedup vs FP32

<!-- TABLE:TRAIN -->

## 6. End-to-end inference speedup vs FP32

<!-- TABLE:INFER -->

### Decode (autoregressive, language model)

<!-- TABLE:DECODE -->

### Quality proxy (logit cosine vs FP32)

<!-- TABLE:QUALITY -->

## 7. Analysis

<!-- ANALYSIS -->

## 8. 中文结论速览

<!-- ZH_SUMMARY -->

## 9. Caveats

- Single GPU, eager mode. `torch.compile`, FlashAttention-3, TransformerEngine
  (delayed-scaling FP8), CUDA graphs, and fused optimizers would all shift the
  absolute numbers (generally *widening* low-precision gains by removing
  overhead that is precision-independent).
- FP8 training here is a minimal per-tensor dynamic-scaling implementation on
  linear layers only; TE with fused cast+transpose kernels does better.
- INT8 `_int_mm` requires M>16, so bs=1 decode pads the activation — stock
  torch has no INT8 GEMV; dedicated kernels (e.g. Marlin, machete) change that
  picture.
- Random-init weights: throughput identical to pretrained; the quality-proxy
  column is a *numerical fidelity* indicator, not task accuracy. Real QAT
  accuracy needs task training, out of scope for a speed study.
- Whisper/VideoMAE include non-Linear compute (convs) that stays in the base
  precision for quantized modes.
