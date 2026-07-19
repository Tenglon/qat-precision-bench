# Precision speedup report: FP32 → TF32 / BF16 / FP16 / FP8 / INT8 / INT4

Measured on **1× NVIDIA H100 64GB (MareNostrum 5 `acc` partition, BSC)**,
torch 2.11.0+cu128, transformers 5.10.2, single GPU, eager mode (no
`torch.compile`). Five ~1B-parameter models, one per modality,
random-initialized from their real HF configs (throughput is
architecture-determined; weight values don't matter). Raw data:
[`../results/*.json`](../results), generated tables: [`tables.md`](tables.md).

> 中文结论速览见 [§8](#8-中文结论速览)。

## 1. Question framing — what "QAT speedup" actually means

The question was "以 FP32 为基准，TF32 / BF16 / TF16 / FP8 / INT4 的 QAT 分别能加速多少".
Two clarifications define the whole report:

1. **"TF16" is not a real format** — we interpret it as **FP16** (IEEE half).
   The measured set is TF32, BF16, FP16, FP8 (E4M3/E5M2), INT8, INT4.
2. **QAT (quantization-aware training) does not speed up training.** QAT
   inserts *fake-quant* ops (quantize→dequantize + straight-through estimator)
   into a full-precision compute graph, so QAT training is *slower* than its
   mixed-precision baseline — we measure exactly how much (§5). What QAT buys
   is **deployment speed at preserved accuracy**: the checkpoint can run real
   INT8/INT4/FP8 kernels at inference (§6) — our fidelity probe (§6.2) shows
   why naive post-training INT4 needs QAT in the first place. So "how much
   does each precision accelerate" decomposes into:
   - **(a)** mixed-precision **training** speedup (TF32/BF16/FP16/FP8 vs FP32),
   - **(b)** **QAT overhead** during training (int8/int4/fp8 fake-quant), and
   - **(c)** real quantized **inference** speedup (FP8/INT8/INT4 kernels).

## 2. Hardware ceiling (H100 datasheet, dense)

| Precision | Engine | Peak (SXM datasheet) | Theoretical × vs FP32 |
|---|---|---:|---:|
| FP32 | CUDA cores | 67 TFLOP/s | 1.0× |
| TF32 | Tensor Core | 495 TFLOP/s | 7.4× |
| BF16 / FP16 | Tensor Core | 989 TFLOP/s | 14.8× |
| FP8 (E4M3) | Tensor Core | 1979 TFLOP/s | 29.5× |
| INT8 | Tensor Core | 1979 TOPS | 29.5× |
| INT4 | — | *no INT4 tensor core on Hopper* | n/a |

Hopper dropped INT4 tensor cores (Ampere had them). INT4 on H100 is a
**memory-bandwidth play**: tinygemm dequantizes INT4 weights into BF16 tensor
cores, so it can only win where weight loading dominates (small-batch decode).

## 3. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA H100 64GB (63.1 GiB usable), SM 9.0, 1 GPU/job |
| Stack | Python 3.12.13, torch 2.11.0+cu128, transformers 5.10.2, eager |
| Kernels | stock torch only: `_scaled_mm` (FP8), `_int_mm` (INT8), `_weight_int4pack_mm` (INT4 tinygemm), autocast (BF16/FP16), TF32 flags |
| Train | AdamW, fixed synthetic batch, 5 warmup + 20 timed steps, median; bs sized so FP32 (memory high-water mark) fits |
| Infer | batch forward, 5 warmup + 20 timed, median; quantized modes swap every eligible `nn.Linear` (≥256 dims, alignment-gated, vocab-size heads excluded from INT4) |
| Decode | lang only: greedy, 128-token prompt + 128 new tokens, bs=1 / bs=32 |
| Quality | logit cosine vs FP32 on a fixed probe batch |

Models (all random-init from vendored configs):

| Modality | Model | Params | train bs | infer bs |
|---|---|---:|---:|---:|
| language | Qwen2.5-1.5B, seq 1024 | 1.5B | 4 | 16 |
| image | DINOv2-giant + cls head, 224px | 1.1B | 16 | 64 |
| video | VideoMAE-huge, 16×224px frames | 0.6B | 4 | 16 |
| audio | Whisper-large-v3, 30 s mel | 1.5B | 4 | 8 |
| multimodal | Qwen2-VL-2B, 224px img + 64 text tok | 2.2B | 4 | 8 |

## 4. GEMM microbenchmark — the measured ceiling

TFLOP/s (speedup vs FP32):

| precision | 4096³ | 8192³ | 16384×1536×8960 (LLM MLP) | 16×8192×8192 (decode-like) |
|---|---|---|---|---|
| `fp32` | 50.2 (1.00×) | 50.4 (1.00×) | 52.3 (1.00×) | 7.5 (1.00×) |
| `tf32` | 378.3 (**7.54×**) | 308.1 (6.12×) | 286.6 (5.48×) | 11.3 (1.52×) |
| `bf16` | 772.7 (**15.39×**) | 718.8 (14.27×) | 695.5 (13.30×) | 23.0 (3.09×) |
| `fp16` | 750.4 (14.95×) | 684.9 (13.60×) | 646.8 (12.37×) | 23.2 (3.12×) |
| `fp8` | 1181.0 (**23.53×**) | 1193.3 (23.69×) | 998.2 (19.09×) | 44.1 (5.92×) |
| `int8` | 126.8 (2.53×) | 128.1 (2.54×) | 124.1 (2.37×) | 8.8 (1.17×) |
| `int4` | 29.4 (0.59×) | 29.7 (0.59×) | 28.3 (0.54×) | 26.9 (**3.61×**) |

Findings:

- TF32/BF16/FP8 track the datasheet hierarchy almost perfectly (7.5× / 15.4× /
  23.5×; FP8 reaches 1.19 PFLOP/s ≈ 60% of dense peak with per-tensor scaling).
- **Stock-torch INT8 (`_int_mm`) is broken-slow on Hopper** (~127 TFLOP/s,
  2.5×): cuBLASLt's row-major int8 path does not engage the fast IMMA
  pipeline the way FP8 does. INT8's 29.5× datasheet peak is unreachable from
  stock eager PyTorch — you need CUTLASS/TensorRT/vLLM kernels.
- **INT4 tinygemm inverts with batch**: 0.59× (slower than FP32!) at large M,
  but 3.61× vs FP32 and **1.17× vs BF16** in the decode-like tall-skinny
  shape — exactly the memory-bound regime it was written for.

## 5. End-to-end training speedup vs FP32

Median optimizer-step time, same batch per column (higher = faster):

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `tf32` | 2.39× | 2.40× | 1.90× | 1.96× | 1.70× |
| `bf16` | **3.90×** | **3.71×** | **5.36×** | **3.75×** | **1.91×** |
| `fp16` | 3.76× | 3.57× | 5.17× | 3.60× | 1.80× |
| `fp8_train` (real FP8 GEMM) | 2.26× | 2.01× | 2.56× | 1.98× | 1.23× |
| `fp8_qat` (fake quant) | 2.52× | 2.31× | 3.04× | 2.22× | 1.22× |
| `int8_qat` (fake quant) | 2.56× | 2.35× | 3.11× | 2.27× | 1.24× |
| `int4_qat` (fake quant) | 2.56× | 2.35× | 3.11× | 2.27× | 1.24× |

Peak train memory (GiB, lang): fp32 51.5 → bf16 39.9 → fp8_train 36.8.

Findings:

- **BF16 mixed precision is the training workhorse: 3.7–5.4×** on
  GEMM-dominated models (1.9× on Qwen2-VL only because its benchmark sequence
  is short, so non-GEMM overhead dominates). FP16 ≈ BF16 (GradScaler costs
  ~3%); on H100 there is no reason to prefer FP16. TF32 gives ~2× for free
  (no code changes, no numerics risk beyond 10-bit mantissa).
- **QAT costs 1.5–1.7× over BF16** (e.g. lang 281 ms → 428 ms/step), landing
  at 2.2–3.1× vs FP32. INT8 and INT4 fake-quant cost the same (both are
  round+clamp in high precision) — bit-width doesn't change QAT overhead.
- Our minimal `fp8_train` (per-tensor dynamic scaling, unfused casts, fp8
  transposes in backward) is *slower* than BF16 — matching torchao/TE
  experience that FP8 training only pays off at ≥7B scale or with fused
  cast+transpose kernels (TransformerEngine, torch.compile).

## 6. End-to-end inference speedup vs FP32

### 6.1 Batch forward (compute-bound)

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `tf32` | 2.75× | 3.18× | 2.37× | 2.41× | 2.76× |
| `bf16` | **7.99×** | **8.10×** | **9.24×** | **6.75×** | **4.68×** |
| `fp16` | 7.92× | 7.94× | 8.98× | 6.66× | 4.67× |
| `fp8` | 3.83× | 2.83× | 2.76× | 2.56× | 2.78× |
| `int8` | 1.28× | 1.04× | 1.14× | 1.10× | 1.09× |
| `int4` | 0.78× | 0.63× | 0.76× | 0.80× | 0.75× |

### Decode (language, autoregressive, tokens/s)

| precision | bs=1 | bs=32 |
|---|---|---|
| `fp32` | 90 (1.00×) | 1612 (1.00×) |
| `tf32` | 90 (1.00×) | 2017 (1.25×) |
| `bf16` | 93 (1.03×) | **3134 (1.94×)** |
| `fp16` | 103 (1.14×) | 2957 (1.83×) |
| `fp8` | 39 (0.43×) | 1410 (0.87×) |
| `int8` | 33 (0.37×) | 1015 (0.63×) |
| `int4` | 95 (1.05×) | **2290 (1.42×)** |

### 6.2 Fidelity probe (logit cosine vs FP32 — why QAT exists)

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `tf32` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `bf16` | 0.9901 | 0.9983 | 1.0000 | 0.9999 | 0.9996 |
| `fp16` | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `fp8` | 0.9264 | 0.8927 | 0.9973 | 0.9935 | 0.9665 |
| `int8` | 0.9778 | 0.9705 | 0.9997 | 0.9991 | 0.9942 |
| `int4` | **0.6520** | **0.5172** | 0.9695 | 0.9188 | 0.7615 |

Findings:

- **BF16 weights are the best end-to-end inference precision in eager stock
  PyTorch: 4.7–9.2× over FP32.** Quantized modes lose to BF16 here because
  the dynamic per-token/per-tensor quantization runs as unfused eager ops,
  and (for INT8) the cuBLAS Hopper path is slow (§4). The FP8 GEMM's 1.5×
  kernel advantage over BF16 is real but is eaten by quantize/dequantize
  launches — recovering it requires kernel fusion (torch.compile, TE, vLLM,
  TRT-LLM), not a different algorithm.
- **INT4 weight-only is the only quantized mode that beats FP32 at decode
  (1.4× at bs=32) and it does so with zero dynamic-quant overhead** — but it
  still trails BF16 (0.73×) in this eager stack; its production 2–3× wins
  come from fused stacks where BF16 is equally accelerated and INT4's 4×
  smaller weight traffic dominates. bs=1 decode is completely
  launch-overhead-bound in eager mode (~90–100 tok/s regardless of
  precision) — precision is irrelevant there without CUDA graphs.
- The fidelity column is the motivation for QAT: naive round-to-nearest INT4
  collapses logit fidelity on 28–40-layer nets (0.52–0.76 cosine on
  lang/image/mm), INT8/FP8 mostly survive. QAT's job is to recover that lost
  accuracy during training — paying the 1.5–1.7× training overhead of §5 —
  so that the INT4/INT8 deployment speedups become usable.

## 7. Synthesis — the answer in one table

Speedup **vs FP32** on ~1B models, H100, stock PyTorch eager:

| Precision | GEMM ceiling | Training (measured) | Batch inference (measured) | Notes |
|---|---:|---:|---:|---|
| TF32 | 5.5–7.5× | **1.7–2.4×** | **2.4–3.2×** | free lunch, flip a flag |
| BF16 | 13–15× | **1.9–5.4×** (typ. ~3.8×) | **4.7–9.2×** | the default choice |
| FP16 | 12–15× | ≈ BF16 −3% | ≈ BF16 | no advantage on H100 |
| FP8 | 19–24× | 2.0–2.6× naive (needs TE/compile to beat BF16) | 2.6–3.8× eager | kernel is 1.5× BF16; fusion required to realize it |
| INT8 (QAT→deploy) | 2.5× (stock) / ~30× (theory) | QAT: 2.2–3.1× (= BF16 ÷ 1.5–1.7) | 1.0–1.3× eager | stock `_int_mm` can't use Hopper IMMA well |
| INT4 (QAT→deploy) | 3.6× at small M only | QAT: same as INT8 | 0.6–0.8× batch, **1.4× decode** | weight-only, memory-bound wins only |

Reading guide: *training* column answers "训练能加速多少"; *inference* column
answers "部署能加速多少"; QAT rows show that QAT itself is an accuracy
investment (×1.5–1.7 training cost), not a training accelerator.

## 8. 中文结论速览

以 FP32 为基准，单卡 H100（MareNostrum 5），~1B 模型五个模态实测（eager，未用
torch.compile）：

1. **训练加速**：TF32 约 **1.7–2.4×**（零代码成本）；BF16 混合精度约
   **1.9–5.4×**（典型 ~3.8×，是训练默认选择）；FP16 与 BF16 基本持平略低，
   H100 上没有理由再用 FP16；朴素 FP8 训练（逐张量动态缩放）只有 2.0–2.6×，
   反而不如 BF16 —— FP8 训练要靠 TransformerEngine/torch.compile 的算子融合
   才能在 ≥7B 规模兑现收益。
2. **QAT 不加速训练，反而有 1.5–1.7× 开销**（相对 BF16；INT4/INT8/FP8 假量化
   开销相同），折算下来约为 FP32 的 2.2–3.1×。QAT 的意义在 6.2 节的保真度表：
   直接 INT4 后训练量化会把 28–40 层网络的 logit 余弦相似度打到 0.52–0.76，
   必须用 QAT 在训练中补回精度，然后在部署端换取量化推理加速。
3. **推理加速（大 batch 前向）**：BF16 权重 **4.7–9.2×** 最优；eager 下
   FP8 只有 2.6–3.8×、INT8 约 1.1×、INT4 批量前向甚至 <1×——GEMM 内核收益
   （FP8 内核确实比 BF16 快 1.5×）被未融合的动态量化开销吃掉；INT8 还受
   stock cuBLAS 在 Hopper 上 int8 路径慢（仅 2.5×）拖累。
4. **INT4 的真正战场是小 batch 解码**（内存带宽受限）：内核级比 BF16 快
   1.17×，端到端 decode bs=32 比 FP32 快 1.42×。要拿到宣传中的 2–3× INT4
   部署加速，需要 vLLM / TensorRT-LLM / torchao+compile 这类融合栈，
   stock eager PyTorch 拿不到。
5. **GEMM 天花板（内核级）**：TF32 7.5×、BF16 15.4×、FP16 15.0×、FP8 23.5×
   （1.19 PFLOP/s）；Hopper 无 INT4 tensor core。
6. **显存**：BF16 训练峰值较 FP32 省约 23%（51.5→39.9 GiB，lang）。

一句话回答原问题：**训练看 BF16（~4×），想再多要靠 FP8+融合栈；QAT 是花
1.5–1.7× 训练时间买部署精度；部署端大 batch 用 BF16（~8×），小 batch 解码用
INT4 权重量化，INT8/FP8 在 stock PyTorch eager 下不划算。**

## 9. Caveats

- Single GPU, eager mode. `torch.compile`, FlashAttention-3, TransformerEngine
  (delayed-scaling FP8), CUDA graphs, and fused optimizers all shift absolute
  numbers — generally *widening* low-precision gains by removing
  precision-independent overhead. Multi-GPU adds communication that dilutes
  compute speedups further.
- FP8 training here is a minimal per-tensor dynamic-scaling implementation on
  linear layers only; TE with fused cast+transpose does substantially better.
- Stock `_int_mm` requires M>16, so bs=1 decode pads activations; dedicated
  INT8/INT4 GEMV kernels (Marlin, machete, TRT-LLM) change the decode picture.
- Random-init weights: throughput identical to pretrained; the fidelity probe
  is a numerical indicator on N(0, 0.02) weights, not task accuracy. Real QAT
  accuracy recovery requires task training — out of scope for a speed study.
- Whisper/VideoMAE/DINOv2 contain convs and other non-Linear compute that
  stays in base precision under all quantized modes.
- The INT4 vocab-head exclusion (N > 65536) mirrors deployment practice; the
  original run crashed tinygemm on the 151936×1536 lm_head.
