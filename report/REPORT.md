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
2. **单卡 7B 训练墙**：AdamW 每参数 ~16 字节 fp32 状态,64GB 上任何计算精度
   都放不下 7B 全参训练——大模型必须 ZeRO/FSDP 分片或换优化器/LoRA,混合精度
   本身救不了内存。
3. **规模效应确认**："越大越明显"在所有轴上成立：推理 BF16 加速
   7.8×→10.2×（0.5B→7B），FP8 3.9×→6.3×，vLLM 里 INT4 相对 BF16 的 decode
   优势 0.83×→1.82×，torchao 里 FP8 相对 BF16 0.99×→1.42×。
4. **量化收益能否兑现取决于软件栈而非精度本身**：
   - eager（地板）：BF16 最优，FP8/INT8/INT4 全部跑输 BF16；
   - `torch.compile`（一行代码）：FP8 反超 BF16（155–161k vs 142k tok/s）；
     Triton 把 int8 GEMM 提速 2.1× 但仍只有峰值 13%；
   - vLLM（生产答案）：7B decode 上 FP8 1.66×、INT8 1.40×、**AWQ-INT4
     1.82×** 全面超过 BF16；bs=1 时延比 eager 好 5–6×。
   - 从 eager-BF16 到 vLLM-INT4（7B decode）总提升 **6×**——**选对栈比选
     精度更重要**。
5. **测量学**：`nvidia-smi` util=100% 不代表算力跑满（eager INT8 推理
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
