# Measured tables

GPU: **NVIDIA H100** (SM 9.0), torch 2.11.0+cu128, CUDA 12.8

- **lang**: Qwen2.5-1.5B (random init) (train bs=4, infer bs=16)
- **image**: DINOv2-giant 1.1B + cls head (random init) (train bs=16, infer bs=64)
- **video**: VideoMAE-huge 0.6B, 16 frames (random init) (train bs=4, infer bs=16)
- **audio**: Whisper-large-v3 1.5B (random init) (train bs=4, infer bs=8)
- **mm**: Qwen2-VL-2B (random init), 224px image + 64 text tokens (train bs=4, infer bs=8)

## GEMM microbenchmark (hardware ceiling)

| precision | 4096x4096x4096 TFLOP/s (x) | 8192x8192x8192 TFLOP/s (x) | 16384x1536x8960 TFLOP/s (x) | 16x8192x8192 TFLOP/s (x) |
|---|---|---|---|---|
| `fp32` | 50.2 (1.00x) | 50.4 (1.00x) | 52.3 (1.00x) | 7.5 (1.00x) |
| `tf32` | 378.3 (7.54x) | 308.1 (6.12x) | 286.6 (5.48x) | 11.3 (1.52x) |
| `bf16` | 772.7 (15.39x) | 718.8 (14.27x) | 695.5 (13.30x) | 23.0 (3.09x) |
| `fp16` | 750.4 (14.95x) | 684.9 (13.60x) | 646.8 (12.37x) | 23.2 (3.12x) |
| `fp8` | 1181.0 (23.53x) | 1193.3 (23.69x) | 998.2 (19.09x) | 44.1 (5.92x) |
| `int8` | 126.8 (2.53x) | 128.1 (2.54x) | 124.1 (2.37x) | 8.8 (1.17x) |
| `int4` | 29.4 (0.59x) | 29.7 (0.59x) | 28.3 (0.54x) | 26.9 (3.61x) |

## Training speedup vs FP32 (step time, higher is faster)

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| `tf32` | 2.40x | 2.40x | 1.90x | 1.96x | 1.70x |
| `bf16` | 3.90x | 3.71x | 5.36x | 3.75x | 1.91x |
| `fp16` | 3.76x | 3.57x | 5.17x | 3.60x | 1.80x |
| `fp8_train` | 2.26x | 2.01x | 2.56x | 1.98x | 1.23x |
| `fp8_qat` | 2.52x | 2.31x | 3.04x | 2.22x | 1.22x |
| `int8_qat` | 2.56x | 2.35x | 3.11x | 2.27x | 1.24x |
| `int4_qat` | 2.56x | 2.35x | 3.11x | 2.27x | 1.24x |

## Inference (batch forward) speedup vs FP32

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| `tf32` | 2.75x | 3.18x | 2.37x | 2.41x | 2.76x |
| `bf16` | 7.95x | 8.10x | 9.24x | 6.75x | 4.68x |
| `fp16` | 7.88x | 7.94x | 8.98x | 6.66x | 4.67x |
| `fp8` | 3.83x | 2.83x | 2.76x | 2.56x | 2.78x |
| `int8` | 1.28x | 1.04x | 1.14x | 1.10x | 1.09x |
| `int4` | 0.78x | 0.63x | 0.76x | 0.80x | 0.75x |

## Decode (lang, autoregressive) speedup vs FP32


### decode_bs1

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 1.00x | — | — | — | — |
| `tf32` | 1.00x | — | — | — | — |
| `bf16` | 1.03x | — | — | — | — |
| `fp16` | 1.14x | — | — | — | — |
| `fp8` | 0.43x | — | — | — | — |
| `int8` | 0.37x | — | — | — | — |
| `int4` | 1.05x | — | — | — | — |

### decode_bs32

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 1.00x | — | — | — | — |
| `tf32` | 1.25x | — | — | — | — |
| `bf16` | 1.94x | — | — | — | — |
| `fp16` | 1.83x | — | — | — | — |
| `fp8` | 0.87x | — | — | — | — |
| `int8` | 0.63x | — | — | — | — |
| `int4` | 1.42x | — | — | — | — |

## Peak memory (GiB), train

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 51.5 | 32.4 | 22.7 | 36.0 | 41.2 |
| `tf32` | 51.5 | 32.4 | 22.7 | 36.0 | 41.2 |
| `bf16` | 39.9 | 26.7 | 18.1 | 33.0 | 41.2 |
| `fp16` | 39.9 | 26.7 | 18.1 | 33.0 | 41.2 |
| `fp8_train` | 36.8 | 24.4 | 16.0 | 29.2 | 41.2 |
| `fp8_qat` | 40.2 | 27.2 | 18.8 | 33.9 | 41.2 |
| `int8_qat` | 40.2 | 27.2 | 18.8 | 33.9 | 41.2 |
| `int4_qat` | 40.2 | 27.2 | 18.8 | 33.9 | 41.2 |

## Logit cosine vs FP32 (inference quality proxy)

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `tf32` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `bf16` | 0.9901 | 0.9983 | 1.0000 | 0.9999 | 0.9996 |
| `fp16` | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `fp8` | 0.9264 | 0.8927 | 0.9973 | 0.9935 | 0.9665 |
| `int8` | 0.9778 | 0.9705 | 0.9997 | 0.9991 | 0.9942 |
| `int4` | 0.6520 | 0.5172 | 0.9695 | 0.9188 | 0.7615 |
