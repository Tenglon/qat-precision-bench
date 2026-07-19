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
| `fp32` | 50.4 (1.00x) | 51.0 (1.00x) | 52.5 (1.00x) | 7.1 (1.00x) |
| `tf32` | 374.1 (7.43x) | 316.6 (6.21x) | 287.5 (5.48x) | 11.1 (1.56x) |
| `bf16` | 743.2 (14.75x) | 709.3 (13.91x) | 689.5 (13.15x) | 23.0 (3.23x) |
| `fp16` | 710.9 (14.11x) | 682.1 (13.38x) | 662.1 (12.62x) | 23.1 (3.25x) |
| `fp8` | 1170.9 (23.24x) | 1199.7 (23.53x) | 1027.5 (19.59x) | 44.0 (6.19x) |
| `int8` | 126.8 (2.52x) | 127.9 (2.51x) | 124.2 (2.37x) | 8.8 (1.23x) |
| `int4` | 29.4 (0.58x) | 29.8 (0.58x) | 28.3 (0.54x) | 27.0 (3.79x) |

## Training speedup vs FP32 (step time, higher is faster)

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| `tf32` | 2.38x | 2.41x | 1.89x | 1.96x | 1.70x |
| `bf16` | 3.89x | 3.73x | 5.36x | 3.74x | 1.92x |
| `fp16` | 3.75x | 3.59x | 5.18x | 3.58x | 1.80x |
| `fp8_train` | 2.25x | 2.02x | 2.55x | 1.96x | 1.23x |
| `fp8_qat` | 2.51x | 2.32x | 3.04x | 2.23x | 1.22x |
| `int8_qat` | 2.55x | 2.36x | 3.10x | 2.27x | 1.24x |
| `int4_qat` | 2.56x | 2.36x | 3.10x | 2.28x | 1.24x |

## Inference (batch forward) speedup vs FP32

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| `tf32` | 2.75x | 3.19x | 2.36x | 2.41x | 2.75x |
| `bf16` | 7.99x | 8.21x | 9.28x | 6.76x | 4.67x |
| `fp16` | 7.92x | 8.07x | 9.06x | 6.66x | 4.65x |
| `fp8` | 3.83x | 2.84x | 2.75x | 2.57x | 2.77x |
| `int8` | 1.28x | 1.04x | 1.13x | 1.11x | 1.09x |
| `int4` | 0.77x | 0.63x | 0.75x | 0.80x | 0.81x |

## Decode (lang, autoregressive) speedup vs FP32


### decode_bs1

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 1.00x | — | — | — | — |
| `tf32` | 1.01x | — | — | — | — |
| `bf16` | 1.07x | — | — | — | — |
| `fp16` | 1.05x | — | — | — | — |
| `fp8` | 0.44x | — | — | — | — |
| `int8` | 0.36x | — | — | — | — |
| `int4` | 1.17x | — | — | — | — |

### decode_bs32

| precision | lang | image | video | audio | mm |
|---|---|---|---|---|---|
| `fp32` | 1.00x | — | — | — | — |
| `tf32` | 1.25x | — | — | — | — |
| `bf16` | 1.91x | — | — | — | — |
| `fp16` | 1.90x | — | — | — | — |
| `fp8` | 0.88x | — | — | — | — |
| `int8` | 0.63x | — | — | — | — |
| `int4` | 1.57x | — | — | — | — |

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
| `int4` | 0.6520 | 0.5172 | 0.9695 | 0.9188 | 0.7653 |
