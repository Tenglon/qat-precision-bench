# Scale sweep + acceleration-route tables (instrumented)

Every throughput cell was measured with a 200 ms nvidia-smi sampler
covering ONLY the timed window; `(NN%)` = average GPU utilization.
Note `utilization.gpu` counts kernel-resident time — power draw and
MFU are the compute-saturation indicators.

## Scale sweep (stock PyTorch eager)


### Training speedup vs FP32 (Qwen2.5 family; 7B = all-precision OOM, AdamW state alone exceeds 64 GB)

| precision | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| `tf32` | 1.95x | 2.38x | 2.35x | OOM |
| `bf16` | 3.98x | 3.89x | 3.19x | OOM |
| `fp16` | 3.91x | 3.75x | 3.02x | OOM |
| `fp8_train` | 2.22x | 2.25x | 2.00x | OOM |
| `int8_qat` | 2.95x | 2.55x | 2.04x | OOM |

### Batch-forward inference speedup vs FP32 (bs=16, seq=1024)

| precision | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| `tf32` | 1.97x | 2.75x | 3.04x | 3.77x |
| `bf16` | 7.78x | 7.99x | 8.74x | 10.16x |
| `fp16` | 7.69x | 7.92x | 8.59x | 10.05x |
| `fp8` | 3.85x | 3.83x | 4.45x | 6.32x |
| `int8` | 1.30x | 1.28x | 1.43x | 1.67x |
| `int4` | 1.15x | 0.77x | 0.73x | 0.64x |

### Decode bs=1 speedup vs FP32

| precision | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| `bf16` | 1.04x | 1.07x | 1.04x | 1.62x |
| `fp8` | 0.50x | 0.44x | 0.42x | 0.88x |
| `int8` | 0.45x | 0.36x | 0.34x | 0.43x |
| `int4` | 1.03x | 1.17x | 1.06x | 2.02x |

### Decode bs=32 speedup vs FP32

| precision | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| `bf16` | 1.20x | 1.91x | 2.14x | 3.19x |
| `fp8` | 0.65x | 0.88x | 0.98x | 2.14x |
| `int8` | 0.56x | 0.63x | 0.67x | 0.88x |
| `int4` | 1.07x | 1.57x | 1.54x | 1.41x |

### Achieved model TFLOP/s and MFU (batch forward, ~2N FLOPs/token)

| scale | precision | tokens/s | model TFLOP/s | MFU vs BF16 peak | gpu util | power |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | `fp32` | 29055 | 28 | 2.9% | 100.0% | 572.3W |
| 0.5B | `bf16` | 226055 | 222 | 22.4% | 99.9% | 454.6W |
| 0.5B | `fp8` | 111943 | 110 | 11.1% | 99.9% | 395.5W |
| 1.5B | `fp32` | 12661 | 39 | 3.9% | 100.0% | 660.7W |
| 1.5B | `bf16` | 101117 | 311 | 31.5% | 99.9% | 586.8W |
| 1.5B | `fp8` | 48460 | 149 | 15.1% | 100.0% | 439.8W |
| 3B | `fp32` | 6562 | 41 | 4.1% | 100.0% | 666.1W |
| 3B | `bf16` | 57384 | 355 | 35.9% | 100.0% | 610.6W |
| 3B | `fp8` | 29223 | 181 | 18.3% | 100.0% | 439.5W |
| 7B | `fp32` | 3172 | 48 | 4.9% | 100.0% | 690.4W |
| 7B | `bf16` | 32229 | 491 | 49.7% | 100.0% | 690.6W |
| 7B | `fp8` | 20042 | 305 | 30.9% | 100.0% | 502.6W |

## Route comparison: eager vs inductor-compile vs torchao vs vLLM


### Batch forward / prefill tokens-per-s (gpu util %)

| stack / precision | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| eager `bf16` | 226056 (99.9%) | 101117 (99.9%) | 57384 (100.0%) | 32230 (100.0%) |
| eager `fp8` | 111944 (99.9%) | 48460 (100.0%) | 29223 (100.0%) | 20043 (100.0%) |
| eager `int8` | 37754 (100.0%) | 16154 (100.0%) | 9360 (100.0%) | 5297 (100.0%) |
| eager `int4` | 33333 (100.0%) | 9800 (100.0%) | 4779 (100.0%) | 2029 (100.0%) |
| compile `bf16` | — | 141628 (99.9%) | — | — |
| compile `fp8` | — | 155220 (99.9%) | — | — |
| compile `int8` | — | 34541 (100.0%) | — | — |
| torchao `bf16` | 335816 (99.2%) | 142370 (99.8%) | — | 39488 (100.0%) |
| torchao `fp8dyn` | 329521 (96.5%) | 161069 (98.2%) | — | 55777 (99.2%) |
| torchao `int8da` | 207466 (96.5%) | 105385 (97.8%) | — | 32692 (99.2%) |
| torchao `int4wo` | — | — | — | — |
| vllm `bf16` | 970707 (19.0%) | 897500 (30.0%) | — | 479641 (100.0%) |
| vllm `fp8` | 975923 (16.0%) | 924008 (38.0%) | — | 651054 (86.0%) |
| vllm `int8` | 998890 (21.0%) | 829058 (43.0%) | — | 581400 (100.0%) |
| vllm `int4` | 944899 (11.0%) | 872323 (39.0%) | — | 695267 (99.0%) |

### Decode bs=1 tokens/s

| stack / precision | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| eager `bf16` | 107 (48.4%) | 96 (63.3%) | 69 (67.1%) | 69 (95.9%) |
| eager `fp8` | 52 (45.8%) | 40 (54.9%) | 28 (50.6%) | 37 (66.4%) |
| eager `int8` | 46 (75.8%) | 32 (97.8%) | 22 (98.5%) | 18 (99.0%) |
| eager `int4` | 105 (43.8%) | 105 (59.4%) | 70 (52.0%) | 86 (69.9%) |
| compile `bf16` | — | — | — | — |
| compile `fp8` | — | — | — | — |
| compile `int8` | — | — | — | — |
| torchao `bf16` | 205 (52.8%) | 165 (70.8%) | — | 77 (92.5%) |
| torchao `fp8dyn` | 122 (29.5%) | 115 (42.0%) | — | 92 (69.5%) |
| torchao `int8da` | — | — | — | — |
| torchao `int4wo` | — | — | — | — |
| vllm `bf16` | 588 (71.8%) | 292 (84.1%) | — | 92 (98.3%) |
| vllm `fp8` | 637 (86.7%) | 368 (89.0%) | — | 150 (97.5%) |
| vllm `int8` | 493 (76.8%) | 290 (86.4%) | — | 134 (99.9%) |
| vllm `int4` | 527 (71.8%) | 329 (93.5%) | — | 175 (94.5%) |

### Decode bs=32 tokens/s

| stack / precision | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| eager `bf16` | 3406 (54.1%) | 3053 (72.3%) | 2140 (76.0%) | 1912 (97.0%) |
| eager `fp8` | 1844 (51.6%) | 1411 (65.0%) | 985 (60.7%) | 1283 (80.9%) |
| eager `int8` | 1598 (82.5%) | 1002 (98.6%) | 673 (99.0%) | 526 (99.1%) |
| eager `int4` | 3054 (59.9%) | 2511 (91.0%) | 1537 (93.7%) | 846 (98.8%) |
| compile `bf16` | — | — | — | — |
| compile `fp8` | — | — | — | — |
| compile `int8` | — | — | — | — |
| torchao `bf16` | 6285 (69.1%) | 4277 (87.4%) | — | 1718 (95.1%) |
| torchao `fp8dyn` | 3060 (39.0%) | 2973 (57.7%) | — | 1908 (80.6%) |
| torchao `int8da` | — | — | — | — |
| torchao `int4wo` | — | — | — | — |
| vllm `bf16` | 16922 (75.8%) | 8650 (86.6%) | — | 2764 (97.1%) |
| vllm `fp8` | 18387 (90.2%) | 10814 (92.2%) | — | 4579 (95.4%) |
| vllm `int8` | 13506 (85.2%) | 8699 (86.7%) | — | 3876 (95.9%) |
| vllm `int4` | 14018 (80.4%) | 9717 (97.3%) | — | 5040 (92.6%) |
