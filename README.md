# qat-precision-bench

**How much speedup do TF32 / BF16 / FP16 / FP8 / INT8 / INT4 give over FP32 —
measured end-to-end on ~1B-parameter models across five modalities, on one
NVIDIA H100 (MareNostrum 5, BSC).**

以 FP32 为基准，实测 TF32 / BF16 / FP16 / FP8 / INT8 / INT4（含 QAT）在
语言 / 图像 / 视频 / 音频 / 多模态 五类 ~1B 模型上的训练与推理加速比。

📊 **Results & analysis: [report/REPORT.md](report/REPORT.md)** (English + 中文结论)

## Headline results (speedup vs FP32, H100)

| Precision | GEMM ceiling | Training e2e | Infer floor (eager) | Infer realized (compile / vLLM) |
|---|---:|---:|---:|---|
| TF32 | 5.5-7.5x | 1.7-2.4x | 2.4-3.2x | - |
| BF16 | 13-15x | 1.9-5.4x (typ ~3.8x) | 4.7-10.2x | stack reference |
| FP16 | 12-15x | = BF16 | = BF16 | - |
| FP8 | 19-24x | 2.0-2.6x (naive) | 2.6-3.8x | beats BF16 w/ compile; 1.66x over BF16 in vLLM 7B decode |
| INT8 | 2.5x (stock torch) | QAT: 2.2-3.1x | 1.0-1.3x | 1.40x over BF16 (vLLM 7B decode) |
| INT4 | 3.6x (small M) | QAT: 2.2-3.1x | 0.6-0.8x | 1.82x over BF16 (vLLM AWQ 7B decode) |

Scale sweep (Qwen2.5 0.5B->7B) and route comparison (eager vs torch.compile
vs torchao vs vLLM) show low-precision benefit **grows with model scale** and
that **stack choice dwarfs precision choice** (eager-BF16 -> vLLM-INT4 = 6x
at 7B decode). Every measurement carries in-window GPU util/power proof.

QAT itself never accelerates training (it costs 1.5–1.7× vs BF16); its payoff
is quantized deployment. Full nuance in the report.

## What is measured

| Axis | Detail |
|---|---|
| GEMM ceiling | `gemm_bench.py`: raw matmul TFLOP/s per precision (the hardware upper bound) |
| Training | optimizer-step throughput: `fp32` `tf32` `bf16` `fp16` (mixed precision), `fp8_train` (real FP8 GEMMs fwd+bwd via `torch._scaled_mm`), `fp8_qat` `int8_qat` `int4_qat` (fake-quant QAT — measures the *overhead* of QAT, since fake quant simulates low-bit in high precision) |
| Inference | batch-forward throughput: `fp32` `tf32` `bf16` `fp16` + real quantized kernels `fp8` (`_scaled_mm`), `int8` (`_int_mm`, W8A8 dynamic), `int4` (tinygemm `_weight_int4pack_mm`, weight-only g128) |
| Decode | (language model) autoregressive generation tokens/s at bs=1 / bs=32 |
| Quality proxy | logit cosine vs FP32 for every quantized inference mode |

**Key framing note (QAT).** QAT (quantization-aware training) itself does not
speed up training — fake-quant ops *add* overhead. Its payoff is that the
resulting checkpoint can be deployed with real INT8/INT4/FP8 kernels at
inference. This repo therefore reports (a) mixed-precision *training* speedups,
(b) QAT training *overhead*, and (c) real quantized *inference* speedups —
which together answer "how much does each precision accelerate."

## Models (all ~1B class, random-init from vendored HF configs)

| Modality | Model | Params |
|---|---|---|
| language | Qwen2.5-1.5B | 1.5B |
| image | DINOv2-giant + cls head | 1.1B |
| video | VideoMAE-huge (16 frames) | 0.6B |
| audio | Whisper-large-v3 | 1.5B |
| multimodal | Qwen2-VL-2B (image+text) | 2.2B |

Weights are random-initialized (`from_config`): throughput depends only on
architecture shapes, not weight values, so results match pretrained
checkpoints while requiring zero model downloads on the air-gapped cluster.

## No extra dependencies

Everything uses kernels that ship inside stock PyTorch ≥ 2.4 (tested on
2.11.0+cu128): `torch._scaled_mm` (FP8), `torch._int_mm` (INT8),
`aten._weight_int4pack_mm` (INT4 tinygemm). No TransformerEngine, torchao, or
bitsandbytes.

## Layout

```
bench/quant.py       QAT fake-quant + real fp8/int8/int4 linear layers (pure torch)
bench/models.py      5 modality model builders + synthetic batches
bench/run_bench.py   end-to-end train/infer/decode benchmark for one modality
bench/gemm_bench.py  GEMM-level ceiling per precision
configs/             vendored HF config.json files
slurm/               Slurm array job (BSC MareNostrum 5, H100, 1 GPU/task)
scripts/             stage / submit / collect / analyze helpers
results/             measured JSONs from the cluster
report/              REPORT.md — the write-up
```

## Run it

On any CUDA box with torch ≥ 2.4 + transformers:

```bash
python bench/gemm_bench.py --out out/gemm.json
python bench/run_bench.py --modality lang --out out/lang.json
```

On BSC via Slurm: `scripts/stage_to_bsc.sh`, then `scripts/submit.sh`
(array of 6 single-GPU jobs), then `scripts/collect.sh` and
`scripts/analyze.py`.

## License

MIT
