# qat-precision-bench

**Precision speedups (FP32 = 1.0×) measured one variable at a time, with
numerical-fidelity columns — on H100 (MareNostrum 5, BSC).**

以 FP32 为基准,单变量控制实验逐表测量 TF32/BF16/FP16/FP8/INT8/INT4 的
训练/推理加速比,每张低精度表都附数值精度列;所有实验保证无 OOM
（放不下就翻倍卡数并在行内注明）。

📊 **Report: [report/REPORT.md](report/REPORT.md)** — one table per variable,
each discussed and extended iteratively.

## Methodology

1. Each table varies exactly ONE variable; all else pinned in the header.
2. No OOM rows — GPU count doubles until the row fits.
3. Low-precision rows carry numerics: training = 30-step update-direction
   cosine / rel-err vs FP32; inference = logit cosine vs FP32.
4. 5 warmup + 20 timed steps (median), in-window GPU util/power sampling.

## Layout

```
bench/quant.py                    QAT fake-quant + real fp8/int8/int4 linears (stock torch)
bench/models.py                   model builders (Qwen2.5 family + 4 modalities), synthetic batches
bench/gpuutil.py                  in-window nvidia-smi sampler
bench/table1_train_precision.py   Table 1: training compute precision
configs/                          vendored HF config.json files
results/                          measured JSONs (one per table)
report/REPORT.md                  the tables + findings
```

Earlier exploratory phase (multi-variable sweeps, routes, distributed,
optimizer studies) lives in git history before the `Report v2` commit.

## License

MIT
