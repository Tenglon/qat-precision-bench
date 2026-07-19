#!/usr/bin/env python3
"""Second-round analysis: scale sweep + route (eager/compile/torchao/vLLM)
comparison, with GPU-utilization proof columns and MFU. Writes
report/tables_scale_routes.md.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "report", "tables_scale_routes.md")

SCALES = [("0.5", "lang05", 0.49e9), ("1.5", "lang", 1.54e9),
          ("3", "lang3", 3.09e9), ("7", "lang7", 7.62e9)]
BF16_PEAK = 989e12   # H100 dense datasheet


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def rec_map(doc):
    return {(r["mode"], r["precision"]): r for r in doc["records"]} if doc else {}


def fmt_ratio(base, rec):
    if rec is None or not rec.get("ok"):
        return "OOM" if rec and "OutOfMemory" in str(rec.get("error", "")) else "—"
    if base is None or not base.get("ok"):
        return "n/a"
    return "%.2fx" % (base["ms_per_iter_median"] / rec["ms_per_iter_median"])


def scale_tables():
    docs = {s: rec_map(load(n)) for s, n, _ in SCALES}
    parts = []
    header = "| precision | " + " | ".join(f"{s}B" for s, _, _ in SCALES) + " |"
    sep = "|---|" + "---|" * len(SCALES)
    for mode, precs, title in [
        ("train", ["tf32", "bf16", "fp16", "fp8_train", "int8_qat"],
         "Training speedup vs FP32 (Qwen2.5 family; 7B = all-precision OOM, "
         "AdamW state alone exceeds 64 GB)"),
        ("infer", ["tf32", "bf16", "fp16", "fp8", "int8", "int4"],
         "Batch-forward inference speedup vs FP32 (bs=16, seq=1024)"),
        ("decode_bs1", ["bf16", "fp8", "int8", "int4"],
         "Decode bs=1 speedup vs FP32"),
        ("decode_bs32", ["bf16", "fp8", "int8", "int4"],
         "Decode bs=32 speedup vs FP32"),
    ]:
        parts.append(f"\n### {title}\n")
        parts.append(header)
        parts.append(sep)
        for p in precs:
            row = [f"`{p}`"]
            for s, _, _ in SCALES:
                m = docs[s]
                row.append(fmt_ratio(m.get((mode, "fp32")), m.get((mode, p))))
            parts.append("| " + " | ".join(row) + " |")
    # MFU for infer bf16/fp32 per scale
    parts.append("\n### Achieved model TFLOP/s and MFU (batch forward, ~2N FLOPs/token)\n")
    parts.append("| scale | precision | tokens/s | model TFLOP/s | MFU vs BF16 peak | gpu util | power |")
    parts.append("|---|---|---:|---:|---:|---:|---:|")
    for s, n, nparam in SCALES:
        m = rec_map(load(n))
        for p in ("fp32", "bf16", "fp8"):
            r = m.get(("infer", p))
            if r and r.get("ok") and r.get("tokens_per_s"):
                tf = 2 * nparam * r["tokens_per_s"] / 1e12
                parts.append("| %sB | `%s` | %d | %.0f | %.1f%% | %s%% | %sW |" % (
                    s, p, r["tokens_per_s"], tf, 100 * tf * 1e12 / BF16_PEAK,
                    r.get("gpu_util_avg"), r.get("power_w_avg")))
    return "\n".join(parts)


def route_tables():
    """eager vs +compile (route2) vs torchao (route1) vs vLLM (route3)."""
    parts = []
    # gather per-scale per-stack tok/s for infer(prefill) and decode
    stacks = {}

    def put(scale, stack, mode, variant, rec):
        stacks.setdefault((scale, mode), {})[(stack, variant)] = rec

    for s, n, _ in SCALES:
        d = load(n)
        if d:
            for r in d["records"]:
                if r.get("ok") and r.get("tokens_per_s"):
                    md = {"infer": "prefill", "decode_bs1": "decode_bs1",
                          "decode_bs32": "decode_bs32"}.get(r["mode"])
                    if md:
                        put(s, "eager", md, r["precision"], r)
    d = load("route2")
    if d:
        for r in d.get("model", []):
            if r.get("ok") and r.get("compiled"):
                md = "prefill" if r["mode"] == "infer" else r["mode"]
                put("1.5", "compile", md, r["precision"], r)
    for f in ("route1_small", "route1_large"):
        d = load(f)
        if d:
            for r in d["records"]:
                if r.get("ok"):
                    md = "prefill" if r["mode"] == "infer_bs16" else r["mode"]
                    put(r["scale_b"], "torchao", md, r["variant"], r)
    d = load("route3")
    if d:
        for r in d["records"]:
            if r.get("ok"):
                md = "prefill" if r["mode"] == "prefill_bs16" else r["mode"]
                put(r["scale_b"], "vllm", md, r["variant"], r)

    def cell(scale, mode, stack, variant):
        r = stacks.get((scale, mode), {}).get((stack, variant))
        if not r:
            return "—"
        u = r.get("gpu_util_avg")
        return "%d (%s%%)" % (round(r["tokens_per_s"]), u if u is not None else "?")

    variants = [("eager", "bf16"), ("eager", "fp8"), ("eager", "int8"), ("eager", "int4"),
                ("compile", "bf16"), ("compile", "fp8"), ("compile", "int8"),
                ("torchao", "bf16"), ("torchao", "fp8dyn"), ("torchao", "int8da"), ("torchao", "int4wo"),
                ("vllm", "bf16"), ("vllm", "fp8"), ("vllm", "int8"), ("vllm", "int4")]
    for mode, title in [("prefill", "Batch forward / prefill tokens-per-s (gpu util %)"),
                        ("decode_bs1", "Decode bs=1 tokens/s"),
                        ("decode_bs32", "Decode bs=32 tokens/s")]:
        parts.append(f"\n### {title}\n")
        parts.append("| stack / precision | " + " | ".join(f"{s}B" for s, _, _ in SCALES) + " |")
        parts.append("|---|" + "---|" * len(SCALES))
        for stack, v in variants:
            row = [f"{stack} `{v}`"]
            for s, _, _ in SCALES:
                row.append(cell(s, mode, stack, v))
            parts.append("| " + " | ".join(row) + " |")
    return "\n".join(parts)


def main():
    parts = ["# Scale sweep + acceleration-route tables (instrumented)\n",
             "Every throughput cell was measured with a 200 ms nvidia-smi sampler",
             "covering ONLY the timed window; `(NN%)` = average GPU utilization.",
             "Note `utilization.gpu` counts kernel-resident time — power draw and",
             "MFU are the compute-saturation indicators.\n",
             "## Scale sweep (stock PyTorch eager)\n",
             scale_tables(),
             "\n## Route comparison: eager vs inductor-compile vs torchao vs vLLM\n",
             route_tables()]
    with open(OUT, "w") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
