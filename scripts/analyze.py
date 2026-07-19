#!/usr/bin/env python3
"""Turn results/*.json into markdown speedup tables (FP32 = 1.00x baseline).

Runs on any python3 (no torch needed). Writes report/tables.md.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "report", "tables.md")

MODALITIES = ["lang", "image", "video", "audio", "mm"]
TRAIN_ORDER = ["fp32", "tf32", "bf16", "fp16",
               "fp8_train", "fp8_qat", "int8_qat", "int4_qat"]
INFER_ORDER = ["fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4"]


def load(modality):
    p = os.path.join(RES, modality + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def by_mode(records, mode):
    out = {}
    for r in records:
        if r.get("mode") == mode:
            out[r["precision"]] = r
    return out


def fmt_speedup(rec, base):
    if rec is None:
        return "—"
    if not rec.get("ok"):
        err = rec.get("error", "failed").split(":")[0]
        return "FAIL(%s)" % err
    if base is None or not base.get("ok"):
        return "?"
    s = base["ms_per_iter_median"] / rec["ms_per_iter_median"]
    return "%.2fx" % s


def table(mode, order, docs):
    lines = ["| precision | " + " | ".join(MODALITIES) + " |",
             "|---|" + "---|" * len(MODALITIES)]
    permod = {}
    for m in MODALITIES:
        d = docs.get(m)
        permod[m] = by_mode(d["records"], mode) if d else {}
    for p in order:
        row = ["`%s`" % p]
        for m in MODALITIES:
            recs = permod[m]
            row.append(fmt_speedup(recs.get(p), recs.get("fp32")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def raw_table(mode, order, docs, key, fmt="%.1f"):
    lines = ["| precision | " + " | ".join(MODALITIES) + " |",
             "|---|" + "---|" * len(MODALITIES)]
    for p in order:
        row = ["`%s`" % p]
        for m in MODALITIES:
            d = docs.get(m)
            rec = by_mode(d["records"], mode).get(p) if d else None
            if rec and rec.get("ok") and rec.get(key) is not None:
                row.append(fmt % rec[key])
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def gemm_table():
    p = os.path.join(RES, "gemm.json")
    if not os.path.exists(p):
        return "(gemm.json missing)"
    with open(p) as f:
        g = json.load(f)
    shapes = list(g["shapes"].keys())
    order = ["fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4"]
    lines = ["| precision | " + " | ".join(
        "%s TFLOP/s (x)" % s for s in shapes) + " |",
        "|---|" + "---|" * len(shapes)]
    for prec in order:
        row = ["`%s`" % prec]
        for s in shapes:
            e = g["shapes"][s].get(prec, {})
            b = g["shapes"][s].get("fp32", {})
            if e.get("ok"):
                sp = (b["us"] / e["us"]) if b.get("ok") else float("nan")
                row.append("%.1f (%.2fx)" % (e["tflops"], sp))
            else:
                row.append("FAIL")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    docs = {m: load(m) for m in MODALITIES}
    have = [m for m in MODALITIES if docs[m]]
    parts = []
    hdr = next((docs[m]["header"] for m in have), {})
    parts.append("# Measured tables\n")
    parts.append("GPU: **%s** (SM %s), torch %s, CUDA %s\n" % (
        hdr.get("gpu"), hdr.get("capability"), hdr.get("torch"), hdr.get("cuda")))
    for m in have:
        h = docs[m]["header"]
        tb = by_mode(docs[m]["records"], "train")
        ib = by_mode(docs[m]["records"], "infer")
        bs_t = next((r["batch_size"] for r in tb.values() if r.get("ok")), "?")
        bs_i = next((r["batch_size"] for r in ib.values() if r.get("ok")), "?")
        parts.append("- **%s**: %s (train bs=%s, infer bs=%s)" %
                     (m, h["model"], bs_t, bs_i))
    parts.append("\n## GEMM microbenchmark (hardware ceiling)\n")
    parts.append(gemm_table())
    parts.append("\n## Training speedup vs FP32 (step time, higher is faster)\n")
    parts.append(table("train", TRAIN_ORDER, docs))
    parts.append("\n## Inference (batch forward) speedup vs FP32\n")
    parts.append(table("infer", INFER_ORDER, docs))
    parts.append("\n## Decode (lang, autoregressive) speedup vs FP32\n")
    for mode in ("decode_bs1", "decode_bs32"):
        parts.append("\n### %s\n" % mode)
        parts.append(table(mode, INFER_ORDER, docs))
    parts.append("\n## Peak memory (GiB), train\n")
    parts.append(raw_table("train", TRAIN_ORDER, docs, "peak_mem_gib", "%.1f"))
    parts.append("\n## Logit cosine vs FP32 (inference quality proxy)\n")
    parts.append(raw_table("infer", INFER_ORDER, docs, "logit_cos_vs_fp32", "%.4f"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
