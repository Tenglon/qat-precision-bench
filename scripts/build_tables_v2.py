#!/usr/bin/env python3
"""Regenerate report/REPORT.md table sections from results/table*.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    p = os.path.join(ROOT, 'results', f'{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def t2():
    d = load('table2')
    if not d or len(d['records']) < 4:
        return "*(pending)*"
    b = d['records'][0]
    rows = []
    for r in d['records']:
        sp = b['ms_per_step_median'] / r['ms_per_step_median']
        rows.append(f"| `{r['scheme']}` | {r['ms_per_step_median']:.1f} | {sp:.2f}× | "
                    f"{r['peak_mem_gib']:.1f} GiB | {r['gpu_util_avg']}% / {r['power_w_avg']} W | "
                    f"{r['update_cos_vs_bf16']:.6f} | {r['update_rel_err_vs_bf16']:.4f} | {r['loss_step30']:.4f} |")
    return ("| scheme | ms/step | vs bf16 | peak mem | util / power | ΔW cos vs bf16 | ΔW rel-err | loss@30 |\n"
            "|---|---:|---:|---:|---|---:|---:|---:|\n" + "\n".join(rows))


def t3():
    rows = []
    for m, label in [('lang05', '0.5B'), ('lang', '1.5B'), ('lang3', '3B'),
                     ('lang7', '7B'), ('lang14', '14B')]:
        d = load(f'table3_{m}')
        if not d:
            rows.append(f"| {label} | *(pending)* | | | | |")
            continue
        r = d['records'][0]
        rows.append(f"| {label} | {r['world_size']} | {r['global_tokens_per_step']} | "
                    f"{r['ms_per_step_median']:.1f} | {round(r['tokens_per_s_aggregate'])} | "
                    f"{r['peak_mem_gib_rank0']} GiB | {r['gpu_util_avg']}% |")
    return ("| model | GPUs | tokens/step | ms/step | agg tokens/s | peak/rank | util |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows))


def t4():
    d = load('table4')
    b = d['records'][0]
    rows = []
    for r in d['records']:
        sp = b['ms_per_iter_median'] / r['ms_per_iter_median']
        rows.append(f"| `{r['precision']}` | {r['ms_per_iter_median']:.1f} | {sp:.2f}× | "
                    f"{r['gpu_util_avg']}% / {r['power_w_avg']} W | "
                    f"{r['logit_cos_vs_fp32']:.4f} | {r['logit_mean_rel_err']:.4f} |")
    return ("| precision | ms/fwd | speedup | util / power | logit cos vs fp32 | mean rel-err |\n"
            "|---|---:|---:|---|---:|---:|\n" + "\n".join(rows))


def t5():
    d = load('table5')
    out = []
    for mode in ('train', 'infer'):
        base = next(r for r in d['records'] if r['mode'] == mode and r['level'] == 'eager')
        for r in d['records']:
            if r['mode'] != mode or not r.get('ok'):
                continue
            sp = base['ms_median'] / r['ms_median']
            out.append(f"| {mode} | `{r['level']}` | {r['ms_median']:.1f} | {sp:.2f}× | "
                       f"{round(r['tokens_per_s'])} | {r['gpu_util_avg']}% |")
    return ("| mode | level | ms | vs eager | tokens/s | util |\n"
            "|---|---|---:|---:|---:|---:|\n" + "\n".join(out))


def t6():
    rows = []
    for f, lab in [('table6_fsdp_1node', 'FSDP, 4×1 node'),
                   ('table6_tp_1node', 'TP=4, 1 node'),
                   ('table6_fsdp_2node', 'FSDP, 2+2 across 2 nodes'),
                   ('table6_tp_2node', 'TP=4, 2+2 across 2 nodes')]:
        d = load(f)
        if not d:
            rows.append(f"| {lab} | *(pending)* | | | |")
            continue
        r = d['records'][0]
        agg = r.get('tokens_per_s_aggregate') or r.get('tokens_per_s')
        pk = (r.get('peak_mem_gib_per_rank') or [None])[0]
        rows.append(f"| {lab} | {r['ms_per_step_median']:.1f} | {round(agg)} | "
                    f"{pk} GiB | {r['gpu_util_avg']}% |")
    return ("| layout (7B, world=4, std recipe) | ms/step | agg tokens/s | peak/rank | util |\n"
            "|---|---:|---:|---:|---:|\n" + "\n".join(rows))


def t7():
    er = {}
    d = load('table7_eager')
    for r in d['records']:
        if r.get('ok') and r['precision'] == 'bf16':
            er[r['mode']] = r
    rows = [f"| eager | {round(er['infer']['tokens_per_s'])} | "
            f"{round(er['decode_bs1']['tokens_per_s'])} | {round(er['decode_bs32']['tokens_per_s'])} |"]
    for f, lab in [('table7_torchao', 'torchao+compile'), ('table7_vllm', 'vLLM')]:
        d = load(f)
        recs = {r['mode']: r for r in d['records'] if r.get('ok') and r.get('variant') == 'bf16'}
        pre = recs.get('infer_bs16') or recs.get('prefill_bs16')
        rows.append(f"| {lab} | {round(pre['tokens_per_s'])} | "
                    f"{round(recs['decode_bs1']['tokens_per_s'])} | {round(recs['decode_bs32']['tokens_per_s'])} |")
    return ("| stack (1.5B, bf16 pinned) | batch fwd / prefill tok/s | decode bs=1 | decode bs=32 |\n"
            "|---|---:|---:|---:|\n" + "\n".join(rows))


def t8():
    rows = []
    for m, label in [('image', 'image DINOv2-g 1.1B'), ('video', 'video VideoMAE-h 0.6B'),
                     ('audio', 'audio Whisper-l-v3 1.5B'), ('mm', 'mm Qwen2-VL 2.2B')]:
        d = load(f'table8_{m}')
        recs = {(r['mode'], r['precision']): r for r in d['records'] if r.get('ok')}
        def sp(mode, p):
            b, r = recs.get((mode, 'fp32')), recs.get((mode, p))
            return f"{b['ms_per_iter_median']/r['ms_per_iter_median']:.2f}×" if b and r else "—"
        i4 = recs.get(('infer', 'int4'), {})
        rows.append(f"| {label} | {sp('train','bf16')} | {sp('train','fp16')} | "
                    f"{sp('infer','bf16')} | {sp('infer','fp8')} | {sp('infer','int8')} | "
                    f"{i4.get('logit_cos_vs_fp32', float('nan')):.4f} |")
    return ("| modality | train bf16 | train fp16 | infer bf16 | infer fp8 | infer int8 | int4 logit-cos |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows))


if __name__ == "__main__":
    for name, fn in [("T2", t2), ("T3", t3), ("T4", t4), ("T5", t5),
                     ("T6", t6), ("T7", t7), ("T8", t8)]:
        print(f"<<<{name}>>>")
        print(fn())
        print()
