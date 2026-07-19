"""CPU smoke test: verify model construction, synthetic batches, forward,
backward, and the fake-quant QAT swap for all 5 modalities using 2-layer
configs (QATBENCH_TINY=1). Runs each modality in its own subprocess so peak
RSS stays within login-node limits; catches transformers-API mistakes before
burning a GPU allocation.

  python bench/smoke_cpu.py            # all modalities
  python bench/smoke_cpu.py --one lang # single modality (child mode)
"""
import os
import subprocess
import sys

os.environ["QATBENCH_TINY"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODALITIES = ["lang", "image", "video", "audio", "mm"]


def run_one(name):
    import torch
    from models import get_spec
    from quant import swap_linears

    spec = get_spec(name)
    model = spec.build()
    model.train()
    loss = model(**spec.make_batch(1, train=True)).loss
    loss.backward()
    assert torch.isfinite(loss), "non-finite loss"
    model.zero_grad(set_to_none=True)
    r, s = swap_linears(model, "int4_qat")
    loss2 = model(**spec.make_batch(1, train=True)).loss
    loss2.backward()
    assert torch.isfinite(loss2), "non-finite QAT loss"
    model.eval()
    with torch.inference_mode():
        out = model(**spec.make_batch(1, train=False))
    print(f"OK   {name:6s} loss={loss:.3f} qat_loss={loss2:.3f} "
          f"swapped={r} skipped={s} logits={tuple(out.logits.shape)}", flush=True)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        run_one(sys.argv[2])
        return
    ok = True
    for name in MODALITIES:
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--one", name], env=os.environ)
        if p.returncode != 0:
            ok = False
            print(f"FAIL {name} (rc={p.returncode})", flush=True)
    print("SMOKE_PASS" if ok else "SMOKE_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
