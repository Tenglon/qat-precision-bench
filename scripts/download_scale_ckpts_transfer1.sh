#!/bin/bash
# Runs ON transfer1.bsc.es: download the remaining Qwen2.5 scale-sweep
# checkpoints (0.5B/3B/7B x bf16/GPTQ-Int8/AWQ). Uses the existing mmeb_omni
# venv's huggingface_hub (independent of the venv_vllm build).
set -euo pipefail
Q=/gpfs/scratch/ehpc821/tlong/qat_bench
PY=/gpfs/scratch/ehpc821/tlong/mmeb_omni/venv/bin/python

"$PY" - <<PY
import os
from huggingface_hub import snapshot_download
repos = []
for s in ["0.5", "3", "7"]:
    for suf in ["", "-GPTQ-Int8", "-AWQ"]:
        repos.append(f"Qwen/Qwen2.5-{s}B-Instruct{suf}")
for repo in repos:
    name = repo.split("/")[-1]
    p = snapshot_download(repo, local_dir=os.path.join("$Q/models", name))
    print("downloaded", name, flush=True)
PY
echo DOWNLOAD_SCALE_DONE
