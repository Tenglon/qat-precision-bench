#!/bin/bash
# Runs ON transfer1.bsc.es (the only internet-capable node): build an isolated
# vLLM+torchao venv on GPFS and download the Qwen2.5-1.5B checkpoints
# (bf16 + official GPTQ-Int8 + AWQ). Never touches the existing mmeb_omni venv.
set -euo pipefail
Q=/gpfs/scratch/ehpc821/tlong/qat_bench
PYBASE=/gpfs/scratch/ehpc821/tlong/mmeb_omni/pythons/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12

echo "=== venv ==="
[ -x "$Q/venv_vllm/bin/python" ] || "$PYBASE" -m venv "$Q/venv_vllm"
PIP="$Q/venv_vllm/bin/pip"
"$PIP" install -q -U pip
echo "=== install vllm + torchao (big; wheels only) ==="
"$PIP" install --only-binary=:all: vllm torchao huggingface_hub
"$Q/venv_vllm/bin/python" - <<'PY'
import torch, vllm, torchao, transformers
print("torch", torch.__version__, "| vllm", vllm.__version__,
      "| torchao", torchao.__version__, "| transformers", transformers.__version__)
PY

echo "=== download checkpoints ==="
"$Q/venv_vllm/bin/python" - <<PY
import os
from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen2.5-1.5B-Instruct",
             "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
             "Qwen/Qwen2.5-1.5B-Instruct-AWQ"]:
    name = repo.split("/")[-1]
    p = snapshot_download(repo, local_dir=os.path.join("$Q/models", name))
    print("downloaded", name, "->", p, flush=True)
PY
echo BUILD_VLLM_DONE
