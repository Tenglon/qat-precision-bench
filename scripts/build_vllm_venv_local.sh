#!/bin/bash
# Build the relocatable vLLM+torchao venv LOCALLY (transfer1 has no DNS), then
# ship with scripts/ship_vllm_venv.sh. Pattern follows
# /exp/tlong/scale26/bsc_stage/build_env.sh (uv relocatable venv + the same
# standalone CPython 3.12.13 base that already exists on BSC).
set -euo pipefail
STAGE=/exp/tlong/qat_stage
UV=/exp/tlong/scale26/bsc_stage/uvbin/uv
export UV_PYTHON_INSTALL_DIR=/exp/tlong/scale26/bsc_stage/pythons
VENV=$STAGE/venv_vllm
mkdir -p "$STAGE"

echo "=== create relocatable venv (python 3.12) ==="
rm -rf "$VENV"
"$UV" venv --relocatable --python 3.12 "$VENV"
PYBIN="$VENV/bin/python"

echo "=== install vllm + torchao (wheels only) ==="
"$UV" pip install --python "$PYBIN" --only-binary=:all: vllm torchao

echo "=== import smoke (CPU) ==="
"$PYBIN" - <<'PY'
import torch, torchao, transformers
print("torch", torch.__version__, "| cuda_build", torch.version.cuda)
print("torchao", torchao.__version__, "| transformers", transformers.__version__)
import importlib.metadata as md
print("vllm", md.version("vllm"))  # importing vllm needs a GPU env; metadata is enough here
PY

echo "=== tar ==="
tar -C "$STAGE" -czf "$STAGE/venv_vllm.tar.gz" venv_vllm
ls -lh "$STAGE/venv_vllm.tar.gz"
echo BUILD_LOCAL_DONE
