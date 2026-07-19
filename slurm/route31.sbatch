#!/bin/bash
#SBATCH --job-name=qatroute31
#SBATCH --account=ehpc688
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --array=0-2
#SBATCH --output=/gpfs/scratch/ehpc821/tlong/qat_bench/logs/%x-%A_%a.log

set -Eeuo pipefail
umask 077
readonly Q=/gpfs/scratch/ehpc821/tlong/qat_bench
readonly PYTHON=$Q/venv_vllm/bin/python

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=$Q/inductor_cache
export VLLM_LOGGING_LEVEL=WARNING
export PATH="$Q/venv_vllm/bin:$PATH"   # flashinfer JIT needs ninja/cmake

mkdir -p "$Q/out" "$Q/inductor_cache"
nvidia-smi -L

case "$SLURM_ARRAY_TASK_ID" in
  0) exec "$PYTHON" "$Q/code/bench/route3_vllm_bench.py" \
        --models-dir "$Q/models" --scales 0.5,1.5,7 \
        --out "$Q/out/route3.json" ;;
  1) exec "$PYTHON" "$Q/code/bench/route1_torchao_bench.py" \
        --models-dir "$Q/models" --scales 0.5,1.5 \
        --out "$Q/out/route1_small.json" ;;
  2) exec "$PYTHON" "$Q/code/bench/route1_torchao_bench.py" \
        --models-dir "$Q/models" --scales 7 \
        --out "$Q/out/route1_large.json" ;;
esac
