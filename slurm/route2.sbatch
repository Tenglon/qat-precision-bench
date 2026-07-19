#!/bin/bash
#SBATCH --job-name=qatroute2
#SBATCH --account=ehpc688
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scratch/ehpc821/tlong/qat_bench/logs/%x-%j.log

set -Eeuo pipefail
umask 077
readonly Q=/gpfs/scratch/ehpc821/tlong/qat_bench
readonly PYTHON=/gpfs/scratch/ehpc821/tlong/mmeb_omni/venv/bin/python

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QATBENCH_CFG_DIR=$Q/code/configs
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=$Q/inductor_cache

mkdir -p "$Q/out" "$Q/inductor_cache"
nvidia-smi -L
exec "$PYTHON" "$Q/code/bench/route2_compile_bench.py" --out "$Q/out/route2.json"
