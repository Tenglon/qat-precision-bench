#!/bin/bash
#SBATCH --job-name=qatbench
#SBATCH --account=ehpc688
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --array=0-5
#SBATCH --output=/gpfs/scratch/ehpc821/tlong/qat_bench/logs/%x-%A_%a.log

set -Eeuo pipefail
umask 077

readonly Q=/gpfs/scratch/ehpc821/tlong/qat_bench
readonly R=/gpfs/scratch/ehpc821/tlong/mmeb_omni
readonly PYTHON=$R/venv/bin/python   # read-only reuse of the existing venv

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QATBENCH_CFG_DIR=$Q/code/configs
export TOKENIZERS_PARALLELISM=false

TASKS=(gemm lang image video audio mm)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$Q/out"
nvidia-smi -L

if [[ "$T" == "gemm" ]]; then
  exec "$PYTHON" "$Q/code/bench/gemm_bench.py" --out "$Q/out/gemm.json"
else
  exec "$PYTHON" "$Q/code/bench/run_bench.py" --modality "$T" --out "$Q/out/$T.json"
fi
