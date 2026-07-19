#!/bin/bash
#SBATCH --job-name=qatscale
#SBATCH --account=ehpc688
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --array=0-2
#SBATCH --output=/gpfs/scratch/ehpc821/tlong/qat_bench/logs/%x-%A_%a.log

set -Eeuo pipefail
umask 077
readonly Q=/gpfs/scratch/ehpc821/tlong/qat_bench
readonly PYTHON=/gpfs/scratch/ehpc821/tlong/mmeb_omni/venv/bin/python

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QATBENCH_CFG_DIR=$Q/code/configs
export TOKENIZERS_PARALLELISM=false

TASKS=(lang05 lang3 lang7)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$Q/out"
nvidia-smi -L
exec "$PYTHON" "$Q/code/bench/run_bench.py" --modality "$T" --out "$Q/out/$T.json"
