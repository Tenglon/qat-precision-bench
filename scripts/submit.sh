#!/bin/bash
# Submit the benchmark array on BSC (single ssh connection).
set -euo pipefail
BSC=uoa994647@alogin1.bsc.es
Q=/gpfs/scratch/ehpc821/tlong/qat_bench
ssh -F /dev/null -i "$HOME/.ssh/id_rsa" -o BatchMode=yes -o ConnectionAttempts=1 \
    -o ConnectTimeout=20 -o IdentitiesOnly=yes "$BSC" \
    "cd $Q && sbatch code/slurm/qat_bench_array.sbatch && squeue -u \$USER -h -n qatbench"
