#!/bin/bash
# Check job status and pull finished result JSONs in ONE ssh connection:
# status goes to stderr (visible), tar of out/*.json streams over stdout.
set -euo pipefail
BSC=uoa994647@alogin1.bsc.es
Q=/gpfs/scratch/ehpc821/tlong/qat_bench
DST="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$DST"
ssh -F /dev/null -i "$HOME/.ssh/id_rsa" -o BatchMode=yes -o ConnectionAttempts=1 \
    -o ConnectTimeout=20 -o IdentitiesOnly=yes "$BSC" "
  { echo '== queue =='; squeue -u \$USER -n qatbench -h || true
    echo '== last log lines =='
    for f in $Q/logs/qatbench-*.log; do
      [ -e \"\$f\" ] && echo \"--- \$f\" && tail -n 3 \"\$f\"
    done
  } 1>&2
  cd $Q/out 2>/dev/null && ls *.json >/dev/null 2>&1 && tar -cf - *.json || true
" | { tar -xf - -C "$DST" 2>/dev/null || true; }
ls -la "$DST"
echo COLLECT_DONE
