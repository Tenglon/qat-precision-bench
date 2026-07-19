#!/bin/bash
# Stage the benchmark code to BSC MareNostrum 5. One rsync = one ssh
# connection (per the >=20-30s BSC login-node spacing convention used in
# ~/scale26). Submission is a separate, later connection: scripts/submit.sh.
set -euo pipefail
BSC=uoa994647@alogin1.bsc.es
Q=/gpfs/scratch/ehpc821/tlong/qat_bench
SRC="$(cd "$(dirname "$0")/.." && pwd)"

rsync -av --delete \
  -e "ssh -F /dev/null -i $HOME/.ssh/id_rsa -o BatchMode=yes -o ConnectTimeout=20 -o IdentitiesOnly=yes" \
  --exclude results --exclude report --exclude .git \
  "$SRC/" "$BSC:$Q/code/"
echo STAGE_DONE
