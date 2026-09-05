#!/usr/bin/env bash
set -euo pipefail
cd -- "${GRASP_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")}"
source runtime.sh
nproc=$(python experiment_config.py hardware.gpus_per_node)
nodes=$(python experiment_config.py hardware.nodes)
if [[ "$nodes" != 1 ]]; then
  echo 'Multi-node SFT requires an external torchrun rendezvous launcher; see MIGRATION.md.' >&2
  exit 2
fi
exec torchrun --standalone --nnodes=1 --nproc-per-node="$nproc" train_sft.py "$@"
