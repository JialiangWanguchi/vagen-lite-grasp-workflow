#!/usr/bin/env bash
set -euo pipefail
cd -- "${GRASP_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")}"
source runtime.sh
exec python train_sft_grpo.py "$@"
