#!/usr/bin/env bash
set -euo pipefail
cd -- "${GRASP_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")}"
export GRASP_CONFIG="${GRASP_CONFIG:-$PWD/profiles/evaluation_2048_3060.json}"
source runtime.sh
prefix="${1:-eval2048}"
[[ "$prefix" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid prefix" >&2; exit 2; }
# Same held-out rows and existing weights. No training commands are invoked.
bash run_logged.sh "${prefix}_preflight" python preflight_evaluation_budget.py --prefix "$prefix"
for arm in base sft grpo sft_grpo; do
  adapter_args=()
  if [[ "$arm" != base ]]; then
    run="runs/$arm"
    [[ "$arm" != sft_grpo ]] || run="runs/sft_grpo/grpo_stage"
    adapter=$(python audit_adapter.py "$run" --path-only)
    adapter_args=(--adapter "$adapter")
  fi
  bash run_logged.sh "${prefix}_${arm}_test" python evaluate_vllm.py \
    --name "${prefix}_${arm}" --split test "${adapter_args[@]}"
done
echo EVALUATION_BUDGET_COMPLETE
