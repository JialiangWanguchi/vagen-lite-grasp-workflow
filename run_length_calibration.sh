#!/usr/bin/env bash
set -euo pipefail
cd -- "${GRASP_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")}" 
export GRASP_CONFIG="${GRASP_CONFIG:-$PWD/profiles/length_calibration_2048_3060.json}"
source runtime.sh
prefix="${1:-lengthcal}"
[[ "$prefix" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid prefix" >&2; exit 2; }

# Hyperparameter selection uses validation only. Test remains sealed.
inputs=()
for arm in base sft grpo sft_grpo; do
  adapter_args=()
  if [[ "$arm" != base ]]; then
    run="runs/$arm"
    [[ "$arm" != sft_grpo ]] || run="runs/sft_grpo/grpo_stage"
    adapter=$(python audit_adapter.py "$run" --path-only)
    adapter_args=(--adapter "$adapter")
  fi
  name="${prefix}_${arm}"
  bash run_logged.sh "${name}_val" python evaluate_vllm.py \
    --name "$name" --split val "${adapter_args[@]}"
  inputs+=(--input "$arm=results/$name")
done
python calibrate_output_length.py "${inputs[@]}" --split val --quantum 512 \
  --output "results/${prefix}_recommendation.json"
echo LENGTH_CALIBRATION_COMPLETE
