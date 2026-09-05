#!/usr/bin/env bash
set -euo pipefail
cd -- "${GRASP_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")}"
source runtime.sh
# Launch only after the smoke tests have been manually verified by the operator.
python preflight.py
python snapshot_environment.py
bash run_logged.sh base_test python evaluate_vllm.py --name base --split test
bash run_logged.sh sft_train bash run_sft.sh --output runs/sft
python audit_adapter.py runs/sft
sft_adapter=$(python audit_adapter.py runs/sft --path-only)
bash run_logged.sh sft_test python evaluate_vllm.py --name sft --adapter "$sft_adapter" --split test
bash run_logged.sh grpo_train bash run_grpo.sh --output runs/grpo
python audit_adapter.py runs/grpo
grpo_adapter=$(python audit_adapter.py runs/grpo --path-only)
bash run_logged.sh grpo_test python evaluate_vllm.py --name grpo --adapter "$grpo_adapter" --split test
bash run_logged.sh sft_grpo_train bash run_sft_grpo.sh
python audit_adapter.py runs/sft_grpo/grpo_stage --reference runs/sft_grpo/sft_stage
combined_adapter=$(python audit_adapter.py runs/sft_grpo/grpo_stage --path-only)
bash run_logged.sh sft_grpo_test python evaluate_vllm.py --name sft_grpo --adapter "$combined_adapter" --split test
bash finalize_results.sh
echo ALL_THREE_ARMS_COMPLETE
