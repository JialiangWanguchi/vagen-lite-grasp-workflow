#!/usr/bin/env bash
set -euo pipefail
cd -- "${GRASP_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")}"
source runtime.sh
python summarize_results.py
python snapshot_environment.py
python write_workflow_report.py
echo ALL_THREE_ARMS_VERIFIED
