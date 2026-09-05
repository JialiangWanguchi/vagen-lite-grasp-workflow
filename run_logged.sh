#!/usr/bin/env bash
set -euo pipefail
cd -- "${GRASP_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")}"
name="$1"
shift
mkdir -p reports/logs
if [[ -e "reports/logs/${name}.log" ]]; then
  echo "Refusing to overwrite existing log: $name" >&2
  exit 2
fi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 2 > "reports/logs/${name}_gpu.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
date -u +%FT%TZ > "reports/logs/${name}_started.txt"
set +e
"$@" 2>&1 | tee "reports/logs/${name}.log"
status=${PIPESTATUS[0]}
set -e
echo "$status" > "reports/logs/${name}_exit_code.txt"
date -u +%FT%TZ > "reports/logs/${name}_finished.txt"
exit "$status"
