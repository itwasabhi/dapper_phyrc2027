#!/usr/bin/env bash
# Runs torchrl_smoke.py at several worker counts to compare collection throughput.
set -euo pipefail
dapper_dir=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
frames_per_batch=24

for num_workers in 1 2 3 4; do
  echo "=== num_workers=$num_workers ==="
  "$dapper_dir/run_sim.sh" python /project/dapper/src/rl/scripts/torchrl_smoke.py \
    environment=dressing environment.num_envs="$num_workers" \
    smoke.frames_per_batch="$frames_per_batch" smoke.iterations=50
done
