#!/usr/bin/env bash
# Runs torchrl_smoke.py at several worker counts to compare collection throughput.
set -euo pipefail
dapper_dir=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
frames_per_batch=24

for num_workers in 1 2 3 4; do
  echo "=== num_workers=$num_workers ==="
  "$dapper_dir/run_sim.sh" python /project/dapper/scripts/torchrl_smoke.py \
    --num-workers "$num_workers" --frames-per-batch "$frames_per_batch" --iterations 50
done
