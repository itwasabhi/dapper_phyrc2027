#!/usr/bin/env bash
# Runs python (no Isaac sim, no run.sh lock) in a dapper image (DAPPER_IMAGE, default dapper-dev).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/output_link.sh"
exec docker run --rm --gpus all --network=host --user "1234:$(id -g)" \
  -e PYTHONPATH=/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:/opt/isaac-compat \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$root:/project:ro" -v "$root/output:/output:rw" -w /project \
  --entrypoint /isaac-sim/python.sh "${DAPPER_IMAGE:-dapper-dev}" "$@"
