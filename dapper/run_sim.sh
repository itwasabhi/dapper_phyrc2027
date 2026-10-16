#!/usr/bin/env bash
# Runs upstream ./run.sh in a dapper image (DAPPER_IMAGE, default dapper-dev).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/output_link.sh"
export PHYRC_IMAGE=${DAPPER_IMAGE:-dapper-dev}
export PHYRC_EXTRA_DOCKER_ARGS="-v $assets_dir:/assets:ro -e PYTHONPATH=/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:/opt/isaac-compat:/project/dapper/src:/project/src/PhyRC_Sim:/scripts"
exec "$(dirname "${BASH_SOURCE[0]}")/../run.sh" "$@"
