#!/usr/bin/env bash
# Runs upstream ./run.sh in a dapper image (DAPPER_IMAGE, default dapper-dev).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/output_link.sh"
export PHYRC_IMAGE=${DAPPER_IMAGE:-dapper-dev}
exec "$(dirname "${BASH_SOURCE[0]}")/../run.sh" "$@"
