#!/usr/bin/env bash
# Usage: build.sh [prod|dev|all]. Needs ./run.sh build to have made the base image.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
target=${1:-all}
for name in prod dev; do
  [[ "$target" == all || "$target" == "$name" ]] || continue
  cp constraints.txt "$name/constraints.txt"
  docker build -t "dapper-$name" "$name"
  rm "$name/constraints.txt"
done
