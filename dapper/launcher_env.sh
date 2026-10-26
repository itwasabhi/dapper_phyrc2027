# Sourced by the launchers. Resolves the host dirs mounted at /output and /assets.
root=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=$root/output
mkdir -p "$output_dir"
chmod g+rwx "$output_dir"
# Downloaded open models; mounted at /assets (see dapper/src/vision/MODELS.md).
assets_dir=${DAPPER_ASSETS_DIR:-$HOME/data/physrc_assets}
mkdir -p "$assets_dir"
