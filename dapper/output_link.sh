# Sourced by the launchers. Points the repo's output/ at DAPPER_OUTPUT_DIR
# (default ~/data/dapper_phyrc), which upstream run.sh and run_py.sh mount at /output.
root=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${DAPPER_OUTPUT_DIR:-$HOME/data/dapper_phyrc}
mkdir -p "$output_dir"
if [[ ! -e "$root/output" && ! -L "$root/output" ]]; then
  ln -s "$output_dir" "$root/output"
elif [[ -L "$root/output" && "$(readlink -f "$root/output")" != "$(readlink -f "$output_dir")" ]]; then
  echo "warning: $root/output -> $(readlink -f "$root/output"), not $output_dir" >&2
elif [[ -d "$root/output" && ! -L "$root/output" ]]; then
  echo "warning: $root/output is a real directory; move it to $output_dir and symlink it" >&2
fi
