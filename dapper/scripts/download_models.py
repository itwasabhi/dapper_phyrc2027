"""Downloads pinned Hugging Face models into the assets dir (the only networked step).

Run: dapper/run_py.sh /project/dapper/scripts/download_models.py [name ...]
"""

import argparse
import os
import pathlib
import sys

# The images default to offline mode; this script is the one place that goes online.
os.environ["HF_HUB_OFFLINE"] = "0"

PROJECT_ROOT = pathlib.Path("/project")
if not (PROJECT_ROOT / "dapper/src").is_dir():
  PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dapper/src"))

import huggingface_hub  # noqa: E402

from vision import model_registry  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("names", nargs="*", help="Default: every registered model.")
  parser.add_argument(
    "--models-root", type=pathlib.Path, default=model_registry.DEFAULT_MODELS_ROOT
  )
  args = parser.parse_args()
  names = args.names or sorted(model_registry.REGISTRY)
  for name in names:
    spec = model_registry.REGISTRY[name]
    path = huggingface_hub.snapshot_download(
      repo_id=spec.repo_id,
      revision=spec.revision,
      allow_patterns=list(spec.allow_patterns),
      local_dir=args.models_root / name,
    )
    print(f"{name}: {spec.repo_id}@{spec.revision[:8]} -> {path}")


if __name__ == "__main__":
  main()
