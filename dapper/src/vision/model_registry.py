"""Pinned Hugging Face models and where they live on disk."""

import dataclasses
import pathlib
from typing import Mapping, Sequence

DEFAULT_MODELS_ROOT = pathlib.Path("/assets/models")


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  repo_id: str
  revision: str
  license: str
  allow_patterns: Sequence[str]


REGISTRY: Mapping[str, ModelSpec] = {
  "omdet_turbo": ModelSpec(
    repo_id="omlab/omdet-turbo-swin-tiny-hf",
    revision="7fe93cecfb770c4d76cf71163956221249cab566",
    license="apache-2.0",
    allow_patterns=("*.json", "*.txt", "model.safetensors"),
  ),
  "grounding_dino_tiny": ModelSpec(
    repo_id="IDEA-Research/grounding-dino-tiny",
    revision="a2bb814dd30d776dcf7e30523b00659f4f141c71",
    license="apache-2.0",
    allow_patterns=("*.json", "*.txt", "model.safetensors"),
  ),
}


def local_path(
  name: str, models_root: pathlib.Path = DEFAULT_MODELS_ROOT
) -> pathlib.Path:
  """Directory of a downloaded model; raises if it has not been downloaded."""
  if name not in REGISTRY:
    raise KeyError(f"Unknown model {name!r}; known: {sorted(REGISTRY)}")
  path = models_root / name
  if not (path / "config.json").is_file():
    raise FileNotFoundError(
      f"{path} is missing. Download with: dapper/run_py.sh "
      f"/project/dapper/scripts/download_models.py {name}"
    )
  return path
