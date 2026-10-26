"""Per-run output directories: <root>/<script_name>/[<prefix>-]<uid>."""

import argparse
import functools
import pathlib
import sys
import uuid

from omegaconf import OmegaConf

OUTPUT_ROOT = pathlib.Path("/output")
_HYDRA_FLAGS = {"--log-dir-prefix": "log_dir_prefix", "--log-dir-name": "log_dir_name"}


@functools.cache
def process_uid() -> str:
  return uuid.uuid4().hex[:8]


def resolve_log_dir(
  script_name: str,
  prefix: str = "",
  name: str | None = None,
  root: pathlib.Path = OUTPUT_ROOT,
) -> pathlib.Path:
  """`name` replaces the whole run directory name; `prefix` is ignored then."""
  if name:
    return root / script_name / name
  return root / script_name / (f"{prefix}-{process_uid()}" if prefix else process_uid())


def add_log_dir_flags(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--log-dir-prefix",
    default="",
    help="Prefix for the generated run directory name (<prefix>-<uid>).",
  )
  parser.add_argument(
    "--log-dir-name",
    help="Exact run directory name under /output/<script_name>, instead of a uid.",
  )


def log_dir_from_args(script_name: str, args: argparse.Namespace) -> pathlib.Path:
  return resolve_log_dir(script_name, args.log_dir_prefix, args.log_dir_name)


def rewrite_hydra_argv(argv: list[str]) -> list[str]:
  """Turns `--log-dir-prefix X` / `--log-dir-name=X` into Hydra `key=X` overrides."""
  rewritten = []
  tokens = iter(argv)
  for token in tokens:
    flag, equals, value = token.partition("=")
    if flag in _HYDRA_FLAGS:
      if not equals:
        value = next(tokens, "")
      rewritten.append(f"{_HYDRA_FLAGS[flag]}={value}")
    else:
      rewritten.append(token)
  return rewritten


def install_hydra_support(script_name: str) -> None:
  """Registers `${run_dir:}` and rewrites sys.argv for the log-dir flags."""
  OmegaConf.register_new_resolver(
    "run_dir",
    lambda prefix="", name=None: str(resolve_log_dir(script_name, prefix, name)),
    replace=True,
  )
  sys.argv[1:] = rewrite_hydra_argv(sys.argv[1:])
