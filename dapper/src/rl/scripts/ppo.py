"""Trains a behavior with PPO; the behavior is chosen by the Hydra config.

Run: dapper/run_sim.sh python /project/dapper/src/rl/scripts/ppo.py [overrides]
Example: ... ppo.py ppo.total_frames=256 ppo.frames_per_batch=64 ppo.minibatch_size=32
"""

import json
import os
import pathlib
import subprocess

import hydra
from hydra.core import hydra_config
from omegaconf import OmegaConf
import torch

from rl import ppo_builder
from rl.envs import factory

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _git(*args: str):
  try:
    return subprocess.check_output(
      ["git", "-c", f"safe.directory={_REPO_ROOT}", "-C", str(_REPO_ROOT), *args],
      text=True,
      stderr=subprocess.DEVNULL,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def _source_metadata():
  commit = _git("rev-parse", "HEAD") or os.environ.get("PHYRC_SOURCE_COMMIT")
  status = _git("status", "--porcelain")
  dirty = (
    os.environ.get("PHYRC_SOURCE_DIRTY") == "1" if status is None else bool(status)
  )
  return {"source_commit": commit, "source_is_dirty": dirty}


@hydra.main(
  config_path="../configs", config_name="joint_grasp_trained0", version_base="1.3"
)
def main(config):
  run_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "run_metadata.json").write_text(
    json.dumps(
      {**_source_metadata(), "config": OmegaConf.to_container(config, resolve=True)},
      indent=2,
    )
  )
  torch.manual_seed(config.seed)
  env = factory.make_torchrl_env(config.environment, config.seed)
  trainer = ppo_builder.build_trainer(config.ppo, env, run_dir)
  trainer.train()
  ppo_builder.save_policy(trainer, run_dir)
  print(f"Saved run to {run_dir}", flush=True)
  factory.close_simulation_app(config.environment)


if __name__ == "__main__":
  main()
