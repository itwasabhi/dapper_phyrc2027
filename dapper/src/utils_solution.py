"""Loading and querying a deployable solution directory (policy.py layout)."""

import importlib.util
import pathlib

import numpy as np


def find_checkpoint(solution_dir: pathlib.Path) -> pathlib.Path | None:
  matches = sorted(solution_dir.glob("checkpoint.*"))
  return matches[0] if matches else None


def load_policy(
  solution_dir: pathlib.Path, checkpoint, observation_space, action_space
):
  spec = importlib.util.spec_from_file_location(
    "phyrc_submitted_policy", solution_dir / "policy.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  policy = module.load_policy(
    checkpoint=checkpoint,
    observation_space=observation_space,
    action_space=action_space,
  )
  if not callable(policy):
    raise TypeError("load_policy must return a callable policy(observation)")
  return policy


def validate_action(action, action_shape: tuple[int, ...]) -> np.ndarray:
  action = np.asarray(action, dtype=np.float32)
  if action.shape != action_shape:
    raise ValueError(f"Action shape {action.shape} != {action_shape}")
  if not np.all(np.isfinite(action)):
    raise ValueError("Action contains NaN or Inf")
  return np.clip(action, -1.0, 1.0)
