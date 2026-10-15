"""Random-action sampling and output-directory handling for rollouts."""

import pathlib
import shutil

import numpy as np

NUM_ROBOTS = 2
ACTION_DIM = 9
GRIPPER_COLUMN = 8


def sample_action(
  rng: np.random.Generator,
  previous_action: np.ndarray | None,
  step: int,
  hold_steps: int,
  action_scale: float,
  gripper_resample_prob: float,
) -> np.ndarray:
  """Velocity columns are resampled every `hold_steps`; the gripper rarely."""
  action = np.zeros((NUM_ROBOTS, ACTION_DIM), np.float32)
  if previous_action is not None:
    action[:] = previous_action
  if previous_action is None or step % hold_steps == 0:
    velocities = rng.uniform(-1.0, 1.0, (NUM_ROBOTS, ACTION_DIM))
    action[:, :GRIPPER_COLUMN] = action_scale * velocities[:, :GRIPPER_COLUMN]
  resample = rng.random(NUM_ROBOTS) < gripper_resample_prob
  if previous_action is None:
    resample[:] = True
  action[resample, GRIPPER_COLUMN] = rng.choice([-1.0, 1.0], resample.sum())
  return action


def prepare_output_dir(output_dir: pathlib.Path, overwrite: bool) -> None:
  """Leaves `output_dir` absent; the dataset writer creates it."""
  if output_dir.exists() and any(output_dir.iterdir()):
    if not overwrite:
      raise FileExistsError(f"{output_dir} is not empty; pass --overwrite.")
    shutil.rmtree(output_dir)
  elif output_dir.exists():
    output_dir.rmdir()
