import gymnasium as gym
import numpy as np
import torch
import pytest
from torchrl import envs

import utils_torchrl_env

NUM_CAMERAS = 5
RESOLUTION = 8


def _observation_space() -> gym.spaces.Dict:
  """Matches DressingEnv: every entry is a Box, including bools."""
  inf = np.inf
  return gym.spaces.Dict(
    {
      "joint_position": gym.spaces.Box(-inf, inf, (2, 13), np.float32),
      "gripper_close_command": gym.spaces.Box(0, 1, (2, 1), bool),
      "simulation_time_s": gym.spaces.Box(-inf, inf, (), np.float64),
      "rgb": gym.spaces.Box(0, 255, (NUM_CAMERAS, RESOLUTION, RESOLUTION, 3), np.uint8),
      "depth": gym.spaces.Box(
        -inf, inf, (NUM_CAMERAS, RESOLUTION, RESOLUTION, 1), np.float32
      ),
      "depth_valid": gym.spaces.Box(
        0, 1, (NUM_CAMERAS, RESOLUTION, RESOLUTION, 1), bool
      ),
    }
  )


class _FakeDressingEnv(gym.Env):
  """Mimics DressingEnv's spaces and its reset/step/info contract."""

  def __init__(self, max_episode_steps: int = 5):
    self.observation_space = _observation_space()
    self.action_space = gym.spaces.Box(-1.0, 1.0, (2, 9), np.float32)
    self._max_episode_steps = max_episode_steps
    self._step_id = 0
    self._rng = np.random.default_rng(0)

  def _observe(self):
    obs = self.observation_space.sample()
    obs["simulation_time_s"] = np.float64(self._step_id * 0.05)
    obs["rgb"][:] = self._rng.integers(0, 255, obs["rgb"].shape)
    return obs

  def reset(self, *, seed=None, options=None):
    if options is not None:
      raise ValueError("options unsupported")
    if seed is not None:
      self._rng = np.random.default_rng(seed)
    self._step_id = 0
    return self._observe(), {"source_commit": "abc", "termination_reason": None}

  def step(self, action):
    self._step_id += 1
    truncated = self._step_id >= self._max_episode_steps
    info = {"termination_reason": "time_limit" if truncated else None}
    return self._observe(), 0.0, False, truncated, info


def test_check_env_specs():
  envs.check_env_specs(utils_torchrl_env.make_env(_FakeDressingEnv()))


def test_rollout_keys_shapes_and_dtypes():
  env = utils_torchrl_env.make_env(_FakeDressingEnv())
  rollout = env.rollout(max_steps=3)
  assert rollout.batch_size == (3,)
  assert rollout["rgb"].shape == (3, NUM_CAMERAS, RESOLUTION, RESOLUTION, 3)
  assert rollout["rgb"].dtype == torch.uint8
  assert rollout["action"].shape == (3, 2, 9)
  assert rollout["next", "depth_valid"].dtype == torch.bool
  # torchrl promotes scalar observations to shape (1,).
  assert rollout["next", "simulation_time_s"].shape == (3, 1)
  assert rollout["next", "simulation_time_s"].dtype == torch.float64


def test_truncation_sets_done():
  env = utils_torchrl_env.make_env(_FakeDressingEnv(max_episode_steps=2))
  rollout = env.rollout(max_steps=10)
  assert rollout.batch_size == (2,)
  assert bool(rollout["next", "truncated"][-1])
  assert bool(rollout["next", "done"][-1])


def test_seed_is_repeatable():
  env = utils_torchrl_env.make_env(_FakeDressingEnv())
  env.set_seed(3)
  first = env.reset()["rgb"].clone()
  env.set_seed(3)
  second = env.reset()["rgb"]
  assert (first == second).all()
