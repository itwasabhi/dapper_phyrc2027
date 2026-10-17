"""Sim-free gym env with DressingEnv's observation/action spaces, for tests and smoke runs."""

import gymnasium as gym
import numpy as np

NUM_ROBOTS = 2
NUM_JOINTS = 13


def observation_space(num_cameras: int, resolution: int) -> gym.spaces.Dict:
  inf = np.inf
  image = (num_cameras, resolution, resolution)
  return gym.spaces.Dict(
    {
      "joint_position": gym.spaces.Box(-inf, inf, (NUM_ROBOTS, NUM_JOINTS), np.float32),
      "joint_velocity": gym.spaces.Box(-inf, inf, (NUM_ROBOTS, NUM_JOINTS), np.float32),
      "gripper_close_command": gym.spaces.Box(0, 1, (NUM_ROBOTS, 1), bool),
      "previous_action": gym.spaces.Box(-1.0, 1.0, (NUM_ROBOTS, 9), np.float32),
      "simulation_time_s": gym.spaces.Box(-inf, inf, (), np.float64),
      "rgb": gym.spaces.Box(0, 255, (*image, 3), np.uint8),
      "depth": gym.spaces.Box(0.0, inf, (*image, 1), np.float32),
      "depth_valid": gym.spaces.Box(0, 1, (*image, 1), bool),
    }
  )


class FakeShirtGraspEnv(gym.Env):
  def __init__(
    self, num_cameras: int = 5, resolution: int = 16, max_episode_steps: int = 8
  ):
    self.observation_space = observation_space(num_cameras, resolution)
    self.action_space = gym.spaces.Box(-1.0, 1.0, (NUM_ROBOTS, 9), np.float32)
    self._max_episode_steps = max_episode_steps
    self._step_id = 0
    self._previous_action = np.zeros(self.action_space.shape, np.float32)
    self._rng = np.random.default_rng(0)

  def _observe(self):
    observation = self.observation_space.sample()
    observation["simulation_time_s"] = np.float64(self._step_id * 0.05)
    observation["rgb"][:] = self._rng.integers(0, 255, observation["rgb"].shape)
    observation["depth"][:] = self._rng.uniform(0.2, 5.0, observation["depth"].shape)
    observation["joint_position"][:] = self._rng.normal(
      size=observation["joint_position"].shape
    )
    observation["previous_action"] = self._previous_action
    return observation

  def reset(self, *, seed=None, options=None):
    if options is not None:
      raise ValueError("options unsupported")
    if seed is not None:
      self._rng = np.random.default_rng(seed)
    self._step_id = 0
    self._previous_action = np.zeros(self.action_space.shape, np.float32)
    return self._observe(), {}

  def step(self, action):
    self._step_id += 1
    self._previous_action = np.asarray(action, np.float32)
    truncated = self._step_id >= self._max_episode_steps
    reward = -float(np.abs(action).mean())
    return self._observe(), reward, False, truncated, {}
