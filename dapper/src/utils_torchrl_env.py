"""Wraps a Gymnasium env as a TorchRL env."""

import gymnasium as gym
from torchrl import envs


def make_env(env: gym.Env) -> envs.TransformedEnv:
  """Wraps `env` in a GymWrapper plus a StepCounter."""
  return envs.TransformedEnv(
    envs.GymWrapper(env, device="cpu"),
    envs.StepCounter(),
  )
