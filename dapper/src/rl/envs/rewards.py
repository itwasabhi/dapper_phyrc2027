"""Reward functions: (previous_observation, action, observation) -> (reward, terminated)."""

from typing import Mapping, Tuple

import numpy as np

Observation = Mapping[str, np.ndarray]


def zero_reward(
  previous_observation: Observation, action: np.ndarray, observation: Observation
) -> Tuple[float, bool]:
  """Placeholder until a grasp reward is designed."""
  del previous_observation, action, observation
  return 0.0, False
