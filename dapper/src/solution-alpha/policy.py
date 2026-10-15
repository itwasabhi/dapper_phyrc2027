"""solution-alpha: both robots rotate slowly in place."""

import numpy as np

BASE_YAW_COLUMN = 2
# 0.1 * 2.86 rad/s ~= 0.29 rad/s, about 22 s per revolution.
YAW_COMMAND = 0.1


def load_policy(checkpoint, observation_space, action_space):
  del checkpoint, observation_space

  action = np.zeros(action_space.shape, dtype=np.float32)
  action[:, BASE_YAW_COLUMN] = YAW_COMMAND

  class SpinPolicy:
    def reset(self):
      pass

    def __call__(self, observation):
      return action.copy()

  return SpinPolicy()
