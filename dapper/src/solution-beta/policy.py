"""solution-beta: spins like alpha while running open-vocab detection on the cameras.

Set DAPPER_DEBUG_DIR (run_solution.py --debug does) to write annotated frames there.
"""

import os
import pathlib

import numpy as np

from vision import debug_writer
from vision import detector_factory

BASE_YAW_COLUMN = 2
YAW_COMMAND = 0.1
DETECT_EVERY_N_STEPS = 4
POLICY_HZ = 20
DEBUG_DIR_ENV = "DAPPER_DEBUG_DIR"


def load_policy(checkpoint, observation_space, action_space):
  del checkpoint, observation_space

  action = np.zeros(action_space.shape, dtype=np.float32)
  action[:, BASE_YAW_COLUMN] = YAW_COMMAND
  detector = detector_factory.make_omdet_turbo()
  debug_dir = os.environ.get(DEBUG_DIR_ENV)

  class PerceptionSpinPolicy:
    def __init__(self):
      self._step = 0
      self.detections = []
      self._writer = None
      if debug_dir:
        self._writer = debug_writer.DebugWriter(
          pathlib.Path(debug_dir), POLICY_HZ / DETECT_EVERY_N_STEPS
        )

    def reset(self):
      self._step = 0
      self.detections = []

    def __call__(self, observation):
      if self._step % DETECT_EVERY_N_STEPS == 0:
        rgb_batch = observation["rgb"]
        self.detections = detector.detect(rgb_batch)
        if self._writer:
          self._writer.write(self._step, rgb_batch, self.detections)
      self._step += 1
      return action.copy()

    def close(self):
      if self._writer:
        self._writer.close()

  return PerceptionSpinPolicy()
