"""solution-alpha: raises the wrist cameras, spins once and integrates a scene estimate.

Set DAPPER_DEBUG_DIR (run_solution.py --debug does) to write annotated frames and
scene_estimate.json there.
"""

import json
import math
import os
import pathlib

import numpy as np

from vision import camera_geometry
from vision import debug_writer
from vision import detector_factory
from vision import scene as scene_lib
from vision import scene_integrator

BASE_YAW_COLUMN = 2
LIFT_COLUMN = 3
WRIST_PITCH_COLUMN = 6
BASE_ANGULAR_RATE = 2.86
YAW_COMMAND = 0.2
LIFT_COMMAND = 1.0
# Sign needed to tilt the wrist camera toward the table tops.
WRIST_PITCH_COMMAND = -1.0
POLICY_HZ = 20
UPDATE_EVERY_N_STEPS = 4
RAISE_STEPS = 40
SPIN_MARGIN = 1.1
SPIN_STEPS = math.ceil(
  SPIN_MARGIN * 2 * math.pi / (YAW_COMMAND * BASE_ANGULAR_RATE) * POLICY_HZ
)
DEBUG_DIR_ENV = "DAPPER_DEBUG_DIR"


def load_policy(checkpoint, observation_space, action_space):
  del checkpoint, observation_space

  integrator = scene_integrator.BasicSceneIntegrator(
    detector=detector_factory.make_omdet_turbo(),
    mounts=camera_geometry.load_mounts(),
    config=scene_integrator.GridConfig(update_every_n_calls=UPDATE_EVERY_N_STEPS),
  )
  debug_dir = os.environ.get(DEBUG_DIR_ENV)

  class ScenePolicy:
    def __init__(self):
      self.scene = scene_lib.SceneEstimate()
      self._step = 0
      self._writer = None
      if debug_dir:
        self._writer = debug_writer.DebugWriter(
          pathlib.Path(debug_dir), POLICY_HZ / UPDATE_EVERY_N_STEPS
        )

    def reset(self):
      self._step = 0
      self.scene = scene_lib.SceneEstimate()
      integrator.reset()

    def __call__(self, observation):
      self.scene = integrator.update(observation)
      if self._writer and self._step % UPDATE_EVERY_N_STEPS == 0:
        self._writer.write(
          self._step,
          observation["rgb"],
          self.scene.debug_detections,
          self.scene.debug_image,
        )
      action = np.zeros(action_space.shape, dtype=np.float32)
      if self._step < RAISE_STEPS:
        action[:, LIFT_COLUMN] = LIFT_COMMAND
        action[:, WRIST_PITCH_COLUMN] = WRIST_PITCH_COMMAND
      elif self._step < RAISE_STEPS + SPIN_STEPS:
        action[:, BASE_YAW_COLUMN] = YAW_COMMAND
      self._step += 1
      return action

    def close(self):
      if self._writer:
        self._writer.close()
        out = pathlib.Path(debug_dir) / "scene_estimate.json"
        out.write_text(json.dumps(scene_lib.to_jsonable(self.scene), indent=2))

  return ScenePolicy()
