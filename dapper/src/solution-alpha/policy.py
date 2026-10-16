"""solution-alpha: scans the room, then drives both robots to the shirt.

Set DAPPER_DEBUG_DIR (run_solution.py --debug does) so scan_room writes annotated frames and
scene_estimate.json under <debug-dir>/scan_room/.
"""

import os
import pathlib

from behaviors import base
from behaviors import drive_to_shirt
from behaviors import execution_utils
from behaviors import scan_room
from vision import camera_geometry
from vision import detector_factory
from vision import scene_integrator

DEBUG_DIR_ENV = "DAPPER_DEBUG_DIR"


def load_policy(checkpoint, observation_space, action_space):
  del checkpoint, observation_space

  cfg = base.BehaviorCfg()
  integrator = scene_integrator.BasicSceneIntegrator(
    detector=detector_factory.make_omdet_turbo(),
    mounts=camera_geometry.load_mounts(),
    config=scene_integrator.GridConfig(
      update_every_n_calls=scan_room.UPDATE_EVERY_N_STEPS
    ),
  )
  debug_dir = os.environ.get(DEBUG_DIR_ENV)
  scan = scan_room.ScanRoom(
    integrator,
    cfg=cfg,
    debug_dir=pathlib.Path(debug_dir) / "scan_room" if debug_dir else None,
  )
  drive = drive_to_shirt.DriveToShirt(cfg=cfg)

  class SequencedPolicy:
    def __init__(self, behaviors):
      self._behaviors = behaviors
      self._index = 0
      self._state = None

    def reset(self):
      self._index = 0
      self._state = execution_utils.create_initial_state()
      self._behaviors[0].reset(self._state)

    def __call__(self, observation):
      action, self._index = execution_utils.execute_sequence(
        self._behaviors, self._index, observation, self._state
      )
      return action

    def close(self):
      for behavior in self._behaviors:
        behavior.close()

  return SequencedPolicy([scan, drive])
