"""solution-test1: scans the room, drives the robots to two corners of the shirt's table, then
moves them as a pair to the front of the manikin.

Set DAPPER_DEBUG_DIR (run_solution.py --debug does) so scan_room writes annotated frames and
scene_estimate.json under <debug-dir>/scan_room/.
"""

import os
import pathlib

from behaviors import base
from behaviors import drive_to_manikin
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
  corners = drive_to_shirt.DriveToShirtCorners(cfg=cfg)
  manikin = drive_to_manikin.DriveToManikin(cfg=cfg)

  return execution_utils.SequencedPolicy([scan, corners, manikin])
