"""solution-alpha: scans the room, grasps the shirt's hem corners, and carries it to the manikin.

Set DAPPER_DEBUG_DIR (run_solution.py --debug does) so scan_room writes annotated frames and
scene_estimate.json under <debug-dir>/scan_room/.
"""

import os
import pathlib

from behaviors import base
from behaviors import down_and_pinch
from behaviors import drive_shirt_to_manikin
from behaviors import drive_to_shirt
from behaviors import execution_utils
from behaviors import scan_room
from behaviors import shirt_grasp
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
  drive = drive_to_shirt.DriveToShirtHemCorners(cfg=cfg)
  pinch = down_and_pinch.DownAndPinch(shirt_grasp.get_hem_grasp_targets, cfg=cfg)

  carry = drive_shirt_to_manikin.DriveShirtToManikin(cfg=cfg)

  return execution_utils.SequencedPolicy([scan, drive, pinch, carry])
