"""drive_to_manikin: moves both robots as a rigid pair to the standoff in front of the manikin."""

import math

import numpy as np

from behaviors import base
from behaviors import drive_formation
from behaviors import motion
from vision import scene as scene_lib

Config = drive_formation.Config


def _get_manikin_front_goal(
  scene: scene_lib.SceneEstimate, base_pose: np.ndarray
) -> scene_lib.Pose2D | None:
  manikin = scene.manikin
  if manikin is None:
    return None
  yaws = np.array([motion.yaw_from_base_pose(pose) for pose in base_pose])
  # Pair frame heading is robot 0's; offset the goal so the robots' mean heading faces the manikin.
  offsets = np.array([motion.wrap_angle(yaw - yaws[0]) for yaw in yaws])
  heading_offset = math.atan2(np.sin(offsets).mean(), np.cos(offsets).mean())
  forward = np.array([math.cos(manikin.yaw), math.sin(manikin.yaw)])
  position = np.array([manikin.x, manikin.y]) + scene_lib.FRONT_STANDOFF_M * forward
  return scene_lib.Pose2D(
    float(position[0]),
    float(position[1]),
    motion.wrap_angle(manikin.yaw + math.pi - heading_offset),
    0.0,
    0.0,
    manikin.confidence,
  )


def DriveToManikin(
  cfg: base.BehaviorCfg = base.BehaviorCfg(), config: Config = Config()
) -> drive_formation.DriveFormation:
  return drive_formation.DriveFormation(_get_manikin_front_goal, cfg=cfg, config=config)
