"""drive_to_shirt: drives each robot to its own standoff pose beside the shirt scan_room found."""

import functools

import numpy as np

from behaviors import base
from behaviors import drive_independent
from behaviors import shirt_grasp
from vision import scene as scene_lib

Config = drive_independent.Config


def _shirt_goals(
  scene: scene_lib.SceneEstimate, base_pose: np.ndarray
) -> tuple[scene_lib.Pose2D | None, scene_lib.Pose2D | None]:
  # No shirt detected yields None goals, so the robots hold rather than drive blindly.
  return tuple(
    scene_lib.shirt_approach_pose(scene, tuple(base_pose[robot, :2]))
    for robot in range(2)
  )


def DriveToShirt(
  cfg: base.BehaviorCfg = base.BehaviorCfg(), config: Config = Config()
) -> drive_independent.DriveIndependent:
  return drive_independent.DriveIndependent(_shirt_goals, cfg=cfg, config=config)


def _get_corner_goals(
  scene: scene_lib.SceneEstimate, base_pose: np.ndarray
) -> tuple[scene_lib.Pose2D | None, scene_lib.Pose2D | None]:
  positions = base_pose[:, :2].astype(float)
  target = (
    (scene.manikin.x, scene.manikin.y)
    if scene.manikin is not None
    else tuple(positions.mean(axis=0))
  )
  corners = scene_lib.get_shirt_table_corner_poses(scene, target)
  if corners is None:
    return None, None

  def total_distance(order):
    return sum(
      np.linalg.norm(positions[robot] - [corners[index].x, corners[index].y])
      for robot, index in enumerate(order)
    )

  first, second = min(((0, 1), (1, 0)), key=total_distance)
  return corners[first], corners[second]


def DriveToShirtCorners(
  cfg: base.BehaviorCfg = base.BehaviorCfg(), config: Config = Config()
) -> drive_independent.DriveIndependent:
  return drive_independent.DriveIndependent(_get_corner_goals, cfg=cfg, config=config)


def _get_hem_corner_goals(
  scene: scene_lib.SceneEstimate,
  base_pose: np.ndarray,
  config: Config,
  grasp_config: shirt_grasp.HemGraspConfig,
) -> tuple[scene_lib.Pose2D | None, scene_lib.Pose2D | None]:
  grasps = shirt_grasp.get_hem_grasps(
    scene, config.collision_config.obstacle_clearance_m, grasp_config
  )
  goals = tuple(grasp.base_pose if grasp else None for grasp in grasps)
  goal_xy = tuple(np.array([g.x, g.y]) if g else None for g in goals)
  return shirt_grasp.assign_to_robots(goals, goal_xy, base_pose)


def DriveToShirtHemCorners(
  cfg: base.BehaviorCfg = base.BehaviorCfg(),
  config: Config = Config(),
  grasp_config: shirt_grasp.HemGraspConfig = shirt_grasp.HemGraspConfig(),
) -> drive_independent.DriveIndependent:
  """Each robot stands beside one hem corner of the shirt, facing across it, within arm reach."""
  return drive_independent.DriveIndependent(
    functools.partial(_get_hem_corner_goals, config=config, grasp_config=grasp_config),
    cfg=cfg,
    config=config,
  )
