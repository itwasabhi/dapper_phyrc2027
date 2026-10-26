"""Grasp points at the shirt's bottom (hem) corners and base poses beside them that reach them.

Each robot stands at the side of the shirt, facing across it, so its fingers (which close along
the base's y axis) close across the hem edge.
"""

import dataclasses
import math

import numpy as np

from behaviors import arm
from behaviors import collision
from vision import scene as scene_lib


@dataclasses.dataclass(frozen=True)
class HemGraspConfig:
  # Grasp point offsets from each hem corner: toward the collar, and toward the shirt's midline.
  inset_along_m: float = 0.0
  inset_across_m: float = 0.04
  # Gripper reach searched for a base pose clear of obstacles, as arm extension at full pitch.
  min_arm_extension_m: float = 0.08
  max_arm_extension_m: float = 0.45
  reach_step_m: float = 0.02
  # Extra clearance beyond the obstacle inflation for the base pose.
  base_margin_m: float = 0.05
  wrist_pitch: float = 1.1


@dataclasses.dataclass(frozen=True)
class HemGrasp:
  """World grasp point and the base pose that puts the gripper over it."""

  point_xy: np.ndarray
  base_pose: scene_lib.Pose2D


def get_hem_grasp_points(
  scene: scene_lib.SceneEstimate, config: HemGraspConfig = HemGraspConfig()
) -> tuple[np.ndarray, np.ndarray] | None:
  """World (x, y) grasp points near the left and right hem corners (facing the collar)."""
  if scene.shirt is None or scene.shirt_hem_corners is None:
    return None
  axis = np.array([math.cos(scene.shirt.yaw), math.sin(scene.shirt.yaw)])
  left = np.array([-axis[1], axis[0]])
  left_corner, right_corner = (
    np.asarray(c, dtype=float) for c in scene.shirt_hem_corners
  )
  along = config.inset_along_m * axis
  return (
    left_corner + along - config.inset_across_m * left,
    right_corner + along + config.inset_across_m * left,
  )


def get_hem_grasps(
  scene: scene_lib.SceneEstimate,
  clearance_m: float,
  config: HemGraspConfig = HemGraspConfig(),
) -> tuple[HemGrasp | None, HemGrasp | None]:
  """Grasps for the left and right hem corners; None for a corner with no clear base pose."""
  points = get_hem_grasp_points(scene, config)
  if points is None:
    return None, None
  left = np.array([-math.sin(scene.shirt.yaw), math.cos(scene.shirt.yaw)])
  obstacles = collision.obstacles(scene)
  grasps = []
  # The left-corner robot stands further left and faces right, and vice versa.
  for point, outward in ((points[0], left), (points[1], -left)):
    facing = -outward
    heading = math.atan2(facing[1], facing[0])
    robot_left = np.array([-facing[1], facing[0]])
    grasp = None
    for extension in np.arange(
      config.min_arm_extension_m, config.max_arm_extension_m + 1e-9, config.reach_step_m
    ):
      reach, _ = arm.get_gripper_center(0.0, float(extension), config.wrist_pitch)
      base_xy = point - reach * facing - arm.GRIPPER_ORIGIN_Y * robot_left
      if not any(
        collision.contains(rect, base_xy, clearance_m + config.base_margin_m)
        for rect in obstacles
      ):
        grasp = HemGrasp(
          point_xy=point,
          base_pose=scene_lib.Pose2D(
            float(base_xy[0]),
            float(base_xy[1]),
            heading,
            0.0,
            0.0,
            scene.shirt.confidence,
          ),
        )
        break
    grasps.append(grasp)
  return tuple(grasps)


def assign_to_robots(items: tuple, item_xy: tuple, base_pose: np.ndarray) -> tuple:
  """Orders a (left, right) pair so each robot gets the one that minimizes total distance."""
  positions = np.asarray(base_pose)[:, :2].astype(float)

  def total_distance(order):
    return sum(
      np.linalg.norm(positions[robot] - item_xy[index])
      for robot, index in enumerate(order)
      if item_xy[index] is not None
    )

  first, second = min(((0, 1), (1, 0)), key=total_distance)
  return items[first], items[second]


def get_hem_grasp_targets(
  scene: scene_lib.SceneEstimate,
  base_pose: np.ndarray,
  config: HemGraspConfig = HemGraspConfig(),
) -> tuple[np.ndarray | None, np.ndarray | None]:
  """Hem grasp points ordered by robot, each to the nearer robot."""
  points = get_hem_grasp_points(scene, config)
  if points is None:
    return None, None
  return assign_to_robots(points, points, base_pose)
