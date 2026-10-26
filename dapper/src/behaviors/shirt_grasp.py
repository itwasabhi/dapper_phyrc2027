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
  # If the side approach is blocked, swing it around the grasp point in these steps.
  swing_step_deg: float = 15.0
  max_swing_deg: float = 90.0


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
  robot_clearance_m: float = collision.CollisionConfig().robot_clearance_m,
) -> tuple[HemGrasp | None, HemGrasp | None]:
  """Grasps for the left and right hem corners; None for a corner with no clear base pose.

  Each robot approaches from the shirt's side; if obstacles leave no base pose within reach,
  the approach swings around the grasp point toward the hem side. The two base poses are kept
  `robot_clearance_m` apart.
  """
  points = get_hem_grasp_points(scene, config)
  if points is None:
    return None, None
  axis = np.array([math.cos(scene.shirt.yaw), math.sin(scene.shirt.yaw)])
  left = np.array([-axis[1], axis[0]])
  obstacles = collision.obstacles(scene)
  grasps = []
  # The left-corner robot stands further left and faces right, and vice versa.
  for point, outward in ((points[0], left), (points[1], -left)):
    other = grasps[0].base_pose if grasps and grasps[0] else None
    grasp = None
    for swing in _get_swing_angles(outward, -axis, config):
      direction = _rotate(outward, swing)
      grasp = _find_base_pose(
        point, -direction, obstacles, clearance_m, scene.shirt.confidence, config
      )
      if grasp is not None and other is not None:
        gap = math.hypot(grasp.base_pose.x - other.x, grasp.base_pose.y - other.y)
        if gap < robot_clearance_m:
          grasp = None
      if grasp is not None:
        break
    grasps.append(grasp)
  return tuple(grasps)


def _get_swing_angles(
  outward: np.ndarray, hem_side: np.ndarray, config: HemGraspConfig
) -> list[float]:
  """0 first, then growing swings, each trying the hem side of `outward` before the other."""
  hem_sign = 1.0 if _rotate(outward, 0.1) @ hem_side > 0 else -1.0
  angles = [0.0]
  step = math.radians(config.swing_step_deg)
  for count in range(1, round(config.max_swing_deg / config.swing_step_deg) + 1):
    angles += [hem_sign * count * step, -hem_sign * count * step]
  return angles


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
  cos_angle, sin_angle = math.cos(angle), math.sin(angle)
  return np.array(
    [
      cos_angle * vector[0] - sin_angle * vector[1],
      sin_angle * vector[0] + cos_angle * vector[1],
    ]
  )


def _find_base_pose(
  point: np.ndarray,
  facing: np.ndarray,
  obstacles,
  clearance_m: float,
  confidence: float,
  config: HemGraspConfig,
) -> HemGrasp | None:
  """Nearest base pose facing `facing` whose gripper reaches `point`, clear of obstacles."""
  heading = math.atan2(facing[1], facing[0])
  robot_left = np.array([-facing[1], facing[0]])
  for extension in np.arange(
    config.min_arm_extension_m, config.max_arm_extension_m + 1e-9, config.reach_step_m
  ):
    reach, _ = arm.get_gripper_center(0.0, float(extension), config.wrist_pitch)
    base_xy = point - reach * facing - arm.GRIPPER_ORIGIN_Y * robot_left
    if not any(
      collision.contains(rect, base_xy, clearance_m + config.base_margin_m)
      for rect in obstacles
    ):
      return HemGrasp(
        point_xy=point,
        base_pose=scene_lib.Pose2D(
          float(base_xy[0]), float(base_xy[1]), heading, 0.0, 0.0, confidence
        ),
      )
  return None


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
