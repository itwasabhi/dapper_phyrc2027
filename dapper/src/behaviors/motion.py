"""Reusable pose/velocity helpers for behaviors that drive the base toward a Pose2D goal."""

import dataclasses
import math

import numpy as np

from vision import camera_geometry
from vision import scene as scene_lib

FORWARD_COLUMN = 0
LEFT_COLUMN = 1
YAW_COLUMN = 2


@dataclasses.dataclass(frozen=True)
class DriveConfig:
  position_gain: float = 1.0
  yaw_gain: float = 1.0
  position_tolerance_m: float = 0.15
  yaw_tolerance_rad: float = 0.1


def wrap_angle(angle: float) -> float:
  return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_from_base_pose(pose_wxyz: np.ndarray) -> float:
  rotation = camera_geometry.pose_matrix(pose_wxyz)[:3, :3]
  return math.atan2(rotation[1, 0], rotation[0, 0])


def drive_toward(
  current_xy: np.ndarray,
  current_yaw: float,
  goal: scene_lib.Pose2D,
  config: DriveConfig,
) -> np.ndarray:
  """Body-frame (forward, left, yaw) command in [-1, 1] toward a world-frame goal pose."""
  error_world = np.array([goal.x, goal.y]) - np.asarray(current_xy, dtype=float)
  cos_yaw, sin_yaw = math.cos(current_yaw), math.sin(current_yaw)
  forward_error = cos_yaw * error_world[0] + sin_yaw * error_world[1]
  left_error = -sin_yaw * error_world[0] + cos_yaw * error_world[1]
  yaw_error = wrap_angle(goal.yaw - current_yaw)
  command = np.array(
    [
      forward_error * config.position_gain,
      left_error * config.position_gain,
      yaw_error * config.yaw_gain,
    ],
    dtype=np.float32,
  )
  return np.clip(command, -1.0, 1.0)


def within_tolerance(
  current_xy: np.ndarray,
  current_yaw: float,
  goal: scene_lib.Pose2D,
  config: DriveConfig,
) -> bool:
  position_error = float(
    np.linalg.norm(np.array([goal.x, goal.y]) - np.asarray(current_xy))
  )
  yaw_error = abs(wrap_angle(goal.yaw - current_yaw))
  return (
    position_error <= config.position_tolerance_m
    and yaw_error <= config.yaw_tolerance_rad
  )
