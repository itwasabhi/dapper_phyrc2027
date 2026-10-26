"""Reusable pose/velocity helpers for behaviors that drive the base toward a Pose2D goal."""

import dataclasses
import math

import numpy as np

from vision import camera_geometry
from vision import scene as scene_lib

FORWARD_COLUMN = 0
LEFT_COLUMN = 1
YAW_COLUMN = 2
BASE_LINEAR_RATE = 0.616
BASE_ANGULAR_RATE = 2.86


@dataclasses.dataclass(frozen=True)
class DriveConfig:
  # Gains are in 1/s: commanded velocity (m/s, rad/s) = gain * error (m, rad).
  position_gain: float = 1.5
  yaw_gain: float = 2.0
  position_tolerance_m: float = 0.15
  yaw_tolerance_rad: float = 0.1


def wrap_angle(angle: float) -> float:
  return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_from_base_pose(pose_wxyz: np.ndarray) -> float:
  rotation = camera_geometry.pose_matrix(pose_wxyz)[:3, :3]
  return math.atan2(rotation[1, 0], rotation[0, 0])


def world_to_body(vector_world_xy: np.ndarray, yaw: float) -> np.ndarray:
  """Rotates a world-frame planar vector into the (forward, left) frame of a base at `yaw`."""
  cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
  x, y = vector_world_xy
  return np.array([cos_yaw * x + sin_yaw * y, -sin_yaw * x + cos_yaw * y])


def drive_toward(
  current_xy: np.ndarray,
  current_yaw: float,
  goal: scene_lib.Pose2D,
  config: DriveConfig,
) -> np.ndarray:
  """Body-frame (forward, left, yaw) command in [-1, 1] toward a world-frame goal pose."""
  error_world = np.array([goal.x, goal.y]) - np.asarray(current_xy, dtype=float)
  forward_error, left_error = world_to_body(error_world, current_yaw)
  yaw_error = wrap_angle(goal.yaw - current_yaw)
  command = np.array(
    [
      forward_error * config.position_gain / BASE_LINEAR_RATE,
      left_error * config.position_gain / BASE_LINEAR_RATE,
      yaw_error * config.yaw_gain / BASE_ANGULAR_RATE,
    ],
    dtype=np.float32,
  )
  return np.clip(command, -1.0, 1.0)


def formation_pose(
  xy_0: np.ndarray, xy_1: np.ndarray, yaw_0: float
) -> scene_lib.Pose2D:
  """Rigid-pair frame: origin at the bases' midpoint, heading of robot 0."""
  midpoint = (np.asarray(xy_0, dtype=float) + np.asarray(xy_1, dtype=float)) / 2
  return scene_lib.Pose2D(float(midpoint[0]), float(midpoint[1]), yaw_0, 0.0, 0.0, 1.0)


def compose(
  frame: scene_lib.Pose2D, local_xy: np.ndarray, local_yaw: float
) -> tuple[np.ndarray, float]:
  """World (xy, yaw) of a pose given in `frame`."""
  local_x, local_y = np.asarray(local_xy, dtype=float)
  cos_yaw, sin_yaw = math.cos(frame.yaw), math.sin(frame.yaw)
  rotated = np.array(
    [
      cos_yaw * local_x - sin_yaw * local_y,
      sin_yaw * local_x + cos_yaw * local_y,
    ]
  )
  return np.array([frame.x, frame.y]) + rotated, wrap_angle(frame.yaw + local_yaw)


def relative(
  frame: scene_lib.Pose2D, xy: np.ndarray, yaw: float
) -> tuple[np.ndarray, float]:
  """Inverse of `compose`: a world pose expressed in `frame`."""
  offset = np.asarray(xy, dtype=float) - np.array([frame.x, frame.y])
  return world_to_body(offset, frame.yaw), wrap_angle(yaw - frame.yaw)


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
