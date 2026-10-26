"""Point-base collision checks against the scene's rectangular obstacles."""

import dataclasses
import math
from typing import Sequence

import numpy as np

from vision import scene as scene_lib


@dataclasses.dataclass(frozen=True)
class CollisionConfig:
  # Footprint inflation standing in for the base radius.
  obstacle_clearance_m: float = 0.25
  # Minimum distance between the two base points.
  robot_clearance_m: float = 0.6


def obstacles(scene: scene_lib.SceneEstimate) -> tuple[scene_lib.Pose2D, ...]:
  if scene.manikin is None:
    return tuple(scene.tables)
  return (*scene.tables, scene.manikin)


def _to_rect_frame(rect: scene_lib.Pose2D, point: np.ndarray) -> np.ndarray:
  offset = np.asarray(point, dtype=float) - np.array([rect.x, rect.y])
  cos_yaw, sin_yaw = math.cos(rect.yaw), math.sin(rect.yaw)
  return np.array(
    [
      cos_yaw * offset[0] + sin_yaw * offset[1],
      -sin_yaw * offset[0] + cos_yaw * offset[1],
    ]
  )


def _half_extents(rect: scene_lib.Pose2D, margin: float) -> np.ndarray:
  return np.array([rect.length / 2 + margin, rect.width / 2 + margin])


def contains(rect: scene_lib.Pose2D, point: np.ndarray, margin: float) -> bool:
  local = _to_rect_frame(rect, point)
  return bool(np.all(np.abs(local) <= _half_extents(rect, margin)))


def segment_hits_rect(
  start: np.ndarray, end: np.ndarray, rect: scene_lib.Pose2D, margin: float
) -> bool:
  """Liang-Barsky clip of the segment against the rect inflated by `margin`."""
  local_start = _to_rect_frame(rect, start)
  direction = _to_rect_frame(rect, end) - local_start
  half_extents = _half_extents(rect, margin)
  enter, leave = 0.0, 1.0
  for axis in range(2):
    if abs(direction[axis]) < 1e-12:
      if abs(local_start[axis]) > half_extents[axis]:
        return False
      continue
    t_low = (-half_extents[axis] - local_start[axis]) / direction[axis]
    t_high = (half_extents[axis] - local_start[axis]) / direction[axis]
    enter = max(enter, min(t_low, t_high))
    leave = min(leave, max(t_low, t_high))
    if enter > leave:
      return False
  return True


def path_is_clear(
  start: np.ndarray,
  end: np.ndarray,
  rects: Sequence[scene_lib.Pose2D],
  margin: float,
) -> bool:
  """True if the straight path avoids every rect; empty `rects` is always clear.

  A start already inside a rect's inflated box is tested against the raw box, so a base
  that begins within the clearance band can still leave it.
  """
  for rect in rects:
    effective_margin = 0.0 if contains(rect, start, margin) else margin
    if segment_hits_rect(start, end, rect, effective_margin):
      return False
  return True


def rects_overlap(
  first: scene_lib.Pose2D, second: scene_lib.Pose2D, margin: float
) -> bool:
  """Separating-axis test of two oriented rects, `second` inflated by `margin`."""
  inflated = dataclasses.replace(
    second, length=second.length + 2 * margin, width=second.width + 2 * margin
  )
  corners = (scene_lib.get_corners(first), scene_lib.get_corners(inflated))
  for rect in (first, inflated):
    for angle in (rect.yaw, rect.yaw + math.pi / 2):
      axis = np.array([math.cos(angle), math.sin(angle)])
      first_range, second_range = (
        [float(corner @ axis) for corner in rect_corners] for rect_corners in corners
      )
      if max(first_range) < min(second_range) or max(second_range) < min(first_range):
        return False
  return True
