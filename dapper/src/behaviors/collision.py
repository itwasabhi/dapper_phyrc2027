"""Point-base collision checks and paths around the scene's rectangular obstacles."""

import dataclasses
import heapq
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


def path_clear_of_point(
  start: np.ndarray, end: np.ndarray, point: np.ndarray, radius: float
) -> bool:
  """True if the segment keeps `radius` from `point`; a start already within it is exempt."""
  start, end, point = (np.asarray(v, dtype=float) for v in (start, end, point))
  if np.linalg.norm(point - start) < radius:
    return True
  segment = end - start
  length_squared = float(segment @ segment)
  fraction = (
    0.0
    if length_squared == 0.0
    else np.clip((point - start) @ segment / length_squared, 0, 1)
  )
  return float(np.linalg.norm(point - (start + fraction * segment))) >= radius


def plan_path(
  start: np.ndarray,
  goal: np.ndarray,
  rects: Sequence[scene_lib.Pose2D],
  margin: float,
  corner_margin: float = 0.05,
) -> list[np.ndarray] | None:
  """Shortest waypoint list (ending at `goal`) around rects inflated by `margin`, or None.

  Waypoints are corners of the rects inflated by `margin + corner_margin` (a visibility graph).
  """
  start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
  if path_is_clear(start, goal, rects, margin):
    return [goal]
  corners = [
    corner
    for rect in rects
    for corner in scene_lib.get_corners(
      dataclasses.replace(
        rect,
        length=rect.length + 2 * (margin + corner_margin),
        width=rect.width + 2 * (margin + corner_margin),
      )
    )
    if not any(contains(other, corner, margin) for other in rects)
  ]
  nodes = [start, *corners, goal]
  goal_index = len(nodes) - 1
  distances = [math.inf] * len(nodes)
  previous = [None] * len(nodes)
  distances[0] = 0.0
  queue = [(0.0, 0)]
  while queue:
    distance, index = heapq.heappop(queue)
    if index == goal_index:
      break
    if distance > distances[index]:
      continue
    for neighbor in range(1, len(nodes)):
      if neighbor == index or not path_is_clear(
        nodes[index], nodes[neighbor], rects, margin
      ):
        continue
      candidate = distance + float(np.linalg.norm(nodes[neighbor] - nodes[index]))
      if candidate < distances[neighbor]:
        distances[neighbor] = candidate
        previous[neighbor] = index
        heapq.heappush(queue, (candidate, neighbor))
  if math.isinf(distances[goal_index]):
    return None
  path, index = [], goal_index
  while index != 0:
    path.append(nodes[index])
    index = previous[index]
  return path[::-1]
