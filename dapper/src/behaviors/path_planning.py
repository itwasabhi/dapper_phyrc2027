"""Crude 2D planners for holonomic bases among the scene's rectangular obstacles.

`plan_point_path` is a visibility graph for one base; `plan_formation_path` is an RRT-Connect over
the pose of a rigid pair of bases (see motion.formation_pose).
"""

import dataclasses
import heapq
import math
from typing import Sequence

import numpy as np

from behaviors import collision
from behaviors import motion
from vision import scene as scene_lib


@dataclasses.dataclass(frozen=True)
class PlannerConfig:
  # The room's extent is unknown, so RRT samples a box around the start, goal and obstacles.
  sample_padding_m: float = 2.0
  rrt_step_m: float = 0.3
  rrt_goal_bias: float = 0.15
  rrt_max_iterations: int = 3000
  edge_resolution_m: float = 0.05
  shortcut_iterations: int = 100
  seed: int = 0


def get_robot_obstacle(
  xy: np.ndarray, robot_clearance_m: float, margin: float
) -> scene_lib.Pose2D:
  """Square that, once inflated by `margin`, keeps `robot_clearance_m` from `xy`."""
  side = 2 * max(robot_clearance_m - margin, 0.0)
  return scene_lib.Pose2D(float(xy[0]), float(xy[1]), 0.0, side, side, 1.0)


def plan_point_path(
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
  if collision.path_is_clear(start, goal, rects, margin):
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
    if not any(collision.contains(other, corner, margin) for other in rects)
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
      if neighbor == index or not collision.path_is_clear(
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


class _FormationSpace:
  """Validity, distance and interpolation for pair-frame states (x, y, yaw)."""

  def __init__(
    self,
    start: scene_lib.Pose2D,
    slots: Sequence[tuple[np.ndarray, float]],
    rects: Sequence[scene_lib.Pose2D],
    margin: float,
    config: PlannerConfig,
    payload: Sequence[scene_lib.Pose2D] = (),
    payload_rects: Sequence[scene_lib.Pose2D] = (),
    payload_margin: float = 0.0,
  ):
    self._slots = slots
    self._rects = rects
    self._margin = margin
    self._payload = payload
    self._payload_rects = payload_rects
    self._payload_margin = payload_margin
    self._config = config
    self._yaw_radius = max(float(np.linalg.norm(slot_xy)) for slot_xy, _ in slots)
    # Pairs already inside an inflated box at the start may leave it, as in path_is_clear.
    self._exempt = {
      (robot, index)
      for robot, point in enumerate(self.get_base_points(self.to_array(start)))
      for index, rect in enumerate(rects)
      if collision.contains(rect, point, margin)
    }
    self._payload_exempt = {
      (item, index)
      for item, carried in enumerate(self.get_payload(self.to_array(start)))
      for index, rect in enumerate(payload_rects)
      if collision.rects_overlap(carried, rect, payload_margin)
    }

  @staticmethod
  def to_array(pose: scene_lib.Pose2D) -> np.ndarray:
    return np.array([pose.x, pose.y, pose.yaw])

  @staticmethod
  def to_pose(state: np.ndarray) -> scene_lib.Pose2D:
    return scene_lib.Pose2D(
      float(state[0]),
      float(state[1]),
      motion.wrap_angle(float(state[2])),
      0.0,
      0.0,
      1.0,
    )

  def get_base_points(self, state: np.ndarray) -> list[np.ndarray]:
    frame = self.to_pose(state)
    return [motion.compose(frame, *slot)[0] for slot in self._slots]

  def get_payload(self, state: np.ndarray) -> list[scene_lib.Pose2D]:
    """World rects of the carried payload, given in the pair frame."""
    frame = self.to_pose(state)
    carried = []
    for rect in self._payload:
      center, yaw = motion.compose(frame, np.array([rect.x, rect.y]), rect.yaw)
      carried.append(
        dataclasses.replace(rect, x=float(center[0]), y=float(center[1]), yaw=yaw)
      )
    return carried

  def is_valid(self, state: np.ndarray) -> bool:
    for robot, point in enumerate(self.get_base_points(state)):
      for index, rect in enumerate(self._rects):
        margin = 0.0 if (robot, index) in self._exempt else self._margin
        if collision.contains(rect, point, margin):
          return False
    for item, carried in enumerate(self.get_payload(state)):
      for index, rect in enumerate(self._payload_rects):
        margin = 0.0 if (item, index) in self._payload_exempt else self._payload_margin
        if collision.rects_overlap(carried, rect, margin):
          return False
    return True

  def get_distance(self, first: np.ndarray, second: np.ndarray) -> float:
    yaw_error = motion.wrap_angle(second[2] - first[2])
    return math.hypot(
      second[0] - first[0], second[1] - first[1], self._yaw_radius * yaw_error
    )

  def interpolate(
    self, first: np.ndarray, second: np.ndarray, fraction: float
  ) -> np.ndarray:
    yaw_error = motion.wrap_angle(second[2] - first[2])
    return np.array(
      [
        first[0] + fraction * (second[0] - first[0]),
        first[1] + fraction * (second[1] - first[1]),
        motion.wrap_angle(first[2] + fraction * yaw_error),
      ]
    )

  def is_edge_valid(self, first: np.ndarray, second: np.ndarray) -> bool:
    num_checks = max(
      1, math.ceil(self.get_distance(first, second) / self._config.edge_resolution_m)
    )
    return all(
      self.is_valid(self.interpolate(first, second, index / num_checks))
      for index in range(1, num_checks + 1)
    )

  def steer(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    distance = self.get_distance(first, second)
    if distance <= self._config.rrt_step_m:
      return second
    return self.interpolate(first, second, self._config.rrt_step_m / distance)


class _Tree:
  def __init__(self, root: np.ndarray):
    self.states = [root]
    self.parents = [-1]

  def get_nearest(self, space: _FormationSpace, state: np.ndarray) -> int:
    return min(
      range(len(self.states)),
      key=lambda index: space.get_distance(self.states[index], state),
    )

  def add(self, state: np.ndarray, parent: int) -> int:
    self.states.append(state)
    self.parents.append(parent)
    return len(self.states) - 1

  def get_root_path(self, index: int) -> list[np.ndarray]:
    """States from the root to `index`."""
    path = []
    while index != -1:
      path.append(self.states[index])
      index = self.parents[index]
    return path[::-1]


def _shortcut(
  space: _FormationSpace,
  path: list[np.ndarray],
  iterations: int,
  rng: np.random.Generator,
) -> list[np.ndarray]:
  for _ in range(iterations):
    if len(path) < 3:
      break
    first, second = sorted(rng.choice(len(path), size=2, replace=False))
    if second - first > 1 and space.is_edge_valid(path[first], path[second]):
      path = path[: first + 1] + path[second:]
  return path


def plan_formation_path(
  start: scene_lib.Pose2D,
  goal: scene_lib.Pose2D,
  slots: Sequence[tuple[np.ndarray, float]],
  rects: Sequence[scene_lib.Pose2D],
  margin: float,
  config: PlannerConfig = PlannerConfig(),
  payload: Sequence[scene_lib.Pose2D] = (),
  payload_rects: Sequence[scene_lib.Pose2D] = (),
  payload_margin: float = 0.0,
) -> list[scene_lib.Pose2D] | None:
  """Pair-frame waypoints (ending at `goal`) keeping both bases clear of `rects`, or None.

  `slots` are each base's (xy, yaw) in the pair frame. `payload` rects, also in the pair frame
  (e.g. the carried cloth and the span between the bases), must stay `payload_margin` clear of
  `payload_rects`. Plans with RRT-Connect and shortcuts.
  """
  space = _FormationSpace(
    start, slots, rects, margin, config, payload, payload_rects, payload_margin
  )
  start_state, goal_state = space.to_array(start), space.to_array(goal)
  if not space.is_valid(goal_state):
    return None
  if space.is_edge_valid(start_state, goal_state):
    return [goal]

  rng = np.random.default_rng(config.seed)
  extent = np.array(
    [corner for rect in rects for corner in scene_lib.get_corners(rect)]
    + [start_state[:2], goal_state[:2]]
  )
  low = np.array([*(extent.min(axis=0) - config.sample_padding_m), -math.pi])
  high = np.array([*(extent.max(axis=0) + config.sample_padding_m), math.pi])
  trees = [_Tree(start_state), _Tree(goal_state)]
  # trees[0] is the one being extended; they swap each iteration.
  swapped = False
  for _ in range(config.rrt_max_iterations):
    extending, connecting = trees
    sample = rng.uniform(low, high)
    if rng.uniform() < config.rrt_goal_bias:
      sample = connecting.states[0]
    nearest = extending.get_nearest(space, sample)
    new_state = space.steer(extending.states[nearest], sample)
    if not space.is_edge_valid(extending.states[nearest], new_state):
      trees.reverse()
      swapped = not swapped
      continue
    new_index = extending.add(new_state, nearest)
    # Greedily grow the other tree toward the new state.
    reached = False
    while True:
      nearest_other = connecting.get_nearest(space, new_state)
      other_state = space.steer(connecting.states[nearest_other], new_state)
      if not space.is_edge_valid(connecting.states[nearest_other], other_state):
        break
      other_index = connecting.add(other_state, nearest_other)
      if space.get_distance(other_state, new_state) < 1e-9:
        reached = True
        break
    if reached:
      from_extending = extending.get_root_path(new_index)
      from_connecting = connecting.get_root_path(other_index)
      path = from_extending + from_connecting[::-1][1:]
      if swapped:
        path = path[::-1]
      path = _shortcut(space, path, config.shortcut_iterations, rng)
      return [space.to_pose(state) for state in path[1:]]
    trees.reverse()
    swapped = not swapped
  return None
