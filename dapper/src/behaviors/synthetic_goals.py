"""Synthetic drive goals sampled from the scene estimate, and a per-leg outcome recorder.

For exercising the drive behaviors; goals are free of the estimated obstacles but otherwise random.
"""

import dataclasses
import json
import math
import pathlib

import numpy as np

from behaviors import collision
from behaviors import motion
from vision import scene as scene_lib


@dataclasses.dataclass(frozen=True)
class SyntheticGoalConfig:
  seed: int = 0
  # Goals are sampled in the obstacles' and robots' bounding box grown by this much.
  sample_padding_m: float = 1.0
  min_goal_separation_m: float = 1.0
  max_attempts: int = 500
  # Prefer goals whose straight path hits an obstacle, so legs have to plan around it; after
  # half the attempts any free goal is accepted.
  require_detour: bool = True
  # Side-by-side spacing of a gathered pair.
  pair_spacing_m: float = 1.0


def _get_pose(xy: np.ndarray, yaw: float) -> scene_lib.Pose2D:
  return scene_lib.Pose2D(float(xy[0]), float(xy[1]), float(yaw), 0.0, 0.0, 1.0)


def _get_yaws(base_pose: np.ndarray) -> np.ndarray:
  return np.array([motion.yaw_from_base_pose(pose) for pose in base_pose])


def _has_reached_goals(goals, base_pose, drive_config) -> bool:
  return all(
    goal is None
    or motion.within_tolerance(
      base_pose[robot, :2],
      motion.yaw_from_base_pose(base_pose[robot]),
      goal,
      drive_config,
    )
    for robot, goal in enumerate(goals)
  )


class IndependentLeg:
  """One goal pose per robot, sampled on the first call and then held."""

  def __init__(
    self,
    rng: np.random.Generator,
    config: SyntheticGoalConfig,
    collision_config: collision.CollisionConfig,
    drive_config: motion.DriveConfig,
  ):
    self._rng = rng
    self._config = config
    self._collision_config = collision_config
    self._drive_config = drive_config
    self._goals: tuple[scene_lib.Pose2D | None, scene_lib.Pose2D | None] = (None, None)
    self._sampled = False

  def get_goals(
    self, scene: scene_lib.SceneEstimate, base_pose: np.ndarray
  ) -> tuple[scene_lib.Pose2D | None, scene_lib.Pose2D | None]:
    if not self._sampled:
      self._goals = self._sample_goals(scene, base_pose)
      self._sampled = True
    return self._goals

  def get_sampled_goals(self) -> list[scene_lib.Pose2D | None]:
    return list(self._goals)

  def has_reached(self, base_pose: np.ndarray) -> bool:
    return _has_reached_goals(self._goals, base_pose, self._drive_config)

  def _sample_goals(self, scene, base_pose):
    rects = collision.obstacles(scene)
    margin = self._collision_config.obstacle_clearance_m
    low, high = _get_sample_box(rects, base_pose[:, :2], self._config.sample_padding_m)
    goals: list[scene_lib.Pose2D | None] = []
    for robot in range(2):
      for attempt in range(self._config.max_attempts):
        xy = self._rng.uniform(low, high)
        if (
          not any(collision.contains(rect, xy, margin) for rect in rects)
          and (
            not _needs_detour(self._config, attempt)
            or not collision.path_is_clear(base_pose[robot, :2], xy, rects, margin)
          )
          and np.linalg.norm(xy - base_pose[robot, :2])
          >= self._config.min_goal_separation_m
          and all(
            np.linalg.norm(xy - np.array([goal.x, goal.y]))
            >= self._collision_config.robot_clearance_m
            for goal in goals
          )
        ):
          goals.append(_get_pose(xy, self._rng.uniform(-math.pi, math.pi)))
          break
      else:
        goals.append(None)
    return tuple(goals)


class GatherLeg:
  """Goals that put the robots side by side, facing the same way, at a random free spot."""

  def __init__(
    self,
    rng: np.random.Generator,
    config: SyntheticGoalConfig,
    collision_config: collision.CollisionConfig,
    drive_config: motion.DriveConfig,
  ):
    self._rng = rng
    self._config = config
    self._collision_config = collision_config
    self._drive_config = drive_config
    self._goals: tuple[scene_lib.Pose2D | None, scene_lib.Pose2D | None] = (None, None)
    self._sampled = False

  def get_goals(
    self, scene: scene_lib.SceneEstimate, base_pose: np.ndarray
  ) -> tuple[scene_lib.Pose2D | None, scene_lib.Pose2D | None]:
    if not self._sampled:
      self._goals = self._sample_goals(scene, base_pose)
      self._sampled = True
    return self._goals

  def get_sampled_goals(self) -> list[scene_lib.Pose2D | None]:
    return list(self._goals)

  def has_reached(self, base_pose: np.ndarray) -> bool:
    return _has_reached_goals(self._goals, base_pose, self._drive_config)

  def _sample_goals(self, scene, base_pose):
    rects = collision.obstacles(scene)
    margin = self._collision_config.obstacle_clearance_m
    positions = base_pose[:, :2].astype(float)
    low, high = _get_sample_box(rects, positions, self._config.sample_padding_m)
    half_spacing = self._config.pair_spacing_m / 2
    slots = [
      (np.array([0.0, half_spacing]), 0.0),
      (np.array([0.0, -half_spacing]), 0.0),
    ]
    for attempt in range(self._config.max_attempts):
      frame = _get_pose(
        self._rng.uniform(low, high), self._rng.uniform(-math.pi, math.pi)
      )
      goals = [_get_pose(*motion.compose(frame, *slot)) for slot in slots]
      goal_points = [np.array([goal.x, goal.y]) for goal in goals]
      if (
        not any(
          collision.contains(rect, point, margin)
          for rect in rects
          for point in goal_points
        )
        and all(
          np.linalg.norm(point - positions[robot]) >= self._config.min_goal_separation_m
          for robot, point in enumerate(goal_points)
        )
        and (
          not _needs_detour(self._config, attempt)
          or not all(
            collision.path_is_clear(positions[robot], goal_points[robot], rects, margin)
            for robot in range(2)
          )
        )
      ):
        return tuple(goals)
    return (None, None)


class FormationLeg:
  """One pair-frame goal pose, sampled on the first call and then held."""

  def __init__(
    self,
    rng: np.random.Generator,
    config: SyntheticGoalConfig,
    collision_config: collision.CollisionConfig,
    drive_config: motion.DriveConfig,
  ):
    self._rng = rng
    self._config = config
    self._collision_config = collision_config
    self._drive_config = drive_config
    self._goal: scene_lib.Pose2D | None = None
    self._sampled = False

  def get_goal(
    self, scene: scene_lib.SceneEstimate, base_pose: np.ndarray
  ) -> scene_lib.Pose2D | None:
    if not self._sampled:
      self._goal = self._sample_goal(scene, base_pose)
      self._sampled = True
    return self._goal

  def get_sampled_goals(self) -> list[scene_lib.Pose2D | None]:
    return [self._goal]

  def has_reached(self, base_pose: np.ndarray) -> bool:
    if self._goal is None:
      return True
    frame = motion.formation_pose(
      base_pose[0, :2], base_pose[1, :2], motion.yaw_from_base_pose(base_pose[0])
    )
    return motion.within_tolerance(
      np.array([frame.x, frame.y]), frame.yaw, self._goal, self._drive_config
    )

  def _sample_goal(self, scene, base_pose):
    rects = collision.obstacles(scene)
    margin = self._collision_config.obstacle_clearance_m
    positions, yaws = base_pose[:, :2].astype(float), _get_yaws(base_pose)
    frame = motion.formation_pose(positions[0], positions[1], yaws[0])
    slots = [
      motion.relative(frame, positions[robot], yaws[robot]) for robot in range(2)
    ]
    low, high = _get_sample_box(rects, positions, self._config.sample_padding_m)
    for attempt in range(self._config.max_attempts):
      xy = self._rng.uniform(low, high)
      goal = _get_pose(xy, self._rng.uniform(-math.pi, math.pi))
      slot_points = [motion.compose(goal, *slot)[0] for slot in slots]
      if np.linalg.norm(
        xy - [frame.x, frame.y]
      ) >= self._config.min_goal_separation_m and not any(
        collision.contains(rect, point, margin)
        for rect in rects
        for point in slot_points
      ):
        return goal
    return None


def _needs_detour(config: SyntheticGoalConfig, attempt: int) -> bool:
  return config.require_detour and attempt < config.max_attempts // 2


def _get_sample_box(rects, positions, padding_m):
  points = np.array(
    [corner for rect in rects for corner in scene_lib.get_corners(rect)]
    + list(positions)
  )
  return points.min(axis=0) - padding_m, points.max(axis=0) + padding_m


class SyntheticGoalSampler:
  """Creates legs whose goals come from one seeded stream."""

  def __init__(
    self,
    config: SyntheticGoalConfig = SyntheticGoalConfig(),
    collision_config: collision.CollisionConfig = collision.CollisionConfig(),
    drive_config: motion.DriveConfig = motion.DriveConfig(),
  ):
    self._rng = np.random.default_rng(config.seed)
    self._args = (config, collision_config, drive_config)

  def create_independent_leg(self) -> IndependentLeg:
    return IndependentLeg(self._rng, *self._args)

  def create_gather_leg(self) -> GatherLeg:
    return GatherLeg(self._rng, *self._args)

  def create_formation_leg(self) -> FormationLeg:
    return FormationLeg(self._rng, *self._args)


class LegRecorder:
  """Wraps a drive behavior; logs one JSON line (and prints a summary) when its leg finishes."""

  def __init__(self, behavior, leg, name: str, log_path: pathlib.Path | None = None):
    self._behavior = behavior
    self._leg = leg
    self._name = name
    self._log_path = log_path
    self._steps = 0
    self._logged = False
    self._trajectory: list[list[list[float]]] = []

  def reset(self, state) -> None:
    self._behavior.reset(state)
    self._steps = 0
    self._logged = False
    self._trajectory = []

  def step(self, observation) -> np.ndarray:
    base_pose = np.asarray(observation["base_pose_world"])
    if not self._logged:
      self._trajectory.append(
        [
          [*base_pose[robot, :2].tolist(), motion.yaw_from_base_pose(base_pose[robot])]
          for robot in range(2)
        ]
      )
    action = self._behavior.step(observation)
    self._steps += 1
    if self._behavior.done() and not self._logged:
      self._logged = True
      self._write_record(np.asarray(observation["base_pose_world"]))
    return action

  def done(self) -> bool:
    return self._behavior.done()

  def close(self) -> None:
    self._behavior.close()

  def _write_record(self, base_pose: np.ndarray) -> None:
    goals = self._leg.get_sampled_goals()
    record = {
      "name": self._name,
      "steps": self._steps,
      "blocked": self._behavior.blocked(),
      "reached": bool(self._leg.has_reached(base_pose)),
      "goals": [None if goal is None else [goal.x, goal.y, goal.yaw] for goal in goals],
      "trajectory": self._trajectory,
      "final_poses": [
        [*base_pose[robot, :2].tolist(), motion.yaw_from_base_pose(base_pose[robot])]
        for robot in range(2)
      ],
    }
    print(
      f"[leg] {self._name}: reached={record['reached']} blocked={record['blocked']} "
      f"steps={self._steps}"
    )
    if self._log_path is not None:
      self._log_path.parent.mkdir(parents=True, exist_ok=True)
      with self._log_path.open("a") as log:
        log.write(json.dumps(record) + "\n")
