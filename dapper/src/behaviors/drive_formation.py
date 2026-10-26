"""drive_formation: moves both robots as a rigid pair toward a goal pose of the pair's frame.

The pair frame has its origin at the midpoint of the two bases and the heading of robot 0
(see motion.formation_pose). The relative pose of the robots at the first step is preserved.
"""

import dataclasses
import math
from typing import Callable, Sequence

import numpy as np

from behaviors import base
from behaviors import collision
from behaviors import motion
from behaviors import path_planning
from vision import scene as scene_lib

GoalFn = Callable[[scene_lib.SceneEstimate, np.ndarray], scene_lib.Pose2D | None]
# Pair-frame rects the formation carries, which must also stay clear of the manikin.
PayloadFn = Callable[[], Sequence[scene_lib.Pose2D]]


@dataclasses.dataclass(frozen=True)
class Config:
  drive: motion.DriveConfig = motion.DriveConfig()
  formation_gain: float = 1.0  # Pulls each robot back to its slot to cancel drift.
  collision_config: collision.CollisionConfig = collision.CollisionConfig()
  planner: path_planning.PlannerConfig = path_planning.PlannerConfig()
  payload_clearance_m: float = 0.1
  waypoint_tolerance_m: float = 0.2
  waypoint_tolerance_rad: float = 0.2
  timeout_s: float = 60.0


class DriveFormation:
  """Tracks a pair-frame goal along a planned path; holds if no path keeps both bases clear."""

  def __init__(
    self,
    goal_fn: GoalFn,
    cfg: base.BehaviorCfg = base.BehaviorCfg(),
    config: Config = Config(),
    payload_fn: PayloadFn | None = None,
  ):
    self._goal_fn = goal_fn
    self._payload_fn = payload_fn
    self._config = config
    self._timeout_steps = round(config.timeout_s * cfg.policy_hz)
    self._state: base.BehaviorState | None = None
    self._step = 0
    self._slots: list[tuple[np.ndarray, float]] | None = None
    self._path: list[scene_lib.Pose2D] | None = None
    self._path_index = 0
    self._planned_goal: scene_lib.Pose2D | None = None
    self._finished = False
    self._blocked = False

  def reset(self, state: base.BehaviorState) -> None:
    self._state = state
    self._step = 0
    self._slots = None
    self._path = None
    self._path_index = 0
    self._planned_goal = None
    self._finished = False
    self._blocked = False

  def step(self, observation) -> np.ndarray:
    self._step += 1
    action = np.zeros((2, 9), dtype=np.float32)
    base_pose = np.asarray(observation["base_pose_world"])
    goal = self._goal_fn(self._state.scene, base_pose)
    if goal is None:
      self._finished = True
      return action

    positions = base_pose[:, :2].astype(float)
    yaws = np.array([motion.yaw_from_base_pose(pose) for pose in base_pose])
    frame = motion.formation_pose(positions[0], positions[1], yaws[0])
    if self._slots is None:
      self._slots = [
        motion.relative(frame, positions[robot], yaws[robot]) for robot in range(2)
      ]

    drive = self._config.drive
    self._finished = motion.within_tolerance(
      np.array([frame.x, frame.y]), frame.yaw, goal, drive
    )
    if self._finished:
      return action

    if self._goal_moved(goal):
      scene = self._state.scene
      payload = self._payload_fn() if self._payload_fn else ()
      self._path = path_planning.plan_formation_path(
        frame,
        goal,
        self._slots,
        collision.obstacles(self._state.scene),
        self._config.collision_config.obstacle_clearance_m,
        self._config.planner,
        payload=payload,
        payload_rects=(scene.manikin,) if scene.manikin is not None else (),
        payload_margin=self._config.payload_clearance_m,
      )
      self._path_index = 0
      self._planned_goal = goal
    self._blocked = self._path is None
    if self._blocked:
      self._finished = True
      return action

    last_index = len(self._path) - 1
    while self._path_index < last_index and self._is_near_waypoint(
      frame, self._path[self._path_index]
    ):
      self._path_index += 1
    waypoint = self._path[self._path_index]
    is_final = self._path_index == last_index

    center = np.array([frame.x, frame.y])
    linear = drive.position_gain * (np.array([waypoint.x, waypoint.y]) - center)
    angular = drive.yaw_gain * motion.wrap_angle(waypoint.yaw - frame.yaw)
    commands = np.zeros((2, 3))
    for robot in range(2):
      slot_xy, slot_yaw = motion.compose(frame, *self._slots[robot])
      offset = positions[robot] - center
      velocity = (
        linear
        + angular * np.array([-offset[1], offset[0]])
        + self._config.formation_gain * (slot_xy - positions[robot])
      )
      yaw_rate = angular + self._config.formation_gain * motion.wrap_angle(
        slot_yaw - yaws[robot]
      )
      forward, left = motion.world_to_body(velocity, yaws[robot])
      commands[robot] = [
        forward / motion.BASE_LINEAR_RATE,
        left / motion.BASE_LINEAR_RATE,
        yaw_rate / motion.BASE_ANGULAR_RATE,
      ]
    # One shared scale keeps the robots' relative motion intact when saturating. Intermediate
    # waypoints are driven at full speed instead of slowing at each one.
    peak = float(np.abs(commands).max())
    commands /= peak if not is_final and peak > 1e-6 else max(1.0, peak)
    action[:, motion.FORWARD_COLUMN] = commands[:, 0]
    action[:, motion.LEFT_COLUMN] = commands[:, 1]
    action[:, motion.YAW_COLUMN] = commands[:, 2]
    return action

  def _goal_moved(self, goal: scene_lib.Pose2D) -> bool:
    planned = self._planned_goal
    if planned is None:
      return True
    drive = self._config.drive
    return (
      math.hypot(goal.x - planned.x, goal.y - planned.y) > drive.position_tolerance_m
      or abs(motion.wrap_angle(goal.yaw - planned.yaw)) > drive.yaw_tolerance_rad
    )

  def _is_near_waypoint(
    self, frame: scene_lib.Pose2D, waypoint: scene_lib.Pose2D
  ) -> bool:
    return (
      math.hypot(waypoint.x - frame.x, waypoint.y - frame.y)
      <= self._config.waypoint_tolerance_m
      and abs(motion.wrap_angle(waypoint.yaw - frame.yaw))
      <= self._config.waypoint_tolerance_rad
    )

  def blocked(self) -> bool:
    return self._blocked

  def done(self) -> bool:
    return self._finished or self._step >= self._timeout_steps

  def close(self) -> None:
    pass
