"""drive_formation: moves both robots as a rigid pair toward a goal pose of the pair's frame.

The pair frame has its origin at the midpoint of the two bases and the heading of robot 0
(see motion.formation_pose). The relative pose of the robots at the first step is preserved.
"""

import dataclasses
from typing import Callable

import numpy as np

from behaviors import base
from behaviors import collision
from behaviors import motion
from vision import scene as scene_lib

GoalFn = Callable[[scene_lib.SceneEstimate, np.ndarray], scene_lib.Pose2D | None]


@dataclasses.dataclass(frozen=True)
class Config:
  drive: motion.DriveConfig = motion.DriveConfig()
  formation_gain: float = 1.0  # Pulls each robot back to its slot to cancel drift.
  collision_config: collision.CollisionConfig = collision.CollisionConfig()
  timeout_s: float = 60.0


class DriveFormation:
  """Tracks a pair-frame goal; holds if either robot's path to its final slot is blocked."""

  def __init__(
    self,
    goal_fn: GoalFn,
    cfg: base.BehaviorCfg = base.BehaviorCfg(),
    config: Config = Config(),
  ):
    self._goal_fn = goal_fn
    self._config = config
    self._timeout_steps = round(config.timeout_s * cfg.policy_hz)
    self._state: base.BehaviorState | None = None
    self._step = 0
    self._slots: list[tuple[np.ndarray, float]] | None = None
    self._finished = False
    self._blocked = False

  def reset(self, state: base.BehaviorState) -> None:
    self._state = state
    self._step = 0
    self._slots = None
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

    margin = self._config.collision_config.obstacle_clearance_m
    rects = collision.obstacles(self._state.scene)
    goal_slots = [motion.compose(goal, *slot) for slot in self._slots]
    self._blocked = not all(
      collision.path_is_clear(positions[robot], goal_slots[robot][0], rects, margin)
      for robot in range(2)
    )
    if self._blocked:
      self._finished = True
      return action

    center = np.array([frame.x, frame.y])
    linear = drive.position_gain * (np.array([goal.x, goal.y]) - center)
    angular = drive.yaw_gain * motion.wrap_angle(goal.yaw - frame.yaw)
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
    # One shared scale keeps the robots' relative motion intact when saturating.
    commands /= max(1.0, float(np.abs(commands).max()))
    action[:, motion.FORWARD_COLUMN] = commands[:, 0]
    action[:, motion.LEFT_COLUMN] = commands[:, 1]
    action[:, motion.YAW_COLUMN] = commands[:, 2]
    return action

  def blocked(self) -> bool:
    return self._blocked

  def done(self) -> bool:
    return self._finished or self._step >= self._timeout_steps

  def close(self) -> None:
    pass
