"""drive_independent: drives each robot to its own goal pose in parallel, around obstacles."""

import dataclasses
from typing import Callable

import numpy as np

from behaviors import base
from behaviors import collision
from behaviors import motion
from vision import scene as scene_lib

GoalFn = Callable[
  [scene_lib.SceneEstimate, np.ndarray],
  tuple[scene_lib.Pose2D | None, scene_lib.Pose2D | None],
]


@dataclasses.dataclass(frozen=True)
class Config:
  drive: motion.DriveConfig = motion.DriveConfig()
  collision_config: collision.CollisionConfig = collision.CollisionConfig()
  timeout_s: float = 60.0  # Safety bound if convergence stalls.


class DriveIndependent:
  """Drives each robot to its goal along a path around the scene's obstacles, re-planned each step.

  A robot with no goal holds. A robot with no obstacle-free path (e.g. a goal inside an
  obstacle's clearance) holds and counts as finished (blocked). A robot whose next path leg
  passes the other base waits.
  """

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
    self._finished = (False, False)
    self._blocked = (False, False)

  def reset(self, state: base.BehaviorState) -> None:
    self._state = state
    self._step = 0
    self._finished = (False, False)
    self._blocked = (False, False)

  def step(self, observation) -> np.ndarray:
    base_pose = np.asarray(observation["base_pose_world"])
    scene = self._state.scene
    goals = self._goal_fn(scene, base_pose)
    rects = collision.obstacles(scene)
    action = np.zeros((2, 9), dtype=np.float32)
    finished = [False, False]
    blocked = [False, False]
    for robot, goal in enumerate(goals):
      if goal is None:
        finished[robot] = True
        continue
      current_xy = base_pose[robot, :2]
      current_yaw = motion.yaw_from_base_pose(base_pose[robot])
      if motion.within_tolerance(current_xy, current_yaw, goal, self._config.drive):
        finished[robot] = True
        continue
      path = collision.plan_path(
        current_xy,
        np.array([goal.x, goal.y]),
        rects,
        self._config.collision_config.obstacle_clearance_m,
      )
      if path is None:
        finished[robot] = blocked[robot] = True
        continue
      waypoint = path[0]
      if not collision.path_clear_of_point(
        current_xy,
        waypoint,
        base_pose[1 - robot, :2],
        self._config.collision_config.robot_clearance_m,
      ):
        continue
      target = scene_lib.Pose2D(
        float(waypoint[0]), float(waypoint[1]), goal.yaw, 0.0, 0.0, goal.confidence
      )
      command = motion.drive_toward(current_xy, current_yaw, target, self._config.drive)
      if len(path) > 1:
        # Keep full speed through intermediate waypoints instead of slowing at each one.
        command[:2] /= max(float(np.max(np.abs(command[:2]))), 1e-6)
      action[robot, motion.FORWARD_COLUMN] = command[0]
      action[robot, motion.LEFT_COLUMN] = command[1]
      action[robot, motion.YAW_COLUMN] = command[2]
    self._finished = tuple(finished)
    self._blocked = tuple(blocked)
    self._step += 1
    return action

  def blocked(self) -> tuple[bool, bool]:
    return self._blocked

  def done(self) -> bool:
    return all(self._finished) or self._step >= self._timeout_steps

  def close(self) -> None:
    pass
