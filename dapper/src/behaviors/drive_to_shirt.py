"""drive_to_shirt: drives each robot to its own standoff corner beside the shirt scan_room found."""

import dataclasses

import numpy as np

from behaviors import base
from behaviors import motion
from vision import scene as scene_lib


@dataclasses.dataclass(frozen=True)
class Config:
  drive: motion.DriveConfig = motion.DriveConfig()
  timeout_s: float = 20.0  # Safety bound if convergence stalls.


class DriveToShirt:
  """Reads state.scene (populated by scan_room) and drives both robots toward the shirt."""

  def __init__(
    self, cfg: base.BehaviorCfg = base.BehaviorCfg(), config: Config = Config()
  ):
    self._config = config
    self._timeout_steps = round(config.timeout_s * cfg.policy_hz)
    self._state: base.BehaviorState | None = None
    self._step = 0
    self._reached = (False, False)

  def reset(self, state: base.BehaviorState) -> None:
    self._state = state
    self._step = 0
    self._reached = (False, False)

  def step(self, observation) -> np.ndarray:
    base_pose = np.asarray(observation["base_pose_world"])
    action = np.zeros((2, 9), dtype=np.float32)
    reached = []
    for robot in range(2):
      current_xy = base_pose[robot, :2]
      goal = scene_lib.shirt_approach_pose(self._state.scene, tuple(current_xy))
      if goal is None:
        # No shirt detected: hold position rather than drive blindly.
        reached.append(True)
        continue
      current_yaw = motion.yaw_from_base_pose(base_pose[robot])
      command = motion.drive_toward(current_xy, current_yaw, goal, self._config.drive)
      action[robot, motion.FORWARD_COLUMN] = command[0]
      action[robot, motion.LEFT_COLUMN] = command[1]
      action[robot, motion.YAW_COLUMN] = command[2]
      reached.append(
        motion.within_tolerance(current_xy, current_yaw, goal, self._config.drive)
      )
    self._reached = tuple(reached)
    self._step += 1
    return action

  def done(self) -> bool:
    return all(self._reached) or self._step >= self._timeout_steps

  def close(self) -> None:
    pass
