"""down_and_pinch: each robot lowers its tilted gripper onto a world point on a surface and pinches.

Open loop with respect to vision: the grasp points come from the scene estimate once, at the
first step. Proprioception (gripper pose, joint and controller targets) closes the loop on the
arm, lift and a base strafe that line the gripper up with the point.
"""

import dataclasses
import enum
from typing import Callable

import numpy as np

from behaviors import arm
from behaviors import base
from behaviors import motion
from vision import scene as scene_lib

# (scene, base_pose_world) -> per-robot world (x, y) grasp point, or None to hold.
TargetFn = Callable[
  [scene_lib.SceneEstimate, np.ndarray], tuple[np.ndarray | None, np.ndarray | None]
]


class Phase(enum.Enum):
  PREPARE = 0
  DESCEND = 1
  CLOSE = 2
  RAISE = 3
  DONE = 4


@dataclasses.dataclass(frozen=True)
class Config:
  wrist_pitch: float = 1.1
  hover_above_m: float = 0.1
  # Descent stops at this gripper-center height relative to the surface, or on contact.
  grasp_below_surface_m: float = 0.03
  descend_speed_mps: float = 0.05
  # Contact with the surface bends the wrist up from its target (the first sign in the sim) or,
  # failing that, makes the lift lag its controller target.
  contact_pitch_deflection_rad: float = 0.01
  contact_lag_m: float = 0.015
  # After the descent the lift holds this far above where it stopped, so the fingers are not
  # pressed into the surface while they close.
  close_backoff_m: float = 0.005
  # Closing ends once the fingers have closed by `min_closing_m` (they start after a delay) and
  # then changed less than `settle_tolerance_m` over `settle_s`.
  min_closing_m: float = 0.02
  min_close_s: float = 1.0
  # Fingers pressed onto the surface do not close; if they have not started after this long,
  # the lift rises slowly until they do.
  unjam_after_s: float = 0.75
  unjam_speed_mps: float = 0.01
  max_close_s: float = 3.0
  settle_s: float = 0.25
  settle_tolerance_m: float = 0.002
  # Fingers closed on nothing end about 0.044 m apart; wider means something is held.
  empty_grip_distance_m: float = 0.05
  raise_m: float = 0.15
  position_tolerance_m: float = 0.01
  pitch_tolerance_rad: float = 0.03
  reach_gain: float = 2.0
  max_command: float = 0.5
  # A robot skips a point outside this reach box (base frame) at the start.
  max_lateral_offset_m: float = 0.25
  reach_slack_m: float = 0.1
  prepare_timeout_s: float = 10.0
  descend_timeout_s: float = 6.0
  raise_timeout_s: float = 5.0
  default_surface_height_m: float = 0.6


@dataclasses.dataclass
class _RobotState:
  phase: Phase = Phase.PREPARE
  phase_steps: int = 0
  target_xy: np.ndarray | None = None
  hold_lift: float = 0.0
  raise_lift: float = 0.0
  finger_distances: list[float] = dataclasses.field(default_factory=list)
  grasped: bool = False


class DownAndPinch:
  """Per-robot prepare -> descend -> close -> raise; done once both robots finish."""

  def __init__(
    self,
    target_fn: TargetFn,
    cfg: base.BehaviorCfg = base.BehaviorCfg(),
    config: Config = Config(),
  ):
    self._target_fn = target_fn
    self._policy_hz = cfg.policy_hz
    self._config = config
    self._state: base.BehaviorState | None = None
    self._robots: list[_RobotState] = []
    self._surface_z = config.default_surface_height_m
    self._started = False

  def reset(self, state: base.BehaviorState) -> None:
    self._state = state
    self._robots = [_RobotState(), _RobotState()]
    self._started = False

  def phases(self) -> tuple[Phase, Phase]:
    return tuple(robot.phase for robot in self._robots)

  def grasped(self) -> tuple[bool, bool]:
    """Per robot, whether the fingers closed but stayed apart after the raise (holding cloth)."""
    return tuple(robot.grasped for robot in self._robots)

  def step(self, observation) -> np.ndarray:
    base_pose = np.asarray(observation["base_pose_world"])
    if not self._started:
      self._started = True
      scene = self._state.scene
      if scene.shirt_height_m is not None:
        self._surface_z = scene.shirt_height_m
      targets = self._target_fn(scene, base_pose)
      for robot, target, pose in zip(self._robots, targets, base_pose):
        if target is None or not self._is_reachable(np.asarray(target, float), pose):
          robot.phase = Phase.DONE
        else:
          robot.target_xy = np.asarray(target, dtype=float)
    action = np.zeros((2, 9), dtype=np.float32)
    for index, robot in enumerate(self._robots):
      if robot.phase is not Phase.DONE:
        action[index] = self._step_robot(index, robot, observation)
      robot.phase_steps += 1
    return action

  def _step_robot(self, index, robot, observation) -> np.ndarray:
    config = self._config
    joints = np.asarray(observation["joint_position"]).reshape(2, -1)[index]
    targets = np.asarray(observation["controller_target"]).reshape(2, -1)[index]
    gripper = np.asarray(observation["grasp_pose_base"]).reshape(2, -1)[index]
    finger_distance = float(
      np.asarray(observation["fingertip_origin_distance"]).reshape(2)[index]
    )
    base_pose = np.asarray(observation["base_pose_world"])[index]
    yaw = motion.yaw_from_base_pose(base_pose)
    target_base = motion.world_to_body(robot.target_xy - base_pose[:2], yaw)
    reach_error = target_base[0] - gripper[0]
    lateral_error = target_base[1] - gripper[1]
    lift = float(joints[arm.LIFT_JOINT])

    command = np.zeros(9, dtype=np.float32)
    command[arm.WRIST_PITCH_COLUMN] = self._to_target(
      config.wrist_pitch, targets[arm.WRIST_PITCH_TARGET], arm.WRIST_RATE
    )
    command[arm.WRIST_YAW_COLUMN] = self._to_target(
      0.0, targets[arm.WRIST_YAW_TARGET], arm.WRIST_RATE
    )
    command[arm.WRIST_ROLL_COLUMN] = self._to_target(
      0.0, targets[arm.WRIST_ROLL_TARGET], arm.WRIST_RATE
    )
    if robot.phase in (Phase.PREPARE, Phase.DESCEND):
      command[arm.ARM_COLUMN] = np.clip(
        config.reach_gain * reach_error / arm.ARM_RATE,
        -config.max_command,
        config.max_command,
      )
      command[motion.LEFT_COLUMN] = np.clip(
        config.reach_gain * lateral_error / motion.BASE_LINEAR_RATE,
        -config.max_command,
        config.max_command,
      )

    if robot.phase is Phase.PREPARE:
      command[arm.GRIPPER_COLUMN] = arm.GRIPPER_OPEN
      hover_lift = arm.get_lift_for_height(
        self._surface_z + config.hover_above_m, config.wrist_pitch
      )
      command[arm.LIFT_COLUMN] = self._to_target(
        hover_lift, targets[arm.LIFT_TARGET], arm.LIFT_RATE
      )
      ready = (
        abs(lift - hover_lift) < config.position_tolerance_m
        and abs(joints[arm.WRIST_PITCH_JOINT] - config.wrist_pitch)
        < config.pitch_tolerance_rad
        and abs(reach_error) < config.position_tolerance_m
        and abs(lateral_error) < config.position_tolerance_m
      )
      if ready or robot.phase_steps >= config.prepare_timeout_s * self._policy_hz:
        self._advance(robot, Phase.DESCEND)
    elif robot.phase is Phase.DESCEND:
      command[arm.LIFT_COLUMN] = -config.descend_speed_mps / arm.LIFT_RATE
      pitch_deflection = targets[arm.WRIST_PITCH_TARGET] - joints[arm.WRIST_PITCH_JOINT]
      in_contact = (
        pitch_deflection > config.contact_pitch_deflection_rad
        or lift - targets[arm.LIFT_TARGET] > config.contact_lag_m
      )
      deep_enough = gripper[2] <= self._surface_z - config.grasp_below_surface_m
      timed_out = robot.phase_steps >= config.descend_timeout_s * self._policy_hz
      if in_contact or deep_enough or timed_out:
        robot.hold_lift = lift + config.close_backoff_m
        robot.raise_lift = lift + config.raise_m
        self._advance(robot, Phase.CLOSE)
    elif robot.phase is Phase.CLOSE:
      command[arm.GRIPPER_COLUMN] = arm.GRIPPER_CLOSE
      command[arm.LIFT_COLUMN] = self._to_target(
        robot.hold_lift, targets[arm.LIFT_TARGET], arm.LIFT_RATE
      )
      robot.finger_distances.append(finger_distance)
      settle_steps = max(round(config.settle_s * self._policy_hz), 1)
      recent = robot.finger_distances[-settle_steps - 1 :]
      closing = robot.finger_distances[0] - finger_distance > config.min_closing_m
      elapsed_s = robot.phase_steps / self._policy_hz
      if not closing and elapsed_s >= config.unjam_after_s:
        robot.hold_lift += config.unjam_speed_mps / self._policy_hz
      settled = (
        closing
        and len(recent) > settle_steps
        and max(recent) - min(recent) < config.settle_tolerance_m
      )
      if (
        settled and elapsed_s >= config.min_close_s
      ) or elapsed_s >= config.max_close_s:
        self._advance(robot, Phase.RAISE)
    elif robot.phase is Phase.RAISE:
      command[arm.GRIPPER_COLUMN] = arm.GRIPPER_CLOSE
      command[arm.LIFT_COLUMN] = self._to_target(
        robot.raise_lift, targets[arm.LIFT_TARGET], arm.LIFT_RATE
      )
      raised = lift >= robot.raise_lift - config.position_tolerance_m
      if raised or robot.phase_steps >= config.raise_timeout_s * self._policy_hz:
        closed = robot.finger_distances[0] - finger_distance > config.min_closing_m
        robot.grasped = closed and finger_distance > config.empty_grip_distance_m
        self._advance(robot, Phase.DONE)
    return command

  def _is_reachable(self, target_xy: np.ndarray, base_pose: np.ndarray) -> bool:
    config = self._config
    target_base = motion.world_to_body(
      target_xy - base_pose[:2], motion.yaw_from_base_pose(base_pose)
    )
    nearest, _ = arm.get_gripper_center(0.0, 0.0, config.wrist_pitch)
    farthest, _ = arm.get_gripper_center(0.0, arm.MAX_ARM_EXTENSION, config.wrist_pitch)
    return (
      nearest - config.reach_slack_m
      <= target_base[0]
      <= farthest + config.reach_slack_m
      and abs(target_base[1] - arm.GRIPPER_ORIGIN_Y) <= config.max_lateral_offset_m
    )

  def _to_target(self, desired: float, current_target: float, rate: float) -> float:
    return arm.command_to_target(
      desired, float(current_target), rate, self._policy_hz, self._config.max_command
    )

  def _advance(self, robot: _RobotState, phase: Phase) -> None:
    robot.phase = phase
    robot.phase_steps = -1

  def done(self) -> bool:
    return bool(self._robots) and all(
      robot.phase is Phase.DONE for robot in self._robots
    )

  def close(self) -> None:
    pass
