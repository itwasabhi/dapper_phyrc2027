"""Stretch4 arm indices, rates and gripper kinematics, measured from joint_position/grasp_pose_base.

Base frame: x forward, y left, z up from the floor. The telescoping arm extends along +x and the
fingers close along y. Positive wrist pitch tilts the gripper down, up to about 65 degrees.
"""

import math

import numpy as np

# Action columns (docs/competition/DATA.md).
LIFT_COLUMN = 3
ARM_COLUMN = 4
WRIST_YAW_COLUMN = 5
WRIST_PITCH_COLUMN = 6
WRIST_ROLL_COLUMN = 7
GRIPPER_COLUMN = 8
GRIPPER_CLOSE = 1.0
GRIPPER_OPEN = -1.0

# controller_target columns.
LIFT_TARGET = 0
ARM_TARGET = 1
WRIST_YAW_TARGET = 2
WRIST_PITCH_TARGET = 3
WRIST_ROLL_TARGET = 4

# joint_position indices within a robot.
LIFT_JOINT = 3
ARM_JOINTS = slice(4, 8)
WRIST_YAW_JOINT = 8
WRIST_PITCH_JOINT = 9
WRIST_ROLL_JOINT = 10

LIFT_RATE = 1.54
ARM_RATE = 1.21
WRIST_RATE = 5.5

# Gripper center at zero lift, arm and wrist angles.
GRIPPER_ORIGIN_X = 0.466
GRIPPER_ORIGIN_Y = -0.087
GRIPPER_ORIGIN_Z = 0.177
# Wrist pitch pivot to gripper center.
WRIST_LENGTH = 0.297
MAX_ARM_EXTENSION = 0.52
MAX_WRIST_PITCH = 1.14


def get_gripper_center(lift: float, arm: float, pitch: float) -> tuple[float, float]:
  """Base-frame (x, z) of the gripper center with zero wrist yaw and roll."""
  pivot_x = GRIPPER_ORIGIN_X - WRIST_LENGTH + arm
  return (
    pivot_x + WRIST_LENGTH * math.cos(pitch),
    GRIPPER_ORIGIN_Z + lift - WRIST_LENGTH * math.sin(pitch),
  )


def get_lift_for_height(gripper_z: float, pitch: float) -> float:
  return gripper_z - GRIPPER_ORIGIN_Z + WRIST_LENGTH * math.sin(pitch)


def get_arm_for_reach(gripper_x: float, pitch: float) -> float:
  return gripper_x - (GRIPPER_ORIGIN_X - WRIST_LENGTH + WRIST_LENGTH * math.cos(pitch))


def command_to_target(
  desired: float, current_target: float, rate: float, policy_hz: int, max_command: float
) -> float:
  """Velocity command that moves a joint's controller target toward `desired` this step.

  The controller integrates velocity commands into its target, so this lands on `desired` in
  one step when within `max_command * rate / policy_hz`.
  """
  command = (desired - current_target) * policy_hz / rate
  return float(np.clip(command, -max_command, max_command))
