import math

import numpy as np
import pytest

from behaviors import motion
from vision import scene


def test_drive_toward_is_near_zero_at_the_goal():
  goal = scene.Pose2D(1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
  command = motion.drive_toward((1.0, 0.0), 0.0, goal, motion.DriveConfig())
  assert np.allclose(command, 0.0, atol=1e-6)


def test_drive_toward_moves_forward_toward_a_goal_ahead():
  goal = scene.Pose2D(1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
  command = motion.drive_toward((0.0, 0.0), 0.0, goal, motion.DriveConfig())
  assert command[motion.FORWARD_COLUMN] > 0
  assert command[motion.LEFT_COLUMN] == pytest.approx(0.0, abs=1e-6)


def test_within_tolerance():
  goal = scene.Pose2D(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
  config = motion.DriveConfig()
  assert motion.within_tolerance((0.0, 0.0), 0.0, goal, config)
  assert not motion.within_tolerance((1.0, 0.0), 0.0, goal, config)


def test_yaw_from_base_pose_identity_and_quarter_turn():
  identity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  assert motion.yaw_from_base_pose(identity) == pytest.approx(0.0)
  quarter_turn = np.array(
    [0.0, 0.0, 0.0, math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
  )
  assert motion.yaw_from_base_pose(quarter_turn) == pytest.approx(math.pi / 2)
