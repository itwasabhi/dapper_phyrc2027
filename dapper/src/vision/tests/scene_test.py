import math

import pytest

from vision import scene


def _pose(x, y, yaw, length=0.8, width=0.44):
  return scene.Pose2D(x, y, yaw, length, width, 1.0)


def test_manikin_front_poses_stand_ahead_and_face_the_manikin():
  estimate = scene.SceneEstimate(manikin=_pose(0.0, 1.0, math.pi / 2))
  left_pose, right_pose = scene.manikin_front_poses(estimate)
  assert left_pose.x == pytest.approx(-scene.FRONT_LATERAL_M)
  assert right_pose.x == pytest.approx(scene.FRONT_LATERAL_M)
  assert left_pose.y == pytest.approx(1.0 + scene.FRONT_STANDOFF_M)
  assert left_pose.yaw == pytest.approx(-math.pi / 2)


def test_shirt_approach_pose_picks_side_facing_the_robot():
  estimate = scene.SceneEstimate(shirt=_pose(0.0, -1.1, 0.0))
  pose = scene.shirt_approach_pose(estimate, robot_xy=(0.0, 0.1))
  assert pose.x == pytest.approx(0.0)
  assert pose.y == pytest.approx(-1.1 + 0.22 + scene.TABLE_STANDOFF_M)
  assert pose.yaw == pytest.approx(-math.pi / 2)


def test_goals_are_none_without_estimate():
  empty = scene.SceneEstimate()
  assert scene.manikin_front_poses(empty) is None
  assert scene.shirt_approach_pose(empty, (0.0, 0.0)) is None
