import math

import numpy as np
import pytest

from behaviors import base
from behaviors import drive_to_manikin
from behaviors import motion
from vision import scene as scene_lib


def _observation(poses):
  rows = [
    [x, y, 0.0, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)] for x, y, yaw in poses
  ]
  return {"base_pose_world": np.array(rows, np.float32)}


def _scene():
  return scene_lib.SceneEstimate(
    manikin=scene_lib.Pose2D(0.0, 3.0, math.pi / 2, 0.4, 0.4, 1.0)
  )


def test_goal_is_in_front_of_the_manikin_facing_it():
  observation = _observation([(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)])
  goal = drive_to_manikin._get_manikin_front_goal(
    _scene(), observation["base_pose_world"]
  )
  assert goal.x == pytest.approx(0.0, abs=1e-6)
  assert goal.y == pytest.approx(3.0 + scene_lib.FRONT_STANDOFF_M)
  assert goal.yaw == pytest.approx(-math.pi / 2)


def test_goal_yaw_faces_the_manikin_on_average():
  observation = _observation([(-0.5, 0.0, 0.0), (0.5, 0.0, 0.2)])
  goal = drive_to_manikin._get_manikin_front_goal(
    _scene(), observation["base_pose_world"]
  )
  assert goal.yaw == pytest.approx(-math.pi / 2 - 0.1, abs=1e-3)


def test_pair_drives_toward_the_manikin_front():
  behavior = drive_to_manikin.DriveToManikin()
  behavior.reset(base.BehaviorState(scene=_scene()))
  # Robots abreast on the manikin's front side, facing -y: the pair must drive forward.
  yaw = -math.pi / 2
  action = behavior.step(_observation([(-0.5, 6.0, yaw), (0.5, 6.0, yaw)]))
  assert np.all(action[:, motion.FORWARD_COLUMN] > 0)
  assert not behavior.blocked()
  assert not behavior.done()


def test_pair_holds_when_the_manikin_blocks_the_path():
  behavior = drive_to_manikin.DriveToManikin()
  behavior.reset(base.BehaviorState(scene=_scene()))
  action = behavior.step(_observation([(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)]))
  assert not action.any()
  assert behavior.blocked()


def test_done_immediately_without_a_manikin():
  behavior = drive_to_manikin.DriveToManikin()
  behavior.reset(base.BehaviorState())
  action = behavior.step(_observation([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]))
  assert not action.any()
  assert behavior.done()
