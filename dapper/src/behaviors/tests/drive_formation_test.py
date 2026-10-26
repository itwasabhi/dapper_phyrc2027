import math

import numpy as np
import pytest

from behaviors import base
from behaviors import drive_formation
from behaviors import motion
from vision import scene as scene_lib


def _observation(poses):
  """poses: [(x, y, yaw), (x, y, yaw)]."""
  rows = [
    [x, y, 0.0, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)] for x, y, yaw in poses
  ]
  return {"base_pose_world": np.array(rows, np.float32)}


def _make(goal, scene=scene_lib.SceneEstimate()):
  behavior = drive_formation.DriveFormation(lambda scene, base_pose: goal)
  behavior.reset(base.BehaviorState(scene=scene))
  return behavior


def _world_velocities(action, poses):
  velocities = []
  for robot, (_, _, yaw) in enumerate(poses):
    forward, left = (
      action[robot, motion.FORWARD_COLUMN],
      action[robot, motion.LEFT_COLUMN],
    )
    velocities.append(
      [
        math.cos(yaw) * forward - math.sin(yaw) * left,
        math.sin(yaw) * forward + math.cos(yaw) * left,
      ]
    )
  return np.array(velocities)


def _goal(x, y, yaw):
  return scene_lib.Pose2D(x, y, yaw, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
  "poses",
  [
    [(0.0, 0.5, 0.0), (0.0, -0.5, 0.0)],
    [(0.0, 0.5, 0.0), (0.0, -0.5, math.pi)],
    [(0.0, 0.5, 1.0), (0.0, -0.5, -2.0)],
  ],
)
def test_translation_goal_moves_both_robots_identically_in_world_frame(poses):
  behavior = _make(_goal(3.0, 0.0, poses[0][2]))
  action = behavior.step(_observation(poses))
  velocities = _world_velocities(action, poses)
  assert np.allclose(velocities[0], velocities[1], atol=1e-5)
  assert velocities[0][0] > 0
  assert np.allclose(action[:, motion.YAW_COLUMN], 0.0, atol=1e-5)


def test_rotation_goal_swings_the_robots_in_opposite_directions():
  poses = [(0.0, 0.5, 0.0), (0.0, -0.5, 0.0)]
  behavior = _make(_goal(0.0, 0.0, 0.5))
  action = behavior.step(_observation(poses))
  assert action[0, motion.FORWARD_COLUMN] == pytest.approx(
    -action[1, motion.FORWARD_COLUMN]
  )
  assert action[0, motion.YAW_COLUMN] == pytest.approx(action[1, motion.YAW_COLUMN])
  assert action[0, motion.YAW_COLUMN] > 0


def test_saturation_scales_both_robots_together():
  poses = [(0.0, 0.5, 0.0), (0.0, -0.5, 0.0)]
  action = _make(_goal(50.0, 0.0, 0.0)).step(_observation(poses))
  assert np.abs(action).max() == pytest.approx(1.0)
  assert action[0, motion.FORWARD_COLUMN] == pytest.approx(
    action[1, motion.FORWARD_COLUMN]
  )


def test_holds_and_finishes_at_the_goal():
  poses = [(0.0, 0.5, 0.0), (0.0, -0.5, 0.0)]
  behavior = _make(_goal(0.0, 0.0, 0.0))
  action = behavior.step(_observation(poses))
  assert np.all(action == 0)
  assert behavior.done()


def test_no_goal_holds_and_finishes():
  poses = [(0.0, 0.5, 0.0), (0.0, -0.5, 0.0)]
  behavior = _make(None)
  assert np.all(behavior.step(_observation(poses)) == 0)
  assert behavior.done()


def test_table_on_either_robots_slot_path_holds_the_pair():
  poses = [(0.0, 0.5, 0.0), (0.0, -0.5, 0.0)]
  table = scene_lib.Pose2D(2.0, -0.5, 0.0, 0.5, 0.5, 1.0)  # Only robot 1's lane.
  behavior = _make(_goal(4.0, 0.0, 0.0), scene_lib.SceneEstimate(tables=(table,)))
  action = behavior.step(_observation(poses))
  assert np.all(action == 0)
  assert behavior.blocked()
  assert behavior.done()
