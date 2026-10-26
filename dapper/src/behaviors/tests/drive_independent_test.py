import numpy as np

from behaviors import base
from behaviors import drive_independent
from behaviors import motion
from vision import scene as scene_lib


def _observation(robot_0_xy, robot_1_xy):
  base_pose = np.array(
    [[*robot_0_xy, 0.0, 1.0, 0.0, 0.0, 0.0], [*robot_1_xy, 0.0, 1.0, 0.0, 0.0, 0.0]],
    np.float32,
  )
  return {"base_pose_world": base_pose}


def _goals(scene, base_pose):
  del scene, base_pose
  return (
    scene_lib.Pose2D(4.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    scene_lib.Pose2D(4.0, 5.0, 0.0, 0.0, 0.0, 1.0),
  )


def _make(scene):
  behavior = drive_independent.DriveIndependent(_goals)
  behavior.reset(base.BehaviorState(scene=scene))
  return behavior


def test_both_robots_drive_toward_their_own_goals_without_obstacles():
  behavior = _make(scene_lib.SceneEstimate())
  action = behavior.step(_observation((0.0, 0.0), (0.0, 5.0)))
  assert action[0, motion.FORWARD_COLUMN] > 0
  assert action[1, motion.FORWARD_COLUMN] > 0
  assert not behavior.done()


def test_table_on_one_robots_path_is_driven_around():
  table = scene_lib.Pose2D(2.0, 0.0, 0.0, 1.0, 1.0, 1.0)
  behavior = _make(scene_lib.SceneEstimate(tables=(table,)))
  action = behavior.step(_observation((0.0, 0.0), (0.0, 5.0)))
  # Heads for a corner of the inflated table, so it moves sideways as well as forward.
  assert action[0, motion.FORWARD_COLUMN] > 0
  assert abs(action[0, motion.LEFT_COLUMN]) > 0.1
  assert behavior.blocked() == (False, False)


def test_goal_inside_an_obstacle_blocks_only_that_robot():
  table = scene_lib.Pose2D(4.0, 0.0, 0.0, 1.0, 1.0, 1.0)
  behavior = _make(scene_lib.SceneEstimate(tables=(table,)))
  action = behavior.step(_observation((0.0, 0.0), (0.0, 5.0)))
  assert np.all(action[0] == 0)
  assert action[1, motion.FORWARD_COLUMN] > 0
  assert behavior.blocked() == (True, False)
  assert not behavior.done()


def test_done_once_every_robot_is_blocked():
  table = scene_lib.Pose2D(4.0, 0.0, 0.0, 1.0, 1.0, 1.0)
  other = scene_lib.Pose2D(4.0, 5.0, 0.0, 1.0, 1.0, 1.0)
  behavior = _make(scene_lib.SceneEstimate(tables=(table, other)))
  behavior.step(_observation((0.0, 0.0), (0.0, 5.0)))
  assert behavior.blocked() == (True, True)
  assert behavior.done()


def test_robot_detours_around_the_other_base_on_its_path():
  behavior = _make(scene_lib.SceneEstimate())
  action = behavior.step(_observation((0.0, 0.0), (2.0, 0.1)))
  assert action[0, motion.FORWARD_COLUMN] > 0
  assert abs(action[0, motion.LEFT_COLUMN]) > 0.1
  assert behavior.blocked() == (False, False)
  assert not behavior.done()


def test_robot_waits_while_the_other_base_sits_on_its_goal():
  behavior = _make(scene_lib.SceneEstimate())
  action = behavior.step(_observation((0.0, 0.0), (4.0, 0.1)))
  assert np.all(action[0] == 0)
  assert behavior.blocked() == (False, False)
  assert not behavior.done()


def test_robots_on_each_others_goals_do_not_deadlock():
  def swap_goals(scene, base_pose):
    del scene
    return tuple(
      scene_lib.Pose2D(*base_pose[1 - robot, :2], 0.0, 0.0, 0.0, 1.0)
      for robot in range(2)
    )

  behavior = drive_independent.DriveIndependent(swap_goals)
  behavior.reset(base.BehaviorState())
  action = behavior.step(_observation((0.0, 0.0), (2.0, 0.0)))
  assert action[0, motion.FORWARD_COLUMN] > 0
  assert np.all(action[1] == 0)
