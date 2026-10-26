import numpy as np

from behaviors import base
from behaviors import drive_to_shirt
from behaviors import motion
from vision import scene as scene_lib


def _observation(robot_0_xy, robot_1_xy=(5.0, 5.0)):
  base_pose = np.array(
    [[*robot_0_xy, 0.0, 1.0, 0.0, 0.0, 0.0], [*robot_1_xy, 0.0, 1.0, 0.0, 0.0, 0.0]],
    np.float32,
  )
  return {"base_pose_world": base_pose}


def test_drives_toward_the_shirt_and_eventually_reports_done():
  state = base.BehaviorState(
    scene=scene_lib.SceneEstimate(shirt=scene_lib.Pose2D(2.0, 0.0, 0.0, 0.5, 0.3, 1.0))
  )
  behavior = drive_to_shirt.DriveToShirt(config=drive_to_shirt.Config(timeout_s=1.0))
  behavior.reset(state)

  action = behavior.step(_observation(robot_0_xy=(0.0, 0.0)))
  assert action[0, motion.FORWARD_COLUMN] > 0
  assert not behavior.done()

  # A fake robot that never actually moves will not converge; the timeout still fires.
  for _ in range(30):
    behavior.step(_observation(robot_0_xy=(0.0, 0.0)))
  assert behavior.done()


def test_holds_position_when_no_shirt_was_found():
  state = base.BehaviorState(scene=scene_lib.SceneEstimate())
  behavior = drive_to_shirt.DriveToShirt()
  behavior.reset(state)

  action = behavior.step(_observation(robot_0_xy=(0.0, 0.0)))
  assert np.array_equal(action, np.zeros((2, 9), np.float32))
  assert behavior.done()


def test_corners_send_each_robot_to_a_different_corner():
  table = scene_lib.Pose2D(0.0, 0.0, 0.0, 2.0, 1.0, 1.0)
  state = base.BehaviorState(
    scene=scene_lib.SceneEstimate(
      tables=(table,),
      shirt_table=0,
      manikin=scene_lib.Pose2D(0.0, 5.0, 0.0, 0.4, 0.4, 1.0),
    )
  )
  # Robot 0 starts on the +x side, so it takes the +x corner.
  goals = drive_to_shirt._get_corner_goals(
    state.scene, _observation((1.0, 3.0), (-1.0, 3.0))["base_pose_world"]
  )
  assert goals[0].x > 0 > goals[1].x


def test_corners_hold_without_a_shirt_table():
  behavior = drive_to_shirt.DriveToShirtCorners()
  behavior.reset(base.BehaviorState(scene=scene_lib.SceneEstimate()))
  action = behavior.step(_observation(robot_0_xy=(0.0, 0.0)))
  assert np.array_equal(action, np.zeros((2, 9), np.float32))
  assert behavior.done()
