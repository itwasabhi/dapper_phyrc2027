import dataclasses
import math

import numpy as np

from behaviors import arm
from behaviors import collision
from behaviors import motion
from behaviors import shirt_grasp
from vision import scene as scene_lib

CLEARANCE_M = 0.25


def _scene(heading=-math.pi / 2):
  """Shirt at the origin on a 0.94 x 0.72 table, collar toward `heading`."""
  axis = np.array([math.cos(heading), math.sin(heading)])
  left = np.array([-axis[1], axis[0]])
  hem_center = -0.28 * axis
  return scene_lib.SceneEstimate(
    tables=(scene_lib.Pose2D(0.0, 0.0, 0.0, 0.94, 0.72, 1.0),),
    shirt=scene_lib.Pose2D(0.0, 0.0, heading, 0.6, 0.76, 0.5),
    shirt_table=0,
    shirt_hem_corners=(
      tuple(hem_center + 0.25 * left),
      tuple(hem_center - 0.25 * left),
    ),
    shirt_hem_seen=True,
    shirt_height_m=0.6,
  )


def test_grasp_points_are_inset_from_the_corners():
  left_point, right_point = shirt_grasp.get_hem_grasp_points(_scene())
  # Collar toward -y puts the hem at y = +0.28; facing -y, left is +x.
  np.testing.assert_allclose(left_point, (0.21, 0.28), atol=1e-6)
  np.testing.assert_allclose(right_point, (-0.21, 0.28), atol=1e-6)


def test_grasp_base_poses_put_the_gripper_on_the_point_and_clear_the_table():
  scene = _scene()
  for grasp in shirt_grasp.get_hem_grasps(scene, CLEARANCE_M):
    assert grasp is not None
    pose = grasp.base_pose
    assert not collision.contains(
      scene.tables[0], np.array([pose.x, pose.y]), CLEARANCE_M
    )
    point_base = motion.world_to_body(
      grasp.point_xy - np.array([pose.x, pose.y]), pose.yaw
    )
    assert abs(point_base[1] - arm.GRIPPER_ORIGIN_Y) < 1e-6
    extension = arm.get_arm_for_reach(point_base[0], 1.1)
    assert 0.0 <= extension <= arm.MAX_ARM_EXTENSION


def test_robots_face_across_the_shirt_from_either_side():
  left_grasp, right_grasp = shirt_grasp.get_hem_grasps(_scene(), CLEARANCE_M)
  # Left corner is at +x, so that robot stands at larger x and faces -x.
  assert (
    left_grasp.base_pose.x > 0.47
    and abs(abs(left_grasp.base_pose.yaw) - math.pi) < 1e-6
  )
  assert right_grasp.base_pose.x < -0.47 and abs(right_grasp.base_pose.yaw) < 1e-6


def test_no_grasps_without_hem_corners():
  assert shirt_grasp.get_hem_grasps(scene_lib.SceneEstimate(), CLEARANCE_M) == (
    None,
    None,
  )


def test_targets_go_to_the_nearer_robot():
  base_pose = np.zeros((2, 7))
  base_pose[0, :2] = (-2.0, 0.0)
  base_pose[1, :2] = (2.0, 0.0)
  first, second = shirt_grasp.get_hem_grasp_targets(_scene(), base_pose)
  assert first[0] < 0 < second[0]


def test_blocked_side_approach_swings_toward_the_hem():
  # Collar toward -y on a table rotated 45 degrees, so the straight side approach to each
  # corner runs into the table.
  scene = dataclasses.replace(
    _scene(), tables=(scene_lib.Pose2D(0.0, 0.0, math.pi / 4, 1.4, 1.4, 1.0),)
  )
  grasps = shirt_grasp.get_hem_grasps(scene, CLEARANCE_M)
  assert all(grasp is not None for grasp in grasps)
  for grasp in grasps:
    pose = grasp.base_pose
    assert not collision.contains(
      scene.tables[0], np.array([pose.x, pose.y]), CLEARANCE_M
    )
    point_base = motion.world_to_body(
      grasp.point_xy - np.array([pose.x, pose.y]), pose.yaw
    )
    assert abs(point_base[1] - arm.GRIPPER_ORIGIN_Y) < 1e-6
  # Swung toward the hem side (+y), not straight across.
  assert all(grasp.base_pose.y > 0.4 for grasp in grasps)
  gap = math.hypot(
    grasps[0].base_pose.x - grasps[1].base_pose.x,
    grasps[0].base_pose.y - grasps[1].base_pose.y,
  )
  assert gap >= collision.CollisionConfig().robot_clearance_m
