import math

import numpy as np
import pytest

from behaviors import base
from behaviors import collision
from behaviors import drive_shirt_to_manikin
from behaviors import motion
from vision import scene as scene_lib

MANIKIN_LENGTH_M = 0.4


def _observation():
  # Robots face each other across the shirt, grippers 0.4 m ahead of the bases.
  rows = [[-0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
  grasp = [[0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]] * 2
  return {
    "base_pose_world": np.array(rows, np.float32),
    "grasp_pose_base": np.array(grasp, np.float32),
  }


def _scene(manikin_width=0.4, with_shirt=True):
  return scene_lib.SceneEstimate(
    manikin=scene_lib.Pose2D(
      0.0, 3.0, math.pi / 2, MANIKIN_LENGTH_M, manikin_width, 1.0
    ),
    shirt=scene_lib.Pose2D(0.0, 0.8, math.pi / 2, 0.7, 0.8, 1.0)
    if with_shirt
    else None,
  )


def _plan(scene, config=drive_shirt_to_manikin.Config()):
  behavior = drive_shirt_to_manikin.DriveShirtToManikin(config=config)
  behavior.reset(base.BehaviorState(scene=scene))
  return behavior, behavior._plan_goal(_observation())


def test_hem_is_parallel_to_the_manikin_front_with_collar_away():
  _, goal = _plan(_scene())
  # The hem midpoint is the pair origin here; the manikin faces +y, so the collar (+y in the
  # start pose, where the shirt lies) already points away and the pair needs no rotation.
  assert goal.x == pytest.approx(0.0, abs=1e-6)
  assert goal.y == pytest.approx(3.0 + MANIKIN_LENGTH_M / 2 + 0.3)
  assert goal.yaw == pytest.approx(0.0, abs=1e-6)


def test_pair_turns_so_the_collar_points_away_from_the_manikin():
  scene = _scene()
  scene = scene_lib.SceneEstimate(
    manikin=scene_lib.Pose2D(0.0, 3.0, -math.pi / 2, MANIKIN_LENGTH_M, 0.4, 1.0),
    shirt=scene.shirt,
  )
  _, goal = _plan(scene)
  assert goal.y == pytest.approx(3.0 - MANIKIN_LENGTH_M / 2 - 0.3)
  assert abs(goal.yaw) == pytest.approx(math.pi, abs=1e-6)


def test_standoff_grows_until_the_bases_clear_the_manikin():
  config = drive_shirt_to_manikin.Config(hem_standoff_m=0.1)
  _, goal = _plan(_scene(manikin_width=1.4), config)
  assert goal.y - (3.0 + MANIKIN_LENGTH_M / 2) >= 0.25 - 1e-6


def test_blocked_and_done_without_a_manikin():
  behavior = drive_shirt_to_manikin.DriveShirtToManikin()
  behavior.reset(base.BehaviorState())
  action = behavior.step(_observation())
  assert not action.any()
  assert behavior.blocked()
  assert behavior.done()


def test_first_step_drives_the_pair_without_arm_commands():
  behavior = drive_shirt_to_manikin.DriveShirtToManikin()
  behavior.reset(base.BehaviorState(scene=_scene()))
  action = behavior.step(_observation())
  assert action[:, : motion.YAW_COLUMN + 1].any()
  assert not action[:, motion.YAW_COLUMN + 1 :].any()
  assert not behavior.blocked()
  assert not behavior.done()


def test_payload_covers_the_span_between_bases_and_the_shirt_body():
  behavior, _ = _plan(_scene())
  span, body = behavior._get_payload()
  assert span.length == pytest.approx(1.0)
  assert (span.x, span.y) == pytest.approx((0.0, 0.0))
  # Collar side is +y, so the shirt body lies past the hem on that side.
  assert body.y == pytest.approx(0.35)


def test_pair_does_not_sweep_its_span_through_the_manikin():
  # Start on the manikin's far side: a straight drive would pass the manikin between the bases.
  behavior = drive_shirt_to_manikin.DriveShirtToManikin()
  behavior.reset(base.BehaviorState(scene=_scene()))
  behavior.step(_observation())
  path = behavior._drive._path
  assert path is not None
  manikin = _scene().manikin
  for pose in path:
    span = scene_lib.Pose2D(pose.x, pose.y, pose.yaw, 1.0, 0.3, 1.0)
    assert not collision.rects_overlap(span, manikin, 0.0)
