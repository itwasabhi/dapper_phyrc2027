import math

import numpy as np

from behaviors import collision
from vision import scene


def _table(x=0.0, y=0.0, yaw=0.0, length=2.0, width=1.0):
  return scene.Pose2D(x, y, yaw, length, width, 1.0)


def test_empty_scene_has_no_obstacles_and_every_path_is_clear():
  assert collision.obstacles(scene.SceneEstimate()) == ()
  assert collision.path_is_clear(np.array([0.0, 0.0]), np.array([9.0, 9.0]), (), 0.25)


def test_obstacles_include_tables_and_manikin():
  table, manikin = _table(), _table(x=3.0)
  estimate = scene.SceneEstimate(tables=(table,), manikin=manikin)
  assert collision.obstacles(estimate) == (table, manikin)


def test_path_through_rect_is_blocked_and_path_around_is_clear():
  rects = (_table(),)
  assert not collision.path_is_clear(
    np.array([-3.0, 0.0]), np.array([3.0, 0.0]), rects, 0.0
  )
  assert collision.path_is_clear(
    np.array([-3.0, 2.0]), np.array([3.0, 2.0]), rects, 0.0
  )


def test_margin_inflates_the_rect():
  rects = (_table(),)  # Half-width 0.5.
  start, end = np.array([-3.0, 0.6]), np.array([3.0, 0.6])
  assert collision.path_is_clear(start, end, rects, 0.0)
  assert not collision.path_is_clear(start, end, rects, 0.25)


def test_rotated_rect_is_respected():
  rects = (_table(yaw=math.pi / 2),)  # Long axis along world y.
  assert not collision.path_is_clear(
    np.array([0.0, -3.0]), np.array([0.0, 3.0]), rects, 0.0
  )
  assert collision.path_is_clear(
    np.array([1.5, -3.0]), np.array([1.5, 3.0]), rects, 0.0
  )


def test_start_inside_the_clearance_band_can_leave():
  rects = (_table(),)
  start = np.array([0.0, 0.6])  # Outside the raw box, inside the 0.25 band.
  assert collision.path_is_clear(start, np.array([0.0, 3.0]), rects, 0.25)
  assert not collision.path_is_clear(start, np.array([0.0, -3.0]), rects, 0.25)


def test_path_clear_of_point():
  start, end = np.array([0.0, 0.0]), np.array([4.0, 0.0])
  assert not collision.path_clear_of_point(start, end, np.array([2.0, 0.2]), 0.6)
  assert collision.path_clear_of_point(start, end, np.array([2.0, 1.0]), 0.6)
  # Already close at the start: exempt so two nearby robots cannot lock each other.
  assert collision.path_clear_of_point(start, end, np.array([0.0, 0.3]), 0.6)


def test_plan_path_is_direct_when_clear():
  goal = np.array([3.0, 2.0])
  path = collision.plan_path(np.array([-3.0, 2.0]), goal, (_table(),), 0.25)
  assert len(path) == 1 and np.allclose(path[0], goal)


def test_plan_path_goes_around_a_blocking_rect():
  start, goal = np.array([-3.0, 0.0]), np.array([3.0, 0.0])
  rects = (_table(),)
  path = collision.plan_path(start, goal, rects, 0.25)
  assert path is not None and len(path) >= 2 and np.allclose(path[-1], goal)
  for begin, end in zip([start, *path[:-1]], path):
    assert collision.path_is_clear(begin, end, rects, 0.25)


def test_plan_path_returns_none_for_a_goal_inside_an_obstacle():
  path = collision.plan_path(np.array([-3.0, 0.0]), np.zeros(2), (_table(),), 0.25)
  assert path is None
