import math

import numpy as np

from behaviors import collision
from behaviors import motion
from behaviors import path_planning
from vision import scene as scene_lib

MARGIN = 0.25
SLOTS = [(np.array([0.0, 0.5]), 0.0), (np.array([0.0, -0.5]), 0.0)]


def _table(x=0.0, y=0.0, yaw=0.0, length=2.0, width=1.0):
  return scene_lib.Pose2D(x, y, yaw, length, width, 1.0)


def _pose(x, y, yaw=0.0):
  return scene_lib.Pose2D(x, y, yaw, 0.0, 0.0, 1.0)


def _assert_formation_path_is_clear(start, path, rects):
  states = [start, *path]
  for first, second in zip(states[:-1], states[1:]):
    for fraction in np.linspace(0, 1, 40):
      x = first.x + fraction * (second.x - first.x)
      y = first.y + fraction * (second.y - first.y)
      yaw = first.yaw + fraction * motion.wrap_angle(second.yaw - first.yaw)
      for slot in SLOTS:
        point, _ = motion.compose(_pose(x, y, yaw), *slot)
        assert not any(collision.contains(rect, point, MARGIN) for rect in rects)


def test_point_path_is_direct_when_clear():
  goal = np.array([3.0, 2.0])
  path = path_planning.plan_point_path(np.array([-3.0, 2.0]), goal, (_table(),), MARGIN)
  assert len(path) == 1 and np.allclose(path[0], goal)


def test_point_path_goes_around_a_blocking_rect():
  start, goal = np.array([-3.0, 0.0]), np.array([3.0, 0.0])
  rects = (_table(),)
  path = path_planning.plan_point_path(start, goal, rects, MARGIN)
  assert path is not None and len(path) >= 2 and np.allclose(path[-1], goal)
  for begin, end in zip([start, *path[:-1]], path):
    assert collision.path_is_clear(begin, end, rects, MARGIN)


def test_point_path_is_none_for_a_goal_inside_an_obstacle():
  path = path_planning.plan_point_path(
    np.array([-3.0, 0.0]), np.zeros(2), (_table(),), MARGIN
  )
  assert path is None


def test_robot_obstacle_keeps_the_clearance_once_inflated():
  obstacle = path_planning.get_robot_obstacle(np.array([1.0, 1.0]), 0.6, MARGIN)
  assert collision.contains(obstacle, np.array([1.0, 1.59]), MARGIN)
  assert not collision.contains(obstacle, np.array([1.0, 1.62]), MARGIN)


def test_formation_path_is_direct_when_clear():
  goal = _pose(3.0, 0.0)
  path = path_planning.plan_formation_path(_pose(-3.0, 0.0), goal, SLOTS, (), MARGIN)
  assert path == [goal]


def test_formation_path_goes_around_a_table():
  start, goal = _pose(-3.0, 0.0), _pose(3.0, 0.0)
  rects = (_table(length=1.0, width=2.0),)
  path = path_planning.plan_formation_path(start, goal, SLOTS, rects, MARGIN)
  assert path is not None and len(path) >= 2
  assert path[-1] == goal
  _assert_formation_path_is_clear(start, path, rects)


def test_formation_path_is_none_when_a_slot_goal_is_inside_an_obstacle():
  rects = (_table(x=3.0, y=0.5, length=0.5, width=0.5),)
  assert (
    path_planning.plan_formation_path(
      _pose(-3.0, 0.0), _pose(3.0, 0.0), SLOTS, rects, MARGIN
    )
    is None
  )


def test_formation_path_is_none_when_walled_in():
  # A wall longer than the sampled region, with a gap narrower than the pair's footprint.
  rects = (
    _table(y=20.2, length=40.0, width=1.0, yaw=math.pi / 2),
    _table(y=-20.2, length=40.0, width=1.0, yaw=math.pi / 2),
  )
  config = path_planning.PlannerConfig(rrt_max_iterations=300)
  assert (
    path_planning.plan_formation_path(
      _pose(-3.0, 0.0), _pose(3.0, 0.0), SLOTS, rects, MARGIN, config
    )
    is None
  )


def test_formation_start_inside_the_clearance_band_can_leave():
  rects = (_table(),)
  start = _pose(0.0, 1.2)  # Lower slot at y=0.7 is within the 0.25 band of the table.
  path = path_planning.plan_formation_path(start, _pose(0.0, 2.0), SLOTS, rects, MARGIN)
  assert path is not None


def test_formation_path_is_reproducible_for_a_seed():
  rects = (_table(length=1.0, width=2.0),)
  args = (_pose(-3.0, 0.0), _pose(3.0, 0.0), SLOTS, rects, MARGIN)
  assert path_planning.plan_formation_path(*args) == path_planning.plan_formation_path(
    *args
  )


def test_formation_path_keeps_the_payload_span_clear_of_a_rect():
  slots = [(np.array([-1.0, 0.0]), 0.0), (np.array([1.0, 0.0]), 0.0)]
  blocker = _table(x=0.0, y=0.0, length=0.4, width=0.4)
  start, goal = _pose(0.0, -2.0), _pose(0.0, 2.0)
  span = scene_lib.Pose2D(0.0, 0.0, 0.0, 2.0, 0.2, 1.0)
  # Bases (x = +-1) clear the blocker on a straight line, but the span between them does not.
  assert path_planning.plan_formation_path(start, goal, slots, [blocker], 0.1) == [goal]
  path = path_planning.plan_formation_path(
    start, goal, slots, [blocker], 0.1, payload=[span], payload_rects=[blocker]
  )
  assert path is not None and path != [goal]
  for pose in path:
    carried = scene_lib.Pose2D(pose.x, pose.y, pose.yaw, 2.0, 0.2, 1.0)
    assert not collision.rects_overlap(carried, blocker, 0.0)
