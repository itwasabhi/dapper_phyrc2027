import math

import numpy as np
import pytest

from vision import scene as scene_lib
from vision import shirt_orientation

CELL_M = 0.02


def _grid(u_range, v_range):
  u_values = np.arange(u_range[0], u_range[1], CELL_M)
  v_values = np.arange(v_range[0], v_range[1], CELL_M)
  u_grid, v_grid = np.meshgrid(u_values, v_values)
  return np.column_stack([u_grid.ravel(), v_grid.ravel()])


def _tshirt():
  """Shirt-frame (u toward collar, v left) footprint and trim points of a flat T-shirt."""
  body = _grid((-0.3, 0.3), (-0.25, 0.25))
  sleeves = np.concatenate(
    [_grid((0.1, 0.3), (0.25, 0.37)), _grid((0.1, 0.3), (-0.37, -0.25))]
  )
  hem = _grid((-0.3, -0.26), (-0.25, 0.25))
  collar = _grid((0.22, 0.3), (-0.04, 0.04))
  return np.concatenate([body, sleeves]), np.concatenate([hem, collar])


def _to_world(points, heading, center):
  rotation = np.array(
    [[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]]
  )
  return points @ rotation.T + center


@pytest.mark.parametrize("heading_deg", range(-180, 180, 25))
def test_recovers_heading_at_any_rotation(heading_deg):
  footprint, trim = _tshirt()
  heading = math.radians(heading_deg)
  center = np.array([1.5, -2.0])
  result = shirt_orientation.estimate_orientation(
    _to_world(footprint, heading, center),
    _to_world(trim, heading, center),
    0.5,
  )
  assert abs(scene_lib.wrap_angle(result.pose.yaw - heading)) < math.radians(3)
  assert result.margin > 0.2
  assert result.pose.length < result.pose.width
  assert np.hypot(result.pose.x - 1.5, result.pose.y + 2.0) < 0.05


def test_footprint_alone_points_to_sleeves():
  footprint, _ = _tshirt()
  result = shirt_orientation.estimate_orientation(footprint, np.zeros((0, 2)), 0.5)
  assert abs(scene_lib.wrap_angle(result.pose.yaw)) < math.radians(5)


def test_hem_trim_alone_points_away_from_hem():
  square = _grid((-0.3, 0.3), (-0.3, 0.3))
  hem = _grid((-0.3, -0.26), (-0.3, 0.3))
  result = shirt_orientation.estimate_orientation(square, hem, 0.5)
  assert abs(scene_lib.wrap_angle(result.pose.yaw)) < math.radians(5)


def test_symmetric_square_is_ambiguous():
  square = _grid((-0.3, 0.3), (-0.3, 0.3))
  result = shirt_orientation.estimate_orientation(square, np.zeros((0, 2)), 0.5)
  assert result.margin < 0.05


def test_hem_band_refines_a_tilted_coarse_heading():
  _, trim = _tshirt()
  hem = shirt_orientation.get_hem_line(trim, np.zeros(2), math.radians(15))
  assert abs(hem.heading) < math.radians(1)


def test_short_hem_is_not_used():
  collar_only = _grid((0.22, 0.3), (-0.04, 0.04)) - [0.5, 0.0]
  assert shirt_orientation.get_hem_line(collar_only, np.zeros(2), 0.2) is None


@pytest.mark.parametrize("heading_deg", (0, 70, -130))
def test_hem_corners_are_the_ends_of_the_hem_band(heading_deg):
  footprint, trim = _tshirt()
  heading = math.radians(heading_deg)
  center = np.array([0.5, 1.0])
  result = shirt_orientation.estimate_orientation(
    _to_world(footprint, heading, center), _to_world(trim, heading, center), 0.5
  )
  assert result.hem_seen
  # Hem band runs at u = -0.28 from v = -0.25 (right) to v = +0.25 (left).
  expected_left = _to_world(np.array([[-0.28, 0.24]]), heading, center)[0]
  expected_right = _to_world(np.array([[-0.28, -0.24]]), heading, center)[0]
  np.testing.assert_allclose(result.hem_corners[0], expected_left, atol=0.03)
  np.testing.assert_allclose(result.hem_corners[1], expected_right, atol=0.03)


def test_hem_corners_fall_back_to_the_footprint_without_trim():
  footprint, _ = _tshirt()
  result = shirt_orientation.estimate_orientation(footprint, np.zeros((0, 2)), 0.5)
  assert not result.hem_seen
  left, right = result.hem_corners
  assert left[0] < -0.25 and right[0] < -0.25
  assert left[1] > 0 > right[1]
