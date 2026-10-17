import numpy as np
import pytest

from rl.envs import spawn

CENTER = (1.0, -1.5)
SIZE = (0.93, 0.70)


def _outside_table(position, margin=0.0):
  half = np.asarray(SIZE) / 2 + margin
  return bool(np.any(np.abs(position[:2] - np.asarray(CENTER)) > half))


@pytest.mark.parametrize("pairing", ["random_pair", "same_edge_pair"])
def test_corners_distinct_and_outside_table(pairing):
  config = spawn.SpawnConfig(pairing=pairing)
  for seed in range(50):
    spawns = spawn.sample_robot_spawns(
      CENTER, SIZE, np.random.default_rng(seed), config
    )
    assert len(spawns) == 2
    assert spawns[0].corner_index != spawns[1].corner_index
    assert all(_outside_table(s.position) for s in spawns)


def test_same_edge_pair_shares_an_edge():
  config = spawn.SpawnConfig(pairing="same_edge_pair")
  for seed in range(50):
    first, second = spawn.sample_robot_spawns(
      CENTER, SIZE, np.random.default_rng(seed), config
    )
    assert (first.corner_index - second.corner_index) % 4 in (1, 3)


def test_position_within_noise_bounds_of_nominal():
  config = spawn.SpawnConfig(standoff_m=0.8, position_noise_m=0.1)
  for seed in range(50):
    for robot in spawn.sample_robot_spawns(
      CENTER, SIZE, np.random.default_rng(seed), config
    ):
      signs = spawn._CORNER_SIGNS[robot.corner_index]
      nominal = (
        np.asarray(CENTER)
        + signs * np.asarray(SIZE) / 2
        + signs / np.linalg.norm(signs) * 0.8
      )
      assert np.all(np.abs(robot.position[:2] - nominal) <= 0.1 + 1e-6)


def test_yaw_faces_table_center_within_noise():
  config = spawn.SpawnConfig(yaw_noise_deg=15.0, yaw_offset_deg=0.0)
  for seed in range(50):
    for robot in spawn.sample_robot_spawns(
      CENTER, SIZE, np.random.default_rng(seed), config
    ):
      to_center = np.asarray(CENTER) - robot.position[:2]
      expected = np.degrees(np.arctan2(to_center[1], to_center[0]))
      error = (robot.yaw_deg - expected + 180) % 360 - 180
      assert abs(error) <= 15.0 + 1e-6


def test_quaternion_is_unit_and_matches_yaw():
  quat = spawn.yaw_to_quat_wxyz(270.0)
  np.testing.assert_allclose(quat, [-0.7071068, 0, 0, 0.7071068], atol=1e-6)
  assert np.isclose(np.linalg.norm(quat), 1.0)


def test_same_seed_is_deterministic():
  config = spawn.SpawnConfig()
  first = spawn.sample_robot_spawns(CENTER, SIZE, np.random.default_rng(3), config)
  second = spawn.sample_robot_spawns(CENTER, SIZE, np.random.default_rng(3), config)
  for a, b in zip(first, second):
    np.testing.assert_array_equal(a.position, b.position)


def test_unknown_pairing_raises():
  with pytest.raises(ValueError):
    spawn.sample_robot_spawns(
      CENTER, SIZE, np.random.default_rng(0), spawn.SpawnConfig(pairing="x")
    )
