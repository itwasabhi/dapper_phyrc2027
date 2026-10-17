"""Samples robot start poses near the corners of a rectangular table."""

import dataclasses
import math
from typing import Sequence

import numpy as np

# Corner signs (x, y) in the table frame; indices 0/1 share the -y long edge.
_CORNER_SIGNS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
_SAME_EDGE_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))
_ROBOT_Z_M = 0.03


@dataclasses.dataclass(frozen=True)
class SpawnConfig:
  num_robots: int = 2
  pairing: str = "random_pair"  # random_pair | same_edge_pair
  standoff_m: float = 0.8
  position_noise_m: float = 0.1
  yaw_noise_deg: float = 15.0
  yaw_offset_deg: float = 0.0


@dataclasses.dataclass(frozen=True)
class RobotSpawn:
  position: np.ndarray  # (3,) world metres
  quat_wxyz: np.ndarray  # (4,)
  yaw_deg: float
  corner_index: int


def _choose_corners(
  rng: np.random.Generator, pairing: str, num_robots: int
) -> Sequence[int]:
  if pairing == "same_edge_pair":
    if num_robots != 2:
      raise ValueError("same_edge_pair requires num_robots == 2")
    pair = _SAME_EDGE_PAIRS[rng.integers(len(_SAME_EDGE_PAIRS))]
    return pair if rng.random() < 0.5 else pair[::-1]
  if pairing == "random_pair":
    return tuple(
      int(c) for c in rng.choice(len(_CORNER_SIGNS), num_robots, replace=False)
    )
  raise ValueError(f"Unknown pairing {pairing!r}")


def yaw_to_quat_wxyz(yaw_deg: float) -> np.ndarray:
  half = math.radians(yaw_deg) / 2
  return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def sample_robot_spawns(
  table_center_xy: Sequence[float],
  table_size_xy: Sequence[float],
  rng: np.random.Generator,
  config: SpawnConfig,
) -> Sequence[RobotSpawn]:
  """One spawn per robot, each near a distinct corner and facing the table."""
  center = np.asarray(table_center_xy, dtype=np.float64)
  half_size = np.asarray(table_size_xy, dtype=np.float64) / 2
  spawns = []
  for corner_index in _choose_corners(rng, config.pairing, config.num_robots):
    signs = _CORNER_SIGNS[corner_index]
    outward = signs / np.linalg.norm(signs)
    noise = rng.uniform(-config.position_noise_m, config.position_noise_m, size=2)
    position_xy = center + signs * half_size + outward * config.standoff_m + noise
    to_center = center - position_xy
    yaw_deg = (
      math.degrees(math.atan2(to_center[1], to_center[0]))
      + config.yaw_offset_deg
      + rng.uniform(-config.yaw_noise_deg, config.yaw_noise_deg)
    )
    spawns.append(
      RobotSpawn(
        position=np.array([*position_xy, _ROBOT_Z_M], dtype=np.float32),
        quat_wxyz=yaw_to_quat_wxyz(yaw_deg),
        yaw_deg=float(yaw_deg),
        corner_index=int(corner_index),
      )
    )
  return tuple(spawns)
