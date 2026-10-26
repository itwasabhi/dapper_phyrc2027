"""Replays recorded policy.hdf5 samples as the observation dicts a policy receives at runtime."""

import pathlib
from typing import Iterator, Mapping

import numpy as np

from Policy import contract as contract_lib
from Policy import dataset_reader

IMAGE_FIELDS = ("rgb", "depth", "depth_valid")
NUM_ROBOTS = 2
NUM_JOINTS = 13


def get_camera_ids(contract: Mapping) -> list[str]:
  return [camera["id"] for camera in contract["cameras"]]


def to_policy_observation(
  sample: Mapping[str, np.ndarray], contract: Mapping
) -> dict[str, np.ndarray]:
  """Stacks per-camera keys into (cameras, H, W, C) and restores low-dim shapes."""
  shapes = contract_lib.observation_shapes(
    contract, robots=NUM_ROBOTS, joints=NUM_JOINTS, profile="actor_rgbd"
  )
  cameras = get_camera_ids(contract)
  observation = {}
  for name, shape in shapes.items():
    if name in IMAGE_FIELDS:
      observation[name] = np.stack([sample[f"{camera}_{name}"] for camera in cameras])
    else:
      observation[name] = np.asarray(sample[name]).reshape(shape)
  return observation


def iterate_observations(
  path: pathlib.Path, start: int = 0, stop: int | None = None, stride: int = 1
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
  """Yields (step, observation) for the first episode-ordered samples in [start, stop)."""
  dataset = dataset_reader.PolicyDataset(path)
  try:
    stop = len(dataset) if stop is None else min(stop, len(dataset))
    for step in range(start, stop, stride):
      sample, _ = dataset[step]
      yield step, to_policy_observation(sample, dataset.contract)
  finally:
    dataset.close()


def rotate_about_z(observation: Mapping[str, np.ndarray], angle: float) -> dict:
  """Rotates the world frame about the z axis through the origin by `angle` (rad).

  Only world-frame base poses change; camera and gripper poses follow, since they are relative.
  """
  rotated = dict(observation)
  base_pose = np.array(observation["base_pose_world"], dtype=np.float64)
  cos_angle, sin_angle = np.cos(angle), np.sin(angle)
  x, y = base_pose[:, 0].copy(), base_pose[:, 1].copy()
  base_pose[:, 0] = cos_angle * x - sin_angle * y
  base_pose[:, 1] = sin_angle * x + cos_angle * y
  half_cos, half_sin = np.cos(angle / 2), np.sin(angle / 2)
  w, i, j, k = (base_pose[:, index].copy() for index in range(3, 7))
  # (half_cos, 0, 0, half_sin) * (w, i, j, k), quaternions in wxyz order.
  base_pose[:, 3] = half_cos * w - half_sin * k
  base_pose[:, 4] = half_cos * i - half_sin * j
  base_pose[:, 5] = half_cos * j + half_sin * i
  base_pose[:, 6] = half_cos * k + half_sin * w
  rotated["base_pose_world"] = base_pose.astype(np.float32)
  return rotated
