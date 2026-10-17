"""Contract-style observations (batched over envs) from an Isaac Lab grasp scene."""

from typing import Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils import warp as lab_warp


def as_torch(value) -> torch.Tensor:
  if isinstance(value, lab_warp.ProxyArray):
    return value.torch
  return value


def observation_space(
  num_robots: int, num_joints: int, num_cameras: int, width: int, height: int
) -> gym.spaces.Dict:
  inf = np.inf
  image = (num_cameras, height, width)
  return gym.spaces.Dict(
    {
      "joint_position": gym.spaces.Box(-inf, inf, (num_robots, num_joints), np.float32),
      "joint_velocity": gym.spaces.Box(-inf, inf, (num_robots, num_joints), np.float32),
      "previous_action": gym.spaces.Box(-1.0, 1.0, (num_robots, 9), np.float32),
      "simulation_time_s": gym.spaces.Box(-inf, inf, (), np.float64),
      "rgb": gym.spaces.Box(0, 255, (*image, 3), np.uint8),
      "depth": gym.spaces.Box(0.0, inf, (*image, 1), np.float32),
      "depth_valid": gym.spaces.Box(0, 1, (*image, 1), bool),
    }
  )


def build_observation(
  robots: Sequence,
  cameras: Sequence,
  clip_ranges: Sequence[Sequence[float]],
  previous_action: torch.Tensor,
  simulation_time_s: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
  rgb, depth, valid = [], [], []
  for camera, (near, far) in zip(cameras, clip_ranges):
    image = as_torch(camera.data.output["rgb"])[..., :3]
    distance = as_torch(camera.data.output["distance_to_image_plane"])
    is_valid = torch.isfinite(distance) & (distance >= near) & (distance <= far)
    rgb.append(image)
    depth.append(torch.where(is_valid, distance, torch.zeros_like(distance)))
    valid.append(is_valid)
  return {
    "joint_position": torch.stack(
      [as_torch(robot.data.joint_pos) for robot in robots], dim=1
    ),
    "joint_velocity": torch.stack(
      [as_torch(robot.data.joint_vel) for robot in robots], dim=1
    ),
    "previous_action": previous_action,
    "simulation_time_s": simulation_time_s,
    "rgb": torch.stack(rgb, dim=1),
    "depth": torch.stack(depth, dim=1),
    "depth_valid": torch.stack(valid, dim=1),
  }
