"""Depth back-projection and world poses of the robot-mounted cameras."""

import dataclasses
import json
import pathlib

import numpy as np

ROBOT_CAMERA_FOV_DEG = 70.0
WRIST_CAMERA_NAMES = ("robot_0_wrist", "robot_1_wrist")
HEAD_CAMERA_NAMES = ("robot_0_head", "robot_1_head")


@dataclasses.dataclass(frozen=True)
class CameraMounts:
  """Static camera extrinsics shared by both robots (4x4, optical frame: x right, y down, z forward)."""

  base_from_head_optical: np.ndarray
  grasp_from_wrist_optical: np.ndarray


DEFAULT_MOUNTS_PATH = pathlib.Path(__file__).with_name("camera_mounts.json")


def load_mounts(path: pathlib.Path = DEFAULT_MOUNTS_PATH) -> CameraMounts:
  raw = json.loads(pathlib.Path(path).read_text())
  return CameraMounts(
    base_from_head_optical=np.asarray(raw["base_from_head_optical"], dtype=float),
    grasp_from_wrist_optical=np.asarray(raw["grasp_from_wrist_optical"], dtype=float),
  )


def intrinsics(width: int, fov_deg: float = ROBOT_CAMERA_FOV_DEG) -> np.ndarray:
  focal = width / (2 * np.tan(np.deg2rad(fov_deg) / 2))
  return np.array([[focal, 0, width / 2], [0, focal, width / 2], [0, 0, 1]])


def pose_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
  """x,y,z,qw,qx,qy,qz to a 4x4 local-to-parent matrix."""
  x, y, z, w, i, j, k = pose_wxyz
  norm = np.sqrt(w * w + i * i + j * j + k * k)
  w, i, j, k = w / norm, i / norm, j / norm, k / norm
  out = np.eye(4)
  out[:3, :3] = [
    [1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 * (i * k + j * w)],
    [2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w)],
    [2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 * (i * i + j * j)],
  ]
  out[:3, 3] = (x, y, z)
  return out


def camera_world_poses(
  base_pose_world: np.ndarray, grasp_pose_base: np.ndarray, mounts: CameraMounts
) -> dict[str, np.ndarray]:
  """World-from-optical matrices for the head and wrist cameras of each robot."""
  poses = {}
  for robot in range(len(base_pose_world)):
    world_from_base = pose_matrix(base_pose_world[robot])
    poses[HEAD_CAMERA_NAMES[robot]] = world_from_base @ mounts.base_from_head_optical
    poses[WRIST_CAMERA_NAMES[robot]] = (
      world_from_base
      @ pose_matrix(grasp_pose_base[robot])
      @ mounts.grasp_from_wrist_optical
    )
  return poses


def backproject(
  depth: np.ndarray, valid: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Returns optical-frame points (N,3) and their flat pixel indices (N,) for valid pixels."""
  depth = depth[..., 0] if depth.ndim == 3 else depth
  valid = valid[..., 0] if valid.ndim == 3 else valid
  pixel_index = np.flatnonzero(valid.ravel())
  rows, cols = np.divmod(pixel_index, depth.shape[1])
  z = depth.ravel()[pixel_index]
  x = (cols - intrinsic[0, 2]) / intrinsic[0, 0] * z
  y = (rows - intrinsic[1, 2]) / intrinsic[1, 1] * z
  return np.stack([x, y, z], axis=1), pixel_index


def get_world_points(
  depth: np.ndarray, valid: np.ndarray, world_from_optical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """World-frame points (N,3) of valid pixels and their flat pixel indices (N,)."""
  points, pixel_index = backproject(depth, valid, intrinsics(depth.shape[1]))
  world = points @ world_from_optical[:3, :3].T + world_from_optical[:3, 3]
  return world, pixel_index
