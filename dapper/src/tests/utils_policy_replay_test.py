import json
import pathlib

import numpy as np

from Policy import contract as contract_lib

import utils_policy_replay

CONTRACT_PATH = (
  pathlib.Path(__file__).resolve().parents[3] / "config/policy_interface.json"
)


def _recorded_sample(contract):
  """Per-camera image keys and flattened low-dim keys, as PolicyDataset returns them."""
  shapes = contract_lib.observation_shapes(
    contract, robots=2, joints=13, profile="actor_rgbd"
  )
  sample = {}
  for name, shape in shapes.items():
    if name in utils_policy_replay.IMAGE_FIELDS:
      for index, camera in enumerate(utils_policy_replay.get_camera_ids(contract)):
        sample[f"{camera}_{name}"] = np.full(shape[1:], index, np.float32)
    else:
      sample[name] = np.arange(max(int(np.prod(shape)), 1), dtype=np.float32)
  return sample


def test_to_policy_observation_stacks_cameras_and_reshapes_low_dim():
  contract = json.loads(CONTRACT_PATH.read_text())
  observation = utils_policy_replay.to_policy_observation(
    _recorded_sample(contract), contract
  )
  num_cameras = len(contract["cameras"])
  assert observation["rgb"].shape == (num_cameras, 256, 256, 3)
  assert observation["depth"][2, 0, 0, 0] == 2
  assert observation["base_pose_world"].shape == (2, 7)
  np.testing.assert_array_equal(observation["base_pose_world"][1], np.arange(7, 14))
  assert observation["simulation_time_s"].shape == ()


def test_rotate_about_z_rotates_position_and_heading():
  half = np.sqrt(0.5)
  # Robot at (1, 0) facing +y.
  base_pose = np.array([[1.0, 0.0, 0.1, half, 0.0, 0.0, half]] * 2, np.float32)
  rotated = utils_policy_replay.rotate_about_z(
    {"base_pose_world": base_pose}, np.pi / 2
  )["base_pose_world"]
  np.testing.assert_allclose(rotated[0, :3], [0.0, 1.0, 0.1], atol=1e-6)
  # Now facing -x: yaw pi, i.e. quaternion (0, 0, 0, +-1).
  np.testing.assert_allclose(np.abs(rotated[0, 3:]), [0.0, 0.0, 0.0, 1.0], atol=1e-6)
