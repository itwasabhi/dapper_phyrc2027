import numpy as np

from vision import camera_geometry


def test_pose_matrix_applies_yaw_and_translation():
  half = np.sqrt(0.5)
  matrix = camera_geometry.pose_matrix(np.array([1.0, 2.0, 3.0, half, 0.0, 0.0, half]))
  np.testing.assert_allclose(matrix[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-9)
  np.testing.assert_allclose(matrix[:3, 3], [1, 2, 3])


def test_backproject_inverts_pinhole_projection():
  intrinsic = camera_geometry.intrinsics(64)
  point = np.array([0.3, -0.2, 2.0])
  col = intrinsic[0, 0] * point[0] / point[2] + intrinsic[0, 2]
  row = intrinsic[1, 1] * point[1] / point[2] + intrinsic[1, 2]
  depth = np.zeros((64, 64, 1), np.float32)
  valid = np.zeros((64, 64, 1), bool)
  depth[int(row), int(col), 0] = point[2]
  valid[int(row), int(col), 0] = True
  points, pixel_index = camera_geometry.backproject(depth, valid, intrinsic)
  assert pixel_index.tolist() == [int(row) * 64 + int(col)]
  np.testing.assert_allclose(points[0], point, atol=0.05)


def test_camera_world_poses_chain_base_grasp_and_mounts():
  head = np.eye(4)
  head[:3, 3] = [0.0, 0.0, 1.0]
  wrist = np.eye(4)
  wrist[:3, 3] = [0.0, 0.1, 0.0]
  mounts = camera_geometry.CameraMounts(head, wrist)
  base_pose = np.array([[5.0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
  grasp_pose = np.array([[0.5, 0, 0.8, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
  poses = camera_geometry.camera_world_poses(base_pose, grasp_pose, mounts)
  np.testing.assert_allclose(poses["robot_0_head"][:3, 3], [5.0, 0.0, 1.0])
  np.testing.assert_allclose(poses["robot_0_wrist"][:3, 3], [5.5, 0.1, 0.8])
  assert set(poses) == {
    "robot_0_head",
    "robot_1_head",
    "robot_0_wrist",
    "robot_1_wrist",
  }
