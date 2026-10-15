import json
import pathlib

import cv2
import h5py
import numpy as np

import utils_video

CAMERAS = [
  {"id": "cam_a", "clip_m": [0.05, 10.0]},
  {"id": "cam_b", "clip_m": [0.05, 5.0]},
]
SIZE = 16
NUM_SAMPLES = 5


def _write_hdf5(path: pathlib.Path) -> None:
  with h5py.File(path, "w") as f:
    f.attrs["contract_json"] = json.dumps({"cameras": CAMERAS})
    demo = f.create_group("data/demo_0")
    demo.attrs["num_samples"] = NUM_SAMPLES
    for group in ("obs", "next_obs"):
      g = demo.create_group(group)
      for camera in CAMERAS:
        name = camera["id"]
        g[f"{name}_rgb"] = np.full((NUM_SAMPLES, SIZE, SIZE, 3), 100, np.uint8)
        g[f"{name}_depth"] = np.full((NUM_SAMPLES, SIZE, SIZE, 1), 2.0, np.float32)
        g[f"{name}_depth_valid"] = np.ones((NUM_SAMPLES, SIZE, SIZE, 1), bool)


def _num_frames(path):
  capture = cv2.VideoCapture(str(path))
  count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  capture.release()
  return count


def test_tile_pads_to_grid():
  frames = [np.ones((4, 4, 3), np.uint8)] * 2
  assert utils_video.tile(frames, num_columns=3).shape == (4, 12, 3)
  assert utils_video.tile(frames * 2, num_columns=3).shape == (8, 12, 3)


def test_invalid_depth_is_black():
  depth = np.ones((2, 2, 1), np.float32)
  valid = np.array([[[True], [False]], [[True], [True]]])
  image = utils_video.depth_to_bgr(depth, valid, far_m=2.0)
  assert image[0, 1].sum() == 0
  assert image[0, 0].sum() > 0


def test_write_videos_includes_final_observation(tmp_path):
  _write_hdf5(tmp_path / "policy.hdf5")
  paths = utils_video.write_videos(tmp_path / "policy.hdf5", tmp_path / "v")
  assert sorted(p.name for p in paths) == ["demo_0_depth.mp4", "demo_0_rgb.mp4"]
  assert _num_frames(paths[0]) == NUM_SAMPLES + 1


def test_stride_and_no_depth(tmp_path):
  _write_hdf5(tmp_path / "policy.hdf5")
  paths = utils_video.write_videos(
    tmp_path / "policy.hdf5", tmp_path / "v", stride=2, include_depth=False
  )
  assert [p.name for p in paths] == ["demo_0_rgb.mp4"]
  assert _num_frames(paths[0]) == 3 + 1
