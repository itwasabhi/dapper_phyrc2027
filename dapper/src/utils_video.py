"""Renders camera videos from a policy.hdf5 written by teleop or the recorder."""

import json
import pathlib
from typing import Sequence

import cv2
import h5py
import numpy as np

NUM_COLUMNS = 3
CODEC = "mp4v"  # The only MPEG-4 codec OpenCV can encode in the sim image.
POLICY_HZ = 20


def depth_to_bgr(
  depth: np.ndarray, valid: np.ndarray, far_m: float
) -> np.ndarray:
  """Colormaps metric depth; invalid pixels are black."""
  scaled = np.clip(depth[..., 0] / far_m, 0.0, 1.0)
  colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
  colored[~valid[..., 0]] = 0
  return colored


def tile(frames: Sequence[np.ndarray], num_columns: int = NUM_COLUMNS) -> np.ndarray:
  """Lays out equal-size BGR frames row-major, padding with black."""
  num_rows = -(-len(frames) // num_columns)
  blank = np.zeros_like(frames[0])
  padded = list(frames) + [blank] * (num_rows * num_columns - len(frames))
  rows = [
    np.concatenate(padded[row * num_columns : (row + 1) * num_columns], axis=1)
    for row in range(num_rows)
  ]
  return np.concatenate(rows, axis=0)


def _label(frame: np.ndarray, text: str) -> np.ndarray:
  frame = np.ascontiguousarray(frame)
  cv2.putText(
    frame, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
  )
  return frame


def _frame(
  group: h5py.Group,
  step: int,
  cameras: Sequence[dict],
  depth: bool,
) -> np.ndarray:
  frames = []
  for camera in cameras:
    name = camera["id"]
    if depth:
      frame = depth_to_bgr(
        group[f"{name}_depth"][step],
        group[f"{name}_depth_valid"][step],
        camera["clip_m"][1],
      )
    else:
      frame = cv2.cvtColor(group[f"{name}_rgb"][step], cv2.COLOR_RGB2BGR)
    frames.append(_label(frame, name))
  return tile(frames)


def _write(
  path: pathlib.Path,
  demo: h5py.Group,
  cameras: Sequence[dict],
  depth: bool,
  stride: int,
) -> None:
  num_samples = int(demo.attrs["num_samples"])
  steps = list(range(0, num_samples, stride))
  first = _frame(demo["obs"], 0, cameras, depth)
  writer = cv2.VideoWriter(
    str(path),
    cv2.VideoWriter_fourcc(*CODEC),
    POLICY_HZ / stride,
    (first.shape[1], first.shape[0]),
  )
  if not writer.isOpened():
    raise RuntimeError(f"VideoWriter would not open {path}")
  try:
    for step in steps:
      writer.write(_frame(demo["obs"], step, cameras, depth))
    # next_obs of the last sample is the final observation of the episode.
    writer.write(_frame(demo["next_obs"], num_samples - 1, cameras, depth))
  finally:
    writer.release()


def write_videos(
  hdf5_path: pathlib.Path,
  output_dir: pathlib.Path,
  stride: int = 1,
  demos: Sequence[str] | None = None,
  include_depth: bool = True,
) -> list[pathlib.Path]:
  """Writes `<demo>_rgb.mp4` (and `_depth.mp4`) tiling all cameras."""
  if stride < 1:
    raise ValueError("stride must be >= 1")
  output_dir.mkdir(parents=True, exist_ok=True)
  written = []
  with h5py.File(hdf5_path, "r") as f:
    cameras = json.loads(f.attrs["contract_json"])["cameras"]
    for name, demo in f["data"].items():
      if demos is not None and name not in demos:
        continue
      if int(demo.attrs["num_samples"]) == 0:
        continue
      for kind in ("rgb", "depth") if include_depth else ("rgb",):
        path = output_dir / f"{name}_{kind}.mp4"
        _write(path, demo, cameras, kind == "depth", stride)
        written.append(path)
  return written
