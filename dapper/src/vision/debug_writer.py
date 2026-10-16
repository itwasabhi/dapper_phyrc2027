"""Writes annotated detection frames, a video and a JSONL log for a rollout."""

import json
import pathlib
from typing import Sequence

import cv2
import numpy as np

from vision import annotate
from vision import detections as detections_lib

import utils_video

PNG_EVERY_N_FRAMES = 10


class DebugWriter:
  """Streams outputs incrementally; files are created on the first write."""

  def __init__(self, output_dir: pathlib.Path, video_fps: float):
    self._dir = output_dir
    self._video_fps = video_fps
    self._video = None
    self._log = None
    self._num_frames = 0

  def write(
    self,
    step: int,
    rgb_batch: np.ndarray,
    detections_per_camera: Sequence[Sequence[detections_lib.Detection]],
  ) -> None:
    frame = annotate.annotate_batch(rgb_batch, detections_per_camera)
    if self._log is None:
      self._dir.mkdir(parents=True, exist_ok=True)
      self._log = open(self._dir / "detections.jsonl", "w")
    if self._video is None:
      self._video = cv2.VideoWriter(
        str(self._dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*utils_video.CODEC),
        self._video_fps,
        (frame.shape[1], frame.shape[0]),
      )
      if not self._video.isOpened():
        raise RuntimeError("VideoWriter would not open annotated.mp4")
    self._video.write(frame)
    if self._num_frames % PNG_EVERY_N_FRAMES == 0:
      cv2.imwrite(str(self._dir / f"step_{step:05d}.png"), frame)
    record = {
      "step": step,
      "detections": detections_lib.to_jsonable(detections_per_camera),
    }
    self._log.write(json.dumps(record) + "\n")
    self._log.flush()
    self._num_frames += 1

  def close(self) -> None:
    if self._video is not None:
      self._video.release()
      self._video = None
    if self._log is not None:
      self._log.close()
      self._log = None
