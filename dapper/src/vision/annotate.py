"""Draws detections on camera images."""

from typing import Mapping, Sequence

import cv2
import numpy as np

from vision import detections as detections_lib

import utils_video

# BGR.
CLASS_COLORS: Mapping[str, tuple[int, int, int]] = {
  "box": (0, 165, 255),
  "shirt": (255, 0, 0),
  "manikin": (0, 255, 0),
  "chair": (0, 255, 255),
  "robot": (255, 0, 255),
}
DEFAULT_COLOR = (255, 255, 255)


def annotate_frame(
  rgb: np.ndarray, detections: Sequence[detections_lib.Detection]
) -> np.ndarray:
  """Returns a BGR copy of an RGB image with boxes and labels drawn."""
  frame = np.ascontiguousarray(rgb[..., ::-1])
  for detection in detections:
    color = CLASS_COLORS.get(detection.class_name, DEFAULT_COLOR)
    x_min, y_min, x_max, y_max = (int(round(v)) for v in detection.box_xyxy)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 1)
    cv2.putText(
      frame,
      f"{detection.class_name} {detection.score:.2f}",
      (x_min, max(y_min - 3, 8)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.35,
      color,
      1,
      cv2.LINE_AA,
    )
  return frame


def annotate_batch(
  rgb_batch: np.ndarray,
  detections_per_camera: Sequence[Sequence[detections_lib.Detection]],
) -> np.ndarray:
  """Tiles annotated cameras, each titled with its name, into one BGR image."""
  frames = [
    utils_video.label(annotate_frame(rgb, detections), name)
    for rgb, detections, name in zip(
      rgb_batch, detections_per_camera, detections_lib.CAMERA_NAMES
    )
  ]
  return utils_video.tile(frames)
