"""Shirt labels on public camera frames, a reference footprint lifted from them, and detector metrics.

Labels are proposed from the color prior (`color_mask`) and reviewed by a human. They never come
from simulator state, so they may be used for evaluation and training (docs/competition/RULES.md).
"""

import dataclasses
import json
import pathlib
from typing import Iterable, Sequence

import cv2
import numpy as np

from vision import color_mask
from vision import scene as scene_lib

LABELS_FILE = "labels.json"
MASKS_FILE = "masks.npz"
# Optional human annotation: {"<policy.hdf5 path>": hem-to-collar heading in world degrees}.
HEADINGS_FILE = "headings.json"
# Review states; "reject" frames are dropped from evaluation and training.
REVIEW_STATES = ("unreviewed", "ok", "reject")


@dataclasses.dataclass(frozen=True)
class FrameLabel:
  id: int
  source: str
  step: int
  camera: str
  box_xyxy: tuple[float, float, float, float] | None
  mask_area_px: int
  review: str = "unreviewed"


def write_labels(
  output_dir: pathlib.Path,
  labels: Sequence[FrameLabel],
  masks: dict[int, np.ndarray],
  config: color_mask.ShirtColorConfig,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  payload = {
    "color_config": dataclasses.asdict(config),
    "frames": [dataclasses.asdict(label) for label in labels],
  }
  (output_dir / LABELS_FILE).write_text(json.dumps(payload, indent=1))
  np.savez_compressed(
    output_dir / MASKS_FILE,
    **{str(index): np.packbits(mask, axis=None) for index, mask in masks.items()},
  )


def read_labels(label_dir: pathlib.Path) -> list[FrameLabel]:
  payload = json.loads((label_dir / LABELS_FILE).read_text())
  labels = []
  for frame in payload["frames"]:
    if frame["review"] not in REVIEW_STATES:
      raise ValueError(f"Frame {frame['id']}: review must be one of {REVIEW_STATES}")
    box = tuple(frame["box_xyxy"]) if frame["box_xyxy"] is not None else None
    labels.append(FrameLabel(**{**frame, "box_xyxy": box}))
  return labels


def read_mask(
  label_dir: pathlib.Path, label_id: int, shape: tuple[int, int]
) -> np.ndarray:
  with np.load(label_dir / MASKS_FILE) as masks:
    if str(label_id) not in masks:
      return np.zeros(shape, bool)
    bits = np.unpackbits(masks[str(label_id)])[: shape[0] * shape[1]]
  return bits.reshape(shape).astype(bool)


def read_headings(label_dir: pathlib.Path) -> dict[str, float]:
  path = label_dir / HEADINGS_FILE
  if not path.is_file():
    return {}
  return {
    source: float(heading) for source, heading in json.loads(path.read_text()).items()
  }


def get_reference_footprint(
  world_points: np.ndarray, trim_fraction: float = 0.05
) -> scene_lib.Pose2D | None:
  """Rectangle around lifted shirt pixels after dropping the farthest points from the median."""
  if len(world_points) < 10:
    return None
  xy = world_points[:, :2]
  distance = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
  kept = xy[distance <= np.quantile(distance, 1 - trim_fraction)]
  return scene_lib.fit_rect(kept, 1.0)


def get_footprint_iou(first: scene_lib.Pose2D, second: scene_lib.Pose2D) -> float:
  polygons = [
    np.array(scene_lib.get_corners(pose), np.float32) for pose in (first, second)
  ]
  intersection, _ = cv2.intersectConvexConvex(*polygons)
  union = sum(cv2.contourArea(polygon) for polygon in polygons) - intersection
  return float(intersection / union) if union > 0 else 0.0


def get_box_iou(first: Sequence[float], second: Sequence[float]) -> float:
  width = min(first[2], second[2]) - max(first[0], second[0])
  height = min(first[3], second[3]) - max(first[1], second[1])
  intersection = max(width, 0.0) * max(height, 0.0)
  area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
  union = area(first) + area(second) - intersection
  return intersection / union if union > 0 else 0.0


@dataclasses.dataclass(frozen=True)
class FrameDetections:
  """Labeled box (None when the shirt is not visible) and scored shirt-class detection boxes."""

  label_box: tuple[float, float, float, float] | None
  scored_boxes: tuple[tuple[float, tuple[float, float, float, float]], ...]


def summarize(
  frames: Iterable[FrameDetections],
  score_threshold: float,
  iou_threshold: float = 0.5,
) -> dict[str, float]:
  """Recall/precision at a score threshold and all-point AP; each frame has at most one shirt."""
  frames = list(frames)
  num_positive = sum(frame.label_box is not None for frame in frames)
  num_negative = len(frames) - num_positive
  ranked = []
  for index, frame in enumerate(frames):
    for score, box in frame.scored_boxes:
      iou = get_box_iou(box, frame.label_box) if frame.label_box else 0.0
      ranked.append((score, index, iou >= iou_threshold))
  ranked.sort(key=lambda item: -item[0])
  matched = set()
  true_positive_flags = []
  for _, index, overlaps in ranked:
    is_match = overlaps and index not in matched
    if is_match:
      matched.add(index)
    true_positive_flags.append(is_match)
  flags = np.array(true_positive_flags, bool)
  scores = np.array([item[0] for item in ranked], float)
  above = scores >= score_threshold
  true_positives = int(flags[above].sum())
  false_positives = int((~flags[above]).sum())
  negative_frames_with_detection = sum(
    1
    for frame in frames
    if frame.label_box is None
    and any(score >= score_threshold for score, _ in frame.scored_boxes)
  )
  return {
    "frames": len(frames),
    "positives": num_positive,
    "recall": true_positives / num_positive if num_positive else float("nan"),
    "precision": (
      true_positives / (true_positives + false_positives)
      if true_positives + false_positives
      else float("nan")
    ),
    "negative_frame_fp_rate": (
      negative_frames_with_detection / num_negative if num_negative else float("nan")
    ),
    "average_precision": _average_precision(flags, num_positive),
  }


def _average_precision(flags: np.ndarray, num_positive: int) -> float:
  if num_positive == 0:
    return float("nan")
  if len(flags) == 0:
    return 0.0
  true_positives = np.cumsum(flags)
  precision = true_positives / np.arange(1, len(flags) + 1)
  recall = true_positives / num_positive
  # Monotone precision envelope, integrated over recall steps.
  envelope = np.maximum.accumulate(precision[::-1])[::-1]
  recall_steps = np.diff(np.concatenate([[0.0], recall]))
  return float((envelope * recall_steps).sum())
