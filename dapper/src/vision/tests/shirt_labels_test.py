import math

import numpy as np
import pytest

from vision import color_mask
from vision import scene as scene_lib
from vision import shirt_labels

BOX = (10.0, 10.0, 30.0, 30.0)


def test_labels_and_masks_round_trip(tmp_path):
  mask = np.zeros((16, 16), bool)
  mask[2:5, 3:9] = True
  labels = [
    shirt_labels.FrameLabel(0, "a.hdf5", 8, "robot_0_head", (3.0, 2.0, 9.0, 5.0), 18),
    shirt_labels.FrameLabel(1, "a.hdf5", 8, "robot_1_head", None, 0),
  ]
  shirt_labels.write_labels(tmp_path, labels, {0: mask}, color_mask.ShirtColorConfig())
  assert shirt_labels.read_labels(tmp_path) == labels
  np.testing.assert_array_equal(shirt_labels.read_mask(tmp_path, 0, mask.shape), mask)
  assert not shirt_labels.read_mask(tmp_path, 1, mask.shape).any()


def test_read_labels_rejects_unknown_review(tmp_path):
  label = shirt_labels.FrameLabel(0, "a", 0, "robot_0_head", None, 0, review="maybe")
  shirt_labels.write_labels(tmp_path, [label], {}, color_mask.ShirtColorConfig())
  with pytest.raises(ValueError):
    shirt_labels.read_labels(tmp_path)


def test_reference_footprint_ignores_outliers():
  rng = np.random.default_rng(0)
  points = np.column_stack(
    [rng.uniform(-0.3, 0.3, 500) + 1.0, rng.uniform(-0.2, 0.2, 500), np.full(500, 0.6)]
  )
  points[:5, :2] = 5.0
  reference = shirt_labels.get_reference_footprint(points)
  assert abs(reference.x - 1.0) < 0.03 and abs(reference.y) < 0.03
  assert abs(reference.length - 0.6) < 0.05 and abs(reference.width - 0.4) < 0.05


def test_footprint_iou():
  pose = scene_lib.Pose2D(0.0, 0.0, 0.0, 2.0, 1.0, 1.0)
  shifted = scene_lib.Pose2D(1.0, 0.0, 0.0, 2.0, 1.0, 1.0)
  rotated = scene_lib.Pose2D(0.0, 0.0, math.pi, 2.0, 1.0, 1.0)
  assert shirt_labels.get_footprint_iou(pose, rotated) == pytest.approx(1.0, abs=1e-3)
  assert shirt_labels.get_footprint_iou(pose, shifted) == pytest.approx(1 / 3, abs=1e-3)


def test_summarize_counts_matches_and_duplicates():
  frames = [
    # Hit plus a lower-scored duplicate, which is a false positive.
    shirt_labels.FrameDetections(BOX, ((0.9, BOX), (0.5, (11.0, 11.0, 30.0, 30.0)))),
    # Miss: detection in the wrong place.
    shirt_labels.FrameDetections(BOX, ((0.6, (50.0, 50.0, 60.0, 60.0)),)),
    # Negative frame with a detection.
    shirt_labels.FrameDetections(None, ((0.4, BOX),)),
    shirt_labels.FrameDetections(None, ()),
  ]
  summary = shirt_labels.summarize(frames, score_threshold=0.45)
  assert summary["positives"] == 2
  assert summary["recall"] == 0.5
  assert summary["precision"] == pytest.approx(1 / 3)
  assert summary["negative_frame_fp_rate"] == 0.0
  assert shirt_labels.summarize(frames, 0.3)["negative_frame_fp_rate"] == 0.5
  assert summary["average_precision"] == pytest.approx(0.5)
