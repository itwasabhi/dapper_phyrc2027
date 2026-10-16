import numpy as np

from vision import annotate
from vision import detections


def _batch():
  return np.zeros((5, 256, 256, 3), np.uint8)


def test_annotate_batch_tiles_five_cameras():
  tiled = annotate.annotate_batch(_batch(), [[]] * 5)
  assert tiled.shape == (512, 768, 3)


def test_annotate_frame_draws_box_in_class_color():
  detection = detections.Detection(0, "shirt", "t-shirt", 0.9, (10, 20, 100, 120))
  frame = annotate.annotate_frame(_batch()[0], [detection])
  assert tuple(frame[70, 10]) == annotate.CLASS_COLORS["shirt"]
  assert frame[70, 50].sum() == 0
