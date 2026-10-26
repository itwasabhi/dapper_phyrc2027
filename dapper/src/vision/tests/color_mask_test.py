import cv2
import numpy as np

from vision import color_mask

SHIRT_BLUE = (20, 30, 230)
TRIM_YELLOW = (230, 215, 15)
WOOD = (150, 110, 80)


def _image():
  return np.full((64, 64, 3), 128, np.uint8)


def test_mask_covers_body_and_adjacent_trim():
  rgb = _image()
  rgb[20:40, 10:30] = SHIRT_BLUE
  rgb[20:22, 30:34] = TRIM_YELLOW
  mask = color_mask.get_shirt_mask(rgb)
  assert mask[30, 20] and mask[21, 32]
  assert color_mask.get_mask_box(mask) == (10.0, 20.0, 34.0, 40.0)


def test_ignores_isolated_yellow_wood_and_gray():
  rgb = _image()
  rgb[5:20, 40:60] = TRIM_YELLOW
  rgb[40:60, 5:25] = WOOD
  mask = color_mask.get_shirt_mask(rgb)
  assert not mask.any()
  assert color_mask.get_mask_box(mask) is None


def test_drops_masks_below_min_area():
  rgb = _image()
  rgb[0:4, 0:4] = SHIRT_BLUE
  assert not color_mask.get_shirt_mask(rgb).any()


def test_shadowed_blue_still_counts():
  rgb = _image()
  rgb[20:40, 20:40] = SHIRT_BLUE
  hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
  hsv[20:40, 20:40, 2] = 70
  assert color_mask.get_shirt_mask(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)).sum() >= 300


def test_parts_separate_body_and_trim():
  rgb = _image()
  rgb[20:40, 10:30] = SHIRT_BLUE
  rgb[20:22, 30:34] = TRIM_YELLOW
  rgb[50:60, 50:60] = TRIM_YELLOW
  body, trim = color_mask.get_shirt_parts(rgb)
  assert body[30, 20] and not body[21, 32]
  assert trim[21, 32] and not trim[55, 55]
