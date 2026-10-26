"""Color prior for the T-shirt (saturated blue body, yellow trim) measured from public RGB frames."""

import dataclasses

import cv2
import numpy as np

# Box (x0, y0, x1, y1) in pixels, inclusive-exclusive like the detector boxes.
BoxXyxy = tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class ShirtColorConfig:
  """OpenCV HSV ranges (hue 0-179). Wood (hue 10-14, low saturation) is the nearest distractor."""

  body_hue: tuple[int, int] = (110, 135)
  body_min_saturation: int = 100
  body_min_value: int = 40
  trim_hue: tuple[int, int] = (20, 38)
  trim_min_saturation: int = 90
  trim_min_value: int = 120
  # Trim only counts next to the body, so stray yellow elsewhere is ignored.
  trim_max_distance_px: int = 5
  close_px: int = 3
  min_area_px: int = 40


def get_shirt_parts(
  rgb: np.ndarray, config: ShirtColorConfig = ShirtColorConfig()
) -> tuple[np.ndarray, np.ndarray]:
  """Raw boolean (H, W) body and trim masks; trim is limited to pixels near the body."""
  hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
  hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
  body = (
    (hue >= config.body_hue[0])
    & (hue <= config.body_hue[1])
    & (saturation >= config.body_min_saturation)
    & (value >= config.body_min_value)
  )
  trim = (
    (hue >= config.trim_hue[0])
    & (hue <= config.trim_hue[1])
    & (saturation >= config.trim_min_saturation)
    & (value >= config.trim_min_value)
  )
  near_body = cv2.dilate(
    body.astype(np.uint8), _disk(config.trim_max_distance_px)
  ).astype(bool)
  return body, trim & near_body


def get_shirt_mask(
  rgb: np.ndarray, config: ShirtColorConfig = ShirtColorConfig()
) -> np.ndarray:
  """Boolean (H, W) mask of shirt-colored pixels; empty if fewer than min_area_px."""
  body, trim = get_shirt_parts(rgb, config)
  mask = (body | trim).astype(np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disk(config.close_px)).astype(bool)
  if mask.sum() < config.min_area_px:
    return np.zeros_like(mask)
  return mask


def get_mask_box(mask: np.ndarray) -> BoxXyxy | None:
  """Bounding box of all mask pixels; the scene has a single shirt, so parts are merged."""
  rows, cols = np.nonzero(mask)
  if len(rows) == 0:
    return None
  return (
    float(cols.min()),
    float(rows.min()),
    float(cols.max() + 1),
    float(rows.max() + 1),
  )


def _disk(radius: int) -> np.ndarray:
  return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
