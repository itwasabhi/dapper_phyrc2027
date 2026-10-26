"""Hem-to-collar heading of a flat T-shirt from its top-down footprint and trim.

Cues, in the frame of a candidate heading (u toward the collar):
- the footprint centroid sits toward the collar, where the sleeves widen the T;
- the trim centroid sits toward the hem, whose band spans the full width while the collar
  trim is a small V.
The winning heading is then refined to the normal of the hem band, a long straight line of trim,
whose ends are the shirt's bottom corners.
"""

import dataclasses
import math

import numpy as np

from vision import scene as scene_lib


@dataclasses.dataclass(frozen=True)
class OrientationConfig:
  footprint_weight: float = 1.0
  trim_weight: float = 1.0
  # Below this many trim points the trim cue is skipped.
  min_trim_points: int = 5
  # The hem band refines the heading when it is long and thin enough and close to the coarse
  # heading; otherwise the min-area rectangle axis is kept.
  hem_min_length_m: float = 0.2
  hem_min_elongation: float = 3.0
  # Beyond 45 degrees another quarter-turn of the rectangle would be closer.
  hem_max_correction_deg: float = 45.0
  # Hem ends are taken at these quantiles along the line, ignoring stray trim cells.
  hem_end_quantile: float = 0.02
  # Without a usable hem band, corners sit this fraction of the full width (sleeves included)
  # apart, at the hem edge of the footprint.
  fallback_hem_width_fraction: float = 0.65


@dataclasses.dataclass(frozen=True)
class HemLine:
  """Hem band fitted as a line; corners are its ends, left/right as seen facing the collar."""

  heading: float
  left_xy: np.ndarray
  right_xy: np.ndarray


@dataclasses.dataclass(frozen=True)
class ShirtOrientation:
  """`pose.yaw` points from hem to collar; `pose.length` runs along it."""

  pose: scene_lib.Pose2D
  # Best minus second-best candidate score; near 0 means the heading is ambiguous.
  margin: float
  scores: tuple[float, float, float, float]
  # Bottom corners of the shirt, left then right as seen facing the collar.
  hem_corners: tuple[np.ndarray, np.ndarray]
  # False when the hem band was not usable and the corners come from the footprint alone.
  hem_seen: bool


def estimate_orientation(
  footprint_xy: np.ndarray,
  trim_xy: np.ndarray,
  confidence: float,
  config: OrientationConfig = OrientationConfig(),
) -> ShirtOrientation:
  """Picks the best of the 4 min-area-rectangle headings, then aligns it to the hem band."""
  rect = scene_lib.fit_rect(footprint_xy, confidence)
  center = np.array([rect.x, rect.y])
  use_trim = len(trim_xy) >= config.min_trim_points
  scores = []
  for quarter in range(4):
    heading = rect.yaw + quarter * math.pi / 2
    axis = np.array([math.cos(heading), math.sin(heading)])
    half_extent = (rect.length if quarter % 2 == 0 else rect.width) / 2
    footprint_shift = float(np.mean((footprint_xy - center) @ axis)) / half_extent
    score = config.footprint_weight * footprint_shift
    if use_trim:
      trim_shift = float(np.mean((trim_xy - center) @ axis)) / half_extent
      score -= config.trim_weight * trim_shift
    scores.append(score)
  order = np.argsort(scores)[::-1]
  best = int(order[0])
  heading = rect.yaw + best * math.pi / 2
  hem = get_hem_line(trim_xy, center, heading, config) if use_trim else None
  if hem is not None:
    heading = hem.heading
  axis = np.array([math.cos(heading), math.sin(heading)])
  left = np.array([-axis[1], axis[0]])
  offsets = footprint_xy - center
  along = float(np.ptp(offsets @ axis))
  across = float(np.ptp(offsets @ left))
  if hem is not None:
    corners = (hem.left_xy, hem.right_xy)
  else:
    hem_center = center + float(np.min(offsets @ axis)) * axis
    half_hem = config.fallback_hem_width_fraction * across / 2
    corners = (hem_center + half_hem * left, hem_center - half_hem * left)
  return ShirtOrientation(
    pose=dataclasses.replace(
      rect, yaw=scene_lib.wrap_angle(heading), length=along, width=across
    ),
    margin=float(scores[best] - scores[order[1]]),
    scores=tuple(scores),
    hem_corners=corners,
    hem_seen=hem is not None,
  )


def get_hem_line(
  trim_xy: np.ndarray,
  center: np.ndarray,
  coarse_heading: float,
  config: OrientationConfig = OrientationConfig(),
) -> HemLine | None:
  """Line through the hem-side trim; its normal toward the collar is the refined heading."""
  axis = np.array([math.cos(coarse_heading), math.sin(coarse_heading)])
  hem = trim_xy[(trim_xy - center) @ axis < 0]
  if len(hem) < config.min_trim_points:
    return None
  hem_mean = hem.mean(axis=0)
  eigenvalues, eigenvectors = np.linalg.eigh(np.cov(hem.T))
  line = eigenvectors[:, 1]
  along_line = (hem - hem_mean) @ line
  elongation = math.sqrt(eigenvalues[1] / max(eigenvalues[0], 1e-12))
  if (
    float(np.ptp(along_line)) < config.hem_min_length_m
    or elongation < config.hem_min_elongation
  ):
    return None
  normal = np.array([-line[1], line[0]])
  if normal @ axis < 0:
    normal = -normal
  heading = math.atan2(normal[1], normal[0])
  if abs(scene_lib.wrap_angle(heading - coarse_heading)) > math.radians(
    config.hem_max_correction_deg
  ):
    return None
  left = np.array([-normal[1], normal[0]])
  if line @ left < 0:
    line, along_line = -line, -along_line
  low, high = np.quantile(
    along_line, [config.hem_end_quantile, 1 - config.hem_end_quantile]
  )
  return HemLine(heading, hem_mean + high * line, hem_mean + low * line)
