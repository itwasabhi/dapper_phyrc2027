"""Scene estimate produced by a scene integrator and consumed by solutions."""

import dataclasses
import math

import cv2
import numpy as np

from vision import detections as detections_lib

FRONT_STANDOFF_M = 0.8
FRONT_LATERAL_M = 0.35
TABLE_STANDOFF_M = 0.7


@dataclasses.dataclass(frozen=True)
class Pose2D:
  """World-frame footprint; `yaw` is the heading of the `length` axis."""

  x: float
  y: float
  yaw: float
  length: float
  width: float
  confidence: float


@dataclasses.dataclass(frozen=True)
class SceneEstimate:
  tables: tuple[Pose2D, ...] = ()
  # yaw points from hem to collar; length runs along it.
  shirt: Pose2D | None = None
  shirt_table: int | None = None
  # Score gap between the best and second-best shirt heading; near 0 means ambiguous.
  shirt_heading_margin: float = 0.0
  # Bottom (hem) corners as world (x, y), left then right as seen facing the collar.
  shirt_hem_corners: tuple[tuple[float, float], tuple[float, float]] | None = None
  # False when the hem band was not visible and the corners are inferred from the footprint.
  shirt_hem_seen: bool = False
  # World z of the shirt's top surface.
  shirt_height_m: float | None = None
  manikin: Pose2D | None = None  # yaw is the direction the manikin faces.
  debug_image: np.ndarray | None = None  # RGB top-down map.
  debug_detections: tuple[tuple[detections_lib.Detection, ...], ...] = ()


def to_jsonable(scene: SceneEstimate) -> dict:
  return {
    "tables": [dataclasses.asdict(table) for table in scene.tables],
    "shirt": dataclasses.asdict(scene.shirt) if scene.shirt else None,
    "shirt_table": scene.shirt_table,
    "shirt_heading_margin": scene.shirt_heading_margin,
    "shirt_hem_corners": scene.shirt_hem_corners,
    "shirt_hem_seen": scene.shirt_hem_seen,
    "shirt_height_m": scene.shirt_height_m,
    "manikin": dataclasses.asdict(scene.manikin) if scene.manikin else None,
  }


def manikin_front_poses(scene: SceneEstimate) -> list[Pose2D] | None:
  """Two standoff poses in front of the manikin, facing it."""
  manikin = scene.manikin
  if manikin is None:
    return None
  forward = np.array([math.cos(manikin.yaw), math.sin(manikin.yaw)])
  left = np.array([-forward[1], forward[0]])
  center = np.array([manikin.x, manikin.y])
  poses = []
  for lateral in (FRONT_LATERAL_M, -FRONT_LATERAL_M):
    position = center + FRONT_STANDOFF_M * forward + lateral * left
    poses.append(
      Pose2D(
        float(position[0]),
        float(position[1]),
        wrap_angle(manikin.yaw + math.pi),
        0.0,
        0.0,
        manikin.confidence,
      )
    )
  return poses


def shirt_approach_pose(
  scene: SceneEstimate, robot_xy: tuple[float, float]
) -> Pose2D | None:
  """Standoff pose beside the shirt on the side closest to the robot, facing it."""
  shirt = scene.shirt
  if shirt is None:
    return None
  shirt_xy = np.array([shirt.x, shirt.y])
  to_robot = np.asarray(robot_xy, dtype=float) - shirt_xy
  axes = [
    np.array([math.cos(shirt.yaw), math.sin(shirt.yaw)]),
    np.array([-math.sin(shirt.yaw), math.cos(shirt.yaw)]),
  ]
  half_extents = (shirt.length / 2, shirt.width / 2)
  best = max(
    (
      (axis * sign, extent)
      for axis, extent in zip(axes, half_extents)
      for sign in (1, -1)
    ),
    key=lambda candidate: float(candidate[0] @ to_robot),
  )
  direction, extent = best
  position = shirt_xy + direction * (extent + TABLE_STANDOFF_M)
  heading = math.atan2(-direction[1], -direction[0])
  return Pose2D(
    float(position[0]), float(position[1]), heading, 0.0, 0.0, shirt.confidence
  )


def get_shirt_table_corner_poses(
  scene: SceneEstimate, toward_xy: tuple[float, float]
) -> list[Pose2D] | None:
  """Standoff poses at both ends of the shirt table's edge facing `toward_xy`, facing the table."""
  if scene.shirt_table is None:
    return None
  table = scene.tables[scene.shirt_table]
  table_xy = np.array([table.x, table.y])
  to_target = np.asarray(toward_xy, dtype=float) - table_xy
  axes = [
    np.array([math.cos(table.yaw), math.sin(table.yaw)]),
    np.array([-math.sin(table.yaw), math.cos(table.yaw)]),
  ]
  half_extents = (table.length / 2, table.width / 2)
  candidates = [
    (axes[index] * sign, half_extents[index], axes[1 - index], half_extents[1 - index])
    for index in range(2)
    for sign in (1, -1)
  ]
  normal, normal_extent, tangent, tangent_extent = max(
    candidates, key=lambda candidate: float(candidate[0] @ to_target)
  )
  heading = math.atan2(-normal[1], -normal[0])
  edge_center = table_xy + normal * (normal_extent + TABLE_STANDOFF_M)
  return [
    Pose2D(float(position[0]), float(position[1]), heading, 0.0, 0.0, table.confidence)
    for position in (
      edge_center + tangent * tangent_extent,
      edge_center - tangent * tangent_extent,
    )
  ]


def fit_rect(points: np.ndarray, confidence: float) -> Pose2D:
  """Minimum-area rectangle of (N, 2) points; yaw follows the long edge, in [0, pi)."""
  corners = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
  edges = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
  long_edge = max(edges[:2], key=np.linalg.norm)
  other = min(edges[:2], key=np.linalg.norm)
  center = corners.mean(axis=0)
  return Pose2D(
    float(center[0]),
    float(center[1]),
    math.atan2(long_edge[1], long_edge[0]) % math.pi,
    float(np.linalg.norm(long_edge)),
    float(np.linalg.norm(other)),
    confidence,
  )


def get_corners(pose: Pose2D) -> list[np.ndarray]:
  forward = np.array([math.cos(pose.yaw), math.sin(pose.yaw)]) * pose.length / 2
  left = np.array([-math.sin(pose.yaw), math.cos(pose.yaw)]) * pose.width / 2
  center = np.array([pose.x, pose.y])
  return [
    center + forward + left,
    center - forward + left,
    center - forward - left,
    center + forward - left,
  ]


def contains_point(pose: Pose2D, xy: np.ndarray) -> bool:
  offset = np.asarray(xy) - np.array([pose.x, pose.y])
  along = offset @ np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
  across = offset @ np.array([-math.sin(pose.yaw), math.cos(pose.yaw)])
  return abs(along) <= pose.length / 2 and abs(across) <= pose.width / 2


def wrap_angle(angle: float) -> float:
  return (angle + math.pi) % (2 * math.pi) - math.pi
