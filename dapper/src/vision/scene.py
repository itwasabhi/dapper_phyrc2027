"""Scene estimate produced by a scene integrator and consumed by solutions."""

import dataclasses
import math

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
  shirt: Pose2D | None = None
  shirt_table: int | None = None
  manikin: Pose2D | None = None  # yaw is the direction the manikin faces.
  debug_image: np.ndarray | None = None  # RGB top-down map.
  debug_detections: tuple[tuple[detections_lib.Detection, ...], ...] = ()


def to_jsonable(scene: SceneEstimate) -> dict:
  return {
    "tables": [dataclasses.asdict(table) for table in scene.tables],
    "shirt": dataclasses.asdict(scene.shirt) if scene.shirt else None,
    "shirt_table": scene.shirt_table,
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
        _wrap(manikin.yaw + math.pi),
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


def _wrap(angle: float) -> float:
  return (angle + math.pi) % (2 * math.pi) - math.pi
