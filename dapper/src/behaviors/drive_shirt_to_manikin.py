"""drive_shirt_to_manikin: carries the grasped shirt, as a rigid pair, to the manikin's front.

The goal is chosen once, from the measured gripper positions at the first step: the hem line
(between the grippers) ends up parallel to the manikin's front, `hem_standoff_m` from its face,
with the collar side pointing away from the manikin.
"""

import dataclasses
import math

import numpy as np

from behaviors import base
from behaviors import collision
from behaviors import drive_formation
from behaviors import motion
from vision import scene as scene_lib


@dataclasses.dataclass(frozen=True)
class Config:
  drive: drive_formation.Config = drive_formation.Config()
  # Hem midpoint distance from the manikin's front face; grows if the base poses collide.
  hem_standoff_m: float = 0.3
  max_hem_standoff_m: float = 1.0
  standoff_step_m: float = 0.05
  # Width of the strip covering the arms and hem between the two bases.
  span_width_m: float = 0.3


class DriveShirtToManikin:
  """Wraps DriveFormation with a goal that squares the held hem to the manikin's front."""

  def __init__(
    self,
    cfg: base.BehaviorCfg = base.BehaviorCfg(),
    config: Config = Config(),
  ):
    self._config = config
    self._drive = drive_formation.DriveFormation(
      self._get_goal, cfg=cfg, config=config.drive, payload_fn=self._get_payload
    )
    self._payload: list[scene_lib.Pose2D] = []
    self._state: base.BehaviorState | None = None
    self._goal: scene_lib.Pose2D | None = None
    self._planned = False

  def reset(self, state: base.BehaviorState) -> None:
    self._state = state
    self._goal = None
    self._payload = []
    self._planned = False
    self._drive.reset(state)

  def step(self, observation) -> np.ndarray:
    if not self._planned:
      self._planned = True
      self._goal = self._plan_goal(observation)
    return self._drive.step(observation)

  def _get_goal(self, scene, base_pose) -> scene_lib.Pose2D | None:
    del scene, base_pose
    return self._goal

  def _get_payload(self) -> list[scene_lib.Pose2D]:
    return self._payload

  def _plan_goal(self, observation) -> scene_lib.Pose2D | None:
    scene = self._state.scene
    manikin = scene.manikin
    if manikin is None:
      return None
    base_pose = np.asarray(observation["base_pose_world"])
    grasp_pose = np.asarray(observation["grasp_pose_base"]).reshape(2, -1)
    positions = base_pose[:, :2].astype(float)
    yaws = np.array([motion.yaw_from_base_pose(pose) for pose in base_pose])
    frame = motion.formation_pose(positions[0], positions[1], yaws[0])
    slots = [
      motion.relative(frame, positions[robot], yaws[robot]) for robot in range(2)
    ]
    grippers = np.array(
      [
        positions[robot] + _rotate(grasp_pose[robot, :2], yaws[robot])
        for robot in range(2)
      ]
    )
    hem_center = grippers.mean(axis=0)
    hem_local, _ = motion.relative(frame, hem_center, 0.0)
    collar_angle = self._get_collar_angle(grippers, hem_center, scene) - frame.yaw
    self._payload = self._get_payload_rects(slots, hem_local, collar_angle, scene)

    # Collar direction (world) = manikin facing direction, so the shirt trails away from it.
    goal_yaw = motion.wrap_angle(manikin.yaw - collar_angle)
    forward = np.array([math.cos(manikin.yaw), math.sin(manikin.yaw)])
    center = np.array([manikin.x, manikin.y])
    clearance = self._config.drive.collision_config.obstacle_clearance_m
    payload_clearance = self._config.drive.payload_clearance_m
    obstacles = collision.obstacles(scene)
    config = self._config
    for standoff in np.arange(
      config.hem_standoff_m, config.max_hem_standoff_m + 1e-9, config.standoff_step_m
    ):
      hem_target = center + (manikin.length / 2 + standoff) * forward
      origin = hem_target - _rotate(hem_local, goal_yaw)
      goal = scene_lib.Pose2D(
        float(origin[0]), float(origin[1]), goal_yaw, 0.0, 0.0, manikin.confidence
      )
      base_xys = [motion.compose(goal, *slot)[0] for slot in slots]
      carried = [_to_world(goal, rect) for rect in self._payload]
      if not any(
        collision.contains(rect, base_xy, clearance)
        for rect in obstacles
        for base_xy in base_xys
      ) and not any(
        collision.rects_overlap(rect, manikin, payload_clearance) for rect in carried
      ):
        return goal
    return None

  def _get_payload_rects(
    self, slots, hem_local: np.ndarray, collar_angle: float, scene
  ) -> list[scene_lib.Pose2D]:
    """Pair-frame rects for the span between the bases and the shirt body past the hem."""
    first, second = (np.asarray(xy, dtype=float) for xy, _ in slots)
    span = second - first
    rects = [
      scene_lib.Pose2D(
        *(first + second) / 2,
        math.atan2(span[1], span[0]),
        float(np.linalg.norm(span)),
        self._config.span_width_m,
        1.0,
      )
    ]
    if scene.shirt is not None:
      direction = np.array([math.cos(collar_angle), math.sin(collar_angle)])
      body_center = hem_local + direction * scene.shirt.length / 2
      rects.append(
        scene_lib.Pose2D(
          float(body_center[0]),
          float(body_center[1]),
          collar_angle,
          scene.shirt.length,
          scene.shirt.width,
          1.0,
        )
      )
    return rects

  def _get_collar_angle(
    self, grippers: np.ndarray, hem_center: np.ndarray, scene
  ) -> float:
    """World angle of the hem-to-collar direction, perpendicular to the gripper line."""
    line = grippers[1] - grippers[0]
    normal = np.array([-line[1], line[0]])
    # Without a shirt estimate the side is arbitrary.
    if scene.shirt is not None:
      toward_collar = np.array([scene.shirt.x, scene.shirt.y]) - hem_center
      if normal @ toward_collar < 0:
        normal = -normal
    return math.atan2(normal[1], normal[0])

  def blocked(self) -> bool:
    return (self._planned and self._goal is None) or self._drive.blocked()

  def done(self) -> bool:
    return self._drive.done()

  def close(self) -> None:
    self._drive.close()


def _rotate(vector: np.ndarray, yaw: float) -> np.ndarray:
  cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
  return np.array(
    [
      cos_yaw * vector[0] - sin_yaw * vector[1],
      sin_yaw * vector[0] + cos_yaw * vector[1],
    ]
  )


def _to_world(frame: scene_lib.Pose2D, rect: scene_lib.Pose2D) -> scene_lib.Pose2D:
  center, yaw = motion.compose(frame, np.array([rect.x, rect.y]), rect.yaw)
  return dataclasses.replace(rect, x=float(center[0]), y=float(center[1]), yaw=yaw)
