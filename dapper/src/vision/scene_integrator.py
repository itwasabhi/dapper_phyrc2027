"""Fuses per-frame detections and depth from the robot cameras into a world-frame scene estimate."""

import dataclasses
import math

import cv2
import numpy as np

from vision import camera_geometry
from vision import detections as detections_lib
from vision import detector_factory
from vision import scene as scene_lib

CLASSES = tuple(detections_lib.DEFAULT_PROMPTS)
ROBOT_CAMERAS = detections_lib.CAMERA_NAMES[1:]


@dataclasses.dataclass(frozen=True)
class GridConfig:
  cell_m: float = 0.02
  x_range_m: tuple[float, float] = (-4.0, 4.0)
  y_range_m: tuple[float, float] = (-3.0, 3.0)
  update_every_n_calls: int = 4
  pixel_stride: int = 2
  wrist_min_depth_m: float = 0.3
  wrist_near_box_fraction: float = 0.5
  robot_radius_m: float = 0.5
  floor_z_m: float = 0.03
  box_shrink: float = 0.7
  min_seen: int = 2
  table_height_range_m: tuple[float, float] = (0.3, 1.0)
  table_band_m: float = 0.06
  table_close_cells: int = 9
  min_table_area_m2: float = 0.15
  min_table_belief: float = 0.1
  shirt_belief: float = 0.3
  manikin_belief: float = 0.2
  manikin_min_height_m: float = 0.9
  head_band_m: float = 0.2
  limb_below_head_m: float = 0.4
  limb_min_height_m: float = 0.3
  limb_radius_m: float = 1.2


class BasicSceneIntegrator:
  """Accumulates depth-lifted detection votes on a world grid and fits rectangles to it."""

  def __init__(
    self,
    detector: detector_factory.Detector,
    mounts: camera_geometry.CameraMounts,
    config: GridConfig = GridConfig(),
  ):
    self._detector = detector
    self._mounts = mounts
    self._config = config
    self._origin = np.array([config.x_range_m[0], config.y_range_m[0]])
    self._shape = (
      round((config.y_range_m[1] - config.y_range_m[0]) / config.cell_m),
      round((config.x_range_m[1] - config.x_range_m[0]) / config.cell_m),
    )
    self.reset()

  def reset(self) -> None:
    self._calls = 0
    self._seen = np.zeros(self._shape, np.float32)
    self._height = np.zeros(self._shape, np.float32)
    self._votes = np.zeros((len(CLASSES), *self._shape), np.float32)
    self._scene = scene_lib.SceneEstimate()

  def update(self, observation) -> scene_lib.SceneEstimate:
    run = self._calls % self._config.update_every_n_calls == 0
    self._calls += 1
    if not run:
      return self._scene
    rgb = np.asarray(observation["rgb"])
    depth = np.asarray(observation["depth"])
    valid = np.asarray(observation["depth_valid"])
    base_pose = np.asarray(observation["base_pose_world"])
    grasp_pose = np.asarray(observation["grasp_pose_base"])
    detections = self._detector.detect(rgb[1:])
    world_from_optical = camera_geometry.camera_world_poses(
      base_pose, grasp_pose, self._mounts
    )
    for index, name in enumerate(ROBOT_CAMERAS):
      self._integrate(
        name,
        depth[index + 1],
        valid[index + 1],
        detections[index],
        world_from_optical[name],
        base_pose[:, :2],
      )
    self._scene = self._fit(base_pose[:, :2], detections)
    return self._scene

  def _integrate(self, name, depth, valid, detections, world_from_optical, robot_xy):
    config = self._config
    stride = config.pixel_stride
    depth = depth[::stride, ::stride, 0]
    valid = valid[::stride, ::stride, 0]
    is_wrist = name in camera_geometry.WRIST_CAMERA_NAMES
    near = (depth > 0) & (depth < config.wrist_min_depth_m)
    if is_wrist:
      valid = valid & ~near
    score_map = self._score_map(detections, near if is_wrist else None, depth.shape)

    points, pixel_index = camera_geometry.backproject(
      depth, valid, camera_geometry.intrinsics(depth.shape[1])
    )
    world = points @ world_from_optical[:3, :3].T + world_from_optical[:3, 3]
    keep = world[:, 2] > config.floor_z_m
    for center in robot_xy:
      keep &= np.linalg.norm(world[:, :2] - center, axis=1) > config.robot_radius_m
    cell = np.floor((world[:, :2] - self._origin) / config.cell_m).astype(int)
    keep &= (
      (cell >= 0).all(axis=1)
      & (cell[:, 0] < self._shape[1])
      & (cell[:, 1] < self._shape[0])
    )
    world, cell, pixel_index = world[keep], cell[keep], pixel_index[keep]
    flat = cell[:, 1] * self._shape[1] + cell[:, 0]
    size = self._seen.size
    self._seen += np.bincount(flat, minlength=size).reshape(self._shape)
    np.maximum.at(self._height.reshape(-1), flat, world[:, 2].astype(np.float32))
    for class_index in range(len(CLASSES)):
      weights = score_map[class_index].reshape(-1)[pixel_index]
      self._votes[class_index] += np.bincount(
        flat, weights=weights, minlength=size
      ).reshape(self._shape)

  def _score_map(self, detections, near, image_shape):
    config = self._config
    stride = config.pixel_stride
    score_map = np.zeros((len(CLASSES), *image_shape), np.float32)
    for detection in detections:
      x0, y0, x1, y1 = (value / stride for value in detection.box_xyxy)
      if near is not None:
        box_near = near[max(int(y0), 0) : int(y1) + 1, max(int(x0), 0) : int(x1) + 1]
        if box_near.size and box_near.mean() > config.wrist_near_box_fraction:
          continue
      half_w, half_h = (
        (x1 - x0) / 2 * config.box_shrink,
        (y1 - y0) / 2 * config.box_shrink,
      )
      center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
      rows = slice(max(int(center_y - half_h), 0), int(center_y + half_h) + 1)
      cols = slice(max(int(center_x - half_w), 0), int(center_x + half_w) + 1)
      region = score_map[CLASSES.index(detection.class_name), rows, cols]
      np.maximum(region, detection.score, out=region)
    return score_map

  def debug_layers(self) -> dict[str, np.ndarray]:
    """Accumulated grid layers (rows are y, columns are x): seen, height and per-class belief."""
    layers = {"seen": self._seen.copy(), "height": self._height.copy()}
    for index, name in enumerate(CLASSES):
      layers[f"belief_{name}"] = self._votes[index] / np.maximum(self._seen, 1)
    return layers

  def _world_xy(self, mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return self._origin + (np.stack([cols, rows], axis=1) + 0.5) * self._config.cell_m

  def _components(self, mask: np.ndarray, min_cells: int):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    return [
      labels == label for label in order if stats[label, cv2.CC_STAT_AREA] >= min_cells
    ]

  def _fit(self, robot_xy, detections) -> scene_lib.SceneEstimate:
    config = self._config
    seen = self._seen >= config.min_seen
    height = np.where(seen, self._height, 0.0)
    belief = {
      name: self._votes[index] / np.maximum(self._seen, 1)
      for index, name in enumerate(CLASSES)
    }
    tables, table_mask = self._fit_tables(seen, height, belief["box"])
    shirt, shirt_table = self._fit_shirt(seen, belief["shirt"], table_mask, tables)
    manikin = self._fit_manikin(seen, height, belief, table_mask)
    result = scene_lib.SceneEstimate(tables, shirt, shirt_table, manikin)
    return dataclasses.replace(
      result,
      debug_image=self._render(height, belief, result, robot_xy),
      debug_detections=_align_to_all_cameras(detections),
    )

  def _fit_tables(self, seen, height, box_belief):
    config = self._config
    low, high = config.table_height_range_m
    candidate = seen & (height >= low) & (height <= high)
    if not candidate.any():
      return (), np.zeros(self._shape, bool)
    bins = np.arange(low, high + 0.02, 0.02)
    counts, edges = np.histogram(height[candidate], bins=bins)
    peak = edges[counts.argmax()] + 0.01
    mask = candidate & (np.abs(height - peak) <= config.table_band_m)
    kernel = np.ones((config.table_close_cells,) * 2, np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    min_cells = config.min_table_area_m2 / config.cell_m**2
    tables, kept = [], np.zeros(self._shape, bool)
    for component in self._components(mask, int(min_cells)):
      mean_belief = float(box_belief[component].mean())
      if mean_belief < config.min_table_belief:
        continue
      tables.append(_fit_rect(self._world_xy(component), mean_belief))
      kept |= component
    tables.sort(key=lambda pose: (pose.x, pose.y))
    return tuple(tables), kept

  def _fit_shirt(self, seen, shirt_belief, table_mask, tables):
    config = self._config
    near_table = cv2.dilate(
      table_mask.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    cells = seen & (shirt_belief >= config.shirt_belief) & near_table
    components = self._components(cells, 10)
    if not components:
      return None, None
    component = components[0]
    points = self._world_xy(component)
    center = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
    major = eigenvectors[:, 1]
    yaw = math.atan2(major[1], major[0]) % math.pi
    along = (points - center) @ major
    across = (points - center) @ eigenvectors[:, 0]
    pose = scene_lib.Pose2D(
      float(center[0]),
      float(center[1]),
      yaw,
      float(np.ptp(along)),
      float(np.ptp(across)),
      float(shirt_belief[component].mean()),
    )
    containing = [i for i, table in enumerate(tables) if _contains(table, center)]
    return pose, (containing[0] if containing else None)

  def _fit_manikin(self, seen, height, belief, table_mask):
    config = self._config
    tall = (
      seen
      & (height >= config.manikin_min_height_m)
      & (belief["manikin"] >= config.manikin_belief)
    )
    components = self._components(tall, 20)
    if not components:
      return None
    component = components[0]
    peak = np.percentile(height[component], 98)
    head = self._world_xy(component & (height >= peak - config.head_band_m)).mean(
      axis=0
    )
    limb_cells = (
      seen
      & (height >= config.limb_min_height_m)
      & (height <= peak - config.limb_below_head_m)
      & ((belief["manikin"] + belief["chair"]) >= config.manikin_belief)
      & ~table_mask
    )
    limb_xy = self._world_xy(limb_cells)
    limb_xy = limb_xy[np.linalg.norm(limb_xy - head, axis=1) <= config.limb_radius_m]
    if len(limb_xy) == 0:
      return None
    # Arms and legs reach forward of the head, so head -> limb centroid is the facing direction.
    limbs = limb_xy.mean(axis=0)
    facing = math.atan2(limbs[1] - head[1], limbs[0] - head[0])
    footprint = np.concatenate([self._world_xy(component), limb_xy])
    rect = _fit_rect(footprint, float(belief["manikin"][component].mean()))
    return dataclasses.replace(
      rect, x=float(footprint[:, 0].mean()), y=float(footprint[:, 1].mean()), yaw=facing
    )

  def _render(self, height, belief, scene, robot_xy) -> np.ndarray:
    gray = np.clip(height / 1.8, 0, 1)
    image = np.stack([gray] * 3, axis=-1) * 0.5
    image[..., 0] += 0.5 * belief["box"]
    image[..., 1] += 0.5 * belief["manikin"]
    image[..., 2] += 0.5 * belief["shirt"]
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    def to_pixel(xy):
      return tuple(
        int(v) for v in (np.asarray(xy) - self._origin) / self._config.cell_m
      )

    footprints = [(t, (255, 160, 0)) for t in scene.tables]
    footprints += [
      (p, color)
      for p, color in ((scene.shirt, (0, 0, 255)), (scene.manikin, (0, 255, 0)))
      if p
    ]
    for pose, color in footprints:
      corners = np.array([to_pixel(c) for c in _corners(pose)], np.int32)
      cv2.polylines(image, [corners], True, color, 1)
    if scene.manikin is not None:
      tip = [
        scene.manikin.x + 0.5 * math.cos(scene.manikin.yaw),
        scene.manikin.y + 0.5 * math.sin(scene.manikin.yaw),
      ]
      cv2.arrowedLine(
        image,
        to_pixel([scene.manikin.x, scene.manikin.y]),
        to_pixel(tip),
        (255, 255, 0),
        1,
      )
    for center in robot_xy:
      cv2.circle(image, to_pixel(center), 5, (255, 0, 255), -1)
    return np.ascontiguousarray(image[::-1])


def _align_to_all_cameras(detections):
  """Re-indexes detections of the robot cameras to the full camera order."""
  return ((),) + tuple(
    tuple(dataclasses.replace(d, camera=d.camera + 1) for d in camera)
    for camera in detections
  )


def _fit_rect(points: np.ndarray, confidence: float) -> scene_lib.Pose2D:
  corners = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
  edges = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
  long_edge = max(edges[:2], key=np.linalg.norm)
  other = min(edges[:2], key=np.linalg.norm)
  center = corners.mean(axis=0)
  return scene_lib.Pose2D(
    float(center[0]),
    float(center[1]),
    math.atan2(long_edge[1], long_edge[0]) % math.pi,
    float(np.linalg.norm(long_edge)),
    float(np.linalg.norm(other)),
    confidence,
  )


def _corners(pose: scene_lib.Pose2D) -> list[np.ndarray]:
  forward = np.array([math.cos(pose.yaw), math.sin(pose.yaw)]) * pose.length / 2
  left = np.array([-math.sin(pose.yaw), math.cos(pose.yaw)]) * pose.width / 2
  center = np.array([pose.x, pose.y])
  return [
    center + forward + left,
    center - forward + left,
    center - forward - left,
    center + forward - left,
  ]


def _contains(pose: scene_lib.Pose2D, xy: np.ndarray) -> bool:
  offset = np.asarray(xy) - np.array([pose.x, pose.y])
  along = offset @ np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
  across = offset @ np.array([-math.sin(pose.yaw), math.cos(pose.yaw)])
  return abs(along) <= pose.length / 2 and abs(across) <= pose.width / 2
