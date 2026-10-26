"""Fuses per-frame detections and depth from the robot cameras into a world-frame scene estimate."""

import dataclasses
import math

import cv2
import numpy as np

from vision import camera_geometry
from vision import color_mask
from vision import detections as detections_lib
from vision import detector_factory
from vision import scene as scene_lib
from vision import shirt_orientation

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
  # A shirt box votes only on its color-mask pixels when at least this many fall inside it.
  shirt_min_color_pixels: int = 10
  # With a color prior, drop shirt boxes holding too few shirt-colored pixels (e.g. the manikin's
  # own striped top) instead of voting on the whole box.
  shirt_require_color: bool = True
  # Box filter (cells) over shirt votes and seen counts; fills gaps from sparse far-range samples.
  shirt_smooth_cells: int = 5
  # A shirt cell is trim when this share of its shirt-colored samples are trim-colored.
  shirt_trim_share: float = 0.5
  shirt_trim_min_samples: int = 2
  # Trim is searched this far around the shirt component, which can miss sparsely seen edges.
  shirt_trim_search_m: float = 0.1
  orientation: shirt_orientation.OrientationConfig = (
    shirt_orientation.OrientationConfig()
  )
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
    shirt_color: color_mask.ShirtColorConfig | None = color_mask.ShirtColorConfig(),
  ):
    """`shirt_color` refines shirt boxes to shirt-colored pixels; None votes on whole boxes."""
    self._detector = detector
    self._mounts = mounts
    self._config = config
    self._shirt_color = shirt_color
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
    self._shirt_body = np.zeros(self._shape, np.float32)
    self._shirt_trim = np.zeros(self._shape, np.float32)
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
        rgb[index + 1],
        depth[index + 1],
        valid[index + 1],
        detections[index],
        world_from_optical[name],
        base_pose[:, :2],
      )
    self._scene = self._fit(base_pose[:, :2], detections)
    return self._scene

  def _integrate(
    self, name, rgb, depth, valid, detections, world_from_optical, robot_xy
  ):
    config = self._config
    stride = config.pixel_stride
    shirt_pixels = shirt_body = shirt_trim = None
    if self._shirt_color is not None and any(
      detection.class_name == "shirt" for detection in detections
    ):
      shirt_pixels = color_mask.get_shirt_mask(rgb, self._shirt_color)[
        ::stride, ::stride
      ]
      shirt_body, shirt_trim = (
        part[::stride, ::stride]
        for part in color_mask.get_shirt_parts(rgb, self._shirt_color)
      )
    depth = depth[::stride, ::stride, 0]
    valid = valid[::stride, ::stride, 0]
    is_wrist = name in camera_geometry.WRIST_CAMERA_NAMES
    near = (depth > 0) & (depth < config.wrist_min_depth_m)
    if is_wrist:
      valid = valid & ~near
    score_map = self._score_map(
      detections, near if is_wrist else None, depth.shape, shirt_pixels
    )

    world, pixel_index = camera_geometry.get_world_points(
      depth, valid, world_from_optical
    )
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
    if shirt_pixels is not None:
      in_shirt_box = score_map[CLASSES.index("shirt")] > 0
      for part, accumulator in (
        (shirt_body, self._shirt_body),
        (shirt_trim, self._shirt_trim),
      ):
        weights = (in_shirt_box & part).reshape(-1)[pixel_index]
        accumulator += np.bincount(flat, weights=weights, minlength=size).reshape(
          self._shape
        )

  def _score_map(self, detections, near, image_shape, shirt_pixels):
    config = self._config
    stride = config.pixel_stride
    score_map = np.zeros((len(CLASSES), *image_shape), np.float32)
    for detection in detections:
      x0, y0, x1, y1 = (value / stride for value in detection.box_xyxy)
      if near is not None:
        box_near = near[max(int(y0), 0) : int(y1) + 1, max(int(x0), 0) : int(x1) + 1]
        if box_near.size and box_near.mean() > config.wrist_near_box_fraction:
          continue
      class_map = score_map[CLASSES.index(detection.class_name)]
      if detection.class_name == "shirt" and shirt_pixels is not None:
        rows = slice(max(int(y0), 0), int(math.ceil(y1)))
        cols = slice(max(int(x0), 0), int(math.ceil(x1)))
        inside = shirt_pixels[rows, cols]
        if inside.sum() >= config.shirt_min_color_pixels:
          region = class_map[rows, cols]
          region[inside] = np.maximum(region[inside], detection.score)
          continue
        if config.shirt_require_color:
          continue
      half_w, half_h = (
        (x1 - x0) / 2 * config.box_shrink,
        (y1 - y0) / 2 * config.box_shrink,
      )
      center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
      rows = slice(max(int(center_y - half_h), 0), int(center_y + half_h) + 1)
      cols = slice(max(int(center_x - half_w), 0), int(center_x + half_w) + 1)
      region = class_map[rows, cols]
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
    shirt_fields = self._fit_shirt(seen, height, table_mask, tables)
    manikin = self._fit_manikin(seen, height, belief, table_mask)
    result = scene_lib.SceneEstimate(tables=tables, manikin=manikin, **shirt_fields)
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
      tables.append(scene_lib.fit_rect(self._world_xy(component), mean_belief))
      kept |= component
    tables.sort(key=lambda pose: (pose.x, pose.y))
    return tuple(tables), kept

  def _fit_shirt(self, seen, height, table_mask, tables) -> dict:
    """Shirt fields of SceneEstimate; empty when no shirt is found."""
    config = self._config
    size = (config.shirt_smooth_cells,) * 2
    votes = cv2.boxFilter(
      self._votes[CLASSES.index("shirt")], -1, size, normalize=False
    )
    counts = cv2.boxFilter(self._seen, -1, size, normalize=False)
    shirt_belief = votes / np.maximum(counts, 1)
    if table_mask.any():
      support = cv2.dilate(
        table_mask.astype(np.uint8), np.ones((5, 5), np.uint8)
      ).astype(bool)
    else:
      low, high = config.table_height_range_m
      support = (height >= low) & (height <= high + config.table_band_m)
    cells = seen & (shirt_belief >= config.shirt_belief) & support
    components = self._components(cells, 10)
    if not components:
      return {}
    component = components[0]
    orientation = shirt_orientation.estimate_orientation(
      self._world_xy(component),
      self._world_xy(self._get_trim_cells(component)),
      float(shirt_belief[component].mean()),
      config.orientation,
    )
    # Cell centers span one cell less than the cells themselves.
    pose = dataclasses.replace(
      orientation.pose,
      length=orientation.pose.length + config.cell_m,
      width=orientation.pose.width + config.cell_m,
    )
    center = np.array([pose.x, pose.y])
    containing = [
      i for i, table in enumerate(tables) if scene_lib.contains_point(table, center)
    ]
    return {
      "shirt": pose,
      "shirt_table": containing[0] if containing else None,
      "shirt_heading_margin": orientation.margin,
      "shirt_hem_corners": tuple(
        (float(corner[0]), float(corner[1])) for corner in orientation.hem_corners
      ),
      "shirt_hem_seen": orientation.hem_seen,
      "shirt_height_m": float(np.median(height[component])),
    }

  def _get_trim_cells(self, component: np.ndarray) -> np.ndarray:
    """Cells near the shirt component whose shirt-colored samples are mostly trim."""
    config = self._config
    samples = self._shirt_body + self._shirt_trim
    trim = (samples >= config.shirt_trim_min_samples) & (
      self._shirt_trim >= config.shirt_trim_share * samples
    )
    radius = round(config.shirt_trim_search_m / config.cell_m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    near = cv2.dilate(component.astype(np.uint8), kernel)
    return trim & near.astype(bool)

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
    rect = scene_lib.fit_rect(footprint, float(belief["manikin"][component].mean()))
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
      corners = np.array([to_pixel(c) for c in scene_lib.get_corners(pose)], np.int32)
      cv2.polylines(image, [corners], True, color, 1)
    for corner in scene.shirt_hem_corners or ():
      cv2.circle(image, to_pixel(corner), 3, (255, 255, 0), -1)
    if scene.shirt is not None:
      tip = [
        scene.shirt.x + 0.4 * math.cos(scene.shirt.yaw),
        scene.shirt.y + 0.4 * math.sin(scene.shirt.yaw),
      ]
      cv2.arrowedLine(
        image, to_pixel([scene.shirt.x, scene.shirt.y]), to_pixel(tip), (0, 255, 255), 1
      )
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
