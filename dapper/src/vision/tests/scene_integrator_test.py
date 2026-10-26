import math

import numpy as np

from vision import camera_geometry
from vision import color_mask
from vision import detections as detections_lib
from vision import detector_factory
from vision import scene_integrator
from vision import shirt_orientation

SIZE = 256
CAMERA_XY = (1.0, 0.5)
CAMERA_HEIGHT = 3.0
FOCAL = camera_geometry.intrinsics(SIZE)[0, 0]
# Coarse cells keep the test fast; shirt smoothing is off since shirts span few cells.
# Trim cells link into clusters only within trim_link_m, so it must exceed the cell size.
CONFIG = scene_integrator.GridConfig(
  cell_m=0.05,
  robot_radius_m=0.1,
  update_every_n_calls=1,
  shirt_smooth_cells=1,
  orientation=shirt_orientation.OrientationConfig(trim_link_m=0.08),
)
HEAD_0, WRIST_0 = (
  2,
  0,
)  # Index into the 4 robot cameras (wrist_0, wrist_1, head_0, head_1).


def _look_down(camera_xy):
  matrix = np.eye(4)
  matrix[:3, :3] = np.diag([1.0, -1.0, -1.0])
  matrix[:3, 3] = [camera_xy[0], camera_xy[1], CAMERA_HEIGHT]
  return matrix


def _render_depth(camera_xy, boxes):
  """Top-down depth of upright boxes given as (cx, cy, length, width, yaw, height)."""
  rows, cols = np.mgrid[0:SIZE, 0:SIZE]
  depth = np.full((SIZE, SIZE), CAMERA_HEIGHT, np.float32)
  for cx, cy, length, width, yaw, height in sorted(boxes, key=lambda box: box[-1]):
    plane_depth = CAMERA_HEIGHT - height
    x = camera_xy[0] + (cols - SIZE / 2) / FOCAL * plane_depth - cx
    y = camera_xy[1] - (rows - SIZE / 2) / FOCAL * plane_depth - cy
    along = x * math.cos(yaw) + y * math.sin(yaw)
    across = -x * math.sin(yaw) + y * math.cos(yaw)
    inside = (abs(along) <= length / 2) & (abs(across) <= width / 2)
    depth[inside] = plane_depth
  return depth[..., None]


def _pixel_box(camera_xy, plane_depth, x_range, y_range):
  to_col = lambda x: SIZE / 2 + (x - camera_xy[0]) / plane_depth * FOCAL
  to_row = lambda y: SIZE / 2 - (y - camera_xy[1]) / plane_depth * FOCAL
  return (
    to_col(x_range[0]),
    to_row(y_range[1]),
    to_col(x_range[1]),
    to_row(y_range[0]),
  )


def _detection(camera, class_name, score, box):
  return detections_lib.Detection(camera, class_name, class_name, score, box)


def _integrator(
  detections, camera_xy=CAMERA_XY, shirt_color=color_mask.ShirtColorConfig()
):
  mounts = camera_geometry.CameraMounts(
    base_from_head_optical=_look_down(camera_xy),
    grasp_from_wrist_optical=_look_down(camera_xy),
  )
  detector = detector_factory.Detector("fake", lambda rgb_batch: detections)
  return scene_integrator.BasicSceneIntegrator(detector, mounts, CONFIG, shirt_color)


def _observation(head_depth=None, wrist_depth=None, head_rgb=None):
  depth = np.zeros((5, SIZE, SIZE, 1), np.float32)
  if wrist_depth is not None:
    depth[1] = wrist_depth
  if head_depth is not None:
    depth[3] = head_depth
  base_pose = np.array([[0, 0, 0, 1, 0, 0, 0], [50, 50, 0, 1, 0, 0, 0]], np.float32)
  rgb = np.zeros((5, SIZE, SIZE, 3), np.uint8)
  if head_rgb is not None:
    rgb[3] = head_rgb
  return {
    "rgb": rgb,
    "depth": depth,
    "depth_valid": depth > 0,
    "base_pose_world": base_pose,
    "grasp_pose_base": np.tile([0, 0, 0, 1, 0, 0, 0], (2, 1)).astype(np.float32),
  }


def test_fits_table_and_shirt_pose():
  table = (1.0, 0.5, 0.8, 0.6, 0.3, 0.57)
  depth = _render_depth(CAMERA_XY, [table])
  table_depth = CAMERA_HEIGHT - 0.57
  detections = [[], [], [], []]
  detections[HEAD_0] = [
    _detection(HEAD_0, "box", 0.9, (0, 0, SIZE, SIZE)),
    _detection(
      HEAD_0, "shirt", 0.9, _pixel_box(CAMERA_XY, table_depth, (0.8, 1.2), (0.4, 0.6))
    ),
  ]
  # Black images carry no shirt color, so fit from the boxes alone.
  integrator = _integrator(detections, shirt_color=None)
  scene = integrator.update(_observation(head_depth=depth))

  [fitted] = scene.tables
  assert abs(fitted.x - 1.0) < 0.1 and abs(fitted.y - 0.5) < 0.1
  assert abs(fitted.length - 0.8) < 0.15 and abs(fitted.width - 0.6) < 0.15
  assert abs(fitted.yaw - 0.3) < 0.1
  assert scene.shirt is not None and scene.shirt_table == 0
  assert abs(scene.shirt.x - 1.0) < 0.15 and abs(scene.shirt.y - 0.5) < 0.15
  # A plain rectangle has no hem or collar, so the heading may be any of its 4 axis directions.
  assert max(scene.shirt.length, scene.shirt.width) > min(
    scene.shirt.length, scene.shirt.width
  )
  quarter_turn = math.pi / 2
  assert abs((scene.shirt.yaw + 0.15) % quarter_turn - 0.15) < 0.15


def test_manikin_faces_from_torso_toward_lap():
  torso = (0.0, 1.0, 0.3, 0.3, 0.0, 1.3)
  lap = (0.0, 1.4, 0.3, 0.5, 0.0, 0.6)
  detections = [[], [], [], []]
  detections[HEAD_0] = [_detection(HEAD_0, "manikin", 0.9, (0, 0, SIZE, SIZE))]
  camera_xy = (0.0, 1.2)
  scene = _integrator(detections, camera_xy).update(
    _observation(head_depth=_render_depth(camera_xy, [torso, lap]))
  )
  assert scene.manikin is not None
  assert abs(scene.manikin.yaw - math.pi / 2) < 0.2
  assert abs(scene.manikin.x) < 0.1
  # The box is in the facing frame: length runs from the torso to the lap's far edge.
  assert scene.manikin.length > scene.manikin.width


def test_ignores_detections_that_mostly_cover_the_gripper():
  table = (1.0, 0.5, 2.0, 1.4, 0.0, 0.57)
  depth = _render_depth(CAMERA_XY, [table])
  # Left 60% of the wrist image is the gripper at 0.2 m; a whole-image shirt box mostly covers it.
  wrist_depth = depth.copy()
  wrist_depth[:, : int(SIZE * 0.6)] = 0.2
  detections = [[], [], [], []]
  detections[WRIST_0] = [
    _detection(WRIST_0, "box", 0.9, (SIZE * 0.6, 0, SIZE, SIZE)),
    _detection(WRIST_0, "shirt", 0.9, (0, 0, SIZE, SIZE)),
  ]
  scene = _integrator(detections).update(_observation(wrist_depth=wrist_depth))
  assert scene.shirt is None
  assert len(scene.tables) == 1


def test_update_every_n_calls_reuses_last_scene():
  calls = []
  mounts = camera_geometry.CameraMounts(_look_down(CAMERA_XY), _look_down(CAMERA_XY))

  def detect(rgb_batch):
    calls.append(1)
    return [[], [], [], []]

  config = scene_integrator.GridConfig(update_every_n_calls=3)
  integrator = scene_integrator.BasicSceneIntegrator(
    detector_factory.Detector("fake", detect), mounts, config
  )
  observation = _observation()
  scenes = [integrator.update(observation) for _ in range(4)]
  assert len(calls) == 2
  assert scenes[1] is scenes[0] and scenes[2] is scenes[0]


def test_debug_detections_use_full_camera_order():
  detections = [[], [], [], []]
  detections[HEAD_0] = [_detection(HEAD_0, "box", 0.9, (0, 0, SIZE, SIZE))]
  scene = _integrator(detections).update(_observation())
  assert len(scene.debug_detections) == 5
  [detection] = scene.debug_detections[3]
  assert detections_lib.CAMERA_NAMES[detection.camera] == "robot_0_head"


def _shirt_on_table_scene(shirt_xy_range):
  """Head-camera depth of a table, and an RGB image with the shirt region painted blue."""
  table = (1.0, 0.5, 1.2, 0.8, 0.0, 0.57)
  depth = _render_depth(CAMERA_XY, [table])
  table_depth = CAMERA_HEIGHT - 0.57
  x0, y0, x1, y1 = (int(v) for v in _pixel_box(CAMERA_XY, table_depth, *shirt_xy_range))
  rgb = np.full((SIZE, SIZE, 3), 200, np.uint8)
  rgb[y0:y1, x0:x1] = (20, 30, 230)
  return depth, rgb, table_depth


def test_shirt_votes_only_on_shirt_colored_pixels():
  shirt_range = ((0.8, 1.0), (0.4, 0.6))
  depth, rgb, table_depth = _shirt_on_table_scene(shirt_range)
  detections = [[], [], [], []]
  detections[HEAD_0] = [
    _detection(HEAD_0, "box", 0.9, (0, 0, SIZE, SIZE)),
    # Loose box, three times the shirt's size.
    _detection(
      HEAD_0, "shirt", 0.9, _pixel_box(CAMERA_XY, table_depth, (0.6, 1.2), (0.2, 0.8))
    ),
  ]
  scene = _integrator(detections).update(_observation(head_depth=depth, head_rgb=rgb))
  assert scene.shirt is not None
  assert abs(scene.shirt.x - 0.9) < 0.06 and abs(scene.shirt.y - 0.5) < 0.06
  assert scene.shirt.length < 0.3 and scene.shirt.width < 0.3


def test_finds_shirt_at_table_height_without_tables():
  depth, rgb, table_depth = _shirt_on_table_scene(((0.8, 1.0), (0.4, 0.6)))
  detections = [[], [], [], []]
  detections[HEAD_0] = [
    _detection(
      HEAD_0, "shirt", 0.9, _pixel_box(CAMERA_XY, table_depth, (0.8, 1.0), (0.4, 0.6))
    ),
  ]
  scene = _integrator(detections).update(_observation(head_depth=depth, head_rgb=rgb))
  assert scene.tables == ()
  assert scene.shirt is not None and scene.shirt_table is None


def test_drops_shirt_boxes_without_shirt_color():
  depth, _, table_depth = _shirt_on_table_scene(((0.8, 1.0), (0.4, 0.6)))
  gray = np.full((SIZE, SIZE, 3), 200, np.uint8)
  detections = [[], [], [], []]
  detections[HEAD_0] = [
    _detection(HEAD_0, "box", 0.9, (0, 0, SIZE, SIZE)),
    _detection(
      HEAD_0, "shirt", 0.9, _pixel_box(CAMERA_XY, table_depth, (0.8, 1.0), (0.4, 0.6))
    ),
  ]
  scene = _integrator(detections).update(_observation(head_depth=depth, head_rgb=gray))
  assert scene.tables and scene.shirt is None


def test_shirt_heading_points_from_hem_to_collar():
  table = (1.0, 0.5, 1.4, 1.0, 0.0, 0.57)
  depth = _render_depth(CAMERA_XY, [table])
  table_depth = CAMERA_HEIGHT - 0.57
  rgb = np.full((SIZE, SIZE, 3), 200, np.uint8)

  def paint(x_range, y_range, color):
    x0, y0, x1, y1 = (
      int(v) for v in _pixel_box(CAMERA_XY, table_depth, x_range, y_range)
    )
    rgb[y0:y1, x0:x1] = color

  # Collar toward -y: sleeves widen the -y end and the hem band runs along +y.
  paint((0.75, 1.25), (0.2, 0.8), (20, 30, 230))
  paint((0.63, 1.37), (0.2, 0.4), (20, 30, 230))
  paint((0.75, 1.25), (0.74, 0.8), (230, 215, 15))
  detections = [[], [], [], []]
  detections[HEAD_0] = [
    _detection(HEAD_0, "box", 0.9, (0, 0, SIZE, SIZE)),
    _detection(
      HEAD_0, "shirt", 0.9, _pixel_box(CAMERA_XY, table_depth, (0.6, 1.4), (0.15, 0.85))
    ),
  ]
  scene = _integrator(detections).update(_observation(head_depth=depth, head_rgb=rgb))
  assert scene.shirt is not None
  assert abs(scene.shirt.yaw + math.pi / 2) < 0.2
  assert scene.shirt_heading_margin > 0.1
  # Facing the collar (-y), the left hem corner is at +x.
  assert scene.shirt_hem_seen
  left, right = scene.shirt_hem_corners
  np.testing.assert_allclose(left, (1.25, 0.77), atol=0.08)
  np.testing.assert_allclose(right, (0.75, 0.77), atol=0.08)
  assert abs(scene.shirt_height_m - 0.57) < 0.03
