"""Replays the scan_room span of recorded rollouts through the scene integrator, without the sim.

With --labels (build_shirt_labels.py output), the shirt pixels labeled on those frames are lifted
with depth into a reference footprint and the estimate is scored against it. Each recording also
gets reference_topdown.png (lifted body/trim pixels, +x right, +y up) for annotating the shirt's
hem-to-collar heading in <labels>/headings.json, which is then scored too.

--rotate-world-deg turns the recorded world about z (robots, cameras and so the shirt), to test
other shirt headings on real images; widen the grid with --grid x_range_m=-5,5 --grid y_range_m=-5,5.

To run:
dapper/run_py.sh /project/dapper/scripts/replay_scene.py /output/run_solution/scan-seed*/policy.hdf5 \
    --labels /output/build_shirt_labels/v1 [--prompt "shirt=blue t-shirt"]
"""

import argparse
import collections
import dataclasses
import json
import math
import pathlib

import cv2
import numpy as np

import utils_policy_replay
import utils_run_dir
from behaviors import base
from behaviors import scan_room
from vision import camera_geometry
from vision import color_mask
from vision import detections as detections_lib
from vision import detector_factory
from vision import scene as scene_lib
from vision import scene_integrator
from vision import shirt_labels

ROBOT_CAMERAS = detections_lib.CAMERA_NAMES[1:]
MIN_SHIRT_HEIGHT_M = 0.1
TOPDOWN_CELL_M = 0.01
ESTIMATE_ARROW_BGR = (0, 0, 255)
LABEL_ARROW_BGR = (0, 255, 0)


def get_grid_config(overrides: list[str]) -> scene_integrator.GridConfig:
  """Applies "field=value" overrides to the scan_room GridConfig; tuples are comma-separated."""
  config = scene_integrator.GridConfig(
    update_every_n_calls=scan_room.UPDATE_EVERY_N_STEPS
  )
  fields = {field.name: field for field in dataclasses.fields(config)}
  changes = {}
  for override in overrides:
    name, _, value = override.partition("=")
    if name not in fields:
      raise ValueError(f"Unknown GridConfig field {name!r}")
    current = getattr(config, name)
    if isinstance(current, tuple):
      changes[name] = tuple(float(part) for part in value.split(","))
    else:
      changes[name] = type(current)(value)
  return dataclasses.replace(config, **changes)


def get_labels_by_step(label_dir: pathlib.Path | None, source: pathlib.Path):
  if label_dir is None:
    return {}
  by_step = collections.defaultdict(list)
  for label in shirt_labels.read_labels(label_dir):
    if (
      label.source == str(source)
      and label.review != "reject"
      and label.box_xyxy is not None
      and label.camera in ROBOT_CAMERAS
    ):
      by_step[label.step].append(label)
  return by_step


def lift_labels(label_dir, labels, observation, mounts):
  """World points (N, 3) of the labeled shirt pixels and whether each is trim-colored (N,)."""
  world_from_optical = camera_geometry.camera_world_poses(
    observation["base_pose_world"], observation["grasp_pose_base"], mounts
  )
  points, is_trim = [np.zeros((0, 3))], [np.zeros(0, bool)]
  for label in labels:
    camera = detections_lib.CAMERA_NAMES.index(label.camera)
    depth = observation["depth"][camera][..., 0]
    mask = shirt_labels.read_mask(label_dir, label.id, depth.shape)
    valid = observation["depth_valid"][camera][..., 0] & mask
    world, pixel_index = camera_geometry.get_world_points(
      depth, valid, world_from_optical[label.camera]
    )
    _, trim = color_mask.get_shirt_parts(observation["rgb"][camera])
    keep = world[:, 2] > MIN_SHIRT_HEIGHT_M
    points.append(world[keep])
    is_trim.append(trim.reshape(-1)[pixel_index][keep])
  return np.concatenate(points), np.concatenate(is_trim)


def render_reference_topdown(points, is_trim, center, headings_deg):
  """1 cm top-down image of lifted shirt pixels with an arrow per (BGR color, heading) pair."""
  half_cells = 60
  cells = np.floor((points[:, :2] - center) / TOPDOWN_CELL_M).astype(int) + half_cells
  keep = ((cells >= 0) & (cells < 2 * half_cells)).all(axis=1)
  image = np.full((2 * half_cells, 2 * half_cells, 3), 60, np.uint8)
  for trim_value, color in ((False, (255, 80, 0)), (True, (0, 230, 255))):
    selected = cells[keep & (is_trim == trim_value)]
    image[selected[:, 1], selected[:, 0]] = color
  image = np.ascontiguousarray(image[::-1])
  image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
  middle = image.shape[0] // 2
  for color, heading_deg in headings_deg:
    heading = math.radians(heading_deg)
    tip = (
      int(middle + 0.8 * middle * math.cos(heading)),
      int(middle - 0.8 * middle * math.sin(heading)),
    )
    cv2.arrowedLine(image, (middle, middle), tip, color, 2)
  return image


def score(estimate, reference, expected_heading_deg):
  metrics = {"tables": len(estimate.tables), "shirt_found": estimate.shirt is not None}
  shirt = estimate.shirt
  if shirt is None:
    return metrics
  metrics["heading_margin"] = estimate.shirt_heading_margin
  if expected_heading_deg is not None:
    metrics["heading_error_deg"] = abs(
      math.degrees(scene_lib.wrap_angle(shirt.yaw - math.radians(expected_heading_deg)))
    )
  if reference is None:
    return metrics
  return {
    **metrics,
    "center_error_m": math.hypot(shirt.x - reference.x, shirt.y - reference.y),
    "footprint_iou": shirt_labels.get_footprint_iou(shirt, reference),
    "area_ratio": shirt.length * shirt.width / (reference.length * reference.width),
  }


def replay(path, args, integrator, mounts, headings, output_dir) -> dict:
  integrator.reset()
  labels_by_step = get_labels_by_step(args.labels, path)
  rotation = math.radians(args.rotate_world_deg)
  shirt_points, shirt_is_trim = [], []
  estimate = scene_lib.SceneEstimate()
  for step, observation in utils_policy_replay.iterate_observations(
    path, stop=args.steps
  ):
    if rotation:
      observation = utils_policy_replay.rotate_about_z(observation, rotation)
    estimate = integrator.update(observation)
    if step in labels_by_step:
      points, is_trim = lift_labels(
        args.labels, labels_by_step[step], observation, mounts
      )
      shirt_points.append(points)
      shirt_is_trim.append(is_trim)
  points = np.concatenate(shirt_points) if shirt_points else np.zeros((0, 3))
  reference = shirt_labels.get_reference_footprint(points)
  expected_heading_deg = (
    headings[str(path)] + args.rotate_world_deg if str(path) in headings else None
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  if estimate.debug_image is not None:
    cv2.imwrite(str(output_dir / "top_down.png"), estimate.debug_image[..., ::-1])
  if reference is not None:
    arrows = []
    if estimate.shirt is not None:
      arrows.append((ESTIMATE_ARROW_BGR, math.degrees(estimate.shirt.yaw)))
    if expected_heading_deg is not None:
      arrows.append((LABEL_ARROW_BGR, expected_heading_deg))
    cv2.imwrite(
      str(output_dir / "reference_topdown.png"),
      render_reference_topdown(
        points,
        np.concatenate(shirt_is_trim),
        np.array([reference.x, reference.y]),
        arrows,
      ),
    )
  result = {
    "source": str(path),
    "estimate": scene_lib.to_jsonable(estimate),
    "reference_shirt": dataclasses.asdict(reference) if reference else None,
    "expected_heading_deg": expected_heading_deg,
    "metrics": score(estimate, reference, expected_heading_deg),
  }
  (output_dir / "scene_estimate.json").write_text(json.dumps(result, indent=1))
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("hdf5_paths", type=pathlib.Path, nargs="+")
  utils_run_dir.add_log_dir_flags(parser)
  parser.add_argument("--labels", type=pathlib.Path)
  parser.add_argument(
    "--backend", choices=sorted(detector_factory.BACKENDS), default="omdet"
  )
  parser.add_argument(
    "--prompt",
    action="append",
    default=[],
    help='"class=prompt1|prompt2"; repeat per class. Other classes keep DEFAULT_PROMPTS.',
  )
  parser.add_argument(
    "--class-threshold",
    action="append",
    default=[],
    help='"class=score"; overrides DEFAULT_CLASS_THRESHOLDS for one class.',
  )
  parser.add_argument(
    "--grid",
    action="append",
    default=[],
    help='"field=value" GridConfig override, e.g. --grid shirt_belief=0.2.',
  )
  parser.add_argument(
    "--rotate-world-deg",
    type=float,
    default=0.0,
    help="Rotate the recorded world about z through the origin before integrating.",
  )
  parser.add_argument(
    "--no-shirt-color",
    action="store_true",
    help="Vote on whole shirt boxes instead of their shirt-colored pixels.",
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=scan_room.get_total_steps(base.BehaviorCfg()),
    help="Replay steps [0, steps); defaults to the scan_room span.",
  )
  args = parser.parse_args()

  output_dir = utils_run_dir.log_dir_from_args("replay_scene", args)
  mounts = camera_geometry.load_mounts()
  class_thresholds = dict(detections_lib.DEFAULT_CLASS_THRESHOLDS)
  for spec in args.class_threshold:
    class_name, _, value = spec.partition("=")
    class_thresholds[class_name] = float(value)
  integrator = scene_integrator.BasicSceneIntegrator(
    detector=detector_factory.BACKENDS[args.backend](
      detections_lib.parse_prompt_specs(args.prompt),
      class_thresholds=class_thresholds,
    ),
    mounts=mounts,
    config=get_grid_config(args.grid),
    shirt_color=None if args.no_shirt_color else color_mask.ShirtColorConfig(),
  )
  headings = shirt_labels.read_headings(args.labels) if args.labels else {}
  results = [
    replay(path, args, integrator, mounts, headings, output_dir / path.parent.name)
    for path in args.hdf5_paths
  ]
  for result in results:
    metrics = result["metrics"]
    print(
      f"{pathlib.Path(result['source']).parent.name:24s} "
      + " ".join(
        f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in metrics.items()
      )
    )
  heading_errors = [
    result["metrics"]["heading_error_deg"]
    for result in results
    if "heading_error_deg" in result["metrics"]
  ]
  found = [
    result["metrics"] for result in results if "center_error_m" in result["metrics"]
  ]
  summary = {
    "args": {key: str(value) for key, value in vars(args).items()},
    "shirt_found_rate": sum(r["metrics"]["shirt_found"] for r in results)
    / len(results),
    "median_center_error_m": (
      float(np.median([m["center_error_m"] for m in found])) if found else None
    ),
    "median_footprint_iou": (
      float(np.median([m["footprint_iou"] for m in found])) if found else None
    ),
    "median_heading_error_deg": (
      float(np.median(heading_errors)) if heading_errors else None
    ),
    "max_heading_error_deg": max(heading_errors) if heading_errors else None,
    "results": results,
  }
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
  print(
    f"found {summary['shirt_found_rate']:.2f}, median center error "
    f"{summary['median_center_error_m']}, median IoU {summary['median_footprint_iou']}, "
    f"heading error median {summary['median_heading_error_deg']} "
    f"max {summary['max_heading_error_deg']} "
    f"-> {output_dir}"
  )


if __name__ == "__main__":
  main()
