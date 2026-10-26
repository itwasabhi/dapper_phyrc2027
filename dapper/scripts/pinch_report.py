"""Summarizes down_and_pinch outcomes of solution runs from their recorded public observations.

Per run and robot: the shirt heading from scan_room's scene estimate, where the gripper was when it
first commanded a close versus that robot's hem grasp point, and whether the fingers still hold
something after the lift. Also writes a frame sheet per run (at the close and at the end).

To run:
dapper/run_py.sh /project/dapper/scripts/pinch_report.py /output/run_solution/<run> [...] \
    --log-dir-name <name>
"""

import argparse
import json
import math
import pathlib

import cv2
import numpy as np

from Policy import dataset_reader

import utils_run_dir
from behaviors import arm
from behaviors import down_and_pinch
from behaviors import motion
from behaviors import shirt_grasp
from vision import scene as scene_lib

SHEET_CAMERAS = ("overview", "robot_0_wrist", "robot_1_wrist", "robot_0_head")
LOW_DIM_KEYS = (
  "base_pose_world",
  "grasp_pose_base",
  "fingertip_origin_distance",
  "joint_position",
)


def read_scene(run_dir: pathlib.Path) -> scene_lib.SceneEstimate:
  raw = json.loads((run_dir / "debug/scan_room/scene_estimate.json").read_text())

  def to_pose(values):
    return scene_lib.Pose2D(**values) if values else None

  corners = raw.get("shirt_hem_corners")
  return scene_lib.SceneEstimate(
    tables=tuple(to_pose(table) for table in raw["tables"]),
    shirt=to_pose(raw["shirt"]),
    shirt_table=raw["shirt_table"],
    manikin=to_pose(raw["manikin"]),
    shirt_heading_margin=raw.get("shirt_heading_margin", 0.0),
    shirt_hem_corners=tuple(tuple(c) for c in corners) if corners else None,
    shirt_hem_seen=raw.get("shirt_hem_seen", False),
    shirt_height_m=raw.get("shirt_height_m"),
  )


def summarize_run(run_dir: pathlib.Path, sheet_dir: pathlib.Path) -> dict:
  scene = read_scene(run_dir)
  keys = list(LOW_DIM_KEYS) + [f"{camera}_rgb" for camera in SHEET_CAMERAS]
  dataset = dataset_reader.PolicyDataset(run_dir / "policy.hdf5", observation_keys=keys)
  low_dim = {key: [] for key in LOW_DIM_KEYS}
  actions = []
  for step in range(len(dataset)):
    sample, action = dataset[step]
    for key in LOW_DIM_KEYS:
      low_dim[key].append(sample[key])
    actions.append(action.reshape(2, 9))
  base_pose = np.stack(low_dim["base_pose_world"]).reshape(-1, 2, 7)
  grasp = np.stack(low_dim["grasp_pose_base"]).reshape(-1, 2, 7)
  finger = np.stack(low_dim["fingertip_origin_distance"]).reshape(-1, 2)
  lift = np.stack(low_dim["joint_position"]).reshape(-1, 2, 13)[:, :, arm.LIFT_JOINT]
  actions = np.stack(actions)
  closes = actions[:, :, arm.GRIPPER_COLUMN] > 0.5
  first_close = [
    int(np.argmax(closes[:, robot])) if closes[:, robot].any() else None
    for robot in range(2)
  ]
  start = min((step for step in first_close if step is not None), default=None)
  targets = (
    shirt_grasp.get_hem_grasp_targets(scene, base_pose[start])
    if start is not None
    else (None, None)
  )
  empty = down_and_pinch.Config().empty_grip_distance_m
  robots = []
  for robot in range(2):
    step = first_close[robot]
    result = {"closed": step is not None}
    if step is not None:
      yaw = motion.yaw_from_base_pose(base_pose[step, robot])
      offset = grasp[step, robot, :2]
      world = base_pose[step, robot, :2] + np.array(
        [
          math.cos(yaw) * offset[0] - math.sin(yaw) * offset[1],
          math.sin(yaw) * offset[0] + math.cos(yaw) * offset[1],
        ]
      )
      result.update(
        close_step=step,
        close_xy=world.round(4).tolist(),
        close_error_m=(
          float(np.linalg.norm(world - targets[robot]))
          if targets[robot] is not None
          else None
        ),
        final_finger_distance_m=float(finger[-1, robot]),
        lift_gain_m=float(lift[-1, robot] - lift[step, robot]),
        holding=bool(
          finger[step, robot] - finger[-1, robot] > 0.02 and finger[-1, robot] > empty
        ),
      )
    robots.append(result)
  frames = [index for index in (start, len(dataset) - 1) if index is not None]
  rows = []
  for index in frames:
    sample, _ = dataset[index]
    rows.append(np.concatenate([sample[f"{c}_rgb"] for c in SHEET_CAMERAS], axis=1))
  dataset.close()
  if rows:
    sheet_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
      str(sheet_dir / f"{run_dir.name}.png"),
      cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR),
    )
  return {
    "run": run_dir.name,
    "shirt_heading_deg": (
      math.degrees(scene.shirt.yaw) if scene.shirt is not None else None
    ),
    "heading_margin": scene.shirt_heading_margin,
    "hem_seen": scene.shirt_hem_seen,
    "robots": robots,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run_dirs", type=pathlib.Path, nargs="+")
  utils_run_dir.add_log_dir_flags(parser)
  args = parser.parse_args()
  output_dir = utils_run_dir.log_dir_from_args("pinch_report", args)
  output_dir.mkdir(parents=True, exist_ok=True)
  summaries = []
  for run_dir in args.run_dirs:
    summary = summarize_run(run_dir, output_dir / "frames")
    summaries.append(summary)
    heading = summary["shirt_heading_deg"]
    cells = []
    for robot in summary["robots"]:
      if not robot["closed"]:
        cells.append("no close")
        continue
      error = robot["close_error_m"]
      error_text = f"{error * 100:.1f}cm" if error is not None else "?"
      cells.append(
        f"err {error_text} gap {robot['final_finger_distance_m'] * 100:.1f}cm "
        f"{'HOLD' if robot['holding'] else 'miss'}"
      )
    print(
      f"{summary['run']:28s} heading "
      f"{heading if heading is None else round(heading, 1)} "
      f"margin {summary['heading_margin']:.2f} | " + " | ".join(cells)
    )
  (output_dir / "summary.json").write_text(json.dumps(summaries, indent=1))
  print(f"-> {output_dir}")


if __name__ == "__main__":
  main()
