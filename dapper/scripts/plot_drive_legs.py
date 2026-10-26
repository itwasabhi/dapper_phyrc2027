"""Plots each drive leg of a solution-test3 run over the scanned obstacles.

Run: dapper/run_py.sh /project/dapper/scripts/plot_drive_legs.py <run>/debug <out.png>
Reads <debug>/drive_legs.jsonl and <debug>/scan_room/scene_estimate.json. The dashed line is the
planner's initial path from the leg's start pose; the solid line is the path actually driven.
"""

import argparse
import json
import math
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import patches  # pylint: disable=wrong-import-position
from matplotlib import pyplot as plt  # pylint: disable=wrong-import-position

from behaviors import collision  # pylint: disable=wrong-import-position
from behaviors import motion  # pylint: disable=wrong-import-position
from behaviors import path_planning  # pylint: disable=wrong-import-position
from vision import scene as scene_lib  # pylint: disable=wrong-import-position

ROBOT_COLORS = ("#0072B2", "#D55E00")
ARROW_M = 0.5
NUM_COLUMNS = 3


def _load_obstacles(scene_path: pathlib.Path) -> list[scene_lib.Pose2D]:
  scene = json.loads(scene_path.read_text())
  rects = [scene_lib.Pose2D(**table) for table in scene["tables"]]
  if scene["manikin"]:
    rects.append(scene_lib.Pose2D(**scene["manikin"]))
  return rects


def _get_planned_paths(leg, start_poses, rects, margin) -> list[np.ndarray]:
  """Per-robot polylines (N, 2) of the planner's initial path from `start_poses`."""
  starts = [np.array(pose[:2]) for pose in start_poses]
  if leg["name"].endswith("formation"):
    positions = np.array(starts)
    yaws = [pose[2] for pose in start_poses]
    frame = motion.formation_pose(positions[0], positions[1], yaws[0])
    slots = [
      motion.relative(frame, positions[robot], yaws[robot]) for robot in range(2)
    ]
    goal = scene_lib.Pose2D(*leg["goals"][0][:3], 0.0, 0.0, 1.0)
    path = path_planning.plan_formation_path(frame, goal, slots, rects, margin)
    if path is None:
      return [starts[robot][None] for robot in range(2)]
    return [
      np.array(
        [starts[robot]] + [motion.compose(pose, *slots[robot])[0] for pose in path]
      )
      for robot in range(2)
    ]
  paths = []
  for robot in range(2):
    goal = leg["goals"][robot]
    path = (
      None
      if goal is None
      else path_planning.plan_point_path(
        starts[robot], np.array(goal[:2]), rects, margin
      )
    )
    paths.append(
      starts[robot][None] if path is None else np.array([starts[robot], *path])
    )
  return paths


def _draw_pose(axis, pose, color, marker):
  axis.plot(pose[0], pose[1], marker, color=color, markersize=8)
  axis.arrow(
    pose[0],
    pose[1],
    ARROW_M * math.cos(pose[2]),
    ARROW_M * math.sin(pose[2]),
    color=color,
    head_width=0.1,
    length_includes_head=True,
  )


def _draw_rect(axis, rect, margin, filled):
  for inflate, style in ((0.0, "-"), (2 * margin, "--")):
    inflated = scene_lib.Pose2D(
      rect.x, rect.y, rect.yaw, rect.length + inflate, rect.width + inflate, 1.0
    )
    corners = scene_lib.get_corners(inflated)
    axis.add_patch(
      patches.Polygon(
        corners,
        closed=True,
        linestyle=style,
        facecolor=filled if inflate == 0.0 else "none",
        edgecolor="#555555",
        linewidth=1,
        zorder=1,
      )
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("debug_dir", type=pathlib.Path)
  parser.add_argument("out_path", type=pathlib.Path)
  args = parser.parse_args()
  legs = [
    json.loads(line)
    for line in (args.debug_dir / "drive_legs.jsonl").read_text().splitlines()
  ]
  rects = _load_obstacles(args.debug_dir / "scan_room" / "scene_estimate.json")
  margin = collision.CollisionConfig().obstacle_clearance_m

  points = np.array(
    [corner for rect in rects for corner in scene_lib.get_corners(rect)]
    + [pose[:2] for leg in legs for step in leg["trajectory"] for pose in step]
  )
  low, high = points.min(axis=0) - 0.5, points.max(axis=0) + 0.5

  num_rows = math.ceil(len(legs) / NUM_COLUMNS)
  figure, axes = plt.subplots(
    num_rows, NUM_COLUMNS, figsize=(5 * NUM_COLUMNS, 4.2 * num_rows), squeeze=False
  )
  for axis, leg in zip(axes.flat, legs):
    for index, rect in enumerate(rects):
      _draw_rect(axis, rect, margin, "#BBBBBB" if index < len(rects) - 1 else "#888888")
    trajectory = np.array(leg["trajectory"])  # (steps, robot, xyyaw)
    planned = _get_planned_paths(leg, trajectory[0], rects, margin)
    for robot, color in enumerate(ROBOT_COLORS):
      axis.plot(*planned[robot].T, "--", color=color, linewidth=1.5, zorder=2)
      axis.plot(*planned[robot][1:-1].T, ".", color=color, markersize=6, zorder=2)
      axis.plot(
        trajectory[:, robot, 0],
        trajectory[:, robot, 1],
        "-",
        color=color,
        linewidth=2,
        alpha=0.8,
        zorder=3,
      )
      _draw_pose(axis, trajectory[0, robot], color, "o")
      _draw_pose(axis, leg["final_poses"][robot], color, "x")
      if not leg["name"].endswith("formation") and leg["goals"][robot] is not None:
        _draw_pose(axis, leg["goals"][robot], color, "*")
    if leg["name"].endswith("formation"):
      axis.plot(*leg["goals"][0][:2], "k+", markersize=12)
    detours = max(len(path) - 2 for path in planned)
    axis.set_title(
      f"{leg['name']}: {'reached' if leg['reached'] else 'NOT reached'}, "
      f"{leg['steps']} steps, {max(detours, 0)} detour waypoints",
      fontsize=10,
    )
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
  for axis in list(axes.flat)[len(legs) :]:
    axis.axis("off")
  figure.text(
    0.5,
    0.005,
    "o start   x final   * goal   + pair-frame goal   dashed = initial plan (dots = waypoints)   "
    "solid = driven   gray = tables/manikin, dashed outline = clearance   blue robot 0, orange robot 1",
    ha="center",
    fontsize=8,
  )
  figure.tight_layout(rect=(0, 0.02, 1, 1))
  figure.savefig(args.out_path, dpi=110)


if __name__ == "__main__":
  main()
