"""Solves the static head/wrist camera mounts and writes them to --output.

Compares mounts across both robots and two arm configurations; fails if they differ.
The result is a calibration constant; the policy never reads sim camera matrices.

Run: ./run.sh python /project/dapper/scripts/extract_camera_mounts.py --seed 0
Then: cp output/camera_mounts.json dapper/src/vision/camera_mounts.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path("/project")
if not (PROJECT_ROOT / "src/PhyRC_Sim").is_dir():
  PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src/PhyRC_Sim"))
sys.path.insert(0, str(PROJECT_ROOT / "dapper/src"))
sys.path.insert(0, "/scripts")  # Upstream scripts dir in the container.

from vision import camera_geometry  # noqa: E402
from vision import detections as detections_lib  # noqa: E402

LIFT_COLUMN, WRIST_YAW_COLUMN, WRIST_PITCH_COLUMN = 3, 5, 6
MOVE_STEPS = 20
TOLERANCE_M = 1e-3


def _solve_mounts(observation, info):
  world_from_optical = {
    camera["id"]: np.asarray(camera["world_from_optical_column_vectors"])
    for camera in info["cameras"]
  }
  mounts = []
  for robot in range(2):
    world_from_base = camera_geometry.pose_matrix(observation["base_pose_world"][robot])
    world_from_grasp = world_from_base @ camera_geometry.pose_matrix(
      observation["grasp_pose_base"][robot]
    )
    head = world_from_optical[camera_geometry.HEAD_CAMERA_NAMES[robot]]
    wrist = world_from_optical[camera_geometry.WRIST_CAMERA_NAMES[robot]]
    mounts.append(
      (np.linalg.inv(world_from_base) @ head, np.linalg.inv(world_from_grasp) @ wrist)
    )
  return mounts


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--output", type=pathlib.Path, default="/output/camera_mounts.json"
  )
  args = parser.parse_args()

  import policy_cli  # pylint: disable=import-outside-toplevel

  policy_cli.install_failure_handler()
  from Policy import environment  # pylint: disable=import-outside-toplevel

  assert tuple(detections_lib.CAMERA_NAMES[1:]) == (
    "robot_0_wrist",
    "robot_1_wrist",
    "robot_0_head",
    "robot_1_head",
  )
  env = environment.DressingEnv(profile="actor_rgbd", max_episode_steps=MOVE_STEPS + 2)
  observation, info = env.reset(seed=args.seed)
  configs = [_solve_mounts(observation, info)]
  action = np.zeros(env.action_space.shape, np.float32)
  action[:, LIFT_COLUMN] = 0.5
  action[:, WRIST_YAW_COLUMN] = 0.5
  action[:, WRIST_PITCH_COLUMN] = -0.5
  for _ in range(MOVE_STEPS):
    observation, _, _, _, info = env.step(action)
  configs.append(_solve_mounts(observation, info))

  reference_head, reference_wrist = configs[0][0]
  worst = 0.0
  for robot_mounts in configs:
    for head, wrist in robot_mounts:
      worst = max(
        worst,
        float(np.abs(head - reference_head).max()),
        float(np.abs(wrist - reference_wrist).max()),
      )
  print(f"max mount deviation across robots/configs: {worst:.6f}")
  if worst > TOLERANCE_M:
    raise SystemExit("Mounts are not constant; the camera chain assumption is wrong.")
  args.output.write_text(
    json.dumps(
      {
        "base_from_head_optical": reference_head.tolist(),
        "grasp_from_wrist_optical": reference_wrist.tolist(),
      },
      indent=2,
    )
  )
  print(f"Wrote {args.output}")
  env.close()


if __name__ == "__main__":
  main()
