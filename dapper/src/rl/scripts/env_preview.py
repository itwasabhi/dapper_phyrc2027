"""Resets ShirtGraspEnv several times and saves camera tiles to check robot/shirt placement.

Run: dapper/run_sim.sh python /project/dapper/src/rl/scripts/env_preview.py --output-dir /output/rl/env_preview
"""

import argparse
import pathlib

import numpy as np
from PIL import Image


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=pathlib.Path, required=True)
  parser.add_argument("--resets", type=int, default=6)
  parser.add_argument("--steps", type=int, default=3)
  parser.add_argument("--resolution", type=int, default=256)
  parser.add_argument("--table-index", type=int, default=None)
  parser.add_argument("--pairing", default="random_pair")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  import policy_cli  # pylint: disable=import-outside-toplevel

  policy_cli.install_failure_handler()
  from rl.envs import shirt_grasp_env  # pylint: disable=import-outside-toplevel
  from rl.envs import spawn  # pylint: disable=import-outside-toplevel

  env = shirt_grasp_env.ShirtGraspEnv(
    resolution=(args.resolution, args.resolution),
    table_index=args.table_index,
    shirt_xy_jitter_m=0.05,
    spawn_config=spawn.SpawnConfig(pairing=args.pairing),
  )
  for reset_index in range(args.resets):
    observation, info = env.reset(seed=args.seed + reset_index)
    print(f"reset {reset_index}: {info}", flush=True)
    for robot_index, robot_spawn in enumerate(env.robot_spawns):
      print(
        f"  robot {robot_index}: corner {robot_spawn.corner_index} "
        f"position {np.round(robot_spawn.position, 3).tolist()} yaw {robot_spawn.yaw_deg:.1f} "
        f"measured base pose {np.round(observation['base_pose_world'][robot_index], 3).tolist()}",
        flush=True,
      )
    for _ in range(args.steps):
      observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample() * 0.0
      )
    tile = np.concatenate(list(observation["rgb"]), axis=1)
    Image.fromarray(tile).save(args.output_dir / f"reset_{reset_index:02d}.png")
  print(f"Wrote tiles to {args.output_dir}", flush=True)
  env.close()
  shirt_grasp_env.close_simulation_app()


if __name__ == "__main__":
  main()
