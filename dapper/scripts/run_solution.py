"""Rolls out a solution dir (policy.py layout) in DressingEnv and records it.

Run: ./run.sh python /project/dapper/scripts/run_solution.py \
    --solution-dir /project/dapper/src/solution-alpha \
    --output-dir /output/solution_alpha/run0 --write-video
"""

import argparse
import os
import pathlib
import sys

import utils_random_rollout
import utils_rollout_recorder
import utils_solution
import utils_video

PROJECT_ROOT = pathlib.Path("/project")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--solution-dir",
    type=pathlib.Path,
    default=PROJECT_ROOT / "dapper/src/solution-alpha",
    help="Directory containing policy.py and optionally checkpoint.*.",
  )
  parser.add_argument("--checkpoint", type=pathlib.Path)
  parser.add_argument(
    "--output-dir",
    type=pathlib.Path,
    required=True,
    help="Where to write the trajectory; must be under /output in docker.",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Tell the policy to write debug files (e.g. annotated frames) to <output-dir>/debug.",
  )
  parser.add_argument("--overwrite", action="store_true")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--steps", type=int, default=100, help="20 Hz steps.")
  parser.add_argument("--resolution", type=int, nargs=2, metavar=("W", "H"))
  utils_rollout_recorder.add_recording_flags(parser)
  args = parser.parse_args()
  if args.steps < 1 or args.video_stride < 1:
    parser.error("--steps and --video-stride must be >= 1")
  if not (args.solution_dir / "policy.py").is_file():
    parser.error(f"{args.solution_dir}/policy.py not found")
  return args


def main() -> None:
  args = parse_args()
  utils_random_rollout.prepare_output_dir(args.output_dir, args.overwrite)
  checkpoint = args.checkpoint or utils_solution.find_checkpoint(args.solution_dir)
  if args.debug:
    os.environ["DAPPER_DEBUG_DIR"] = str(args.output_dir / "debug")

  import policy_cli  # pylint: disable=import-outside-toplevel

  policy_cli.install_failure_handler()
  from Policy import environment  # pylint: disable=import-outside-toplevel

  env = environment.DressingEnv(
    profile="actor_rgbd",
    resolution=tuple(args.resolution) if args.resolution else None,
    max_episode_steps=args.steps,
  )
  # Loaded after the Isaac app starts, as in the official evaluator.
  sys.path.insert(0, str(args.solution_dir))
  policy = utils_solution.load_policy(
    args.solution_dir, checkpoint, env.observation_space, env.action_space
  )

  observation, info = env.reset(seed=args.seed)
  if hasattr(policy, "reset"):
    policy.reset()
  recorder = utils_rollout_recorder.RolloutRecorder(
    utils_rollout_recorder.open_writer(
      PROJECT_ROOT,
      args.output_dir,
      {
        "args": vars(args),
        "source_commit": info["source_commit"],
        "source_is_dirty": info["source_is_dirty"],
      },
    )
  )
  recorder.start_episode(observation, info)
  for _ in range(args.steps):
    action = utils_solution.validate_action(policy(observation), env.action_space.shape)
    observation, _, terminated, truncated, step_info = env.step(action)
    recorder.record_step(action, observation)
    if terminated or truncated:
      recorder.end_episode(step_info["termination_reason"])
      break

  # Save before close(): it can fast-exit the process.
  if hasattr(policy, "close"):
    policy.close()
  recorder.close()
  print(f"Wrote {recorder.num_steps} steps to {args.output_dir}")
  if args.write_video:
    for path in utils_video.write_videos(
      args.output_dir / "policy.hdf5",
      args.output_dir / "videos",
      stride=args.video_stride,
    ):
      print(f"Wrote {path}")
  env.close()


if __name__ == "__main__":
  main()
