"""Rolls out random actions in DressingEnv and records it as policy.hdf5 (+ videos).

Run: ./run.sh python /project/dapper/scripts/random_rollout.py \
    --output-dir /output/random_rollout/run0 --write-video
"""

import argparse
import pathlib

import numpy as np

import utils_random_rollout
import utils_rollout_recorder
import utils_video

PROJECT_ROOT = pathlib.Path("/project")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir",
    type=pathlib.Path,
    required=True,
    help="Where to write the trajectory; must be under /output in docker.",
  )
  parser.add_argument("--overwrite", action="store_true")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--steps", type=int, default=100, help="20 Hz steps.")
  parser.add_argument("--resolution", type=int, nargs=2, metavar=("W", "H"))
  parser.add_argument("--action-scale", type=float, default=1.0)
  parser.add_argument(
    "--hold-steps",
    type=int,
    default=10,
    help="Resample velocity commands every this many steps.",
  )
  parser.add_argument("--gripper-resample-prob", type=float, default=0.05)
  utils_rollout_recorder.add_recording_flags(parser)
  args = parser.parse_args()
  if args.steps < 1 or args.hold_steps < 1 or args.video_stride < 1:
    parser.error("--steps, --hold-steps and --video-stride must be >= 1")
  if not 0.0 <= args.action_scale <= 1.0:
    parser.error("--action-scale must be in [0, 1]")
  return args


def main() -> None:
  args = parse_args()
  utils_random_rollout.prepare_output_dir(args.output_dir, args.overwrite)

  import policy_cli  # pylint: disable=import-outside-toplevel

  policy_cli.install_failure_handler()
  # Importing Policy.environment before construction is fine; the Isaac app
  # starts inside the DressingEnv constructor.
  from Policy import environment  # pylint: disable=import-outside-toplevel

  env = environment.DressingEnv(
    profile="actor_rgbd",
    resolution=tuple(args.resolution) if args.resolution else None,
    max_episode_steps=args.steps,
  )
  rng = np.random.default_rng(args.seed)

  observation, info = env.reset(seed=args.seed)
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
  recorder.start_episode(
    observation, utils_rollout_recorder.dressing_episode_metadata(info)
  )
  action = None
  for step in range(args.steps):
    action = utils_random_rollout.sample_action(
      rng,
      action,
      step,
      args.hold_steps,
      args.action_scale,
      args.gripper_resample_prob,
    )
    observation, _, terminated, truncated, step_info = env.step(action)
    recorder.record_step(action, observation)
    if terminated or truncated:
      recorder.end_episode(step_info["termination_reason"])
      break

  # Save before close(): it can fast-exit the process.
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
