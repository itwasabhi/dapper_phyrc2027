"""Renders RGB/depth camera videos from a policy.hdf5.

Run: dapper/run_py.sh /project/dapper/scripts/make_video.py \
    /output/random_rollout/<uid>/policy.hdf5
"""

import argparse
import pathlib

import utils_run_dir
import utils_video


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("hdf5", type=pathlib.Path)
  utils_run_dir.add_log_dir_flags(parser)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--demo", action="append", help="e.g. demo_0; repeatable.")
  parser.add_argument("--no-depth", action="store_true")
  args = parser.parse_args()
  output_dir = utils_run_dir.log_dir_from_args("make_video", args)
  paths = utils_video.write_videos(
    args.hdf5,
    output_dir,
    stride=args.stride,
    demos=args.demo,
    include_depth=not args.no_depth,
  )
  for path in paths:
    print(path)


if __name__ == "__main__":
  main()
