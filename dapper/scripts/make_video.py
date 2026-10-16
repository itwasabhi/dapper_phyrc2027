"""Renders RGB/depth camera videos from a policy.hdf5.

Run: dapper/run_py.sh /project/dapper/scripts/make_video.py \
    /output/random_rollout/run0/policy.hdf5 --output-dir /output/videos/run0
"""

import argparse
import pathlib

import utils_video


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("hdf5", type=pathlib.Path)
  parser.add_argument("--output-dir", type=pathlib.Path, required=True)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--demo", action="append", help="e.g. demo_0; repeatable.")
  parser.add_argument("--no-depth", action="store_true")
  args = parser.parse_args()
  paths = utils_video.write_videos(
    args.hdf5,
    args.output_dir,
    stride=args.stride,
    demos=args.demo,
    include_depth=not args.no_depth,
  )
  for path in paths:
    print(path)


if __name__ == "__main__":
  main()
