"""Renders videos for an entry of tracked_data.json (see make_video.py).

Run: dapper/run_py.sh /project/dapper/scripts/make_tracked_video.py <name>
"""

import argparse
import json
import pathlib

import utils_run_dir
import utils_video

_TRACKED_DATA = pathlib.Path(__file__).resolve().parent.parent / "tracked_data.json"


def main() -> None:
  entries = json.loads(_TRACKED_DATA.read_text())
  names = [entry["name"] for entry in entries]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("name", choices=names)
  utils_run_dir.add_log_dir_flags(parser)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--demo", action="append", help="e.g. demo_0; repeatable.")
  parser.add_argument("--no-depth", action="store_true")
  args = parser.parse_args()
  entry = entries[names.index(args.name)]
  output_dir = utils_run_dir.log_dir_from_args("make_tracked_video", args)
  paths = utils_video.write_videos(
    pathlib.Path(entry["path"]) / "policy.hdf5",
    output_dir,
    stride=args.stride,
    demos=args.demo,
    include_depth=not args.no_depth,
  )
  for path in paths:
    print(path)


if __name__ == "__main__":
  main()
