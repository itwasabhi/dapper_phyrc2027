"""Runs a detector over a recorded policy.hdf5 and writes annotated debug outputs.

To run: 
dapper/run_py.sh /project/dapper/scripts/perception_debug.py \
    /output/solution_alpha/run0/policy.hdf5 --output-dir /output/perception/omdet \
    --backend omdet --stride 4 [--prompt "box=white box|cardboard box" ...]
"""

import argparse
import pathlib
import time

import numpy as np

from Policy import dataset_reader

import utils_video
from vision import debug_writer
from vision import detector_factory
from vision import detections as detections_lib


def _parse_prompts(specs: list[str] | None):
  if not specs:
    return detections_lib.DEFAULT_PROMPTS
  prompts = {}
  for spec in specs:
    class_name, _, joined = spec.partition("=")
    prompts[class_name] = tuple(joined.split("|"))
  return prompts


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("hdf5_path", type=pathlib.Path)
  parser.add_argument("--output-dir", type=pathlib.Path, required=True)
  parser.add_argument(
    "--backend", choices=sorted(detector_factory.BACKENDS), default="omdet"
  )
  parser.add_argument("--stride", type=int, default=4, help="Use every Nth 20 Hz step.")
  parser.add_argument(
    "--prompt",
    action="append",
    help='"class=prompt1|prompt2"; repeat per class. Default: detections.DEFAULT_PROMPTS.',
  )
  args = parser.parse_args()

  detector = detector_factory.BACKENDS[args.backend](_parse_prompts(args.prompt))
  keys = [f"{name}_rgb" for name in detections_lib.CAMERA_NAMES]
  dataset = dataset_reader.PolicyDataset(args.hdf5_path, observation_keys=keys)
  writer = debug_writer.DebugWriter(
    args.output_dir, utils_video.POLICY_HZ / args.stride
  )
  latencies = []
  for index in range(0, len(dataset), args.stride):
    observation, _ = dataset[index]
    rgb_batch = np.stack([observation[key] for key in keys])
    start = time.perf_counter()
    detections = detector.detect(rgb_batch)
    latencies.append(time.perf_counter() - start)
    writer.write(index, rgb_batch, detections)
  writer.close()
  dataset.close()
  print(
    f"{args.backend}: {len(latencies)} frames, median latency "
    f"{1e3 * np.median(latencies):.1f} ms -> {args.output_dir}"
  )


if __name__ == "__main__":
  main()
