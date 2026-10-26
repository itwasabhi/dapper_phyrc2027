"""Scores shirt detection against reviewed labels (build_shirt_labels.py) for several prompt sets.

Each --shirt-prompts value is one variant ("p1|p2" = several prompts for the shirt class); all other
classes keep DEFAULT_PROMPTS (or --prompt overrides), since classes compete in the detector.

To run:
dapper/run_py.sh /project/dapper/scripts/eval_detector.py /output/build_shirt_labels/v1 \
    --shirt-prompts "t-shirt with yellow collar" "blue t-shirt" "blue t-shirt|blue cloth"
"""

import argparse
import collections
import json
import pathlib
import time

import cv2
import numpy as np

from Policy import dataset_reader

import utils_run_dir
import utils_video
from vision import annotate
from vision import detections as detections_lib
from vision import detector_factory
from vision import shirt_labels

BATCH_SIZE = 8
LABEL_COLOR_BGR = (0, 255, 0)
# Detector-side floor; summaries apply their own thresholds above it.
MIN_SCORE = 0.1


def load_frames(labels: list[shirt_labels.FrameLabel]) -> np.ndarray:
  by_source = collections.defaultdict(list)
  for index, label in enumerate(labels):
    by_source[label.source].append(index)
  frames = [None] * len(labels)
  for source, indices in by_source.items():
    cameras = sorted({labels[index].camera for index in indices})
    dataset = dataset_reader.PolicyDataset(
      pathlib.Path(source), observation_keys=[f"{camera}_rgb" for camera in cameras]
    )
    for index in indices:
      sample, _ = dataset[labels[index].step]
      frames[index] = sample[f"{labels[index].camera}_rgb"]
    dataset.close()
  return np.stack(frames)


def get_camera_group(camera: str) -> str:
  return "wrist" if "wrist" in camera else "head"


def get_error_tiles(frames, labels, detections, threshold, iou_threshold):
  """Annotated tiles of missed shirts and of frames with a shirt detection but no shirt."""
  misses, false_positives = [], []
  for rgb, label, frame_detections in zip(frames, labels, detections):
    shirts = [
      d for d in frame_detections if d.class_name == "shirt" and d.score >= threshold
    ]
    hit = label.box_xyxy is not None and any(
      shirt_labels.get_box_iou(d.box_xyxy, label.box_xyxy) >= iou_threshold
      for d in shirts
    )
    if label.box_xyxy is not None and not hit:
      target = misses
    elif label.box_xyxy is None and shirts:
      target = false_positives
    else:
      continue
    tile = annotate.annotate_frame(
      rgb, [d for d in frame_detections if d.score >= threshold]
    )
    if label.box_xyxy is not None:
      x0, y0, x1, y1 = (int(round(value)) for value in label.box_xyxy)
      cv2.rectangle(tile, (x0, y0), (x1 - 1, y1 - 1), LABEL_COLOR_BGR, 1)
    target.append(utils_video.label(tile, f"{label.id} {label.camera}"))
  return misses, false_positives


def evaluate_variant(detector, frames, labels, score_thresholds, iou_threshold):
  results, latencies, all_detections = [], [], []
  for start in range(0, len(frames), BATCH_SIZE):
    batch = frames[start : start + BATCH_SIZE]
    begin = time.perf_counter()
    detections = detector.detect(batch)
    latencies.append((time.perf_counter() - begin) / len(batch))
    all_detections.extend(detections)
    for label, frame_detections in zip(labels[start : start + BATCH_SIZE], detections):
      results.append(
        shirt_labels.FrameDetections(
          label_box=label.box_xyxy,
          scored_boxes=tuple(
            (detection.score, detection.box_xyxy)
            for detection in frame_detections
            if detection.class_name == "shirt"
          ),
        )
      )
  summary = {"latency_ms_per_image": 1e3 * float(np.median(latencies))}
  groups = {"all": range(len(labels))}
  for index, label in enumerate(labels):
    groups.setdefault(get_camera_group(label.camera), []).append(index)
  for group, indices in groups.items():
    for threshold in score_thresholds:
      summary[f"{group}@{threshold}"] = shirt_labels.summarize(
        [results[index] for index in indices], threshold, iou_threshold
      )
  return summary, all_detections


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("label_dir", type=pathlib.Path)
  utils_run_dir.add_log_dir_flags(parser)
  parser.add_argument(
    "--backend", choices=sorted(detector_factory.BACKENDS), default="omdet"
  )
  parser.add_argument(
    "--shirt-prompts",
    nargs="+",
    default=[detections_lib.DEFAULT_PROMPTS["shirt"][0]],
    help='One variant per value; "p1|p2" gives the shirt class several prompts.',
  )
  parser.add_argument(
    "--prompt",
    action="append",
    default=[],
    help='"class=prompt1|prompt2" overrides for the other classes.',
  )
  parser.add_argument(
    "--score-thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4]
  )
  parser.add_argument("--iou-threshold", type=float, default=0.5)
  parser.add_argument(
    "--dump-errors",
    action="store_true",
    help="Write miss/false-positive contact sheets per variant at the first score threshold.",
  )
  parser.add_argument(
    "--verified-only", action="store_true", help='Only use frames reviewed "ok".'
  )
  args = parser.parse_args()

  labels = [
    label
    for label in shirt_labels.read_labels(args.label_dir)
    if label.review != "reject" and (label.review == "ok" or not args.verified_only)
  ]
  frames = load_frames(labels)
  output_dir = utils_run_dir.log_dir_from_args("eval_detector", args)
  output_dir.mkdir(parents=True, exist_ok=True)
  base_prompts = detections_lib.parse_prompt_specs(args.prompt)
  report = {"label_dir": str(args.label_dir), "backend": args.backend, "variants": {}}
  for shirt_spec in args.shirt_prompts:
    prompts = {**base_prompts, "shirt": tuple(shirt_spec.split("|"))}
    detector = detector_factory.BACKENDS[args.backend](
      prompts, score_threshold=MIN_SCORE, class_thresholds={}
    )
    summary, detections = evaluate_variant(
      detector, frames, labels, args.score_thresholds, args.iou_threshold
    )
    if args.dump_errors:
      misses, false_positives = get_error_tiles(
        frames, labels, detections, args.score_thresholds[0], args.iou_threshold
      )
      error_dir = output_dir / "errors" / shirt_spec.replace(" ", "_").replace("|", "+")
      utils_video.write_contact_sheets(error_dir, "miss", misses)
      utils_video.write_contact_sheets(error_dir, "false_positive", false_positives)
    report["variants"][shirt_spec] = {"prompts": prompts, **summary}
    del detector
    for key, metrics in summary.items():
      if isinstance(metrics, dict):
        print(
          f"{shirt_spec[:40]:40s} {key:10s} R={metrics['recall']:.2f} "
          f"P={metrics['precision']:.2f} negFP={metrics['negative_frame_fp_rate']:.2f} "
          f"AP={metrics['average_precision']:.2f} (pos {metrics['positives']}/{metrics['frames']})"
        )
  (output_dir / "report.json").write_text(json.dumps(report, indent=1))
  print(f"-> {output_dir / 'report.json'}")


if __name__ == "__main__":
  main()
