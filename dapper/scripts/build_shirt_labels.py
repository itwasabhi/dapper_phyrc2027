"""Proposes shirt box/mask labels on recorded camera frames from the color prior, for human review.

Writes <run dir>/labels.json, masks.npz and review/*.png contact sheets (tile titles are label ids).
Review by editing each frame's "review" field in labels.json to "ok" or "reject"; fix a wrong box
by editing "box_xyxy" (or set it to null when no shirt is visible) and marking it "ok".

To run:
dapper/run_py.sh /project/dapper/scripts/build_shirt_labels.py \
    /output/run_solution/scan-seed*/policy.hdf5 --stride 8 --log-dir-name v1
"""

import argparse
import pathlib

import cv2
import numpy as np

from Policy import dataset_reader

import utils_run_dir
import utils_video
from vision import color_mask
from vision import detections as detections_lib
from vision import shirt_labels

ROBOT_CAMERAS = detections_lib.CAMERA_NAMES[1:]
MASK_COLOR_BGR = (0, 0, 255)
BOX_COLOR_BGR = (0, 255, 0)


def get_review_tile(rgb: np.ndarray, mask: np.ndarray, label: shirt_labels.FrameLabel):
  tile = np.ascontiguousarray(rgb[..., ::-1])
  contours, _ = cv2.findContours(
    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
  )
  cv2.drawContours(tile, contours, -1, MASK_COLOR_BGR, 1)
  if label.box_xyxy is not None:
    x0, y0, x1, y1 = (int(round(value)) for value in label.box_xyxy)
    cv2.rectangle(tile, (x0, y0), (x1 - 1, y1 - 1), BOX_COLOR_BGR, 1)
  return utils_video.label(tile, f"{label.id} {label.camera}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("hdf5_paths", type=pathlib.Path, nargs="+")
  utils_run_dir.add_log_dir_flags(parser)
  parser.add_argument("--stride", type=int, default=8, help="Use every Nth 20 Hz step.")
  parser.add_argument("--cameras", nargs="+", default=list(ROBOT_CAMERAS))
  parser.add_argument(
    "--negative-sheet-every",
    type=int,
    default=4,
    help="Put every Nth no-shirt frame on the review sheets (all are labeled).",
  )
  args = parser.parse_args()
  unknown = set(args.cameras) - set(detections_lib.CAMERA_NAMES)
  if unknown:
    parser.error(f"Unknown cameras {sorted(unknown)}")

  output_dir = utils_run_dir.log_dir_from_args("build_shirt_labels", args)
  config = color_mask.ShirtColorConfig()
  labels, masks = [], {}
  positive_tiles, negative_tiles = [], []
  for path in args.hdf5_paths:
    dataset = dataset_reader.PolicyDataset(
      path, observation_keys=[f"{camera}_rgb" for camera in args.cameras]
    )
    for step in range(0, len(dataset), args.stride):
      sample, _ = dataset[step]
      for camera in args.cameras:
        rgb = sample[f"{camera}_rgb"]
        mask = color_mask.get_shirt_mask(rgb, config)
        label = shirt_labels.FrameLabel(
          id=len(labels),
          source=str(path),
          step=step,
          camera=camera,
          box_xyxy=color_mask.get_mask_box(mask),
          mask_area_px=int(mask.sum()),
        )
        labels.append(label)
        if label.box_xyxy is not None:
          masks[label.id] = mask
          positive_tiles.append(get_review_tile(rgb, mask, label))
        elif label.id % args.negative_sheet_every == 0:
          negative_tiles.append(get_review_tile(rgb, mask, label))
    dataset.close()

  shirt_labels.write_labels(output_dir, labels, masks, config)
  review_dir = output_dir / "review"
  utils_video.write_contact_sheets(review_dir, "positive", positive_tiles)
  utils_video.write_contact_sheets(review_dir, "negative", negative_tiles)
  print(
    f"{len(labels)} frames, {len(masks)} with a shirt, from {len(args.hdf5_paths)} "
    f"recordings -> {output_dir}"
  )


if __name__ == "__main__":
  main()
