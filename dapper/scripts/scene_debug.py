"""Replays a recorded policy.hdf5 through BasicSceneIntegrator and dumps the grid layers.

Run: dapper/run_py.sh /project/dapper/scripts/scene_debug.py /output/solution_gamma/run0/policy.hdf5
Writes to <run dir>/scene_debug unless --output-dir is given.
"""

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path("/project")
if not (PROJECT_ROOT / "src/PhyRC_Sim").is_dir():
  PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src/PhyRC_Sim"))
sys.path.insert(0, str(PROJECT_ROOT / "dapper/src"))

from Policy import dataset_reader  # noqa: E402

import utils_video  # noqa: E402
from vision import camera_geometry  # noqa: E402
from vision import detections as detections_lib  # noqa: E402
from vision import detector_factory  # noqa: E402
from vision import scene as scene_lib  # noqa: E402
from vision import scene_integrator  # noqa: E402

LAYER_SCALE = 2
HEIGHT_MAX_M = 1.8


def _layer_image(name: str, layer: np.ndarray) -> np.ndarray:
  scale = HEIGHT_MAX_M if name == "height" else (layer.max() or 1.0)
  gray = (np.clip(layer / scale, 0, 1) * 255).astype(np.uint8)
  image = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
  image = cv2.resize(
    image, None, fx=LAYER_SCALE, fy=LAYER_SCALE, interpolation=cv2.INTER_NEAREST
  )
  return utils_video.label(np.ascontiguousarray(image[::-1]), name)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("hdf5_path", type=pathlib.Path)
  parser.add_argument("--output-dir", type=pathlib.Path)
  parser.add_argument("--stride", type=int, default=4, help="Use every Nth 20 Hz step.")
  args = parser.parse_args()
  output_dir = args.output_dir or args.hdf5_path.parent / "scene_debug"

  integrator = scene_integrator.BasicSceneIntegrator(
    detector=detector_factory.make_omdet_turbo(),
    mounts=camera_geometry.load_mounts(),
    config=scene_integrator.GridConfig(update_every_n_calls=1),
  )
  names = detections_lib.CAMERA_NAMES
  keys = [
    f"{name}_{kind}" for name in names for kind in ("rgb", "depth", "depth_valid")
  ] + ["base_pose_world", "grasp_pose_base"]
  dataset = dataset_reader.PolicyDataset(args.hdf5_path, observation_keys=keys)
  scene = scene_lib.SceneEstimate()
  for index in range(0, len(dataset), args.stride):
    raw, _ = dataset[index]
    observation = {
      kind: np.stack([raw[f"{name}_{kind}"] for name in names])
      for kind in ("rgb", "depth", "depth_valid")
    }
    observation["base_pose_world"] = raw["base_pose_world"].reshape(2, 7)
    observation["grasp_pose_base"] = raw["grasp_pose_base"].reshape(2, 7)
    scene = integrator.update(observation)
  dataset.close()

  output_dir.mkdir(parents=True, exist_ok=True)
  layers = integrator.debug_layers()
  images = [_layer_image(name, layer) for name, layer in layers.items()]
  images.append(
    np.ascontiguousarray(
      cv2.resize(scene.debug_image[..., ::-1], None, fx=LAYER_SCALE, fy=LAYER_SCALE)
    )
  )
  cv2.imwrite(str(output_dir / "layers.png"), utils_video.tile(images, num_columns=4))
  (output_dir / "scene_estimate.json").write_text(
    json.dumps(scene_lib.to_jsonable(scene), indent=2)
  )
  print(f"Wrote {output_dir}")


if __name__ == "__main__":
  main()
