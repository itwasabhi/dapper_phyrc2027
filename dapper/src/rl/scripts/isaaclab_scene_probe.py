"""Builds the exported grasp scene in Isaac Lab for N envs and times physics and cameras.

Run: DAPPER_IMAGE=dapper-lab dapper/run_sim.sh python /project/dapper/src/rl/scripts/isaaclab_scene_probe.py \
    --scene-dir /output/rl/scenes/grasp --num-envs 4
"""

import argparse
import time

from isaaclab import app

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene-dir", required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--policy-steps", type=int, default=10)
parser.add_argument("--save-images", help="Directory for a PNG of env 0 cameras.")
parser.add_argument(
  "--no-shirt", action="store_true", help="Sanity check: drop the cloth."
)
args = parser.parse_args()
launcher = app.AppLauncher(headless=True, enable_cameras=True)

import torch  # pylint: disable=wrong-import-position
from isaaclab import scene as lab_scene  # pylint: disable=wrong-import-position
from isaaclab import sim as sim_utils  # pylint: disable=wrong-import-position

from rl.envs.isaaclab import observations  # pylint: disable=wrong-import-position
from rl.envs.isaaclab import scene as grasp_scene  # pylint: disable=wrong-import-position

SUBSTEPS = 12


def main() -> None:
  meta = grasp_scene.load_scene_meta(args.scene_dir)
  if args.no_shirt:
    meta = grasp_scene.without_shirt(meta)
  sim = sim_utils.SimulationContext(grasp_scene.make_sim_cfg(meta, "cuda:0"))
  if not args.no_shirt:
    grasp_scene.configure_physics_scene(sim, args.num_envs)
  t0 = time.perf_counter()
  scene_cfg = grasp_scene.make_scene_cfg(
    meta, args.num_envs, (args.resolution, args.resolution)
  )
  if args.no_shirt:
    del scene_cfg.shirt
  scene = lab_scene.InteractiveScene(scene_cfg)
  sim.reset()
  print(f"PROBE build+reset {time.perf_counter() - t0:.1f}s", flush=True)
  dt = sim.get_physics_dt()
  for index in range(len(meta.robots)):
    robot = scene[f"robot_{index}"]
    pose = robot.data.default_root_pose.torch.clone()
    pose[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim_index(root_pose=pose)
  shirt = None if args.no_shirt else scene["shirt"]

  def report(tag: str) -> None:
    print(
      f"PROBE {tag}: robot0 z={scene['robot_0'].data.root_pos_w.torch[:, 2].tolist()} "
      f"shirt z={None if shirt is None else shirt.data.nodal_pos_w.torch[..., 2].mean(1).tolist()}",
      flush=True,
    )

  def run(render: bool, steps: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
      for _ in range(SUBSTEPS):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
      if render:
        sim.render()
        scene.update(dt)
        for index in range(len(meta.meta["cameras"])):
          scene[f"camera_{index}"].data.output["rgb"]
    torch.cuda.synchronize()
    return time.perf_counter() - start

  report("initial")
  run(False, 2)
  report("after 2 policy steps")
  physics_s = run(False, args.policy_steps)
  print(
    f"PROBE physics-only: {args.policy_steps} policy steps in {physics_s:.2f}s "
    f"= {args.policy_steps * args.num_envs / physics_s:.2f} env-steps/s",
    flush=True,
  )
  report("after physics-only phase")
  run(True, 2)
  render_s = run(True, args.policy_steps)
  print(
    f"PROBE with cameras: {args.policy_steps} policy steps in {render_s:.2f}s "
    f"= {args.policy_steps * args.num_envs / render_s:.2f} env-steps/s",
    flush=True,
  )
  report("end")
  if args.save_images:
    import numpy as np  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel

    tiles = [
      observations.as_torch(scene[f"camera_{i}"].data.output["rgb"])[0, ..., :3]
      .cpu()
      .numpy()
      for i in range(len(meta.meta["cameras"]))
    ]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(args.save_images)
  rgb = scene["camera_0"].data.output["rgb"]
  print("PROBE rgb", tuple(rgb.shape), rgb.dtype, float(rgb.float().mean()), flush=True)
  print(
    "PROBE robot0 root", scene["robot_0"].data.root_pos_w.torch[0].tolist(), flush=True
  )


main()
launcher.app.close()
