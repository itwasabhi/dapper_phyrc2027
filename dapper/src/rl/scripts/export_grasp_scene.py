"""Exports the ShirtGraspEnv scene as a USD plus meta.json for the Isaac Lab env.

Run: dapper/run_sim.sh python /project/dapper/src/rl/scripts/export_grasp_scene.py \
    --output-dir /output/rl/scenes/grasp
"""

import argparse
import json
import pathlib

import numpy as np

from rl.envs import shirt_grasp_env

_EXCLUDED_PRIMS = ("/World/PolicySensors",)
_GRASP_CONSTANTS = (
  "GRIPPER_OPEN",
  "GRIPPER_CLOSED",
  "GRIPPER_RATE",
  "GRASP_TOLERANCE",
  "GRASP_MIN_PARTICLES",
  "GRASP_DRIVEN_TARGET",
  "GRAB_CLOSED_EPS",
  "GRAB_SETTLE_FRAMES",
  "GRAB_ANCHOR_COUNT",
  "SINGLE_LAYER_GRASP",
  "SINGLE_LAYER_GAP",
  "BASE_LINEAR_ACCEL",
  "BASE_ANGULAR_ACCEL",
  "PREVENT_WHEEL_LIFT",
)


def _to_list(value):
  return np.asarray(value, dtype=np.float64).tolist()


def _find_prim_path(stage, root: str, name: str) -> str:
  for prim in stage.Traverse():
    path = prim.GetPath().pathString
    if prim.GetName() == name and path.startswith(root + "/"):
      return path
  raise ValueError(f"No prim named {name} under {root}")


def _robot_meta(env, rig, index: int, prim_root: str) -> dict:
  robot = rig["robot"]
  ctrl = rig["ctrl"]
  M = env.M
  kps, kds = ctrl.get_gains()
  initial_position, initial_quat, initial_joints = env._initial_robots[index]
  limits = np.asarray(ctrl.get_joint_limits().cpu())
  return {
    "prim_root": prim_root,
    "articulation_root": robot.prim_path,
    "dof_names": list(robot.dof_names),
    "kps": _to_list(kps.cpu()),
    "kds": _to_list(kds.cpu()),
    "joint_limits": _to_list(limits),
    "max_efforts": _to_list(ctrl.get_max_efforts().cpu()),
    "initial_position": _to_list(initial_position),
    "initial_quat_wxyz": _to_list(initial_quat),
    "initial_joint_positions": _to_list(initial_joints),
    "indices": {
      key: _to_list(rig[key])
      for key in (
        "lift_idx",
        "arm_joint_idx",
        "yaw_idx",
        "pitch_idx",
        "roll_idx",
        "grip_idx",
      )
    },
    "limits": {
      key: float(rig[key])
      for key in rig
      if key.endswith(("_lo", "_hi")) and isinstance(rig[key], float)
    },
    "links": {
      "grasp_center": _find_prim_path(env.stage, prim_root, "grasp_center_link"),
      "fingertip_left": _find_prim_path(
        env.stage, prim_root, "gripper_fingertip_left_link"
      ),
      "fingertip_right": _find_prim_path(
        env.stage, prim_root, "gripper_fingertip_right_link"
      ),
    },
    "mounts": {
      role: _find_prim_path(env.stage, prim_root, role)
      for role in ("gripper_camera_link", "camera_center_link")
    },
    "wheel_lift_prevented": bool(M.PREVENT_WHEEL_LIFT),
  }


def _write_usd(env, output_path: pathlib.Path, robot_roots: list[str]) -> None:
  from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # pylint: disable=import-outside-toplevel

  flat = env.stage.Flatten()
  layer = Sdf.Layer.CreateNew(str(output_path))
  Sdf.CopySpec(flat, Sdf.Path("/World"), layer, Sdf.Path("/World"))
  layer.defaultPrim = "World"
  stage = Usd.Stage.Open(layer)
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  for path in _EXCLUDED_PRIMS:
    stage.RemovePrim(path)
  # configure_stretch4 sets these through the physics view, which is not saved in the USD.
  for root in robot_roots:
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
      if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        api.CreateSolverPositionIterationCountAttr().Set(32)
        api.CreateSolverVelocityIterationCountAttr().Set(4)
        api.CreateSleepThresholdAttr().Set(0.0)
  layer.Save()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=pathlib.Path, required=True)
  parser.add_argument("--resolution", type=int, default=128)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  import policy_cli  # pylint: disable=import-outside-toplevel

  policy_cli.install_failure_handler()
  env = shirt_grasp_env.ShirtGraspEnv(
    resolution=(args.resolution, args.resolution), defer_app_close=True
  )
  M = env.M
  robot_roots = [rig["robot"].prim_path.rsplit("/", 1)[0] for rig in env.rigs]
  _write_usd(env, args.output_dir / "scene.usd", robot_roots)

  contract = env.contract
  meta = {
    "scene_usd": "scene.usd",
    "physics_dt": float(env.world.get_physics_dt()),
    "physics_per_tick": env._physics_per_tick,
    "table_centers": _to_list(env.backend.garment_table_centers),
    "box_size": _to_list(M.BOX_SIZE),
    "floor_size": float(M.FLOOR_SIZE),
    "garment": {
      "mesh_prim": "/World/Garment/garment/mesh",
      "initial_position": _to_list(env._initial_garment_position),
      "initial_nodes": "garment_nodes.npy",
    },
    "robots": [
      _robot_meta(env, rig, index, root)
      for index, (rig, root) in enumerate(zip(env.rigs, robot_roots))
    ],
    "runtime": dict(env._runtime),
    "constants": {name: float(getattr(M, name)) for name in _GRASP_CONSTANTS},
    "cameras": contract["cameras"],
    "default_dimensions": contract["default_dimensions"],
  }
  np.save(args.output_dir / "garment_nodes.npy", env._initial_cloths[0])
  (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
  print(f"Wrote scene to {args.output_dir}", flush=True)
  shirt_grasp_env.close_simulation_app()


if __name__ == "__main__":
  main()
