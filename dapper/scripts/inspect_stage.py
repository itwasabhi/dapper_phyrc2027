"""Prints the ShirtGraspEnv stage hierarchy (throwaway exploration aid).

Run: dapper/run_sim.sh python /project/dapper/scripts/inspect_stage.py
"""

from rl.envs import shirt_grasp_env

env = shirt_grasp_env.ShirtGraspEnv(resolution=(64, 64))
stage = env.stage
for child in stage.GetPrimAtPath("/World").GetChildren():
  print(
    "W",
    child.GetPath(),
    child.GetTypeName(),
    len(list(child.GetChildren())),
    flush=True,
  )
print("DEFAULT", stage.GetDefaultPrim().GetPath(), flush=True)
cloth = env.cloths[0]
print("CLOTH", cloth.prim_path, flush=True)
for prim in stage.Traverse():
  path = str(prim.GetPath())
  if path.startswith("/World/Garment") or "Looks" in path and "Hem" in path:
    print("G", path, prim.GetTypeName(), list(prim.GetAppliedSchemas()), flush=True)
for prim in stage.GetPrimAtPath("/World/PolicySensors").GetChildren():
  print("CAM", prim.GetPath(), prim.GetTypeName(), flush=True)
rig = env.rigs[0]
print("ROBOT", rig["robot"].prim_path, rig["robot"].dof_names, flush=True)
print("PHYS", env.world.get_physics_dt(), flush=True)
shirt_grasp_env.close_simulation_app()
