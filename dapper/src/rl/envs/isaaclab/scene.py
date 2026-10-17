"""Isaac Lab scene and simulation configs for the scene exported by export_grasp_scene.py.

Isaac Lab quaternions are (x, y, z, w); the exported meta.json uses (w, x, y, z).
"""

import dataclasses
import json
import pathlib
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import transform

from isaaclab import actuators
from isaaclab import assets
from isaaclab import scene as lab_scene
from isaaclab import sensors
from isaaclab import sim as sim_utils
from isaaclab_physx import physics as physx_physics

from Policy import sensors as policy_sensors

CAMERA_APERTURE = 20.955
# Same value as the upstream /OmniverseKit dome light, which is not part of /World.
_DOME_LIGHT_INTENSITY = 1000.0
ENV_SPACING_M = 10.0
_UNLIMITED_EFFORT = 1e9


@dataclasses.dataclass(frozen=True)
class GraspSceneMeta:
  """The exported scene directory: USD path plus the parsed meta.json."""

  directory: pathlib.Path
  meta: Mapping[str, Any]

  @property
  def scene_usd(self) -> str:
    return str(self.directory / self.meta["scene_usd"])

  @property
  def robots(self) -> Sequence[Mapping[str, Any]]:
    return self.meta["robots"]


def load_scene_meta(directory: str | pathlib.Path) -> GraspSceneMeta:
  directory = pathlib.Path(directory)
  return GraspSceneMeta(directory, json.loads((directory / "meta.json").read_text()))


def wxyz_to_xyzw(quat_wxyz: Sequence[float]) -> tuple[float, float, float, float]:
  w, x, y, z = quat_wxyz
  return (x, y, z, w)


def _pose_from_matrix(
  matrix: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
  quat = transform.Rotation.from_matrix(matrix[:3, :3]).as_quat()
  return tuple(float(v) for v in matrix[:3, 3]), tuple(float(v) for v in quat)


def _robot_cfg(robot: Mapping[str, Any], prim_name: str) -> assets.ArticulationCfg:
  names = robot["dof_names"]
  # Unlisted joints would otherwise get a zero effort limit; "unlimited" is float max.
  efforts = {
    name: min(effort, _UNLIMITED_EFFORT)
    for name, effort in zip(names, robot["max_efforts"])
  }
  limits = np.asarray(robot["joint_limits"])
  initial_joints = np.clip(
    robot["initial_joint_positions"], limits[:, 0], limits[:, 1]
  ).tolist()
  return assets.ArticulationCfg(
    prim_path=f"{{ENV_REGEX_NS}}/World/{prim_name}/base_footprint",
    spawn=None,
    init_state=assets.ArticulationCfg.InitialStateCfg(
      pos=tuple(robot["initial_position"]),
      rot=wxyz_to_xyzw(robot["initial_quat_wxyz"]),
      joint_pos=dict(zip(names, initial_joints)),
    ),
    actuators={
      "all": actuators.ImplicitActuatorCfg(
        joint_names_expr=list(names),
        stiffness=dict(zip(names, robot["kps"])),
        damping=dict(zip(names, robot["kds"])),
        effort_limit_sim=efforts,
      )
    },
  )


def _camera_cfg(
  spec: Mapping[str, Any],
  robots: Sequence[Mapping[str, Any]],
  width: int,
  height: int,
) -> sensors.TiledCameraCfg:
  fov = np.deg2rad(spec["horizontal_fov_deg"])
  focal = CAMERA_APERTURE / (2 * np.tan(fov / 2))
  pinhole = sim_utils.PinholeCameraCfg(
    focal_length=float(focal),
    horizontal_aperture=CAMERA_APERTURE,
    clipping_range=tuple(spec["clip_m"]),
  )
  if "eye_world_m" in spec:
    parent = "{ENV_REGEX_NS}/World"
    matrix = policy_sensors.look_at(
      np.array(spec["eye_world_m"], dtype=float), spec["target_world_m"], [0, 0, 1]
    )
  else:
    robot_index = int(spec["id"].split("_")[1])
    robot = robots[robot_index]
    parent = "{ENV_REGEX_NS}" + robot["mounts"][spec["mount_role"]]
    matrix = policy_sensors.look_at(
      np.array(spec["eye_mount_m"], dtype=float),
      spec["target_mount_m"],
      spec["up_mount"],
    )
  position, rotation = _pose_from_matrix(matrix)
  return sensors.TiledCameraCfg(
    prim_path=f"{parent}/policy_{spec['id']}",
    spawn=pinhole,
    offset=sensors.TiledCameraCfg.OffsetCfg(
      pos=position, rot=rotation, convention="opengl"
    ),
    data_types=["rgb", "distance_to_image_plane"],
    width=width,
    height=height,
  )


def without_shirt(scene_meta: GraspSceneMeta) -> GraspSceneMeta:
  """A copy of the scene whose USD has no garment (needs the app running for pxr)."""
  from pxr import Usd  # pylint: disable=import-outside-toplevel

  stage = Usd.Stage.Open(scene_meta.scene_usd)
  stage.RemovePrim("/World/Garment")
  path = scene_meta.directory / "scene_noshirt.usd"
  stage.Export(str(path))
  return GraspSceneMeta(
    scene_meta.directory, {**scene_meta.meta, "scene_usd": path.name}
  )


def make_scene_cfg(
  scene_meta: GraspSceneMeta,
  num_envs: int,
  resolution: tuple[int, int],
  include_shirt: bool = True,
) -> lab_scene.InteractiveSceneCfg:
  """`resolution` is (width, height). Entities: robot_<i>, shirt, camera_<i>."""
  width, height = resolution
  cfg = lab_scene.InteractiveSceneCfg(
    # Deformables and collision groups cannot be physics-replicated.
    num_envs=num_envs,
    env_spacing=ENV_SPACING_M,
    replicate_physics=False,
    # With filtering on, the in-env floor collider does not collide; envs are far apart anyway.
    filter_collisions=False,
  )
  cfg.light = assets.AssetBaseCfg(
    prim_path="/World/Light",
    spawn=sim_utils.DomeLightCfg(intensity=_DOME_LIGHT_INTENSITY),
  )
  cfg.world = assets.AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/World",
    spawn=sim_utils.UsdFileCfg(usd_path=scene_meta.scene_usd),
  )
  for index, robot in enumerate(scene_meta.robots):
    prim_name = robot["prim_root"].rsplit("/", 1)[1]
    setattr(cfg, f"robot_{index}", _robot_cfg(robot, prim_name))
  if include_shirt:
    cfg.shirt = assets.DeformableObjectCfg(
      prim_path="{ENV_REGEX_NS}/World/Garment", spawn=None
    )
  for index, spec in enumerate(scene_meta.meta["cameras"]):
    setattr(
      cfg,
      f"camera_{index}",
      _camera_cfg(spec, scene_meta.robots, width, height),
    )
  return cfg


def make_sim_cfg(
  scene_meta: GraspSceneMeta, device: str, render_interval: int = 1
) -> sim_utils.SimulationCfg:
  return sim_utils.SimulationCfg(
    dt=scene_meta.meta["physics_dt"],
    device=device,
    render_interval=render_interval,
    physics=physx_physics.PhysxCfg(
      solver_type=1,
      enable_ccd=True,
      enable_stabilization=True,
      enable_enhanced_determinism=True,
      gpu_max_rigid_contact_count=10240000,
      gpu_max_rigid_patch_count=10240000,
    ),
  )


def configure_physics_scene(sim: sim_utils.SimulationContext, num_envs: int) -> None:
  """Scene attributes that SimulationCfg does not expose."""
  from pxr import PhysxSchema  # pylint: disable=import-outside-toplevel

  api = PhysxSchema.PhysxSceneAPI.Apply(sim.stage.GetPrimAtPath("/physicsScene"))
  # The upstream env uses 4M for one dense shirt on a table; the pool is shared by all envs.
  api.GetGpuMaxDeformableSurfaceContactsAttr().Set(1048576 * max(4, num_envs))
