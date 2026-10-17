"""Simplified single-scene training env: two robots start near corners of the shirt's table.

Adapted from `Policy/environment.py` (DressingEnv); observations and actions follow
the same public contract. One instance per process (Isaac owns the application).
"""

import copy
import os
import pathlib
from typing import Callable, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from rl.envs import rewards
from rl.envs import spawn

RewardFn = Callable[..., Tuple[float, bool]]

_MIN_SETTLE_TICKS = 60
_POLICY_STEP_S = 0.05
_simulation_module = None


class SettleError(RuntimeError):
  """The scene did not come to rest after placement (e.g. a robot spawned in contact)."""


def close_simulation_app() -> None:
  """Closes Isaac if a ShirtGraspEnv started it in this process; may fast-exit."""
  if _simulation_module is not None:
    _simulation_module.simulation_app.close()


def _no_human_randomization(stage, path):
  del stage, path
  return {
    "offset_m": [0.0, 0.0],
    "yaw_deg": 0.0,
    "seed": None,
    "pivot_m": [0.0, 0.0],
    "randomized": False,
  }


class ShirtGraspEnv(gym.Env):
  metadata = {"render_modes": [], "render_fps": 20}
  _claimed = False

  def __init__(
    self,
    *,
    resolution: Sequence[int] = (128, 128),
    max_episode_steps: int = 200,
    settle_ticks: int = 90,
    table_index: Optional[int] = None,
    shirt_xy_jitter_m: float = 0.0,
    spawn_config: spawn.SpawnConfig = spawn.SpawnConfig(),
    reward_fn: RewardFn = rewards.zero_reward,
    defer_app_close: bool = True,
    max_reset_attempts: int = 5,
  ):
    global _simulation_module
    # pylint: disable=import-outside-toplevel
    from Policy import contract as policy_contract
    from Policy import sensors as policy_sensors
    from Policy import state as policy_state

    if max_episode_steps < 1 or settle_ticks < _MIN_SETTLE_TICKS:
      raise ValueError(
        "Positive episode horizon and at least 60 settling ticks required"
      )
    if ShirtGraspEnv._claimed:
      raise RuntimeError("Isaac supports one env per process; launch a new process")
    width, height = (int(v) for v in resolution)
    contract_path = (
      pathlib.Path(policy_contract.__file__).resolve().parents[3]
      / "config/policy_interface.json"
    )
    self.contract = policy_contract.load_contract(contract_path)
    self.contract["default_dimensions"].update(width=width, height=height)
    self._contract_lib, self._snapshot = policy_contract, policy_state.snapshot
    self.max_episode_steps, self.settle_ticks = (
      int(max_episode_steps),
      int(settle_ticks),
    )
    self.table_index, self.shirt_xy_jitter_m = table_index, float(shirt_xy_jitter_m)
    self.spawn_config, self.reward_fn = spawn_config, reward_fn
    self.defer_app_close = defer_app_close
    self.max_reset_attempts = max_reset_attempts
    self._done, self._closed = True, False

    os.environ["STRETCH4_HEADLESS"] = "1"
    os.environ["STRETCH4_SHOW_COLLIDER"] = "0"
    import Env_StandAlone.Teleop_TShirt_Stretch4_Env as M

    ShirtGraspEnv._claimed = True
    _simulation_module = self.M = M
    original_placement = M.randomize_human_and_chair
    M.randomize_human_and_chair = _no_human_randomization
    try:
      self.backend = M.TeleopTShirtStretch4_Env()
    finally:
      M.randomize_human_and_chair = original_placement
    self.world, self.stage = self.backend.world, self.backend.stage
    self.rigs = (self.backend.rig, self.backend.rig2)
    self.cloths, self.faces, self._post = M.initialize_manipulation(self.backend)
    self._initial_states = [copy.deepcopy(rig["state"]) for rig in self.rigs]
    self._initial_robots = [
      (
        policy_state.array(rig["robot"].get_world_pose()[0]),
        policy_state.array(rig["robot"].get_world_pose()[1]),
        policy_state.array(rig["robot"].get_joint_positions()),
      )
      for rig in self.rigs
    ]
    self._initial_cloths = [
      policy_state.array(cloth.get_world_positions()) for cloth in self.cloths
    ]
    self._initial_garment_position = np.array(
      self.backend.garment_spawn["spawn_position_world_m"]
    )
    self._array = policy_state.array
    self._toggles = [
      M.make_gripper_toggle(rig, f"policy_robot_{i}", self.cloths, self.faces)
      for i, rig in enumerate(self.rigs)
    ]
    self._runtime = {
      key: float(getattr(M, key))
      for key in (
        "BASE_LINEAR_RATE",
        "BASE_ANGULAR_RATE",
        "LIFT_RATE",
        "ARM_RATE",
        "WRIST_RATE",
      )
    }
    self._physics_per_tick = round((1 / 60) / self.world.get_physics_dt())
    if not np.isclose(self._physics_per_tick * self.world.get_physics_dt(), 1 / 60):
      raise ValueError("Physics frequency must be an integer multiple of 60 Hz")

    self.action_space = gym.spaces.Box(-1, 1, (len(self.rigs), 9), dtype=np.float32)
    shapes = policy_contract.observation_shapes(
      self.contract,
      robots=len(self.rigs),
      joints=len(self.rigs[0]["robot"].dof_names),
      profile="actor_rgbd",
    )
    spaces = {}
    for name, shape in shapes.items():
      spec = self.contract["observations"][name]
      dtype = np.dtype(spec["dtype"])
      low, high = spec.get("range", (0, 1) if dtype == bool else (-np.inf, np.inf))
      spaces[name] = gym.spaces.Box(low, high, shape, dtype=dtype)
    self.observation_space = gym.spaces.Dict(spaces)
    self.sensors = policy_sensors.RGBDSensors(
      self.stage, self.world, self.rigs, self.contract
    )

  def _attach_callbacks(self):
    self.world.add_physics_callback(
      "stretch4_grab_follow",
      lambda dt: self.M.cloth_pre_step(self.cloths, self.rigs, dt),
    )
    self._post = self.world._physics_context._physics_sim_interface.subscribe_physics_on_step_events(
      pre_step=False,
      order=100,
      on_update=lambda dt, context: self.M.cloth_post_step(self.cloths, self.rigs, dt),
    )

  def _detach_callbacks(self):
    self.world.remove_physics_callback("stretch4_grab_follow")
    self._post = None

  def _tick(self, decoded):
    from Env_Config.Garment.ContinuousClothControl import cloth_control_batch  # pylint: disable=import-outside-toplevel

    with cloth_control_batch():
      for rig, command in zip(self.rigs, decoded):
        self.M.drive_robot(
          rig,
          self.M.ROBOT1_KEYMAP,
          set(),
          self.cloths,
          1 / 60,
          0,
          commands=command["velocity_commands"],
        )
    for rig in self.rigs:
      state = rig["state"]
      if state.get("pending_grab"):
        if abs(state["grip_pos"] - self.M.GRIPPER_CLOSED) > self.M.GRAB_CLOSED_EPS:
          state["_settle"] = 0
        else:
          state["_settle"] = state.get("_settle", 0) + 1
          if state["_settle"] >= self.M.GRAB_SETTLE_FRAMES:
            state["pending_grab"], state["_settle"] = False, 0
            rig["attempt_grab"]()
    for _ in range(self._physics_per_tick):
      self.world.step(render=False)

  def _place_robots(self, spawns: Sequence[spawn.RobotSpawn]):
    for rig, (initial_position, _, joint_positions), robot_spawn in zip(
      self.rigs, self._initial_robots, spawns
    ):
      robot = rig["robot"]
      robot.initialize()
      self.M.stretch4_tuning(rig, restore=True)
      position = robot_spawn.position.copy()
      position[2] = initial_position[2]
      robot.set_world_pose(self.M._to_t(position), self.M._to_t(robot_spawn.quat_wxyz))
      robot.set_joint_positions(self.M._to_t(joint_positions))
      robot.set_joint_velocities(self.M._to_t(np.zeros_like(joint_positions)))
      robot._articulation_view.set_velocities(self.M._to_t(np.zeros((1, 6))))

  def _place_shirt(self, table_center_xyz: np.ndarray):
    target = table_center_xyz + np.array([0.0, 0.0, self.M.BOX_SIZE[2] / 2 + 0.2])
    target[:2] += self.np_random.uniform(
      -self.shirt_xy_jitter_m, self.shirt_xy_jitter_m, size=2
    )
    offset = target - self._initial_garment_position
    positions = [(p + offset).astype(np.float32) for p in self._initial_cloths]
    for cloth, cloth_positions in zip(self.cloths, positions):
      cloth.initialize()
      cloth.set_world_positions(cloth_positions.copy())
      cloth.set_velocities(np.zeros_like(cloth_positions))
    restore_error = max(
      float(np.max(np.abs(self._array(c.get_world_positions()) - p)))
      for c, p in zip(self.cloths, positions)
    )
    if restore_error > 1e-7:
      raise RuntimeError(f"Cloth reset restore error: {restore_error} m")

  def reset(self, *, seed=None, options=None):
    if self._closed:
      raise RuntimeError("Environment is closed")
    if options:
      raise ValueError("No reset options supported; use constructor configuration")
    super().reset(seed=seed)
    self._done = True
    failures = []
    for _ in range(self.max_reset_attempts):
      try:
        observation = self._reset_scene()
      except SettleError as error:
        failures.append(str(error))
        continue
      self.reset_attempts = len(failures) + 1
      self._done = False
      return observation, self._info()
    raise SettleError(f"Reset failed {len(failures)} times: {failures}")

  def _reset_scene(self):
    from Env_Config.Garment.NativeGrasp import clear_native_grasp  # pylint: disable=import-outside-toplevel

    self._detach_callbacks()
    for rig, initial in zip(self.rigs, self._initial_states):
      clear_native_grasp(rig, self.stage)
      rig["state"].clear()
      rig["state"].update(copy.deepcopy(initial))
      rig["held"].clear()
    self.world.stop()
    self.backend.reset()
    self.M.simulation_app.update()

    centers = np.asarray(self.backend.garment_table_centers, dtype=np.float64)
    self.current_table_index = (
      int(self.table_index)
      if self.table_index is not None
      else int(self.np_random.integers(len(centers)))
    )
    table_center = centers[self.current_table_index]
    self.robot_spawns = spawn.sample_robot_spawns(
      table_center[:2], self.M.BOX_SIZE[:2], self.np_random, self.spawn_config
    )
    self._place_robots(self.robot_spawns)
    self._place_shirt(table_center)
    self.M.rebuild_contact_guards(self.backend, self.cloths)
    self._attach_callbacks()

    neutral = self._contract_lib.decode_action(
      np.zeros(self.action_space.shape, np.float32), self.contract, self._runtime
    )
    for _ in range(self.settle_ticks):
      self._tick(neutral)
    self._previous_action = np.zeros(self.action_space.shape, np.float32)
    self._step_id = 0
    self.sensors.reset_episode()
    observation = self._observe()
    speeds = np.linalg.norm(observation["base_twist_world"][:, :3], axis=1)
    finger_error = max(
      float(
        np.max(
          np.abs(
            self._array(r["robot"].get_joint_positions())[r["grip_idx"]]
            - self.M.GRIPPER_OPEN
          )
        )
      )
      for r in self.rigs
    )
    if speeds.max() > 0.05 or finger_error > 0.03:
      raise SettleError(
        f"table {self.current_table_index} corners {[s.corner_index for s in self.robot_spawns]}: "
        f"max base speed {speeds.max():.3f} m/s, finger error {finger_error:.3f} rad"
      )
    self._last_observation = {key: value.copy() for key, value in observation.items()}
    return observation

  def _info(self) -> Mapping[str, object]:
    return {
      "table_index": self.current_table_index,
      "robot_corners": [s.corner_index for s in self.robot_spawns],
      "step_id": self._step_id,
    }

  def _observe(self):
    meta, _ = self._snapshot(self.M, self.cloths, self.rigs)
    if not meta["all_arrays_finite"]:
      raise RuntimeError("Nonfinite physics state; episode invalid, call reset")
    observation = self._contract_lib.pack_measured_state(meta, self.contract)
    sensor_values, _ = self.sensors.capture()
    observation.update(sensor_values, previous_action=self._previous_action.copy())
    self._contract_lib.validate_observation(
      observation,
      self.contract,
      robots=len(self.rigs),
      joints=len(meta["robots"][0]["joints"]),
      profile="actor_rgbd",
    )
    return observation

  def step(self, action):
    if self._closed or self._done:
      raise RuntimeError("Call reset before step (also after termination/truncation)")
    decoded = self._contract_lib.decode_action(
      action, self.contract, self._runtime, robots=len(self.rigs)
    )
    before = float(self.world.current_time)
    try:
      for rig, command, toggle in zip(self.rigs, decoded, self._toggles):
        closed = bool(rig["state"]["gripper_closed"])
        if (
          self._contract_lib.resolve_gripper_intent(closed, command["gripper_intent"])
          != closed
        ):
          toggle()
      for _ in range(3):
        self._tick(decoded)
      if not np.isclose(
        float(self.world.current_time) - before, _POLICY_STEP_S, atol=1e-7, rtol=0
      ):
        raise RuntimeError("Policy step did not advance exactly 0.05 simulated seconds")
      self._previous_action = np.array(action, copy=True)
      self._step_id += 1
      observation = self._observe()
      reward, terminated = self.reward_fn(
        self._last_observation, np.array(action, copy=True), observation
      )
      if not np.isfinite(reward):
        raise ValueError("reward_fn returned a nonfinite reward")
      truncated = self._step_id >= self.max_episode_steps and not terminated
      self._done = bool(terminated) or truncated
      self._last_observation = {key: value.copy() for key, value in observation.items()}
      return observation, float(reward), bool(terminated), truncated, self._info()
    except Exception:
      self._done = True
      raise

  def close(self):
    if self._closed:
      return
    self._closed = self._done = True
    self._detach_callbacks()
    self.sensors.close()
    if not self.defer_app_close:
      self.M.simulation_app.close()
