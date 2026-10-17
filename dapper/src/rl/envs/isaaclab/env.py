"""Batched Isaac Lab env over the exported grasp scene.

Currently a scene-only env: actions are accepted and ignored, the reward is zero, and
observations carry joint state plus the five camera images.
"""

import gymnasium as gym
import numpy as np
import torch

from isaaclab import envs as lab_envs

from rl.envs.isaaclab import observations
from rl.envs.isaaclab import scene as grasp_scene

NUM_ROBOTS = 2
DECIMATION = 12


class SceneEnv(lab_envs.DirectRLEnv):
  """Adapter between Isaac Lab's DirectRLEnv hooks and the injected scene functions."""

  def __init__(self, cfg: lab_envs.DirectRLEnvCfg, clip_ranges, has_shirt: bool):
    self._clip_ranges = clip_ranges
    self._has_shirt = has_shirt
    super().__init__(cfg)
    self._robots = [self.scene[f"robot_{i}"] for i in range(NUM_ROBOTS)]
    self._cameras = [self.scene[f"camera_{i}"] for i in range(len(clip_ranges))]
    self._previous_action = torch.zeros(
      self.num_envs, NUM_ROBOTS, 9, device=self.device
    )

  def _setup_scene(self) -> None:
    # Runs before the physics reset, which is when the pool capacity must be set.
    if self._has_shirt:
      grasp_scene.configure_physics_scene(self.sim, self.cfg.scene.num_envs)

  def _pre_physics_step(self, actions: torch.Tensor) -> None:
    self._previous_action = actions.clone()

  def _apply_action(self) -> None:
    pass

  def _get_observations(self):
    time_s = self.episode_length_buf.to(torch.float64) * self.step_dt
    # Flat (not under "policy"): IsaacLabWrapper derives its specs from the inner space.
    return observations.build_observation(
      self._robots, self._cameras, self._clip_ranges, self._previous_action, time_s
    )

  def _get_rewards(self) -> torch.Tensor:
    return torch.zeros(self.num_envs, device=self.device)

  def _get_dones(self):
    terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return terminated, self.episode_length_buf >= self.max_episode_length - 1

  def _reset_idx(self, env_ids) -> None:
    super()._reset_idx(env_ids)
    for robot in self._robots:
      pose = observations.as_torch(robot.data.default_root_pose)[env_ids].clone()
      pose[:, :3] += self.scene.env_origins[env_ids]
      robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
      robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
      )
      robot.write_joint_position_to_sim_index(
        position=observations.as_torch(robot.data.default_joint_pos)[env_ids],
        env_ids=env_ids,
      )
      robot.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(observations.as_torch(robot.data.default_joint_vel))[
          env_ids
        ],
        env_ids=env_ids,
      )
    if self._has_shirt:
      shirt = self.scene["shirt"]
      shirt.write_nodal_state_to_sim_index(
        observations.as_torch(shirt.data.default_nodal_state_w)[env_ids],
        env_ids=env_ids,
      )


def make_scene_env(
  scene_dir: str,
  num_envs: int,
  resolution: tuple[int, int],
  max_episode_steps: int,
  device: str = "cuda:0",
  include_shirt: bool = True,
) -> SceneEnv:
  """Call after the Isaac app is launched."""
  meta = grasp_scene.load_scene_meta(scene_dir)
  if not include_shirt:
    meta = grasp_scene.without_shirt(meta)
  sim_cfg = grasp_scene.make_sim_cfg(meta, device, render_interval=DECIMATION)
  scene_cfg = grasp_scene.make_scene_cfg(
    meta, num_envs, tuple(resolution), include_shirt=include_shirt
  )
  cameras = meta.meta["cameras"]
  num_joints = len(meta.robots[0]["dof_names"])
  width, height = resolution
  cfg = lab_envs.DirectRLEnvCfg(
    sim=sim_cfg,
    scene=scene_cfg,
    decimation=DECIMATION,
    episode_length_s=max_episode_steps * DECIMATION * meta.meta["physics_dt"],
    action_space=gym.spaces.Box(-1.0, 1.0, (NUM_ROBOTS, 9), np.float32),
    observation_space=observations.observation_space(
      NUM_ROBOTS, num_joints, len(cameras), width, height
    ),
  )
  return SceneEnv(cfg, [spec["clip_m"] for spec in cameras], include_shirt)
