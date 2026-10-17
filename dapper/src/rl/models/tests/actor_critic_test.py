import functools

import pytest
import tensordict
import torch
from torchrl import envs

import utils_torchrl_env
from rl.envs import fake_env
from rl.models import actor_critic
from rl.models import encoders

PROPRIO_KEYS = (
  "joint_position",
  "joint_velocity",
  "gripper_close_command",
  "previous_action",
)
RESOLUTION = 64


def _make_env():
  return utils_torchrl_env.make_env(
    fake_env.FakeShirtGraspEnv(num_cameras=3, resolution=RESOLUTION)
  )


def _build(env):
  shapes = actor_critic.sample_shapes_from_env(
    env, [*actor_critic.IMAGE_KEYS, *PROPRIO_KEYS]
  )
  encoder_factory = functools.partial(
    encoders.RgbdCnnEncoder, cameras=[0, 1, 2], feature_dim=16
  )
  return shapes, actor_critic.build_actor_critic(
    encoder_factory, shapes, env.action_spec_unbatched, PROPRIO_KEYS, hidden_dims=(32,)
  )


def _random_observation(shapes, batch_shape):
  data = {}
  for key, sample_shape in shapes.observations.items():
    shape = (*batch_shape, *sample_shape)
    if key == "rgb":
      data[key] = torch.randint(0, 255, shape, dtype=torch.uint8)
    elif key in ("depth_valid", "gripper_close_command"):
      data[key] = torch.rand(shape) > 0.5
    else:
      data[key] = torch.rand(shape)
  return tensordict.TensorDict(data, batch_size=batch_shape)


def test_sample_shapes_strip_env_batch_dims():
  single = actor_critic.sample_shapes_from_env(_make_env(), ["rgb", "joint_position"])
  batched = actor_critic.sample_shapes_from_env(
    envs.SerialEnv(2, _make_env), ["rgb", "joint_position"]
  )
  assert single == batched
  assert single.observations["rgb"] == (3, RESOLUTION, RESOLUTION, 3)
  assert single.action == (2, 9)


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 4)])
def test_actor_and_critic_outputs_follow_batch_shape(batch_shape):
  shapes, (actor, critic) = _build(_make_env())
  observation = _random_observation(shapes, batch_shape)
  actor_out = actor(observation.clone())
  assert actor_out["action"].shape == (*batch_shape, 2, 9)
  assert actor_out["action"].abs().max() <= 1.0
  assert actor_out["action_log_prob"].shape == batch_shape
  assert critic(observation.clone())["state_value"].shape == (*batch_shape, 1)


def test_actor_gradients_reach_encoder():
  shapes, (actor, _) = _build(_make_env())
  out = actor(_random_observation(shapes, (2,)))
  out["action_log_prob"].sum().backward()
  assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())


def test_policy_runs_in_batched_env_rollout():
  env = envs.SerialEnv(2, _make_env)
  _, (actor, _) = _build(env)
  rollout = env.rollout(max_steps=4, policy=actor)
  assert rollout.batch_size == (2, 4)
  assert rollout["action"].shape == (2, 4, 2, 9)
