"""Builds the PPO actor and critic from environment specs.

Both networks take the raw observation tensors and accept any leading batch
shape; per-sample shapes come from the environment specs.
"""

import dataclasses
import math
from typing import Callable, Mapping, Sequence, Tuple

import torch
from torch import nn
from tensordict import nn as td_nn
from torchrl import envs
from torchrl import modules as rl_modules

from rl.models import batching

IMAGE_KEYS = ("rgb", "depth", "depth_valid")
EncoderFactory = Callable[..., nn.Module]


@dataclasses.dataclass(frozen=True)
class SampleShapes:
  """Per-sample (batch-free) shapes of the env keys the models read."""

  observations: Mapping[str, Tuple[int, ...]]
  action: Tuple[int, ...]


def sample_shapes_from_env(env: envs.EnvBase, keys: Sequence[str]) -> SampleShapes:
  batch_ndim = len(env.batch_size)
  observations = {
    key: tuple(env.observation_spec[key].shape[batch_ndim:]) for key in keys
  }
  return SampleShapes(observations, tuple(env.action_spec.shape[batch_ndim:]))


class _ObservationTrunk(nn.Module):
  """Image encoder plus proprioceptive MLP, concatenated into one feature vector."""

  def __init__(
    self,
    encoder: nn.Module,
    proprio_sample_ndims: Sequence[int],
    proprio_dim: int,
    hidden_dims: Sequence[int],
    feature_dim: int,
  ):
    super().__init__()
    self.encoder = encoder
    self.proprio_sample_ndims = tuple(proprio_sample_ndims)
    self.proprio_mlp = rl_modules.MLP(
      in_features=proprio_dim,
      out_features=feature_dim,
      num_cells=list(hidden_dims),
      activation_class=nn.ReLU,
      activate_last_layer=True,
    )
    self.output_dim = encoder.output_dim + feature_dim

  def forward(self, rgb, depth, depth_valid, *proprio) -> torch.Tensor:
    image_features = self.encoder(rgb, depth, depth_valid)
    flat_proprio = torch.cat(
      [
        batching.flatten_sample(value.to(image_features.dtype), sample_ndim)
        for value, sample_ndim in zip(proprio, self.proprio_sample_ndims)
      ],
      dim=-1,
    )
    return torch.cat([image_features, self.proprio_mlp(flat_proprio)], dim=-1)


class _PolicyNet(nn.Module):
  """Trunk to (loc, scale), each shaped `batch + action_shape`."""

  def __init__(
    self, trunk: _ObservationTrunk, hidden_dims: Sequence[int], action_shape
  ):
    super().__init__()
    self.trunk = trunk
    self.action_shape = tuple(action_shape)
    self.head = nn.Sequential(
      rl_modules.MLP(
        in_features=trunk.output_dim,
        out_features=2 * math.prod(self.action_shape),
        num_cells=list(hidden_dims),
        activation_class=nn.ReLU,
      ),
      td_nn.NormalParamExtractor(),
    )

  def forward(self, *observations):
    loc, scale = self.head(self.trunk(*observations))
    batch_shape = loc.shape[:-1]
    return loc.reshape(*batch_shape, *self.action_shape), scale.reshape(
      *batch_shape, *self.action_shape
    )


class _ValueNet(nn.Module):
  def __init__(self, trunk: _ObservationTrunk, hidden_dims: Sequence[int]):
    super().__init__()
    self.trunk = trunk
    self.head = rl_modules.MLP(
      in_features=trunk.output_dim,
      out_features=1,
      num_cells=list(hidden_dims),
      activation_class=nn.ReLU,
    )

  def forward(self, *observations):
    return self.head(self.trunk(*observations))


def _make_trunk(
  encoder_factory: EncoderFactory,
  shapes: SampleShapes,
  proprio_keys: Sequence[str],
  hidden_dims: Sequence[int],
  feature_dim: int,
) -> _ObservationTrunk:
  proprio_dim = sum(math.prod(shapes.observations[key]) for key in proprio_keys)
  return _ObservationTrunk(
    encoder=encoder_factory(camera_shape=_camera_shape(shapes)),
    proprio_sample_ndims=[len(shapes.observations[key]) for key in proprio_keys],
    proprio_dim=proprio_dim,
    hidden_dims=hidden_dims,
    feature_dim=feature_dim,
  )


def _camera_shape(shapes: SampleShapes) -> Tuple[int, int, int]:
  num_cameras, height, width, _ = shapes.observations["rgb"]
  return num_cameras, height, width


def build_actor_critic(
  encoder_factory: EncoderFactory,
  shapes: SampleShapes,
  action_spec,
  proprio_keys: Sequence[str],
  hidden_dims: Sequence[int] = (256, 256),
  proprio_feature_dim: int = 64,
) -> Tuple[td_nn.TensorDictModuleBase, td_nn.TensorDictModuleBase]:
  """Returns (actor, critic) with separate encoders.

  `encoder_factory(camera_shape=(cameras, height, width))` must return a module
  with an `output_dim` attribute.
  """
  in_keys = [*IMAGE_KEYS, *proprio_keys]
  actor_net = _PolicyNet(
    _make_trunk(
      encoder_factory, shapes, proprio_keys, hidden_dims, proprio_feature_dim
    ),
    hidden_dims,
    shapes.action,
  )
  actor = rl_modules.ProbabilisticActor(
    module=td_nn.TensorDictModule(
      actor_net, in_keys=in_keys, out_keys=["loc", "scale"]
    ),
    in_keys=["loc", "scale"],
    spec=action_spec,
    distribution_class=rl_modules.TanhNormal,
    distribution_kwargs={"low": -1.0, "high": 1.0, "event_dims": len(shapes.action)},
    return_log_prob=True,
  )
  critic = rl_modules.ValueOperator(
    _ValueNet(
      _make_trunk(
        encoder_factory, shapes, proprio_keys, hidden_dims, proprio_feature_dim
      ),
      hidden_dims,
    ),
    in_keys=in_keys,
  )
  return actor, critic
