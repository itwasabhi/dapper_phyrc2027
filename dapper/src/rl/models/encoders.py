"""Image encoders mapping (rgb, depth, depth_valid) camera stacks to a feature vector.

Inputs have shape `batch + (cameras, height, width, channels)` for any `batch`;
the output has shape `batch + (num_selected_cameras * feature_dim,)`.
"""

import pathlib
from typing import Sequence, Tuple

from safetensors import torch as safetensors_torch
import timm
import torch
from torch import nn
import torch.nn.functional as torch_functional
from torchrl import modules as rl_modules

from rl.models import batching
from vision import model_registry

_CAMERA_SAMPLE_NDIM = 4
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _select_cameras(images: torch.Tensor, cameras: Sequence[int]) -> torch.Tensor:
  index = torch.as_tensor(cameras, device=images.device)
  return images.index_select(images.ndim - _CAMERA_SAMPLE_NDIM, index)


def _to_channels_first(images: torch.Tensor) -> torch.Tensor:
  """(batch, cameras, H, W, C) -> (batch * cameras, C, H, W)."""
  num_batch, num_cameras, height, width, channels = images.shape
  return images.reshape(num_batch * num_cameras, height, width, channels).permute(
    0, 3, 1, 2
  )


class RgbdCnnEncoder(nn.Module):
  """Nature-CNN style encoder over RGB plus normalised depth, shared across cameras."""

  def __init__(
    self,
    camera_shape: Tuple[int, int, int],
    cameras: Sequence[int],
    feature_dim: int = 128,
    max_depth_m: float = 4.0,
    num_cells: Sequence[int] = (32, 64, 64),
    kernel_sizes: Sequence[int] = (8, 4, 3),
    strides: Sequence[int] = (4, 2, 1),
  ):
    super().__init__()
    _, height, width = camera_shape
    self.cameras = tuple(cameras)
    self.max_depth_m = max_depth_m
    self.output_dim = len(self.cameras) * feature_dim
    self.conv = rl_modules.ConvNet(
      in_features=4,
      num_cells=list(num_cells),
      kernel_sizes=list(kernel_sizes),
      strides=list(strides),
      activation_class=nn.ReLU,
    )
    with torch.no_grad():
      flat_dim = self.conv(torch.zeros(1, 4, height, width)).shape[-1]
    self.project = nn.Sequential(nn.Linear(flat_dim, feature_dim), nn.ReLU())

  def forward(
    self, rgb: torch.Tensor, depth: torch.Tensor, depth_valid: torch.Tensor
  ) -> torch.Tensor:
    rgb, batch_shape = batching.fold_batch(
      _select_cameras(rgb, self.cameras), _CAMERA_SAMPLE_NDIM
    )
    depth, _ = batching.fold_batch(
      _select_cameras(depth, self.cameras), _CAMERA_SAMPLE_NDIM
    )
    valid, _ = batching.fold_batch(
      _select_cameras(depth_valid, self.cameras), _CAMERA_SAMPLE_NDIM
    )
    normalised_depth = (
      depth.clamp(0.0, self.max_depth_m) / self.max_depth_m
    ) * valid.to(depth.dtype)
    images = torch.cat([rgb.float() / 255.0, normalised_depth], dim=-1)
    features = self.project(self.conv(_to_channels_first(images)))
    features = features.reshape(rgb.shape[0], -1)
    return batching.unfold_batch(features, batch_shape)


class TimmFrozenEncoder(nn.Module):
  """Frozen timm backbone on RGB with a trainable per-camera projection."""

  def __init__(
    self,
    camera_shape: Tuple[int, int, int],
    cameras: Sequence[int],
    feature_dim: int = 128,
    model_name: str = "resnet18",
    registry_name: str = "resnet18_timm",
    input_size: int = 224,
    models_root: pathlib.Path = model_registry.DEFAULT_MODELS_ROOT,
  ):
    super().__init__()
    del camera_shape
    self.cameras = tuple(cameras)
    self.input_size = input_size
    self.output_dim = len(self.cameras) * feature_dim
    self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    weights_path = (
      model_registry.local_path(registry_name, models_root) / "model.safetensors"
    )
    load_result = self.backbone.load_state_dict(
      safetensors_torch.load_file(weights_path), strict=False
    )
    if load_result.missing_keys:
      raise ValueError(
        f"{weights_path} is missing backbone weights: {load_result.missing_keys}"
      )
    self.backbone.requires_grad_(False)
    self.backbone.eval()
    self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
    self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
    self.project = nn.Sequential(
      nn.Linear(self.backbone.num_features, feature_dim), nn.ReLU()
    )

  def train(self, mode: bool = True):
    super().train(mode)
    self.backbone.eval()
    return self

  def forward(
    self, rgb: torch.Tensor, depth: torch.Tensor, depth_valid: torch.Tensor
  ) -> torch.Tensor:
    del depth, depth_valid
    rgb, batch_shape = batching.fold_batch(
      _select_cameras(rgb, self.cameras), _CAMERA_SAMPLE_NDIM
    )
    images = _to_channels_first(rgb.float() / 255.0)
    images = torch_functional.interpolate(
      images,
      size=(self.input_size, self.input_size),
      mode="bilinear",
      align_corners=False,
    )
    with torch.no_grad():
      # GAE calls the value net under vmap, which rejects non-contiguous conv inputs.
      pooled = self.backbone(((images - self.mean) / self.std).contiguous())
    features = self.project(pooled).reshape(rgb.shape[0], -1)
    return batching.unfold_batch(features, batch_shape)
