"""Helpers for modules that accept any number of leading batch dimensions."""

from typing import Sequence, Tuple

import torch


def fold_batch(
  tensor: torch.Tensor, sample_ndim: int
) -> Tuple[torch.Tensor, torch.Size]:
  """Flattens all leading batch dims into one; `()` batches become size 1."""
  split = tensor.ndim - sample_ndim
  return tensor.reshape(-1, *tensor.shape[split:]), tensor.shape[:split]


def unfold_batch(tensor: torch.Tensor, batch_shape: Sequence[int]) -> torch.Tensor:
  return tensor.reshape(*batch_shape, *tensor.shape[1:])


def flatten_sample(tensor: torch.Tensor, sample_ndim: int) -> torch.Tensor:
  """Flattens the trailing `sample_ndim` dims into one, keeping every batch dim."""
  split = tensor.ndim - sample_ndim
  return tensor.reshape(*tensor.shape[:split], -1)
