import pytest
import torch

from rl.models import encoders
from vision import model_registry

NUM_CAMERAS = 5
RESOLUTION = 64
CAMERA_SHAPE = (NUM_CAMERAS, RESOLUTION, RESOLUTION)
BATCH_SHAPES = [(), (3,), (2, 4)]


def _images(batch_shape):
  shape = (*batch_shape, NUM_CAMERAS, RESOLUTION, RESOLUTION)
  rgb = torch.randint(0, 255, (*shape, 3), dtype=torch.uint8)
  depth = torch.rand(*shape, 1) * 6
  depth_valid = torch.rand(*shape, 1) > 0.2
  return rgb, depth, depth_valid


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_rgbd_cnn_preserves_batch_shape(batch_shape):
  encoder = encoders.RgbdCnnEncoder(CAMERA_SHAPE, cameras=[0, 2, 3], feature_dim=16)
  features = encoder(*_images(batch_shape))
  assert features.shape == (*batch_shape, 3 * 16)
  assert encoder.output_dim == 3 * 16


def test_rgbd_cnn_batched_matches_per_sample():
  encoder = encoders.RgbdCnnEncoder(CAMERA_SHAPE, cameras=[0, 1], feature_dim=8).eval()
  rgb, depth, valid = _images((2, 3))
  batched = encoder(rgb, depth, valid)
  single = encoder(rgb[1, 2], depth[1, 2], valid[1, 2])
  torch.testing.assert_close(batched[1, 2], single, atol=1e-5, rtol=1e-4)


def test_rgbd_cnn_masks_invalid_depth():
  encoder = encoders.RgbdCnnEncoder(CAMERA_SHAPE, cameras=[0], feature_dim=8).eval()
  rgb, depth, _ = _images(())
  invalid = torch.zeros_like(depth, dtype=torch.bool)
  torch.testing.assert_close(
    encoder(rgb, depth, invalid), encoder(rgb, depth * 0 + 3.0, invalid)
  )


def test_timm_encoder_requires_downloaded_weights(tmp_path):
  with pytest.raises(FileNotFoundError):
    encoders.TimmFrozenEncoder(CAMERA_SHAPE, cameras=[0], models_root=tmp_path)


_TIMM_WEIGHTS = (
  model_registry.DEFAULT_MODELS_ROOT / "resnet18_timm" / "model.safetensors"
)


@pytest.mark.skipif(not _TIMM_WEIGHTS.is_file(), reason="resnet18_timm not downloaded")
@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_timm_encoder_is_frozen_and_preserves_batch_shape(batch_shape):
  encoder = encoders.TimmFrozenEncoder(CAMERA_SHAPE, cameras=[0, 3], feature_dim=16)
  features = encoder(*_images(batch_shape))
  assert features.shape == (*batch_shape, 2 * 16)
  assert not any(p.requires_grad for p in encoder.backbone.parameters())
  assert all(p.requires_grad for p in encoder.project.parameters())
  assert not encoder.train().backbone.training
