import pathlib

import hydra
from omegaconf import OmegaConf
import pytest
import torch

from rl import ppo_builder
from rl.envs import factory
from vision import model_registry

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "configs"
FRAMES_PER_BATCH = 16
_TIMM_WEIGHTS = (
  model_registry.DEFAULT_MODELS_ROOT / "resnet18_timm" / "model.safetensors"
)


def _compose(num_envs: int, device: str = "cpu", encoder: str = "RgbdCnnEncoder"):
  with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
    config = hydra.compose(
      config_name="joint_grasp_trained0",
      overrides=[
        f"ppo.device={device}",
        "ppo.total_frames=32",
        f"ppo.frames_per_batch={FRAMES_PER_BATCH}",
        "ppo.minibatch_size=8",
        "ppo.progress_bar=false",
        "ppo.model.hidden_dims=[32]",
        "ppo.model.encoder.cameras=[0,1]",
        "ppo.model.encoder.feature_dim=16",
        f"ppo.model.encoder._target_=rl.models.encoders.{encoder}",
        f"environment.num_envs={num_envs}",
        "environment.install_failure_handler=false",
      ],
    )
  OmegaConf.set_struct(config, False)
  config.environment.env = {
    "_target_": "rl.envs.fake_env.FakeShirtGraspEnv",
    "num_cameras": 3,
    "resolution": 64,
    "max_episode_steps": 6,
  }
  return config


def test_default_config_instantiates_sim_env_arguments():
  with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
    config = hydra.compose(config_name="joint_grasp_trained0")
  spawn_config = hydra.utils.instantiate(config.environment.env.spawn_config)
  assert spawn_config.num_robots == 2
  reward_fn = hydra.utils.instantiate(config.environment.env.reward_fn)
  assert reward_fn({}, None, {}) == (0.0, False)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_trainer_runs_and_saves(tmp_path, num_envs, device):
  if device == "cuda" and not torch.cuda.is_available():
    pytest.skip("no GPU")
  config = _compose(num_envs, device)
  env = factory.make_torchrl_env(config.environment, seed=0)
  trainer = ppo_builder.build_trainer(config.ppo, env, tmp_path)
  initial = {
    k: v.clone() for k, v in trainer.loss_module.actor_network.state_dict().items()
  }
  trainer.train()
  ppo_builder.save_policy(trainer, tmp_path)
  assert (tmp_path / "actor.pt").is_file() and (tmp_path / "critic.pt").is_file()
  updated = trainer.loss_module.actor_network.state_dict()
  assert any(not torch.equal(initial[k], updated[k]) for k in initial)


def test_every_epoch_runs_all_minibatches(tmp_path):
  config = _compose(num_envs=1)
  env = factory.make_torchrl_env(config.environment, seed=0)
  trainer = ppo_builder.build_trainer(config.ppo, env, tmp_path)
  updates = []
  trainer.register_op("post_loss", lambda batch: updates.append(1))
  trainer.train()
  num_batches = config.ppo.total_frames // config.ppo.frames_per_batch
  minibatches_per_epoch = config.ppo.frames_per_batch // config.ppo.minibatch_size
  assert len(updates) == num_batches * config.ppo.num_epochs * minibatches_per_epoch


@pytest.mark.skipif(not _TIMM_WEIGHTS.is_file(), reason="resnet18_timm not downloaded")
def test_trainer_runs_with_frozen_timm_encoder(tmp_path):
  config = _compose(num_envs=1, encoder="TimmFrozenEncoder")
  env = factory.make_torchrl_env(config.environment, seed=0)
  trainer = ppo_builder.build_trainer(config.ppo, env, tmp_path)
  trainer.train()
  backbone = trainer.loss_module.actor_network.module[0].module.trunk.encoder.backbone
  assert not any(p.requires_grad for p in backbone.parameters())
