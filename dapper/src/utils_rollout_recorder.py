"""Writes DressingEnv rollouts in the same HDF5 layout as teleop recordings."""

import argparse
import json
import pathlib
from typing import Any, Mapping

import numpy as np

from Policy import contract as policy_contract
from Policy import training_dataset

TICKS_PER_STEP = 12
CONTRACT_PATH = "config/policy_interface.json"


def add_recording_flags(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--write-video",
    action="store_true",
    help="Render videos from the saved HDF5 once the rollout ends.",
  )
  parser.add_argument(
    "--video-stride",
    type=int,
    default=1,
    help="Use every Nth policy step in videos.",
  )


def _jsonable(value: Any) -> Any:
  return json.loads(json.dumps(value, default=str))


def open_writer(
  project_root: pathlib.Path,
  output_dir: pathlib.Path,
  metadata: Mapping[str, Any],
) -> training_dataset.DatasetWriter:
  """Creates `output_dir`, which must not exist yet."""
  contract = policy_contract.load_contract(project_root / CONTRACT_PATH)
  return training_dataset.DatasetWriter(output_dir, contract, _jsonable(metadata))


class RolloutRecorder:
  """Records policy-rate transitions; one `demo_N` per reset."""

  def __init__(self, writer: training_dataset.DatasetWriter):
    self._writer = writer
    self._observation = None
    self._tick = 0
    self.num_steps = 0

  @property
  def episode_active(self) -> bool:
    return self._observation is not None

  def start_episode(
    self, observation: Mapping[str, np.ndarray], info: Mapping[str, Any]
  ) -> None:
    if self.episode_active:
      raise RuntimeError("Call end_episode before starting another episode.")
    if "rgb" not in observation:
      raise ValueError("Recording requires an actor_rgbd environment.")
    robots = info["state"]["robots"]
    self._writer.start_episode(
      _jsonable(
        {
          "start_tick": 0,
          "episode_seed": info["episode_seed"],
          "human_spawn": info["human_spawn"],
          "garment_spawn": info["garment_spawn"],
          "joint_names": [[j["name"] for j in r["joints"]] for r in robots],
          "joint_units": [[j["unit"] for j in r["joints"]] for r in robots],
        }
      )
    )
    self._observation = _copy(observation)
    self._tick = 0

  def record_step(
    self, action: np.ndarray, next_observation: Mapping[str, np.ndarray]
  ) -> None:
    if not self.episode_active:
      raise RuntimeError("Call start_episode before record_step.")
    next_observation = _copy(next_observation)
    self._writer.append(
      self._observation,
      np.array(action, np.float32),
      next_observation,
      start_tick=self._tick,
      end_tick=self._tick + TICKS_PER_STEP,
      audit={},
    )
    self._observation = next_observation
    self._tick += TICKS_PER_STEP
    self.num_steps += 1

  def end_episode(self, reason: str) -> None:
    if not self.episode_active:
      return
    self._writer.finish_episode(reason)
    self._observation = None

  def close(self) -> None:
    if self.episode_active:
      self.end_episode("closed")
    self._writer.close(complete=True)


def _copy(observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
  return {key: np.array(value, copy=True) for key, value in observation.items()}
