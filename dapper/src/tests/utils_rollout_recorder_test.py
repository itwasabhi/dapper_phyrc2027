import json
import pathlib

import numpy as np
import pytest

from Policy import contract as policy_contract
from Policy import dataset_reader

import utils_rollout_recorder

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SIZE = 8


_LOAD_CONTRACT = policy_contract.load_contract


def _contract():
  contract = _LOAD_CONTRACT(PROJECT_ROOT / utils_rollout_recorder.CONTRACT_PATH)
  contract["default_dimensions"].update(width=SIZE, height=SIZE)
  return contract


def _observation(time_s, previous_action):
  contract = _contract()
  shapes = policy_contract.observation_shapes(
    contract, robots=2, joints=13, profile="actor_rgbd"
  )
  obs = {
    name: np.zeros(shape, contract["observations"][name]["dtype"])
    for name, shape in shapes.items()
  }
  obs["simulation_time_s"] = np.array(time_s, np.float64)
  obs["previous_action"] = previous_action.copy()
  obs["depth"][:] = 1.0
  return obs


def _info():
  joints = [{"name": f"j{i}", "unit": "rad"} for i in range(13)]
  return {
    "state": {"robots": [{"joints": joints}] * 2},
    "episode_seed": 3,
    "human_spawn": {"x": np.float32(1.0)},
    "garment_spawn": {},
  }


class FakeWriter:
  def __init__(self):
    self.calls = []

  def start_episode(self, metadata):
    self.calls.append(("start", metadata))

  def append(self, obs, action, next_obs, **kwargs):
    self.calls.append(("append", obs, action, next_obs, kwargs))

  def finish_episode(self, reason):
    self.calls.append(("finish", reason))

  def close(self, complete):
    self.calls.append(("close", complete))


def test_ticks_advance_by_twelve_and_observations_are_copied():
  writer = FakeWriter()
  recorder = utils_rollout_recorder.RolloutRecorder(writer)
  obs = _observation(0.0, np.zeros((2, 9), np.float32))
  recorder.start_episode(obs, _info())
  obs["joint_position"][:] = 5
  action = np.full((2, 9), 0.5, np.float32)
  for step in range(3):
    recorder.record_step(action, _observation(0.05 * (step + 1), action))
  appends = [call for call in writer.calls if call[0] == "append"]
  assert [c[4]["start_tick"] for c in appends] == [0, 12, 24]
  assert [c[4]["end_tick"] for c in appends] == [12, 24, 36]
  assert appends[0][1]["joint_position"].max() == 0
  assert json.dumps(writer.calls[0][1])


def test_requires_rgbd_observation():
  recorder = utils_rollout_recorder.RolloutRecorder(FakeWriter())
  obs = _observation(0.0, np.zeros((2, 9), np.float32))
  del obs["rgb"]
  with pytest.raises(ValueError):
    recorder.start_episode(obs, _info())


def test_record_step_needs_episode():
  recorder = utils_rollout_recorder.RolloutRecorder(FakeWriter())
  with pytest.raises(RuntimeError):
    recorder.record_step(np.zeros((2, 9), np.float32), {})


def test_round_trip_through_real_writer(tmp_path, monkeypatch):
  monkeypatch.setattr(
    policy_contract,
    "load_contract",
    lambda path: _contract(),
  )
  out = tmp_path / "run"
  recorder = utils_rollout_recorder.RolloutRecorder(
    utils_rollout_recorder.open_writer(PROJECT_ROOT, out, {"seed": 1})
  )
  recorder.start_episode(_observation(0.0, np.zeros((2, 9), np.float32)), _info())
  action = np.full((2, 9), 0.25, np.float32)
  for step in range(2):
    recorder.record_step(action, _observation(0.05 * (step + 1), action))
    action = -action
  recorder.close()
  dataset = dataset_reader.PolicyDataset(out / "policy.hdf5")
  assert len(dataset) == 2
