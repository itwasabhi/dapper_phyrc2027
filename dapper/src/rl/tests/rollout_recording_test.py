import pathlib

import numpy as np
import tensordict
import torch

from rl import rollout_recording

SHAPES = {"joint_position": ()}


class FakeRecorder:
  def __init__(self):
    self.episode_active = False
    self.starts = []
    self.steps = []
    self.ends = []

  def start_episode(self, observation, episode_metadata):
    self.episode_active = True
    self.starts.append((observation["joint_position"].copy(), dict(episode_metadata)))

  def record_step(self, action, next_observation):
    self.steps.append((action.copy(), next_observation["joint_position"].copy()))

  def end_episode(self, reason):
    self.episode_active = False
    self.ends.append(reason)

  def close(self):
    pass


def _batch(num_envs, num_steps, done_at=None, terminated=False):
  # joint_position encodes env * 100 + step so rows are traceable.
  step_ids = torch.arange(num_steps, dtype=torch.float32)
  env_ids = torch.arange(num_envs, dtype=torch.float32)
  position = (env_ids[:, None] * 100 + step_ids[None])[..., None]
  done = torch.zeros(num_envs, num_steps, 1, dtype=torch.bool)
  if done_at is not None:
    done[:, done_at] = True  # int or list of step indices
  return tensordict.TensorDict(
    {
      "joint_position": position,
      "action": position * 0.5,
      "next": {
        "joint_position": position + 1,
        "done": done,
        "terminated": done & terminated,
      },
    },
    batch_size=[num_envs, num_steps],
  )


def _make(num_envs, num_samples, max_episodes):
  recorders = {}

  def make_recorder(index):
    recorders[index] = FakeRecorder()
    return recorders[index], pathlib.Path(f"env_{index}")

  recorder = rollout_recording.BatchRecorder(
    num_envs,
    num_samples,
    max_episodes,
    SHAPES,
    make_recorder,
    lambda index: {"env": index},
  )
  return recorder, recorders


def test_sample_indices_are_evenly_spaced_and_capped():
  assert rollout_recording.sample_indices(16, 5) == [0, 4, 8, 11, 15]
  assert rollout_recording.sample_indices(3, 5) == [0, 1, 2]
  assert rollout_recording.sample_indices(1, 5) == [0]


def test_only_sampled_envs_are_recorded_and_episodes_split_on_done():
  recorder, recorders = _make(num_envs=16, num_samples=5, max_episodes=2)
  recorder.add_batch(_batch(16, 6, done_at=[2, 5], terminated=True))
  assert sorted(recorders) == [0, 4, 8, 11, 15]
  fake = recorders[4]
  assert fake.ends == ["terminated", "terminated"]
  assert len(fake.steps) == 6
  # The second episode starts from the root observation after the done step.
  assert [float(start[0]) for start in fake.starts] == [400.0, 403.0]
  assert fake.starts[0][1] == {"env": 4}


def test_recording_stops_after_max_episodes_and_ignores_later_batches():
  recorder, recorders = _make(num_envs=2, num_samples=5, max_episodes=1)
  recorder.add_batch(_batch(2, 5, done_at=1))
  assert recorder.finished
  steps_before = len(recorders[0].steps)
  recorder.add_batch(_batch(2, 5))
  assert len(recorders[0].steps) == steps_before == 2
  assert recorders[0].ends == ["truncated"]


def test_episode_continues_across_batches_and_single_env_batch_dims():
  recorder, recorders = _make(num_envs=1, num_samples=5, max_episodes=1)
  recorder.add_batch(_batch(1, 3)[0])
  recorder.add_batch(_batch(1, 3, done_at=0)[0])
  fake = recorders[0]
  assert len(fake.starts) == 1
  assert len(fake.steps) == 4
  assert np.isclose(fake.steps[-1][1], 1.0)
