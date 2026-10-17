"""Records a few sampled envs out of collector batches as teleop-style policy.hdf5 files."""

import dataclasses
import pathlib
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import tensordict

import utils_video


class Recorder(Protocol):
  episode_active: bool

  def start_episode(
    self, observation: Mapping[str, np.ndarray], episode_metadata: Mapping[str, object]
  ) -> None: ...

  def record_step(
    self, action: np.ndarray, next_observation: Mapping[str, np.ndarray]
  ) -> None: ...

  def end_episode(self, reason: str) -> None: ...

  def close(self) -> None: ...


def sample_indices(num_envs: int, num_samples: int) -> list[int]:
  """Evenly spaced env indices, at most `num_envs` of them."""
  count = min(num_samples, num_envs)
  return sorted({int(i) for i in np.linspace(0, num_envs - 1, count).round()})


@dataclasses.dataclass
class _Sample:
  recorder: Recorder
  output_dir: pathlib.Path
  episodes_left: int


class BatchRecorder:
  """Feeds `[num_envs, num_steps]` (or `[num_steps]`) collector batches to per-env recorders.

  Once every sampled env has recorded `max_episodes` episodes, `add_batch` is a no-op.
  """

  def __init__(
    self,
    num_envs: int,
    num_samples: int,
    max_episodes: int,
    observation_shapes: Mapping[str, Sequence[int]],
    make_recorder: Callable[[int], tuple[Recorder, pathlib.Path]],
    episode_metadata: Callable[[int], Mapping[str, object]],
  ):
    self._observation_shapes = {k: tuple(v) for k, v in observation_shapes.items()}
    self._episode_metadata = episode_metadata
    self._samples = {}
    for index in sample_indices(num_envs, num_samples):
      recorder, output_dir = make_recorder(index)
      self._samples[index] = _Sample(recorder, output_dir, max_episodes)

  @property
  def output_dirs(self) -> list[pathlib.Path]:
    return [sample.output_dir for sample in self._samples.values()]

  @property
  def finished(self) -> bool:
    return all(sample.episodes_left == 0 for sample in self._samples.values())

  def add_batch(self, batch: tensordict.TensorDictBase) -> None:
    if self.finished:
      return
    if batch.batch_dims == 1:
      batch = batch.unsqueeze(0)
    active = [i for i, s in self._samples.items() if s.episodes_left > 0]
    selected = batch[active].cpu()
    for row, index in enumerate(active):
      self._add_rollout(index, selected[row])

  def _add_rollout(self, index: int, rollout: tensordict.TensorDictBase) -> None:
    sample = self._samples[index]
    num_steps = rollout.batch_size[0]

    def to_numpy(source, key):
      # TorchRL gives scalar observations a trailing singleton dim.
      shape = self._observation_shapes[key]
      return source[key].numpy().reshape(num_steps, *shape)

    observations = {key: to_numpy(rollout, key) for key in self._observation_shapes}
    next_observations = {
      key: to_numpy(rollout["next"], key) for key in self._observation_shapes
    }
    actions = rollout["action"].numpy()
    done = rollout["next", "done"].numpy().reshape(len(actions))
    terminated = rollout["next", "terminated"].numpy().reshape(len(actions))
    for step in range(len(actions)):
      if sample.episodes_left == 0:
        return
      if not sample.recorder.episode_active:
        sample.recorder.start_episode(
          {key: value[step] for key, value in observations.items()},
          self._episode_metadata(index),
        )
      sample.recorder.record_step(
        actions[step], {key: value[step] for key, value in next_observations.items()}
      )
      if done[step]:
        sample.recorder.end_episode("terminated" if terminated[step] else "truncated")
        sample.episodes_left -= 1

  def close(self) -> None:
    for sample in self._samples.values():
      sample.recorder.close()

  def write_videos(self, stride: int = 1) -> list[pathlib.Path]:
    written = []
    for sample in self._samples.values():
      written += utils_video.write_videos(
        sample.output_dir / "policy.hdf5", sample.output_dir / "videos", stride=stride
      )
    return written
