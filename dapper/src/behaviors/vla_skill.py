"""vla_skill: runs a language-conditioned VLA on each robot, replaying its action chunks."""

import collections
import dataclasses
from typing import Callable, Sequence

import numpy as np

from behaviors import base

WRIST_CAMERA_OFFSET = 1
HEAD_CAMERA_OFFSET = 3
NUM_ROBOTS = 2
ACTION_DIM = 9


@dataclasses.dataclass(frozen=True)
class RobotObservation:
  """One robot's slice of the public observation, as the VLA sees it."""

  head_rgb: np.ndarray
  wrist_rgb: np.ndarray
  state: np.ndarray


# (robot observation, instruction) -> (num_actions, 9) actions in [-1, 1].
PolicyFn = Callable[[RobotObservation, str], np.ndarray]


@dataclasses.dataclass(frozen=True)
class Config:
  max_steps: int = 400
  idle_threshold: float = 0.02
  idle_steps: int = 20


def get_robot_observation(observation, robot: int) -> RobotObservation:
  rgb = np.asarray(observation["rgb"])
  state = np.concatenate(
    [
      np.asarray(observation["joint_position"])[robot],
      np.asarray(observation["previous_action"])[robot],
    ]
  ).astype(np.float32)
  return RobotObservation(
    head_rgb=rgb[HEAD_CAMERA_OFFSET + robot],
    wrist_rgb=rgb[WRIST_CAMERA_OFFSET + robot],
    state=state,
  )


class VlaSkill:
  """Per robot, queues the VLA's action chunk and re-queries once it runs out.

  A robot whose instruction is None holds still. The skill is done after `max_steps`, or once
  every instructed robot has commanded near-zero motion for `idle_steps` in a row.
  """

  def __init__(
    self,
    policy_fn: PolicyFn,
    instructions: Sequence[str | None],
    cfg: base.BehaviorCfg = base.BehaviorCfg(),
    config: Config = Config(),
  ):
    if len(instructions) != NUM_ROBOTS:
      raise ValueError(f"Expected {NUM_ROBOTS} instructions, got {len(instructions)}.")
    del cfg
    self._policy_fn = policy_fn
    self._instructions = tuple(instructions)
    self._config = config
    self._queues = [collections.deque() for _ in range(NUM_ROBOTS)]
    self._step = 0
    self._idle_count = 0

  def reset(self, state: base.BehaviorState) -> None:
    del state
    for queue in self._queues:
      queue.clear()
    self._step = 0
    self._idle_count = 0

  def step(self, observation) -> np.ndarray:
    action = np.zeros((NUM_ROBOTS, ACTION_DIM), dtype=np.float32)
    for robot, instruction in enumerate(self._instructions):
      if instruction is None:
        continue
      queue = self._queues[robot]
      if not queue:
        chunk = self._policy_fn(get_robot_observation(observation, robot), instruction)
        queue.extend(np.asarray(chunk, dtype=np.float32).reshape(-1, ACTION_DIM))
      action[robot] = queue.popleft()
    action = np.clip(action, -1.0, 1.0)

    self._step += 1
    moving = np.abs(action[:, :-1]).max() > self._config.idle_threshold
    self._idle_count = 0 if moving else self._idle_count + 1
    return action

  def done(self) -> bool:
    return (
      self._step >= self._config.max_steps
      or self._idle_count >= self._config.idle_steps
    )

  def close(self) -> None:
    pass
