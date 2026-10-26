import numpy as np
import pytest

from behaviors import base
from behaviors import vla_skill


def _observation():
  rgb = np.zeros((5, 4, 4, 3), np.uint8)
  for camera in range(5):
    rgb[camera] = camera
  return {
    "rgb": rgb,
    "joint_position": np.arange(26, dtype=np.float32).reshape(2, 13),
    "previous_action": np.zeros((2, 9), np.float32),
  }


class _FakePolicy:
  def __init__(self, chunk):
    self.chunk = np.asarray(chunk, np.float32)
    self.calls = []

  def __call__(self, robot_observation, instruction):
    self.calls.append((robot_observation, instruction))
    return self.chunk


def _make(policy, instructions=("up", "down"), config=vla_skill.Config()):
  skill = vla_skill.VlaSkill(policy, instructions, config=config)
  skill.reset(base.BehaviorState())
  return skill


def test_get_robot_observation_picks_each_robots_cameras_and_joints():
  robot_1 = vla_skill.get_robot_observation(_observation(), 1)
  assert robot_1.wrist_rgb[0, 0, 0] == 2
  assert robot_1.head_rgb[0, 0, 0] == 4
  assert robot_1.state[0] == 13
  assert robot_1.state.shape == (22,)


def test_chunk_is_replayed_before_requerying_the_policy():
  chunk = np.full((3, 9), 0.5)
  policy = _FakePolicy(chunk)
  skill = _make(policy)
  for _ in range(3):
    skill.step(_observation())
  assert len(policy.calls) == 2  # One query per robot.
  skill.step(_observation())
  assert len(policy.calls) == 4


def test_each_robot_gets_its_own_instruction():
  policy = _FakePolicy(np.zeros((1, 9)))
  _make(policy).step(_observation())
  assert [instruction for _, instruction in policy.calls] == ["up", "down"]


def test_robot_without_instruction_holds_and_is_never_queried():
  policy = _FakePolicy(np.full((1, 9), 0.5))
  action = _make(policy, instructions=("up", None)).step(_observation())
  assert len(policy.calls) == 1
  assert np.all(action[1] == 0)
  assert np.all(action[0] == 0.5)


def test_actions_are_clipped_to_valid_range():
  action = _make(_FakePolicy(np.full((1, 9), 5.0))).step(_observation())
  assert action.max() == 1.0


def test_done_after_idle_steps():
  config = vla_skill.Config(idle_steps=3)
  skill = _make(_FakePolicy(np.zeros((1, 9))), config=config)
  for _ in range(2):
    skill.step(_observation())
  assert not skill.done()
  skill.step(_observation())
  assert skill.done()


def test_done_after_max_steps_while_moving():
  skill = _make(_FakePolicy(np.full((1, 9), 0.5)), config=vla_skill.Config(max_steps=2))
  skill.step(_observation())
  assert not skill.done()
  skill.step(_observation())
  assert skill.done()


def test_wrong_instruction_count_raises():
  with pytest.raises(ValueError):
    vla_skill.VlaSkill(_FakePolicy(np.zeros((1, 9))), ("up",))
