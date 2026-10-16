from behaviors import base
from behaviors import execution_utils


class _FakeBehavior:
  def __init__(self, name, done_after):
    self.name = name
    self._done_after = done_after
    self._step = 0
    self.reset_calls = 0

  def reset(self, state):
    self.reset_calls += 1
    self._step = 0

  def step(self, observation):
    self._step += 1
    return self.name

  def done(self):
    return self._step >= self._done_after

  def close(self):
    pass


def test_execute_sequence_advances_and_resets_the_next_behavior():
  first, second = (
    _FakeBehavior("first", done_after=1),
    _FakeBehavior("second", done_after=5),
  )
  state = base.BehaviorState()

  action, index = execution_utils.execute_sequence([first, second], 0, {}, state)
  assert action == "first"
  assert index == 1
  assert second.reset_calls == 1


def test_execute_sequence_stays_on_the_last_behavior_once_done():
  only = _FakeBehavior("only", done_after=1)
  action, index = execution_utils.execute_sequence([only], 0, {}, base.BehaviorState())
  assert action == "only"
  assert index == 0
