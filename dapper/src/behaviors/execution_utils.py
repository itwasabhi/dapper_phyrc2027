"""Runs an ordered list of behaviors as a single solution policy."""

from behaviors import base


def create_initial_state() -> base.BehaviorState:
  return base.BehaviorState()


def execute_sequence(behaviors, index, observation, state):
  """Steps the active behavior; advances and resets the next one once it's done.

  Returns (action, next_index).
  """
  active = behaviors[index]
  action = active.step(observation)
  if active.done() and index < len(behaviors) - 1:
    index += 1
    behaviors[index].reset(state)
  return action, index


class SequencedPolicy:
  """Policy callable that runs `behaviors` in order via execute_sequence."""

  def __init__(self, behaviors):
    self._behaviors = behaviors
    self._index = 0
    self._state = None

  def reset(self):
    self._index = 0
    self._state = create_initial_state()
    self._behaviors[0].reset(self._state)

  def __call__(self, observation):
    action, self._index = execute_sequence(
      self._behaviors, self._index, observation, self._state
    )
    return action

  def close(self):
    for behavior in self._behaviors:
      behavior.close()
