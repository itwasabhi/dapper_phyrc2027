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
