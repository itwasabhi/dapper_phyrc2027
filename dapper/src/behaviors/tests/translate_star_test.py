import numpy as np
import pytest

from behaviors import base
from behaviors import motion
from behaviors import translate_star


def _run(config):
  behavior = translate_star.TranslateStar(config=config)
  behavior.reset(base.BehaviorState())
  actions = []
  while not behavior.done():
    actions.append(behavior.step({}))
  return np.array(actions)


def test_each_spoke_goes_out_and_back_and_returns_to_start():
  config = translate_star.Config(leg_s=1.0, pause_s=0.5, num_directions=4)
  actions = _run(config)
  assert len(actions) == 4 * 2 * (20 + 10)
  assert actions[:, :, motion.FORWARD_COLUMN].sum() == pytest.approx(0.0, abs=1e-4)
  assert actions[:, :, motion.LEFT_COLUMN].sum() == pytest.approx(0.0, abs=1e-4)
  # Second spoke (90 degrees) first leg is purely leftward.
  second_spoke = actions[60]
  assert second_spoke[0, motion.FORWARD_COLUMN] == pytest.approx(0.0, abs=1e-6)
  assert second_spoke[0, motion.LEFT_COLUMN] == pytest.approx(1.0)


def test_both_robots_get_the_same_command_and_commands_stay_in_range():
  actions = _run(translate_star.Config(num_directions=8))
  assert np.array_equal(actions[:, 0], actions[:, 1])
  assert np.abs(actions).max() <= 1.0
