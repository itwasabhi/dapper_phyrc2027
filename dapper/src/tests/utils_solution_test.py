import pathlib
import types

import numpy as np
import pytest

import utils_solution

SOLUTION_ALPHA = pathlib.Path(__file__).resolve().parents[1] / "solution-alpha"
ACTION_SPACE = types.SimpleNamespace(shape=(2, 9))


def _load():
  return utils_solution.load_policy(SOLUTION_ALPHA, None, None, ACTION_SPACE)


def test_solution_alpha_only_yaws():
  policy = _load()
  policy.reset()
  action = utils_solution.validate_action(policy({}), (2, 9))
  assert action.dtype == np.float32
  assert np.all(action[:, 2] > 0)
  assert np.abs(action[:, 2]).max() <= 0.2
  assert np.all(np.delete(action, 2, axis=1) == 0)


def test_find_checkpoint_none_for_alpha():
  assert utils_solution.find_checkpoint(SOLUTION_ALPHA) is None


def test_validate_action_rejects_bad_input():
  with pytest.raises(ValueError):
    utils_solution.validate_action(np.zeros((2, 8)), (2, 9))
  with pytest.raises(ValueError):
    utils_solution.validate_action(np.full((2, 9), np.nan), (2, 9))
  clipped = utils_solution.validate_action(np.full((2, 9), 5.0), (2, 9))
  assert clipped.max() == 1.0
