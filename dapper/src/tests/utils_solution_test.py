import pathlib

import numpy as np
import pytest

import utils_solution

SOLUTION_ALPHA = pathlib.Path(__file__).resolve().parents[1] / "solution-alpha"


def test_find_checkpoint_none_for_alpha():
  assert utils_solution.find_checkpoint(SOLUTION_ALPHA) is None


def test_validate_action_rejects_bad_input():
  with pytest.raises(ValueError):
    utils_solution.validate_action(np.zeros((2, 8)), (2, 9))
  with pytest.raises(ValueError):
    utils_solution.validate_action(np.full((2, 9), np.nan), (2, 9))
  clipped = utils_solution.validate_action(np.full((2, 9), 5.0), (2, 9))
  assert clipped.max() == 1.0
