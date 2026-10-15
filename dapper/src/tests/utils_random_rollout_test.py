import numpy as np
import pytest

import utils_random_rollout


def _rollout(num_steps=40, **kwargs):
  rng = np.random.default_rng(0)
  params = dict(hold_steps=10, action_scale=1.0, gripper_resample_prob=0.05)
  params.update(kwargs)
  actions, action = [], None
  for step in range(num_steps):
    action = utils_random_rollout.sample_action(rng, action, step, **params)
    actions.append(action)
  return np.stack(actions)


def test_actions_are_valid_for_contract():
  actions = _rollout()
  assert actions.shape == (40, 2, 9)
  assert actions.dtype == np.float32
  assert np.abs(actions).max() <= 1.0


def test_velocities_held_between_resamples():
  actions = _rollout()
  np.testing.assert_array_equal(actions[0, :, :8], actions[9, :, :8])
  assert not np.array_equal(actions[9, :, :8], actions[10, :, :8])


def test_gripper_does_not_toggle_every_step():
  gripper = _rollout(num_steps=200)[:, :, 8]
  assert (gripper[1:] != gripper[:-1]).mean() < 0.2


def test_action_scale_applies_to_velocities():
  actions = _rollout(action_scale=0.25)
  assert np.abs(actions[:, :, :8]).max() <= 0.25


def test_refuses_nonempty_dir(tmp_path):
  (tmp_path / "x").write_text("")
  with pytest.raises(FileExistsError):
    utils_random_rollout.prepare_output_dir(tmp_path, overwrite=False)


def test_overwrite_clears_dir_for_writer(tmp_path):
  (tmp_path / "x").write_text("")
  utils_random_rollout.prepare_output_dir(tmp_path, overwrite=True)
  assert not tmp_path.exists()
