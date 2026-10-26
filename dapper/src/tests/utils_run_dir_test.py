import pathlib

import utils_run_dir


def test_default_is_script_name_slash_uid():
  path = utils_run_dir.resolve_log_dir("ppo", root=pathlib.Path("/r"))
  assert path == pathlib.Path("/r/ppo") / utils_run_dir.process_uid()


def test_prefix_precedes_uid():
  path = utils_run_dir.resolve_log_dir("ppo", prefix="exp1", root=pathlib.Path("/r"))
  assert path.name == f"exp1-{utils_run_dir.process_uid()}"


def test_name_overrides_prefix_and_uid():
  path = utils_run_dir.resolve_log_dir("ppo", "exp1", "mine", pathlib.Path("/r"))
  assert path == pathlib.Path("/r/ppo/mine")


def test_rewrite_hydra_argv():
  argv = ["--log-dir-prefix", "a", "x=1", "--log-dir-name=b"]
  assert utils_run_dir.rewrite_hydra_argv(argv) == [
    "log_dir_prefix=a",
    "x=1",
    "log_dir_name=b",
  ]
