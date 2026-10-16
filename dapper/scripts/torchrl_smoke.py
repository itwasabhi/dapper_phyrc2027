"""Checks DressingEnv through torchrl's GymWrapper and times it against the raw env.

Run: dapper/run_sim.sh python /project/dapper/scripts/torchrl_smoke.py --steps 20
"""

import argparse
import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path("/project")
if not (PROJECT_ROOT / "src/PhyRC_Sim").is_dir():
  PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src/PhyRC_Sim"))
sys.path.insert(0, str(PROJECT_ROOT / "dapper/src"))
sys.path.insert(0, "/scripts")  # Upstream scripts dir in the container.

import numpy as np  # noqa: E402
from torchrl import envs  # noqa: E402

import utils_random_rollout  # noqa: E402
import utils_torchrl_env  # noqa: E402


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--steps", type=int, default=20, help="20 Hz steps.")
  parser.add_argument("--resolution", type=int, nargs=2, metavar=("W", "H"))
  args = parser.parse_args()
  if args.steps < 1:
    parser.error("--steps must be >= 1")
  return args


def time_raw(env, seed: int, steps: int) -> tuple[float, float]:
  """Returns (reset seconds, mean step seconds) for the unwrapped env."""
  rng = np.random.default_rng(seed)
  start = time.perf_counter()
  env.reset(seed=seed)
  reset_s = time.perf_counter() - start
  action = None
  start = time.perf_counter()
  for step in range(steps):
    action = utils_random_rollout.sample_action(rng, action, step, 10, 1.0, 0.05)
    _, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
      break
  return reset_s, (time.perf_counter() - start) / (step + 1)


def time_wrapped(env, seed: int, steps: int):
  """Returns (reset seconds, mean step seconds, rollout) for the torchrl env."""
  env.set_seed(seed)
  start = time.perf_counter()
  env.reset()
  reset_s = time.perf_counter() - start
  start = time.perf_counter()
  rollout = env.rollout(max_steps=steps, break_when_any_done=True)
  return reset_s, (time.perf_counter() - start) / rollout.shape[0], rollout


def main() -> None:
  args = parse_args()

  import policy_cli  # pylint: disable=import-outside-toplevel

  policy_cli.install_failure_handler()
  from Policy import environment  # pylint: disable=import-outside-toplevel

  gym_env = environment.DressingEnv(
    profile="actor_rgbd",
    resolution=tuple(args.resolution) if args.resolution else None,
    max_episode_steps=args.steps,
  )
  raw_reset_s, raw_step_s = time_raw(gym_env, args.seed, args.steps)

  env = utils_torchrl_env.make_env(gym_env)
  envs.check_env_specs(env)
  wrapped_reset_s, wrapped_step_s, rollout = time_wrapped(env, args.seed, args.steps)

  print(f"rollout: {rollout.shape[0]} steps")
  for key, value in rollout.items(include_nested=True, leaves_only=True):
    print(f"  {key}: {tuple(value.shape)} {value.dtype}")
  print(f"final done={bool(rollout['next', 'done'][-1])}")
  print(f"raw     reset {raw_reset_s:.2f}s  step {raw_step_s * 1e3:.1f}ms")
  print(f"wrapped reset {wrapped_reset_s:.2f}s  step {wrapped_step_s * 1e3:.1f}ms")
  # close() can fast-exit the process; everything is printed above.
  gym_env.close()


if __name__ == "__main__":
  main()
