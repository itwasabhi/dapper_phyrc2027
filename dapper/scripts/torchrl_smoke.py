"""Benchmarks DressingEnv frame-collection throughput with a torchrl Collector.

Runs a random policy (torchrl's default when `policy=None`) through a Collector
for several iterations and reports frames_per_second.

Run (single env, no subprocess):
  dapper/run_sim.sh python /project/dapper/scripts/torchrl_smoke.py

Run (4 parallel envs, one container, 4 worker subprocesses):
  dapper/run_sim.sh python /project/dapper/scripts/torchrl_smoke.py --num-workers 4
"""

import argparse
import functools
import time

from torchrl import collectors, envs

import utils_torchrl_env


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--num-workers", type=int, default=1, help="Parallel DressingEnv subprocesses."
  )
  parser.add_argument("--frames-per-batch", type=int, default=24)
  parser.add_argument("--iterations", type=int, default=50)
  parser.add_argument("--max-episode-steps", type=int, default=400)
  args = parser.parse_args()
  if args.num_workers < 1:
    parser.error("--num-workers must be >= 1")
  if args.frames_per_batch < 1:
    parser.error("--frames-per-batch must be >= 1")
  if args.iterations < 1:
    parser.error("--iterations must be >= 1")
  if args.max_episode_steps < 1:
    parser.error("--max-episode-steps must be >= 1")
  return args


def build_env(seed: int, max_episode_steps: int) -> envs.TransformedEnv:
  """Constructs one DressingEnv and wraps it; safe to call in a spawned subprocess."""
  import policy_cli  # pylint: disable=import-outside-toplevel

  policy_cli.install_failure_handler()
  from Policy import environment  # pylint: disable=import-outside-toplevel

  gym_env = environment.DressingEnv(
    profile="actor_rgbd",
    max_episode_steps=max_episode_steps,
  )
  env = utils_torchrl_env.make_env(gym_env)
  env.set_seed(seed)
  return env


def main() -> None:
  args = parse_args()

  import policy_cli  # pylint: disable=import-outside-toplevel

  policy_cli.install_failure_handler()

  if args.num_workers == 1:
    env = build_env(args.seed, args.max_episode_steps)
    envs.check_env_specs(env)
    create_env_fn = env
  else:
    worker_fns = [
      functools.partial(
        build_env, seed=args.seed + i, max_episode_steps=args.max_episode_steps
      )
      for i in range(args.num_workers)
    ]
    create_env_fn = envs.ParallelEnv(
      args.num_workers, worker_fns, mp_start_method="spawn"
    )

  # total_frames=-1 (collect indefinitely): on the batch that satisfies a finite
  # total_frames, Collector closes the env before yielding it, and close() can
  # fast-exit the process (see DressingEnv.close()) -- losing that last batch and
  # every print below it. We stop ourselves after args.iterations instead, so
  # `collector.shutdown()` only ever runs under our own control, after printing.
  collector = collectors.Collector(
    create_env_fn,
    policy=None,
    frames_per_batch=args.frames_per_batch,
    total_frames=-1,
  )

  iteration_seconds = []
  previous = time.perf_counter()
  for batch_index, _ in enumerate(collector):
    now = time.perf_counter()
    iteration_seconds.append(now - previous)
    previous = now
    if batch_index + 1 >= args.iterations:
      break

  warmup_seconds = iteration_seconds[0]
  steady_seconds = sum(iteration_seconds[1:])
  steady_frames = args.frames_per_batch * (len(iteration_seconds) - 1)
  warmup_frames_per_second = args.frames_per_batch / warmup_seconds

  print(
    f"num_workers={args.num_workers} frames_per_batch={args.frames_per_batch} "
    f"iterations={len(iteration_seconds)}"
  )
  print(
    f"warmup : {args.frames_per_batch} frames in {warmup_seconds:.2f}s "
    f"(frames_per_second={warmup_frames_per_second:.1f}, includes reset/startup)"
  )
  if steady_seconds > 0:
    frames_per_second = steady_frames / steady_seconds
    frames_per_second_per_env = frames_per_second / args.num_workers
    print(
      f"steady : {steady_frames} frames in {steady_seconds:.2f}s "
      f"(frames_per_second={frames_per_second:.1f}, "
      f"frames_per_second_per_env={frames_per_second_per_env:.1f})"
    )
  else:
    print("steady : not available (need --iterations >= 2)")

  # shutdown() can fast-exit the process; everything is printed above.
  collector.shutdown()


if __name__ == "__main__":
  main()
