"""Benchmarks (and optionally records) collection from any `environment` config group entry.

Runs a random policy through a Collector and reports frames_per_second; with
`record.enabled=true` it also saves a few sampled envs as policy.hdf5 (+ videos).

Run: dapper/run_sim.sh python /project/dapper/src/rl/scripts/torchrl_smoke.py environment=<name> [overrides]
Example: ... torchrl_smoke.py environment=dressing environment.num_envs=4 smoke.iterations=20
"""

import pathlib
import time

import hydra
from hydra.core import hydra_config
from torchrl import collectors
from torchrl import envs

from Policy import contract as policy_contract

import utils_rollout_recorder
import utils_run_dir
from rl import rollout_recording
from rl.envs import factory

_PROJECT_ROOT = pathlib.Path("/project")


def _make_batch_recorder(
  config, env: envs.EnvBase, run_dir: pathlib.Path
) -> rollout_recording.BatchRecorder:
  rgb_shape = env.observation_spec["rgb"].shape
  resolution = (rgb_shape[-2], rgb_shape[-3])
  robots, joints = env.observation_spec["joint_position"].shape[-2:]
  contract = policy_contract.load_contract(
    _PROJECT_ROOT / utils_rollout_recorder.CONTRACT_PATH
  )
  contract["default_dimensions"].update(width=resolution[0], height=resolution[1])
  # Contract keys only: env transforms add keys such as step_count that the writer rejects.
  observation_shapes = policy_contract.observation_shapes(
    contract, robots=robots, joints=joints, profile="actor_rgbd"
  )

  def make_recorder(index: int):
    output_dir = run_dir / f"env_{index}"
    writer = utils_rollout_recorder.open_writer(
      _PROJECT_ROOT, output_dir, {"env_index": index, "seed": config.seed}, resolution
    )
    return utils_rollout_recorder.RolloutRecorder(writer), output_dir

  return rollout_recording.BatchRecorder(
    num_envs=env.batch_size.numel(),
    num_samples=config.record.num_samples,
    max_episodes=config.record.max_episodes_per_sample,
    observation_shapes=observation_shapes,
    make_recorder=make_recorder,
    episode_metadata=lambda index: {"env_index": index, "start_tick": 0},
  )


@hydra.main(config_path="../configs", config_name="torchrl_smoke", version_base="1.3")
def main(config) -> None:
  run_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)
  env = factory.make_torchrl_env(config.environment, config.seed)
  num_envs = env.batch_size.numel()
  envs.check_env_specs(env)
  batch_recorder = (
    _make_batch_recorder(config, env, run_dir) if config.record.enabled else None
  )

  # total_frames=-1 (collect indefinitely): on the batch that satisfies a finite
  # total_frames, Collector closes the env before yielding it, and close() can
  # fast-exit the process -- losing that last batch and every print below it. We stop
  # ourselves after `iterations` instead, so `collector.shutdown()` only ever runs
  # under our own control, after printing.
  frames_per_batch = config.smoke.frames_per_batch
  collector = collectors.Collector(
    env, policy=None, frames_per_batch=frames_per_batch, total_frames=-1
  )

  iteration_seconds = []
  previous = time.perf_counter()
  for batch_index, batch in enumerate(collector):
    now = time.perf_counter()
    iteration_seconds.append(now - previous)
    if batch_recorder is not None:
      batch_recorder.add_batch(batch)
    previous = time.perf_counter()
    if batch_index + 1 >= config.smoke.iterations:
      break

  warmup_seconds = iteration_seconds[0]
  steady_seconds = sum(iteration_seconds[1:])
  steady_frames = frames_per_batch * (len(iteration_seconds) - 1)
  print(
    f"num_envs={num_envs} frames_per_batch={frames_per_batch} "
    f"iterations={len(iteration_seconds)}"
  )
  print(
    f"warmup : {frames_per_batch} frames in {warmup_seconds:.2f}s "
    f"(frames_per_second={frames_per_batch / warmup_seconds:.1f}, "
    "includes reset/startup)"
  )
  if steady_seconds > 0:
    frames_per_second = steady_frames / steady_seconds
    print(
      f"steady : {steady_frames} frames in {steady_seconds:.2f}s "
      f"(frames_per_second={frames_per_second:.1f}, "
      f"frames_per_second_per_env={frames_per_second / num_envs:.1f})"
    )
  else:
    print("steady : not available (need smoke.iterations >= 2)")

  if batch_recorder is not None:
    batch_recorder.close()
    for output_dir in batch_recorder.output_dirs:
      print(f"Wrote {output_dir / 'policy.hdf5'}", flush=True)
    if config.record.write_video:
      for path in batch_recorder.write_videos(config.record.video_stride):
        print(f"Wrote {path}", flush=True)

  # shutdown() can fast-exit the process; everything is printed above.
  collector.shutdown()
  factory.close_simulation_app(config.environment)


if __name__ == "__main__":
  utils_run_dir.install_hydra_support("torchrl_smoke")
  main()
