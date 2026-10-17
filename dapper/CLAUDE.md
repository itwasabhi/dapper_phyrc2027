# Overview
Research-and-development for a robotic caregiving task. 

## Rules
Python scripts and tests run in the dapper Docker images (see Python dependencies below), not in a host virtualenv.
The launchers set `PYTHONPATH` (`dapper/src`, `src/PhyRC_Sim`, `/scripts`), so scripts import modules directly; do not add `sys.path` manipulation.

## Tips
When running a test file (`dapper/run_py.sh -m pytest -p no:cacheprovider /project/dapper/src/tests -c /project/dapper/pyproject.toml -k ...`), be sure to use test filters to only run the test that is relevant for that particular interaction. 

This project primarily uses the following tools: 
- isaacsim for robotics simulation
- pytorch for general ML
- torchrl for RL implementations such as an environment wrapper, environment transforms, data collection, and training code 
- hydra for config management 

## Style Guidelines
- Stick to the google3 python style guide. 
- Use import statements for packages and modules only, not for individual types, classes, or functions.
- Only use comments for non-obvious lines of implementation; comments should not be overly verbose and should not describe what can be easily inferred from code. 
- Never use inheritance; prefer depedency injection and a library of reusable code. 
- Tests for file `<MODULE>.py` should be in `tests/<MODULE>_test.py` in the same directory as the module (under `dapper/src/`). Tests should be implemented with pytest. 
- prefer type checks over testing `getattr`
- Do not use unreadable variable names like `L` or `K` even if they map to a known tensor dim [.., T, ...]. Prefer descriptive names like num_timesteps, or num_clusters
- Keep docstrings / comments MINIMIAL and informative. Make sure docstrings of methods do not refer to callers of that method.

## Python dependencies
All project code runs in Docker images layered on the upstream Isaac image (Python 3.12). There is no project virtualenv on the host.
- **`dapper-dev`** (`dapper/docker/dev/`, `requirements-dev.txt`): training, experiments, tests and sim scripts. Run with `dapper/run_py.sh` (no Isaac, no sim lock) or `dapper/run_sim.sh` (starts Isaac; wraps `./run.sh`).
- **`dapper-prod`** (`dapper/docker/prod/`, `requirements-prod.txt`): only what `policy.py` needs at inference time. The submission `Dockerfile` is a copy of this one. `dapper-dev` is built on top of it.
- Host `uv` is only for `uvx --no-env-file ruff format dapper` (from the project root); ruff is not a project dependency and `pyproject.toml` only holds its config. `--no-env-file` is needed because `UV_ENV_FILE` is set in the shell to a missing file.
- Everything written to `/output` lands in `~/data/dapper_phyrc` (override with `DAPPER_OUTPUT_DIR`); the launchers symlink `output/` to it.
- Build with `dapper/docker/build.sh [prod|dev|all]` (needs `./run.sh build` first).

Rules:
- Put a dependency in the narrowest image that needs it: `requirements-dev.txt` unless the deployed policy imports it.
- Never edit the upstream `Dockerfile` or `requirements-runtime.txt`, and never install torch/torchvision or change the version of any package already in the base image (the submission is rejected otherwise). Torch (2.11.0+cu128) is not in site-packages; it comes from Isaac's `omni.isaac.ml_archive/pip_prebundle`, which the images and launchers put on `PYTHONPATH`. Pin every addition (`pkg==x.y.z`) and install with `-c constraints.txt`.
- Do not `pip install` inside a running container; edit the requirements file and rebuild.
- Anything a deployed policy imports must be in `requirements-prod.txt`, and its weights must be in the submission zip (no downloads at evaluation; pre-download Hugging Face models into the zip and load from a local path).
- Before submitting, test against `dapper-prod` (`DAPPER_IMAGE=dapper-prod dapper/run_sim.sh python /scripts/evaluate_policy.py ...`), not `dapper-dev`.
- If a dependency cannot coexist with Isaac's Python, run it as a separate process or image instead of loosening constraints.


## Code features

### Behaviors and solutions
A solution's `policy.py` composes a sequence of **behaviors** from `dapper/src/behaviors/`: each
owns both robots' actions for a span of steps. `behaviors/execution_utils.py` steps the active one
and advances once it's done (see `dapper/src/solution-alpha/policy.py` for the reference wiring).
Implement new or modified behaviors here (e.g. "pick up the shirt", "a second dressing policy"),
not inline in a solution's `policy.py`.

- `behaviors/<name>.py` for a plain behavior; `behaviors/<name>/` once it needs its own training
  scripts/configs (e.g. a future `behaviors/dress_1/`).
- No declared interface — by convention a behavior implements `reset(state)`,
  `step(observation) -> np.ndarray` (2,9), `done() -> bool`, `close()`, called duck-typed by
  `execution_utils`.
- Share state only via `behaviors.base.BehaviorState` (passed into `reset`); put shared global
  settings (e.g. control rate) in `behaviors.base.BehaviorCfg`, injected into each constructor.
- Dependency injection only, as in `vision/scene_integrator.BasicSceneIntegrator` — never subclass
  a behavior. Per-behavior tunables go in a frozen dataclass; reuse `behaviors/motion.py` for
  pose/control math.
- Tests in `behaviors/tests/<name>_test.py`, kept minimal, with fakes instead of real models.

### Vision and open models
`dapper/src/vision/` holds detection/annotation code. Open models are downloaded from Hugging Face only via `dapper/scripts/download_models.py` into `~/data/physrc_assets` (mounted at `/assets`); everything else runs offline. See `dapper/src/vision/MODELS.md`.
Run any solution with `--debug` (`run_solution.py`) to get annotated frames, `annotated.mp4` and `detections.jsonl` in `<output-dir>/debug/`. `dapper/scripts/perception_debug.py` does the same over a recorded `policy.hdf5`.
Never put code in `~/data` / `output/`; throwaway scripts go under `dapper/`.

### Recording rollouts
Any script driving `DressingEnv(profile="actor_rgbd")` can record the same `policy.hdf5` as teleop (loadable with `Policy.dataset_reader.PolicyDataset`) via `utils_rollout_recorder` (see `scripts/random_rollout.py`). Pass `--write-video` (via `add_recording_flags`) to also render `<output-dir>/videos/*.mp4`, or render later:
```
dapper/run_py.sh /project/dapper/scripts/make_video.py /output/random_rollout/run0/policy.hdf5 --output-dir /output/videos/run0
```

### RL training
`dapper/src/rl/` is a Hydra + torchrl PPO stack shared by all learned behaviors; only configs and (if needed) env/reward/encoder modules change per behavior.
- Layout: `configs/<name>.yaml` (top level; composes the `environment` and `ppo` groups), `configs/environment/<name>.yaml`, `configs/ppo/<name>.yaml` (hyperparameters plus the `model` section), `envs/` (sim env, spawn sampling, rewards, factory), `models/` (encoders, actor/critic), `ppo_builder.py`, `scripts/ppo.py`.
- Train: `dapper/run_sim.sh python /project/dapper/src/rl/scripts/ppo.py [hydra overrides]`. Runs land in `/output/rl/<behavior>/<timestamp>/` (`actor.pt`, `critic.pt`, `trainer.pt`, `run_metadata.json` with commit and dirty flag, CSV scalars). Commit before runs whose results may be cited.
- Check spawns visually: `dapper/run_sim.sh python /project/dapper/src/rl/scripts/env_preview.py --output-dir /output/rl/env_preview`.
- New behavior: add `configs/environment/<name>.yaml` and `configs/ppo/<name>.yaml`, a top-level `configs/<name>.yaml`, and run with `--config-name <name>`. Select env, reward fn and encoder with `_target_`.
- Rewards are `(previous_observation, action, observation) -> (reward, terminated)` callables injected into the env (`envs/rewards.py`); the default `zero_reward` is a placeholder.
- Models accept any number of leading batch dims (`()`, `[B]`, `[num_envs, T]`) and read per-sample shapes from the env specs.
- The training env picks the shirt's table itself and places robots near its corners. The policy never sees the table index; do not feed it to the policy.
- Encoders: `RgbdCnnEncoder` (default, from scratch) or `TimmFrozenEncoder` (`ppo.model.encoder._target_=rl.models.encoders.TimmFrozenEncoder`; first run `download_models.py resnet18_timm`).
- `torchrl==0.14.1` has no `PPOTrainer.from_env`; `ppo_builder.build_trainer` plays that role. `optim_steps_per_batch=None` is deliberate (a fixed count skips every second epoch).
- `environment.num_envs>1` (one Isaac process per env via `ParallelEnv`) works with the fake env but failed in the sim at worker start (articulation init); treat as unverified.
- Tests sit next to modules (`rl/envs/tests`, `rl/models/tests`, `rl/tests`); run with `dapper/run_py.sh -m pytest -p no:cacheprovider /project/dapper/src/rl -c /project/dapper/pyproject.toml`.
