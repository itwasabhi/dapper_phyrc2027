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
- Method names should lead with actions or verbs (i.e get_new_position())
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
- Everything written to `/output` lands in the repo's `output/` (gitignored), the same mount upstream uses. No symlinks.
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
- `behaviors/arm.py`: action/joint/controller_target indices, rates, and gripper kinematics measured in the sim (arm extends along base +x, fingers close along base y, wrist pitch tops out near +1.14 rad = 65 deg down). Velocity commands integrate into `controller_target`; `arm.command_to_target` sets a joint's target directly.
- `DriveIndependent` re-plans each step around the scene's obstacles (`collision.plan_path`, a visibility graph over inflated table/manikin corners); a robot is blocked only when no path exists.
- `drive_to_shirt.DriveToShirtHemCorners` + `down_and_pinch.DownAndPinch` (with `shirt_grasp.get_hem_grasp_targets`): each robot stands at the side of the shirt facing across it, then lowers its tilted gripper onto a hem corner, closes, and lifts. `DownAndPinch.grasped()` reports per robot whether the fingers stopped apart (holding cloth).

### Vision and open models
`dapper/src/vision/` holds detection/annotation code. Open models are downloaded from Hugging Face only via `dapper/scripts/download_models.py` into `~/data/physrc_assets` (mounted at `/assets`); everything else runs offline. See `dapper/src/vision/MODELS.md`.
Run any solution with `--debug` (`run_solution.py`) to get annotated frames, `annotated.mp4` and `detections.jsonl` in `<run dir>/debug/`. `dapper/scripts/perception_debug.py` does the same over a recorded `policy.hdf5`.
Never put code in `~/data`; throwaway scripts go under `dapper/`.

### Shirt perception
The shirt is always the same blue T-shirt with yellow trim; only its table is randomized. `vision/color_mask.py` holds that color prior. In `BasicSceneIntegrator`, a shirt detection votes only on shirt-colored pixels inside its box (boxes without any are dropped, e.g. the manikin's striped top), and shirt votes are box-filtered before fitting a min-area rectangle.
`SceneEstimate.shirt.yaw` is the hem-to-collar heading (`length` runs along it), from `vision/shirt_orientation.py`: the T's sleeve end and the trim (wide hem band vs. small collar V) pick one of the rectangle's 4 headings, then the hem band's line refines it. `shirt_heading_margin` near 0 means the heading is ambiguous (e.g. the hem is not visible).
Iterate offline instead of in the sim:
- Record scans once: `run_solution.py --steps 300 --seed k --log-dir-name scan-seed<k>`.
- `scripts/build_shirt_labels.py <policy.hdf5 ...>`: color-proposed box/mask labels plus `review/*.png` sheets; set each frame's `review` to `ok`/`reject` in `labels.json`.
- `scripts/eval_detector.py <label dir> --shirt-prompts ... [--dump-errors]`: 2D recall/precision/AP per prompt set.
- `scripts/replay_scene.py <policy.hdf5 ...> --labels <label dir> [--prompt/--grid/--class-threshold ...]`: replays the scan through the integrator and scores the shirt footprint against one lifted from the labels, and the heading against human annotations in `<label dir>/headings.json` (annotate from each recording's `reference_topdown.png`). `--rotate-world-deg` tests other shirt headings on the same real images.
Labels come only from public RGB-D, the color prior and human review, never from sim state (RULES.md), so they may be used for evaluation and fine-tuning.

### Run directories
Scripts that produce artifacts (rollouts, `ppo.py`, `torchrl_smoke.py`, `env_preview.py`, `make_video.py`, `perception_debug.py`) write to `/output/<script_name>/<uid>/` via `utils_run_dir`. Every one accepts `--log-dir-prefix P` (dir becomes `P-<uid>`) and `--log-dir-name N` (exact name, replaces the uid; rollout scripts also need `--overwrite` to reuse it). For Hydra scripts the flags are rewritten to `log_dir_prefix=`/`log_dir_name=` overrides, which also work directly.

### Recording rollouts
Any script driving `DressingEnv(profile="actor_rgbd")` can record the same `policy.hdf5` as teleop (loadable with `Policy.dataset_reader.PolicyDataset`) via `utils_rollout_recorder` (see `scripts/random_rollout.py`). Pass `--write-video` (via `add_recording_flags`) to also render `<run dir>/videos/*.mp4`, or render later:
```
dapper/run_py.sh /project/dapper/scripts/make_video.py /output/random_rollout/<uid>/policy.hdf5
```

### RL training
`dapper/src/rl/` is a Hydra + torchrl PPO stack shared by all learned behaviors; only configs and (if needed) env/reward/encoder modules change per behavior.
- Layout: `configs/<name>.yaml` (top level; composes the `environment` and `ppo` groups), `configs/environment/<name>.yaml`, `configs/ppo/<name>.yaml` (hyperparameters plus the `model` section), `envs/` (sim env, spawn sampling, rewards, factory), `models/` (encoders, actor/critic), `ppo_builder.py`, `scripts/ppo.py`.
- Train: `dapper/run_sim.sh python /project/dapper/src/rl/scripts/ppo.py [hydra overrides]`. Runs land in `/output/ppo/<uid>/` (`actor.pt`, `critic.pt`, `trainer.pt`, `run_metadata.json` with commit and dirty flag, CSV scalars). Commit before runs whose results may be cited.
- Check spawns visually: `dapper/run_sim.sh python /project/dapper/src/rl/scripts/env_preview.py`.
- New behavior: add `configs/environment/<name>.yaml` and `configs/ppo/<name>.yaml`, a top-level `configs/<name>.yaml`, and run with `--config-name <name>`. Select env, reward fn and encoder with `_target_`.
- Rewards are `(previous_observation, action, observation) -> (reward, terminated)` callables injected into the env (`envs/rewards.py`); the default `zero_reward` is a placeholder.
- Models accept any number of leading batch dims (`()`, `[B]`, `[num_envs, T]`) and read per-sample shapes from the env specs.
- The training env picks the shirt's table itself and places robots near its corners. The policy never sees the table index; do not feed it to the policy.
- Encoders: `RgbdCnnEncoder` (default, from scratch) or `TimmFrozenEncoder` (`ppo.model.encoder._target_=rl.models.encoders.TimmFrozenEncoder`; first run `download_models.py resnet18_timm`).
- `torchrl==0.14.1` has no `PPOTrainer.from_env`; `ppo_builder.build_trainer` plays that role. `optim_steps_per_batch=None` is deliberate (a fixed count skips every second epoch).
- `environment.num_envs>1` (one Isaac process per env via `ParallelEnv`) works with the fake env but failed in the sim at worker start (articulation init); treat as unverified.
- Tests sit next to modules (`rl/envs/tests`, `rl/models/tests`, `rl/tests`); run with `dapper/run_py.sh -m pytest -p no:cacheprovider /project/dapper/src/rl -c /project/dapper/pyproject.toml`.
