# dapper

Home of our dev efforts for the PhyRC challenge. 

Here we define our own docker image with additional python requirements to aid for development (docker/dev) and deployment to the official evaluation (docker/prod). 

All commands below run from the project root.

Setup (needed any time requirements change). 
```
./run.sh build && dapper/docker/build.sh all
```

Running common scripts: 
```
dapper/run_py.sh -m pytest -p no:cacheprovider /project/dapper/src/tests -c /project/dapper/pyproject.toml
dapper/run_sim.sh python /project/dapper/scripts/run_solution.py --solution-dir /project/dapper/src/solution-alpha --write-video --debug --log-dir-prefix alpha
```

Drive-planner check (`solution-test3`: scan, then synthetic independent, gather and formation legs), then plot each leg's plan vs. driven path:
```
dapper/run_sim.sh python /project/dapper/scripts/run_solution.py --solution-dir /project/dapper/src/solution-test3 --steps 3200 --seed 0 --log-dir-name synth-goals-pairs-seed0
dapper/run_py.sh /project/dapper/scripts/plot_drive_legs.py /output/run_solution/synth-goals-pairs-seed0/debug /output/run_solution/synth-goals-pairs-seed0/debug/drive_legs.png
```

Debugging: 
```
dapper/run_sim.sh python /project/dapper/scripts/random_rollout.py --write-video --log-dir-prefix test
dapper/run_py.sh /project/dapper/scripts/make_video.py /output/random_rollout/<uid>/policy.hdf5

```

Training a behavior: 
```
dapper/run_sim.sh python /project/dapper/src/rl/scripts/ppo.py --log-dir-prefix grasp ppo.total_frames=256
```

## Force-stopping a sim run
`run.sh` allows one sim at a time. "This repository already has an active prepare/run" means a sim container still holds `cache/runtime.lock`. Kill it (the lock releases when `docker run` exits):
```
docker kill $(docker ps -q --filter label=phyrc.project=PhyRC_2027)
flock -n cache/runtime.lock -c 'echo lock free' || echo lock held
```
If the lock is still held, find the holder with `lsof cache/runtime.lock` and `kill` that PID.

## Run directories
Scripts that produce artifacts write to `/output/<script_name>/<uid>/` (host: `output/<script_name>/<uid>/`, gitignored). Flags accepted by all of them:
- `--log-dir-prefix P`: directory becomes `P-<uid>`.
- `--log-dir-name N`: exact directory name instead of a uid (rollout scripts also need `--overwrite` to reuse an existing one).

For Hydra scripts (`ppo.py`, `torchrl_smoke.py`) the flags become `log_dir_prefix=` / `log_dir_name=` overrides.

## Layout
- `src/`: library modules (`utils_*.py`), with tests in `src/tests/<module>_test.py`.
- `src/solution-*/`: deployable solutions (`policy.py` at the root, zip the folder contents to submit).
- `scripts/`: entry points that run in the sim or dev image.
- `docker/`: `dapper-prod` and `dapper-dev` images and their pinned requirements.

## Formatting
Run on the host with `uv`, from the project root (so upstream code is not touched):
```
uvx --no-env-file ruff format dapper
uvx --no-env-file ruff check dapper
```
