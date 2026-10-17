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
dapper/run_sim.sh python /project/dapper/scripts/random_rollout.py --output-dir /output/random_rollout/run0 --overwrite --write-video
dapper/run_sim.sh python /project/dapper/scripts/run_solution.py --solution-dir /project/dapper/src/solution-alpha --output-dir /output/solution_alpha/run0 --overwrite --write-video --debug
```

Training a behavior: 
```
dapper/run_sim.sh python /project/dapper/src/rl/scripts/ppo.py
```

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
