# Overview
Fork of the PhyRC 2027 competition repo (upstream: https://github.com/rirolab/PhyRC_2027_Challenge) plus my own work in `dapper/`.
Task: two Stretch4 robots dress a manikin in a T-shirt in Isaac Sim 6.0.1. Policies see public observations at 20 Hz and output a `(2,9)` action.

## Layout
- `dapper/`: all of my code (models, training, scripts, tests, submission packaging). Python style and tooling rules are in `dapper/CLAUDE.md`.
- Everything else is upstream (`run.sh`, `Dockerfile`, `src/PhyRC_Sim`, `scripts/`, `config/`, `docs/`). Treat it as read-only.
- If an upstream file must change, keep the edit minimal and record it in `dapper/PATCHES.md` so it can be re-applied after merging upstream.

## Running things
- The sim only runs in the Docker image, via `./run.sh` (see `./run.sh help`). The repo is mounted read-only at `/project`; write outputs to `/output` (host: `output/`).
- Sim-side scripts: `./run.sh python /project/dapper/scripts/<script>.py`. Do not add files to the top-level `scripts/`; it is mounted separately at `/scripts` and belongs to upstream.
- Host-side code that does not import Isaac (model code, dataset loading, tests): `uv run --project dapper ...`.
- Only one sim process can run at a time (`run.sh` takes a lock), and `DressingEnv` allows one instance per process.

## Competition rules (docs/competition/RULES.md)
- Policy inputs are public observations only; never use cloth vertices, exact shirt/manikin/chair poses, ground-truth segmentation, or evaluator internals.
- Do not use internal state to create action labels, targets, automatic demos, or a teacher policy.
- Do not change physics, geometry, actuation limits, reset rules, or evaluation functions to improve results.
- Commit before runs whose results may be cited; the commit hash and dirty flag are recorded with each run.
