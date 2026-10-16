# External models

All open models come from the public Hugging Face Hub and are listed here as the external-model training record (RULES.md).
Repo ids, licenses and pinned commit shas live only in `model_registry.py`.

| name | repo | license | use |
|---|---|---|---|
| `omdet_turbo` | `omlab/omdet-turbo-swin-tiny-hf` | apache-2.0 | zero-shot open-vocab 2D detection (default) |
| `grounding_dino_tiny` | `IDEA-Research/grounding-dino-tiny` | apache-2.0 | slower zero-shot detector for comparison |

## When
Only when you run the download script, with network. Nothing downloads at import, in tests, or in sim/eval runs:
the dev image sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and loaders pass `local_files_only=True`.
```
dapper/run_py.sh /project/dapper/scripts/download_models.py [name ...]   # default: all
```

## Where
Host `~/data/physrc_assets/models/<name>/` (override the root with `DAPPER_ASSETS_DIR`), mounted at `/assets`:
read-write under `dapper/run_py.sh`, read-only under `dapper/run_sim.sh` (via `PHYRC_EXTRA_DOCKER_ARGS`, see `dapper/PATCHES.md`).
Code reads from `/assets/models` by default (`model_registry.DEFAULT_MODELS_ROOT`).

## How
`huggingface_hub.snapshot_download(repo_id, revision=<sha>, allow_patterns=..., local_dir=/assets/models/<name>)`.
Add a model by adding a `ModelSpec` to `REGISTRY`, then re-run the download script.

## Submission
The zip must contain the weights; nothing is downloaded at evaluation. Copy only the models the policy uses into
`models/` next to `policy.py` and pass that as `models_root`. Not packaged yet.
