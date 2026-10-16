# Upstream patches

Minimal edits to upstream files. Re-apply after merging upstream.

## `run.sh`: extra docker args
After the `for mapping in ...` loop that adds the cache mounts, append the words of `PHYRC_EXTRA_DOCKER_ARGS` to `ARGS`:

```bash
read -ra EXTRA_DOCKER_ARGS <<< "${PHYRC_EXTRA_DOCKER_ARGS:-}"
ARGS+=("${EXTRA_DOCKER_ARGS[@]}")
```

Why: `dapper/run_sim.sh` uses it to mount the model assets dir at `/assets` (read-only).
