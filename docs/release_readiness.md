# Release checks

Release readiness has two separate gates: source quality and simulator
behavior. Passing lint or unit tests does not establish expert success rates.

## Source checks

Install `requirements-dev.txt` in the pinned simulator environment. From the
repository root (CPU only, no GPU needed):

```bash
source scripts/env.sh
GENESIS_BACKEND=cpu bash scripts/check_quality.sh
```

On a shared cluster, submit the same command as a job instead of running it on
a login node ([cluster guide](cluster.md#source-checks)).

The command runs correctness-only Ruff over `envs`, `scripts`, `tests`, and
`baseline/servers`, then the complete pytest suite, including release
inventory checks. Vendored baseline sources are excluded. Both gates run even
if one fails, and the script exits nonzero if either fails. `RUFF_BIN` can name
an existing Ruff executable when installing into the pinned environment is not
appropriate.

Shell syntax, portable paths, and log destinations are checked for the working
tree, including new nonignored scripts. The inventory tests also require every
file listed in `compat/release_files_v1.txt` (and every bowl collision hull in
the package manifest under `envs/task_bases/stack_bowls_collision/`) to be
tracked by Git.

## Package contents

- Runtime code under `envs/` and `scripts/`, its tests, `config/`, and
  `eval_sets/`.
- The asset pack is distributed separately and mounted at `assets/`; see the
  installation section of the [README](../README.md).
- Local environments, `.tmp/`, recordings, caches, `runs/`, `data/`, and logs
  are not part of the package.

## Runtime behavior

Use the pinned Genesis environment with
[`check_latest_genesis_integration.py`](../scripts/check_latest_genesis_integration.py)
and the [registered-task probe](../scripts/probe_latest_genesis_robot_tasks.py).
Framework changes require the registered reset and step sweep. Renderer,
contact, grasp, or planner changes also require focused native checks; tests
that only mock the simulator do not replace those checks.

Preserve task names, public imports, seeded draw order, configuration defaults,
collision checks, and scoring. When behavior changes intentionally, version the
contract in `compat/release_contract_v1.json`.

## Known open test failures

Two assertions in `tests/test_take_handover_stance.py` fail on the release
code: an unreachable presentation can still trigger avatar repositioning
during execution, and the task no longer applies its previous physical-time
motion setting. Both tests are marked `pytest.mark.xfail(strict=True)`, so
`check_quality.sh` exits 0 while they fail and reports them as `xfailed`; a
change that makes either pass turns it into a failure until the marker is
removed. The task behaviour is unchanged until the intended behaviour is
resolved and verified natively.
