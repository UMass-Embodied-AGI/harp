# Contributing

Thanks for improving HARP-Bench. This page covers the checks and conventions
a change must follow; the [README](README.md) covers installation.

## Before you send a change

1. Set up the pinned environment (`requirements.txt`; the simulator lock in
   `requirements-genesis-latest.txt` pins `genesis-world==1.2.0`) and
   `source scripts/env.sh`.
2. Run the source checks: `GENESIS_BACKEND=cpu bash scripts/check_quality.sh`
   (Ruff plus the full pytest suite; it must exit 0).
3. For framework, task, planner, physics or renderer changes, also run the
   native checks in [release checks](docs/release_readiness.md): the
   all-registered reset/step probe, and the fixed-set expert evaluation
   (`scripts/eval_expert_set.py`) for the affected tasks. Unit tests mock the
   simulator and do not measure expert success.

## Compatibility contracts

- `TASK_DESCRIPTORS` in [`envs/tasks/registry.py`](envs/tasks/registry.py) is
  the authoritative task list. Task names, aliases, no-human mappings, public
  imports, CLI arguments and the files in `compat/release_files_v1.txt` are
  frozen by the tests under `tests/contracts/`. Change them only together with
  an explicit, versioned migration in `compat/release_contract_v1.json`.
- Preserve seeded draw order, configuration defaults, collision checks and
  scoring. A change to any of them changes benchmark results.

## Task-code conventions

See [running jobs and task rules](docs/general/running_jobs.md) and the
[task framework](docs/task_framework.md). In short:

- Quaternions are `(w, x, y, z)`.
- Scene construction happens in `reset()`; every reset rebuilds the scene.
- Scripted experts use mplib for planning and IK. Do not add Genesis IK
  fallbacks; `scripts/eval_vla.py`'s `genesis_ik` servo control is a separate
  contract for policies.
- New scripted grasps use `BaseTask.select_and_execute_grasp`; grasp targets
  go object-local -> world TCP -> robot link through the configured TCP offset.
- The avatar must be in mplib's obstacle point cloud before every robot plan.
- Do not teleport the robot arm or tether held objects during execution. The
  debug switches that do so (`DEBUG_TELEPORT`, `USE_SETQPOS`,
  `DEBUG_BOUNDED_JOINT_PATHS`) are refused by the eval and collection entry
  points.

## Where things go

Logs go under `job_logs/`, recordings and media under `data/<task-or-run>/`,
training and evaluation outputs under `runs/`. All of these are gitignored.
Machine-specific paths belong in the gitignored `scripts/env.local.sh`.
