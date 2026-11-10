# Pinned runtime

## Runtime contract

The release target is Python 3.10 with `genesis-world==1.2.0`. Install
[`requirements.txt`](../requirements.txt), the release entry point documented
in the root [`README.md`](../README.md). It includes
[`requirements-genesis-latest.txt`](../requirements-genesis-latest.txt), the
narrow simulator/planner/renderer lock, plus the repository's task and
offline-data helpers. Do not loosen the simulator lock without rerunning the
robot, task, planner, and renderer compatibility sweep.

Core pinned versions are:

- `genesis-world==1.2.0`
- `gs-nyx==0.1.3`
- `gs-nyx-plugin==0.1.4`
- `mplib==0.2.1`
- `numpy==1.26.4`

Genesis 1.2.0 emits an upstream warning for the validated
`torch==2.4.1+cu121` pin because it recommends Torch 2.8 or newer. The release
keeps the end-to-end-tested Torch/NYX/CUDA combination for now; a Torch upgrade
requires a separate renderer, planner, and VLA compatibility sweep.

Use the repository environment:

```bash
source scripts/env.sh
printf 'MAWM_PY=%s\n' "$MAWM_PY"
```

Commands below are local; on a shared cluster run them through the scheduler
([cluster guide](cluster.md)).

## Public compatibility surface

The release exposes 49 canonical task names and 32 accepted compatibility
names (the pre-HARP names; see
[task names](task_framework.md#task-names)). The authoritative metadata is
`TASK_DESCRIPTORS` in [`envs/tasks/registry.py`](../envs/tasks/registry.py);
`TASK_MAP`, `TASK_ALIASES`, and the no-human maps are compatibility projections
exported by [`envs/tasks/__init__.py`](../envs/tasks/__init__.py).

`BaseTask` accepts these robot embodiment keys:

- `piper`
- `franka`
- `stretch`
- `xarm7`
- `arx_x5`
- `ur5_wsg`

That list describes loader support; individual tasks may still require a
task-specific embodiment calibration. Robot loader support does not add extra
task names to the registry. See [`envs/base_task.py`](../envs/base_task.py)
and [`envs/robot/`](../envs/robot/) for the implementations.

## Planner contract

Scripted experts use mplib for collision-aware path planning and IK. This rule
does not apply to high-rate VLA EE servoing: [`scripts/eval_vla.py`](../scripts/eval_vla.py)
intentionally sets `planner_override: genesis_ik` so each single-step target is
solved locally from the current qpos. The rationale and task-development rules
are in [`general/running_jobs.md`](general/running_jobs.md).

## Validation

The repository has an automated pytest suite under [`tests/`](../tests/),
including frozen registry/release-contract tests and latest-Genesis integration
tests. Run the static and unit contracts with a rasterizer-safe backend:

```bash
source scripts/env.sh
GENESIS_BACKEND=cpu python -m pytest -q
```

Probe every accepted task name in isolated subprocesses, including one reset
and one post-reset physics step:

```bash
source scripts/env.sh
GENESIS_BACKEND=cpu python scripts/probe_latest_genesis_robot_tasks.py \
  --all-registered --settle-steps 1 --steps 1 --timeout 300
```

This is a startup/reset/physics-step contract for all 81 accepted names, not a
claim that every scripted expert succeeds on every seed. Release gates are
listed in [release readiness](release_readiness.md); task-quality claims must
use the fixed evaluation workflow described in
[`general/running_jobs.md`](general/running_jobs.md).

For a focused robot/task integration probe:

```bash
source scripts/env.sh
GENESIS_BACKEND=cpu python scripts/check_latest_genesis_integration.py \
  --python "$MAWM_PY" --robots franka xarm7 \
  --tasks blocks_ranking_rgb dump_bin --steps 1
```

NYX rendering has separate GPU constraints and one-scene-build-per-process
requirements. Follow [`general/nyx_render_howto.md`](general/nyx_render_howto.md)
and [`general/gpu_resources.md`](general/gpu_resources.md) before running it.
