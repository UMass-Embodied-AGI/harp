# Code map

## Runtime ownership

```text
envs/
  tasks/                 Public scenario classes and authoritative registry
  task_bases/            Shared family cores and local helpers
    categorize_support/  Categorize-only motion, calibration, and scoring helpers
    stack_bowls_collision/  Packaged bowl collision URDF and hulls (runtime data)
  interaction/           Human-interaction controllers and eval-mode hooks
  base_task.py           Shared reset, execution, observation, and policy lifecycle
  base_task_parts/       Scene, object, observation, and grasp-selection components
  avatar/                Avatar loading, motion, geometry, and collision checks
  robot/                 Embodiment implementations
  planning/              Planner adapters (mplib, Genesis IK, optional cuRobo)
  scenes/                Reusable kitchen and factory scene builders
  contact.py             Read-only native contact snapshots
```

Task names and aliases come from [`tasks/registry.py`](../envs/tasks/registry.py),
not filenames. Public imports through `envs.tasks` and existing task class paths
remain supported. See [the task framework](task_framework.md) for the
controller lifecycle.

Family-specific code belongs with its core under `envs/task_bases/`:

| Area | Start here | Supporting modules |
| --- | --- | --- |
| Categorize | `categorize.py` | `categorize_support/` |
| Dump bin | `dump_bin.py` | `dump_bin_grasp.py`, `dump_bin_motion.py`, `dump_bin_milk.py` (side-transport helpers), `dump_neutral_layout.py` |
| Block ranking | `blocks_ranking_rgb.py`, `blocks_ranking_size.py` | `block_grasp.py`, `block_neutral_public_grasp.py` |
| Co-presence human work | `neutral_avatar_table_work.py` | `neutral_avatar_spec.py`, clip/transform, layout, and prop modules |
| Cabinet, shoes, bowls | Their family core modules | `cabinet_ordinary_motion.py`, `place_dual_shoes_motion.py`, `stack_bowls_motion.py`, `stack_bowls_collision/` |
| Human handover | `envs/tasks/take_from_human_easy.py` | `take_hold.py`, `take_release.py`, `take_release_plan.py`, `take_handover_observe.py` |

General grasp and planner behavior stays in the shared framework; object-specific
corrections stay within their task family.

## Evaluation, collection, and baselines

| Responsibility | Module |
| --- | --- |
| Scripted demo collection | [`scripts/collect.py`](../scripts/collect.py), [`scripts/collect_vla.sh`](../scripts/collect_vla.sh) |
| One training episode: reset, expert, recording | [`scripts/vla_data/collect_vla.py`](../scripts/vla_data/collect_vla.py) |
| Collector metadata and H5 contract | [`scripts/vla_data/collection_contract.py`](../scripts/vla_data/collection_contract.py) |
| NYX episode validation | [`scripts/vla_data/validate_nyx.py`](../scripts/vla_data/validate_nyx.py) |
| Bounded NYX collection campaigns | [`scripts/vla_data/nyx_campaign.py`](../scripts/vla_data/nyx_campaign.py) |
| Dataset conversion (RLDS, LeRobot) | `scripts/data_conversion/`, `build_genesis_hr_bench_rlds.py`, `convert_genesis_hr_bench_to_lerobot.py` |
| Eval-set manifests | `scripts/generate_eval_set.py`, `scripts/validate_eval_set_init.py`, `eval_sets/` |
| Fixed-manifest scripted expert evaluation | [`scripts/eval_expert_set.py`](../scripts/eval_expert_set.py) |
| Policy/VLA evaluation | `scripts/eval.py`, `scripts/eval_vla.py`, `scripts/vla_client.py`, `scripts/run_vla_eval.sh` |
| Batched VLA evaluation of released checkpoints | `scripts/vla_train37/` |
| Baseline training | `scripts/finetune*.sh`, `patches/lerobot-genesis-hr-bench.patch` |
| Model server | `baseline/servers/lerobot_server.py` (LeRobot source in the `baseline/lerobot` submodule) |

See the [script index](../scripts/README.md) for entrypoints.

## Source and configuration

- `config/`: [named runtime profiles](../config/README.md).
- `tests/`: task and utility regressions; `tests/contracts/`: public API,
  registry, documentation, packaging, and source-hygiene checks. See the
  [test map](../tests/README.md).
- `compat/`: the frozen release contract and the required-file inventory.
- `docs/general/`: execution and implementation rules.
- `data/` (ignored): recordings and run outputs; `job_logs/` (ignored): logs.

When changing shared runtime code, preserve public imports, seeded random-draw
order, configuration semantics, and scoring, and run the registered-task
reset/step probe described in [running jobs](general/running_jobs.md).
