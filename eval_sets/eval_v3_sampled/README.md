# eval_v3_sampled

The HARP-Bench release evaluation set: 3,403 episode initializations over the
37 benchmark tasks (the train37 task set). By paper condition, with the
manifest `category` value in parentheses: Cueing (`cueing`) 7 tasks / 662
rows, Contribution (`contribution`) 10 / 899, Interruption (`interruption`)
10 / 974, and Co-presence (`copresence`) 10 / 868. Task names, categories,
and row ids use the HARP-Bench names; seeds and `config_overrides` are
unchanged from the pre-HARP manifest (old `id` =
`eval_v3_sampled__<old category>__<old task>__seed_<seed>`, mapped by
`envs.tasks.LEGACY_TASK_NAMES` / `LEGACY_CATEGORY_VALUES`). Per-task counts are in
[`counts.json`](counts.json); `envs.tasks.paper_label(task)` gives each task's
paper condition and name (see the task table in the repository README). Like
every eval set, it stores initializations only: no rollouts, videos, actions
or success labels.

## Selection

Each row is a seed that passed seed sampling on this release's code:

- the privileged full-state expert (`scripts/eval_expert_set.py --policy expert_full_state`)
  succeeds;
- a hold-still policy (the robot never moves) does **not** succeed and makes no
  scored contact with the human, so the task cannot be solved by waiting;
- the scene resets without a layout error.

Seeds come from the `9000000+` evaluation range and do not overlap any
training range. Most tasks keep 63-100 of their first 100 candidate seeds.
`put_object_in_cabinet_copresence` and `dump_bin_copresence` were topped up to 90 seeds
from further candidates (seeds `9001000+` and `9002000+`) screened the same way.

An independent second full-state expert run on these rows succeeds on about
99% of them. This is therefore an **expert-solvable** set, not an unbiased
sample of each task's scene distribution: a policy's score measures how often
it succeeds where the scripted expert reliably can.

## Settings

Every row carries its full configuration in `config_overrides`: the NYX
renderer (spp 16, denoise, random environment map), 640x480 head and
right-wrist cameras, a randomized avatar (seeded per episode), strict
Co-presence layout, avatar collision tracking with `collision_check_stride` 5,
`planner_override` null and `eval_mode` true. `max_steps` and
`action_substeps` are per row. NYX needs the GPU backend; see
[`docs/nyx_render.md`](../../docs/nyx_render.md).

## Running it

On one machine with an NYX-capable GPU (see the repository README):

```bash
source scripts/env.sh

# full-state expert reference (each row runs in a fresh process because NYX
# builds one scene per process; results under
# runs/expert_eval_set/eval_v3_sampled/expert_full_state/)
python scripts/eval_expert_set.py --eval-set eval_sets/eval_v3_sampled \
  --policy expert_full_state

# one shard of rows (zero-based, end exclusive), e.g. to split the set
# across GPUs or machines
python scripts/eval_expert_set.py --eval-set eval_sets/eval_v3_sampled \
  --policy expert_full_state --start-index 0 --end-index 500

# the released VLA baselines (after scripts/vla_train37/fetch_ckpts.py)
scripts/vla_train37/launch_full.sh eval_sets/eval_v3_sampled/manifest.jsonl <run_name> --local
```

With `GENESIS_BACKEND=cpu` the NYX rows fall back to the rasterizer (the
script prints a notice); those are not the reference results. To run shards
as SLURM jobs, see [`docs/cluster.md`](../../docs/cluster.md).

Report success together with the safety metrics written per episode
(`safety` block: collision, near-miss, intrusion time below 0.05/0.10/0.20 m,
minimum and median robot-human separation).
