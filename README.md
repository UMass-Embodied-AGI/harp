# HARP-Bench

**Website:** [harp-bench.vercel.app](https://harp-bench.vercel.app/) ·
**Paper:** [arXiv:2611.11123](https://arxiv.org/abs/2611.11123) ([PDF](paper/harp_bench.pdf))

HARP-Bench is a simulation benchmark for robot manipulation around active
humans, built on Genesis. Animated human avatars change what the robot should
do, which objects are available, when it can act, and how it must move.
The repository (Python packages, paths, and dataset IDs) keeps its historical
`genesis-hr-bench` / `genesis_hr_bench` names.

Ships 49 canonical manipulation tasks plus 32 compatibility aliases (the
pre-HARP task names), with six robot embodiments. The 37 benchmark tasks
evaluated in the paper are the rows of
[`eval_sets/eval_v3_sampled`](eval_sets/eval_v3_sampled/README.md); the
remaining registered tasks are no-human base tasks and factory work-cell
scenarios. The release runtime is pinned to `genesis-world==1.2.0` with the
NYX GPU renderer.

Start with [installation](#installation), [quick start](#quick-start), and the
[documentation index](docs/README.md). The [code map](docs/code_layout.md)
explains module ownership.

## Benchmark tasks

The 37 tasks are task-condition instances over four human-present
conditions. Task names follow the paper: a Cueing task uses the paper task
name, and every other task is `<paper task>_<condition>`. The condition value
(in parentheses) is what `TaskVariant`, eval-set manifest `category` fields,
and summaries store.

- **Cueing** (`cueing`): the robot infers its objective from human activity,
  e.g. a human carrying, offering, or using an object.
- **Contribution** (`contribution`, suffix `_contribution`): the human
  completes part of the task, and the robot finishes the remaining subtasks.
- **Interruption** (`interruption`, suffix `_interruption`): the human
  temporarily changes execution conditions, e.g. by inspecting or holding the
  object the robot needs, and the robot must wait or adapt and then resume.
- **Co-presence** (`copresence`, suffix `_copresence`): the robot completes
  its original goal while a human does unrelated work nearby.

That is 7 Cueing, 10 Contribution, 10 Interruption, and 10 Co-presence tasks.
`envs.tasks.paper_label(task_name)` returns `(condition, paper task name)` for
any of them. The pre-HARP names (`*_assist`, `*_interrupt`, `*_neutral`,
`categorize_cooperative`, `deliver_to_human_easy`, `stamp_documents`) still resolve to the same
classes, so old commands, manifests, checkpoints and datasets keep working;
the [task framework](docs/task_framework.md#task-names) has the old-to-new
table, also exported as `envs.tasks.LEGACY_TASK_NAMES`.

### Cueing (`cueing`)

| Paper task | Task name | Description |
|---|---|---|
| Deliver to Human | `deliver_to_human` | Pick up the requested object from the table and deliver it to the human. |
| Pour Water | `pour_water` | Pour water from the mug into the cup. |
| Oil Bottle Recovery | `oil_bottle_recovery` | Recover the knocked-over oil bottle from the cooking area and place it upright. |
| Open Microwave | `open_microwave` | Open the microwave door for the human carrying a plate. |
| Feed Documents | `feed_documents` | Push each document into the stamp zone so the human can stamp it. |
| Take from Human (Easy) | `take_from_human_easy` | Take the requested compact object from the person and place it on the table. |
| Take from Human (Safety) | `take_from_human_safety` | Take the kitchen knife offered by the chopping human and place it in the knife box. |

### Contribution / Interruption / Co-presence

| Paper task | Contribution (`contribution`) | Interruption (`interruption`) | Co-presence (`copresence`) |
|---|---|---|---|
| Categorize Objects | `categorize_objects_contribution`: Sort the objects into matching baskets. | `categorize_objects_interruption`: Sort objects into matching baskets while yielding when the human inspects one. | `categorize_objects_copresence`: Sort objects while the human works nearby. |
| Put Object in Cabinet | `put_object_in_cabinet_contribution`: Place the object in a human-opened cabinet drawer. | `put_object_in_cabinet_interruption`: Place the object after the human inspects it. | `put_object_in_cabinet_copresence`: Put the object in the cabinet while the human works nearby. |
| Stack Three Bowls | `stack_three_bowls_contribution`: Stack one bowl while the human stacks another. | `stack_three_bowls_interruption`: Stack bowls while waiting if the human inspects one. | `stack_three_bowls_copresence`: Stack bowls while the human works nearby. |
| Place Bread in Basket | `place_bread_in_basket_contribution`: Place one bread while the human places the other. | `place_bread_in_basket_interruption`: Place breads while waiting if the human inspects one. | `place_bread_in_basket_copresence`: Place breads while the human works nearby. |
| Dump Bin | `dump_bin_contribution`: Empty the table cubes while the human tosses another cube into the bin. | `dump_bin_interruption`: Place each cube into the trash can, waiting if the human inspects one. | `dump_bin_copresence`: Put table objects into the bin while the human works nearby. |
| Place Burger and Fries | `place_burger_and_fries_contribution`: Place meal items while the human adds one item. | `place_burger_and_fries_interruption`: Place meal items while waiting if the human inspects one. | `place_burger_and_fries_copresence`: Place meal items while the human works nearby. |
| Place Pair of Shoes | `place_pair_of_shoes_contribution`: Put one shoe in the shoebox while the human puts the other. | `place_pair_of_shoes_interruption`: Put shoes in the shoebox while waiting if the human inspects one. | `place_pair_of_shoes_copresence`: Place shoes in the shoebox while the human works nearby. |
| Place Food in Skillet | `place_food_in_skillet_contribution`: Place the skillet on the stove and put foods in it while the human helps place one food. | `place_food_in_skillet_interruption`: Place the skillet and foods while waiting if the human inspects one. | `place_food_in_skillet_copresence`: Place food in the skillet while the human works nearby. |
| Rank Blocks by Color | `rank_blocks_by_color_contribution`: Place red, green, and blue blocks in left-to-right order while the human helps place one. | `rank_blocks_by_color_interruption`: Order RGB blocks while waiting if the human inspects one. | `rank_blocks_by_color_copresence`: Order RGB blocks while the human works nearby. |
| Rank Blocks by Size | `rank_blocks_by_size_contribution`: Place blocks from largest to smallest while the human helps place one. | `rank_blocks_by_size_interruption`: Order blocks by size while waiting if the human inspects one. | `rank_blocks_by_size_copresence`: Order blocks by size while the human works nearby. |

Each family also has a no-human base task, which keeps its historical name
(`put_object_cabinet`, `stack_bowls_three`, `blocks_ranking_rgb`, `dump_bin`,
...; the Categorize Objects tasks collapse to `categorize` under
`--no-human`). The factory work-cell scenarios (`factory_inspect_pack`, `factory_kit_packing`,
`factory_line_feeding`) are registered extras outside the 37-task benchmark;
see the [task inventory](docs/factory_task_inventory.md).

Supported robots are Franka Panda, Piper, Stretch, ARX X5, UR5+WSG, and xArm7.
Scripted experts use mplib planning and IK; policy evaluation has a separate
high-rate control path. See [the task framework](docs/task_framework.md).

## Installation

The commands in this README run on one Linux machine with an NVIDIA GPU. To
use a SLURM cluster instead, see [running on a SLURM cluster](docs/cluster.md).

### Requirements

- **Python 3.10** and an **NVIDIA GPU with a CUDA driver** for the benchmark
  itself. Every `eval_v3_sampled` row renders with the **NYX** renderer, which
  runs on the Genesis GPU backend (`GENESIS_BACKEND=gpu`) and draws through
  **Vulkan**, so it needs a graphics-capable card. RTX 2080 Ti, RTX 8000,
  RTX A4000, A16, A40, L4 and L40S have been verified. The compute-only data
  centre parts (**A100, H100, H200**) have no Vulkan device and **cannot run
  NYX**. Pre-Turing cards (GTX 1080 Ti, Titan X, M40) are too old for the
  Genesis GPU backend. See [GPU requirements](docs/general/gpu_resources.md).
- **CPU-only fallback.** With `GENESIS_BACKEND=cpu` and `--renderer rasterizer`,
  scripted experts, data-collection debugging and the test suite run without a
  GPU. A row that asks for NYX falls back to the rasterizer on the CPU backend,
  and `scripts/eval_expert_set.py` prints a notice when that happens. CPU
  rasterizer results are not the reference numbers, and the VLA baselines need
  a GPU.
- **Memory:** at least 32 GB of RAM; some scene builds need 64 GB.

### Environment

```bash
# 1. Genesis runtime env (genesis-world 1.2.0 + NYX, all from PyPI).
python3.10 -m venv .venv-genesis-latest
.venv-genesis-latest/bin/python -m pip install -U pip wheel
.venv-genesis-latest/bin/python -m pip install -r requirements.txt

# Optional: the test runner and correctness linter.
.venv-genesis-latest/bin/python -m pip install -r requirements-dev.txt
```

`requirements.txt` is the installation entry point. It includes
`requirements-genesis-latest.txt`, the narrow simulator/planner/renderer lock,
and adds the task and offline-data helpers. Keep the simulator lock pinned:
changing it requires a new robot, task, planner, and renderer compatibility
sweep.

`source scripts/env.sh` activates `.venv-genesis-latest`, exports `MAWM_PY`
(the simulator interpreter), and sets `PYTHONPATH` and `GENESIS_BACKEND=gpu`.
If your interpreters live elsewhere, set them in `scripts/env.local.sh`: copy
[`scripts/env.local.sh.example`](scripts/env.local.sh.example). With an
external `MAWM_PY`, run `"$MAWM_PY"` wherever this README says `python`.

The VLA baselines run their policy servers in separate LeRobot environments
(`LEROBOT_PY`, `PI0_PY`); see [VLA fine-tuning](docs/vla-finetune-pipeline.md)
for how to create them.

### Downloads

Everything below is public and ungated, so no Hugging Face login or token is
needed.

1. **Asset pack (required).** The Hugging Face dataset
   [`zhouqh/harp-bench-assets`](https://huggingface.co/datasets/zhouqh/harp-bench-assets)
   holds objects, avatars, motions, robot descriptions and HDRIs. Download it
   to `<repo>/assets`, or download it elsewhere and symlink it there:

   ```bash
   pip install -U huggingface_hub   # provides the `hf` command
   hf download zhouqh/harp-bench-assets --repo-type dataset --local-dir assets
   ```

   The pack is self-contained and ships a `manifest.json` with a SHA-256 for
   every file. Its dataset card lists per-asset provenance and licensing.

2. **Released baseline checkpoints (for VLA evaluation).** These are the
   Hugging Face model repos `zimplex/harp-vla-train37-{act,diffusion,pi0,pi05,smolvla}`.
   Run [`scripts/vla_train37/fetch_ckpts.py`](scripts/vla_train37/fetch_ckpts.py)
   with the LeRobot environment's interpreter, which already has
   `huggingface_hub`:

   ```bash
   source scripts/env.sh
   "$LEROBOT_PY" scripts/vla_train37/fetch_ckpts.py                  # all five
   "$LEROBOT_PY" scripts/vla_train37/fetch_ckpts.py --models act,pi0 # a subset
   ```

   Each snapshot lands in `checkpoints/train37/<model>/` and is checked
   against its `SHA256SUMS`. The servers load
   `checkpoints/train37_eval/<model>/`, a view of symlinks plus JSON configs
   whose training-machine paths point at public Hugging Face model ids.
   `--root DIR` puts both trees somewhere other than `checkpoints/`.

3. **LeRobot source (for training or serving the baselines).**

   ```bash
   git submodule update --init --recursive   # baseline/lerobot
   ```

The LeRobot training datasets are not published as a download. To train,
record your own demos with the scripted experts; see
[portable data collection](docs/collect-data-portable.md).

## Quick start

```bash
source scripts/env.sh

# Collect one scripted demo (writes data/pour_water/ and a video).
python scripts/collect.py --task pour_water \
    --config config/latest_genesis/pour_water_no_avatar.yml --episodes 1

# Run the privileged scripted expert with NYX on one eval-set row
# (row 1176 is pour_water; rows are zero-based, --end-index is exclusive).
python scripts/eval_expert_set.py --eval-set eval_sets/eval_v3_sampled \
    --policy expert_full_state --start-index 1176 --end-index 1177
```

Without a usable GPU, add `--renderer rasterizer` and prefix the commands
with `GENESIS_BACKEND=cpu` (debugging only; see [Requirements](#requirements)).

Tasks are declared in `envs/tasks/registry.py` (`TASK_DESCRIPTORS`), which
projects the compatibility maps exported from `envs.tasks` (`TASK_MAP` for
canonical tasks, `TASK_ALIASES` for compatibility names).

## Evaluation

### Scripted experts on an eval set

`scripts/eval_expert_set.py` runs the `expert_full_state` (privileged) and
`expert` (blind to the human) policies on a fixed manifest. The release
evaluation set is `eval_sets/eval_v3_sampled`: 3,403 expert-solvable seeds
over the 37 benchmark tasks (see its [README](eval_sets/eval_v3_sampled/README.md)).

```bash
source scripts/env.sh
python scripts/eval_expert_set.py --eval-set eval_sets/eval_v3_sampled \
    --policy expert_full_state
```

- **Renderer.** Each row's `renderer` (NYX for `eval_v3_sampled`) is used
  unless you pass `--renderer {nyx,rasterizer}`, which overrides every row.
  If `GENESIS_BACKEND=cpu` while rows ask for NYX, the script prints a
  one-line notice that those rows fall back to the rasterizer.
- **One scene per process.** NYX can build only one scene per Python process,
  so when more than one row renders with NYX each row runs in a fresh
  subprocess (`--subprocess-per-episode auto`, the default).
- **Output.** Results go to `runs/expert_eval_set/<eval-set name>/<policy>/`
  (`results.jsonl`, `results.csv`, `summary.json`), or to `--output-dir`.
  The console summary reports success per paper condition (Cueing,
  Contribution, Interruption, Co-presence); `summary.json` keys keep the
  manifest `category` values.
- **Sharding.** The full set takes days on one GPU. `--start-index` and
  `--end-index` select a zero-based row range (end exclusive), and each
  shard writes to its own `.../shard_<start>_<end>/` directory, so shards
  can run in parallel on several GPUs or machines:

  ```bash
  for start in 0 500 1000 1500 2000 2500 3000; do
    python scripts/eval_expert_set.py --eval-set eval_sets/eval_v3_sampled \
        --policy expert_full_state --start-index $start --end-index $((start + 500))
  done
  ```

Other eval sets live under `eval_sets/`; `eval_v0_smoke` has one seed per task.
`scripts/generate_eval_set.py` builds new manifests and
`scripts/validate_eval_set_init.py` checks that every row resets.

### VLA baselines

- **One checkpoint, one task.** `scripts/run_vla_eval.sh` starts the LeRobot
  policy server for a checkpoint, waits for it, runs `scripts/eval_vla.py`,
  and stops the server:

  ```bash
  source scripts/env.sh
  ACT_SERVER_ENV=lerobot scripts/run_vla_eval.sh --model act \
      --checkpoint checkpoints/train37_eval/act --task pour_water --episodes 1
  ```

  `ACT_SERVER_ENV=lerobot` serves the released ACT checkpoint from
  `LEROBOT_PY`; without it ACT uses the legacy `PI0_PY` environment.

- **Released checkpoints on an eval set.** `scripts/vla_train37/` shards a
  manifest into per-model batch files, runs each batch with NYX rendering,
  and aggregates success and safety tables:

  ```bash
  source scripts/env.sh
  scripts/vla_train37/launch_full.sh eval_sets/eval_v3_sampled/manifest.jsonl <run_name> --local
  python scripts/vla_train37/summarize.py data/vla_train37_eval/<run_name>
  ```

See [VLA inference and control](docs/vla-inference-control-pipeline.md) for
running a subset of rows, per-model defaults and GPU memory needs.

## Data collection

`scripts/vla_data/collect_vla.py` records one expert episode per seed as
images, proprioception and actions (H5). `scripts/collect_vla.sh` wraps
`scripts/collect.py --record-stride` for the per-seed pickle layout under
`data/<task>/vla_raw/`. Convert recordings with
`scripts/build_genesis_hr_bench_rlds.py` (RLDS) and
`scripts/convert_genesis_hr_bench_to_lerobot.py` (LeRobot). See
[portable data collection](docs/collect-data-portable.md).
`scripts/vla_data/nyx_campaign.py` is an optional SLURM campaign runner for
large collections ([cluster guide](docs/cluster.md)).

## Baselines

The released baselines are LeRobot policies: **ACT**, **LeRobot Diffusion**,
**pi0**, **pi0.5**, and **SmolVLA**. They are trained with `scripts/finetune.sh`
(which dispatches to `finetune_lerobot.sh` for ACT/Diffusion and
`finetune_lerobot_v2.sh` for pi0/pi0.5/SmolVLA) and served by
[`baseline/servers/lerobot_server.py`](baseline/servers/lerobot_server.py);
the benchmark talks to the server through `scripts/vla_client.py`. Apply
`patches/lerobot-genesis-hr-bench.patch` to the `baseline/lerobot` submodule as
described in [VLA fine-tuning](docs/vla-finetune-pipeline.md).

Core versions (pinned in `requirements-genesis-latest.txt`):

- Python 3.10
- `torch==2.4.1+cu121`
- `genesis-world==1.2.0`
- `gs-nyx==0.1.3` / `gs-nyx-plugin==0.1.4`
- `numpy==1.26.4` for `mplib==0.2.1` compatibility
- `h5py==3.16.0` for VLA data recording

## Rendering

Genesis provides two camera paths here: **NYX** (default; GPU sensor renderer
that needs `GENESIS_BACKEND=gpu` on a Vulkan-capable GPU, and falls back to
the rasterizer when it cannot run) and the **rasterizer** (fast, also runs on
CPU). Switch with `--renderer`; see
[NYX rendering](docs/general/nyx_render_howto.md).

## Verify

Run the correctness linter and the complete regression suite (CPU only, a few
minutes):

```bash
source scripts/env.sh
GENESIS_BACKEND=cpu bash scripts/check_quality.sh
```

It exits 0 when Ruff and every test pass. Two known open failures in
`tests/test_take_handover_stance.py` are marked `xfail(strict=True)` (see
[release checks](docs/release_readiness.md)).

Native integration check and the reset/step smoke for every registered task
(the second takes hours):

```bash
source scripts/env.sh
GENESIS_BACKEND=cpu python scripts/check_latest_genesis_integration.py \
  --python "$MAWM_PY" --robots franka --tasks blocks_ranking_rgb --steps 1

GENESIS_BACKEND=cpu python scripts/probe_latest_genesis_robot_tasks.py \
  --all-registered --settle-steps 1 --timeout 300
```

A cold first kernel/JIT compile can exceed the default timeouts; retry with
larger `--robot-timeout` / `--task-timeout`. See
[release checks](docs/release_readiness.md) and the
[pinned runtime](docs/latest_genesis_status.md).

## Documentation

| Topic | Doc |
|---|---|
| Documentation index | [`docs/README.md`](docs/README.md) |
| Code layout and module ownership | [`docs/code_layout.md`](docs/code_layout.md) |
| Collect VLA training data (RLDS / LeRobot) | [`docs/collect-data-portable.md`](docs/collect-data-portable.md) |
| Fine-tune the LeRobot baselines | [`docs/vla-finetune-pipeline.md`](docs/vla-finetune-pipeline.md) |
| Serve and evaluate policies | [`docs/vla-inference-control-pipeline.md`](docs/vla-inference-control-pipeline.md) |
| Task registry and interaction architecture | [`docs/task_framework.md`](docs/task_framework.md) |
| GPU requirements for NYX | [`docs/general/gpu_resources.md`](docs/general/gpu_resources.md) |
| Running jobs, outputs and task rules | [`docs/general/running_jobs.md`](docs/general/running_jobs.md) |
| Optional: running on a SLURM cluster | [`docs/cluster.md`](docs/cluster.md) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
