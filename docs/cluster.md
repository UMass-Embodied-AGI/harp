# Running on a SLURM cluster (optional)

Everything in this repository runs on one machine (see the
[README](../README.md)). This page is for users who want to spread an eval
set, a data collection or a training run over a SLURM cluster. Partition
names, GPU feature names and bad nodes are cluster-specific: replace every
`<placeholder>` with your cluster's values, and put persistent values in
`scripts/env.local.sh` (copy [`scripts/env.local.sh.example`](../scripts/env.local.sh.example)).

## Ground rules

- Keep the login node light. Run Python (rollouts, tests, scene builds,
  training, data conversion) through `srun`/`sbatch`, not on a shared login
  node. Create the virtual environment from an interactive allocation too:
  `srun -p <partition> --gres=gpu:1 --mem=32G -c 4 -t 2:00:00 --pty bash`.
- Point both `-o` and `-e` at `job_logs/` (for example
  `-o job_logs/<job_name>_%j.log`) rather than the default
  `slurm-<id>.out` in the current directory.
- Bind thread counts to the allocation (`OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`)
  and cap job-array concurrency (for example `--array=0-99%10`).
- The checked-in `.sbatch` wrappers set only portable resources (GPU count,
  CPUs, memory, time, log paths). Add the partition, GPU constraint and any
  node exclusions on the `sbatch` command line, or through `sbatch`'s own
  `SBATCH_PARTITION` / `SBATCH_CONSTRAINT` environment variables.

## Picking GPUs for NYX

NYX needs the Genesis GPU backend on a Vulkan-capable (graphics) GPU; see
[GPU requirements](general/gpu_resources.md). On a cluster, this usually means
a `--constraint` over the graphics GPU types and an `--exclude` of nodes
whose driver cannot run Genesis or Vulkan:

```bash
--gres=gpu:1 --constraint='<nyx-gpu-constraint>' --exclude='<bad-nodes>'
```

Never put compute-only A100/H100/H200 types in an NYX constraint. Check a GPU
type (or a suspect node) with a one-episode NYX smoke before a large
submission:

```bash
source scripts/env.sh
sbatch -p <partition> --gres=gpu:1 --constraint='<gpu-type>' --mem=32G -c 4 -t 0:20:00 \
  -o job_logs/nyx_check_%j.log -e job_logs/nyx_check_%j.log \
  --wrap 'GENESIS_BACKEND=gpu srun "$MAWM_PY" scripts/collect.py --task pour_water --config config/latest_genesis/pour_water_no_avatar.yml --episodes 1 --save-dir data/nyx_check'
```

A pass writes an NYX-rendered episode under `data/nyx_check/`. A Vulkan
failure shows `Vulkan API: Failed to create the device` in the log.

## Source checks

```bash
source scripts/env.sh
sbatch -p <cpu-partition> -c 4 --mem=32G -t 1:00:00 -o job_logs/quality_%j.log \
  --wrap 'GENESIS_BACKEND=cpu bash scripts/check_quality.sh'
```

## Scripted experts on an eval set

[`scripts/eval_expert_set.sbatch`](../scripts/eval_expert_set.sbatch) runs
`scripts/eval_expert_set.py` with the arguments it is given, using
`GENESIS_BACKEND=gpu` unless you set another backend. Submit one job per shard
with `--start-index/--end-index`, and each shard writes to
`runs/expert_eval_set/<eval-set>/shard_<start>_<end>/`:

```bash
for start in $(seq 0 250 3400); do
  sbatch -p <partition> --constraint='<nyx-gpu-constraint>' \
    scripts/eval_expert_set.sbatch --eval-set eval_sets/eval_v3_sampled \
    --policy expert_full_state --start-index $start --end-index $((start + 250))
done
```

Each job requests 8 hours (`-t` changes it). Time a small shard first and
size shards so they finish within the limit. For a CPU-only rasterizer check, submit to a CPU partition
with `GENESIS_BACKEND=cpu sbatch ... --renderer rasterizer`.
[`scripts/validate_eval_set_init.sbatch`](../scripts/validate_eval_set_init.sbatch)
wraps the reset-only manifest check in the same way.

## VLA baselines

- One checkpoint: [`scripts/eval.sbatch`](../scripts/eval.sbatch) wraps
  `scripts/run_vla_eval.sh`, for example
  `sbatch -p <partition> --constraint='<nyx-gpu-constraint>' scripts/eval.sbatch act checkpoints/train37_eval/act --task pour_water`.
- Released checkpoints on an eval set:
  [`scripts/vla_train37/launch_full.sh`](../scripts/vla_train37/launch_full.sh)
  builds the batch files and prints (default) or submits (`--submit`) one
  [`run_batch.sbatch`](../scripts/vla_train37/run_batch.sbatch) array per
  model. Configure it in `scripts/env.local.sh`:

  ```bash
  T37_SBATCH_ARGS="-p <partition> --exclude=<bad-nodes>"
  NYX_GPU_CONSTRAINT="<nyx-gpu-constraint>"                 # every NYX GPU type
  NYX_LARGE_GPU_CONSTRAINT="<nyx-gpus-with-at-least-16gb>"  # pi0 / pi05
  NYX_LARGE_MULTI_GPU_CONSTRAINT="<nyx-gpus-with-about-48gb>"  # pi0 / pi05 at PARALLEL>=2
  NYX_SMALL_GPU_FEATURES="<feature-of-an-11gb-gpu>"         # refused for pi0 / pi05
  ```

  Unset constraints are omitted. Other knobs are environment variables of
  `launch_full.sh`: `PER_BATCH` (episodes per array task, default 37),
  `PARALLEL` (episodes per GPU, default 2), `CONC_PER_MODEL` (array
  concurrency per model, default 2, so at most 10 GPU jobs for 5 models),
  `JOB_TIME` (default `20:00:00`) and `MODELS`.
  [`scripts/vla_train37/submit.sh`](../scripts/vla_train37/submit.sh)
  submits a single batch directory with the same constraint logic. At
  runtime, `run_batch.py` refuses pi0/pi0.5 on a GPU under 16 GB whatever
  constraint was used.

## Data collection campaigns

[`scripts/vla_data/nyx_campaign.py`](../scripts/vla_data/nyx_campaign.py) runs
a bounded, resumable NYX collection over SLURM: its manifest names the
constraint, the node exclusions and the per-job resources. A job array over
`scripts/vla_data/collect_vla.py` is the simpler alternative; see
[portable data collection](collect-data-portable.md#42-parallel--slurm-cluster).

## Fine-tuning

`scripts/finetune.sh --slurm` builds the `sbatch` command for
[`scripts/finetune.sbatch`](../scripts/finetune.sbatch) from `FT_PARTITION`,
`FT_QOS`, `FT_CONSTRAINT`, `FT_ACCOUNT` and `FT_GRES`, which are empty
(omitted) by default. Training does not render, so compute-only GPUs work for
it. See [VLA fine-tuning](vla-finetune-pipeline.md).
[`scripts/finetune_act.sbatch`](../scripts/finetune_act.sbatch) is a direct ACT
wrapper; add `-p <partition>` when you submit it.
