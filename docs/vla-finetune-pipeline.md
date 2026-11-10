# Baseline fine-tuning

All released baselines are LeRobot policies. One launcher,
`scripts/finetune.sh`, dispatches to the per-family drivers. It trains on the
GPUs of the current machine; `--slurm` submits the same run to a SLURM
cluster instead. Machine-specific paths (and the optional SLURM names) live in
the gitignored `scripts/env.local.sh`.

| Baseline (`--model`) | LeRobot version | Driver | Multi-node |
|---|---|---|---|
| `pi0`, `pi05`, `smolvla` | v0.5.2 (`lerobot` venv, `LEROBOT_PY`) | `finetune_lerobot_v2.sh` (accelerate) | ✅ |
| `act` | v0.1 (`pi0` venv, `PI0_PY`) | `finetune_act.sh` → `finetune_lerobot.sh` | single GPU |
| `lerobot_diffusion` | v0.1 (`pi0` venv, `PI0_PY`) | `finetune_lerobot_diffusion.sh` → `finetune_lerobot.sh` | single GPU |

`LEROBOT_TRAIN_PY` overrides the interpreter used by `finetune_lerobot.sh`.

---

## 0. TL;DR

```bash
git clone --recurse-submodules <repo-url> genesis-hr-bench
cd genesis-hr-bench
cp scripts/env.local.sh.example scripts/env.local.sh   # edit for your machine
# ... build venvs + datasets (sections 2 and 4) ...
scripts/finetune.sh --model smolvla --sanity                  # 1-GPU smoke test
scripts/finetune.sh --model pi0 --gpus-per-node 4 --full       # 4 GPUs, this machine
```

---

## 1. Get the code

```bash
git submodule update --init --recursive   # baseline/lerobot
```

The LeRobot source is the `baseline/lerobot` submodule;
`patches/lerobot-genesis-hr-bench.patch` holds the repository's changes to it.
The policy HTTP server is `baseline/servers/lerobot_server.py`.

---

## 2. Set up the environments

Each model family has its own venv because their `torch` / `transformers`
pins conflict:

| venv | Python | Used by |
|---|---|---|
| simulator (`MAWM_PY`) | 3.10 | dataset converters + HTTP eval client |
| `lerobot` | 3.12 | `pi0`, `pi05`, `smolvla` (and v0.5 ACT/Diffusion checkpoints) |
| `pi0` | per LeRobot 0.1 | `act`, `lerobot_diffusion` training and legacy ACT checkpoints |

Put the model environments under one parent and point `VLA_ENV_ROOT` at it;
`scripts/env.sh` derives `LEROBOT_PY` and `PI0_PY` from that root. `MAWM_PY` is
the pinned simulator environment from the repository README.

```bash
uv venv --python 3.12 "$ENVS/lerobot"
uv pip install --python "$ENVS/lerobot/bin/python" \
    -e "baseline/lerobot[smolvla,pi,training]" flask
"$ENVS/lerobot/bin/python" -c "import lerobot, scipy, accelerate; print('lerobot venv OK')"
```

The simulator venv additionally needs `requests`, `h5py`, `tensorflow` and
`tensorflow_datasets` for the converters and the HTTP client (see
[data collection](collect-data-portable.md#22-build-the-genesis-environment)).

---

## 3. Configure `scripts/env.local.sh`

```bash
cp scripts/env.local.sh.example scripts/env.local.sh
$EDITOR scripts/env.local.sh
```

Set either `VLA_ENV_ROOT=/path/to/envs` (holding `lerobot/` and `pi0/`) or the
interpreters individually (`MAWM_PY`, `LEROBOT_PY`, `PI0_PY`). For
`finetune.sh --slurm` only, also set the cluster knobs (all empty by default,
which omits the flag):

```bash
FT_PARTITION=<partition>          # sbatch -p
FT_CONSTRAINT="<gpu-constraint>"  # sbatch --constraint
FT_ACCOUNT=<account>              # sbatch -A
FT_GRES=gpu                       # sbatch --gres=${FT_GRES}:<gpus-per-node>
```

`scripts/finetune.sbatch` carries no static cluster names. Training does not
render, so any CUDA GPU with enough memory works, including A100/H100. See
the [cluster guide](cluster.md#fine-tuning).

---

## 4. Build the LeRobot dataset

Collect episodes and build the RLDS dataset first
([data collection](collect-data-portable.md)). Then:

```bash
# 1. RLDS -> LeRobot v2.1 (simulator venv; needs tensorflow + tensorflow_datasets)
$MAWM_PY scripts/convert_genesis_hr_bench_to_lerobot.py \
    --lerobot-naming --repo-id genesis-hr-bench/genesis_hr_bench_lerobot
DS=~/.cache/huggingface/lerobot/genesis-hr-bench
# 2. migrate v2.1 -> v3.0 for LeRobot v0.5.2 (lerobot venv)
cp -r "$DS/genesis_hr_bench_lerobot" "$DS/genesis_hr_bench_lerobot_v30"
$LEROBOT_PY -m lerobot.scripts.convert_dataset_v21_to_v30 \
    --repo-id genesis-hr-bench/genesis_hr_bench_lerobot_v30 \
    --root "$DS/genesis_hr_bench_lerobot_v30" --push-to-hub=false
rm -rf "$DS/genesis_hr_bench_lerobot_v30_old"
# 3. add quantile stats needed by pi0.5 (the trailing 403 hub-push error is harmless)
$LEROBOT_PY -m lerobot.scripts.augment_dataset_quantile_stats \
    --repo-id genesis-hr-bench/genesis_hr_bench_lerobot_v30 \
    --root "$DS/genesis_hr_bench_lerobot_v30" || true
```

`finetune_lerobot_v2.sh` defaults `DATASET_REPO_ID` to the v3.0 dataset;
`finetune_lerobot.sh` (ACT / Diffusion on LeRobot 0.1) defaults to the v2.1
`genesis-hr-bench/genesis_hr_bench_lerobot`. Override either with
`DATASET_REPO_ID`.

---

## 5. Run fine-tuning

**Local** (this machine's GPUs):

```bash
scripts/finetune.sh --model smolvla --sanity                 # 1 GPU
scripts/finetune.sh --model pi0 --gpus-per-node 2 --full      # 2 GPUs, this node
scripts/finetune.sh --model act --full                        # single-GPU ACT
```

**Optional: SLURM** (`--slurm`, VLA models only; submits `finetune.sbatch`):

```bash
scripts/finetune.sh --model <pi0|pi05|smolvla> --slurm \
    --nodes N --gpus-per-node M \
    [--time HH:MM:SS] [--mem 80G] [--cpus 8] \
    [--partition P] [--constraint C] [--account A] \
    [--sanity | --full] [--exp-name NAME] \
    [-- <extra lerobot_train args>]
```

For ACT on SLURM submit `scripts/finetune_act.sbatch --sanity|--full`.

- `--sanity` is a 200-step smoke run; `--full` is the real run.
- Anything after `--` is forwarded verbatim to the trainer.
- Full runs log to W&B project `<policy>_genesis_hr_bench`; set
  `WANDB_ENTITY` / `WANDB_PROJECT` to change the target.
- Multi-node: `finetune.sbatch` derives `MASTER_ADDR` from
  `SLURM_JOB_NODELIST` and exports `NNODES`/`GPUS_PER_NODE`;
  `scripts/_dist.sh` builds the `accelerate launch` prefix. Without SLURM, run
  `finetune.sh` on each node with `NNODES`, `GPUS_PER_NODE`, `MASTER_ADDR`,
  `MASTER_PORT` and `NODE_RANK` set.

---

## 6. Outputs

```
runs/<model>/finetune/<exp-name>/        training checkpoints
job_logs/finetune_<model>_%j.log         SLURM stdout (--slurm only)
job_logs/finetune_<model>_%j.err         SLURM stderr (--slurm only)
```

`<exp-name>` defaults to `<mode>_<slurm_job_id>` under SLURM and
`<mode>_<YYYYmmdd_HHMMSS>` locally. The SLURM job id is stable across
requeues, so a preempted job resumes from its last checkpoint (`--resume`).

---

## 7. Evaluation

See [VLA inference and control](vla-inference-control-pipeline.md). In short:

```bash
scripts/run_vla_eval.sh --model pi0 \
    --checkpoint runs/pi0/finetune/<exp>/checkpoints/last/pretrained_model \
    --task pour_water --episodes 10        # on an NYX-capable GPU
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `lerobot venv python not found` / missing interpreter | fix `VLA_ENV_ROOT` / `*_PY` in `env.local.sh` |
| `sbatch: invalid partition` / constraint errors | fix `FT_PARTITION` / `FT_CONSTRAINT` / `FT_GRES` |
| lerobot `--dataset.repo_id` not found | the LeRobot dataset isn't under `$HF_LEROBOT_HOME` (section 4) |
| job preempted | it resumes on requeue |
| multi-node job hangs at startup | check the nodes can reach `MASTER_ADDR:MASTER_PORT` and your NCCL settings |
