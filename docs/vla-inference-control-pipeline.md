# VLA inference and control

Policies run as out-of-process HTTP servers; the benchmark talks to them from
the simulator environment through the clients in `scripts/vla_client.py`.

## Models and defaults

Every model is served by `baseline/servers/lerobot_server.py` and needs a
finetuned checkpoint (`--checkpoint`). Defaults in `scripts/run_vla_eval.sh`:

| `--model` | Server `--policy-type` | Port | Default action type | Server interpreter |
|---|---|---|---|---|
| `pi0` | `pi0` | 8767 | `qpos_abs` | `LEROBOT_PY` |
| `pi05` | `pi05` | 8768 | `qpos_abs` | `LEROBOT_PY` |
| `act` | `act` | 8775 | `qpos_abs` | `PI0_PY` (`ACT_SERVER_ENV=lerobot` for `LEROBOT_PY`) |
| `lerobot_diffusion` | `diffusion` | 8777 | `ee` | `LEROBOT_PY` |
| `smolvla` | `smolvla` | 8778 | `ee` | `LEROBOT_PY` |

`template` (zero actions) is a client-only example for writing new policies.
Pick the `--action-type` the checkpoint was trained on: `qpos_abs` for
absolute joint-target datasets, `ee` for EE-delta datasets.

## Run one evaluation

```bash
source scripts/env.sh
# on a Vulkan-capable NVIDIA GPU (NYX; see docs/general/gpu_resources.md):
scripts/run_vla_eval.sh --model pi0 \
    --checkpoint runs/pi0/finetune/<exp>/checkpoints/last/pretrained_model \
    --task pour_water --episodes 10 --max-steps 600 \
    -- --output-json runs/pi0/eval/pour_water.json
```

On a SLURM cluster, `scripts/eval.sbatch` wraps the same command; see the
[cluster guide](cluster.md#vla-baselines).

`run_vla_eval.sh` starts the server, waits for `/health`, runs
`scripts/eval_vla.py`, and stops the server on exit. Server logs go to
`$SERVER_LOG_DIR` (default `job_logs/`). Arguments after `--` go to
`eval_vla.py` (e.g. `--config`, `--vla-config`, `--start-seed`,
`--output-json`, `--video-dir`).

## Evaluate the released checkpoints on an eval set

`scripts/vla_train37/` evaluates the five released checkpoints (`act`,
`diffusion`, `pi0`, `pi05`, `smolvla`) on an eval-set manifest with NYX
rendering:

```bash
source scripts/env.sh
# 1. download and verify the checkpoints (checkpoints/train37/ + the
#    checkpoints/train37_eval/ view the servers load)
"$LEROBOT_PY" scripts/vla_train37/fetch_ckpts.py
# 2. build the batch files and run them one after another on this GPU
scripts/vla_train37/launch_full.sh <eval_set.jsonl> <run_name> --local
# 3. aggregate success and safety tables
python scripts/vla_train37/summarize.py data/vla_train37_eval/<run_name>
```

Without `--local`, `launch_full.sh` only builds the batches and prints the
SLURM commands; `--submit` submits them ([cluster guide](cluster.md#vla-baselines)).
Environment knobs: `MODELS` (default all five), `PER_BATCH` (episodes per
batch file, default 37), `PARALLEL` (concurrent episodes per GPU, one policy
server each, default 2) and `VIDEO` (default 1).

To run a few rows, build batch files for them and run `run_batch.py`
directly:

```bash
source scripts/env.sh
python scripts/vla_train37/make_batches.py \
    --rows eval_sets/eval_v3_sampled/manifest.jsonl \
    --run-dir data/vla_train37_eval/<run_name> --models act \
    --select pour_water:9000000 --per-batch 1
LEROBOT_PY="${T37_LEROBOT_PY:-$LEROBOT_PY}" "$NYX_PY" scripts/vla_train37/run_batch.py \
    data/vla_train37_eval/<run_name>/batches/act/b_0.json
```

The episode lands in
`data/vla_train37_eval/<run_name>/episodes/<model>/<task>/seed_<seed>/`
(`result.json`, `episode.log`, `config.yml`, video).

`NYX_PY` (the simulator interpreter for these runs) defaults to `MAWM_PY`,
which includes NYX when installed from `requirements.txt`; the policy server
runs `T37_LEROBOT_PY` if set, else `LEROBOT_PY`. Set either in
`scripts/env.local.sh`. Each episode runs in a fresh
`eval_vla.py` process with the row's `config_overrides`, the task's training
instruction (`train_instructions.json`), `--action-type qpos_abs`, and a
per-task step budget from `train_demo_steps.json`. `gpu_guard.py` keeps
pi0/pi05 off GPUs with less than 16 GB.

`summary.md` labels task rows with the paper's condition and task names
(e.g. "Co-presence: Dump Bin (`dump_bin_copresence`)") and adds a per-condition
success table (Cueing, Contribution, Interruption, Co-presence);
`summary.json` keys keep the code task names.

## Control path

Observations: head camera (`primary_camera`) and wrist camera
(`wrist_camera`) RGB, resized for the policy (`config/vla.yml`;
`image_pipeline: jpeg92_bilinear` reproduces the training-data JPEG/resize
path), plus proprio:

- `qpos` / `qpos_abs`: 7 measured arm joints + gripper (the gripper is not
  appended twice when `joint_state` already contains it);
- `ee`: `[x, y, z, roll, pitch, yaw, gripper_closed]`.

Actions:

- `qpos_abs`: absolute joint PD targets + gripper, applied for
  `action_substeps` simulator steps.
- `ee`: the model's EE delta `[dx, dy, dz, droll, dpitch, dyaw, gripper_closed]`
  is composed into an absolute target `[x, y, z, qw, qx, qy, qz, gripper_open]`.
  By default the delta is integrated on the previously **commanded** pose,
  not the lagged measured TCP; set `ee_delta_integration: measured` to change
  this. `envs/base_task.py` skips the IK solve when the requested pose is
  already within 1 mm / 1 mrad of the current pose, so no-op actions do not
  cause null-space drift.

VLA evaluation sets `planner_override: genesis_ik` for single-step EE servo
control. Scripted experts use mplib instead.

## LeRobot server

`lerobot_server.py` loads the saved `policy_preprocessor.json` /
`policy_postprocessor.json` when present (required for pi0/pi0.5, which refuse
to run without their tokenizer), feeds the training-time image keys
(`observation.image`, `observation.wrist_image`), keeps the action-chunk queue
server-side, and resets policy and processors when the client sends
`reset=true` at the start of each episode. Concurrent episodes must use
separate servers.

`patches/lerobot-genesis-hr-bench.patch` records the LeRobot submodule changes
used for training: a longer distributed init timeout, selected-episode
normalization stats for `--dataset.episodes`, and episode-aware sampler
boundaries.
