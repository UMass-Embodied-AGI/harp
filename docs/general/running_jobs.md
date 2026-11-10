# Running jobs

Operational rules for running code in this repository. They apply to all
tasks and scripts unless a doc explicitly overrides them. Commands are shown
as local commands on one machine; to use a SLURM cluster, wrap them as
described in [running on a SLURM cluster](../cluster.md).

## Resources

- A single-episode rollout needs about 4 CPU cores and 32 GB of RAM; scene
  builds of some tasks need 64 GB.
- NYX rendering needs the GPU backend on a Vulkan-capable GPU; see
  [GPU requirements](gpu_resources.md). CPU-backend runs
  (`GENESIS_BACKEND=cpu`) do not need a GPU.
- Bind thread counts to the cores you want to use (`OMP_NUM_THREADS`) and
  avoid "all cores" settings when several episodes run side by side.

## Genesis backend

With `GENESIS_BACKEND` unset, `BaseTask` selects GPU for NYX when CUDA is
available and CPU otherwise. `scripts/env.sh` defaults the release
environment to `gpu`. Set `GENESIS_BACKEND=cpu` for rasterizer/debug runs. NYX
requires `GENESIS_BACKEND=gpu`.

## Python environment

Use the pinned Genesis 1.2.0 environment created as documented in the root
README:

```
source scripts/env.sh
python ...        # or "$MAWM_PY" ... when MAWM_PY points outside the repo venv
```

Older Genesis releases have different gripper/contact physics; grasp and
contact behaviour must be verified on the pinned environment. Sanity check:
`envs.genesis_compat.is_latest_genesis()` returns `True` and the installed
version matches `requirements-genesis-latest.txt`.

Example:

```bash
GENESIS_BACKEND=cpu python scripts/collect.py --task pour_water --episodes 1 \
    --renderer rasterizer
```

### Scoring vs. debugging

`scripts/collect.py` (random seeds, default config) is a debugging harness,
not the benchmark score. The benchmark runs `scripts/eval_expert_set.py`
(scripted experts) or `scripts/eval_vla.py` (policies) over a fixed eval-set
manifest with the manifest's `config_overrides` — notably `eval_mode: true`,
`track_avatar_collision: true`, and `neutral_avatar_strict_layout: true`,
which change success/failure relative to a bare `collect.py` run.
`eval_sets/eval_v0_smoke/` is a one-episode-per-task smoke set.
`scripts/eval_expert_set.py --start-index/--end-index` splits an eval set into
shards that can run in parallel.

## Where to write logs and outputs

- All `*.log` files go under `job_logs/` (gitignored apart from `.gitkeep`);
  the launchers write server and batch logs there by default.
- Videos, images and episode data go under `data/<task-or-run>/` (gitignored).
  Temporary scratch goes under `data/tmp/`.
- Training runs and evaluation outputs go under `runs/` (gitignored).
- Do not create new top-level output folders.

## Task implementation rules

These rules apply to every task class under `envs/tasks/`.

### Robot arm: PD control only — no kinematic shortcuts

The robot arm must reach poses through real physics — the planned trajectory
plays through `set_arm_joints` (PD position targets) and the simulator's
contact dynamics. Do not:

- call `entity.set_qpos(...)` or `teleport_arm_joints(...)` to jump the arm to
  a target during task execution (setting the initial qpos at scene reset is
  fine);
- set `USE_SETQPOS=1` / `DEBUG_TELEPORT=1` / `DEBUG_BOUNDED_JOINT_PATHS=1`
  outside debugging. The eval and collection entry points refuse to run with
  any of them set (`envs/debug_switches.py`); `MPLIB_PLAN_DIAG=1` only adds
  planner log lines and is allowed;
- use `attach_to_gripper` / `gripper_tether` or any kinematic constraint that
  pins a held object to the gripper without contact dynamics.

The avatar's hand-attached items are a separate system and not subject to
this rule.

### Scripted expert path planning + IK: use mplib

Use mplib for both motion planning and inverse kinematics:

- `arm.planner.plan_path(current_qpos, target_pose7)` — RRT-Connect through
  `mplib.plan_pose`;
- `arm.planner.solve_ik(...)` — mplib IK.

Do not use `arm.planner.solve_ik_cartesian(...)` or any wrapper around
`entity.inverse_kinematics(...)`: Genesis's damped-least-squares IK returns the
closest reachable qpos with no convergence gate and ignores the planner
obstacle cloud. mplib's `plan_pose` can also relax the IK silently; FK-verify
the end qpos and retry with a different RRT seed rather than falling back to
Genesis IK.

VLA evaluation is a separate control mode: `scripts/eval_vla.py` selects the
current-qpos-seeded `genesis_ik` controller for high-rate single-step EE
actions. Do not interchange the two planner contracts.

### Avatar must be in the planner's obstacle cloud

Pass the avatar's current capsule point cloud into
`arm.planner.update_obstacles(...)` before each plan via
`_refresh_planner_obstacles`. If a motion snapshot blocks every IK candidate,
step the simulation forward, refresh obstacles and retry — don't drop the
avatar from the cloud.

## Reproducibility notes

When comparing two code versions for behaviour preservation, compare per-seed
episode records, not just success counts:

- CPU-backend rollouts are bit-reproducible only when nothing else runs on the
  machine; run both versions sequentially on an otherwise idle machine.
- Some Co-presence (`*_copresence`) tasks (`place_food_in_skillet_copresence`,
  `place_pair_of_shoes_copresence`, `categorize_objects_copresence`) vary between identical
  runs; compare those by success rate over many seeds.
- OMPL seeds its RNG once per process: `BaseTask.reset` calls
  `pymp.set_global_seed`, but episodes 2..N of a multi-episode process do not
  get a fresh planner seed. Compare like positions across runs (episode k of a
  sweep with episode k), not a sweep episode with an isolated rerun.
- The first planning process in a job can differ from later ones; repeat the
  control run (control → treatment → control) before attributing a
  difference to code.
- Sourcing `scripts/env.sh` twice keeps the first `REPO_ROOT`; drivers outside
  the repository must put their own repo root first on `sys.path`.
