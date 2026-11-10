# Script entrypoints

Run the scripts locally in the
[pinned environment](../docs/general/running_jobs.md); the `.sbatch` files are
optional SLURM wrappers ([cluster guide](../docs/cluster.md)). Every launcher sources
`env.sh`, which reads machine-local overrides from `env.local.sh`
(copy `env.local.sh.example`).

| Area | Location | Responsibility |
| --- | --- | --- |
| Environment | `env.sh`, `env.local.sh.example` | Portable defaults and machine-local overrides |
| Scripted collection | `collect.py`, `collect_vla.sh`, `launch/rasterizer.sh` | Expert demo rollouts |
| Training collection | `vla_data/` | Per-seed recording (`collect_vla.py`), validation, NYX campaigns |
| Dataset conversion | `build_genesis_hr_bench_rlds.py`, `convert_genesis_hr_bench_to_lerobot.py`, `data_conversion/` | RLDS and LeRobot formats |
| Eval sets | `generate_eval_set.py`, `validate_eval_set_init.py` (`.sbatch`) | Build and check eval-set manifests |
| Expert evaluation | `eval_expert_set.py` (`.sbatch`) | Scripted experts on a fixed manifest |
| Policy evaluation | `eval.py`, `eval_vla.py`, `vla_client.py`, `run_vla_eval.sh`, `eval.sbatch` | Environment evaluation and model-server client |
| Released-checkpoint evaluation | `vla_train37/` | Fetch checkpoints, batch an eval set per model, run (locally or on SLURM), summarize |
| Training | `finetune.sh` (`finetune.sbatch`), `finetune_lerobot.sh`, `finetune_lerobot_v2.sh`, `finetune_act.sh` (`.sbatch`), `finetune_lerobot_diffusion.sh`, `_dist.sh` | LeRobot baseline training |
| Release checks | `check_quality.sh`, `check_latest_genesis_integration.py`, `probe_latest_genesis_*.py` | Source and native runtime gates |

`build_genesis_hr_bench_rlds.py` and `convert_genesis_hr_bench_to_lerobot.py`
delegate to `data_conversion/`. `finetune_lerobot.sh` (ACT, LeRobot Diffusion)
and `finetune_lerobot_v2.sh` (pi0, pi0.5, SmolVLA) target different LeRobot
versions; they are not interchangeable.

The model server lives in `../baseline/servers/`. Underscore-prefixed local
scripts and untracked launch experiments are not entrypoints. Logs belong
under `job_logs/`; generated media and run outputs under `data/` or `runs/`.
