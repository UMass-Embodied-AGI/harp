# General Rules

`docs/general/` collects the repository-wide rules for running jobs and
changing task code. Read [`running_jobs.md`](running_jobs.md) before running
jobs, collecting data, or writing outputs.

Required reading:

- [`running_jobs.md`](running_jobs.md) — resources, backend, log and output
  locations, and task implementation rules.

Read when relevant:

- [`grasp_helper_howto.md`](grasp_helper_howto.md) — before adding or changing
  robot grasp logic in a task.
- [`nyx_render_howto.md`](nyx_render_howto.md) — the default NYX (`gs-nyx`)
  renderer: environment, fallback behavior, cameras, speed, and
  troubleshooting.
- [`gpu_resources.md`](gpu_resources.md) — which GPU types run NYX
  (A100/H100 do not) and how to check a GPU.
- [`../cluster.md`](../cluster.md) — optional: running on a SLURM cluster.

Task-specific implementation notes belong in `docs/<task-or-feature>.md`.
Generated recordings belong under `data/` and logs under `job_logs/`.
