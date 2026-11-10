# Documentation

Start with the [repository README](../README.md) for installation and a first
rollout.

## Evaluate, collect, and train

- [Portable data collection](collect-data-portable.md): scripted rollouts,
  RLDS and LeRobot conversion.
- [VLA fine-tuning](vla-finetune-pipeline.md): LeRobot baselines (ACT,
  LeRobot Diffusion, pi0, pi0.5, SmolVLA).
- [VLA inference and control](vla-inference-control-pipeline.md): model
  servers, clients, and policy evaluation.
- [Running jobs](general/running_jobs.md): resources, outputs and task rules.
- [Running on a SLURM cluster](cluster.md) (optional).

## Tasks, scenes, and rendering

- [Task framework](task_framework.md): task registry and interaction lifecycle.
- [Task inventory](factory_task_inventory.md), [object catalog](object_catalog.md),
  and [kitchen scene](kitchen_scene.md).
- [NYX rendering](general/nyx_render_howto.md) and
  [GPU requirements](general/gpu_resources.md).
- [Grasp helper](general/grasp_helper_howto.md) for task authors.

## Develop

- [Code map](code_layout.md): module ownership.
- [Pinned runtime](latest_genesis_status.md): environment and compatibility scope.
- [Release checks](release_readiness.md): source-quality and runtime gates.
- [Tests](../tests/README.md), [scripts](../scripts/README.md), and
  [configuration](../config/README.md).
