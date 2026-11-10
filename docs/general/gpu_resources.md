# Which GPUs can run NYX?

NYX, the default renderer and the one every `eval_v3_sampled` row uses, has
two independent requirements. A GPU must meet both.

## 1. The Genesis GPU backend needs a modern architecture

NYX runs on the Genesis GPU backend (`GENESIS_BACKEND=gpu`), which needs a
Turing-or-newer NVIDIA GPU (compute capability 7.5+) and a working CUDA
driver. Older cards such as GTX 1080 Ti (sm_61), Titan X and M40 (sm_52)
cannot run it. V100 (sm_70) sits at the floor and has not been measured with
the release stack.

## 2. NYX needs a graphics-capable GPU (Vulkan)

NYX renders through Vulkan. Data centre compute parts (**A100, H100, H200**)
have no graphics engine, so Vulkan cannot create a device and NYX dies at
scene build with:

```
[ERROR][NYX][ASSERT_FAILURE] Vulkan API: Failed to create the device.
[Vulkan Loader] ERROR: vkGetDeviceQueue: Invalid device
  ...or...
[ERROR][NYX][ASSERT_FAILURE] Vulkan API: Failed to allocate buffer memory
```

Compute capability is not the criterion: RTX 8000 and RTX 2080 Ti (sm_75)
render fine, while A100 (sm_80) and H100 (sm_90) fail. What matters is whether
the card has a render engine.

## Measured GPU types

Each type was checked with a one-episode NYX render of `pour_water`:

| GPU | Memory | Compute capability | NYX |
|---|---|---|---|
| RTX 2080 Ti | 11 GB | 7.5 | works |
| RTX 8000 | 48 GB | 7.5 | works |
| RTX A4000 | 16 GB | 8.6 | works |
| A16 | 16 GB | 8.6 | works |
| A40 | 48 GB | 8.6 | works |
| L4 | 24 GB | 8.9 | works |
| L40S | 48 GB | 8.9 | works |
| V100 | 16/32 GB | 7.0 | not measured |
| A100 | 40/80 GB | 8.0 | fails (compute-only, no Vulkan) |
| H100 / H200 | 80+ GB | 9.0 | fails (compute-only, no Vulkan) |
| GTX 1080 Ti, Titan X, M40 | | 5.2-6.1 | fails (architecture too old) |

Other graphics-capable Turing-or-newer cards (consumer RTX, RTX A-series,
L-series) are expected to work but have not been measured.

Memory for the VLA baselines: one NYX episode plus a pi0 or pi0.5 policy
server needs about 10-11 GB, so pi0/pi0.5 need a GPU with at least 16 GB;
`scripts/vla_train37/run_batch.py` refuses smaller ones. ACT, LeRobot
Diffusion and SmolVLA fit on an 11 GB card.

Individual machines can also fail through their driver rather than their GPU
model (CUDA PTX JIT errors or a Vulkan loader without the NVIDIA ICD). If one
host fails while others with the same GPU work, check its driver and
`vulkaninfo`.

## Don't need NYX? Use the rasterizer

For debug and expert-only work, the rasterizer on the CPU backend
(`GENESIS_BACKEND=cpu --renderer rasterizer`) runs anywhere. NYX is a smart
default: it falls back to the rasterizer when it cannot run (CPU backend, no
CUDA, an incompatible plugin). Note that the CPU and GPU backends are
different physics runs, so CPU rasterizer rollouts are not the reference NYX
results. See [`nyx_render_howto.md`](nyx_render_howto.md).

## How to check a GPU

```bash
source scripts/env.sh
GENESIS_BACKEND=gpu python scripts/collect.py --task pour_water \
  --config config/latest_genesis/pour_water_no_avatar.yml --episodes 1 \
  --save-dir data/nyx_check
```

A pass writes an NYX-rendered episode under `data/nyx_check/`. A warning
`renderer='nyx' unavailable (...); falling back to 'rasterizer'` or a Vulkan
error means this GPU or driver cannot run NYX. To check GPU types on a
cluster, see [running on a SLURM cluster](../cluster.md#picking-gpus-for-nyx).
