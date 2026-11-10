# Using the NYX renderer

NYX (the `gs-nyx` / `gs-nyx-plugin` GPU sensor renderer) is the **default**
renderer for tasks, VLA data collection, and eval. This doc is the practical
how-to: which env to use, how the smart default behaves, what to watch out for.

> NYX here means the **`gs-nyx` camera-sensor renderer** — it adds GPU camera
> sensors (`scene.add_sensor(NyxCameraOptions(...))`) while the scene itself
> still uses `gs.renderers.Rasterizer()`. It is **not** the Madrona
> `BatchRenderer`, which this release does not pin or support.

## TL;DR

1. Run on a Vulkan-capable NVIDIA GPU with `GENESIS_BACKEND=gpu` (the
   `scripts/env.sh` default; see [GPU requirements](gpu_resources.md)).
2. Use the pinned release environment selected by `scripts/env.sh`; it installs
   the compatible NYX packages from `requirements.txt`.
3. NYX is already the default (`config/default.yml: renderer: nyx`), so you do
   **not** need `--renderer nyx`. Just run collect/eval as usual.

```bash
source scripts/env.sh
GENESIS_BACKEND=gpu python scripts/collect.py --task pour_water --episodes 1
```

On a compatible GPU this renders with NYX. When NYX is unavailable, the task
renderer can transparently fall back to the rasterizer (see "Smart default"
below); use `GENESIS_BACKEND=cpu` for an intentional CPU debug run.

## Release environment

The supported environment is the repo-local `.venv-genesis-latest` described
in the root `README.md`. `source scripts/env.sh` resolves it as `$MAWM_PY` and
allows a host-specific override using the checked-in
`scripts/env.local.sh.example` template.
The release pins Genesis 1.2.0, `gs-nyx` 0.1.3, and `gs-nyx-plugin` 0.1.4.
Older Genesis releases are not supported.

## Smart default (auto-fallback)

`renderer: nyx` is the default, but `BaseTask._resolve_renderer()` makes it a
*smart* default: when NYX cannot run, it transparently falls back to
`rasterizer` (with a warning) instead of crashing. Fallback triggers on:

- CPU backend / `GENESIS_BACKEND=cpu` (debug runs),
- no CUDA GPU available,
- a missing or incompatible `gs_nyx_plugin` installation.

So **CPU debug runs and environments without usable NYX keep working** on the
rasterizer. To force a specific renderer regardless, pass
`--renderer {rasterizer,nyx}` (collect/eval) or set `renderer:`
in the config.

Caveat: the fallback probe checks that the plugin is *compatible*; it cannot
catch a mid-render GPU/Vulkan crash. If NYX starts rendering and then dies (see
"Bad nodes"), that is a hard failure, not a fallback.

## Avatar (human) rendering

NYX **renders the animated avatar** (verified). The avatar is a latest-Genesis
visualization-only `KinematicEntity` whose skinned vertices are driven by
`set_vverts`; gs-nyx-plugin 0.1.x only streams dynamic vertices for FEM/PBD/MPM/
SPH entities, so without help the skin gets exported as a (non-existent) static
file mesh → `Failed to fetch mesh .unknown` → segfault. `envs/nyx_kinematic_patch.py`
(`apply_nyx_kinematic_patch()`, called from `NyxCameraManager.__init__`) fixes
this by streaming the avatar's custom verts as a dynamic mesh. So avatar tasks
(Cueing/Contribution/Interruption/Co-presence) render correctly under NYX. The avatar fix also
made the real animated `AvatarController` required (the old static fallback was
removed; `strict_avatar_init: true`).

## Articulated URDFs: use the NYX-safe load recipe

The gs-nyx-plugin scene exporter renders a `gs.morphs.URDF` entity as a
**sub-scene reference**: it stores a pointer to the original `.urdf` file
(`instance.subscene_uri`) and re-parses it at render time to rebuild the
link/joint tree, then animates each link from Genesis's per-link transforms.
This assumes a **1:1 mapping between the file's links and Genesis's simulated
links**.

Genesis's default `merge_fixed_links=True` *collapses* every fixed-jointed link
into its parent at load time, so the simulated body has fewer links than the
file declares. NYX can't map the file tree onto the collapsed set, so it bails
out at `scene.build()`:

```
RuntimeError: The Nyx Renderer does not support exporting an URDF entity
with merged fixed links: .../044_microwave/7221/mobility.urdf
```

(This is a plugin-side *build* error, raised after `_resolve_renderer()` has run,
so the smart default cannot pre-empt it.)

Setting `merge_fixed_links=False` prevents that build error, but it is not
sufficient for SAPIEN assets with a fixed root and many `<visual>` elements per
link: NYX can silently omit the fixed child or show only one visual, producing a
detached microwave door, flat cabinet panels, chair wheels without a chair, or
a laptop lid without its base.

Use `prepare_articulated_urdf_for_renderer()` for task-loaded articulated
assets. Under NYX it pre-merges fixed joints in the file, consolidates each
link into one textured GLB visual, bakes runtime scale into the URDF, and
returns `merge_fixed_links=False`. Other renderers keep their ordinary runtime
scale and Genesis defaults:

```python
path, scale, merge = prepare_articulated_urdf_for_renderer(
    path, renderer=self.config.get("renderer", ""), scale=scale,
)
urdf_kwargs = dict(file=str(path), pos=..., quat=..., scale=scale, fixed=True)
if merge is not None:
    urdf_kwargs["merge_fixed_links"] = merge
entity = self.scene.add_entity(gs.morphs.URDF(**urdf_kwargs))
```

This is used by `open_microwave` and all `put_object_cabinet` implementations.
Any new task that loads a multi-link `mobility.urdf` directly should use the
same helper rather than adding a task-local NYX condition.

**The Piper robot** (`piper_robot.py` loads its URDF with
`merge_fixed_links=True`) hits the same wall; the **Franka** robot is MJCF, so
all Franka tasks are unaffected (which is why the avatar tasks above render
without any change). Mesh/primitive objects (`gs.morphs.Mesh` / `Primitive`) are
never affected. Fallback if you don't want the un-merge: `--renderer rasterizer`.

## Known limitation: one NYX scene build per process

NYX initializes GPU/Vulkan global singletons on the **first** `scene.build()` in
a process. Building a **second** scene in the same process re-triggers that init
and asserts → segfault:

```
[ERROR][NYX][ASSERT_FAILURE] Memory Globals: Memory globals already initialized.
[ERROR][NYX][ASSERT_FAILURE] GraphicsAPI: The backend has already been set.
[ERROR][NYX][ASSERT_FAILURE] Singleton already initialized.
```

Genesis rebuilds the scene from scratch on **every** `reset()`,
so any driver that loops multiple episodes/tasks **in one process** with NYX will
segfault on the 2nd `reset()` (episode 1 renders fine, episode 2 segfaults at
build).

**Workaround: one NYX scene build per process.** Run one episode/task per Python
invocation (loop in the shell or scheduler, not in Python). For `collect.py` /
`eval.py`, use `--episodes 1` per invocation. `scripts/eval_expert_set.py`
runs each NYX row in a fresh subprocess automatically, and
`scripts/vla_train37/run_batch.py` starts one `eval_vla.py` process per
episode. This process
isolation is unrelated to the `merge_fixed_links` URDF issue above.

## HDRI background (EXR environment map)

By default a NYX scene has no background (grey void + checkered floor). Set
`nyx.env_texture` to an equirectangular HDRI and NYX uses it as both the
**background** and **image-based lighting** (the HDRIs ship in `assets/exr/`):

```yaml
nyx:
  spp: 1
  denoise: false
  env_texture: "assets/exr/brown_photostudio_02_4k.exr"
  # env_rotation: 0.0      # spin about the up axis, degrees
  # env_multiplier: 1.0    # radiance scale (brightness)
```

Notes:

- When `env_texture` is set, the ground plane + checkered floor **visuals are
  hidden automatically** (`BaseTask._hide_ground_visuals`) so the HDRI's own
  ground shows under the table; the collision plane stays active.
- Measured cost: **zero** steady-state render overhead (spp1 and spp4+denoise
  identical ms/frame); one-time scene build grows ~4–9 s per process for the
  4K EXR load.
- Implementation: `NyxCameraManager` pops the three `env_*` keys from the
  `nyx:` block and attaches one `EnvironmentMapAsset` to the **first** sensor
  only (the exporter concatenates env maps across all sensors, so per-sensor
  attachment would stack the map 5×).

### Per-EXR background offsets

NYX treats an environment map as infinite-distance lighting. It has no XYZ
position and therefore cannot be translated like a mesh. Use `env_offset` for the visual operation normally
wanted here: shifting photographed furniture or the horizon away from the
simulated task table.

```yaml
nyx:
  env_texture: random
  # Global [horizontal_deg, vertical_deg], positive = image-right / image-up.
  env_offset: [0.0, 0.0]
  # Applied after random selection and added to the global offset. Keys may be
  # configured paths, filenames, or stems.
  env_offsets:
    abandoned_garage_4k.exr: [12.0, 8.0]
    university_workshop_4k.exr: [-20.0, -5.0]
```

Offsets are measured in panorama degrees rather than pixels, so they are
resolution-independent. Nonzero variants are generated once under
`__nyx_cache__/env_offsets/`; source EXRs remain unchanged. The horizontal
component is a wrapped equirectangular shift. The vertical component is also an
image shift (not a physical camera translation), which is intentional for
removing background-table overlap.

## Cameras + output parity

NYX provides the **same 5-camera set** as the rasterizer:
`head_camera`, `left_wrist`, `right_wrist`, `recording`, `side`. The wrist
cameras track the end-effector each step (same RealSense-D435 mount as the
rasterizer). `get_obs()` returns the usual `rgb` / `depth` /
`camera_intrinsics` / `camera_extrinsics` dicts keyed by camera name, so VLA
collection (which consumes `image_primary` + `image_wrist`) works unchanged.

Ordinary scripted-demo videos use a separate, configurable two-camera pair.
The default `demo_video_cameras` block maps `recording` to the tuned
three-quarter-left pose and `side` to the tuned three-quarter-right pose; the
saved MP4 concatenates them left-to-right. The straight view behind the robot
is not included. Enabling the pair guarantees both panes for demos, even when
a legacy task config has `side_video: false`. This override is automatically
disabled when `record_stride`
is set or `vla_recording.enabled` is true, so VLA camera poses are unaffected.
Set `demo_video_cameras.enabled: false` to restore each task's authored
recording/side poses.

## Config

NYX options live under the `nyx:` block (defaults in `config/default.yml`):

```yaml
renderer: nyx
nyx:
  spp: 1            # samples per pixel (single-bounce ray tracer)
  denoise: false
  open_window: false
```

NYX is a **single-bounce ray tracer**, so `spp: 1` + `denoise: false` gives
**grainy** output. If you need cleaner images, raise `spp` and/or set
`denoise: true` (slower). Any keys under `nyx:` are passed to `NyxCameraOptions`
— keep them compatible with the installed `gs-nyx-plugin`.

## Speed

NYX render is fast (it is a normal GPU sensor renderer). Per-frame render cost
@640x480, spp=1 (steady state, excludes the one-time scene build):

| GPU | render ms | FPS |
|---|---|---|
| A16 | 2.6 | ~380 |
| L4  | 2.9 | ~350 |
| RTX 8000 | 3.5 | ~280 |
| A4000 | 4.1 | ~245 |

There is a one-time **scene build** cost (renderer setup + cache warmup,
~1–4 min depending on scene/GPU) paid once per process; it can recompile on a
fresh cache dir or a new GPU arch.

## Machines that fail NYX

Some machines fail Genesis or NYX because of driver/PTX/Vulkan problems even
with a supported GPU model; see [`gpu_resources.md`](gpu_resources.md). On a
cluster, exclude such nodes ([cluster guide](../cluster.md)).

## Building or refreshing the environment

Create the repo-local `.venv-genesis-latest` from `requirements.txt` as shown
in the root `README.md`. Do not upgrade Genesis, torch, or the NYX packages
independently: their compatible versions are pinned as one release contract.
After installation, verify NYX on a graphics-capable GPU:

```bash
source scripts/env.sh
GENESIS_BACKEND=gpu python scripts/collect.py \
  --task pour_water --config config/latest_genesis/pour_water_no_avatar.yml \
  --episodes 1 --save-dir data/nyx_smoke
```

## Troubleshooting

- **Renders look grainy** — expected at `spp: 1`; raise `spp` / set
  `denoise: true`.
- **`renderer='nyx' unavailable (...); falling back to 'rasterizer'`** — you are
  on CPU, on a non-GPU node, or using an environment without the compatible
  plugin. Use a graphics-capable GPU, `GENESIS_BACKEND=gpu`, and `$MAWM_PY`
  from `scripts/env.sh`.
- **`Vulkan API: Failed to allocate buffer memory` / segfault** — a compute-only
  GPU or a broken driver on this machine; see [`gpu_resources.md`](gpu_resources.md).
- **Segfault on the 2nd episode + `Memory globals already initialized` /
  `backend has already been set`** — NYX only supports one scene build per
  process. Run one episode/task per process and loop externally. See "Known
  limitation: one NYX scene build per process" above.
- **`NyxCameraData.__new__() missing 'rgb'` or `__init__() takes 4 ... 6 given`**
  — the NYX plugin does not match Genesis. Recreate the pinned release
  environment instead of patching one package in place.
- **`does not support exporting an URDF entity with merged fixed links`** — a
  task-loaded articulated asset bypassed
  `prepare_articulated_urdf_for_renderer()`, or the task uses a URDF robot that
  still merges links. Use the shared helper for task assets; otherwise use
  `--renderer rasterizer`. See the articulated-URDF section above.
- **Want plain rasterizer** — `--renderer rasterizer` (or run on CPU; it falls
  back automatically).

## References

- Code: `envs/base_task.py` (`_resolve_renderer`, `_nyx_renderer_available`),
  `envs/camera.py` (`NyxCameraManager`, wrist-camera support).
- Original team doc: `docs/nyx_render.md`.
