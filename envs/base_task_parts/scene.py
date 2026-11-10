"""Scene setup: table, robot mount, cameras, avatar and object loading.

Split from BaseTask (verbatim member move); composed back into it in
envs/base_task.py.
"""


import os
import zlib
import numpy as np
import torch
import genesis as gs
from copy import deepcopy
from pathlib import Path

from ..utils import Pose, create_primitive, ASSETS_PATH
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..table_assets import add_table_variant, table_randomization_enabled
from ..robot import PiperRobot, FrankaRobot, StretchRobot, XArm7Robot, ARXX5Robot, UR5WSGRobot
from ..camera import Camera, NyxCameraManager
from ..exr_envmap import normalize_env_offset
from ..avatar.model_list import DEFAULT_AVATAR_POOL

from .shared import DEFAULT_RENDERER, TABLE_HEIGHT



def isolate_add_entity_rng(scene) -> None:
    """Make ``scene.add_entity`` leave the global ``np.random`` stream untouched.

    Asset loading can consume global draws depending on the renderer and on
    on-disk mesh caches (e.g. a put_object_cabinet_neutral seed
    9000095 drew a stapler on the rasterizer and a cube on NYX with the same
    config, because adding the cabinet URDF consumed draws only on the
    rasterizer path).  Every task-level draw is made by task code with
    already-decided parameters, so restoring the state after each call keeps
    a seed's scene identical across renderers, backends and cache states.
    """
    if getattr(scene, "_global_rng_isolated", False):
        return
    orig = scene.add_entity

    def add_entity(*args, **kwargs):
        state = np.random.get_state()
        try:
            return orig(*args, **kwargs)
        finally:
            np.random.set_state(state)

    scene.add_entity = add_entity
    scene._global_rng_isolated = True

class _SceneSetupMixin:
    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Initialize Genesis and create the scene."""
        renderer_type = self._resolve_renderer()

        renderer = self._build_renderer()

        scene_kwargs = dict(
            show_viewer=self.config.get("show_viewer", False),
            sim_options=gs.options.SimOptions(dt=0.002),
        )
        if renderer is not None:
            scene_kwargs["renderer"] = renderer
        if self.RIGID_OPTIONS:
            scene_kwargs["rigid_options"] = gs.options.RigidOptions(**self.RIGID_OPTIONS)

        self.scene = gs.Scene(**scene_kwargs)
        if bool(self.config.get("isolate_asset_load_rng", True)):
            isolate_add_entity_rng(self.scene)
        if renderer_type in ("batch", "madrona"):
            self._add_batch_render_lights()
        floor_center = np.asarray(self.config.get("floor_center", [0.0, 0.0, 0.0]), dtype=float)
        self.scene.add_entity(
            gs.morphs.Plane(
                pos=tuple(floor_center[:3]),
                visualization=not self._hide_ground_visuals(),
            )
        )
        self._create_floor()

    def _hide_ground_visuals(self) -> bool:
        """Whether the ground plane/floor visuals should be hidden.

        True when the renderer supplies its own environment ground — the
        raytracer's emissive env sphere, or nyx with an HDRI env map
        (``nyx.env_texture``) — so the local floor would only occlude it.
        Collision is unaffected; only the visuals are dropped.
        """
        renderer_type = self.config.get("renderer", DEFAULT_RENDERER).lower()
        if renderer_type == "raytracer":
            return True
        if renderer_type == "nyx" and (self.config.get("nyx") or {}).get("env_texture"):
            return True
        return False

    def _resolve_renderer(self):
        """Init Genesis backend and resolve the effective renderer.

        nyx is the default renderer, but it needs a CUDA GPU and a working
        gs_nyx_plugin. This is a *smart* default: when nyx can't run here
        (CPU/debug runs, GENESIS_BACKEND=cpu, no GPU, or an env whose
        gs_nyx_plugin is incompatible) it transparently falls back to the
        rasterizer so nothing breaks. batch/madrona stay explicit opt-in and
        still hard-error without CUDA. Returns the effective renderer string
        and rewrites ``config['renderer']`` so all downstream reads agree.
        """
        import warnings
        requested = self.config.get("renderer", DEFAULT_RENDERER).lower()

        # Backend decision. nyx/batch/madrona want the GPU; for the nyx default
        # we only force GPU if one is actually usable and the user hasn't pinned
        # CPU — otherwise nyx falls back to rasterizer below (rasterizer is fine
        # on CPU). GENESIS_BACKEND=gpu always forces GPU.
        env_backend = (os.environ.get("GENESIS_BACKEND") or "").lower()
        cpu_pinned = env_backend == "cpu"
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        want_gpu = (
            env_backend == "gpu"
            or requested in ("batch", "madrona")
            or (requested == "nyx" and cuda_available and not cpu_pinned)
        )
        try:
            gs.init(backend=(gs.gpu if want_gpu else gs.cpu), logging_level="error")
        except Exception:
            pass  # already initialized

        # Smart default: fall back nyx -> rasterizer when nyx can't run.
        if requested == "nyx":
            reason = None
            if gs.backend != gs.cuda:
                reason = (f"backend={gs.backend!r} (nyx needs CUDA; set "
                          "GENESIS_BACKEND=gpu on a GPU node)")
            else:
                ok, why = self._nyx_renderer_available()
                if not ok:
                    reason = why
            if reason is not None:
                warnings.warn(
                    f"renderer='nyx' unavailable ({reason}); falling back to "
                    "'rasterizer'. Use a GPU node with a working gs_nyx_plugin "
                    "to enable nyx.")
                self.config["renderer"] = "rasterizer"
                requested = "rasterizer"

        # batch/madrona remain explicit opt-in: hard error without CUDA.
        if requested in ("batch", "madrona") and gs.backend != gs.cuda:
            raise RuntimeError(
                f"renderer={requested!r} requires the CUDA backend, but Genesis "
                f"initialized with backend={gs.backend!r}. Run on a GPU node "
                "(sm_70+) — e.g. allocate --gres=gpu:1."
            )
        # Always write the EFFECTIVE renderer back, not just on the fallback
        # path: a task built from a bare dict (no config file) resolves to
        # DEFAULT_RENDERER but left config["renderer"] unset, so downstream reads
        # (and anything logging the config) saw None while nyx was actually used.
        self.config["renderer"] = requested
        return requested

    @staticmethod
    def _nyx_renderer_available():
        """Cheaply probe whether gs_nyx_plugin works with the installed Genesis.

        Genesis resolves a sensor's ``_metadata_cls`` from its generic params;
        a plugin/Genesis version mismatch resolves it to the wrong class so
        instantiating it raises (e.g. gs_nyx_plugin 0.1.2 vs Genesis 1.2.0).
        This catches both the missing-plugin and incompatible-plugin cases
        without building a scene. Must be called after gs.init().
        """
        try:
            from gs_nyx_plugin import nyx_camera_sensor as _ncs  # noqa: F401
            _ncs.NyxCameraSensor._metadata_cls()  # raises on version mismatch
            return True, None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _build_renderer(self):
        """Build renderer from config. Returns None for default rasterizer."""
        renderer_type = self.config.get("renderer", DEFAULT_RENDERER).lower()
        if renderer_type == "rasterizer":
            return gs.renderers.Rasterizer()
        elif renderer_type == "nyx":
            return gs.renderers.Rasterizer()
        elif renderer_type in ("batch", "madrona"):
            # Madrona batch renderer (GPU-only). Renders all cameras in a
            # single CUDA call; n_envs stays 0 (1 world) in this Phase-1
            # drop-in.
            #
            # use_rasterizer defaults True: Madrona's built-in rasterizer skips
            # the ray-tracer BVH kernels, which fail to load on sm_80/86/90
            # (A100/A40/H100) with CUDA_ERROR_NO_BINARY_FOR_GPU. The rasterizer
            # path is also the right semantics for VLA images (pyrender is a
            # rasterizer too). Override with batch_render.use_rasterizer: false
            # only on an arch where the RT kernels are known to load.
            br_cfg = self.config.get("batch_render", {}) or {}
            return gs.renderers.BatchRenderer(
                use_rasterizer=bool(br_cfg.get("use_rasterizer", True)),
            )
        elif renderer_type == "raytracer":
            rt_cfg = self.config.get("raytracer", {})
            rt_kwargs = {}
            for key in ("tracing_depth", "rr_depth", "rr_threshold",
                        "env_radius", "env_pos", "env_euler",
                        "normal_diff_clamp", "logging_level"):
                if key in rt_cfg:
                    rt_kwargs[key] = rt_cfg[key]
            if "lights" in rt_cfg:
                rt_kwargs["lights"] = rt_cfg["lights"]
            if rt_cfg.get("env_texture"):
                env_tex_path = rt_cfg["env_texture"]
                # Resolve relative paths against project root
                if not os.path.isabs(env_tex_path):
                    env_tex_path = str(ASSETS_PATH.parent / env_tex_path)
                rt_kwargs["env_surface"] = gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path=env_tex_path,
                    ),
                )
            return gs.renderers.RayTracer(**rt_kwargs)
        else:
            raise ValueError(
                f"Unknown renderer: {renderer_type}. Use 'rasterizer', 'raytracer', 'batch', or 'nyx'."
            )

    def _add_batch_render_lights(self):
        """Add lights for the Madrona batch renderer.

        Unlike the rasterizer/raytracer (which read VisOptions / sphere
        lights), the batch renderer renders black unless lights are added
        via ``scene.add_light`` before ``scene.build()``. Defaults give a
        bright, mostly top-down key light plus a softer fill so the tabletop
        scene reads similarly to the pyrender rasterizer output. Override
        with ``config['batch_render']['lights']`` (list of dicts matching the
        ``scene.add_light`` kwargs).
        """
        br_cfg = self.config.get("batch_render", {}) or {}
        lights = br_cfg.get("lights")
        if not lights:
            lights = [
                # Key light: directional, from above and slightly front.
                dict(pos=(0.0, 0.0, 3.0), dir=(0.2, 0.3, -1.0),
                     color=(1.0, 1.0, 1.0), intensity=4.0,
                     directional=True, castshadow=True, cutoff=45.0, attenuation=0.0),
                # Fill light: softer, from the opposite side, no shadow.
                dict(pos=(0.0, 0.0, 3.0), dir=(-0.3, -0.2, -1.0),
                     color=(1.0, 1.0, 1.0), intensity=1.5,
                     directional=True, castshadow=False, cutoff=45.0, attenuation=0.0),
            ]
        for light in lights:
            self.scene.add_light(**light)

    def _setup_batch_render_cache(self):
        """Point Madrona at a PER-GPU-ARCH kernel cache dir before scene.build().

        Madrona compiles a GPU megakernel at runtime (NVRTC) and caches the
        cubin under ``MADRONA_ROOT_CACHE_DIR`` (default ``~/.cache/madrona``).
        That cubin is arch-specific: an sm_89 (L40S) cubin loaded on an sm_80
        (A100) device fails with CUDA_ERROR_NO_BINARY_FOR_GPU. The default
        single shared dir therefore POISONS across mixed-GPU clusters. Keying
        the cache dir by the running GPU's compute capability isolates them so
        each arch keeps (and reuses) its own valid cubin.

        gs_madrona's ``renderer_gs`` sets ``MADRONA_ROOT_CACHE_DIR`` at import
        time, so we import it first (best-effort) then override — the C++ reads
        the env var live at build. No-op unless renderer=batch.
        """
        if self.config.get("renderer", DEFAULT_RENDERER).lower() not in ("batch", "madrona"):
            return
        try:
            import os
            import torch
            try:
                import gs_madrona.renderer_gs  # noqa: F401  (triggers its env default)
            except Exception:
                pass
            if not torch.cuda.is_available():
                return
            cc = torch.cuda.get_device_capability(0)
            arch = f"sm{cc[0]}{cc[1]}"
            override = self.config.get("batch_render", {}).get("cache_dir")
            cache_dir = override or os.path.expanduser(f"~/.cache/madrona_{arch}")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["MADRONA_ROOT_CACHE_DIR"] = cache_dir
        except Exception:
            pass  # cache keying is an optimization; never block scene build

    table_offset = np.array([0.0, 0.0])  # (dx, dy) override in subclass to shift table
    recording_camera_pos = None    # override in subclass: [x, y, z]
    recording_camera_lookat = None  # override in subclass: [x, y, z]
    side_camera_pos = None         # override in subclass: [x, y, z]
    side_camera_lookat = None      # override in subclass: [x, y, z]

    def _demo_video_camera_pair_enabled(self) -> bool:
        """Whether this run is an ordinary demo using the tuned camera pair."""
        demo_cfg = self.config.get("demo_video_cameras", {}) or {}
        vla_cfg = self.config.get("vla_recording", {}) or {}
        return bool(demo_cfg.get("enabled", False)) and not (
            self.config.get("record_stride") is not None
            or bool(vla_cfg.get("enabled"))
        )

    def _resolve_video_camera_specs(self):
        """Resolve the two cameras used only for rendered demonstration video.

        Task-authored ``recording_camera_*`` / ``side_camera_*`` poses remain
        the fallback and the VLA contract.  For ordinary demo collection, the
        optional ``demo_video_cameras`` config replaces them with a consistent
        left/right pair.  VLA collection is identified either by trajectory
        ``record_stride`` or by the HDF5 ``VLARecorder`` being enabled.
        """
        recording = {
            "position": self.recording_camera_pos,
            "lookat": self.recording_camera_lookat,
            "fov": 60.0,
        }
        side = {
            "position": self.side_camera_pos,
            "lookat": self.side_camera_lookat,
            "fov": 60.0,
        }

        demo_cfg = self.config.get("demo_video_cameras", {}) or {}
        if not self._demo_video_camera_pair_enabled():
            return recording, side

        def _read_view(name):
            raw = demo_cfg.get(name, {}) or {}
            position = np.asarray(raw.get("position"), dtype=float).ravel()
            lookat = np.asarray(raw.get("lookat"), dtype=float).ravel()
            fov = float(raw.get("fov", 40.0))
            if position.size != 3 or lookat.size != 3:
                raise ValueError(
                    f"demo_video_cameras.{name} requires 3D position and lookat"
                )
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(lookat)):
                raise ValueError(f"demo_video_cameras.{name} pose must be finite")
            if not np.isfinite(fov) or fov <= 0.0 or fov >= 180.0:
                raise ValueError(f"demo_video_cameras.{name}.fov must be in (0, 180)")
            return {
                "position": position.tolist(),
                "lookat": lookat.tolist(),
                "fov": fov,
            }

        return _read_view("left"), _read_view("right")

    def _create_floor(self):
        """Add a visible ground-level floor while leaving plane collision intact."""
        if self._hide_ground_visuals():
            self.floor = None
            return
        if not bool(self.config.get("show_floor", True)):
            self.floor = None
            return
        half_size = self.config.get("floor_half_size", self.FLOOR_HALF_SIZE)
        hx, hy = [float(v) for v in half_size[:2]]
        thickness = float(self.config.get("floor_thickness", self.FLOOR_THICKNESS))
        color = tuple(float(v) for v in self.config.get("floor_color", self.FLOOR_COLOR))
        center = np.asarray(self.config.get("floor_center", [0.0, 0.0, 0.0]), dtype=float)
        self.floor = create_primitive(
            self.scene,
            "box",
            Pose(p=[float(center[0]), float(center[1]), float(center[2]) - thickness / 2.0]),
            size={"half_size": (hx, hy, thickness / 2.0)},
            color=color,
            is_static=True,
            collision=False,
        )

    def _create_table(self, table_height=TABLE_HEIGHT):
        """Add table to scene."""
        self._create_rectangular_table(
            table_height=table_height,
            half_size=(0.6, 0.35),
        )

    def _create_rectangular_table(
        self,
        *,
        table_height: float,
        half_size: tuple[float, float],
        center_xy: tuple[float, float] | None = None,
        table_thickness: float = 0.05,
        color: tuple[float, float, float] = (0.8, 0.75, 0.65),
        leg_inset: tuple[float, float] = (0.05, 0.05),
        leg_radius: float = 0.025,
        leg_xy: tuple[tuple[float, float], ...] | None = None,
        remember_bounds_before_variant: bool = True,
        allow_variant: bool = True,
    ):
        """Build one table geometry contract for primitives and mesh variants.

        ``half_size``, ``center_xy``, and ``table_height`` are the authoritative
        dimensions.  The procedural table and every debug mesh variant consume
        the same values, so changing a task footprint cannot leave its visual
        mesh, collision top, random-object bounds, or legs at different sizes.
        ``leg_xy`` may provide ordered absolute world coordinates for families
        whose calibrated layouts cannot be reconstructed from a symmetric inset.
        ``remember_bounds_before_variant`` is a transitional internal switch for
        legacy wrappers whose variant call was already their sole bounds writer.
        ``allow_variant=False`` keeps a family's fixed procedural table while
        still routing its geometry through this single builder.
        """
        half_x, half_y = (float(half_size[0]), float(half_size[1]))
        if half_x <= 0.0 or half_y <= 0.0:
            raise ValueError(f"table half_size must be positive, got {half_size!r}")
        if center_xy is None:
            center_xy = tuple(np.asarray(self.table_offset, dtype=float).ravel()[:2])
        dx, dy = (float(center_xy[0]), float(center_xy[1]))

        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = float(table_thickness)
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self._table_top_z_fixed = False
        if remember_bounds_before_variant:
            self._remember_table_bounds(
                half_size=(half_x, half_y), center_xy=(dx, dy),
            )
        if allow_variant:
            if self._try_create_table_variant(
                table_height,
                half_size=(half_x, half_y),
                center_xy=(dx, dy),
            ):
                return self.table
        elif not remember_bounds_before_variant:
            self._remember_table_bounds(
                half_size=(half_x, half_y), center_xy=(dx, dy),
            )

        # Table top
        self.table = create_primitive(
            self.scene, "box",
            Pose(p=[dx, dy, table_height]),
            size={"half_size": (half_x, half_y, self.TABLE_THICKNESS / 2)},
            color=color,
            is_static=True,
        )
        # Table legs
        leg_h = table_height - self.TABLE_THICKNESS / 2
        if leg_xy is None:
            inset_x = min(max(float(leg_inset[0]), 0.0), half_x)
            inset_y = min(max(float(leg_inset[1]), 0.0), half_y)
            leg_x = max(half_x - inset_x, 0.0)
            leg_y = max(half_y - inset_y, 0.0)
            leg_xy = (
                (dx - leg_x, dy - leg_y),
                (dx + leg_x, dy - leg_y),
                (dx - leg_x, dy + leg_y),
                (dx + leg_x, dy + leg_y),
            )
        for x, y in leg_xy:
            create_primitive(
                self.scene, "cylinder",
                Pose(p=[float(x), float(y), leg_h / 2]),
                size={"radius": float(leg_radius), "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5),
                is_static=True,
            )
        return self.table

    def _try_create_table_variant(
        self,
        table_height: float,
        half_size: tuple[float, float] = (0.6, 0.35),
        center_xy: tuple[float, float] | None = None,
    ) -> bool:
        """Debug-only randomized SAPIEN table replacement."""
        self._remember_table_bounds(half_size=half_size, center_xy=center_xy)
        if not table_randomization_enabled(self.config):
            return False
        add_table_variant(self, table_height, half_size=half_size, center_xy=center_xy)
        return True

    def _load_robot(self):
        """Add robot to scene.

        Robot selection is driven by ``config["robot_type"]`` (default
        ``"piper"``). Per-robot kwargs can be passed via
        ``config["robot_kwargs"]`` (e.g. ``{"mobile_base": True}`` for
        Stretch, ``{"pos": [...], "quat": [...]}`` for Franka).

        ``config["planner_override"]`` (optional, default absent) forces a
        specific planner across robot types. When absent, behavior is
        unchanged — each robot uses the planner declared in its embodiment
        YAML. Set by ``scripts/eval_vla.py`` only; data collection never
        sets it.
        """
        robot_type = str(self.config.get("robot_type", "piper")).lower()
        robot_config = self.config.get("robot_config", None)
        robot_kwargs = dict(self.config.get("robot_kwargs", {}) or {})

        planner_override = self.config.get("planner_override")
        if planner_override and robot_type in ("franka", "xarm7"):
            robot_kwargs.setdefault("planner_type", planner_override)

        if robot_type == "piper":
            self.robot = PiperRobot(config_path=robot_config, **robot_kwargs)
        elif robot_type == "franka":
            self.robot = FrankaRobot(**robot_kwargs)
        elif robot_type == "stretch":
            robot_kwargs.setdefault("mobile_base", True)
            self.robot = StretchRobot(**robot_kwargs)
        elif robot_type == "xarm7":
            self.robot = XArm7Robot(**robot_kwargs)
        elif robot_type in ("arx_x5", "arx-x5", "arxx5"):
            self.robot = ARXX5Robot(config_path=robot_config, **robot_kwargs)
        elif robot_type in ("ur5_wsg", "ur5-wsg", "ur5wsg"):
            self.robot = UR5WSGRobot(config_path=robot_config, **robot_kwargs)
        else:
            raise ValueError(
                f"Unknown robot_type: {robot_type!r}. "
                f"Expected one of: piper, franka, stretch, xarm7, arx_x5, ur5_wsg."
            )

        # For robots whose Arm reads `planner` from an embodiment-yaml dict
        # (Piper, Stretch, ...), patch the in-memory config before add_to_scene
        # so Arm.__init__ picks up the override.
        if planner_override and hasattr(self.robot, "config") and isinstance(self.robot.config, dict):
            self.robot.config["planner"] = planner_override

        self.robot.add_to_scene(self.scene)

    # HDRI environment maps sampled when nyx.env_texture is "random".
    # Any *.exr under assets/exr is eligible; override per-task with
    # config["nyx"]["env_texture_pool"].
    DEFAULT_ENV_TEXTURE_POOL = (
        "assets/exr/abandoned_garage_4k.exr",
        "assets/exr/brown_photostudio_02_4k.exr",
        "assets/exr/church_meeting_room_4k.exr",
        "assets/exr/ferndale_studio_11_4k.exr",
        "assets/exr/glasshouse_interior_4k.exr",
        "assets/exr/marry_hall_4k.exr",
        "assets/exr/pav_studio_03_4k.exr",
        "assets/exr/photo_studio_01_4k.exr",
        "assets/exr/university_workshop_4k.exr",
    )

    def episode_rng(self, stream: str) -> np.random.Generator:
        """Return a deterministic RNG isolated from task/physics sampling.

        Visual domain randomization must not consume NumPy's global stream:
        cameras are loaded before task actors, so doing that changes object and
        layout selection for the same episode seed.  Named streams also keep
        independent visual features from perturbing each other when one is
        enabled or disabled.
        """
        key = str(stream)
        rng = self._episode_rngs.get(key)
        if rng is None:
            stream_id = int(zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)
            seed = int(self._episode_seed) & 0xFFFFFFFF
            rng = np.random.default_rng(np.random.SeedSequence([seed, stream_id]))
            self._episode_rngs[key] = rng
        return rng

    def _resolve_nyx_options(self) -> dict:
        """Per-episode nyx options.

        ``env_texture: random`` samples a fresh HDRI from
        ``nyx.env_texture_pool`` (default: DEFAULT_ENV_TEXTURE_POOL) on every
        reset. ``nyx.env_offsets`` then selects the image offset belonging to
        that EXR.  Keys may be a full configured path, filename, or stem; each
        value is ``[horizontal_deg, vertical_deg]``.  A global
        ``nyx.env_offset`` is added to the per-EXR value.
        """
        nyx = dict(self.config.get("nyx", {}) or {})
        pool_override = nyx.pop("env_texture_pool", None)
        if str(nyx.get("env_texture", "")).lower() == "random":
            pool = list(pool_override or self.DEFAULT_ENV_TEXTURE_POOL)
            if not pool:
                nyx.pop("env_texture", None)
            else:
                nyx["env_texture"] = str(
                    self.episode_rng("nyx_env_texture").choice(pool)
                )
                print(
                    f"[nyx] randomized env_texture="
                    f"{os.path.basename(nyx['env_texture'])}"
                )

        per_texture = nyx.pop("env_offsets", {}) or {}
        if not isinstance(per_texture, dict):
            raise ValueError("nyx.env_offsets must be a mapping keyed by EXR name")
        global_x, global_y = normalize_env_offset(
            nyx.pop("env_offset", (0.0, 0.0))
        )
        texture = nyx.get("env_texture")
        texture_offset = (0.0, 0.0)
        if texture:
            texture_path = Path(str(texture))
            for key in (str(texture), texture_path.name, texture_path.stem):
                if key in per_texture:
                    texture_offset = normalize_env_offset(per_texture[key])
                    break
        resolved_offset = (
            global_x + texture_offset[0],
            global_y + texture_offset[1],
        )
        nyx["env_offset"] = resolved_offset
        if texture:
            print(
                f"[nyx] env_offset {Path(str(texture)).name}="
                f"[{resolved_offset[0]:g}, {resolved_offset[1]:g}] deg"
            )
        return nyx

    def _load_cameras(self):
        """Add cameras to scene."""
        # Get static camera config from robot config
        static_cameras = deepcopy(
            self.config.get("static_camera_list")
            or getattr(self, "static_camera_list", None)
            or []
        )
        if self.robot and hasattr(self.robot, 'config'):
            static_cameras = static_cameras or deepcopy(
                self.robot.config.get("static_camera_list", [])
            )
        if not static_cameras:
            # Default head camera. Bridge-V2-style third-person view: behind the
            # robot base (Franka at y=-0.65), raised slightly above shoulder, and
            # looking forward + slightly down so the whole arm + tabletop are
            # in frame. This is the view OpenVLA-format collection consumes.
            static_cameras = [{
                "name": "head_camera",
                "position": [0.0, 0.2, 2.0],
                "forward": [0.0, -0.4, -0.9],
            }]
        static_cameras = self._maybe_adjust_vla_head_camera(static_cameras)

        camera_config = self.config.get("camera_config", {"default": {"w": 640, "h": 480, "fovy": 60}})
        rt_cfg = self.config.get("raytracer", {}) or {}
        is_raytracer = self.config.get("renderer", DEFAULT_RENDERER).lower() == "raytracer"
        spp = rt_cfg.get("spp") if is_raytracer else None
        denoise = rt_cfg.get("denoise") if is_raytracer and "denoise" in rt_cfg else None
        recording_resolution = self.config.get("recording_resolution")
        side_resolution = self.config.get("side_resolution")
        # A demo pair always means two panes, even if a legacy task config
        # disabled its old side camera. VLA runs still honor side_video.
        enable_side = (
            bool(self.config.get("side_video", True))
            or self._demo_video_camera_pair_enabled()
        )
        recording_view, side_view = self._resolve_video_camera_specs()
        renderer_type = self.config.get("renderer", DEFAULT_RENDERER).lower()
        if renderer_type == "nyx":
            self.cameras = NyxCameraManager(
                self.scene,
                static_cameras=static_cameras,
                camera_config=camera_config,
                recording_pos=recording_view["position"],
                recording_lookat=recording_view["lookat"],
                side_pos=side_view["position"],
                side_lookat=side_view["lookat"],
                recording_fov=recording_view["fov"],
                side_fov=side_view["fov"],
                recording_resolution=recording_resolution,
                side_resolution=side_resolution,
                enable_side=enable_side,
                nyx_options=self._resolve_nyx_options(),
            )
            return

        batch_render = renderer_type in ("batch", "madrona")
        self.cameras = Camera(
            self.scene, static_cameras=static_cameras, camera_config=camera_config,
            recording_pos=recording_view["position"],
            recording_lookat=recording_view["lookat"],
            side_pos=side_view["position"],
            side_lookat=side_view["lookat"],
            recording_fov=recording_view["fov"],
            side_fov=side_view["fov"],
            spp=spp,
            recording_resolution=recording_resolution,
            side_resolution=side_resolution,
            enable_side=enable_side,
            denoise=denoise,
            batch_render=batch_render,
        )

    def _maybe_adjust_vla_head_camera(self, static_cameras: list) -> list:
        """Use a centered top-down head camera for VLA collection.

        This fixes the VLA visual orientation at the source. It is gated by
        config["vla_recording"]["opposite_head_camera"] so normal videos and
        non-recording eval keep their task-authored camera poses.
        """
        vla_cfg = self.config.get("vla_recording", {}) or {}
        use_vla_head_camera = bool(
            self.config.get("vla_head_camera", False)
            or (vla_cfg.get("enabled") and vla_cfg.get("opposite_head_camera"))
        )
        if not use_vla_head_camera:
            return static_cameras
        for cam_info in static_cameras:
            if cam_info.get("name") != "head_camera":
                continue
            try:
                if bool(self.config.get("xarm_calibrated_head_camera", False)):
                    cam_info["position"] = [0.9112, -0.2669, 0.5065]
                    cam_info["lookat"] = [0.0074, -0.2728, -0.1215]
                    cam_info["up"] = [-0.4280, 0.0030, 1.0038]
                    cam_info["fov"] = 64.81
                    cam_info["vla_top_down_head_camera"] = True
                    continue
                center = np.asarray(
                    vla_cfg.get(
                        "head_camera_table_center_xy",
                        getattr(self, "TABLE_CENTER_XY", getattr(self, "table_offset", (0.0, 0.0))),
                    ),
                    dtype=np.float64,
                ).ravel()[:2]
                table_z = float(getattr(self, "TABLE_TOP_Z", TABLE_HEIGHT + 0.025))
                z = float(vla_cfg.get("head_camera_top_down_z", table_z + 1.60))
                cam_info["position"] = [float(center[0]), float(center[1]), z]
                cam_info["forward"] = [0.0, 0.0, -1.0]
                cam_info["vla_top_down_head_camera"] = True
            except Exception:
                continue
        return static_cameras

    def _resolve_avatar_skin(self) -> dict:
        """Build the skin_options dict for this episode.

        Starts from DEFAULT_AVATAR_SKIN, overlays any config overrides, and —
        if config["randomize_avatar"] is set — samples a fresh glb_path from
        AVATAR_POOL (task pool if defined, else DEFAULT_AVATAR_POOL).
        """
        skin = dict(self.DEFAULT_AVATAR_SKIN)
        skin.update(self.config.get("avatar", {}).get("skin_options", {}))
        randomized = False
        if self.config.get("randomize_avatar", False):
            pool = self.AVATAR_POOL or DEFAULT_AVATAR_POOL
            if pool:
                skin["glb_path"] = str(self.episode_rng("avatar_skin").choice(pool))
                randomized = True
                print(f"[avatar] randomized glb={os.path.basename(skin['glb_path'])}")
        self._vla_avatar_model = str(skin["glb_path"])
        self._vla_avatar_randomized = randomized
        return skin

    def _init_avatar(self):
        """Initialize avatar if task uses one."""
        if not self.use_avatar or self.no_human_enabled():
            return
        try:
            from ..avatar import AvatarController
        except ImportError as exc:
            raise RuntimeError("AvatarController import failed; real animated avatar is required.") from exc

        avatar_cfg = self.config.get("avatar", {})
        if self.config.get("debug_avatar_init", False):
            setattr(self.scene, "_debug_avatar_init", True)
        if self.config.get("skip_avatar_walk_modules", False):
            setattr(self.scene, "_skip_avatar_walk_modules", True)
        skin_options = self._resolve_avatar_skin()

        try:
            self.avatar = AvatarController(
                scene=self.scene,
                motion_data_path=avatar_cfg.get("motion_data", "avatars/motions/motion.pkl"),
                skin_options=skin_options,
                frame_ratio=avatar_cfg.get("frame_ratio", 1.0),
                name="human",
                assets_dir=str(ASSETS_PATH),
                generated_motion_path=avatar_cfg.get("generated_motion_data", "avatars/motions/generated_motions.pkl"),
            )
        except Exception as exc:
            raise RuntimeError("AvatarController init failed; real animated avatar is required.") from exc

    def _avatar_collision_enabled(self) -> bool:
        """Is avatar-collision tracking turned on for this run?  Checked
        once in reset(); not an expensive property."""
        return (
            bool(self.config.get("track_avatar_collision", False))
            and not self.no_human_enabled()
        )

    def _avatar_retreat_enabled(self) -> bool:
        """Whether the active safety controller should override robot motion."""
        return (
            bool(self.config.get("avatar_retreat_enabled", False))
            and not self.no_human_enabled()
        )

    def _apply_policy_gripper_gains(self) -> None:
        """Apply task-level finger gains for policy/eval rollouts.

        Scripted experts often call a task-local ``_boost_finger_pd`` before
        grasping. Learned policies drive the robot through ``take_action`` and
        never enter those expert hooks, so mirror the task constants once after
        robot initialization.
        """
        if not bool(self.config.get("eval_mode", False)):
            return
        if not bool(self.config.get("apply_task_gripper_gains", True)):
            return

        if (
            "policy_gripper_kp" not in self.config
            and "policy_gripper_kv" not in self.config
            and "policy_gripper_force_limit" not in self.config
            and hasattr(self, "_boost_finger_pd")
        ):
            applied = False
            for arm_tag in ("left", "right"):
                try:
                    arm = self.robot.get_arm(arm_tag)
                except Exception:
                    arm = None
                if arm is None:
                    continue
                try:
                    self._boost_finger_pd(arm_tag)
                    applied = True
                except Exception:
                    pass
            if applied:
                return

        kp_value = self.config.get("policy_gripper_kp", getattr(self, "FINGER_KP", None))
        kv_value = self.config.get("policy_gripper_kv", getattr(self, "FINGER_KV", None))
        force_value = self.config.get(
            "policy_gripper_force_limit",
            getattr(self, "FINGER_FORCE_LIMIT", getattr(self, "FINGER_FORCE_N", None)),
        )
        if kp_value is None and kv_value is None and force_value is None:
            return

        for arm_tag in ("left", "right"):
            try:
                arm = self.robot.get_arm(arm_tag)
            except Exception:
                arm = None
            if arm is None or not getattr(arm, "_finger_dof_indices", None):
                continue
            if kp_value is not None or kv_value is not None:
                update_dofs_kp_kv_compat(
                    arm.entity,
                    arm._finger_dof_indices,
                    kp_value=kp_value,
                    kv_value=kv_value,
                )
            if force_value is not None:
                update_dofs_force_range_compat(
                    arm.entity,
                    arm._finger_dof_indices,
                    lower_value=-float(force_value),
                    upper_value=+float(force_value),
                )

    def no_human_enabled(self) -> bool:
        """Whether this rollout/eval should omit the human avatar entirely."""
        return bool(
            self.config.get("no_human", False)
            or self.config.get("no_avatar", False)
            or self.config.get("disable_avatar", False)
        )

    def _init_avatar_collider(self):
        """Build the bone-cylinder rig BEFORE scene.build().  Created
        when either the collision checker is on OR the rig is explicitly
        asked for (e.g. --show-collider)."""
        if self.avatar is None:
            return
        want_rig = (self._avatar_collision_enabled()
                    or self._avatar_retreat_enabled()
                    or bool(self.config.get("use_avatar_collider", False))
                    or bool(self.config.get("collider_visualization", False)))
        if not want_rig:
            return
        try:
            from ..avatar import AvatarCollider
        except ImportError:
            print("[avatar] AvatarCollider import failed")
            return
        self.avatar_collider = AvatarCollider(
            scene=self.scene,
            avatar_robot=self.avatar.robot,
            visualization=bool(self.config.get("collider_visualization", False)),
            collision=False,   # stay out of the physics world — detection is analytic
        )

    def _init_avatar_collision_checker(self):
        """Build the analytic checker AFTER scene.build() + robot joints.
        Binds to every arm entity on the robot so dual-arm robots
        (Piper) are covered uniformly."""
        if self.avatar_collider is None:
            return
        safety_margin = self.config.get("avatar_safety_margin", None)
        want_retreat = self._avatar_retreat_enabled()
        if not self._avatar_collision_enabled() and safety_margin is None and not want_retreat:
            return
        try:
            from ..avatar import AvatarCollisionChecker
        except ImportError:
            print("[avatar] AvatarCollisionChecker import failed")
            return
        # Collect every arm entity the robot exposes.  Most tasks use a
        # single-arm robot (Franka/Stretch); Piper exposes left+right.
        entities = []
        for tag in ("left", "right"):
            try:
                arm = self.robot.get_arm(tag)
            except Exception:
                arm = None
            if arm is not None and getattr(arm, "entity", None) is not None:
                if arm.entity not in entities:
                    entities.append(arm.entity)
        if not entities:
            return
        n_pts = int(self.config.get("collision_n_points_per_geom", 30))
        if self._avatar_collision_enabled():
            margin = float(self.config.get("collision_margin",
                                           self.DEFAULT_COLLISION_MARGIN))
            self.avatar_collision_checker = AvatarCollisionChecker(
                avatar_collider=self.avatar_collider,
                robot_entity=entities,
                n_points_per_geom=n_pts,
                margin=margin,
                ignored_capsules=self.config.get("collision_ignore_capsules", ()),
                ignored_capsule_prefixes=self.config.get(
                    "collision_ignore_capsule_prefixes", ()
                ),
            )
        if safety_margin is not None:
            self.avatar_safety_checker = AvatarCollisionChecker(
                avatar_collider=self.avatar_collider,
                robot_entity=entities,
                n_points_per_geom=n_pts,
                margin=float(safety_margin),
            )
        if want_retreat:
            retreat_margin = self.config.get("avatar_retreat_margin", safety_margin)
            if retreat_margin is None:
                retreat_margin = 0.10
            self.avatar_retreat_checker = AvatarCollisionChecker(
                avatar_collider=self.avatar_collider,
                robot_entity=entities,
                n_points_per_geom=n_pts,
                margin=float(retreat_margin),
            )
