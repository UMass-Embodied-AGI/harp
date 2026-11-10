"""Base task: slim foundation for all manipulation tasks."""

import contextlib
import os
import pickle
import random
import numpy as np
import torch
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from .utils import Pose, to_numpy
from .scene_init import (
    apply_scene_init_post_settle,
    apply_scene_init_pre_avatar,
    prepare_config_for_scene_init,
)
from .interaction import (
    InteractionController,
    LegacyInteractionController,
    PolicyDirective,
)

# Redundant aliases mark these as re-exports: TABLE_HEIGHT / DEFAULT_RENDERER
# have always been importable from envs.base_task and task code relies on that.
from .base_task_parts.shared import (
    DEFAULT_RENDERER as DEFAULT_RENDERER,
    TABLE_HEIGHT as TABLE_HEIGHT,
)
from .base_task_parts.objects import _TargetObjectMixin
from .safety_metrics import DEFAULT_DISTANCE_THRESHOLDS, compute_safety_metrics
from .base_task_parts.scene import _SceneSetupMixin
from .base_task_parts.observation import _ObservationActionMixin
from .base_task_parts.grasp_select import _GraspSelectionMixin



@dataclass
class TargetSpec:
    """Per-task evaluation target.

    Tasks opt in by setting ``self.target = TargetSpec(...)`` in ``load_actors``.
    ``BaseTask.evaluate()`` then reports position / orientation / collision
    metrics; the default ``check_success()`` passes when ``plan_success`` is
    True and every configured threshold holds, so simple tasks don't need
    their own success logic.

    Geometry hooks:
      object:          actor/entity whose final pose is measured (exposes
                       ``.entity.get_pos()``). None disables target metrics.
      position:        ``np.ndarray`` for a fixed target, or a zero-arg
                       callable returning one (use a callable when the
                       target moves — e.g. the avatar's palm).
      label:           human-readable tag used in console/pickle output
                       (e.g. "plate", "hand", "cup_rim").
      object_pos_fn:   optional zero-arg callable returning the object's
                       world position, used instead of ``entity.get_pos()``
                       (e.g. bbox centre instead of mesh origin).
      object_axis_fn:  optional zero-arg callable returning the object's
                       "up" direction in world frame (unit vector).  Enables
                       tilt metrics / thresholds.

    Success thresholds (all optional; set only what matters for the task):
      success_dist_xy:         pass if lateral distance <= this (m)
      success_dist_3d:         pass if 3-D distance <= this (m)
      success_dz_min:          pass if obj.z - tgt.z >= this (m)
      success_tilt_min_deg:    pass if angle(object_axis, world +Z) >= this
      success_tilt_max_deg:    pass if angle(object_axis, world +Z) <= this
      success_require_no_avatar_collision:
          pass only if the avatar-collision tracker was enabled and never
          triggered this episode.  No-op when tracking is off.
    """
    object: Any = None
    position: Optional[Union[np.ndarray, Callable[[], np.ndarray]]] = None
    label: str = "target"
    object_pos_fn: Optional[Callable[[], np.ndarray]] = None
    object_axis_fn: Optional[Callable[[], np.ndarray]] = None
    success_dist_xy: Optional[float] = None
    success_dist_3d: Optional[float] = None
    success_dz_min: Optional[float] = None
    success_tilt_min_deg: Optional[float] = None
    success_tilt_max_deg: Optional[float] = None
    success_require_no_avatar_collision: bool = False

class _PlanningBlindCollider:
    """Avatar collider view that reports no shapes (see hide_avatar_from_planning)."""

    def __init__(self, collider):
        self._collider = collider

    def current_capsules(self):
        return []

    def current_boxes(self):
        return []

    def __getattr__(self, name):
        return getattr(self._collider, name)


class BaseTask(
    _TargetObjectMixin,
    _SceneSetupMixin,
    _ObservationActionMixin,
    _GraspSelectionMixin,
):
    """Base class for all manipulation tasks.

    Lifecycle:
        1. __init__(config) — parse config
        2. reset(seed) — build scene, load actors, return initial obs
        3. play_once() — scripted rollout (data collection)
           OR take_action(action) loop — policy evaluation
        4. check_success() — evaluate task completion

    Subclasses must override:
        - load_actors() — add task-specific objects to scene
        - play_once() — scripted task execution
        - check_success() — success evaluation
    """

    use_avatar = False  # Override in subclass to enable avatar
    use_table = True    # Override to False for floor-only tasks (mobile pick, etc.)

    # Shared easy/hard object-count knobs for tasks whose difficulty is mainly
    # the number of task-relevant objects. Subclasses override these values.
    EASY_NUM_OBJECTS = 1
    HARD_NUM_OBJECTS = 2

    # Universal tabletop clutter defaults. These are real dynamic collision
    # bodies when table_random_objects is enabled.
    TABLE_RANDOM_OBJECT_COUNT = 2
    TABLE_RANDOM_OBJECT_TARGET_SIZE = 0.09
    TABLE_RANDOM_OBJECT_TABLE_MARGIN = 0.07
    TABLE_RANDOM_OBJECT_EXCLUSION_RADIUS = 0.20
    TABLE_RANDOM_OBJECT_ROBOT_BASE_RADIUS = 0.30
    TABLE_RANDOM_OBJECT_TYPE_ALIASES = {
        "block": ("086_woodenblock", "108_block"),
        "cube": ("086_woodenblock", "108_block"),
        "woodenblock": ("086_woodenblock",),
        "bowl": ("002_bowl",),
        "shoe": ("041_shoe",),
        "bread": ("075_bread",),
        "burger": ("006_hamburg",),
        "hamburg": ("006_hamburg",),
        "fries": ("005_french-fries",),
        "french-fries": ("005_french-fries",),
        "apple": ("035_apple",),
        "can": ("071_can", "105_sauce-can"),
        "cup": ("021_cup",),
        "mug": ("039_mug",),
        "phone": ("077_phone",),
        "stapler": ("048_stapler",),
        "tissue": ("023_tissue-box",),
        "milk": ("038_milk-box",),
        "jam": ("031_jam-jar",),
        "oil": ("029_olive-oil",),
        "seal": ("100_seal",),
    }

    # Short imperative natural-language instruction (Bridge V2 style).
    # Required for OpenVLA / Bridge-format data collection. Subclasses must
    # set this; collect.py writes it into each episode record.  May contain a
    # ``{object}`` slot, filled with the resolved target's display name by
    # ``resolve_target_object`` (see envs/object_catalog.py).
    INSTRUCTION = None

    # Target-object selection (see envs/object_catalog.py).  Either an inline
    # white-list of object keys (``["041_shoe"]``) or a category name
    # (``"cabinet_drawer_items"``).  Tasks that pick/place a catalog object set
    # this and call ``resolve_target_object()``; None means the task does not
    # use the catalog (e.g. primitive-only blocks tasks, multi-object trays).
    OBJECT_SET = None

    # Single global reset-settle budget: sim steps to run after scene build so
    # dynamic objects fall and settle before any task begins.  Keep this
    # uniform across tasks; override only via config["settle_steps"] for an
    # explicit experiment/debug run.
    SETTLE_STEPS = 400

    # Capture one video frame every VIDEO_STRIDE sim steps. 1 = every step.
    # Tasks with long rollouts should bump this to 6+ to cut render cost —
    # rendering is the dominant per-step wall-clock cost. Override via subclass
    # class-attr or via config["video_stride"].
    VIDEO_STRIDE = 1

    # Record one observation (image + proprio) every RECORD_STRIDE step_sim
    # calls into self.traj_data. None = recording disabled (default; preserves
    # legacy behavior where only tasks that explicitly call self.record_frame()
    # populate traj_data). Override via config["record_stride"]. Genesis default
    # dt=0.002s, so 50 ≈ 10 Hz, 100 ≈ 5 Hz (Bridge-V2-style cadence).
    RECORD_STRIDE = None

    # Pool of avatar glb paths (relative to ASSETS_PATH). Consulted when
    # config["randomize_avatar"] is True. Subclasses can override with a
    # task-specific subset; None falls back to DEFAULT_AVATAR_POOL below.
    AVATAR_POOL = None

    # Extra ``gs.options.RigidOptions`` for scenes whose contact count outgrows
    # Genesis's fixed collision buffers (e.g. many-hull objects piling up).
    # Larger buffers only add capacity; empty keeps Genesis's defaults.
    RIGID_OPTIONS: dict = {}

    # Default skin settings used when config doesn't specify them.
    DEFAULT_AVATAR_SKIN = {
        "glb_path": "avatars/models/custom_Adrian_Keller.glb",
        "euler": (-90, 0, 90),
        "pos": (0.0, 0.0, -0.959008030),
    }

    FLOOR_HALF_SIZE = (2.2, 2.2)
    FLOOR_COLOR = (0.64, 0.65, 0.62)
    FLOOR_THICKNESS = 0.004

    # ------------------------------------------------------------------
    # Avatar-collision tracking (opt-in, all default off).  Turn on via
    # `config["track_avatar_collision"] = True` (or the matching
    # --track-avatar-collision CLI flag).  See
    # envs/avatar/collision_checker.py for the analytic method and
    # scripts/debug_avatar_collision.py for a standalone demo.
    DEFAULT_COLLISION_CHECK_STRIDE = 30   # ~60 ms real-time at 500 Hz sim
    DEFAULT_COLLISION_MARGIN = 0.0        # 0 = strict overlap; +ve = near-miss

    def __init__(self, config: dict = None):
        self.config = prepare_config_for_scene_init(config or {}, (config or {}).get("scene_init"))
        self.instruction = self.config.get("instruction") or type(self).INSTRUCTION
        # Base instruction string (may contain a ``{object}`` slot); kept so
        # resolve_target_object can re-template it per reset.
        self._instruction_template = self.instruction
        self._target_entry = None          # last-resolved catalog ObjectEntry
        self._episode_seed = 0
        self._episode_rngs = {}
        self.scene = None
        self.robot = None
        self.cameras = None
        self.avatar = None
        self.plan_success = True
        self.eval_success = False
        self.take_action_cnt = 0
        self.FRAME_IDX = 0

        # Avatar-collision tracking state.  `avatar_collider` is built
        # pre-build iff `track_avatar_collision` (or `use_avatar_collider`)
        # is on; `avatar_collision_checker` is built post-build.
        self.avatar_collider = None
        self.avatar_collision_checker = None
        self.avatar_safety_checker = None
        self.avatar_retreat_checker = None
        self.avatar_collided = False       # sticky — first hit sets True
        self.avatar_collision_log = []     # list of per-check dicts
        self.avatar_safety_log = []        # inflated-envelope near-miss checks
        self.avatar_safety_intervention_log = []
        self._avatar_retreat_hold_count = 0
        self._avatar_retreat_tick = 0
        self._avatar_retreat_active_cached = False
        self._avatar_retreat_release_state = {}
        self._avatar_collision_stride = int(self.config.get(
            "collision_check_stride", self.DEFAULT_COLLISION_CHECK_STRIDE))
        self._avatar_collision_stride = max(1, self._avatar_collision_stride)
        self._avatar_collision_tick = 0
        self.SIM_STEP_IDX = 0   # incremented every step_sim(); episode-scoped
        # Separation thresholds (m) for post-hoc intrusion stats
        # (envs/safety_metrics.py); derived from the logged distance series.
        self._safety_distance_thresholds = sorted(
            float(x) for x in self.config.get(
                "safety_distance_thresholds", DEFAULT_DISTANCE_THRESHOLDS))

        # Trajectory recording
        self.left_joint_path = []
        self.right_joint_path = []
        self.traj_data = []

        # Video recording
        self._video_frames = []
        self._video_path = None
        self._video_stride = int(self.config.get("video_stride", self.VIDEO_STRIDE))
        self._video_stride = max(1, self._video_stride)
        self._video_cap_tick = 0
        # When True, step_sim records nothing (video/VLA/trajectory).  Used to
        # exclude pre-task scene settling + avatar/scene setup from the
        # recordings — see BaseTask.suppress_recording().
        self._record_suppressed = False

        # Trajectory recording stride. Off by default; set via config to opt in.
        rs = self.config.get("record_stride", self.RECORD_STRIDE)
        self._record_stride = int(rs) if rs is not None else None
        self._record_tick = 0

        # Optional whitelist of camera names to keep in recorded obs.rgb.
        # None = keep all cameras (legacy behavior). Set to e.g. ["head_camera"]
        # to dump only the cameras you actually need — cuts pickle size by ~5x
        # on a 5-camera scene.
        rc = self.config.get("record_cameras")
        self._record_cameras = list(rc) if rc else None

        # Gripper attachment (for DEBUG_TELEPORT / kinematic grasp)
        # {arm_tag: (entity, relative_pose)} where relative_pose is Pose of object in EE frame
        self._gripper_attached = {}

        # Target-object evaluation spec (see `TargetSpec` docstring).
        # Tasks opt in by replacing this in `load_actors`; the default
        # spec has no object/position so target metrics are omitted.
        self.target = TargetSpec()
        self.table_variant_name = None
        self._table_top_z_fixed = False
        self._table_center_xy = np.asarray(getattr(self, "table_offset", (0.0, 0.0)), dtype=float)[:2]
        self._table_half_size_xy = np.asarray((0.6, 0.35), dtype=float)
        self._table_random_object_actors = []
        self._table_random_object_exclusions = []
        self.table_random_objects = bool(
            self.config.get(
                "table_random_objects",
                self.config.get(
                    "task_irrelevant_objects",
                    self.config.get("with_irrelevant_objects", False),
                ),
            )
        )
        # Backward-compatible attribute name used by older task code.
        self.task_irrelevant_objects = self.table_random_objects
        difficulty = str(self.config.get("difficulty", "")).strip().lower()
        self.hard = bool(
            self.config.get("hard", self.config.get("task_hard", False))
            or difficulty == "hard"
        )

        # Optional VLA training-data recorder.  Built in reset() iff
        # config["vla_recording"]["enabled"]; ticked from step_sim.
        self.vla_recorder = None
        self.scene_init_replay = {"enabled": False}

        # Controller creation is lazy until reset so subclass __init__ methods
        # have completed. Existing task behavior remains in legacy classes and
        # mixins during the compatibility-first migration.
        self._interaction_controller = None
        self._interaction_sim_step = 0
        self._interaction_policy_step = 0
        self.last_policy_directive = PolicyDirective()

    # ------------------------------------------------------------------
    # Simulation step helper
    # ------------------------------------------------------------------

    def step_sim(self):
        """One simulation step: physics + video capture + avatar + attached objects."""
        self.scene.step()
        self.SIM_STEP_IDX += 1
        if self.robot is not None:
            self.robot.on_post_step(self.scene)
        self._sync_gripper_attached()
        if self.avatar is not None:
            self.avatar.step()
        self._notify_interaction_sim_step()
        self._tick_avatar_checks()
        # Skip ALL recording (VLA, video, trajectory) while the scene is being
        # set up / settled before the task actually starts — see
        # `suppress_recording()`.  Pre-task object-settling and avatar/scene
        # layout must not pollute the video or the training data.
        if not self._record_suppressed:
            if self.vla_recorder is not None:
                self.vla_recorder.tick(self)
            self.capture_frame()
            if self._record_stride is not None:
                if self._record_tick % self._record_stride == 0:
                    self.record_frame()
            self._record_tick += 1

    def _tick_avatar_checks(self):
        """Per-sim-step avatar collider update + stride-sampled checks.

        The single owner of robot-avatar collision/safety bookkeeping.
        ``step_sim`` overrides that do not call ``super().step_sim()`` must
        increment ``SIM_STEP_IDX`` after ``scene.step()`` and call this
        helper, so every record carries ``sim_step`` / ``t`` / ``pretask``.

        The bone-cylinder rig tracks the avatar skin every sim step even
        when we're not actively checking — cheap, and it keeps the
        visualization aligned if --show-collider is on.  The analytic check
        runs every ``collision_check_stride`` steps (default 30, ~60 ms at
        500 Hz).  It never gates the task; it only appends to
        ``avatar_collision_log`` / ``avatar_safety_log``.
        """
        if self.avatar_collider is not None:
            self.avatar_collider.update()
        if self.avatar_collision_checker is None and self.avatar_safety_checker is None:
            return
        self._avatar_collision_tick += 1
        if self._avatar_collision_tick < self._avatar_collision_stride:
            return
        self._avatar_collision_tick = 0
        sim_t = self.SIM_STEP_IDX * self._sim_dt()
        if self.avatar_collision_checker is not None:
            result = self.avatar_collision_checker.check()
            result["frame"] = self.FRAME_IDX
            result["sim_step"] = self.SIM_STEP_IDX
            result["t"] = sim_t
            result["pretask"] = bool(self._record_suppressed)
            self.avatar_collision_log.append(result)
            if result["collided"] and not self.avatar_collided:
                self.avatar_collided = True
        if self.avatar_safety_checker is not None:
            result = self.avatar_safety_checker.check()
            result["frame"] = self.FRAME_IDX
            result["sim_step"] = self.SIM_STEP_IDX
            result["t"] = sim_t
            self.avatar_safety_log.append(result)

    # ------------------------------------------------------------------
    # Gripper attachment (kinematic grasp for DEBUG_TELEPORT)
    # ------------------------------------------------------------------

    def attach_to_gripper(self, entity, arm_tag: str):
        """Attach entity to gripper, preserving current relative pose."""
        arm = self.robot.get_arm(arm_tag)
        ee_pose = Pose.from_pose7(arm.get_ee_pose())
        obj_pos = to_numpy(entity.get_pos()).ravel()[:3]
        obj_quat = to_numpy(entity.get_quat()).ravel()[:4]
        obj_pose = Pose(obj_pos, obj_quat)
        # relative_pose = inv(T_world_ee) @ T_world_obj = T_ee_obj
        relative_pose = ee_pose.inv() * obj_pose
        self._gripper_attached[arm_tag] = (entity, relative_pose)

    def detach_from_gripper(self, arm_tag: str):
        """Detach entity from gripper."""
        self._gripper_attached.pop(arm_tag, None)

    def _sync_gripper_attached(self):
        """Update positions of objects attached to grippers."""
        for arm_tag, (entity, relative_pose) in self._gripper_attached.items():
            arm = self.robot.get_arm(arm_tag)
            ee_pose = Pose.from_pose7(arm.get_ee_pose())
            obj_world = ee_pose * relative_pose
            entity.set_pos(obj_world.p.astype(float))
            entity.set_quat(obj_world.q.astype(float))

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def create_interaction_controller(self) -> InteractionController:
        """Create this task's composed human-interaction behavior.

        The legacy adapter is intentionally a no-op: current task classes and
        mixins retain identical behavior until each family is migrated.
        """
        return LegacyInteractionController()

    @property
    def interaction_controller(self) -> InteractionController:
        """Return the lazily created controller after validating its type."""

        if self._interaction_controller is None:
            controller = self.create_interaction_controller()
            if not isinstance(controller, InteractionController):
                raise TypeError(
                    "create_interaction_controller() must return an "
                    "InteractionController, got "
                    f"{type(controller).__name__}"
                )
            self._interaction_controller = controller
        return self._interaction_controller

    def _notify_interaction_sim_step(self) -> None:
        """Emit one monotonic BaseTask-core simulation-step notification."""

        self._interaction_sim_step += 1
        self.interaction_controller.on_base_sim_step(
            self, step_index=int(self._interaction_sim_step)
        )

    def advance_interaction_policy_step(self) -> bool:
        """Advance a composed controller at one policy-step boundary.

        Returns ``True`` when the active controller owns policy ticks.  Legacy
        scripted-rollout adapters use this signal to retain their private-hook
        fallback until each remaining task family is migrated.
        """

        controller = self.interaction_controller
        if not bool(controller.handles_policy_ticks):
            return False
        self._interaction_policy_step += 1
        controller.on_policy_tick(
            self, step_index=int(self._interaction_policy_step)
        )
        return True

    def reset(self, seed: int = 0):
        """Build scene and return initial observation.

        Call order: setup_scene → create_table → load_robot → load_cameras →
                    load_actors → init_avatar → scene.build() → init_joints → homestate
        """
        # Seed every RNG that any task or planner might pull from so a stage-1
        # rasterizer success verdict deterministically predicts the stage-2
        # raytracer outcome (renderer choice doesn't affect physics).
        self._episode_seed = int(seed)
        self._episode_rngs = {}
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import mplib.pymp as pymp
            pymp.set_global_seed(int(seed))
        except Exception:
            pass
        self._interaction_sim_step = 0
        self._interaction_policy_step = 0
        self.interaction_controller.on_episode_seeded(self, int(seed))
        self.last_policy_directive = PolicyDirective()

        self._setup_scene()
        if self.use_table:
            self._create_table()
        else:
            self.table = None
            self.TABLE_HEIGHT = 0.0
            self.TABLE_THICKNESS = 0.0
            self.TABLE_TOP_Z = 0.0
        self._load_robot()
        self._load_cameras()
        self.load_actors()
        self.interaction_controller.augment_scene_before_build(self)
        self._spawn_table_random_objects()
        apply_scene_init_pre_avatar(self)
        if self.config.get("debug_reset", False):
            print("[base_reset] init_avatar_begin", flush=True)
        self._init_avatar()
        self.interaction_controller.on_avatar_created_before_build(self)
        if self.config.get("debug_reset", False):
            print("[base_reset] init_avatar_done", flush=True)
            print("[base_reset] init_avatar_collider_begin", flush=True)
        self._init_avatar_collider()
        if self.config.get("debug_reset", False):
            print("[base_reset] init_avatar_collider_done", flush=True)
            print("[base_reset] scene_build_begin", flush=True)

        self._setup_batch_render_cache()
        self.scene.build()
        if self.config.get("debug_reset", False):
            print("[base_reset] scene_build_done", flush=True)

        self.robot.init_joints(self.scene)
        if self.config.get("robot_type", "franka") == "franka" and self.cameras is not None:
            self.cameras.attach_franka_wrist_camera(self.robot)
        self.robot.move_to_homestate()
        self._init_avatar_collision_checker()
        if self.robot.left_arm is not None:
            self.robot.open_gripper("left")
        self.robot.open_gripper("right")
        self._apply_policy_gripper_gains()

        # Reset avatar position after scene build
        if self.avatar is not None:
            avatar_pos = np.asarray(
                getattr(self, "avatar_init_pos", [0.0, 0.0, 0.0]), dtype=np.float64
            )
            avatar_rot = np.asarray(
                getattr(self, "avatar_init_rot", np.eye(3)), dtype=np.float64
            )
            if self.config.get("debug_reset", False):
                print("[base_reset] avatar_reset_begin", flush=True)
            self.avatar.reset(
                avatar_pos,
                avatar_rot,
                update_mesh=not bool(self.config.get("skip_avatar_reset_update", False)),
            )
        if self.config.get("debug_reset", False):
            print("[base_reset] avatar_reset_done", flush=True)

        # Let dynamic objects fall and settle before the task starts.  This
        # intentionally uses BaseTask.SETTLE_STEPS, not subclass attributes, so
        # all tasks share one reset-settle budget.
        settle_steps = int(self.config.get("settle_steps", BaseTask.SETTLE_STEPS))
        if self.config.get("debug_reset", False):
            print(f"[base_reset] settle_begin steps={settle_steps}", flush=True)
        self._settle_scene(settle_steps)
        self.scene_init_replay = apply_scene_init_post_settle(self)
        if self.config.get("debug_reset", False):
            print("[base_reset] settle_done", flush=True)

        self.robot.set_origin_endpose()
        if self.config.get("debug_reset", False):
            print("[base_reset] origin_endpose_done", flush=True)

        # Reset state
        self.plan_success = True
        self.eval_success = False
        self.take_action_cnt = 0
        self.FRAME_IDX = 0
        self.SIM_STEP_IDX = 0
        self.left_joint_path = []
        self.right_joint_path = []
        self.traj_data = []

        # Avatar-collision episode state.  Collider + checker objects
        # themselves stay built; only the recorded results are cleared.
        self.avatar_collided = False
        self.avatar_collision_log = []
        self.avatar_safety_log = []
        self.avatar_safety_intervention_log = []
        self._avatar_retreat_hold_count = 0
        self._avatar_retreat_tick = 0
        self._avatar_retreat_active_cached = False
        self._avatar_retreat_release_state = {}
        self._avatar_collision_tick = 0

        # Build VLA recorder if requested.  Off by default so existing
        # collect.py / eval.py runs are zero-overhead.
        vla_cfg = self.config.get("vla_recording", {}) or {}
        if vla_cfg.get("enabled"):
            from .vla_recorder import VLARecorder
            self.vla_recorder = VLARecorder(
                out_dir=vla_cfg["out_dir"],
                cams=vla_cfg.get("cams", ("recording", "right_wrist")),
                jpeg_quality=int(vla_cfg.get("jpeg_quality", 92)),
                capture_stride=int(vla_cfg.get("capture_stride",
                                               self.config.get("action_substeps", 25))),
            )
        else:
            self.vla_recorder = None

        # Compute actual table top Z
        if self.table is not None and not getattr(self, "_table_top_z_fixed", False):
            pos = to_numpy(self.table.get_pos()).ravel()
            self.TABLE_TOP_Z = float(pos[2]) + self.TABLE_THICKNESS / 2

        self.interaction_controller.on_base_reset_ready(self)

        if self.config.get("skip_reset_obs", False):
            if self.config.get("debug_reset", False):
                print("[base_reset] skip_reset_obs returning", flush=True)
            observation = None
        else:
            observation = self.get_obs()
        self.interaction_controller.after_base_reset_observation(
            self, observation=observation
        )
        return observation

    def _settle_scene(self, n_steps: int = None):
        """Run sim steps so gravity-loaded dynamic objects fall and come to
        rest before the task begins.  Uses the global BaseTask.SETTLE_STEPS by
        default; callers can pass n_steps explicitly for debug/special cases."""
        steps = BaseTask.SETTLE_STEPS if n_steps is None else n_steps
        for _ in range(int(steps)):
            self.scene.step()

    def _align_avatar_head_hand_to_xy(
        self,
        target_xy,
        hand_id: int = 1,
        motion_name: str | None = None,
        motion_frame: int = -1,
    ) -> float | None:
        """Yaw the avatar so the Head→Palm XY vector points at target_xy.

        Returns the applied yaw delta in degrees.  Call after `avatar.reset`
        has populated skin transforms and before scene settling.
        """
        if self.avatar is None or getattr(self.avatar.robot, "skin", None) is None:
            return None
        target_xy = np.asarray(target_xy, dtype=np.float64).ravel()[:2]
        motion = None
        if motion_name is not None:
            self.avatar.play_animation(str(motion_name))
            motion = self.avatar.motion_modules.get(str(motion_name))
            if motion is None or not getattr(motion, "data", None):
                return None
            frame = int(motion_frame)
            if frame < 0:
                frame = len(motion.data) + frame
            frame = int(np.clip(frame, 0, len(motion.data) - 1))
            def apply_motion_frame():
                robot = self.avatar.robot
                robot.pose = motion.data[frame]
                robot.node_trans = motion.node_data[frame]
                robot.global_mat = motion.global_mat
                robot.global_mat_inv = motion.global_mat_inv
                robot.update()
        else:
            def apply_motion_frame():
                return None

        skin = self.avatar.robot.skin
        side = "Left" if int(hand_id) == 0 else "Right"
        total_yaw = 0.0
        pos = np.asarray(self.avatar.robot.global_trans, dtype=np.float64).copy()
        rot = np.asarray(self.avatar.robot.global_rot, dtype=np.float64).copy()
        for _ in range(10):
            apply_motion_frame()
            try:
                neck = to_numpy(skin.get_global_translation("Neck")[0]).ravel()[:3]
                head_top = to_numpy(skin.get_global_translation("HeadTop_End")[0]).ravel()[:3]
                head = 0.5 * (neck + head_top)
                wrist = to_numpy(skin.get_global_translation(f"{side}Hand")[0]).ravel()[:3]
                mid1 = to_numpy(skin.get_global_translation(f"{side}HandMiddle1")[0]).ravel()[:3]
                hand = 0.5 * (wrist + mid1)
            except Exception:
                return None

            current = np.asarray(hand[:2] - head[:2], dtype=np.float64)
            desired = np.asarray(target_xy - head[:2], dtype=np.float64)
            cur_n = float(np.linalg.norm(current))
            dst_n = float(np.linalg.norm(desired))
            if cur_n < 1e-8 or dst_n < 1e-8:
                return None
            current /= cur_n
            desired /= dst_n
            cross = current[0] * desired[1] - current[1] * desired[0]
            dot = float(np.clip(np.dot(current, desired), -1.0, 1.0))
            yaw = float(np.arctan2(cross, dot))
            if abs(yaw) < np.deg2rad(0.05):
                break
            cz, sz = np.cos(yaw), np.sin(yaw)
            rot_z = np.array(
                [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            rot = rot_z @ rot
            total_yaw += yaw
            self.avatar.reset(pos, rot)

        self.avatar.reset(pos, rot)
        self.avatar_init_pos = pos
        self.avatar_init_rot = rot
        if self.avatar_collider is not None:
            self.avatar_collider.update()
        return float(np.rad2deg(total_yaw))

    def load_actors(self):
        """Override in subclass: add task-specific objects to scene."""
        pass

    def play_once(self) -> bool:
        """Override in subclass: scripted task rollout. Return True on success."""
        raise NotImplementedError

    def check_success(self) -> bool:
        """Default success check.

        Passes when ``plan_success`` is True **and** every threshold set on
        ``self.target`` holds (distance / tilt / collision).  Tasks with
        bespoke criteria (grip forces, joint angles, multi-object …) should
        override this.
        """
        if not self.plan_success:
            return False
        spec = self.target
        metrics = self._compute_target_metrics()

        if spec.success_dist_xy is not None:
            if metrics is None or metrics["dist_xy"] > spec.success_dist_xy:
                return False
        if spec.success_dist_3d is not None:
            if metrics is None or metrics["dist_3d"] > spec.success_dist_3d:
                return False
        if spec.success_dz_min is not None:
            if metrics is None or metrics["dz"] < spec.success_dz_min:
                return False
        if spec.success_tilt_min_deg is not None:
            if metrics is None or "tilt_deg" not in metrics \
                    or metrics["tilt_deg"] < spec.success_tilt_min_deg:
                return False
        if spec.success_tilt_max_deg is not None:
            if metrics is None or "tilt_deg" not in metrics \
                    or metrics["tilt_deg"] > spec.success_tilt_max_deg:
                return False
        if spec.success_require_no_avatar_collision:
            summary = self.avatar_collision_summary()
            if not summary.get("enabled") or summary.get("any_collision"):
                return False
        return True

    # ------------------------------------------------------------------
    # Avatar-collision query API
    # ------------------------------------------------------------------

    def avatar_collision_occurred(self) -> bool:
        """Did any robot↔avatar collision register this episode?"""
        return bool(self.avatar_collided)

    def latest_avatar_collision(self) -> dict | None:
        """Most recent checker result, or None if never checked."""
        return self.avatar_collision_log[-1] if self.avatar_collision_log else None

    def hide_avatar_from_planning(self) -> None:
        """Blind-expert ablation: planning code sees no avatar body.

        Replaces the task-facing ``avatar_collider`` with a stand-in that
        reports no capsules or boxes, so obstacle clouds built from it omit
        the human. The collision/safety checkers keep their own reference to
        the real collider, so scoring is unchanged. Call after ``reset()``.
        """
        if self.avatar_collider is not None and not isinstance(
                self.avatar_collider, _PlanningBlindCollider):
            self.avatar_collider = _PlanningBlindCollider(self.avatar_collider)

    def avatar_collision_summary(self) -> dict:
        """Episode-level roll-up of the per-step log.

        Returns keys:
          enabled, n_checks, n_collisions, any_collision,
          deepest_depth_m, deepest_pair, first_collision_frame,
          per_capsule_frames  (dict: capsule_name → n_frames)
        """
        if self.avatar_collision_checker is None:
            return {"enabled": False}
        log = self.avatar_collision_log
        n_checks = len(log)
        collided = [r for r in log if r.get("collided")]
        deepest = None
        deepest_pair = None
        if collided:
            worst = min(collided, key=lambda r: r["min_distance"])
            deepest = float(worst["min_distance"])
            deepest_pair = worst.get("closest_pair")
        first_frame = collided[0]["frame"] if collided else None
        # FRAME_IDX only advances in record_frame(), so first_collision_frame
        # is 0 whenever trajectory recording is off (VLA eval).  The sim-step
        # index is the reliable timestamp.
        first_sim_step = collided[0].get("sim_step") if collided else None
        per_capsule: dict = {}
        ignored_capsule: dict = {}
        for r in collided:
            # Every (link, capsule) pair in `pairs` contributes one
            # frame-hit to that capsule's count.
            for _link, cap, _d in (r.get("pairs") or []):
                per_capsule[cap] = per_capsule.get(cap, 0) + 1
        for r in log:
            for _link, cap, _d in (r.get("ignored_pairs") or []):
                ignored_capsule[cap] = ignored_capsule.get(cap, 0) + 1
        return {
            "enabled": True,
            "n_checks": n_checks,
            "n_collisions": len(collided),
            "n_ignored_collisions": sum(
                1 for r in log if (r.get("ignored_pairs") or [])
            ),
            "any_collision": bool(collided),
            "deepest_depth_m": deepest,
            "deepest_pair": deepest_pair,
            "first_collision_frame": first_frame,
            "first_collision_sim_step": first_sim_step,
            "per_capsule_frames": per_capsule,
            "ignored_capsule_frames": ignored_capsule,
        }

    def avatar_safety_summary(self) -> dict:
        """Episode-level summary for the inflated avatar safety envelope."""
        if self.avatar_safety_checker is None:
            return {"enabled": False}
        log = self.avatar_safety_log
        n_checks = len(log)
        violations = [r for r in log if r.get("collided")]
        closest = None
        closest_pair = None
        if log:
            best = min(log, key=lambda r: r["min_distance"])
            closest = float(best["min_distance"])
            closest_pair = best.get("closest_pair")
        first_frame = violations[0]["frame"] if violations else None
        return {
            "enabled": True,
            "margin_m": float(self.avatar_safety_checker.margin),
            "n_checks": n_checks,
            "n_violations": len(violations),
            "any_violation": bool(violations),
            "min_distance_m": closest,
            "closest_pair": closest_pair,
            "first_violation_frame": first_frame,
        }

    def avatar_safety_intervention_summary(self) -> dict:
        """Episode-level summary for active retreat overrides."""
        if self.avatar_retreat_checker is None:
            return {"enabled": False}
        log = self.avatar_safety_intervention_log
        closest = None
        if log:
            closest = float(min(r["min_distance"] for r in log))
        return {
            "enabled": True,
            "margin_m": float(self.avatar_retreat_checker.margin),
            "n_interventions": len(log),
            "any_intervention": bool(log),
            "min_distance_m": closest,
            "first_intervention_frame": log[0]["frame"] if log else None,
        }

    def _sim_dt(self) -> float:
        """Physics dt (s) from the live scene; a few tasks override 0.002."""
        try:
            return float(self.scene.dt)
        except Exception:
            return 0.002

    def safety_metrics_summary(self) -> dict:
        """Post-hoc safety metrics (separation stats, intrusion ratios,
        collision rates, approach speed) over the task phase of the
        per-check distance log.  See envs/safety_metrics.py."""
        if self.avatar_collision_checker is None:
            return {"enabled": False}
        records = []
        for r in self.avatar_collision_log:
            if r.get("pretask"):
                continue
            rec = dict(r)
            if "scored_min_distance" in r:
                rec["min_distance"] = r["scored_min_distance"]
                rec["closest_pair"] = r.get("scored_closest_pair")
            records.append(rec)
        out = compute_safety_metrics(
            records,
            thresholds=self._safety_distance_thresholds,
            check_interval_s=self._avatar_collision_stride * self._sim_dt(),
        )
        out["n_pretask_checks_excluded"] = len(self.avatar_collision_log) - len(records)
        out["sim_dt"] = self._sim_dt()
        return out

    def safe_distance_summary(self) -> dict:
        """Episode-level closest robot-avatar signed distance.

        Positive values are clearance, zero is contact, and negative values
        are penetration depth.  Distances use the same sampled robot collision
        geometry and avatar capsule / palm-box model as avatar_collision.
        """
        if self.avatar_collision_checker is None:
            return {"enabled": False}

        def _distance_or_inf(record: dict) -> float:
            try:
                d = float(record.get("min_distance", float("inf")))
            except (TypeError, ValueError):
                return float("inf")
            return d if np.isfinite(d) else float("inf")

        n_checks = len(self.avatar_collision_log)
        log = [
            r for r in self.avatar_collision_log
            if np.isfinite(_distance_or_inf(r))
        ]
        if not log:
            return {
                "enabled": True,
                "n_checks": n_checks,
                "n_valid_distance_checks": 0,
                "safe_distance_m": None,
                "closest_pair": None,
                "closest_frame": None,
            }
        best = min(log, key=_distance_or_inf)
        return {
            "enabled": True,
            "n_checks": n_checks,
            "n_valid_distance_checks": len(log),
            "safe_distance_m": _distance_or_inf(best),
            "closest_pair": best.get("closest_pair"),
            "closest_frame": best.get("frame"),
        }

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    def _compute_target_metrics(self) -> Optional[dict]:
        """Resolve the spec into a raw-metrics dict, or None if unavailable.

        Keys: object_pos, target_pos, dist_xy, dist_3d, dz, and (if the spec
        supplies ``object_axis_fn``) tilt_deg = angle(axis, world +Z).
        """
        spec = self.target
        if spec.object is None or spec.position is None:
            return None

        if callable(spec.object_pos_fn):
            try:
                obj_pos = np.asarray(spec.object_pos_fn(),
                                     dtype=float).ravel()[:3]
            except Exception:
                return None
        else:
            entity = getattr(spec.object, "entity", spec.object)
            try:
                obj_pos = np.asarray(to_numpy(entity.get_pos()),
                                     dtype=float).ravel()[:3]
            except Exception:
                return None

        try:
            tgt_raw = spec.position() if callable(spec.position) else spec.position
            tgt_pos = np.asarray(tgt_raw, dtype=float).ravel()[:3]
        except Exception:
            return None

        metrics = {
            "object_pos": obj_pos,
            "target_pos": tgt_pos,
            "dist_xy": float(np.linalg.norm(obj_pos[:2] - tgt_pos[:2])),
            "dist_3d": float(np.linalg.norm(obj_pos - tgt_pos)),
            "dz": float(obj_pos[2] - tgt_pos[2]),
        }

        if callable(spec.object_axis_fn):
            try:
                axis = np.asarray(spec.object_axis_fn(),
                                  dtype=float).ravel()[:3]
                n = float(np.linalg.norm(axis))
                if n > 1e-8:
                    axis = axis / n
                    metrics["tilt_deg"] = float(np.degrees(
                        np.arccos(np.clip(axis[2], -1.0, 1.0))))
            except Exception:
                pass

        return metrics

    # Mapping from TargetSpec threshold attrs → output-dict keys.  Used by
    # evaluate() so thresholds that are set on the spec are echoed into the
    # pickled metrics dict for downstream analysis.
    _THRESHOLD_KEYS = (
        ("success_dist_xy",      "target_dist_xy_threshold"),
        ("success_dist_3d",      "target_dist_3d_threshold"),
        ("success_dz_min",       "target_dz_min_threshold"),
        ("success_tilt_min_deg", "target_tilt_min_deg_threshold"),
        ("success_tilt_max_deg", "target_tilt_max_deg_threshold"),
    )

    def evaluate(self) -> dict:
        """Standard per-episode metrics.

        Keys always present:
          success, plan_success, avatar_collision.
        Keys present iff ``self.target`` has an object + position:
          target_label, target_object_pos, target_pos,
          target_dist_xy, target_dist_3d, target_dz,
          target_tilt_deg (if object_axis_fn set),
          target_*_threshold (one per configured spec threshold).
        """
        out = {"plan_success": bool(self.plan_success), "success": False}
        try:
            out["success"] = bool(self.check_success())
        except NotImplementedError:
            out["check_success_error"] = "not implemented"
        except Exception as e:
            out["check_success_error"] = str(e)

        metrics = self._compute_target_metrics()
        if metrics is not None:
            spec = self.target
            out["target_label"] = spec.label
            out["target_object_pos"] = metrics["object_pos"].tolist()
            out["target_pos"] = metrics["target_pos"].tolist()
            out["target_dist_xy"] = metrics["dist_xy"]
            out["target_dist_3d"] = metrics["dist_3d"]
            out["target_dz"] = metrics["dz"]
            if "tilt_deg" in metrics:
                out["target_tilt_deg"] = metrics["tilt_deg"]
            for spec_key, out_key in self._THRESHOLD_KEYS:
                v = getattr(spec, spec_key)
                if v is not None:
                    out[out_key] = v
            if self.target.success_require_no_avatar_collision:
                out["target_require_no_avatar_collision"] = True

        avatar_collision = self.avatar_collision_summary()
        if avatar_collision.get("any_collision"):
            out["success"] = False
        out["avatar_collision"] = avatar_collision
        out["safe_distance"] = self.safe_distance_summary()
        out["avatar_safety"] = self.avatar_safety_summary()
        out["avatar_safety_intervention"] = self.avatar_safety_intervention_summary()
        out["safety"] = self.safety_metrics_summary()
        return out

    # ------------------------------------------------------------------
    # Motion helpers (for scripted rollouts)
    # ------------------------------------------------------------------

    def move_to_pose(self, pose, arm_tag: str):
        """Plan and execute motion to target pose. Returns plan result."""
        if not self.plan_success:
            return None
        if pose is None:
            self.plan_success = False
            return None

        pose7 = np.asarray(pose, dtype=np.float64).ravel()[:7]
        arm = self.robot.get_arm(arm_tag)
        result = arm.planner.plan_path(arm.get_arm_qpos(), pose7)

        if arm_tag == "left":
            self.left_joint_path.append(deepcopy(result.as_dict()))
        else:
            self.right_joint_path.append(deepcopy(result.as_dict()))

        if not result.success:
            self.plan_success = False
            return None

        return result

    def execute_plan(self, result, arm_tag: str, gripper_result=None):
        """Execute a planned trajectory."""
        if result is None:
            return

        teleport = os.environ.get("DEBUG_TELEPORT", "").strip() == "1"
        use_setqpos = os.environ.get("USE_SETQPOS", "").strip() == "1"
        n_steps = result.position.shape[0] if result.position.size > 0 else 0
        print(f"[execute_plan] trajectory steps={n_steps}, teleport={teleport}, setqpos={use_setqpos}")
        gripper_steps = len(gripper_result) if gripper_result is not None else 0
        max_steps = max(n_steps, gripper_steps)

        # Hold current gripper value throughout motion unless gripper_result overrides
        arm = self.robot.get_arm(arm_tag)
        hold_gripper_val = arm.gripper_val

        i = 0
        extra_steps = 0
        max_extra_cfg = self.config.get("avatar_retreat_max_extra_steps", 800)
        max_extra_steps = None if max_extra_cfg is None else int(max_extra_cfg)
        while i < max_steps:
            retreating = self._apply_avatar_retreat_if_needed((arm_tag,))
            if i < n_steps and not retreating:
                if teleport or use_setqpos:
                    self.robot.teleport_arm_joints(result.position[i], arm_tag)
                else:
                    self.robot.set_arm_joints(result.position[i], arm_tag)
            if retreating:
                pass
            elif gripper_result is not None and i < gripper_steps:
                self.robot.set_gripper(float(gripper_result[i]), arm_tag)
            else:
                self.robot.set_gripper(hold_gripper_val, arm_tag)

            self.step_sim()
            if retreating:
                extra_steps += 1
                if max_extra_steps is not None and extra_steps > max_extra_steps:
                    print("[avatar-retreat] max extra retreat steps reached; resuming nominal plan")
                    self._avatar_retreat_hold_count = 0
                    self._avatar_retreat_active_cached = False
                    i += 1
            else:
                i += 1

    def move_and_execute(self, pose, arm_tag: str):
        """Plan + execute in one call."""
        result = self.move_to_pose(pose, arm_tag)
        if result is not None:
            self.execute_plan(result, arm_tag)
        return result

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_video(self, path: str):
        """Start recording video frames from the recording camera."""
        self._video_path = path
        self._video_frames = []

    @contextlib.contextmanager
    def suppress_recording(self):
        """Context manager: inside it, ``step_sim`` records nothing (video,
        VLA, trajectory).  Wrap pre-task scene settling / avatar+scene layout
        so those static "let everything fall to rest" steps never land in the
        video or the training data.  Nests safely (restores the prior state)."""
        prev = self._record_suppressed
        self._record_suppressed = True
        try:
            yield
        finally:
            self._record_suppressed = prev

    def capture_frame(self):
        """Capture one frame from the recording + side cameras for video."""
        if self._video_path is None:
            return
        self._video_cap_tick += 1
        if self._video_cap_tick % self._video_stride != 0:
            return
        from .camera import _finalize_rgb
        cam_entry = self.cameras._cameras.get("recording")
        if cam_entry is None:
            return
        cam = cam_entry[0]
        out = cam.render(rgb=True)
        rgb_raw = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else None
        rec_frame = _finalize_rgb(cam, rgb_raw)
        if rec_frame is None:
            return

        # Tile with side camera if available
        side_entry = self.cameras._cameras.get("side")
        if side_entry is not None:
            side_cam = side_entry[0]
            side_out = side_cam.render(rgb=True)
            side_raw = side_out[0] if isinstance(side_out, (list, tuple)) and len(side_out) > 0 else None
            side_frame = _finalize_rgb(side_cam, side_raw)
            if side_frame is not None:
                rec_frame = np.concatenate([rec_frame, side_frame], axis=1)

        self._video_frames.append(rec_frame)

    def save_video(self, fps: int = 30):
        """Save captured frames as MP4 video."""
        if not self._video_frames or self._video_path is None:
            return
        import imageio
        os.makedirs(os.path.dirname(self._video_path), exist_ok=True)
        writer_kwargs = {"fps": fps}
        quality = self.config.get("video_quality")
        if quality is not None:
            writer_kwargs["quality"] = int(quality)
        ffmpeg_params = self.config.get("video_ffmpeg_params")
        if ffmpeg_params:
            writer_kwargs["ffmpeg_params"] = [str(x) for x in ffmpeg_params]
        try:
            writer = imageio.get_writer(
                self._video_path,
                codec=self.config.get("video_codec", "libx264"),
                macro_block_size=int(self.config.get("video_macro_block_size", 1)),
                **writer_kwargs,
            )
        except TypeError:
            writer = imageio.get_writer(self._video_path, **writer_kwargs)
        for frame in self._video_frames:
            writer.append_data(frame)
        writer.close()
        print(f"  Video saved: {self._video_path} ({len(self._video_frames)} frames)")
        self._video_frames = []

    # ------------------------------------------------------------------
    # Data saving
    # ------------------------------------------------------------------

    def record_frame(self):
        """Record current state as a trajectory frame."""
        obs = self.get_obs()
        if self._record_cameras is not None:
            if "rgb" in obs:
                obs["rgb"] = {k: v for k, v in obs["rgb"].items() if k in self._record_cameras}
            if "depth" in obs:
                obs["depth"] = {k: v for k, v in obs["depth"].items() if k in self._record_cameras}
        self.traj_data.append(obs)
        self.FRAME_IDX += 1

    def save_trajectory(self, path: str):
        """Save recorded trajectory to pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "observations": self.traj_data,
            "left_joint_path": self.left_joint_path,
            "right_joint_path": self.right_joint_path,
            "instruction": self.instruction,
            "evaluation": self.evaluate(),
        }
        # Only attach the per-step collision log when the checker was
        # active this episode.  (The summary is already in the
        # `evaluation` dict via `avatar_collision_summary()`.)
        if payload["evaluation"].get("avatar_collision", {}).get("enabled"):
            payload["avatar_collision_log"] = self.avatar_collision_log
        if payload["evaluation"].get("avatar_safety_intervention", {}).get("enabled"):
            payload["avatar_safety_intervention_log"] = self.avatar_safety_intervention_log
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def _move_to_pose(self, pose, arm_tag: str):
        """IK-based single-step move (for take_action ee mode)."""
        arm = self.robot.get_arm(arm_tag)
        result = arm.planner.solve_ik(arm.get_arm_qpos(), pose, num_waypoints=1)
        if result.success and result.position.size > 0:
            self.robot.set_arm_joints(result.position[-1], arm_tag)
