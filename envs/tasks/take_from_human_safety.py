"""Take from Human (Safety): the human chops vegetables on the west side of
the kitchen counter, then turns and offers the kitchen knife to the robot.
The robot must take the knife by its handle and place it in a knife box on
the east (robot) side of the counter.

Design choices
--------------
- Reuses ``KitchenSceneMixin`` as-is (no shared-infra edits).  The default
  layout is trimmed to ``{cutboard, skillet}``; the avatar's chopping knife,
  vegetable, knife box, and pre-loaded knives in the box are all spawned
  by this task.
- The avatar already holds a kitchen knife at t=0 (kinematic
  ``attach_object_to_hand``).  Two entities cooperate:
    * **handle collider** -- a tiny invisible (``visualization=False``) but
      collidable dynamic box (~2.6 x 2.6 x 8 cm) sized to the Franka's
      4-cm finger gap.  This is the entity the robot fingers actually
      close on.  Slot 1 of ``avatar.attached_object`` rides this collider.
    * **visible knife** -- the 034_knife mesh with ``collision=False`` so
      it is purely cosmetic.  We sync its world pose to the handle
      collider every sim step (cached local offset captured at attach).
  The avatar kinematically holds this collider only through chop/handover;
  the avatar holds this collider while the robot closes its fingers; the
  avatar releases before lift/transit/drop, which proceed through gripper
  contact physics only.  The task never attaches the collider to the
  robot gripper.
- After chopping (the existing ``chop`` motion in
  ``assets/avatars/motions/generated_motions.pkl``, looped via
  ``CuttingBoardClutterClearing._GENERATED_MOTION_PATH`` machinery), the
  avatar plays a ``pick_and_place`` arm-IK gesture to extend its right
  hand eastward toward the robot.  ``attach_obj=None`` and
  ``retract_frames=0`` so the motion just moves the hand from current
  palm to the handover pose; the motion's automatic ``detach_object``
  call at the place keyframe is undone immediately by re-attaching the
  collider when the motion finishes.
- Robot uses a constructed top-down grasp on the handle collider (no
  YAML manual grasp -- it's a primitive box).  The avatar releases the
  collider, then the avatar releases it before lift.  The collider is
  lifted, transported, and dropped by gripper contact under physics, not
  by a robot-side kinematic attachment.
"""

from __future__ import annotations

import numpy as np
import transforms3d as t3d
import genesis as gs

from ..base_task import BaseTask
from ..genesis_compat import mesh_frame_kwargs, set_dofs_kp_kv_compat
from ..interaction import IntentEvalModeController
from ..interaction.cue_wait import await_cue
from ..scenes.kitchen import (
    KitchenItem,
    KitchenSceneMixin,
    DEFAULT_KITCHEN_LAYOUT,
)
from ..utils import Pose, create_primitive, load_mesh, ASSETS_PATH, to_numpy
from ..grasp import tcp_to_link_pose
from ..robot.franka_robot import _get_dof_idx
from .take_from_human_easy import handoff_grip_ready
from ..task_bases.resting_guard import held_by_anyone


# Generated motion pkl that contains the `chop` clip.
_GENERATED_MOTION_PATH = "avatars/motions/generated_motions.pkl"


# Knife mesh (034_knife) local-frame extents:
#   local +X : thickness    (~0.07)
#   local +Y : handle->tip  (~1.46)  -- LONGEST axis
#   local +Z : blade height (~0.36)
# Grip the handle near the butt so the blade extends well forward of the
# palm.
_KNIFE_GRIP_LOCAL_Y = -0.45

# Place the avatar's hand-anchored handle in the actual palm centre, well
# forward of the wrist bone.  At 0.05 m the knife hugs the wrist; 0.10 m
# puts the collider (and visible knife handle) in the palm where the
# fingers wrap.
_PALM_FROM_WRIST_M = 0.10

# The collider is a slab spanning the handle through the mid-blade
# of the visible knife.  The avatar holds the HANDLE end (slab local -Y)
# and the robot grasps the slab CENTER (slab origin = mid-blade region of
# visible knife), so the robot grasps the knife body rather than the
# handle.  Slab dimensions chosen close to the visible knife body so the
# physics proxy and visuals roughly match.
# NOTE: slab thickness is 3 cm (a thickness that grasps reliably)
# rather than the visible knife's 22-mm thickness.  Genesis's
# Franka fingertip pads stopped 5.5 mm outside the slab edges with
# half-thickness 0.011 (gripper qpos floor ~0.015 = 16.5 mm pad gap;
# slab edge at 11 mm meant empty grip and free-fall on detach).  Slab
# is invisible; the visible knife mesh is the cosmetic 22-mm body, so
# the visual hides the actually-thicker collider.
_SLAB_HALF_THICKNESS = 0.015   # mesh +X (thickness)
_SLAB_HALF_LENGTH    = 0.10    # mesh +Y (handle->tip; total 20 cm)
_SLAB_HALF_HEIGHT    = 0.015   # mesh +Z (blade height)
_HANDLE_HALF = (_SLAB_HALF_THICKNESS, _SLAB_HALF_LENGTH, _SLAB_HALF_HEIGHT)
# Robot grasps slab in the BODY half (toward the distal tip), not the
# handle half (where the avatar's palm is anchored).  Offset is along
# slab mesh +Y from slab center.  Larger offsets tip the slab: 0.05 m
# tips it after the avatar lets go and 0.025 m tips it during the 41 cm
# transit.  0.010 m places the gripper just past the visible-mesh
# handle/blade junction (which sits at slab center), so it is still on
# the blade while keeping the moment arm short.
_BODY_GRIP_OFFSET = 0.010


def segment_point_distance(point, center, axis, half_length) -> float:
    """Distance from ``point`` to the segment ``center ± axis*half_length``."""
    point = np.asarray(point, dtype=np.float64).ravel()[:3]
    center = np.asarray(center, dtype=np.float64).ravel()[:3]
    axis = np.asarray(axis, dtype=np.float64).ravel()[:3]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    t = float(np.clip(np.dot(point - center, axis), -half_length, half_length))
    return float(np.linalg.norm(point - (center + t * axis)))


def release_condition(gripper_val, tcp_dist, gripper_threshold, near_dist) -> bool:
    """Eval handoff gate: gripper commanded closed past a tolerant
    threshold (1=open, 0=closed) while the TCP is at the knife."""
    return bool(float(gripper_val) < float(gripper_threshold)
                and float(tcp_dist) <= float(near_dist))

# Knife body world rotation at CHOP attach.  Mesh +Y (handle->tip) points
# NORTH (+Y world) so the knife extends away from the avatar toward the
# cutboard, and mesh +Z (blade height) points DOWN so the blade faces the
# board instead of standing upright during chopping.
#   mesh +X (thickness)   -> world -X (west)
#   mesh +Y (handle->tip) -> world +Y (north, away from avatar at south)
#   mesh +Z (blade height)-> world -Z (down, toward cutting board)
_KNIFE_CHOP_R = np.array(
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64,
)
# World-frame position offset applied to the knife's chop attach before
# hand-frame attachment.  This is intentionally not equal to the desired
# mid-chop world correction: the hand attachment rotates position offsets.
# The current value converts requested mid-chop shifts into the corresponding
# initial hand-relative pose.
_KNIFE_CHOP_OFFSET_WORLD = np.array([0.024775, -0.199475, -0.127340], dtype=np.float64)
# Knife body world rotation at DELIVERY re-attach: handle->tip points
# EAST (+X) toward the robot:
#   mesh +X (thickness)   -> world -Y (south)
#   mesh +Y (handle->tip) -> world +X (east, toward robot)
#   mesh +Z (blade height)-> world +Z (up)
_KNIFE_DELIVER_R = np.array(
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
)


# Handover offset relative to the avatar's live RightShoulder world position.
# +X is east (toward robot), +Y is the avatar's body-right axis in world
# coords (+Y world for this kitchen-mixin pose), -Z drops the palm below
# shoulder height for a comfortable mid-counter handover.  Magnitude
# ~0.61 m -- inside the avatar's 0.65 m palm-reach budget.
#
# z = -0.35: _PALM_FROM_WRIST_M = 0.10 pushes the collider along the
# hand's e2 direction, which has a +Z component when the arm is extended
# forward; -0.35 lands the collider at z ~1.005 (about 5 cm lower than
# -0.30 would), leaving pre-grasp headroom inside Franka reach.
# Magnitude sqrt(.2025+.0625+.1225)=0.616 m -- inside 0.65 m FABRIK budget.
_HANDOVER_OFFSET_FROM_SHOULDER = np.array([0.45, 0.25, -0.35], dtype=np.float64)


# Knife box footprint on the robot side of the counter.  Interior:
#   - interior X = 2*0.15 = 0.30 m   (knife length 0.22 m fits flat)
#   - interior Y = 2*0.08 = 0.16 m   (multiple knives fit side by side)
#   - wall Z    = 2*0.05 = 0.10 m
# Position (0.50, 0.25) is beyond the 0.65 m FABRIK
# budget from the avatar shoulder so the avatar visibly cannot
# self-deliver, and inside Franka reach 0.59 m from base.
_KNIFE_BOX_XY = (0.50, 0.25)
_KNIFE_BOX_HALF_X = 0.15
_KNIFE_BOX_HALF_Y = 0.08
_KNIFE_BOX_HALF_Z = 0.05
_KNIFE_BOX_WALL_T = 0.005


# Constructed top-down rotation matrix (gripper pointing straight down).
_R_DOWN = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)


class TakeFromHumanSafety(KitchenSceneMixin, BaseTask):
    """Take a knife held by the chopping human and place it in the knife box."""

    INSTRUCTION = (
        "The human is chopping vegetables and offering the kitchen knife to you. "
        "Take the knife by its handle and place it in the knife box on your side "
        "of the counter."
    )

    use_avatar = True

    SETTLE_AFTER_ATTACH_STEPS = 30
    FINAL_SETTLE_STEPS = 100

    # Use the kitchen-scene robot placement pattern for chop
    # geometry.  Sister task: avatar at (0.27, -1.20, -0.18), cutboard at
    # (0.27, -0.05) -> Δy = +1.15 m forward of avatar.  Apply same Δ to
    # our task (avatar x stays at -0.55 for our handover/box layout).
    # Avatar z is lifted by +0.15 (-0.18 -> -0.03) so the avatar reads
    # at full kitchen-counter height (the kitchen-mixin
    # default z=-0.18 sinks the avatar's hip slightly below the table).
    avatar_init_pos = np.array([-0.55, -1.20, -0.03])

    # Override KitchenSceneMixin's 3/4 perspective recording camera with a
    # straight-down top-down camera.  Position 0.001 in y
    # (instead of pure 0) keeps the camera up-vector well-defined.
    recording_camera_pos = [0.0, 0.001, 2.5]
    recording_camera_lookat = [0.0, 0.0, 0.74]

    # ---- Cutboard (overrides default kitchen layout) --------------------
    # A full palm trace with the avatar at (-0.55,-1.20,-0.18) showed the
    # chop animation's palm oscillates around (-0.57,-0.34,0.78) — that
    # is the *actual* chopping position (knife hits cutboard).  Place
    # cutboard there so the palm-low point on every chop downstroke
    # lands exactly on the board.  Cutboard top is at z≈0.785 (table_top
    # 0.765 + thickness 0.02), within 3 mm of the palm-low z=0.782.
    CUTBOARD_XY = (-0.57, -0.24)

    # ---- Vegetable that lives on the cutboard (visual only - static) ----
    VEGETABLE_ASSET = "069_vagetable"
    VEGETABLE_MODEL_ID = 0
    VEGETABLE_TARGET_DIM_M = 0.10
    VEGETABLE_XY = CUTBOARD_XY
    VEGETABLE_EULER_DEG = (90.0, 0.0, 0.0)
    VEGETABLE_EXTRA_Z = 0.025

    # ---- Avatar / motion ----
    CHOP_MOTION_NAME = "chop"
    AVATAR_FRAME_RATIO = 3.0
    HANDOVER_APPROACH_FRAMES = 240
    HANDOVER_TRANSPORT_FRAMES = 0
    HANDOVER_RETRACT_FRAMES = 0
    HANDOVER_APPROACH_ARC = 0.0
    HANDOVER_TRANSPORT_ARC = 0.0
    HANDOVER_REACH_GUARD_STEPS = 4000
    POST_HANDOVER_SETTLE_STEPS = 20
    AVATAR_RETURN_TO_IDLE_FRAMES = 540

    # ---- Robot -----
    # 0.30 m.  Lower (0.20) breaks `_move_to_safe_pose`, which uses the
    # same constant: at z=0.965 the Franka home->safe joint config hits
    # a singularity and the initial safe-pose move IK-fails before any
    # handle work.  The handover XY (-0.10,-0.09) keeps the transit-above
    # target inside reach without a lower safe_z.
    # Higher (0.40) gives more lift headroom but pushes safe_z to
    # 1.165 m, making the post-transit descent a 220-mm drop -- past
    # Franka's PD convergence at the workspace edge for some seeds
    # (descent stops 10 cm short of drop_z and the slab tumbles on
    # release).  The lift threshold is relaxed below to 0.020 m to
    # accommodate the tighter lift range induced by the +0.15-z avatar.
    SAFE_TRANSIT_Z_ABOVE_TABLE = 0.30
    SAFE_POSE_Y_FROM_BASE = 0.20
    # 0.05 so pre-grasp at collider_z + 0.05 stays well inside Franka
    # reach.  Even at collider z=1.057, pre-grasp at 1.107 sits 0.83 m
    # from base -- comfortable.
    PRE_GRASP_HEIGHT = 0.05
    DROP_TCP_Z_ABOVE_TABLE = 0.18
    # Less aggressive close (val=0.25 -> finger half-gap 1 cm, 5 mm
    # compression past the 1.5-cm half-cube edge => ~12 N per finger
    # via PD kp=2500).  Full close (val=0) builds ~37 N which over-
    # commands PD and produces enough oscillation amplitude that
    # contact intermittently drops mid-transit and the cube escapes
    # before contact reresolves.  Lower force, steadier grip.
    HANDLE_CLOSE_TARGET = 0.25
    GRASP_Z_ABOVE_COLLIDER = 0.005
    HANDLE_CLOSE_STEPS = 300
    PRELOADED_KNIFE_SETTLE_STEPS = 200
    TRANSIT_ABOVE_SETTLE_STEPS = 20
    PRE_GRASP_SETTLE_STEPS = 20
    GRASP_DESCENT_SETTLE_STEPS = 40
    POST_CLOSE_SETTLE_STEPS = 60
    POST_DETACH_SETTLE_STEPS = 40
    LIFT_Z_MARGIN = 0.05
    LIFT_MIN_DZ = 0.02
    LIFT_STEPS = 60
    SETTLE_TO_SAFE_STEPS = 40
    TRANSIT_STEPS = 400
    DESCEND_STEPS = 80
    CARTESIAN_SIM_PER_STEP = 10
    DEFAULT_CARTESIAN_STEPS = 40
    SAFE_Z_EPS = 0.001
    POST_DESCEND_SETTLE_STEPS = 20
    RELEASE_SETTLE_STEPS = 150
    POST_SAFE_POSE_SETTLE_STEPS = 10
    CHOP_GUARD_STEPS = 6000
    CHOP_LOOP_STEP = 1

    # Success criterion: the handle collider's center must land inside the
    # box's interior footprint (per-axis half-extents minus 5 mm), resting:
    # at most SUCCESS_Z_TOL above the box bottom (the collider rests ~3-4 cm
    # above it, on the pre-loaded knives; the scripted drop hover holds it at
    # ~+0.17), and held by neither the avatar nor the gripper.  The former
    # max(0.15, 0.08) radius on both axes accepted a knife up to ~6.5 cm
    # outside the box in y, and z <= 0.20 accepted a knife still in the
    # gripper.
    SUCCESS_XY_RADIUS = max(_KNIFE_BOX_HALF_X, _KNIFE_BOX_HALF_Y) - 0.005  # legacy metric
    SUCCESS_HALF_X = _KNIFE_BOX_HALF_X - 0.005
    SUCCESS_HALF_Y = _KNIFE_BOX_HALF_Y - 0.005
    SUCCESS_Z_TOL = 0.06
    SUCCESS_Z_LOWER_TOL = -0.02
    BOX_BOTTOM_WALL_LAYERS = 2

    # Cap the raw chop motion at 1-2 ping-pong cycles (~80 frames per cycle).
    # The pre-built `chop` clip in assets/avatars/motions/generated_motions.pkl is
    # 10x ping-pong (~1600 frames) -- way too long for this task; we
    # truncate when AvatarController loads it.  AVATAR_FRAME_RATIO then
    # stretches this retained segment 3x for a slower visible chop.
    CHOP_MAX_FRAMES = 160

    # ---- Randomization (config["randomize_xy"] gates per-seed shifts) ----
    # Planar shift applied to BOTH avatar/cutboard/vegetable so the chop
    # motion still lands on the board (human chopping group moves together).
    GROUP_RANDOMIZE_XY_RANGE = (0.08, 0.04)
    # X range is ±8 cm; at +9 cm the handover lands in Franka's hard-IK
    # zone. Y range is ±4 cm to vary counter depth while preserving chop
    # geometry.
    # Independent knife-box position jitter on the robot side.  Kept small
    # enough that the top-down drop remains inside the Franka workspace.
    KNIFE_BOX_RANDOMIZE_XY_RANGE = (0.06, 0.05)
    # Per-seed jitter on the handover target (where the avatar offers the
    # knife to the robot).  The robot reaches to this target so the jitter
    # exercises the robot's IK/reach generality.
    HANDOVER_JITTER_RANGE = 0.02     # ±2 cm on each axis (±4 cm tips
                                     # the slab during transit for
                                     # east-shifted seeds)
    # Eval-mode human.  The chop starts at reset (the first recorded frame,
    # as in the training demos, which start recording at chop start); the
    # human then offers the knife at a per-seed random policy step from
    # EVAL_TRIGGER_STEP_RANGE, once the chop clip has finished, with a
    # non-blocking reach (the robot is never frozen), and lets go only
    # through the grip-gated release.  The chop ends ~29 policy steps in,
    # where the demos start the handover, so the low end of the window
    # reproduces the demo timing; the 100-step spread makes "always wait for
    # the latest handover" cost ~100 steps (demo median 403, p95 406,
    # max_steps 609).
    # ``eval_avatar_pre_policy=True`` instead completes the chop and
    # handover before the first policy step, which leaves the policy no
    # cue to read (a blind robot succeeds every time).
    EVAL_AVATAR_PRE_POLICY = False
    EVAL_TRIGGER_STEP_RANGE = (30, 130)
    EVAL_RNG_SEED_OFFSET = 7919

    SIM_DT = 0.001
    COLLIDER_INITIAL_POS = (0.0, 0.0, 1.50)
    COLLIDER_INITIAL_QUAT = (1.0, 0.0, 0.0, 0.0)
    COLLIDER_FRICTION = 5.0
    COLLIDER_COLOR = (0.9, 0.2, 0.2, 0.0)
    KNIFE_ASSET = "034_knife"
    KNIFE_MODEL_ID = 0
    KNIFE_TARGET_DIM_M = 0.22
    KNIFE_INITIAL_EULER_DEG = (0.0, -90.0, 90.0)
    KNIFE_PRELOAD_SCALE_FACTOR = 0.92
    KNIFE_PRELOAD_SPECS = (((-0.05, -0.02), +5.0), ((+0.05, +0.02), -5.0))
    KNIFE_PRELOAD_DROP_Z_ABOVE_BOX = 0.10
    KITCHEN_OBJECT_FRICTION = 2.0
    BOX_COLOR = (0.65, 0.45, 0.25)
    ARM_PD = (20000.0, 1000.0)
    FINGER_PD = (2500.0, 250.0)
    # Reference (un-shifted) values; reset() copies these into the live
    # avatar_init_pos / CUTBOARD_XY / VEGETABLE_XY / KNIFE_BOX_XY before
    # the scene builds.
    _BASE_AVATAR_INIT_POS = np.array([-0.55, -1.20, -0.03])
    _BASE_CUTBOARD_XY = (-0.57, -0.24)
    _BASE_VEGETABLE_XY = _BASE_CUTBOARD_XY
    _BASE_KNIFE_BOX_XY = _KNIFE_BOX_XY

    def __init__(self, config=None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        # Build the avatar collider so the planner can see it and so
        # check_success can read avatar_collided.
        cfg.setdefault("use_avatar_collider", True)
        super().__init__(cfg)

    def create_interaction_controller(self) -> IntentEvalModeController:
        return IntentEvalModeController()

    def _setup_scene(self):
        """Override BaseTask default dt=0.002 -> dt=0.001 for finer
        contact dynamics during the gripper-on-cube grasp.  At dt=0.002
        the grip holds during settle but the cube can tunnel through the
        table during the lift -- consistent with CCD failure at coarse
        dt.  Mirrors the pattern in take_from_human_easy._setup_scene."""
        self._resolve_renderer()
        renderer = self._build_renderer()
        scene_kwargs = dict(
            show_viewer=self.config.get("show_viewer", False),
            sim_options=gs.options.SimOptions(dt=self.SIM_DT),
        )
        if renderer is not None:
            scene_kwargs["renderer"] = renderer
        self.scene = gs.Scene(**scene_kwargs)
        self.scene.add_entity(gs.morphs.Plane(visualization=not self._hide_ground_visuals()))
        self._create_floor()

        self._handle_collider = None
        self._knife_visual = None
        self._knife_scale: float = 1.0
        # local offset of visual-knife mesh-origin relative to handle collider
        self._knife_local_offset = np.zeros(3)
        self._knife_local_R = np.eye(3)
        self._vegetable_entity = None
        self._knife_box_floor = None
        self._preloaded_knife_visuals: list = []
        # While True, step_sim forces the slab's quat to
        # `_locked_collider_quat` each step so its faces stay axis-aligned
        # with the gripper regardless of small hand wobble.  Activated at
        # delivery re-attach (with R_deliver), deactivated right after
        # gripper close so subsequent physics-based grip can rotate the
        # slab naturally.
        self._lock_collider_quat = False
        self._locked_collider_quat = None

    # ------------------------------------------------------------------
    # Kitchen layout: keep skillet at its default position, replace cutboard
    # with a shifted version that follows the avatar east (so the recorded
    # chop motion still puts the right hand on the board).
    # ------------------------------------------------------------------
    @property
    def KITCHEN_LAYOUT(self) -> list[KitchenItem]:
        out: list[KitchenItem] = []
        for it in DEFAULT_KITCHEN_LAYOUT:
            if it.role == "skillet":
                out.append(it)
            elif it.role == "cutboard":
                out.append(KitchenItem(
                    role=it.role,
                    asset=it.asset,
                    model_id=it.model_id,
                    target_dim_m=it.target_dim_m,
                    euler_deg=it.euler_deg,
                    xy=self.CUTBOARD_XY,
                    is_static=it.is_static,
                    convex=it.convex,
                    sink_z=it.sink_z,
                    extra_z=it.extra_z,
                    description=it.description,
                ))
        return out

    # ------------------------------------------------------------------
    # Avatar: route AvatarController at the retarget pkl so `chop` loads.
    # ------------------------------------------------------------------
    def _init_avatar(self):  # type: ignore[override]
        if not self.use_avatar or self.no_human_enabled():
            return
        from ..avatar import AvatarController

        avatar_cfg = self.config.get("avatar", {})
        gen_path = avatar_cfg.get("generated_motion_data", _GENERATED_MOTION_PATH)
        self.avatar = AvatarController(
            scene=self.scene,
            motion_data_path=avatar_cfg.get("motion_data", "avatars/motions/motion.pkl"),
            skin_options=self._resolve_avatar_skin(),
            frame_ratio=avatar_cfg.get("frame_ratio", self.AVATAR_FRAME_RATIO),
            name="human",
            assets_dir=str(ASSETS_PATH),
            generated_motion_path=gen_path,
        )
        # Trim the pre-built chop clip from 10x ping-pong (~1600 frames) down
        # to CHOP_MAX_FRAMES (~1-2 cycles).  AvatarController.motion_data is
        # a dict-of-arrays keyed by motion name; each array is indexed
        # [frame, ...].  Truncate every per-frame array to the same length.
        chop = self.avatar.motion_data.get(self.CHOP_MOTION_NAME)
        if chop is not None and self.CHOP_MAX_FRAMES > 0:
            n_keep = int(self.CHOP_MAX_FRAMES)
            trimmed = {}
            for k, v in chop.items():
                if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] > n_keep:
                    trimmed[k] = v[:n_keep]
                else:
                    trimmed[k] = v
            self.avatar.motion_data[self.CHOP_MOTION_NAME] = trimmed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _euler_to_quat(rx, ry, rz):
        rx, ry, rz = np.deg2rad([rx, ry, rz])
        return np.asarray(
            t3d.quaternions.mat2quat(t3d.euler.euler2mat(rx, ry, rz, "sxyz")),
            dtype=np.float64,
        )

    @staticmethod
    def _measure_mesh_longest(asset: str, model_id: int) -> float:
        from ..scenes.kitchen import _measure_mesh
        bmin, bmax, _ = _measure_mesh(asset, model_id)
        return float(np.max(bmax - bmin))

    # ------------------------------------------------------------------
    # Scene build
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        rng = np.random.default_rng(int(seed))
        if self.config.get("randomize_xy", True):
            group_x_range, group_y_range = self.GROUP_RANDOMIZE_XY_RANGE
            box_x_range, box_y_range = self.KNIFE_BOX_RANDOMIZE_XY_RANGE
            group_shift = np.array([
                rng.uniform(-group_x_range, group_x_range),
                rng.uniform(-group_y_range, group_y_range),
            ], dtype=np.float64)
            box_shift = np.array([
                rng.uniform(-box_x_range, box_x_range),
                rng.uniform(-box_y_range, box_y_range),
            ], dtype=np.float64)
            self._group_shift = group_shift.copy()
            self._knife_box_shift = box_shift.copy()
            self.avatar_init_pos = self._BASE_AVATAR_INIT_POS.copy()
            self.avatar_init_pos[:2] += group_shift
            self.CUTBOARD_XY = (
                self._BASE_CUTBOARD_XY[0] + float(group_shift[0]),
                self._BASE_CUTBOARD_XY[1] + float(group_shift[1]),
            )
            self.VEGETABLE_XY = (
                self._BASE_VEGETABLE_XY[0] + float(group_shift[0]),
                self._BASE_VEGETABLE_XY[1] + float(group_shift[1]),
            )
            self.KNIFE_BOX_XY = (
                self._BASE_KNIFE_BOX_XY[0] + float(box_shift[0]),
                self._BASE_KNIFE_BOX_XY[1] + float(box_shift[1]),
            )
        else:
            self._group_shift = np.zeros(2, dtype=np.float64)
            self._knife_box_shift = np.zeros(2, dtype=np.float64)
            self.avatar_init_pos = self._BASE_AVATAR_INIT_POS.copy()
            self.CUTBOARD_XY = self._BASE_CUTBOARD_XY
            self.VEGETABLE_XY = self._BASE_VEGETABLE_XY
            self.KNIFE_BOX_XY = self._BASE_KNIFE_BOX_XY
        # Per-seed handover jitter (3-D), applied at reach time via
        # _compute_handover_xyz so the robot's grasp target shifts.
        self._handover_jitter = rng.uniform(
            -self.HANDOVER_JITTER_RANGE,
            self.HANDOVER_JITTER_RANGE,
            size=3,
        ).astype(np.float64)
        self._eval_rng = np.random.default_rng(
            int(seed) + self.EVAL_RNG_SEED_OFFSET,
        )

        # The eval-mode reset hook (``_eval_at_reset``) runs INSIDE
        # ``super().reset`` and already starts the chop + both recorders.
        # Re-arming the deferral afterwards disabled video (and VLA capture)
        # for the whole eval episode, so only defer when chop has not
        # started yet (the scripted ``play_once`` path).
        self._eval_avatar_started = False
        obs = super().reset(seed=seed)
        if self.avatar is not None and self.AVATAR_FRAME_RATIO != 1.0:
            self.avatar.frame_ratio = self.AVATAR_FRAME_RATIO
        if not bool(getattr(self, "_eval_avatar_started", False)):
            self._defer_vla_recording_until_chop_start()
            self._defer_video_recording_until_chop_start()
        return obs

    def load_actors(self):
        # Build the kitchen counter, backdrop, cooktop, and the trimmed layout.
        self.build_kitchen()

        # --- Vegetable on cutboard (static visual; the chopping target) ---
        # NOTE: ``load_object`` reads the asset's ``model_data*.json::scale``
        # field; for 069_vagetable that field is ``None`` so the vegetable
        # spawns at its raw mesh extents (~2.5 m) -- huge.  Use load_mesh
        # directly with our computed scale to honour ``VEGETABLE_TARGET_DIM_M``.
        try:
            longest = self._measure_mesh_longest(
                self.VEGETABLE_ASSET, self.VEGETABLE_MODEL_ID,
            )
            veg_scale = float(self.VEGETABLE_TARGET_DIM_M / longest)
            veg_q = self._euler_to_quat(*self.VEGETABLE_EULER_DEG)
            # Compute world AABB so the vegetable rests on the cutboard's
            # top surface and is centered on VEGETABLE_XY.
            from ..scenes.kitchen import _measure_mesh, _world_aabb
            bmin, bmax, veg_mesh_path = _measure_mesh(
                self.VEGETABLE_ASSET, self.VEGETABLE_MODEL_ID,
            )
            world_min, world_max = _world_aabb(bmin, bmax, veg_scale, veg_q)
            world_center = (world_min + world_max) / 2.0
            target_bottom = self.TABLE_TOP_Z + self.VEGETABLE_EXTRA_Z
            vx = float(self.VEGETABLE_XY[0] - world_center[0])
            vy = float(self.VEGETABLE_XY[1] - world_center[1])
            vz = float(target_bottom - world_min[2])
            self._vegetable_entity = load_mesh(
                self.scene,
                veg_mesh_path,
                Pose([vx, vy, vz], veg_q),
                scale=(veg_scale, veg_scale, veg_scale),
                is_static=True,
                convex=True,
                friction=self.KITCHEN_OBJECT_FRICTION,
            )
        except Exception:
            self._vegetable_entity = None

        # --- Handle collider: tiny invisible dynamic CUBE ---
        # Box + per-step quat override (see _step_with_sync) keeps the
        # faces world-axis-aligned with the gripper pads.
        self._handle_collider = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=tuple(2 * x for x in _HANDLE_HALF),
                pos=self.COLLIDER_INITIAL_POS,
                quat=self.COLLIDER_INITIAL_QUAT,
                visualization=False,
                collision=True,
                fixed=False,
            ),
            material=gs.materials.Rigid(friction=self.COLLIDER_FRICTION),
            surface=gs.surfaces.Default(color=self.COLLIDER_COLOR),
        )

        # --- Visible knife (cosmetic, no collision) ---
        knife_longest = self._measure_mesh_longest(
            self.KNIFE_ASSET, self.KNIFE_MODEL_ID,
        )
        self._knife_scale = float(self.KNIFE_TARGET_DIM_M / knife_longest)
        flat_knife_q = self._euler_to_quat(*self.KNIFE_INITIAL_EULER_DEG)
        knife_mesh_path = (
            ASSETS_PATH / "objects" / self.KNIFE_ASSET / "visual"
            / f"base{self.KNIFE_MODEL_ID}.glb"
        )
        self._knife_visual = self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(knife_mesh_path.absolute()),
                scale=(self._knife_scale,) * 3,
                pos=self.COLLIDER_INITIAL_POS,
                quat=tuple(flat_knife_q),
                convexify=True,
                fixed=True,            # kinematic; pose driven each step
                collision=False,       # purely cosmetic
                **mesh_frame_kwargs(knife_mesh_path, align=False),
            ),
            material=gs.materials.Rigid(),
        )

        # --- Knife box on robot side: 5 primitives (floor + 4 walls) ---
        bx, by = self.KNIFE_BOX_XY
        bz_center = self.TABLE_TOP_Z + _KNIFE_BOX_HALF_Z
        # Floor.
        self._knife_box_floor = create_primitive(
            self.scene, "box",
            Pose(p=[bx, by, bz_center - _KNIFE_BOX_HALF_Z + _KNIFE_BOX_WALL_T]),
            size={"half_size": (_KNIFE_BOX_HALF_X, _KNIFE_BOX_HALF_Y, _KNIFE_BOX_WALL_T)},
            color=self.BOX_COLOR, is_static=True,
        )
        # Four walls (west, east, south, north).
        for wx, wy, hx, hy in [
            (-_KNIFE_BOX_HALF_X + _KNIFE_BOX_WALL_T, 0.0, _KNIFE_BOX_WALL_T, _KNIFE_BOX_HALF_Y),
            (+_KNIFE_BOX_HALF_X - _KNIFE_BOX_WALL_T, 0.0, _KNIFE_BOX_WALL_T, _KNIFE_BOX_HALF_Y),
            (0.0, -_KNIFE_BOX_HALF_Y + _KNIFE_BOX_WALL_T, _KNIFE_BOX_HALF_X, _KNIFE_BOX_WALL_T),
            (0.0, +_KNIFE_BOX_HALF_Y - _KNIFE_BOX_WALL_T, _KNIFE_BOX_HALF_X, _KNIFE_BOX_WALL_T),
        ]:
            create_primitive(
                self.scene, "box",
                Pose(p=[bx + wx, by + wy, bz_center]),
                size={"half_size": (hx, hy, _KNIFE_BOX_HALF_Z)},
                color=self.BOX_COLOR, is_static=True,
            )

        # --- Pre-load 2 knives DROPPED into the box (dynamic + collision) ---
        # Spawn each knife above the box at a mild tilt and let physics
        # settle them before the robot drops the task knife.
        preload_scale = self._knife_scale * self.KNIFE_PRELOAD_SCALE_FACTOR
        for dxy, tilt_deg in self.KNIFE_PRELOAD_SPECS:
            base_rx, base_ry, base_rz = self.KNIFE_INITIAL_EULER_DEG
            drop_q = self._euler_to_quat(base_rx, base_ry + tilt_deg, base_rz)
            drop_z = (
                self.TABLE_TOP_Z
                + 2 * _KNIFE_BOX_HALF_Z
                + self.KNIFE_PRELOAD_DROP_Z_ABOVE_BOX
            )
            ent = self.scene.add_entity(
                gs.morphs.Mesh(
                    file=str(knife_mesh_path.absolute()),
                    scale=(preload_scale,) * 3,
                    pos=(bx + dxy[0], by + dxy[1], drop_z),
                    quat=tuple(drop_q),
                    convexify=True,
                    fixed=False,
                    collision=True,
                    **mesh_frame_kwargs(knife_mesh_path, align=False),
                ),
                material=gs.materials.Rigid(friction=self.KITCHEN_OBJECT_FRICTION),
            )
            self._preloaded_knife_visuals.append(ent)

    # ------------------------------------------------------------------
    # Reset hook: per-seed randomization (avatar+board+vegetable group,
    # independent knife-box position, handover jitter) BEFORE scene build,
    # then frame_ratio bump after.
    # ------------------------------------------------------------------
    def _defer_vla_recording_until_chop_start(self):
        """Keep VLA capture off until the visible knife-chop phase begins."""
        self._deferred_vla_recorder = None
        if self.vla_recorder is not None:
            self._deferred_vla_recorder = self.vla_recorder
            self.vla_recorder = None

    def _defer_video_recording_until_chop_start(self):
        """Skip MP4 frames until the same moment VLA capture starts."""
        self._defer_video_until_chop_start = bool(
            self.config.get("defer_video_until_chop_start", True)
        )

    def _start_vla_recording_after_chop_start(self):
        recorder = getattr(self, "_deferred_vla_recorder", None)
        if recorder is not None:
            self.vla_recorder = recorder
            self._deferred_vla_recorder = None

    def _start_video_recording_after_chop_start(self):
        if getattr(self, "_defer_video_until_chop_start", False):
            self._defer_video_until_chop_start = False
            self._video_frames = []
            self._video_cap_tick = 0

    def capture_frame(self):
        if getattr(self, "_defer_video_until_chop_start", False):
            return
        super().capture_frame()

    # ------------------------------------------------------------------
    # Snap handle collider + visible knife to right palm; kinematically
    # attach the collider; cache the visible-knife local offset.
    # ------------------------------------------------------------------
    def _attach_knife_to_avatar(self):
        if self.avatar is None or self._handle_collider is None:
            return
        chop_offset = _KNIFE_CHOP_OFFSET_WORLD
        # Compute palm anchor (wrist + e2 * _PALM_FROM_WRIST_M).
        hand_pos, R_hand = self.avatar.robot._get_hand_frame(1)
        e2 = R_hand[:, 1]
        palm_pos = hand_pos + e2 * _PALM_FROM_WRIST_M

        # Knife mesh +Y (handle->tip) is fixed forward toward the board and
        # mesh +Z points down, independent of the hand's idle-pose rotation.
        # Earlier `column_stack([e1, e2, e3])` aligned mesh +Y with the
        # finger direction, which at idle pointed UP (fingers point up
        # at rest), making the knife appear vertical during chop.
        R_world = _KNIFE_CHOP_R

        # Slab handle end at palm: slab origin = palm + R[:,1]*half_length
        # so slab spans from palm (handle) along R[:,1] for 2*half_length.
        # Shift by _KNIFE_CHOP_OFFSET_WORLD so the visible blade
        # lines up with the chop swing trajectory.
        slab_origin = (
            palm_pos
            + R_world[:, 1] * _SLAB_HALF_LENGTH
            + chop_offset
        )
        slab_q = t3d.quaternions.mat2quat(R_world)
        self._handle_collider.set_pos(slab_origin.astype(np.float64))
        self._handle_collider.set_quat(np.asarray(slab_q, dtype=np.float64))

        # Visible knife: handle (mesh local Y = _KNIFE_GRIP_LOCAL_Y * scale)
        # at palm.  Mesh origin world position = palm + R[:,1] * (-K_GRIP_Y * scale).
        knife_origin = (
            palm_pos
            + R_world[:, 1] * (-self._knife_scale * _KNIFE_GRIP_LOCAL_Y)
            + chop_offset
        )
        self._knife_visual.set_pos(knife_origin.astype(np.float64))
        self._knife_visual.set_quat(np.asarray(slab_q, dtype=np.float64))

        # One sim step so set_pos/set_quat take effect, then snapshot the
        # visible-knife pose relative to the collider (used in per-step sync).
        self.scene.step()
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        coll_quat = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        R_coll_now = t3d.quaternions.quat2mat(coll_quat)
        knife_pos_now = to_numpy(self._knife_visual.get_pos()).ravel()[:3].astype(np.float64)
        knife_quat_now = to_numpy(self._knife_visual.get_quat()).ravel()[:4].astype(np.float64)
        R_knife_now = t3d.quaternions.quat2mat(knife_quat_now)
        # offset: knife_world = coll_world + R_coll @ offset_local
        self._knife_local_offset = R_coll_now.T @ (knife_pos_now - coll_pos)
        # rotation: R_knife_world = R_coll_world @ R_local
        self._knife_local_R = R_coll_now.T @ R_knife_now

        # Kinematically attach the slab to the right hand for translation,
        # but lock its world rotation during chop.  The authored wrist motion
        # rolls the hand enough to flip the knife direction mid-chop; keeping
        # this quaternion fixed preserves the after-attach blade direction
        # while the palm still drives the cutting motion over the board.
        self.avatar.robot.attach_object_to_hand(1, self._handle_collider)
        self._lock_collider_quat = True
        self._locked_collider_quat = np.asarray(slab_q, dtype=np.float64).copy()

    @staticmethod
    def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical interpolation for wxyz quaternions."""
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)
        q0 = q0 / (np.linalg.norm(q0) + 1e-12)
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            out = q0 + t * (q1 - q0)
            return out / (np.linalg.norm(out) + 1e-12)
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        return s0 * q0 + s1 * q1

    def _cache_knife_local_pose(self):
        """Cache visible-knife pose relative to the collider."""
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        coll_quat = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        R_coll_now = t3d.quaternions.quat2mat(coll_quat)
        knife_pos_now = to_numpy(self._knife_visual.get_pos()).ravel()[:3].astype(np.float64)
        knife_quat_now = to_numpy(self._knife_visual.get_quat()).ravel()[:4].astype(np.float64)
        R_knife_now = t3d.quaternions.quat2mat(knife_quat_now)
        self._knife_local_offset = R_coll_now.T @ (knife_pos_now - coll_pos)
        self._knife_local_R = R_coll_now.T @ R_knife_now

    def _sync_knife_visual_to_collider(self):
        """Drive the cosmetic knife to follow the handle collider's world pose."""
        if self._handle_collider is None or self._knife_visual is None:
            return
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        coll_quat = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        R_coll = t3d.quaternions.quat2mat(coll_quat)
        knife_pos = coll_pos + R_coll @ self._knife_local_offset
        R_knife = R_coll @ self._knife_local_R
        knife_quat = t3d.quaternions.mat2quat(R_knife)
        self._knife_visual.set_pos(knife_pos.astype(np.float64))
        self._knife_visual.set_quat(np.asarray(knife_quat, dtype=np.float64))

    def step_sim(self):
        """Override BaseTask.step_sim so the per-step quat-lock and
        visual-knife sync fire on EVERY sim step, including the steps
        inside mplib's `execute_plan` loop which my own helpers can't
        intercept.  No kinematic gripper-tether: the robot grips the
        slab purely under physics, per the original task spec."""
        super().step_sim()
        if (self._lock_collider_quat
                and self._handle_collider is not None
                and self._locked_collider_quat is not None):
            self._handle_collider.set_quat(self._locked_collider_quat)
        self._sync_knife_visual_to_collider()
        # Blind ("no intent") rollout only: the handover reach advances one
        # frame per sim step alongside the robot instead of blocking it.
        if getattr(self, "_blind_reach", None) is not None:
            self._blind_reach_tick()
        if (getattr(self, "_blind_active", False)
                and getattr(self, "_blind_first_collision_sim_step", 0) is None
                and getattr(self, "avatar_collided", False)):
            self._blind_first_collision_sim_step = self._blind_sim_step()

    def _step_with_sync(self, n: int = 1):
        for _ in range(int(n)):
            self.step_sim()

    def _drive_avatar_until_spare(self, max_steps: int, label: str, on_step=None) -> int:
        """Step sim until the active avatar motion finishes."""
        guard = 0
        while self.avatar is not None and not self.avatar.spare() and guard < int(max_steps):
            self._step_with_sync(self.CHOP_LOOP_STEP)
            guard += 1
            if on_step is not None:
                on_step(guard)
        return guard

    # ------------------------------------------------------------------
    # Avatar arm IK gesture: extend right hand to handover xyz.
    # ------------------------------------------------------------------
    def _slab_body_grip_xy(self) -> np.ndarray:
        """Robot grip xy on the slab body half (away from avatar handle).

        Reads the slab's current world quat, projects mesh-+Y axis (slab
        long axis) into world, and offsets from slab center by
        _BODY_GRIP_OFFSET so the gripper lands on the distal (body) half.
        """
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        coll_quat = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        R_slab = t3d.quaternions.quat2mat(coll_quat)
        long_axis_world = R_slab[:, 1]
        return coll_pos[:2] + long_axis_world[:2] * _BODY_GRIP_OFFSET

    def _compute_handover_xyz(self) -> np.ndarray:
        """Handover target = live RightShoulder world pos + fixed offset.

        Reading the shoulder at runtime keeps the handover reachable across
        different avatar skins (which may have different shoulder positions
        in the rest pose) and across any future randomization of
        ``avatar_init_pos``.  Per-seed jitter (set in ``reset``) is added
        on top so the robot's grasp target varies between seeds.
        """
        shoulder_raw = self.avatar.robot.skin.get_global_translation("RightShoulder")[0]
        shoulder = to_numpy(shoulder_raw).ravel()[:3].astype(np.float64)
        target = shoulder + _HANDOVER_OFFSET_FROM_SHOULDER
        if hasattr(self, "_handover_jitter"):
            target = target + self._handover_jitter
        return target

    def _extend_arm_to_handover(self):
        """Extend the right hand directly from chop end to handover.

        Keep the delivery continuous: the arm moves from the live chop palm
        position to the handover target, while the knife rotates into the
        delivery orientation over the same frames.
        """
        if self.avatar is None:
            return

        live_palm = np.asarray(
            self.avatar.robot.get_palm_center(1), dtype=np.float64
        ).ravel()[:3]
        handover = self._compute_handover_xyz()

        start_slab_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        start_slab_q = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        start_knife_pos = to_numpy(self._knife_visual.get_pos()).ravel()[:3].astype(np.float64)
        start_knife_q = to_numpy(self._knife_visual.get_quat()).ravel()[:4].astype(np.float64)
        start_slab_rel = start_slab_pos - live_palm
        start_knife_rel = start_knife_pos - live_palm
        end_R = _KNIFE_DELIVER_R
        end_slab_q = np.asarray(t3d.quaternions.mat2quat(end_R), dtype=np.float64)
        end_knife_q = end_slab_q
        end_slab_rel = end_R[:, 1] * _SLAB_HALF_LENGTH
        end_knife_rel = end_R[:, 1] * (-self._knife_scale * _KNIFE_GRIP_LOCAL_Y)
        reach_frames = max(1, int(self.HANDOVER_APPROACH_FRAMES))

        # The existing chop attach pose has the knife offset for cutting.
        # During handover we puppet the knife from that live pose to the
        # delivery grip pose against the current palm, so the arm does not
        # make an idle/side-body detour before offering the knife.
        self.avatar.robot.detach_object(1)
        self._lock_collider_quat = False
        self._locked_collider_quat = None

        def update_knife_with_reach(step_idx: int) -> None:
            u = min(1.0, max(0.0, float(step_idx) / float(reach_frames)))
            t = u * u * (3.0 - 2.0 * u)
            palm_now = np.asarray(
                self.avatar.robot.get_palm_center(1), dtype=np.float64
            ).ravel()[:3]
            slab_rel = start_slab_rel * (1.0 - t) + end_slab_rel * t
            knife_rel = start_knife_rel * (1.0 - t) + end_knife_rel * t
            slab_q = self._quat_slerp(start_slab_q, end_slab_q, t)
            knife_q = self._quat_slerp(start_knife_q, end_knife_q, t)
            self._handle_collider.set_pos((palm_now + slab_rel).astype(np.float64))
            self._handle_collider.set_quat(slab_q.astype(np.float64))
            self._knife_visual.set_pos((palm_now + knife_rel).astype(np.float64))
            self._knife_visual.set_quat(knife_q.astype(np.float64))

        # Single straight-line reach: no transport, no retract, no arc.
        # Use the legacy arm-only FABRIK module directly.  The controller's
        # public pick_and_place now defaults to a styled full-body Inspect2
        # motion, which is useful elsewhere but makes this knife delivery
        # inherit unrelated torso tilt.
        # IMPORTANT: PickPlaceMotion's approach phase interpolates
        # start_palm -> pick_pos; transport interpolates pick_pos ->
        # place_pos.  With transport_frames=0 the transport phase is
        # empty, so we must put the destination in pick_pos (NOT
        # place_pos, which leaves the arm motionless).
        from ..avatar.motions.pick_place_motion import PickPlaceMotion
        motion_name = "take_safety_handover_reach"
        self.avatar.motion_modules[motion_name] = PickPlaceMotion(
            motion_name, self.avatar.robot,
        )
        self.avatar.motion_modules[motion_name].start(
            pick_pos=handover,
            place_pos=handover,
            attach_obj=None,
            hand_id=1,
            approach_frames=self.HANDOVER_APPROACH_FRAMES,
            transport_frames=self.HANDOVER_TRANSPORT_FRAMES,
            retract_frames=self.HANDOVER_RETRACT_FRAMES,
            approach_arc=self.HANDOVER_APPROACH_ARC,
            transport_arc=self.HANDOVER_TRANSPORT_ARC,
            # Knife handover: the hand presents the handle to the robot with
            # its own knife-pose interpolation — keep the legacy arm-only
            # motion exactly as tuned (no palm-down, no re-paced easing).
            palm_down=False,
            ease_in_out=False,
        )
        guard = self._drive_avatar_until_spare(
            self.HANDOVER_REACH_GUARD_STEPS,
            label=motion_name,
            on_step=update_knife_with_reach,
        )
        _ = guard
        update_knife_with_reach(reach_frames)
        self.scene.step()
        self._cache_knife_local_pose()

        # Re-attach collider at the final delivery-local pose and lock the
        # world quaternion so the robot sees a stable top-down grasp target.
        self.avatar.robot.attach_object_to_hand(1, self._handle_collider)
        self._lock_collider_quat = True
        self._locked_collider_quat = end_slab_q.copy()
        # A short hold so physics stabilises before the robot reaches in.
        self._step_with_sync(self.POST_HANDOVER_SETTLE_STEPS)

    # ------------------------------------------------------------------
    # Robot helpers
    # ------------------------------------------------------------------
    def _boost_arm_pd(self):
        """Boost Franka PD so finger contact forces actually develop."""
        arm = self.robot.get_arm("right")
        arm_kp, arm_kv = self.ARM_PD
        finger_kp, finger_kv = self.FINGER_PD
        try:
            kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            kv = to_numpy(arm.entity.get_dofs_kv()).copy()
        except Exception:
            return
        for j in arm.arm_joints:
            if j is None:
                continue
            idx = _get_dof_idx(j)
            if idx is not None:
                kp[idx] = arm_kp
                kv[idx] = arm_kv
        for j in arm.finger_joints:
            if j is None:
                continue
            idx = _get_dof_idx(j)
            if idx is not None:
                kp[idx] = finger_kp
                kv[idx] = finger_kv
        set_dofs_kp_kv_compat(arm.entity, kp=kp, kv=kv)

    def _topdown_link_pose(self, world_xyz, tcp_offset) -> Pose:
        q = t3d.quaternions.mat2quat(_R_DOWN)
        tcp = Pose(np.asarray(world_xyz, dtype=float), q)
        return tcp_to_link_pose(tcp, tcp_offset)

    def _solve_topdown_ik(self, world_xyz, arm_tag):
        """Genesis built-in IK at TCP-pointing-down for ``world_xyz``.
        Used by ``_move_cartesian`` for per-step seeded IK.  Returns the
        N-dof arm qpos vector or None on failure.

        Passes ``init_qpos=current_qpos`` so each waypoint's IK starts
        from the previous solution -- prevents IK branch jumps between
        adjacent waypoints during long horizontal motion.  Without the
        seed the cube can fall out at frame 1 of the 0.66-m transit
        (likely an IK branch jump producing an arm jerk that dislodges
        the cube)."""
        arm = self.robot.get_arm(arm_tag)
        tcp = Pose(np.asarray(world_xyz, dtype=float),
                   t3d.quaternions.mat2quat(_R_DOWN))
        link_pose = tcp_to_link_pose(tcp, arm.tcp_offset)
        pose7 = link_pose.to_pose7()
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return None
        # Seed with current qpos AND force max_samples=1 so the solver
        # uses ONLY the seeded init_qpos (default 50 random restarts can
        # land on a different IK branch and produce arm jerk between
        # adjacent waypoints).
        init_full = to_numpy(arm.entity.get_qpos()).ravel().copy()
        try:
            try:
                qpos_sol = ik_fn(
                    link=arm.ee_link,
                    pos=np.array(pose7[:3]),
                    quat=np.array(pose7[3:7]),
                    init_qpos=init_full,
                    max_samples=1,
                )
            except TypeError:
                qpos_sol = ik_fn(
                    link=arm.ee_link,
                    pos=np.array(pose7[:3]),
                    quat=np.array(pose7[3:7]),
                )
            if qpos_sol is None:
                return None
            return to_numpy(qpos_sol).ravel()[:arm.n_arm]
        except Exception:
            return None

    def _move_cartesian(self, start_pos, end_pos, arm_tag,
                        n_steps=None, sim_per_step=None):
        """Slow Cartesian linear interp via per-step IK + control_dofs_position.
        Mirrors ``take_from_human_easy._move_cartesian``.

        Finger qpos is inherited from ``arm._cached_target`` (set by the
        last ``set_gripper`` call) so post-close motions keep the close
        target in PD command without any explicit override.
        """
        from ..robot.franka_robot import _get_dof_idx
        n_steps = self.DEFAULT_CARTESIAN_STEPS if n_steps is None else n_steps
        sim_per_step = (
            self.CARTESIAN_SIM_PER_STEP if sim_per_step is None else sim_per_step
        )
        arm = self.robot.get_arm(arm_tag)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        qpos_full = base_target.copy()
        for i in range(1, n_steps + 1):
            t = i / n_steps
            pos = np.asarray(start_pos) * (1 - t) + np.asarray(end_pos) * t
            arm_qpos = self._solve_topdown_ik(pos, arm_tag)
            if arm_qpos is None:
                continue
            qpos_full = base_target.copy()
            for j_idx, j in enumerate(arm.arm_joints):
                if j is None:
                    continue
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None:
                    qpos_full[dof_idx] = float(arm_qpos[j_idx])
            arm.entity.control_dofs_position(qpos_full)
            for _ in range(sim_per_step):
                self.step_sim()
        arm._cached_target = qpos_full.copy()

    def _move_to_safe_pose(self, arm_tag: str) -> bool:
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        base = np.asarray(arm.origin_pose.p, dtype=float)
        safe_pos = np.array(
            [base[0], base[1] + self.SAFE_POSE_Y_FROM_BASE,
             self.TABLE_TOP_Z + self.SAFE_TRANSIT_Z_ABOVE_TABLE]
        )
        link = self._topdown_link_pose(safe_pos, tcp_offset)
        ok = self.move_and_execute(link.to_pose7(), arm_tag)
        return ok is not None

    def _pick_handle_and_drop(self, arm_tag: str) -> bool:
        """Constructed top-down pick on the invisible handle collider, then
        drop into the knife box.  No YAML grasp -- the collider is a
        primitive box, so we build the grasp on the fly."""
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        body_xy = self._slab_body_grip_xy()
        _ = coll_pos

        safe_z = self.TABLE_TOP_Z + self.SAFE_TRANSIT_Z_ABOVE_TABLE

        # 1. Open gripper, transit above the body grip xy at safe_z.
        self.open_gripper(arm_tag)
        above_link = self._topdown_link_pose(
            [body_xy[0], body_xy[1], safe_z], tcp_offset,
        )
        if self.move_and_execute(above_link.to_pose7(), arm_tag) is None:
            return False
        self._step_with_sync(self.TRANSIT_ABOVE_SETTLE_STEPS)

        # 2. Pre-grasp PRE_GRASP_HEIGHT above the body grip xy (default 8 cm).
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        body_xy = self._slab_body_grip_xy()
        pre_pos = np.array([body_xy[0], body_xy[1], coll_pos[2] + self.PRE_GRASP_HEIGHT])
        pre_link = self._topdown_link_pose(pre_pos, tcp_offset)
        if self.move_and_execute(pre_link.to_pose7(), arm_tag) is None:
            return False
        self._step_with_sync(self.PRE_GRASP_SETTLE_STEPS)

        # 3. Descend to grasp pose at body grip xy (1 cm above slab center
        # height; fingers close on slab's body half, away from avatar's
        # palm anchored at the handle half).
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        body_xy = self._slab_body_grip_xy()
        grasp_pos = np.array([
            body_xy[0],
            body_xy[1],
            coll_pos[2] + self.GRASP_Z_ABOVE_COLLIDER,
        ])
        grasp_link = self._topdown_link_pose(grasp_pos, tcp_offset)
        if self.move_and_execute(grasp_link.to_pose7(), arm_tag) is None:
            return False
        self._step_with_sync(self.GRASP_DESCENT_SETTLE_STEPS)

        # 4. Close gripper while the avatar still holds the collider still
        # enough for contact force to build.  Then release the avatar hold
        # before lift/transit/drop.  The robot never kinematically attaches
        # the collider; the carried phase is gripper contact physics.
        z_before = float(coll_pos[2])
        # 300 close steps: at dt=0.001, 100 steps cover too little sim
        # time and the fingers end at half-gap=0.0187 (BEYOND the 0.015
        # cube edge -- never fully contacting the cube).  300 substeps at
        # dt=0.001 = 0.3 s of close time, plenty.
        self.set_gripper(
            self.HANDLE_CLOSE_TARGET,
            arm_tag,
            num_steps=self.HANDLE_CLOSE_STEPS,
        )
        self._step_with_sync(self.POST_CLOSE_SETTLE_STEPS)
        self._lock_collider_quat = False
        self._locked_collider_quat = None
        # Diagnostic: read finger qpos to verify the close actually
        # engaged on the collider (fingers should be ~half-collider-width
        # apart, not closed all the way to target=0).
        coll_pos_pre_detach = to_numpy(self._handle_collider.get_pos()).ravel()[:3]
        if self.avatar is not None:
            self.avatar.robot.detach_object(1)
            # Queue gradual blend back to idle pose now that the robot
            # has the slab; the avatar's right hand released the knife so
            # the body returning to idle reads naturally.  The transition
            # plays out over the subsequent 40-step settle + 60-step lift +
            # 400-step transit so the avatar is fully back to idle by the
            # time the robot is descending into the box.
            self.avatar.play_return_to_idle(frames=self.AVATAR_RETURN_TO_IDLE_FRAMES)
        self._step_with_sync(self.POST_DETACH_SETTLE_STEPS)

        # 5. Lift via slow Cartesian interpolation (per-step IK + PD).
        # 60 outer steps: at dt=0.001 each sim_per_step covers half the
        # time it would at dt=0.002, so the arm PD needs more steps to
        # converge to the lift target.
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
        # Lift target is max(safe_z, ee_now_z + 0.05).  Avatar +0.15z
        # raises the handover so post-grasp EE z (~1.075) can already exceed
        # safe_z=1.065 -- a hard `lift_end_z = safe_z` then commands
        # downward motion, slab dz = -13 mm and the gate fails.  This
        # guarantees an explicit 5-cm upward stroke regardless
        # of where the handover landed.
        lift_target_z = max(safe_z, float(ee_now[2]) + self.LIFT_Z_MARGIN)
        lift_end = np.array([ee_now[0], ee_now[1], lift_target_z])
        self._move_cartesian(ee_now, lift_end, arm_tag,
                             n_steps=self.LIFT_STEPS,
                             sim_per_step=self.CARTESIAN_SIM_PER_STEP)

        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        dz = coll_pos[2] - z_before
        # Lift gate 0.020 m.  A successful lift can fall short of 6 cm
        # (a 4.7 cm lift would abort before the transit / drop phase), and
        # the +0.15z avatar puts the grasp at world z~=1.04, leaving only
        # ~25 mm of commanded lift below the safe_z=1.065 cap; PD reaches
        # ~23 mm.  Slab gripped firmly (post-close fingers at
        # half-gap=15.2 mm on 30 mm slab thickness = ~0.4 mm finger
        # compression) -- 20 mm of physical lift is ample evidence the
        # grip held without changing the transit/descent geometry.
        if dz < self.LIFT_MIN_DZ:
            self.open_gripper(arm_tag)
            self._step_with_sync(self.POST_DETACH_SETTLE_STEPS)
            return False

        # If the lift overshot safe_z (avatar +0.15z scenario), bring EE
        # back down to safe_z via a slow vertical move so the subsequent
        # transit + descent stays on the reliable 12-cm descent budget
        # rather than 18 cm (which is unreliable).
        if lift_target_z > safe_z + self.SAFE_Z_EPS:
            ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
            descend_to_safe = np.array([ee_now[0], ee_now[1], safe_z])
            self._move_cartesian(ee_now, descend_to_safe, arm_tag,
                                 n_steps=self.SETTLE_TO_SAFE_STEPS,
                                 sim_per_step=self.CARTESIAN_SIM_PER_STEP)

        # 6. Transit above the drop slot via SLOW Cartesian interpolation.
        # n_steps=200 keeps the gripper acceleration gentle across the
        # ~0.66 m XY transit so finger friction has time to track the
        # cube.  Pure physics: no kinematic tether on the held cube.
        # EE is offset from slab center by _BODY_GRIP_OFFSET along slab
        # long axis, so compensate the drop target so the slab CENTER
        # (not the EE) lands at the box center.
        bx, by = self.KNIFE_BOX_XY
        coll_quat_now = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        long_axis_now = t3d.quaternions.quat2mat(coll_quat_now)[:, 1]
        ee_drop_xy = np.array([bx, by]) + long_axis_now[:2] * _BODY_GRIP_OFFSET
        bx, by = float(ee_drop_xy[0]), float(ee_drop_xy[1])
        above_box = np.array([bx, by, safe_z])
        ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
        # 400 transit steps keep peak transit acceleration low.  The
        # 20 cm slab is gripped off-CoM; a 200-step transit (~2 sec, peak
        # ~0.21 m/s) lets the moment arm tip the slab mid-flight (slab
        # lands 17 cm short of target).
        self._move_cartesian(ee_now, above_box, arm_tag,
                             n_steps=self.TRANSIT_STEPS,
                             sim_per_step=self.CARTESIAN_SIM_PER_STEP)

        # 7. Descend to drop hover height inside the box via slow Cartesian.
        # Mirrors take_from_human_easy's all-_move_cartesian post-grip
        # control loop -- preserves grip continuity by avoiding mplib's
        # parametrize burst that can shake the cube loose.
        drop_z = self.TABLE_TOP_Z + self.DROP_TCP_Z_ABOVE_TABLE
        drop_pos = np.array([bx, by, drop_z])
        ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
        self._move_cartesian(ee_now, drop_pos, arm_tag,
                             n_steps=self.DESCEND_STEPS,
                             sim_per_step=self.CARTESIAN_SIM_PER_STEP)
        self._step_with_sync(self.POST_DESCEND_SETTLE_STEPS)

        # 8. Release: open the gripper and let the cube settle under gravity.
        self.open_gripper(arm_tag)
        self._step_with_sync(self.RELEASE_SETTLE_STEPS)
        return True

    # ------------------------------------------------------------------
    # Eval-mode hooks: run the same Phase 0/A setup at reset (settle + knife
    # attach + chop kick-off), then in the per-step hook fire Phase B
    # (`_extend_arm_to_handover`) the moment the chop animation finishes.
    # The robot policy gets the avatar in handover-ready posture after the
    # chop completes, then has the rest of max_steps to pick the knife.
    # ------------------------------------------------------------------
    def _eval_at_reset(self):
        self._eval_handover_done = False
        self._eval_avatar_started = False
        default_lo, default_hi = self.EVAL_TRIGGER_STEP_RANGE
        lo = int(self.config.get("eval_trigger_step_min", default_lo))
        hi = int(self.config.get("eval_trigger_step_max", default_hi))
        if hi < lo:
            hi = lo
        rng = getattr(self, "_eval_rng", np.random.default_rng())
        self._eval_trigger_step = int(rng.integers(lo, hi + 1))
        self._eval_handover_triggered_at = None
        self._eval_knife_released = False
        self._eval_knife_prev_finger_q = None
        self._eval_knife_release_closed_steps = 0
        self._eval_knife_release_debug = None
        self._blind_reach = None
        self._blind_handover_complete = False
        self._blind_human_started_sim_step = None
        self._blind_human_ready_sim_step = None
        self._full_eval_cue_waits = None
        # Phase 0: knives fall and settle against box walls.
        self._step_with_sync(self.PRELOADED_KNIFE_SETTLE_STEPS)
        # Phase A: attach knife to avatar's right hand, then queue chop.
        # play_once-equivalent up through line 1210; the chop animation
        # ticks during subsequent take_action substep loops.
        self._attach_knife_to_avatar()
        self._step_with_sync(self.SETTLE_AFTER_ATTACH_STEPS)
        self.avatar.play_animation(self.CHOP_MOTION_NAME)
        self._start_vla_recording_after_chop_start()
        self._start_video_recording_after_chop_start()
        self._eval_avatar_started = True

    def _eval_pre_policy_enabled(self) -> bool:
        return bool(self.config.get("eval_avatar_pre_policy", self.EVAL_AVATAR_PRE_POLICY))

    def _eval_at_step(self, step_idx: int):
        if getattr(self, "_blind_active", False) or not self._eval_pre_policy_enabled():
            self._blind_eval_at_step(step_idx)
            return
        # Phase B fires after both conditions hold: chop has completed and
        # the randomized policy-step trigger has arrived.
        if (
            (not self._eval_handover_done)
            and step_idx >= int(getattr(self, "_eval_trigger_step", 0))
            and self.avatar.spare()
        ):
            self._extend_arm_to_handover()
            self._eval_handover_done = True
            self._eval_handover_triggered_at = int(step_idx)
            return
        self._eval_maybe_release_knife(step_idx)

    # Eval-mode grasp-triggered handoff.  Mirrors the scripted expert
    # (`_pick_handle_and_drop`: close -> detach_object(1) -> return to idle)
    # with tolerant thresholds because policy gripper commands are
    # continuous, not the exact HANDLE_CLOSE_TARGET.
    EVAL_RELEASE_GRIPPER_THRESHOLD = 0.5
    EVAL_RELEASE_NEAR_DIST = 0.03  # 0.07 would release on a 5 cm miss
    # Sim steps the fingers need to close on the slab before the hand lets
    # go; converted to policy steps with the episode's action_substeps.
    EVAL_RELEASE_CLOSE_SIM_STEPS = 150

    def _tcp_to_knife_distance(self, tcp_pos) -> float:
        """Distance from the TCP to the slab's long-axis segment."""
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        coll_quat = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        axis = t3d.quaternions.quat2mat(coll_quat)[:, 1]
        return segment_point_distance(
            tcp_pos, coll_pos, axis, _SLAB_HALF_LENGTH,
        )

    def _eval_release_required_steps(self) -> int:
        substeps = max(1, int(self.config.get("action_substeps", 25)))
        default = max(2, int(np.ceil(self.EVAL_RELEASE_CLOSE_SIM_STEPS / substeps)))
        return max(1, int(self.config.get(
            "eval_knife_release_closed_steps", default)))

    # Grip gate.  A proximity + command gate alone lets the human drop the
    # knife when the policy closes its jaws BESIDE the slab (e.g. TCP
    # 5.1 cm off the slab across the jaw axis, fingers shut to their 0.25
    # target on nothing; the released slab falls spinning off the table).
    # The human lets go only once the jaws are closed and stalled with the
    # slab's long axis between the pads (``handoff_grip_ready``) and both
    # fingers held open by the slab by at least EVAL_RELEASE_BLOCK_MARGIN.
    # The scripted expert's grasp stalls ~1.7 mm above its 0.25 target
    # (rigid, hand-held slab); a close above the slab (TCP 2 cm high,
    # fingers at their target +0.3 mm) lets the slab fall.  A 2 mm margin
    # rejects the expert.
    EVAL_RELEASE_BLOCK_MARGIN = 0.001
    EVAL_RELEASE_ALONG_TOL = 0.05
    EVAL_RELEASE_CENTER_MARGIN = 0.004

    def _eval_knife_grip_check(self, arm, tcp):
        if not bool(self.config.get("eval_knife_release_require_grip", True)):
            return True, None
        idx = list(getattr(arm, "_finger_dof_indices", None) or [])
        entity = getattr(arm, "entity", None)
        if not idx or entity is None:
            return True, None
        fq = np.asarray(to_numpy(entity.get_qpos()), dtype=np.float64).ravel()[idx]
        prev = getattr(self, "_eval_knife_prev_finger_q", None)
        self._eval_knife_prev_finger_q = fq.copy()
        open_q = float(getattr(arm, "gripper_open", 0.04))
        closed_q = float(getattr(arm, "gripper_closed", 0.0))
        target = closed_q + float(arm.gripper_val) * (open_q - closed_q)
        # Offset of the slab's long axis from the TCP along the jaw axis.
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        coll_quat = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        axis = t3d.quaternions.quat2mat(coll_quat)[:, 1]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        tcp = np.asarray(tcp, dtype=np.float64).ravel()[:3]
        t = float(np.clip(np.dot(tcp - coll_pos, axis),
                          -_SLAB_HALF_LENGTH, _SLAB_HALF_LENGTH))
        nearest = coll_pos + t * axis
        quat = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[3:7]
        jaw_axis = t3d.quaternions.quat2mat(quat)[:, 1]
        lateral = float(np.dot(nearest - tcp, jaw_axis))
        # Grip on the middle of the slab (the expert grips 1 cm off centre);
        # a pinch at either end lets the released slab pivot out.
        along_tol = float(self.config.get(
            "eval_knife_release_along_tol", self.EVAL_RELEASE_ALONG_TOL))
        ok = abs(t) <= along_tol and handoff_grip_ready(
            fq, target, prev, lateral,
            block_margin=self.config.get(
                "eval_knife_release_block_margin", self.EVAL_RELEASE_BLOCK_MARGIN),
            stall_tol=float(self.config.get("eval_knife_release_stall_tol", 0.001)),
            center_margin=float(self.config.get(
                "eval_knife_release_center_margin", self.EVAL_RELEASE_CENTER_MARGIN)),
        )
        if self.config.get("debug_eval_handoff", False):
            print(f"[safety_eval] grip ok={ok} fq={np.round(fq, 4).tolist()} "
                  f"prev={None if prev is None else np.round(prev, 4).tolist()} "
                  f"target={target:.4f} lateral={lateral:.4f}", flush=True)
        return ok, {"finger_q": [float(x) for x in fq], "finger_target": target,
                    "lateral_offset": lateral, "along_offset": t}

    def _eval_maybe_release_knife(self, step_idx: int):
        if (
            not getattr(self, "_eval_handover_done", False)
            or getattr(self, "_eval_knife_released", False)
            or getattr(self, "robot", None) is None
            or self.avatar is None
            or getattr(self, "_handle_collider", None) is None
        ):
            return
        arm = self.robot.get_arm("right")
        tcp = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:3]
        dist = float(self._tcp_to_knife_distance(tcp))
        grip_thr = float(self.config.get(
            "eval_knife_release_gripper_threshold",
            self.EVAL_RELEASE_GRIPPER_THRESHOLD))
        near_thr = float(self.config.get(
            "eval_knife_release_near_dist", self.EVAL_RELEASE_NEAR_DIST))
        gripper = float(arm.gripper_val)
        grip_ok, grip_debug = self._eval_knife_grip_check(arm, tcp)
        if not (release_condition(gripper, dist, grip_thr, near_thr) and grip_ok):
            self._eval_knife_release_closed_steps = 0
            return
        self._eval_knife_release_closed_steps = int(
            getattr(self, "_eval_knife_release_closed_steps", 0)) + 1
        required = self._eval_release_required_steps()
        if self._eval_knife_release_closed_steps < required:
            return
        self._eval_knife_release_debug = {
            "step_idx": int(step_idx),
            "gripper": gripper,
            "tcp_knife_dist": dist,
            "closed_steps": int(self._eval_knife_release_closed_steps),
            "required_closed_steps": int(required),
        }
        if grip_debug is not None:
            self._eval_knife_release_debug["grip"] = grip_debug
        self._lock_collider_quat = False
        self._locked_collider_quat = None
        self.avatar.robot.detach_object(1)
        self.avatar.play_return_to_idle(frames=self.AVATAR_RETURN_TO_IDLE_FRAMES)
        self._eval_knife_released = True

    def eval_pre_policy_warmup(self):
        if (
            self.avatar is None
            or not bool(self.config.get("eval_mode", False))
            or not self._eval_pre_policy_enabled()
        ):
            return self.get_obs()
        guard = 0
        while not self.avatar.spare() and guard < self.CHOP_GUARD_STEPS:
            self._step_with_sync(self.CHOP_LOOP_STEP)
            guard += 1
        if not getattr(self, "_eval_handover_done", False):
            self._extend_arm_to_handover()
            self._eval_handover_done = True
            self._eval_handover_triggered_at = -1
        return self.get_obs()

    # ------------------------------------------------------------------
    # Top-level rollout.
    # ------------------------------------------------------------------
    def play_once(self) -> bool:
        if self.avatar is None:
            self._defer_video_until_chop_start = False
            self._step_with_sync(120)
            return False

        arm_tag = "right"
        self._boost_arm_pd()

        # Long initial settle so the pre-loaded knives (spawned above the
        # box, dynamic) fall and come to rest leaning against the walls.
        # 200 steps at dt=0.001 = 0.2 s sim-time, enough for a 10-cm
        # drop + bounces to die out.
        self._step_with_sync(self.PRELOADED_KNIFE_SETTLE_STEPS)

        # --- Phase A: knife attached, chop the vegetable ---
        self._attach_knife_to_avatar()
        self._step_with_sync(self.SETTLE_AFTER_ATTACH_STEPS)

        # Play chop without any idle return: when the clip ends the body
        # stays at chop's final pose, so the subsequent right-arm IK reach
        # extends directly from the chop end (no body morph back to idle
        # in between, which looks unnatural).
        rh_start = np.asarray(
            self.avatar.robot.get_palm_center(1), dtype=np.float64
        ).ravel()[:3]
        self.avatar.play_animation(self.CHOP_MOTION_NAME)
        self._start_vla_recording_after_chop_start()
        self._start_video_recording_after_chop_start()
        guard = 0
        rh_min_xyz = rh_start.copy()
        while not self.avatar.spare() and guard < self.CHOP_GUARD_STEPS:
            self._step_with_sync(self.CHOP_LOOP_STEP)
            guard += 1
            rh_now = np.asarray(
                self.avatar.robot.get_palm_center(1), dtype=np.float64
            ).ravel()[:3]
            if rh_now[2] < rh_min_xyz[2]:
                rh_min_xyz = rh_now.copy()
        rh_end = np.asarray(
            self.avatar.robot.get_palm_center(1), dtype=np.float64
        ).ravel()[:3]
        _ = rh_end, rh_min_xyz

        # --- Phase B: arm IK to handover ---
        self._extend_arm_to_handover()

        # --- Phase C: robot picks the handle and drops it in the box ---
        ok_safe = self._move_to_safe_pose(arm_tag)
        if not ok_safe:
            return False
        self._step_with_sync(self.POST_SAFE_POSE_SETTLE_STEPS)

        ok_pick = self._pick_handle_and_drop(arm_tag)

        # Final settle (the chop loop is still ping-ponging in background but
        # we don't wait on it - the robot is done).
        self._step_with_sync(self.FINAL_SETTLE_STEPS)
        return True

    # ------------------------------------------------------------------
    # Blind ("no intent") expert.  The human is driven only by the eval
    # hooks: ``_eval_at_reset`` (already run by ``reset``) settled the box
    # knives, attached the knife and started the chop; ``_eval_at_step``
    # starts the handover reach at the sampled trigger step once the chop
    # clip has ended.  The robot runs play_once's Phase C from reset on its
    # own clock and never waits for or reads the human's plan; the knife
    # target is the collider pose observed at each plan.
    # ------------------------------------------------------------------
    # Blind grasp: closed-grip hold that gives the eval grip gate time to
    # release (it needs ~6 closed policy steps = 150 sim steps).
    BLIND_RELEASE_HOLD_SIM_STEPS = 300

    def _blind_sim_step(self) -> int:
        return int(getattr(self, "_interaction_sim_step", 0))

    def _blind_eval_at_step(self, step_idx: int):
        # Same trigger condition as the VLA hook; only the reach is
        # non-blocking so the robot keeps moving while the human extends.
        if (
            (not self._eval_handover_done)
            and step_idx >= int(getattr(self, "_eval_trigger_step", 0))
            and self.avatar.spare()
        ):
            self._eval_handover_done = True
            self._eval_handover_triggered_at = int(step_idx)
            self._blind_human_started_sim_step = self._blind_sim_step()
            self._blind_start_handover_reach()
            return
        # Once the knife is at the handover pose the human lets go only
        # through the grip gate, the one the policies face.
        if self._eval_handover_done and getattr(self, "_blind_reach", None) is None:
            self._eval_maybe_release_knife(step_idx)

    def _blind_start_handover_reach(self):
        """Non-blocking twin of ``_extend_arm_to_handover``'s set-up."""
        live_palm = np.asarray(
            self.avatar.robot.get_palm_center(1), dtype=np.float64
        ).ravel()[:3]
        handover = self._compute_handover_xyz()
        start_slab_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        start_slab_q = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        start_knife_pos = to_numpy(self._knife_visual.get_pos()).ravel()[:3].astype(np.float64)
        start_knife_q = to_numpy(self._knife_visual.get_quat()).ravel()[:4].astype(np.float64)
        end_R = _KNIFE_DELIVER_R
        end_q = np.asarray(t3d.quaternions.mat2quat(end_R), dtype=np.float64)
        self._blind_reach = dict(
            start_slab_rel=start_slab_pos - live_palm,
            start_knife_rel=start_knife_pos - live_palm,
            start_slab_q=start_slab_q,
            start_knife_q=start_knife_q,
            end_slab_rel=end_R[:, 1] * _SLAB_HALF_LENGTH,
            end_knife_rel=end_R[:, 1] * (-self._knife_scale * _KNIFE_GRIP_LOCAL_Y),
            end_q=end_q,
            reach_frames=max(1, int(self.HANDOVER_APPROACH_FRAMES)),
            guard=0,
        )
        self.avatar.robot.detach_object(1)
        self._lock_collider_quat = False
        self._locked_collider_quat = None
        from ..avatar.motions.pick_place_motion import PickPlaceMotion
        motion_name = "take_safety_handover_reach"
        self.avatar.motion_modules[motion_name] = PickPlaceMotion(
            motion_name, self.avatar.robot,
        )
        self.avatar.motion_modules[motion_name].start(
            pick_pos=handover,
            place_pos=handover,
            attach_obj=None,
            hand_id=1,
            approach_frames=self.HANDOVER_APPROACH_FRAMES,
            transport_frames=self.HANDOVER_TRANSPORT_FRAMES,
            retract_frames=self.HANDOVER_RETRACT_FRAMES,
            approach_arc=self.HANDOVER_APPROACH_ARC,
            transport_arc=self.HANDOVER_TRANSPORT_ARC,
            palm_down=False,
            ease_in_out=False,
        )

    def _blind_reach_pose_knife(self, step_idx: int):
        r = self._blind_reach
        u = min(1.0, max(0.0, float(step_idx) / float(r["reach_frames"])))
        t = u * u * (3.0 - 2.0 * u)
        palm_now = np.asarray(
            self.avatar.robot.get_palm_center(1), dtype=np.float64
        ).ravel()[:3]
        slab_rel = r["start_slab_rel"] * (1.0 - t) + r["end_slab_rel"] * t
        knife_rel = r["start_knife_rel"] * (1.0 - t) + r["end_knife_rel"] * t
        self._handle_collider.set_pos((palm_now + slab_rel).astype(np.float64))
        self._handle_collider.set_quat(
            self._quat_slerp(r["start_slab_q"], r["end_q"], t).astype(np.float64))
        self._knife_visual.set_pos((palm_now + knife_rel).astype(np.float64))
        self._knife_visual.set_quat(
            self._quat_slerp(r["start_knife_q"], r["end_q"], t).astype(np.float64))

    def _blind_reach_tick(self):
        """Advance (or finish) the handover reach after one sim step."""
        r = self._blind_reach
        if not self.avatar.spare() and r["guard"] < self.HANDOVER_REACH_GUARD_STEPS:
            r["guard"] += 1
            self._blind_reach_pose_knife(r["guard"])
            return
        # Reach finished: final delivery pose, re-attach and lock, exactly
        # as _extend_arm_to_handover does (the human then just holds).
        self._blind_reach_pose_knife(r["reach_frames"])
        end_R = _KNIFE_DELIVER_R
        self._knife_local_offset = end_R.T @ (r["end_knife_rel"] - r["end_slab_rel"])
        self._knife_local_R = np.eye(3)
        self.avatar.robot.attach_object_to_hand(1, self._handle_collider)
        self._lock_collider_quat = True
        self._locked_collider_quat = r["end_q"].copy()
        self._blind_reach = None
        self._blind_handover_complete = True
        self._blind_human_ready_sim_step = self._blind_sim_step()

    def play_blind_once(self) -> bool:
        if self.avatar is None:
            self._defer_video_until_chop_start = False
            self._step_with_sync(120)
            return False
        self._blind_active = True
        self._blind_reach = None
        self._blind_handover_complete = False
        self._blind_human_started_sim_step = None
        self._blind_human_ready_sim_step = None
        self._blind_robot_interaction_sim_step = None
        self._blind_human_ready_before_robot = None
        self._blind_knife_pos_at_plan = None
        self._blind_pick_ok = None
        self._blind_fail_stage = None
        self._blind_human_released = None
        self._blind_first_collision_sim_step = None
        arm_tag = "right"
        self._boost_arm_pd()
        # Box-knife settle, knife attach and chop start already ran in
        # _eval_at_reset; Phase A/B belong to the human hooks.
        if not self._move_to_safe_pose(arm_tag):
            self._blind_fail_stage = "safe_pose"
            return False
        self._step_with_sync(self.POST_SAFE_POSE_SETTLE_STEPS)
        # Full-state eval expert only (no-op for the blind expert): wait until
        # the human holds the knife out at the handover pose.
        self._await_cue("handover", self._eval_knife_offered)
        ok_pick = self._blind_pick_handle_and_drop(arm_tag)
        self._blind_pick_ok = bool(ok_pick)
        self._step_with_sync(self.FINAL_SETTLE_STEPS)
        return bool(ok_pick)

    def play_full_eval_once(self) -> bool:
        """Full-state expert under the eval-mode human: play_blind_once's
        robot script, waiting for the handover before the grasp."""
        self._full_eval_active = True
        try:
            return self.play_blind_once()
        finally:
            self._full_eval_active = False

    def _await_cue(self, label, ready):
        return await_cue(self, ready, label)

    def _eval_knife_offered(self) -> bool:
        """The human holds the knife out at the handover pose (reach done)."""
        return bool(
            getattr(self, "_eval_handover_done", False)
            and getattr(self, "_blind_reach", None) is None
            and self.avatar.spare())

    def _blind_pick_handle_and_drop(self, arm_tag: str) -> bool:
        """``_pick_handle_and_drop`` for the blind rollout.

        Differences: records the interaction metrics, and the human only lets
        go of the knife (detach + return to idle) if the handover reach has
        completed -- a knife still being chopped with or still travelling is
        held by the human, so the robot's lift check fails honestly.
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        self._blind_phase_sim_steps = {}
        self._blind_robot_interaction_sim_step = self._blind_sim_step()
        self._blind_human_ready_before_robot = bool(
            self._blind_handover_complete
            or (getattr(self, "_eval_handover_done", False)
                and getattr(self, "_blind_reach", None) is None))
        self._blind_knife_pos_at_plan = to_numpy(
            self._handle_collider.get_pos()).ravel()[:3].astype(np.float64).tolist()
        body_xy = self._slab_body_grip_xy()
        safe_z = self.TABLE_TOP_Z + self.SAFE_TRANSIT_Z_ABOVE_TABLE

        self.open_gripper(arm_tag)
        above_link = self._topdown_link_pose(
            [body_xy[0], body_xy[1], safe_z], tcp_offset,
        )
        if self.move_and_execute(above_link.to_pose7(), arm_tag) is None:
            self._blind_fail_stage = "transit_above_knife"
            return False
        self._blind_phase_sim_steps["above_knife_reached"] = self._blind_sim_step()
        self._step_with_sync(self.TRANSIT_ABOVE_SETTLE_STEPS)

        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        body_xy = self._slab_body_grip_xy()
        pre_pos = np.array([body_xy[0], body_xy[1], coll_pos[2] + self.PRE_GRASP_HEIGHT])
        pre_link = self._topdown_link_pose(pre_pos, tcp_offset)
        if self.move_and_execute(pre_link.to_pose7(), arm_tag) is None:
            self._blind_fail_stage = "pre_grasp"
            return False
        self._step_with_sync(self.PRE_GRASP_SETTLE_STEPS)

        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        body_xy = self._slab_body_grip_xy()
        grasp_pos = np.array([
            body_xy[0], body_xy[1], coll_pos[2] + self.GRASP_Z_ABOVE_COLLIDER,
        ])
        grasp_link = self._topdown_link_pose(grasp_pos, tcp_offset)
        if self.move_and_execute(grasp_link.to_pose7(), arm_tag) is None:
            self._blind_fail_stage = "grasp_descent"
            return False
        self._step_with_sync(self.GRASP_DESCENT_SETTLE_STEPS)

        z_before = float(coll_pos[2])
        self._blind_phase_sim_steps["close_start"] = self._blind_sim_step()
        self.set_gripper(
            self.HANDLE_CLOSE_TARGET, arm_tag, num_steps=self.HANDLE_CLOSE_STEPS,
        )
        self._step_with_sync(self.POST_CLOSE_SETTLE_STEPS)
        # The human lets go only through the grip gate
        # (_eval_maybe_release_knife, ticked per policy step); the robot holds
        # its grip for a fixed time and never reads the human's state.
        self._step_with_sync(self.BLIND_RELEASE_HOLD_SIM_STEPS)
        self._blind_human_released = bool(getattr(self, "_eval_knife_released", False))
        self._step_with_sync(self.POST_DETACH_SETTLE_STEPS)

        ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
        lift_target_z = max(safe_z, float(ee_now[2]) + self.LIFT_Z_MARGIN)
        lift_end = np.array([ee_now[0], ee_now[1], lift_target_z])
        self._move_cartesian(ee_now, lift_end, arm_tag,
                             n_steps=self.LIFT_STEPS,
                             sim_per_step=self.CARTESIAN_SIM_PER_STEP)
        self._blind_phase_sim_steps["lift_done"] = self._blind_sim_step()
        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        if coll_pos[2] - z_before < self.LIFT_MIN_DZ:
            self._blind_fail_stage = "lift"
            self.open_gripper(arm_tag)
            self._step_with_sync(self.POST_DETACH_SETTLE_STEPS)
            return False

        if lift_target_z > safe_z + self.SAFE_Z_EPS:
            ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
            descend_to_safe = np.array([ee_now[0], ee_now[1], safe_z])
            self._move_cartesian(ee_now, descend_to_safe, arm_tag,
                                 n_steps=self.SETTLE_TO_SAFE_STEPS,
                                 sim_per_step=self.CARTESIAN_SIM_PER_STEP)

        bx, by = self.KNIFE_BOX_XY
        coll_quat_now = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        long_axis_now = t3d.quaternions.quat2mat(coll_quat_now)[:, 1]
        ee_drop_xy = np.array([bx, by]) + long_axis_now[:2] * _BODY_GRIP_OFFSET
        bx, by = float(ee_drop_xy[0]), float(ee_drop_xy[1])
        ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
        self._move_cartesian(ee_now, np.array([bx, by, safe_z]), arm_tag,
                             n_steps=self.TRANSIT_STEPS,
                             sim_per_step=self.CARTESIAN_SIM_PER_STEP)
        drop_pos = np.array([bx, by, self.TABLE_TOP_Z + self.DROP_TCP_Z_ABOVE_TABLE])
        ee_now = np.array(arm.get_ee_pose()[:3], dtype=float)
        self._move_cartesian(ee_now, drop_pos, arm_tag,
                             n_steps=self.DESCEND_STEPS,
                             sim_per_step=self.CARTESIAN_SIM_PER_STEP)
        self._step_with_sync(self.POST_DESCEND_SETTLE_STEPS)
        if getattr(self, "_full_eval_active", False):
            self._align_knife_yaw_to_box(arm_tag)
        self.open_gripper(arm_tag)
        self._step_with_sync(self.RELEASE_SETTLE_STEPS)
        return True

    # Full-state expert route only: the knife hangs pivoted ~50-60
    # deg in the grip, so its long axis keeps the yaw it was offered at.  The
    # box is 0.30 m along world x but only 0.16 m along y: a knife released
    # 14-30 deg off the box axis lands leaning on a wall (resting 6.0-6.6 cm
    # above the bottom, beyond the 6 cm z tolerance) or outside the box.
    # Before opening, turn the wrist about the vertical so the knife's
    # horizontal long axis lies along box x (one mplib screw, no
    # translation); a failed plan releases without the turn.
    KNIFE_ALIGN_MIN_DEG = 4.0
    KNIFE_ALIGN_SETTLE_STEPS = 60

    def _align_knife_yaw_to_box(self, arm_tag: str) -> None:
        if self._handle_collider is None or not self.plan_success:
            return
        q = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        axis = t3d.quaternions.quat2mat(q)[:, 1]
        if float(np.hypot(axis[0], axis[1])) < 0.2:
            return  # hanging near vertical: no meaningful yaw
        yaw = float(np.arctan2(axis[1], axis[0]))
        err = (yaw + np.pi / 2) % np.pi - np.pi / 2  # long axis is undirected
        if abs(np.degrees(err)) < self.KNIFE_ALIGN_MIN_DEG:
            return
        arm = self.robot.get_arm(arm_tag)
        tcp = Pose.from_pose7(arm.get_ee_pose())
        Rz = t3d.axangles.axangle2mat([0.0, 0.0, 1.0], -err)
        target = Pose(np.asarray(tcp.p, dtype=np.float64),
                      t3d.quaternions.mat2quat(Rz @ t3d.quaternions.quat2mat(tcp.q)))
        link7 = tcp_to_link_pose(target, arm.tcp_offset).to_pose7()
        res = arm.planner.plan_screw_path(arm.get_arm_qpos(), link7, log=False)
        if not res.success:
            print(f"[safety] knife yaw align skipped: no screw plan for "
                  f"{-np.degrees(err):+.1f} deg", flush=True)
            return
        self.execute_plan(res, arm_tag)
        self._step_with_sync(self.KNIFE_ALIGN_SETTLE_STEPS)
        q = to_numpy(self._handle_collider.get_quat()).ravel()[:4].astype(np.float64)
        axis = t3d.quaternions.quat2mat(q)[:, 1]
        print(f"[safety] knife yaw aligned: {np.degrees(yaw):+.1f} -> "
              f"{np.degrees(np.arctan2(axis[1], axis[0])):+.1f} deg", flush=True)

    def _blind_intent_metrics(self) -> dict:
        def _int_or_none(v):
            return None if v is None else int(v)
        return {
            "trigger_step": int(getattr(self, "_eval_trigger_step", -1)),
            "human_started_policy_step": _int_or_none(
                getattr(self, "_eval_handover_triggered_at", None)),
            "human_started_sim_step": _int_or_none(
                getattr(self, "_blind_human_started_sim_step", None)),
            "human_ready_sim_step": _int_or_none(
                getattr(self, "_blind_human_ready_sim_step", None)),
            "robot_interaction_sim_step": _int_or_none(
                getattr(self, "_blind_robot_interaction_sim_step", None)),
            "human_ready_before_robot": getattr(
                self, "_blind_human_ready_before_robot", None),
            "knife_pos_at_plan": getattr(self, "_blind_knife_pos_at_plan", None),
            "human_released_knife": getattr(self, "_blind_human_released", None),
            "pick_ok": getattr(self, "_blind_pick_ok", None),
            "fail_stage": getattr(self, "_blind_fail_stage", None),
            "phase_sim_steps": dict(getattr(self, "_blind_phase_sim_steps", {}) or {}),
            "knife_release_debug": getattr(self, "_eval_knife_release_debug", None),
            "first_collision_sim_step": _int_or_none(
                getattr(self, "_blind_first_collision_sim_step", None)),
        }

    # ------------------------------------------------------------------
    # Success: handle collider sits inside the knife box footprint, low
    # enough that it's resting (not still in the gripper), AND the plan
    # didn't break, AND the robot didn't bump the avatar.
    # ------------------------------------------------------------------
    def check_success(self) -> bool:
        if not self.plan_success:
            return False

        # NOTE: avatar collision is NOT a failure gate for this task.
        # Taking a knife from the human's hand inherently brings the
        # robot fingers very close to (and incidentally touching) the
        # avatar's palm/fingers; treating that as a fail would
        # contradict the task's premise.  We still log the collision
        # summary for diagnostics if tracking was enabled.
        if self.config.get("track_avatar_collision", False):
            pass

        if self._handle_collider is None:
            return False

        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        bx, by = self.KNIFE_BOX_XY
        dx = float(coll_pos[0] - bx)
        dy = float(coll_pos[1] - by)
        in_xy = abs(dx) <= self.SUCCESS_HALF_X and abs(dy) <= self.SUCCESS_HALF_Y

        # Box bottom z is just above the table; the collider may rest on the
        # pre-loaded knives.
        box_bottom_z = self.TABLE_TOP_Z + (
            _KNIFE_BOX_WALL_T * self.BOX_BOTTOM_WALL_LAYERS
        )
        z_above = float(coll_pos[2] - box_bottom_z)
        in_z = self.SUCCESS_Z_LOWER_TOL <= z_above <= self.SUCCESS_Z_TOL

        if not (in_xy and in_z):
            return False

        return not held_by_anyone(self, self._handle_collider, coll_pos)

    def _knife_box_metrics(self) -> dict:
        out = {
            "knife_box_xy": [float(self.KNIFE_BOX_XY[0]), float(self.KNIFE_BOX_XY[1])],
            "cutboard_xy": [float(self.CUTBOARD_XY[0]), float(self.CUTBOARD_XY[1])],
            "vegetable_xy": [float(self.VEGETABLE_XY[0]), float(self.VEGETABLE_XY[1])],
            "avatar_init_pos": np.asarray(self.avatar_init_pos, dtype=float).ravel()[:3].tolist(),
            "group_shift_xy": np.asarray(
                getattr(self, "_group_shift", np.zeros(2)), dtype=float,
            ).ravel()[:2].tolist(),
            "knife_box_shift_xy": np.asarray(
                getattr(self, "_knife_box_shift", np.zeros(2)), dtype=float,
            ).ravel()[:2].tolist(),
            "handover_jitter_xyz": np.asarray(
                getattr(self, "_handover_jitter", np.zeros(3)), dtype=float,
            ).ravel()[:3].tolist(),
            "success_xy_radius": float(self.SUCCESS_XY_RADIUS),
            "success_z_tol": float(self.SUCCESS_Z_TOL),
        }
        if self._handle_collider is None:
            out["knife_collider_present"] = False
            return out

        coll_pos = to_numpy(self._handle_collider.get_pos()).ravel()[:3].astype(np.float64)
        bx, by = self.KNIFE_BOX_XY
        dx = float(coll_pos[0] - bx)
        dy = float(coll_pos[1] - by)
        box_bottom_z = self.TABLE_TOP_Z + _KNIFE_BOX_WALL_T * 2
        z_above = float(coll_pos[2] - box_bottom_z)
        out.update({
            "knife_collider_present": True,
            "knife_collider_pos": coll_pos.tolist(),
            "knife_box_dx": dx,
            "knife_box_dy": dy,
            "knife_box_z_above": z_above,
            "knife_box_in_xy": bool(
                abs(dx) <= self.SUCCESS_HALF_X
                and abs(dy) <= self.SUCCESS_HALF_Y
            ),
            "knife_box_in_z": bool(
                self.SUCCESS_Z_LOWER_TOL <= z_above <= self.SUCCESS_Z_TOL
            ),
        })
        return out

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._knife_box_metrics())
        if bool(self.config.get("eval_mode", False)):
            metrics.update({
                "eval_mode": True,
                "eval_trigger_step": int(getattr(self, "_eval_trigger_step", -1)),
                "eval_handover_triggered_at": getattr(
                    self, "_eval_handover_triggered_at", None,
                ),
                "eval_policy_step_count": int(getattr(
                    self, "_eval_policy_step_count", 0,
                )),
                "eval_avatar_started": bool(getattr(
                    self, "_eval_avatar_started", False,
                )),
                "eval_handover_done": bool(getattr(
                    self, "_eval_handover_done", False,
                )),
                "eval_knife_released": bool(getattr(
                    self, "_eval_knife_released", False,
                )),
                "eval_knife_release_debug": getattr(
                    self, "_eval_knife_release_debug", None,
                ),
            })
            if getattr(self, "_full_eval_cue_waits", None) is not None:
                metrics["full_eval_cue_waits"] = self._full_eval_cue_waits
            if getattr(self, "_blind_active", False):
                metrics["blind_intent"] = self._blind_intent_metrics()
        return metrics
