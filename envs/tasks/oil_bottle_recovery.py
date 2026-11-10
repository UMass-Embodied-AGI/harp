"""Oil Bottle Recovery: a knocked-over oil bottle next to a frying human;
the robot rights it and moves it to a safe zone away from the cooking area.

Scene
-----
- Kitchen counter (``KitchenSceneMixin``) with the same drops as the
  shared frying scene (``FryingSceneMixin``; skillet/mug/wineglass/plate/
  bread cleared from the cooktop footprint), plus the default ``oil`` entry dropped.
  The recovered bottle is placed upright in a middle-counter safe zone,
  away from the back-row vinegar / soy-sauce bottles.
- Avatar holds pan + fork and runs the ``frying`` shake-loop, identical
  to the shared frying scene.
- A single ``029_olive-oil`` bottle is loaded dynamically near the
  avatar's right-hand swing arc (east of the cooktop). After the
  animation has been running long enough for the right hand to swing
  past, the bottle is "knocked over": gravity takes the rest.

Flow
----
1. Ease into frame 0 of the frying motion, snap pan + fork to hands,
   and start the long ``frying`` shake-loop.
2. Settle, then trigger knock-over (tilt + small lateral velocity in
   the swing direction).  Step physics until the bottle rests on its
   side.
3. Robot picks the fallen bottle (``select_and_execute_grasp`` on the
   live tipped pose), with avatar capsule pointcloud fed to the planner
   so the transit avoids the avatar's arm.
4. Lift, restore upright in mid-air over the safe zone.
5. Lateral transit + descend to the safe-zone xy.  Release.
6. Drain the rest of the frying loop.
"""

from __future__ import annotations

import json
import numpy as np
import transforms3d as t3d

from ..base_task import BaseTask
from ..interaction.cue_wait import await_cue
from ..interaction.eval_mode import TaskDrivenEvalTickController
from ..genesis_compat import set_dofs_kp_kv_compat, update_dofs_kp_kv_compat
from ..utils import Pose, to_numpy
from ..scenes.kitchen import (
    KitchenSceneMixin,
    DEFAULT_KITCHEN_LAYOUT,
    KitchenItem,
)
from ..task_bases.frying_scene import (
    _FRYING_MOTION_PKL,
    _PAN_ASSET,
    _FORK_ASSET,
    _OIL_TARGET_DIM_M,
    _PAN_TARGET_LEN_M,
    _FORK_TARGET_LEN_M,
    _AVATAR_FRAME_RATIO,
    _DROP_ROLES,
    _upright_q,
    _mesh_fit_scale,
    FryingSceneMixin,
)

from ..object_catalog import get_entry
from ..task_bases.resting_guard import held_by_anyone

_RECOVERY_OIL_ENTRY = get_entry("029_olive-oil")
_RECOVERY_OIL_ID = _RECOVERY_OIL_ENTRY.object_id
_RECOVERY_OIL_MODEL_ID = _RECOVERY_OIL_ENTRY.model_id
_OIL_JITTER_XY = 0.015
_SAFE_TARGET_JITTER_XY = 0.02

# Spawn the bottle east of the cooktop (cooktop is at (0.0, -0.20)) inside
# the avatar's right-hand swing arc.  The right hand drives the fork
# above the pan during frying; this xy puts the bottle on the counter
# slightly east-and-south of the pan so the right hand passes overhead
# during shake cycles.
_BOTTLE_INIT_XY = (0.25, -0.35)

# Safe zone — middle counter, not the back-row oil slot.  This keeps
# the carried bottle away from the vinegar / soy-sauce bottles at y=+0.35.
_SAFE_TARGET_XY = (0.36, 0.08)

# Tilt seed applied to the upright bottle once the avatar's right hand
# has swung close enough to "look like it bumped it".  At 50° the bottle
# is well past its angle of repose for a straight-walled cylinder and
# falls under gravity rather than re-righting itself.
_KNOCK_TILT_DEG = 50.0

# Small lateral push velocity applied at the same moment so the bottle
# slides a few centimetres in the swing direction.  Genesis exposes
# ``set_dofs_velocity`` on the entity's free root-joint dofs (3 trans +
# 3 rot); we wrap the call in try/except because the API is not
# guaranteed across Genesis versions.
_KNOCK_PUSH_VEL = 0.30  # m/s, +x (east — bottle falls into open counter, away from west cooktop / avatar)

# Hand-to-bottle xy distance at which we trigger the knock-over.  The
# avatar's right hand sweeps across an ~20 cm arc above the pan during
# each shake cycle; 0.25 m is the closest approach we can expect when
# the bottle is parked at the east edge of that arc.
_KNOCK_TRIGGER_DIST_M = 0.25

# Upper bound on number of physics steps we'll wait for the trigger
# before falling back to an unconditional knock-over.  The 2120-frame
# frying motion at frame_ratio=4 covers ~17 s = ~8500 physics steps, so
# 1500 is well within one shake cycle.
_KNOCK_TRIGGER_TIMEOUT_STEPS = 1500

# Settle window after the knock.  The gradual-tilt approach gives the
# bottle nontrivial momentum (a short window left it drifting ~2 cm in x
# during the pre-grasp move).  With 600 steps the bottle could still be
# mid-fall (~22° tilt) when grasp sampling started, and gripper contact
# during descent kept rolling it.
_KNOCK_SETTLE_STEPS = 1500  # 1500 steps (3 s real-time) lets it settle into a stable horizontal pose before sampling.

# Friction 2.0 keeps the bottle in the gripper through a 30 cm
# horizontal swing without slip.  The bottle is not carried upright, so
# the frying scene's 4.0 is unnecessary; 4.0 with close=0.30 slipped on
# the lateral place because close=0.30 leaves a 2.4 cm gap (less
# compression than a full close + PD boost).
_OIL_FRICTION = 2.0

# Partial close avoids crushing the bottle fully shut in the gripper while
# still giving enough contact for the short carry to the middle counter.
_OIL_GRIP_CLOSE_TARGET = 0.04
_OIL_FINGER_KP = 9000.0
_OIL_FINGER_KV = 250.0
_PRE_UPRIGHT_LIFT_M = 0.12
# Plan rounds for the checked clearance lift (same as the non-clearance lift's
# wait-and-retry rounds; 300 sim steps of waiting between rounds).
_OIL_CLEARANCE_LIFT_ROUNDS = 3
_OIL_CLEARANCE_LIFT_WAIT_STEPS = 300
_OIL_CLEARANCE_LIFT_FALLBACK_M = (0.09, 0.06)
_UPRIGHT_PLACE_HOVER_M = 0.13
_UPRIGHT_ENDPOINT_HOLD_STEPS = 360
_SUCCESS_DIST_XY_M = 0.15
_SUCCESS_TILT_DEG = 30.0
# The recovered bottle rests on the counter: its origin within this of the
# upright table-resting height, and held by neither the avatar nor the
# gripper (closed gripper with the TCP within _SUCCESS_HELD_TCP_RADIUS_M of
# the bottle's bbox centre).  Without it a bottle held upright over the zone
# counted.
_SUCCESS_REST_DZ_M = 0.02
_SUCCESS_HELD_TCP_RADIUS_M = 0.12
# Blind-intent diagnostics: the human's knock counts as complete once the
# bottle lies on its side (tilt above this) and has come to rest.
_BLIND_READY_TILT_DEG = 60.0
_BLIND_READY_SPEED_M_S = 0.05

def _read_object_scale(asset: str, model_id: int = 0) -> float:
    from ..utils import ASSETS_PATH

    path = ASSETS_PATH / "objects" / asset / f"model_data{model_id}.json"
    if not path.exists():
        return 1.0
    raw = json.load(open(path)).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)


def _read_object_height(
    asset: str,
    model_id: int = 0,
    fallback: float = _OIL_TARGET_DIM_M,
) -> float:
    from ..utils import ASSETS_PATH

    path = ASSETS_PATH / "objects" / asset / f"model_data{model_id}.json"
    if not path.exists():
        return fallback
    data = json.load(open(path))
    extents = data.get("extents")
    if not extents:
        return fallback
    scale = _read_object_scale(asset, model_id)
    return float(max(extents) * scale)


def _upright_origin_z_for_bottom(
    asset: str,
    model_id: int,
    bottom_z: float,
    fallback_height: float = _OIL_TARGET_DIM_M,
) -> float:
    from ..utils import ASSETS_PATH

    path = ASSETS_PATH / "objects" / asset / f"model_data{model_id}.json"
    if not path.exists():
        return float(bottom_z + fallback_height / 2)
    data = json.load(open(path))
    center = np.asarray(data.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
    extents = np.asarray(data.get("extents", [0.0, fallback_height, 0.0]), dtype=np.float64)
    scale = _read_object_scale(asset, model_id)
    bmin = (center - extents / 2.0) * scale
    bmax = (center + extents / 2.0) * scale
    corners = np.array([
        [x, y, z]
        for x in (bmin[0], bmax[0])
        for y in (bmin[1], bmax[1])
        for z in (bmin[2], bmax[2])
    ], dtype=np.float64)
    R = t3d.quaternions.quat2mat(_upright_q())
    world_min = (R @ corners.T).T.min(axis=0)
    return float(bottom_z - world_min[2])


class OilBottleRecovery(KitchenSceneMixin, BaseTask):
    """Robot recovers a knocked-over oil bottle from the cooking zone."""

    INSTRUCTION = "recover the knocked-over oil bottle from the cooking area and place it upright"
    OBJECT_SET = ["029_olive-oil"]
    use_avatar = True
    # Eval-mode human.  The human starts frying at a per-seed random policy
    # step (25 sim steps) from [EVAL_TRIGGER_STEP_MIN, EVAL_TRIGGER_STEP_MAX]
    # and knocks the bottle over when its hand passes it (~8 steps later);
    # the robot must see the bottle fall and settle before it grasps.  In the
    # training demos frying starts at the first recorded frames and the robot
    # starts its recovery ~70 steps in (knock + 1500-sim-step settle), so the
    # low end of the window reproduces the demo timing; the 120-step spread
    # makes "always wait for the latest knock" cost ~120 steps (demo median
    # 284, p95 323, max_steps 485).
    # ``eval_avatar_pre_policy=True`` instead knocks and settles the bottle
    # before the first policy step (no cue to read).
    EVAL_AVATAR_PRE_POLICY = False
    EVAL_TRIGGER_STEP_MIN = 0
    EVAL_TRIGGER_STEP_MAX = 120
    EVAL_KNOCK_TIMEOUT_STEPS = 80
    _EVAL_RNG_SALT = 0x0B17

    # Drop the same conflict items as frying, the default ``oil`` (we
    # spawn our own dynamic copy), and the ``baguette`` (default
    # position xy=(-0.05, -0.33) sits right next to the cooktop and gets
    # in the way of the gripper's approach to the fallen bottle).
    KITCHEN_LAYOUT = tuple(
        KitchenItem(
            role=it.role, asset=it.asset, model_id=it.model_id,
            target_dim_m=it.target_dim_m,
            euler_deg=it.euler_deg, xy=it.xy,
            is_static=it.is_static, convex=it.convex,
            sink_z=it.sink_z, extra_z=it.extra_z, description=it.description,
        )
        for it in DEFAULT_KITCHEN_LAYOUT
        if it.role not in (_DROP_ROLES | {"oil", "baguette"})
    )

    # Default kitchen mount on the south side of the cooktop.  A mount at
    # (+0.55, +0.30) — north of the cooktop — would put the fallen bottle
    # (y≈-0.33) directly *behind* the robot; mplib's IK then has to flip
    # joint1 by ~180°, producing
    # approximate solutions whose execution landed the gripper 0.9 m off
    # target.  Mounting at (0.60, -0.30) (same as the frying scene
    # and the rest of the kitchen tasks) puts both the bottle and the
    # safe zone (0.48, 0.35) inside Franka's natural forward reach, with
    # joint1 swings under 100°.
    KITCHEN_ROBOT_KWARGS = {"pos": [0.60, -0.30, 0.765]}

    # Same cooktop position as frying so the held pan lands above it.
    KITCHEN_COOKTOP = {"xy": (0.0, -0.20)}

    # Avatar pose copied from frying.
    avatar_init_pos = np.array([+0.05, -1.05, -0.03])

    # Cameras roughly mirror frying but pulled back a bit so the bottle
    # at (0.25, -0.35), the avatar, and the safe zone at (0.48, 0.35)
    # are all in frame.
    recording_camera_pos = [-0.55, -0.55, 1.10]
    recording_camera_lookat = [+0.20, -0.05, 0.90]
    side_camera_pos = [+0.20, -0.10, 2.05]
    side_camera_lookat = [+0.20, -0.10, 0.85]

    def __init__(self, config=None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        # Avatar capsules feed the planner as obstacle pointcloud each
        # plan, as in the frying scene.
        cfg.setdefault("use_avatar_collider", True)
        cfg.setdefault("randomize_avatar", True)
        cfg.setdefault("oil_clearance_lift", True)
        avatar_cfg = dict(cfg.get("avatar") or {})
        avatar_cfg.setdefault("generated_motion_data", _FRYING_MOTION_PKL)
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)
        self.resolve_target_object()   # registry/override hook (size-1 set)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._init_eval_recovery_state()
            obs = self.get_obs()
        return obs

    def create_interaction_controller(self) -> TaskDrivenEvalTickController:
        return TaskDrivenEvalTickController()

    def take_action(self, action, action_type: str = "qpos"):
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
        # Success is scored by the evaluator's N-step hold rule
        # (envs/success_hold.py) on check_success, not latched here.
        return super().take_action(action, action_type=action_type)

    def load_actors(self):
        from ..utils import load_mesh, ASSETS_PATH

        self.build_kitchen()

        if bool(self.config.get("randomize_layout", False)):
            bottle_jitter = np.random.uniform(
                -_OIL_JITTER_XY, _OIL_JITTER_XY, size=2,
            )
            safe_jitter = np.random.uniform(
                -_SAFE_TARGET_JITTER_XY, _SAFE_TARGET_JITTER_XY, size=2,
            )
        else:
            bottle_jitter = np.zeros(2, dtype=np.float64)
            safe_jitter = np.zeros(2, dtype=np.float64)
        self._bottle_init_xy = (
            float(_BOTTLE_INIT_XY[0] + bottle_jitter[0]),
            float(_BOTTLE_INIT_XY[1] + bottle_jitter[1]),
        )
        self._safe_target_xy = (
            float(_SAFE_TARGET_XY[0] + safe_jitter[0]),
            float(_SAFE_TARGET_XY[1] + safe_jitter[1]),
        )

        # --- Oil bottle: dynamic, using the current recovery asset and
        #     runtime-scale body grasp annotations. ---
        oil_scale = _read_object_scale(_RECOVERY_OIL_ID, _RECOVERY_OIL_MODEL_ID)
        oil_height = _read_object_height(_RECOVERY_OIL_ID, _RECOVERY_OIL_MODEL_ID)
        self._oil_scale = float(oil_scale)
        self._oil_target_dim_m = oil_height
        self._oil_upright_origin_z_on_table = _upright_origin_z_for_bottom(
            _RECOVERY_OIL_ID,
            _RECOVERY_OIL_MODEL_ID,
            self.TABLE_TOP_Z + 0.005,
            oil_height,
        )
        oil_init_pose = Pose(
            np.array([
                self._bottle_init_xy[0],
                self._bottle_init_xy[1],
                self._oil_upright_origin_z_on_table,
            ]),
            _upright_q(),
        )
        self.oil_entity = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / _RECOVERY_OIL_ID / "visual" / f"base{_RECOVERY_OIL_MODEL_ID}.glb",
            oil_init_pose,
            scale=(oil_scale, oil_scale, oil_scale),
            convex=True,
            is_static=False,
        )
        try:
            self.oil_entity.set_friction(_OIL_FRICTION)
        except Exception:
            pass

        # --- Pan + fork: same kinematic-driven props as frying. ---
        self._pan_scale = _mesh_fit_scale(_PAN_ASSET, 0, _PAN_TARGET_LEN_M)
        self._fork_scale = _mesh_fit_scale(_FORK_ASSET, 0, _FORK_TARGET_LEN_M)
        hover_pan = Pose(
            [self.avatar_init_pos[0] - 0.30, self.avatar_init_pos[1] + 0.30,
             self.TABLE_TOP_Z + 0.30],
            _upright_q(),
        )
        hover_fork = Pose(
            [self.avatar_init_pos[0] + 0.30, self.avatar_init_pos[1] + 0.30,
             self.TABLE_TOP_Z + 0.30],
            _upright_q(),
        )
        self.pan_entity = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / _PAN_ASSET / "visual" / "base0.glb",
            hover_pan, scale=(self._pan_scale,) * 3, convex=True,
            is_static=True,
        )
        self.fork_entity = load_mesh(
            self.scene,
            ASSETS_PATH / "objects" / _FORK_ASSET / "visual" / "base0.glb",
            hover_fork, scale=(self._fork_scale,) * 3, convex=True,
            is_static=True,
        )

    # ------------------------------------------------------------------
    # Borrow the pan + fork drivers, hand-frame helpers, planner-obstacle
    # refresh, and bottle-to-link transform from FryingSceneMixin.
    # ------------------------------------------------------------------

    _PAN_GRIP_LOCAL_Z = FryingSceneMixin._PAN_GRIP_LOCAL_Z
    _FORK_GRIP_LOCAL_X = FryingSceneMixin._FORK_GRIP_LOCAL_X
    _FORK_FORWARD_GRIP_SHIFT_M = FryingSceneMixin._FORK_FORWARD_GRIP_SHIFT_M
    _FORK_RIGHT_GRIP_SHIFT_M = FryingSceneMixin._FORK_RIGHT_GRIP_SHIFT_M
    _PAN_BODY_CENTER_LOCAL_Z = FryingSceneMixin._PAN_BODY_CENTER_LOCAL_Z
    # These spatula attach nudges only affect the spatula's visible pose
    # (grasp quality depends on ``_AVATAR_INFLATE_FACTOR``, see below).
    #
    # The parent's fork drive
    # aims the blade at the PAN BODY CENTRE with a forced >=10 cm downward
    # component, which puts the whole visible blade (15-23.5 cm out along the
    # aim) inside the 3.25 cm-deep skillet — buried.  Raise the palm anchor
    # and tilt the aim back up so the blade hovers just above the rim while
    # still pointing at the pan.
    _FORK_NE_SHIFT_M = 0.03
    _FORK_UP_SHIFT_M = 0.06
    _FORK_EXTRA_DOWNTILT_DEG = -25.0
    _AVATAR_OBSTACLE_RES = FryingSceneMixin._AVATAR_OBSTACLE_RES
    # Override the parent's inflate (1.6, sized for the frying-pour swing)
    # down to 1.0.  At 1.6 the avatar's right-arm capsule near the cooktop
    # fattens enough to push mplib's pre-grasp IK into a wider RRT branch
    # search that lands the grasp 36–40 cm off target (silent IK relax);
    # the slimmer obstacle pcd snaps to the bottle on the first attempt.
    # OBR's grasp at (0.34,
    # −0.35, 0.79) is south of the cooktop and doesn't need the extra
    # padding the pour-and-tilt swing does.
    _AVATAR_INFLATE_FACTOR = 1.0
    # FryingSceneMixin._avatar_obstacle_points reads this when sampling
    # arm/hand capsules. OilBottleRecovery reuses that method but keeps the
    # base avatar inflation slimmer, so use no extra task-local arm padding.
    _AVATAR_ACTIVE_ARM_EXTRA_INFLATE_M = 0.0
    # The transit can produce a ~24 000-step plan_screw trajectory (toppra
    # duration ~97 s); 32 000 lets the long but legitimate cartesian
    # transit from grasp xy back to the safe-zone xy complete.
    _MAX_TRAJECTORY_STEPS = 32000
    _CABINET_X_HALF = FryingSceneMixin._CABINET_X_HALF
    _CABINET_Y_RANGE = FryingSceneMixin._CABINET_Y_RANGE
    _CABINET_Z_RANGE = FryingSceneMixin._CABINET_Z_RANGE

    _get_entity_pose = FryingSceneMixin._get_entity_pose
    _attach_items_to_hands = FryingSceneMixin._attach_items_to_hands
    _drive_pan_horizontal = FryingSceneMixin._drive_pan_horizontal
    _drive_fork_at_pan = FryingSceneMixin._drive_fork_at_pan
    _avatar_obstacle_points = FryingSceneMixin._avatar_obstacle_points
    _cabinet_obstacle_points = FryingSceneMixin._cabinet_obstacle_points
    _bottle_to_link = FryingSceneMixin._bottle_to_link
    step_sim = FryingSceneMixin.step_sim

    # Cooktop occupies x=(-0.76,-0.34), y=(0.07,0.49), z up to 0.78.
    # The parent's `_refresh_planner_obstacles` adds avatar + cabinet
    # but NOT the cooktop, so plan_path can report Success on
    # paths that sweep the elbow through the cooktop region — Genesis
    # physics then blocks j1 at runtime.  This override appends a
    # cooktop pcd to the parent's pcd so plan_path actually avoids it.
    _COOKTOP_X = (-0.78, -0.32)  # 2-cm margin around (-0.76, -0.34)
    _COOKTOP_Y = (+0.05, +0.51)
    _COOKTOP_Z = (0.74, 0.80)

    def _cooktop_obstacle_points(self) -> np.ndarray:
        step = self._AVATAR_OBSTACLE_RES
        xs = np.arange(self._COOKTOP_X[0], self._COOKTOP_X[1] + step, step)
        ys = np.arange(self._COOKTOP_Y[0], self._COOKTOP_Y[1] + step, step)
        zs = np.arange(self._COOKTOP_Z[0], self._COOKTOP_Z[1] + step, step)
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1).astype(np.float64)

    def _refresh_planner_obstacles(self, arm_tag: str):
        """Parent's avatar+cabinet pcd + cooktop pcd."""
        avatar_pts = self._avatar_obstacle_points()
        cabinet_pts = self._cabinet_obstacle_points()
        cooktop_pts = self._cooktop_obstacle_points()
        parts = [p for p in (avatar_pts, cabinet_pts, cooktop_pts) if p is not None]
        if not parts:
            return
        pts = np.concatenate(parts, axis=0)
        arm = self.robot.get_arm(arm_tag)
        try:
            arm.planner.update_obstacles(pts, resolution=self._AVATAR_OBSTACLE_RES)
        except Exception:
            pass

    def _plan_path_with_fk_check(
        self, link_target, arm_tag, arm,
        max_fk_retries=5, max_snap_retries=10, snap_steps=100,
        fk_tol=0.005, log=True, max_plan_steps=None,
    ):
        """``plan_path`` (mplib RRT + plan_pose IK) with **two layers**
        of retry:

        1. **Snapshot retry**: if plan_path returns "invalid start
           state" / "Cannot find valid solution" because the live
           avatar capsules block IK or the start qpos collides with
           the avatar that moved since sampling, step the sim
           ``snap_steps`` frames (avatar continues frying) and refresh
           the obstacle pcd, then retry.  Up to ``max_snap_retries``
           snapshots.

        2. **FK retry within a snapshot**: even when plan_path returns
           Success, mplib can silently relax IK — the trajectory's end
           qpos has FK 1-3 cm off the requested pose (project memory
           ``mplib_silent_ik_relax``).  At each snapshot we try up to
           ``max_fk_retries`` times and keep the trajectory whose end
           qpos has the smallest FK error.  Each plan_path call uses a
           fresh RRT seed, so different attempts often converge to
           different IK branches.

        No Genesis IK fallback (user constraint).  No set_qpos for the
        robot — the trajectory is played through the standard PD
        controller via ``execute_plan``.
        """
        planner = arm.planner
        backend = planner._mplib
        target = (backend.robot.get_base_pose().inv()
                  * planner._to_mplib_pose(link_target.to_pose7()))
        best_err = float("inf")
        clearance_mode = self.config.get("oil_clearance_lift", False) is True
        for snap_idx in range(max_snap_retries):
            if clearance_mode and getattr(self, "avatar_collided", False):
                self.plan_success = False
                return None, best_err
            if clearance_mode:
                from ..task_bases.oil_recovery import checked_oil_carry_start

                try:
                    plan_start = checked_oil_carry_start(self, arm)
                    measured_start = plan_start[np.asarray(backend.move_group_joint_indices)]
                except Exception as exc:
                    print(f"[oil-clearance-lift] rejected measured carry start: {exc}", flush=True)
                    self.plan_success = False
                    return None, best_err
            else:
                self._refresh_planner_obstacles(arm_tag)
                plan_start = measured_start = np.asarray(arm.get_arm_qpos(), dtype=float).copy()
            for attempt in range(max_fk_retries):
                res = arm.planner.plan_path(
                    plan_start, link_target.to_pose7(), log=False,
                )
                if res is None or not res.success or res.position.size == 0:
                    continue
                if (clearance_mode
                        and not np.allclose(res.position[0], measured_start, atol=1e-6, rtol=0)):
                    print("[oil-clearance-lift] rejected repaired native start", flush=True)
                    self.plan_success = False
                    return None, best_err
                # While HOLDING the bottle, a grotesquely long RRT plan is a
                # branch-flip joint unwind that spins/drops it (1600+-step
                # lift fallbacks lost the bottle every time).  With a cap set, such plans are never
                # eligible; keep retrying for a short one instead.
                if (max_plan_steps is not None
                        and res.position.shape[0] > max_plan_steps):
                    if log:
                        print(f"[obr] plan {res.position.shape[0]} steps > "
                              f"{max_plan_steps} cap = branch flip; retrying",
                              flush=True)
                    continue
                if clearance_mode:
                    end_q = plan_start.copy()
                    end_q[np.asarray(backend.move_group_joint_indices)] = res.position[-1]
                else:
                    end_q = backend.pad_move_group_qpos(
                        np.asarray(res.position[-1], dtype=float),
                    )
                backend.pinocchio_model.compute_forward_kinematics(end_q)
                actual = backend.pinocchio_model.get_link_pose(
                    backend.link_name_2_idx[backend.move_group],
                )
                fk_err = float(np.linalg.norm(
                    np.asarray(actual.p) - np.asarray(target.p),
                ))
                dot = float(abs(np.dot(actual.q, target.q)))
                angle_err = float(2 * np.arccos(np.clip(dot, 0.0, 1.0)))
                if fk_err < best_err:
                    best_err = fk_err
                if (np.isfinite(fk_err) and np.isfinite(angle_err)
                        and fk_err < fk_tol and angle_err < 0.05):
                    return res, fk_err
            # Avatar continues animating; step forward and try again so
            # the live frying motion has a chance to clear the IK region.
            for _ in range(snap_steps):
                self.step_sim()
        # No inaccurate "best" trajectory may escape the gate. Some lift
        # callers rely on result.success and do not separately inspect error.
        return None, best_err

    # ------------------------------------------------------------------
    # Knock-over
    # ------------------------------------------------------------------

    def _right_hand_xy(self) -> np.ndarray | None:
        """World-XY of the avatar's right-hand frame, or None if no avatar."""
        if self.avatar is None:
            return None
        try:
            hand_pos, _ = self.avatar.robot._get_hand_frame(1)
        except Exception:
            return None
        return np.asarray(hand_pos, dtype=np.float64)[:2]

    def _knock_over_bottle(self):
        """Apply a gradual angular velocity about world +y to topple the
        bottle smoothly eastward (long axis ends along world +x).  This
        avoids a +y-tilt + post-knock ``set_quat`` snap, which
        looks discontinuous; the bottle falls under physics from
        the moment of contact and settles naturally with long axis +x."""
        # Angular velocity = 50° / 0.4 s ≈ 2.18 rad/s about world +y; the
        # top of the upright bottle swings in +x direction and past the
        # angle-of-repose (~30°) gravity completes the topple.
        omega = float(np.deg2rad(_KNOCK_TILT_DEG)) / 0.4
        try:
            # 6 free-body dofs: [vx, vy, vz, wx, wy, wz].  +x push for
            # the slide eastward and +y angular for the tilt direction.
            vel = np.array(
                [_KNOCK_PUSH_VEL, 0.0, 0.0, 0.0, +omega, 0.0],
                dtype=np.float64,
            )
            self.oil_entity.set_dofs_velocity(vel)
        except Exception:
            pass

    def _wait_for_knock_trigger(self) -> bool:
        """Step the sim until the right-hand passes within
        ``_KNOCK_TRIGGER_DIST_M`` of the bottle xy, or the timeout
        expires.  Returns True on trigger, False on timeout."""
        for i in range(_KNOCK_TRIGGER_TIMEOUT_STEPS):
            self.step_sim()
            if i % 10 != 0:
                continue
            hand_xy = self._right_hand_xy()
            if hand_xy is None:
                continue
            bottle = self._get_entity_pose(
                self.oil_entity, Pose(np.zeros(3), _upright_q()),
            )
            d = float(np.linalg.norm(hand_xy - np.asarray(bottle.p[:2])))
            if d < _KNOCK_TRIGGER_DIST_M:
                return True
        return False

    # ------------------------------------------------------------------
    # Eval/testbed mode: policy eval runs a pre-policy warmup by default so
    # the bottle is already knocked over for the first policy observation.
    # If that hook is disabled/unavailable, take_action keeps the older
    # delayed randomized knock-over as a fallback.
    # ------------------------------------------------------------------

    def _eval_pre_policy_enabled(self) -> bool:
        return bool(self.config.get("eval_avatar_pre_policy", self.EVAL_AVATAR_PRE_POLICY))

    def _eval_knock_settled(self) -> bool:
        """The bottle was knocked over and has had the scripted settle time."""
        knock = getattr(self, "_eval_knock_sim_step", None)
        if not getattr(self, "_eval_knock_done", False) or knock is None:
            return False
        settle = int(self.config.get("eval_knock_settle_steps", _KNOCK_SETTLE_STEPS))
        return int(getattr(self, "_interaction_sim_step", 0)) - int(knock) >= settle

    def _await_cue(self, label, ready):
        return await_cue(self, ready, label)

    def _init_eval_recovery_state(self):
        self._attach_items_to_hands()
        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        lo = int(self.config.get("eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
        hi = int(self.config.get("eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
        if hi < lo:
            hi = lo
        # Private per-seed stream: the draw never shifts the global RNG.
        rng = np.random.default_rng(
            [int(getattr(self, "_episode_seed", 0)), self._EVAL_RNG_SALT])
        self._eval_trigger_step = int(rng.integers(lo, hi + 1))
        self._full_eval_cue_waits = None
        self._eval_frying_started = False
        self._eval_knock_done = False
        self._eval_knock_wait_steps = 0
        self._eval_policy_step_count = 0
        # Diagnostics only (blind-intent metrics); no behaviour reads these.
        self._eval_human_started_sim_step = None
        self._eval_knock_sim_step = None
        self._blind_robot_interaction_sim_step = None
        self._blind_bottle_state_at_robot_step = None

    def eval_pre_policy_warmup(self):
        """Match the scripted data-collection start state for policy eval.

        ``play_once`` knocks the oil bottle over and lets it settle before the
        robot begins recovery.  Policy eval bypasses ``play_once``, so do the
        same physics setup here before the first policy observation.
        """
        if (
            self.avatar is None
            or not bool(self.config.get("eval_mode", False))
            or not self._eval_pre_policy_enabled()
        ):
            return self.get_obs()

        if not getattr(self, "_eval_frying_started", False):
            self.avatar.play_animation("frying", return_to_idle_after=30)
            self._eval_frying_started = True
            self._eval_human_started_sim_step = int(
                getattr(self, "_interaction_sim_step", 0))

        if not getattr(self, "_eval_knock_done", False):
            self._wait_for_knock_trigger()
            self._knock_over_bottle()
            self._eval_knock_done = True
            self._eval_knock_wait_steps = 0
            self._eval_knock_sim_step = int(
                getattr(self, "_interaction_sim_step", 0))

        for _ in range(int(self.config.get(
            "eval_knock_settle_steps", _KNOCK_SETTLE_STEPS,
        ))):
            self.step_sim()

        # Keep this counter policy-relative; warmup should not consume eval
        # policy steps or shift any downstream policy-step diagnostics.
        self._eval_policy_step_count = 0
        return self.get_obs()

    def _eval_step_avatar(self):
        self._eval_policy_step_count += 1
        if (not self._eval_frying_started and
                self._eval_policy_step_count >= self._eval_trigger_step):
            self.avatar.play_animation("frying", return_to_idle_after=30)
            self._eval_frying_started = True
            self._eval_human_started_sim_step = int(
                getattr(self, "_interaction_sim_step", 0))

        if not self._eval_frying_started or self._eval_knock_done:
            return

        self._eval_knock_wait_steps += 1
        hand_xy = self._right_hand_xy()
        bottle = self._get_entity_pose(
            self.oil_entity, Pose(np.zeros(3), _upright_q()),
        )
        d = float("inf") if hand_xy is None else float(
            np.linalg.norm(hand_xy - np.asarray(bottle.p[:2]))
        )
        timeout = self._eval_knock_wait_steps >= int(
            self.config.get("eval_knock_timeout_steps", self.EVAL_KNOCK_TIMEOUT_STEPS)
        )
        if d < _KNOCK_TRIGGER_DIST_M or timeout:
            self._knock_over_bottle()
            self._eval_knock_done = True
            self._eval_knock_sim_step = int(
                getattr(self, "_interaction_sim_step", 0))

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------

    # The knock leaves the bottle at the SAME pose every episode (physics
    # is deterministic), so the per-seed outcome is otherwise mplib
    # process-RNG branch luck at the pre-grasp RRT.  Staging to one
    # calibrated known-good pre-grasp joint configuration (collision-checked
    # plan_qpos, PD-executed) makes the branch deterministic.  Calibrated
    # from a successful run via the pre-arrival print below; of the two
    # branch families observed — j5 ~ -2.05 succeeds, j5 ~ +2.5 fails —
    # this pins the good one.  None disables staging.
    _REFERENCE_PRE_QPOS = (
        1.133545, 0.3879068, 0.003493995, -2.715581,
        -2.073593, 3.554543, -0.6910609,
    )

    def _plan_one_grasp(self, *args, **kwargs):
        plans = super()._plan_one_grasp(*args, **kwargs)
        if plans is not None:
            result_pre = plans[0]
            if getattr(result_pre, "position", np.zeros((0,))).size > 0:
                q = np.asarray(result_pre.position[-1], dtype=np.float64)
                print("[obr] pre-arrival qpos="
                      + np.array2string(q[:7], precision=6, separator=", "),
                      flush=True)
        return plans

    def _select_grasp_once(self, arm_tag: str, fallen_pose: Pose):
        """One full grasp-chain selection (plans + executes on success)."""
        return self.select_and_execute_grasp(
            object_name=_RECOVERY_OIL_ID,
            object_pose=fallen_pose,
            arm_tag=arm_tag,
            model_id=_RECOVERY_OIL_MODEL_ID,
            object_scale=self._oil_scale,
            categories=["pour_side_primary"],
            max_candidates=15,
            # No post_plan_check: a lift-feasibility probe ANTI-correlates
            # with runtime success (strict or advisory, it dropped the
            # success rate from ~11/20 to ~6/20 by steering execution onto
            # later, worse branches).  The runtime lift chain
            # (screw -> IK hop -> two-stage -> capped RRT -> half height ->
            # hover-direct) handles lift trouble better than pre-selection.
            # 0.04: longer axis-aligned approaches (0.10, 0.06) collapse the
            # pre-pose RRT/screw planning on this cluttered counter.  The
            # short approach is a deliberate plannability trade-off.
            pre_dist=0.04,
            # ~1/4 of episodes otherwise die with "screw failed" on
            # the SINGLE feasible candidate — a pure planning failure caused
            # by the pre-RRT's random arrival branch.  Retrying the plan
            # (nothing executed yet) re-rolls that branch.
            plan_attempts=6,
        )

    def _stage_to_reference_pre(self, arm_tag: str) -> bool:
        """Joint-plan to the calibrated pre-grasp branch before grasping."""
        ref = self._REFERENCE_PRE_QPOS
        if ref is None:
            return False
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        if not hasattr(planner, "_mplib"):
            return False
        self._refresh_planner_obstacles(arm_tag)
        try:
            result = planner._mplib.plan_qpos(
                goal_qposes=[planner._pad_qpos(
                    np.asarray(ref, dtype=np.float64),
                )],
                current_qpos=planner._pad_qpos(
                    np.asarray(arm.get_arm_qpos(), dtype=np.float64),
                ),
                time_step=1 / 250,
                planning_time=10,
                rrt_range=0.3,
                verbose=False,
            )
        except Exception:
            return False
        if not (result and result.get("status") == "Success"):
            print("[obr] staging plan_qpos failed; falling through to the "
                  "unstaged grasp", flush=True)
            return False
        from ..planning.base import PlanResult
        pos = result["position"]
        vel = result.get("velocity", np.zeros_like(pos))
        print(f"[obr] staging to calibrated pre-grasp branch "
              f"({pos.shape[0]} steps)", flush=True)
        self.execute_plan(PlanResult(True, pos, vel), arm_tag)
        return True

    def _start_frying_animation(self):
        """Start the clip with a clock that honors later grasp/wait rate changes."""
        from ..task_bases.oil_recovery import OilFryingMotionClock

        self.avatar.frame_ratio = _AVATAR_FRAME_RATIO
        self.avatar.play_animation("frying", return_to_idle_after=30)
        motion = self.avatar.motion_modules["frying"]
        self.avatar.motion_modules["frying"] = OilFryingMotionClock(
            motion, self.avatar, _AVATAR_FRAME_RATIO,
        )

    def play_once(self) -> bool:
        if self.avatar is None:
            with self.suppress_recording():  # pre-task settle, excluded from recordings
                for _ in range(120):
                    self.step_sim()
            return False

        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)

        # --- Phase 1: ease into frying-start pose, snap pan/fork. ---
        self._attach_items_to_hands()

        # Brief settle so the bottle (loaded above the counter) lands.
        for _ in range(60):
            self.step_sim()

        # --- Phase 2: start the long shake-loop, wait for the trigger,
        # then knock the bottle over. ---
        self._start_frying_animation()

        self._wait_for_knock_trigger()
        self._knock_over_bottle()

        for _ in range(_KNOCK_SETTLE_STEPS):
            self.step_sim()

        # No post-knock set_quat / set_pos.  The eastward angular kick
        # (about world +y) carries the bottle past angle-of-repose; pure
        # physics then drops the long axis to world +x and the cylinder
        # rolls into a stable resting pose.  A `set_quat` snap would
        # produce a visible discontinuity in the video; it is only
        # needed with a knock about -x (toward +y), which
        # leaves the bottle in a metastable cap-prop equilibrium at 22°.

        return self._recover_fallen_bottle(arm_tag, arm)

    def _plan_clearance_lift_rounds(self, arm, arm_tag, saved_frame_ratio):
        """Checked clearance-lift plan with wait-and-retry rounds.

        Same rounds as the non-clearance lift: the frying avatar's arm over
        the lift corridor can leave the single checked screw with no solution
        (mplib "unsuccessful native plan").  Nothing has
        executed yet, so wait 300 steps at NORMAL avatar speed with the bottle
        held statically and re-plan; ``plan_oil_clearance_lift`` refreshes the
        obstacle cloud and re-measures the egress each round and keeps all of
        its path/collision checks.  A latched failure (plan_success False or
        avatar contact) is not retried.  Returns ``(lift_res, lift_link_pose)``
        or None after marking the rejection.
        """
        from ..task_bases.oil_recovery import plan_oil_clearance_lift

        last_exc = None
        rounds = int(_OIL_CLEARANCE_LIFT_ROUNDS)
        for lift_round in range(rounds):
            if lift_round > 0:
                print(f"[oil-clearance-lift] corridor blocked; waiting for the "
                      f"avatar arm to clear (round {lift_round + 1}/{rounds})", flush=True)
                self.avatar.frame_ratio = saved_frame_ratio
                for _ in range(_OIL_CLEARANCE_LIFT_WAIT_STEPS):
                    self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                    self.step_sim()
                self.avatar.frame_ratio = saved_frame_ratio * 4
            try:
                planned = plan_oil_clearance_lift(self, arm, arm_tag, _PRE_UPRIGHT_LIFT_M)
            except Exception as exc:
                last_exc = exc
                print(f"[oil-clearance-lift] round {lift_round + 1} rejected: {exc}",
                      flush=True)
                if not self.plan_success or getattr(self, "avatar_collided", False):
                    break  # latched failure: waiting cannot help
                continue
            self._oil_clearance_lift_diagnostics["rounds_used"] = lift_round + 1
            return planned
        # On some episodes the 12 cm egress lift
        # is infeasible at every screw step size in all rounds (the side
        # grasp's joint 6 runs out of range), while the same egress at a
        # lower height plans.  Nothing has executed yet: try lower lifts once.
        if self.plan_success and not getattr(self, "avatar_collided", False):
            for height in _OIL_CLEARANCE_LIFT_FALLBACK_M:
                try:
                    planned = plan_oil_clearance_lift(self, arm, arm_tag, height)
                except Exception as exc:
                    last_exc = exc
                    print(f"[oil-clearance-lift] {height:.2f} m lift rejected: {exc}",
                          flush=True)
                    if not self.plan_success or getattr(self, "avatar_collided", False):
                        break
                    continue
                print(f"[oil-clearance-lift] lift at reduced height {height:.2f} m",
                      flush=True)
                self._oil_clearance_lift_diagnostics["rounds_used"] = rounds
                self._oil_clearance_lift_diagnostics["reduced_height_m"] = height
                return planned
        print(f"[oil-clearance-lift] rejected: {last_exc}", flush=True)
        self._oil_clearance_lift_diagnostics = dict(
            getattr(self, "_oil_clearance_lift_diagnostics", {}),
            stage="rejected", reason=str(last_exc), rounds_used=lift_round + 1,
        )
        self.plan_success = False
        self.avatar.frame_ratio = saved_frame_ratio
        return None

    def _recover_fallen_bottle(self, arm_tag: str, arm) -> bool:
        """play_once Phases 3-6: grasp the fallen bottle, right it, place it
        upright in the safe zone and retreat.  Shared by the scripted
        rollout and the full-state eval expert."""
        fallen_pose = self._get_entity_pose(
            self.oil_entity, Pose(np.zeros(3), _upright_q()),
        )
        print(f"[obr] fallen bottle pos={np.asarray(fallen_pose.p).round(3).tolist()} "
              f"quat={np.asarray(fallen_pose.q).round(3).tolist()}", flush=True)

        # --- Phase 3: pick the fallen bottle via the shared helper. ---
        # `BaseTask.select_and_execute_grasp` is the repo-standard robot
        # grasp path: curr→pre via RRT, pre→grasp via screw, FK-checked,
        # then executed with PD control. No robot-side attach/pin is used.
        self._refresh_planner_obstacles(arm_tag)
        self.open_gripper(arm_tag)

        # Boost arm-joint PD gains 4× before the pre-grasp move (fixes
        # 15-18° steady-state error on
        # joints 1-3 under gravity load with bent-shoulder configs).
        try:
            saved_arm_kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            saved_arm_kv = to_numpy(arm.entity.get_dofs_kv()).copy()
            new_kp = saved_arm_kp.copy()
            new_kv = saved_arm_kv.copy()
            from ..robot.franka_robot import _get_dof_idx as _dof_idx
            arm_idxs = []
            for j in arm.arm_joints:
                if j is None:
                    continue
                idx = _dof_idx(j)
                if idx is None or idx < 0 or idx >= len(new_kp):
                    continue
                arm_idxs.append(idx)
                new_kp[idx] = 16000.0
                new_kv[idx] = 800.0
            set_dofs_kp_kv_compat(arm.entity, kp=new_kp, kv=new_kv)
        except Exception:
            pass

        # Slow the avatar 4× during the grasp (still playing — user
        # directive — but moves less per sim step so its arm doesn't
        # sweep into the robot's planned path during execution).
        saved_frame_ratio = self.avatar.frame_ratio
        self.avatar.frame_ratio = saved_frame_ratio * 4

        # The pre pose can be UNREACHABLE for a whole planning
        # window — "IK Failed" on every attempt — because the frying
        # avatar's arm capsules sit over the bottle in the obstacle cloud
        # (retrying back-to-back hits
        # the SAME frozen avatar pose).  Let sim time
        # pass so the 4x-slowed frying motion clears, refresh the pcd,
        # re-stage, and retry.  Nothing has executed until a full grasp
        # chain plans, so the retries stay invisible.
        grasp_choice = None
        for outer in range(4):
            if outer > 0:
                print(f"[obr] grasp window blocked; waiting for the avatar "
                      f"arm to clear (outer retry {outer + 1}/4)", flush=True)
                # Wait at NORMAL avatar speed — the grasp-phase 4x slowdown
                # makes a slowed wait nearly useless for clearing.
                self.avatar.frame_ratio = saved_frame_ratio
                for _ in range(250):
                    self.step_sim()
                self.avatar.frame_ratio = saved_frame_ratio * 4
                self._refresh_planner_obstacles(arm_tag)
            self._stage_to_reference_pre(arm_tag)
            grasp_choice = self._select_grasp_once(arm_tag, fallen_pose)
            if grasp_choice is not None:
                break
        if grasp_choice is None:
            return False
        grasp_link, _pre_link, _chosen_g = grasp_choice

        desired_tcp = grasp_link * arm.tcp_offset
        actual_tcp = Pose.from_pose7(arm.get_ee_pose())
        tcp_err = float(np.linalg.norm(actual_tcp.p - desired_tcp.p))
        if tcp_err > 0.015:
            self._refresh_planner_obstacles(arm_tag)
            refine = arm.planner.plan_screw_path(
                arm.get_arm_qpos(), grasp_link.to_pose7(),
            )
            if not refine.success:
                return False
            self.execute_plan(refine, arm_tag)
            for _ in range(30):
                self.step_sim()
            actual_tcp = Pose.from_pose7(arm.get_ee_pose())
            tcp_err = float(np.linalg.norm(actual_tcp.p - desired_tcp.p))

        # Soft, partial close.  Do this before the finger PD change so the
        # visible close is not a high-force full crush.
        self.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag, num_steps=120)
        for _ in range(30):
            self.step_sim()

        # Moderate per-DOF gripper PD: enough to hold the side grasp, but
        # far below a 50k/1k full-crush setting.
        update_dofs_kp_kv_compat(
            arm.entity,
            arm._finger_dof_indices,
            kp_value=_OIL_FINGER_KP,
            kv_value=_OIL_FINGER_KV,
        )
        self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
        for _ in range(30):
            self.step_sim()

        # Clear the table before commanding the large rotation that rights the
        # bottle.  A single-shot reorientation can sweep the bottle
        # through the counter, so physics contact blocks the arm far from the
        # planned endpoint even though mplib found a valid robot-only path.
        clearance_lift = self.config.get("oil_clearance_lift", False) is True
        if clearance_lift:
            planned = self._plan_clearance_lift_rounds(arm, arm_tag, saved_frame_ratio)
            if planned is None:
                return False
            lift_res, lift_link_pose = planned
        else:
            lift_tcp0 = Pose.from_pose7(arm.get_ee_pose())
            lift_tcp_pose = Pose(
                lift_tcp0.p + np.array([0.0, 0.0, _PRE_UPRIGHT_LIFT_M]),
                lift_tcp0.q,
            )
            lift_link_pose = lift_tcp_pose * arm.tcp_offset.inv()
            self._refresh_planner_obstacles(arm_tag)
            # The lift can be blocked the same way the grasp window
            # is — the frying avatar's arm over the lift corridor makes even
            # collision-checked IK return "no solution" (a staged
            # grasp on the calibrated branch can get zero lift IK solutions and
            # only 2000+-step RRT unwinds).  Wait at NORMAL avatar speed with
            # the bottle held statically, refresh the pcd, and retry the
            # screw + gated-hop rungs before degrading further.
            lift_res = None
            for lift_round in range(3):
                if lift_round > 0:
                    print(f"[obr] lift corridor blocked; waiting for the avatar "
                          f"arm to clear (round {lift_round + 1}/3)", flush=True)
                    self.avatar.frame_ratio = saved_frame_ratio
                    for _ in range(300):
                        self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                        self.step_sim()
                    self.avatar.frame_ratio = saved_frame_ratio * 4
                    self._refresh_planner_obstacles(arm_tag)
                lift_res = arm.planner.plan_screw_path(
                    arm.get_arm_qpos(), lift_link_pose.to_pose7(),
                )
                if lift_res.success:
                    break
                # The screw's failure means goal-IK seeded from the
                # grasp branch failed — but mplib's multi-seed closest-solution
                # IK often still finds a nearby branch.  A shortest-path joint
                # interpolation over 12 cm straight up is safe IF the joint
                # delta is small; gate it, else fall through to the step-capped
                # RRT.  (An uncapped RRT fallback executes 1600+-step
                # unwinds holding the bottle and drops it.)
                ik_res = arm.planner.solve_ik(
                    arm.get_arm_qpos(), lift_link_pose.to_pose7(), log=False,
                )
                if (getattr(ik_res, "success", False)
                        and getattr(ik_res, "position", np.zeros((0,))).shape[0] > 0):
                    start_q = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
                    end_q = np.asarray(ik_res.position[-1], dtype=np.float64)
                    n = min(len(start_q), len(end_q))
                    dq = float(np.abs(end_q[:n] - start_q[:n]).max())
                    if dq < 1.0:
                        print("[obr] lift via gated IK hop", flush=True)
                        lift_res = ik_res
                        break
                    print(f"[obr] lift IK hop refused (dq={dq:.2f} rad)",
                          flush=True)
                else:
                    print("[obr] lift IK hop: no IK solution", flush=True)
            if not lift_res.success:
                # Two-stage screw — the full 12 cm goal's IK can fail
                # from the grasp branch while the 6 cm midpoint succeeds; after
                # executing the half lift the arm is in a fresh branch and the
                # remainder usually screws cleanly.
                half_link = Pose(
                    lift_tcp0.p + np.array([0.0, 0.0, _PRE_UPRIGHT_LIFT_M / 2]),
                    lift_tcp0.q,
                ) * arm.tcp_offset.inv()
                half_res = arm.planner.plan_screw_path(
                    arm.get_arm_qpos(), half_link.to_pose7(),
                )
                if half_res.success:
                    print("[obr] lift via two-stage screw", flush=True)
                    self.execute_plan(half_res, arm_tag)
                    for _ in range(30):
                        self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                        self.step_sim()
                    lift_res = arm.planner.plan_screw_path(
                        arm.get_arm_qpos(), lift_link_pose.to_pose7(),
                    )
            if not lift_res.success:
                lift_res, _ = self._plan_path_with_fk_check(
                    lift_link_pose, arm_tag, arm, fk_tol=0.008,
                    max_plan_steps=600,
                )
            if lift_res is None or not lift_res.success:
                # Reduced-height lift.  On some deterministic failing
                # seeds the FULL 12 cm lift pose has no IK solution at all
                # (side-grasp wrist configuration runs out of joint range /
                # clashes with the cooktop pcd higher up); half the clearance
                # is often still solvable and beats no lift.
                short_link = Pose(
                    lift_tcp0.p + np.array([0.0, 0.0, _PRE_UPRIGHT_LIFT_M / 2]),
                    lift_tcp0.q,
                ) * arm.tcp_offset.inv()
                lift_res, _ = self._plan_path_with_fk_check(
                    short_link, arm_tag, arm, fk_tol=0.010,
                    max_plan_steps=600, max_snap_retries=3,
                )
                if lift_res is not None and lift_res.success:
                    print("[obr] lift at reduced height (6 cm)", flush=True)
            if lift_res is None or not lift_res.success:
                # Last resort: skip the lift and let the hover reorientation
                # plan directly from the grasp configuration.  Riskier (the
                # reorientation can sweep the bottle near the counter) but a
                # certain improvement over returning False untouched.
                print("[obr] lift unplannable at every height; proceeding to "
                      "hover directly", flush=True)
                lift_res = None
        if lift_res is not None:
            self.execute_plan(lift_res, arm_tag)
            if lift_res.position.size > 0:
                final_q = lift_res.position[-1]
                for _ in range(180):
                    self.robot.set_arm_joints(final_q, arm_tag)
                    self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                    self.step_sim()

        if clearance_lift:
            from ..task_bases.oil_recovery import pose_error

            error = pose_error(arm.get_ee_pose(), (lift_link_pose * arm.tcp_offset).to_pose7())
            self._oil_clearance_lift_diagnostics.update(stage="held", arrival_error=error)
            if (getattr(self, "avatar_collided", False) or not self.plan_success
                    or error["position_m"] > .005 or error["orientation_rad"] > .01):
                print("[oil-clearance-lift] rejected physical arrival/contact", flush=True)
                self._oil_clearance_lift_diagnostics.update(
                    stage="rejected_after_hold", reason="physical arrival/contact",
                )
                self.plan_success = False
                self.avatar.frame_ratio = saved_frame_ratio
                return False

        # Diagnostics: dump TCP pose + bottle pose at cache time so we can
        # debug bottle-frame target divergence (wildly off
        # link targets have been seen; _bottle_to_link is not trusted here).
        ee_pose = Pose.from_pose7(arm.get_ee_pose())
        bottle_now = self._get_entity_pose(self.oil_entity, fallen_pose)
        self._T_ee_bottle = ee_pose.inv() * bottle_now

        # Experimental task-local carry only; acquisition/lift/defaults unchanged.
        if self.config.get("oil_rotated_placement", False):
            from ..task_bases.oil_recovery import place_rotated_oil
            return place_rotated_oil(self, arm_tag, saved_frame_ratio)

        # --- Phase 4: carry to the middle counter and place upright. ---

        def _link_pose_at_tcp(tcp_p, tcp_q) -> Pose:
            return (Pose(np.asarray(tcp_p, dtype=np.float64),
                         np.asarray(tcp_q, dtype=np.float64))
                    * arm.tcp_offset.inv())

        def _link_at_tcp(tcp_p, tcp_q):
            return _link_pose_at_tcp(tcp_p, tcp_q).to_pose7()

        # Compute the TCP pose that would put the bottle's own frame upright
        # at the middle-counter target.  This is pure motion planning; the
        # bottle remains held by contact, not attached to the gripper.
        upright_bottle_hover = Pose(
            np.array([
                self._safe_target_xy[0],
                self._safe_target_xy[1],
                self._oil_upright_origin_z_on_table + _UPRIGHT_PLACE_HOVER_M,
            ], dtype=np.float64),
            _upright_q(),
        )
        hover_tcp = upright_bottle_hover * self._T_ee_bottle.inv()
        hover_link_pose = hover_tcp * arm.tcp_offset.inv()
        self._refresh_planner_obstacles(arm_tag)
        hover_res, hover_err = self._plan_path_with_fk_check(
            hover_link_pose, arm_tag, arm, fk_tol=0.005,
        )
        if hover_res is None or hover_err > 0.05:
            return False
        self.execute_plan(hover_res, arm_tag)
        if hover_res.position.size > 0:
            final_q = hover_res.position[-1]
            for _ in range(_UPRIGHT_ENDPOINT_HOLD_STEPS):
                self.robot.set_arm_joints(final_q, arm_tag)
                self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                self.step_sim()
        self._correct_held_bottle_at_hover(arm_tag, arm, fallen_pose, upright_bottle_hover)

        upright_bottle_place = Pose(
            np.array([
                self._safe_target_xy[0],
                self._safe_target_xy[1],
                self._oil_upright_origin_z_on_table,
            ], dtype=np.float64),
            _upright_q(),
        )
        place_tcp = upright_bottle_place * self._T_ee_bottle.inv()
        place_link_pose = place_tcp * arm.tcp_offset.inv()
        self._refresh_planner_obstacles(arm_tag)
        place_res = arm.planner.plan_screw_path(
            arm.get_arm_qpos(), place_link_pose.to_pose7(),
        )
        if not place_res.success:
            # Short descent while holding — same step cap as the lift so a
            # branch-flip unwind can never execute with the bottle gripped.
            place_res, place_err = self._plan_path_with_fk_check(
                place_link_pose, arm_tag, arm, fk_tol=0.006,
                max_plan_steps=600,
            )
            if place_res is None or place_err > 0.05:
                return False
        self.execute_plan(place_res, arm_tag)
        if place_res.position.size > 0:
            final_q = place_res.position[-1]
            for _ in range(_UPRIGHT_ENDPOINT_HOLD_STEPS):
                self.robot.set_arm_joints(final_q, arm_tag)
                self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                self.step_sim()

        self.open_gripper(arm_tag)
        for _ in range(30):
            self.step_sim()
        for _ in range(120):
            self.step_sim()

        # Retreat: small +z lift to clear the released bottle.
        ee_after = Pose.from_pose7(arm.get_ee_pose())
        retreat_tcp_p = ee_after.p + np.array([0.0, 0.0, 0.10])
        retreat_link = _link_at_tcp(retreat_tcp_p, ee_after.q)
        self._refresh_planner_obstacles(arm_tag)
        self.move_and_execute(retreat_link, arm_tag)

        # Restore avatar frame_ratio after the grasp phase.
        self.avatar.frame_ratio = saved_frame_ratio

        # --- Phase 6: short tail after recovery.  Do not drain the full
        # frying loop; the result is visible once the robot retreats.
        for _ in range(180):
            if self.avatar.spare():
                break
            self.step_sim()

        return True

    # The long righting carry to the hover can let the bottle
    # turn in the side grip (a carry that swings wide can end the
    # hover with the bottle 10-40 deg off upright and a stale grasp-time
    # T_ee_bottle; the place then sets it down tilted/high and it topples at
    # release).  After the hover hold, re-measure the held bottle; if it is
    # off upright, re-plan the hover once with the measured in-hand transform
    # (short capped plan), and place with the measured transform either way.
    _HOVER_RETILT_DEG = 6.0
    _HOVER_HELD_MAX_DIST_M = 0.12

    def _correct_held_bottle_at_hover(self, arm_tag, arm, fallen_pose, upright_hover):
        def measure():
            ee = Pose.from_pose7(arm.get_ee_pose())
            bottle = self._get_entity_pose(self.oil_entity, fallen_pose)
            return ee, bottle

        ee, bottle = measure()
        if float(np.linalg.norm(np.asarray(ee.p) - np.asarray(bottle.p))) > self._HOVER_HELD_MAX_DIST_M:
            return  # not held any more: nothing to correct
        tilt = float(self._bottle_state()["tilt_deg"])
        if tilt > self._HOVER_RETILT_DEG:
            T_now = ee.inv() * bottle
            link = (upright_hover * T_now.inv()) * arm.tcp_offset.inv()
            self._refresh_planner_obstacles(arm_tag)
            plan_ok = bool(self.plan_success)
            res, err = self._plan_path_with_fk_check(
                link, arm_tag, arm, fk_tol=0.005, max_plan_steps=600,
                max_snap_retries=2,
            )
            if res is None and not getattr(self, "avatar_collided", False):
                # Nothing executed: a refused correction plan is not a failure.
                self.plan_success = plan_ok
            if res is not None and err <= 0.05 and res.position.size > 0:
                self.execute_plan(res, arm_tag)
                final_q = res.position[-1]
                for _ in range(_UPRIGHT_ENDPOINT_HOLD_STEPS):
                    self.robot.set_arm_joints(final_q, arm_tag)
                    self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                    self.step_sim()
                print(f"[obr] hover re-tilt: {tilt:.1f} -> "
                      f"{self._bottle_state()['tilt_deg']:.1f} deg", flush=True)
            else:
                print(f"[obr] hover re-tilt skipped ({tilt:.1f} deg): no capped plan",
                      flush=True)
            ee, bottle = measure()
        self._T_ee_bottle = ee.inv() * bottle

    # ------------------------------------------------------------------
    # Blind ("no intent") expert
    # ------------------------------------------------------------------

    def _bottle_state(self) -> dict:
        """Observed bottle tilt (deg from upright) and linear speed."""
        pose = self._get_entity_pose(
            self.oil_entity, Pose(np.zeros(3), _upright_q()),
        )
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=np.float64))
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 1], -1.0, 1.0))))
        try:
            speed = float(np.linalg.norm(
                np.asarray(to_numpy(self.oil_entity.get_vel()), dtype=np.float64).ravel()[:3]
            ))
        except Exception:
            speed = float("nan")
        return {"pos": np.asarray(pose.p, dtype=np.float64).round(4).tolist(),
                "tilt_deg": tilt, "speed_m_s": speed}

    def play_full_eval_once(self) -> bool:
        """Full-state expert under the eval-mode human.

        The human (eval hooks) fries and knocks the bottle at its random
        step; the expert waits for the knock and the scripted settle (the
        privileged cue), then runs play_once's recovery unchanged, including
        its staging to the calibrated pre-grasp branch and its
        wait-for-the-arm-to-clear retries.
        """
        if self.avatar is None:
            return False
        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)
        self._full_eval_active = True
        try:
            # Same non-human settle as play_once: the bottle lands.
            for _ in range(60):
                self.step_sim()
            self._await_cue("knock_settled", self._eval_knock_settled)
            return self._recover_fallen_bottle(arm_tag, arm)
        finally:
            self._full_eval_active = False

    def play_blind_once(self) -> bool:
        """Blind expert: ``play_once``'s robot motions on the robot's own
        clock, never waiting for or reading the human's trigger or knock.

        The human (eval hooks) starts frying at a random step of the eval
        trigger window and knocks the bottle when its hand comes close.  The
        robot grasps the bottle at the pose observed at grasp-plan time
        (upright, falling or fallen), with single-snapshot plans and no
        human-dependent waits.  A failed step returns False.
        """
        if self.avatar is None:
            return self.play_once()

        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)

        # Non-human setup: the bottle (loaded above the counter) lands.
        for _ in range(60):
            self.step_sim()
        # Full-state eval expert only (no-op for the blind expert): wait for
        # the knock and the scripted settle.
        self._await_cue("knock_settled", self._eval_knock_settled)

        self._refresh_planner_obstacles(arm_tag)
        self.open_gripper(arm_tag)

        # Same 4x arm PD boost as play_once (robot-side only).
        try:
            new_kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            new_kv = to_numpy(arm.entity.get_dofs_kv()).copy()
            from ..robot.franka_robot import _get_dof_idx as _dof_idx
            for j in arm.arm_joints:
                if j is None:
                    continue
                idx = _dof_idx(j)
                if idx is None or idx < 0 or idx >= len(new_kp):
                    continue
                new_kp[idx] = 16000.0
                new_kv[idx] = 800.0
            set_dofs_kp_kv_compat(arm.entity, kp=new_kp, kv=new_kv)
        except Exception:
            pass

        # One grasp selection, without play_once's staging to the reference
        # pre-grasp qpos: that target was fit on the post-knock bottle pose
        # (it encodes the human's outcome) and sweeps the arm into an upright
        # bottle.
        self._refresh_planner_obstacles(arm_tag)
        observed = self._get_entity_pose(
            self.oil_entity, Pose(np.zeros(3), _upright_q()),
        )
        self._blind_robot_interaction_sim_step = int(
            getattr(self, "_interaction_sim_step", 0))
        self._blind_bottle_state_at_robot_step = self._bottle_state()
        print(f"[obr-blind] grasp plan at sim step "
              f"{self._blind_robot_interaction_sim_step}: bottle "
              f"{self._blind_bottle_state_at_robot_step} knock_done="
              f"{bool(getattr(self, '_eval_knock_done', False))}", flush=True)
        grasp_choice = self._select_grasp_once(arm_tag, observed)
        if grasp_choice is None:
            return False
        grasp_link, _pre_link, _chosen_g = grasp_choice

        desired_tcp = grasp_link * arm.tcp_offset
        actual_tcp = Pose.from_pose7(arm.get_ee_pose())
        if float(np.linalg.norm(actual_tcp.p - desired_tcp.p)) > 0.015:
            self._refresh_planner_obstacles(arm_tag)
            refine = arm.planner.plan_screw_path(
                arm.get_arm_qpos(), grasp_link.to_pose7(),
            )
            if not refine.success:
                return False
            self.execute_plan(refine, arm_tag)
            for _ in range(30):
                self.step_sim()

        self.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag, num_steps=120)
        for _ in range(30):
            self.step_sim()
        update_dofs_kp_kv_compat(
            arm.entity,
            arm._finger_dof_indices,
            kp_value=_OIL_FINGER_KP,
            kv_value=_OIL_FINGER_KV,
        )
        self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
        for _ in range(30):
            self.step_sim()

        # Lift: the same single checked clearance screw as play_once (the
        # live forearm capsule sets the egress direction at plan time).
        clearance_lift = self.config.get("oil_clearance_lift", False) is True
        if clearance_lift:
            from ..task_bases.oil_recovery import plan_oil_clearance_lift

            try:
                lift_res, lift_link_pose = plan_oil_clearance_lift(
                    self, arm, arm_tag, _PRE_UPRIGHT_LIFT_M,
                )
            except Exception as exc:
                print(f"[oil-clearance-lift] rejected: {exc}", flush=True)
                self._oil_clearance_lift_diagnostics = dict(
                    getattr(self, "_oil_clearance_lift_diagnostics", {}),
                    stage="rejected", reason=str(exc),
                )
                self.plan_success = False
                return False
        else:
            lift_tcp0 = Pose.from_pose7(arm.get_ee_pose())
            lift_link_pose = Pose(
                lift_tcp0.p + np.array([0.0, 0.0, _PRE_UPRIGHT_LIFT_M]),
                lift_tcp0.q,
            ) * arm.tcp_offset.inv()
            self._refresh_planner_obstacles(arm_tag)
            lift_res = arm.planner.plan_screw_path(
                arm.get_arm_qpos(), lift_link_pose.to_pose7(),
            )
        if lift_res is None or not lift_res.success:
            return False
        self.execute_plan(lift_res, arm_tag)
        if lift_res.position.size > 0:
            final_q = lift_res.position[-1]
            for _ in range(180):
                self.robot.set_arm_joints(final_q, arm_tag)
                self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                self.step_sim()
        if clearance_lift:
            from ..task_bases.oil_recovery import pose_error

            error = pose_error(arm.get_ee_pose(), (lift_link_pose * arm.tcp_offset).to_pose7())
            self._oil_clearance_lift_diagnostics.update(stage="held", arrival_error=error)
            if (getattr(self, "avatar_collided", False) or not self.plan_success
                    or error["position_m"] > .005 or error["orientation_rad"] > .01):
                print("[oil-clearance-lift] rejected physical arrival/contact", flush=True)
                self._oil_clearance_lift_diagnostics.update(
                    stage="rejected_after_hold", reason="physical arrival/contact",
                )
                self.plan_success = False
                return False

        ee_pose = Pose.from_pose7(arm.get_ee_pose())
        bottle_now = self._get_entity_pose(self.oil_entity, observed)
        self._T_ee_bottle = ee_pose.inv() * bottle_now

        # Carry to the middle counter and place upright (same as play_once,
        # but one planning snapshot per leg — no sim-time re-plan waits).
        upright_bottle_hover = Pose(
            np.array([
                self._safe_target_xy[0],
                self._safe_target_xy[1],
                self._oil_upright_origin_z_on_table + _UPRIGHT_PLACE_HOVER_M,
            ], dtype=np.float64),
            _upright_q(),
        )
        hover_link_pose = (upright_bottle_hover * self._T_ee_bottle.inv()
                           * arm.tcp_offset.inv())
        self._refresh_planner_obstacles(arm_tag)
        hover_res, hover_err = self._plan_path_with_fk_check(
            hover_link_pose, arm_tag, arm, fk_tol=0.005,
            max_snap_retries=1, snap_steps=0,
        )
        if hover_res is None or hover_err > 0.05:
            return False
        self.execute_plan(hover_res, arm_tag)
        if hover_res.position.size > 0:
            final_q = hover_res.position[-1]
            for _ in range(_UPRIGHT_ENDPOINT_HOLD_STEPS):
                self.robot.set_arm_joints(final_q, arm_tag)
                self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                self.step_sim()

        upright_bottle_place = Pose(
            np.array([
                self._safe_target_xy[0],
                self._safe_target_xy[1],
                self._oil_upright_origin_z_on_table,
            ], dtype=np.float64),
            _upright_q(),
        )
        place_link_pose = (upright_bottle_place * self._T_ee_bottle.inv()
                           * arm.tcp_offset.inv())
        self._refresh_planner_obstacles(arm_tag)
        place_res = arm.planner.plan_screw_path(
            arm.get_arm_qpos(), place_link_pose.to_pose7(),
        )
        if not place_res.success:
            place_res, place_err = self._plan_path_with_fk_check(
                place_link_pose, arm_tag, arm, fk_tol=0.006,
                max_plan_steps=600, max_snap_retries=1, snap_steps=0,
            )
            if place_res is None or place_err > 0.05:
                return False
        self.execute_plan(place_res, arm_tag)
        if place_res.position.size > 0:
            final_q = place_res.position[-1]
            for _ in range(_UPRIGHT_ENDPOINT_HOLD_STEPS):
                self.robot.set_arm_joints(final_q, arm_tag)
                self.robot.set_gripper(_OIL_GRIP_CLOSE_TARGET, arm_tag)
                self.step_sim()

        self.open_gripper(arm_tag)
        for _ in range(150):
            self.step_sim()

        ee_after = Pose.from_pose7(arm.get_ee_pose())
        retreat_link = (Pose(ee_after.p + np.array([0.0, 0.0, 0.10]), ee_after.q)
                        * arm.tcp_offset.inv()).to_pose7()
        self._refresh_planner_obstacles(arm_tag)
        self.move_and_execute(retreat_link, arm_tag)

        # Bounded settle (fixed length; does not read the avatar's state).
        for _ in range(180):
            self.step_sim()
        return True

    def _blind_intent_metrics(self) -> dict:
        state = getattr(self, "_blind_bottle_state_at_robot_step", None)
        knock_step = getattr(self, "_eval_knock_sim_step", None)
        robot_step = getattr(self, "_blind_robot_interaction_sim_step", None)
        knocked_before = (knock_step is not None and robot_step is not None
                          and knock_step <= robot_step)
        lying_still = bool(
            state is not None
            and state["tilt_deg"] > _BLIND_READY_TILT_DEG
            and np.isfinite(state["speed_m_s"])
            and state["speed_m_s"] < _BLIND_READY_SPEED_M_S
        )
        return {
            "trigger_step": getattr(self, "_eval_trigger_step", None),
            "human_started_sim_step": getattr(self, "_eval_human_started_sim_step", None),
            "knock_sim_step": knock_step,
            "robot_interaction_sim_step": robot_step,
            "bottle_at_robot_step": state,
            # The full-state expert waits for the knock AND the bottle coming
            # to rest on its side before it plans the grasp.
            "human_ready_before_robot": bool(knocked_before and lying_still),
        }

    def check_success(self) -> bool:
        """Bottle ends upright inside the safe zone with no avatar collisions."""
        m = self._recovery_metrics()
        ok = bool(
            self.plan_success
            and m["target_dist_xy"] < _SUCCESS_DIST_XY_M
            and m["target_tilt_deg"] < _SUCCESS_TILT_DEG
            and m["avatar_ok"]
            and m["target_resting"]
        )
        return ok

    def _oil_resting(self, dz: float) -> bool:
        if abs(float(dz)) > _SUCCESS_REST_DZ_M or self.oil_entity is None:
            return False
        try:
            aabb = np.asarray(to_numpy(self.oil_entity.get_AABB()), dtype=float)
            center = 0.5 * (aabb[0] + aabb[1])
        except Exception:
            center = to_numpy(self.oil_entity.get_pos()).ravel()[:3]
        return not held_by_anyone(
            self, self.oil_entity, center, radius_m=_SUCCESS_HELD_TCP_RADIUS_M,
        )

    def _recovery_metrics(self) -> dict:
        safe_xy = np.asarray(
            getattr(self, "_safe_target_xy", _SAFE_TARGET_XY),
            dtype=np.float64,
        )
        if self.oil_entity is None:
            pose = Pose(np.zeros(3), _upright_q())
        else:
            pose = self._get_entity_pose(
                self.oil_entity, Pose(np.zeros(3), _upright_q()),
            )
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=np.float64))
        bottle_up_world = R[:, 1]
        cos_tilt = float(np.clip(bottle_up_world[2], -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
        dxy = float(np.linalg.norm(np.asarray(pose.p[:2]) - safe_xy))
        dist_3d = float(np.linalg.norm(np.asarray(pose.p) - np.array([
            safe_xy[0],
            safe_xy[1],
            getattr(self, "_oil_upright_origin_z_on_table", self.TABLE_TOP_Z),
        ], dtype=np.float64)))
        dz = float(pose.p[2] - getattr(
            self, "_oil_upright_origin_z_on_table", self.TABLE_TOP_Z,
        ))
        avatar_ok = not getattr(self, "avatar_collided", False)
        return {
            "target_label": "recovered_oil_bottle",
            "target_pos": [
                float(safe_xy[0]),
                float(safe_xy[1]),
                float(getattr(self, "_oil_upright_origin_z_on_table", self.TABLE_TOP_Z)),
            ],
            "target_object_pos": np.asarray(pose.p, dtype=np.float64).tolist(),
            "target_dist_xy": dxy,
            "target_dist_3d": dist_3d,
            "target_dz": dz,
            "target_tilt_deg": tilt_deg,
            "target_dist_xy_threshold": _SUCCESS_DIST_XY_M,
            "target_tilt_deg_threshold": _SUCCESS_TILT_DEG,
            "target_resting": bool(self._oil_resting(dz)),
            "avatar_ok": bool(avatar_ok),
            "eval_mode": bool(self.config.get("eval_mode", False)),
            "eval_trigger_step": getattr(self, "_eval_trigger_step", None),
            "eval_knock_done": bool(getattr(self, "_eval_knock_done", False)),
            "bottle_init_xy": list(getattr(self, "_bottle_init_xy", _BOTTLE_INIT_XY)),
            "safe_target_xy": safe_xy.tolist(),
        }

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._recovery_metrics())
        if bool(self.config.get("eval_mode", False)):
            metrics["blind_intent"] = self._blind_intent_metrics()
            if getattr(self, "_full_eval_cue_waits", None) is not None:
                metrics["full_eval_cue_waits"] = self._full_eval_cue_waits
        return metrics


# ------------------------------------------------------------------
# Quaternion helpers (Pose convention: w, x, y, z)
# ------------------------------------------------------------------

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 in (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)
