"""Put object in cabinet (Contribution, ``put_object_in_cabinet_contribution``;
single Franka + avatar).

RoboTwin's ``put_object_cabinet`` is bimanual: one arm opens a drawer,
the other places an object inside.  This variant maps that split across
actors: the avatar pulls the cabinet's top drawer open and the Franka
picks a table object and drops it into the human-opened drawer.

Pure physics for the robot — no kinematic teleport, no
``attach_to_gripper``.  The avatar uses FABRIK palm IK; the drawer DOF is
advanced in sync with the avatar hand because this is a non-robot
articulation and the Contribution behavior being graded is the robot's object
placement into the opened drawer.

Coordinate notes (cabinet asset 036_cabinet/46653):

  - URDF has a fixed ``joint_0`` with rpy (1.57, 0, -1.57) so URDF-Z
    (the prismatic drawer axis) maps to world -X under an *identity*
    cabinet quat.
  - We rotate the cabinet by Rz(+π/2) so URDF-Z maps to world -Y, i.e.
    the drawers slide *toward* the Franka (which is at world Y < 0,
    facing +Y).  After that:
        cabinet world X span ≈ ±0.43  (left/right of cabinet origin)
        cabinet world Y span ≈ [-0.47, +0.43]  (front-to-back depth)
        cabinet world Z span ≈ [-0.80, +0.81]  (height around origin)
  - We anchor the cabinet so it stands on the floor with the front
    face facing the robot, *behind* the table.

The asset has 3 prismatic drawers (link_1, link_2, link_3) all with
joint origin (0, 0, 0) and the same range [0, 0.66].  They differ only
in mesh translation, so which one is the "top drawer" must be detected
at runtime by reading link world Z.  ``_pick_top_drawer`` does that.
"""

from __future__ import annotations

import numpy as np

from ..genesis_compat import hold_dof_position_compat
from ..interaction.put_object_cabinet import CabinetAssistController
from ..manipulation import PickSpec, PlaceSpec
from ..task_bases.put_object_cabinet import CabinetTaskCore
from ..task_bases.cabinet_ordinary_motion import CabinetOrdinaryMotionMixin
from ..utils import to_numpy

class PutObjectInCabinetContribution(CabinetOrdinaryMotionMixin, CabinetTaskCore):
    """Avatar opens the top drawer while the robot picks a table object
    and drops it inside the open drawer."""

    # Jam jar: the catalog close (.55) lets it slide out on lift/carry; the
    # family's width grip (live jaw-axis width minus 8 mm) holds it.
    ORDINARY_WIDTH_GRIP_EXTRA_OBJECTS = ("031_jam-jar",)

    INSTRUCTION = "place the object inside the human-opened cabinet drawer"
    # Shared tabletop pool minus 038_milk-box, like the interrupt variant
    # (fragile carton side-grasp: the full-state expert fails on it; see
    # object_catalog).
    OBJECT_SET = "cabinet_interrupt_pick_pool"
    # The open-drawer phase is NOT physically driven on this single-arm
    # config — the cabinet handle's IK lands on a Franka-7DOF qpos that
    # mplib + Genesis IK can't solve consistently (repeated attempts
    # left the arm 15-30 cm off target with non-deterministic
    # joint solutions across runs).  Instead the drawer is pre-opened
    # via cabinet.set_qpos at scene init (kinematic control is allowed
    # for non-robot articulations) and the robot's job is to
    # pick the table object and drop it in the open drawer.
    use_avatar = True

    _CABINET_TASK_NAME = "put_object_in_cabinet_contribution"
    _CABINET_ALLOW_TABLE_VARIANTS = False

    # Avatar stands on the cabinet's +X side, facing left toward the
    # handle.  This keeps the body outside the Franka's central
    # object-pick corridor and clear of the drawer drop line.
    AVATAR_SIDE_ROT = np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    avatar_init_pos = np.array([0.82, -0.56, -0.18], dtype=np.float64)
    avatar_init_rot = AVATAR_SIDE_ROT

    AVATAR_HAND_ID = 1
    AVATAR_HANDLE_SIDE_OFFSET = -0.005
    AVATAR_OPEN_APPROACH_FRAMES = 300
    AVATAR_OPEN_TRANSPORT_FRAMES = 500
    AVATAR_OPEN_RETRACT_FRAMES = 300
    AVATAR_RETURN_IDLE_FRAMES = 360
    AVATAR_OPEN_ARC = 0.04
    AVATAR_NATURAL_BODY_MARGIN = 0.36
    AVATAR_NATURAL_YAW_LIMIT_DEG = 25.0
    AVATAR_BODY_X_BOUNDS = (0.70, 0.98)
    AVATAR_BODY_Y_BOUNDS = (-0.78, -0.38)

    # Eval/testbed mode: policy calls take_action(action)->obs, while the
    # avatar starts the assist motion once at a randomized policy step.
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        cfg.setdefault("track_avatar_collision", False)
        super().__init__(cfg)
        self._object_grasp_z_offset = 0.0
        self._spawn_x_range = tuple(float(v) for v in cfg.get("object_spawn_x_range", self._SPAWN_X))
        self._spawn_y_range = tuple(float(v) for v in cfg.get("object_spawn_y_range", self._SPAWN_Y))
        self._avatar_drawer_motion = None
        self._avatar_drawer_sync_active = False
        self._avatar_idle_return_started = False
        self._avatar_body_return_active = False
        self._avatar_body_return_from = None
        self._avatar_body_return_to = None
        self._avatar_body_return_frame = 0
        self._avatar_body_return_frames = 1
        self._eval_trigger_step = None
        self._eval_avatar_started = False
        self._eval_avatar_failed = False
        self._policy_step_count = 0

    def create_interaction_controller(self):
        return CabinetAssistController()

    def _assist_controller(self) -> CabinetAssistController:
        controller = self.interaction_controller
        if not isinstance(controller, CabinetAssistController):
            raise TypeError("task does not use cabinet assist behavior")
        return controller

    # Reset ordering (drawer pick, eval-state arming, avatar drawer pull start)
    # lives in the controller's after_base_reset_observation hook.

    def _eval_step_avatar(self):
        self._assist_controller().advance_eval_step(self)

    def _drawer_qpos(self) -> float:
        q = to_numpy(self.cabinet.entity.get_qpos()).ravel()
        return float(q[self._drawer_dof_idx])

    def _hold_drawer_open(self) -> bool:
        target = (
            self._drawer_open_qpos_target
            if self._drawer_held_qpos is None
            else self._drawer_held_qpos
        )
        ok = hold_dof_position_compat(
            self.cabinet.entity,
            self._drawer_dof_idx,
            target,
            kp=self._DRAWER_HOLD_KP,
            kv=self._DRAWER_HOLD_KV,
            force_limit=self._DRAWER_HOLD_FORCE,
        )
        if ok:
            self.sync_support_proxy(
                self._drawer_floor_proxy, self._drawer_link, self._drawer_support_surface_z(),
            )
        return ok

    def _set_drawer_qpos(self, value: float) -> None:
        q = to_numpy(self.cabinet.entity.get_qpos()).ravel().copy()
        q[self._drawer_dof_idx] = float(value)
        self.cabinet.entity.set_qpos(q)

    def _start_avatar_body_return(self) -> None:
        self._assist_controller().start_avatar_body_return(self)

    def _step_avatar_body_return_pre_avatar_update(self) -> None:
        self._assist_controller().step_avatar_body_return(self)

    def step_sim(self):
        """Step sim and keep the avatar-owned drawer motion synchronized."""
        if self._avatar_drawer_sync_active and self._avatar_drawer_motion is not None:
            progress = float(self._avatar_drawer_motion.pull_progress())
            self._set_drawer_qpos(self._drawer_open_qpos_target * progress)
        self._step_avatar_body_return_pre_avatar_update()

        super().step_sim()

        if (self._avatar_drawer_sync_active
                and self._avatar_drawer_motion is not None
                and self.avatar is not None
                and self.avatar.spare()):
            self._set_drawer_qpos(self._drawer_open_qpos_target)
            self._drawer_held_qpos = self._drawer_open_qpos_target
            self._hold_drawer_open()
            self._avatar_drawer_sync_active = False
            self._avatar_drawer_motion = None
            if not self._avatar_idle_return_started:
                self._avatar_idle_return_started = True
                self.avatar.play_return_to_idle(frames=self.AVATAR_RETURN_IDLE_FRAMES)
                self._start_avatar_body_return()

    def _start_open_drawer_with_avatar(self) -> bool:
        """Start the avatar drawer pull; drawer qpos is synced by step_sim."""
        return self._assist_controller().start_open_drawer_with_avatar(self)

    def _wait_for_avatar_drawer_open(self) -> bool:
        return self._assist_controller().wait_for_avatar_drawer_open(self)

    def _open_drawer_with_avatar(self) -> bool:
        """Blocking helper for drawer-only debug scripts."""
        if not self._start_open_drawer_with_avatar():
            return False
        return self._wait_for_avatar_drawer_open()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        # Settle the object on the table.
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(80):
                self.step_sim()

        if self.cabinet is None or self.object is None:
            return False

        self._pick_drawer()

        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        # Phase A — normally the avatar opens the drawer while the robot
        # begins the pickup. In no-human mode, fall back to the task's
        # robot-only drawer-opening primitive.
        if self.avatar is None:
            self._set_arm_pd(arm_tag, arm_kp=15000.0, arm_kv=800.0)
            ok_open = self._open_drawer_physics(arm_tag)
            self._set_arm_pd(arm_tag, arm_kp=4000.0, arm_kv=300.0)
        else:
            ok_open = self._start_open_drawer_with_avatar()
        if not ok_open:
            return False
        if self.avatar is None and not self._hold_drawer_open():
            return False

        # Phase B+C — pick the randomized object and drop into open drawer via the
        # mixin's ``pick_and_place`` helper.  This uses the proven
        # seeded-IK + xy_err retry pattern from dump_bin_bigbin /
        # place_bread_in_basket; bypasses the silent IK relax that
        # affects plan_path-based moves.
        def _object_center():
            return self._live_object_center().copy()

        def _object_half_height():
            return self.live_entity_half_height(self.object.entity)

        self._soften_object_gripper_pd(arm_tag)

        # Drop target = inside the final open drawer cavity, computed from
        # the planned final handle position because the robot begins before
        # the drawer has physically finished opening.  X comes from the live
        # drawer AABB; Y keeps the payload's full live footprint behind the
        # final front edge.
        drawer_aabb = to_numpy(self._drawer_link.get_AABB())
        drop_x = 0.5 * (float(drawer_aabb[0, 0]) + float(drawer_aabb[1, 0]))
        final_front_y = float(
            self._drawer_handle_world0[1] + self._drawer_open_qpos_target
        )
        object_aabb = to_numpy(self.object.entity.get_AABB())
        object_depth_y = float(object_aabb[1, 1] - object_aabb[0, 1])
        drop_y = self.contained_center_from_edge(
            final_front_y,
            object_depth_y,
            inward_sign=-1.0,
            clearance=self._DRAWER_PLACEMENT_CLEARANCE,
            held_inward_offset=self._DRAWER_HELD_INWARD_ALLOWANCE,
        )
        drop_xyz = np.array([
            drop_x, drop_y,
            self._drawer_support_surface_z() + self._DROP_HOVER_ABOVE_DRAWER_FLOOR,
        ], dtype=float)

        pick = PickSpec(
            get_center=_object_center,
            # Use the object's true cross-section radius (base + interrupt
            # variants do this). A stale hardcoded 0.005 m made the lift target
            # `obj_center + radius` rise only 5 mm — below a 7 cm block's top —
            # so the object never lifted clear of the table (dz≈0).
            radius=float(self._object_radius),
            label=self._object_id,
            get_half_height=_object_half_height,
            after_close=lambda: self._boost_finger_pd(arm_tag),
            require_lift=True,
            transport_retention_tolerance=0.08,
            **self._payload_pick_options(),
        )
        # transport_z must clear the cabinet top (≈ rim_z + 0.03 = 0.99 m)
        # AND the drawer rim during the lateral transit + hover-release.
        place = PlaceSpec(
            pos=drop_xyz,
            label="drawer",
            transport_z=self._drawer_rim_z + 0.10,
            support_z=self._drawer_support_surface_z(),
            support_clearance=self._DRAWER_SUPPORT_CLEARANCE,
        )
        ok = self.pick_and_place(pick, place, arm_tag)
        if not ok:
            return False

        # Let everything settle so check_success reads steady state.
        for _ in range(60):
            self.step_sim()
        return True

    # Blind expert: the drawer counts as open within this distance of the
    # open target, and the wait gives up after this many sim steps.
    _BLIND_DRAWER_OPEN_TOL_M = 0.005
    _BLIND_DRAWER_WAIT_MAX_STEPS = 3000

    def _wait_for_observed_drawer_open(self) -> bool:
        target = abs(float(self._drawer_open_qpos_target))
        for _ in range(self._BLIND_DRAWER_WAIT_MAX_STEPS):
            if abs(self._drawer_qpos()) >= target - self._BLIND_DRAWER_OPEN_TOL_M:
                return True
            self.step_sim()
        return False

    def play_blind_once(self) -> bool:
        for _ in range(80):
            self.step_sim()

        if self.cabinet is None or self.object is None:
            return False

        self._pick_drawer()

        arm_tag = "right"
        self._boost_finger_pd(arm_tag)

        if self.avatar is None:
            self._set_arm_pd(arm_tag, arm_kp=15000.0, arm_kv=800.0)
            ok_open = self._open_drawer_physics(arm_tag)
            self._set_arm_pd(arm_tag, arm_kp=4000.0, arm_kv=300.0)
            if not ok_open:
                return False
        else:
            # The eval-mode avatar opens the drawer on its own schedule; wait
            # until the drawer is observed open rather than assuming when.
            if not self._wait_for_observed_drawer_open():
                return False
        if not self._hold_drawer_open():
            return False

        def _object_center():
            return self._live_object_center().copy()

        def _object_half_height():
            return self.live_entity_half_height(self.object.entity)

        self._soften_object_gripper_pd(arm_tag)

        drawer_aabb = to_numpy(self._drawer_link.get_AABB())
        drop_x = 0.5 * (float(drawer_aabb[0, 0]) + float(drawer_aabb[1, 0]))
        final_front_y = float(
            self._drawer_handle_world0[1] + self._drawer_open_qpos_target
        )
        object_aabb = to_numpy(self.object.entity.get_AABB())
        object_depth_y = float(object_aabb[1, 1] - object_aabb[0, 1])
        drop_y = self.contained_center_from_edge(
            final_front_y,
            object_depth_y,
                inward_sign=-1.0,
                clearance=self._DRAWER_PLACEMENT_CLEARANCE,
                held_inward_offset=self._DRAWER_HELD_INWARD_ALLOWANCE,
            )
        drop_xyz = np.array([
            drop_x, drop_y,
            self._drawer_support_surface_z() + self._DROP_HOVER_ABOVE_DRAWER_FLOOR,
        ], dtype=float)

        pick = PickSpec(
            get_center=_object_center,
            # Use the object's true cross-section radius (base + interrupt
            # variants do this). A stale hardcoded 0.005 m made the lift target
            # `obj_center + radius` rise only 5 mm — below a 7 cm block's top —
            # so the object never lifted clear of the table (dz≈0).
            radius=float(self._object_radius),
                label=self._object_id,
                get_half_height=_object_half_height,
                after_close=lambda: self._boost_finger_pd(arm_tag),
                require_lift=True,
                transport_retention_tolerance=0.08,
                **self._payload_pick_options(),
        )
        place = PlaceSpec(
            pos=drop_xyz,
            label="drawer",
                transport_z=self._drawer_rim_z + 0.10,
                support_z=self._drawer_support_surface_z(),
                support_clearance=self._DRAWER_SUPPORT_CLEARANCE,
            )
        ok = self.pick_and_place(pick, place, arm_tag)
        if not ok:
            return False

        if self.avatar is not None:
            tail = 0
            while not self.avatar.spare() and tail < 1200:
                self.step_sim()
                tail += 1
        for _ in range(60):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def _object_in_drawer_metrics(self) -> dict:
        if self._drawer_link is None or self.object is None:
            return {
                "object_in_drawer": False,
                "object_in_drawer_error": "missing drawer link or object",
            }

        obj_p = self._live_object_center()
        # Object must lie inside the *current* drawer cavity, defined
        # by the live drawer-body AABB (which moves with the drawer's
        # prismatic joint).  Using ``get_pos()`` (joint origin) instead
        # would let an object sitting BEHIND the open drawer (inside
        # the cabinet body) pass.
        drawer_aabb = to_numpy(self._drawer_link.get_AABB())
        margin = 0.005  # 5 mm wall margin
        in_aabb_xy = (
            drawer_aabb[0, 0] + margin < obj_p[0] < drawer_aabb[1, 0] - margin
            and drawer_aabb[0, 1] + margin < obj_p[1] < drawer_aabb[1, 1] - margin
        )
        # Cabinet may have a ceiling above the drawer body (the
        # next-drawer-up's top panel for non-top drawers, or the
        # cabinet's own top panel above the top drawer).  Require the
        # object to ALSO be in front of the cabinet's body so it sits
        # in the *exposed* cavity slice — otherwise it's wedged
        # against / on the ceiling.  Cabinet front face is at
        # _CABINET_POS.y + half-Y * cabinet_scale.
        cabinet_front_y = (float(self._cabinet_pos[1])
                           + 0.094)  # body half-Y at scale 0.20
        exposed = bool(obj_p[1] > cabinet_front_y)
        in_xy = bool(in_aabb_xy and exposed)
        drawer_p = to_numpy(self._drawer_link.get_pos()).ravel()[:3]
        rim_z = self._drawer_rim_z
        in_z = bool(obj_p[2] > self._drawer_floor_z - 0.05
                    and obj_p[2] < rim_z + 0.02)
        ok = bool(in_xy and in_z)
        try:
            drawer_qpos = float(to_numpy(self.cabinet.entity.get_qpos()).ravel()[
                self._drawer_dof_idx])
        except Exception:
            drawer_qpos = None
        return {
            "object_in_drawer": ok,
            "object_in_drawer_xy": in_xy,
            "object_in_drawer_aabb_xy": bool(in_aabb_xy),
            "object_in_drawer_exposed": exposed,
            "object_in_drawer_z": in_z,
            "object_name": self._object_id,
            "object_model_id": int(self._object_model_id),
            "object_grasp_z_offset": float(getattr(self, "_object_grasp_z_offset", 0.0)),
            "object_center": obj_p.tolist(),
            "drawer_center": drawer_p.tolist(),
            "drawer_aabb_min": drawer_aabb[0].tolist(),
            "drawer_aabb_max": drawer_aabb[1].tolist(),
            "drawer_floor_z": float(self._drawer_floor_z),
            "drawer_rim_z": float(rim_z),
            "drawer_qpos": drawer_qpos,
            "cabinet_pos": self._cabinet_pos.tolist(),
            "cabinet_front_y": float(cabinet_front_y),
        }

    def check_success(self) -> bool:
        metrics = self._object_in_drawer_metrics()
        if "object_in_drawer_error" in metrics:
            return False
        return bool(metrics["object_in_drawer"])

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._object_in_drawer_metrics())
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
