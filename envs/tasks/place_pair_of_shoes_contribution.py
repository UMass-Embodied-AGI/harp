"""Place pair of shoes (Contribution, ``place_pair_of_shoes_contribution``) over the shared
family core."""


import numpy as np
import transforms3d as t3d

from ..interaction.place_dual_shoes import DualShoesAssistController
from ..task_bases.place_dual_shoes import DualShoesTaskCore
from ..utils import Pose, to_numpy


_REGION_X = (-0.22, 0.22)
_ASSIST_SHOE_REGION_Y = (-0.16, -0.04)
_ASSIST_BOX_REGION_Y = (-0.18, -0.08)
_ASSIST_MIN_SEP_M = 0.20


class PlacePairOfShoesContribution(DualShoesTaskCore):
    """Avatar and robot cooperatively place one shoe each in the shoebox.

    The human side (shoe allocation, blind order draw, avatar pick-and-place,
    eval start) is owned by the composed :class:`DualShoesAssistController`;
    this class keeps the robot choreography and thin ``_`` delegators.
    """

    INSTRUCTION = (
        "the human will put one shoe in the shoebox; the robot puts the "
        "other shoe in the shoebox — work in parallel"
    )
    OBJECT_SET = ["041_shoe"]
    use_avatar = True

    _DUAL_SHOES_USE_DIFFICULTY_COUNT = False
    _DUAL_SHOES_ALLOW_TABLE_VARIANTS = False
    RELEASE_SETTLE_STEPS = 220
    RELEASE_FORCE_N = None

    # Avatar at +Y side of the table, facing -Y toward the table.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    LIFT_HEIGHT = 0.20         # m above grasp before transit
    APPROACH_HEIGHT = 0.22     # m above box rim before lowering
    DROP_HEIGHT = 0.08         # m above box rim at release
    # Box URDF is placed at table_top + this clearance (the fork hardcodes
    # the placement; the constant is needed by the pendant-drop depth calc).
    BOX_FLOOR_Z_CLEARANCE = 0.001

    AVATAR_APPROACH_FRAMES = 30
    AVATAR_TRANSPORT_FRAMES = 45
    AVATAR_RETRACT_FRAMES = 30
    AVATAR_FRAME_REPEAT = 6
    AVATAR_REFINE_ITERS = 1
    AVATAR_DROP_ABOVE_RIM = 0.10
    AVATAR_NATURAL_BODY_MARGIN = 0.40
    AVATAR_NATURAL_YAW_LIMIT_DEG = 30.0
    AVATAR_NATURAL_BODY_Y_BOUNDS = (0.22, 0.70)
    AVATAR_NATURAL_SETTLE_STEPS = 20

    # Eval/testbed mode: policy calls take_action(action)->obs while the
    # avatar starts its shoe assist once at a randomized policy step.
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 60

    # Both shoes must end up inside the box footprint for success.
    # Shoebox interior is ~16 × 25 cm.  Half the diagonal ≈ 0.149 m;
    # 0.18 m gives some slack for shoes that drop near the rim.
    SUCCESS_DIST_XY_M = 0.18
    # Shoe must end up above the box floor (within 1 cm tolerance).
    # Shoes resting on the box floor sit ~5 cm above the floor (their
    # half-height); a stricter z-floor check would reject those.
    SUCCESS_Z_FLOOR_MARGIN_M = -0.01

    recording_camera_pos = [1.1, 0.0, 1.95]
    recording_camera_lookat = [0.0, 0.0, 0.85]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("track_avatar_collision", False)
        super().__init__(cfg)
        self._eval_trigger_step = None
        self._eval_avatar_started = False
        self._eval_avatar_failed = False
        self._policy_step_count = 0
        self._blind_robot_order = None
        self._blind_robot_first_idx = None

    def create_interaction_controller(self):
        return DualShoesAssistController()

    def _assist_controller(self) -> DualShoesAssistController:
        controller = self.interaction_controller
        if not isinstance(controller, DualShoesAssistController):
            raise TypeError("task does not use dual-shoes assist behavior")
        return controller

    def _eval_step_avatar(self) -> None:
        self._assist_controller().advance_eval_step(self)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _sample_layout_positions(self, rng):
        return self._assist_controller().sample_layout_positions(
            self, rng,
            region_x=_REGION_X,
            box_region_y=_ASSIST_BOX_REGION_Y,
            shoe_region_y=_ASSIST_SHOE_REGION_Y,
            min_separation=_ASSIST_MIN_SEP_M,
        )

    def _assign_shoes(self, placed) -> None:
        self._assist_controller().assign_shoes(self, placed)

    def _sample_blind_robot_order(self) -> list[int]:
        return self._assist_controller().sample_blind_robot_order(self)

    # ------------------------------------------------------------------
    # Pick + place primitive
    # ------------------------------------------------------------------

    # In-box slot offset (along the box's mesh-x = SHORT axis, world-frame
    # rotated by box yaw).  Keep both targets closer to center than the old
    # 4.5 cm slots; seed 1 showed edge drops can leave a shoe outside.
    SLOT_OFFSET_M = 0.030
    AVATAR_SLOT_OFFSET_M = 0.015

    def _shoe_world_center(self, shoe) -> np.ndarray:
        origin, q = self._shoe_pos_quat(shoe)
        R = t3d.quaternions.quat2mat(q)
        scale = self._shoe_scale
        if isinstance(scale, np.ndarray):
            offset = self._shoe_center_mesh * scale
        else:
            offset = self._shoe_center_mesh * float(scale)
        return origin + R @ offset

    def _slow_avatar_pick_place(self, repeat: int) -> None:
        self._assist_controller().slow_avatar_pick_place(self, repeat)

    def _start_avatar_assist(self) -> bool:
        return self._assist_controller().start_avatar_assist(self)

    # ------------------------------------------------------------------
    # Full-state acquisition: wait out the human before a fallback grasp
    # ------------------------------------------------------------------

    # Physics steps the full-state expert may wait for the human to finish
    # its own pick-and-place when the avatar's arm blocks the robot's
    # midbody grasp (same bound as the core's post-pool wait).
    MIDBODY_AVATAR_WAIT_STEPS = 2000
    # The robot's shoe must still be where the candidates were generated.
    MIDBODY_RETRY_MAX_SHIFT_M = 0.005
    MIDBODY_RETRY_MAX_TURN_RAD = 0.05
    # Planner-cloud capsule inflation for the retry once the human is idle
    # (the core uses 1.8 for a moving human).  An idle human does not move
    # into the path, so the second retry (and the rest of the rollout while
    # the human stays idle) plans with a smaller margin; the human is still
    # in the cloud and any real contact is still scored.
    MIDBODY_IDLE_AVATAR_INFLATE = 1.3

    def _topdown_pick_shoe(self, shoe, arm_tag: str, label: str) -> bool:
        self._assist_pick_shoe = shoe
        try:
            return super()._topdown_pick_shoe(shoe, arm_tag, label)
        finally:
            self._assist_pick_shoe = None

    def select_and_execute_grasp(self, *args, **kwargs):
        """Full-state expert only: retry the midbody pool once after the
        human has finished, before the core falls back to other pools.

        The core tries every grasp pool (midbody, then yaw/oblique/end
        fallbacks) before it waits for a busy avatar.  In this task the
        human starts its own pick-and-place at reset, and its arm (inflated
        into the planner's point cloud) can sit over the robot's shoe while
        the robot plans: e.g. seed 9000015 planned midbody only with
        the avatar removed from the cloud, fell back to a heel end pinch
        and the shoe did not rise.  The fallback pinches pivot the shoe
        instead of lifting it (in a 10-seed CPU run both failures
        used one, all 8 successes used midbody), so the full-state expert
        waits for the human (as the core already does after the last
        pool) and plans midbody again on the refreshed cloud.  Nothing has
        executed before the retry.  The blind expert never waits for the
        human (``_assist_wait_for_human`` is set by ``play_once`` only).
        """
        result = super().select_and_execute_grasp(*args, **kwargs)
        pool = kwargs.get("candidate_grasps") or []
        shoe = getattr(self, "_assist_pick_shoe", None)
        avatar = getattr(self, "avatar", None)
        if (result is not None or shoe is None or avatar is None
                or not getattr(self, "_assist_wait_for_human", False)
                or getattr(self, "_assist_midbody_retried", False)
                or not pool or not all(g.name.startswith("shoe_midbody_") for g in pool)
                or avatar.spare()):
            return result
        self._assist_midbody_retried = True
        before = Pose(*self._shoe_pos_quat(shoe))
        waited = 0
        while not avatar.spare() and waited < self.MIDBODY_AVATAR_WAIT_STEPS:
            self.step_sim()
            waited += 1
        after = Pose(*self._shoe_pos_quat(shoe))
        shift = float(np.linalg.norm(after.p - before.p))
        turn = float(2 * np.arccos(np.clip(abs(np.dot(after.q, before.q)), 0.0, 1.0)))
        print(f"[shoe_assist] midbody blocked while the human works; waited {waited} "
              f"steps (spare={avatar.spare()}, shoe shift={shift:.4f} m)", flush=True)
        if (not avatar.spare() or shift > self.MIDBODY_RETRY_MAX_SHIFT_M
                or turn > self.MIDBODY_RETRY_MAX_TURN_RAD):
            return None
        arm_tag = kwargs.get("arm_tag", "right")
        kwargs["object_pose"] = after
        result = None
        for idle_margin in (False, True):
            # The second plan, and every later move of this rollout while
            # the human stays idle, use the idle-human margin
            # (_shoe_refresh_avatar below): the arm must not start a later
            # move inside a cloud it was planned around.
            self._assist_idle_margin = idle_margin
            self._shoe_refresh_avatar(arm_tag)
            result = super().select_and_execute_grasp(*args, **kwargs)
            print(f"[shoe_assist] midbody retry after the human "
                  f"({'idle-human' if idle_margin else 'core'} margin): "
                  f"{'planned' if result is not None else 'failed'}", flush=True)
            if result is not None:
                break
        if result is None:
            self._assist_idle_margin = False
            self._shoe_refresh_avatar(arm_tag)
        return result

    def _shoe_refresh_avatar(self, arm_tag):
        """Core margin (1.8x capsule radius) unless the idle-human margin was
        needed for the midbody retry and the human is still idle."""
        avatar = getattr(self, "avatar", None)
        if not (getattr(self, "_assist_idle_margin", False)
                and avatar is not None and avatar.spare()):
            return super()._shoe_refresh_avatar(arm_tag)
        from ..avatar.collision_checker import sample_capsule_planner_points

        self._shoe_refresh_fixtures()
        collider = getattr(self, "avatar_collider", None)
        if collider is not None:
            points = sample_capsule_planner_points(
                collider.current_capsules(),
                inflate_factor=float(self.MIDBODY_IDLE_AVATAR_INFLATE))
            if points is not None and points.size:
                self.robot.get_arm(arm_tag).planner.update_obstacles(points, resolution=.04)

    def play_once(self) -> bool:
        self._assist_wait_for_human = True
        self._assist_midbody_retried = False
        self._assist_idle_margin = False
        try:
            return self._play_once_full_state()
        finally:
            self._assist_wait_for_human = False
            self._assist_idle_margin = False

    def _play_once_full_state(self) -> bool:
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(60):
                self.step_sim()

        if not self.shoes or self.shoebox is None:
            return False

        arm_tag = "right"
        if self.avatar is None:
            return self._play_robot_only_fallback(arm_tag)

        avatar_idx = int(getattr(self, "_avatar_shoe_idx", 0))
        robot_idx = int(getattr(self, "_robot_shoe_idx", 1 - avatar_idx))
        robot_shoe = self.shoes[robot_idx]
        robot_label = chr(ord("A") + robot_idx)
        robot_slot = 1


        # ---- Avatar pick + place (non-blocking) -----------------------
        if not self._start_avatar_assist():
            return False

        # ---- Full-state yield: let the human finish first ---------------
        # The expert knows which shoe the human takes, so it holds still
        # until the human's pick-and-place has ended (as the skillet and
        # bowls Contribution experts do).  Planning around the moving human
        # failed carries on its inflated arm cloud (CPU seed
        # 9000043: hand vs scene_pcd mid-carry) and forced the end-pinch
        # fallbacks; an idle human plans with the idle-human margin.
        waited = 0
        while not self.avatar.spare() and waited < self.MIDBODY_AVATAR_WAIT_STEPS:
            self.step_sim()
            waited += 1
        self._expert_human_wait_sim_steps = waited
        if self.avatar.spare():
            self._assist_idle_margin = True

        # ---- Robot pick + place (pure-physics primitive) --------------
        try:
            ok_robot = self._pick_and_place_shoe(
                robot_shoe, arm_tag, robot_label, slot_idx=robot_slot,
            )
        except Exception as e:
            print(f"[shoe_assist] {robot_label}: pick_and_place raised: {e!r}")
            ok_robot = False

        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(150):
            self.step_sim()
        return ok_robot

    def _shoe_in_box(self, shoe) -> bool:
        box_pos = to_numpy(self.shoebox.entity.get_pos()).ravel()[:3]
        p = np.asarray(self._shoe_pos_quat(shoe)[0])
        dxy = float(np.linalg.norm(p[:2] - box_pos[:2]))
        return bool(
            dxy <= self.SUCCESS_DIST_XY_M
            and p[2] > self.TABLE_TOP_Z + self.SUCCESS_Z_FLOOR_MARGIN_M
        )

    def play_blind_once(self) -> bool:
        for _ in range(60):
            self.step_sim()

        if not self.shoes or self.shoebox is None:
            return False

        arm_tag = "right"
        if self.avatar is None:
            return self._play_robot_only_fallback(arm_tag)

        ok_any = False
        order = list(self._blind_robot_order or self._sample_blind_robot_order())
        if not order:
            return False
        self._blind_robot_order = order
        self._blind_robot_first_idx = int(order[0])
        # Blind expert commits to one shoe.  It must not recover by watching
        # which shoe the avatar already placed and then choosing the other.
        idx = int(order[0])
        shoe = self.shoes[idx]
        label = chr(ord("A") + idx)
        try:
            ok_any = bool(self._pick_and_place_shoe(
                shoe, arm_tag, label, slot_idx=0,
            ))
        except Exception as e:
            print(f"[shoe_assist] {label}: blind pick_and_place raised: {e!r}")
            ok_any = False
        if self.avatar is not None:
            tail = 0
            while not self.avatar.spare() and tail < 1200:
                self.step_sim()
                tail += 1
        for _ in range(150):
            self.step_sim()
        return ok_any

    def _play_robot_only_fallback(self, arm_tag: str) -> bool:
        """Serial parent-style fallback for debug configs with no avatar."""
        box_xy = to_numpy(self.shoebox.entity.get_pos()).ravel()[:2]
        order = sorted(
            range(len(self.shoes)),
            key=lambda i: float(np.linalg.norm(
                to_numpy(self.shoes[i].entity.get_pos()).ravel()[:2] - box_xy,
            )),
        )
        any_success = False
        for slot, idx in enumerate(order):
            label = chr(ord("A") + idx)
            ok = self._pick_and_place_shoe(
                self.shoes[idx], arm_tag, label, slot_idx=slot,
            )
            any_success = any_success or ok
        return any_success

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        """Both shoes must end up within `SUCCESS_DIST_XY_M` (xy) of
        the shoebox centre and at z above the box floor."""
        if (self.avatar_collision_checker is not None
                and self.avatar_collided):
            return False
        return super().check_success()

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self._shoe_in_box_metrics())
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
