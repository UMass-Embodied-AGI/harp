"""Stack three bowls (Interruption, ``stack_three_bowls_interruption``): same
scene as ``stack_bowls_three`` but a human avatar picks up + inspects one
of the two stacked bowls mid-task; the robot must wait, then resume on the
bowl's settled-after-detach pose.

Flow per episode:
  1. The task samples whether the avatar interrupts the 2nd or 3rd
     stack layer.  For layer 2, the interruption happens before any
     non-base bowl is stacked; for layer 3, the robot first stacks the
     other non-base bowl, then the interruption happens.
  2. Robot shows intent above the interrupted bowl (top-down EE pose,
     no commit) so the video reads as "robot was reaching for it".
  3. Avatar starts the ``Bowl_clip`` motion with ``attach_obj=`` the
     interrupted bowl — bowl attaches to a two-hand palm frame at frame
     ATTACH and detaches at DETACH.  Avatar lifts, holds, sets back.
  4. Robot retreats toward the robot base (out of the avatar's lean
     cone) and waits.
  5. ``avatar.spare()`` → bowl is back on the table at a slightly
     different pose; settle ~120 steps.
  6. Robot reads the bowl's NEW pose and runs the same
     ``_pick_and_stack_one`` primitive on it.

Geometry calibration (same recipe as ``dump_bin_interruption`` /
``place_bread_in_basket_interruption``):
  * Pre-bake: dry-run the Bowl_clip motion at ``avatar_init_pos`` and
    measure the avatar's two-hand bowl frame at the (scaled) attach
    frame. Cached on the class — depends only on motion + frame_ratio
    + home pose.
  * Constraint solve: avatar shift = bowl_pos − bowl_frame_default.
    After shifting, the two-hand frame meets the interrupted bowl at
    attach.
  * Both non-base bowls spawn in the same avatar-reachable subregion so
    placement does not reveal which bowl the human will inspect.

Eval inherits the parent's loose "stacked bowl center inside bowl 0's
horizontal footprint and higher z" gate, plus a hard fail on
``self.avatar_collided`` (avatar collision tracking is forced on in __init__).

Robot motion is pure physics (PD only — no kinematic teleport on the
arm or the held bowl).  The avatar attaches the bowl via
``play_animation(attach_obj=...)``, which is the sanctioned avatar
mechanism.
"""

import contextlib

import numpy as np
import transforms3d as t3d

# The YAML grasp loader consulted by BaseTask's grasp selection lives in
# the grasp_select split module; patch THAT binding (patching
# envs.base_task would be inert since the base_task split).
from ..base_task_parts import grasp_select as _grasp_select
from ..grasp import GraspPose, tcp_to_link_pose
from ..interaction.stack_bowls_three import StackBowlsThreeInterruptController
from ..utils import Pose, load_object
from ..task_bases.stack_bowls_three import (
    StackBowlsThree,
    _BOWL_ID,
    _BOWL_MODEL_ID,
    _MIN_SEP_M,
    _quat_mul,
    _read_extents,
    _read_scale,
    load_fixed_collision_bowl,
)


# Shared avatar-reachable subregion for non-base bowls.  Strict subset of
# the parent's full bowl-spawn region chosen so the Bowl_clip shift keeps the
# avatar beside the table while not revealing which non-base bowl is targeted.
_REACH_X_LO, _REACH_X_HI = -0.12, 0.18
_REACH_Y_LO, _REACH_Y_HI = -0.16, 0.04
_INT_X_LO, _INT_X_HI = _REACH_X_LO, _REACH_X_HI
_INT_Y_LO, _INT_Y_HI = _REACH_Y_LO, _REACH_Y_HI
_BASE_X_LO, _BASE_X_HI = -0.08, 0.10
_BASE_Y_LO, _BASE_Y_HI = -0.10, 0.02
_LIFT_X_LO, _LIFT_X_HI = _REACH_X_LO, _REACH_X_HI
_LIFT_Y_LO, _LIFT_Y_HI = _REACH_Y_LO, _REACH_Y_HI
_INTERRUPT_SAMPLE_ATTEMPTS = 256
_INTERRUPT_FALLBACK_XY = [(-0.12, 0.08), (0.14, 0.06), (0.02, -0.12)]


def _sample_bowl_xy_with_interrupt(rng, n_bowls):
    """Compatibility alias: the layout now lives on the interrupt controller."""
    return _layout_controller().sample_bowl_xy(rng, n_bowls)


def _layout_controller() -> StackBowlsThreeInterruptController:
    return StackBowlsThreeInterruptController(
        sample_attempts=_INTERRUPT_SAMPLE_ATTEMPTS,
        fallback_xy=_INTERRUPT_FALLBACK_XY,
        base_x=(_BASE_X_LO, _BASE_X_HI),
        base_y=(_BASE_Y_LO, _BASE_Y_HI),
        lift_x=(_LIFT_X_LO, _LIFT_X_HI),
        lift_y=(_LIFT_Y_LO, _LIFT_Y_HI),
        reach_x=(_REACH_X_LO, _REACH_X_HI),
        reach_y=(_REACH_Y_LO, _REACH_Y_HI),
        min_separation=_MIN_SEP_M,
    )


class StackThreeBowlsInterruption(StackBowlsThree):
    """Robot stacks three bowls; a human avatar picks up + inspects
    one of the two stacked bowls (bowl 1 or 2) mid-task — robot must
    wait, then resume on the bowl's settled-after-detach pose.

    The interruption layer is randomized between the 2nd and 3rd stack
    layer.  Bowl 0 stays the base; the avatar can handle either bowl 1
    or bowl 2 before the robot stacks it.
    """

    INSTRUCTION = "stack the three bowls on top of each other; if the human inspects one bowl midway through, wait and then continue"
    use_avatar = True
    # Expert side: also offer the finger-swapped rim pinch and require wrist
    # headroom for the carry (as the Contribution variant does).  Grasp candidates
    # only; the scene and the bowl collision are unchanged.
    BOWL_WRIST_FLIP = True
    # Fixed 200-hull bowl collision (as in the Contribution variant).
    # Runtime CoACD hulls form a ~4.5 deg wedge wall on which the rim pinch
    # creeps out (held offset 2 -> 7 -> 29 mm, then lost); on the fixed
    # hulls it stays at 1.5 mm.
    BOWL_FIXED_COLLISION = True

    # Avatar at +Y side of the table, facing -Y (toward the table).
    # Maps mesh-+X (avatar forward) to world-(-Y).  Same convention as
    # dump_bin_interruption / place_bread_in_basket_interruption.
    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 0.55, -0.18])
    avatar_init_rot = AVATAR_BASE_ROT

    BOWL_MOTION_NAME = "Bowl_clip"
    BOWL_ATTACH_FRAME = 45
    BOWL_DETACH_FRAME = 168
    AVATAR_MOTION_SLOW = 5.0
    BOWL_TWO_HAND_OFFSET = np.array([0.0, 0.0, -0.015], dtype=np.float64)

    # In the authored bowl frame, object-local +y is world-up.  A small
    # lift keeps the fingers clear of the lower wall while retaining
    # substantially more rim engagement than a 15 mm correction.
    GRASP_Z_LIFT = 0.005
    INTERRUPT_SEED_OFFSET = 4104
    DRYRUN_RESET_SETTLE_STEPS = 20
    CALIBRATION_SETTLE_STEPS = 30
    RETREAT_Y_FROM_BASE = 0.25
    RETREAT_Z_ABOVE_TABLE = 0.30
    INTENT_Z_ABOVE_TABLE = 0.20
    # None ⇒ avatar freezes on the bowl motion's last frame (pose + position).
    BOWL_RETURN_TO_IDLE_FRAMES = None
    PRE_ATTACH_WAIT_MARGIN_STEPS = 100
    POST_RELEASE_SETTLE_STEPS = 120
    # Bounds of the retired random-step trigger.  Only its RNG draw remains
    # (burned at reset so seeded scenes keep their release-v1 stream).
    EVAL_TRIGGER_STEP_MIN = 20
    EVAL_TRIGGER_STEP_MAX = 100
    # Progress-trigger interrupt contract (envs/interaction/interrupt_progress.py):
    # 3-D TCP to bowl-centre progress trigger; the bowl rim is wide.
    # Fitted to the demonstrated intent hover (TABLE_TOP_Z + 0.20 m, 0.162-
    # 0.166 m from the bowl, progress 0.45-0.46): 0.5 reached the hover only
    # through the scripted path's dip, so a policy that goes straight to the
    # hover was interrupted only on the rim grasp.  0.35 fires on the hover.
    INTERRUPT_COMMIT_RADIUS_M = 0.24
    INTERRUPT_TRIGGER_PROGRESS = 0.35
    INTERRUPT_MIN_POLICY_STEPS = 20
    # A rim grasp holds the TCP ~7 cm from the bowl centre.
    INTERRUPT_HELD_TCP_RADIUS_M = 0.10

    def __init__(self, config: dict = None):
        # Force avatar-collision tracking on so check_success can hard-
        # fail any episode where the robot bumps the avatar.  Caller may
        # still pass `track_avatar_collision` explicitly to override.
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", True)
        avatar_cfg = dict(cfg.get("avatar", {}))
        avatar_cfg.setdefault(
            "generated_motion_data",
            "avatars/motions/generated_motions.pkl",
        )
        cfg["avatar"] = avatar_cfg
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Scene setup — re-implements parent's `load_actors` so the non-base
    # bowls' xy positions are constrained to the shared avatar-reach subregion.
    # ------------------------------------------------------------------

    def create_interaction_controller(self) -> StackBowlsThreeInterruptController:
        return _layout_controller()

    def _interrupt_controller(self) -> StackBowlsThreeInterruptController:
        controller = self.interaction_controller
        if not isinstance(controller, StackBowlsThreeInterruptController):
            raise TypeError("task does not use stack-bowls interrupt behavior")
        return controller

    # Reset ordering (frame ratio, calibration, eval-state arming, trigger
    # draw) lives in the controller's after_base_reset_observation hook; the
    # episode seed is recorded by its on_episode_seeded hook.

    def load_actors(self):
        # Pick the interrupted bowl ∈ {1, 2}.  Bowl 0 is the stack base —
        # moving the base ruins the stack target (per brief).  Separately
        # pick whether the interruption occurs on the 2nd or 3rd stack layer.
        # Tie this high-level task choice directly to the episode seed,
        # instead of the global RNG state after robot/camera setup, so seed
        # sweeps are reproducible and small sweeps cover both interruption
        # timings.
        self._interrupt_controller().choose_interrupt_plan(self)

        upright_q = self.BOWL_UPRIGHT_QUAT
        table_top = self.TABLE_TOP_Z

        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = self._interrupt_controller().sample_bowl_xy(rng, self._num_bowls)

        self._bowl_scale = _read_scale(_BOWL_ID, _BOWL_MODEL_ID)
        bowl_extents = _read_extents(_BOWL_ID, _BOWL_MODEL_ID)
        self._bowl_extents = bowl_extents
        self._bowl_height = float(bowl_extents[1]) * self._bowl_scale

        yaws = [
            float(rng.uniform(-self.BOWL_YAW_LIMIT, self.BOWL_YAW_LIMIT))
            for _ in range(self._num_bowls)
        ]

        spawn_z = table_top + self.BOWL_SPAWN_Z_CLEARANCE
        self.bowls = []
        self._spawn_xy = []
        for i, ((cx, cy), yaw) in enumerate(zip(positions, yaws)):
            yaw_q = np.asarray(
                t3d.quaternions.mat2quat(
                    t3d.euler.euler2mat(0, 0, yaw, "sxyz")
                ),
                dtype=np.float64,
            )
            full_q = _quat_mul(yaw_q, upright_q)
            loader = (load_fixed_collision_bowl if self.BOWL_FIXED_COLLISION
                      else load_object)
            actor = loader(
                self.scene,
                Pose([cx, cy, spawn_z], full_q),
                _BOWL_ID, model_id=_BOWL_MODEL_ID,
                convex=True, is_static=False,
                friction=self.BOWL_FRICTION,
                density=self.BOWL_DENSITY,
            )
            self.bowls.append(actor)
            self._spawn_xy.append((cx, cy))

        self._stacked_indices = list(range(1, self._num_bowls))

    # ------------------------------------------------------------------
    # Avatar calibration — delegated to the controller
    # ------------------------------------------------------------------

    @property
    def _attach_frame_scaled(self) -> int:
        return self._interrupt_controller().attach_frame_scaled(self)

    @property
    def _detach_frame_scaled(self) -> int:
        return self._interrupt_controller().detach_frame_scaled(self)

    def _bowl_frame_center(self):
        return self._interrupt_controller().bowl_frame_center(self)

    def _dryrun_capture_bowl_frame(self):
        return self._interrupt_controller().dryrun_capture_bowl_frame(self)

    def _calibrate_avatar_for_bowl_motion(self):
        self._interrupt_controller().calibrate_avatar(self)

    # ------------------------------------------------------------------
    # Grasp up-lift override — wraps `_pick_and_stack_one` so the rim
    # grasp TCP sits `GRASP_Z_LIFT` higher than the YAML pose.
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _with_grasp_z_lift(self):
        """Monkey-patch ``base_task_parts.grasp_select.load_grasp_poses`` to return
        copies whose up component is shifted by ``GRASP_Z_LIFT`` (in
        world metres).  The authored bowl mesh uses object-local +y as
        world-up.  ``_scaled_pose`` later multiplies by ``object_scale``
        for ``mesh_unit`` poses, so we pre-divide to land the world shift
        at exactly ``GRASP_Z_LIFT``.
        """
        delta = float(self.GRASP_Z_LIFT)
        if delta == 0.0:
            yield
            return

        original = _grasp_select.load_grasp_poses
        scale = float(self._bowl_scale)

        def _patched(*args, **kwargs):
            grasps = original(*args, **kwargs)
            lifted = []
            for g in grasps:
                local_up = (
                    delta / scale if g.scale_frame == "mesh_unit" else delta
                )
                new_pos = np.asarray(g.pose.p, dtype=np.float64).copy()
                new_pos[1] += local_up
                ng = GraspPose(
                    name=g.name,
                    position=new_pos,
                    quaternion=np.asarray(g.pose.q, dtype=np.float64).copy(),
                    pre_distance=g.pre_distance,
                    source=g.source,
                    scale_frame=g.scale_frame,
                    category=g.category,
                    kinematic_pin=g.kinematic_pin,
                )
                lifted.append(ng)
            return lifted

        _grasp_select.load_grasp_poses = _patched
        try:
            yield
        finally:
            _grasp_select.load_grasp_poses = original

    def _pick_and_stack_one(self, src_idx, stack_position):
        """Execute one physical grasp using the interrupt-safe lifted pose."""
        with self._with_grasp_z_lift():
            return super()._pick_and_stack_one(src_idx, stack_position)

    # ------------------------------------------------------------------
    # Robot retreat — pull the EE toward the robot base + lift, so the
    # gripper is well clear of the avatar's forward-lean cone before the
    # avatar's attach frame fires.
    # ------------------------------------------------------------------

    def _retreat_safe(self, arm_tag: str):
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        retreat_p = np.array([
            arm_base[0],
            arm_base[1] + self.RETREAT_Y_FROM_BASE,
            self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE,
        ])
        # Top-down TCP rotation (same convention as parent
        # _pick_and_stack_one's place pose).
        R_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        q_tcp_down = t3d.quaternions.mat2quat(R_tcp_down)
        retreat_link = tcp_to_link_pose(
            Pose(retreat_p, q_tcp_down), tcp_offset,
        )
        self.move_and_execute(retreat_link.to_pose7(), arm_tag)

    def _wait_until_avatar_attach(self):
        self._interrupt_controller().wait_until_avatar_attach(self)

    def take_action(self, action, action_type: str = "qpos"):
        # Tick the interrupt, then apply the policy action
        # unmodified (no env-forced retreat/hold/return).
        if bool(self.config.get("eval_mode", False)) and self.avatar is not None:
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    # ------------------------------------------------------------------
    # Eval-mode interrupt contract — delegated to the controller.
    # ------------------------------------------------------------------

    def _closest_interrupt_bowl_to_ee(self):
        return self._interrupt_controller().closest_interrupt_bowl_to_ee(self)

    def _eval_step_avatar(self):
        self._interrupt_controller().advance_eval_step(self)

    def eval_skip_policy_action(self) -> bool:
        return self._interrupt_controller().eval_skip_policy_action(self)

    def consume_eval_policy_reset_requested(self) -> bool:
        return self._interrupt_controller().consume_eval_policy_reset_requested(self)

    # ------------------------------------------------------------------
    # Full-state expert under the eval-mode interrupt contract
    # (driven by envs.interaction.interrupt_progress.run_full_state_interrupt_expert)
    # ------------------------------------------------------------------

    def interrupt_expert_attempt(self) -> bool:
        """Idempotent scripted rollout: stack every non-base bowl that is
        not yet on the stack, reading live poses."""
        if not getattr(self, "_interrupt_expert_started", False):
            self._interrupt_expert_started = True
            with self.suppress_recording():
                for _ in range(self.INITIAL_SETTLE_STEPS):
                    self.step_sim()
        if any(b is None for b in self.bowls):
            return False
        controller = self._interrupt_controller()
        placed = [0] + [
            i for i in self._stacked_indices if controller.bowl_stacked(self, i)
        ]
        for idx in self._stacked_indices:
            if idx in placed:
                continue
            target = self._stack_target_for_placed(placed)
            if not self._pick_and_stack_one(idx, target):
                return False
            placed.append(idx)
        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    def interrupt_expert_retreat(self) -> None:
        self.open_gripper("right", num_steps=40)
        self._retreat_safe("right")

    def _run_avatar_interrupt(self, interrupted_idx: int, arm_tag: str) -> bool:
        """Show robot intent, let the avatar operate the bowl, then wait
        until the bowl has been returned to the table."""
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        # ---- Robot shows intent above the interrupted bowl -----------
        self.open_gripper(arm_tag)
        int_pos = self._bowl_xyz(interrupted_idx)
        intent_z = self.TABLE_TOP_Z + self.INTENT_Z_ABOVE_TABLE
        above_p = np.array([int_pos[0], int_pos[1], intent_z])
        R_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        q_tcp_down = t3d.quaternions.mat2quat(R_tcp_down)
        above_link = tcp_to_link_pose(
            Pose(above_p, q_tcp_down), tcp_offset,
        )
        self.move_and_execute(above_link.to_pose7(), arm_tag)

        # ---- Avatar plays Bowl_clip with two-hand attach -------------
        int_bowl = self.bowls[interrupted_idx]
        self.avatar.play_bowl_motion(
            attach_obj=int_bowl.entity,
            return_to_idle_after=self.BOWL_RETURN_TO_IDLE_FRAMES,
            snap_to_hands=False,
        )

        # ---- Robot retreats so the avatar can grab the bowl ----------
        self._retreat_safe(arm_tag)
        # Step until the avatar reaches the attach frame so the bowl is
        # visibly held before the robot moves on.
        self._wait_until_avatar_attach()
        for _ in range(self.PRE_ATTACH_WAIT_MARGIN_STEPS):
            self.step_sim()

        # ---- Wait for avatar to release the bowl ---------------------
        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        # Settle the released bowl on the table.
        for _ in range(self.POST_RELEASE_SETTLE_STEPS):
            self.step_sim()
        return True

    def _stack_target_for_placed(self, placed_indices):
        base_xyz = self._bowl_xyz(0)
        top_z = max(self._bowl_xyz(i)[2] for i in placed_indices)
        return (base_xyz[0], base_xyz[1], top_z + self._bowl_height)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def play_once(self) -> bool:
        # Settle the bowls before any robot motion.
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()
        if any(b is None for b in self.bowls):
            return False

        if self.avatar is None:
            return super().play_once()

        arm_tag = "right"

        interrupted_idx = self._interrupted_idx
        # _stacked_indices = [1, 2]; the OTHER stacked bowl is whichever
        # of {1, 2} isn't the interrupted one.
        other_idx = 3 - interrupted_idx

        if self._interrupt_stack_layer == 2:
            # Avatar interrupts the first bowl the robot would stack.
            if not self._run_avatar_interrupt(interrupted_idx, arm_tag):
                return False

            first_target = self._stack_target_for_placed([0])
            if not self._pick_and_stack_one(interrupted_idx, first_target):
                return False

            if self.simplified_mode_enabled():
                for _ in range(self.FINAL_SETTLE_STEPS):
                    self.step_sim()
                return True

            second_target = self._stack_target_for_placed([0, interrupted_idx])
            if not self._pick_and_stack_one(other_idx, second_target):
                return False

        else:
            # Avatar interrupts the final bowl; this is the original flow.
            first_target = self._stack_target_for_placed([0])
            if not self._pick_and_stack_one(other_idx, first_target):
                return False

            if not self._run_avatar_interrupt(interrupted_idx, arm_tag):
                return False

            second_target = self._stack_target_for_placed([0, other_idx])
            if not self._pick_and_stack_one(interrupted_idx, second_target):
                return False

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    # ------------------------------------------------------------------
    # Success — inherit parent's loose footprint + higher-z gate
    # plus a hard-fail on avatar collision.
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        if self.avatar_collided:
            return False
        # No robot-yield phase gates; a bowl the human still
        # holds is simply not stacked.
        return super().check_success()

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        metrics.update(self.interaction_controller.interaction_metrics(self))
        return metrics
