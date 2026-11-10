"""Place food in skillet (Interruption, ``place_food_in_skillet_interruption``).

The robot first places the skillet on the stove.  When it starts reaching
for one food item, a human avatar inspects that same item, sets it back on
the table, and the robot resumes from the item's latest physics pose.

The human side (interrupted-food choice, calibration, the reset-order hook,
the eval-mode retreat/glide/inspect machine) is owned by the composed
:class:`SkilletInterruptController`; this class keeps the robot
choreography, the ``take_action`` application boundary, and thin
``_``-prefixed delegators.
"""

from __future__ import annotations

import numpy as np
import transforms3d as t3d

from ..avatar.inspect_motion_mixin import STANDARD_INSPECT_POOL
from ..grasp import tcp_to_link_pose
from ..interaction.place_food_in_skillet import SkilletInterruptController
from ..utils import Pose
from ..task_bases.place_food_in_skillet import FoodSpec, SkilletTaskCore
from ._inspect_motion_interrupt_facade import InspectMotionInterruptFacade


class PlaceFoodInSkilletInterruption(
    InspectMotionInterruptFacade,
    SkilletTaskCore,
):
    """Robot places food in a skillet while a human briefly inspects one item."""

    INSTRUCTION = (
        "place the skillet on the stove and put the foods in it; the human "
        "may inspect one food item while you are reaching for it"
    )
    use_avatar = True

    # The avatar inspects the FIRST selected food (burger).  Its hard-mode
    # spawn (0.42, -0.12) sits ~0.6 m inboard of the avatar's table edge, and
    # the inspection reach translates the whole avatar body to the palm
    # target — dragging it through the tabletop.  Pull the
    # burger toward the +Y edge: (0.40, +0.05) stays clear of the counter
    # pan spawn (x <= 0.355) and within the Franka annulus (0.57 m from the
    # base at [0, -0.35]).
    FOOD_SPAWN_XY_OVERRIDE = {"burger": (0.40, 0.05)}

    # The bread starts already in the skillet, so the robot
    # manipulates exactly ONE object — the burger the human inspects.
    PREPLACED_FOOD_NAMES = ("bread",)

    AVATAR_BASE_ROT = np.array(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64,
    )
    avatar_init_pos = np.array([0.0, 1.05, -0.34])
    avatar_init_rot = AVATAR_BASE_ROT

    INSPECT_MOTION_NAME = "Inspect1"
    INSPECT_ATTACH_FRAME = 34
    INSPECT_DETACH_FRAME = 187
    INSPECT_POOL = STANDARD_INSPECT_POOL
    AVATAR_MOTION_SLOW = 10.0
    AVATAR_HAND_ID = 1
    AVATAR_EXTRA_Z = 0.04

    # Bounds of the retired random-step trigger.  Only its RNG draw remains
    # (burned at reset so seeded scenes keep their release-v1 stream).
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200
    # Progress-trigger interrupt contract (envs/interaction/interrupt_progress.py):
    # progress trigger gated on the pan being on the cooktop.
    #
    # Thresholds are matched to the demonstrated reach (training data
    # hr_v2_nyx37 / skillet interrupt): the robot's intent move hovers the TCP
    # at TABLE_TOP_Z + FOOD_TRANSPORT_ABOVE_TABLE (0.28 m), i.e. 0.26-0.30 m
    # (3-D) from the food centre with progress 0.27-0.40 from the ~0.43 m
    # reset distance, around policy step 10-20 for the trained policies.
    # The generic 0.22 m / 0.5 thresholds cannot fire on that hover; they
    # fire only on the later pick descent, so the human (1.5 s glide) lands
    # on the gripper mid-grasp -- a situation absent from every demo
    # (evaluated ACT/SmolVLA policies: 0% success, 63-80% contacts < 5 cm).
    # Retreat/home poses stay >= 0.45 m from the food.
    INTERRUPT_COMMIT_RADIUS_M = 0.32
    INTERRUPT_TRIGGER_PROGRESS = 0.25
    INTERRUPT_MIN_POLICY_STEPS = 10
    PACE_REQUIRE_PAN_READY = True
    PACE_PAN_READY_XY = 0.12

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("track_avatar_collision", True)
        # Two foods even in the easy (fixed-pan) layout: the interrupt story
        # inspects one food while the robot still has another to place.
        cfg.setdefault("foods", ["burger", "bread"])
        super().__init__(cfg)
        if "force_dynamic_pan" in cfg:
            self._force_dynamic_pan = bool(cfg["force_dynamic_pan"])
        self._interrupted_idx = 0

    def create_interaction_controller(self) -> SkilletInterruptController:
        return SkilletInterruptController()

    def _interrupt_controller(self) -> SkilletInterruptController:
        controller = self.interaction_controller
        if not isinstance(controller, SkilletInterruptController):
            raise TypeError("task does not use skillet interrupt behavior")
        return controller

    # ------------------------------------------------------------------
    # Legacy private helpers, delegated to the controller.  Reset ordering
    # (inspect resolve -> select -> eval init or calibrate) lives in the
    # controller's after_base_reset_observation hook.
    # ------------------------------------------------------------------

    def _select_interrupted_food(self) -> None:
        self._interrupt_controller().select_interrupted_food(self)

    def _dryrun_capture_palm(self):
        return self._interrupt_controller().dryrun_capture_palm(self)

    def _calibrate_avatar_for_inspect(self):
        self._interrupt_controller().calibrate_avatar(self)

    def _retreat_safe(self, arm_tag: str) -> None:
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p, dtype=float)
        retreat_p = np.array([
            arm_base[0],
            arm_base[1] + 0.25,
            self.TABLE_TOP_Z + 0.30,
        ])
        r_tcp_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        retreat_link = tcp_to_link_pose(
            Pose(retreat_p, t3d.quaternions.mat2quat(r_tcp_down)),
            arm.tcp_offset,
        )
        self.move_and_execute(retreat_link.to_pose7(), arm_tag)

    def _move_above_food_intent(self, actor, spec: FoodSpec, arm_tag: str) -> None:
        arm = self.robot.get_arm(arm_tag)
        center = self._actor_center(actor, spec.asset_id, spec.model_id)
        above = np.array([
            center[0],
            center[1],
            self.TABLE_TOP_Z + self.FOOD_TRANSPORT_ABOVE_TABLE,
        ])
        above_link = tcp_to_link_pose(
            self._rotated_top_down_tcp(above, spec.tcp_yaw_deg),
            arm.tcp_offset,
        )
        self.open_gripper(arm_tag, num_steps=80)
        self._move_seeded(above_link.to_pose7(), arm_tag)

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_finger_pd(arm_tag)
        with self.suppress_recording():  # pre-task settle, excluded from recordings
            for _ in range(60):
                self.step_sim()

        if self.avatar is None:
            return super().play_once()
        if not self._pan_starts_on_cooktop() and not self._place_pan_on_cooktop(
            arm_tag
        ):
            return False

        interrupted_actor, interrupted_spec = self.food_actors[self._interrupted_idx]
        self._move_above_food_intent(interrupted_actor, interrupted_spec, arm_tag)

        self._play_inspect_motion(
            attach_obj=interrupted_actor.entity,
            hand_id=self.AVATAR_HAND_ID,
            attach_frame=self._attach_frame_scaled,
            detach_frame=self._detach_frame_scaled,
            return_to_idle_after=None,  # freeze on last inspect frame (pose + position)
        )
        self._retreat_safe(arm_tag)
        self._wait_until_avatar_attach()
        for _ in range(100):
            self.step_sim()

        ok_any = False
        for idx, (actor, spec) in enumerate(self.food_actors):
            if idx == self._interrupted_idx:
                continue
            if self._food_in_skillet(actor, spec):
                # Pre-placed in the pan — nothing to do.
                ok_any = True
                continue
            ok = self._pick_and_drop_food(actor, spec, arm_tag)
            ok_any = ok_any or ok

        if not self.avatar.spare():
            while not self.avatar.spare():
                self.step_sim()
        for _ in range(120):
            self.step_sim()

        ok = self._pick_and_drop_food(interrupted_actor, interrupted_spec, arm_tag)
        ok_any = ok_any or ok

        for _ in range(180):
            self.step_sim()
        return ok_any

    def _init_eval_interrupt_state(self) -> None:
        self._interrupt_controller().init_eval_state(self)

    def _eval_step_avatar(self) -> None:
        self._interrupt_controller().advance_eval_step(self)

    def _pace_pan_ready(self) -> bool:
        return self._interrupt_controller().pace_pan_ready(self)

    def _find_closest_table_food(self):
        best_idx, _ = self._closest_table_food_candidate()
        return best_idx

    def _closest_table_food_candidate(self):
        return self._interrupt_controller().closest_table_food_candidate(self)

    def check_success(self) -> bool:
        if self.avatar_collided:
            return False
        return super().check_success()

    def evaluate(self) -> dict:
        out = super().evaluate()
        out.update(self.interaction_controller.interaction_metrics(self))
        return out

    def take_action(self, action, action_type: str = "qpos"):
        # Tick the interrupt, then apply the policy action unmodified.
        if self.avatar is not None and bool(self.config.get("eval_mode", False)):
            self._eval_step_avatar()
        return super().take_action(action, action_type=action_type)

    # ------------------------------------------------------------------
    # Full-state expert under the eval-mode interrupt contract
    # (driven by envs.interaction.interrupt_progress.run_full_state_interrupt_expert)
    # ------------------------------------------------------------------

    def interrupt_expert_attempt(self) -> bool:
        """Idempotent scripted rollout: place the pan if it is not on the
        cooktop, then pick every food that is not yet in the skillet from its
        live pose."""
        arm_tag = "right"
        if not getattr(self, "_interrupt_expert_started", False):
            self._interrupt_expert_started = True
            self._boost_finger_pd(arm_tag)
            with self.suppress_recording():
                for _ in range(60):
                    self.step_sim()
        if not self._pace_pan_ready():
            if not self._place_pan_on_cooktop(arm_tag):
                return False
        ok_all = True
        for actor, spec in self.food_actors:
            if self._food_in_skillet(actor, spec):
                continue
            ok = self._pick_and_drop_food(actor, spec, arm_tag)
            ok_all = ok_all and bool(ok)
        for _ in range(180):
            self.step_sim()
        return ok_all

    def interrupt_expert_retreat(self) -> None:
        self.open_gripper("right", num_steps=40)
        self._retreat_safe("right")
